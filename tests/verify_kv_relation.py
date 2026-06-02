"""
verify_kv_relation.py — End-to-end correctness and efficiency verification
for the KV nopeless cache optimisation.

Target model: DeepSeek-V2-Lite (deepseek-ai/DeepSeek-V2-Lite)

Tests
──────────────────────────────────────────────────────────────────────────────
  Test 1  N matrix accuracy
          Hook kv_b_proj output during one forward pass.
          Split into k_nope_ref and v_ref.
          Check max |k_nope_ref - v_ref @ N| and top-1 argmax match.
          Validates compute_n_matrix.py independently of the cache patch.

  Test 2  Output equivalence
          Generate tokens with the unpatched model → ref logits.
          Patch the model; generate with KVNopelessCache.
          Assert top-1 token matches at every position.

  Test 3  Cache shape
          After one prefill + one decode step, inspect key tensors in
          KVNopelessCache. Assert shape[-1] == qk_rope_head_dim, not
          qk_nope_head_dim + qk_rope_head_dim.

  Test 4  Memory savings across context lengths
          Measure DynamicCache (pre-patch) and KVNopelessCache (post-patch)
          at context lengths [128, 512, 1024, 2048, 4096].
          Print a table and assert the saving % is constant (within 1%).

  Test 5  Throughput  (informational, no pass/fail)
          Tokens/sec for the patched vs unpatched model.
          Gated with --skip-throughput.

Usage
──────────────────────────────────────────────────────────────────────────────
    python verify_kv_relation.py --model_path ./models/DeepSeek-V2-Lite
    python verify_kv_relation.py --model_path ./models/DeepSeek-V2-Lite --skip-throughput
    python verify_kv_relation.py --model_path ./models/DeepSeek-V2-Lite --layer_idx 2
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from src.compute_n_matrix import compute_N_matrix
from ops.kv_nopeless_cache import KVNopelessCache
from kv_patch import patch_kv_model


# ── Compatibility patch for DynamicCache ─────────────────────────────────────
#
# DeepSeek-V2-Lite's modeling_deepseek.py calls:
#     past_key_values.get_usable_length(seq_length)
#
# This method was added in transformers 4.40 and removed again in 4.47+.
# Patching the class globally fixes all instances, including those created
# internally by the model.

if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = _get_usable_length


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify KV nopeless cache optimisation")
    p.add_argument(
        "--model_path", type=str,
        default="./models/DeepSeek-V2-Lite",
        help="Path to the HuggingFace model directory.",
    )
    p.add_argument(
        "--prompt", type=str,
        default=(
            "The history of artificial intelligence begins in antiquity, "
            "with myths, stories and rumors of artificial beings endowed with "
            "intelligence or consciousness by master craftsmen."
        ),
        help="Prompt used for all generation tests.",
    )
    p.add_argument(
        "--layer_idx", type=int, default=0,
        help="Decoder layer index used for Test 1 hook (default: 0).",
    )
    p.add_argument(
        "--context_lengths", type=int, nargs="+",
        default=[2048, 4096, 8192, 16384, 32768],
        help="Token counts for Test 4 memory table.",
    )
    p.add_argument(
        "--gen_tokens", type=int, default=64,
        help="Number of new tokens generated in Tests 2 and 5.",
    )
    p.add_argument(
        "--throughput_tokens", type=int, default=256,
        help="Tokens generated per model in the Test 5 throughput benchmark.",
    )
    p.add_argument(
        "--skip-throughput", dest="skip_throughput", action="store_true",
        help="Skip Test 5 (throughput benchmark).",
    )
    p.add_argument(
        "--n_abs_tol", type=float, default=3e-1,
        help="Max |Δ| tolerance for Test 1 N-accuracy check (default: 3e-1).",
    )
    p.add_argument(
        "--output_abs_tol", type=float, default=4e-1,
        help="Max |Δlogit| tolerance for Test 2 output-equivalence (default: 4e-1).",
    )
    return p.parse_args()


# ── Pretty-print helpers ──────────────────────────────────────────────────────

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
INFO = "\033[94m INFO \033[0m"
SEP  = "─" * 64


def header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def log(label: str, passed: Optional[bool] = None, detail: str = "") -> None:
    if passed is None:
        tag = f"[{INFO}]"
    elif passed:
        tag = f"[{PASS}]"
    else:
        tag = f"[{FAIL}]"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag}  {label}{suffix}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str):
    print(f"\nLoading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).to(torch.bfloat16)
    model.eval()
    print(f"  dtype : {next(model.parameters()).dtype}")
    print(f"  device: {next(model.parameters()).device}")
    return model, tokenizer


def tokenize(tokenizer, prompt: str, device: torch.device) -> dict:
    return tokenizer(prompt, return_tensors="pt").to(device)


# ── Generation helper ─────────────────────────────────────────────────────────

def run_generation_logits(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    n_new_tokens: int,
    cache_type: str = "dynamic",
) -> torch.Tensor:
    """
    Generate n_new_tokens and return the stacked logits for each new token.

    cache_type: 'dynamic'  → model's internal cache (past_key_values=None on step 0)
                'nopeless' → KVNopelessCache passed explicitly

    attention_mask is extended by 1 at each decode step — DeepSeek-V2-Lite's
    modeling code requires attention_mask on every forward call.
    """
    device    = next(model.parameters()).device
    inputs    = tokenize(tokenizer, prompt, device)
    attn_mask = inputs["attention_mask"]

    all_logits = []
    past_kv    = KVNopelessCache() if cache_type == "nopeless" else None

    with torch.no_grad():
        for step in range(n_new_tokens):
            if step == 0:
                out = model(**inputs, past_key_values=past_kv, use_cache=True)
            else:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_ones((1, 1))], dim=1
                )
                out = model(
                    input_ids=next_token,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )
            logits     = out.logits[:, -1, :]
            past_kv    = out.past_key_values
            next_token = logits.argmax(dim=-1, keepdim=True)
            all_logits.append(logits.cpu())

    return torch.stack(all_logits, dim=0)   # [T, B, vocab]


# ── Test 1 — N matrix accuracy ────────────────────────────────────────────────

def test_n_accuracy(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    layer_idx: int = 0,
    max_abs_tol: float = 3e-1,
) -> bool:
    """
    Hook into kv_b_proj at layer_idx and capture its output.

    The hook captures the output of kv_b_proj, shape:
        [batch, seq, num_heads * (qk_nope_head_dim + v_head_dim)]

    Split into k_nope_ref and v_ref, then check:
        max |k_nope_ref  -  v_ref @ N|

    Runs PRE-PATCH to validate compute_n_matrix.py independently.
    """
    header("Test 1 — N matrix accuracy  (pre-patch)")

    attn     = model.model.layers[layer_idx].self_attn
    num_heads = int(attn.num_heads)
    v_head_dim = attn.v_head_dim
    device   = next(model.parameters()).device

    N = compute_N_matrix(attn, out_dtype=torch.bfloat16, verbose=False).to(device)
    qk_nope_head_dim = N.shape[2]

    log(f"N shape   : {tuple(N.shape)}", detail=f"layer {layer_idx}")
    log(f"N dtype   : {N.dtype}")

    if not hasattr(attn, "kv_b_proj"):
        log("Cannot find kv_b_proj — skipping Test 1", passed=False)
        return False

    log("Weight source: kv_b_proj.weight")

    captured: dict = {}

    def _hook(module, input, output):
        captured["kv_b_out"] = output.detach()

    handle = attn.kv_b_proj.register_forward_hook(_hook)

    inputs_tok = tokenize(tokenizer, prompt, device)
    with torch.no_grad():
        model(**inputs_tok, use_cache=False)

    handle.remove()

    if "kv_b_out" not in captured:
        log("Hook did not fire — module not reached during forward", passed=False)
        return False

    kv_out = captured["kv_b_out"]   # [B, S, num_heads*(qk_nope+v_dim)]
    B, S, _ = kv_out.shape

    kv_reshaped = kv_out.view(B, S, num_heads, qk_nope_head_dim + v_head_dim)
    k_nope_ref  = kv_reshaped[..., :qk_nope_head_dim].transpose(1, 2)  # [B, H, S, qk_nope]
    v_ref       = kv_reshaped[..., qk_nope_head_dim:].transpose(1, 2)  # [B, H, S, v_dim]

    k_nope_hat = torch.matmul(v_ref, N)

    diff       = (k_nope_hat - k_nope_ref).abs()
    max_diff   = diff.max().item()
    mean_diff  = diff.mean().item()
    top1_match = (
        k_nope_hat.argmax(dim=-1) == k_nope_ref.argmax(dim=-1)
    ).all().item()

    abs_ok = max_diff < max_abs_tol

    log(
        f"max  |k_nope - v @ N| < {max_abs_tol}",
        passed=abs_ok,
        detail=f"max={max_diff:.4f}  mean={mean_diff:.4f}",
    )
    log(
        "top-1 argmax match  ← primary correctness gate",
        passed=top1_match,
    )

    passed = top1_match
    print(f"\n  Overall Test 1: {'PASS' if passed else 'FAIL'}")
    return passed


# ── Test 2 — Output equivalence ───────────────────────────────────────────────

def test_output_equivalence(
    model: torch.nn.Module,
    tokenizer,
    ref_logits: torch.Tensor,
    prompt: str,
    n_new_tokens: int,
    max_abs_tol: float = 4e-1,
) -> bool:
    """
    Generate n_new_tokens with the already-patched model using KVNopelessCache.
    Compare against ref_logits collected before patching.

    Primary gate: top-1 token must match at every decode step.
    Secondary metric: max |Δlogit| reported but not used as gate.
    """
    header("Test 2 — Output equivalence  (patched vs unpatched)")

    pat_logits = run_generation_logits(
        model, tokenizer, prompt, n_new_tokens, cache_type="nopeless"
    )

    max_diff   = (pat_logits - ref_logits).abs().max().item()
    top1_ref   = ref_logits.argmax(dim=-1)
    top1_pat   = pat_logits.argmax(dim=-1)
    top1_match = (top1_ref == top1_pat).all().item()
    abs_ok     = max_diff < max_abs_tol

    log(
        f"max |Δlogit| < {max_abs_tol}",
        passed=abs_ok,
        detail=f"max={max_diff:.4f}",
    )
    log(
        "top-1 token unchanged  ← primary correctness gate",
        passed=top1_match,
    )
    log(f"tokens compared: {n_new_tokens}")

    passed = top1_match
    print(f"\n  Overall Test 2: {'PASS' if passed else 'FAIL'}")
    return passed


# ── Test 3 — Cache shape ──────────────────────────────────────────────────────

def test_cache_shape(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
) -> bool:
    """
    Run prefill + one decode step with KVNopelessCache.
    Assert stored key tensors have shape [..., qk_rope_head_dim],
    NOT [..., qk_nope_head_dim + qk_rope_head_dim].
    """
    header("Test 3 — Cache shape  (k_nope must not be stored)")

    device  = next(model.parameters()).device
    attn0   = model.model.layers[0].self_attn
    qk_rope = attn0.qk_rope_head_dim
    qk_nope = attn0.kv_relation.qk_nope_head_dim
    full_kd = qk_nope + qk_rope

    inputs_tok = tokenize(tokenizer, prompt, device)
    past_kv    = KVNopelessCache()

    with torch.no_grad():
        out      = model(**inputs_tok, past_key_values=past_kv, use_cache=True)
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)

        attn_mask = torch.cat(
            [inputs_tok["attention_mask"],
             inputs_tok["attention_mask"].new_ones((1, 1))], dim=1
        )
        out = model(
            input_ids=next_tok,
            attention_mask=attn_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
        past_kv = out.past_key_values

    stored_kd = past_kv.key_head_dim(layer_idx=0)

    log(f"Expected key head dim : {qk_rope}  (qk_rope_head_dim only)")
    log(f"Full key head dim     : {full_kd}  (what DynamicCache would store)")
    log(
        f"Actual stored key dim : {stored_kd}",
        passed=(stored_kd == qk_rope),
        detail="k_nope absent ✓" if stored_kd == qk_rope else "WRONG — k_nope is leaking into cache",
    )

    passed = (stored_kd == qk_rope)
    print(f"\n  Overall Test 3: {'PASS' if passed else 'FAIL'}")
    return passed


# ── Test 4 — Memory savings ───────────────────────────────────────────────────

def _run_to_length_standard(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    target_tokens: int,
) -> float:
    """
    Run generation to target_tokens using the model's default internal cache.
    Returns total cache size in MB.

    Handles both DynamicCache format and DeepSeek's legacy tuple format
    ((k0, v0), (k1, v1), ...).
    """
    device   = next(model.parameters()).device
    inputs   = tokenize(tokenizer, prompt, device)
    n_new    = max(1, target_tokens - inputs["input_ids"].shape[1])

    attn_mask = inputs["attention_mask"]
    with torch.no_grad():
        out      = model(**inputs, past_key_values=None, use_cache=True)
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(n_new - 1):
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=1)
            out      = model(
                input_ids=next_tok,
                attention_mask=attn_mask,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv  = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    if hasattr(past_kv, "key_cache"):
        key_bytes   = sum(t.nbytes for t in past_kv.key_cache   if t is not None)
        value_bytes = sum(t.nbytes for t in past_kv.value_cache if t is not None)
    else:
        # Legacy tuple format: ((k0, v0), (k1, v1), ...)
        key_bytes   = sum(kv[0].nbytes for kv in past_kv if kv is not None)
        value_bytes = sum(kv[1].nbytes for kv in past_kv if kv is not None)

    return (key_bytes + value_bytes) / 1e6


def _run_to_length_nopeless(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    target_tokens: int,
) -> float:
    """Fill a KVNopelessCache to target_tokens. Returns total cache size in MB."""
    device   = next(model.parameters()).device
    inputs   = tokenize(tokenizer, prompt, device)
    n_new    = max(1, target_tokens - inputs["input_ids"].shape[1])

    attn_mask = inputs["attention_mask"]
    past_kv   = KVNopelessCache()
    with torch.no_grad():
        out      = model(**inputs, past_key_values=past_kv, use_cache=True)
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(n_new - 1):
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=1)
            out      = model(
                input_ids=next_tok,
                attention_mask=attn_mask,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv  = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    return past_kv.cache_size_bytes()["total_bytes"] / 1e6


def test_memory_savings(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    context_lengths: list,
    std_sizes: dict,
) -> bool:
    """
    Print a memory table and assert saving % is constant across context lengths.

    std_sizes is a pre-collected dict {ctx: mb} measured before patching.

    The saving is purely a function of model dimensions:
        saving = qk_nope / (qk_nope + qk_rope)  ≈ 67% for DeepSeek-V2-Lite
    It should be constant at every context length; drift signals a bug.
    """
    header("Test 4 — Memory savings across context lengths")

    attn0           = model.model.layers[0].self_attn
    qk_nope         = attn0.kv_relation.qk_nope_head_dim
    qk_rope         = attn0.qk_rope_head_dim
    expected_saving = 100.0 * qk_nope / (qk_nope + qk_rope)

    log(f"qk_nope_head_dim : {qk_nope}")
    log(f"qk_rope_head_dim : {qk_rope}")
    log(f"Expected saving  : {expected_saving:.1f}%  (key cache only, not value)")
    print()

    col_w = [10, 18, 18, 10]
    header_row = (
        f"  {'Context':<{col_w[0]}}"
        f"{'Standard cache':>{col_w[1]}}"
        f"{'Nopeless cache':>{col_w[2]}}"
        f"{'Saving':>{col_w[3]}}"
    )
    rule = "  " + "─" * (sum(col_w) + 2)
    print(header_row)
    print(rule)

    savings = []
    for ctx in context_lengths:
        std_mb      = std_sizes[ctx]
        nopeless_mb = _run_to_length_nopeless(model, tokenizer, prompt, ctx)
        saving_pct  = 100.0 * (1.0 - nopeless_mb / std_mb) if std_mb > 0 else 0.0
        savings.append(saving_pct)
        print(
            f"  {f'{ctx} tok':<{col_w[0]}}"
            f"{f'{std_mb:.2f} MB':>{col_w[1]}}"
            f"{f'{nopeless_mb:.2f} MB':>{col_w[2]}}"
            f"{f'{saving_pct:.1f}%':>{col_w[3]}}"
        )

    print(rule)
    print()

    drift     = max(savings) - min(savings)
    drift_ok  = drift < 1.0
    mean_sav  = sum(savings) / len(savings)
    approx_ok = abs(mean_sav - expected_saving) < 5.0

    log(
        "Saving is constant across lengths (drift < 1pp)",
        passed=drift_ok,
        detail=f"min={min(savings):.1f}%  max={max(savings):.1f}%  drift={drift:.2f}pp",
    )
    log(
        f"Observed saving ≈ theoretical {expected_saving:.1f}%  (within 5pp)",
        passed=approx_ok,
        detail=f"observed mean={mean_sav:.1f}%",
    )

    passed = drift_ok and approx_ok
    print(f"\n  Overall Test 4: {'PASS' if passed else 'FAIL'}")
    return passed


# ── Test 5 — Throughput ───────────────────────────────────────────────────────

def _measure_tps(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    n_tokens: int,
    cache_type: str = "dynamic",
) -> float:
    """Generate n_tokens and return tokens/sec (decode phase only, not prefill)."""
    device    = next(model.parameters()).device
    inputs    = tokenize(tokenizer, prompt, device)
    attn_mask = inputs["attention_mask"]
    past_kv   = KVNopelessCache() if cache_type == "nopeless" else None

    with torch.no_grad():
        out      = model(**inputs, past_key_values=past_kv, use_cache=True)
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    # Warm-up step
    with torch.no_grad():
        attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=1)
        out      = model(
            input_ids=next_tok,
            attention_mask=attn_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(n_tokens):
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=1)
            out      = model(
                input_ids=next_tok,
                attention_mask=attn_mask,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv  = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    if device.type == "cuda":
        torch.cuda.synchronize()

    return n_tokens / (time.perf_counter() - t0)


def test_throughput(
    model_unpatched: torch.nn.Module,
    model_patched: torch.nn.Module,
    tokenizer,
    prompt: str,
    n_tokens: int = 256,
) -> None:
    """
    Informational only — no pass/fail gate.
    Reports decode tokens/sec for DynamicCache vs KVNopelessCache.
    """
    header("Test 5 — Throughput  (informational, no pass/fail)")
    log(f"Generating {n_tokens} decode tokens per model...")
    print()

    tps_std = _measure_tps(model_unpatched, tokenizer, prompt, n_tokens, "dynamic")
    tps_pat = _measure_tps(model_patched,   tokenizer, prompt, n_tokens, "nopeless")

    ratio = tps_pat / tps_std if tps_std > 0 else float("nan")

    col_w = [26, 14]
    print(f"  {'Model':<{col_w[0]}}{'Tokens / sec':>{col_w[1]}}")
    print("  " + "─" * (sum(col_w) + 2))
    print(f"  {'Unpatched (DynamicCache)':<{col_w[0]}}{tps_std:>{col_w[1]}.1f}")
    print(f"  {'Patched  (KVNopelessCache)':<{col_w[0]}}{tps_pat:>{col_w[1]}.1f}")
    print("  " + "─" * (sum(col_w) + 2))
    print(f"\n  Relative speed : {ratio:.3f}x")
    if ratio < 1.0:
        print(f"  Phase-1 overhead: {100.0*(1.0-ratio):.1f}%  (v@N matmul — expected at this phase)")
    else:
        print(f"  Patched is faster by {100.0*(ratio-1.0):.1f}%  (cache bandwidth reduction dominating)")
    print()
    log("Throughput is informational — no pass/fail gate")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 64)
    print("  KV Nopeless Cache — Verification Suite")
    print("  Target: DeepSeek-V2-Lite")
    print("=" * 64)
    print(f"  model     : {args.model_path}")
    print(f"  prompt    : {args.prompt[:60]}...")
    print(f"  layer_idx : {args.layer_idx}  (Test 1 hook)")
    print(f"  gen_tokens: {args.gen_tokens}")

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = next(model.parameters()).device

    # Pre-patch: collect reference logits and standard cache sizes
    print(f"\n{SEP}")
    print("  Pre-patch: collecting reference logits  (unpatched model)")
    print(SEP)
    ref_logits = run_generation_logits(
        model, tokenizer, args.prompt, args.gen_tokens, cache_type="dynamic"
    )
    print(f"  Reference logits shape: {tuple(ref_logits.shape)}")

    print(f"\n{SEP}")
    print("  Pre-patch: measuring standard DynamicCache sizes")
    print(SEP)
    std_sizes: dict = {}
    for ctx in args.context_lengths:
        mb = _run_to_length_standard(model, tokenizer, args.prompt, ctx)
        std_sizes[ctx] = mb
        print(f"    {ctx:5d} tokens → {mb:.2f} MB")

    # Test 1 — N accuracy (pre-patch)
    t1_pass = test_n_accuracy(
        model, tokenizer, args.prompt,
        layer_idx=args.layer_idx,
        max_abs_tol=args.n_abs_tol,
    )

    # Patch the model
    print(f"\n{SEP}")
    print("  Patching model with patch_kv_model()...")
    print(SEP)
    patch_kv_model(model, device=device)

    # Test 2 — Output equivalence
    t2_pass = test_output_equivalence(
        model, tokenizer, ref_logits, args.prompt,
        n_new_tokens=args.gen_tokens,
        max_abs_tol=args.output_abs_tol,
    )

    # Test 3 — Cache shape
    t3_pass = test_cache_shape(model, tokenizer, args.prompt)

    # Test 4 — Memory savings
    t4_pass = test_memory_savings(
        model, tokenizer, args.prompt,
        context_lengths=args.context_lengths,
        std_sizes=std_sizes,
    )

    # Test 5 — Throughput
    if args.skip_throughput:
        header("Test 5 — Throughput  (skipped via --skip-throughput)")
    else:
        print(f"\n{SEP}")
        print("  Test 5: reloading unpatched model for throughput comparison...")
        print(f"  (pass --skip-throughput to skip this if VRAM is limited)")
        print(SEP)
        model_ref, _ = load_model_and_tokenizer(args.model_path)
        test_throughput(
            model_ref, model,
            tokenizer, args.prompt,
            n_tokens=args.throughput_tokens,
        )
        del model_ref
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 64}")
    print("  SUMMARY")
    print(f"{'=' * 64}")

    results = {
        "Test 1 — N matrix accuracy    ": t1_pass,
        "Test 2 — Output equivalence   ": t2_pass,
        "Test 3 — Cache shape          ": t3_pass,
        "Test 4 — Memory savings       ": t4_pass,
    }
    all_pass = True
    for name, ok in results.items():
        tag = PASS if ok else FAIL
        print(f"  [{tag}]  {name}")
        if not ok:
            all_pass = False

    if args.skip_throughput:
        print(f"  [{INFO}]  Test 5 — Throughput         (skipped)")
    else:
        print(f"  [{INFO}]  Test 5 — Throughput         (informational only)")

    print(f"\n{'=' * 64}")
    print("  ALL TESTS PASSED" if all_pass else "  SOME TESTS FAILED — see details above")
    print(f"{'=' * 64}\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()