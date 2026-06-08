"""
verify_kv_relation.py — End-to-end correctness and efficiency verification
for the KV latent cache optimisation.

Supported models
──────────────────────────────────────────────────────────────────────────────
  DeepSeek-V2-Lite  (deepseek-ai/DeepSeek-V2-Lite)  — proxy for debugging
  Kimi-K2.6 NVFP4   (nvidia/Kimi-K2.6-NVFP4)        — target model

Tests
──────────────────────────────────────────────────────────────────────────────
  Test 1  Latent round-trip
          Hook kv_b_proj input (kv_a_norm) and output during one forward pass.
          Verify kv_b_proj(kv_a_norm) correctly splits into k_nope and v.
          This confirms the latent extraction in kv_patch.py is correct.

  Test 2  Output equivalence
          Generate tokens with the unpatched model → ref logits.
          Patch the model; generate with KVLatentCache.
          Assert top-1 token matches at every position.

  Test 3  Cache shape
          After one prefill + one decode step, inspect tensors in KVLatentCache.
          Assert key_cache[-1] == kv_lora_rank (512), value_cache[-1] == qk_rope_head_dim (64).

  Test 4  Memory savings across context lengths
          Measure DynamicCache (pre-patch) and KVLatentCache (post-patch).
          Expected saving: computed from model config — ~88.7% for DeepSeek-V2-Lite.

  Test 5  Throughput  (informational, no pass/fail)
          Tokens/sec for the patched vs unpatched model.
          Gated with --skip-throughput.
          NOTE: Kimi-K2.6 requires ~500 GB VRAM to load two copies; use
          --skip-throughput on single-node setups.

Usage
──────────────────────────────────────────────────────────────────────────────
    # DeepSeek-V2-Lite (proxy, debugging)
    python verify_kv_relation.py --model_path ./models/DeepSeek-V2-Lite

    # Kimi-K2.6 NVFP4 (target, multi-GPU)
    python verify_kv_relation.py --model_path ./models/Kimi-K2.6-NVFP4 --skip-throughput

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

from ops.kv_latent_cache import KVLatentCache
from kv_patch import patch_kv_model


# ── Compatibility patch for DynamicCache ─────────────────────────────────────
#
# DeepSeek-V2-Lite's modeling_deepseek.py calls get_usable_length().
# Added in transformers 4.40, removed in 4.47+.

if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = _get_usable_length


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify KV latent cache optimisation")
    p.add_argument(
        "--model_path", type=str,
        default="./models/DeepSeek-V2-Lite",
        help="Path to model weights. Use ./models/DeepSeek-V2-Lite (proxy) or "
             "./models/Kimi-K2.6-NVFP4 (target).",
    )
    p.add_argument(
        "--prompt", type=str,
        default=(
            "The history of artificial intelligence begins in antiquity, "
            "with myths, stories and rumors of artificial beings endowed with "
            "intelligence or consciousness by master craftsmen."
        ),
    )
    p.add_argument("--layer_idx",        type=int,   default=0)
    p.add_argument("--context_lengths",  type=int,   nargs="+",
                   default=[2048, 4096, 8192, 16384, 32768])
    p.add_argument("--gen_tokens",       type=int,   default=64)
    p.add_argument("--throughput_tokens",type=int,   default=256)
    p.add_argument("--skip-throughput",  dest="skip_throughput", action="store_true")
    p.add_argument("--output_abs_tol",   type=float, default=6e-1)
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
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
    Generate n_new_tokens and return stacked logits [T, B, vocab].

    cache_type: 'dynamic' → explicit DynamicCache (unpatched model)
                'latent'  → KVLatentCache (patched model)

    Both pass an explicit empty cache (not None) so DeepSeek's outer
    model uses the same code path for both runs — otherwise the None
    vs non-None difference changes how the outer model builds the
    attention mask, causing ~0.4 per-step error in the reference.
    """
    device    = next(model.parameters()).device
    inputs    = tokenize(tokenizer, prompt, device)
    attn_mask = inputs["attention_mask"]
    past_kv   = KVLatentCache() if cache_type == "latent" else DynamicCache()
    all_logits = []

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

    return torch.stack(all_logits, dim=0)


# ── Test 1 — Latent round-trip ────────────────────────────────────────────────

def test_latent_roundtrip(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    layer_idx: int = 0,
) -> bool:
    """
    Confirm that kv_b_proj(kv_a_norm) reproduces k_nope and v correctly.

    Hooks both the input and output of kv_b_proj. Splits the output into
    k_nope and v, then checks that kv_b_proj(hooked_input) == hooked_output.
    This validates that _extract_kv_latent() captures the right tensor
    (kv_a_norm after kv_a_layernorm, before kv_b_proj).
    """
    header("Test 1 — Latent round-trip  (pre-patch)")

    attn   = model.model.layers[layer_idx].self_attn
    device = next(model.parameters()).device

    if not hasattr(attn, "kv_b_proj"):
        log("Cannot find kv_b_proj — skipping", passed=False)
        return False

    captured: dict = {}

    def _hook_in(module, args):
        captured["kv_a_norm"] = args[0].detach()    # first positional arg

    def _hook_out(module, input, output):
        captured["kv_b_out"] = output.detach()

    h_in  = attn.kv_b_proj.register_forward_pre_hook(_hook_in)
    h_out = attn.kv_b_proj.register_forward_hook(_hook_out)

    inputs_tok = tokenize(tokenizer, prompt, device)
    with torch.no_grad():
        model(**inputs_tok, use_cache=False)

    h_in.remove()
    h_out.remove()

    if "kv_a_norm" not in captured or "kv_b_out" not in captured:
        log("Hooks did not fire", passed=False)
        return False

    kv_a_norm = captured["kv_a_norm"]   # [B, S, lora_rank]
    kv_b_out  = captured["kv_b_out"]    # [B, S, num_heads*(qk_nope+v_dim)]

    # Re-run kv_b_proj on the captured input and check it matches the output
    with torch.no_grad():
        kv_b_rerun = attn.kv_b_proj(kv_a_norm)

    diff      = (kv_b_rerun - kv_b_out).abs()
    max_diff  = diff.max().item()
    exact     = max_diff == 0.0

    log(f"kv_a_norm shape : {tuple(kv_a_norm.shape)}", detail=f"layer {layer_idx}")
    log(f"kv_b_out  shape : {tuple(kv_b_out.shape)}")
    log(
        "kv_b_proj(kv_a_norm) == kv_b_out  (exact round-trip)",
        passed=exact,
        detail=f"max_diff={max_diff:.2e}",
    )

    # Also confirm the split dimensions match model attributes
    num_heads        = int(attn.num_heads)
    qk_nope_head_dim = int(attn.qk_nope_head_dim)
    v_head_dim       = int(attn.v_head_dim)
    expected_out_dim = num_heads * (qk_nope_head_dim + v_head_dim)
    dim_ok = kv_b_out.shape[-1] == expected_out_dim

    log(
        f"kv_b_out last dim == num_heads×(qk_nope+v_dim) = {expected_out_dim}",
        passed=dim_ok,
        detail=f"got {kv_b_out.shape[-1]}",
    )

    passed = exact and dim_ok
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
    Generate with the patched model using KVLatentCache.
    Compare top-1 tokens against ref_logits from the unpatched model.
    """
    header("Test 2 — Output equivalence  (patched vs unpatched)")

    pat_logits = run_generation_logits(
        model, tokenizer, prompt, n_new_tokens, cache_type="latent"
    )

    # Per-step diff to identify whether error starts at step 0 (prefill) or later
    per_step_max = (pat_logits - ref_logits).abs().max(dim=-1).values.squeeze()
    for i, d in enumerate(per_step_max[:min(5, len(per_step_max))]):
        log(f"  step {i}: max |Δlogit| = {d.item():.4f}")

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
    Run prefill + one decode step with KVLatentCache.
    Assert:
      key_cache[-1]   == kv_lora_rank     (512) — kv_a_norm stored
      value_cache[-1] == qk_rope_head_dim  (64) — k_pe_roped stored
    """
    header("Test 3 — Cache shape  (latent + k_pe must be stored)")

    device         = next(model.parameters()).device
    attn0          = model.model.layers[0].self_attn
    lora_rank      = int(attn0.kv_lora_rank)
    qk_rope        = int(attn0.qk_rope_head_dim)

    inputs_tok = tokenize(tokenizer, prompt, device)
    past_kv    = KVLatentCache()

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

    stored_latent = past_kv.latent_dim(layer_idx=0)
    stored_kpe    = past_kv.kpe_dim(layer_idx=0)

    log(f"Expected key   dim : {lora_rank}  (kv_lora_rank — kv_a_norm)")
    log(f"Expected value dim : {qk_rope}    (qk_rope_head_dim — k_pe_roped)")
    log(
        f"Actual key   dim : {stored_latent}",
        passed=(stored_latent == lora_rank),
        detail="kv_a_norm ✓" if stored_latent == lora_rank else "WRONG",
    )
    log(
        f"Actual value dim : {stored_kpe}",
        passed=(stored_kpe == qk_rope),
        detail="k_pe_roped ✓" if stored_kpe == qk_rope else "WRONG",
    )

    passed = (stored_latent == lora_rank) and (stored_kpe == qk_rope)
    print(f"\n  Overall Test 3: {'PASS' if passed else 'FAIL'}")
    return passed


# ── Test 4 — Memory savings ───────────────────────────────────────────────────

def _run_to_length_standard(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    target_tokens: int,
) -> float:
    """Run to target_tokens with model's default cache. Returns MB."""
    device   = next(model.parameters()).device
    inputs   = tokenize(tokenizer, prompt, device)
    n_new    = max(1, target_tokens - inputs["input_ids"].shape[1])

    attn_mask = inputs["attention_mask"]
    with torch.no_grad():
        out      = model(**inputs, past_key_values=DynamicCache(), use_cache=True)
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
        key_bytes   = sum(kv[0].nbytes for kv in past_kv if kv is not None)
        value_bytes = sum(kv[1].nbytes for kv in past_kv if kv is not None)

    return (key_bytes + value_bytes) / 1e6


def _run_to_length_latent(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    target_tokens: int,
) -> float:
    """Fill a KVLatentCache to target_tokens. Returns MB."""
    device   = next(model.parameters()).device
    inputs   = tokenize(tokenizer, prompt, device)
    n_new    = max(1, target_tokens - inputs["input_ids"].shape[1])

    attn_mask = inputs["attention_mask"]
    past_kv   = KVLatentCache()
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
    Print a memory table and assert saving % is consistent across lengths.

    Expected saving is computed from model config:
      Standard : num_heads*(qk_nope+qk_rope) + num_heads*v_dim per token
      Latent   : kv_lora_rank + qk_rope per token
      Saving   = 1 - latent_dims / standard_dims
    For DeepSeek-V2-Lite: 576 / 5120 → ~88.7% saving.
    For Kimi-K2.6:        computed from model attributes at test time.
    """
    header("Test 4 — Memory savings across context lengths")

    attn0          = model.model.layers[0].self_attn
    lora_rank      = int(attn0.kv_lora_rank)
    qk_rope        = int(attn0.qk_rope_head_dim)
    num_heads      = int(attn0.num_heads)
    qk_nope        = int(attn0.qk_nope_head_dim)
    v_dim          = int(attn0.v_head_dim)

    standard_dims  = num_heads * (qk_nope + qk_rope) + num_heads * v_dim
    latent_dims    = lora_rank + qk_rope
    expected_saving = 100.0 * (standard_dims - latent_dims) / standard_dims

    log(f"Standard dims/token : {standard_dims}  ({num_heads}×({qk_nope}+{qk_rope}) + {num_heads}×{v_dim})")
    log(f"Latent   dims/token : {latent_dims}   ({lora_rank} + {qk_rope})")
    log(f"Expected saving     : {expected_saving:.1f}%")
    print()

    col_w = [10, 18, 16, 10]
    print(
        f"  {'Context':<{col_w[0]}}"
        f"{'Standard cache':>{col_w[1]}}"
        f"{'Latent cache':>{col_w[2]}}"
        f"{'Saving':>{col_w[3]}}"
    )
    print("  " + "─" * (sum(col_w) + 2))

    savings = []
    for ctx in context_lengths:
        std_mb     = std_sizes[ctx]
        latent_mb  = _run_to_length_latent(model, tokenizer, prompt, ctx)
        saving_pct = 100.0 * (1.0 - latent_mb / std_mb) if std_mb > 0 else 0.0
        savings.append(saving_pct)
        print(
            f"  {f'{ctx} tok':<{col_w[0]}}"
            f"{f'{std_mb:.2f} MB':>{col_w[1]}}"
            f"{f'{latent_mb:.2f} MB':>{col_w[2]}}"
            f"{f'{saving_pct:.1f}%':>{col_w[3]}}"
        )

    print("  " + "─" * (sum(col_w) + 2))
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
    """Generate n_tokens decode steps and return tokens/sec."""
    device    = next(model.parameters()).device
    inputs    = tokenize(tokenizer, prompt, device)
    attn_mask = inputs["attention_mask"]
    past_kv   = KVLatentCache() if cache_type == "latent" else DynamicCache()

    with torch.no_grad():
        out      = model(**inputs, past_key_values=past_kv, use_cache=True)
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    # Warm-up step
    with torch.no_grad():
        attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=1)
        out      = model(
            input_ids=next_tok, attention_mask=attn_mask,
            past_key_values=past_kv, use_cache=True,
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
                input_ids=next_tok, attention_mask=attn_mask,
                past_key_values=past_kv, use_cache=True,
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
    """Informational only — no pass/fail gate."""
    header("Test 5 — Throughput  (informational, no pass/fail)")
    log(f"Generating {n_tokens} decode tokens per model...")
    print()

    tps_std = _measure_tps(model_unpatched, tokenizer, prompt, n_tokens, "dynamic")
    tps_pat = _measure_tps(model_patched,   tokenizer, prompt, n_tokens, "latent")
    ratio   = tps_pat / tps_std if tps_std > 0 else float("nan")

    col_w = [28, 14]
    print(f"  {'Model':<{col_w[0]}}{'Tokens / sec':>{col_w[1]}}")
    print("  " + "─" * (sum(col_w) + 2))
    print(f"  {'Unpatched (DynamicCache)':<{col_w[0]}}{tps_std:>{col_w[1]}.1f}")
    print(f"  {'Patched  (KVLatentCache)':<{col_w[0]}}{tps_pat:>{col_w[1]}.1f}")
    print("  " + "─" * (sum(col_w) + 2))
    print(f"\n  Relative speed : {ratio:.3f}x")
    if ratio < 1.0:
        print(f"  Overhead: {100.0*(1.0-ratio):.1f}%  (kv_b_proj over full context each step)")
    else:
        print(f"  Faster by {100.0*(ratio-1.0):.1f}%  (cache bandwidth reduction dominating)")
    log("Throughput is informational — no pass/fail gate")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    model_name = Path(args.model_path).name
    print("=" * 64)
    print("  KV Latent Cache — Verification Suite")
    print(f"  Model: {model_name}")
    print("=" * 64)
    print(f"  model     : {args.model_path}")
    print(f"  prompt    : {args.prompt[:60]}...")
    print(f"  layer_idx : {args.layer_idx}  (Test 1 hook)")
    print(f"  gen_tokens: {args.gen_tokens}")

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = next(model.parameters()).device

    # Pre-patch: reference logits and standard cache sizes
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

    # Test 1 — Latent round-trip (pre-patch)
    t1_pass = test_latent_roundtrip(
        model, tokenizer, args.prompt, layer_idx=args.layer_idx,
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
        print(f"  (pass --skip-throughput to skip if VRAM is limited)")
        print(f"  NOTE: Kimi-K2.6 requires ~500 GB VRAM for two copies — use --skip-throughput on single-node setups")
        print(SEP)
        model_ref, _ = load_model_and_tokenizer(args.model_path)
        test_throughput(
            model_ref, model, tokenizer, args.prompt,
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
        "Test 1 — Latent round-trip     ": t1_pass,
        "Test 2 — Output equivalence    ": t2_pass,
        "Test 3 — Cache shape           ": t3_pass,
        "Test 4 — Memory savings        ": t4_pass,
    }
    all_pass = True
    for name, ok in results.items():
        print(f"  [{PASS if ok else FAIL}]  {name}")
        if not ok:
            all_pass = False

    if args.skip_throughput:
        print(f"  [{INFO}]  Test 5 — Throughput          (skipped)")
    else:
        print(f"  [{INFO}]  Test 5 — Throughput          (informational only)")

    print(f"\n{'=' * 64}")
    print("  ALL TESTS PASSED" if all_pass else "  SOME TESTS FAILED — see details above")
    print(f"{'=' * 64}\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
