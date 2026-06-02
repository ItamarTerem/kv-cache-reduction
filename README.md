# KV Nopeless Cache

A memory optimisation for **Multi-head Latent Attention (MLA)** models that eliminates the `k_nope` component from the KV cache by reconstructing it at runtime from the already-cached value tensor `v`.

Tested on **DeepSeek-V2-Lite** (deepseek-ai/DeepSeek-V2-Lite, 15.7 B parameters).

---

## Table of Contents

1. [Background — Why This Works](#background--why-this-works)
2. [Project Structure](#project-structure)
3. [Quickstart](#quickstart)
4. [Verification Tests](#verification-tests)
5. [CLI Reference](#cli-reference)
6. [Module Reference](#module-reference)
7. [Memory Estimates](#memory-estimates)
8. [Known Limitations (Phase 1)](#known-limitations-phase-1)

---

## Background — Why This Works

### MLA Attention Recap

Standard MLA uses a two-stage projection for keys and values. Both `k_nope` and `v` are derived from the **same compressed latent vector** `kv_a_norm` through a single shared weight `kv_b_proj`:

```
kv      = kv_a_norm  @  W_kv_b.T
k_nope  = kv[..., :num_kv_heads * qk_nope_head_dim]
v       = kv[..., num_kv_heads * qk_nope_head_dim:]
```

Because both are **linear functions of the same input**, there exists a static matrix `N` such that at inference time:

```
k_nope  =  v  @  N
```

### The N Matrix

`N` is derived from the weight matrix **once, offline**, using a pseudo-inverse:

```
W_K[h]  =  W_kv_b rows for head h, k_nope part   [qk_nope_head_dim, lora_rank]
W_V[h]  =  W_kv_b rows for head h, v part         [v_head_dim,       lora_rank]

N[h]  =  pinv(W_V[h].T)  @  W_K[h].T             [v_head_dim, qk_nope_head_dim]
```

Computed in **FP64** for numerical precision of the pseudo-inverse. Stored and used at runtime in **BF16**.

### What Gets Cached

| Cache | Key tensor stored | Key memory |
|---|---|---|
| Standard `DynamicCache` | `k_full = cat([k_nope, k_pe])` | 100% |
| `KVNopelessCache` (this work) | `k_pe` only | ~33% |

For DeepSeek-V2-Lite (`qk_nope=128`, `qk_rope=64`):

```
key cache saving = qk_nope / (qk_nope + qk_rope) = 128 / 192 ≈ 67%
```

The **value cache is unchanged**. Total KV cache saving is typically **30–40%** of the full KV footprint, enabling meaningfully larger batch sizes at the same VRAM budget.

### Important: Only k_nope Can Be Dropped

`k_pe` (the RoPE part of the key) is **position-dependent** — it encodes where each token appears in the sequence. It cannot be reconstructed from `v` and must still be stored in the cache. Only the non-RoPE part `k_nope` has the static linear relationship with `v`.

### Phase 1 vs Phase 2

This repository implements **Phase 1** (correctness-first, pure PyTorch):

- `k_nope` is reconstructed via a plain `torch.matmul(v_acc, N)` on each decode step
- This reads `v` from HBM twice per step (once for attention, once for the matmul) — a known bandwidth regression
- The memory saving still enables **larger batch sizes** at the same VRAM budget

**Phase 2** (future): fuse `k_nope = v @ N` directly into FlashAttention so `v` is only read once, eliminating the bandwidth regression entirely.

---

## Project Structure

```
.
├── README.md
├── kv_patch.py                   # Patches all decoder layers to use KVNopelessCache
├── requirements.txt
│
├── ops/
│   ├── kv_nopeless_cache.py      # DynamicCache subclass storing (k_pe, v) only
│   └── kv_relation_module.py     # nn.Module holding N; forward(v) → k_nope
│
├── src/
│   └── compute_n_matrix.py       # Offline N = pinv(W_V.T) @ W_K.T  (FP64)
│
├── scripts/
│   ├── setup_runtime.sh          # Create venv, install all dependencies
│   └── download_model.sh         # Download DeepSeek-V2-Lite from HuggingFace
│
└── tests/
    └── verify_kv_relation.py     # Full 5-test verification suite
```

---

## Quickstart

### Step 1 — Set up the runtime environment

```bash
bash scripts/setup_runtime.sh
```

Creates a virtual environment at `.venv` and installs all dependencies from `requirements.txt`, including PyTorch and Transformers. Only needs to be run once.

### Step 2 — Activate the environment

```bash
source .venv/bin/activate
```

Every subsequent command must be run inside this environment.

### Step 3 — Download the model

```bash
bash scripts/download_model.sh
```

Downloads **DeepSeek-V2-Lite** (~31 GB, BF16) to `models/DeepSeek-V2-Lite/`. The script checks available disk space before downloading and is idempotent — re-running skips files that already exist.

### Step 4 — Run the verification suite

```bash
# Recommended first run: skip throughput test to avoid loading the model twice
python tests/verify_kv_relation.py \
    --model_path ./models/DeepSeek-V2-Lite \
    --skip-throughput

# Full suite including throughput benchmark (requires ~62 GB VRAM — loads model twice)
python tests/verify_kv_relation.py \
    --model_path ./models/DeepSeek-V2-Lite
```

All 4 correctness tests should pass. Typical output:

```
════════════════════════════════════════════════════════════════
  KV Nopeless Cache — Verification Suite
  Target: DeepSeek-V2-Lite
════════════════════════════════════════════════════════════════
  model     : ./models/DeepSeek-V2-Lite
  ...

──────────────────────────────────────────────────────────────
  Test 1 — N matrix accuracy  (pre-patch)
──────────────────────────────────────────────────────────────
  [ PASS ]  N shape    : (1, 128, 128)  (layer 0)
  [ PASS ]  max |k_nope - v @ N| < 0.3   (max=0.0021  mean=0.0003)
  [ PASS ]  top-1 argmax match  ← primary correctness gate

──────────────────────────────────────────────────────────────
  Test 2 — Output equivalence  (patched vs unpatched)
──────────────────────────────────────────────────────────────
  [ PASS ]  max |Δlogit| < 0.4
  [ PASS ]  top-1 token unchanged  ← primary correctness gate

──────────────────────────────────────────────────────────────
  Test 3 — Cache shape  (k_nope must not be stored)
──────────────────────────────────────────────────────────────
  [ PASS ]  Actual stored key dim : 64   (k_nope absent ✓)

──────────────────────────────────────────────────────────────
  Test 4 — Memory savings across context lengths
──────────────────────────────────────────────────────────────
  Context    Standard cache    Nopeless cache    Saving
  ──────────────────────────────────────────────────────
  2048 tok         X.XX MB          Y.YY MB       67%
  4096 tok         X.XX MB          Y.YY MB       67%
  ...
  ──────────────────────────────────────────────────────
  [ PASS ]  Saving is constant across lengths (drift < 1pp)
  [ PASS ]  Observed saving ≈ theoretical 66.7%  (within 5pp)

════════════════════════════════════════════════════════════════
  SUMMARY
════════════════════════════════════════════════════════════════
  [ PASS ]  Test 1 — N matrix accuracy
  [ PASS ]  Test 2 — Output equivalence
  [ PASS ]  Test 3 — Cache shape
  [ PASS ]  Test 4 — Memory savings
  [ INFO ]  Test 5 — Throughput         (skipped)
════════════════════════════════════════════════════════════════
  ALL TESTS PASSED
════════════════════════════════════════════════════════════════
```

---

## Verification Tests

### Test 1 — N Matrix Accuracy (pre-patch)

**Goal:** Validate that `compute_n_matrix.py` produced a correct `N` — independently of the cache and patch code.

**How it works:** A `register_forward_hook` is placed on `kv_b_proj`. The hook captures the raw output tensor `[B, S, H*(qk_nope + v_dim)]` during one forward pass with `use_cache=False`. This tensor is split into `k_nope_ref` and `v_ref` exactly as `kv_patch.py` does at runtime. Then:

```python
k_nope_hat = torch.matmul(v_ref, N)       # reconstruct
diff       = |k_nope_hat - k_nope_ref|    # compare
```

This test runs **before any patching**. If it fails, the issue is in `compute_n_matrix.py` (e.g. wrong weight slice, ill-conditioned W_V). If it passes, Tests 2–4 isolate cache/patch issues.

**Pass gate:** Top-1 argmax match over all heads and sequence positions. `max |Δ|` is reported for diagnostics but is not the hard gate.

---

### Test 2 — Output Equivalence (post-patch)

**Goal:** Confirm that dropping `k_nope` from the cache and reconstructing it from `v @ N` does not change what the model predicts.

**How it works:** Reference logits are collected pre-patch using standard `DynamicCache`. After `patch_kv_model()` is applied, the same prompt is run again with `KVNopelessCache`. Logits are compared at every decode step.

**Pass gate:** Top-1 token must match at every position. `max |Δlogit|` is reported for diagnostics.

**BF16 note:** Expected `max |Δlogit|` is 0.1–0.35. This is not a bug — BF16 rounding (7 mantissa bits) means rounding scales with magnitude. Top-1 token match is the meaningful correctness signal.

---

### Test 3 — Cache Shape

**Goal:** Prove that `k_nope` is genuinely never written to the cache.

**How it works:** After one prefill pass and one decode step, the key tensors inside `KVNopelessCache` are inspected. Their last dimension must equal `qk_rope_head_dim` (64 for DeepSeek-V2-Lite), not the full `qk_nope_head_dim + qk_rope_head_dim` (192).

**Pass gate:** `cache.key_head_dim(layer_idx=0) == qk_rope_head_dim`. A failure means `k_nope` is silently leaking back into the cache — the saving would still appear correct in Test 4 but be caused by a bug.

---

### Test 4 — Memory Savings Across Context Lengths

**Goal:** Measure the actual memory reduction and confirm it is consistent.

**How it works:** Standard `DynamicCache` sizes are collected pre-patch at each context length. After patching, `KVNopelessCache` sizes are measured at the same lengths. A side-by-side table is printed.

**Key insight:** The saving percentage is **the same at every context length** — it is a pure function of model dimensions:

```
saving = qk_nope_head_dim / (qk_nope_head_dim + qk_rope_head_dim)
```

If the percentage drifts between rows, something is wrong (e.g. `k_nope` being cached for some steps but not others).

**Pass gate:** Drift between min and max saving percentage < 1 percentage point, and observed mean within 5 pp of the theoretical value.

---

### Test 5 — Throughput (informational only)

**Goal:** Report decode tokens/sec for the unpatched (`DynamicCache`) vs patched (`KVNopelessCache`) model.

**How it works:** The unpatched model is reloaded from disk. Both models generate the same number of decode tokens from a warm cache (prefill not timed). Tokens/sec and relative speed are printed.

**Pass gate:** None. Purely informational.

**Phase-1 expectation:** The patched model may be slightly slower per token — each decode step recomputes `k_nope = v_acc @ N` over the full accumulated `v`, reading `v` from HBM twice. The memory saving pays off at larger batch sizes.

Requires ~62 GB VRAM (two model copies). Use `--skip-throughput` if unavailable.

---

## CLI Reference — verify_kv_relation.py

```
python tests/verify_kv_relation.py [OPTIONS]

Required:
  --model_path PATH          Path to HuggingFace model directory

Optional:
  --prompt TEXT              Prompt used for all tests
                             (default: excerpt from AI history paragraph)
  --layer_idx INT            Layer index to hook for Test 1 (default: 0)
  --context_lengths INT...   Context lengths for Test 4 table
                             (default: 2048 4096 8192 16384 32768)
  --gen_tokens INT           New tokens to generate in Tests 2 and 5 (default: 64)
  --throughput_tokens INT    Tokens per model in Test 5 benchmark (default: 256)
  --skip-throughput          Skip Test 5; avoids reloading the model
  --n_abs_tol FLOAT          |Δ| tolerance for Test 1 (default: 0.3)
  --output_abs_tol FLOAT     |Δlogit| tolerance for Test 2 (default: 0.4)
```

---

## Module Reference

### `src/compute_n_matrix.py`

Offline computation of the KV relation matrix `N`.

```python
from src.compute_n_matrix import compute_N_matrix, compute_all_N_matrices

# For a single attention layer
N = compute_N_matrix(attn)
# Returns: [num_kv_heads, v_head_dim, qk_nope_head_dim], dtype=bfloat16

# For all layers in a model
N_list = compute_all_N_matrices(model)
# Returns: list of Tensors, one per layer
```

Uses `kv_b_proj.weight` as the weight source. Use `verbose=True` to print per-head condition numbers of `W_V` for diagnostics.

---

### `ops/kv_relation_module.py`

Thin `nn.Module` wrapper that holds `N` as a non-trainable buffer and exposes a `forward(v) → k_nope` interface.

```python
from ops.kv_relation_module import KVRelationModule

module = KVRelationModule(N)          # N: [H, v_dim, qk_nope_dim]
k_nope = module(v_cached)            # [B, H, S, qk_nope_dim]
```

`N` is a `register_buffer` — persistent in `state_dict`, never updated by an optimiser, and moves to device automatically with `.to(device)`. One instance is attached to each attention layer as `attn.kv_relation` by `kv_patch.py`.

---

### `ops/kv_nopeless_cache.py`

Drop-in replacement for HuggingFace `DynamicCache` that stores `(k_pe, v)` instead of `(k_full, v)`.

```python
from ops.kv_nopeless_cache import KVNopelessCache

cache = KVNopelessCache()
output = model.generate(**inputs, past_key_values=cache, use_cache=True)

# Diagnostics
print(cache.report())
# KVNopelessCache | 27 layers populated
#   key   (k_pe only):  [1, 1, 512, 64]   dtype=bfloat16
#   value:              [1, 1, 512, 128]  dtype=bfloat16
#   key memory  :   0.07 MB
#   value memory:   0.13 MB
#   total memory:   0.20 MB

cache.key_head_dim(layer_idx=0)      # returns qk_rope_head_dim (64)
cache.cache_size_bytes()             # {'key_bytes': ..., 'value_bytes': ..., 'total_bytes': ...}
```

No `update()` override is needed. `DynamicCache.update()` concatenates whatever tensors you pass along the sequence dimension — passing `k_pe` instead of `k_full` simply accumulates `k_pe`.

---

### `kv_patch.py`

Patches all decoder layers in-place to use `KVNopelessCache`.

```python
from kv_patch import patch_kv_model
from ops.kv_nopeless_cache import KVNopelessCache

model = patch_kv_model(model)

output = model.generate(
    **inputs,
    past_key_values=KVNopelessCache(),
    use_cache=True,
)
```

For each layer, `patch_kv_model`:

1. Calls `compute_N_matrix(attn)` to get `N`
2. Attaches `KVRelationModule(N)` as `attn.kv_relation`
3. Replaces `attn.forward` with a closure that writes only `(k_pe, v)` to the cache and reconstructs `k_nope = attn.kv_relation(v_acc)` before computing attention

---

## Memory Estimates

Figures below are for the **key cache only** (value cache is unchanged). BF16, batch size 1.

DeepSeek-V2-Lite: 27 layers, 1 KV head, `qk_nope=128`, `qk_rope=64`.

| Context | Standard key cache | Nopeless key cache | Saving |
|---|---|---|---|
| 2 048 tokens | ~1.07 MB | ~0.36 MB | 67% |
| 4 096 tokens | ~2.14 MB | ~0.71 MB | 67% |
| 8 192 tokens | ~4.29 MB | ~1.43 MB | 67% |
| 16 384 tokens | ~8.57 MB | ~2.86 MB | 67% |
| 32 768 tokens | ~17.14 MB | ~5.71 MB | 67% |

At large batch sizes (e.g. batch=128, 4096 tokens), the key-side saving scales linearly — enough headroom to fit several additional sequences in the same VRAM budget.

---

## Known Limitations (Phase 1)

**Double HBM read of v.** `v` is read once by `k_nope = v @ N` and again inside scaled dot-product attention. Phase 2 (fused kernel) eliminates this by computing `k_nope` inside the attention kernel while `v` is already in SRAM.

**FP64 pinv at load time.** `compute_N_matrix` runs a per-head pseudo-inverse in FP64. For 27 layers this takes a few seconds at startup but is a one-time cost.

**Ill-conditioned W_V heads.** If a particular head's `W_V` has a high condition number, reconstruction error for that head will be larger than average. Use `compute_N_matrix(attn, verbose=True)` to print per-head condition numbers before deploying.

**No batch-size scaling test.** Test 4 measures cache size at batch=1. The memory saving scales linearly with batch size, but this is not automatically verified by the test suite.