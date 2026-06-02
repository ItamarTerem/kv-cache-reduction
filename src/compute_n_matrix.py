"""
compute_n_matrix.py — Offline computation of the KV relation matrix N.

Background
──────────────────────────────────────────────────────────────────────────────
In MLA attention, both k_nope and v are linear projections of the same
compressed latent vector (kv_a_norm) through the shared weight kv_b_proj:

    kv      = kv_a_norm  @  W_kv_b.T
    k_nope  = kv[..., :num_kv_heads * qk_nope_head_dim]
    v       = kv[..., num_kv_heads * qk_nope_head_dim:]

Since both are linear functions of the same input, there exists a static
matrix N such that:

    k_nope  =  v  @  N

N is derived from the weight matrix once, offline:

    W_K[h]  =  W_kv_b rows for head h, k_nope part   [qk_nope_head_dim, lora_rank]
    W_V[h]  =  W_kv_b rows for head h, v part         [v_head_dim,       lora_rank]

    N[h]  =  pinv(W_V[h].T)  @  W_K[h].T             [v_head_dim, qk_nope_head_dim]

Computed in FP64 for numerical precision of the pseudo-inverse.
Stored and used at runtime in BF16.

Note: only k_nope (the non-RoPE part of K) can be reconstructed this way.
k_pe (the RoPE part) is position-dependent and must still be stored in the cache.

Target model
──────────────────────────────────────────────────────────────────────────────
  DeepSeek-V2-Lite  (deepseek-ai/DeepSeek-V2-Lite)

Weight layout (blocked):
  kv_b_proj.weight rows: [All K heads | All V heads]
  i.e. [ K_h0 ... K_hN | V_h0 ... V_hN ]
"""

import torch
import torch.nn as nn


# ── Attribute helpers ────────────────────────────────────────────────────────

def _get_num_kv_heads(attn: nn.Module) -> int:
    """
    Return the number of KV heads for DeepSeek-V2-Lite.
    DeepSeek-V2 MLA decompresses keys/values to all attention heads.
    """
    if hasattr(attn, "num_heads"):
        return int(attn.num_heads)
    raise AttributeError(
        f"Cannot determine head count on {type(attn).__name__}. "
        f"Expected attribute 'num_heads'."
    )


# ── Weight extraction ─────────────────────────────────────────────────────────

def _get_kv_weight(attn: nn.Module) -> torch.Tensor:
    """
    Return the kv_b projection weight matrix from kv_b_proj.
    """
    if hasattr(attn, "kv_b_proj"):
        return attn.kv_b_proj.weight
    raise AttributeError(
        f"Cannot find 'kv_b_proj' on {type(attn).__name__}."
    )


def _split_kv_weight(
    W: torch.Tensor,
    num_kv_heads: int,
    v_head_dim: int,
    qk_nope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split the kv_b weight matrix into K and V blocks.

    DeepSeek-V2-Lite uses a flat blocked layout:
        rows 0 .. total_k_dim-1          → all K heads
        rows total_k_dim .. end          → all V heads

    Returns:
        W_K: [num_kv_heads, qk_nope_head_dim, lora_rank]
        W_V: [num_kv_heads, v_head_dim,       lora_rank]
    """
    lora_rank   = W.shape[1]
    total_k_dim = num_kv_heads * qk_nope_head_dim
    total_v_dim = num_kv_heads * v_head_dim
    expected_rows = total_k_dim + total_v_dim

    if W.shape[0] != expected_rows:
        raise ValueError(
            f"Weight shape mismatch: W.shape[0]={W.shape[0]}, "
            f"expected {expected_rows} ({num_kv_heads} heads × "
            f"({qk_nope_head_dim} + {v_head_dim}))."
        )

    W_double = W.double()
    W_K = W_double[:total_k_dim, :].view(num_kv_heads, qk_nope_head_dim, lora_rank)
    W_V = W_double[total_k_dim:,  :].view(num_kv_heads, v_head_dim,       lora_rank)
    return W_K, W_V


# ── Per-layer N computation ───────────────────────────────────────────────────

def compute_N_matrix(
    attn: nn.Module,
    out_dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute the KV relation matrix N for a single MLA attention module.

    Returns:
        N: [num_kv_heads, v_head_dim, qk_nope_head_dim]
    """
    W                = _get_kv_weight(attn)
    num_kv_heads     = _get_num_kv_heads(attn)
    v_head_dim       = attn.v_head_dim
    qk_nope_head_dim = attn.qk_nope_head_dim

    W_K, W_V = _split_kv_weight(W, num_kv_heads, v_head_dim, qk_nope_head_dim)

    N_heads = []
    for h in range(num_kv_heads):
        # pinv(W_V[h].T): [v_head_dim, lora_rank]
        # W_K[h].T:       [lora_rank, qk_nope_head_dim]
        # N_h:            [v_head_dim, qk_nope_head_dim]
        pinv_WV_T = torch.linalg.pinv(W_V[h].T)
        N_h       = pinv_WV_T @ W_K[h].T
        N_heads.append(N_h)

        if verbose:
            cond = torch.linalg.cond(W_V[h].float()).item()
            print(f"    head {h:2d}: cond(W_V)={cond:.1f}")

    N = torch.stack(N_heads, dim=0).to(dtype=out_dtype)
    return N


# ── Model-level helper ────────────────────────────────────────────────────────

def compute_all_N_matrices(
    model: nn.Module,
    out_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    verbose: bool = False,
) -> list[torch.Tensor]:
    """
    Compute N matrices for every decoder layer in the model.

    Returns:
        List of N tensors, one per layer, each [num_kv_heads, v_head_dim, qk_nope_head_dim].
    """
    if device is None:
        device = next(model.parameters()).device

    layers = model.model.layers
    print(f"Computing N matrices for {len(layers)} layers...")
    N_matrices = []

    for i, layer in enumerate(layers):
        N = compute_N_matrix(layer.self_attn, out_dtype=out_dtype, verbose=verbose)
        N_target = N.to(device=device, non_blocking=True).detach()
        N_matrices.append(N_target)

        if i == 0 or verbose:
            print(f"  Layer {i:3d}: N shape={tuple(N_target.shape)}  "
                  f"dtype={N_target.dtype}  device={N_target.device}")

    if not verbose:
        print(f"  ... ({len(N_matrices)} layers total)")

    print("Done.")
    return N_matrices