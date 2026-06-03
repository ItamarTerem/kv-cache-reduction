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

    DeepSeek-V2-Lite uses an interleaved layout — matching its forward pass:

        kv = kv_b_proj(compressed_kv)
        kv = kv.view(B, S, num_heads, qk_nope_head_dim + v_head_dim)
        k_nope, v = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)

    Weight rows are therefore ordered:
        [ K_h0 | V_h0 | K_h1 | V_h1 | ... | K_hN | V_hN ]

    Returns:
        W_K: [num_kv_heads, qk_nope_head_dim, lora_rank]
        W_V: [num_kv_heads, v_head_dim,       lora_rank]
    """
    lora_rank     = W.shape[1]
    per_head_dim  = qk_nope_head_dim + v_head_dim
    expected_rows = num_kv_heads * per_head_dim

    if W.shape[0] != expected_rows:
        raise ValueError(
            f"Weight shape mismatch: W.shape[0]={W.shape[0]}, "
            f"expected {expected_rows} ({num_kv_heads} heads × "
            f"({qk_nope_head_dim} + {v_head_dim}))."
        )

    W_double = W.double().view(num_kv_heads, per_head_dim, lora_rank)
    W_K = W_double[:, :qk_nope_head_dim, :]   # [H, qk_nope, lora_rank]
    W_V = W_double[:, qk_nope_head_dim:,  :]  # [H, v_dim,   lora_rank]
    return W_K, W_V


# ── Per-layer N computation ───────────────────────────────────────────────────

def compute_N_matrix(
    attn: nn.Module,
    out_dtype: torch.dtype = torch.float32,
    svd_threshold: float = 1e-5,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute the KV relation matrix N for a single MLA attention module.

    Uses truncated SVD to compute the pseudo-inverse of W_V, discarding
    singular values below svd_threshold * max(s). This avoids dividing by
    near-zero singular values that would amplify noise in ill-conditioned heads.

    Args:
        svd_threshold: relative cutoff — singular values smaller than
                       svd_threshold * s.max() are treated as zero.
                       Default 1e-5 keeps all numerically meaningful components.

    Returns:
        N: [num_kv_heads, v_head_dim, qk_nope_head_dim], dtype=out_dtype
    """
    W                = _get_kv_weight(attn)
    num_kv_heads     = _get_num_kv_heads(attn)
    v_head_dim       = attn.v_head_dim
    qk_nope_head_dim = attn.qk_nope_head_dim

    W_K, W_V = _split_kv_weight(W, num_kv_heads, v_head_dim, qk_nope_head_dim)

    N_heads = []
    for h in range(num_kv_heads):
        # W_V[h]:   [v_head_dim,       lora_rank]  →  W_V[h].T: [lora_rank, v_head_dim]
        # W_K[h]:   [qk_nope_head_dim, lora_rank]  →  W_K[h].T: [lora_rank, qk_nope_head_dim]
        #
        # SVD of W_V[h]  (fat matrix [v_dim, lora_rank], full_matrices=False):
        #   U:  [v_dim, v_dim]
        #   s:  [v_dim]
        #   Vh: [v_dim, lora_rank]
        #
        # pinv(W_V[h]) = Vh.T @ diag(1/s) @ U.T   shape [lora_rank, v_dim]
        # pinv(W_V[h].T) = U @ diag(1/s) @ Vh      shape [v_dim, lora_rank]
        #
        # N_h = pinv(W_V[h].T) @ W_K[h].T           shape [v_dim, qk_nope]

        U, s, Vh = torch.linalg.svd(W_V[h], full_matrices=False)
        # U:  [v_dim, v_dim],  s: [v_dim],  Vh: [v_dim, lora_rank]

        s_inv = torch.where(
            s > svd_threshold * s.max(),
            1.0 / s,
            torch.zeros_like(s),
        )

        # pinv(W_V[h].T) = U @ diag(s_inv) @ Vh
        pinv_WV_T = U @ torch.diag(s_inv) @ Vh        # [v_dim, lora_rank]
        N_h       = pinv_WV_T @ W_K[h].T              # [v_dim, qk_nope]
        N_heads.append(N_h)

        if verbose:
            n_kept = (s > svd_threshold * s.max()).sum().item()
            print(f"    head {h:2d}: max_s={s.max():.3f}  min_s={s.min():.3f}"
                  f"  kept={n_kept}/{len(s)}  cond={s.max()/s[s > 0].min():.1f}")

    N = torch.stack(N_heads, dim=0).to(dtype=out_dtype)
    return N


# ── Model-level helper ────────────────────────────────────────────────────────

def compute_all_N_matrices(
    model: nn.Module,
    out_dtype: torch.dtype = torch.float32,
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
