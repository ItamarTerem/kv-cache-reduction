"""
kv_patch.py — Patch MLA attention to use KVLatentCache.

Target model: DeepSeek-V2-Lite (deepseek-ai/DeepSeek-V2-Lite)

What this file does
──────────────────────────────────────────────────────────────────────────────
For each decoder layer, replaces attn.forward so that:
  • (kv_a_norm, k_pe_roped) are written to KVLatentCache — not (k_full, v)
  • At each decode step, k_nope and v are recomputed exactly from the
    accumulated kv_a_norm via kv_b_proj
  • RoPE is applied explicitly to q_pe and k_pe before caching

Cache layout
──────────────────────────────────────────────────────────────────────────────
  key_cache[layer]   = kv_a_norm   [B, 1, S, kv_lora_rank=512]
  value_cache[layer] = k_pe_roped  [B, 1, S, qk_rope_head_dim=64]

Reconstruction at attention time (exact, not approximate)
──────────────────────────────────────────────────────────────────────────────
  kv_a_norm_acc  →  kv_b_proj  →  [B, S, 16 × 256]
                 →  reshape    →  [B, 16, S, 256]
                 →  split      →  k_nope_acc [B, 16, S, 128]
                                  v_acc      [B, 16, S, 128]
  k_pe_acc.expand(16 heads)   →  [B, 16, S, 64]
  k = cat([k_nope_acc, k_pe_acc_exp], dim=-1)  →  [B, 16, S, 192]

Usage
──────────────────────────────────────────────────────────────────────────────
    from kv_patch import patch_kv_model
    from ops.kv_latent_cache import KVLatentCache

    model = patch_kv_model(model)
    output = model.generate(
        **inputs,
        past_key_values=KVLatentCache(),
        use_cache=True,
    )
"""

import math
import importlib
import torch
import torch.nn as nn

from src.compute_n_matrix import compute_N_matrix  # kept for verify_kv_relation Test 1 compat
from ops.kv_relation_module import KVRelationModule  # kept for verify_kv_relation Test 1 compat


# ── Model-difference helpers (resolved once at patch time) ───────────────────

def _resolve_rope_fn(attn: nn.Module):
    """Return the module-level apply_rotary_pos_emb for DeepSeek-V2-Lite."""
    mod = importlib.import_module(type(attn).__module__)
    fn  = getattr(mod, "apply_rotary_pos_emb", None)
    if fn is None:
        raise AttributeError(
            f"Could not find apply_rotary_pos_emb in {type(attn).__module__}."
        )
    return fn


def _resolve_scale(attn: nn.Module) -> float:
    """
    Return the softmax scale for this attention module.
    DeepSeek-V2-Lite stores it as attn.softmax_scale (a custom value,
    NOT simply 1/sqrt(q_head_dim)). Fall back to computing if absent.
    """
    if hasattr(attn, "softmax_scale"):
        return float(attn.softmax_scale)
    if hasattr(attn, "scaling"):
        return float(attn.scaling)
    if hasattr(attn, "q_head_dim"):
        return 1.0 / math.sqrt(attn.q_head_dim)
    return 1.0 / math.sqrt(attn.qk_nope_head_dim + attn.qk_rope_head_dim)


def _resolve_q_head_dim(attn: nn.Module) -> int:
    """Full Q head dimension (nope + rope)."""
    if hasattr(attn, "q_head_dim"):
        return int(attn.q_head_dim)
    return attn.q_b_proj.weight.shape[0] // attn.num_heads


# ── Projection helpers ────────────────────────────────────────────────────────

def _project_q(attn: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Q projection for DeepSeek-V2-Lite.
    Some layers use two-stage (q_a_proj → q_a_layernorm → q_b_proj);
    others use a single q_proj.
    """
    if hasattr(attn, "q_a_proj"):
        q_a = attn.q_a_proj(hidden_states)
        q_a = attn.q_a_layernorm(q_a)
        return attn.q_b_proj(q_a)
    return attn.q_proj(hidden_states)


def _extract_kv_latent(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract kv_a_norm (the shared KV latent) and k_pe — stopping BEFORE kv_b_proj.

    Returns:
        kv_a_norm : [B, S, kv_lora_rank]  — shared latent for all KV heads
        k_pe      : [B, S, qk_rope_head_dim]  — RoPE key component (pre-RoPE)
    """
    kv_a = attn.kv_a_proj_with_mqa(hidden_states)
    kv_a_out, k_pe = torch.split(
        kv_a,
        [attn.kv_lora_rank, qk_rope_head_dim],
        dim=-1,
    )
    kv_a_norm = attn.kv_a_layernorm(kv_a_out)
    return kv_a_norm, k_pe


# ── Per-layer patching ────────────────────────────────────────────────────────

def _patch_attention_forward(attn: nn.Module, debug: bool = False) -> None:
    """
    Replace attn.forward with a latent-cache forward for DeepSeek-V2-Lite.
    All model-specific values are resolved once here and captured in the closure.
    """
    _apply_rope  = _resolve_rope_fn(attn)
    _scale       = _resolve_scale(attn)
    _q_head_dim  = _resolve_q_head_dim(attn)
    _num_heads   = int(attn.num_heads)

    if debug:
        print(f"  [patch debug] layer {attn.layer_idx}:")
        print(f"    num_heads={_num_heads}  q_head_dim={_q_head_dim}  scale={_scale:.6f}")
        print(f"    qk_nope={attn.qk_nope_head_dim}  qk_rope={attn.qk_rope_head_dim}  v_head_dim={attn.v_head_dim}")
        print(f"    kv_lora_rank={attn.kv_lora_rank}")
        # Check if model stores its own scale
        for attr in ("softmax_scale", "scaling", "scale"):
            if hasattr(attn, attr):
                print(f"    attn.{attr} = {getattr(attn, attr)}")

    def patched_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,       # DeepSeek-V2-Lite uses singular form
        past_key_values=None,      # accepted for compatibility
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        # Normalise cache argument — DeepSeek passes `past_key_value` (singular)
        cache = past_key_value if past_key_value is not None else past_key_values

        bsz, q_len, _ = hidden_states.shape
        qk_rope_head_dim = attn.qk_rope_head_dim
        qk_nope_head_dim = attn.qk_nope_head_dim
        v_head_dim       = attn.v_head_dim

        # ── Q projection ──────────────────────────────────────────────────────
        q = _project_q(attn, hidden_states)
        q = q.view(bsz, q_len, _num_heads, _q_head_dim).transpose(1, 2)
        # Split into non-RoPE and RoPE parts
        q_nope, q_pe = torch.split(
            q,
            [_q_head_dim - qk_rope_head_dim, qk_rope_head_dim],
            dim=-1,
        )

        # ── KV latent extraction (stop before kv_b_proj) ──────────────────────
        kv_a_norm, k_pe = _extract_kv_latent(attn, hidden_states, qk_rope_head_dim)
        # kv_a_norm: [B, S, lora_rank]  →  add "1 head" dim for cache storage
        kv_a_norm_1h = kv_a_norm.unsqueeze(1)          # [B, 1, S, lora_rank]
        k_pe_1h = k_pe.unsqueeze(1)                    # [B, 1, S, qk_rope]

        # ── RoPE ──────────────────────────────────────────────────────────────
        kv_seq_len = q_len
        if cache is not None:
            kv_seq_len += cache.get_usable_length(q_len, attn.layer_idx)

        cos, sin = attn.rotary_emb(q_pe, seq_len=kv_seq_len)

        # Apply RoPE to q_pe and k_pe separately so k_pe stays contiguous
        # with 1 head — matching the original forward's tensor layout exactly.
        # Passing k_pe expanded to 16 heads produces different BF16 rounding
        # in rotate_half due to non-contiguous memory layout.
        q_pe_roped, k_pe_roped_1h = _apply_rope(
            q_pe,
            k_pe_1h,          # [B, 1, S, qk_rope] — 1 head, contiguous
            cos, sin, position_ids,
        )

        # Reassemble full Q with RoPE applied
        q = torch.cat([q_nope, q_pe_roped], dim=-1)    # [B, H, S_q, q_head_dim]

        # ── Cache (latent store) ───────────────────────────────────────────────
        if cache is not None:
            # Store (kv_a_norm, k_pe_roped) — DynamicCache concatenates along S dim
            kv_a_norm_acc, k_pe_acc = cache.update(
                kv_a_norm_1h,
                k_pe_roped_1h,
                attn.layer_idx,
            )
            # kv_a_norm_acc: [B, 1, S_total, lora_rank]
            # k_pe_acc:      [B, 1, S_total, qk_rope]

            # Reconstruct k_nope and v EXACTLY from accumulated latent
            kv_a_acc_2d = kv_a_norm_acc[:, 0, :, :]   # [B, S_total, lora_rank]
            kv_acc = attn.kv_b_proj(kv_a_acc_2d)       # [B, S_total, H*(nope+v)]
            kv_acc = kv_acc.view(
                bsz, -1, _num_heads, qk_nope_head_dim + v_head_dim
            ).transpose(1, 2)                           # [B, H, S_total, nope+v]
            k_nope_acc, v_acc = torch.split(
                kv_acc, [qk_nope_head_dim, v_head_dim], dim=-1
            )

            k_pe_acc_exp = k_pe_acc.expand(-1, _num_heads, -1, -1)
            k          = torch.cat([k_nope_acc, k_pe_acc_exp], dim=-1)
            v_for_attn = v_acc

        else:
            # No cache: compute k and v directly from current tokens
            kv_cur = attn.kv_b_proj(kv_a_norm)         # [B, S, H*(nope+v)]
            kv_cur = kv_cur.view(
                bsz, q_len, _num_heads, qk_nope_head_dim + v_head_dim
            ).transpose(1, 2)                           # [B, H, S, nope+v]
            k_nope_cur, v_cur = torch.split(
                kv_cur, [qk_nope_head_dim, v_head_dim], dim=-1
            )
            k_pe_cur_exp = k_pe_roped_1h.expand(-1, _num_heads, -1, -1)
            k          = torch.cat([k_nope_cur, k_pe_cur_exp], dim=-1)
            v_for_attn = v_cur

        # ── Attention ─────────────────────────────────────────────────────────
        # Replicate the original DeepSeek-V2-Lite forward exactly:
        # manual matmul + float32 softmax cast back to input dtype.
        # Using F.scaled_dot_product_attention gives different BF16 numerics
        # due to its fused kernel using different internal precision.
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * _scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        if attn.training and attn.attention_dropout > 0.0:
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=attn.attention_dropout
            )
        attn_output = torch.matmul(attn_weights, v_for_attn)

        attn_output = (
            attn_output.transpose(1, 2)
            .reshape(bsz, q_len, -1)
            .contiguous()
        )
        attn_output = attn.o_proj(attn_output)

        # DeepSeek expects (attn_output, attn_weights, present_key_value)
        return attn_output, None, cache

    attn.forward = patched_forward


# ── Public API ────────────────────────────────────────────────────────────────

def patch_kv_model(
    model: nn.Module,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Patch all decoder layers in a DeepSeek-V2-Lite model to use KVLatentCache.

    For each layer, replaces attn.forward with a closure that:
      1. Extracts kv_a_norm (before kv_b_proj) and k_pe
      2. Applies RoPE to q_pe and k_pe
      3. Caches (kv_a_norm, k_pe_roped) — not (k_pe, v)
      4. Reconstructs k_nope and v exactly via kv_b_proj at attention time

    Args:
        model:  DeepSeek-V2-Lite HuggingFace causal LM
        device: target device (defaults to model's current device)

    Returns:
        The patched model (modified in-place).

    Example:
        model = patch_kv_model(model)
        output = model.generate(
            **inputs,
            past_key_values=KVLatentCache(),
            use_cache=True,
        )
    """
    if device is None:
        device = next(model.parameters()).device

    n_layers = len(model.model.layers)
    print(f"Patching {n_layers} layers for KV latent cache...")

    for i, layer in enumerate(model.model.layers):
        _patch_attention_forward(layer.self_attn, debug=(i == 0))
        if i == 0:
            print(f"  Layer 0: patched (kv_a_norm + k_pe_roped caching)")

    print(f"Done. {n_layers} layers patched.")
    print()
    print("Usage:")
    print("  from ops.kv_latent_cache import KVLatentCache")
    print("  output = model.generate(**inputs, past_key_values=KVLatentCache(), use_cache=True)")

    return model
