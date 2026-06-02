"""
kv_patch.py — Patch MLA attention to use KVNopelessCache.

Target model: DeepSeek-V2-Lite (deepseek-ai/DeepSeek-V2-Lite)

What this file does
──────────────────────────────────────────────────────────────────────────────
For each decoder layer:
  1. Computes the KV relation matrix N  (via compute_n_matrix.py)
  2. Wraps N in a KVRelationModule and attaches it to the attention module
  3. Replaces the attention forward so that:
       • only (k_pe, v) are written to the cache  — not k_full
       • k_nope is reconstructed from cached v via  k_nope = attn.kv_relation(v)
       • everything else (RoPE, SDPA, o_proj) is unchanged

Usage
──────────────────────────────────────────────────────────────────────────────
    from kv_patch import patch_kv_model
    from kv_nopeless_cache import KVNopelessCache

    model = patch_kv_model(model)

    output = model.generate(
        **inputs,
        past_key_values=KVNopelessCache(),
        use_cache=True,
    )
"""

import math
import importlib
import torch
import torch.nn as nn

from src.compute_n_matrix import compute_N_matrix
from ops.kv_relation_module import KVRelationModule


# ── Model-difference helpers (resolved once at patch time) ───────────────────

def _resolve_rope_fn(attn: nn.Module):
    """
    Return the module-level apply_rotary_pos_emb function for DeepSeek-V2-Lite.
    It lives in the same module as the attention class.
    """
    mod = importlib.import_module(type(attn).__module__)
    fn  = getattr(mod, "apply_rotary_pos_emb", None)
    if fn is None:
        raise AttributeError(
            f"Could not find apply_rotary_pos_emb in module "
            f"{type(attn).__module__}."
        )
    return fn


def _resolve_scale(attn: nn.Module) -> float:
    """
    Return the softmax scale.
    DeepSeek-V2-Lite stores the full Q head dim as attn.q_head_dim.
    Falls back to computing from nope + rope dims.
    """
    if hasattr(attn, "q_head_dim"):
        return 1.0 / math.sqrt(attn.q_head_dim)
    return 1.0 / math.sqrt(attn.qk_nope_head_dim + attn.qk_rope_head_dim)


def _resolve_q_head_dim(attn: nn.Module) -> int:
    """
    Return the full Q head dimension (nope + rope).
    DeepSeek-V2-Lite stores this as attn.q_head_dim.
    """
    if hasattr(attn, "q_head_dim"):
        return int(attn.q_head_dim)
    # Fallback: derive from weight shape
    return attn.q_b_proj.weight.shape[0] // attn.num_heads


# ── Projection helpers ────────────────────────────────────────────────────────

def _project_q(attn: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Two-stage Q projection for DeepSeek-V2-Lite:
        hidden → q_a_proj → q_a_layernorm → q_b_proj
    """
    q_a = attn.q_a_proj(hidden_states)
    q_a = attn.q_a_layernorm(q_a)
    return attn.q_b_proj(q_a)


def _project_kv(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    KV projection for DeepSeek-V2-Lite:
        hidden → kv_a_proj_with_mqa → split (kv_a, k_pe) → kv_a_layernorm → kv_b_proj

    Returns:
        kv   : projected KV tensor (k_nope + v interleaved per head)
        k_pe : rope component, shape [B, S, 1, qk_rope_head_dim] pre-transpose
    """
    kv_a = attn.kv_a_proj_with_mqa(hidden_states)
    kv_a_out, k_pe = torch.split(
        kv_a,
        [attn.kv_lora_rank, qk_rope_head_dim],
        dim=-1,
    )
    kv_a_out = attn.kv_a_layernorm(kv_a_out)
    kv = attn.kv_b_proj(kv_a_out)
    return kv, k_pe


# ── Per-layer patching ────────────────────────────────────────────────────────

def _attach_kv_relation(attn: nn.Module, device: torch.device) -> None:
    """
    Compute N for this layer and attach a KVRelationModule to attn.kv_relation.
    """
    N = compute_N_matrix(attn)
    attn.kv_relation = KVRelationModule(N).to(device)


def _patch_attention_forward(attn: nn.Module) -> None:
    """
    Replace attn.forward with a nopeless-cache forward for DeepSeek-V2-Lite.
    All model-specific values are resolved once here and captured in the closure.
    """
    _apply_rope  = _resolve_rope_fn(attn)
    _scale       = _resolve_scale(attn)
    _q_head_dim  = _resolve_q_head_dim(attn)
    _num_heads   = int(attn.num_heads)

    def patched_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape
        qk_rope_head_dim = attn.qk_rope_head_dim

        # ── Q projection ──────────────────────────────────────────────────────
        q = _project_q(attn, hidden_states)
        q = q.view(bsz, q_len, _num_heads, _q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q,
            [_q_head_dim - qk_rope_head_dim, qk_rope_head_dim],
            dim=-1,
        )

        # ── KV projection ─────────────────────────────────────────────────────
        kv, k_pe = _project_kv(attn, hidden_states, qk_rope_head_dim)
        k_pe = k_pe.view(bsz, q_len, 1, qk_rope_head_dim).transpose(1, 2)

        qk_nope_head_dim = attn.kv_relation.qk_nope_head_dim
        v_head_dim       = attn.v_head_dim

        kv = kv.view(bsz, q_len, _num_heads, qk_nope_head_dim + v_head_dim)
        k_nope, v = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)
        v      = v.transpose(1, 2)

        # ── RoPE ──────────────────────────────────────────────────────────────
        kv_seq_len = q_len
        if past_key_values is not None:
            kv_seq_len += past_key_values.get_usable_length(q_len, attn.layer_idx)

        # DeepSeek-V2-Lite rotary_emb expects a dummy tensor + seq_len kwarg
        cos, sin = attn.rotary_emb(q_pe, seq_len=kv_seq_len)

        # ── Cache (KVNopeless core) ───────────────────────────────────────────
        if past_key_values is not None:
            k_pe_acc, v_acc = past_key_values.update(
                k_pe,
                v,
                attn.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )
            k_nope_acc = attn.kv_relation(v_acc)
            k_pe_acc   = k_pe_acc.expand(-1, _num_heads, -1, -1)
            k          = torch.cat([k_nope_acc, k_pe_acc], dim=-1)
            v_for_attn = v_acc
        else:
            k_pe_exp   = k_pe.expand(-1, _num_heads, -1, -1)
            k          = torch.cat([k_nope, k_pe_exp], dim=-1)
            v_for_attn = v

        # ── Attention ─────────────────────────────────────────────────────────
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v_for_attn,
            attn_mask=attention_mask,
            dropout_p=attn.attention_dropout if attn.training else 0.0,
            scale=_scale,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .reshape(bsz, q_len, -1)
            .contiguous()
        )
        attn_output = attn.o_proj(attn_output)

        # HF expects (attn_output, attn_weights, past_key_values)
        return attn_output, None, past_key_values

    attn.forward = patched_forward


# ── Public API ────────────────────────────────────────────────────────────────

def patch_kv_model(
    model: nn.Module,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Patch all decoder layers in a DeepSeek-V2-Lite model to use KVNopelessCache.

    For each layer:
      1. Computes N = pinv(W_V.T) @ W_K.T  (from kv_b_proj)
      2. Attaches KVRelationModule(N) as attn.kv_relation
      3. Replaces attn.forward with the nopeless-cache forward

    Args:
        model:  DeepSeek-V2-Lite HuggingFace causal LM
        device: target device (defaults to model's current device)

    Returns:
        The patched model (modified in-place).

    Example:
        model = patch_kv_model(model)
        output = model.generate(
            **inputs,
            past_key_values=KVNopelessCache(),
            use_cache=True,
        )
    """
    if device is None:
        device = next(model.parameters()).device

    n_layers = len(model.model.layers)
    print(f"Patching {n_layers} layers for KV nopeless cache...")

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        _attach_kv_relation(attn, device)
        _patch_attention_forward(attn)
        if i == 0:
            print(f"  Layer 0: {attn.kv_relation}")

    print(f"Done. {n_layers} layers patched.")
    print()
    print("Usage:")
    print("  from kv_nopeless_cache import KVNopelessCache")
    print("  output = model.generate(**inputs, past_key_values=KVNopelessCache(), use_cache=True)")

    return model