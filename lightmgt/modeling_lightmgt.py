"""LightMGT Transformer Model.

Lightweight Masked Generative Transformer with:
- 4 Double-stream blocks (softmax, text-image joint attention)
- 14 Single-stream blocks (GLA, gated linear attention)
- 6 Single-stream blocks (softmax, sharp discriminative attention)
- LFQ Embedding (18-bit → 1024D projection, ~19K params)
- Factorized Gen Head (2×512 for 262144 LFQ codebook)
- 3D Unified RoPE (Z-Image style)
- Shared AdaLN, SwiGLU MLP, Sandwich Norm, Parallel Blocks, QK-Norm
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging

from lightmgt.configuration_lightmgt import LightMGTConfig
from lightmgt.modeling_gla import GLAAttention, build_gla_attention
from lightmgt.modeling_rope import (
    LightMGTRoPE3D,
    apply_rotary_emb,
    build_position_ids,
    get_3d_rotary_embedding,
)

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * norm).to(dtype) * self.weight


class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network (FLUX.2 style).

    Effective expansion ratio ≈ 2.67x (8/3).
    Uses gate * swish(gate_proj(x)) * up_proj(x) pattern.
    """

    def __init__(self, hidden_size: int, intermediate_size: int = None,
                 mlp_ratio: float = 2.6875, bias: bool = False):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * mlp_ratio)
            # Round to nearest multiple of 256 for efficiency
            intermediate_size = ((intermediate_size + 255) // 256) * 256
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SharedAdaLN(nn.Module):
    """Shared Adaptive Layer Normalization (FLUX.2 style).

    Produces scale, shift, and gate parameters from timestep + pooled text embeddings.
    Shared across all layers to save parameters.

    Outputs 6 modulation vectors for double blocks (2 norms x 3 params each)
    and 3 modulation vectors for single blocks (1 norm x 3 params).
    """

    def __init__(self, hidden_size: int, cond_dim: int = None):
        super().__init__()
        if cond_dim is None:
            cond_dim = hidden_size
        self.silu = nn.SiLU()
        # 6 vectors for double block: (shift1, scale1, gate1, shift2, scale2, gate2)
        # These are for the image stream
        self.double_proj = nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        # 6 vectors for the text stream in double blocks
        self.double_text_proj = nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        # 3 vectors for single block: (shift, scale, gate)
        self.single_proj = nn.Linear(cond_dim, 3 * hidden_size, bias=True)

        # Zero-init weights, gate bias set to small positive value.
        # Actual values are applied in LightMGTTransformer.__init__ after
        # _init_weights runs, so this is just the default structure.
        nn.init.zeros_(self.double_proj.weight)
        nn.init.zeros_(self.double_text_proj.weight)
        nn.init.zeros_(self.single_proj.weight)
        nn.init.zeros_(self.double_proj.bias)
        nn.init.zeros_(self.double_text_proj.bias)
        nn.init.zeros_(self.single_proj.bias)

    def forward_double(self, cond: torch.Tensor):
        """Get modulation params for double block.

        Returns:
            img_mods: (shift1, scale1, gate1, shift2, scale2, gate2) for image
            txt_mods: (shift1, scale1, gate1, shift2, scale2, gate2) for text
        """
        img_mods = self.double_proj(self.silu(cond)).chunk(6, dim=-1)
        txt_mods = self.double_text_proj(self.silu(cond)).chunk(6, dim=-1)
        return img_mods, txt_mods

    def forward_single(self, cond: torch.Tensor):
        """Get modulation params for single block.

        Returns:
            mods: (shift, scale, gate) tuple
        """
        return self.single_proj(self.silu(cond)).chunk(3, dim=-1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + MLP projection."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_dim = freq_dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep: [B] float tensor (0-1000 range).
        Returns:
            [B, hidden_size] embedding.
        """
        half_dim = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=timestep.device, dtype=torch.float32) / half_dim
        )
        args = timestep.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # Cast to model dtype for mixed-precision compatibility
        emb = emb.to(next(self.mlp.parameters()).dtype)
        return self.mlp(emb)


class ConditionEmbedding(nn.Module):
    """Combine timestep + pooled text embeddings into conditioning vector."""

    def __init__(self, hidden_size: int, text_pooled_dim: int, freq_dim: int = 256):
        super().__init__()
        self.timestep_embed = TimestepEmbedding(hidden_size, freq_dim)
        self.text_proj = nn.Linear(text_pooled_dim, hidden_size, bias=True)

    def forward(self, timestep: torch.Tensor, pooled_text: torch.Tensor) -> torch.Tensor:
        t_emb = self.timestep_embed(timestep)
        txt_emb = self.text_proj(pooled_text)
        return t_emb + txt_emb


# ---------------------------------------------------------------------------
# Attention blocks
# ---------------------------------------------------------------------------


class SoftmaxAttention(nn.Module):
    """Standard multi-head softmax attention with QK-Norm and optional RoPE."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 bias: bool = False, qk_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size, bias=bias)

        self.q_norm = RMSNorm(head_dim) if qk_norm else None
        self.k_norm = RMSNorm(head_dim) if qk_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        rotary_cos: Optional[torch.Tensor] = None,
        rotary_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, N, D] image tokens (or concatenated txt+img for single blocks)
            encoder_hidden_states: [B, T, D] text tokens (only for double blocks)
            rotary_cos, rotary_sin: [N_total, head_dim] RoPE embeddings
        """
        B = hidden_states.shape[0]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if encoder_hidden_states is not None:
            # Double block: separate text Q/K/V
            txt_q = self.q_proj(encoder_hidden_states)
            txt_k = self.k_proj(encoder_hidden_states)
            txt_v = self.v_proj(encoder_hidden_states)

        # Reshape to [B, N, H, D]
        q = q.view(B, -1, self.num_heads, self.head_dim)
        k = k.view(B, -1, self.num_heads, self.head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if encoder_hidden_states is not None:
            txt_q = txt_q.view(B, -1, self.num_heads, self.head_dim)
            txt_k = txt_k.view(B, -1, self.num_heads, self.head_dim)
            txt_v = txt_v.view(B, -1, self.num_heads, self.head_dim)
            if self.q_norm is not None:
                txt_q = self.q_norm(txt_q)
                txt_k = self.k_norm(txt_k)

        # Apply RoPE
        if rotary_cos is not None:
            if encoder_hidden_states is not None:
                # Double block: split RoPE for text and image
                txt_len = txt_q.shape[1]
                img_len = q.shape[1]
                txt_cos = rotary_cos[:txt_len]
                txt_sin = rotary_sin[:txt_len]
                img_cos = rotary_cos[txt_len:txt_len + img_len]
                img_sin = rotary_sin[txt_len:txt_len + img_len]

                txt_q = apply_rotary_emb(txt_q, txt_cos, txt_sin)
                txt_k = apply_rotary_emb(txt_k, txt_cos, txt_sin)
                q = apply_rotary_emb(q, img_cos, img_sin)
                k = apply_rotary_emb(k, img_cos, img_sin)
            else:
                q = apply_rotary_emb(q, rotary_cos, rotary_sin)
                k = apply_rotary_emb(k, rotary_cos, rotary_sin)

        # Transpose for attention: [B, H, N, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if encoder_hidden_states is not None:
            txt_q = txt_q.transpose(1, 2)
            txt_k = txt_k.transpose(1, 2)
            txt_v = txt_v.transpose(1, 2)
            # Joint attention: concatenate text and image
            full_q = torch.cat([txt_q, q], dim=2)
            full_k = torch.cat([txt_k, k], dim=2)
            full_v = torch.cat([txt_v, v], dim=2)
        else:
            full_q, full_k, full_v = q, k, v

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            full_q, full_k, full_v, dropout_p=0.0, is_causal=False
        )

        # [B, H, N, D] -> [B, N, H*D]
        attn_output = attn_output.transpose(1, 2).reshape(B, -1, self.inner_dim)

        if encoder_hidden_states is not None:
            txt_len = encoder_hidden_states.shape[1]
            txt_output = attn_output[:, :txt_len]
            img_output = attn_output[:, txt_len:]
            img_output = self.out_proj(img_output)
            txt_output = self.out_proj(txt_output)
            return img_output, txt_output
        else:
            return self.out_proj(attn_output)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class DoubleBlock(nn.Module):
    """Double-stream transformer block (text + image joint softmax attention).

    FLUX.2 style with:
    - Separate MLPs per modality
    - Joint attention (concatenated Q/K/V)
    - AdaLN modulation from shared conditioning
    - Sandwich norm (RMSNorm before + after attention and MLP)
    """

    def __init__(self, config: LightMGTConfig, is_last: bool = False):
        super().__init__()
        hidden = config.hidden_size

        # Image stream norms
        self.norm1_img = RMSNorm(hidden)
        self.norm2_img = RMSNorm(hidden)
        self.post_attn_norm_img = RMSNorm(hidden) if config.use_sandwich_norm else nn.Identity()
        self.post_mlp_norm_img = RMSNorm(hidden) if config.use_sandwich_norm else nn.Identity()

        # Text stream norms (skip MLP-related norms for last block to avoid unused params)
        self.norm1_txt = RMSNorm(hidden)
        self.post_attn_norm_txt = RMSNorm(hidden) if config.use_sandwich_norm else nn.Identity()
        if not is_last:
            self.norm2_txt = RMSNorm(hidden)
            self.post_mlp_norm_txt = RMSNorm(hidden) if config.use_sandwich_norm else nn.Identity()

        # Shared attention for joint text-image
        self.attn = SoftmaxAttention(
            hidden_size=hidden,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            bias=config.use_bias,
            qk_norm=config.use_qk_norm,
        )

        # Separate MLPs
        self.mlp_img = SwiGLUMLP(hidden, mlp_ratio=config.mlp_ratio, bias=config.use_bias)
        self.mlp_txt = SwiGLUMLP(hidden, mlp_ratio=config.mlp_ratio, bias=config.use_bias) if not is_last else None
        self.is_last = is_last

    def forward(
        self,
        img_hidden: torch.Tensor,
        txt_hidden: torch.Tensor,
        cond: torch.Tensor,
        img_mods: Tuple,
        txt_mods: Tuple,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_hidden: [B, N_img, D]
            txt_hidden: [B, N_txt, D]
            cond: [B, D] conditioning vector (unused here, mods are pre-computed)
            img_mods: (shift1, scale1, gate1, shift2, scale2, gate2)
            txt_mods: (shift1, scale1, gate1, shift2, scale2, gate2)
        """
        i_shift1, i_scale1, i_gate1, i_shift2, i_scale2, i_gate2 = img_mods
        t_shift1, t_scale1, t_gate1, t_shift2, t_scale2, t_gate2 = txt_mods

        # Pre-attention norm + modulation
        img_normed = self.norm1_img(img_hidden) * (1 + i_scale1.unsqueeze(1)) + i_shift1.unsqueeze(1)
        txt_normed = self.norm1_txt(txt_hidden) * (1 + t_scale1.unsqueeze(1)) + t_shift1.unsqueeze(1)

        # Joint attention
        img_attn, txt_attn = self.attn(
            hidden_states=img_normed,
            encoder_hidden_states=txt_normed,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
        )

        # Post-attention sandwich norm
        img_attn = self.post_attn_norm_img(img_attn)
        txt_attn = self.post_attn_norm_txt(txt_attn)

        # Residual with gate
        img_hidden = img_hidden + i_gate1.unsqueeze(1) * img_attn
        txt_hidden = txt_hidden + t_gate1.unsqueeze(1) * txt_attn

        # MLP with pre-norm + modulation
        img_normed2 = self.norm2_img(img_hidden) * (1 + i_scale2.unsqueeze(1)) + i_shift2.unsqueeze(1)
        img_mlp = self.post_mlp_norm_img(self.mlp_img(img_normed2))
        img_hidden = img_hidden + i_gate2.unsqueeze(1) * img_mlp

        if not self.is_last:
            txt_normed2 = self.norm2_txt(txt_hidden) * (1 + t_scale2.unsqueeze(1)) + t_shift2.unsqueeze(1)
            txt_mlp = self.post_mlp_norm_txt(self.mlp_txt(txt_normed2))
            txt_hidden = txt_hidden + t_gate2.unsqueeze(1) * txt_mlp

        return img_hidden, txt_hidden


class SingleSoftmaxBlock(nn.Module):
    """Single-stream softmax attention block.

    Used for the 6 blocks near the output where sharp discriminative attention
    is needed for mask prediction. Uses parallel attn||MLP pattern.
    """

    def __init__(self, config: LightMGTConfig):
        super().__init__()
        hidden = config.hidden_size

        self.norm = RMSNorm(hidden)
        self.post_attn_norm = RMSNorm(hidden) if config.use_sandwich_norm else nn.Identity()

        self.attn = SoftmaxAttention(
            hidden_size=hidden,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            bias=config.use_bias,
            qk_norm=config.use_qk_norm,
        )

        self.use_parallel = config.use_parallel_block
        if self.use_parallel:
            self.mlp = SwiGLUMLP(hidden, mlp_ratio=config.mlp_ratio, bias=config.use_bias)
            self.post_mlp_norm = RMSNorm(hidden) if config.use_sandwich_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        mods: Tuple,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale, gate = mods
        residual = hidden_states

        # Pre-norm + modulation
        normed = self.norm(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        if self.use_parallel:
            # Parallel: attn || MLP, then combine
            attn_out = self.post_attn_norm(self.attn(normed, rotary_cos=rotary_cos, rotary_sin=rotary_sin))
            mlp_out = self.post_mlp_norm(self.mlp(normed))
            hidden_states = residual + gate.unsqueeze(1) * (attn_out + mlp_out)
        else:
            attn_out = self.post_attn_norm(self.attn(normed, rotary_cos=rotary_cos, rotary_sin=rotary_sin))
            hidden_states = residual + gate.unsqueeze(1) * attn_out

        return hidden_states


class SingleGLABlock(nn.Module):
    """Single-stream GLA attention block.

    Used for the 14 middle blocks. Gated Linear Attention provides sub-quadratic
    complexity (4-9x speedup at 4096 tokens). Uses parallel attn||MLP pattern.
    """

    def __init__(self, config: LightMGTConfig):
        super().__init__()
        hidden = config.hidden_size

        self.norm = RMSNorm(hidden)
        self.post_attn_norm = RMSNorm(hidden) if config.use_sandwich_norm else nn.Identity()

        self.attn = build_gla_attention(config)

        self.use_parallel = config.use_parallel_block
        if self.use_parallel:
            self.mlp = SwiGLUMLP(hidden, mlp_ratio=config.mlp_ratio, bias=config.use_bias)
            self.post_mlp_norm = RMSNorm(hidden) if config.use_sandwich_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        mods: Tuple,
        rotary_cos: torch.Tensor = None,
        rotary_sin: torch.Tensor = None,
    ) -> torch.Tensor:
        """GLA blocks don't use RoPE (linear attention has its own position handling via DWConv)."""
        shift, scale, gate = mods
        residual = hidden_states

        normed = self.norm(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        if self.use_parallel:
            attn_out = self.post_attn_norm(self.attn(normed))
            mlp_out = self.post_mlp_norm(self.mlp(normed))
            hidden_states = residual + gate.unsqueeze(1) * (attn_out + mlp_out)
        else:
            attn_out = self.post_attn_norm(self.attn(normed))
            hidden_states = residual + gate.unsqueeze(1) * attn_out

        return hidden_states


# ---------------------------------------------------------------------------
# Gen Head
# ---------------------------------------------------------------------------


class FactorizedGenHead(nn.Module):
    """MaskBit-style Factorized Generation Head for LFQ codebooks.

    Factorizes the 262144-way classification into two 512-way classifications:
        token_id = group2_pred * 512 + group1_pred
    where group1 = lower 9 bits, group2 = upper 9 bits of the 18-bit LFQ code.

    This avoids the expensive 262K-way softmax. 2×512 = 1.05M params total.

    Args:
        hidden_size: Input hidden dimension.
        groups: Number of factorization groups (2).
        group_vocab: Vocabulary per group (512 = 2^9).
    """

    def __init__(self, hidden_size: int, groups: int = 2, group_vocab: int = 512):
        super().__init__()
        self.groups = groups
        self.group_vocab = group_vocab
        self.codebook_size = group_vocab ** groups  # 512^2 = 262144

        # No norm here — final_norm in the main model already normalizes
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, group_vocab, bias=False) for _ in range(groups)
        ])

    def forward(self, hidden_states: torch.Tensor) -> list:
        """
        Args:
            hidden_states: [B, N, D] (already normalized by final_norm)

        Returns:
            List of group logits, each [B, N, group_vocab]
        """
        return [head(hidden_states) for head in self.heads]

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        mask: torch.Tensor,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        """Compute factorized cross-entropy loss.

        Args:
            hidden_states: [B, N, D]
            target_ids: [B, N] token IDs in [0, codebook_size)
            mask: [B, N] bool, True for positions to compute loss
            label_smoothing: Label smoothing parameter.

        Returns:
            Scalar loss.
        """
        logits_list = self.forward(hidden_states)

        # Factorize targets: id = g1 * group_vocab + g2
        target_g1 = target_ids // self.group_vocab
        target_g2 = target_ids % self.group_vocab

        targets = [target_g1, target_g2]
        total_loss = 0.0

        for logits, target in zip(logits_list, targets):
            # Only compute loss on masked positions
            logits_masked = logits[mask]  # [M, group_vocab]
            target_masked = target[mask]  # [M]
            loss = F.cross_entropy(
                logits_masked, target_masked, label_smoothing=label_smoothing
            )
            total_loss = total_loss + loss

        return total_loss / self.groups

    @torch.no_grad()
    def sample(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample token IDs and compute confidence scores.

        Args:
            hidden_states: [B, N, D]
            temperature: Sampling temperature.

        Returns:
            pred_ids: [B, N] predicted token IDs
            confidence: [B, N] confidence scores (product of group probs)
        """
        logits_list = self.forward(hidden_states)

        pred_ids = torch.zeros(
            hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device
        )
        confidence = torch.ones(
            hidden_states.shape[:2], dtype=hidden_states.dtype, device=hidden_states.device
        )

        multiplier = 1
        for i in reversed(range(self.groups)):
            logits = logits_list[i]
            probs = F.softmax(logits / temperature, dim=-1)

            probs_flat = probs.reshape(-1, probs.size(-1))
            if probs_flat.dtype != torch.float32:
                probs_flat = probs_flat.float()
            if generator is not None:
                sampled = torch.multinomial(probs_flat, 1, generator=generator)
            else:
                sampled = torch.multinomial(probs_flat, 1)
            sampled = sampled.view(*hidden_states.shape[:2])

            # Confidence = prob of selected token
            group_conf = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
            confidence = confidence * group_conf

            pred_ids = pred_ids + sampled * multiplier
            multiplier *= self.group_vocab

        return pred_ids, confidence


# ---------------------------------------------------------------------------
# VQ Embedding
# ---------------------------------------------------------------------------


class LFQEmbedding(nn.Module):
    """LFQ-aware token embedding (MaskBit-style embedding-free approach).

    For LFQ tokens, each token_id encodes an 18-bit binary code on the
    hypercube {-1, +1}^18. Instead of a 262K embedding table (~268M params),
    we convert indices to bit vectors and project to hidden dim (~19K params).

    Args:
        num_bits: Number of LFQ bits (18 for Open-MAGVIT2 262K).
        hidden_size: Output embedding dimension (1024).
        mask_token_id: ID for [MASK] token (262144).
    """

    def __init__(self, num_bits: int = 18, hidden_size: int = 1024,
                 mask_token_id: int = 262144):
        super().__init__()
        self.num_bits = num_bits
        self.mask_token_id = mask_token_id
        self.bit_proj = nn.Linear(num_bits, hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(hidden_size))
        self.norm = RMSNorm(hidden_size)

        # Precompute bit shift values (buffer, not parameter)
        self.register_buffer(
            'bit_shifts', torch.arange(num_bits, dtype=torch.long)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to hidden representations.

        Args:
            token_ids: [B, N] integer token IDs (0..262143 for real, 262144 for mask).
        Returns:
            [B, N, D] embeddings.
        """
        is_mask = (token_ids == self.mask_token_id)
        # Clamp mask tokens to valid range for bit extraction
        safe_ids = token_ids.clone()
        safe_ids[is_mask] = 0

        # Convert indices to 18-bit binary vectors: {-1, +1}^18
        bits = ((safe_ids.unsqueeze(-1) >> self.bit_shifts) & 1).float() * 2 - 1
        hidden = self.bit_proj(bits)  # [B, N, D]

        # Replace mask positions with learned mask token (match dtype for bf16)
        hidden[is_mask] = self.mask_token.to(hidden.dtype)

        return self.norm(hidden)


# ---------------------------------------------------------------------------
# Main Transformer
# ---------------------------------------------------------------------------


@dataclass
class LightMGTOutput(BaseOutput):
    """Output of LightMGTTransformer."""
    logits: Optional[list] = None  # List of [B, N, group_vocab] per group
    hidden_states: Optional[torch.Tensor] = None  # [B, N, D] last hidden states
    loss: Optional[torch.Tensor] = None


class LightMGTTransformer(ModelMixin, ConfigMixin):
    """LightMGT: Lightweight Masked Generative Transformer.

    Architecture: FLUX.2-style with hybrid attention.
    - 4 double-stream blocks (softmax) for text-image alignment
    - 14 single-stream blocks (GLA) for efficient feature extraction
    - 6 single-stream blocks (softmax) for sharp mask prediction

    Args:
        config: LightMGTConfig instance with all hyperparameters.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 1024,
        num_double_blocks: int = 4,
        num_single_gla_blocks: int = 14,
        num_single_softmax_blocks: int = 6,
        num_attention_heads: int = 16,
        head_dim: int = 64,
        mlp_ratio: float = 2.6875,
        codebook_size: int = 262144,
        mask_token_id: int = 262144,
        vocab_size: int = 262145,
        num_lfq_bits: int = 18,
        gen_head_groups: int = 2,
        gen_head_vocab: int = 512,
        text_hidden_size: int = 1024,
        text_max_length: int = 256,
        rope_axes_dim: tuple = (8, 28, 28),
        rope_theta: float = 10000.0,
        label_smoothing: float = 0.1,
        cfg_dropout: float = 0.1,
        use_sandwich_norm: bool = True,
        use_parallel_block: bool = True,
        use_qk_norm: bool = True,
        use_bias: bool = False,
        gla_num_heads: int = None,
        gla_expand_ratio: float = 1.0,
        gla_conv_size: int = 4,
    ):
        super().__init__()

        # Build a config-like namespace for sub-module constructors
        cfg = LightMGTConfig(
            hidden_size=hidden_size,
            num_double_blocks=num_double_blocks,
            num_single_gla_blocks=num_single_gla_blocks,
            num_single_softmax_blocks=num_single_softmax_blocks,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            codebook_size=codebook_size,
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            gen_head_groups=gen_head_groups,
            gen_head_vocab=gen_head_vocab,
            text_hidden_size=text_hidden_size,
            text_max_length=text_max_length,
            rope_axes_dim=rope_axes_dim,
            rope_theta=rope_theta,
            label_smoothing=label_smoothing,
            cfg_dropout=cfg_dropout,
            use_sandwich_norm=use_sandwich_norm,
            use_parallel_block=use_parallel_block,
            use_qk_norm=use_qk_norm,
            use_bias=use_bias,
            gla_num_heads=gla_num_heads,
            gla_expand_ratio=gla_expand_ratio,
            gla_conv_size=gla_conv_size,
        )

        # --- LFQ Embedding (18-bit → 1024D, ~19K params) ---
        self.vq_embed = LFQEmbedding(
            num_bits=num_lfq_bits,
            hidden_size=hidden_size,
            mask_token_id=mask_token_id,
        )

        # --- Text Projector ---
        if text_hidden_size != hidden_size:
            self.text_proj = nn.Sequential(
                nn.Linear(text_hidden_size, hidden_size, bias=use_bias),
                RMSNorm(hidden_size),
            )
        else:
            # Qwen3.5-0.8B hidden=1024 == transformer hidden=1024, just norm
            self.text_proj = RMSNorm(hidden_size)

        # --- Conditioning ---
        self.cond_embed = ConditionEmbedding(
            hidden_size=hidden_size,
            text_pooled_dim=text_hidden_size,
        )

        # --- Shared AdaLN ---
        self.shared_adaln = SharedAdaLN(hidden_size)

        # --- 3D RoPE ---
        self.rope = LightMGTRoPE3D(
            axes_dim=tuple(rope_axes_dim),
            theta=rope_theta,
        )

        # --- Double blocks (softmax, text-image joint attention) ---
        self.double_blocks = nn.ModuleList([
            DoubleBlock(cfg, is_last=(i == num_double_blocks - 1))
            for i in range(num_double_blocks)
        ])

        # --- Single GLA blocks (linear attention) ---
        self.single_gla_blocks = nn.ModuleList([
            SingleGLABlock(cfg) for _ in range(num_single_gla_blocks)
        ])

        # --- Single softmax blocks (sharp attention near output) ---
        self.single_softmax_blocks = nn.ModuleList([
            SingleSoftmaxBlock(cfg) for _ in range(num_single_softmax_blocks)
        ])

        # --- Factorized Gen Head (2×512 for 18-bit LFQ) ---
        self.gen_head = FactorizedGenHead(
            hidden_size=hidden_size,
            groups=gen_head_groups,
            group_vocab=gen_head_vocab,
        )
        self._num_lfq_bits = num_lfq_bits

        # --- Final norm ---
        self.final_norm = RMSNorm(hidden_size)

        self.gradient_checkpointing = False

        # Store for quick access
        self._hidden_size = hidden_size
        self._mask_token_id = mask_token_id
        self._label_smoothing = label_smoothing

        # Initialize weights — then restore custom inits that apply() overwrites
        self.apply(self._init_weights)
        # Restore SharedAdaLN: zero-init weights, gate bias=0.01
        # (FLUX-style arch needs non-zero gates; MaskBit hyper alignment is separate)
        _gate_init = 0.01
        for m in self.modules():
            if isinstance(m, SharedAdaLN):
                H = hidden_size
                nn.init.zeros_(m.double_proj.weight)
                nn.init.zeros_(m.double_text_proj.weight)
                nn.init.zeros_(m.single_proj.weight)
                with torch.no_grad():
                    m.double_proj.bias.zero_()
                    m.double_proj.bias[2*H:3*H] = _gate_init
                    m.double_proj.bias[5*H:6*H] = _gate_init
                    m.double_text_proj.bias.zero_()
                    m.double_text_proj.bias[2*H:3*H] = _gate_init
                    m.double_text_proj.bias[5*H:6*H] = _gate_init
                    m.single_proj.bias.zero_()
                    m.single_proj.bias[2*H:3*H] = _gate_init
        # Restore GLA gate_proj.bias to 1.0
        from lightmgt.modeling_gla import GLAAttention
        for m in self.modules():
            if isinstance(m, GLAAttention):
                m._reset_parameters()

    def _init_weights(self, module):
        """Initialize weights following GPT-2/FLUX style for stable bf16 training.

        Skips GLAAttention (has _reset_parameters) and SharedAdaLN (has custom init).
        """
        from lightmgt.modeling_gla import GLAAttention
        # Skip modules with their own init
        if isinstance(module, (GLAAttention, SharedAdaLN)):
            return
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        token_ids: torch.Tensor,
        text_hidden_states: torch.Tensor,
        text_pooled: torch.Tensor,
        timestep: torch.Tensor,
        img_h: int = 64,
        img_w: int = 64,
        num_ref_images: int = 0,
        target_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[LightMGTOutput, Tuple]:
        """Forward pass of the LightMGT transformer.

        Args:
            token_ids: [B, N] VQ token IDs (with [MASK] tokens).
            text_hidden_states: [B, T, 1024] text encoder hidden states.
            text_pooled: [B, 1024] pooled text embeddings.
            timestep: [B] timestep values (0-1000 range).
            img_h: Image height in tokens.
            img_w: Image width in tokens.
            num_ref_images: Number of reference images in the sequence.
            target_ids: [B, N] ground truth token IDs (for training loss).
            mask: [B, N] bool mask, True for positions that are [MASK]ed.
            return_dict: Whether to return LightMGTOutput or tuple.

        Returns:
            LightMGTOutput with logits, hidden_states, and optional loss.
        """
        B, N = token_ids.shape
        text_len = text_hidden_states.shape[1]

        # --- Embed tokens ---
        img_hidden = self.vq_embed(token_ids)  # [B, N, D]

        # --- Project text ---
        txt_hidden = self.text_proj(text_hidden_states)  # [B, T, D]

        # --- Conditioning ---
        cond = self.cond_embed(timestep, text_pooled)  # [B, D]

        # --- 3D RoPE ---
        device = token_ids.device
        dtype = img_hidden.dtype
        rotary_cos, rotary_sin = self.rope(
            text_len=text_len,
            img_h=img_h,
            img_w=img_w,
            num_ref_images=num_ref_images,
            device=device,
            dtype=dtype,
        )

        # =====================================================
        # Stage 1: Double blocks (text-image joint attention)
        # =====================================================
        for block in self.double_blocks:
            img_mods, txt_mods = self.shared_adaln.forward_double(cond)

            if self.training and self.gradient_checkpointing:
                img_hidden, txt_hidden = torch.utils.checkpoint.checkpoint(
                    block,
                    img_hidden, txt_hidden, cond, img_mods, txt_mods,
                    rotary_cos, rotary_sin,
                    use_reentrant=False,
                )
            else:
                img_hidden, txt_hidden = block(
                    img_hidden, txt_hidden, cond, img_mods, txt_mods,
                    rotary_cos, rotary_sin,
                )

        # =====================================================
        # Stage 2: Concatenate text + image for single-stream blocks
        # =====================================================
        hidden = torch.cat([txt_hidden, img_hidden], dim=1)  # [B, T+N, D]

        # Single-stream RoPE covers entire [text + image] sequence
        # (already computed above as rotary_cos/sin)

        # =====================================================
        # Stage 2a: GLA single blocks (14 blocks, no RoPE needed)
        # =====================================================
        for block in self.single_gla_blocks:
            mods = self.shared_adaln.forward_single(cond)

            if self.training and self.gradient_checkpointing:
                hidden = torch.utils.checkpoint.checkpoint(
                    block, hidden, mods, None, None,
                    use_reentrant=False,
                )
            else:
                hidden = block(hidden, mods)

        # =====================================================
        # Stage 2b: Softmax single blocks (6 blocks, with RoPE)
        # =====================================================
        for block in self.single_softmax_blocks:
            mods = self.shared_adaln.forward_single(cond)

            if self.training and self.gradient_checkpointing:
                hidden = torch.utils.checkpoint.checkpoint(
                    block, hidden, mods, rotary_cos, rotary_sin,
                    use_reentrant=False,
                )
            else:
                hidden = block(hidden, mods, rotary_cos, rotary_sin)

        # =====================================================
        # Stage 3: Extract image tokens, generate output
        # =====================================================
        img_hidden = hidden[:, text_len:, :]  # [B, N, D]
        img_hidden = self.final_norm(img_hidden)

        # --- Gen Head ---
        logits_list = self.gen_head(img_hidden)

        # --- Loss ---
        loss = None
        if target_ids is not None and mask is not None:
            loss = self.gen_head.compute_loss(
                img_hidden, target_ids, mask, self._label_smoothing
            )

        if not return_dict:
            return (logits_list, img_hidden, loss)

        return LightMGTOutput(
            logits=logits_list,
            hidden_states=img_hidden,
            loss=loss,
        )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters by component for debugging."""
    counts = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        counts[name] = n
    counts["total"] = sum(p.numel() for p in model.parameters())
    return counts
