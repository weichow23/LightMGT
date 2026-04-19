"""Gated Linear Attention (GLA) with Depthwise Convolution.

Implements the GLA mechanism used in the 14 middle single-stream blocks of
LightMGT, replacing standard softmax attention with sub-quadratic linear
attention for 4-9x speedup at 1024px resolution (4096 tokens).

IMPORTANT: MaskGIT is non-autoregressive — all tokens must attend to all other
tokens (bidirectional). This implementation uses the kernel trick for O(N*D^2)
bidirectional linear attention, NOT causal masking.

References:
    - DiG: Scalable Diffusion with Gated Linear Attention (CVPR 2025)
    - LiT: Large-scale Image Tokenizer (ICCV 2025)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        dim: Feature dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * norm).to(dtype) * self.weight


def bidirectional_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Bidirectional linear attention via the kernel trick.

    MaskGIT uses non-autoregressive masked token prediction where every token
    must attend to every other token. Causal masking is WRONG for this setting.

    Complexity: O(N * D^2) via associativity of matrix multiplication, compared
    to O(N^2 * D) for explicit attention. For typical D=64, N=4096 this gives
    ~64x speedup over naive quadratic attention.

    Computed in float32 to avoid bf16 overflow.

    Args:
        q: ``[B, H, N, D]`` queries (non-negative after activation).
        k: ``[B, H, N, D]`` keys (non-negative after 1+elu).
        v: ``[B, H, N, D_v]`` values.

    Returns:
        ``[B, H, N, D_v]`` output.
    """
    orig_dtype = q.dtype
    q = q.float()
    k = k.float()
    v = v.float()

    # Kernel trick: output_i = Q_i @ S / (Q_i @ z)
    # where S = K^T @ V (global key-value summary) and z = sum(K) (normalization)
    kv = torch.einsum("bhnd,bhnv->bhdv", k, v)   # [B, H, D, D_v]
    qkv = torch.einsum("bhnd,bhdv->bhnv", q, kv)  # [B, H, N, D_v]

    z = k.sum(dim=2)                                # [B, H, D]
    denom = torch.einsum("bhnd,bhd->bhn", q, z)     # [B, H, N]
    denom = denom.clamp(min=1e-6).unsqueeze(-1)

    return (qkv / denom).to(orig_dtype)


class DepthwiseConv1d(nn.Module):
    """Depthwise separable 1-D convolution for injecting local context.

    Applied along the sequence dimension independently per channel, giving
    each value head a short-range receptive field (default kernel_size=4).

    Uses symmetric (same) padding for bidirectional context, since MaskGIT
    is non-autoregressive and tokens should attend to both left and right
    neighbors in the flattened raster-scan sequence.

    Args:
        channels: Number of input/output channels (= hidden_size).
        kernel_size: Convolution kernel size.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        padding: Optional[int] = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,  # depthwise
            bias=True,
        )
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise conv along the sequence dimension.

        Args:
            x: Input tensor of shape ``[B, N, D]``.

        Returns:
            Output tensor of shape ``[B, N, D]``.
        """
        N = x.shape[1]
        x = rearrange(x, "b n d -> b d n")
        x = self.conv(x)
        # Trim to original length if padding produced extra samples
        if x.shape[2] > N:
            x = x[:, :, :N]
        x = rearrange(x, "b d n -> b n d")
        return x


class GLAAttention(nn.Module):
    """Gated Linear Attention with Depthwise Convolution.

    Replaces standard softmax attention in the 14 middle single-stream blocks
    of LightMGT. Uses non-negative key activations (``1 + elu(k)``), a
    depthwise convolution on values for local context, and element-wise gating
    on the output.

    When the ``fla`` library is available, the efficient ``chunk_gla`` kernel
    is used for O(N) chunk-wise recurrence. Otherwise, a naive O(N^2) fallback
    is used.

    Args:
        hidden_size: Model hidden dimension (default: 1024).
        num_heads: Number of attention heads (default: 16).
        head_dim: Dimension per head (default: 64).
        conv_size: Depthwise conv kernel size (default: 4).
        expand_ratio: Gate expansion ratio (default: 1.0).
        use_qk_norm: Whether to apply RMSNorm to Q and K (default: True).
        use_bias: Whether to use bias in linear projections (default: False).

    Example::

        >>> gla = GLAAttention(hidden_size=1024, num_heads=16, head_dim=64)
        >>> x = torch.randn(2, 4096, 1024)
        >>> output = gla(x)
        >>> output.shape
        torch.Size([2, 4096, 1024])
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        conv_size: int = 4,
        expand_ratio: float = 1.0,
        use_qk_norm: bool = True,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.use_qk_norm = use_qk_norm

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, self.inner_dim, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, self.inner_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, self.inner_dim, bias=use_bias)

        # Gate projection (element-wise output gating)
        self.gate_proj = nn.Linear(hidden_size, self.inner_dim, bias=True)

        # Depthwise conv on values for local context
        self.dwconv = DepthwiseConv1d(
            channels=self.inner_dim,
            kernel_size=conv_size,
        )

        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, hidden_size, bias=use_bias)

        # Optional QK-Norm (per-head RMSNorm)
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize projections with scaled normal initialization."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        # Gate bias initialized to positive value so gates start near-open.
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.ones_(self.gate_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of Gated Linear Attention (bidirectional).

        Args:
            hidden_states: Input tensor of shape ``[B, N, D]``.

        Returns:
            Output tensor of shape ``[B, N, D]``.
        """
        B, N, D = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        gate = torch.sigmoid(self.gate_proj(hidden_states))

        v = self.dwconv(v)

        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.num_heads)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k = 1.0 + F.elu(k)

        # Bidirectional linear attention via kernel trick — O(N * D^2)
        # NOTE: chunk_gla from fla is causal and WRONG for MaskGIT.
        # torch.compile is also incompatible with the fla naive fallback.
        q_t = rearrange(q, "b n h d -> b h n d")
        k_t = rearrange(k, "b n h d -> b h n d")
        v_t = rearrange(v, "b n h d -> b h n d")
        output = bidirectional_linear_attention(q_t, k_t, v_t)
        output = rearrange(output, "b h n d -> b n (h d)")

        output = gate * output
        output = self.out_proj(output)

        return output


def build_gla_attention(config) -> GLAAttention:
    """Factory function to build a GLAAttention layer from a LightMGTConfig.

    Args:
        config: A ``LightMGTConfig`` instance.

    Returns:
        A configured ``GLAAttention`` module.
    """
    num_heads = getattr(config, "gla_num_heads", None) or config.num_attention_heads
    return GLAAttention(
        hidden_size=config.hidden_size,
        num_heads=num_heads,
        head_dim=config.head_dim,
        conv_size=config.gla_conv_size,
        expand_ratio=config.gla_expand_ratio,
        use_qk_norm=config.use_qk_norm,
        use_bias=config.use_bias,
    )
