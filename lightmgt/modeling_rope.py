"""3D Unified RoPE for LightMGT.

Implements Z-Image style 3D Rotary Position Embeddings that unify text and image
tokens into a single (temporal, height, width) coordinate system.

Position assignment:
    - Text tokens:      [t=0..L-1,  0, 0]  — temporal increments, spatial fixed at (0, 0)
    - Reference image:  [t=L,       h, w]  — shares spatial grid with target
    - Target image:     [t=L+1,     h, w]  — same spatial coords, temporal + 1
    - Multi-ref images: [t=L+k,     h, w]  — temporal + k for k-th additional reference

The head dimension is partitioned across axes:
    axes_dim = [8, 28, 28]  →  8d temporal | 28d height | 28d width = 64d total

Each axis gets its own set of sinusoidal frequencies, and the resulting cos/sin
embeddings are concatenated along the head dimension.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def build_position_ids(
    text_len: int,
    img_h: int,
    img_w: int,
    num_ref_images: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build 3D position IDs for a text + image sequence.

    Args:
        text_len: Number of text tokens.
        img_h: Image height in tokens (after VQ encoding).
        img_w: Image width in tokens (after VQ encoding).
        num_ref_images: Number of reference images (0 = unconditional generation).
        device: Target device for the output tensor.

    Returns:
        position_ids: ``[N, 3]`` tensor of ``(t, h, w)`` coordinates where
            ``N = text_len + (1 + num_ref_images) * img_h * img_w``.
    """
    ids = []

    # --- Text tokens: (t=0..L-1, h=0, w=0) ---
    if text_len > 0:
        text_t = torch.arange(text_len, device=device)
        text_hw = torch.zeros(text_len, 2, dtype=torch.long, device=device)
        ids.append(torch.cat([text_t.unsqueeze(1), text_hw], dim=1))

    # --- Image tokens ---
    # Spatial grid shared by all images
    h_coords = torch.arange(img_h, device=device)
    w_coords = torch.arange(img_w, device=device)
    grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")  # [H, W]
    grid_h = grid_h.reshape(-1)  # [H*W]
    grid_w = grid_w.reshape(-1)  # [H*W]

    num_img_tokens = img_h * img_w

    # Reference images: t = text_len, text_len+1, ...
    for k in range(num_ref_images):
        t_val = text_len + k
        img_t = torch.full((num_img_tokens,), t_val, dtype=torch.long, device=device)
        ids.append(torch.stack([img_t, grid_h, grid_w], dim=1))

    # Target image: t = text_len + num_ref_images
    t_target = text_len + num_ref_images
    target_t = torch.full((num_img_tokens,), t_target, dtype=torch.long, device=device)
    ids.append(torch.stack([target_t, grid_h, grid_w], dim=1))

    return torch.cat(ids, dim=0)  # [N, 3]


def _compute_axis_freqs(dim: int, theta: float, device: torch.device) -> torch.Tensor:
    """Compute frequency bands for a single RoPE axis.

    Args:
        dim: Dimension allocated to this axis (must be even).
        theta: Base frequency.
        device: Target device.

    Returns:
        freqs: ``[dim // 2]`` frequency values.
    """
    # freq_i = 1.0 / (theta ** (2i / dim)) for i = 0, 1, ..., dim//2 - 1
    exponents = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
    return 1.0 / (theta ** exponents)


def get_3d_rotary_embedding(
    position_ids: torch.Tensor,
    axes_dim: Tuple[int, int, int] = (8, 28, 28),
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 3D RoPE cos/sin embeddings from position IDs.

    For each axis (temporal, height, width), we compute sinusoidal embeddings
    using that axis's position coordinates and frequency bands, then concatenate
    along the head dimension.

    Args:
        position_ids: ``[N, 3]`` or ``[B, N, 3]`` integer tensor of
            ``(t, h, w)`` coordinates.
        axes_dim: Dimension allocation per axis ``[d_t, d_h, d_w]``.
            Must sum to ``head_dim`` and each must be even.
        theta: Base frequency for sinusoidal encoding.

    Returns:
        cos: ``[N, head_dim]`` or ``[B, N, head_dim]`` cosine embeddings.
        sin: ``[N, head_dim]`` or ``[B, N, head_dim]`` sine embeddings.
    """
    assert position_ids.shape[-1] == 3, (
        f"Expected last dim of position_ids to be 3, got {position_ids.shape[-1]}"
    )
    assert all(d % 2 == 0 for d in axes_dim), (
        f"All axes_dim must be even, got {axes_dim}"
    )

    device = position_ids.device
    has_batch = position_ids.dim() == 3

    cos_parts = []
    sin_parts = []

    for axis_idx, dim in enumerate(axes_dim):
        # Position values for this axis: [..., N]
        pos = position_ids[..., axis_idx].float()

        # Frequency bands: [dim // 2]
        freqs = _compute_axis_freqs(dim, theta, device)

        # Outer product: angles = pos * freqs → [..., N, dim // 2]
        if has_batch:
            # pos: [B, N] → [B, N, 1], freqs: [dim//2] → [1, 1, dim//2]
            angles = pos.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        else:
            # pos: [N] → [N, 1], freqs: [dim//2] → [1, dim//2]
            angles = pos.unsqueeze(-1) * freqs.unsqueeze(0)

        # Duplicate angles for the full dim (interleaved pairs share the same angle)
        # We use the "non-interleaved" layout: [cos, cos, ...] [sin, sin, ...]
        # so each half of the dim gets the same angles.
        cos_parts.append(torch.cos(angles).repeat_interleave(2, dim=-1))
        sin_parts.append(torch.sin(angles).repeat_interleave(2, dim=-1))

    # Concatenate across axes: [temporal | height | width]
    cos = torch.cat(cos_parts, dim=-1)  # [..., head_dim]
    sin = torch.cat(sin_parts, dim=-1)  # [..., head_dim]

    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of elements: (x0, x1) → (-x1, x0).

    This implements the rotation needed for RoPE when using the
    non-interleaved (paired) layout.
    """
    x1 = x[..., ::2]   # even indices
    x2 = x[..., 1::2]  # odd indices
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to query or key tensors.

    Args:
        x: Input tensor of shape ``[B, num_heads, N, head_dim]`` or
            ``[B, N, num_heads, head_dim]``.
        cos: Cosine embeddings ``[B, N, head_dim]`` or ``[N, head_dim]``.
        sin: Sine embeddings ``[B, N, head_dim]`` or ``[N, head_dim]``.

    Returns:
        Tensor of the same shape as ``x`` with RoPE applied.
    """
    # Determine which dimension is the sequence dimension in x
    # Common layouts: [B, heads, N, D] or [B, N, heads, D]
    if x.dim() == 4:
        if cos.dim() == 2:
            # cos: [N, D] → broadcast to match x
            if x.shape[1] == cos.shape[0]:
                # x is [B, N, heads, D]
                cos = cos.unsqueeze(0).unsqueeze(2)  # [1, N, 1, D]
                sin = sin.unsqueeze(0).unsqueeze(2)
            else:
                # x is [B, heads, N, D]
                cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, N, D]
                sin = sin.unsqueeze(0).unsqueeze(1)
        elif cos.dim() == 3:
            # cos: [B, N, D]
            if x.shape[1] == cos.shape[1]:
                # x is [B, N, heads, D]
                cos = cos.unsqueeze(2)  # [B, N, 1, D]
                sin = sin.unsqueeze(2)
            else:
                # x is [B, heads, N, D]
                cos = cos.unsqueeze(1)  # [B, 1, N, D]
                sin = sin.unsqueeze(1)
    elif x.dim() == 3:
        # x: [B, N, D] — no head dimension
        if cos.dim() == 2:
            cos = cos.unsqueeze(0)  # [1, N, D]
            sin = sin.unsqueeze(0)

    return x * cos + _rotate_half(x) * sin


class LightMGTRoPE3D(nn.Module):
    """3D Unified Rotary Position Embedding module for LightMGT.

    Precomputes and caches position IDs and rotary embeddings for a given
    sequence configuration. Useful as a reusable module inside the transformer.

    Args:
        axes_dim: Dimension allocation per axis ``(d_t, d_h, d_w)``.
        theta: Base frequency for sinusoidal encoding.
    """

    def __init__(
        self,
        axes_dim: Tuple[int, int, int] = (8, 28, 28),
        theta: float = 10000.0,
    ):
        super().__init__()
        self.axes_dim = axes_dim
        self.theta = theta
        self.head_dim = sum(axes_dim)

    def forward(
        self,
        text_len: int,
        img_h: int,
        img_w: int,
        num_ref_images: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build position IDs and compute rotary embeddings.

        Args:
            text_len: Number of text tokens.
            img_h: Image height in tokens.
            img_w: Image width in tokens.
            num_ref_images: Number of reference images.
            device: Target device.
            dtype: Target dtype for cos/sin (default: float32).

        Returns:
            cos: ``[N, head_dim]`` cosine embeddings.
            sin: ``[N, head_dim]`` sine embeddings.
        """
        position_ids = build_position_ids(
            text_len, img_h, img_w, num_ref_images, device=device
        )
        cos, sin = get_3d_rotary_embedding(
            position_ids, self.axes_dim, self.theta
        )
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        return cos, sin

    def extra_repr(self) -> str:
        return f"axes_dim={self.axes_dim}, theta={self.theta}, head_dim={self.head_dim}"
