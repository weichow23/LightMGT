"""VQ-VAE Wrapper for LightMGT.

Open-MAGVIT2-PT: LFQ 18-bit, 262144 codes, trained on ~100M diverse images.

Uses 16x downsampling:
- 256px -> 16x16 = 256 tokens
- 512px -> 32x32 = 1024 tokens
- 1024px -> 64x64 = 4096 tokens
- Image normalization: [-1, 1]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


class OpenMAGVIT2Wrapper(nn.Module):
    """Wrapper around Open-MAGVIT2-PT 262K (18-bit LFQ) VQ-VAE.

    Loads the model from SEED-Voken codebase and exposes simple APIs.
    LFQ properties:
    - 18-bit Lookup-Free Quantization: each token = 18 binary decisions
    - Codebook: 2^18 = 262144 implicit codes (no learned codebook table)
    - Structured: Hamming-distance-close indices = visually similar
    - Factorized head compatible: split 18 bits into 2 groups of 9
    """

    def __init__(
        self,
        ckpt_path: str,
        seed_voken_dir: str = "/mnt/bn/search-auto-eval-v2/zhouwei/SEED-Voken",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # Add SEED-Voken to path (for Encoder, Decoder, LFQ modules only)
        if seed_voken_dir not in sys.path:
            sys.path.insert(0, seed_voken_dir)

        # Import only the components we need (avoids lightning dependency)
        from src.Open_MAGVIT2.modules.diffusionmodules.improved_model import Encoder, Decoder
        from src.Open_MAGVIT2.modules.vqvae.lookup_free_quantize import LFQ

        # Hardcoded config from pretrain_lfqgan_256_262144.yaml
        ddconfig = {
            "double_z": False,
            "z_channels": 18,
            "resolution": 128,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 1, 2, 2, 4],
            "num_res_blocks": 4,
        }

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = LFQ(
            dim=18,
            codebook_size=262144,
            sample_minimization_weight=1.0,
            batch_maximization_weight=1.0,
        )

        # Load checkpoint
        self._load_checkpoint(ckpt_path)
        self.eval()
        self.requires_grad_(False)

    def _load_checkpoint(self, ckpt_path: str):
        """Load Open-MAGVIT2-PT checkpoint, filtering to encoder/decoder/quantize."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        # Filter: only keep encoder, decoder, quantize weights
        filtered = {}
        for k, v in sd.items():
            if k.startswith(("encoder.", "decoder.", "quantize.")):
                filtered[k] = v

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        real_missing = [k for k in missing if not k.startswith("quantize.")]
        if real_missing:
            print(f"Open-MAGVIT2: {len(real_missing)} missing keys: {real_missing[:5]}")
        print(f"Open-MAGVIT2 loaded: {len(filtered)} params from {ckpt_path}")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode images to discrete LFQ token indices.

        Args:
            x: [B, 3, H, W] images normalized to [-1, 1].

        Returns:
            token_ids: [B, h*w] discrete token IDs in [0, 262143].
            quant: [B, 18, h, w] quantized continuous representation.
        """
        h = self.encoder(x)
        (quant, _, info), _ = self.quantize(h, return_loss_breakdown=True)
        indices = info
        B = x.shape[0]
        token_ids = indices.reshape(B, -1)
        return token_ids, quant

    @torch.no_grad()
    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """Decode from quantized representation.

        Args:
            quant: [B, 18, h, w] quantized representation.

        Returns:
            images: [B, 3, H, W] in [-1, 1].
        """
        return self.decoder(quant)

    @torch.no_grad()
    def decode_tokens(self, token_ids: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        """Decode from discrete token IDs by converting to LFQ bit vectors.

        Args:
            token_ids: [B, N] token IDs in [0, 262143].
            img_h: Token grid height.
            img_w: Token grid width.

        Returns:
            images: [B, 3, H, W] in [-1, 1].
        """
        B = token_ids.shape[0]
        num_bits = 18
        # Convert indices to binary vectors: {-1, +1}^18
        bit_shifts = torch.arange(num_bits, device=token_ids.device)
        bits = ((token_ids.unsqueeze(-1) >> bit_shifts) & 1).float() * 2 - 1
        # Reshape to spatial: [B, h, w, 18] -> [B, 18, h, w]
        quant = bits.reshape(B, img_h, img_w, num_bits).permute(0, 3, 1, 2).contiguous()
        return self.decode(quant)

    @property
    def codebook_size(self) -> int:
        return 262144

    @property
    def num_bits(self) -> int:
        return 18

    @property
    def embed_dim(self) -> int:
        return 18

    @property
    def downsample_factor(self) -> int:
        return 16

    def get_token_grid_size(self, image_size: int) -> int:
        return image_size // self.downsample_factor
