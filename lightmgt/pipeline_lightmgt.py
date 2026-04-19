"""LightMGT Inference Pipeline.

Text-to-Image and Image Editing pipeline with:
- Qwen3.5-0.8B text encoding (last-token pooling for causal LM)
- Open-MAGVIT2-PT 262K VQ-VAE encode/decode
- MaskGIT iterative unmasking
- Classifier-free guidance with Time Interval CFG
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import logging
from PIL import Image

from lightmgt.modeling_lightmgt import LightMGTTransformer
from lightmgt.scheduler_maskgit import MaskGITScheduler, should_apply_cfg

logger = logging.get_logger(__name__)


class LightMGTPipeline(DiffusionPipeline):
    """Pipeline for LightMGT text-to-image generation and editing.

    Components:
        transformer: LightMGTTransformer model.
        scheduler: MaskGITScheduler for iterative unmasking.
        text_encoder: Qwen3.5-0.8B (frozen) text encoder.
        tokenizer: Corresponding tokenizer.
        vq_model: OpenMAGVIT2Wrapper for encoding/decoding images.
    """

    def __init__(
        self,
        transformer: LightMGTTransformer,
        scheduler: MaskGITScheduler,
        text_encoder=None,
        tokenizer=None,
        vq_model=None,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vq_model=vq_model,
        )

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt using Qwen3.5-0.8B.

        Uses last-token pooling (matching training) since Qwen is a causal LM:
        the last non-pad token has attended to all preceding tokens and carries
        the sequence summary.

        Args:
            prompt: Input text prompt(s).
            device: Target device.
            dtype: Target dtype.
            max_length: Max token length.

        Returns:
            hidden_states: [B, T, 1024] text hidden states.
            pooled: [B, 1024] pooled text embeddings (last-token + L2 renorm).
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1].to(dtype)  # [B, T, 1024]

        # Last-token pooling (consistent with training)
        B = hidden_states.shape[0]
        seq_lengths = inputs.attention_mask.sum(dim=1).long() - 1  # [B]
        pooled = hidden_states[torch.arange(B, device=device), seq_lengths]  # [B, D]
        pooled = F.normalize(pooled, dim=-1) * (pooled.shape[-1] ** 0.5)

        return hidden_states, pooled

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to VQ tokens using Open-MAGVIT2-PT 262K.

        Args:
            image: [B, 3, H, W] normalized to [-1, 1].

        Returns:
            token_ids: [B, h*w] discrete token IDs in [0, 262143].
        """
        with torch.no_grad():
            token_ids, _ = self.vq_model.encode(image)
        return token_ids

    def decode_tokens(self, token_ids: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        """Decode VQ tokens back to image.

        Args:
            token_ids: [B, N] token IDs.
            img_h: Token grid height.
            img_w: Token grid width.

        Returns:
            images: [B, 3, H, W] pixel values in [-1, 1].
        """
        mask_id = self.scheduler.config.mask_token_id
        token_ids = torch.where(
            token_ids == mask_id,
            torch.zeros_like(token_ids),
            token_ids,
        )

        with torch.no_grad():
            images = self.vq_model.decode_tokens(token_ids, img_h, img_w)

        return images

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        temperature: Union[float, Tuple[float, float]] = (2.0, 0.0),
        use_time_interval_cfg: bool = True,
        cfg_start_ratio: float = 0.3,
        cfg_end_ratio: float = 0.7,
        generator: Optional[torch.Generator] = None,
        reference_image: Optional[torch.Tensor] = None,
        edit_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """Generate or edit an image.

        Args:
            prompt: Text prompt(s).
            height: Output image height (must be divisible by 16).
            width: Output image width (must be divisible by 16).
            num_inference_steps: Number of unmasking steps.
            guidance_scale: CFG scale (>1 for stronger guidance).
            temperature: Sampling temperature (start, end) schedule.
            use_time_interval_cfg: Whether to use Time Interval CFG (saves ~40%).
            cfg_start_ratio: Start ratio for time-interval CFG.
            cfg_end_ratio: End ratio for time-interval CFG.
            generator: Random generator for reproducibility.
            reference_image: [B, 3, H, W] reference image for editing (normalized [-1,1]).
            edit_mask: [B, h, w] bool mask for editing (True = edit region).

        Returns:
            images: List of PIL Images or tensor.
        """
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        img_h = height // 16
        img_w = width // 16
        num_tokens = img_h * img_w

        # --- Encode text ---
        text_hidden, text_pooled = self.encode_prompt(prompt, device, dtype)

        # --- Handle editing ---
        num_ref_images = 0
        if reference_image is not None:
            ref_tokens = self.encode_image(reference_image.to(device))
            num_ref_images = 1
        else:
            ref_tokens = None

        # --- Initialize all tokens as [MASK] ---
        mask_token_id = self.scheduler.config.mask_token_id
        tokens = torch.full(
            (batch_size, num_tokens), mask_token_id,
            dtype=torch.long, device=device,
        )

        # For editing: copy unmasked region from reference
        if ref_tokens is not None and edit_mask is not None:
            flat_mask = edit_mask.reshape(batch_size, -1)  # [B, N]
            tokens = torch.where(flat_mask, tokens, ref_tokens)

        # --- Setup scheduler ---
        self.scheduler.set_timesteps(num_inference_steps, temperature, device)

        # --- Null embeddings for CFG ---
        if guidance_scale > 1.0:
            null_hidden = torch.zeros_like(text_hidden)
            null_pooled = torch.zeros_like(text_pooled)

        # --- Iterative unmasking ---
        for i, timestep in enumerate(self.scheduler.timesteps):
            apply_cfg = (
                guidance_scale > 1.0
                and (
                    not use_time_interval_cfg
                    or should_apply_cfg(i, num_inference_steps, cfg_start_ratio, cfg_end_ratio)
                )
            )

            t = timestep.float() / num_inference_steps * 1000

            if apply_cfg:
                double_tokens = tokens.repeat(2, 1)
                double_text = torch.cat([text_hidden, null_hidden], dim=0)
                double_pooled = torch.cat([text_pooled, null_pooled], dim=0)
                double_t = t.expand(batch_size * 2)

                output = self.transformer(
                    token_ids=double_tokens,
                    text_hidden_states=double_text,
                    text_pooled=double_pooled,
                    timestep=double_t,
                    img_h=img_h,
                    img_w=img_w,
                    num_ref_images=num_ref_images,
                )

                cond_hidden = output.hidden_states[:batch_size]
                uncond_hidden = output.hidden_states[batch_size:]
                guided_hidden = uncond_hidden + guidance_scale * (cond_hidden - uncond_hidden)
            else:
                t_batch = t.expand(batch_size)
                output = self.transformer(
                    token_ids=tokens,
                    text_hidden_states=text_hidden,
                    text_pooled=text_pooled,
                    timestep=t_batch,
                    img_h=img_h,
                    img_w=img_w,
                    num_ref_images=num_ref_images,
                )
                guided_hidden = output.hidden_states

            scheduler_output = self.scheduler.step(
                model_output_hidden=guided_hidden,
                timestep=timestep,
                sample=tokens,
                gen_head=self.transformer.gen_head,
                generator=generator,
            )

            tokens = scheduler_output.prev_sample

            if ref_tokens is not None and edit_mask is not None:
                tokens = torch.where(flat_mask, tokens, ref_tokens)

        # --- Decode tokens to image ---
        if self.vq_model is not None:
            images = self.decode_tokens(tokens, img_h, img_w)  # [-1, 1]
            images = ((images + 1) / 2 * 255).clamp(0, 255).byte()
            images_pil = []
            for img in images:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                images_pil.append(Image.fromarray(img_np))
            return images_pil
        else:
            return tokens
