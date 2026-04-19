"""MaskGIT Scheduler for LightMGT.

Implements the iterative unmasking schedule for masked generative transformers:
- Training: arccos mask schedule (MaskBit-aligned)
- Inference: cosine unmask schedule with confidence-based remasking
- Factorized token sampling support (2x512 for 262K LFQ)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class MaskGITSchedulerOutput(BaseOutput):
    """Output of the MaskGIT scheduler step.

    Attributes:
        prev_sample: [B, N] token IDs after this step (with re-masking).
        pred_original_sample: [B, N] predicted clean token IDs.
        confidence: [B, N] confidence scores for each prediction.
    """
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor
    confidence: Optional[torch.Tensor] = None


def gumbel_noise(t: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Generate Gumbel noise for stochastic sampling."""
    device = generator.device if generator is not None else t.device
    noise = torch.zeros_like(t, device=device).uniform_(0, 1, generator=generator).to(t.device)
    return -torch.log((-torch.log(noise.clamp(1e-20))).clamp(1e-20))


def mask_by_random_topk(
    mask_len: torch.Tensor,
    probs: torch.Tensor,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Select lowest-confidence tokens to re-mask.

    Args:
        mask_len: [B, 1] number of tokens to mask.
        probs: [B, N] confidence scores.
        temperature: Stochasticity temperature.
        generator: Random generator.

    Returns:
        masking: [B, N] bool tensor, True for positions to re-mask.
    """
    confidence = torch.log(probs.clamp(1e-20)) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = torch.sort(confidence, dim=-1).values
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    return confidence < cut_off


class MaskGITScheduler(SchedulerMixin, ConfigMixin):
    """MaskGIT iterative unmasking scheduler.

    Training: Uses arccos schedule to determine mask ratio per sample.
    Inference: Uses cosine schedule to progressively unmask tokens.

    Args:
        mask_token_id: Token ID for [MASK] (262144 for LFQ 18-bit).
        masking_schedule: Schedule type ("cosine" or "arccos").
        gen_head_groups: Number of factorized groups (2).
        gen_head_vocab: Vocabulary per group (512).
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        mask_token_id: int = 262144,
        masking_schedule: str = "cosine",
        gen_head_groups: int = 2,
        gen_head_vocab: int = 512,
    ):
        self.temperatures = None
        self.timesteps = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        temperature: Union[float, Tuple[float, float], List[float]] = (2.0, 0.0),
        device: Union[str, torch.device] = None,
    ):
        """Set up the inference schedule.

        Args:
            num_inference_steps: Number of unmasking steps.
            temperature: Sampling temperature schedule (start, end).
            device: Target device.
        """
        self.timesteps = torch.arange(num_inference_steps, device=device).flip(0)

        if isinstance(temperature, (tuple, list)):
            self.temperatures = torch.linspace(
                temperature[0], temperature[1], num_inference_steps, device=device
            )
        else:
            self.temperatures = torch.linspace(
                temperature, 0.01, num_inference_steps, device=device
            )

    # -----------------------------------------------------------------
    # Training utilities
    # -----------------------------------------------------------------

    def get_train_mask_ratio(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample mask ratios using arccos schedule (MaskBit).

        Args:
            batch_size: Number of samples.
            device: Target device.

        Returns:
            mask_ratio: [B] float tensor in (0, 1].
        """
        r = torch.rand(batch_size, device=device)
        mask_ratio = torch.arccos(r) / (math.pi / 2)
        # Full arccos range (MaskBit recipe): no clamp, let the schedule
        # naturally concentrate on high mask ratios while allowing easy tasks
        mask_ratio = mask_ratio.clamp(min=1e-4)  # only prevent exact zero
        return mask_ratio

    def add_noise(
        self,
        sample: torch.Tensor,
        mask_ratio: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking for training.

        Args:
            sample: [B, N] clean token IDs.
            mask_ratio: [B] mask ratios. If None, sampled via arccos schedule.
            generator: Random generator.

        Returns:
            masked_sample: [B, N] with [MASK] tokens.
            mask: [B, N] bool tensor, True for masked positions.
        """
        B, N = sample.shape
        device = sample.device

        if mask_ratio is None:
            mask_ratio = self.get_train_mask_ratio(B, device)

        num_mask = (mask_ratio * N).long().clamp(min=1)

        # Generate random scores and mask the lowest
        gen_device = generator.device if generator is not None else device
        rand_scores = torch.rand(B, N, device=gen_device, generator=generator).to(device)
        sorted_indices = rand_scores.argsort(dim=-1)

        # Create mask: first num_mask[i] positions (after sorting by random score) are masked
        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        mask = positions < num_mask.unsqueeze(1)

        # Scatter back to original positions
        scatter_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        scatter_mask.scatter_(1, sorted_indices, mask)

        masked_sample = sample.clone()
        masked_sample[scatter_mask] = self.config.mask_token_id

        return masked_sample, scatter_mask

    # -----------------------------------------------------------------
    # Inference step
    # -----------------------------------------------------------------

    def step(
        self,
        model_output_hidden: torch.Tensor,
        timestep: torch.long,
        sample: torch.LongTensor,
        gen_head,
        starting_mask_ratio: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[MaskGITSchedulerOutput, Tuple]:
        """One step of iterative unmasking.

        Args:
            model_output_hidden: [B, N, D] hidden states from transformer.
            timestep: Current timestep index.
            sample: [B, N] current token IDs (with masks).
            gen_head: FactorizedGenHead module for sampling.
            starting_mask_ratio: Initial mask ratio (default 1.0).
            generator: Random generator.
            return_dict: Whether to return dataclass or tuple.

        Returns:
            MaskGITSchedulerOutput with updated tokens.
        """
        unknown_map = (sample == self.config.mask_token_id)

        # Sample predictions from gen head
        pred_ids, confidence = gen_head.sample(
            model_output_hidden, temperature=1.0, generator=generator
        )

        # Keep already-unmasked tokens unchanged
        pred_original_sample = torch.where(unknown_map, pred_ids, sample)

        if timestep == 0:
            # Last step: accept all predictions
            prev_sample = pred_original_sample
        else:
            seq_len = sample.shape[1]
            step_idx = (self.timesteps == timestep).nonzero()
            ratio = (step_idx + 1) / len(self.timesteps)

            if self.config.masking_schedule == "cosine":
                mask_ratio = torch.cos(ratio * math.pi / 2)
            elif self.config.masking_schedule == "arccos":
                mask_ratio = torch.arccos(ratio) / (math.pi / 2)
            else:
                mask_ratio = 1 - ratio

            mask_ratio = starting_mask_ratio * mask_ratio
            mask_len = (seq_len * mask_ratio).floor()

            # Get confidence for predictions
            selected_confidence = confidence

            # Maximize confidence for already-unmasked positions (don't re-mask them)
            selected_confidence = torch.where(
                unknown_map, selected_confidence,
                torch.finfo(selected_confidence.dtype).max
            )

            # Don't mask more than currently masked
            mask_len = torch.min(
                unknown_map.sum(dim=-1, keepdim=True).float() - 1, mask_len
            )
            mask_len = torch.max(
                torch.tensor([1.0], device=sample.device), mask_len
            )

            # Re-mask lowest confidence positions
            masking = mask_by_random_topk(
                mask_len, selected_confidence,
                self.temperatures[step_idx],
                generator=generator,
            )
            prev_sample = torch.where(
                masking, self.config.mask_token_id, pred_original_sample
            )

        if not return_dict:
            return (prev_sample, pred_original_sample, confidence)

        return MaskGITSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
            confidence=confidence,
        )


# -----------------------------------------------------------------
# Time Interval CFG (eMIGM)
# -----------------------------------------------------------------


def should_apply_cfg(step: int, total_steps: int, start_ratio: float = 0.3, end_ratio: float = 0.7) -> bool:
    """Determine if CFG should be applied at this step (Time Interval CFG).

    Only applies CFG in the middle portion of the schedule (30%-70% by default),
    saving ~40% inference compute.
    """
    ratio = step / max(total_steps - 1, 1)
    return start_ratio <= ratio <= end_ratio
