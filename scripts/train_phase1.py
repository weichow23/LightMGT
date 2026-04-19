"""Phase 1 T2I Training Script for LightMGT.

Online encoding: images -> Open-MAGVIT2-PT 262K LFQ -> VQ tokens on-the-fly each step.
Multi-scale training with aspect-ratio bucketing.
Uses torchrun + manual DDP (not accelerate).

Usage:
    # Single GPU debug
    python scripts/train_phase1.py --resolution 256 --batch_size 4 --max_steps 100

    # 2x H800
    torchrun --nproc_per_node=2 scripts/train_phase1.py \
        --resolution 256 --batch_size 32 --gradient_accumulation_steps 2
"""

import argparse
import contextlib
import logging
import math
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightmgt.modeling_lightmgt import LightMGTTransformer
from lightmgt.scheduler_maskgit import MaskGITScheduler
from train.dataset import build_dataloader


# ─── DDP Utilities ───────────────────────────────────────────────────

def setup_ddp():
    """Initialize DDP if launched via torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    # Single GPU fallback
    return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, nargs="+",
                        default=["/mnt/hdfs/weichow/maskedit/t2i"],
                        help="JSON directory(ies) containing data shards")
    parser.add_argument("--resolution", type=int, default=256, choices=[128, 256, 512, 1024])
    parser.add_argument("--multi_scale", action="store_true", default=True)
    parser.add_argument("--no_multi_scale", dest="multi_scale", action="store_false")

    parser.add_argument("--text_encoder", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--vq_ckpt", type=str,
                        default="/mnt/bn/search-auto-eval-v2/zhouwei/nextmgt/tokenizer_ckpts/pretrain256_262144.ckpt",
                        help="Path to Open-MAGVIT2-PT 262K checkpoint")
    parser.add_argument("--seed_voken_dir", type=str,
                        default="/mnt/bn/search-auto-eval-v2/zhouwei/SEED-Voken")
    parser.add_argument("--tar_pattern", type=str, default=None, nargs="+",
                        help="Glob pattern(s) for image tar archives, e.g. /path/to/camera_good_images_*.tar /path/to/fine_t2i_images_*.tar")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1.5e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=300000)
    parser.add_argument("--weight_decay", type=float, default=0.045)
    parser.add_argument("--beta2", type=float, default=0.96)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--split_vae_encode", type=int, default=None,
                        help="Split VQ encoding into chunks to save VRAM")

    parser.add_argument("--output_dir", type=str,
                        default="/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/checkpoints")
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--max_checkpoints", type=int, default=5)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--resume_weights_only", action="store_true",
                        help="Only load model weights from checkpoint, reset optimizer/scheduler/step (for resolution change)")

    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for ~20-30%% training speedup (PyTorch 2.0+). "
                             "Currently INCOMPATIBLE with GLA naive fallback — do not enable "
                             "until fla bidirectional kernel is available.")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--text_max_length", type=int, default=256)

    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="EMA decay rate (MaskBit uses EMA for stable training)")
    parser.add_argument("--no_ema", action="store_true")

    parser.add_argument("--wandb_project", type=str, default="lightmgt")
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    # Inference sampling during training
    parser.add_argument("--sample_steps", type=int, default=1000,
                        help="Generate sample images every N steps (0 to disable)")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of sample images to generate")
    parser.add_argument("--sample_inference_steps", type=int, default=20,
                        help="MaskGIT unmasking steps for sample generation")
    parser.add_argument("--sample_cfg_scale", type=float, default=5.0,
                        help="CFG scale for sample generation")

    return parser.parse_args()


def get_cosine_schedule(optimizer, warmup_steps, total_steps, floor_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(floor_ratio, cosine)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def encode_images_to_tokens(images, vq_model, split_batch=None):
    """Online VQ encoding: [B, 3, H, W] -> [B, N] token IDs.

    Args:
        images: [B, 3, H, W] normalized to [-1, 1].
        vq_model: OpenMAGVIT2Wrapper instance.
        split_batch: Split into chunks of this size (saves VRAM for large batches).

    Returns:
        token_ids: [B, N] long tensor.
    """
    B = images.shape[0]
    if split_batch is None or split_batch >= B:
        token_ids, _ = vq_model.encode(images)
        return token_ids

    # Split encoding for memory efficiency
    chunks = []
    for i in range(0, B, split_batch):
        end = min(i + split_batch, B)
        ids, _ = vq_model.encode(images[i:end])
        chunks.append(ids)
    return torch.cat(chunks, dim=0)


SAMPLE_PROMPTS = [
    "A photo of a cute golden retriever puppy sitting in a sunny meadow",
    "A futuristic city skyline at sunset with flying cars and neon lights",
    "A bowl of ramen with steam rising, studio food photography",
]


@torch.no_grad()
def generate_samples(
    model, text_encoder, tokenizer, vq_model, scheduler,
    resolution, device, amp_dtype,
    prompts=None, num_samples=3, num_steps=20, cfg_scale=5.0,
):
    """Generate sample images using MaskGIT iterative unmasking.

    Returns list of PIL Images.
    """
    from PIL import Image as PILImage

    if prompts is None:
        prompts = SAMPLE_PROMPTS[:num_samples]
    else:
        prompts = prompts[:num_samples]

    raw_model = model.module if hasattr(model, "module") else model
    # Handle torch.compile wrapping
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    was_training = raw_model.training
    raw_model.eval()

    B = len(prompts)
    img_h = img_w = resolution // 16
    num_tokens = img_h * img_w
    mask_token_id = scheduler.config.mask_token_id

    # Encode text
    tokens = tokenizer(
        prompts, padding="max_length", max_length=256,
        truncation=True, return_tensors="pt",
    ).to(device)

    text_out = text_encoder(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        output_hidden_states=True,
    )
    text_hidden = text_out.hidden_states[-1].to(torch.bfloat16)
    seq_lengths = tokens.attention_mask.sum(dim=1).long() - 1
    text_pooled = text_hidden[torch.arange(B, device=device), seq_lengths]
    text_pooled = torch.nn.functional.normalize(text_pooled, dim=-1) * (text_pooled.shape[-1] ** 0.5)

    # Null embeddings for CFG
    null_hidden = torch.zeros_like(text_hidden)
    null_pooled = torch.zeros_like(text_pooled)

    # Initialize all [MASK]
    sample = torch.full((B, num_tokens), mask_token_id, dtype=torch.long, device=device)

    # Setup scheduler
    scheduler.set_timesteps(num_steps, temperature=(2.0, 0.0), device=device)

    # Iterative unmasking with CFG
    for i, timestep in enumerate(scheduler.timesteps):
        t = timestep.float() / num_steps * 1000

        # Double forward for CFG
        double_tokens = sample.repeat(2, 1)
        double_text = torch.cat([text_hidden, null_hidden], dim=0)
        double_pooled = torch.cat([text_pooled, null_pooled], dim=0)
        double_t = t.expand(B * 2)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            output = raw_model(
                token_ids=double_tokens,
                text_hidden_states=double_text,
                text_pooled=double_pooled,
                timestep=double_t,
                img_h=img_h,
                img_w=img_w,
            )

        cond_hidden = output.hidden_states[:B]
        uncond_hidden = output.hidden_states[B:]
        guided = uncond_hidden + cfg_scale * (cond_hidden - uncond_hidden)

        step_out = scheduler.step(
            model_output_hidden=guided,
            timestep=timestep,
            sample=sample,
            gen_head=raw_model.gen_head,
        )
        sample = step_out.prev_sample

    # Decode tokens to images via VQ
    images = vq_model.decode_tokens(sample, img_h, img_w)  # [-1, 1]
    images = ((images + 1) / 2 * 255).clamp(0, 255).byte()

    # Convert to PIL
    pil_images = []
    for img in images:
        img_np = img.permute(1, 2, 0).cpu().numpy()
        pil_images.append(PILImage.fromarray(img_np))

    if was_training:
        raw_model.train()

    return pil_images, prompts


def main():
    args = parse_args()

    # --- DDP setup ---
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(42)

    if is_main(rank):
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Load text encoder (frozen) ---
    if is_main(rank):
        print("Loading Qwen3.5-0.8B text encoder...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_encoder = AutoModelForCausalLM.from_pretrained(
        args.text_encoder, torch_dtype=torch.bfloat16,
    )
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    text_encoder = text_encoder.to(device)

    # --- Load Open-MAGVIT2-PT 262K (frozen, 18-bit LFQ) ---
    if is_main(rank):
        print("Loading Open-MAGVIT2-PT 262K (18-bit LFQ)...")
    from lightmgt.vq_wrapper import OpenMAGVIT2Wrapper
    vq_model = OpenMAGVIT2Wrapper(
        ckpt_path=args.vq_ckpt,
        seed_voken_dir=args.seed_voken_dir,
    ).to(device)

    # --- Build transformer ---
    if is_main(rank):
        print("Building LightMGT transformer...")
    model = LightMGTTransformer(
        text_hidden_size=1024,
        label_smoothing=args.label_smoothing,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing = True

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main(rank):
        print(f"LightMGT: {num_params:.1f}M parameters")

    # Move to device
    model = model.to(device)

    # torch.compile: must be BEFORE DDP wrapping
    if args.compile:
        if is_main(rank):
            print(f"Compiling model with torch.compile(mode='{args.compile_mode}')...")
        model = torch.compile(model, mode=args.compile_mode)
        if is_main(rank):
            print("Model compiled (first step will be slow due to JIT compilation)")

    # DDP wrapping (after compile)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    if is_main(rank):
        print(f"DDP: find_unused_parameters=False, {world_size} GPU(s)")

    # --- EMA (MaskBit recipe: stabilizes training and improves convergence) ---
    ema_model = None
    if not args.no_ema:
        import copy
        raw = model.module if hasattr(model, "module") else model
        ema_model = copy.deepcopy(raw).to(device)
        ema_model.eval()
        ema_model.requires_grad_(False)
        if is_main(rank):
            print(f"EMA enabled: decay={args.ema_decay}")

    scheduler = MaskGITScheduler(mask_token_id=262144)

    # --- Dataset (online encoding, multi-scale) ---
    # NOTE: Do NOT use DistributedSampler — BucketBatchSampler already
    # handles rank/world_size distribution for multi-scale batching.
    if is_main(rank):
        print(f"Building dataset: {args.data_dir}, resolution={args.resolution}, multi_scale={args.multi_scale}")
    dataloader = build_dataloader(
        json_dir=args.data_dir if len(args.data_dir) > 1 else args.data_dir[0],
        tokenizer=tokenizer,
        center_resolution=args.resolution,
        multi_scale=args.multi_scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tar_pattern=args.tar_pattern,
        text_max_length=args.text_max_length,
        rank=rank,
        world_size=world_size,
    )

    # --- Optimizer (MaskBit recipe: uniform LR, AdamW) ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, args.beta2),
        weight_decay=args.weight_decay,
    )
    lr_scheduler = get_cosine_schedule(optimizer, args.warmup_steps, args.max_steps)

    # --- Mixed precision: manual autocast + GradScaler ---
    # bf16 doesn't need loss scaling, but GradScaler with enabled=False
    # keeps the code path clean for potential fp16 use
    use_bf16 = (args.mixed_precision == "bf16" and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(not use_bf16))
    if is_main(rank):
        print(f"Mixed precision: {amp_dtype}, GradScaler enabled={not use_bf16}")

    # --- Wandb (main process only) ---
    wandb_run = None
    if not args.no_wandb and is_main(rank):
        import wandb
        wandb_entity = os.environ.get("WANDB_ENTITY", None)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=wandb_entity,
            name=args.wandb_run or f"phase1_{args.resolution}px",
            config=vars(args),
        )

    # --- Resume ---
    global_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        raw_model = model.module if hasattr(model, "module") else model
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
            # Strip torch.compile's _orig_mod. prefix if present
            cleaned = {}
            stripped = 0
            for k, v in sd.items():
                if k.startswith("_orig_mod."):
                    cleaned[k[len("_orig_mod."):]] = v
                    stripped += 1
                else:
                    cleaned[k] = v
            if stripped > 0 and is_main(rank):
                print(f"Stripped _orig_mod. prefix from {stripped}/{len(sd)} keys")
            missing, unexpected = raw_model.load_state_dict(cleaned, strict=False)
            if is_main(rank) and missing:
                print(f"WARNING: {len(missing)} missing keys in checkpoint: {missing[:5]}")
            if is_main(rank) and unexpected:
                print(f"WARNING: {len(unexpected)} unexpected keys in checkpoint: {unexpected[:5]}")
        if args.resume_weights_only:
            if is_main(rank):
                print(f"Loaded weights only from: {args.resume_from} (optimizer/scheduler/step reset)")
        else:
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                lr_scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler"])
            global_step = ckpt.get("step", 0)
            if is_main(rank):
                print(f"Resumed from step {global_step}: {args.resume_from}")
        del ckpt

    # --- Training loop ---
    grad_accum = args.gradient_accumulation_steps
    if is_main(rank):
        eff_batch = args.batch_size * world_size * grad_accum
        print(f"\nPhase 1 Training:")
        print(f"  Resolution: {args.resolution}px, multi_scale={args.multi_scale}")
        print(f"  Effective batch: {args.batch_size} x {world_size} GPU x {grad_accum} accum = {eff_batch}")
        print(f"  Steps: {global_step}/{args.max_steps}")
        print(f"  LR: {args.learning_rate}, warmup={args.warmup_steps}")
        print()

    model.train()
    epoch = 0
    data_iter = iter(dataloader)
    running_loss = 0.0
    start_time = time.time()
    grad_norm = 0.0
    last_logits = None
    last_target = None
    last_mask = None
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    # EMA metrics for stable wandb curves (single-batch acc is too noisy)
    ema_acc = 0.0
    ema_g1 = 0.0
    ema_g2 = 0.0
    ema_beta = 0.95  # ~20-step effective window

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "set_epoch"):
                dataloader.batch_sampler.set_epoch(epoch)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        prompt_ids = batch["prompt_input_ids"].to(device)
        prompt_mask = batch["prompt_attention_mask"].to(device)
        target_size = batch["target_size"]
        img_h = target_size[0].item() // 16
        img_w = target_size[1].item() // 16

        # --- Data diagnostic (first batch only) ---
        if global_step == 0 and micro_step == 0 and is_main(rank) and not getattr(main, '_diag_done', False):
            main._diag_done = True
            print(f"\n=== DATA DIAGNOSTIC (first batch) ===")
            print(f"  images: shape={images.shape}, range=[{images.min():.3f}, {images.max():.3f}], std={images.std():.3f}")
            print(f"  prompt_ids: shape={prompt_ids.shape}, non-pad tokens={prompt_mask.sum(1).float().mean():.1f}")
            per_img_std = images.view(images.shape[0], -1).std(dim=1)
            gray_count = (per_img_std < 0.01).sum().item()
            print(f"  gray/dummy images: {gray_count}/{images.shape[0]} (should be 0)")
            print(f"  per-image std: min={per_img_std.min():.4f}, max={per_img_std.max():.4f}, mean={per_img_std.mean():.4f}")
            print(f"  target_size: {target_size[0].item()}x{target_size[1].item()}")
            print(f"=== END DIAGNOSTIC ===\n")

        # --- Online VQ encoding (frozen Open-MAGVIT2 262K) ---
        vq_tokens = encode_images_to_tokens(
            images, vq_model, split_batch=args.split_vae_encode
        )

        # VQ token diagnostic (first batch)
        if global_step == 0 and micro_step == 0 and is_main(rank) and not getattr(main, '_vq_diag_done', False):
            main._vq_diag_done = True
            print(f"  VQ tokens: shape={vq_tokens.shape}, range=[{vq_tokens.min()}, {vq_tokens.max()}], "
                  f"unique={vq_tokens.unique().numel()}/262144")

        # --- Encode text (frozen Qwen3.5-0.8B) ---
        B = images.shape[0]
        with torch.no_grad():
            text_out = text_encoder(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                output_hidden_states=True,
            )
            text_hidden = text_out.hidden_states[-1].to(torch.bfloat16)
            # Last-token pooling: for causal LMs, the last non-pad token has
            # attended to all preceding tokens and carries the sequence summary.
            seq_lengths = prompt_mask.sum(dim=1).long() - 1  # [B]
            text_pooled = text_hidden[torch.arange(B, device=device), seq_lengths]  # [B, D]
            text_pooled = torch.nn.functional.normalize(text_pooled, dim=-1) * (text_pooled.shape[-1] ** 0.5)

        # --- CFG dropout ---
        if args.cfg_dropout > 0:
            cfg_mask = torch.rand(B, device=device) < args.cfg_dropout
            if cfg_mask.any():
                text_hidden[cfg_mask] = 0.0
                text_pooled[cfg_mask] = 0.0

        # --- NaN diagnostic (first 5 steps, ALL ranks) ---
        if global_step < 5 and micro_step < 5 * grad_accum:
            has_nan_vq = torch.isnan(vq_tokens.float()).any().item()
            has_nan_text = torch.isnan(text_hidden).any().item()
            has_nan_pooled = torch.isnan(text_pooled).any().item()
            text_max = text_hidden.abs().max().item()
            pooled_max = text_pooled.abs().max().item()
            param_nan = any(torch.isnan(p).any().item() for p in model.parameters())
            param_max = max(p.abs().max().item() for p in model.parameters())
            print(f"[NaN check R{rank} step {global_step} micro {micro_step}] "
                  f"vq_nan={has_nan_vq} text_nan={has_nan_text} pooled_nan={has_nan_pooled} "
                  f"text_max={text_max:.1f} pooled_max={pooled_max:.1f} "
                  f"param_nan={param_nan} param_max={param_max:.1f}")

        # --- MaskGIT masking ---
        masked_tokens, mask = scheduler.add_noise(vq_tokens)
        mask_ratio = mask.float().mean(dim=-1)
        timestep = mask_ratio * 1000

        # --- Gradient accumulation with no_sync for intermediate micro-steps ---
        is_last_micro = ((micro_step + 1) % grad_accum == 0)
        ctx = model.no_sync() if (world_size > 1 and not is_last_micro) else contextlib.nullcontext()

        with ctx:
            # --- Forward with manual autocast ---
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                output = model(
                    token_ids=masked_tokens,
                    text_hidden_states=text_hidden,
                    text_pooled=text_pooled,
                    timestep=timestep,
                    img_h=img_h,
                    img_w=img_w,
                    target_ids=vq_tokens,
                    mask=mask,
                )

                loss = output.loss

            # NaN diagnostic after forward (ALL ranks)
            if global_step < 5 and torch.isnan(loss):
                print(f"[NaN DETECTED R{rank} step {global_step} micro {micro_step}] loss={loss.item()}")

            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()

        micro_step += 1

        # Save last batch info for metrics (cheap, no_grad)
        last_logits = output.logits
        last_target = vq_tokens
        last_mask = mask

        # --- Optimizer step at accumulation boundary ---
        if is_last_micro:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running_loss += loss.detach().item()

            # --- EMA update ---
            if ema_model is not None:
                raw_model = model.module if hasattr(model, "module") else model
                decay = args.ema_decay
                with torch.no_grad():
                    for ema_p, model_p in zip(ema_model.parameters(), raw_model.parameters()):
                        ema_p.lerp_(model_p, 1.0 - decay)

            # --- Compute per-step acc (for EMA, cheap) ---
            if is_main(rank) and last_logits is not None:
                with torch.no_grad():
                    group_vocab = 512
                    _lg1 = last_logits[0]
                    _lg2 = last_logits[1]
                    _pg1 = _lg1[last_mask].argmax(-1)
                    _pg2 = _lg2[last_mask].argmax(-1)
                    _preds = _pg1 * group_vocab + _pg2
                    _tg1 = last_target[last_mask] // group_vocab
                    _tg2 = last_target[last_mask] % group_vocab
                    _targets = last_target[last_mask]

                    _acc = (_preds == _targets).float().mean().item()
                    _ag1 = (_pg1 == _tg1).float().mean().item()
                    _ag2 = (_pg2 == _tg2).float().mean().item()

                ema_acc = ema_beta * ema_acc + (1 - ema_beta) * _acc
                ema_g1 = ema_beta * ema_g1 + (1 - ema_beta) * _ag1
                ema_g2 = ema_beta * ema_g2 + (1 - ema_beta) * _ag2

            # --- Logging ---
            if global_step % args.log_steps == 0 and is_main(rank):
                avg_loss = running_loss / args.log_steps
                elapsed = time.time() - start_time
                steps_per_sec = args.log_steps / elapsed
                lr = optimizer.param_groups[0]["lr"]

                # Use the latest per-step values for instantaneous metrics
                with torch.no_grad():
                    group_vocab = 512
                    logits_g1 = last_logits[0]
                    logits_g2 = last_logits[1]
                    pred_g1 = logits_g1[last_mask].argmax(-1)
                    pred_g2 = logits_g2[last_mask].argmax(-1)
                    preds = pred_g1 * group_vocab + pred_g2
                    targets_flat = last_target[last_mask]

                    acc = (preds == targets_flat).float().mean().item()
                    acc_g1 = (pred_g1 == (last_target[last_mask] // group_vocab)).float().mean().item()
                    acc_g2 = (pred_g2 == (last_target[last_mask] % group_vocab)).float().mean().item()
                    diversity = preds.unique().numel()
                    avg_mask_ratio = last_mask.float().mean().item()

                # Bias-correct EMA for early steps
                bc = 1.0 - ema_beta ** global_step if global_step > 0 else 1.0
                ema_acc_bc = ema_acc / bc
                ema_g1_bc = ema_g1 / bc
                ema_g2_bc = ema_g2 / bc

                gn = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                res_tag = f"{target_size[0].item()}x{target_size[1].item()}"

                print(f"step {global_step:>7d} | loss {avg_loss:.4f} | "
                      f"acc {acc:.4f} (ema {ema_acc_bc:.4f}) g1 {acc_g1:.4f} g2 {acc_g2:.4f} | "
                      f"div {diversity}/262144 | mask {avg_mask_ratio:.2f} | "
                      f"gnorm {gn:.2f} | lr {lr:.2e} | {steps_per_sec:.2f} s/s | "
                      f"res {res_tag}")
                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss": avg_loss,
                        "train/top1_acc": acc,
                        "train/top1_acc_ema": ema_acc_bc,
                        "train/acc_g1": acc_g1,
                        "train/acc_g1_ema": ema_g1_bc,
                        "train/acc_g2": acc_g2,
                        "train/acc_g2_ema": ema_g2_bc,
                        "train/diversity": diversity,
                        "train/mask_ratio": avg_mask_ratio,
                        "train/grad_norm": gn,
                        "train/lr": lr,
                        "train/steps_per_sec": steps_per_sec,
                        f"train/acc_by_res/{res_tag}": acc,
                    }, step=global_step)
                running_loss = 0.0
                start_time = time.time()

            # --- Checkpoint saving ---
            if global_step % args.save_steps == 0 and is_main(rank):
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                raw_model = model.module if hasattr(model, "module") else model
                save_dict = {
                    "step": global_step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                }
                if ema_model is not None:
                    save_dict["ema_state_dict"] = ema_model.state_dict()
                torch.save(save_dict, os.path.join(save_path, "training_state.pt"))
                print(f"Saved checkpoint: {save_path}")

                # Prune old checkpoints — only consider those from current run
                # (weights-only resume resets step to 0, so old checkpoints may
                # have higher step numbers and should not participate in pruning)
                import shutil
                ckpts = sorted(
                    [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
                    key=lambda x: int(x.split("-")[1]),
                )
                current_run_ckpts = [d for d in ckpts if int(d.split("-")[1]) <= global_step]
                while len(current_run_ckpts) > args.max_checkpoints:
                    old = current_run_ckpts.pop(0)
                    shutil.rmtree(os.path.join(args.output_dir, old))
                    print(f"Pruned old checkpoint: {old}")

            # --- Sample generation & wandb image logging ---
            if (args.sample_steps > 0
                    and global_step % args.sample_steps == 0
                    and is_main(rank)):
                print(f"Generating {args.num_samples} sample images at step {global_step}...")
                try:
                    sample_start = time.time()
                    pil_images, sample_prompts = generate_samples(
                        model=model,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        vq_model=vq_model,
                        scheduler=scheduler,
                        resolution=args.resolution,
                        device=device,
                        amp_dtype=amp_dtype,
                        num_samples=args.num_samples,
                        num_steps=args.sample_inference_steps,
                        cfg_scale=args.sample_cfg_scale,
                    )
                    sample_time = time.time() - sample_start
                    print(f"Sample generation done in {sample_time:.1f}s")

                    if wandb_run is not None:
                        import wandb
                        wandb_images = []
                        for img, prompt in zip(pil_images, sample_prompts):
                            wandb_images.append(
                                wandb.Image(img, caption=f"step={global_step}: {prompt[:80]}")
                            )
                        wandb_run.log({"samples/generated": wandb_images})
                        print(f"Uploaded {len(wandb_images)} images to wandb")
                except Exception as e:
                    print(f"WARNING: Sample generation failed: {e}")
                    import traceback
                    traceback.print_exc()

    # --- Dump data load errors for offline repair ---
    if is_main(rank) and hasattr(dataloader, "dataset"):
        ds = dataloader.dataset
        errors = ds.get_load_errors() if hasattr(ds, "get_load_errors") else []
        if errors:
            import json as _json
            err_path = os.path.join(args.output_dir, "data_load_errors.json")
            try:
                with open(err_path, "w") as f:
                    _json.dump(errors, f, indent=2, ensure_ascii=False)
                print(f"Dumped {len(errors)} data load errors to {err_path}")
            except Exception as e:
                print(f"WARNING: Failed to dump load errors: {e}")

    # --- Cleanup ---
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_ddp()
    if is_main(rank):
        print(f"\nTraining complete! {global_step} steps.")


if __name__ == "__main__":
    main()
