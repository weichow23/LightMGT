"""Sanity check: verify LightMGT model builds correctly and has expected param count.

Usage:
    python scripts/sanity_check.py
"""

import sys
import torch

sys.path.insert(0, ".")

from lightmgt.modeling_lightmgt import LightMGTTransformer, count_parameters


def main():
    print("=" * 60)
    print("LightMGT Sanity Check")
    print("=" * 60)

    # Build model
    print("\n[1] Building model...")
    model = LightMGTTransformer(
        hidden_size=1024,
        num_double_blocks=4,
        num_single_gla_blocks=14,
        num_single_softmax_blocks=6,
        num_attention_heads=16,
        head_dim=64,
        codebook_size=262144,
        mask_token_id=262144,
        vocab_size=262145,
        num_lfq_bits=18,
        gen_head_groups=2,
        gen_head_vocab=512,
        text_hidden_size=1024,
        text_max_length=256,
        rope_axes_dim=(8, 28, 28),
    )

    # Parameter count
    print("\n[2] Parameter counts:")
    counts = count_parameters(model)
    for name, count in sorted(counts.items()):
        print(f"  {name:35s} {count:>12,d} ({count/1e6:.1f}M)")

    total = counts["total"]
    print(f"\n  TOTAL: {total:,d} ({total/1e6:.1f}M)")
    print(f"  Target: ~384M (mlp_ratio=2.6875)")

    # Forward pass
    print("\n[3] Forward pass test...")
    B, N, T = 2, 256, 64  # 256px: 16x16=256 tokens, 64 text tokens
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Mix of real tokens (0..262143) and mask tokens (262144)
    token_ids = torch.randint(0, 262144, (B, N), device=device)
    token_ids[:, ::3] = 262144  # mask every 3rd token
    text_hidden = torch.randn(B, T, 1024, device=device)
    text_pooled = torch.randn(B, 1024, device=device)
    timestep = torch.rand(B, device=device) * 1000
    target_ids = torch.randint(0, 262144, (B, N), device=device)
    mask = torch.rand(B, N) > 0.5

    with torch.no_grad():
        output = model(
            token_ids=token_ids,
            text_hidden_states=text_hidden,
            text_pooled=text_pooled,
            timestep=timestep,
            img_h=16,
            img_w=16,
            target_ids=target_ids,
            mask=mask,
        )

    print(f"  Logits: {len(output.logits)} groups, each {output.logits[0].shape}")
    print(f"  Hidden: {output.hidden_states.shape}")
    print(f"  Loss: {output.loss.item():.4f}")

    # Gen head sampling
    print("\n[4] Gen head sampling test...")
    pred_ids, confidence = model.gen_head.sample(output.hidden_states)
    print(f"  Pred IDs: {pred_ids.shape}, range [{pred_ids.min()}, {pred_ids.max()}]")
    print(f"  Confidence: {confidence.shape}, mean={confidence.mean():.4f}")

    # Scheduler test
    print("\n[5] Scheduler test...")
    from lightmgt.scheduler_maskgit import MaskGITScheduler

    scheduler = MaskGITScheduler(mask_token_id=262144)

    # Training mask
    clean_tokens = torch.randint(0, 262144, (B, N))
    masked_tokens, mask = scheduler.add_noise(clean_tokens)
    mask_ratio = mask.float().mean()
    print(f"  Masked ratio: {mask_ratio:.3f}")
    print(f"  Mask token count: {(masked_tokens == 262144).sum()}")

    # RoPE test
    print("\n[6] RoPE test...")
    from lightmgt.modeling_rope import build_position_ids, get_3d_rotary_embedding

    pos_ids = build_position_ids(text_len=T, img_h=16, img_w=16)
    print(f"  Position IDs: {pos_ids.shape}")
    cos, sin = get_3d_rotary_embedding(pos_ids, axes_dim=(8, 28, 28))
    print(f"  cos: {cos.shape}, sin: {sin.shape}")

    # GLA test
    print("\n[7] GLA test...")
    from lightmgt.modeling_gla import GLAAttention

    gla = GLAAttention(hidden_size=1024, num_heads=16, head_dim=64)
    x = torch.randn(2, 256, 1024)
    y = gla(x)
    print(f"  Input: {x.shape} -> Output: {y.shape}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
