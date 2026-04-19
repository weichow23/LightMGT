"""Evaluation utilities for LightMGT.

Model initialization, metric helpers (L1, CLIP-I, CLIP-T).
"""

import os
import sys
import json

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def init_lightmgt(ckpt_path, device="cuda:0", dtype=torch.bfloat16,
                  text_encoder="Qwen/Qwen3.5-0.8B",
                  vq_ckpt="/mnt/bn/search-auto-eval-v2/zhouwei/nextmgt/tokenizer_ckpts/pretrain256_262144.ckpt",
                  seed_voken_dir="/mnt/bn/search-auto-eval-v2/zhouwei/SEED-Voken"):
    """Initialize LightMGT pipeline for evaluation.

    Args:
        ckpt_path: Path to LightMGT transformer checkpoint.
        device: CUDA device.
        dtype: Model dtype (bf16 recommended).
        text_encoder: HF text encoder name/path.
        vq_ckpt: Path to Open-MAGVIT2-PT 262K checkpoint.

    Returns:
        LightMGTPipeline instance.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lightmgt.modeling_lightmgt import LightMGTTransformer
    from lightmgt.scheduler_maskgit import MaskGITScheduler
    from lightmgt.pipeline_lightmgt import LightMGTPipeline
    from lightmgt.vq_wrapper import OpenMAGVIT2Wrapper

    # Text encoder
    tokenizer = AutoTokenizer.from_pretrained(text_encoder)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_enc = AutoModelForCausalLM.from_pretrained(text_encoder, dtype=dtype)
    text_enc.eval().requires_grad_(False).to(device)

    # VQ-VAE (Open-MAGVIT2-PT 262K, LFQ 18-bit)
    vq_model = OpenMAGVIT2Wrapper(ckpt_path=vq_ckpt, seed_voken_dir=seed_voken_dir).to(device)

    # Transformer
    model = LightMGTTransformer(text_hidden_size=1024)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "model" in state:
            state = state["model"]
        # Strip torch.compile _orig_mod. prefix if present
        cleaned = {}
        for k, v in state.items():
            cleaned[k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"WARNING: {len(missing)} missing keys: {missing[:5]}")
        print(f"Loaded LightMGT from {ckpt_path}")
    model.eval().to(device, dtype)

    # Scheduler (262144 = Open-MAGVIT2 LFQ codebook size)
    scheduler = MaskGITScheduler(mask_token_id=262144)

    # Pipeline
    pipe = LightMGTPipeline(
        transformer=model,
        scheduler=scheduler,
        text_encoder=text_enc,
        tokenizer=tokenizer,
        vq_model=vq_model,
    )
    pipe = pipe.to(device)

    return pipe


# Resolution buckets (same as training, all multiples of 16)
ASPECT_RATIO_1024 = [
    (1024, 1024), (1088, 960), (960, 1088), (1152, 896), (896, 1152),
    (1216, 832), (832, 1216), (1280, 800), (800, 1280), (1344, 768),
    (768, 1344), (1408, 704), (704, 1408), (1472, 672), (672, 1472),
    (1536, 640), (640, 1536),
]

import math

def find_nearest_bucket(h, w, buckets=None):
    if buckets is None:
        buckets = ASPECT_RATIO_1024
    ar = w / max(h, 1)
    best = buckets[0]
    best_d = float("inf")
    for bh, bw in buckets:
        d = abs(math.log(max(ar, 1e-6)) - math.log(bw / bh))
        if d < best_d:
            best_d = d
            best = (bh, bw)
    return best


# GPT-4o API for ByteDance internal
GPT_API_URL = "https://search-va.byteintl.net/gpt/openapi/online/multimodal/crawl"
GPT_API_KEY = "I5RRgGh9v5JX40yYqDRg4uD2z4UosDBx"
GPT_MODEL = "gpt-4o-2024-08-06"


def call_gpt4o_vision(prompt, images=None, max_tokens=500, temperature=0.0):
    """Call GPT-4o with optional images via ByteDance API.

    Args:
        prompt: Text prompt.
        images: List of PIL images or file paths.
        max_tokens: Max response tokens.

    Returns:
        Response text string.
    """
    import base64
    import requests

    messages = []
    content = []

    if images:
        for img in images:
            if isinstance(img, str):
                with open(img, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
            else:
                import io
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})

    payload = {
        "model": GPT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {"Authorization": f"Bearer {GPT_API_KEY}", "Content-Type": "application/json"}

    for retry in range(3):
        try:
            resp = requests.post(GPT_API_URL, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if retry == 2:
                raise
            import time
            time.sleep(2 ** retry)

    return ""
