"""DreamBench++ evaluation for LightMGT.

Measures subject-driven generation quality via:
  - CLIP-T: text alignment (cosine sim between generated image & prompt in CLIP space)
  - CLIP-I: image alignment (cosine sim between generated & reference in CLIP space)
  - DINO:   subject preservation (DINOv2 feature similarity)

Dataset: yuangpeng/dreambench_plus_plus
  Remote: /mnt/hdfs/weichow/maskedit/eval_data/dreambench_plus_plus/

Usage:
    python -m eval.dreambench_pp.run --mode infer --ckpt <path> --results_dir <dir>
    python -m eval.dreambench_pp.run --mode score --results_dir <dir>
    python -m eval.dreambench_pp.run --mode stat  --results_dir <dir>
    python -m eval.dreambench_pp.run --mode all   --ckpt <path> --results_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from eval.utils import init_lightmgt, find_nearest_bucket, ASPECT_RATIO_1024

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATA_ROOT = "/mnt/hdfs/weichow/maskedit/eval_data/dreambench_plus/data"


def load_dataset(data_root: str = DATA_ROOT) -> List[Dict]:
    """Load DreamBench++ metadata from JSON or parquet.

    Expected fields per sample:
        - prompt (str)
        - image_path (str): path to reference subject image
        - category (str): e.g. live_subject, non-live_subject
        - sample_id / id (str|int): unique identifier

    Returns:
        List of sample dicts.
    """
    json_path = os.path.join(data_root, "metadata.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            samples = json.load(f)
        if isinstance(samples, dict) and "data" in samples:
            samples = samples["data"]
        return samples

    # Try parquet (HF datasets format)
    parquet_candidates = list(Path(data_root).rglob("*.parquet"))
    if parquet_candidates:
        import pandas as pd
        dfs = [pd.read_parquet(p) for p in parquet_candidates]
        df = pd.concat(dfs, ignore_index=True)
        return df.to_dict(orient="records")

    # Try JSONL
    jsonl_path = os.path.join(data_root, "metadata.jsonl")
    if os.path.exists(jsonl_path):
        samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    raise FileNotFoundError(
        f"No metadata found in {data_root}. "
        "Expected metadata.json, metadata.jsonl, or *.parquet files."
    )


def _get_sample_id(sample: Dict) -> str:
    """Extract a unique string ID from a sample dict."""
    for key in ("sample_id", "id", "index", "idx"):
        if key in sample:
            return str(sample[key])
    return str(hash(json.dumps(sample, sort_keys=True, default=str)))


def _get_reference_image_path(sample: Dict, data_root: str) -> str:
    """Resolve the reference image path."""
    for key in ("image_path", "reference_image", "image", "ref_image"):
        if key in sample and sample[key]:
            p = sample[key]
            if os.path.isabs(p):
                return p
            return os.path.join(data_root, p)
    raise KeyError(f"No image path found in sample: {list(sample.keys())}")


def _get_prompt(sample: Dict) -> str:
    """Extract the generation prompt."""
    for key in ("prompt", "text", "caption"):
        if key in sample and sample[key]:
            return sample[key]
    raise KeyError(f"No prompt found in sample: {list(sample.keys())}")


def _get_category(sample: Dict) -> str:
    """Extract category, defaulting to 'unknown'."""
    for key in ("category", "type", "class"):
        if key in sample and sample[key]:
            return str(sample[key])
    return "unknown"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def infer(args):
    """Generate images for all DreamBench++ samples."""
    print("Loading dataset...")
    samples = load_dataset(args.data_root)
    print(f"Loaded {len(samples)} samples from {args.data_root}")

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    gen_dir = os.path.join(results_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    # Resume support: collect already-generated sample IDs
    done_ids = set()
    for fname in os.listdir(gen_dir):
        if fname.endswith(".png"):
            done_ids.add(fname.replace(".png", ""))

    todo = []
    for s in samples:
        sid = _get_sample_id(s)
        if sid not in done_ids:
            todo.append(s)
    print(f"Skipping {len(samples) - len(todo)} already generated, {len(todo)} remaining.")

    if not todo:
        print("All samples generated. Skipping inference.")
        return

    # Initialize model
    print("Initializing LightMGT pipeline...")
    pipe = init_lightmgt(args.ckpt, device=args.device)

    # Save metadata mapping for scoring later
    meta_path = os.path.join(results_dir, "metadata.json")
    if not os.path.exists(meta_path):
        meta = []
        for s in samples:
            meta.append({
                "sample_id": _get_sample_id(s),
                "prompt": _get_prompt(s),
                "category": _get_category(s),
                "reference_image": _get_reference_image_path(s, args.data_root),
            })
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    for sample in tqdm(todo, desc="Generating"):
        sid = _get_sample_id(sample)
        prompt = _get_prompt(sample)

        # Load reference image to determine aspect ratio
        ref_path = _get_reference_image_path(sample, args.data_root)
        ref_img = Image.open(ref_path).convert("RGB")
        h, w = ref_img.height, ref_img.width
        bucket_h, bucket_w = find_nearest_bucket(h, w)

        images = pipe(
            prompt=prompt,
            height=bucket_h,
            width=bucket_w,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )

        out_path = os.path.join(gen_dir, f"{sid}.png")
        images[0].save(out_path)

    print(f"Inference complete. Results saved to {gen_dir}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _load_clip_model(device: str):
    """Load OpenCLIP ViT-L-14 for CLIP-T and CLIP-I scores."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    return model, preprocess, tokenizer


def _load_dino_model(device: str):
    """Load DINOv2 ViT-L/14 for subject preservation score."""
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    dino = dino.eval().to(device)

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return dino, preprocess


@torch.no_grad()
def _clip_image_features(model, preprocess, image: Image.Image, device: str) -> torch.Tensor:
    """Extract normalized CLIP image features."""
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    features = model.encode_image(img_tensor)
    features = features / features.norm(dim=-1, keepdim=True)
    return features


@torch.no_grad()
def _clip_text_features(model, tokenizer, text: str, device: str) -> torch.Tensor:
    """Extract normalized CLIP text features."""
    tokens = tokenizer([text]).to(device)
    features = model.encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)
    return features


@torch.no_grad()
def _dino_features(model, preprocess, image: Image.Image, device: str) -> torch.Tensor:
    """Extract normalized DINOv2 features."""
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    features = model(img_tensor)
    features = features / features.norm(dim=-1, keepdim=True)
    return features


def score(args):
    """Compute CLIP-T, CLIP-I, and DINO scores for generated images."""
    results_dir = args.results_dir
    gen_dir = os.path.join(results_dir, "generated")
    scores_path = os.path.join(results_dir, "scores.json")

    # Load metadata
    meta_path = os.path.join(results_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata not found at {meta_path}. Run inference first."
        )
    with open(meta_path) as f:
        metadata = json.load(f)

    # Resume support: load existing scores
    existing_scores: Dict[str, Dict] = {}
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            scored_list = json.load(f)
        for entry in scored_list:
            existing_scores[entry["sample_id"]] = entry

    todo = []
    for m in metadata:
        sid = m["sample_id"]
        gen_path = os.path.join(gen_dir, f"{sid}.png")
        if sid in existing_scores:
            continue
        if not os.path.exists(gen_path):
            print(f"Warning: generated image missing for {sid}, skipping.")
            continue
        todo.append(m)

    print(f"Scoring: {len(todo)} new samples, {len(existing_scores)} already scored.")

    if not todo:
        print("All samples scored. Skipping.")
        return

    device = args.device
    print("Loading CLIP model (OpenCLIP ViT-L-14)...")
    clip_model, clip_preprocess, clip_tokenizer = _load_clip_model(device)
    print("Loading DINOv2 model...")
    dino_model, dino_preprocess = _load_dino_model(device)

    all_scores = list(existing_scores.values())

    for m in tqdm(todo, desc="Scoring"):
        sid = m["sample_id"]
        prompt = m["prompt"]
        category = m["category"]
        ref_path = m["reference_image"]

        gen_path = os.path.join(gen_dir, f"{sid}.png")
        gen_img = Image.open(gen_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")

        # CLIP-T: generated image vs text prompt
        gen_clip_feat = _clip_image_features(clip_model, clip_preprocess, gen_img, device)
        text_clip_feat = _clip_text_features(clip_model, clip_tokenizer, prompt, device)
        clip_t = (gen_clip_feat @ text_clip_feat.T).item()

        # CLIP-I: generated image vs reference image
        ref_clip_feat = _clip_image_features(clip_model, clip_preprocess, ref_img, device)
        clip_i = (gen_clip_feat @ ref_clip_feat.T).item()

        # DINO: generated vs reference
        gen_dino_feat = _dino_features(dino_model, dino_preprocess, gen_img, device)
        ref_dino_feat = _dino_features(dino_model, dino_preprocess, ref_img, device)
        dino_score = (gen_dino_feat @ ref_dino_feat.T).item()

        entry = {
            "sample_id": sid,
            "prompt": prompt,
            "category": category,
            "clip_t": clip_t,
            "clip_i": clip_i,
            "dino": dino_score,
        }
        all_scores.append(entry)

        # Incremental save every 50 samples
        if len(all_scores) % 50 == 0:
            with open(scores_path, "w") as f:
                json.dump(all_scores, f, indent=2)

    # Final save
    with open(scores_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    print(f"Scoring complete. {len(all_scores)} scores saved to {scores_path}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def stat(args):
    """Print per-category and overall DreamBench++ scores."""
    scores_path = os.path.join(args.results_dir, "scores.json")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Scores file not found: {scores_path}. Run scoring first.")

    with open(scores_path) as f:
        scores = json.load(f)

    # Aggregate by category
    from collections import defaultdict
    cat_scores: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"clip_t": [], "clip_i": [], "dino": []}
    )

    for entry in scores:
        cat = entry["category"]
        cat_scores[cat]["clip_t"].append(entry["clip_t"])
        cat_scores[cat]["clip_i"].append(entry["clip_i"])
        cat_scores[cat]["dino"].append(entry["dino"])

    # Print table
    header = f"{'Category':<25} {'N':>5} {'CLIP-T':>8} {'CLIP-I':>8} {'DINO':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    all_clip_t, all_clip_i, all_dino = [], [], []

    for cat in sorted(cat_scores.keys()):
        s = cat_scores[cat]
        n = len(s["clip_t"])
        ct = sum(s["clip_t"]) / n
        ci = sum(s["clip_i"]) / n
        di = sum(s["dino"]) / n
        print(f"{cat:<25} {n:>5} {ct:>8.4f} {ci:>8.4f} {di:>8.4f}")
        all_clip_t.extend(s["clip_t"])
        all_clip_i.extend(s["clip_i"])
        all_dino.extend(s["dino"])

    print(sep)
    n_total = len(all_clip_t)
    print(
        f"{'Overall':<25} {n_total:>5} "
        f"{sum(all_clip_t)/n_total:>8.4f} "
        f"{sum(all_clip_i)/n_total:>8.4f} "
        f"{sum(all_dino)/n_total:>8.4f}"
    )
    print(sep)

    # Also save summary JSON
    summary = {"total_samples": n_total}
    for cat in sorted(cat_scores.keys()):
        s = cat_scores[cat]
        n = len(s["clip_t"])
        summary[cat] = {
            "n": n,
            "clip_t": sum(s["clip_t"]) / n,
            "clip_i": sum(s["clip_i"]) / n,
            "dino": sum(s["dino"]) / n,
        }
    summary["overall"] = {
        "clip_t": sum(all_clip_t) / n_total,
        "clip_i": sum(all_clip_i) / n_total,
        "dino": sum(all_dino) / n_total,
    }

    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="DreamBench++ evaluation for LightMGT")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["infer", "score", "stat", "all"],
        help="Evaluation stage to run.",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Path to LightMGT checkpoint.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiments/dreambench_pp",
        help="Directory for generated images and scores.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=DATA_ROOT,
        help="Path to DreamBench++ dataset.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    parser.add_argument(
        "--num_steps", type=int, default=20, help="Number of inference steps."
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale."
    )

    args = parser.parse_args()

    if args.mode in ("infer", "all") and args.ckpt is None:
        parser.error("--ckpt is required for inference.")

    if args.mode == "infer":
        infer(args)
    elif args.mode == "score":
        score(args)
    elif args.mode == "stat":
        stat(args)
    elif args.mode == "all":
        infer(args)
        score(args)
        stat(args)


if __name__ == "__main__":
    main()
