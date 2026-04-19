"""GenEval benchmark evaluation for LightMGT.

553 compositional prompts evaluating object count, position, color,
and relationships via Mask2Former detection + OpenCLIP color classification.

Usage:
    python -m eval.geneval.run --mode infer --ckpt /path/to/ckpt --results_dir /path/to/results
    python -m eval.geneval.run --mode score --results_dir /path/to/results
    python -m eval.geneval.run --mode all --ckpt /path/to/ckpt --results_dir /path/to/results
"""

import argparse
import json
import os
import random
import sys

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from eval.utils import ASPECT_RATIO_1024, init_lightmgt

# --- Paths ---
PROMPTS_JSONL = "/mnt/hdfs/weichow/maskedit/eval_data/evaluation_metadata_long.jsonl"
MASK2FORMER_CONFIG = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
MASK2FORMER_CKPT = "/mnt/hdfs/weichow/maskedit/eval_data/geneval_models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
OBJECT_NAMES_FILE = "/mnt/hdfs/weichow/maskedit/eval_data/geneval/evaluation/object_names.txt"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer(args):
    """Generate images for all GenEval prompts using LightMGT."""
    # Load prompts
    prompts = []
    with open(PROMPTS_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_JSONL}")

    # Initialize pipeline
    pipe = init_lightmgt(args.ckpt, device=args.device)

    os.makedirs(args.results_dir, exist_ok=True)
    metadata_path = os.path.join(args.results_dir, "metadata.jsonl")

    # Load already-generated items for resume
    done_ids = set()
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    done_ids.add(entry["id"])
        print(f"Resuming: {len(done_ids)} images already generated")

    # Seed for reproducibility
    rng = random.Random(42)

    meta_f = open(metadata_path, "a")

    for idx, item in enumerate(tqdm(prompts, desc="GenEval infer")):
        item_id = item.get("id", idx)
        if item_id in done_ids:
            continue

        prompt_text = item["prompt"]

        # Multi-scale random bucket selection
        bucket_h, bucket_w = rng.choice(ASPECT_RATIO_1024)

        generator = torch.Generator(device=args.device).manual_seed(42 + idx)

        for img_idx in range(args.num_images):
            img_path = os.path.join(args.results_dir, f"{item_id}_{img_idx}.png")
            if os.path.exists(img_path):
                continue

            images = pipe(
                prompt=prompt_text,
                height=bucket_h,
                width=bucket_w,
                num_inference_steps=args.step,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            images[0].save(img_path)

        # Write metadata
        meta_entry = {
            "id": item_id,
            "prompt": prompt_text,
            "resolution": [bucket_h, bucket_w],
            "num_images": args.num_images,
            **{k: v for k, v in item.items() if k not in ("id", "prompt")},
        }
        meta_f.write(json.dumps(meta_entry) + "\n")
        meta_f.flush()

    meta_f.close()
    print(f"Inference complete. Results saved to {args.results_dir}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _load_object_names(path: str) -> list:
    """Load COCO object names for Mask2Former."""
    with open(path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def _init_mask2former(device: str):
    """Load Mask2Former (Swin-S) via mmdet for object detection."""
    from mmdet.apis import init_detector, inference_detector

    # Build config path from mmdet
    import mmdet
    mmdet_dir = os.path.dirname(mmdet.__file__)
    config_dir = os.path.join(os.path.dirname(mmdet_dir), ".mim", "configs", "mask2former")
    config_file = os.path.join(config_dir, f"{MASK2FORMER_CONFIG}.py")

    if not os.path.exists(config_file):
        # Fallback: try mmdet internal configs
        from mmengine.config import Config
        config_file = os.path.join(
            mmdet_dir, ".mim", "configs", "mask2former",
            f"{MASK2FORMER_CONFIG}.py",
        )

    model = init_detector(config_file, MASK2FORMER_CKPT, device=device)
    return model


def _init_clip_model(device: str):
    """Load OpenCLIP ViT-L-14 for color classification."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    return model, preprocess, tokenizer


COLOR_NAMES = [
    "red", "orange", "yellow", "green", "blue", "purple",
    "pink", "brown", "black", "white", "gray", "silver",
    "gold", "beige", "cyan", "magenta",
]


def _classify_color_clip(crop: Image.Image, clip_model, preprocess, tokenizer, device: str) -> str:
    """Zero-shot color classification using CLIP."""
    img_tensor = preprocess(crop).unsqueeze(0).to(device)
    text_prompts = [f"a photo of a {c} object" for c in COLOR_NAMES]
    text_tokens = tokenizer(text_prompts).to(device)

    with torch.no_grad():
        img_feat = clip_model.encode_image(img_tensor)
        txt_feat = clip_model.encode_text(text_tokens)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ txt_feat.T).squeeze(0)

    return COLOR_NAMES[sim.argmax().item()]


def _detect_objects(image_path: str, detector, score_thresh: float = 0.3):
    """Run Mask2Former on an image, return list of (label_id, bbox, score, mask)."""
    from mmdet.apis import inference_detector
    import numpy as np

    result = inference_detector(detector, image_path)

    detections = []
    pred_instances = result.pred_instances
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    bboxes = pred_instances.bboxes.cpu().numpy()
    masks = pred_instances.masks.cpu().numpy() if hasattr(pred_instances, "masks") else [None] * len(scores)

    for score, label, bbox, mask in zip(scores, labels, bboxes, masks):
        if score >= score_thresh:
            detections.append({
                "label": int(label),
                "bbox": bbox.tolist(),
                "score": float(score),
                "mask": mask,
            })

    return detections


def _check_count(detections: list, target_name: str, target_count: int, object_names: list) -> bool:
    """Check if the correct number of target objects are detected."""
    target_name_lower = target_name.lower()
    count = 0
    for det in detections:
        obj_name = object_names[det["label"]].lower()
        if target_name_lower in obj_name or obj_name in target_name_lower:
            count += 1
    return count == target_count


def _check_color(
    detections: list,
    target_name: str,
    target_color: str,
    object_names: list,
    image: Image.Image,
    clip_model,
    preprocess,
    tokenizer,
    device: str,
) -> bool:
    """Check if detected objects of target type have the target color."""
    target_name_lower = target_name.lower()
    target_color_lower = target_color.lower()

    for det in detections:
        obj_name = object_names[det["label"]].lower()
        if target_name_lower in obj_name or obj_name in target_name_lower:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            crop = image.crop((x1, y1, x2, y2))
            if crop.size[0] < 5 or crop.size[1] < 5:
                continue
            pred_color = _classify_color_clip(crop, clip_model, preprocess, tokenizer, device)
            if pred_color == target_color_lower:
                return True
    return False


def _check_position(detections: list, requirements: dict, object_names: list, img_w: int, img_h: int) -> bool:
    """Check spatial relationships between detected objects.

    Supports: left_of, right_of, above, below.
    """
    relation = requirements.get("relation")
    obj_a_name = requirements.get("object_a", "").lower()
    obj_b_name = requirements.get("object_b", "").lower()

    if not relation or not obj_a_name or not obj_b_name:
        return False

    # Find center positions
    def find_center(name):
        for det in detections:
            obj_name = object_names[det["label"]].lower()
            if name in obj_name or obj_name in name:
                x1, y1, x2, y2 = det["bbox"]
                return ((x1 + x2) / 2, (y1 + y2) / 2)
        return None

    pos_a = find_center(obj_a_name)
    pos_b = find_center(obj_b_name)

    if pos_a is None or pos_b is None:
        return False

    if relation == "left_of":
        return pos_a[0] < pos_b[0]
    elif relation == "right_of":
        return pos_a[0] > pos_b[0]
    elif relation == "above":
        return pos_a[1] < pos_b[1]  # y increases downward
    elif relation == "below":
        return pos_a[1] > pos_b[1]
    else:
        return False


def score(args):
    """Score generated images using Mask2Former + OpenCLIP."""
    metadata_path = os.path.join(args.results_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata found at {metadata_path}. Run infer first.")

    # Load metadata
    entries = []
    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Scoring {len(entries)} items")

    # Load models
    print("Loading Mask2Former...")
    detector = _init_mask2former(args.device)
    object_names = _load_object_names(OBJECT_NAMES_FILE)
    print(f"Loaded {len(object_names)} object names")

    print("Loading OpenCLIP ViT-L-14...")
    clip_model, clip_preprocess, clip_tokenizer = _init_clip_model(args.device)

    # Per-tag tracking
    tag_correct = {}
    tag_total = {}

    for entry in tqdm(entries, desc="GenEval score"):
        item_id = entry["id"]
        num_images = entry.get("num_images", args.num_images)

        # Collect tags and requirements from metadata
        tags = entry.get("tags", [])
        if not tags:
            # Build tags from structured fields
            if "count" in entry:
                tags.append({"type": "count", "object": entry.get("object", ""), "count": entry["count"]})
            if "color" in entry:
                tags.append({"type": "color", "object": entry.get("object", ""), "color": entry["color"]})
            if "position" in entry:
                tags.append({"type": "position", **entry["position"]})

        for tag in tags:
            tag_type = tag.get("type", "unknown")
            if tag_type not in tag_total:
                tag_total[tag_type] = 0
                tag_correct[tag_type] = 0

        # Score each generated image, take best
        best_results = {tag.get("type", "unknown"): False for tag in tags}

        for img_idx in range(num_images):
            img_path = os.path.join(args.results_dir, f"{item_id}_{img_idx}.png")
            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert("RGB")
            detections = _detect_objects(img_path, detector)

            for tag in tags:
                tag_type = tag.get("type", "unknown")

                if tag_type == "count":
                    ok = _check_count(detections, tag["object"], tag["count"], object_names)
                elif tag_type == "color":
                    ok = _check_color(
                        detections, tag["object"], tag["color"],
                        object_names, image, clip_model, clip_preprocess, clip_tokenizer, args.device,
                    )
                elif tag_type == "position":
                    ok = _check_position(detections, tag, object_names, image.width, image.height)
                else:
                    ok = False

                if ok:
                    best_results[tag_type] = True

        # Accumulate
        for tag_type, passed in best_results.items():
            tag_total[tag_type] += 1
            if passed:
                tag_correct[tag_type] += 1

    # Print results
    print("\n" + "=" * 60)
    print("GenEval Results")
    print("=" * 60)

    total_correct = 0
    total_count = 0
    results = {}

    for tag_type in sorted(tag_total.keys()):
        correct = tag_correct[tag_type]
        total = tag_total[tag_type]
        acc = correct / total if total > 0 else 0.0
        results[tag_type] = {"correct": correct, "total": total, "accuracy": acc}
        total_correct += correct
        total_count += total
        print(f"  {tag_type:20s}: {correct:4d}/{total:4d} = {acc:.4f}")

    overall = total_correct / total_count if total_count > 0 else 0.0
    results["overall"] = {"correct": total_correct, "total": total_count, "accuracy": overall}
    print(f"  {'overall':20s}: {total_correct:4d}/{total_count:4d} = {overall:.4f}")
    print("=" * 60)

    # Save results
    results_path = os.path.join(args.results_dir, "geneval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GenEval benchmark for LightMGT")
    parser.add_argument("--mode", type=str, default="all", choices=["infer", "score", "all"],
                        help="Run mode: infer, score, or all")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to LightMGT checkpoint")
    parser.add_argument("--results_dir", type=str, default="results/geneval",
                        help="Directory to save/load results")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device")
    parser.add_argument("--step", type=int, default=20,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=9.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num_images", type=int, default=1,
                        help="Number of images to generate per prompt")

    args = parser.parse_args()

    if args.mode in ("infer", "all"):
        if args.ckpt is None:
            parser.error("--ckpt is required for infer mode")
        infer(args)

    if args.mode in ("score", "all"):
        score(args)


if __name__ == "__main__":
    main()
