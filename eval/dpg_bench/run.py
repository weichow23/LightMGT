"""DPG-Bench evaluation for LightMGT.

1065 dense prompts with GPT-4o VQA scoring and dependency-aware zero-out.

Usage:
    python -m eval.dpg_bench.run --mode infer --ckpt /path/to/ckpt --results_dir /path/to/results
    python -m eval.dpg_bench.run --mode score --results_dir /path/to/results
    python -m eval.dpg_bench.run --mode all --ckpt /path/to/ckpt --results_dir /path/to/results
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from eval.utils import ASPECT_RATIO_1024, call_gpt4o_vision, init_lightmgt

# --- Paths ---
PROMPTS_DIR = "/mnt/hdfs/weichow/maskedit/eval_data/ELLA/dpg_bench/prompts"
DPG_BENCH_CSV = "/mnt/hdfs/weichow/maskedit/eval_data/ELLA/dpg_bench/dpg_bench.csv"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _load_prompts(prompts_dir: str) -> list:
    """Load prompts from individual .txt files in the prompts directory.

    Returns list of dicts with 'id' and 'prompt' keys, sorted by id.
    """
    prompts = []
    txt_files = sorted(glob.glob(os.path.join(prompts_dir, "*.txt")))
    for txt_path in txt_files:
        item_id = os.path.splitext(os.path.basename(txt_path))[0]
        with open(txt_path, "r") as f:
            prompt_text = f.read().strip()
        prompts.append({"id": item_id, "prompt": prompt_text})
    return prompts


def infer(args):
    """Generate images for all DPG-Bench prompts using LightMGT."""
    prompts = _load_prompts(PROMPTS_DIR)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_DIR}")

    # Initialize pipeline
    pipe = init_lightmgt(args.ckpt, device=args.device)

    os.makedirs(args.results_dir, exist_ok=True)
    metadata_path = os.path.join(args.results_dir, "metadata.jsonl")

    # Resume support
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

    for idx, item in enumerate(tqdm(prompts, desc="DPG-Bench infer")):
        item_id = item["id"]
        if item_id in done_ids:
            continue

        prompt_text = item["prompt"]

        # Multi-scale random bucket selection
        bucket_h, bucket_w = rng.choice(ASPECT_RATIO_1024)

        generator = torch.Generator(device=args.device).manual_seed(42 + idx)

        img_path = os.path.join(args.results_dir, f"{item_id}.png")
        if not os.path.exists(img_path):
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
        }
        meta_f.write(json.dumps(meta_entry) + "\n")
        meta_f.flush()

    meta_f.close()
    print(f"Inference complete. Results saved to {args.results_dir}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _load_questions(csv_path: str) -> dict:
    """Load DPG-Bench questions from CSV.

    Returns:
        dict mapping item_id -> list of question dicts, each with:
            proposition_id, question, dependency, tuple_str
    """
    items = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = str(row["item_id"]).strip()
            items[item_id].append({
                "proposition_id": str(row["proposition_id"]).strip(),
                "question": row["question"].strip(),
                "dependency": str(row.get("dependency", "")).strip(),
                "tuple_str": row.get("tuple", "").strip(),
            })
    return dict(items)


def _ask_gpt_question(image_path: str, question: str) -> bool:
    """Ask GPT-4o a yes/no question about an image.

    Returns True if the answer is 'yes', False otherwise.
    """
    prompt = (
        f"Look at the image carefully. Answer the following question with "
        f"ONLY 'yes' or 'no', nothing else.\n\n"
        f"Question: {question}"
    )

    try:
        response = call_gpt4o_vision(prompt, images=[image_path], max_tokens=10)
        answer = response.strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print(f"GPT-4o API error: {e}")
        return False


def _score_image_with_dependencies(image_path: str, questions: list) -> float:
    """Score a single image with dependency-aware zero-out.

    If a parent question fails, all dependent child questions auto-fail.

    Returns mean score across all questions.
    """
    # Map proposition_id -> result
    results = {}

    # Sort questions by proposition_id to process parents first
    sorted_qs = sorted(questions, key=lambda q: q["proposition_id"])

    for q in sorted_qs:
        pid = q["proposition_id"]
        dep = q["dependency"]

        # Check if parent dependency failed
        if dep and dep in results and not results[dep]:
            results[pid] = False
            continue

        # Ask GPT-4o
        results[pid] = _ask_gpt_question(image_path, q["question"])

    if not results:
        return 0.0

    return sum(1.0 for v in results.values() if v) / len(results)


def score(args):
    """Score generated images using GPT-4o VQA with dependency-aware zero-out."""
    # Load questions
    questions_by_item = _load_questions(DPG_BENCH_CSV)
    print(f"Loaded questions for {len(questions_by_item)} items from {DPG_BENCH_CSV}")

    # Load metadata to get generated item IDs
    metadata_path = os.path.join(args.results_dir, "metadata.jsonl")
    if os.path.exists(metadata_path):
        generated_ids = []
        with open(metadata_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    generated_ids.append(entry["id"])
    else:
        # Fallback: scan for PNG files
        generated_ids = [
            os.path.splitext(os.path.basename(p))[0]
            for p in sorted(glob.glob(os.path.join(args.results_dir, "*.png")))
        ]

    print(f"Found {len(generated_ids)} generated images")

    # Resume: load existing scores
    scores_path = os.path.join(args.results_dir, "dpg_scores.jsonl")
    scored_ids = set()
    image_scores = []
    if os.path.exists(scores_path):
        with open(scores_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    scored_ids.add(entry["id"])
                    image_scores.append(entry["score"])
        print(f"Resuming: {len(scored_ids)} images already scored")

    scores_f = open(scores_path, "a")

    for item_id in tqdm(generated_ids, desc="DPG-Bench score"):
        if item_id in scored_ids:
            continue

        img_path = os.path.join(args.results_dir, f"{item_id}.png")
        if not os.path.exists(img_path):
            print(f"Warning: image not found for {item_id}, skipping")
            continue

        questions = questions_by_item.get(item_id, [])
        if not questions:
            print(f"Warning: no questions for {item_id}, skipping")
            continue

        img_score = _score_image_with_dependencies(img_path, questions)
        image_scores.append(img_score)

        score_entry = {"id": item_id, "score": img_score, "num_questions": len(questions)}
        scores_f.write(json.dumps(score_entry) + "\n")
        scores_f.flush()

    scores_f.close()

    # Compute final score
    if image_scores:
        final_score = sum(image_scores) / len(image_scores)
    else:
        final_score = 0.0

    print("\n" + "=" * 60)
    print("DPG-Bench Results")
    print("=" * 60)
    print(f"  Images scored : {len(image_scores)}")
    print(f"  Mean score    : {final_score:.4f}")
    print("=" * 60)

    # Save summary
    results = {
        "num_images": len(image_scores),
        "mean_score": final_score,
    }
    results_path = os.path.join(args.results_dir, "dpg_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DPG-Bench evaluation for LightMGT")
    parser.add_argument("--mode", type=str, default="all", choices=["infer", "score", "all"],
                        help="Run mode: infer, score, or all")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to LightMGT checkpoint")
    parser.add_argument("--results_dir", type=str, default="results/dpg_bench",
                        help="Directory to save/load results")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device")
    parser.add_argument("--step", type=int, default=20,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=9.0,
                        help="Classifier-free guidance scale")

    args = parser.parse_args()

    if args.mode in ("infer", "all"):
        if args.ckpt is None:
            parser.error("--ckpt is required for infer mode")
        infer(args)

    if args.mode in ("score", "all"):
        score(args)


if __name__ == "__main__":
    main()
