"""ImgEdit-Bench evaluation for LightMGT.

785 editing samples: 737 singleturn + 48 hard.
Modes: infer / rate / stat / all
"""

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = "/mnt/hdfs/weichow/maskedit/eval_data/Benchmark"
SINGLETURN_JSON = os.path.join(DATA_ROOT, "singleturn", "singleturn.json")
HARD_JSONL = os.path.join(DATA_ROOT, "hard", "annotation.jsonl")
JUDGE_PROMPT_JSON = os.path.join(DATA_ROOT, "singleturn", "judge_prompt.json")

SCORE_DIMS = ["instruction_following", "detail_preservation", "quality"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_singleturn(json_path: str) -> list[dict]:
    """Load singleturn split, returning list of dicts with keys:
    id, source_path, instruction, edit_type, split='singleturn'
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        samples.append({
            "id": item.get("id", item.get("img_id", str(len(samples)))),
            "source_path": item["source_image"],
            "instruction": item["instruction"],
            "edit_type": item.get("edit_type", "unknown"),
            "split": "singleturn",
        })
    return samples


def _load_hard(jsonl_path: str) -> list[dict]:
    """Load hard split from JSONL."""
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            samples.append({
                "id": item.get("id", item.get("img_id", str(len(samples)))),
                "source_path": item["source_image"],
                "instruction": item["instruction"],
                "edit_type": item.get("edit_type", "unknown"),
                "split": "hard",
            })
    return samples


def load_all_samples() -> list[dict]:
    samples = _load_singleturn(SINGLETURN_JSON)
    samples += _load_hard(HARD_JSONL)
    return samples


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def infer(args):
    from eval.utils import init_lightmgt, find_nearest_bucket

    pipe = init_lightmgt(args.ckpt, device=args.device)

    samples = load_all_samples()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    for sample in tqdm(samples, desc="ImgEdit infer"):
        out_path = results_dir / f"{sample['id']}.png"
        if out_path.exists() and args.resume:
            done += 1
            continue

        src = Image.open(sample["source_path"]).convert("RGB")
        h, w = src.size[1], src.size[0]  # PIL gives (w, h)
        bh, bw = find_nearest_bucket(h, w)

        edited = pipe(
            prompt=sample["instruction"],
            reference_image=src,
            height=bh,
            width=bw,
            guidance_scale=6.0,
            num_inference_steps=args.step,
        ).images[0]

        edited.save(out_path)
        done += 1

    print(f"Inference complete: {done}/{len(samples)} images saved to {results_dir}")


# ---------------------------------------------------------------------------
# Rating with GPT-4o
# ---------------------------------------------------------------------------
def _build_judge_prompt(edit_type: str, instruction: str,
                        judge_prompts: dict) -> str:
    """Build the judge prompt for a given edit type."""
    template = judge_prompts.get(edit_type, judge_prompts.get("default", ""))
    if not template:
        template = (
            "You are an expert image editing judge. Given a source image and "
            "an edited image, evaluate the editing quality based on the "
            "instruction: '{instruction}'.\n\n"
            "Rate on three dimensions (1-5):\n"
            "1. instruction_following: How well does the edit follow the instruction?\n"
            "2. detail_preservation: How well are unrelated details preserved?\n"
            "3. quality: Overall image quality and naturalness?\n\n"
            "Output format:\n"
            "instruction_following: <score>\n"
            "detail_preservation: <score>\n"
            "quality: <score>"
        )
    # Some templates use {instruction} placeholder
    return template.replace("{instruction}", instruction)


def _parse_scores(response: str) -> dict[str, int] | None:
    """Extract 3 dimension scores from GPT response using regex."""
    scores = {}
    for dim in SCORE_DIMS:
        m = re.search(rf"{dim}\s*:\s*(\d)", response, re.IGNORECASE)
        if m:
            scores[dim] = int(m.group(1))

    if len(scores) == len(SCORE_DIMS):
        return scores

    # Fallback: look for any three scores in order
    all_scores = re.findall(r":\s*(\d)", response)
    if len(all_scores) >= 3:
        return {
            SCORE_DIMS[0]: int(all_scores[0]),
            SCORE_DIMS[1]: int(all_scores[1]),
            SCORE_DIMS[2]: int(all_scores[2]),
        }
    return None


def _rate_one(sample: dict, results_dir: Path, judge_prompts: dict) -> dict | None:
    from eval.utils import call_gpt4o_vision

    edited_path = results_dir / f"{sample['id']}.png"
    if not edited_path.exists():
        return None

    prompt = _build_judge_prompt(sample["edit_type"], sample["instruction"],
                                 judge_prompts)

    try:
        response = call_gpt4o_vision(
            prompt=prompt,
            images=[sample["source_path"], str(edited_path)],
            max_tokens=300,
            temperature=0.0,
        )
    except Exception as e:
        print(f"GPT error for {sample['id']}: {e}")
        return None

    scores = _parse_scores(response)
    if scores is None:
        print(f"Parse error for {sample['id']}: {response[:200]}")
        return None

    return {
        "id": sample["id"],
        "split": sample["split"],
        "edit_type": sample["edit_type"],
        **scores,
    }


def rate(args):
    with open(JUDGE_PROMPT_JSON, "r") as f:
        judge_prompts = json.load(f)

    samples = load_all_samples()
    results_dir = Path(args.results_dir)
    scores_path = Path(args.scores_csv)
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load already-rated IDs
    rated_ids: set[str] = set()
    existing_rows: list[dict] = []
    if scores_path.exists() and args.resume:
        with open(scores_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                rated_ids.add(row["id"])
        print(f"Resuming: {len(rated_ids)} already rated")

    todo = [s for s in samples if s["id"] not in rated_ids]
    print(f"Rating {len(todo)} samples ({len(rated_ids)} done)")

    new_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_rate_one, s, results_dir, judge_prompts): s
            for s in todo
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Rating"):
            result = fut.result()
            if result:
                new_rows.append(result)

    all_rows = existing_rows + new_rows
    fieldnames = ["id", "split", "edit_type"] + SCORE_DIMS
    with open(scores_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} scores to {scores_path}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def stat(args):
    scores_path = Path(args.scores_csv)
    if not scores_path.exists():
        print(f"Scores file not found: {scores_path}")
        return

    with open(scores_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No scores found.")
        return

    # Group by (split, edit_type)
    from collections import defaultdict
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    all_by_split: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        key = (row["split"], row["edit_type"])
        groups[key].append(row)
        all_by_split[row["split"]].append(row)

    def avg_scores(row_list: list[dict]) -> dict[str, float]:
        result = {}
        for dim in SCORE_DIMS:
            vals = [int(r[dim]) for r in row_list if r.get(dim)]
            result[dim] = sum(vals) / max(len(vals), 1)
        result["avg"] = sum(result.values()) / len(result)
        return result

    # Print per edit_type table
    header = f"{'Split':<12} {'Edit Type':<25} {'N':>5}  " + "  ".join(
        f"{d:>22}" for d in SCORE_DIMS
    ) + f"  {'avg':>6}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for (split, etype) in sorted(groups.keys()):
        row_list = groups[(split, etype)]
        avgs = avg_scores(row_list)
        dims_str = "  ".join(f"{avgs[d]:>22.3f}" for d in SCORE_DIMS)
        print(f"{split:<12} {etype:<25} {len(row_list):>5}  {dims_str}  {avgs['avg']:>6.3f}")

    print("-" * len(header))

    # Per-split summary
    for split in sorted(all_by_split.keys()):
        avgs = avg_scores(all_by_split[split])
        dims_str = "  ".join(f"{avgs[d]:>22.3f}" for d in SCORE_DIMS)
        print(f"{split:<12} {'ALL':<25} {len(all_by_split[split]):>5}  {dims_str}  {avgs['avg']:>6.3f}")

    # Overall
    avgs = avg_scores(rows)
    dims_str = "  ".join(f"{avgs[d]:>22.3f}" for d in SCORE_DIMS)
    print(f"{'OVERALL':<12} {'ALL':<25} {len(rows):>5}  {dims_str}  {avgs['avg']:>6.3f}")
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ImgEdit-Bench evaluation")
    parser.add_argument("--mode", choices=["infer", "rate", "stat", "all"],
                        default="all")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to LightMGT checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--step", type=int, default=20,
                        help="Number of inference steps")
    parser.add_argument("--results_dir", type=str, default="results/imgedit",
                        help="Directory to save generated images")
    parser.add_argument("--scores_csv", type=str, default="results/imgedit/scores.csv",
                        help="Path to scores CSV")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from existing results")
    args = parser.parse_args()

    if args.mode in ("infer", "all"):
        infer(args)
    if args.mode in ("rate", "all"):
        rate(args)
    if args.mode in ("stat", "all"):
        stat(args)


if __name__ == "__main__":
    main()
