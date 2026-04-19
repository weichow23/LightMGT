"""GEditBench evaluation for LightMGT.

~600 samples, 11 edit categories, VIEScore rating.
Modes: infer / rate / stat / all
"""

import argparse
import csv
import json
import math
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
LOCAL_DATA_DIR = "/mnt/hdfs/weichow/maskedit/eval_data/geditbench"
HF_DATASET_NAME = "stepfun-ai/GEdit-Bench"

TASK_TYPES = [
    "add", "remove", "replace", "change_attribute", "change_color",
    "change_material", "change_style", "change_background",
    "change_weather", "change_expression", "action_change",
]


# ---------------------------------------------------------------------------
# VIEScore prompts
# ---------------------------------------------------------------------------
VIESCORE_SC_PROMPT = """\
You are an expert image editing evaluator. You will be given:
1. A source image (before editing)
2. An edited image (after editing)
3. The editing instruction

Evaluate the **Semantic Correctness (SC)** of the edit: how well does the \
edited image follow the given instruction while maintaining coherence?

Consider:
- Is the requested edit clearly applied?
- Does the edit match the instruction precisely?
- Are there unintended changes that contradict the instruction?

Rate on a scale of 1-5:
1: Completely wrong / no visible edit
2: Partially attempted but mostly incorrect
3: Edit is present but with notable issues
4: Good edit with minor imperfections
5: Perfect edit, fully matches instruction

Editing instruction: {instruction}

Output ONLY: SC: <score>
"""

VIESCORE_PQ_PROMPT = """\
You are an expert image quality evaluator. You will be given:
1. A source image (before editing)
2. An edited image (after editing)
3. The editing instruction

Evaluate the **Perceptual Quality (PQ)** of the edited image: how natural \
and artifact-free is the result?

Consider:
- Are there visible artifacts, distortions, or blurring?
- Is the lighting consistent?
- Are edges clean and natural?
- Does the image look realistic / plausible?
- Are unedited regions well-preserved?

Rate on a scale of 1-5:
1: Severe artifacts, unnatural
2: Notable artifacts or distortions
3: Acceptable but with visible imperfections
4: Good quality, minor issues only
5: Excellent, no visible artifacts

Editing instruction: {instruction}

Output ONLY: PQ: <score>
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_geditbench(data_dir: str | None = None) -> list[dict]:
    """Load GEditBench samples. Tries local dir first, then HF.

    Returns list of dicts: id, source_path (or source_image PIL),
    instruction, task_type, language.
    """
    data_dir = data_dir or LOCAL_DATA_DIR
    local_meta = os.path.join(data_dir, "metadata.jsonl")

    samples = []

    if os.path.exists(local_meta):
        # Local loading
        with open(local_meta, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # Filter English only
                if item.get("language", "en") != "en":
                    continue
                source_path = item.get("source_image", "")
                if not os.path.isabs(source_path):
                    source_path = os.path.join(data_dir, source_path)
                samples.append({
                    "id": str(item.get("id", len(samples))),
                    "source_path": source_path,
                    "instruction": item["instruction"],
                    "task_type": item.get("task_type", item.get("edit_type", "unknown")),
                })
    else:
        # HF dataset loading
        try:
            from datasets import load_dataset
            ds = load_dataset(HF_DATASET_NAME, split="test")
        except Exception as e:
            print(f"Failed to load HF dataset: {e}")
            print("Please download data to:", LOCAL_DATA_DIR)
            return []

        for idx, item in enumerate(ds):
            if item.get("language", "en") != "en":
                continue
            samples.append({
                "id": str(item.get("id", idx)),
                "source_image": item["source_image"],  # PIL Image from HF
                "instruction": item["instruction"],
                "task_type": item.get("task_type", item.get("edit_type", "unknown")),
            })

    print(f"Loaded {len(samples)} English samples from GEditBench")
    return samples


def _get_source_image(sample: dict) -> Image.Image:
    """Get source image as PIL, handling both path and PIL variants."""
    if "source_path" in sample:
        return Image.open(sample["source_path"]).convert("RGB")
    return sample["source_image"].convert("RGB")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def infer(args):
    from eval.utils import init_lightmgt, find_nearest_bucket

    pipe = init_lightmgt(args.ckpt, device=args.device)

    samples = load_geditbench(args.data_dir)
    results_dir = Path(args.results_dir)

    done = 0
    for sample in tqdm(samples, desc="GEditBench infer"):
        task_dir = results_dir / sample["task_type"]
        task_dir.mkdir(parents=True, exist_ok=True)
        out_path = task_dir / f"{sample['id']}.png"

        if out_path.exists() and args.resume:
            done += 1
            continue

        src = _get_source_image(sample)
        h, w = src.size[1], src.size[0]
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
# VIEScore Rating
# ---------------------------------------------------------------------------
def _parse_sc(response: str) -> int | None:
    m = re.search(r"SC\s*:\s*(\d)", response, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _parse_pq(response: str) -> int | None:
    m = re.search(r"PQ\s*:\s*(\d)", response, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _rate_one(sample: dict, results_dir: Path) -> dict | None:
    from eval.utils import call_gpt4o_vision

    task_dir = results_dir / sample["task_type"]
    edited_path = task_dir / f"{sample['id']}.png"
    if not edited_path.exists():
        return None

    source_path = sample.get("source_path")
    if source_path:
        images = [source_path, str(edited_path)]
    else:
        # HF dataset: save source temporarily
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        sample["source_image"].save(tmp.name)
        images = [tmp.name, str(edited_path)]

    instruction = sample["instruction"]

    # SC score
    sc_prompt = VIESCORE_SC_PROMPT.format(instruction=instruction)
    try:
        sc_resp = call_gpt4o_vision(
            prompt=sc_prompt, images=images, max_tokens=100, temperature=0.0,
        )
        sc = _parse_sc(sc_resp)
    except Exception as e:
        print(f"SC error for {sample['id']}: {e}")
        sc = None

    # PQ score
    pq_prompt = VIESCORE_PQ_PROMPT.format(instruction=instruction)
    try:
        pq_resp = call_gpt4o_vision(
            prompt=pq_prompt, images=images, max_tokens=100, temperature=0.0,
        )
        pq = _parse_pq(pq_resp)
    except Exception as e:
        print(f"PQ error for {sample['id']}: {e}")
        pq = None

    # Clean up temp file if created
    if not source_path:
        try:
            os.unlink(images[0])
        except OSError:
            pass

    if sc is None or pq is None:
        print(f"Parse error for {sample['id']}: SC={sc}, PQ={pq}")
        return None

    overall = math.sqrt(sc * pq)

    return {
        "id": sample["id"],
        "task_type": sample["task_type"],
        "sc": sc,
        "pq": pq,
        "overall": round(overall, 4),
    }


def rate(args):
    samples = load_geditbench(args.data_dir)
    results_dir = Path(args.results_dir)
    scores_path = Path(args.scores_csv)
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
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
            pool.submit(_rate_one, s, results_dir): s
            for s in todo
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="VIEScore"):
            result = fut.result()
            if result:
                new_rows.append(result)

    all_rows = existing_rows + new_rows
    fieldnames = ["id", "task_type", "sc", "pq", "overall"]
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

    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        groups[row["task_type"]].append(row)

    def avg(row_list: list[dict], key: str) -> float:
        vals = [float(r[key]) for r in row_list if r.get(key)]
        return sum(vals) / max(len(vals), 1)

    header = f"{'Category':<25} {'N':>5}  {'SC':>6}  {'PQ':>6}  {'Overall':>8}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    all_sc, all_pq, all_overall = [], [], []

    for task in sorted(groups.keys()):
        row_list = groups[task]
        sc = avg(row_list, "sc")
        pq = avg(row_list, "pq")
        ov = avg(row_list, "overall")
        all_sc.extend([float(r["sc"]) for r in row_list if r.get("sc")])
        all_pq.extend([float(r["pq"]) for r in row_list if r.get("pq")])
        all_overall.extend([float(r["overall"]) for r in row_list if r.get("overall")])
        print(f"{task:<25} {len(row_list):>5}  {sc:>6.3f}  {pq:>6.3f}  {ov:>8.4f}")

    print("-" * len(header))
    n = len(rows)
    mean_sc = sum(all_sc) / max(len(all_sc), 1)
    mean_pq = sum(all_pq) / max(len(all_pq), 1)
    mean_ov = sum(all_overall) / max(len(all_overall), 1)
    print(f"{'OVERALL':<25} {n:>5}  {mean_sc:>6.3f}  {mean_pq:>6.3f}  {mean_ov:>8.4f}")
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GEditBench evaluation")
    parser.add_argument("--mode", choices=["infer", "rate", "stat", "all"],
                        default="all")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to LightMGT checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--step", type=int, default=20,
                        help="Number of inference steps")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Local data dir (default: HDFS path)")
    parser.add_argument("--results_dir", type=str, default="results/geditbench",
                        help="Directory to save generated images")
    parser.add_argument("--scores_csv", type=str,
                        default="results/geditbench/scores.csv",
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
