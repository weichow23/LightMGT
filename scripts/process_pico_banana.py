"""Process pico-banana-400K SFT subset into LightMGT Edit format.

Output format: JSON [{uid, instruction, input_image, output_image, height, width}]
Output path: /mnt/hdfs/weichow/maskedit/edit/pico_banana_sft_*.json

Usage:
    python scripts/process_pico_banana.py \
        --input_dir /path/to/pico-banana-400k/sft \
        --output_dir /mnt/hdfs/weichow/maskedit/edit \
        --num_workers 232
"""

import argparse
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm


SHARD_SIZE = 10000
PREFIX = "pico_banana_sft"


def process_single_sample(args):
    """Process a single edit sample."""
    idx, row, image_dir = args

    try:
        # pico-banana SFT has: input_image, output_image, instruction
        input_path = row.get("input_image") or row.get("source_image")
        output_path = row.get("output_image") or row.get("edited_image") or row.get("target_image")
        instruction = row.get("instruction") or row.get("edit_instruction") or row.get("prompt")

        if not all([input_path, output_path, instruction]):
            return None

        # Resolve paths
        if not os.path.isabs(input_path):
            input_path = os.path.join(image_dir, input_path)
        if not os.path.isabs(output_path):
            output_path = os.path.join(image_dir, output_path)

        if not os.path.exists(input_path) or not os.path.exists(output_path):
            return None

        # Validate output image
        img = Image.open(output_path)
        img.verify()
        img = Image.open(output_path)
        w, h = img.size

        if min(w, h) < 128:
            return None

        uid = f"{PREFIX}_{idx:06d}"

        return {
            "uid": uid,
            "instruction": instruction,
            "input_image": f"images/{PREFIX}/{uid}_input.jpg",
            "output_image": f"images/{PREFIX}/{uid}_output.jpg",
            "height": h,
            "width": w,
            "source_input_path": input_path,
            "source_output_path": output_path,
        }
    except Exception:
        return None


def load_samples(input_dir: str):
    """Load samples from pico-banana directory."""
    samples = []
    input_path = Path(input_dir)

    # Try parquet
    for pf in sorted(input_path.glob("*.parquet")):
        import pyarrow.parquet as pq
        table = pq.read_table(pf)
        samples.extend(table.to_pandas().to_dict("records"))

    if samples:
        return samples

    # Try jsonl
    for jf in sorted(input_path.glob("*.jsonl")):
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

    if samples:
        return samples

    # Try json
    for jf in sorted(input_path.glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
            if isinstance(data, list):
                samples.extend(data)
            else:
                samples.append(data)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Process pico-banana SFT for LightMGT")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=232)
    parser.add_argument("--make_tar", action="store_true")
    args = parser.parse_args()

    image_dir = args.image_dir or args.input_dir
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, f"images/{PREFIX}"), exist_ok=True)

    print(f"Loading samples from {args.input_dir}...")
    raw_samples = load_samples(args.input_dir)
    print(f"Loaded {len(raw_samples)} raw samples")

    tasks = [(i, row, image_dir) for i, row in enumerate(raw_samples)]

    processed = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_sample, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                processed.append(result)

    processed.sort(key=lambda x: x["uid"])
    print(f"Valid: {len(processed)}/{len(raw_samples)}")

    num_shards = (len(processed) + SHARD_SIZE - 1) // SHARD_SIZE
    for shard_idx in range(num_shards):
        start = shard_idx * SHARD_SIZE
        end = min(start + SHARD_SIZE, len(processed))
        shard = processed[start:end]

        clean = [{k: v for k, v in s.items() if not k.startswith("source_")} for s in shard]
        out_file = os.path.join(args.output_dir, f"{PREFIX}_{shard_idx:04d}.json")
        with open(out_file, "w") as f:
            json.dump(clean, f, ensure_ascii=False)
        print(f"Wrote {out_file} ({len(shard)} samples)")

        if args.make_tar:
            tar_path = os.path.join(args.output_dir, f"{PREFIX}_images_{shard_idx:04d}.tar")
            with tarfile.open(tar_path, "w") as tar:
                for s in shard:
                    for key in ["source_input_path", "source_output_path"]:
                        src = s[key]
                        dst = s["input_image"] if "input" in key else s["output_image"]
                        if os.path.exists(src):
                            tar.add(src, arcname=dst)
            print(f"Wrote {tar_path}")

    print(f"Done! {len(processed)} samples in {num_shards} shards")


if __name__ == "__main__":
    main()
