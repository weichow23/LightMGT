"""Process TextAtlas5M subsets into LightMGT T2I format.

Processes: CoverBook, LongWordsSubset-A, TextScenesHQ
Output format: JSON arrays [{uid, caption, image, height, width}]
Output path: /mnt/hdfs/weichow/maskedit/t2i/textatlas_{subset}_*.json

Usage:
    python scripts/process_textatlas.py \
        --subset coverbook \
        --input_dir /path/to/textatlas5m/CoverBook \
        --output_dir /mnt/hdfs/weichow/maskedit/t2i \
        --num_workers 232
"""

import argparse
import io
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm


SUBSETS = {
    "coverbook": "textatlas_coverbook",
    "longwords": "textatlas_longwords",
    "textscenes": "textatlas_textscenes",
}

SHARD_SIZE = 10000  # samples per shard


def process_single_sample(args):
    """Process a single sample: validate image, extract metadata."""
    idx, row, subset_prefix, image_dir = args

    try:
        # Locate image
        if isinstance(row.get("image"), str):
            img_path = row["image"]
            if not os.path.isabs(img_path):
                img_path = os.path.join(image_dir, img_path)
        elif isinstance(row.get("image_path"), str):
            img_path = row["image_path"]
            if not os.path.isabs(img_path):
                img_path = os.path.join(image_dir, img_path)
        else:
            return None

        if not os.path.exists(img_path):
            return None

        # Validate image
        img = Image.open(img_path)
        img.verify()
        img = Image.open(img_path)
        w, h = img.size

        # Skip very small images
        if min(w, h) < 128:
            return None

        # Get caption
        caption = row.get("caption") or row.get("text") or row.get("description", "")
        if not caption:
            return None

        uid = f"{subset_prefix}_{idx:06d}"

        # Output image path (relative)
        out_img_name = f"images/{subset_prefix}/{uid}.jpg"

        return {
            "uid": uid,
            "caption": caption,
            "image": out_img_name,
            "height": h,
            "width": w,
            "source_path": img_path,
        }
    except Exception:
        return None


def load_samples(input_dir: str):
    """Load all samples from input directory (parquet, json, jsonl, csv)."""
    samples = []
    input_path = Path(input_dir)

    # Try parquet first
    parquet_files = list(input_path.glob("*.parquet"))
    if parquet_files:
        import pyarrow.parquet as pq
        for pf in sorted(parquet_files):
            table = pq.read_table(pf)
            df = table.to_pandas()
            samples.extend(df.to_dict("records"))
        return samples

    # Try jsonl
    jsonl_files = list(input_path.glob("*.jsonl"))
    if jsonl_files:
        for jf in sorted(jsonl_files):
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        return samples

    # Try json
    json_files = list(input_path.glob("*.json"))
    for jf in sorted(json_files):
        with open(jf) as f:
            data = json.load(f)
            if isinstance(data, list):
                samples.extend(data)
            else:
                samples.append(data)

    return samples


def write_shard(samples, shard_idx, output_dir, subset_prefix):
    """Write a shard of samples to JSON file."""
    out_file = os.path.join(output_dir, f"{subset_prefix}_{shard_idx:04d}.json")
    # Remove source_path before writing
    clean_samples = [{k: v for k, v in s.items() if k != "source_path"} for s in samples]
    with open(out_file, "w") as f:
        json.dump(clean_samples, f, ensure_ascii=False)
    return out_file


def copy_images_to_tar(samples, shard_idx, output_dir, subset_prefix):
    """Pack images into a tar archive."""
    tar_path = os.path.join(output_dir, f"{subset_prefix}_images_{shard_idx:04d}.tar")
    with tarfile.open(tar_path, "w") as tar:
        for sample in samples:
            src = sample["source_path"]
            dst = sample["image"]
            if os.path.exists(src):
                tar.add(src, arcname=dst)
    return tar_path


def main():
    parser = argparse.ArgumentParser(description="Process TextAtlas5M subsets for LightMGT")
    parser.add_argument("--subset", required=True, choices=list(SUBSETS.keys()))
    parser.add_argument("--input_dir", required=True, help="Path to raw TextAtlas subset")
    parser.add_argument("--output_dir", required=True, help="Output dir (e.g., /mnt/hdfs/.../t2i)")
    parser.add_argument("--image_dir", default=None, help="Image directory (if different from input_dir)")
    parser.add_argument("--num_workers", type=int, default=232, help="Number of parallel workers")
    parser.add_argument("--make_tar", action="store_true", help="Also pack images into tar")
    args = parser.parse_args()

    subset_prefix = SUBSETS[args.subset]
    image_dir = args.image_dir or args.input_dir
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, f"images/{subset_prefix}"), exist_ok=True)

    print(f"Loading samples from {args.input_dir}...")
    raw_samples = load_samples(args.input_dir)
    print(f"Loaded {len(raw_samples)} raw samples")

    # Process in parallel
    print(f"Processing with {args.num_workers} workers...")
    tasks = [
        (i, row, subset_prefix, image_dir)
        for i, row in enumerate(raw_samples)
    ]

    processed = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_sample, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                processed.append(result)

    # Sort by uid for determinism
    processed.sort(key=lambda x: x["uid"])
    print(f"Valid samples: {len(processed)}/{len(raw_samples)}")

    # Write shards
    num_shards = (len(processed) + SHARD_SIZE - 1) // SHARD_SIZE
    for shard_idx in range(num_shards):
        start = shard_idx * SHARD_SIZE
        end = min(start + SHARD_SIZE, len(processed))
        shard = processed[start:end]

        json_path = write_shard(shard, shard_idx, args.output_dir, subset_prefix)
        print(f"Wrote {json_path} ({len(shard)} samples)")

        if args.make_tar:
            tar_path = copy_images_to_tar(shard, shard_idx, args.output_dir, subset_prefix)
            print(f"Wrote {tar_path}")

    print(f"Done! {len(processed)} samples in {num_shards} shards")


if __name__ == "__main__":
    main()
