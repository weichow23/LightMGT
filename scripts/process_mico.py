"""Process MICo-150K into LightMGT Multi-Image format.

Output format: JSON arrays [{uid, image:[], height:[], width:[],
    target_image, target_height, target_width, conversations:[], task_type}]
Output path: /mnt/hdfs/weichow/maskedit/vqa/mico_*.json

Usage:
    python scripts/process_mico.py \
        --input_dir /path/to/MICo-150K \
        --output_dir /mnt/hdfs/weichow/maskedit/vqa \
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
PREFIX = "mico"


def process_single_sample(args):
    """Process a single multi-image sample."""
    idx, row, image_dir = args

    try:
        # MICo has: input_images (list), output_image, prompt/instruction
        input_images = row.get("input_images") or row.get("images") or row.get("image")
        target_image = row.get("output_image") or row.get("target_image") or row.get("target")
        prompt = (
            row.get("prompt") or row.get("instruction")
            or row.get("text") or row.get("caption") or ""
        )

        if not isinstance(input_images, list):
            input_images = [input_images]

        if not target_image or len(input_images) == 0:
            return None

        # Resolve paths
        resolved_inputs = []
        heights, widths = [], []
        for img_path in input_images:
            if not os.path.isabs(img_path):
                img_path = os.path.join(image_dir, img_path)
            if not os.path.exists(img_path):
                return None

            img = Image.open(img_path)
            w, h = img.size
            resolved_inputs.append(img_path)
            heights.append(h)
            widths.append(w)

        if not os.path.isabs(target_image):
            target_image = os.path.join(image_dir, target_image)
        if not os.path.exists(target_image):
            return None

        target_img = Image.open(target_image)
        target_w, target_h = target_img.size

        uid = f"{PREFIX}_{idx:06d}"

        # Build output paths
        out_inputs = [
            f"images/fusion/{uid}_in{k}.jpg" for k in range(len(resolved_inputs))
        ]
        out_target = f"images/fusion/{uid}_out.jpg"

        conversations = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": "Here is the generated image."},
        ]

        return {
            "uid": uid,
            "image": out_inputs,
            "height": heights,
            "width": widths,
            "target_image": out_target,
            "target_height": target_h,
            "target_width": target_w,
            "conversations": conversations,
            "source": "mico_150k",
            "task_type": "multi_image_fusion",
            # Internal: source paths for tar packing
            "_source_inputs": resolved_inputs,
            "_source_target": target_image,
        }
    except Exception:
        return None


def load_samples(input_dir: str):
    """Load samples from MICo directory."""
    samples = []
    input_path = Path(input_dir)

    for pf in sorted(input_path.glob("*.parquet")):
        import pyarrow.parquet as pq
        table = pq.read_table(pf)
        samples.extend(table.to_pandas().to_dict("records"))

    if samples:
        return samples

    for jf in sorted(input_path.glob("*.jsonl")):
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

    if samples:
        return samples

    for jf in sorted(input_path.glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
            if isinstance(data, list):
                samples.extend(data)
            else:
                samples.append(data)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Process MICo-150K for LightMGT")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=232)
    parser.add_argument("--make_tar", action="store_true")
    args = parser.parse_args()

    image_dir = args.image_dir or args.input_dir
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images/fusion"), exist_ok=True)

    print(f"Loading from {args.input_dir}...")
    raw = load_samples(args.input_dir)
    print(f"Loaded {len(raw)} raw samples")

    tasks = [(i, row, image_dir) for i, row in enumerate(raw)]

    processed = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_sample, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                processed.append(result)

    processed.sort(key=lambda x: x["uid"])
    print(f"Valid: {len(processed)}/{len(raw)}")

    num_shards = (len(processed) + SHARD_SIZE - 1) // SHARD_SIZE
    for shard_idx in range(num_shards):
        start = shard_idx * SHARD_SIZE
        end = min(start + SHARD_SIZE, len(processed))
        shard = processed[start:end]

        clean = [{k: v for k, v in s.items() if not k.startswith("_")} for s in shard]
        out_file = os.path.join(args.output_dir, f"{PREFIX}_{shard_idx:04d}.json")
        with open(out_file, "w") as f:
            json.dump(clean, f, ensure_ascii=False)
        print(f"Wrote {out_file} ({len(shard)} samples)")

        if args.make_tar:
            tar_path = os.path.join(args.output_dir, f"{PREFIX}_images_{shard_idx:04d}.tar")
            with tarfile.open(tar_path, "w") as tar:
                for s in shard:
                    for src, dst in zip(s["_source_inputs"], s["image"]):
                        if os.path.exists(src):
                            tar.add(src, arcname=dst)
                    if os.path.exists(s["_source_target"]):
                        tar.add(s["_source_target"], arcname=s["target_image"])
            print(f"Wrote {tar_path}")

    print(f"Done! {len(processed)} samples in {num_shards} shards")


if __name__ == "__main__":
    main()
