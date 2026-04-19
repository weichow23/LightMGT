"""Re-download and process TextAtlas5M CoverBook and LongWordsSubset-A.

These subsets were lost during a data migration accident.
Source: CSU-JPG/TextAtlas5M on HuggingFace (MIT License).

Usage (run on cpu_light):
    python3 scripts/fix_textatlas.py
"""

import hashlib
import json
import os

from datasets import load_dataset
from PIL import Image

OUTPUT_BASE = "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/data/textatlas5m_fix"
SHARD_SIZE = 10000

SUBSETS = ["CoverBook", "LongWordsSubset-A"]


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    for subset_name in SUBSETS:
        print(f"\n{'='*60}")
        print(f"Downloading {subset_name} from CSU-JPG/TextAtlas5M...")
        print(f"{'='*60}")

        ds = load_dataset("CSU-JPG/TextAtlas5M", subset_name, split="train")
        print(f"  {len(ds)} rows loaded")

        image_dir = os.path.join(OUTPUT_BASE, f"images/textatlas_{subset_name}")
        os.makedirs(image_dir, exist_ok=True)

        records = []
        skipped = 0
        for i, item in enumerate(ds):
            try:
                img = item["image"]
                annotation = item.get("annotation", "")
                if not annotation:
                    skipped += 1
                    continue

                if img.mode != "RGB":
                    img = img.convert("RGB")
                w, h = img.size

                uid = hashlib.md5(f"textatlas_{subset_name}_{i}".encode()).hexdigest()
                ext = "jpg"
                save_path = os.path.join(image_dir, f"{uid}.{ext}")

                if not os.path.exists(save_path):
                    img.save(save_path, quality=95)

                records.append({
                    "uid": uid,
                    "caption": annotation,
                    "image": f"images/textatlas_{subset_name}/{uid}.{ext}",
                    "height": h,
                    "width": w,
                })
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  Error at {i}: {e}")

            if (i + 1) % 50000 == 0:
                print(f"  Progress: {i+1}/{len(ds)}, {len(records)} valid, {skipped} skipped")

        print(f"  {subset_name}: {len(records)} records, {skipped} skipped")

        # Write JSON shards
        for j in range(0, len(records), SHARD_SIZE):
            shard = records[j:j + SHARD_SIZE]
            shard_name = f"textatlas5m_{subset_name}_{j // SHARD_SIZE:04d}.json"
            with open(os.path.join(OUTPUT_BASE, shard_name), "w") as f:
                json.dump(shard, f)

        n_shards = (len(records) + SHARD_SIZE - 1) // SHARD_SIZE
        print(f"  Written {n_shards} shards")

    # Create tar
    import tarfile
    tar_path = os.path.join(OUTPUT_BASE, "textatlas5m_fix_images.tar")
    print(f"\nCreating tar: {tar_path}")
    image_base = os.path.join(OUTPUT_BASE, "images")
    with tarfile.open(tar_path, "w") as tf:
        for subset_name in SUBSETS:
            subset_dir = f"textatlas_{subset_name}"
            full_dir = os.path.join(image_base, subset_dir)
            if os.path.isdir(full_dir):
                tf.add(full_dir, arcname=subset_dir)
                print(f"  Added {subset_dir}")
    print(f"Tar created: {os.path.getsize(tar_path) / 1e9:.1f} GB")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
