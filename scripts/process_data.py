"""Process expanded data sources into LightMGT training format.

Converts TextAtlas5M, MICo-150K, and pico-banana datasets into:
- JSON shards: [{uid, caption, image, height, width}, ...]
- Image directory: images/{subset}/{uid}.{ext}
- Tar archives: {subset}_images.tar (for TarImageReader)

Run on cpu_light (no GPU needed):
    python3 scripts/process_data.py --dataset textatlas
    python3 scripts/process_data.py --dataset mico
    python3 scripts/process_data.py --dataset pico

Or all at once:
    python3 scripts/process_data.py --dataset all
"""

import argparse
import hashlib
import io
import json
import os
import struct
import sys
import tarfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DATA = "/mnt/bn/search-auto-eval-v2/zhouwei/eval_data"
OUTPUT_BASE = "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/data"

TEXTATLAS_DIR = f"{BASE_DATA}/textatlas5m"
MICO_DIR = f"{BASE_DATA}/mico-150k"
PICO_DIR = f"{BASE_DATA}/pico-banana-400k"

SHARD_SIZE = 10000  # samples per JSON shard


def save_image_from_bytes(img_bytes, save_path):
    """Save image bytes to file, return (width, height) or None on failure."""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        img.save(save_path, quality=95)
        return w, h
    except Exception:
        return None


def write_shards(records, output_dir, prefix, shard_size=SHARD_SIZE):
    """Write records to JSON shard files."""
    os.makedirs(output_dir, exist_ok=True)
    shard_idx = 0
    for i in range(0, len(records), shard_size):
        chunk = records[i:i + shard_size]
        shard_path = os.path.join(output_dir, f"{prefix}_{shard_idx:04d}.json")
        with open(shard_path, "w") as f:
            json.dump(chunk, f)
        shard_idx += 1
    print(f"  Wrote {shard_idx} shards ({len(records)} records) to {output_dir}/{prefix}_*.json")


def create_tar_from_images(image_dir, tar_path, prefix="images"):
    """Create a tar archive from an image directory."""
    if os.path.exists(tar_path):
        print(f"  Tar exists, skipping: {tar_path}")
        return
    print(f"  Creating tar: {tar_path} from {image_dir}...")
    with tarfile.open(tar_path, "w") as tf:
        for root, dirs, files in os.walk(image_dir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, os.path.dirname(image_dir))
                tf.add(fpath, arcname=arcname)
    size_gb = os.path.getsize(tar_path) / 1e9
    print(f"  Created tar: {tar_path} ({size_gb:.1f} GB)")


# ─── TextAtlas5M ─────────────────────────────────────────────────────

def process_textatlas_parquet(args):
    """Process a single parquet file. Returns list of records."""
    parquet_path, subset_name, image_dir = args
    records = []
    try:
        table = pq.read_table(parquet_path)
        for i in range(len(table)):
            row = table.slice(i, 1)
            annotation = row.column("annotation")[0].as_py()
            img_data = row.column("image")[0].as_py()

            if isinstance(img_data, dict):
                img_bytes = img_data.get("bytes", b"")
            elif isinstance(img_data, bytes):
                img_bytes = img_data
            else:
                continue

            if not img_bytes:
                continue

            uid = hashlib.md5(img_bytes[:1024]).hexdigest()
            ext = "png"
            save_path = os.path.join(image_dir, f"{uid}.{ext}")

            if not os.path.exists(save_path):
                result = save_image_from_bytes(img_bytes, save_path)
                if result is None:
                    continue
                w, h = result
            else:
                try:
                    img = Image.open(save_path)
                    w, h = img.size
                except Exception:
                    continue

            records.append({
                "uid": uid,
                "caption": annotation,
                "image": f"images/textatlas_{subset_name}/{uid}.{ext}",
                "height": h,
                "width": w,
            })
    except Exception as e:
        print(f"  Error processing {parquet_path}: {e}")
    return records


def process_textatlas(num_workers=8):
    """Process all TextAtlas5M subsets."""
    print("=" * 60)
    print("Processing TextAtlas5M")
    print("=" * 60)

    output_dir = os.path.join(OUTPUT_BASE, "textatlas5m")
    os.makedirs(output_dir, exist_ok=True)

    all_records = []
    subsets = sorted([d for d in os.listdir(TEXTATLAS_DIR)
                      if os.path.isdir(os.path.join(TEXTATLAS_DIR, d))])

    for subset in subsets:
        subset_dir = os.path.join(TEXTATLAS_DIR, subset)
        parquet_files = sorted(Path(subset_dir).glob("*.parquet"))
        if not parquet_files:
            continue

        image_dir = os.path.join(output_dir, f"images/textatlas_{subset}")
        os.makedirs(image_dir, exist_ok=True)

        print(f"\n  Subset: {subset} ({len(parquet_files)} parquet files)")

        tasks = [(str(pf), subset, image_dir) for pf in parquet_files]
        subset_records = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_textatlas_parquet, t): t for t in tasks}
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    recs = future.result()
                    subset_records.extend(recs)
                    if done % 10 == 0 or done == len(tasks):
                        print(f"    Progress: {done}/{len(tasks)} files, {len(subset_records)} records")
                except Exception as e:
                    print(f"    Worker error: {e}")

        all_records.extend(subset_records)
        print(f"  Subset {subset}: {len(subset_records)} records")

    # Write JSON shards
    write_shards(all_records, output_dir, "textatlas5m")

    # Create tar archive
    image_base = os.path.join(output_dir, "images")
    tar_path = os.path.join(output_dir, "textatlas5m_images.tar")
    create_tar_from_images(image_base, tar_path)

    print(f"\nTextAtlas5M complete: {len(all_records)} total records")
    return len(all_records)


# ─── MICo-150K ───────────────────────────────────────────────────────

def process_mico_parquet(args):
    """Process a single MICo parquet file. Returns list of records."""
    parquet_path, subset_name, image_dir = args
    records = []
    try:
        table = pq.read_table(parquet_path)
        for i in range(len(table)):
            row = table.slice(i, 1)
            instruction = row.column("instruction")[0].as_py()
            output_img = row.column("output_image")[0].as_py()
            ed_idx = row.column("ed_idx")[0].as_py()

            if isinstance(output_img, dict):
                img_bytes = output_img.get("bytes", b"")
            elif isinstance(output_img, bytes):
                img_bytes = output_img
            else:
                continue

            if not img_bytes:
                continue

            uid = f"mico_{subset_name}_{ed_idx}_{i}"
            uid_hash = hashlib.md5(uid.encode()).hexdigest()
            ext = "png"
            save_path = os.path.join(image_dir, f"{uid_hash}.{ext}")

            if not os.path.exists(save_path):
                result = save_image_from_bytes(img_bytes, save_path)
                if result is None:
                    continue
                w, h = result
            else:
                try:
                    img = Image.open(save_path)
                    w, h = img.size
                except Exception:
                    continue

            # Use instruction as caption for T2I training
            records.append({
                "uid": uid_hash,
                "caption": instruction,
                "image": f"images/mico_{subset_name}/{uid_hash}.{ext}",
                "height": h,
                "width": w,
            })
    except Exception as e:
        print(f"  Error processing {parquet_path}: {e}")
    return records


def process_mico(num_workers=8):
    """Process all MICo-150K subsets."""
    print("=" * 60)
    print("Processing MICo-150K")
    print("=" * 60)

    output_dir = os.path.join(OUTPUT_BASE, "mico150k")
    os.makedirs(output_dir, exist_ok=True)

    all_records = []
    subsets = sorted([d for d in os.listdir(MICO_DIR)
                      if os.path.isdir(os.path.join(MICO_DIR, d))])

    for subset in subsets:
        subset_dir = os.path.join(MICO_DIR, subset)
        parquet_files = sorted(Path(subset_dir).glob("*.parquet"))
        if not parquet_files:
            continue

        image_dir = os.path.join(output_dir, f"images/mico_{subset}")
        os.makedirs(image_dir, exist_ok=True)

        print(f"\n  Subset: {subset} ({len(parquet_files)} parquet files)")

        tasks = [(str(pf), subset, image_dir) for pf in parquet_files]
        subset_records = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_mico_parquet, t): t for t in tasks}
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    recs = future.result()
                    subset_records.extend(recs)
                    if done % 5 == 0 or done == len(tasks):
                        print(f"    Progress: {done}/{len(tasks)} files, {len(subset_records)} records")
                except Exception as e:
                    print(f"    Worker error: {e}")

        all_records.extend(subset_records)
        print(f"  Subset {subset}: {len(subset_records)} records")

    write_shards(all_records, output_dir, "mico150k")

    image_base = os.path.join(output_dir, "images")
    tar_path = os.path.join(output_dir, "mico150k_images.tar")
    create_tar_from_images(image_base, tar_path)

    print(f"\nMICo-150K complete: {len(all_records)} total records")
    return len(all_records)


# ─── pico-banana ─────────────────────────────────────────────────────

def process_pico(num_workers=4):
    """Process pico-banana SFT editing dataset."""
    print("=" * 60)
    print("Processing pico-banana-400k")
    print("=" * 60)

    output_dir = os.path.join(OUTPUT_BASE, "pico_banana")
    image_dir = os.path.join(output_dir, "images/pico_banana")
    os.makedirs(image_dir, exist_ok=True)

    jsonl_path = os.path.join(PICO_DIR, "sft.jsonl")
    edited_images_dir = os.path.join(PICO_DIR, "edited_images")

    records = []
    skipped = 0

    with open(jsonl_path) as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            caption = item.get("text", "")
            output_image_path = item.get("output_image", "")

            if not caption or not output_image_path:
                skipped += 1
                continue

            # output_image is like "images/positive-edit/1.png"
            full_path = os.path.join(PICO_DIR, output_image_path)
            if not os.path.exists(full_path):
                # Try edited_images dir
                full_path = os.path.join(edited_images_dir, os.path.basename(output_image_path))
                if not os.path.exists(full_path):
                    skipped += 1
                    continue

            try:
                img = Image.open(full_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                w, h = img.size
            except Exception:
                skipped += 1
                continue

            uid = hashlib.md5(f"pico_{line_num}".encode()).hexdigest()
            ext = "png"
            save_path = os.path.join(image_dir, f"{uid}.{ext}")

            if not os.path.exists(save_path):
                try:
                    img.save(save_path, quality=95)
                except Exception:
                    skipped += 1
                    continue

            records.append({
                "uid": uid,
                "caption": caption,
                "image": f"images/pico_banana/{uid}.{ext}",
                "height": h,
                "width": w,
            })

            if (line_num + 1) % 50000 == 0:
                print(f"  Progress: {line_num + 1} lines, {len(records)} records, {skipped} skipped")

    write_shards(records, output_dir, "pico_banana")

    image_base = os.path.join(output_dir, "images")
    tar_path = os.path.join(output_dir, "pico_banana_images.tar")
    create_tar_from_images(image_base, tar_path)

    print(f"\npico-banana complete: {len(records)} records, {skipped} skipped")
    return len(records)


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Process expanded data sources for LightMGT")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["textatlas", "mico", "pico", "all"],
                        help="Which dataset to process")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    if args.dataset in ("textatlas", "all"):
        process_textatlas(num_workers=args.workers)

    if args.dataset in ("mico", "all"):
        process_mico(num_workers=args.workers)

    if args.dataset in ("pico", "all"):
        process_pico(num_workers=args.workers)

    print("\n" + "=" * 60)
    print("All processing complete!")
    print(f"Output: {OUTPUT_BASE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
