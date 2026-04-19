#!/usr/bin/env python3
"""Process Open-Bee/Honey-Data-15M VQA dataset into JSON shards + tar archives.

Schema: images (list of {bytes,path}), conversations (list of {from,value}),
        id, img_phash, img_size, source

Output per split:
  JSON: {Category}_{Split}_{idx:04d}.json  → HDFS /mnt/hdfs/weichow/maskedit/t2i-pt/
  Tar:  {Category}_{Split}_images_{idx}.tar → BN

Usage:
    python3 process_honey_data.py --workers 150
    python3 process_honey_data.py --download-only
    python3 process_honey_data.py --process-only --workers 150
"""

import argparse
import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BN_BASE = "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data"
HDFS_TARGET = "/mnt/hdfs/weichow/maskedit/t2i-pt"
HF_CACHE = "/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache"
HF_TOKEN = "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"
HF_ID = "Open-Bee/Honey-Data-15M"

SHARD_SIZE = 10000
PARQUET_BATCH = 200  # smaller batch — multi-image rows are heavier


def detect_image_format(img_bytes):
    if img_bytes[:3] == b"\xff\xd8\xff":
        return "jpg"
    if img_bytes[:4] == b"\x89PNG":
        return "png"
    return "unknown"


def ensure_jpeg(img_bytes):
    """Returns (jpeg_bytes, w, h) or None."""
    fmt = detect_image_format(img_bytes)
    if fmt == "jpg":
        return img_bytes
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception:
        return None


def download():
    """Download full dataset to BN."""
    from huggingface_hub import snapshot_download
    local_dir = os.path.join(BN_BASE, "downloads", "honey_15m")
    os.makedirs(local_dir, exist_ok=True)
    existing = list(Path(local_dir).rglob("*.parquet"))
    log.info(f"Downloading {HF_ID} → {local_dir} (existing: {len(existing)} files)")
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["HF_TOKEN"] = HF_TOKEN
    snapshot_download(repo_id=HF_ID, repo_type="dataset", local_dir=local_dir)
    n = len(list(Path(local_dir).rglob("*.parquet")))
    log.info(f"Download complete: {n} parquet files")
    return local_dir


def _worker_process_parquet(args):
    """Worker: stream one parquet → one tar + return JSON records.

    Honey schema: images (list), conversations (list), id, img_size, source.
    Each row can have multiple images.
    """
    parquet_path, tar_path, split_prefix, worker_id = args
    import pyarrow.parquet as pq

    records = []
    errors = 0
    written_imgs = 0

    try:
        pf = pq.ParquetFile(parquet_path)
        with tarfile.open(tar_path, "w") as tf:
            for batch in pf.iter_batches(batch_size=PARQUET_BATCH):
                cols = batch.column_names

                for i in range(len(batch)):
                    try:
                        row_id = batch.column("id")[i].as_py() if "id" in cols else f"row_{worker_id}_{i}"
                        source = batch.column("source")[i].as_py() if "source" in cols else ""
                        convs = batch.column("conversations")[i].as_py() if "conversations" in cols else []
                        img_sizes = batch.column("img_size")[i].as_py() if "img_size" in cols else []

                        # Process images (list of {bytes, path})
                        images_raw = batch.column("images")[i].as_py() if "images" in cols else []
                        if not images_raw:
                            errors += 1
                            continue

                        image_paths = []
                        for img_idx, img_val in enumerate(images_raw):
                            if isinstance(img_val, dict):
                                img_bytes = img_val.get("bytes")
                            elif isinstance(img_val, bytes):
                                img_bytes = img_val
                            else:
                                continue

                            if not img_bytes or len(img_bytes) < 100:
                                continue

                            uid = hashlib.md5(img_bytes[:2048]).hexdigest()
                            jpeg_bytes = ensure_jpeg(img_bytes)
                            if jpeg_bytes is None:
                                continue

                            member_name = f"images/{split_prefix}/{uid}.jpg"
                            info = tarfile.TarInfo(name=member_name)
                            info.size = len(jpeg_bytes)
                            tf.addfile(info, io.BytesIO(jpeg_bytes))
                            image_paths.append(member_name)
                            written_imgs += 1

                        if not image_paths:
                            errors += 1
                            continue

                        record = {
                            "id": row_id,
                            "images": image_paths,
                            "conversations": convs,
                            "source": source,
                        }
                        if img_sizes:
                            record["img_size"] = img_sizes

                        records.append(record)
                    except Exception:
                        errors += 1

    except Exception as e:
        return [], errors, 0, str(e)

    return records, errors, written_imgs, None


def process(num_workers=150):
    """Process all splits: discover splits → parallel parquet processing → JSON shards."""
    dl_dir = os.path.join(BN_BASE, "downloads", "honey_15m")
    output_dir = os.path.join(BN_BASE, "output", "honey_15m")
    os.makedirs(output_dir, exist_ok=True)

    # Discover splits (Category/SplitName directories containing parquets)
    all_parquets = sorted(Path(dl_dir).rglob("*.parquet"))
    if not all_parquets:
        log.error(f"No parquet files in {dl_dir}")
        return

    # Group by split: path like Caption/COYO-Recaption/train-00000-of-00110.parquet
    from collections import defaultdict
    splits = defaultdict(list)
    for pf in all_parquets:
        rel = pf.relative_to(dl_dir)
        parts = rel.parts
        if len(parts) >= 3:
            split_name = f"{parts[0]}_{parts[1]}"  # e.g. Caption_COYO-Recaption
        elif len(parts) == 2:
            split_name = parts[0]
        else:
            split_name = "root"
        splits[split_name].append(pf)

    log.info(f"Found {len(all_parquets)} parquets in {len(splits)} splits")
    for s, pqs in sorted(splits.items()):
        log.info(f"  {s}: {len(pqs)} parquets")

    # Process each split
    grand_total = 0
    for split_name, parquet_files in sorted(splits.items()):
        split_output = os.path.join(output_dir, split_name)
        os.makedirs(split_output, exist_ok=True)

        log.info(f"\n{'='*60}")
        log.info(f"Split: {split_name} ({len(parquet_files)} parquets)")

        # Build tasks (skip existing tars)
        tasks = []
        for i, pf in enumerate(parquet_files):
            tar_path = os.path.join(split_output, f"{split_name}_images_{i}.tar")
            if os.path.exists(tar_path) and os.path.getsize(tar_path) > 512:
                continue
            tasks.append((str(pf), tar_path, f"honey_{split_name}", i))

        if not tasks:
            log.info(f"  All tars exist, skipping processing")
            # Still need to collect records from existing JSON
            existing_jsons = sorted(Path(split_output).glob(f"{split_name}_*.json"))
            if existing_jsons:
                count = sum(len(json.load(open(j))) for j in existing_jsons)
                grand_total += count
                log.info(f"  Existing: {count} records in {len(existing_jsons)} shards")
            continue

        log.info(f"  Processing {len(tasks)} parquets with {min(num_workers, len(tasks))} workers")

        all_records = []
        total_errors = 0
        total_imgs = 0
        t0 = time.time()
        done = 0

        eff_workers = min(num_workers, len(tasks))
        with ProcessPoolExecutor(max_workers=eff_workers) as executor:
            futures = {executor.submit(_worker_process_parquet, t): t for t in tasks}
            for future in as_completed(futures):
                done += 1
                try:
                    records, errors, imgs, err = future.result()
                    if err:
                        log.error(f"  Worker error: {err}")
                    all_records.extend(records)
                    total_errors += errors
                    total_imgs += imgs
                    if done % 10 == 0 or done == len(tasks):
                        elapsed = time.time() - t0
                        log.info(f"  [{done}/{len(tasks)}] {len(all_records)} records, {total_imgs} imgs, {total_errors} errors, {elapsed:.0f}s")
                except Exception as e:
                    log.error(f"  Future error: {e}")

        log.info(f"  Split done: {len(all_records)} records, {total_imgs} imgs, {total_errors} errors")

        # Write JSON shards: {split_name}_{idx:04d}.json
        shard_idx = 0
        for i in range(0, len(all_records), SHARD_SIZE):
            chunk = all_records[i:i + SHARD_SIZE]
            path = os.path.join(split_output, f"{split_name}_{shard_idx:04d}.json")
            with open(path, "w") as f:
                json.dump(chunk, f)
            shard_idx += 1
        log.info(f"  Wrote {shard_idx} JSON shards")
        grand_total += len(all_records)

    log.info(f"\n{'='*60}")
    log.info(f"ALL SPLITS DONE: {grand_total} total records")


def upload():
    """Upload all JSON shards to HDFS."""
    output_dir = os.path.join(BN_BASE, "output", "honey_15m")
    os.makedirs(HDFS_TARGET, exist_ok=True)

    total = 0
    failed = 0
    for split_dir in sorted(Path(output_dir).iterdir()):
        if not split_dir.is_dir():
            continue
        jsons = sorted(split_dir.glob("*.json"))
        for jf in jsons:
            dest = os.path.join(HDFS_TARGET, jf.name)
            try:
                shutil.copy2(str(jf), dest)
                total += 1
            except Exception as e:
                log.error(f"  FAIL {jf.name}: {e}")
                failed += 1

    log.info(f"HDFS upload: {total} succeeded, {failed} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=150)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    args = parser.parse_args()

    if args.upload_only:
        upload()
        return

    if not args.process_only:
        download()

    if args.download_only:
        return

    process(num_workers=args.workers)
    upload()
    log.info("=== All done! ===")


if __name__ == "__main__":
    main()
