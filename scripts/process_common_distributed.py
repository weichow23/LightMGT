#!/usr/bin/env python3
"""Distributed commoncatalog half-2: each machine downloads+processes its own shard.

Only downloads parquets with index >= HALF1_COUNT (already processed in half-1).
Each machine downloads 1/N of the remaining parquets using hf_hub_download.

Usage:
    python3 process_common_distributed.py --shard 0 --total-shards 4 --workers 150
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
HF_ID = "common-canvas/commoncatalog-cc-by"
HALF1_COUNT = 2602  # Already processed in half-1
MAX_DIM = 1536
SHARD_SIZE = 10000
PARQUET_BATCH = 500


def download_my_parquets(all_parquet_names, shard, total_shards):
    """Download only this shard's parquet files."""
    from huggingface_hub import hf_hub_download
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["HF_TOKEN"] = HF_TOKEN

    # Only half-2 parquets (index >= HALF1_COUNT)
    half2 = all_parquet_names[HALF1_COUNT:]
    my_files = [f for i, f in enumerate(half2) if i % total_shards == shard]
    log.info(f"Shard {shard}: downloading {len(my_files)} parquets (of {len(half2)} half-2)")

    dl_dir = os.path.join(BN_BASE, "downloads", f"commoncatalog_s{shard}")
    os.makedirs(dl_dir, exist_ok=True)

    downloaded = []
    for i, fname in enumerate(my_files):
        local_path = os.path.join(dl_dir, fname.replace("/", "_"))
        if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
            downloaded.append(local_path)
            continue
        try:
            tmp = hf_hub_download(HF_ID, fname, repo_type="dataset",
                                  cache_dir=os.path.join(dl_dir, ".hf_cache"))
            shutil.copy2(tmp, local_path)
            downloaded.append(local_path)
            if (i + 1) % 50 == 0:
                log.info(f"  Downloaded [{i+1}/{len(my_files)}]")
        except Exception as e:
            log.error(f"  Failed {fname}: {e}")

    log.info(f"Downloaded {len(downloaded)}/{len(my_files)} parquets")
    return downloaded


def process_one_parquet(args):
    """Worker: one parquet → one tar + records."""
    parquet_path, tar_path, prefix, img_col, cap_col = args
    import pyarrow.parquet as pq
    from PIL import Image

    records = []
    errors = 0
    seen = set()

    try:
        pf = pq.ParquetFile(parquet_path)
        with tarfile.open(tar_path, "w") as tf:
            for batch in pf.iter_batches(batch_size=PARQUET_BATCH, columns=[img_col, cap_col]):
                for i in range(len(batch)):
                    try:
                        img_val = batch.column(img_col)[i].as_py()
                        cap_val = batch.column(cap_col)[i].as_py()

                        img_bytes = img_val.get("bytes") if isinstance(img_val, dict) else img_val
                        if not img_bytes or len(img_bytes) < 100:
                            errors += 1; continue

                        caption = cap_val if isinstance(cap_val, str) else ""
                        uid = hashlib.md5(img_bytes[:2048]).hexdigest()
                        if uid in seen: continue
                        seen.add(uid)

                        img = Image.open(io.BytesIO(img_bytes))
                        if img.mode != "RGB": img = img.convert("RGB")
                        w, h = img.size
                        if max(w, h) > MAX_DIM:
                            scale = MAX_DIM / max(w, h)
                            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                            w, h = img.size
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=95)

                        name = f"images/{prefix}/{uid}.jpg"
                        info = tarfile.TarInfo(name=name)
                        info.size = buf.tell()
                        buf.seek(0)
                        tf.addfile(info, buf)
                        records.append({"uid": uid, "caption": caption, "image": name, "height": h, "width": w})
                    except Exception:
                        errors += 1
    except Exception as e:
        return [], errors, str(e)
    return records, errors, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--total-shards", type=int, default=4)
    parser.add_argument("--workers", type=int, default=150)
    args = parser.parse_args()

    # List all parquet files from HF
    from huggingface_hub import HfApi
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["HF_TOKEN"] = HF_TOKEN
    api = HfApi()

    log.info("Listing all parquet files...")
    all_files = api.list_repo_files(HF_ID, repo_type="dataset")
    all_parquets = sorted([f for f in all_files if f.endswith(".parquet")])
    log.info(f"Total: {len(all_parquets)}, half-1 done: {HALF1_COUNT}, half-2: {len(all_parquets) - HALF1_COUNT}")

    # Download this shard's parquets
    my_paths = download_my_parquets(all_parquets, args.shard, args.total_shards)
    if not my_paths:
        log.error("No parquets to process")
        return

    # Detect schema
    import pyarrow.parquet as pq
    sample = pq.ParquetFile(my_paths[0]).read_row_group(0)
    cols_lower = {c.lower(): c for c in sample.column_names}
    img_col = cols_lower.get("jpg") or cols_lower.get("image")
    cap_col = cols_lower.get("caption") or cols_lower.get("blip2_caption")
    log.info(f"Schema: img={img_col}, cap={cap_col}")

    # Process
    output_dir = os.path.join(BN_BASE, "output", f"commoncatalog_s{args.shard}")
    os.makedirs(output_dir, exist_ok=True)
    prefix = "commoncatalog"

    tasks = []
    for i, pf in enumerate(my_paths):
        tar_path = os.path.join(output_dir, f"{prefix}_s{args.shard}_{i}.tar")
        if os.path.exists(tar_path) and os.path.getsize(tar_path) > 512:
            continue
        tasks.append((pf, tar_path, prefix, img_col, cap_col))

    log.info(f"Processing {len(tasks)} parquets with {min(args.workers, max(len(tasks),1))} workers")

    all_records = []
    total_errors = 0
    t0 = time.time()
    done = 0

    eff = min(args.workers, max(len(tasks), 1))
    with ProcessPoolExecutor(max_workers=eff) as executor:
        futures = {executor.submit(process_one_parquet, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            records, errors, err = future.result()
            if err: log.error(f"  Error: {err}")
            all_records.extend(records)
            total_errors += errors
            if done % 50 == 0 or done == len(tasks):
                log.info(f"  [{done}/{len(tasks)}] {len(all_records):,} records, {total_errors:,} err, {time.time()-t0:.0f}s")

    # Dedup + JSON
    seen = set()
    deduped = [r for r in all_records if r["uid"] not in seen and not seen.add(r["uid"])]
    log.info(f"Dedup: {len(all_records)} → {len(deduped)}")

    idx = 0
    for i in range(0, len(deduped), SHARD_SIZE):
        chunk = deduped[i:i + SHARD_SIZE]
        path = os.path.join(output_dir, f"{prefix}_h2s{args.shard}_{idx:04d}.json")
        with open(path, "w") as f:
            json.dump(chunk, f)
        idx += 1
    log.info(f"Wrote {idx} JSON shards")

    # Upload to HDFS
    os.makedirs(HDFS_TARGET, exist_ok=True)
    for jf in sorted(Path(output_dir).glob("*.json")):
        shutil.copy2(str(jf), os.path.join(HDFS_TARGET, jf.name))
    log.info(f"Uploaded to HDFS")

    # Cleanup
    shutil.rmtree(os.path.join(BN_BASE, "downloads", f"commoncatalog_s{args.shard}"), ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    log.info(f"Shard {args.shard} DONE: {len(deduped):,} records, cleaned BN")


if __name__ == "__main__":
    main()
