#!/usr/bin/env python3
"""Process a SHARD of commoncatalog parquets (for parallel 3-machine processing).

Usage:
    python3 process_common_shard.py --shard 0 --total-shards 3 --workers 150
    python3 process_common_shard.py --shard 1 --total-shards 3 --workers 150
    python3 process_common_shard.py --shard 2 --total-shards 3 --workers 150

Each shard processes parquets where index % total_shards == shard.
"""
import argparse
import hashlib
import io
import json
import logging
import os
import shutil
import struct
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BN_BASE = "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data"
HDFS_TARGET = "/mnt/hdfs/weichow/maskedit/t2i-pt"
PARQUET_BATCH = 500
MAX_DIM = 1536
SHARD_SIZE = 10000


def resize_if_needed(img):
    from PIL import Image
    w, h = img.size
    if max(w, h) <= MAX_DIM:
        return img
    scale = MAX_DIM / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def process_one_parquet(args):
    parquet_path, tar_path, prefix, img_col, cap_col, w_col, h_col, worker_id = args
    import pyarrow.parquet as pq
    from PIL import Image

    records = []
    errors = 0
    written = 0
    seen = set()

    try:
        pf = pq.ParquetFile(parquet_path)
        columns = [img_col, cap_col]
        if w_col: columns.append(w_col)
        if h_col: columns.append(h_col)

        with tarfile.open(tar_path, "w") as tf:
            for batch in pf.iter_batches(batch_size=PARQUET_BATCH, columns=columns):
                img_arr = batch.column(img_col)
                cap_arr = batch.column(cap_col)

                for i in range(len(batch)):
                    try:
                        img_val = img_arr[i].as_py()
                        cap_val = cap_arr[i].as_py()

                        if isinstance(img_val, dict):
                            img_bytes = img_val.get("bytes")
                        elif isinstance(img_val, bytes):
                            img_bytes = img_val
                        else:
                            errors += 1; continue

                        if not img_bytes or len(img_bytes) < 100:
                            errors += 1; continue

                        caption = cap_val if isinstance(cap_val, str) and cap_val else ""
                        uid = hashlib.md5(img_bytes[:2048]).hexdigest()
                        if uid in seen: continue
                        seen.add(uid)

                        img = Image.open(io.BytesIO(img_bytes))
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        img = resize_if_needed(img)
                        w, h = img.size
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=95)
                        jpeg = buf.getvalue()

                        name = f"images/{prefix}/{uid}.jpg"
                        info = tarfile.TarInfo(name=name)
                        info.size = len(jpeg)
                        tf.addfile(info, io.BytesIO(jpeg))

                        records.append({"uid": uid, "caption": caption, "image": name, "height": h, "width": w})
                        written += 1
                    except Exception:
                        errors += 1
    except Exception as e:
        return [], errors, 0, str(e)
    return records, errors, written, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--total-shards", type=int, default=3)
    parser.add_argument("--workers", type=int, default=150)
    args = parser.parse_args()

    dl_dir = os.path.join(BN_BASE, "downloads", "commoncatalog")
    output_dir = os.path.join(BN_BASE, "output", f"commoncatalog_s{args.shard}")
    os.makedirs(output_dir, exist_ok=True)
    prefix = "commoncatalog"

    all_parquets = sorted(Path(dl_dir).rglob("*.parquet"))
    my_parquets = [p for i, p in enumerate(all_parquets) if i % args.total_shards == args.shard]
    log.info(f"Shard {args.shard}/{args.total_shards}: {len(my_parquets)} parquets (of {len(all_parquets)} total)")

    if not my_parquets:
        log.error("No parquets for this shard")
        return

    # Detect schema
    import pyarrow.parquet as pq
    sample = pq.ParquetFile(str(my_parquets[0])).read_row_group(0)
    cols_lower = {c.lower(): c for c in sample.column_names}
    img_col = cols_lower.get("jpg") or cols_lower.get("image") or cols_lower.get("img")
    cap_col = cols_lower.get("caption") or cols_lower.get("blip2_caption") or cols_lower.get("text")
    w_col = cols_lower.get("width")
    h_col = cols_lower.get("height")
    log.info(f"Schema: img={img_col}, cap={cap_col}, w={w_col}, h={h_col}")

    tasks = []
    for i, pf in enumerate(my_parquets):
        tar_path = os.path.join(output_dir, f"{prefix}_s{args.shard}_{i}.tar")
        if os.path.exists(tar_path) and os.path.getsize(tar_path) > 512:
            continue
        tasks.append((str(pf), tar_path, prefix, img_col, cap_col, w_col, h_col, i))

    log.info(f"Processing {len(tasks)} parquets with {min(args.workers, len(tasks))} workers")

    all_records = []
    total_errors = 0
    total_written = 0
    t0 = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = {executor.submit(process_one_parquet, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            records, errors, written, err = future.result()
            if err: log.error(f"  Worker error: {err}")
            all_records.extend(records)
            total_errors += errors
            total_written += written
            if done % 50 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                log.info(f"  [{done}/{len(tasks)}] {total_written:,} imgs, {total_errors:,} err, {elapsed:.0f}s")

    # Dedup
    seen = set()
    deduped = [r for r in all_records if r["uid"] not in seen and not seen.add(r["uid"])]
    log.info(f"Dedup: {len(all_records)} → {len(deduped)}")

    # Write JSON shards
    idx = 0
    for i in range(0, len(deduped), SHARD_SIZE):
        chunk = deduped[i:i + SHARD_SIZE]
        path = os.path.join(output_dir, f"{prefix}_s{args.shard}_{idx:04d}.json")
        with open(path, "w") as f:
            json.dump(chunk, f)
        idx += 1
    log.info(f"Wrote {idx} JSON shards ({len(deduped):,} records)")

    # Upload to HDFS
    os.makedirs(HDFS_TARGET, exist_ok=True)
    for jf in sorted(Path(output_dir).glob("*.json")):
        shutil.copy2(str(jf), os.path.join(HDFS_TARGET, jf.name))
    log.info(f"Uploaded {idx} JSONs to HDFS")

    # Cleanup output (tars stay for now, cleanup later)
    log.info(f"Shard {args.shard} DONE: {len(deduped):,} records")


if __name__ == "__main__":
    main()
