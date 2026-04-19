#!/usr/bin/env python3
"""Pipeline commoncatalog: download batch → process → upload → cleanup → repeat.

Each machine handles its shard. Downloads in batches of BATCH_SIZE,
processes immediately, uploads to HDFS, deletes local files. Never accumulates.

Usage:
    python3 process_common_pipeline.py --shard 0 --total-shards 4 --workers 150
"""
import argparse, hashlib, io, json, logging, os, shutil, tarfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BN_BASE = "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data"
HDFS_TARGET = "/mnt/hdfs/weichow/maskedit/t2i-pt"
HF_CACHE = "/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache"
HF_TOKEN = "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"
HF_ID = "common-canvas/commoncatalog-cc-by"
HALF1_COUNT = 2602
MAX_DIM = 1536
SHARD_SIZE = 10000
BATCH_SIZE = 50  # Download+process in batches of 50


def process_one_parquet(args):
    parquet_path, tar_path, img_col, cap_col = args
    import pyarrow.parquet as pq
    from PIL import Image
    records, errors, seen = [], 0, set()
    try:
        pf = pq.ParquetFile(parquet_path)
        with tarfile.open(tar_path, "w") as tf:
            for batch in pf.iter_batches(batch_size=500, columns=[img_col, cap_col]):
                for i in range(len(batch)):
                    try:
                        v = batch.column(img_col)[i].as_py()
                        c = batch.column(cap_col)[i].as_py()
                        b = v.get("bytes") if isinstance(v, dict) else v
                        if not b or len(b) < 100: errors += 1; continue
                        uid = hashlib.md5(b[:2048]).hexdigest()
                        if uid in seen: continue
                        seen.add(uid)
                        img = Image.open(io.BytesIO(b))
                        if img.mode != "RGB": img = img.convert("RGB")
                        w, h = img.size
                        if max(w, h) > MAX_DIM:
                            s = MAX_DIM / max(w, h)
                            img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
                            w, h = img.size
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=95)
                        name = f"images/commoncatalog/{uid}.jpg"
                        info = tarfile.TarInfo(name=name); info.size = buf.tell(); buf.seek(0)
                        tf.addfile(info, buf)
                        records.append({"uid": uid, "caption": c if isinstance(c, str) else "", "image": name, "height": h, "width": w})
                    except: errors += 1
    except Exception as e:
        return [], errors, str(e)
    return records, errors, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--total-shards", type=int, default=4)
    p.add_argument("--workers", type=int, default=150)
    args = p.parse_args()

    os.environ["HF_HOME"] = HF_CACHE
    os.environ["HF_TOKEN"] = HF_TOKEN
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()

    log.info("Listing parquet files...")
    all_files = sorted([f for f in api.list_repo_files(HF_ID, repo_type="dataset") if f.endswith(".parquet")])
    half2 = all_files[HALF1_COUNT:]
    my_files = [f for i, f in enumerate(half2) if i % args.total_shards == args.shard]
    log.info(f"Shard {args.shard}/{args.total_shards}: {len(my_files)} parquets")

    dl_dir = os.path.join(BN_BASE, "downloads", f"common_pipe_s{args.shard}")
    out_dir = os.path.join(BN_BASE, "output", f"common_pipe_s{args.shard}")
    os.makedirs(dl_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(HDFS_TARGET, exist_ok=True)

    # Detect schema from first file
    log.info("Detecting schema...")
    tmp = hf_hub_download(HF_ID, my_files[0], repo_type="dataset", cache_dir=os.path.join(dl_dir, ".cache"))
    import pyarrow.parquet as pq
    sample = pq.ParquetFile(tmp).read_row_group(0)
    cols = {c.lower(): c for c in sample.column_names}
    img_col = cols.get("jpg") or cols.get("image")
    cap_col = cols.get("caption") or cols.get("blip2_caption")
    log.info(f"Schema: img={img_col}, cap={cap_col}")

    grand_records = 0
    grand_json_idx = 0
    t0 = time.time()

    # Process in batches
    for batch_start in range(0, len(my_files), BATCH_SIZE):
        batch_files = my_files[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE
        log.info(f"\n=== Batch {batch_num} ({batch_start}-{batch_start+len(batch_files)}/{len(my_files)}) ===")

        # Download batch
        local_paths = []
        for fname in batch_files:
            local = os.path.join(dl_dir, f"b{batch_num}_{len(local_paths)}.parquet")
            if os.path.exists(local) and os.path.getsize(local) > 100:
                local_paths.append(local)
                continue
            try:
                tmp = hf_hub_download(HF_ID, fname, repo_type="dataset", cache_dir=os.path.join(dl_dir, ".cache"))
                shutil.copy2(tmp, local)
                local_paths.append(local)
            except Exception as e:
                log.error(f"  DL fail: {e}")
        log.info(f"  Downloaded {len(local_paths)}/{len(batch_files)}")

        # Process batch
        tasks = []
        for i, lp in enumerate(local_paths):
            tar_path = os.path.join(out_dir, f"cc_s{args.shard}_b{batch_num}_{i}.tar")
            tasks.append((lp, tar_path, img_col, cap_col))

        batch_records = []
        batch_errors = 0
        eff = min(args.workers, max(len(tasks), 1))
        with ProcessPoolExecutor(max_workers=eff) as executor:
            futures = {executor.submit(process_one_parquet, t): t for t in tasks}
            for future in as_completed(futures):
                records, errors, err = future.result()
                if err: log.error(f"  Proc err: {err}")
                batch_records.extend(records)
                batch_errors += errors

        log.info(f"  Processed: {len(batch_records)} records, {batch_errors} errors")

        # Dedup
        seen = set()
        deduped = [r for r in batch_records if r["uid"] not in seen and not seen.add(r["uid"])]

        # Write JSON shard + upload
        if deduped:
            for i in range(0, len(deduped), SHARD_SIZE):
                chunk = deduped[i:i + SHARD_SIZE]
                jname = f"commoncatalog_h2s{args.shard}_{grand_json_idx:04d}.json"
                jpath = os.path.join(out_dir, jname)
                with open(jpath, "w") as f:
                    json.dump(chunk, f)
                shutil.copy2(jpath, os.path.join(HDFS_TARGET, jname))
                grand_json_idx += 1
            grand_records += len(deduped)
            log.info(f"  Uploaded {grand_json_idx} JSONs so far, {grand_records:,} total records")

        # Cleanup batch files (free BN space)
        for lp in local_paths:
            os.remove(lp) if os.path.exists(lp) else None
        for t in tasks:
            os.remove(t[1]) if os.path.exists(t[1]) else None

        elapsed = time.time() - t0
        rate = grand_records / max(elapsed, 1)
        log.info(f"  Running: {grand_records:,} records, {rate:.0f} rec/s, {elapsed:.0f}s elapsed")

    # Final cleanup
    shutil.rmtree(dl_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    log.info(f"\nShard {args.shard} DONE: {grand_records:,} records, {grand_json_idx} JSONs, BN cleaned")


if __name__ == "__main__":
    main()
