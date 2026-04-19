#!/usr/bin/env python3
"""Commoncatalog full reprocess: download → process → tar+json to HDFS → cleanup.

Streams in batches of BATCH_SIZE parquets. Each batch:
1. Download parquets from HF
2. Process: parquet → tar (bucket resize for >1024) + JSON records
3. Upload tar + JSON to HDFS via FUSE
4. Delete BN local files
5. Next batch

Usage (4-machine parallel, each handles 1/4 of parquets):
    python3 process_common_final.py --shard 0 --total-shards 4 --workers 150
"""
import argparse, hashlib, io, json, logging, math, os, shutil, tarfile, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BN = "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data"
HDFS_JSON = "/mnt/hdfs/weichow/maskedit/t2i-pt"
HDFS_TAR = "/mnt/hdfs/weichow/maskedit/t2i-pt/commoncatalog_tars"
HF_CACHE = "/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache"
HF_TOKEN = "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"
HF_ID = "common-canvas/commoncatalog-cc-by"
SHARD_SIZE = 10000
BATCH_SIZE = 50

# 1024-tier aspect ratio buckets (all dims multiples of 16)
ASPECT_RATIO_1024 = [
    (1024, 1024),  # 1:1
    (1088, 960),   # ~1.13:1
    (960, 1088),
    (1152, 896),   # 4:3
    (896, 1152),
    (1216, 832),   # 3:2
    (832, 1216),
    (1280, 800),   # 16:10
    (800, 1280),
    (1344, 768),   # ~16:9
    (768, 1344),
    (1408, 704),   # 2:1
    (704, 1408),
    (1472, 672),   # ~2.2:1
    (672, 1472),
    (1536, 640),   # ~2.4:1
    (640, 1536),
]


def find_nearest_bucket(h, w):
    """Find nearest 1024-tier bucket by log-ratio distance."""
    ar = w / max(h, 1)
    best = ASPECT_RATIO_1024[0]
    min_d = float("inf")
    for bh, bw in ASPECT_RATIO_1024:
        d = abs(math.log(max(ar, 1e-6)) - math.log(bw / bh))
        if d < min_d:
            min_d = d
            best = (bh, bw)
    return best


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
                        if max(w, h) > 1024:
                            # Bucket resize: snap to nearest 1024-tier bucket
                            bh, bw = find_nearest_bucket(h, w)
                            img = img.resize((bw, bh), Image.LANCZOS)
                            w, h = bw, bh
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


def upload_to_hdfs(src, dst):
    """Upload file to HDFS via FUSE with 64MB chunks."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as s, open(dst, "wb") as d:
        while True:
            chunk = s.read(64 * 1024 * 1024)
            if not chunk: break
            d.write(chunk)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--total-shards", type=int, default=4)
    p.add_argument("--workers", type=int, default=150)
    p.add_argument("--batch-start", type=int, default=None, help="Start from this batch index (inclusive)")
    p.add_argument("--batch-end", type=int, default=None, help="Stop at this batch index (exclusive)")
    args = p.parse_args()

    os.environ["HF_HOME"] = HF_CACHE
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 min timeout per download
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()

    log.info("Listing parquet files...")
    all_pq = sorted([f for f in api.list_repo_files(HF_ID, repo_type="dataset") if f.endswith(".parquet")])
    # Skip 4096+ resolution parquets (only 12K records but 83GB download, not worth it)
    all_pq = [f for f in all_pq if "4096+" not in f]
    log.info(f"After filtering out 4096+: {len(all_pq)} parquets")
    my_files = [f for i, f in enumerate(all_pq) if i % args.total_shards == args.shard]
    log.info(f"Shard {args.shard}/{args.total_shards}: {len(my_files)} / {len(all_pq)} parquets")

    dl_dir = os.path.join(BN, "downloads", f"ccfinal_s{args.shard}")
    out_dir = os.path.join(BN, "output", f"ccfinal_s{args.shard}")
    os.makedirs(dl_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    os.makedirs(HDFS_JSON, exist_ok=True); os.makedirs(HDFS_TAR, exist_ok=True)

    # Detect schema from first parquet
    log.info("Detecting schema...")
    tmp = hf_hub_download(HF_ID, my_files[0], repo_type="dataset", cache_dir=dl_dir + "/.c")
    import pyarrow.parquet as pq
    sample = pq.ParquetFile(tmp).read_row_group(0)
    cols = {c.lower(): c for c in sample.column_names}
    img_col = cols.get("jpg") or cols.get("image")
    cap_col = cols.get("caption") or cols.get("blip2_caption")
    log.info(f"Schema: img={img_col}, cap={cap_col}")

    # Resume: detect already-uploaded batches by counting existing HDFS tars
    existing_tars = [f for f in os.listdir(HDFS_TAR) if f.startswith(f"cc_s{args.shard}_")]
    existing_jsons = [f for f in os.listdir(HDFS_JSON) if f.startswith(f"commoncatalog_s{args.shard}_")]
    if existing_tars:
        # Each batch produces up to BATCH_SIZE tars; figure out last completed batch
        completed_batches = set()
        for t in existing_tars:
            # format: cc_s{shard}_b{batch}_{idx}.tar
            parts = t.replace(".tar", "").split("_")
            for p in parts:
                if p.startswith("b"):
                    try: completed_batches.add(int(p[1:]))
                    except: pass
        resume_batch = max(completed_batches) + 1 if completed_batches else 0
        resume_start = resume_batch * BATCH_SIZE
        log.info(f"Resuming: found {len(existing_tars)} tars, {len(existing_jsons)} jsons. Skipping to batch {resume_batch} (parquet {resume_start})")
    else:
        resume_start = 0

    grand_records = 0
    grand_json_idx = len(existing_jsons)
    t0 = time.time()

    for batch_start in range(0, len(my_files), BATCH_SIZE):
        batch_num = batch_start // BATCH_SIZE
        if batch_start < resume_start:
            continue
        if args.batch_start is not None and batch_num < args.batch_start:
            continue
        if args.batch_end is not None and batch_num >= args.batch_end:
            log.info(f"Reached batch-end={args.batch_end}, stopping.")
            break
        batch_files = my_files[batch_start:batch_start + BATCH_SIZE]
        log.info(f"\n=== Batch {batch_num} ({batch_start}-{batch_start + len(batch_files)}/{len(my_files)}) ===")

        # Download (parallel with retries)
        def _download_one(idx_fname):
            idx, fname = idx_fname
            local = os.path.join(dl_dir, f"b{batch_num}_{idx}.parquet")
            if os.path.exists(local) and os.path.getsize(local) > 100:
                return local
            for attempt in range(3):
                try:
                    tmp = hf_hub_download(HF_ID, fname, repo_type="dataset", cache_dir=dl_dir + "/.c")
                    shutil.copy2(tmp, local)
                    return local
                except Exception as e:
                    log.warning(f"  DL attempt {attempt+1}/3 fail ({fname}): {e}")
                    time.sleep(5 * (attempt + 1))
            log.error(f"  DL gave up: {fname}")
            return None

        local_paths = []
        with ThreadPoolExecutor(max_workers=8) as dl_pool:
            results = list(dl_pool.map(_download_one, enumerate(batch_files)))
        local_paths = [r for r in results if r is not None]
        log.info(f"  Downloaded {len(local_paths)}/{len(batch_files)}")

        # Process: each parquet → one tar
        tasks = [(lp, os.path.join(out_dir, f"cc_s{args.shard}_b{batch_num}_{i}.tar"), img_col, cap_col)
                 for i, lp in enumerate(local_paths)]
        batch_records = []; batch_errors = 0
        eff = min(args.workers, max(len(tasks), 1))
        with ProcessPoolExecutor(max_workers=eff) as executor:
            futures = {executor.submit(process_one_parquet, t): t for t in tasks}
            for future in as_completed(futures):
                records, errors, err = future.result()
                if err: log.error(f"  Proc err: {err}")
                batch_records.extend(records); batch_errors += errors
        log.info(f"  Processed: {len(batch_records)} records, {batch_errors} errors")

        # Dedup
        seen = set()
        deduped = [r for r in batch_records if r["uid"] not in seen and not seen.add(r["uid"])]

        # Upload tars to HDFS
        for t in tasks:
            tar_local = t[1]
            if os.path.exists(tar_local) and os.path.getsize(tar_local) > 512:
                tar_hdfs = os.path.join(HDFS_TAR, os.path.basename(tar_local))
                upload_to_hdfs(tar_local, tar_hdfs)
        log.info(f"  Tars uploaded to HDFS")

        # Write JSON + upload
        if deduped:
            for i in range(0, len(deduped), SHARD_SIZE):
                chunk = deduped[i:i + SHARD_SIZE]
                jname = f"commoncatalog_s{args.shard}_{grand_json_idx:04d}.json"
                jpath = os.path.join(out_dir, jname)
                with open(jpath, "w") as f: json.dump(chunk, f)
                shutil.copy2(jpath, os.path.join(HDFS_JSON, jname))
                grand_json_idx += 1
            grand_records += len(deduped)
            log.info(f"  Uploaded {grand_json_idx} JSONs, {grand_records:,} total records")

        # Cleanup batch
        for lp in local_paths:
            if os.path.exists(lp): os.remove(lp)
        for t in tasks:
            if os.path.exists(t[1]): os.remove(t[1])

        elapsed = time.time() - t0
        log.info(f"  Running: {grand_records:,} records, {elapsed:.0f}s elapsed")

    # Final cleanup
    shutil.rmtree(dl_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    log.info(f"\nShard {args.shard} DONE: {grand_records:,} records, {grand_json_idx} JSONs, BN cleaned")


if __name__ == "__main__":
    main()
