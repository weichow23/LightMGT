#!/usr/bin/env python3
"""Reprocess pico-banana-400k with BOTH input + output images.

Previous processing only saved output_image + text. This fixes it to include:
- input_image (original, from Flickr/OpenImages URLs)
- output_image (edited, from Apple CDN via manifest)
- instruction (edit text)
- edit_type (category)
- summarized_text (short version)

Strategy:
1. Download sft.jsonl from Apple CDN (already has all metadata)
2. Download input images from Flickr URLs (parallel, with retry)
3. Reuse existing output images from HDFS tar
4. Create new JSON shards + input image tar → HDFS edit/
"""

import hashlib
import io
import json
import logging
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BN_BASE = "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data"
HDFS_EDIT = "/mnt/hdfs/weichow/maskedit/edit"
HDFS_T2I = "/mnt/hdfs/weichow/maskedit/t2i"
SFT_URL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl"
MANIFEST_URL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/manifest/sft_manifest.txt"
SHARD_SIZE = 10000


def download_file(url, dest):
    """Download a file with retry."""
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30, stream=True)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt == 2:
                return False
            time.sleep(1)
    return False


def download_image_worker(args):
    """Worker: download one input image from URL."""
    url, save_path, uid = args
    if os.path.exists(save_path) and os.path.getsize(save_path) > 100:
        return uid, True, 0  # already exists
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and len(r.content) > 100:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return uid, True, len(r.content)
        return uid, False, 0
    except Exception:
        return uid, False, 0


def load_existing_output_index():
    """Load tar index for existing pico_banana output images on HDFS."""
    import pickle
    import glob as glob_mod

    # Try loading from cache
    cache_dir = "/mnt/bn/search-auto-eval/zhouwei/maskedit_cache/tar_index"
    tar_pattern = f"{HDFS_T2I}/pico_banana_images.tar"
    cache_key = hashlib.md5(tar_pattern.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            idx = pickle.load(f)
        log.info(f"Loaded output tar index: {len(idx)} entries")
        return idx

    # Build index from tar
    tar_path = f"{HDFS_T2I}/pico_banana_images.tar"
    if not os.path.exists(tar_path):
        log.error(f"Output tar not found: {tar_path}")
        return {}

    log.info(f"Building index from {tar_path}...")
    idx = {}
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if member.isfile():
                idx[member.name] = (tar_path, member.offset_data, member.size)

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(idx, f)
    log.info(f"Built and cached index: {len(idx)} entries")
    return idx


def main():
    work_dir = os.path.join(BN_BASE, "pico_banana_reprocess")
    input_img_dir = os.path.join(work_dir, "input_images")
    os.makedirs(input_img_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # 1. Download sft.jsonl
    sft_path = os.path.join(work_dir, "sft.jsonl")
    if not os.path.exists(sft_path):
        log.info(f"Downloading sft.jsonl from {SFT_URL}...")
        if not download_file(SFT_URL, sft_path):
            log.error("Failed to download sft.jsonl")
            return
    else:
        log.info("sft.jsonl already exists")

    # 2. Parse all records
    records = []
    with open(sft_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    log.info(f"Loaded {len(records)} SFT records")

    # 3. Load existing output image index (from HDFS tar)
    output_idx = load_existing_output_index()

    # 4. Download input images (parallel, 100 threads)
    log.info(f"Downloading input images from Flickr URLs (100 threads)...")
    tasks = []
    uid_map = {}  # uid -> record
    for i, rec in enumerate(records):
        url = rec.get("open_image_input_url", "")
        if not url:
            continue
        uid = hashlib.md5(f"pico_{i}".encode()).hexdigest()
        save_path = os.path.join(input_img_dir, f"{uid}.jpg")
        tasks.append((url, save_path, uid))
        uid_map[uid] = (i, rec)

    success = 0
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(download_image_worker, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            uid, ok, sz = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            if done % 10000 == 0:
                elapsed = time.time() - t0
                log.info(f"  [{done}/{len(tasks)}] success={success} failed={failed} {elapsed:.0f}s")

    log.info(f"Input images: {success} downloaded, {failed} failed out of {len(tasks)}")

    # 5. Build final records with both input + output
    log.info("Building edit records with input + output images...")
    edit_records = []
    for uid, (idx, rec) in uid_map.items():
        input_path = os.path.join(input_img_dir, f"{uid}.jpg")
        if not os.path.exists(input_path) or os.path.getsize(input_path) < 100:
            continue

        # Check output image exists in tar
        output_key = f"images/pico_banana/{uid}.png"
        if output_key not in output_idx:
            continue

        # Get input dimensions
        try:
            from PIL import Image
            img = Image.open(input_path)
            w, h = img.size
        except Exception:
            continue

        edit_records.append({
            "uid": uid,
            "instruction": rec["text"],
            "input_image": f"images/pico_banana_input/{uid}.jpg",
            "output_image": output_key,
            "edit_type": rec.get("edit_type", ""),
            "summarized_text": rec.get("summarized_text", ""),
            "height": h,
            "width": w,
        })

    log.info(f"Final edit records: {len(edit_records)} (from {len(records)} original)")

    # 6. Create input image tar
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    tar_path = os.path.join(output_dir, "pico_banana_input_images.tar")
    log.info(f"Creating input image tar: {tar_path}")
    with tarfile.open(tar_path, "w") as tf:
        for rec in edit_records:
            uid = rec["uid"]
            local = os.path.join(input_img_dir, f"{uid}.jpg")
            if os.path.exists(local):
                tf.add(local, arcname=rec["input_image"])
    tar_size = os.path.getsize(tar_path) / 1e9
    log.info(f"Input tar: {tar_size:.1f} GB")

    # 7. Write JSON shards
    shard_idx = 0
    for i in range(0, len(edit_records), SHARD_SIZE):
        chunk = edit_records[i:i + SHARD_SIZE]
        path = os.path.join(output_dir, f"pico_banana_edit_{shard_idx:04d}.json")
        with open(path, "w") as f:
            json.dump(chunk, f)
        shard_idx += 1
    log.info(f"Wrote {shard_idx} JSON shards")

    # 8. Upload to HDFS edit/
    import shutil
    os.makedirs(HDFS_EDIT, exist_ok=True)
    for jf in sorted(Path(output_dir).glob("pico_banana_edit_*.json")):
        shutil.copy2(str(jf), os.path.join(HDFS_EDIT, jf.name))
    log.info(f"Uploaded {shard_idx} JSONs to {HDFS_EDIT}/")

    # Upload input tar to HDFS
    hdfs_tar = os.path.join(HDFS_EDIT, "pico_banana_input_images.tar")
    log.info(f"Uploading input tar to HDFS ({tar_size:.1f} GB)...")
    with open(tar_path, "rb") as src, open(hdfs_tar, "wb") as dst:
        while True:
            chunk = src.read(64 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    log.info("Input tar uploaded to HDFS")

    # 9. Cleanup BN
    shutil.rmtree(input_img_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    log.info("BN cleaned. Done!")


if __name__ == "__main__":
    main()
