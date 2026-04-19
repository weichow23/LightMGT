#!/usr/bin/env python3
"""Process Nano-consistent-150k into json+tar format.

Schema: {task_type, instruction, input_images (list), output_image, category}
All records have input+output images → edit format.

Output → HDFS t2i/ (per user request, high-quality data)
"""
import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BN_BASE = "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data"
HDFS_EDIT = "/mnt/hdfs/weichow/maskedit/edit"
SHARD_SIZE = 10000
MAX_DIM = 1536


def process():
    tgz_path = os.path.join(BN_BASE, "downloads", "nano150k", "Nano-150k.tar.gz")
    output_dir = os.path.join(BN_BASE, "output", "nano150k")
    os.makedirs(output_dir, exist_ok=True)

    # Extract tar.gz first (random access on gzip is O(n²))
    extract_dir = os.path.join(BN_BASE, "tmp_extract", "nano150k")
    if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) < 100:
        log.info(f"Extracting {tgz_path} → {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tgz_path, "r:gz") as tf:
            tf.extractall(extract_dir)
        log.info("Extraction done")
    else:
        log.info(f"Already extracted to {extract_dir}")

    log.info("Building file index from extracted files...")
    members = {}
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, extract_dir)
            members[rel] = fpath
    log.info(f"  {len(members)} files indexed")

    # Dummy context for compatibility
    class DummyTF:
        pass
    tf = DummyTF()

    # Read all JSONLs
    jsonl_paths = [p for k, p in members.items() if k.endswith(".jsonl")]
    all_records = []
    for jp in jsonl_paths:
        with open(jp) as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))
    log.info(f"  {len(all_records)} records from {len(jsonl_paths)} jsonls")

    if True:  # keep indentation level

        # Process: extract images, create output tar + JSON
        tar_path = os.path.join(output_dir, "nano150k_images_0.tar")
        log.info(f"Creating tar: {tar_path}")

        final_records = []
        errors = 0

        with tarfile.open(tar_path, "w") as out_tf:
            for i, rec in enumerate(all_records):
                try:
                    # Output image
                    out_key = rec["output_image"]
                    # Try exact match or with Nano-150k/ prefix stripped
                    out_member = None
                    for candidate in [out_key, f"Nano-150k/{out_key}" if not out_key.startswith("Nano-150k/") else out_key,
                                      out_key.replace("Nano-150k/", ""),
                                      os.path.join("Nano-150k", out_key) if not out_key.startswith("Nano-150k") else out_key]:
                        if candidate in members:
                            out_member = members[candidate]
                            break
                    if out_member is None:
                        basename = os.path.basename(out_key)
                        matches = [k for k in members if k.endswith(basename)]
                        if matches:
                            out_member = members[matches[0]]
                    if out_member is None:
                        errors += 1
                        continue

                    out_bytes = open(out_member, "rb").read()
                    uid = hashlib.md5(out_bytes[:2048]).hexdigest()

                    # Resize if needed
                    from PIL import Image
                    img = Image.open(io.BytesIO(out_bytes))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    w, h = img.size
                    if max(w, h) > MAX_DIM:
                        scale = MAX_DIM / max(w, h)
                        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                        w, h = img.size
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=95)
                    out_jpeg = buf.getvalue()

                    # Add output to tar
                    out_name = f"images/nano150k/{uid}_output.jpg"
                    info = tarfile.TarInfo(name=out_name)
                    info.size = len(out_jpeg)
                    out_tf.addfile(info, io.BytesIO(out_jpeg))

                    # Input images
                    input_paths = []
                    for inp_key in rec.get("input_images", []):
                        inp_member = None
                        for candidate in [inp_key, f"Nano-150k/{inp_key}" if not inp_key.startswith("Nano-150k/") else inp_key,
                                          inp_key.replace("Nano-150k/", "")]:
                            if candidate in members:
                                inp_member = members[candidate]
                                break
                        if inp_member is None:
                            basename = os.path.basename(inp_key)
                            matches = [k for k in members if k.endswith(basename)]
                            if matches:
                                inp_member = members[matches[0]]
                        if inp_member is None:
                            continue

                        inp_bytes = open(inp_member, "rb").read()
                        inp_uid = hashlib.md5(inp_bytes[:2048]).hexdigest()

                        inp_img = Image.open(io.BytesIO(inp_bytes))
                        if inp_img.mode != "RGB":
                            inp_img = inp_img.convert("RGB")
                        iw, ih = inp_img.size
                        if max(iw, ih) > MAX_DIM:
                            scale = MAX_DIM / max(iw, ih)
                            inp_img = inp_img.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
                        buf2 = io.BytesIO()
                        inp_img.save(buf2, format="JPEG", quality=95)
                        inp_jpeg = buf2.getvalue()

                        inp_name = f"images/nano150k/{inp_uid}_input.jpg"
                        info2 = tarfile.TarInfo(name=inp_name)
                        info2.size = len(inp_jpeg)
                        out_tf.addfile(info2, io.BytesIO(inp_jpeg))
                        input_paths.append(inp_name)

                    final_records.append({
                        "uid": uid,
                        "instruction": rec["instruction"],
                        "input_images": input_paths,
                        "output_image": out_name,
                        "task_type": rec.get("task_type", ""),
                        "category": rec.get("category", ""),
                        "height": h,
                        "width": w,
                    })

                    if (i + 1) % 10000 == 0:
                        log.info(f"  [{i+1}/{len(all_records)}] {len(final_records)} ok, {errors} errors")

                except Exception as e:
                    errors += 1

        log.info(f"Processing done: {len(final_records)} records, {errors} errors")
        tar_size = os.path.getsize(tar_path) / 1e9
        log.info(f"Tar: {tar_size:.1f} GB")

    # Write JSON shards
    idx = 0
    for i in range(0, len(final_records), SHARD_SIZE):
        chunk = final_records[i:i + SHARD_SIZE]
        path = os.path.join(output_dir, f"nano150k_{idx:04d}.json")
        with open(path, "w") as f:
            json.dump(chunk, f)
        idx += 1
    log.info(f"Wrote {idx} JSON shards")

    # Upload JSON to HDFS
    os.makedirs(HDFS_EDIT, exist_ok=True)
    for jf in sorted(Path(output_dir).glob("nano150k_*.json")):
        shutil.copy2(str(jf), os.path.join(HDFS_EDIT, jf.name))
    log.info(f"Uploaded {idx} JSONs to {HDFS_EDIT}/")

    # Upload tar to HDFS (CRITICAL: do this BEFORE cleanup!)
    hdfs_tar_dest = os.path.join(HDFS_EDIT, "nano150k_images_0.tar")
    log.info(f"Uploading tar ({tar_size:.1f} GB) to {hdfs_tar_dest}...")
    with open(tar_path, "rb") as src, open(hdfs_tar_dest, "wb") as dst:
        while True:
            chunk = src.read(64 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    log.info("Tar uploaded to HDFS")

    # Cleanup BN (only AFTER tar is on HDFS)
    shutil.rmtree(os.path.join(BN_BASE, "downloads", "nano150k"), ignore_errors=True)
    shutil.rmtree(os.path.join(BN_BASE, "tmp_extract", "nano150k"), ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    log.info("Cleaned BN. Done!")


if __name__ == "__main__":
    process()
