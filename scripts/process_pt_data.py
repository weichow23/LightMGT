#!/usr/bin/env python3
"""Process pre-training data into LightMGT training format (json + tar).

Streams HuggingFace parquet/webdataset → json shards + tar archives.
Memory-safe: each worker streams one parquet → one tar, never holds full dataset.

Output:
  Tar:  /mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/output/{dataset}/
  JSON: /mnt/hdfs/weichow/maskedit/t2i-pt/  (via hdfs dfs -put)

Machine allocation:
  cpu_light:  laion_512, laion_1024, cc12m
  cpu_light2: journeydb_p1, journeydb_p2

Usage:
    python3 process_pt_data.py --dataset laion_512 --download-only
    python3 process_pt_data.py --dataset laion_512 --process-only --workers 50
    python3 process_pt_data.py --dataset laion_512 --upload-only
    python3 process_pt_data.py --dataset laion_512 --workers 50   # all steps
"""

import argparse
import hashlib
import io
import json
import logging
import os
import struct
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────

BN_BASE = "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data"
HDFS_TARGET = "/mnt/hdfs/weichow/maskedit/t2i-pt"
HF_CACHE = "/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache"
HF_TOKEN = "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"

SHARD_SIZE = 10000  # records per JSON shard
PARQUET_BATCH = 500  # rows to read at a time from parquet (memory control)

DATASETS = {
    # ─── P0 ───
    "laion_512": {
        "hf_id": "limingcv/LAION_Aesthetics_512",
        "type": "parquet",
        "prefix": "laion_aes_512",
    },
    "laion_1024": {
        "hf_id": "limingcv/LAION_Aesthetics_1024",
        "type": "parquet",
        "prefix": "laion_aes_1024",
    },
    "journeydb_p1": {
        "hf_id": "limingcv/JourneyDB_part1",
        "type": "parquet",
        "prefix": "journeydb_p1",
    },
    "journeydb_p2": {
        "hf_id": "limingcv/JourneyDB_part2",
        "type": "parquet",
        "prefix": "journeydb_p2",
    },
    "cc12m": {
        "hf_id": "pixparse/cc12m-wds",
        "type": "webdataset",
        "prefix": "cc12m",
    },
    "park": {
        "hf_id": "MeissonFlow/park",
        "type": "parquet",
        "prefix": "meissonic_park",
    },
    # ─── P1 ───
    "multigen": {
        "hf_id": "limingcv/MultiGen-20M_train",
        "type": "parquet",
        "prefix": "multigen_20m",
    },
    "text2image_2m": {
        "hf_id": "jackyhate/text-to-image-2M",
        "type": "webdataset",
        "prefix": "text2img_2m",
    },
    "flux_journey": {
        "hf_id": "WeiChow/FLUX-dev-Journey",
        "type": "parquet",
        "prefix": "flux_journey",
    },
    # ─── P2 ───
    "dalle3_synthetic": {
        "hf_id": "ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions",
        "type": "webdataset",
        "prefix": "dalle3_synth",
    },
    "midjourney_v6": {
        "hf_id": "terminusresearch/midjourney-v6-520k-raw",
        "type": "webdataset",
        "prefix": "midjourney_v6",
    },
    "laion_600k": {
        "hf_id": "woweenie/laion-600k-aesthetic-6.5plus-768-32k",
        "type": "parquet",
        "prefix": "laion_600k",
    },
    # ─── P3 ───
    "commoncatalog": {
        "hf_id": "common-canvas/commoncatalog-cc-by",
        "type": "parquet",
        "prefix": "commoncatalog",
    },
}


# ─── Image Utilities ────────────────────────────────────────────────────


def detect_image_format(img_bytes):
    """Detect image format from magic bytes."""
    if img_bytes[:3] == b"\xff\xd8\xff":
        return "jpg"
    if img_bytes[:4] == b"\x89PNG":
        return "png"
    if img_bytes[:4] == b"RIFF" and len(img_bytes) > 12 and img_bytes[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def _jpeg_dimensions(data):
    """Parse JPEG dimensions from SOF marker."""
    i = 2
    while i < len(data) - 8:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            h = struct.unpack(">H", data[i + 5 : i + 7])[0]
            w = struct.unpack(">H", data[i + 7 : i + 9])[0]
            return w, h
        if marker == 0xD9:
            break
        if marker in range(0xD0, 0xDA) or marker == 0x01:
            i += 2
            continue
        if i + 4 > len(data):
            break
        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
        i += 2 + length
    return None


def get_image_dimensions_fast(img_bytes):
    """Get (width, height) from image bytes without PIL."""
    try:
        if img_bytes[:3] == b"\xff\xd8\xff":
            result = _jpeg_dimensions(img_bytes)
            if result:
                return result
        elif img_bytes[:4] == b"\x89PNG" and len(img_bytes) > 24:
            w = struct.unpack(">I", img_bytes[16:20])[0]
            h = struct.unpack(">I", img_bytes[20:24])[0]
            return w, h
        elif img_bytes[:4] == b"RIFF" and len(img_bytes) > 30 and img_bytes[8:12] == b"WEBP":
            if img_bytes[12:16] == b"VP8 " and len(img_bytes) > 30:
                w = struct.unpack("<H", img_bytes[26:28])[0] & 0x3FFF
                h = struct.unpack("<H", img_bytes[28:30])[0] & 0x3FFF
                return w, h
            elif img_bytes[12:16] == b"VP8L" and len(img_bytes) > 25:
                bits = struct.unpack("<I", img_bytes[21:25])[0]
                w = (bits & 0x3FFF) + 1
                h = ((bits >> 14) & 0x3FFF) + 1
                return w, h
    except Exception:
        pass
    return None


def get_image_dimensions(img_bytes):
    """Get (width, height), with PIL fallback."""
    result = get_image_dimensions_fast(img_bytes)
    if result:
        return result
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        return img.size
    except Exception:
        return 256, 256


MAX_DIM = 1536  # Max dimension for any side (fits within 1024 bucket system)


def resize_if_needed(img, max_dim=MAX_DIM):
    """Resize image so max(w,h) <= max_dim, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def ensure_jpeg(img_bytes):
    """Convert to JPEG, resize if > MAX_DIM. Returns (jpeg_bytes, width, height) or None."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = resize_if_needed(img)
        w, h = img.size

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), w, h
    except Exception:
        return None


# ─── Schema Detection ───────────────────────────────────────────────────


def detect_columns(table):
    """Auto-detect image, caption, width, height columns from a pyarrow table.

    Returns (img_col, cap_col, w_col_or_None, h_col_or_None).
    """
    names_lower = {c.lower(): c for c in table.column_names}

    img_col = None
    for candidate in ["image", "img", "jpg", "photo", "output_image", "png"]:
        if candidate in names_lower:
            img_col = names_lower[candidate]
            break
    if img_col is None:
        raise ValueError(f"No image column. Columns: {table.column_names}")

    cap_col = None
    for candidate in ["caption", "text", "prompt", "description", "annotation"]:
        if candidate in names_lower:
            cap_col = names_lower[candidate]
            break
    if cap_col is None:
        for c in table.column_names:
            if "caption" in c.lower() or "text" in c.lower():
                cap_col = c
                break
    if cap_col is None:
        raise ValueError(f"No caption column. Columns: {table.column_names}")

    # Optional width/height columns
    w_col = names_lower.get("image_width") or names_lower.get("width")
    h_col = names_lower.get("image_height") or names_lower.get("height")

    return img_col, cap_col, w_col, h_col


# ─── Download ───────────────────────────────────────────────────────────


def download_dataset(dataset_key):
    """Download HuggingFace dataset to BN using Python API.

    Always calls snapshot_download which handles resume internally
    (skips already-downloaded files, re-downloads incomplete ones).
    """
    from huggingface_hub import snapshot_download

    cfg = DATASETS[dataset_key]
    hf_id = cfg["hf_id"]
    local_dir = os.path.join(BN_BASE, "downloads", dataset_key)
    os.makedirs(local_dir, exist_ok=True)

    existing = list(Path(local_dir).rglob("*.parquet")) + list(Path(local_dir).rglob("*.tar"))
    log.info(f"Downloading {hf_id} → {local_dir} (existing: {len(existing)} files)")

    os.environ["HF_HOME"] = HF_CACHE
    os.environ["HF_TOKEN"] = HF_TOKEN

    snapshot_download(
        repo_id=hf_id,
        repo_type="dataset",
        local_dir=local_dir,
    )

    n_files = len(list(Path(local_dir).rglob("*.parquet")) + list(Path(local_dir).rglob("*.tar")))
    log.info(f"Download complete: {local_dir} ({n_files} data files)")
    return local_dir


# ─── Parquet Worker ─────────────────────────────────────────────────────


def _worker_process_parquet(args):
    """Worker: stream one parquet file → one tar file + return JSON records.

    Memory: holds ~PARQUET_BATCH images at a time (~250MB), not the whole file.
    """
    parquet_path, tar_path, prefix, img_col, cap_col, w_col, h_col, worker_id = args
    import pyarrow.parquet as pq

    records = []
    errors = 0
    written = 0
    seen_uids = set()

    try:
        pf = pq.ParquetFile(parquet_path)
        columns = [img_col, cap_col]
        if w_col:
            columns.append(w_col)
        if h_col:
            columns.append(h_col)

        with tarfile.open(tar_path, "w") as tf:
            for batch in pf.iter_batches(batch_size=PARQUET_BATCH, columns=columns):
                img_array = batch.column(img_col)
                cap_array = batch.column(cap_col)
                w_array = batch.column(w_col) if w_col else None
                h_array = batch.column(h_col) if h_col else None

                for i in range(len(batch)):
                    try:
                        img_val = img_array[i].as_py()
                        cap_val = cap_array[i].as_py()

                        # Extract image bytes
                        if isinstance(img_val, dict):
                            img_bytes = img_val.get("bytes")
                        elif isinstance(img_val, bytes):
                            img_bytes = img_val
                        else:
                            errors += 1
                            continue

                        if not img_bytes or len(img_bytes) < 100:
                            errors += 1
                            continue

                        # Caption
                        caption = cap_val if isinstance(cap_val, str) and cap_val else ""

                        # UID from content hash (dedup)
                        uid = hashlib.md5(img_bytes[:2048]).hexdigest()
                        if uid in seen_uids:
                            continue
                        seen_uids.add(uid)

                        # Get dimensions from parquet or image bytes
                        w = w_array[i].as_py() if w_array else None
                        h = h_array[i].as_py() if h_array else None

                        # Ensure JPEG + resize to MAX_DIM
                        result = ensure_jpeg(img_bytes)
                        if result is None:
                            errors += 1
                            continue
                        jpeg_bytes, w, h = result

                        # Write to tar
                        member_name = f"images/{prefix}/{uid}.jpg"
                        info = tarfile.TarInfo(name=member_name)
                        info.size = len(jpeg_bytes)
                        tf.addfile(info, io.BytesIO(jpeg_bytes))

                        records.append({
                            "uid": uid,
                            "caption": caption,
                            "image": member_name,
                            "height": int(h),
                            "width": int(w),
                        })
                        written += 1
                    except Exception:
                        errors += 1

    except Exception as e:
        return [], errors, 0, str(e)

    return records, errors, written, None


# ─── Parquet Dataset Pipeline ──────────────────────────────────────────


def process_parquet_dataset(dataset_key, num_workers=50):
    """Process parquet dataset: parallel workers, each parquet → one tar."""
    cfg = DATASETS[dataset_key]
    prefix = cfg["prefix"]
    download_dir = os.path.join(BN_BASE, "downloads", dataset_key)
    output_dir = os.path.join(BN_BASE, "output", dataset_key)
    os.makedirs(output_dir, exist_ok=True)

    # Find parquet files
    parquet_files = sorted(Path(download_dir).rglob("*.parquet"))
    if not parquet_files:
        log.error(f"No parquet files in {download_dir}")
        return 0

    log.info(f"Found {len(parquet_files)} parquet files")

    # Detect schema from first row group of first parquet
    import pyarrow.parquet as pq
    sample = pq.ParquetFile(str(parquet_files[0])).read_row_group(0)
    img_col, cap_col, w_col, h_col = detect_columns(sample)
    log.info(f"Schema: image={img_col}, caption={cap_col}, w={w_col}, h={h_col}")
    log.info(f"Columns: {sample.column_names}")
    del sample

    # Skip already-processed parquet files
    tasks = []
    for i, pf in enumerate(parquet_files):
        tar_path = os.path.join(output_dir, f"{prefix}_images_{i}.tar")
        if os.path.exists(tar_path) and os.path.getsize(tar_path) > 512:
            log.info(f"  Skip (exists): {tar_path}")
            continue
        tasks.append((str(pf), tar_path, prefix, img_col, cap_col, w_col, h_col, i))

    if not tasks:
        log.info("All parquet files already processed. Collecting existing records...")
        return _collect_existing_records(output_dir, prefix, parquet_files)

    log.info(f"Processing {len(tasks)} parquet files with {num_workers} workers")

    all_records = []
    total_errors = 0
    total_written = 0
    t0 = time.time()
    completed = 0

    effective_workers = min(num_workers, len(tasks))
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(_worker_process_parquet, t): t for t in tasks}

        for future in as_completed(futures):
            completed += 1
            task_info = futures[future]
            try:
                records, errors, written, err_msg = future.result()
                if err_msg:
                    log.error(f"  Worker error on {task_info[0]}: {err_msg}")
                all_records.extend(records)
                total_errors += errors
                total_written += written

                if completed % 5 == 0 or completed == len(tasks):
                    elapsed = time.time() - t0
                    rate = total_written / max(elapsed, 1)
                    log.info(
                        f"  [{completed}/{len(tasks)}] "
                        f"{total_written:,} images, "
                        f"{total_errors:,} errors, "
                        f"{rate:.0f} img/s, "
                        f"{elapsed:.0f}s elapsed"
                    )
            except Exception as e:
                log.error(f"  Future exception: {e}")

    elapsed = time.time() - t0
    log.info(f"Processing done: {total_written:,} images, {total_errors:,} errors in {elapsed:.0f}s")

    # Global dedup across all workers
    seen = set()
    deduped = []
    for r in all_records:
        if r["uid"] not in seen:
            seen.add(r["uid"])
            deduped.append(r)
    log.info(f"Global dedup: {len(all_records)} → {len(deduped)} records")
    all_records = deduped

    # Write JSON shards
    _write_json_shards(all_records, output_dir, prefix)

    return len(all_records)


def _collect_existing_records(output_dir, prefix, parquet_files):
    """Collect records from existing JSON shards (for resume)."""
    json_files = sorted(Path(output_dir).glob(f"{prefix}_*.json"))
    all_records = []
    for jf in json_files:
        try:
            with open(jf) as f:
                all_records.extend(json.load(f))
        except Exception:
            pass
    log.info(f"Collected {len(all_records)} existing records from {len(json_files)} JSON shards")
    return len(all_records)


# ─── WebDataset Processing ─────────────────────────────────────────────


def process_webdataset(dataset_key, num_workers=50):
    """Process WebDataset tar files."""
    cfg = DATASETS[dataset_key]
    prefix = cfg["prefix"]
    download_dir = os.path.join(BN_BASE, "downloads", dataset_key)
    output_dir = os.path.join(BN_BASE, "output", dataset_key)
    os.makedirs(output_dir, exist_ok=True)

    src_tars = sorted(Path(download_dir).rglob("*.tar"))
    if not src_tars:
        log.error(f"No tar files in {download_dir}")
        return 0

    log.info(f"Found {len(src_tars)} source tar files")

    all_records = []
    total_errors = 0
    tar_out_idx = 0
    current_tar = None
    current_tar_count = 0
    seen_uids = set()

    for src_idx, src_tar_path in enumerate(src_tars):
        log.info(f"  Reading source tar [{src_idx+1}/{len(src_tars)}]: {src_tar_path.name}")
        try:
            with tarfile.open(str(src_tar_path), "r") as tf_in:
                # Group members by key (stem without extension)
                members_by_key = {}
                for member in tf_in.getmembers():
                    if not member.isfile():
                        continue
                    parts = member.name.rsplit(".", 1)
                    key = parts[0]
                    ext = parts[1].lower() if len(parts) > 1 else ""
                    if key not in members_by_key:
                        members_by_key[key] = {}
                    members_by_key[key][ext] = member

                for key, parts in members_by_key.items():
                    try:
                        # Find image
                        img_member = None
                        for img_ext in ["jpg", "jpeg", "png", "webp"]:
                            if img_ext in parts:
                                img_member = parts[img_ext]
                                break
                        if img_member is None:
                            total_errors += 1
                            continue

                        # Find caption
                        caption = ""
                        for txt_ext in ["txt", "caption", "json"]:
                            if txt_ext in parts:
                                raw = tf_in.extractfile(parts[txt_ext]).read()
                                txt = raw.decode("utf-8", errors="replace").strip()
                                if txt_ext == "json":
                                    try:
                                        j = json.loads(txt)
                                        caption = j.get("caption", j.get("text", j.get("prompt", "")))
                                    except json.JSONDecodeError:
                                        caption = txt
                                else:
                                    caption = txt
                                break

                        img_bytes = tf_in.extractfile(img_member).read()
                        if not img_bytes or len(img_bytes) < 100:
                            total_errors += 1
                            continue

                        uid = hashlib.md5(img_bytes[:2048]).hexdigest()
                        if uid in seen_uids:
                            continue
                        seen_uids.add(uid)

                        result = ensure_jpeg(img_bytes)
                        if result is None:
                            total_errors += 1
                            continue
                        jpeg_bytes, w, h = result

                        # Open new output tar if needed
                        if current_tar is None:
                            tar_out_path = os.path.join(output_dir, f"{prefix}_images_{tar_out_idx}.tar")
                            current_tar = tarfile.open(tar_out_path, "w")
                            current_tar_count = 0

                        member_name = f"images/{prefix}/{uid}.jpg"
                        info = tarfile.TarInfo(name=member_name)
                        info.size = len(jpeg_bytes)
                        current_tar.addfile(info, io.BytesIO(jpeg_bytes))
                        current_tar_count += 1

                        all_records.append({
                            "uid": uid,
                            "caption": caption,
                            "image": member_name,
                            "height": h,
                            "width": w,
                        })

                        # Rotate tar if full
                        if current_tar_count >= 50000:
                            current_tar.close()
                            size_gb = os.path.getsize(
                                os.path.join(output_dir, f"{prefix}_images_{tar_out_idx}.tar")
                            ) / 1e9
                            log.info(f"    Closed tar {tar_out_idx} ({current_tar_count} imgs, {size_gb:.1f}GB)")
                            tar_out_idx += 1
                            current_tar = None

                    except Exception:
                        total_errors += 1

        except Exception as e:
            log.error(f"  Failed: {e}")

    # Close last tar
    if current_tar is not None:
        current_tar.close()
        log.info(f"    Closed final tar {tar_out_idx} ({current_tar_count} imgs)")

    log.info(f"WebDataset done: {len(all_records):,} images, {total_errors:,} errors, {tar_out_idx+1} tars")

    _write_json_shards(all_records, output_dir, prefix)
    return len(all_records)


# ─── JSON Shards ────────────────────────────────────────────────────────


def _write_json_shards(records, output_dir, prefix):
    """Write records to numbered JSON shard files."""
    os.makedirs(output_dir, exist_ok=True)
    shard_idx = 0
    for i in range(0, len(records), SHARD_SIZE):
        chunk = records[i : i + SHARD_SIZE]
        path = os.path.join(output_dir, f"{prefix}_{shard_idx:04d}.json")
        with open(path, "w") as f:
            json.dump(chunk, f)
        shard_idx += 1
    log.info(f"Wrote {shard_idx} JSON shards ({len(records):,} records)")


# ─── HDFS Upload ────────────────────────────────────────────────────────


def migrate_tars_to_hdfs(dataset_key):
    """Move tar archives to HDFS and delete BN copies to free space."""
    import shutil

    cfg = DATASETS[dataset_key]
    prefix = cfg["prefix"]
    output_dir = os.path.join(BN_BASE, "output", dataset_key)

    tar_files = sorted(Path(output_dir).glob(f"{prefix}_images_*.tar"))
    if not tar_files:
        log.info(f"No tar files to migrate for {dataset_key}")
        return

    hdfs_tar_dir = os.path.join(HDFS_TARGET, f"{prefix}_tars")
    os.makedirs(hdfs_tar_dir, exist_ok=True)
    total_size = 0

    log.info(f"Migrating {len(tar_files)} tars → {hdfs_tar_dir}/ (then delete BN copies)")

    for i, tf in enumerate(tar_files):
        dest = os.path.join(hdfs_tar_dir, tf.name)
        try:
            sz = tf.stat().st_size
            # Stream via cat to avoid FUSE cp issues with large files
            with open(str(tf), "rb") as src, open(dest, "wb") as dst:
                while True:
                    chunk = src.read(64 * 1024 * 1024)  # 64MB chunks
                    if not chunk:
                        break
                    dst.write(chunk)
            # Verify size
            dest_sz = os.path.getsize(dest)
            if dest_sz == sz:
                tf.unlink()  # Delete BN copy
                total_size += sz
            else:
                log.warning(f"  Size mismatch {tf.name}: {sz} vs {dest_sz}, keeping BN copy")

            if (i + 1) % 100 == 0 or i == len(tar_files) - 1:
                log.info(f"  Migrated [{i+1}/{len(tar_files)}] tars, freed {total_size/1e9:.1f} GB")
        except Exception as e:
            log.error(f"  FAIL {tf.name}: {e}")

    # Also delete JSON copies from BN (already on HDFS)
    for jf in Path(output_dir).glob(f"{prefix}_*.json"):
        jf.unlink()

    # Remove output dir if empty
    try:
        remaining = list(Path(output_dir).iterdir())
        if not remaining:
            os.rmdir(output_dir)
            log.info(f"  Removed empty dir {output_dir}")
    except Exception:
        pass

    log.info(f"Tar migration done: {total_size/1e12:.2f} TB freed from BN")


def upload_to_hdfs(dataset_key):
    """Upload JSON shards to HDFS via FUSE mount (no hdfs CLI needed)."""
    import shutil

    cfg = DATASETS[dataset_key]
    prefix = cfg["prefix"]
    output_dir = os.path.join(BN_BASE, "output", dataset_key)

    json_files = sorted(Path(output_dir).glob(f"{prefix}_*.json"))
    if not json_files:
        log.error(f"No JSON files in {output_dir}")
        return

    os.makedirs(HDFS_TARGET, exist_ok=True)
    log.info(f"Uploading {len(json_files)} JSON shards → {HDFS_TARGET}/ (FUSE)")

    failed = 0
    for jf in json_files:
        dest = os.path.join(HDFS_TARGET, jf.name)
        try:
            shutil.copy2(str(jf), dest)
            log.info(f"  OK {jf.name}")
        except Exception as e:
            log.error(f"  FAIL {jf.name}: {e}")
            failed += 1

    log.info(f"Upload done: {len(json_files) - failed}/{len(json_files)} succeeded")


# ─── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Process PT data for LightMGT")
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--workers", type=int, default=50,
                        help="Parallel workers for parquet processing (default 50)")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        cfg = DATASETS[ds]
        log.info(f"\n{'='*60}")
        log.info(f"Dataset: {ds} ({cfg['hf_id']}, type={cfg['type']})")
        log.info(f"{'='*60}")

        try:
            if args.upload_only:
                upload_to_hdfs(ds)
                continue

            if not args.process_only:
                download_dataset(ds)

            if args.download_only:
                continue

            if cfg["type"] == "parquet":
                count = process_parquet_dataset(ds, num_workers=args.workers)
            elif cfg["type"] == "webdataset":
                count = process_webdataset(ds, num_workers=args.workers)
            else:
                log.error(f"Unknown type: {cfg['type']}")
                continue

            log.info(f"Processed {count:,} records for {ds}")

            # Auto-upload JSONs + tars to HDFS, then clean BN
            if not args.process_only:
                upload_to_hdfs(ds)
                migrate_tars_to_hdfs(ds)

            # Auto-cleanup downloads to free disk space
            dl_dir = os.path.join(BN_BASE, "downloads", ds)
            if os.path.exists(dl_dir):
                import shutil
                dl_size = sum(f.stat().st_size for f in Path(dl_dir).rglob("*") if f.is_file()) / 1e12
                shutil.rmtree(dl_dir)
                log.info(f"Cleaned downloads for {ds} ({dl_size:.2f} TB freed)")

        except Exception as e:
            log.error(f"FAILED {ds}: {e}")
            import traceback
            traceback.print_exc()

    log.info("\n=== All done! ===")


if __name__ == "__main__":
    main()
