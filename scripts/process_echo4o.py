#!/usr/bin/env python3
"""Process Echo-4o T2I/Surreal + OpenGPT-4o gen/editing into json+tar format.

Downloads tar.gz archives, extracts images, matches with JSONL/JSON metadata,
creates standard json shards + tar archives, uploads to HDFS.

Usage:
    python3 process_echo4o.py --workers 150
"""

import argparse
import gzip
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
HDFS_T2I = "/mnt/hdfs/weichow/maskedit/t2i"
HDFS_EDIT = "/mnt/hdfs/weichow/maskedit/edit"
HF_CACHE = "/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache"
HF_TOKEN = "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"
SHARD_SIZE = 10000


def download_datasets():
    """Download both Echo-4o-Image and OpenGPT-4o-Image."""
    from huggingface_hub import snapshot_download
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["HF_TOKEN"] = HF_TOKEN

    dl_dir = os.path.join(BN_BASE, "downloads", "echo4o")
    os.makedirs(dl_dir, exist_ok=True)

    log.info("Downloading Yejy53/Echo-4o-Image...")
    snapshot_download("Yejy53/Echo-4o-Image", repo_type="dataset",
                      local_dir=os.path.join(dl_dir, "echo4o"))

    log.info("Downloading WINDop/OpenGPT-4o-Image...")
    snapshot_download("WINDop/OpenGPT-4o-Image", repo_type="dataset",
                      local_dir=os.path.join(dl_dir, "opengpt4o"))

    return dl_dir


def extract_images_from_tars(tar_dir, extract_to):
    """Extract all tar.gz files in a directory to extract_to."""
    os.makedirs(extract_to, exist_ok=True)
    count = 0
    for tgz in sorted(Path(tar_dir).glob("*.tar.gz")):
        log.info(f"  Extracting {tgz.name}...")
        with tarfile.open(str(tgz), "r:gz") as tf:
            tf.extractall(extract_to)
            count += len(tf.getmembers())
    return count


def extract_split_targz(file_pattern, extract_to):
    """Extract split tar.gz files (e.g. gen.tar.gz.00, gen.tar.gz.01, ...)."""
    import subprocess
    os.makedirs(extract_to, exist_ok=True)

    parts = sorted(Path(file_pattern).parent.glob(Path(file_pattern).name + ".*"))
    if not parts:
        parts = sorted(Path(file_pattern).parent.glob(Path(file_pattern).name + "[0-9]*"))

    if not parts:
        log.warning(f"  No split files matching {file_pattern}")
        return 0

    log.info(f"  Concatenating {len(parts)} parts and extracting...")
    # Cat all parts and pipe to tar
    cat_cmd = " ".join([f'"{p}"' for p in parts])
    cmd = f"cat {cat_cmd} | tar xzf - -C {extract_to}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"  Extract failed: {result.stderr[:200]}")
        return 0

    count = sum(1 for _ in Path(extract_to).rglob("*") if _.is_file())
    return count


def get_image_dims(img_path):
    """Get (w, h) from an image file."""
    try:
        from PIL import Image
        img = Image.open(img_path)
        return img.size
    except Exception:
        return 256, 256


def create_tar_and_json(records, images_dir, output_dir, prefix):
    """Create tar archives + JSON shards from records."""
    os.makedirs(output_dir, exist_ok=True)

    # Create tar
    tar_path = os.path.join(output_dir, f"{prefix}_images_0.tar")
    log.info(f"  Creating tar: {tar_path}")
    written = 0
    with tarfile.open(tar_path, "w") as tf:
        for r in records:
            img_local = r.pop("_img_local", None)
            if img_local and os.path.exists(img_local):
                member_name = r["image"]
                tf.add(img_local, arcname=member_name)
                written += 1
            # For edit pairs, also add input images
            for inp in r.get("_input_locals", []):
                if os.path.exists(inp["path"]):
                    tf.add(inp["path"], arcname=inp["name"])

    # Clean internal keys
    for r in records:
        r.pop("_img_local", None)
        r.pop("_input_locals", None)

    log.info(f"  Tar: {written} images written")

    # JSON shards
    idx = 0
    for i in range(0, len(records), SHARD_SIZE):
        chunk = records[i:i + SHARD_SIZE]
        path = os.path.join(output_dir, f"{prefix}_{idx:04d}.json")
        with open(path, "w") as f:
            json.dump(chunk, f)
        idx += 1
    log.info(f"  Wrote {idx} JSON shards ({len(records)} records)")

    return written


def process_echo4o_t2i(dl_dir):
    """Process Echo-4o Instruction-Following-Image (T2I, 68K)."""
    log.info("\n" + "=" * 60)
    log.info("Processing Echo-4o T2I (Instruction-Following-Image)")

    base = os.path.join(dl_dir, "echo4o", "Instruction-Following-Image")
    jsonl_path = os.path.join(base, "Instruction-Following-Image.jsonl")
    images_tar_dir = os.path.join(base, "images")
    extract_dir = os.path.join(BN_BASE, "tmp_extract", "echo4o_t2i")

    # Extract images
    count = extract_images_from_tars(images_tar_dir, extract_dir)
    log.info(f"  Extracted {count} files")

    # Read metadata
    records = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line.strip())
            # output_image: /Echo-4o-Image/Instruction-Following-Image/images/00000.jpg
            img_name = os.path.basename(row["output_image"])
            img_local = os.path.join(extract_dir, img_name)

            if not os.path.exists(img_local):
                # Try finding in subdirs
                matches = list(Path(extract_dir).rglob(img_name))
                if matches:
                    img_local = str(matches[0])
                else:
                    continue

            w, h = get_image_dims(img_local)
            uid = hashlib.md5(f"echo4o_t2i_{img_name}".encode()).hexdigest()

            records.append({
                "uid": uid,
                "caption": row["instruction"],
                "image": f"images/echo4o_t2i/{uid}.jpg",
                "height": h,
                "width": w,
                "source": "echo4o_t2i",
                "type": row.get("type", ""),
                "_img_local": img_local,
            })

    log.info(f"  Matched {len(records)} records")

    output_dir = os.path.join(BN_BASE, "output", "echo4o_t2i")
    create_tar_and_json(records, extract_dir, output_dir, "echo4o_t2i")

    # Cleanup extract
    shutil.rmtree(extract_dir, ignore_errors=True)
    return len(records)


def process_echo4o_surreal(dl_dir):
    """Process Echo-4o Surreal-Fantasy-Image (38K)."""
    log.info("\n" + "=" * 60)
    log.info("Processing Echo-4o Surreal (Surrel-Fantasy-Image)")

    base = os.path.join(dl_dir, "echo4o", "Surrel-Fantasy-Image")
    jsonl_path = os.path.join(base, "conflict.jsonl")
    images_tar_dir = os.path.join(base, "images")
    extract_dir = os.path.join(BN_BASE, "tmp_extract", "echo4o_surreal")

    count = extract_images_from_tars(images_tar_dir, extract_dir)
    log.info(f"  Extracted {count} files")

    records = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line.strip())
            img_name = os.path.basename(row["output_image"])
            img_local = os.path.join(extract_dir, img_name)

            if not os.path.exists(img_local):
                matches = list(Path(extract_dir).rglob(img_name))
                if matches:
                    img_local = str(matches[0])
                else:
                    continue

            w, h = get_image_dims(img_local)
            uid = hashlib.md5(f"echo4o_surreal_{img_name}".encode()).hexdigest()

            records.append({
                "uid": uid,
                "caption": row["instruction"],
                "image": f"images/echo4o_surreal/{uid}.jpg",
                "height": h,
                "width": w,
                "source": "echo4o_surreal",
                "type": row.get("type", ""),
                "_img_local": img_local,
            })

    log.info(f"  Matched {len(records)} records")

    output_dir = os.path.join(BN_BASE, "output", "echo4o_surreal")
    create_tar_and_json(records, extract_dir, output_dir, "echo4o_surreal")
    shutil.rmtree(extract_dir, ignore_errors=True)
    return len(records)


def process_opengpt4o_gen(dl_dir):
    """Process OpenGPT-4o gen (T2I, 40K)."""
    log.info("\n" + "=" * 60)
    log.info("Processing OpenGPT-4o gen (T2I)")

    base = os.path.join(dl_dir, "opengpt4o")
    json_path = os.path.join(base, "gen.json")
    extract_dir = os.path.join(BN_BASE, "tmp_extract", "opengpt4o_gen")

    # Extract split tar.gz
    count = extract_split_targz(os.path.join(base, "gen.tar.gz"), extract_dir)
    log.info(f"  Extracted {count} files")

    with open(json_path) as f:
        data = json.load(f)

    records = []
    for row in data:
        img_rel = row["output_image"]  # e.g. "gen/0.png"
        img_local = os.path.join(extract_dir, img_rel)

        if not os.path.exists(img_local):
            matches = list(Path(extract_dir).rglob(os.path.basename(img_rel)))
            if matches:
                img_local = str(matches[0])
            else:
                continue

        w, h = get_image_dims(img_local)
        uid = hashlib.md5(f"opengpt4o_gen_{img_rel}".encode()).hexdigest()

        records.append({
            "uid": uid,
            "caption": row["input_prompt"],
            "image": f"images/opengpt4o_gen/{uid}.png",
            "height": h,
            "width": w,
            "source": "opengpt4o_gen",
            "meta_task": row.get("meta-task", ""),
            "_img_local": img_local,
        })

    log.info(f"  Matched {len(records)} records")

    output_dir = os.path.join(BN_BASE, "output", "opengpt4o_gen")
    create_tar_and_json(records, extract_dir, output_dir, "opengpt4o_gen")
    shutil.rmtree(extract_dir, ignore_errors=True)
    return len(records)


def process_opengpt4o_editing(dl_dir):
    """Process OpenGPT-4o editing (40K edit pairs)."""
    log.info("\n" + "=" * 60)
    log.info("Processing OpenGPT-4o editing")

    base = os.path.join(dl_dir, "opengpt4o")
    json_path = os.path.join(base, "editing.json")
    extract_dir = os.path.join(BN_BASE, "tmp_extract", "opengpt4o_edit")

    count = extract_split_targz(os.path.join(base, "editing.tar.gz"), extract_dir)
    log.info(f"  Extracted {count} files")

    with open(json_path) as f:
        data = json.load(f)

    records = []
    for row in data:
        out_rel = row["output_image"]  # "editing/output_0.png"
        out_local = os.path.join(extract_dir, out_rel)

        if not os.path.exists(out_local):
            matches = list(Path(extract_dir).rglob(os.path.basename(out_rel)))
            if matches:
                out_local = str(matches[0])
            else:
                continue

        w, h = get_image_dims(out_local)
        uid = hashlib.md5(f"opengpt4o_edit_{out_rel}".encode()).hexdigest()

        # Input images
        input_images = row.get("input_image", [])
        input_locals = []
        input_paths = []
        for inp in input_images:
            inp_local = os.path.join(extract_dir, inp)
            if not os.path.exists(inp_local):
                matches = list(Path(extract_dir).rglob(os.path.basename(inp)))
                if matches:
                    inp_local = str(matches[0])
                else:
                    continue
            inp_name = f"images/opengpt4o_edit/input_{hashlib.md5(inp.encode()).hexdigest()}.png"
            input_locals.append({"path": inp_local, "name": inp_name})
            input_paths.append(inp_name)

        records.append({
            "uid": uid,
            "caption": row["input_prompt"],
            "image": f"images/opengpt4o_edit/{uid}.png",
            "input_images": input_paths,
            "height": h,
            "width": w,
            "source": "opengpt4o_edit",
            "_img_local": out_local,
            "_input_locals": input_locals,
        })

    log.info(f"  Matched {len(records)} records")

    output_dir = os.path.join(BN_BASE, "output", "opengpt4o_edit")
    create_tar_and_json(records, extract_dir, output_dir, "opengpt4o_edit")
    shutil.rmtree(extract_dir, ignore_errors=True)
    return len(records)


def upload_all():
    """Upload JSON shards to HDFS: T2I→t2i/, Edit→edit/."""
    # T2I datasets → /mnt/hdfs/weichow/maskedit/t2i/
    t2i_datasets = ["echo4o_t2i", "echo4o_surreal", "opengpt4o_gen"]
    # Edit datasets → /mnt/hdfs/weichow/maskedit/edit/
    edit_datasets = ["opengpt4o_edit"]

    for ds_list, hdfs_dir in [(t2i_datasets, HDFS_T2I), (edit_datasets, HDFS_EDIT)]:
        os.makedirs(hdfs_dir, exist_ok=True)
        for ds in ds_list:
            out_dir = os.path.join(BN_BASE, "output", ds)
            jsons = sorted(Path(out_dir).glob("*.json"))
            for jf in jsons:
                dest = os.path.join(hdfs_dir, jf.name)
                try:
                    shutil.copy2(str(jf), dest)
                except Exception as e:
                    log.error(f"  FAIL {jf.name}: {e}")
            log.info(f"  Uploaded {len(jsons)} JSONs for {ds} → {hdfs_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    args = parser.parse_args()

    dl_dir = os.path.join(BN_BASE, "downloads", "echo4o")

    if not args.process_only:
        download_datasets()

    if args.download_only:
        return

    total = 0
    total += process_echo4o_t2i(dl_dir)
    total += process_echo4o_surreal(dl_dir)
    total += process_opengpt4o_gen(dl_dir)
    total += process_opengpt4o_editing(dl_dir)

    log.info(f"\nTotal: {total} records across 4 splits")

    upload_all()

    # Cleanup downloads
    shutil.rmtree(os.path.join(BN_BASE, "downloads", "echo4o"), ignore_errors=True)
    log.info("Cleaned downloads")
    log.info("=== All done! ===")


if __name__ == "__main__":
    main()
