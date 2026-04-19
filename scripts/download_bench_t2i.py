"""Download and package echo4o_t2i, echo4o_surreal, opengpt4o_gen into JSON + HDFS tar.

Downloads from HuggingFace, maps images to existing JSON UIDs by order,
packages into standard tar format for TarImageReader.

Usage:
    python scripts/download_bench_t2i.py --dataset echo4o_t2i
    python scripts/download_bench_t2i.py --dataset echo4o_surreal
    python scripts/download_bench_t2i.py --dataset opengpt4o_gen
    python scripts/download_bench_t2i.py --dataset all
"""

import argparse
import glob
import io
import json
import os
import subprocess
import tarfile
import tempfile

# ALL writes to /mnt/bn/ only (NEVER /tmp)
WORK_DIR = "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/bench_download"
HDFS_OUT = "/mnt/hdfs/weichow/maskedit/t2i"
JSON_DIR = "/mnt/hdfs/weichow/maskedit/t2i"  # existing JSONs

DATASETS = {
    "echo4o_t2i": {
        "hf_repo": "Yejy53/Echo-4o-Image",
        "hf_jsonl": "Instruction-Following-Image/Instruction-Following-Image.jsonl",
        "hf_tar_dir": "Instruction-Following-Image/images",
        "caption_field": "instruction",
        "img_ext": "jpg",
        "prefix": "echo4o_t2i",
    },
    "echo4o_surreal": {
        "hf_repo": "Yejy53/Echo-4o-Image",
        "hf_jsonl": "Surrel-Fantasy-Image/conflict.jsonl",
        "hf_tar_dir": "Surrel-Fantasy-Image/images",
        "caption_field": "instruction",
        "img_ext": "jpg",
        "prefix": "echo4o_surreal",
    },
    "opengpt4o_gen": {
        "hf_repo": "WINDop/OpenGPT-4o-Image",
        "hf_jsonl": "gen.json",
        "hf_tar_prefix": "gen.tar.gz",
        "caption_field": "input_prompt",
        "img_ext": "png",
        "prefix": "opengpt4o_gen",
    },
}


def load_our_jsons(prefix: str) -> list:
    """Load all JSON shards for a dataset prefix, preserving order."""
    files = sorted(glob.glob(os.path.join(JSON_DIR, f"{prefix}_*.json")))
    samples = []
    for f in files:
        with open(f) as fh:
            samples.extend(json.load(fh))
    print(f"  Loaded {len(samples)} samples from {len(files)} JSON shards")
    return samples


def download_hf_file(repo: str, path: str) -> str:
    """Download a file from HF hub, return local path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset")


def list_hf_files(repo: str, prefix: str) -> list:
    """List files in HF repo matching prefix."""
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(repo, repo_type="dataset")
    return sorted([f for f in all_files if f.startswith(prefix) and f.endswith(".tar.gz")])


def process_echo4o(dataset_name: str):
    """Process echo4o_t2i or echo4o_surreal."""
    cfg = DATASETS[dataset_name]
    prefix = cfg["prefix"]
    img_ext = cfg["img_ext"]
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")

    # Load our JSONs
    our_samples = load_our_jsons(prefix)
    uid_map = {i: s["uid"] for i, s in enumerate(our_samples)}
    total = len(our_samples)

    # Download HF JSONL to verify order
    print(f"  Downloading HF JSONL: {cfg['hf_jsonl']}...")
    jpath = download_hf_file(cfg["hf_repo"], cfg["hf_jsonl"])
    with open(jpath) as f:
        hf_entries = [json.loads(l) for l in f if l.strip()]
    print(f"  HF JSONL: {len(hf_entries)} entries")
    assert len(hf_entries) == total, f"Count mismatch: HF={len(hf_entries)} vs ours={total}"

    # Verify order by caption
    mismatches = 0
    for i in range(min(50, total)):
        hf_cap = hf_entries[i].get(cfg["caption_field"], "")
        our_cap = our_samples[i]["caption"]
        if hf_cap != our_cap:
            mismatches += 1
    print(f"  Caption order verification (first 50): {50 - mismatches}/50 match")
    assert mismatches < 5, f"Too many caption mismatches: {mismatches}/50"

    # List tar.gz files on HF
    tar_files = list_hf_files(cfg["hf_repo"], cfg["hf_tar_dir"])
    print(f"  HF tar.gz files: {len(tar_files)}")

    # Create output tar
    os.makedirs(WORK_DIR, exist_ok=True)
    out_tar_path = os.path.join(WORK_DIR, f"{prefix}_images.tar")
    print(f"  Writing output tar: {out_tar_path}")

    img_idx = 0
    with tarfile.open(out_tar_path, "w") as out_tf:
        for tf_name in tar_files:
            print(f"  Downloading {tf_name}...")
            local_path = download_hf_file(cfg["hf_repo"], tf_name)

            with tarfile.open(local_path, "r:gz") as in_tf:
                members = sorted(in_tf.getmembers(), key=lambda m: m.name)
                for m in members:
                    if not m.isfile():
                        continue
                    if img_idx >= total:
                        break

                    uid = uid_map[img_idx]
                    member_name = f"images/{prefix}/{uid}.{img_ext}"

                    # Read raw bytes
                    raw = in_tf.extractfile(m).read()

                    # Add to output tar with correct name
                    info = tarfile.TarInfo(name=member_name)
                    info.size = len(raw)
                    out_tf.addfile(info, io.BytesIO(raw))

                    img_idx += 1
                    if img_idx % 5000 == 0:
                        print(f"    Processed {img_idx}/{total}")

    print(f"  Total images packed: {img_idx}/{total}")

    # Upload to HDFS via FUSE mount (no hdfs CLI needed)
    hdfs_path = os.path.join(HDFS_OUT, f"{prefix}_images.tar")
    print(f"  Moving to HDFS via FUSE: {hdfs_path}")
    print(f"  Size: {os.path.getsize(out_tar_path) / 1e9:.1f} GB (this may take a while)...")
    subprocess.run(["mv", out_tar_path, hdfs_path], check=True)

    print(f"  Verifying: {os.path.exists(hdfs_path)}")
    if os.path.exists(hdfs_path):
        size_gb = os.path.getsize(hdfs_path) / 1e9
        print(f"  HDFS tar: {hdfs_path} ({size_gb:.1f} GB)")

    print(f"  Done: {dataset_name}")


def process_opengpt4o_gen():
    """Process opengpt4o_gen (multi-part tar.gz)."""
    cfg = DATASETS["opengpt4o_gen"]
    prefix = cfg["prefix"]
    img_ext = cfg["img_ext"]
    print(f"\n{'='*60}")
    print(f"Processing opengpt4o_gen")
    print(f"{'='*60}")

    # Load our JSONs
    our_samples = load_our_jsons(prefix)
    uid_map = {i: s["uid"] for i, s in enumerate(our_samples)}
    total = len(our_samples)

    # Download HF JSON (plain JSON array, all entries are generation)
    print(f"  Downloading HF JSON: {cfg['hf_jsonl']}...")
    jpath = download_hf_file(cfg["hf_repo"], cfg["hf_jsonl"])
    with open(jpath) as f:
        hf_gen = json.load(f)
    print(f"  HF gen entries: {len(hf_gen)}")
    assert len(hf_gen) == total, f"Count mismatch: HF={len(hf_gen)} vs ours={total}"

    # Verify caption order
    mismatches = 0
    for i in range(min(50, total)):
        hf_cap = hf_gen[i].get(cfg["caption_field"], "")
        our_cap = our_samples[i]["caption"]
        if hf_cap != our_cap:
            mismatches += 1
            if i < 5:
                print(f"    MISMATCH at {i}: HF={hf_cap[:50]} vs ours={our_cap[:50]}")
    print(f"  Caption order verification (first 50): {50 - mismatches}/50 match")

    # List gen tar.gz parts
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(cfg["hf_repo"], repo_type="dataset")
    gen_parts = sorted([f for f in all_files if f.startswith("gen.tar.gz.")])
    print(f"  HF gen tar parts: {len(gen_parts)}")

    # Download all parts and concat
    os.makedirs(WORK_DIR, exist_ok=True)
    concat_path = os.path.join(WORK_DIR, "gen_concat.tar.gz")
    print(f"  Downloading and concatenating {len(gen_parts)} parts...")
    with open(concat_path, "wb") as out_f:
        for part in gen_parts:
            print(f"    Downloading {part}...")
            local = download_hf_file(cfg["hf_repo"], part)
            with open(local, "rb") as in_f:
                while True:
                    chunk = in_f.read(1024 * 1024 * 64)  # 64MB chunks
                    if not chunk:
                        break
                    out_f.write(chunk)

    # Extract and repack
    out_tar_path = os.path.join(WORK_DIR, f"{prefix}_images.tar")
    print(f"  Extracting and repacking to {out_tar_path}...")

    img_idx = 0
    with tarfile.open(concat_path, "r:gz") as in_tf, \
         tarfile.open(out_tar_path, "w") as out_tf:
        for m in in_tf:
            if not m.isfile():
                continue
            if img_idx >= total:
                break

            uid = uid_map[img_idx]
            member_name = f"images/{prefix}/{uid}.{img_ext}"

            raw = in_tf.extractfile(m).read()
            info = tarfile.TarInfo(name=member_name)
            info.size = len(raw)
            out_tf.addfile(info, io.BytesIO(raw))

            img_idx += 1
            if img_idx % 5000 == 0:
                print(f"    Processed {img_idx}/{total}")

    print(f"  Total images packed: {img_idx}/{total}")

    # Upload to HDFS via FUSE mount
    hdfs_path = os.path.join(HDFS_OUT, f"{prefix}_images.tar")
    print(f"  Moving to HDFS via FUSE: {hdfs_path}")
    print(f"  Size: {os.path.getsize(out_tar_path) / 1e9:.1f} GB...")
    subprocess.run(["mv", out_tar_path, hdfs_path], check=True)

    print(f"  Verifying: {os.path.exists(hdfs_path)}")
    if os.path.exists(hdfs_path):
        size_gb = os.path.getsize(hdfs_path) / 1e9
        print(f"  HDFS tar: {hdfs_path} ({size_gb:.1f} GB)")

    # Cleanup
    os.remove(concat_path)
    print(f"  Done: opengpt4o_gen")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["echo4o_t2i", "echo4o_surreal", "opengpt4o_gen", "all"])
    args = parser.parse_args()

    targets = ["echo4o_t2i", "echo4o_surreal", "opengpt4o_gen"] if args.dataset == "all" else [args.dataset]

    for name in targets:
        if name == "opengpt4o_gen":
            process_opengpt4o_gen()
        else:
            process_echo4o(name)

    print("\n" + "=" * 60)
    print("ALL DONE. Tar files on HDFS:")
    for name in targets:
        path = os.path.join(HDFS_OUT, f"{DATASETS[name]['prefix']}_images.tar")
        exists = os.path.exists(path)
        size = os.path.getsize(path) / 1e9 if exists else 0
        print(f"  {path}: {'OK' if exists else 'MISSING'} ({size:.1f} GB)")
    print("Now build tar indexes with your preferred method.")


if __name__ == "__main__":
    main()
