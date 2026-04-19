#!/usr/bin/env python3
"""Fix camera_good JSON: filter out entries where image path is not in tar index.

Root cause: JSON references image paths that don't exist in the tar archives,
causing TarImageReader to retry on every miss during training.

Fix: Load tar index → filter each JSON shard → overwrite with cleaned version.
Also applies to ALL datasets in /mnt/hdfs/weichow/maskedit/t2i/ that use tars.
"""

import glob
import hashlib
import json
import os
import pickle
import sys

TAR_INDEX_DIR = "/mnt/bn/search-auto-eval/zhouwei/maskedit_cache/tar_index"
HDFS_T2I = "/mnt/hdfs/weichow/maskedit/t2i"


def load_tar_index(pattern):
    """Load tar index from cache using md5 of pattern as key."""
    cache_key = hashlib.md5(pattern.encode()).hexdigest()
    cache_path = os.path.join(TAR_INDEX_DIR, f"{cache_key}.pkl")
    if not os.path.exists(cache_path):
        print(f"  WARNING: No index cache for pattern {pattern} (key={cache_key})")
        return None
    with open(cache_path, "rb") as f:
        idx = pickle.load(f)
    print(f"  Loaded index: {len(idx)} entries from {cache_path}")
    return idx


def clean_dataset(prefix, tar_pattern):
    """Clean a dataset's JSON shards by filtering against tar index."""
    print(f"\n{'='*60}")
    print(f"Cleaning: {prefix}")
    print(f"Tar pattern: {tar_pattern}")

    idx = load_tar_index(tar_pattern)
    if idx is None:
        print(f"  SKIP: no tar index found")
        return 0, 0

    json_files = sorted(glob.glob(os.path.join(HDFS_T2I, f"{prefix}_*.json")))
    if not json_files:
        print(f"  SKIP: no JSON files found for {prefix}")
        return 0, 0

    total_before = 0
    total_after = 0

    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ERROR reading {jf}: {e}")
            continue

        before = len(data)
        filtered = [e for e in data if e.get("image", e.get("img", "")) in idx]
        after = len(filtered)
        removed = before - after

        total_before += before
        total_after += after

        if removed > 0:
            with open(jf, "w") as f:
                json.dump(filtered, f)
            print(f"  {os.path.basename(jf)}: {before} -> {after} (removed {removed})")
        # Don't print if no change (too verbose)

    pct = (1 - total_after / max(total_before, 1)) * 100
    print(f"  TOTAL: {total_before} -> {total_after} ({pct:.1f}% removed)")
    return total_before, total_after


def main():
    # Datasets to clean (prefix -> tar pattern)
    # These are the datasets in /mnt/hdfs/weichow/maskedit/t2i/ that have tar archives
    datasets = {
        "camera_good": f"{HDFS_T2I}/camera_good_images_*.tar",
        "fine_t2i": f"{HDFS_T2I}/fine_t2i_images_*.tar",
        "art_sft_good": f"{HDFS_T2I}/art_sft_good_images_*.tar",
        "art_crawler_good": f"{HDFS_T2I}/art_crawler_good_images_*.tar",
        "design_good": f"{HDFS_T2I}/design_good_images_*.tar",
        "movie_good": f"{HDFS_T2I}/movie_good_images_*.tar",
        "photograph_good": f"{HDFS_T2I}/photograph_good_images_*.tar",
    }

    # Check which dataset to clean (or all)
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    grand_before = 0
    grand_after = 0

    for prefix, tar_pattern in datasets.items():
        if target != "all" and target != prefix:
            continue
        before, after = clean_dataset(prefix, tar_pattern)
        grand_before += before
        grand_after += after

    if grand_before > 0:
        pct = (1 - grand_after / grand_before) * 100
        print(f"\n{'='*60}")
        print(f"GRAND TOTAL: {grand_before} -> {grand_after} ({pct:.1f}% removed)")
    else:
        print("\nNo data processed.")


if __name__ == "__main__":
    main()
