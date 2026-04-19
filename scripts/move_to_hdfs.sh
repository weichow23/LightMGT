#!/bin/bash
# Download to /mnt/bn/ then tar+move to /mnt/hdfs/ for each dataset
# Usage: bash scripts/move_to_hdfs.sh [dataset_name|all]
#
# Strategy: download → bn (fast local SSD) → tar → cp to hdfs (single write, no lock issues)

set -e

BN_BASE="/mnt/bn/search-auto-eval-v2/zhouwei/eval_data"
HDFS_BASE="/mnt/hdfs/weichow/maskedit"

move_dataset() {
    local name=$1
    local bn_path=$2
    local hdfs_dest=$3

    if [ ! -d "$bn_path" ]; then
        echo "SKIP: $name — $bn_path not found"
        return 1
    fi

    echo "=== Moving $name ==="
    echo "  From: $bn_path"
    echo "  To:   $hdfs_dest"

    # Create target dir
    mkdir -p "$hdfs_dest"

    # Tar pack on bn (fast)
    local tar_path="${bn_path}.tar"
    echo "  Packing to $tar_path ..."
    cd "$(dirname "$bn_path")"
    tar cf "$tar_path" "$(basename "$bn_path")"
    echo "  Tar size: $(du -sh "$tar_path" | cut -f1)"

    # Move tar to hdfs (single sequential write)
    echo "  Copying tar to HDFS..."
    cp "$tar_path" "${hdfs_dest}/$(basename "$tar_path")"

    # Extract on hdfs
    echo "  Extracting on HDFS..."
    cd "$hdfs_dest"
    tar xf "$(basename "$tar_path")"
    rm "$(basename "$tar_path")"

    echo "  Done: $name ✓"
    echo ""
}

case "${1:-all}" in
    textatlas)
        move_dataset "TextAtlas5M" "$BN_BASE/textatlas5m" "$HDFS_BASE/raw_data"
        ;;
    mico)
        move_dataset "MICo-150K" "$BN_BASE/mico-150k" "$HDFS_BASE/raw_data"
        ;;
    pico)
        move_dataset "pico-banana-400k" "$BN_BASE/pico-banana-400k" "$HDFS_BASE/raw_data"
        ;;
    dreambench)
        move_dataset "DreamBench++" "$BN_BASE/dreambench_plus" "$HDFS_BASE/eval_data"
        ;;
    geditbench)
        # GEditBench already downloaded to hdfs directly
        echo "GEditBench already at /mnt/hdfs/weichow/maskedit/eval_data/geditbench"
        ;;
    all)
        echo "Moving all datasets from /mnt/bn/ to /mnt/hdfs/ ..."
        echo ""
        move_dataset "TextAtlas5M" "$BN_BASE/textatlas5m" "$HDFS_BASE/raw_data" || true
        move_dataset "MICo-150K" "$BN_BASE/mico-150k" "$HDFS_BASE/raw_data" || true
        move_dataset "pico-banana-400k" "$BN_BASE/pico-banana-400k" "$HDFS_BASE/raw_data" || true
        move_dataset "DreamBench++" "$BN_BASE/dreambench_plus" "$HDFS_BASE/eval_data" || true
        echo "=== ALL DONE ==="
        ;;
    *)
        echo "Usage: $0 [textatlas|mico|pico|dreambench|geditbench|all]"
        exit 1
        ;;
esac
