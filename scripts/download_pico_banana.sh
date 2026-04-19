#!/bin/bash
# Download pico-banana-400k SFT edited images from Apple CDN
# Uses xargs parallel download with N workers
#
# Usage: bash scripts/download_pico_banana.sh [num_workers]
# Default: 64 workers (conservative for CDN rate limits)

set -e

WORKERS=${1:-64}
BASE="/mnt/bn/search-auto-eval-v2/zhouwei/eval_data/pico-banana-400k"
IMG_DIR="$BASE/edited_images"
MANIFEST="$BASE/sft_manifest.txt"

mkdir -p "$IMG_DIR"

echo "Downloading pico-banana SFT images..."
echo "  Manifest: $MANIFEST ($(wc -l < "$MANIFEST") URLs)"
echo "  Output: $IMG_DIR"
echo "  Workers: $WORKERS"
echo ""

# Each line in manifest is a URL like:
# https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/sft_edited/xxxxx.png
# Download to $IMG_DIR/xxxxx.png

# Skip already downloaded
TOTAL=$(wc -l < "$MANIFEST")
EXISTING=$(ls "$IMG_DIR" 2>/dev/null | wc -l)
echo "  Already downloaded: $EXISTING / $TOTAL"

if [ "$EXISTING" -ge "$TOTAL" ]; then
    echo "  All images already downloaded!"
    exit 0
fi

# Use xargs for parallel download, skip existing
export IMG_DIR
cat "$MANIFEST" | xargs -P "$WORKERS" -I {} bash -c '
    url="{}"
    fname=$(basename "$url")
    out="$IMG_DIR/$fname"
    if [ ! -f "$out" ]; then
        curl -sL -o "$out" "$url" || echo "FAIL: $url"
    fi
'

FINAL=$(ls "$IMG_DIR" | wc -l)
echo ""
echo "Done! Downloaded $FINAL / $TOTAL images"
