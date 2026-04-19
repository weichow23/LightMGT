#!/bin/bash
# Run all data processing on cpu_light (234 CPUs)
# Usage: bash scripts/data_pipeline.sh
set -e

export PATH=$PATH:/home/tiger/.local/bin
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HOME=/mnt/bn/search-auto-eval/zhouwei/hf_cache
export HF_TOKEN="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"

BN_BASE="/mnt/bn/search-auto-eval-v2/zhouwei/eval_data"
HDFS_BASE="/mnt/hdfs/weichow/maskedit"

echo "========================================"
echo "LightMGT Data Pipeline — $(date)"
echo "CPUs: $(nproc)"
echo "========================================"

# ── 1. MICo-150K: Download (not on this machine yet) ──
echo ""
echo "=== [1/3] MICo-150K: Downloading ==="
if [ ! -d "$BN_BASE/mico-150k" ] || [ $(ls "$BN_BASE/mico-150k/" 2>/dev/null | wc -l) -lt 5 ]; then
    pip install -q huggingface_hub 2>/dev/null
    python3 -c "
from huggingface_hub import snapshot_download
print('Downloading MICo-150K...')
path = snapshot_download('kr-cen/MICo-150K', repo_type='dataset',
                         local_dir='$BN_BASE/mico-150k')
print(f'MICo-150K: {path}')
"
    echo "MICo-150K download complete"
else
    echo "MICo-150K already exists at $BN_BASE/mico-150k"
fi

# ── 2. pico-banana: Download images (257K images from Apple CDN) ──
echo ""
echo "=== [2/3] pico-banana: Downloading 257K images ==="
PICO_DIR="$BN_BASE/pico-banana-400k"
IMG_DIR="$PICO_DIR/edited_images"
mkdir -p "$IMG_DIR"

TOTAL=$(wc -l < "$PICO_DIR/sft_manifest.txt")
EXISTING=$(ls "$IMG_DIR" 2>/dev/null | wc -l)
echo "  Already: $EXISTING / $TOTAL"

if [ "$EXISTING" -lt "$TOTAL" ]; then
    cat "$PICO_DIR/sft_manifest.txt" | xargs -P 200 -I {} bash -c '
        url="{}"
        fname=$(basename "$url")
        out="'"$IMG_DIR"'/$fname"
        if [ ! -f "$out" ]; then
            curl -sL --max-time 30 -o "$out" "$url" 2>/dev/null || true
        fi
    '
    FINAL=$(ls "$IMG_DIR" | wc -l)
    echo "  Downloaded: $FINAL / $TOTAL"
else
    echo "  All images already downloaded"
fi

# ── 3. TextAtlas5M: Move to HDFS ──
echo ""
echo "=== [3/3] TextAtlas5M: tar+move to HDFS ==="
TA_SRC="$BN_BASE/textatlas5m"
TA_DST="$HDFS_BASE/raw_data"

if [ -d "$TA_SRC" ] && [ ! -d "$TA_DST/textatlas5m" ]; then
    mkdir -p "$TA_DST"
    echo "  Packing tar..."
    cd "$(dirname "$TA_SRC")"
    tar cf "${TA_SRC}.tar" "$(basename "$TA_SRC")"
    echo "  Copying to HDFS..."
    cp "${TA_SRC}.tar" "$TA_DST/"
    cd "$TA_DST"
    tar xf "$(basename "${TA_SRC}.tar")"
    rm "$(basename "${TA_SRC}.tar")"
    echo "  TextAtlas5M on HDFS ✓"
else
    echo "  Already on HDFS or source not found"
fi

echo ""
echo "========================================"
echo "Data pipeline complete — $(date)"
echo "========================================"
