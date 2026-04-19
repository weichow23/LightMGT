#!/bin/bash
# Process park with only already-downloaded parquets (half dataset, ~3.8K of 7.5K)
# With resize to MAX_DIM=1536 (fits 1024 bucket)
set -uo pipefail

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HOME=/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache
export HF_TOKEN="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"

cd /mnt/bn/search-auto-eval-v2/zhouwei/LightMGT
LOGDIR=/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/logs
mkdir -p $LOGDIR

echo "=== park (half, resize≤1536) START $(date) ==="

# Process-only: skip download, use what's already there
python3 scripts/process_pt_data.py --dataset park --process-only --workers 150 \
    >> "$LOGDIR/park.log" 2>&1

echo "=== park DONE $(date) ==="

# Then multigen
echo "--- multigen START $(date) ---"
python3 scripts/process_pt_data.py --dataset multigen --workers 150 \
    >> "$LOGDIR/multigen.log" 2>&1
echo "--- multigen DONE $(date) ---"
