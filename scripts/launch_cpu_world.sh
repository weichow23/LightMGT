#!/bin/bash
# cpu_world: Sequential processing (one at a time to avoid quota)
set -uo pipefail

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HOME=/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache
export HF_TOKEN="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"

cd /mnt/bn/search-auto-eval-v2/zhouwei/LightMGT
LOGDIR=/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/logs
mkdir -p $LOGDIR

echo "=== cpu_world restart ($(date)) ==="

# park first (biggest, 3.5T+), then remaining
for DS in park multigen text2image_2m laion_600k commoncatalog; do
    echo "--- $DS START $(date) ---"
    python3 scripts/process_pt_data.py --dataset $DS --workers 150 \
        >> "$LOGDIR/${DS}.log" 2>&1 || echo "--- $DS FAILED (exit=$?) ---"
    echo "--- $DS DONE $(date) ---"
done

echo "=== cpu_world ALL DONE ($(date)) ==="
