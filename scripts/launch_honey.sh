#!/bin/bash
# Honey-Data-15M VQA dataset processing
set -uo pipefail

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HOME=/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache
export HF_TOKEN="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"

cd /mnt/bn/search-auto-eval-v2/zhouwei/LightMGT
LOGDIR=/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/logs
mkdir -p $LOGDIR

echo "=== Honey-Data-15M START $(date) ==="
python3 scripts/process_honey_data.py --workers 150 \
    >> "$LOGDIR/honey_15m.log" 2>&1
echo "=== Honey-Data-15M DONE $(date) ==="
