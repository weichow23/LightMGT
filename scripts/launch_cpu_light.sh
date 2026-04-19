#!/bin/bash
# cpu_light: Pick up failed datasets (park + multigen)
set -uo pipefail

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HOME=/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache
export HF_TOKEN="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"

cd /mnt/bn/search-auto-eval-v2/zhouwei/LightMGT
LOGDIR=/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/logs
mkdir -p $LOGDIR

echo "=== cpu_light restart ($(date)) ==="

for DS in park multigen; do
    echo "--- $DS START $(date) ---"
    python3 scripts/process_pt_data.py --dataset $DS --workers 150 \
        >> "$LOGDIR/${DS}.log" 2>&1 || echo "--- $DS FAILED (exit=$?) ---"
    echo "--- $DS DONE $(date) ---"
done

echo "=== cpu_light ALL DONE ($(date)) ==="
