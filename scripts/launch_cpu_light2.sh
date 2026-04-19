#!/bin/bash
# cpu_light2: Sequential processing (one at a time to avoid quota)
set -uo pipefail

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HOME=/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache
export HF_TOKEN="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"

cd /mnt/bn/search-auto-eval-v2/zhouwei/LightMGT
LOGDIR=/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/logs
mkdir -p $LOGDIR

echo "=== cpu_light2 restart ($(date)) ==="

# journeydb_p2 first (3.1T, failed earlier from quota)
# Then smaller datasets that cpu_world was supposed to do
for DS in journeydb_p2 flux_journey midjourney_v6 dalle3_synthetic; do
    echo "--- $DS START $(date) ---"
    python3 scripts/process_pt_data.py --dataset $DS --workers 150 \
        >> "$LOGDIR/${DS}.log" 2>&1 || echo "--- $DS FAILED (exit=$?) ---"
    echo "--- $DS DONE $(date) ---"
done

echo "=== cpu_light2 ALL DONE ($(date)) ==="
