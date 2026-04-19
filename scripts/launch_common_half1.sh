#!/bin/bash
# commoncatalog: process first half (2602 already downloaded parquets)
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HOME=/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache
export HF_TOKEN="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"
cd /mnt/bn/search-auto-eval-v2/zhouwei/LightMGT

echo "=== commoncatalog half-1 process-only START $(date) ==="
python3 scripts/process_pt_data.py --dataset commoncatalog --process-only --workers 150 \
    >> pt_data/logs/commoncatalog.log 2>&1
echo "=== commoncatalog half-1 DONE $(date) ==="
