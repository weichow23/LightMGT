#!/bin/bash
# commoncatalog: download remaining parquets + process second half
# Runs AFTER half-1 processing is done (which frees BN space via auto-cleanup)
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HOME=/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache
export HF_TOKEN="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"
cd /mnt/bn/search-auto-eval-v2/zhouwei/LightMGT

echo "=== commoncatalog half-2 download+process START $(date) ==="
# This will resume download (gets remaining ~2971 parquets) then process them
python3 scripts/process_pt_data.py --dataset commoncatalog --workers 150 \
    >> pt_data/logs/commoncatalog_half2.log 2>&1
echo "=== commoncatalog half-2 DONE $(date) ==="
