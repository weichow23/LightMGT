#!/bin/bash
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HOME=/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache
export HF_TOKEN="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"
cd /mnt/bn/search-auto-eval-v2/zhouwei/LightMGT

# Kill stuck midjourney download
pkill -f "midjourney_v6" 2>/dev/null
sleep 2

# Process midjourney with what's downloaded (108/110 = 98%)
echo "--- midjourney_v6 process-only START $(date) ---"
python3 scripts/process_pt_data.py --dataset midjourney_v6 --process-only --workers 150 \
    >> pt_data/logs/midjourney_v6.log 2>&1
echo "--- midjourney_v6 DONE $(date) ---"

# Then dalle3
echo "--- dalle3_synthetic START $(date) ---"
python3 scripts/process_pt_data.py --dataset dalle3_synthetic --workers 150 \
    >> pt_data/logs/dalle3_synthetic.log 2>&1
echo "--- dalle3_synthetic DONE $(date) ---"
