#!/bin/bash
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HOME=/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache
export HF_TOKEN="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"
cd /mnt/bn/search-auto-eval-v2/zhouwei/LightMGT
python3 scripts/process_echo4o.py >> pt_data/logs/echo4o.log 2>&1
