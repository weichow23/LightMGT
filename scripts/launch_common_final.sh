#!/bin/bash
# Commoncatalog full reprocess: 4-machine parallel
# IMPORTANT: First delete old incomplete JSONs from HDFS, then reprocess everything
#
# Step 1: Clean old commoncatalog JSONs from HDFS (run once on any machine):
#   rm /mnt/hdfs/weichow/maskedit/t2i-pt/commoncatalog_*.json
#   rm /mnt/hdfs/weichow/maskedit/t2i-pt/commoncatalog_h2*.json
#
# Step 2: Launch on 4 machines (each with different --shard):
#   cpu_light:  nohup bash launch_common_final.sh 0 > logs/ccfinal_s0.log 2>&1 &
#   cpu_light2: nohup bash launch_common_final.sh 1 > logs/ccfinal_s1.log 2>&1 &
#   cpu_world:  nohup bash launch_common_final.sh 2 > logs/ccfinal_s2.log 2>&1 &
#   cpu_artic:  nohup bash launch_common_final.sh 3 > logs/ccfinal_s3.log 2>&1 &

SHARD=${1:-0}
EXTRA_ARGS="${@:2}"  # extra args like --batch-start 14 --batch-end 20

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HOME=/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache
export HF_TOKEN="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"

cd /mnt/bn/search-auto-eval-v2/zhouwei/LightMGT
echo "=== Commoncatalog final shard $SHARD START $(date) ==="
python3 scripts/process_common_final.py --shard $SHARD --total-shards 4 --workers 150 $EXTRA_ARGS
echo "=== Commoncatalog final shard $SHARD DONE $(date) ==="
