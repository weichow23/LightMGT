#!/bin/bash
set -e
export PATH=$PATH:/home/tiger/.local/bin
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HOME=/mnt/bn/search-auto-eval/zhouwei/hf_cache
export HF_TOKEN="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"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export PYTHONPATH=./

cd /mnt/bn/search-auto-eval-v2/zhouwei/LightMGT

# Find a free port
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Starting 8-GPU dry run on port $MASTER_PORT..."
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT \
    scripts/train_phase1.py \
    --resolution 256 \
    --batch_size 4 \
    --max_steps 5 \
    --log_steps 2 \
    --no_wandb \
    --warmup_steps 1 \
    --num_workers 0 \
    --gradient_accumulation_steps 1 \
    --multi_scale \
    --output_dir /tmp/lightmgt_test8
