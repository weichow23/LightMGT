#!/bin/bash
# LightMGT Phase 1 Training Launch Script (Multi-GPU)
# Uses torchrun + manual DDP (not accelerate)
# Data: ALL T2I + T2I-PT + Benchmark T2I
# Usage: bash train/train.sh [--resolution 256] [--max_steps 300000] [...]

export PATH=$PATH:/home/tiger/.local/bin
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HOME=/mnt/bn/search-auto-eval/zhouwei/hf_cache
export HF_TOKEN="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"

# Wandb: force entity to 3210103790
export WANDB_API_KEY="694a4f88e896dab0bc5000f60be7881c201d0e98"
export WANDB_PROJECT="LightMGT"
export WANDB_ENTITY="3210103790"

# Cache dirs: ALL on /mnt/bn/ to avoid filling root partition
export WANDB_DIR=/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/wandb
export TRITON_CACHE_DIR=/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/cache/triton

# NCCL config
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

# Multi-node support (defaults to single node)
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-${ARNOLD_ID:-0}}
MASTER_ADDR=${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}
# Auto-find free port if default is in use
if [ -z "$MASTER_PORT" ]; then
    MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
fi

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

echo "========================================"
echo "LightMGT Phase 1 Training (torchrun DDP)"
echo "Data: T2I + T2I-PT + Benchmark T2I"
echo "Node ${NODE_RANK}/${NUM_NODES} | GPUs: ${NUM_GPUS}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "========================================"

# Install wandb if needed
pip install wandb -q 2>/dev/null

# --- Resume from latest checkpoint ---
CKPT_DIR="/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/checkpoints"
if [ -n "$RESUME_CKPT" ]; then
    # Explicit checkpoint path
    LATEST_CKPT="$RESUME_CKPT"
else
    LATEST_CKPT=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
if [ -n "$LATEST_CKPT" ]; then
    RESUME_ARG="--resume_from ${LATEST_CKPT}/training_state.pt"
    echo "Resuming from: ${LATEST_CKPT}"
    if [ "${RESUME_WEIGHTS_ONLY:-0}" = "1" ]; then
        RESUME_ARG="${RESUME_ARG} --resume_weights_only"
        echo "  -> Weights only (optimizer/scheduler/step reset for resolution change)"
    fi
else
    RESUME_ARG=""
    echo "No checkpoint found, training from scratch"
fi

PYTHONPATH='./' torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=${NUM_NODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/train_phase1.py \
    --data_dir \
        /mnt/hdfs/weichow/maskedit/t2i \
        /mnt/hdfs/weichow/maskedit/t2i-pt \
    --tar_pattern \
        "/mnt/hdfs/weichow/maskedit/t2i/camera_good_images_*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i/fine_t2i_images_*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i/art_sft_good_images_*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i/art_crawler_good_images_*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i/design_good_images_*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i/movie_good_images_*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i/photograph_good_images_*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i/textatlas5m_images.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i-pt/journeydb_p1_tars/*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i-pt/journeydb_p2_tars/*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i-pt/laion_512_tars/*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i-pt/laion_1024_tars/*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i-pt/cc12m_tars/*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i-pt/multigen_20m_tars/*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i-pt/text2image_2m_tars/*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i-pt/dalle3_synth_tars/*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i-pt/flux_journey_tars/*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i-pt/commoncatalog_tars/cc_s0_b*_*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i-pt/commoncatalog_tars/cc_s1_b*_*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i-pt/commoncatalog_tars/cc_s2_b*_*.tar" \
        "/mnt/hdfs/weichow/maskedit/t2i-pt/commoncatalog_tars/cc_s3_b*_*.tar" \
        "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/output/echo4o_t2i/*.tar" \
        "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/output/echo4o_surreal/*.tar" \
        "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/output/opengpt4o_gen/*.tar" \
        "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/output/midjourney_v6/*.tar" \
        "/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data/output/park/meissonic_park_images_*.tar" \
    --resolution ${RESOLUTION:-128} \
    --multi_scale \
    --batch_size ${BATCH_SIZE:-32} \
    --gradient_accumulation_steps ${GRAD_ACCUM:-4} \
    --learning_rate 1.5e-4 \
    --warmup_steps 1000 \
    --max_steps ${MAX_STEPS:-300000} \
    --weight_decay 0.045 \
    --beta2 0.96 \
    --grad_clip 1.0 \
    --cfg_dropout 0.1 \
    --label_smoothing 0.1 \
    --output_dir ${CKPT_DIR} \
    --log_steps 50 \
    --save_steps 1000 \
    --max_checkpoints 5 \
    --num_workers 8 \
    --text_max_length 256 \
    --gradient_checkpointing \
    --wandb_project LightMGT \
    --vq_ckpt ${VQ_CKPT:-/mnt/bn/search-auto-eval-v2/zhouwei/nextmgt/tokenizer_ckpts/pretrain256_262144.ckpt} \
    --wandb_run "phase1_${RESOLUTION:-128}px_${NUM_GPUS}gpu_alldata" \
    --sample_steps 1000 \
    --num_samples 3 \
    --sample_inference_steps 20 \
    --sample_cfg_scale 5.0 \
    ${RESUME_ARG} \
    "$@"
