#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0  # 使用4个GPU

# 获取脚本所在目录的绝对路径
THIS_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "$THIS_DIR")

# Tokenizer路径
MINILLM_TOKENIZER_PATH="${PROJECT_ROOT}/assets/minillm_tokenizer"
QWEN_TOKENIZER_PATH="${PROJECT_ROOT}/assets/qwen_tokenizer"
MINIMIND_TOKENIZER_PATH="${PROJECT_ROOT}/assets/minimind_tokenizer"
TOKENIZER_PATH=${MINILLM_TOKENIZER_PATH}

# 训练参数
BATCH_SIZE=80
LEARNING_RATE=9e-4
EPOCHS=2
DIM=512
N_LAYERS=8
MAX_SEQ_LEN=512
NUM_WORKERS=8
ACCUMULATION_STEPS=8
GRAD_CLIP=1.0
WARMUP_ITERS=0
LOG_INTERVAL=50
SAVE_INTERVAL=500
MAX_TOKENS_PER_BATCH=8192

# 数据路径
VL_DATA_PATH="/mnt/d/pretrain/minimind-v_dataset/sft_vlm_data.jsonl"
IMAGE_BASE_DIR="/mnt/d/pretrain/minimind-v_dataset/sft_images"
DATA_PATH=${VL_DATA_PATH}

# 训练模式 (llm 或 llm-vl)
TRAIN_MODEL="llm-vl"
MODE="sft"

# 模型输出目录
OUTPUT_DIR="${PROJECT_ROOT}/assets/mini${TRAIN_MODEL}_output/${MODE}"

# 自动获取最新的checkpoint
MODEL_DIR="${OUTPUT_DIR}/dim_${DIM}/n_layers_${N_LAYERS}/"
LLM_CHECKPOINT_PATH="/mnt/d/linux/LLM/MiniLLM/assets/minillm_output/sft/dim_512/minillm_sft_v2.0_build20250617.pth"
VL_CHECKPOINT_PATH=""

if [ -d "$MODEL_DIR" ]; then
    # 获取最新的checkpoint文件
    LATEST_CHECKPOINT=$(ls -t "$MODEL_DIR" | head -n1)
    if [ ! -z "$LATEST_CHECKPOINT" ]; then
        if [ "$TRAIN_MODEL" = "llm" ]; then
            LLM_CHECKPOINT_PATH="${MODEL_DIR}${LATEST_CHECKPOINT}"
            echo "llm 使用模型: ${LLM_CHECKPOINT_PATH}"
        elif [ "$TRAIN_MODEL" = "llm-vl" ]; then
            VL_CHECKPOINT_PATH="${MODEL_DIR}${LATEST_CHECKPOINT}"
            echo "llm-vl 使用模型: ${VL_CHECKPOINT_PATH}"
        fi
    fi
fi

# 分布式训练参数
DDP="--ddp"
LOCAL_RANK="-1"

# 构建训练命令
python train.py \
    --out_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --mode ${MODE} \
    --train_model ${TRAIN_MODEL} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --device "cuda" \
    --dtype "bfloat16" \
    --use_wandb \
    --wandb_project "MiniLLM" \
    --num_workers ${NUM_WORKERS} \
    ${DDP} \
    --local_rank ${LOCAL_RANK} \
    --accumulation_steps ${ACCUMULATION_STEPS} \
    --grad_clip ${GRAD_CLIP} \
    --warmup_iters ${WARMUP_ITERS} \
    --log_interval ${LOG_INTERVAL} \
    --save_interval ${SAVE_INTERVAL} \
    --dim ${DIM} \
    --n_layers ${N_LAYERS} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --use_moe false \
    --data_path ${DATA_PATH} \
    --image_base_dir ${IMAGE_BASE_DIR} \
    --max_tokens_per_batch ${MAX_TOKENS_PER_BATCH} \
    --use_dynamic_length \
    ${LLM_CHECKPOINT_PATH:+--llm_checkpoint_path ${LLM_CHECKPOINT_PATH}} \
    ${VL_CHECKPOINT_PATH:+--vl_checkpoint_path ${VL_CHECKPOINT_PATH}} 