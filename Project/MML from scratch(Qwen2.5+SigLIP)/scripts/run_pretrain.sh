#!/bin/bash

# ============ Stage 1 Pretrain: 训练视觉-语言投影层 ============
# 冻结 SigLIP + Qwen2.5，只训练 Linear Projector (linear1, linear2)
#
# 前置准备:
#   1. 下载 siglip-base-patch16-224:
#      cd /home/tdh/models && bash hfd.sh google/siglip-base-patch16-224
#   2. 解压预训练图片:
#      cd /home/tdh/datasets/LLaVA-CC3M-Pretrain-595K && unzip images.zip

# ---------- 路径配置 ----------
LLM_MODEL_PATH="/home/tdh/models/Qwen2.5-0.5B-Instruct"
VISION_MODEL_PATH="/home/tdh/models/siglip-base-patch16-224"
IMAGES_PATH="/home/tdh/datasets/LLaVA-CC3M-Pretrain-595K"
DATA_PATH="/home/tdh/datasets/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json"
OUTPUT_DIR="save/pretrain"

# ---------- GPU 配置 ----------
export CUDA_VISIBLE_DEVICES=0

# ---------- 运行训练 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR" || exit 1

python train.py \
    --llm_model_path "$LLM_MODEL_PATH" \
    --vision_model_path "$VISION_MODEL_PATH" \
    --images_path "$IMAGES_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR"
