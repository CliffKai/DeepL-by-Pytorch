#!/bin/bash

# ============ 单图推理 ============

# ---------- 配置 ----------
MODEL_PATH="save/sft/checkpoint-1500"
IMAGE_PATH="/home/tdh/datasets/COCO_images/train2014/COCO_train2014_000000000009.jpg"
PROMPT="描述图片内容"
DEVICE="cuda:0"

# ---------- GPU 配置 ----------
export CUDA_VISIBLE_DEVICES=2

# ---------- 运行推理 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR" || exit 1

python inference.py \
    --model_path "$MODEL_PATH" \
    --image_path "$IMAGE_PATH" \
    --prompt "$PROMPT" \
    --device "$DEVICE"
