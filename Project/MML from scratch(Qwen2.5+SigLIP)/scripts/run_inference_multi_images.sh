#!/bin/bash

# ============ 多图推理 ============

# ---------- 配置 ----------
MODEL_PATH="save/sft_multi_image_v2/checkpoint-2000"
IMAGE_PATHS=(
    "/home/tdh/datasets/COCO_images/train2014/COCO_train2014_000000000009.jpg"
    "/home/tdh/datasets/COCO_images/train2014/COCO_train2014_000000000025.jpg"
)
PROMPT="对比两张图片的内容"
DEVICE="cuda:0"

# ---------- GPU 配置 ----------
export CUDA_VISIBLE_DEVICES=2

# ---------- 运行推理 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR" || exit 1

python inference_multi_images.py \
    --model_path "$MODEL_PATH" \
    --image_paths "${IMAGE_PATHS[@]}" \
    --prompt "$PROMPT" \
    --device "$DEVICE"
