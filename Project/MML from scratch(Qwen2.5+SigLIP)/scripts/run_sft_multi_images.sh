#!/bin/bash

# ============ Stage 3 Multi-Image SFT: 多图指令微调 ============
# 冻结 SigLIP + 投影层，解冻 Qwen2.5 全参数
# 数据: minimind-v_dataset 的 parquet 格式多图对话数据

# ---------- 路径配置 ----------
SFT_MODEL_PATH="save/sft/checkpoint-2500"
IMAGES_PATH="/home/tdh/datasets/minimind-v_dataset"
DATA_PATH="/home/tdh/datasets/minimind-v_dataset/sft_i2t.parquet"
OUTPUT_DIR="save/sft_multi_image_v2"

# ---------- GPU 配置 ----------
export CUDA_VISIBLE_DEVICES=2

# ---------- 运行训练 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR" || exit 1

python sft_train_multi_images.py \
    --sft_model_path "$SFT_MODEL_PATH" \
    --images_path "$IMAGES_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR"
