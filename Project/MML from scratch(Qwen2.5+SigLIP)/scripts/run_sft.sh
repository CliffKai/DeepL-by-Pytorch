#!/bin/bash

# ============ Stage 2 SFT: 单图指令微调 ============
# 冻结 SigLIP + 投影层，解冻 Qwen2.5 全参数
# 图片来源: COCO train2014 (代码硬编码 COCO_train2014_ 前缀)
# 文本来源: llava_instruct_230k.json (150K+80K 合并, 237K 多轮对话)

# ---------- 路径配置 ----------
PRETRAIN_MODEL_PATH="save/pretrain/checkpoint-500"
IMAGES_PATH="/home/tdh/datasets/COCO_images/train2014"
DATA_PATH="/home/tdh/datasets/Chinese-LLaVA-Vision-Instructions/LLaVA-Instruct-150K/translated/llava_instruct_230k.json"
OUTPUT_DIR="save/sft"

# ---------- GPU 配置 ----------
export CUDA_VISIBLE_DEVICES=1

# ---------- 运行训练 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR" || exit 1

python sft_train.py \
    --pretrain_model_path "$PRETRAIN_MODEL_PATH" \
    --images_path "$IMAGES_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR"
