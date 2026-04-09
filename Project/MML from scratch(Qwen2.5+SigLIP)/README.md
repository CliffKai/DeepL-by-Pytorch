# VLM From Scratch - Reproduction

Vision Language Model (VLM) 从零复现，使用 SigLIP 作为视觉编码器，Qwen2.5-0.5B 作为语言模型，通过线性投影层连接两者。

## 模型架构

```
图片 → SigLIP (视觉编码器, frozen) → 196 patches → 每4个合并 → 49 tokens
                                                                  ↓
                                                          linear1 (3072→896)
                                                                  ↓
                                                              SiLU 激活
                                                                  ↓
                                                          linear2 (896→896)
                                                                  ↓
文本 → Tokenizer → Embedding → 替换 <|image_pad|> 位置 ← 图像特征注入
                                          ↓
                                  Qwen2.5-0.5B (LLM)
                                          ↓
                                      logits → 生成文本
```

## 使用的模型

| 组件 | 模型 | 用途 |
|------|------|------|
| 视觉编码器 | [siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) | 图片编码为 patch 特征 |
| 语言模型 | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 文本理解与生成 |

## 训练阶段与数据集

| 阶段 | 文件 | 数据集 | 可训练参数 | 超参数 |
|------|------|--------|-----------|--------|
| Stage 1: 预训练 | `train.py` | [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) (595K图文对) | linear1, linear2 (投影层) | bs=8, lr=1e-4, epoch=5, ga=8 |
| Stage 2: 单图SFT | `sft_train.py` | 图片: COCO train2014, 文本: [Chinese-LLaVA-Vision-Instructions](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions) 中的 llava_instruct_230k.json (150K+80K合并, 237K多轮对话) | Qwen2.5 全参数 | bs=2, lr=1e-4, epoch=5, ga=8 |
| Stage 3: 多图SFT | `sft_train_multi_images.py` | 多图对话数据集 (parquet格式) | Qwen2.5 全参数 | bs=4, lr=1e-5, epoch=1, ga=2 |

### 训练策略说明

- **Stage 1 (预训练)**: 冻结 SigLIP + Qwen2.5，只训练投影层。目标：学习视觉到语言的特征映射。
- **Stage 2 (单图SFT)**: 冻结 SigLIP + 投影层，解冻 Qwen2.5。目标：学习根据图片进行多轮对话。
- **Stage 3 (多图SFT)**: 同 Stage 2，但使用多图数据。目标：学习同时理解多张图片。

## 文件说明

```
├── train.py                    # 核心模型定义 (VLMConfig, VLM) + 预训练脚本
├── sft_train.py                # 单图SFT训练脚本
├── sft_train_multi_images.py   # 多图SFT训练脚本
├── inference.py                # 单图推理
├── inference_multi_images.py   # 多图推理
├── gradio_vlm.py               # Gradio交互界面 (预训练/SFT模型对比)
└── scripts/
    ├── run_pretrain.sh             # Stage 1 预训练启动脚本
    ├── run_sft.sh                  # Stage 2 单图SFT启动脚本
    ├── run_sft_multi_images.sh     # Stage 3 多图SFT启动脚本
    ├── run_inference.sh            # 单图推理脚本
    └── run_inference_multi_images.sh # 多图推理脚本
```

## 数据准备

### 模型下载

所有模型存放在 `/home/tdh/models/` 下：

```bash
cd /home/tdh/models

# Qwen2.5-0.5B-Instruct (已有)
# siglip-base-patch16-224 需要下载:
bash hfd.sh google/siglip-base-patch16-224
```

### 数据集准备

所有数据集存放在 `/home/tdh/datasets/` 下：

**Stage 1 预训练数据:**
- 图片: `LLaVA-CC3M-Pretrain-595K/` — 需解压 `images.zip`（解压后图片直接在根目录下，非 images/ 子目录）
- 文本: `Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json`

```bash
cd /home/tdh/datasets/LLaVA-CC3M-Pretrain-595K
unzip -q images.zip
# 注意: 图片直接解压到当前目录 (GCC_train_*.jpg)，不是 images/ 子目录
```

**Stage 2 SFT 数据:**
- 图片: `COCO_images/train2014/` — 代码中硬编码了 `COCO_train2014_` 前缀 (`sft_train.py:48`)
- 文本: `Chinese-LLaVA-Vision-Instructions/LLaVA-Instruct-150K/translated/llava_instruct_230k.json`
  - 原始数据集只提供 `llava_instruct_150k.json` (157K) 和 `llava_instruct_80k.json` (80K)
  - 需手动合并为 230k 版本:

```python
import json
with open('.../translated/llava_instruct_150k.json') as f:
    d1 = json.load(f)
with open('.../translated/llava_instruct_80k.json') as f:
    d2 = json.load(f)
json.dump(d1 + d2, open('.../translated/llava_instruct_230k.json', 'w'), ensure_ascii=False)
# 合并后: 157,712 + 80,000 = 237,712 条
```

## 使用方法

### 依赖安装

```bash
pip install torch transformers einops pillow pandas gradio sentencepiece protobuf tensorboard
```

### Stage 1: 预训练

```bash
bash scripts/run_pretrain.sh
```

或手动运行:

```bash
python train.py \
    --llm_model_path /home/tdh/models/Qwen2.5-0.5B-Instruct \
    --vision_model_path /home/tdh/models/siglip-base-patch16-224 \
    --images_path /home/tdh/datasets/LLaVA-CC3M-Pretrain-595K \
    --data_path /home/tdh/datasets/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json \
    --output_dir save/pretrain
```

### Stage 2: 单图SFT

```bash
bash scripts/run_sft.sh
```

或手动运行:

```bash
python sft_train.py \
    --pretrain_model_path save/pretrain/checkpoint-500 \
    --images_path /home/tdh/datasets/COCO_images/train2014 \
    --data_path /home/tdh/datasets/Chinese-LLaVA-Vision-Instructions/LLaVA-Instruct-150K/translated/llava_instruct_230k.json \
    --output_dir save/sft
```

### Stage 3: 多图SFT

```bash
bash scripts/run_sft_multi_images.sh
```

或手动运行:

```bash
python sft_train_multi_images.py \
    --sft_model_path save/sft \
    --images_path /home/tdh/datasets/minimind-v_dataset \
    --data_path /home/tdh/datasets/minimind-v_dataset/sft_i2t.parquet \
    --output_dir save/sft_multi_image_v2
```

### 单图推理

```bash
bash scripts/run_inference.sh
```

或手动运行:

```bash
python inference.py \
    --model_path save/sft \
    --image_path /path/to/image.jpg \
    --prompt "描述图片内容" \
    --device cuda:0
```

### 多图推理

```bash
bash scripts/run_inference_multi_images.sh
```

或手动运行:

```bash
python inference_multi_images.py \
    --model_path save/sft_multi_image_v2 \
    --image_paths /path/to/img1.jpg /path/to/img2.jpg \
    --prompt "对比两张图片的内容" \
    --device cuda:0
```

### Gradio Demo

```bash
python gradio_vlm.py \
    --pretrain_model_path save/pretrain \
    --sft_model_path save/sft \
    --device cuda:0 \
    --port 7891
```

## 相比原始代码的改进

### Einops 替换

将原始代码中的 `tensor.view()` 操作替换为语义更清晰的 `einops.rearrange`：

| 位置 | 原始代码 | einops 替换 | 说明 |
|------|---------|------------|------|
| train.py VLM.forward | `image_embeds.view(b, -1, d*4)` | `rearrange(image_embeds, 'b (s g) d -> b s (g d)', g=4)` | patch 压缩 |
| train.py VLM.forward | `logits.view(-1, logits.size(-1))` | `rearrange(logits, 'b s v -> (b s) v')` | loss 计算展平 |
| train.py VLM.forward | `labels.view(-1)` | `rearrange(labels, 'b s -> (b s)')` | loss 计算展平 |
| train.py VLM.merge | `image_features.view(-1, embed_dim)` | `rearrange(image_features, 'n p d -> (n p) d')` | 特征展平 |

### Bug 修复

1. **冻结条件判断错误** (`sft_train.py`, `sft_train_multi_images.py`): 原始代码 `if 'linear' in name or 'vision_model':` 中 `'vision_model'` 是非空字符串，恒为 True。修正为 `if 'linear' in name or 'vision_model' in name:`。
2. **重复惩罚无效** (`inference.py`, `inference_multi_images.py`, `gradio_vlm.py`): 原始代码 `logits[:, token] /= 1.0` 除以 1.0 无任何效果，疑似想实现 repetition penalty 但参数写错。

### 路径参数化

所有硬编码的模型路径和数据路径改为 `argparse` 命令行参数，方便在不同环境下使用。

## 踩坑记录

### 1. Vision 模型选择: siglip-base-patch16-224 vs siglip2-so400m-patch16-384

服务器上已有 `siglip2-so400m-patch16-384`，但本项目代码基于原始 MiniMind-V 架构，投影层维度按 `siglip-base-patch16-224` 的 hidden_size=768 设计 (768\*4=3072 → 896)。如果换用 siglip2-so400m (hidden_size=1152)，投影层维度不匹配会报错。因此需要额外下载 `siglip-base-patch16-224`。

### 2. 预训练图片解压后无 images/ 子目录

`LLaVA-CC3M-Pretrain-595K/images.zip` 解压后，图片 (`GCC_train_*.jpg`) 直接散落在根目录下，而非预期的 `images/` 子目录中。因此 `--images_path` 应直接指向 `LLaVA-CC3M-Pretrain-595K/`，而非 `LLaVA-CC3M-Pretrain-595K/images/`。

### 3. SFT 图片来源: COCO train2014 而非 minimind-v_dataset

`sft_train.py` 第 48 行硬编码了图片名拼接逻辑:

```python
image_name = 'COCO_train2014_' + str(sample['image'])
```

因此 SFT 阶段的图片必须使用 COCO train2014 数据集 (`COCO_images/train2014/`)，图片命名格式为 `COCO_train2014_000000xxxxxx.jpg`。minimind-v_dataset 中的图片是 parquet 内嵌格式，不适用于当前代码的单图 SFT。

### 4. llava_instruct_230k.json 不存在，需手动合并

Chinese-LLaVA-Vision-Instructions 数据集中没有现成的 `llava_instruct_230k.json`，只有:
- `llava_instruct_150k.json` (157,712 条)
- `llava_instruct_80k.json` (80,000 条)

需要将两者合并得到 237,712 条数据 (即 README 中所说的 "230K")。
