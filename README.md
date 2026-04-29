# DeepL-by-Pytorch

基于 PyTorch 的深度学习学习仓库，涵盖 PyTorch 基础教程、论文阅读笔记、项目实践、课程笔记等内容。

## 目录

- [Pytorch Learning Jupyter](#pytorch-learning-jupyter)
- [Project](#project)
- [Paper Reading](#paper-reading)
- [Paper Code](#paper-code)
- [CS336 Note](#cs336-note)
- [Note](#note)
- [Interview](#interview)
- [Vibe Coding](#vibe-coding)
- [Tutorial](#tutorial)
- [说明](#说明)

---

## Pytorch Learning Jupyter

**（已完结）**

在 Jupyter 上实现的土堆教学视频内容，内容来自B站up主[我是土堆](https://space.bilibili.com/203989554)的教学视频[PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】](https://www.bilibili.com/video/BV1hE411t7RN/?share_source=copy_web&vd_source=608471d0e25c02d240b92470bd78f213)。

涵盖内容：Dataset、Tensorboard、Transforms、DataLoader、nn.Module、卷积、池化、激活函数、线性层、Sequential、损失函数与反向传播、优化器、预训练模型、模型保存与加载、GPU 训练等。

## Project

**（在更）**

一些 Pytorch 项目：

- **ViT** — Vision Transformer 实现（含 simple_vit 和完整 vit 两个版本）
- **Transformer from scratch** — 从零实现 Transformer（具体实现讲解可以看 cs336: [assignment1-basics (In Progress)](https://github.com/CliffKai/assignment1-basics) 部分）
  - 包含：BPE 分词器训练、数据集处理、模型训练、文本生成
  - 模型组件：Attention、Embedding、FFN、RMSNorm、Linear
- **MML from scratch (Qwen2.5+SigLIP)** — VLM 从零复现，使用 SigLIP 作为视觉编码器，Qwen2.5-0.5B 作为语言模型
  - 包含：预训练、单图/多图 SFT 训练、推理、Gradio Demo
  - 三阶段训练：投影层预训练 → 单图 SFT → 多图 SFT
- **Build the GPT Tokenizer** (In Progress)
- **Qwen2.5vl 部署** — 推理 demo 及源码解读
- **Qwen3omni 源码** (In Progress) — 源码解读
- **HuggingFace Base Class 讲解** (In Progress)
  - Config — PreTrainedConfig、GenerationConfig
  - IO — PreTrainedTokenizer、PreTrainedTokenizerFast、FeatureExtractionMixin、ImageProcessingMixin、ProcessorMixin
  - Model — modeling_utils
  - Output — ModelOutput
  - Trainer — Trainer

## Paper Reading

**（在更）**

一些论文阅读笔记：

- **CV**
  - MAE
  - ViT
  - MoCo
  - Swin Transformer
  - SAM
- **MML（多模态）**
  - CLIP
  - Flamingo
  - MML 串讲
  - Qwen2.5vl
  - ViLT
  - DeepSeek-OCR
  - Qwen3omni
  - Social Debiasing for Fair MLLMs
  - MiniCPM-V 4.5 Technical Report
  - MLLM-Survey（多模态大模型综述）
  - MME-Survey（多模态评测综述）
  - Multimodal Evaluation Notes（多模态评测笔记）
  - Multimodal Notes 1
  - 多模态论文复习
- **NLP**
  - Transformer
  - BERT
  - BERT_CLS
  - Attention/Self-Attention/Cross-Attention 的本质思考
- **VLA**
  - OpenVLA
  - RT-2
  - BitVLA
- **Memory Agent**
  - RAG
  - MemoryBank
  - MemGPT
  - A-Mem
  - Memory OS of AI Agent
- **Interpretability（可解释性）**
  - Towards Monosemanticity
  - Sparse Autoencoders Find Highly Interpretable Features in Language Models
  - Scaling and evaluating sparse autoencoders
  - Improving Dictionary Learning with Gated Sparse Autoencoders
  - Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models
- **LLM-as-Judge**
  - Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
  - JudgeLM

## Paper Code

**（在更）**

一些论文相关的代码（不一定是论文实现）：

- Attention 机制实现

## CS336 Note

**（在更）**

Stanford CS336 大语言模型课程笔记：

- Lecture 1 ~ Lecture 7
- Lecture 10

## Note

**（在更）**

个人学习与研究笔记：

- Flash Attention
- 什么是反向传播
- 常见激活与损失函数笔记
- Chain of Thought (CoT)
- MoCo 学习有感
- VLA 调研
- 大模型时代科研思路
- 如何从论文中发掘研究思路
- 如何读论文 & 如何获取研究灵感
- 机器学习方式
- 大语言模型拟人化研究（研究1 & 研究2）

## Interview

面试准备材料：

- 多模态面试
- 面经

## Vibe Coding

AI 编程工具使用文档：

- **Claude Code** — 命令详解、扩展资源大全、插件完全指南
- **Codex** — CLI 斜杠命令详解

## Tutorial

教程材料：

- ViT 教程

---

## 说明

Paper Reading 受启发于开源社区的大神李沐老师[跟李沐学AI](https://space.bilibili.com/1567748478?spm_id_from=333.337.0.0)的系列视频：[合集·【更新中】AI 论文精读](https://space.bilibili.com/1567748478/lists/32744?type=season)。

做这个系列的目的是因为我自己是初学者，在看沐神系列视频的时候对于沐神认为稀松平常的概念我需要去找很多资料才能搞明白，所以在学习的过程中我便将很多自己查阅到的许多东西都记录了下来，一方面是为了方便我自己后续的复习，另一方面也是希望能记录下来帮助更多的人（希望可以如此）。

因为我使用的是 VScode 进行的编辑，部分公式 GitHub 官网无法渲染，如果遇到网页端公式渲染失败的情况可以下载到本地查看。
