# 背景

早期视觉-语言预训练模型的目标是：

> **让模型学会理解图像和语言之间的语义关系。**

比如：

* 看图回答问题（VQA）
* 图文匹配（ITM）
* 图文检索（IR/TR）
* 图像字幕生成（Image Captioning）

## 1 标准处理流程（标准范式）

整体结构是“图像 + 文本 → Transformer”的两步式结构：

```
原始图像 ⟶ CNN 提取特征 ⟶ 区域检测（Region Features）
                                  ⬇
                      文本 token ⟶ 词嵌入
                                  ⬇
                拼接图文 token ⟶ Transformer 融合
                                  ⬇
                           多模态输出任务
```

### 1.1 图像处理流程：两阶段处理

(1) CNN 特征提取：使用 ResNet、Faster R-CNN Backbone 等，输出一个二维特征图（feature map）。

(2) 区域检测（Region Features）

* 在 feature map 上使用 RPN 提出多个候选框（Proposals）；
* 对每个候选框使用 ROI Align 提取特征；
* 每个 Region 变成一个高维向量（通常 36 个，每个维度为 2048）；
* 最终形成图像的“token 序列”。

> 每个 Region Feature 是“这张图某个局部区域可能是某种对象”的表示。

### 1.2 文本处理流程

* 原始文本 → Tokenizer；
* 每个 token 通过 embedding 映射为向量；
* 加入位置编码；
* 形成文本 token 序列。

### 1.3 图文 token 融合

* 将图像 Region Features 和文本 token 拼接起来组成统一输入序列：

  ```
  [CLS], region_1, ..., region_36, word_1, ..., word_N, [SEP]
  ```
* 输入到一个 **Transformer** 中（通常是 BERT 或 BERT-style 架构）；
* Transformer 使用自注意力机制，让图像区域与文本词语相互建立联系。

### 1.4 下游任务头（Head）

不同任务对应不同输出方式：

| 任务              | 输出形式                       |
| --------------- | -------------------------- |
| VQA（视觉问答）       | \[CLS] token → 分类头（预测答案）   |
| ITM（图文匹配）       | \[CLS] token → 二分类（匹配/不匹配） |
| 图文检索            | 图文向量 → 相似度打分               |
| Masked Language | MLM 任务，遮盖词预测               |

## 2 代表性模型

| 模型             | 视觉输入            | 模态融合结构          | 备注                 |
| -------------- | --------------- | --------------- | ------------------ |
| **ViLBERT**    | Region Features | 双塔 + Cross-Attn | 图像和文本分别编码后再交互      |
| **LXMERT**     | Region Features | 分离编码 + 融合层      | 类似 ViLBERT，任务分支更复杂 |
| **UNITER**     | Region Features | 单塔 BERT         | 图文拼接 + 自注意力，结构清晰   |
| **VisualBERT** | Region Features | 单塔              | 早期统一模型             |


## 3 标准范式的问题与瓶颈

### 3.1 计算成本高

* Faster R-CNN 非常耗资源；
* 每张图都要提前处理或在线处理 region。

### 3.2 非端到端，结构复杂

* 图像模块和 Transformer 是分开的；
* 不能统一优化，部署不友好。

### 3.3 表达能力受限

* Region 是离散、有限的；
* 一些模糊区域（如背景、属性）很难提取。


## 4 后续转变方向（为 ViLT 奠定基础）

这些问题促使后续研究走向：

* 更轻量的图像表示方式（如 patch embedding）；
* 图文 token 同构（ViLT 结构）；
* 全端到端统一模型（如 BLIP、GIT、OFA）；
* 使用 ViT 代替 CNN；
* 多模态预训练任务更统一（ITM + MLM + CL）。

# ViLT工作