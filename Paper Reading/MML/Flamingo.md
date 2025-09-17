# Abstract

Target:Building models that can be rapidly adapted to novel tasks using only a handful of annotated examples is an open challenge for multimodal machine learning research.

Architectural innovations:
1. bridge powerful pretrained vision-only and language-only models;
2. handle sequences of arbitrarily interleaved visual and textual data;
3. seamlessly ingest images or videos as inputs.

# 1 Introduction

当前难点：
1. **少样本适应性差**
   在视觉和多模态任务中，还没有出现能够像 GPT-3 那样，**只依靠少量标注样本或简单提示，就能快速适应新任务**的通用模型。现有方法大多依赖大规模监督预训练 + 任务特定的微调。

2. **现有方法依赖大量数据和微调**
   多模态模型虽然取得了一些进展，但通常需要**成千上万条标注数据**才能在下游任务上表现良好，并且需要针对不同任务做精细的超参数调整，代价很高。

3. **现有模型的局限性**

   * 对比学习的 VLM（如 CLIP）虽然能零样本迁移，但只能做 **相似度匹配/分类** 这类闭集问题，不会生成语言，因此**不适合开放式任务**（如 VQA、captioning）。
   * 一些探索了“视觉条件下文本生成”的方法，在**低数据场景**下效果仍然不佳。

主要贡献：
1. **架构创新**

   * 可以处理**任意交错的图像/视频 + 文本输入**，并生成开放式输出。
   * 解决了开放式任务（captioning、VQA、visual dialogue 等）的 few-shot 学习问题。

2. **严格的性能评估**

   * 系统性地量化评估 Flamingo 在多任务 few-shot 学习中的表现；
   * 特别保留了大规模 **held-out benchmarks**（完全不参与模型设计/调参），保证了评估的**公平性和无偏性**。

3. **性能突破（SOTA）**

   * 在 **16 个任务**上达到了新的 few-shot SOTA；
   * 在其中 **6 个任务**上甚至超过了全监督微调 SOTA，但只用了 32 个样例（≈少 1000 倍标注）；
   * 若增加标注预算，还能在 **5 个额外任务**上通过 fine-tuning 刷新 SOTA。

# 2 Approach


