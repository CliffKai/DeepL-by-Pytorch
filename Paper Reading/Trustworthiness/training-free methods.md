**training-free methods** 指：**不重新训练、不微调 LVLM 参数，而是在推理阶段通过修改 prompt、解码策略、logits、attention、hidden states，或者增加验证与重采样流程来降低幻觉。**

需要注意：**training-free 不等于完全没有额外计算成本**。有些方法不训练模型，但推理时会多跑几次前向传播、计算对比 logits、调用验证器，甚至像本文的 SGRS 一样需要梯度反向传播。

下面只是大致总结了一下 LVLMs-Saliency 论文中引用的论文的做法。

## 1. 修改解码策略或 logits

这类方法不更新模型参数，而是在生成每个 token 时重新调整候选 token 的概率。

| 方法                           | 论文                | 核心思路                                                                                                                                                         |
| ---------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **VORD**                     | Neo & Chen, 2024  | *Visual Ordinal Calibration for Mitigating Object Hallucinations in Large Vision-Language Models*。通过视觉序数校准，修正模型在对象判断上的偏差。                                    |
| **IMCCD**                    | Li et al., 2025a  | *Mitigating Hallucination for Large Vision Language Model by Inter-Modality Correlation Calibration Decoding*。根据图像与文本之间的跨模态相关性，对解码分数进行校准。                    |
| **Summary-Guided Decoding**  | Min et al., 2024  | *Mitigating Hallucinations in Large Vision-Language Models via Summary-Guided Decoding*。利用摘要信息约束生成，避免输出逐渐偏离图像内容。                                             |
| **CMI-Calibrated Decoding**  | Fang et al., 2025 | *Grounding Language with Vision: A Conditional Mutual Information Calibrated Decoding Strategy for Reducing Hallucinations in LVLMs*。利用条件互信息校准生成，使输出更依赖视觉信息。 |
| **Retrospective Resampling** | Wu et al., 2025b  | *Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling*。先生成，再验证；发现问题后重新采样。                                    |

这类方法可以理解为：

> 模型参数不变，但生成 token 时不直接照搬原始 softmax 概率，而是增加一个校准、验证或重采样步骤。

### 1.1 VORD

**VORD 并不是简单地对“对象是否存在”的最终判断做后处理，而是利用原始图像与扰动图像之间的视觉差异，对生成过程中的 token 概率进行序数校准（ordinal calibration）**。这篇论文由 Dexter Neo 和 Tsuhan Chen 提出，于 2024 年 12 月发布在 arXiv。 ([arXiv][1])

**主要工作：**
该论文针对大型视觉语言模型（LVLM）容易生成图像中不存在对象的问题，提出了视觉序数校准方法 VORD。作者首先对原始图像施加 Mixup、扩散噪声等扰动，构造视觉信息更不可靠的修改图像，并分别计算模型在原始图像和修改图像条件下的 token 生成概率。其基本假设是：对于真正受到图像证据支持的 token，模型在清晰原始图像上的置信度应当高于在扰动图像上的置信度；反之，如果某个 token 在视觉信息被破坏后仍然获得较高概率，甚至概率上升，那么它更可能来自语言先验或数据集偏差，而不是来自可靠的视觉证据。作者据此设计了两种实现形式：无需重新训练的 **VORD Decoding**，以及用于微调模型的 **VORD Loss**。 ([arXiv][2])

**主要观点：**
论文认为，LVLM 的对象幻觉并非完全随机产生，而是与训练数据中的对象共现偏差、语言模型固有的文本先验以及视觉不确定性密切相关。当图像信息变弱时，模型更容易依赖“场景中通常会出现什么对象”来生成文本。例如，在视觉证据不足时，模型可能根据常见搭配生成并不存在的 `person`、`bicycle` 或 `parachute`。因此，缓解幻觉不仅要提高模型输出的准确率，还需要校准模型的置信度：真实视觉对象的概率应当随着视觉证据变弱而有序下降，而缺乏视觉依据的对象 token 不应保持异常高的置信度。VORD 的核心思想，就是利用原始图像与扰动图像之间的这种**序数关系**识别并压制不可信 token。 ([arXiv][2])

**核心贡献：**
VORD 的贡献主要体现在三个方面。第一，论文从置信度校准的角度分析对象幻觉，指出 LVLM 在视觉扰动下并不会自然形成稳定、有序的 token 概率关系。第二，作者提出训练无关的 VORD Decoding：在推理阶段比较原始图像和扰动图像对应的 token 概率，通过序数掩码过滤不满足约束的候选 token，因此可以作为轻量级插件接入现有 LVLM。第三，作者提出可训练的 VORD Loss，并进一步使用原始图像与扰动图像视觉 token 的相似度动态确定惩罚幅度，避免固定 margin 造成过度惩罚。实验在 POPE、MME Hallucination Subset 和 LLaVA-Bench 等基准上表明，该方法能够降低对象幻觉，同时改善模型置信度校准；作者还报告，在 POPE 上使用 VORD Loss 可使准确率和 F1 分数分别最高提升约 2.9 和 2.7 个百分点。 ([arXiv][2])

[1]: https://arxiv.org/abs/2412.15739?utm_source=chatgpt.com "VORD: Visual Ordinal Calibration for Mitigating Object Hallucinations in Large Vision-Language Models"
[2]: https://arxiv.org/html/2412.15739v1 "VORD: Visual Ordinal Calibration for Mitigating Object Hallucinations in Large Vision-Language Models"

### 1.2 IMCCD

PAI 不仅修改 attention，还通过对比“有图像输入”和“无图像输入”时的输出分布，进一步压制由语言先验主导的候选 token。

**主要工作：**
论文首先提出一种称为 **文本惯性（text inertia）** 的现象：在图像描述任务中，LVLM 有时会生成图像中不存在的对象；即使移除图像，仅保留此前已经生成的文本，模型仍然会继续输出相同的幻觉内容。作者认为，这表明部分生成结果主要来自语言模型根据上下文进行的自动补全，而不是视觉证据。为缓解这一问题，作者提出训练无关的 PAI 方法，在不重新训练模型、也不调用外部工具的情况下，直接干预 LVLM 的推理过程。([arXiv][1])

**主要观点：**
PAI 的核心观点是：LVLM 的幻觉不仅源于视觉能力不足，也源于视觉信息与语言先验之间的失衡。随着文本逐步生成，模型可能越来越依赖历史文本，而逐渐忽视图像 token。为此，PAI 包含两个环节。第一个环节是在 decoder 的 self-attention 中，提高当前生成位置对图像 token 的 pre-softmax attention 分数，再通过 softmax 重新分配权重，使视觉表示更充分地进入当前 token 的隐藏状态。第二个环节是 **Image-Centric Logit Refine**：额外执行一次不含图像 token 的纯文本前向传播，并从多模态输出分布中减去部分纯文本输出分布，从而削弱仅靠语言惯性就能获得高概率的候选 token。

**核心贡献：**
论文的核心贡献是提出了一种轻量级、可插拔的推理时干预方案：一方面增强图像 token 在生成过程中的作用，另一方面抑制过强的语言先验。该方法可以与 greedy decoding、beam search 和 nucleus sampling 等不同解码策略结合，并在 LLaVA、MiniGPT-4 和 Shikra 上进行验证。以 LLaVA 的 greedy decoding 为例，使用 PAI 后，描述中包含幻觉对象的句子比例 `CHAIR_S` 从 `46.6` 降至 `24.8`，幻觉对象占全部被提及对象的比例 `CHAIR_I` 从 `13.4` 降至 `6.9`。

与 VORD 的区别：**PAI 并不是把 attention weight 当作幻觉成因的完整解释，而是把 attention 当作可干预的计算路径。** 即使 raw attention weight 不能直接等同于 token 的真实重要性，主动提高图像 token 参与特征混合的程度，仍然可能改善模型输出。

[1]: https://arxiv.org/html/2407.21771v1 "Paying More Attention to Image: A Training-Free Method for Alleviating Hallucination in LVLMs"

## 2. 增强视觉信息的影响

这类方法认为幻觉的重要原因之一是：随着生成过程推进，模型对图像 token 的依赖逐渐减弱，语言先验开始占据主导。因此，它们在推理时增强视觉信号。

| 方法                              | 论文                 | 核心思路                                                                                                                                                       |
| ------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PAI**                         | Liu et al., 2024c  | *Paying More Attention to Image: A Training-Free Method for Alleviating Hallucination in LVLMs*。增强模型对图像 token 的 attention。                                 |
| **DAMRO**                       | Gong et al., 2024  | *Dive into the Attention Mechanism of LVLM to Reduce Object Hallucination*。分析并干预 attention，以减少对象幻觉。                                                        |
| **Visual Information Steering** | Li et al., 2025b   | *The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering*。通过视觉信息 steering，使生成过程持续受到图像约束。         |
| **Latent Space Steering**       | Liu et al., 2024b  | *Reducing Hallucinations in Vision-Language Models via Latent Space Steering*。不是直接改输出概率，而是在隐空间中引导模型生成。                                                     |
| **Attention-Causality Method**  | Zhou et al., 2024  | *Mitigating Modality Prior-Induced Hallucinations in Multimodal Large Language Models via Deciphering Attention Causality*。分析 attention 的因果作用，缓解模态先验造成的幻觉。 |
| **From Pixels to Tokens**       | Shang et al., 2024 | *From Pixels to Tokens: Revisiting Object Hallucinations in Large Vision-Language Models*。重新分析视觉 token 与对象幻觉之间的关系，并据此进行推理期干预。                              |

PAI、DAMRO、latent-space steering 和 summary-guided decoding 的标题可以直接在当前论文的参考文献中看到。

这类方法可以概括为：

> 不训练模型，而是在 forward pass 或 decoding 阶段让图像证据“更有话语权”。

---

## 3. 修改输入约束或利用自我纠错

这类方法不一定直接修改模型内部 attention，而是在输入、反馈或验证环节增加约束。

| 方法                                                    | 论文                  | 核心思路                                                                                                                               |
| ----------------------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Constraint-Aware Prompting**                        | Wu et al., 2025a    | *Mitigating Hallucinations in Multimodal Spatial Relations through Constraint-Aware Prompting*。通过约束感知 prompt，减少空间关系幻觉。             |
| **Self-Correcting Decoding with Generative Feedback** | Zhang et al., 2025a | *Self-Correcting Decoding with Generative Feedback for Mitigating Hallucinations in Large Vision-Language Models*。利用生成反馈进行自我纠错式解码。 |

它们可以理解为：

> 不修改权重，而是让模型在生成前、生成中或生成后增加额外约束。

---

# 这篇论文与上述 training-free 方法的区别

现有 training-free 方法中，很多都关注：

* 模型是否过度依赖语言先验；
* 模型是否忽略图像 token；
* attention sink 是否导致模型把注意力错误集中在某些位置；
* 如何通过修改 decoding 或 attention 减少幻觉。

而这篇论文换了一个观察角度：

> 它不只关注“模型是否还在看图像”，而是关注“模型是否还记得自己刚刚生成了什么”。

作者认为，幻觉出现之前，一个明显现象是：

[
\text{prior output tokens}
;\longrightarrow;
\text{next-token prediction}
]

之间的 saliency 下降。

因此，本文提出两个同样不需要重新训练的方法：

| 本文方法       | 是否需要训练 | 作用                                                 |
| ---------- | -----: | -------------------------------------------------- |
| **SGRS**   |      否 | 计算候选 token 的 saliency；如果候选 token 与前文联系太弱，就拒绝并重新采样。 |
| **LocoRE** |      否 | 增强模型对最近生成 token 的 attention，避免模型在长文本生成时“忘记”前文。     |

其中，LocoRE 仅修改推理过程中的 attention weights，不计算梯度、不修改模型参数；SGRS 也不训练模型，但为了计算 saliency，需要在推理时执行梯度计算，因此成本更高。论文附录指出，SGRS + LocoRE 相较普通 greedy decoding 会增加约 30%–40% 的每 token 推理开销，而单独使用 LocoRE 的延迟增幅低于 2%。

---

# 一句话区分

你可以先这样记忆：

* **训练型方法**：让模型通过数据重新学习如何少产生幻觉。
* **training-free 方法**：模型本身不变，但在推理阶段纠正它的生成过程。
* **本文的方法**：模型本身不变，重点防止模型在自回归生成时失去对先前输出内容的有效依赖。
