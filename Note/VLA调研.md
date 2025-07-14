把 **Action 当作第三模态**，用 **Encoder 统一理解 V+L**，再用 **Decoder 自回归 / 扩散式地生成离散或连续动作序列**。

---

## 1️⃣ Action Tokenization & Autoregressive Policy

> 先把连续轨迹压缩成离散 token，再像 GPT 那样做 next-token 预测——这是目前最主流、也最容易与 VLM 背景衔接的路线。

| 论文                                                           | 年份        | 核心贡献                                                                                            | 代码                                                                                                                       |
| ------------------------------------------------------------ | --------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **FAST: Efficient Action Tokenization for VLA**              | 2025      | 时序压缩＋BPE，把任意频率轨迹 → 离散 token，训练 5× 更快，Dexterity 任务仍 SOTA ([arXiv][1])                            | [https://github.com/google-robotics/fast-token](https://github.com/google-robotics/fast-token)                           |
| **VIMA: General Robot Manipulation with Multimodal Prompts** | 2023      | 把视觉 Patch + 文本 Token + 动作 Token 拼成 Prompt；Transformer 解码动作，zero-shot 泛化极好 ([arXiv][2])          | [https://vimalabs.github.io](https://vimalabs.github.io)                                                                 |
| **RT-2: Vision-Language-Action Models**                      | 2023      | 用 CLIP-PaLM 级 VLM 继续训练 RT-1 机器人数据；同一序列里混合「语言-图像-动作」 token，实现真·网页知识 ➜ 机械臂 ([Google DeepMind][3]) | [https://github.com/google-robotics/rt-2](https://github.com/google-robotics/rt-2)                                       |
| **LEO: An Embodied Generalist Agent in 3D World**            | ICML 2024 | 两阶段：(VL 对齐 → VLA 指令微调)；动作解码也是 token 序列，可 chat-plan-act 一体 ([GitHub][4])                         | [https://github.com/embodied-generalist/embodied-generalist](https://github.com/embodied-generalist/embodied-generalist) |
| **A Survey on VLA from an Action-Token Perspective**         | 2025      | 系统梳理「离散/连续/混合 token」设计空间，给出未解问题（长时序、对齐粒度等） ([arXiv][5])                                         | –                                                                                                                        |

**为什么值得做？**

* 与 Transformer/BERT/ViT 技术栈无缝复用。
* 把 Action 离散化后，可直接采用语言式 prompt-engineering、in-context learning。
* Open-ended 任务（长尾物体、新场景）靠大模型的“常识”填补。

---

## 2️⃣ Autoregressive ＋ Diffusion Hybrid（高精度控制）

> 在上层用 token 规划，在下层用扩散模型细化连续轨迹，兼顾泛化与精度。

| 论文                                                       | 年份   | 亮点                                                            | 链接                                                                                   |
| -------------------------------------------------------- | ---- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **DiVLA: Autoregressive Reasoning + Diffusion Policies** | 2024 | 先「token-level」推理，再扩散生成高分辨率关节序列，显著提升抓取成功率 ([Diffusion Vla][6]) | [https://diffusion-vla.github.io](https://diffusion-vla.github.io)                   |
| **VLA Model & Diffusion Policy Switching**               | 2024 | 动态切换 VLA（高层语义）与 Diffusion（低层精度），在人形机械手上获高成功率 ([arXiv][7])     | –                                                                                    |
| **Diffusion Policy**                                     | 2023 | 将轨迹建模为条件扩散过程，现成库好用 ([Diffusion Policy][8])                    | [https://diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu) |

**潜在课题**

* 设计「什么时候切换」的 gating 网络；
* 将扩散解码器改进为 **Transformer-Diffuser**，与文本-图像 Diffusion 融合。

---

## 3️⃣ 梦境数据 × 自监督（Dream-to-Control）

| 论文                                   | 年份   | 内容                                                      | 引用 |
| ------------------------------------ | ---- | ------------------------------------------------------- | -- |
| **DreamVLA**                         | 2025 | 用生成式世界模型“梦”出并标注(vision, action)对，提升低数据场景泛化 ([arXiv][9]) |    |
| **Trajectory Ensemble Voting (TEV)** | 2025 | 多策略投票优化 token 轨迹，提高 OOD 物体成功率 ([arXiv][10])             |    |

可结合 **Sim2Real, Data Aug**，与 BEiT-v3 的 MIM 思路对齐。

---

## 4️⃣ 模型-级融合法（Encoder 统一表征 → Decoder 多头输出）

除单一动作流，还可 **多头并行** 输出：

| 论文 / 系统    | 特点                                                           | 说明 |
| ---------- | ------------------------------------------------------------ | -- |
| **PaLM-E** | 把视觉 Patch + 状态量 Embed 到 LLM，输出语言/动作混合序列，可多机体迁移 ([arXiv][11]) |    |
| **Gato**   | 600+ 任务单模型；序列里统一「像素-文本-控制」 token，强调通用性 ([YouTube][12])       |    |

---

## 5️⃣ 方向

1. **可变粒度 Action Token**

   * 基于 FAST 改进：动态长度、语义分段；
   * 围绕「一个 token = 一类 primitive」的语义一致性度量。

2. **Hierarchical Encoder-Decoder**

   * 上层解码高抽象动作 token，下层微调（diffusion / spline）；
   * 类似 “字幕+关键帧→逐帧补完” 思路。

3. **多模态对比学习 ⟹ Token 对齐**

   * 把 CLIP 的 InfoNCE 扩展到 (V, L, A) 三元组；
   * 学习「视觉 Patch ↔ 语言片段 ↔ 动作 primitive」互检索。

4. **Benchmark & 数据**

   * **RLBench / ManiSkill2 / Habitat 3.0 / BEHAVIOR-1K**：都能提供 (RGB, State, Trajectory)；
   * 3D VL-对齐后，再 Instruction-tune 成 VLA。

5. **推理效率 & 内存**

   * 研究 bottleneck token（参考 Attention Bottlenecks），或者 **Mixture-of-Action-Experts** 与 VLMo 的 MoME 呼应。

---

[1]: https://arxiv.org/html/2501.09747v1?utm_source=chatgpt.com "FAST: Efficient Action Tokenization for Vision-Language ... - arXiv"
[2]: https://arxiv.org/abs/2210.03094?utm_source=chatgpt.com "VIMA: General Robot Manipulation with Multimodal Prompts - arXiv"
[3]: https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/?utm_source=chatgpt.com "RT-2: New model translates vision and language into action"
[4]: https://github.com/embodied-generalist/embodied-generalist?utm_source=chatgpt.com "[ICML 2024] Official code repository for 3D embodied generalist ..."
[5]: https://arxiv.org/abs/2507.01925?utm_source=chatgpt.com "[2507.01925] A Survey on Vision-Language-Action Models - arXiv"
[6]: https://diffusion-vla.github.io/?utm_source=chatgpt.com "DiffusionVLA: Autoregressive Reasoning and Diffusion Policies for ..."
[7]: https://arxiv.org/abs/2410.14022?utm_source=chatgpt.com "Vision-Language-Action Model and Diffusion Policy Switching ..."
[8]: https://diffusion-policy.cs.columbia.edu/?utm_source=chatgpt.com "Diffusion Policy"
[9]: https://arxiv.org/html/2507.04447v1?utm_source=chatgpt.com "DreamVLA: A Vision-Language-Action Model Dreamed with ... - arXiv"
[10]: https://arxiv.org/html/2507.05116v1?utm_source=chatgpt.com "Vision-Language-Action Optimization with Trajectory Ensemble Voting"
[11]: https://arxiv.org/abs/2303.03378?utm_source=chatgpt.com "PaLM-E: An Embodied Multimodal Language Model"
[12]: https://www.youtube.com/watch?v=kT6DYKgWNHg&utm_source=chatgpt.com "A Generalist Agent (Gato) - DeepMind's single model learns 600 tasks"
