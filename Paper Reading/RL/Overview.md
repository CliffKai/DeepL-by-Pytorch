LLM 的“强化学习训练”通常指 **post-training / alignment 阶段**：先有预训练模型，再做 SFT，然后用人类偏好、AI 偏好、奖励模型或可验证奖励继续优化。经典 RLHF 流程是“收集偏好 → 训练奖励模型 → 用 RL 优化策略”，InstructGPT 和 Llama 2-Chat 都采用了这类思路；Llama 2 还明确提到用 **rejection sampling** 和 **PPO** 迭代优化。([arXiv][1])

## 1. 经典 RLHF：Reward Model + PPO

这是最传统、最“正宗”的 LLM 强化学习路线。

| 方法                                 | 核心思想                                 | 常见算法                              |
| ---------------------------------- | ------------------------------------ | --------------------------------- |
| **RLHF**                           | 人类比较多个回答，训练 reward model，再优化模型输出     | PPO、A2C/Actor-Critic、REINFORCE 变体 |
| **PPO-RLHF**                       | 用奖励模型给回答打分，同时加 KL 惩罚避免模型偏离原始 SFT 模型  | PPO + KL penalty                  |
| **Best-of-N / Rejection Sampling** | 采样多个回答，用 reward model 选最好的，再做蒸馏或 SFT | 严格说更像“选择+监督微调”，不是完整 RL            |

**PPO** 曾是 RLHF 的默认选择，因为它比较稳定，能限制策略更新幅度。但它需要 policy、reference model、reward model、value model 等多个组件，显存和工程复杂度都比较高。([arXiv][1])

## 2. 更轻量的在线 RL：RLOO、ReMax、GRPO

近两年很多方法的目标是：**保留在线 RL 的探索能力，但降低 PPO 的复杂度**。

| 算法        | 重点                                                                         | 适用场景            |
| --------- | -------------------------------------------------------------------------- | --------------- |
| **RLOO**  | REINFORCE Leave-One-Out，用同一 prompt 下其他采样的平均奖励作 baseline，减少方差               | 在线 RLHF，想替代 PPO |
| **ReMax** | 基于 REINFORCE，用更简单的 baseline 技巧，避免训练 value model                            | 资源受限的 RLHF      |
| **GRPO**  | Group Relative Policy Optimization，对同一问题采样一组回答，用组内相对奖励估计优势，不需要 value model | 数学、代码、推理任务      |
| **RLVR**  | Reinforcement Learning with Verifiable Rewards，用答案是否正确、测试是否通过等可验证信号作为奖励    | 数学、编程、逻辑推理      |

**GRPO** 是目前推理模型训练里非常重要的方法。DeepSeekMath 提出 GRPO 作为 PPO 的变体，以提升数学推理并降低 PPO 的内存开销；DeepSeek-R1/R1-Zero 则把大规模 RL 用于激励推理能力，尤其强调可验证奖励和自我演化式推理行为。([arXiv][2])

## 3. RLAIF / Constitutional AI：用 AI 反馈替代人类反馈

**RLAIF** 是 RLHF 的扩展：不完全依赖人类标注，而是让更强的模型或规则系统来评价回答。Anthropic 的 **Constitutional AI** 用“宪法原则”指导模型自我批评、自我修改，并在后续阶段用 AI 生成的偏好训练奖励模型。([arXiv][3])

这类方法常见于安全对齐、风格对齐、拒答策略训练，因为人工标注昂贵且扩展困难。RLAIF 论文也把它定位为扩展 RLHF 的一种方式，用现成 LLM 生成偏好信号来训练 reward model。([arXiv][4])

## 4. 偏好优化：DPO、IPO、KTO、ORPO、SimPO

这些方法经常被放在“RLHF 替代方案”里讨论。严格说，很多不是传统 RL，因为它们 **不显式训练 reward model，也不在线 rollout**，但目标仍是从偏好中优化策略。

| 算法                 | 是否需要 reward model | 是否需要在线采样 | 特点                                            |
| ------------------ | ----------------: | -------: | --------------------------------------------- |
| **DPO**            |                 否 |        否 | 直接用 chosen/rejected 偏好对优化模型，简单稳定              |
| **IPO / ΨPO**      |                 否 |        否 | 从理论上修正 DPO/RLHF 的偏好学习假设                       |
| **KTO**            |                 否 |        否 | 不要求成对偏好，只需“好/坏”二元反馈                           |
| **ORPO**           |                 否 |        否 | 把 SFT 和偏好优化合成一个阶段，无需 reference model          |
| **SimPO**          |                 否 |        否 | 用平均 log probability 作为隐式奖励，去掉 reference model |
| **SLiC-HF / RRHF** |              否或弱化 |        否 | 基于排序、校准或 margin loss 的离线偏好学习                  |

**DPO** 的影响最大：它把 RLHF 的最优策略形式改写成一个简单的分类式损失，因此不需要显式奖励模型、在线采样或 PPO 调参。([arXiv][5])
后续的 **KTO、ORPO、SimPO** 主要是在数据形式、是否需要 reference model、是否合并 SFT 阶段等方面继续简化。KTO 只需要 desirable/undesirable 二元信号；ORPO 直接把偏好惩罚并入 SFT；SimPO 则用序列平均 log probability 作为隐式奖励并去掉 reference model。([arXiv][6])

## 5. 离线 RL / 自训练：ReST、A-LoL 等

还有一类方法介于 RL、数据筛选和自训练之间。

| 方法                                 | 核心思想                                   |
| ---------------------------------- | -------------------------------------- |
| **ReST**                           | 模型先生成大量样本，再用奖励模型或评价器筛选，用离线 RL/监督方式改进模型 |
| **A-LoL**                          | 把完整回答当作一个动作，用已有数据和优势函数做离线策略优化          |
| **Self-Rewarding / Self-Training** | 模型自己生成、评价、筛选数据，再迭代训练                   |

**ReST** 的思路是“Grow + Improve”：先用当前模型生成数据，再用离线 RL 或筛选后的数据改进模型，优势是数据可复用，比典型在线 RLHF 更省。([arXiv][7])
**A-LoL** 则把整个输出序列看成一个动作，用已有数据和序列级奖励做离线策略梯度训练，目标是提升稳定性和样本效率。([arXiv][8])

## 现在最常见的组合

实践中常见的训练路线大致是：

**通用聊天模型**：
预训练 → SFT → 偏好数据 → DPO / ORPO / SimPO，或 RLHF-PPO。

**高质量对话/安全对齐**：
SFT → reward model → PPO / RLOO / ReMax → 安全红队数据继续训练。

**数学、代码、推理模型**：
SFT 或冷启动数据 → 可验证奖励 → GRPO / RLVR → 蒸馏到小模型。

**低成本开源模型对齐**：
SFT → DPO / KTO / ORPO / SimPO，通常比完整 PPO-RLHF 更容易复现。

一句话总结：**PPO-RLHF 是经典路线，DPO 系列是低成本偏好优化主流，GRPO/RLVR 是当前推理模型训练的热点，RLAIF/Constitutional AI 用于降低人类反馈成本并增强安全对齐。**

[1]: https://arxiv.org/abs/2203.02155?utm_source=chatgpt.com "Training language models to follow instructions with human feedback"
[2]: https://arxiv.org/abs/2402.03300?utm_source=chatgpt.com "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open ..."
[3]: https://arxiv.org/abs/2212.08073?utm_source=chatgpt.com "[2212.08073] Constitutional AI: Harmlessness from AI Feedback"
[4]: https://arxiv.org/abs/2309.00267?utm_source=chatgpt.com "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with ..."
[5]: https://arxiv.org/abs/2305.18290?utm_source=chatgpt.com "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
[6]: https://arxiv.org/abs/2402.01306?utm_source=chatgpt.com "KTO: Model Alignment as Prospect Theoretic Optimization"
[7]: https://arxiv.org/abs/2308.08998?utm_source=chatgpt.com "Reinforced Self-Training (ReST) for Language Modeling"
[8]: https://arxiv.org/abs/2305.14718?utm_source=chatgpt.com "Leftover Lunch: Advantage-based Offline Reinforcement Learning for Language Models"
