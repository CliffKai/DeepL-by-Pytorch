# Embodied AI


---

## 1. Embodied Intelligence for Autonomous Systems on the Horizon

一个展望式工作坊，关注“自治系统（如无人车、机器人）中的 Embodied Intelligenc。

* **目标**：思考自动系统如何通过引入基础模型和大规模数据提升感知、规划与决策能力。
* **示范话题**：融合世界模型、奖励学习、视频驱动的自驾模型、无人机竞速等。
* **核心意义**：为未来自主系统勾勒路线—从感知-行动闭环，到通用推理与适应能力。

---

## 2. Foundation Models Meet Embodied Agents

/聚焦于 **Foundation Model（基础模型）如何构建 Embodied Agent** 的专题研讨。

* **组织者**：包括 Fei‑Fei Li、Jiayuan Mao、Shenlong Wang 等一线专家。
* **核心内容**：探讨 LLM/VLM 到 VLA 从基础模型支持 embodied决策的路径，结合 MDP 框架分析决策、子目标分解、动作序列生成与转移建模。
* **形式**：结合理论与实践，展出多模态基础模型在真实 agent 中的作用与挑战。

---

## 3. Multi-Agent Embodied Intelligent Systems Meet Generative-AI Era: Opportunities, Challenges and Futures

/该研讨聚焦在 **多 agent Embodied 系统中融合生成式 AI** 的趋势与难点。

* **内容涵盖**：

  * 基于 Foundation Model 的多智能体生成式协作；
  * 构建含感知、规划、交流、执行的多模态 agent 框架；
  * 分析系统架构与物理/虚拟 embodiment 问题；
* **价值**：系统性揭示 Generative-AI 如何提升多 agent 协作智能，强调现实与仿真环境协同。

---



# Robotic Manipulation

---

## 🧠 1. **3D Vision Language Models for Robotic Manipulation: Opportunities and Challenges**

* **主题**：探讨**3D 视觉-语言模型**（3D VLM）在机器人操作中的潜力与瓶颈 ([Robo 3D VLMs][1])。
* **关注点**：

  * 利用三维信息（如点云、3D 包围框）增强语言理解与推理；
  * 3D 信息是否必要，亦或传统 2D 足够；
  * 如何为机器人 policy 学习预训练 3D VLM；
  * 实时系统的 sensor 校准与推理效率；
* **形式**：融合 invited talk、论文分享，推动跨模态预训练模型与机器人决策的落地。

---

## 🌍 2. **Generalization in Robotics Manipulation Workshop and Challenges**

* **主题**：聚焦**机器操作泛化能力**，尤其从模拟环境到真实场景快速迁移 ([Robo 3D VLMs][1], [CVPR 2025][2])。
* **关注点**：

  * 当前 visual-policy 模型在固定环境表现良好，但泛化能力有限；
  * 基础模型（LLM/VLM）能否为泛化提供支持；
  * Sim2real 挑战：包括 GemBench、Colosseum、真实机器人评测任务 ([RoboMani Grail][3], [CVPR 2025][4])；
* **亮点**：设立模拟竞赛 + 真实平台验证，强调算法在多环境下的连续性能。

---

## 🧬 3. **Workshop on 3D‑LLM/VLA: Bridging Language, Vision and Action in 3D Environments**

* **主题**：聚焦**3D-LLM/VLA 模型**，实现语言、视觉和动作在 3D 环境中的协同 ([3D-LLM/VLA Workshop][5])。
* **关注点**：

  * 利用大语言模型理解 3D 场景、分解任务、规划动作；
  * LLM 融合 point cloud / 3D scene 的输入与输出；
  * 应用于控制、导航、指令理解；
* **形式**：结合论文展示与专家发言，建立跨学科共识。

---

## 🏭 4. **Workshop on Perception for Industrial Robotics Automation**

* **主题**：聚焦**工业自动化**（如 bin picking）中的视觉感知瓶颈 ([pira-workshop.github.io][6], [CVPR 2025][7], [CVPR 2025][4], [CVPR 2025][8])。
* **关注点**：

  * 用于工业场景（贝托识别、物体姿态估计）的 3D 场景理解；
  * 通用可靠且低成本方案；
  * 举办 Bin Picking 感知挑战，奖金 \$60k，吸引 450 支队伍参赛；
  * 实时在真实机器人上验证最终系统效果；
* **亮点**：实现工业环境中端到端的视觉、姿态估计、抓取部署闭环。

---

### 📋 整体对比

| Workshop                  | 核心目标               | 关键挑战              | 实践方式                 |
| ------------------------- | ------------------ | ----------------- | -------------------- |
| **3D VLMs**               | 提升语言-视觉融合中的三维理解    | 是否需 3D；预训练策略；实时性能 | 理论＋实验＋模型实践           |
| **Generalization**        | 强化模型跨场景泛化          | sim2real、环境多样性    | 模拟挑战＋真实验证            |
| **3D‑LLM/VLA**            | 联合语言、视觉、行动在 3D 中协作 | 多模态融合；LLM 控制；动作分解 | 研究交流＋示范论文            |
| **Industrial Perception** | 提高工业 task 的感知表现    | 复杂场景；姿态估计；工业部署    | Challenge 比赛＋真实机器人评测 |

---
