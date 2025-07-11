多模态融合（Language + 3D Vision + Action）用于具身智能体（Embodied Agents）与机器人操控

| 研究方向| 核心问题 & 关键特性 | 主要训练范式 | 代表论文（按发表时间）|
| ------------------------- | ------------------------------------------------------------- | ----------------------------------- | --------------------------------------------------------------- |
| **1. MLLM → 机器人操控（对象/姿态中心）**        | 让多模态大模型直接生成机械臂 6-DoF 姿态或抓取位姿；强调**对象可供性**、**链式推理**与**少量微调**    | 预训练的大模型 + 少量姿态监督微调 | *ManipLLM* (CVPR 24) ([arXiv][1])；*RoboMamba* (NeurIPS 24) ([arXiv][2])                            |
| **2. 行动表示与对齐（Action Tokenization）** | 如何把**连续控制**和**离散指令**映射到 LLM 词表 / 离散 token；比较多粒度量化、VQ-VAE 等方案  | 大规模静态轨迹监督 + 消融比较 | *Grounding Multimodal LLMs in Actions* (NeurIPS 24) ([proceedings.neurips.cc][3])                  |
| **3. 跨模态模仿与数据生成**                   | 借助 LLM 在**文本世界**先学会任务，再蒸馏到视觉-动作环境；或让 LLM 驱动代理主动生成 **多感官交互数据**  | 	文本世界 LLM 生成示范 → 视觉代理模仿学习 |  *EMMA* (CVPR 24) ([arXiv][4])；*MultiPLY* (CVPR 24) ([arXiv][5])                                    |
| **4. 多感官/多模态扩展**                    | 将视觉与**触觉、声音、温度**等感官统一输入/输出，支持更丰富的交互                           | 自监督 3D 感知 + 指令微调 |  *MultiPLY* (CVPR 24) ([arXiv][5])                                                                  |
| **5. 高效推理与部署**                      | 解决机器人端**算力/显存瓶颈**：动态早退出、多级模型、状态空间模型 (SSM)                     | 训练时插入早退出分支；纯监督 | *DeeR-VLA* (NeurIPS 24) ([arXiv][6])；*RoboMamba* (NeurIPS 24) ([arXiv][2])                         |
| **6. 通才 (Generalist) 智能体**          | 单一模型跨**导航、操作、游戏、UI 控制**等多域任务；依赖统一动作词典 + 跨域数据 + 在线 RL           | ① 监督微调 (SFT) ② 可选在线 RL | *LEO* (ICML 24) ([arXiv][7])；*From MLLMs to Generalist Embodied Agents* (CVPR 25) ([CVPR 2025][8]) |
| **7. 多智能体协作与语言通信**                  | 利用 LLM 的语言推理为分布式智能体生成**通信内容**与**协同计划**                         | 	LLM 规划 + 模块化监督训练 | *CoELA* (ICLR 24) ([OpenReview][9])                                                                |
| **8. 规划-可供性-轨迹一体化**                 | 将**任务分解 (HTN)**、**物体可供性检测**与**连续轨迹预测**统一到单一 MLLM，“抽象→具体” 全栈控制  | 	分阶段监督 / 自监督微调 |  *RoboBrain* (CVPR 25) ([arXiv][10])                                                                |

[1]: https://arxiv.org/abs/2312.16217 "[2312.16217] ManipLLM: Embodied Multimodal Large Language Model for Object-Centric Robotic Manipulation"
[2]: https://arxiv.org/abs/2406.04339?utm_source=chatgpt.com "RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation"
[3]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/2406694fd7bc7e7bf257446a14f9ea63-Abstract-Conference.html "Grounding Multimodal Large Language Models in Actions"
[4]: https://arxiv.org/abs/2311.16714 "[2311.16714] Embodied Multi-Modal Agent trained by an LLM from a Parallel TextWorld"
[5]: https://arxiv.org/abs/2401.08577 "[2401.08577] MultiPLY: A Multisensory Object-Centric Embodied Large Language Model in 3D World"
[6]: https://arxiv.org/abs/2411.02359 "[2411.02359] DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution"
[7]: https://arxiv.org/abs/2311.12871 "[2311.12871] An Embodied Generalist Agent in 3D World"
[8]: https://cvpr.thecvf.com/virtual/2025/poster/33823 "CVPR Poster From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons"
[9]: https://openreview.net/forum?id=EnXJfQqy0K "Building Cooperative Embodied Agents Modularly with Large Language Models | OpenReview"
[10]: https://arxiv.org/abs/2502.21257?utm_source=chatgpt.com "RoboBrain: A Unified Brain Model for Robotic Manipulation ... - arXiv"


# 融合语言、3D视觉与行动的 Embodied Agents 最新研究综述（2024–2025）

会议：CVPR、ICCV、NeurIPS、ICML、ICLR 等

方向：Language + 3D Vision + Action

目的：提升 Embodied Agent 真实世界环境理解与操控的能力

时间：2024–2025

## ManipLLM: Embodied Multimodal Large Language Model for Object-Centric Robotic Manipulation (CVPR 2024)

**摘要：** 提出 ManipLLM 方法，将**多模态大语言模型（MLLM）**应用于机器人操作任务，以提升操作的稳定性和泛化性。通过**在预训练视觉-语言大模型中注入适配器并进行微调**，该方法在保持模型原有常识和推理能力的同时，赋予模型物体理解和操作推理能力。ManipLLM 的微调范式包括**物体类别理解**、**可操作部位先验推理**以及**以物体为中心的姿态预测**，以激发大模型在操作任务中的推理能力。推理时，模型接收文本指令和单目RGB图像输入，采用链式思维逐步预测机械臂末端执行器的3D接触点和姿态；初始接触完成后，引入**主动柔顺控制策略**以闭环规划后续路径，并设计了**测试时自适应 (TTA)** 策略使模型更好适应真实环境配置。仿真和真实机器人的实验表明，ManipLLM 在多种物体操作任务上表现出有前景的性能。

**大致内容：** ManipLLM 将图像和自然语言提示输入MLLM模型，以**逐步推理**出机械臂的末端抓取位姿（包括三维接触点以及夹爪朝向）。模型采用**物体中心的3D表示**：先根据视觉识别物体类别和可抓取区域，再由大模型推理输出精确的末端位姿。Inference过程中模型以链式思考方式输出抓取动作序列。此外，ManipLLM 在真实操作中通过在线调整末端阻抗，实现对未知扰动的自适应。在模拟器和真实场景的测试显示，与仅在有限类别数据上训练的传统方法相比，ManipLLM 对**大规模多类别物体**的抓取成功率更高，泛化性更强。

**代码：** 已在 GitHub 开源（地址：[https://github.com/clorislili/ManipLLM）。](https://github.com/clorislili/ManipLLM）。)

## Embodied Multi-Modal Agent trained by an LLM from a Parallel TextWorld (CVPR 2024)

**摘要：** 提出 EMMA（Embodied Multi-Modal Agent）方法，将**纯文本环境中的大语言模型专家**用于指导**视觉环境中的多模态智能体**训练。具体来说，EMMA 在*平行的文本世界*中利用LLM代理先行执行任务、总结错误并给出改进动作（“反思”机制），再将这些改进的动作序列模仿学习到视觉环境中的智能体上。为实现这一跨模态模仿训练，作者设计了**DAgger-DPO 算法**，使视觉-语言模型（VLM）代理能够从文本代理的反思中高效学习。在 ALFWorld 基准的多样任务上，EMMA 相较现有基于视觉语言模型的智能体成功率**提高了 20%–70%**，展现出显著优势。

**大致内容：** EMMA 将**文本环境**与**视觉3D环境**设置为一一对应的平行任务。首先，在纯文本环境中，让强大的GPT-4V（或其他LLM）针对给定指令执行动作、检测并纠正自身错误，从而生成高质量的示范序列。然后，在视觉环境中，以这些示范序列作为行为克隆目标，利用作者提出的\*\*交互模仿学习（DAgger-DPO）\*\*算法微调视觉语言模型代理。这一过程中，文本代理的决策反思（包含错误分析后的改进动作）被不断蒸馏给视觉代理，克服了视觉环境下探索噪声大、缺乏专家指导的难题。最终训练得到的 EMMA 无需进一步依赖LLM指导，就能泛化到各种新任务，在 ALFWorld 可视化环境的一系列*拾取放置*、*工具使用*等复杂操作中取得了远超现有方法的成功率。

**代码：** 官方实现已开放在 GitHub（地址：[https://github.com/stevenyangyj/Emma-Alfworld）](https://github.com/stevenyangyj/Emma-Alfworld）) 。

## MultiPLY: A Multisensory Object-Centric Embodied Large Language Model in 3D World (CVPR 2024)

**摘要：** MultiPLY 提出了一种**多感官、多模态的具身大模型**框架，使大语言模型能够主动**与3D环境交互**并获取多模态感知信息（如视觉、听觉、触觉、温度）。为此，作者构建了大规模的\*\*“多感官宇宙 (Multisensory Universe)”**数据集：由一个LLM驱动的智能体在仿真的3D环境中主动探索500k个交互数据，其中场景基于 Habitat-Matterport 3D 环境，物体来自 ObjectFolder 和 Objaverse，并赋予了丰富的感官属性。模型以**物体为中心**表示3D场景，将环境抽象为一组对象及其感官状态；引入特殊的**动作标记（如 NAVIGATE、OBSERVE、TOUCH、HIT）**来表示智能体在环境中执行的交互动作，以及对应的**状态标记\*\*来封装动作后得到的多感官观测。经过在此数据上的指令微调，MultiPLY 可以生成动作序列指令来引导智能体逐步操作，并在每一步将返回的多感官观测融入模型上下文，持续决策。实验表明，MultiPLY 在多种具身任务上相比基线提升显著，包括物体检索、工具使用、多感官描述和任务分解等，展现了综合的多模态推理和操作能力。

**大致内容：** MultiPLY 的核心是在LLM中**嵌入一套抽象的3D多感官表示和交互机制**。训练阶段，先将3D场景表示为对象列表（带属性），然后在LLM输入序列中加入**视觉 token**（物体3D特征）、**听觉/触觉等感官 token**以及**动作 token**。模型通过预测下一个 token 来决定下一步动作，并接收该动作的感知反馈，以循环方式完成任务。这种设计让模型能灵活地在*全局粗略理解*（通过对象抽象）和*局部细节感知*（通过执行动作获取触觉/声音等信息）之间切换。MultiPLY 在**多感官描述**、**问答对话**、**导航与操作**等任务上大幅超越以往仅有视觉-语言输入的模型。例如，在要求判断微波炉中甜甜圈是否加热好的场景中，模型能够先“听”到微波炉提示音，再导航查看甜甜圈、触摸判断其温度和软硬，从而给出正确回答。

**代码：** 已开源在 GitHub（地址：[https://github.com/UMass-Embodied-AGI/MultiPLY）](https://github.com/UMass-Embodied-AGI/MultiPLY）) 。

## CoELA: Building Cooperative Embodied Agents Modularly with Large Language Models (ICLR 2024)

**摘要：** 本工作面向**多智能体合作**场景，提出一种模块化框架，将**大语言模型**作为核心推理与通信模块，集成感知、记忆、执行等能力，构建合作式具身智能体 CoELA（Cooperative Embodied Language Agent）。CoELA 在*去中心化控制*、*局部感知*、*通信代价高*、*多目标任务*等复杂条件下，充分利用LLM的常识和推理能力，通过**认知模块架构**实现智能体之间的计划协同和语言通信。该框架包括五大模块：**(a) 感知**负责处理原始视觉/听觉输入，**(b) 记忆**存储对环境和其他智能体的知识，**(c) 通信**模块决定何时及如何与同伴交流，**(d) 规划**模块由LLM驱动进行高层推理决策，**(e) 执行**模块将高层计划转化为具体可执行动作。在 ThreeDWorld 多智能体运输任务（TDW-MAT）和 Watch-And-Help 沟通协作任务（C-WAH）中，使用 GPT-4 驱动的 CoELA 智能体表现超过强基线规划方法，涌现出高效的通信策略。即使将LLM换成开源模型（如 LLaMA-2）性能有所下降，但通过收集数据进行微调，CoELA 也能取得有竞争力的成绩。此外，与人类用户的互动实验表明，使用自然语言交流的 CoELA 更受人类信任，在人机协作中效率更高。

**大致内容：** CoELA 的创新在于**模块化地融合LLM**以解决多智能体长视距协作问题。每个智能体由感知模块提取环境状态，并更新内存模块中的**语义记忆**（关于环境常识）、**情景记忆**（关于当前任务进程）和**程序记忆**（动作执行经验）。当需要协作时，智能体通过通信模块**利用LLM拟稿要发送的消息**，并评估发送与否的必要性，从而避免冗余交流。随后，LLM驱动的规划模块基于当前状态和记忆，推理出下一步计划方案，并更新到记忆中。最后执行模块参考程序记忆，将高层计划分解为低层原子动作执行。这种架构下，多个CoELA智能体能够在有限带宽下形成隐式分工，通过自然语言**实时交流意图和信息**，共同完成复杂任务。实验录像和质化分析也验证了智能体间出现如**角色扮演**、**资源共享**等合作行为。

**代码：** 开源代码已发布在 GitHub（地址：[https://github.com/UMass-Foundation-Model/Co-LLM-Agents）](https://github.com/UMass-Foundation-Model/Co-LLM-Agents）) 。

## LEO: An Embodied Generalist Agent in 3D World (ICML 2024)

**摘要：** 该工作提出通用多模态智能体 **LEO**，旨在让人工智能具备在**三维世界**中感知、推理和行动的广泛能力。现有的大模型虽在文本、图像等二维任务上取得成功，但在理解和交互3D环境方面能力有限，阻碍了通用智能的进一步发展。为此，作者构建了 LEO，一个能够在3D世界中执行**多任务、多模态**指令的具身智能体，覆盖*场景理解*、*问答推理*、*路径规划*、*机器人操作*等多种任务。LEO 采用统一的LLM架构，通过两阶段训练获得：首先进行**3D视觉-语言对齐**（LEO-Align），使模型掌握将点云、图像等感知与语言空间对应；其次进行**3D视觉-语言-行动指令微调**（LEO-Instruct），在大量的 embodied 多任务数据上训练模型感知指令并输出相应行动或答案。为支持上述训练，作者精心整理和生成了大规模的数据集，涵盖**物体级**和**场景级**的多模态任务，任务复杂度和规模均超以往。实验结果展示了 LEO 在多项任务上的卓越表现，包括三维场景描述、三维问答、具身推理（如房间中寻找并使用物品）、导航以及机械臂操作等，全面超越各任务的专门模型。消融实验进一步分析了训练数据和模型设计对性能的影响，为未来开发更强大的具身通用智能体提供了有价值的见解。

**大致内容：** LEO 模型将3D环境信息编码为**统一的序列输入**：首先使用3D点云编码器将场景划分为**物体中心的点云表示**（可选地，融合一个2D视觉编码器处理第一人称视图），再将所得的物体特征和视角图像特征离散化为 token。这些视觉 token 与任务指令、一段系统提示共同组成序列喂入LLM，以自回归方式产生输出。训练阶段分两步：在 LEO-Align 阶段，利用现有图文数据和ChatGPT合成的数据对模型进行对比学习，使其能将**3D感知**对齐到**语言空间**；在 LEO-Instruct 阶段，结合公开数据集和新生成的数据（通过提示LLM产生多模态任务），训练模型从感知到行动的映射能力。LEO 最终能够从一段自然语言任务指令出发，理解场景（通过3D/2D视觉）、进行推理，并以**多模态形式**给出响应——可能是一段文字答案，也可能是一系列机器人动作序列。例如，LEO 可以根据口头指令在房间中导航并操纵物体，或回答关于复杂3D房间布局的问题，表现出前所未有的通用性和准确性。

**代码：** 官方实现代码已开源（GitHub仓库：[https://github.com/embodied-generalist/embodied-generalist）。](https://github.com/embodied-generalist/embodied-generalist）。)

## From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons (CVPR 2025)

**摘要：** 这项研究探索了**多模态大语言模型（MLLM）**在传统视觉、语言任务之外的**领域扩展**能力，特别聚焦于**Embodied AI**（具身智能）、电子游戏、UI操作、任务规划等场景。作者提出了将一个强大的 MLLM **适配为通用具身智能体** (Generalist Embodied Agent, GEA) 的流程。GEA 是一个统一的单一模型，通过引入**多主体行动序列的统一离散表示**（multi-embodiment action tokenizer），能够将自身感知和决策**绑定到不同的环境和动作空间**，涵盖机器人操作、导航、游戏控制、UI操作等。训练上，GEA 先在一个大型多领域具身经验数据集上进行**监督微调**（SFT），学习各类任务的示范轨迹；随后，在交互模拟器中通过**在线强化学习**微调，进一步提升其在不同环境中的鲁棒性。实验结果表明：融合跨领域数据进行训练对于构建通用智能体至关重要，在线RL训练可以显著提高模型的稳健性和纠错能力。最终得到的 GEA 模型在**跨多个基准**的未见任务上表现优异，**泛化性能**超越以往的多任务模型以及各单领域的专门策略。例如，在 CALVIN 机械臂操作基准上，GEA 在未见指令和新背景下的成功率比之前最优通用模型提高了显著百分比，接近专用模型的水平；在 Habitat 移动拾取任务上，GEA 在未见场景中的成功率甚至超过了基于真实状态训练的RL策略；在 ProcGen 游戏环境中，GEA 达到了专家水平的高比例得分，明显优于先前的专用模型。

**大致内容：** **GEA** 的构建借鉴了近年来大型视觉-语言-动作模型的经验。作者首先设计了一种**统一动作表示**方案：将连续控制（如机器人关节运动）通过*分层量化*方式编码为若干离散token，以确保既有精度又不至于序列过长；同时将离散动作空间（如游戏按键、UI点击）直接**对齐到LLM的词表**，利用语义关联增强模型输出的有效性。在此基础上，预训练的多模态LLM模型通过上述统一动作token空间进行微调，使其能够从多模态观察输入（如视觉帧、文本指令）预测下一步动作。大量不同领域的数据（超过220万条轨迹，来源于人类演示或已有策略）用于监督微调阶段，涵盖机器人抓取、导航、操纵UI、玩游戏等。由于离线数据难以穷尽所有情况，模型可能缺乏纠错能力，作者进一步在一部分模拟环境中让智能体自主与环境交互并通过强化学习训练，提升其**容错和自我恢复**能力。最终的 GEA 模型在**从未见过的任务**上展现出强大的跨领域泛化能力。值得注意的是，GEA 使用一个共享的模型和权重，就实现了此前需要多个专门模型才能分别完成的任务。这表明通过结合LLM的泛化能力和跨模态训练策略，有望培育出“一模多能”的通用智能代理。作者也强调了**跨领域多样数据**以及**在线交互训练**对于构建通用智能体的重要性。

**代码：** 作者表示将开源训练和评估代码以及模型权重（截至发稿时代码暂未公开）。

## Grounding Multimodal Large Language Models in Actions (NeurIPS 2024)

**摘要：** 本文系统研究了如何将**多模态大语言模型（MLLM）**与**不同形式的行动空间**进行最佳结合。作者在五种具身环境、114个任务上比较了七种动作嵌入与对齐方法，得出以下经验：对于**连续控制**类动作，采用**多尺度的学习型离散化token**来表示动作（例如不同精度分辨率的姿态token集合），可以在保证足够动作精度的同时，让大模型便于生成和学习，从而取得最佳下游性能；对于**离散动作**（如有限的指令集），将动作语义与MLLM原生输出空间对齐（例如把动作映射为模型词表中的特定单词或短语）能显著提升决策效果。这些结论为多模态模型的**行动决策绑定**提供了指南，有助于未来设计能流畅从感知和语言推理过渡到实际操作的智能体。

**大致内容：** 该研究由 Apple 等机构的团队完成，延续并深入了其在通用具身智能体（如 GEA）方向的工作。他们把重点放在**动作空间的表示与对接**上，通过大量实验量化不同策略的优劣。例如，对于机械臂的连续运动，他们尝试了将末端6-DoF动作离散成单词、编码成二进制序列、学习VQVAE码本表示等多种方案，发现**使用多粒度的离散动作token**能让LLM既掌握精细动作又不失泛化。对于游戏等离散按键序列，他们尝试了直接输出键位符号和输出相应描述性单词等方法，结果表明**让LLM以自身词汇来表示动作**效果最佳，因为模型可以利用预训练中学到的语义关联来规划动作序列。通过综合对比114项任务的成功率，论文给出了各方案的性能排名和分析。这项研究为MLLM融合行动决策提供了宝贵的实践教训，证明了**根据动作类型选择恰当的表示**对于提升多模态大模型的控场能力至关重要。

**代码：** 论文主要贡献在方法评估分析上，未直接提供可用代码实现（暂未发现官方代码发布）。

## DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution (NeurIPS 2024)

**摘要：** 针对真实机器人平台**算力和内存有限**而多模态大模型推理代价高的问题，DeeR-VLA 提出了一种**动态早退出推理框架（Dynamic Early-Exit）**。作者观察到：在机器人执行复杂任务的过程中，**大部分步骤其实相对简单**，只需较小模型就能推断正确动作；只有少部分棘手情境才需要完整的大模型能力。据此，DeeR-VLA 在多模态大模型中引入**多级退出架构**：模型在中途若判定当前层次已足够应对眼前情况，便停止进一步计算，以节省算力。同时，作者设计了满足不同资源约束（如平均计算开销、峰值延迟、GPU内存占用）的**早停判据**，确保模型在给定硬件条件下高效运行且性能不显著下降。另外，为了让多级退出的模型依然胜任序列决策，DeeR-VLA 采用特别的训练策略将**时间序列信息**融入各中间退出分支，使其对机器人控制的上下文有所记忆。在 CALVIN 机械臂多任务操作基准上，DeeR-VLA **将推理计算量减少了 5.2–6.5 倍，GPU 显存占用降低一半**，同时成功率与完整模型相当。这表明，通过情境判断动态调节模型规模，可大幅提升大模型在机器人上的部署效率。

**大致内容：** DeeR-VLA 的核心是在一个预训练好的多模态Transformer模型内部插入若干\*\*“退出点”**。这些退出点相当于模型的浅层版本：例如在第N层输出一个小头（head）直接产生动作决策。当当前环境状态简单时，小头已足够给出正确动作，后续层计算就可以略去。为了确定何时退出，作者定义了若干策略：可以根据**预计的平均算力预算**让模型自动调整退出频率，或根据**实时延迟/功耗上限**进行反馈控制，使得在繁简不同场景下模型弹性使用不同深度。此外，他们通过在训练时加入随机截断序列的方法，让每个中间退出都能获得时间顺序信息，保证即使提前停止，模型也考虑了一定历史。实验中，他们详细评测了在**不同计算约束\*\*下模型性能与资源的权衡：结果显示，无论限制平均算力还是峰值延迟，DeeR-VLA 相比不退出的原模型都大幅降低了实际推理耗时和能耗，而任务成功率几乎不变甚至略有提升（因为减少了冗余计算可能缓解过拟合）。作者还公开了代码，方便研究者在自己的多模态大模型上复现该方法。

**代码：** 项目代码和模型已在 GitHub 发布（地址：[https://github.com/yueyang130/DeeR-VLA）](https://github.com/yueyang130/DeeR-VLA）) 。

## RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation (NeurIPS 2024)

**摘要：** RoboMamba 提出了一种高效的**视觉-语言-动作（VLA）**机器人智能体模型，兼顾复杂任务的推理能力和实际部署的计算成本。当前一些VLA模型已经能让机器人理解自然语言指令并执行基本视觉操作任务，但在**复杂推理**和**推理效率**两方面存在不足。为此，RoboMamba 引入了一种**状态空间序列建模架构 Mamba**：它属于*线性时间复杂度*的序列模型，擅长处理长序列依赖。作者将预训练视觉编码器与 Mamba 模型相结合，通过联合训练**对齐视觉特征与语言嵌入空间**，赋予模型**视觉常识**以及复杂任务的推理能力。接着，为了让 RoboMamba 能输出机器人操作所需的**三维姿态（SE(3)）**, 作者只在模型末端添加了一个轻量的**策略头**进行姿态预测，并探索了冻结大部分模型参数、仅微调极少部分（占模型参数0.1%）的高效训练策略。研究发现，当模型具备足够的语言视觉推理能力后，只需极小的微调代价便可以习得准确的操作技能。实验评估显示：RoboMamba 在通用推理和机器人任务基准上取得了**优异的结果**，在模拟和现实环境的抓取姿态预测上表现**出色**，而推理速度比现有VLA模型**提升了约3倍**。

**大致内容：** RoboMamba 模型架构上以**Mamba状态空间模型**为核心。这种模型不同于传统Transformer，更适合长序列推理且推理复杂度低。作者首先对公开的大规模图文数据进行训练，使视觉编码器输出的图像特征可以通过Mamba与文本结合，从而**赋予模型理解复杂场景和指令的能力**。然后，在机器人操作数据上，仅针对最终输出的**姿态预测头**进行微调，这意味着模型主干几乎保持冻结，仅用少量训练就让模型学会了具体的**机械臂抓取和放置**动作。由于模型已经具备强大的推理能力，它能将对任务的高级理解转化为正确的操作参数，无需大规模调整底层权重。实验中，作者在**通用推理基准**（如文本问答、图像推理）以及**机器人专有基准**上都测试了模型：RoboMamba 在推理正确性上显著超过以往方法，同时在**抓取定位**等任务上达到了很高精度。例如，在模拟环境中，它能准确预测多种物体的6-DoF抓取位姿，在真实机器人上也成功完成相应操作。更难能可贵的是，由于架构高效，RoboMamba 的推理速度大幅领先，使其更接近实时部署。

**代码：** 官方代码和项目页面已公布（GitHub仓库：[https://github.com/lmzpai/roboMamba）。模型在GitHub上已有超过百星关注](https://github.com/lmzpai/roboMamba）。模型在GitHub上已有超过百星关注) 。

## RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete (CVPR 2025)

**摘要：** RoboBrain 致力于构建一个统一的大模型，使机器人在执行**长序列复杂操作任务**时具备从高层规划到低层控制的完整能力。作者指出，当前多模态大模型在机器人领域暴露出三大不足：缺乏**任务规划**能力（无法将复杂指令分解为子任务）、缺乏**可供性感知**（难以理解交互对象的操作可能性）、缺乏**轨迹预测**能力（无法预判完成任务所需的连续操作轨迹）。为此，他们构建了**ShareRobot**数据集，涵盖任务规划、物体可供性、机械臂末端轨迹等多维标注信息，由多人精细校对，质量和多样性均很高。基于该数据集，开发了 MLLM 架构的 **RoboBrain 模型**。RoboBrain 通过**多阶段训练策略**将**通用多模态数据**与**机器人专用数据**结合，利用高分辨率图像和长时视频输入提升模型对复杂操作的理解和执行能力。大量实验表明，RoboBrain 在多个机器人任务上达到了**最新的性能水平（SOTA）**，充分展示了其作为机器人“大脑”从抽象理解到具体动作的潜力。

**大致内容：** RoboBrain 的框架融合了**规划、感知、执行**三大模块于一体。首先在训练流程上，模型经历了从**高层规划**到**低层控制**的逐步能力注入：Stage1-2，用大规模通用图像/视频-文本数据预训练模型的基础感知和理解能力；Stage3，引入机器人任务的规划数据（ShareRobot的一部分）微调模型，使其学会根据长文本指令规划子任务序列；Stage4，使用 ShareRobot 中的可供性标注和轨迹标注，通过 LoRA 等高效微调方法，分别赋予模型对物体可操作部位的预测能力和对未来运动轨迹的预测能力。最终，RoboBrain 可以从抽象的任务描述，一路推理出需要操作哪些物品、如何操作（抓哪里、用何工具等），以及具体执行的连续动作序列。例如，给定指令“把桌上的红色杯子放入柜子”，模型会首先规划**子任务**（走到桌前->抓起红杯子->走向柜子->打开柜门->放入杯子），接着识别杯子的**可抓取部位**和柜门的**可开合部位**，最后生成机械臂末端的**轨迹**来完成这些动作。一系列评估显示，RoboBrain 在**任务分解准确性**、**可供区域识别**、**轨迹成功率**等方面均优于此前的单一功能模型，证明了统一框架的优势。RoboBrain 的出现表明，通过结合多源数据和分阶段训练，大模型有望成为机器人领域集感知、认知、行动于一身的“大脑”式核心。

**代码：** RoboBrain 项目已开源，包括模型代码、数据集和预训练权重（ HuggingFace & ModelScope 平台 ）。官方仓库提供了模型推理和评估的脚本，RoboBrain 1.0 入选 CVPR 2025 **Embodied AI Trends**展示。

<br>\*\*参考文献：\*\*本文内容引用了上述论文的公开摘要和相关页面等，以确保信息准确可靠。
