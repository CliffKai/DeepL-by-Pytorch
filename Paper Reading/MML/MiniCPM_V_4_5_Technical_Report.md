# MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data and Training Recipes

---

## 中文翻译

### 标题与作者

**MiniCPM-V 4.5：通过架构、数据和训练方法打造高效的多模态大语言模型**

于天宇、王泽凡、王崇毅、黄福伟、马文硕、何智慧、蔡天驰、陈伟泽、黄宇翔、赵元乾、徐博凯、崔俊博、徐影静、阮立青、张洛源、刘晗宇、唐敬昆、刘洪源、郭启宁、胡文昊、何冰翔、周杰、蔡杰、齐稷、郭宗昊、陈驰、曾国阳、李宇轩、崔淦渠、丁宁、韩旭、姚远\*、刘知远\*、孙茂松\*

MiniCPM-V 团队，OpenBMB

联系邮箱：yiranytianyu@gmail.com，yaoyuanthu@gmail.com

\* 通讯作者。

### 摘要

多模态大语言模型（MLLMs）正在快速发展，代表了人工智能发展的前沿。然而，它们的训练和推理效率已成为让 MLLMs 更加普及和可扩展的核心瓶颈。为了应对这些挑战，我们提出了 MiniCPM-V 4.5，一个专为高效率和强性能而设计的 80 亿参数模型。我们在模型架构、数据策略和训练方法上引入了三个核心改进：一个用于对图像和视频进行高度紧凑编码的统一 3D-Resampler 模型架构；一个无需繁重数据工程即可同时学习文档知识和文本识别的统一学习范式；以及一个兼顾短推理和长推理模式的混合强化学习策略。在 OpenCompass 评测中的综合实验结果表明，MiniCPM-V 4.5 超越了广泛使用的闭源模型（如 GPT-4o-latest），以及规模显著更大的开源模型（如 Qwen2.5-VL 72B）。值得注意的是，这种强性能是以卓越的效率实现的。例如，在广泛采用的 VideoMME 基准上，MiniCPM-V 4.5 在 30B 规模以下的模型中达到了最先进的性能，同时仅使用了 Qwen2.5-VL 7B 46.7% 的 GPU 显存和 8.7% 的推理时间。

### 1 引言

多模态大语言模型（MLLMs）[1, 2, 3, 4, 5, 6, 7] 正在快速推进人工智能的前沿，使机器能够理解和推理不同模态（如文本和图像）的信息。然而，随着 MLLMs 的发展，数据工程、训练和推理的成本也大幅上升。解决这一效率挑战已成为学术界和工业界共同关注的核心议题 [6, 8, 9, 10, 11]，这对于让强大的 MLLMs 更加普及和可扩展至关重要。

我们将这一效率问题分解为三个核心方面：**（1）模型架构。** MLLMs 的主要效率瓶颈来自于高分辨率图像编码所需的大量视觉 token，这给视觉编码器和 LLM 解码器带来了沉重的计算开销。这个问题在视频理解中更加严重，现有模型即使在低帧率采样下，也需要数千个 token 来编码一段短且低分辨率的视频。例如，处理一段分辨率仅为 448×448、2 帧/秒的 6 秒视频，Qwen2.5-VL [7] 需要 1,536 个 token，而 InternVL3 [9] 则需要 3,072 个 token。如此长的视觉 token 序列导致了高昂的训练和推理成本，包括 GPU 显存和计算速度。**（2）训练数据。** 随着我们快速耗尽传统网页数据中的新知识，现代 MLLMs 的一个新基石是从文档中获取高质量的多模态知识 [1, 2]，例如学术论文和教科书。这些文档通常以 PDF 形式存储，包含各个领域的多学科知识，并以文本、图像和表格交错的多样化布局组织。然而，大多数方法依赖于脆弱的外部解析工具，将文档文件转换为交错的图文序列用于训练。这些工具在复杂布局下经常失败，导致知识学习出错，或者需要大量的数据工程工作来修复失败案例。**（3）训练方法。** 强化学习（RL）通过让模型在给出最终答案之前进行逐步显式思考，展现出改进复杂推理能力的潜力 [12, 1]。然而，这种性能提升通常以极端冗长为代价。即使是对于识别显而易见物体等简单任务，大多数现有的思考模型也会产生过长的输出，导致训练和推理效率低下。

为了应对这些挑战，MiniCPM-V 4.5 在模型架构、数据策略和训练方法上引入了三项关键改进：**（1）用于紧凑图像和视频编码的统一 3D-Resampler。** 之前的 MiniCPM-V 系列模型 [6] 通过 2D-Resampler [5, 13] 对高分辨率图像实现了高压缩率（例如，相比大多数 MLLMs 达到 4×）。为了进一步解决视频处理的架构低效问题，我们将 2D-Resampler 扩展为 3D-Resampler，联合压缩视频的时空信息。该模块可以将 6 秒、2 帧/秒、448×448 分辨率的视频仅编码为 128 个视觉 token，相比代表性的 MLLMs [7, 9] 实现了 12×-24× 的 token 成本降低，使高帧率和长视频理解成为可能，并且能统一编码图像。**（2）文档知识和 OCR 的统一学习范式。** 我们提出了一种学习范式，使模型能够直接从文档图像中准确获取知识，无需依赖脆弱的外部解析器。通过对文档中的文本区域动态施加不同程度的噪声干扰，并要求模型重建文本，模型学会了自适应地、恰当地在精确文本识别（当文本大致可见时）和基于多模态上下文的知识推理（当文本严重损坏时）之间切换。**（3）后训练的混合策略。** 与以往专门优化单一长推理模式的模型不同 [2, 1]，我们开发了一种混合 RL 后训练策略，支持用于高效使用的短推理模式和用于复杂任务的长推理模式。在 RL 训练中，我们在 rollout 过程中随机切换两种模式以进行联合优化。这种方法不仅能够灵活控制短推理和长推理模式，还能相互提升性能。在实验中，我们使用更少的训练样本就能在两种模式下都获得更好的推理性能。

OpenCompass 评测的综合实验结果表明，MiniCPM-V 4.5 超越了广泛使用的闭源模型（如 GPT-4o-latest [4]），以及规模显著更大的开源模型（如 Qwen2.5-VL 72B [7]）。值得注意的是，这种强性能是以卓越的效率实现的。例如，在高效统一 3D-Resampler 的支持下，MiniCPM-V 4.5 在 VideoMME [14] 上以仅 9.9% 的推理时间达到了之前最先进 MLLMs [1] 的同等性能。基于混合后训练策略，MiniCPM-V 4.5 在短推理和长推理模式上均表现优异，在 OpenCompass 评测中超越了同期的思考模型 [3, 1]，同时仅使用 42.9%-68.2% 的推理时间。

综上所述，我们的贡献如下：

- 我们开源了 MiniCPM-V 4.5，这是一个高效强大的 MLLM，支持高效的高帧率和长视频理解、强大的 OCR 和文档解析能力，以及可控的混合推理。
- 我们引入了三项关键改进：用于高效图像和视频编码的统一 3D-Resampler、用于文档知识和 OCR 学习的统一范式、以及兼顾性能和效率的混合后训练策略。
- 综合实验证明了所提出的技术改进的有效性和 MiniCPM-V 4.5 的性能。

**图 1：** MiniCPM-V 4.5 架构概览。该模型处理多种视觉输入，如高分辨率图像和高帧率视频。经过图像切分和视频打包处理后，这些输入由视觉编码器编码，然后输入到统一的 3D-Resampler 中。该模块将图像和视频特征高效地压缩为紧凑的 token 序列（图像最高可达 16× 的压缩率，视频再额外实现 6× 的压缩），然后由 LLM 解码器处理。解码器可以以两种不同的风格生成回复：简洁的短推理模式或逐步的长推理模式。

### 2 方法

在本节中，我们描述 MiniCPM-V 4.5 的方法论，包括模型架构和预训练、SFT 及 RL 的配方。

#### 2.1 架构

如图 1 所示，MiniCPM-V 4.5 的架构包含三个主要模块：（1）一个轻量级的视觉编码器，通过特殊的切分策略灵活处理高分辨率图像。（2）一个统一的 3D-Resampler，将图像和视频编码为紧凑特征，利用视觉信息中的时空冗余。（3）一个 LLM 解码器，理解图像、视频、文本并生成文本输出。

##### 2.1.1 统一的 3D-Resampler

为了解决 MLLMs [15, 16] 中图像和视频编码的效率瓶颈，我们将 2D-Resampler 扩展为联合整合时空信息进行压缩的 3D-Resampler。通过这种方式，我们利用连续视频帧的时间冗余性，相较于 2D-Resampler，额外实现了 6× 的时间压缩率。

**图像处理。** 为了处理任意宽高比的高分辨率图像，我们采用了 LLaVA-UHD [13] 的图像切分策略。对于每张图像，我们根据输入分辨率估计理想的切片数量，并选择每切片分辨率最接近视觉编码器预训练设定的切分方案。然后我们使用带有 2D 空间位置编码的可学习查询（queries），通过交叉注意力为每个切片生成固定长度的序列。大多数现有的 MLLMs [7, 9, 1] 采用 MLP 和像素重排（pixel unshuffle）操作进行视觉压缩，通常需要 256 个视觉 token 来编码一张 448×448 的图像。利用 Resampler 架构的灵活性，通过选择少量查询 token，MiniCPM-V 可以在保持良好性能的同时，实现显著更高的视觉 token 压缩率（例如，对 448×448 的图像使用 64 个 token）。

**视频处理。** 为了处理视频数据中的显著冗余，我们采用联合时空压缩策略以获得更高的压缩率。对于每段视频，我们首先沿时间维度将其切分为若干"包"，每个包包含相邻的帧。直观上，同一个包内的视频帧通常共享高度冗余的视觉信息，这些信息在联合建模时可以被识别和压缩。为此，我们通过交叉注意力将每个包中来自视觉编码器的帧特征重采样为固定长度的特征序列。我们为可学习的查询同时增加了与图像编码中相同的 2D 空间位置编码和时间位置编码。最终视频表示是由所有包的 token 序列拼接而成。我们以最高 10 帧/秒的帧率，每段视频最多采样 1080 帧。训练期间，包大小和帧率会被随机增广以提高鲁棒性。这一设计还为推理时提供了灵活性，允许调整这些超参数以满足多样化场景和设备的需求。

基于 3D-Resampler，MiniCPM-V 4.5 可以为视频 token 实现 96× 的压缩率，其中 6 个 448×448 的视频帧² 可以被联合压缩为 64 个视频 token（而对大多数 MLLMs 通常需要 1,536-3,072 个 token）。这意味着模型可以在不增加 LLM 推理成本的情况下感知显著更多的视频帧，从而带来强大的高帧率视频理解和长视频理解能力。

> ² 每一帧从视觉编码器得到 1024 个 token。

**训练效率。** 得益于 Resampler 机制的灵活性（对输入形状不可知），我们可以使用同一个 3D-Resampler 对图像和视频进行统一的视觉编码。这意味着图像和视觉编码共享相同的架构和权重，因此我们可以通过一个轻量级的 SFT 阶段高效地将 2D-Resampler 扩展为 3D-Resampler。此外，这也促进了从图像到视频的高效知识迁移。例如，我们在 MiniCPM-V 4.5 中观察到了合理的视频 OCR 能力，尽管我们并未专门收集这类训练数据。

> **要点**
> 1. 联合时空压缩可以实现更高的视觉压缩率。
> 2. 统一的架构可以通过极少量的额外训练高效适配，并促进从图像到视频的知识迁移。

#### 2.2 预训练

我们的预训练过程旨在通过渐进式多阶段策略系统地构建模型的基础能力。这涉及精心策划的数据组成和一个新颖的文档知识与 OCR 学习的统一范式。

##### 2.2.1 预训练策略

预训练包含三个渐进阶段。每个阶段策略性地解冻不同的模型组件，并引入越来越复杂的数据以优化学习效率。

**阶段 1。** 我们从热身阶段开始，仅训练 Resampler 模块，而所有其他组件保持冻结。该阶段使用图像-caption 数据以最小的训练成本建立视觉与语言模态之间的初始对齐。

**阶段 2。** 然后我们解冻视觉编码器以增强感知基础能力。该阶段消耗富含 OCR 的数据和图像-caption 数据。由于该阶段的数据可能缺乏语言建模所需的流畅性或质量，LLM 解码器在该阶段保持冻结。

**阶段 3。** 在跨模态桥接就位并设定好感知基础后，最终阶段使用我们最高质量的数据端到端训练所有模型参数，包括纯文本语料、图文交错样本、视频以及从前面阶段精选的子集。此时，我们解冻 LLM 解码器，以充分利用数据中的知识和技能，包括多图推理和时间理解。我们采用 Warmup-Stable-Decay 学习率调度器 [17]。在衰减阶段，我们逐步加入更多高质量的指令数据和知识密集型数据。

**图 2：** 通过动态视觉扰动实现的文档知识与 OCR 学习统一范式。我们通过不同级别的扰动创建一系列训练任务：低扰动保留可读性以学习鲁棒 OCR；高扰动迫使模型进行上下文推理；中等扰动则需要从视觉线索和上下文中进行综合推理。

##### 2.2.2 预训练数据

**图像 Caption 数据。** 我们将大规模公开数据集（LAION-2B [18]、COYO [19] 等）与从网络爬取并精心整理的中文图文对相结合。我们过滤掉低分辨率图像，并使用 CLIP [20] 去除不相关的图文对。为了丰富 alt-text 描述，我们在一个子集上采用基于 Capsfusion [21] 的重新描述流程，生成流畅且事实完整的 caption。通过这种方式，我们将原始 caption 中有价值的世界知识转化为更流畅的自然语言。我们使用一个 MLLM 为图像打上概念标签，并确保在不同语言和长尾概念之间保持均衡分布。

**图文交错数据。** 来源于 Common Crawl [22]、OmniCorpus [23] 和 MINT-1T [24]，图文交错数据对于上下文学习和多图理解能力至关重要。我们进行过滤以确保质量，移除图像损坏或图文比例不平衡的样本。我们进一步使用相关性过滤以确保有意义的多模态关联，并采用知识密度过滤为预训练最终衰减阶段选择一个高质量子集。

**OCR 数据。** 我们在预训练早期阶段合成 OCR 数据以增强基础文本识别能力。我们按照 [25] 的方法在自然场景上用各种颜色和字体组合渲染文本，并将真实世界的 HTML 源渲染为图像。

**文档数据。** 我们从网络上收集文档，包括学术论文、科研报告、教科书等。这类数据具有高知识密度并包含视觉上复杂的排版。

**视频 Caption 数据。** 我们聚合了多个公开数据集 [26, 27, 28]，并用更详细的内部视频 caption 进行补充。这一多样化的集合支持视频理解所必需的时间视觉推理能力的发展。

##### 2.2.3 文档知识与 OCR 学习的统一范式

文档（如学术论文和教科书）是学习多样化版式和获取各领域多学科知识的重要资源。然而，大多数 MLLMs 依赖于脆弱的外部解析器将文档 PDF 转换为交错的图文序列用于训练。这种嘈杂且低效的过程经常引入结构性错误，或需要大量数据工程工作来修复失败案例。

OCR 学习的另一个挑战是：虽然更强的图像增广可以创建鲁棒 OCR 所需的更多样化且更难的样本，但过度增广会使文本无法辨认。强迫模型从这种无法辨认的视觉输入中产生真实文本通常会导致幻觉问题。因此，以往我们只能承担较小且安全的增广程度。

为了同时克服这两个挑战，我们提出了一种直接从文档图像中学习的统一训练范式，使用其原始文本作为真值。我们的核心洞察是：文档知识获取和文本识别之间最重要的区别是图像中文本的可见性。我们将两种能力统一为单一的学习目标：从被扰动的文档图像中预测原始文本。通过对文本区域动态施加不同级别的扰动，模型学会了自适应且恰当地在精确文本识别（当文本可辨时）和基于多模态上下文的知识推理（当文本被严重遮挡或掩盖时）之间切换，如图 2 所示。这消除了对脆弱解析器的依赖，并防止了过度增广 OCR 数据带来的幻觉问题。

具体来说，对于每个文档，我们将其文本区域的一个子集视为训练真值。然后我们对每个区域随机应用不同级别的扰动，创建不同的训练任务：

1. **低扰动（增广 OCR）。** 当对文本区域施加轻度噪声时，文本仍然可辨，模型可以通过文本识别有效地预测它们。
2. **中度扰动（综合推理）。** 当对文本区域施加重度噪声时，单个字符变得高度模糊、无法可靠识别。因此模型必须学会将被扰动区域的嘈杂视觉线索与高层次的文档上下文及其内部知识相结合，以重建原始文本。
3. **高扰动（上下文推理与文档知识学习）。** 当文本区域被完全遮盖时，模型无法依赖字符级线索来预测缺失的内容。因此，模型被迫仅从多模态上下文及其内部知识中推断信息。这直接培养了文档级的理解能力。

这种统一方法产生了一个更高效、更有弹性的学习过程。通过直接从文档图像中学习，我们避免了构建复杂的文档解析流水线，并防止了脆弱解析器引入的潜在噪声。此外，这一范式允许我们在同一训练批次内流畅地结合知识学习和 OCR 目标，最大化数据利用率，并产出一个精通广泛文档理解任务的、多功能的单一模型。

> **要点**
> 1. 通过有选择地冻结参数，基础能力可以在不完美的异构数据源上构建。
> 2. 对文档图像文本的简单动态视觉扰动可以有效地将知识学习、鲁棒 OCR 和上下文推理统一为单一学习目标。

#### 2.3 监督微调

监督微调（SFT）阶段旨在激活模型在广泛任务上的能力，并为强化学习做好准备。此外，我们在此阶段将 2D-Resampler 扩展为统一的 3D-Resampler，以增强视频数据的压缩效率。

##### 2.3.1 监督微调策略

我们首先训练通用的交互能力，然后培养高级推理和时间理解的专业技能。

**阶段 1：通用 SFT。** 该阶段旨在激活预训练期间获得的广泛知识，并使其与人类指令对齐。通过在多样化的高质量指令-回复数据混合上微调，模型发展出多模态交互的熟练程度。为了防止纯文本性能的退化并提高训练稳定性，我们在训练混合中加入 10% 的高质量纯文本数据。

**阶段 2：Long-CoT 与 3D-Resampler。** 在前一阶段多功能基础之上，我们培养支持长推理模式、高帧率和长视频理解的专业技能。首先，我们在 SFT 数据中引入 Long-CoT 热身指令。这鼓励模型执行显式的逐步思考过程，融入反思、回溯等认知模式，这对长推理模式至关重要。其次，我们通过将架构从 2D 升级到 3D-Resampler 并引入高帧率和长视频数据来增强其时间理解能力。由于统一的设计，我们发现这种升级可以通过少量高质量视频数据高效实现。

##### 2.3.2 监督微调数据

**STEM 数据。** 为了增强 STEM 推理，我们从网络上精心整理了一个涵盖高中和本科水平多学科问题的数据集，覆盖物理、化学、生物、金融、计算机科学等。为确保数据质量，我们实施了两阶段过滤过程。首先，我们只保留表现出高视觉依赖性的样本（即没有图像信息就无法解决）。其次，我们进行一致性检查以验证答案的正确性。对于每个保留下来的样本，我们通过使用强大 MLLM 的拒绝采样收集干净的推理过程。

**长尾知识数据。** 为了解决模型在不常见主题上常失败的长尾问题，我们从维基百科 [29] 中纳入长尾知识来合成高质量的多模态指令遵循数据。具体来说，对于每个实体页面，我们使用强大的 MLLMs 构建多模态指令和答案，并保留具有高视觉依赖性的样本。

**Long-CoT 数据。** Long-CoT 数据使模型能够获取长推理模式所需的推理模式。我们的数据来自 OpenThoughts [30] 和一个内部流水线。我们通过筛选早期模型难以处理的提示来识别具有挑战性的提示。我们的初步研究表明，聚焦于具有挑战性的问题是发展鲁棒推理能力（而非记忆琐碎模式）的关键。然后每个回复都经过多阶段验证：我们验证其正确性，使用 RLAIF-V [31] 进行 claim 级事实验证以评估可信度，并过滤掉无意义的重复。最后，通过重写对验证过的回复进行增广以提高多样性。

> **要点**
> 过滤简单提示、聚焦于具有挑战性的问题，对于有效的 Long-CoT 热身至关重要。

#### 2.4 强化学习

RL 阶段旨在增强推理性能、实现可控的推理模式并提高可信度。为了提供高效的通用领域奖励，我们将规则验证的奖励（用于简单情况）与来自 RLPR [32] 的通用概率奖励（用于复杂答案）相结合，并添加了一个经过校准的偏好奖励。我们采用混合 RL 策略，允许在短推理和长推理模式之间灵活切换。我们进一步整合 RLAIF-V [31] 以减少幻觉。

##### 2.4.1 强化学习数据

我们的 RL 数据包含跨四个关键领域的高质量样本。每个子集都经过了严格的、人机协同的清洗和去重过程。

**数学。** 我们从学术来源 [33, 34, 35] 收集多模态数学问题，这些问题需要整合视觉感知和逻辑推理。我们观察到许多开源数据集包含严重的标签错误，因此采用了彻底的清洗过程。

**文档、表格和图表。** 为了提升在感知复杂场景下的推理能力，我们精心整理了多样化的真实世界数据集 [36, 37, 38, 39, 40] 和合成数据集 [41, 42, 43] 的混合，以提升领域覆盖范围。

**通用推理。** 为了进一步提升通用推理能力，我们从 VisualWebInstruct [44] 和其他网络资源汇总了一个多样化的问题集合，涵盖逻辑推理和多学科推理任务。这些数据表现出更复杂的参考答案风格，许多问题包含多个子问题。

**指令遵循。** 我们纳入了来自 Llama-Nemotron-Post-Training Dataset [45] 和 MulDimIF 数据集 [46] 的纯文本指令。我们观察到文本指令遵循能力的提升能很好地泛化到多模态指令上。

##### 2.4.2 奖励质量控制

RL 的有效性高度依赖于数据质量。因此，我们实施了严格的质量控制处理，重点关注三个不同的方面：

**标签准确性。** 不正确的标签会引入有缺陷的监督信号。对于每个数据集，我们维护一个小子集来检查标签准确性，并进行人机协同的清洗过程以保持高标签准确性。

**奖励准确性。** 在通用领域验证模型生成的回复是一个非平凡的挑战。手工制定的规则难以应对自然语言的复杂性。为了解决这个问题，我们针对每种情况动态应用最合适的验证方法。对于只包含少量 token 的简单答案，我们采用基于规则的验证系统，实现了 98% 的奖励准确率。对于规则难以适用的复杂自然语言答案（例如，那些包含特定单位或较长措辞的答案），我们使用 RLPR [32] 更通用的基于概率的奖励。

**奖励覆盖率。** 为了补充这些以准确性为核心的信号，我们整合了一个奖励模型，提供密集的、与偏好对齐的信号，引导模型生成更高质量、更像人类的回复。为了避免分布外问题，我们在长推理模式下仅将奖励模型应用于最终答案部分。

##### 2.4.3 混合强化学习

我们为模型采用可控的混合推理设计：用于快速回答的短推理模式和为复杂问题生成显式逐步思考轨迹的长推理模式。模式切换由提示控制。这两种行为在 SFT 期间初始化，然后通过混合 RL 联合优化，其中 rollout 在两种模式之间随机切换。我们应用 GRPO [47] 用这些 rollout 优化模型，并移除 KL 和熵损失以提高稳定性。这种训练方案不仅在保留复杂推理能力的同时保持了短回复的效率，还促进了跨模式泛化——在一种模式中学到的推理能力可以迁移以改进另一种模式。基于这种混合后训练设计，MiniCPM-V 4.5 仅消耗纯长推理策略 70.5% 的训练 token 成本即可获得更好的性能。

##### 2.4.4 奖励塑形

我们设计奖励塑形策略以平衡任务能力、人类偏好和训练稳定性。最终的奖励信号是四个分量的加权组合：准确性奖励 $R_{acc}$、格式奖励 $R_{format}$、重复惩罚奖励 $R_{rep}$ 和偏好奖励 $R_{rm}$。偏好奖励来自用人类偏好数据训练的辅助奖励模型（RM）[48]。然而，在长推理模式下直接应用 RM 会得到不理想的结果，因为标准 RM 难以评估分布外的长推理链，导致更差的对齐和训练不稳定，这也在我们的初步实验中得到了证实。

为了解决这个问题，我们采用了选择性应用策略。RM 只给回复的最终答案部分打分，完全绕过显式思考步骤。这提供了稳定、密集的奖励信号，与人类偏好对齐，同时不会错误地惩罚复杂的推理路径。最终奖励计算如下：

$$R = R_{acc} + R_{format} + R_{rep} + \frac{1}{2} \tilde{R}_{rm}. \quad (1)$$

其中，$\tilde{R}_{rm}$ 是使用 $\frac{R_{rm} - \bar{R}_{rm}}{\sigma(R_{rm})}$ 计算的标准化偏好奖励分数，$\bar{R}_{rm}$ 和 $\sigma(R_{rm})$ 分别表示使用相同提示采样的回复的原始奖励分数的均值和标准差。

##### 2.4.5 RLAIF-V

视觉幻觉仍然是 MLLMs 的一个关键限制，特别是在需要高可靠性的应用中。为了应对这一挑战，我们整合 RLAIF-V [31]，通过可扩展的 AI 反馈对齐使回复更加以视觉输入为依据。值得注意的是，我们将这一方法扩展到视频输入，那里的幻觉问题尤为突出。

**回复采样。** 我们首先在相同的生成条件下从策略模型中采样多个回复。该策略确保对事实准确性的聚焦评估，避免回复之间的分布不匹配。

**反馈收集。** 我们首先将复杂的回复分解为可验证的原子 claim，每个 claim 独立验证。这将复杂的长回复评估转化为更简单的 claim 级验证，解决了整体评估的内在挑战，并提高了事实评估的精度。然后基于聚合的 claim 验证分数构建偏好对，其中包含较少事实错误的回复被视为更优。

**偏好学习。** 得到的偏好数据集（涵盖图像和视频）用于通过 DPO [49] 训练模型。这一阶段对于事实准确性至关重要的视觉任务特别有效，同时不会损害通用性能。

> **要点**
> 1. 将基于规则的奖励（用于简单回复）和基于概率的奖励（用于复杂自然语言回复）相结合，可以为多样化任务提供可靠的奖励系统。
> 2. 混合 RL 实现了长短推理模式之间的跨模式泛化。

### 3 实验

在本节中，我们通过实验评估 MiniCPM-V 4.5 的性能以及所提出方法的有效性。

#### 3.1 基线与基准

我们与多种强大的基线模型进行比较：（1）最先进的开源模型，以 Qwen2.5-VL 72B [7] 为代表；（2）规模相近的强模型，包括 InternVL3 [9]（8B）和 GLM-4.1V [1]（9B）；（3）前沿闭源模型，如 GPT-4o-latest [4]。

我们的评估涵盖了多模态能力的几个关键领域：

**STEM** 包括面向数学和科学的基准，如 MMMU [50]、MathVista [51]、AI2D [52]、MathVerse [53]、LogicVista [54] 和 EMMA [55]，旨在评估逻辑推理、数学问题求解和科学理解能力。

**文档、OCR & 图表** 通过 OCRBench [56]、ChartQA [57]、TextVQA [58]、DocVQA [59] 和 OmniDocBench [60] 涵盖 OCR 相关任务，测试在各种视觉上下文（包括文档和图表）中提取、解读和推理文本信息的能力。

**幻觉** 通过 HallusionBench [61]、ObjHalBench [62] 和 MMHal-Bench [63] 评估模型可靠性，衡量生成虚假或不一致信息的倾向。

**多图 & 真实世界 & 指令遵循** 包括 Mantis [64]、MMT-Bench [65]、RealWorldQA [66] 和 MM-IFEval [67]，评估在涉及多张图像、真实世界理解和指令遵循的复杂场景中的性能。

**视频理解** 涵盖 Video-MME [68]、LVBench [69]、MLVU [70]、LongVideoBench [71]、MotionBench [72] 和 FavorBench [73]，评估各种视频任务中的时间推理和动态视觉理解能力。

**综合多模态理解** 包括 OpenCompass [74]、MMVet [75]、MMStar [76]、MME [77] 和 MMBench V1.1 [78] 等基准，评估跨多样化任务类型的通用视觉-语言理解能力。在 OpenCompass 平均分中，我们对 5 个基准（包括 MMStar、MMVet、HallusionBench、MathVista 和 MMMU）使用长推理模式。

#### 3.2 主要结果

如表 1 所示，MiniCPM-V 4.5 在广泛的视觉-语言能力上表现出色。

**综合能力。** MiniCPM-V 4.5 在 OpenCompass（涵盖 8 个流行基准的综合评估）上取得了 77.0 的平均分。仅凭 80 亿参数，它在视觉-语言能力上就超越了广泛使用的闭源模型（如 GPT-4o-latest）和强大的开源模型（如 Qwen2.5-VL 72B）。

**视频理解。** 该模型在高帧率和细粒度动作动态视频基准（如 MotionBench 和 FavorBench）上表现出色。它在长视频理解基准（如 VideoMME、LVBench、MLVU、LongVideoBench 等）上也展现出有竞争力的性能。

**OCR 和文档分析。** MiniCPM-V 4.5 在 OCRBench 上取得了领先的性能，超越了 GPT-4o-latest 等闭源模型。它在 OmniDocBench 的 PDF 文档解析能力上也在通用 MLLMs 中达到了最先进的性能。

**可信行为。** 由于 RLAIF-V 训练阶段专门增强了可信度水平，该模型在 ObjectHalBench 和 MMHal-Bench 等幻觉基准上胜过其他模型。

#### 3.3 推理效率

我们在 8 块 A100 GPU 的标准配置下，在图像理解和视频理解任务上评估了 MiniCPM-V 4.5 的推理效率。如表 2 所详细显示，我们的模型在达到有竞争力或更优性能的同时，相比其他领先模型显著减少了推理时间和 GPU 显存消耗。在 OpenCompass 上，MiniCPM-V 4.5 不仅在 30B 以下模型中取得了最高的平均分，而且仅用 GLM-4.1V 42.9% 的时间就完成了评估。这种效率得益于模型灵活的短推理和长推理模式。在 VideoMME 上，该模型展现出显著的效率提升。它以 73.6 的强劲性能，将推理时间减少了近 10×（从 2.63 小时降至 0.26 小时），并使用了最少的 28G 显存。这一改进主要归功于高效的 3D-Resampler，它联合考虑空间和时间维度压缩视频。

#### 3.4 消融实验

我们在本节对 MiniCPM-V 4.5 的关键设计选择进行消融。

**混合推理强化学习有助于提升整体性能和效率。** 我们评估混合 RL 策略，该策略在训练期间混合来自长推理和短推理模式的样本。为了进行清晰和公平的比较，我们从同一 SFT 检查点开始训练，并跳过 RLAIF-V 阶段。如表 3 所示，我们观察到：（1）混合策略取得了最佳的长推理性能，即使在评估时禁用长推理也优于 SFT 基线。这表明混合设置有效地激励了两种模式的能力。（2）此外，混合策略仅消耗纯长推理设置 70.5% 的训练 token 成本即可获得更好的性能。我们推测这是因为两种模式共享基础的感知和认知技能，长推理培养的分析深度可以增强短推理，而短推理学到的效率和直接性则可以完善长推理过程。

**基于概率的奖励对规则验证奖励形成互补。** 除了用于易验证回复的基于规则的奖励外，MiniCPM-V 4.5 进一步纳入了来自 RLPR [32] 的基于概率的奖励，为通用领域提供奖励信号。如图 3 所示，结合基于规则和基于概率的奖励（VR + PR）持续且显著地优于仅使用规则的方法，同时在回复长度和熵方面也产生了稳定的训练模式。这证实了基于概率的奖励为规则难以应对的通用推理数据提供了有意义的学习信号，有效地补充了适合规则验证的少量简单数据子集。随着训练步数增加，这种有效性尤为明显——跨多模态场景全谱的鲁棒奖励信号提供了纯基于规则的验证无法提供的必要训练指导。

**文档知识和文本识别的统一学习提升了两种能力。** 我们对所提出的统一学习范式进行消融实验。按照 § 2.2 的三阶段预训练过程，我们在 100 万高质量样本上训练模型，其中 20% 是知识密集型文档。然后我们在相同的 SFT 流水线后将其与基线方法进行比较。如表 4 所示，统一方法在知识密集型评估和文本识别任务上都优于基线。这些增益表明直接从文档图像中学习缓解了脆弱外部解析器引入的噪声。

**3D-Resampler 以更低的 token 成本实现更高性能。** 我们对 3D-Resampler 进行消融以验证其有效性。为了确保与 2D 基线的公平比较，我们在通用 SFT 阶段后微调模型检查点 300 步，将 Resampler 架构作为唯一变量。如表 5 所示，我们的 3D-Resampler 实现了更强的性能，同时每帧仅使用 2D 基线所需视觉 token 的三分之一。

### 4 结论

我们介绍了 MiniCPM-V 4.5，这是一个通过架构、数据和训练配方在训练和推理时都设计为高效的 MLLM。通过统一的 3D-Resampler，它在高帧率和长视频理解上以卓越的编码效率取得了强劲的性能。此外，文档知识和文本识别的统一学习范式使模型能够直接从文档图像中学习。这种方法绕过了脆弱的解析器，显著降低了数据工程的复杂性。最后，混合后训练策略在提高训练和推理效率的同时，也促进了短推理和长推理模式之间的泛化。总体而言，MiniCPM-V 4.5 展示了一条有希望的道路，可用于解决 MLLM 发展中的效率瓶颈。

### 附录 A：实现细节

预训练遵循 WSD 调度 [17]，稳定阶段的固定学习率为 5 × 10⁻⁵，衰减至 1 × 10⁻⁵。SFT 采用从 1 × 10⁻⁵ 到 1 × 10⁻⁶ 的余弦衰减。Long-CoT 和 3D-Resampler 阶段从 SFT 检查点继续，热身至 5 × 10⁻⁶，衰减至 1 × 10⁻⁶。

对于 RL 阶段，我们采用不带熵损失或 KL 惩罚的 GRPO [79]。每个批次包含 128 个提示，每个提示有 8 个回复，最大回复长度为 8192 token 以支持详细推理。Rollout 使用 1.0 的温度，50% 的提示被分配到长推理模式。我们在整个 RL 过程中使用 1 × 10⁻⁶ 的固定学习率。在 RLAIF-V [31] 阶段，我们使用 256 的全局批次大小、1 × 10⁻⁶ 的学习率和 β = 0.1，训练 400 步。

### 附录 B：定性案例

**B.1 综合指令遵循**

- **图 4：** 综合真实世界推理的案例。
- **图 5：** 中文综合真实世界推理的案例。
- **图 6：** 中文创意写作的案例。

**B.2 世界知识**

- **图 7：** 世界知识理解的案例。
- **图 8：** 中文世界知识理解的案例。

**B.3 OCR**

- **图 9：** 手写文本识别的案例。
- **图 10：** 中文手写文本识别的案例。
- **图 11：** 表格内容提取的案例。

**B.4 问题求解**

- **图 12：** 中文化学问题求解的案例。
- **图 13：** 多图统计问题求解的案例。

### 参考文献

完整参考文献列表（79 条）请参阅原始 PDF，涉及 GLM-V、Kimi-VL、MiMo-VL、GPT-4o、Qwen-VL / Qwen2.5-VL、MiniCPM-V、Ovis、InternVL3、Flash-VL、Gemini 2.0、DeepSeek-R1、LLaVA-UHD、Video-MME、MiniCPM、LAION-5B、COYO-700M、CLIP、CapsFusion、Common Crawl、OmniCorpus、MINT-1T、SynthText、Frozen-in-Time、Vript、OpenVid-1M、Wikipedia、OpenThoughts、RLAIF-V、RLPR、G-LLaVA、R-CoT、CLEVR-Math、INFOTABS、PromptPG、WikiTableQuestions、TabFact、TAT-QA、FigureQA、Multimodal-ArXiv、DVQA、VisualWebInstruct、Llama-Nemotron、MulDimIF、DeepSeekMath / GRPO、Skywork-VL Reward、DPO、MMMU、MathVista、AI2D、MathVerse、LogicVista、EMMA、OCRBench、ChartQA、TextVQA、DocVQA、OmniDocBench、HallusionBench、ObjHal、MMHal-Bench、Mantis、MMT-Bench、Grok-1.5V / RealWorldQA、MM-IFEngine、LVBench、MLVU、LongVideoBench、MotionBench、FAVOR-Bench、OpenCompass、MM-Vet、MMStar、MME、MMBench 等工作。

---

## English Original

### Title and Authors

**MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data and Training Recipes**

Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Yuanqian Zhao, Bokai Xu, Junbo Cui, Yingjing Xu, Liqing Ruan, Luoyuan Zhang, Hanyu Liu, Jingkun Tang, Hongyuan Liu, Qining Guo, Wenhao Hu, Bingxiang He, Jie Zhou, Jie Cai, Ji Qi, Zonghao Guo, Chi Chen, Guoyang Zeng, Yuxuan Li, Ganqu Cui, Ning Ding, Xu Han, Yuan Yao*, Zhiyuan Liu*, Maosong Sun*

MiniCPM-V Team, OpenBMB

Contact: yiranytianyu@gmail.com, yaoyuanthu@gmail.com

*Corresponding authors.

### Abstract

Multimodal Large Language Models (MLLMs) are undergoing rapid progress and represent the frontier of AI development. However, their training and inference efficiency have emerged as a core bottleneck in making MLLMs more accessible and scalable. To address the challenges, we present MiniCPM-V 4.5, an 8B parameter model designed for high efficiency and strong performance. We introduce three core improvements in model architecture, data strategy and training method: a unified 3D-Resampler model architecture for highly compact encoding over images and videos, a unified learning paradigm for document knowledge and text recognition without heavy data engineering, and a hybrid reinforcement learning strategy for proficiency in both short and long reasoning modes. Comprehensive experimental results in OpenCompass evaluation show that MiniCPM-V 4.5 surpasses widely used proprietary models such as GPT-4o-latest, and significantly larger open-source models such as Qwen2.5-VL 72B. Notably, the strong performance is achieved with remarkable efficiency. For example, on the widely adopted VideoMME benchmark, MiniCPM-V 4.5 achieves state-of-the-art performance among models under 30B size, using just 46.7% GPU memory cost and 8.7% inference time of Qwen2.5-VL 7B.

### 1 Introduction

Multimodal Large Language Models (MLLMs) [1, 2, 3, 4, 5, 6, 7] are advancing rapidly the frontier of artificial intelligence, enabling machines to understand and reason over different modalities such as text and images. However, as MLLMs evolve, the cost of data engineering, training and inference also increases heavily. Addressing this efficiency challenge is now a central focus of both research and industry [6, 8, 9, 10, 11], essential for making capable MLLMs more accessible and scalable.

We decompose this efficiency problem into three core aspects: **(1) Model Architecture.** A primary efficiency bottleneck in MLLMs comes from the large number of visual tokens for high-resolution image encoding, which brings heavy computation overhead for visual encoders and LLM decoders. The problem is even exacerbated in video understanding, where existing models can take thousands of tokens to encode a short and low-resolution video, even when sampling at a low frame rate. For example, processing a 6-second, 2-fps video at a resolution of just 448×448 requires 1,536 tokens for Qwen2.5-VL [7], and 3,072 tokens for InternVL3 [9]. Such long visual token sequences lead to prohibitive training and inference costs in GPU memory and computation speed. **(2) Training Data.** As we quickly run out of new knowledge from traditional web page data, a new cornerstone of modern MLLMs is harnessing high-quality multimodal knowledge from documents [1, 2], such as scientific papers and textbooks. These documents are often stored as PDFs, containing multi-disciplinary knowledge in various domains and organized in diverse layouts of interleaved texts, images and tables. However, most methods depend on brittle external parsing tools to convert document files into interleaved image-text sequences for training. These tools often fail in complex layouts, leading to either errors in knowledge learning or heavy data engineering efforts to fix failure cases. **(3) Training Methods.** Reinforcement Learning (RL) has shown promise in improving complex reasoning capabilities by enabling a step-by-step explicit thinking process before providing the final answer [12, 1]. However, this performance gain often comes at the expense of extreme verbosity. Even for simple tasks such as identifying obvious objects, most existing thinking models produce excessively long outputs, inducing poor efficiency in both training and inference.

To address the challenges, MiniCPM-V 4.5 introduces three key improvements in model architecture, data strategy and training method: **(1) Unified 3D-Resampler for Compact Image and Video Encoding.** Previous MiniCPM-V series models [6] exhibit high compression rates (e.g., 4× compared with most MLLMs) for high-resolution images via 2D-Resamplers [5, 13]. To further address the architectural inefficiency of video processing, we extend the 2D-Resampler to a 3D-Resampler that jointly compresses spatial-temporal information for videos. This module can encode a 6-second, 2-fps, 448×448 resolution video into only 128 visual tokens, achieving a 12×-24× reduction in token cost compared to representative MLLMs [7, 9], enabling efficient high-frame-rate and long video understanding, and unified encoding for images as well. **(2) Unified Learning Paradigm for Document Knowledge and OCR.** We propose a learning paradigm that enables the model to accurately acquire knowledge directly from document images, eliminating the need for fragile external parsers. By dynamically corrupting text regions in documents with varying noise levels and asking the model to reconstruct the text, the model learns to adaptively and properly switch between accurate text recognition (when text is roughly visible) and multimodal context-based knowledge reasoning (when text is heavily corrupted). **(3) Hybrid Strategy for Post-Training.** Unlike prior models that optimize for a single long reasoning mode [2, 1], we develop a hybrid RL post-training strategy to support both short reasoning mode for efficient usage and long reasoning mode for complex tasks. In RL training, we randomly alternate between two modes during the rollout process for joint optimization. This approach not only enables flexible control over the short and long reasoning modes but also allows for mutual performance enhancement. In experiments, we can achieve better reasoning performance with fewer training samples for both modes.

Comprehensive experimental results in OpenCompass evaluation show that MiniCPM-V 4.5 outperforms widely used proprietary models such as GPT-4o-latest [4], and significantly larger open-source models such as Qwen2.5-VL 72B [7]. Notably, the strong performance is achieved with remarkable efficiency. For example, powered by the efficient unified 3D-Resampler, MiniCPM-V 4.5 achieves equivalent performance on VideoMME [14] using only 9.9% of the inference time of prior state-of-the-art MLLMs [1]. Based on the hybrid post-training strategy, MiniCPM-V 4.5 excels in both short and long reasoning modes, outperforming concurrent thinking models [3, 1] on OpenCompass evaluation while using only 42.9%-68.2% inference time.

In summary, our contributions are as follows:

- We open-source MiniCPM-V 4.5, an efficient and strong MLLM that supports efficient high-frame-rate and long video understanding, robust OCR and strong document parsing capabilities, and controllable hybrid reasoning.
- We introduce three key improvements: a unified 3D-Resampler for efficient image and video encoding, a unified paradigm for document knowledge and OCR learning, and a hybrid strategy for post-training that enhances both performance and efficiency.
- Comprehensive experiments demonstrate the effectiveness of the proposed technical improvements and the performance of MiniCPM-V 4.5.

**Figure 1:** An overview of the MiniCPM-V 4.5 architecture. The model processes diverse visual inputs, such as high-resolution images and high-frame-rate videos. After the image partitioning and video packing processes, these inputs are encoded by a visual encoder and then fed into the unified 3D-Resampler. This module efficiently compresses both image and video features into a compact token sequence (achieving up to 16× compression rate for images and an additional 6× for videos), which is then processed by the LLM decoder. The decoder can generate responses in two distinct styles: a concise, short reasoning mode or a step-by-step, long reasoning mode.

### 2 Approach

In this section, we describe the methodology of MiniCPM-V 4.5, including the model architecture and the recipes for pre-training, SFT and RL.

#### 2.1 Architecture

As shown in Figure 1, the architecture of MiniCPM-V 4.5 comprises three main modules: (1) A lightweight visual encoder that flexibly handles high-resolution images with a special partitioning strategy. (2) A unified 3D-Resampler that encodes images and videos into compact features, exploiting spatial-temporal redundancies in visual information. (3) An LLM decoder that understands images, videos, text, and generates text outputs.

##### 2.1.1 The Unified 3D-Resampler

To tackle the image and video encoding efficiency bottleneck in MLLMs [15, 16], we extend the 2D-Resampler to a 3D-Resampler that jointly incorporate spatial-temporal information for compression. In this way, we achieve an additional 6× temporal compression rate compared to the 2D-Resampler, by leveraging the temporal redundancy of consecutive video frames.

**Image Processing.** To handle high-resolution images in any aspect ratio, we adopt the LLaVA-UHD [13] image partitioning strategy. For each image, we estimate the ideal number of slices from the input resolution and choose the partition whose per-slice resolution deviates least from the visual encoder pretraining setting. We then use learnable queries augmented with 2D spatial positional embeddings to produce a fixed-length sequence for each slice through cross-attention. Most existing MLLMs [7, 9, 1] adopt MLP and pixel unshuffle operation for visual compression, and typically require visual 256 tokens for encoding a 448×448 image. Leveraging the flexibility of resampler architecture, by choosing a small number of query tokens, MiniCPM-V can achieve a significantly higher compression rate for visual tokens (e.g., 64 tokens for a 448×448 image) while maintaining good performance.

**Video Processing.** To handle the significant redundancy in video data, we employ a joint spatial-temporal compression strategy for higher compression rates. For each video, we first split it into packages along the temporal dimension, where each package contains adjacent frames. Intuitively, video frames within the same package typically share highly redundant visual information, which can be identified and compressed when jointly modeled. To this end, we resample the frame features from the visual encoder in each package into a fixed-length feature sequence through cross-attention. We augment the learnable queries with both 2D spatial positional embedding, as used in image encoding, and temporal positional embedding. The final video representation is obtained by concatenating the token sequences from all packages. We sample at most 1080 frames per video at a maximum frame rate of 10. During training, the package size and frame rate are randomly augmented to improve robustness. This design also provides flexibility at inference time, allowing these hyperparameters to be adjusted to meet the demands of diverse scenarios and devices.

Based on the 3D-Resampler, MiniCPM-V 4.5 can achieve 96× compression rate for video tokens, where 6 448×448 video frames² can be jointly compressed into 64 video tokens (normally 1,536-3,072 tokens for most MLLMs). This means that the model can perceive significantly more video frames without increasing the LLM inference cost, which brings strong high-frame-rate video understanding and long video understanding capabilities.

> ² Each frame costs 1024 token from the visual encoder.

**Training Efficiency.** Thanks to the flexibility of the resampler mechanism (agnostic to input shape), we can use the same 3D-Resampler for unified visual encoding over images and videos. This means that image and visual encoding share the same architecture and weights, and therefore, we can achieve the extension from 2D-Resampler to 3D-Resampler efficiently via a lightweight SFT stage. Moreover, this also facilitates efficient knowledge transfer from images to videos. For example, we observe reasonable video OCR capability in MiniCPM-V 4.5, although we did not specifically collect such training data.

> **Takeaway**
> 1. Joint spatial–temporal compression can enable higher visual compression rates.
> 2. A unified architecture can be more efficiently adapted with minimal additional training and facilitates knowledge transfer from images to videos.

#### 2.2 Pre-training

Our pre-training process aims to systematically build the model's foundational capabilities through a progressive, multi-stage strategy. This involves a carefully curated data composition and a novel unified paradigm for document knowledge and OCR learning.

##### 2.2.1 Pre-training Strategy

The pre-training comprises three progressive stages. Each stage strategically unfreezes different model components and introduces increasingly complex data to optimize learning efficiency.

**Stage 1.** We begin with a warm-up stage, training only the resampler module while all other components remain frozen. This stage uses image-caption data to establish an initial alignment between visual and language modalities with minimal training cost.

**Stage 2.** We then unfreeze the vision encoder to enhance the perceptual foundation capability. This stage consumes OCR-rich data and image-caption data. Since the data in this stage may lack the fluency or quality required for language modeling, the LLM decoder remains frozen in this stage.

**Stage 3.** With the cross-modal bridge in place and the perceptual foundation set, the final stage trains all model parameters end-to-end using our highest quality data, including text-only corpora, image-text interleaved samples, videos and a curated subset from earlier stages. At this point, we unfreeze the LLM decoder to fully exploit the knowledge and skills in data, encompassing multi-image reasoning and temporal understanding. We adopt the Warmup-Stable-Decay learning rate scheduler [17]. During the decay phase, we gradually add more high-quality instruction data and knowledge-intensive data.

**Figure 2:** Unified paradigm for document knowledge and OCR learning via dynamic visual corruption. We create a spectrum of training tasks through varied corruption levels: low corruption preserves readability to learn robust OCR, high corruption forces the model to perform contextual inference, and moderate corruption requires integrated inference from visual clues and context.

##### 2.2.2 Pre-training Data

**Image Caption Data.** We combine large-scale public datasets (LAION-2B [18], COYO [19], etc.) with curated Chinese image-text pairs crawled from the web. We filter out low-resolution images and remove irrelevant image-text pairs with CLIP [20]. To enrich alt-text descriptions, we employ a Capsfusion-based [21] re-captioning process on a subset to generate fluent and factually complete captions. In this way, we formulate the valuable world knowledge in raw captions into more fluent natural language. We employ an MLLM to tag images with concept labels and ensure a balanced distribution across languages and long-tail concepts.

**Image-Text Interleaved Data.** Sourced from Common Crawl [22], OmniCorpus [23] and MINT-1T [24], image-text interleaved data is crucial for in-context learning and multi-image understanding capabilities. We apply filtering to ensure quality, removing samples with broken images or imbalanced image-text ratios. We further use relevance filtering to ensure meaningful multimodal associations, and employ knowledge density filtering to select a high-quality subset for the final decay phase of pre-training.

**OCR Data.** We synthesize OCR data to enhance the basic text recognition capability during the early pre-training stage. We render text on natural scenes with various combinations of color and font following [25], and also render real-world HTML sources into images.

**Document Data.** We collect documents, including scientific papers, academic reports, textbooks, etc., from the web. This data exhibits high knowledge density and contains visually complex layouts.

**Video Caption Data.** We aggregate several public datasets [26, 27, 28], and supplement them with more detailed in-house video captions. This diverse collection supports the development of temporal visual reasoning capabilities essential for video comprehension.

##### 2.2.3 Unified Paradigm for Document Knowledge and OCR Learning

Documents, such as scientific papers and textbooks, are vital resources for learning diverse layouts and acquiring multi-disciplinary knowledge in various domains. However, most MLLMs depend on brittle external parsers to convert document PDFs into an interleaved image-text sequence for training. Such a noisy and inefficient process often introduces structural errors or requires heavy data engineering efforts to fix failure cases.

Another challenge for OCR learning is that, while stronger image augmentation can create more diverse and harder samples need for robust OCR, over-augmentation can make the texts indistinguishable. Forcing the model to produce the ground truth text from such indistinguishable visual input typically leads to hallucination problems. Therefore, previously, we could only afford a small and safe level of augmentation.

To overcome both challenges, we propose a unified training paradigm that learns directly from document images, using their original text as ground truth. Our key insight is that the most important difference between document knowledge acquisition and text recognition is the visibility of the text in images. We unify both capabilities into a single learning objective: predicting original text from corrupted document images. By dynamically corrupting text regions with varying corruption levels, the model learns to adaptively and properly switch between precise text recognition (when text is distinguishable) and multimodal context-based knowledge reasoning (when text is heavily obscured or masked), as illustrated in Figure 2. This eliminates reliance on fragile parsers and prevents hallucinations from over-augmented OCR data.

Specifically, for each document, we treat a subset of its text regions as training ground truth. We then stochastically apply different levels of corruption to each region, creating different training tasks:

1. **Low Corruption (Augmented OCR).** When mild noise is applied to a text region, the texts are still recognizable, and the model could effectively predict them via text recognition.
2. **Moderate Corruption (Integrated Inference).** When heavy noise is applied to the text region, individual characters become highly ambiguous and unreliable for recognition. The model must therefore learn to integrate the noisy visual cues from the corrupted region with the high-level document context and its internal knowledge to reconstruct the original text.
3. **High Corruption (Contextual Inference and Document Knowledge Learning).** With the text region completely masked out, the model cannot rely on character-level cues to predict the missing content. Consequently, the model is forced to infer the information only from the multimodal context and its internal knowledge. This directly cultivates document-level understanding.

This unified approach yields a more efficient and resilient learning process. By learning directly from the document image, we avoid building complex document parsing pipelines and prevent potential noise introduced by fragile parsers. Furthermore, this paradigm allows us to fluidly combine knowledge learning and OCR objectives within the same training batch, maximizing data utility and producing a single, versatile model adept at a wide range of document understanding tasks.

> **Takeaway**
> 1. Foundation skills can be built on imperfect heterogeneous data sources by selectively freezing parameters.
> 2. Simple dynamic visual corruption on document image text can effectively unify knowledge learning, robust OCR and contextual inference into a single learning objective.

#### 2.3 Supervised Fine-tuning

The Supervised Fine-Tuning (SFT) stage aims to activate the model's capability on a broad range of tasks and prepares for reinforcement learning. Moreover, we extend the 2D-Resampler to a unified 3D-Resampler at this stage to enhance the compression efficiency of video data.

##### 2.3.1 Supervised Fine-tuning Strategy

We first train the general interaction abilities, and then cultivate specialized skills for advanced reasoning and temporal understanding.

**Stage 1: General SFT.** This stage aims to activate the broad knowledge acquired during pre-training and align it with human instructions. By fine-tuning on a diverse mixture of high-quality instruction-response data, the model develops proficiency in multimodal interaction. To prevent degradation of text-only performance and improve training stability, we include 10% high-quality text-only data in the training mixture.

**Stage 2: Long-CoT & 3D-Resampler.** Building on versatile foundations from the previous stage, we then cultivate specialized skills to support long reasoning mode, high-frame-rate and long video understanding. First, we introduce Long-CoT warm-up instructions into the SFT data. This encourages the model to perform an explicit step-by-step thinking process, incorporating cognitive patterns such as reflection and backtracking, which are vital for the long reasoning mode. Second, we enhance its temporal understanding by upgrading the architecture from 2D to 3D-Resampler and introducing high-frame-rate and long video data. Due to the unified design, we find that such an upgrade can be achieved efficiently with a small amount of high-quality video data.

##### 2.3.2 Supervised Fine-tuning Data

**STEM Data.** To enhance STEM reasoning, we curate a dataset of high-school and undergraduate level multidisciplinary problems from the web, covering physics, chemistry, biology, finance, computer science, etc. To ensure the data quality, we implement a two-stage filtering process. First, we only keep samples that exhibit high visual dependency (i.e., not solvable without image information). Second, we perform a consistency check to validate the correctness of answers. For each remaining sample, we collect a clean reasoning process via rejection sampling with a powerful MLLM.

**Long-tail Knowledge Data.** To address the long-tail problem where models often fail on less common topics, we incorporate long-tail knowledge from Wikipedia [29] to synthesize high-quality multimodal instruction-following data. Specifically, for each entity page, we construct multimodal instructions and answers using strong MLLMs and keep samples with high visual dependency.

**Long-CoT Data.** Long-CoT data enables the model to acquire the necessary reasoning patterns for the long reasoning mode. Our data comes from OpenThoughts [30] and an in-house pipeline. We identify challenging prompts by filtering for those on which our early-stage models struggle. Our pilot studies show that focusing on challenging problems is the key to developing robust reasoning capabilities rather than memorizing trivial patterns. Each response then undergoes a multistage validation: we verify its correctness, assess trustworthiness with claim-level factual verification using RLAIF-V [31], and filter out meaningless repetition. Finally, validated responses are augmented through rewriting to enhance diversity.

> **Takeaway**
> Filtering out easy prompts and focusing on challenging problems is crucial for effective Long-CoT warm-up.

#### 2.4 Reinforcement Learning

The RL stage aims to enhance reasoning performance, enable controllable reasoning modes, and improve trustworthiness. To provide efficient general-domain rewards, we combine rule-verified rewards for straightforward cases with general probability-based rewards from RLPR [32] for complex answers and add a calibrated preference reward. A hybrid RL strategy is adopted to allow flexible switch between short and long reasoning modes. We further integrate RLAIF-V [31] to reduce hallucinations.

##### 2.4.1 Reinforcement Learning Data

Our RL data contains high-quality samples that span four key domains. Each subset underwent a rigorous, human-in-the-loop cleaning and deduplication process.

**Mathematics.** We collect multimodal math problems, which require the integration of visual perception and logical reasoning, from academic sources [33, 34, 35]. We observe that many open-source datasets contain severe label errors and adopt a thorough cleaning process to clean them.

**Documents, Tables and Charts.** To improve reasoning on perceptually complex scenarios, we curate a diverse mix of real-world datasets [36, 37, 38, 39, 40] and synthetic datasets [41, 42, 43] to improve the coverage of domains.

**General Reasoning.** To further improve general reasoning capabilities, we assemble a diverse collection of problems covering logical and multi-disciplinary reasoning tasks from VisualWebInstruct [44] and additional web resources. These data exhibit a more complex reference answer style, and many of the problems have more than one sub-question.

**Instruct Following.** We incorporate text-only instructions from the Llama-Nemotron-Post-Training Dataset [45] and the MulDimIF dataset [46]. We observe that the textual instruction-following improvement generalizes well to multimodal instructions.

##### 2.4.2 Reward Quality Control

The efficacy of RL is highly dependent on data quality. Thus, we implement meticulous quality control processing, focusing on three distinct aspects:

**Label Accuracy.** Incorrect labels can introduce flawed supervision signals. For each dataset, we maintain a small subset to inspect the label accuracy and conduct a human-in-the-loop cleaning process to keep a high label accuracy.

**Rewarding Accuracy.** Verifying model-generated responses in general domains is a nontrivial challenge. Hand-crafted rules struggle to tackle the complexity of natural language. To address this, we dynamically apply the most suitable validation method for each case. For straightforward answers containing only a few tokens, we employ a rule-based verification system, achieving 98% reward accuracy. For complex natural language answers where rules are brittle (e.g., those containing specific units or longer phrasing), we use the more general probability-based rewards of RLPR [32].

**Rewarding Coverage.** To complement these accuracy-focused signals, we integrate a reward model to provide a dense preference-aligned signal that guides the model towards higher-quality human-like responses. We apply the reward model to only the final answer part for the long reasoning mode to avoid the out-of-distribution problem.

##### 2.4.3 Hybrid Reinforcement Learning

We adopt a controllable hybrid reasoning design for our model: a short reasoning mode for quick answers and a long reasoning mode that generates explicit step-by-step thinking traces for complex problems. Mode switching is controlled by prompts. Both behaviors are initialized during SFT and then optimized jointly via hybrid RL, where rollouts randomly alternate between the two modes. We apply GRPO [47] to optimize the model with these rollouts and remove the KL and entropy loss to improve stability. This training schedule not only preserves the efficiency of short responses while retaining complex reasoning capabilities, but also fosters cross-generalization, where reasoning capabilities learned in one mode can transfer to improve the other mode. Based on this hybrid post-training design, MiniCPM-V 4.5 consumes only 70.5% of the training token costs of long reasoning only strategy to achieve better performance.

##### 2.4.4 Reward Shaping

We design the reward shaping strategy to balance task capability, human preference and training stability. The final reward signal is a weighted composite of four components: an accuracy reward $R_{acc}$, a format reward $R_{format}$, a repetition penalty reward $R_{rep}$ and a preference reward $R_{rm}$. The preference reward is derived from an auxiliary RM trained with human preference data [48]. However, directly applying RMs in the long reasoning mode yields unsatisfactory results since standard RMs struggle to evaluate the out-of-distribution long reasoning chains, leading to worse alignment and training instability, which is also confirmed in our preliminary experiments.

To address this, we adopt a selective application strategy. The RM scores only the final answer part of the response, completely bypassing the explicit thinking steps. This provides a stable, dense reward signal that aligns with human preferences without incorrectly penalizing complex reasoning paths. The final reward is calculated as follows.

$$R = R_{acc} + R_{format} + R_{rep} + \frac{1}{2} \tilde{R}_{rm}. \quad (1)$$

Here, $\tilde{R}_{rm}$ is the standardized preference reward score computed using $\frac{R_{rm} - \bar{R}_{rm}}{\sigma(R_{rm})}$, where $\bar{R}_{rm}$ and $\sigma(R_{rm})$ represent the average and standard deviation of raw reward scores of responses sampled with the same prompt.

##### 2.4.5 RLAIF-V

Visual hallucinations remain a critical limitation for MLLMs, particularly in applications requiring high reliability. To address this challenge, we integrate RLAIF-V [31] to make the responses more factually grounded to the visual input through alignment from scalable AI feedback. Notably, we extend this approach to video inputs, where hallucination problems are especially pronounced.

**Response Sampling.** We first sample multiple responses from the policy model under the same generation condition. This strategy ensures focused evaluation of factual accuracy, avoiding distributional mismatches between responses.

**Feedback Collection.** We begin by decomposing complex responses into verifiable atomic claims, where each claim is independently validated. This transforms the complex long response evaluation into simpler claim-level verification, addressing the inherent challenge of holistic assessment and improving the precision of factual evaluation. Preference pairs are then constructed based on aggregated claim verification scores, where responses containing fewer factual errors are preferred.

**Preference Learning.** The resulting preference dataset, encompassing both image and video, is used to train the model with DPO [49]. This stage proves particularly effective for visual tasks where factual accuracy is paramount, without compromising general performance.

> **Takeaway**
> 1. Combining rule-based reward for simple responses and probability-based reward for complex natural language responses enables a reliable reward system for diverse tasks.
> 2. Hybrid RL enables cross-mode generalization between long and short reasoning modes.

### 3 Experiments

In this section, we empirically evaluate the performance of MiniCPM-V 4.5, and the effectiveness of the proposed methods.

#### 3.1 Baselines and Benchmarks

We compare with various strong baseline models: (1) state-of-the-art open-source models, represented by Qwen2.5-VL 72B [7]; (2) strong models of comparable sizes, including InternVL3 [9] (8B) and GLM-4.1V [1] (9B); and (3) frontier proprietary models such as GPT-4o-latest [4].

Our evaluation encompasses several key areas of multimodal capabilities:

**STEM** includes mathematics and science-oriented benchmarks such as MMMU [50], MathVista [51], AI2D [52], MathVerse [53], LogicVista [54] and EMMA [55], designed to evaluate logical reasoning, mathematical problem-solving and scientific understanding capabilities.

**Document, OCR & Chart** covers OCR-related tasks through OCRBench [56], ChartQA [57], TextVQA [58], DocVQA [59], and OmniDocBench [60], testing ability to extract, interpret and reason about textual information in various visual contexts, including documents and charts.

**Hallucination** evaluates model reliability through HallusionBench [61], ObjHalBench [62] and MMHal-Bench [63], measuring the tendency to generate false or inconsistent information.

**Multi-Image & Real-World & Instruction Following** includes Mantis [64], MMT-Bench [65], RealWorldQA [66] and MM-IFEval [67], assessing performance on complex scenarios involving multiple images, real-world understanding and instruction following.

**Video Understanding** encompasses Video-MME [68], LVBench [69], MLVU [70], LongVideoBench [71], MotionBench [72] and FavorBench [73], evaluating temporal reasoning and dynamic visual comprehension across various video tasks.

**Comprehensive Multimodal Understanding** includes benchmarks such as OpenCompass [74], MMVet [75], MMStar [76], MME [77] and MMBench V1.1 [78], which assess general vision-language comprehension across diverse task types. Within the OpenCompass average, we use the long reasoning mode for 5 benchmarks, including MMStar, MMVet, HallusionBench, MathVista and MMMU.

**Table 1:** Evaluation results across diverse vision-language benchmarks. The best performance is marked in bold. * We evaluate officially released checkpoints by ourselves. † Reasoning mode used, where the average score of three runs is reported for robust evaluation. ‡ GPT-4o-latest evaluation results from OpenCompass. Otherwise GPT-4o-1120 is used in evaluation, since GPT-4o-latest is only accessible via Web API.

| Task | Benchmark | MiniCPM-V 4.5 | Qwen2.5-VL | Qwen2.5-VL | InternVL3 | GLM-4.1V | GPT-4o |
|------|-----------|---------------|------------|------------|-----------|----------|--------|
| **Size** | | 8B | 7B | 72B | 8B | 9B | - |
| **Mode** | | hybrid | non-thinking | non-thinking | non-thinking | thinking | non-thinking |
| **Comprehensive Multimodal** | OpenCompass | 77.0† | 70.5 | 76.1 | 73.6 | 76.6 | 75.4‡ |
| | MMVet | 75.5† | 67.1 | 76.9 | 81.3 | 70.5† | 76.9‡ |
| | MMStar | 72.1† | 63.9 | 70.5 | 68.2 | 72.9 | 70.2‡ |
| | MME | 2500 | 2347 | 2483 | 2415 | 2466† | 2318* |
| | MMBench V1.1 | 84.2† | 82.6 | 87.8 | 81.7 | 85.3 | 86.0‡ |
| **STEM** | MMMU | 67.7† | 58.6 | 68.2 | 62.7 | 68.0 | 72.9‡ |
| | MathVista | 79.9† | 68.2 | 74.2 | 71.6 | 80.7 | 71.6‡ |
| | AI2D | 86.5 | 83.9 | 88.5 | 85.2 | 87.9 | 86.3‡ |
| | MathVerse MINI | 58.8† | 49.2 | 47.3 | 39.8 | 68.4 | 40.6 |
| | LogicVista | 57.0† | 44.1 | 55.7 | 44.1 | 60.4 | 52.8 |
| | EMMA | 34.8† | 28.6* | - | - | 35.7† | 32.4 |
| **Document, OCR & Chart** | OCRBench | 89.0 | 86.4 | 88.2 | 88.0 | 84.2 | 82.2‡ |
| | ChartQA | 87.4 | 87.3 | 89.5 | 86.6 | 87.1† | 86.7 |
| | TextVQA | 82.2 | 84.9 | 83.5 | 80.2 | 79.9† | 85.6* |
| | DocVQA | 94.7† | 95.7 | 96.4 | 92.7 | 93.4† | 93.0 |
| | OmniDocBench (EN) ↓ | 0.175 | 0.316 | 0.214 | 0.335* | 0.460* | 0.233 |
| | OmniDocBench (ZH) ↓ | 0.253 | 0.399 | 0.261 | 0.390* | 0.573* | 0.399 |
| **Hallucination** | HallusionBench | 61.2† | 52.9 | 54.6 | 49.9 | 63.2 | 57.0‡ |
| | ObjHalBench (CHAIRs) ↓ | 9.3† | 13.7* | 17.0* | 11.3* | 12.3* | - |
| | ObjHalBench (CHAIRi) ↓ | 5.2† | 7.7* | 8.9* | 6.5* | 6.4* | - |
| | MMHal-Bench (Score) | 5.0† | 4.1* | 4.2* | 4.2* | 4.6* | - |
| | MMHal-Bench (Rate) ↓ | 19.4† | 31.6* | 38.2* | 24.3* | 22.9* | - |
| **Multi-Image & Real World & Instruction Following** | Mantis | 82.5† | 74.7* | 81.1* | 70.1 | 78.8† | - |
| | MMT-Bench | 68.3 | 63.6 | - | 65.0 | 67.6 | 66.7* |
| | RealWorldQA | 72.1† | 68.5 | 75.7 | 70.8 | 70.7† | 76.8* |
| | MM-IFEval | 66.0 | 51.3* | 73.8* | 53.2* | 58.4† | 64.6 |
| **Video Understanding** | Video-MME (w/o subs) | 67.9 | 65.1 | 73.3 | 66.3 | 68.2 | 71.9 |
| | Video-MME (w/ subs) | 73.5 | 71.6 | 79.1 | 68.9 | 73.6 | 77.2 |
| | LVBench | 50.4 | 45.3 | 47.3 | 44.1* | 44.0 | 48.9 |
| | MLVU (M-Avg) | 75.1 | 70.2 | 74.6 | 71.4 | 72.5† | - |
| | LongVideoBench (val) | 63.9 | 56.0 | 60.7 | 58.8 | 65.7 | - |
| | MotionBench | 59.7 | 53.0 | 58.3 | 58.1 | 59.0 | 58.0 |
| | FavorBench | 56.0 | 42.3 | 48.1 | 45.3 | 51.2† | - |

#### 3.2 Main Results

As shown in Table 1, MiniCPM-V 4.5 demonstrates strong performance across a wide range of vision-language capabilities.

**Comprehensive Capability.** MiniCPM-V 4.5 achieves an average score of 77.0 on OpenCompass, a comprehensive evaluation of 8 popular benchmarks. With only 8B parameters, it surpasses widely used proprietary models like GPT-4o-latest and strong open-source models like Qwen2.5-VL 72B for vision-language capabilities.

**Video Understanding.** The model achieves strong performance on high-frame-rate and fine-grained action dynamics video benchmarks such as MotionBench and FlavorBench. It also shows competitive performance on long video understanding benchmarks such as VideoMME, LVBench, MLVLU, LongVideoBench, etc.

**OCR and Document Analysis.** MiniCPM-V 4.5 achieves leading performance on OCRBench, surpassing proprietary models such as GPT-4o-latest. It also achieves state-of-the-art performance for PDF document parsing capability on OmniDocBench among general MLLMs.

**Trustworthy Behavior.** The model outperforms other models on hallucination benchmarks, including ObjectHalBench and MMHal-Bench, since the RLAIF-V training stage specifically enhances the level of trustworthiness.

#### 3.3 Inference Efficiency

We evaluated the inference efficiency of MiniCPM-V 4.5 in a standard configuration of 8 A100 GPUs on both image understanding and video understanding tasks. As detailed in Table 2, our model achieves competitive or superior performance while significantly reducing inference time and GPU memory consumption compared to other leading models. On OpenCompass, MiniCPM-V 4.5 not only achieves the highest average score among models under 30B, but also finishes the evaluation using 42.9% of the time of GLM-4.1V. This efficiency is enabled by the model's flexible short and long reasoning modes. On VideoMME, the model demonstrates remarkable efficiency gains. With a strong performance of 73.6, it also reduces the inference time by nearly 10× (from 2.63h to 0.26h) and uses the least memory of 28G. This improvement is primarily due to the efficient 3D-Resampler, which compresses videos jointly considering spatial and temporal dimensions.

**Table 2:** Inference efficiency on 8 A100 GPUs. Best results are marked in bold.

**(a) OpenCompass results of thinking models**

| Model | Size | Avg Score ↑ | Time ↓ |
|-------|------|-------------|--------|
| GLM-4.1V-9B-thinking | 10.3B | 76.6 | 17.5h |
| MiMo-VL-7B-RL | 8.3B | 76.4 | 11.0h |
| **MiniCPM-V 4.5** | 8.7B | **77.0** | **7.5h** |

**(b) Video-MME results**

| Model | Size | Score ↑ | Time ↓ | Mem ↓ |
|-------|------|---------|--------|-------|
| Qwen2.5-VL-7B | 8.3B | 71.6 | 3.00h | 60G |
| GLM-4.1V-9B-thinking | 10.3B | 73.6 | 2.63h | 32G |
| **MiniCPM-V 4.5** | 8.7B | 73.5 | **0.26h** | **28G** |

#### 3.4 Ablations

We ablate key design choices of MiniCPM-V 4.5 in this section.

**Hybrid reasoning reinforcement learning helps improve overall performance and efficiency.** We evaluate the hybrid RL strategy that mixes samples from both long and short reasoning modes during training. For clear and fair comparison, we train from the same SFT checkpoints and skip the RLAIF-V stage. As shown in Table 3, we observe that: (1) The hybrid strategy achieves the best long reasoning performance, and outperforms the SFT baseline even when long reasoning disabled at evaluation. This demonstrates that the hybrid setup effectively incentivizes capabilities of both modes. (2) Moreover, the hybrid strategy consumes only 70.5% of the training token costs of the pure long reasoning setting to achieve better performance. We hypothesize that this is because both modes share foundational perceptual and cognitive skills, and the analytical depth cultivated by long reasoning could bolster short reasoning, while the efficiency and directness learned from short reasoning refine the long reasoning process.

**Table 3:** Ablation of hybrid reinforcement learning. We report RL training token cost and performance on OpenCompass.

| Training | Evaluate with Long Reasoning | OpenCompass | RL Training Tokens |
|----------|------------------------------|-------------|---------------------|
| SFT Model | ✓ | 73.6 | - |
| Long Reasoning Only | ✓ | 77.0 | 4.4B |
| Hybrid Reasoning | ✗ | 74.9 | 3.1B |
| Hybrid Reasoning | ✓ | 77.1 | 3.1B |

**Probability-based reward complements rule-verification reward.** In addition to rule-based reward for easy-to-verify responses, MiniCPM-V 4.5 further incorporates the probability-based reward from RLPR [32] to supply reward signals for general domains. As shown in Figure 3, combining both rule-based and probability-based rewards (VR + PR) consistently and substantially outperforms the rule-only approach, while also yielding stable training patterns with respect to response length and entropy. This confirms that probability-based reward provides a meaningful learning signal for the general reasoning data that rules struggle with, effectively complementing the limited subset of simple data suitable for rule verification. The effectiveness becomes particularly evident as the training steps scale where the robust reward signals across the full spectrum of multimodal scenarios provide essential training guidance that pure rule-based verification cannot deliver.

**Figure 3:** Ablation results of adding probability-based reward. We report OpenCompass scores, response length and entropy on different training steps.

**Unified learning of document knowledge and text recognition improves both capabilities.** We run an ablation experiment for the proposed unified learning paradigm. Following the three stages pre-training process in § 2.2, we train the model on 1M high-quality samples, 20% of which are knowledge-intensive documents. Then we conduct a comparison against the baseline method after the same SFT pipeline. As shown in Table 4, the unified approach outperforms the baseline on both knowledge-intensive evaluations and text-recognition tasks. These gains indicate that learning directly from document images mitigates the noise introduced by fragile external parsers.

**Table 4:** Ablation of unified learning paradigm for document knowledge and text recognition. We report results on knowledge-intensive, document understanding, and text recognition benchmarks.

| Method | MMMU | AI2D | OCRBench |
|--------|------|------|----------|
| External Parser | 49.0 | 74.9 | 576 |
| Unified Learning | **51.4** | **76.5** | **617** |

**3D-Resampler enables higher performance with lower token cost.** We ablate the 3D-Resampler to verify its effectiveness. To ensure a fair comparison against the 2D baseline, we fine-tuned the model ckpt after the general SFT stage for 300 steps, isolating the resampler architecture as the only variable. As demonstrated in Table 5, our 3D-Resampler achieves stronger performance, while using only one-third of the visual tokens per frame required by the 2D baseline.

**Table 5:** Ablation of the 3D-Resampler. We report scores on VideoMME. w/ sub: using subtitles during evaluation; w/o sub: remove subtitles during evaluation.

| Method | w/ sub | w/o sub | tokens/frame |
|--------|--------|---------|--------------|
| 2D-Resampler | 65.5 | 71.5 | 64.0 |
| 3D-Resampler | **67.3** | **72.5** | **21.3** |

### 4 Conclusion

We introduce MiniCPM-V 4.5, an MLLM designed with high efficiency at both training and inference time via architecture, data and training recipe. With a unified 3D-Resampler, it achieves strong performance on high-frame-rate and long video understanding with superior encoding efficiency. Furthermore, the unified learning paradigm for document knowledge and text recognition allows the model to directly learn from document images. This approach bypasses fragile parsers and significantly reduces the data engineering complexity. Finally, the hybrid post-training strategy improves both training and inference efficiency while also facilitating generalization between short and long reasoning modes. Overall, MiniCPM-V 4.5 demonstrates a promising path toward addressing the efficiency bottlenecks in MLLM development.

### Appendix A: Implementation Details

Pre-training follows a WSD schedule [17] with a fixed learning rate of 5 × 10⁻⁵ in the stable phase, decaying to 1 × 10⁻⁵. SFT applies cosine decay from 1 × 10⁻⁵ to 1 × 10⁻⁶. The Long-CoT and 3D-Resampler stage continues from the SFT checkpoint, warming up to 5 × 10⁻⁶ and decaying to 1 × 10⁻⁶.

For the RL stage, we adopt GRPO [79] without entropy loss or KL penalty. Each batch consists of 128 prompts with 8 responses each, and a max response length of 8192 tokens to support detailed reasoning. Rollouts use a temperature of 1.0, with 50% of prompts assigned to long reasoning mode. We use a fixed learning rate of 1 × 10⁻⁶ throughout RL. In the RLAIF-V [31] stage, we use a global batch size of 256, learning rate of 1 × 10⁻⁶, and β = 0.1 for 400 steps.

### Appendix B: Qualitative Cases

**B.1 Comprehensive Instruction Following**

- **Figure 4:** A case of comprehensive real-world reasoning.
- **Figure 5:** A case of comprehensive real-world reasoning in Chinese.
- **Figure 6:** A case of creative writing in Chinese.

**B.2 World Knowledge**

- **Figure 7:** A case of world knowledge understanding.
- **Figure 8:** A case of world knowledge understanding in Chinese.

**B.3 OCR**

- **Figure 9:** A case of handwritten text recognition.
- **Figure 10:** A case of handwritten text recognition in Chinese.
- **Figure 11:** A case of table content extraction.

**B.4 Problem Solving**

- **Figure 12:** A case of chemistry problem solving in Chinese.
- **Figure 13:** A case of multi-image statistical problem solving.

### References

Please refer to the original PDF for the complete reference list (79 entries), including works on GLM-V, Kimi-VL, MiMo-VL, GPT-4o, Qwen-VL / Qwen2.5-VL, MiniCPM-V, Ovis, InternVL3, Flash-VL, Gemini 2.0, DeepSeek-R1, LLaVA-UHD, Video-MME, MiniCPM, LAION-5B, COYO-700M, CLIP, CapsFusion, Common Crawl, OmniCorpus, MINT-1T, SynthText, Frozen-in-Time, Vript, OpenVid-1M, Wikipedia, OpenThoughts, RLAIF-V, RLPR, G-LLaVA, R-CoT, CLEVR-Math, INFOTABS, PromptPG, WikiTableQuestions, TabFact, TAT-QA, FigureQA, Multimodal-ArXiv, DVQA, VisualWebInstruct, Llama-Nemotron, MulDimIF, DeepSeekMath / GRPO, Skywork-VL Reward, DPO, MMMU, MathVista, AI2D, MathVerse, LogicVista, EMMA, OCRBench, ChartQA, TextVQA, DocVQA, OmniDocBench, HallusionBench, ObjHal, MMHal-Bench, Mantis, MMT-Bench, Grok-1.5V / RealWorldQA, MM-IFEngine, LVBench, MLVU, LongVideoBench, MotionBench, FAVOR-Bench, OpenCompass, MM-Vet, MMStar, MME, MMBench.
