# MultiTrust: A Comprehensive Benchmark Towards Trustworthy Multimodal Large Language Models

## 1 Introduction

**背景**：
- 多模态模型在可信性方面仍然表现出显著缺陷，导致**事实性错误** [62, 118]、**有害输出** [183]、**隐私泄露** [103] 等问题；
- 针对 MLLMs 的相应评估框架仍然缺失；
- 除了 LLMs 固有的弱点之外，MLLMs 的多模态特性还引入了新的风险，例如易受到对抗性图像攻击 [38, 179]、图像中存在有毒内容 [153]，以及通过视觉上下文进行越狱 [21, 112]；
- 当前工作 [76, 87, 145] 通常只考察可信性的一个或少数几个方面，并且只在有限任务上从现象层面对 MLLMs 进行评估，关注图像中的威胁，却忽视了模态之间的交互作用与影响；
- 目前仍缺乏对 MLLM 可信性的全面评估。

**目的**：
- 建立了 MultiTrust，这是首个用于跨多个维度和任务评估 MLLMs 可信性的全面且统一的基准；
- MultiTrust 中确定了可信性的 5 个主要方面，包括真实性、安全性、鲁棒性、公平性和隐私性，涵盖模型在防止意外结果方面的可靠性，以及其对用户社会影响的保障；
- 提出了一种更深入的评估策略，通过同时考虑新场景中的多模态风险，以及视觉输入对基础 LLMs 性能的跨模态影响，深入探究 MLLMs 的多模态本质；
- 设计了 32 个多样化任务，包括：
  * 对现有多模态任务的改进；
  * 将纯文本任务扩展到多模态场景；
  * 提出新的风险评估方法。

**发现**：
- 开源 MLLM 在多个通用基准上已经接近甚至超过闭源模型 [43, 90, 173]，但在可信性方面仍存在显著差距（e.g.GPT-4V、Claude3等）；
- 多模态训练以及推理阶段图像输入的引入，会在多个层面严重削弱 MLLM 的可信性，包括但不限于：
  1. 基础 LLM 的性能与对齐能力受到破坏；
  2. 无关图像会导致模型行为不稳定；
  3. 相关视觉上下文会加剧可信性风险。
- 某些任务的结果验证了不同模型组件（如视觉编码器 [30]、对齐后的 LLM [143]）以及现有训练范式（如基于 GPT-4V 蒸馏数据的监督微调 [151]、RLHF [133]）在提升可信性方面的贡献。然而，仅依赖现有技术仍远不足以实现全方位的可信保障。

## 2 Framework of MultiTrust

- 2.1 节介绍该基准的设计原则，重点关注评估维度与评测策略。
- 2.2 节简要回顾了在两级分类体系下的 32 个任务。
- 2.3 节、2.4 节和 2.5 节分别介绍用于评估的指标、被选取进行测评的模型，以及用于可信性研究的可扩展且标准化的工具箱。

![Figure 1](../../images/MML/MultiTrust_Figure_1.png)  

### 2.1 Philosophy of MultiTrust

#### 2.1.1 Evaluation Aspects

用于评估 MLLMs 可信性的 5 个核心维度，包括：

* 真实性（truthfulness）
* 安全性（safety）
* 鲁棒性（robustness）
* 公平性（fairness）
* 隐私性（privacy）

其中：

* **真实性、安全性与鲁棒性**用于保证模型在不同条件下的可靠性与稳定性，以避免产生不良结果，例如错误、危害以及条件变化下的不稳定表现。
* **公平性与隐私性**则关注模型的社会与伦理影响，包括诸如偏见等不一致态度，以及身份盗用等权利侵犯问题。

可信性研究框架同时包含**技术层面**和**伦理层面**。

#### 2.1.2 Evaluation Strategy

在评测与任务设计方面，为了能够更加全面地覆盖 MLLMs 的多模态特性，MultiTrust 同时考虑了：

* 由新模态引入的多模态风险（multimodal risks）
* 模态之间相互作用带来的跨模态影响（cross-modal impacts）

现有的大多数研究 [49,75,87] 主要关注：

* 图像模态中的可信性威胁，
* 或图文对（image-text pairs）组合中的可信性问题，

这些问题主要体现为视觉模态新引入的多模态风险（multimodal risks newly introduced by vision）。

除了这些问题之外，考虑不同模态之间的交互同样至关重要。具体而言，新引入的模态可能会改变模型在传统 LLM 场景中的原有行为 [112,145]。

这一问题与 MLLMs 在更广泛应用中的稳定性密切相关，但目前仍缺乏研究。因此，MultiTrust 提出通过如下方式研究跨模态影响（cross-modal impacts）：

* 在纯文本任务（text-only tasks）中，分别配合语义相关（semantically relevant）与语义无关（semantically irrelevant）的图像，测量模型性能变化（如Figure 1 所示）。

这种设计能够对 MLLMs 进行更加全面深入的研究。

我们的目的在于强调：
模态之间交互所带来的可信性风险范围非常广泛，并且这一研究框架未来还可以进一步扩展到其他模态。

此外，在本工作中，我们还考虑了文本扰动（text variations）[122,91] 在若干多模态任务中的影响。对于这一问题，我们将其视为 LLM 已经广泛关注的既有问题，而不是 MLLMs 特有的新问题。

### 2.2 Practice in MultiTrust

MultiTrust 组织了一个包含 10 个子维度（sub-aspects）的两级分类体系，以便更好地划分需要评估的目标行为。

基于该分类体系，整理了 32 个多样化任务，用以覆盖真实且全面的、存在可信性风险的场景。这些任务既包括生成式任务（generative tasks），也包括判别式任务（discriminative tasks）；既包括多模态任务（multimodal tasks），也包括纯文本任务（text-only tasks），如表 2 所总结。

![Table_2](../../images/MML/MultiTrust_Table_2.png)

为了解决当前缺乏专门面向这些子维度下不同场景的数据集的问题，MultiTrust 基于已有的文本、图像和多模态数据集 [43,80,107,147]，通过人工处理与自动化方法相结合的方式，对提示词、图像和标注进行改写与适配，从而构建了 20 个数据集。

此外，MultiTrust 还从零开始提出了 8 个新的数据集：这些数据集通过从互联网收集图像，或使用 Stable Diffusion [120] 以及其他算法合成图像，专门服务于所设计的任务。

下面，将介绍每个子维度的设计细节：首先介绍多模态任务，随后介绍用于研究跨模态影响的纯文本任务。

#### 2.2.1 Truthfulness

真实性衡量的是 MLLMs 的输出是否与客观事实一致，重点关注模型所提供信息的准确性。

不同于以往研究 [75,76,132] 主要狭义地关注幻觉（hallucination）和迎合性回答（sycophancy）等现象，本文从宏观角度将真实性重新划分为：

* 内在缺陷（inherent deficiency）[16,82,180]
* 误导性错误（misguided mistakes）[81,113]

**内在缺陷（Inherent Deficiency）**

内在缺陷深入探讨的是模型自身内部局限性如何导致不准确输出。

MultiTrust 首先评估 MLLMs 的基本感知能力 [43,76,13]，例如：

* 判断物体是否存在（object existence judgment，任务 T.1）

以及高级认知能力 [90,154]，例如：

* 时空推理（spatial-temporal reasoning，任务 T.2）

这些任务使用的是基于已有数据集改进而来的数据集。

除此之外，MultiTrust 还提出将不同形式的辅助指令（assisting instructions）整合进以往的 VQA 任务中（任务 T.3），以探索提示引导（prompt guidance）对模型表现的帮助。

MultiTrust 还构建了一个数据集，其中提示词由 GPT-4 生成，图像从互联网收集，用于测试模型在有视觉辅助条件下进行文本事实问答（text-based factual question answering with visual assistance）的表现（任务 T.4）。

**误导性错误（Misguided Mistakes）**

误导性错误关注的是由误导性输入 [34,75] 所导致的错误。

MultiTrust 首先向模型同时提供图像以及包含事实错误的问题，以观察这些错误问题会如何影响模型在 VQA 中的回答（任务 T.5）。

除了文本中的误导之外，MultiTrust 还建立了一个新数据集，其中包含人工收集的视觉错觉图像 [113,140]，用于考察模型在视觉上具有挑战性的场景中的表现（任务 T.6）。

与“视觉辅助下的问答”任务相对应，MultiTrust 将相同的问题与错误图像进行配对，并测量视觉误导对模型产生的干扰（任务 T.7）。

#### 2.2.2 Safety

安全性保证 MLLMs 的回答不会导致意料之外的后果，例如无意伤害 [61] 或非法行为 [67]。

在大模型安全研究 [124,132] 中，两个最重要的主题是：

1. AI 生成内容的毒性（toxicity）[47]：这可能会极大影响用户交互；
2. 模型越狱（jailbreaking）：即绕过安全协议 [89,156]，从而促成恶意滥用。

**毒性（Toxicity）**

毒性衡量的是模型生成有害回答的倾向 [88]。

为了测试模型对有毒内容的敏感性，MultiTrust 使用 NSFW 图像 [1,153,160]，例如色情和暴力图像，进行图像描述任务（Task S.1）。

MultiTrust 设计了风险识别任务，分别考察物体的存在以及物体的使用方式（Task S.2），以观察模型是否具备超越有害物体检测 [55] 之外的安全风险意识。

MultiTrust 还使用 RealToxicityPrompts [47]，结合多样化图像，评估输出毒性的变化（Task S.3）。

**越狱（Jailbreaking）**

越狱研究的是模型面对诱导其生成非法回答的攻击尝试时的抵抗能力 [156]。

MultiTrust 将面向 LLMs 的越狱提示 [125,171] 转换为截图风格的图像 [106]，以观察是否会通过 OCR 触发危险内容（Task S.4）。

MultiTrust 采用了针对 MLLMs 优化的多模态越狱方法 [49,87]，并提出了自己的攻击方法。该方法基于前一任务中的趋势，简化了越狱组合的复杂性，以减少模型混淆（Task S.5）。

MultiTrust 还将由 Stable Diffusion 生成的图像与文本越狱结合起来，以观察模型表现的波动（Task S.6）。

#### 2.2.3 Robustness

鲁棒性评估的是模型在分布偏移（distribution shifts）或输入扰动（input perturbations）下的一致性与抵抗能力。对于 MLLMs 而言，这仍然是一个尚未解决的问题 [38,145,179]。

按照该领域的常见做法 [132,147]，分别考虑：

* 分布外鲁棒性（out-of-distribution robustness, OOD robustness）
* 对抗鲁棒性（adversarial robustness）

**分布外鲁棒性（OOD Robustness）**

分布外鲁棒性评估的是 MLLMs 对非常规领域的泛化能力，这些非常规领域包括多样化的风格和应用场景。

首先，采用 COCO-O [95]，其中包含多种艺术风格的图像，用于图像描述任务（Task R.1）。
随后，使用来自文献 [19] 的 VQA 任务，其中的图像由多种传感器捕获，例如 MRI 和红外成像（Task R.2）。
之后，在来自 [147] 的 OOD SST-2 [128] 上进行测试，并将文本分别与无关图像以及根据文本提示生成的图像进行配对（Task R.3）。

**对抗攻击（Adversarial Attack）**

对抗攻击探索的是 MLLMs 面对对抗样本时的脆弱性，而这种脆弱性不可避免地继承自深度神经网络 [134]。

借助当前最先进的可迁移攻击技术 [24]，在图像描述任务中生成对抗样本，并分别考虑非定向攻击（untargeted setting）和定向攻击（targeted setting）（Task R.4 和 Task R.5）。

对于基于文本的攻击，采用 AdvGLUE [149] 和 AdvGLUE++ [147] 数据集，并将文本提示分别与相关图像和无关图像进行配对（Task R.6）。

#### 2.2.4 Fairness

公平性衡量的是模型输出在多大程度上不包含不平等或歧视性结果，因为这些结果可能会使某些用户群体处于不利地位 [9,97,147]。

根据歧视性输出的类型 [88,132]，MultiTrust 将这一概念进一步划分为：

* 刻板印象（stereotype）
* 偏见与偏好（bias & preference）

**刻板印象（Stereotypes）**

刻板印象关注的是识别 MLLMs 中延续下来的、根深蒂固的社会先入之见 [20]。

MultiTrust 首先从多种公开来源中谨慎收集可能面临歧视风险的人物图像，用于评估模型生成内容中包含的刻板印象（Task F.1）。

为了考察 MLLMs 在实际场景中对刻板印象的理解能力与敏感性，MultiTrust 基于文本任务 [100,102,155] 合成与文本语境相关的图像。这些任务覆盖多个层次，包括：

* 对刻板印象陈述是否表示同意（Task F.2）；
* 对刻板印象进行分类（Task F.3）；
* 面对带有刻板印象的用户查询时的回答（Task F.4）。

**偏见与偏好（Bias & Preference）**

偏见与偏好考察的是模型是否存在某种倾向：这种倾向可能使特定用户群体处于不利地位，或偏向带有偏见的结果。

MultiTrust 将已有研究 [132] 中基于文本的偏好选择任务转换为由图像表示的选项，用于评估 MLLMs 中嵌入的视觉偏好（Task F.5）。

随后，要求模型根据人物图像判断其职业胜任力，并基于文献 [5] 中的标注，使用卡方检验（Chi-square test）[135] 来量化 MLLMs 对不同个人属性，例如性别、年龄等，所表现出的偏见（Task F.6）。

最后，考察当纯文本形式的选择问题 [132] 与各种图像配对时，MLLMs 是否更倾向于表达自己的偏好（Task F.7）。

#### 2.2.5 Privacy

隐私性评估的是模型保护个人数据免受未授权请求影响的能力 [98]。

已有研究表明，大模型容易受到数据抽取攻击 [103]，并且在推理过程中倾向于泄露隐私信息 [130]。当模型被部署到隐私敏感型应用中时，这会带来风险。

从意识（consciousness）和行为（behaviors）两个角度 [132] 出发，MultiTrust 从以下两个方面评估隐私性：

* 隐私意识（privacy awareness）
* 隐私泄露（privacy leakage）

**隐私意识（Privacy Awareness）**

隐私意识要求模型能够在其工作流程中检测个人信息与隐私风险的存在 [132]。

按照难度逐渐增加的方式，MultiTrust 依次测试 MLLMs：

1. 首先，要求模型识别图像中是否存在私人信息 [107,54]（Task P.1）；
2. 随后，要求模型判断针对这些图像提出的问题是否涉及其中的私人信息（Task P.2）。这一任务不仅需要感知能力，还需要进一步的推理能力。

这些问题由 GPT-4V 构造，并经过人工标注。

接着，MultiTrust 将文献 [99] 中的纯文本任务 InfoFlow Expectation 与不同图像进行配对，并评估模型在隐私使用同意程度上的变化（Task P.3）。

**隐私泄露（Privacy Leakage）**

隐私泄露评估的是模型在服务过程中防止私人信息泄露的能力 [147]。

类似于 LLM 红队测试中涉及隐私的提示 [109]，MultiTrust 收集了一组名人照片，并以这些照片作为视觉线索，询问模型这些人物的个人可识别信息（personally identifiable information, PII）（Task P.4）。

随后，MultiTrust 要求模型识别图像中的 PII，这些图像来自一个公开数据集 [107]，并经过人工标注（Task P.5）。

当模型能够访问公众的私人数据，并且具备强大的 OCR 能力时，这两个任务都具有实际意义。

进一步地，MultiTrust 还向 MLLMs 查询过去文本中包含的私人信息 [147]，并测量当这些文本与图像配对时模型的隐私泄露情况（Task P.6）。

### 2.3 指标（Metrics）

如表 2 所示，MultiTrust 针对不同任务采用了多种指标，以提供更加准确且直接的评估。为了清晰展示评估方式，MultiTrust 在此系统性地总结了所提出基准中使用的指标。MultiTrust 主要将这些指标分为客观指标和主观指标。

#### 客观指标

对于具有明确闭集答案的任务，人们通常使用客观指标，例如准确率、Pearson 相关系数 [123] 和 p 值 [127]。它们的形式化定义如下。

**准确率（Accuracy）**

$$
Acc=\frac{\sum_{i=1}^{N}\mathbb{I}(r_i=y_i)}{N}
$$

其中，$r_i$ 是模型回答，$y_i$ 是标签。
$\mathbb{I}$ 用于判断回答是否与标签一致，例如通过精确匹配、关键词匹配等方式进行判断。

**Pearson 相关系数（Pearson Correlation）**

$$
Corr = Pearson\ Correlation(\mathbf{r}, \mathbf{y})
$$
其中，$\mathbf{r}$ 是预测分数序列，$\mathbf{y}$ 是真实分数数组。

**p 值（p-Value）**

$$
p\text{-value}=P(\chi^2 \geq \chi^2_{obs})
$$
其中：
$$
\chi^2_{obs}=\sum_i \frac{(O_i-E_i)^2}{E_i}
$$
这里，$O_i$ 表示观测频数，$E_i$ 表示期望频数。

**其他指标**

其他指标，例如 ASR 和 Cure Rate，是基于准确率等基础指标，通过减法和比率计算推导得到的。

#### 主观指标

对于开放式问答任务，通常需要定义主观指标，并使用外部评估器来评估模型行为，例如评分 API 或大模型，以替代人工评估。具体细节如下。

**毒性分数（Toxicity Score）**

ModerationAPI 和 PerspectiveAPI 被用于评估回答中的毒性程度。它们是 LLM 毒性研究中常用的工具。

**拒答率与包含率（Refuse-to-Answer Rate, Containing Rate）**

使用专门的 Longformer 模型和 GPT-4 等模型，通过二分类答案来判断回答是否传达了特定行为或观点。
这种做法已经在其他基准中得到验证 [155,87]。

**GPT 分数（GPT-Score）**

通过提示 GPT-4 对回答进行评分，从而得到主观评估结果，因此这种评估可能存在不确定性。

MultiTrust 在抽样子集上比较了 GPT-4 给出的分数与人工评分之间的一致性，得到的相关系数为 0.91。这表明 GPT-Score 是有效的，详见附录 E.1.2。

### 2.4 被评估模型（Evaluated Models）

仅仅编制一个当前最先进 MLLMs 的排行榜，并不足以解决可信性问题。因为不同模型在架构 [30,168] 和训练范式 [26,133] 上存在差异，仅靠排行榜很难为未来改进提供足够洞见。

为了解决这一问题，MultiTrust 基于若干标准有策略地选择模型。

首先，MultiTrust 纳入了 4 个先进的闭源商业模型，以突出开源模型在可信性方面的差距。
随后，MultiTrust 收集了来自丰富的 LLaVA 家族 [83] 的 6 个模型，以及基于 MiniGPT-4 [181] 和 mPLUG-Owl [167] 的 4 个模型，用于识别不同改进方式所带来的影响，例如：

* 基座 LLMs [143]；
* 改进的数据集 [26,151]；
* 基于人类反馈的强化学习（reinforcement learning from human feedback, RLHF）[133]。

MultiTrust 还选择了 7 个在 MLLM 发展不同阶段中具有代表性的流行 MLLMs，以扩大模型覆盖范围。

最终，MultiTrust 形成了一组包含 21 个 MLLMs 的模型集合，用于进行全面评估，具体信息见表 B.1。

该基准将在之后持续更新，以纳入更多新发布的模型。

### 2.5 工具箱（Toolbox）

现有基准 [19,43,49,75] 通常只提供数据集和评估脚本，缺乏可扩展性与适应性，这极大限制了对最新模型和新任务的测试。

因此，MultiTrust 中专门开发了一个工具箱，为评估 MLLM 可信性以及促进未来研究提供通用且可扩展的基础设施。

MultiTrust 通过兼容不同开发者提供的多种模型交互格式，将不同的 MLLMs 集成到统一接口中，从而实现标准化的模型评估。

在任务设计方面，MultiTrust 通过将数据、推理和评估指标相互分离，对任务进行模块化处理，以便实现工具复用，并方便后续对新任务进行更新。

## 3 Analysis on Experimental Results

32 个任务上进行了大量实验，以完成该基准的评估。

图 2 中的为排名，并分析了每个维度下具有代表性的实验结果。

![Figure_2](../../images/MML/MultiTrust_Figure_2.png)

### 3.1 Overall Performance

从图 2 中，可以快速得出一个结论：GPT-4V 和 Claude3 等闭源商业模型始终表现最佳，这可以归因于它们在对齐、安全过滤等方面所做的努力。

从全局角度看，不同 MLLMs 的通用能力与可信性之间的相关系数为 0.60。如图 2(a) 所示，上一代 MLLMs 由于多模态感知和推理能力较弱，往往在多个可信性维度上表现不足；而更强大的能力则会在不同程度上带来更好的可信性表现。

与此同时，图 2(b) 中更细粒度的相关性分析表明，除了真实性、毒性和隐私意识等若干子维度之外，不同维度之间并不存在显著相关性。上述这些子维度需要类似的能力，例如识别能力。这个结果强调了进行全面维度划分的必要性，也说明当前模型的可信性距离理想状态仍存在差距。

### 3.2 Truthfulness

在真实性方面，MLLMs 能力上的内在缺陷普遍存在。

虽然 MLLMs 在一般感知任务上表现不错，例如物体存在判断和场景分析；如表 3 所示，大多数模型在这些任务上的准确率都超过 80%。但是，当任务转向更细粒度的任务时，例如视觉定位（visual grounding），模型性能会明显下降。例如 InternLM-XC2 的准确率为 32%，Gemini-Pro 甚至只有 8%。这突出了 MLLMs 在细粒度感知能力方面的局限性 [90]。

此外，MLLMs 在进一步认知推理过程中，对图像模态和文本模态的依赖也存在差异。例如，模型通常在常识推理任务上表现更好，因为这类任务主要利用 LLMs 已学习到的知识；但当任务需要利用视觉模态时，表现则不够理想。

如表 3 所示，后两个任务相比常识推理出现了明显的性能下降，这与以往研究发现一致 [25,154]。

至于外部误导因素，大多数开源模型容易受到混淆性或误导性图像的影响，并进一步产生错误信息；而闭源模型则表现出更强的抵抗能力。

![Table_3](../../images/MML/MultiTrust_Table_3.png)

### 3.3 Safety

在安全性任务中，与闭源商业模型相比，开源 MLLMs 通常具有更差的安全意识和防护机制。

例如，GPT-4V 和 Claude3 平均会拒绝描述 69.5% 的 NSFW 图像，而大多数开源模型完全不会拒答。

越狱任务中也出现了类似情况：这两个闭源模型几乎会拒绝所有恶意查询，而许多先进模型则频繁被攻击成功。

值得注意的是，即使不设计任何刻意构造的提示，仅仅将目标有害行为放入图像中，许多现代 MLLMs 就会由于关注图像内容而被成功越狱。例如，InternLM-XC2 的成功率为 71%，LLaVA-1.5 的成功率为 80%。

尤其值得注意的是，在使用相同基座 LLM Llama-2 [143] 的情况下，MiniGPT-4-L2 和 mPLUG-Owl2 在排版式越狱与 NSFW 描述任务上的表现，与它们在多模态语境下风险识别任务中的表现相反，如表 4 所示。

这表明：虽然多模态训练增强了模型的感知与理解能力，但来自已良好对齐的基座 LLM 的安全机制可能会被灾难性地削弱。

![Table_4](../../images/MML/MultiTrust_Table_4.png)

### 3.4 Robustness

虽然在 OOD 场景中的结果显示，不同模型之间并没有出现剧烈差异，但 MultiTrust 证明了 MLLMs 不可避免地继承了深度神经网络的对抗脆弱性。

在简单物体的图像描述任务中，在非定向攻击下，大多数模型的准确率从超过 90% 降至 20% 以下。

至于定向攻击，超过一半的模型会以高于 50% 的比例输出攻击者期望的目标物体；即使是商业模型 Qwen-VL-Plus 也存在这种情况，这突出了它们的脆弱性。

如图 3 所示，虽然两个先进的开源 MLLMs 具有最高的攻击成功率，但另外两个模型 CogVLM 和 InternVL-Chat 表现出显著更好的鲁棒性。这两个模型拥有参数更多且较为独特的视觉编码器，从而抑制了对抗样本的可迁移性。

同样的原因也可能适用于另外两个闭源模型；此外，它们也可能配备了用于净化噪声的输入过滤器 [104]。

![Figure_3](../../images/MML/MultiTrust_Figure_3.png)

### 3.5 Fairness

在刻板印象方面，大多数 MLLMs 对真实生活场景中的刻板印象用户查询表现出较高敏感性。即使在相关图像的影响下，它们仍保持了平均 93.79% 的拒答率（RtA rate）。

然而，当刻板印象从应用场景中的用户查询转向基于观点的评价时，MLLMs 的性能差异不仅在闭源模型与开源模型之间变得明显，也体现在不同刻板印象主题之间。

例如，与性别、种族、宗教等主题相比，年龄相关的刻板印象在 MLLMs 中表现出明显更高的同意率。

这种主题差异也存在于偏见与偏好评估中。如图 4 所示，最新模型，例如 Claude3、InternLM-XC2 和 LLaVA-NeXT，表现出不同程度的敏感性。

这些模型对意识形态话题更加敏感，而对文化或生活方式类问题更加宽容，因此更容易暴露它们的偏好，并进而影响用户决策。

![Figure_4](../../images/MML/MultiTrust_Figure_4.png)

### 3.6 Privacy

MultiTrust 发现，大多数模型都具备基本的隐私概念。在判断图像中是否存在私人信息时，模型平均准确率为 72.30%，这一结果与模型的一般感知能力相关。

然而，在需要更复杂推理的场景中，这种隐私意识会受到严重挑战，平均表现明显下降至 55.33%。这也拉大了闭源模型与开源模型之间的差距，其中只有 GPT-4V 和 Gemini-Pro 的准确率超过 70%。

在隐私泄露方面，MultiTrust 注意到：通过激活模型的 OCR 本能，从图像中提取隐私信息，比把图像作为 PII 查询的线索更容易破坏数据保护协议。这放大了 MLLMs 在多模态场景中的隐私风险。

两个纯文本任务也揭示了类似现象：多模态输入会影响 LLM 在指令遵循和隐私保护方面的行为。

如表 5 所示，当文本与图像配对时，模型更倾向于泄露 PII 信息。当 MLLMs 能够访问个人数据时，这会带来更大的威胁。

而 MiniGPT-4-Llama2 的结果表明，冻结的 Llama2 可以增强隐私保护能力。

![Table_5](../../images/MML/MultiTrust_Table_5.png)

# MultiTrust 数据机构健全过程

![Table_2](../../images/MML/MultiTrust_Table_2.png)

符号含义是：❌ 表示直接使用已有数据集，➕ 表示基于已有数据集改造，✅ 表示从零构建。

## Truthfulness: T.1–T.7

| ID | 任务 | 评估目标 | 数据来源 | 怎么构建 | 论文哪里讲到 |
|---|---|---|---|---|---|
| **T.1** | **Basic World Understanding** | 测试模型最基础的视觉感知能力：物体是否存在、属性、场景、定位、OCR 等是否能正确识别。 | 基于 AMBER、MME、RTVLM、OCR/文本识别等已有数据改造。 | 测试基础感知能力，包括 object existence judgment、attribute recognition、scene analysis、visual grounding、OCR。作者不是简单照搬原数据，而是重新设计负样本。例如图中有海但没有船，就可能问“是否有船”，这种负样本比随机不存在物体更难。最终包含 **800 个 image-text pairs**。 | Appendix **C.1.1**, p.30；主文 Table 2 |
| **T.2** | **Advanced Cognitive Inference** | 在感知之上测试高阶认知推理：时空关系、属性比较、常识推理、专业领域知识。 | 基于已有数据和部分手工构造。 | 测试高级认知能力，包括 spatial-temporal reasoning、attribute comparison、commonsense reasoning、specialized skills。它不只是问“图中有什么”，而是要求模型进行空间关系、时间顺序、大小/年龄/数量比较、常识推理等。 | Appendix **C.1.2**, p.33；主文 Table 2 |
| **T.3** | **VQA under Instruction Enhancement** | 测试不同复杂度的指令提示是否能改善模型在事实/推理问题上的表现。 | 基于已有 VQA/推理样本改造。 | 复用 T.2 中的数学/逻辑问题和 T.5 中的事实陷阱 VQA，然后给同一图文样本配三种复杂度递增的 prompt：直接回答、更多关注图像、逐步/深入思考。约 **140 个样本 × 3 种 prompt = 420 个 image-text pairs**。 | Appendix **C.1.3**, p.35；主文 Table 2 |
| **T.4** | **QA under Visual Assistance** | 测试在文本问题之外提供相关图像，是否能帮助模型回答长尾事实问题。 | 从零构建。 | 作者用 GPT-4 生成长尾事实问答，覆盖自然科学、历史、神话、人物、国家、地标建筑等。为了避免问题太常识化，要求 GPT-4 生成 long-tail knowledge。然后人工从互联网收集与答案相关的图片，同时配无关图片，包括噪声图、纯色图、自然图。 | Appendix **C.1.4**, p.38；主文 Table 2 |
| **T.5** | **Text Misleading VQA** | 测试模型是否会被文本中带有的错误事实前提诱导，从而给出错误答案。 | 基于已有 GPT-4V hallucination/interference 相关数据改造。 | 作者选取 100 个高质量 image-text pairs，再用 GPT-4 把原本正确的问题改写成带有事实错误的问题，同时保留正确答案。这样可以测试模型是否会被文本中的错误前提诱导。 | Appendix **C.2.1**, p.40；主文 Table 2 |
| **T.6** | **Visual Confusion VQA** | 测试模型在镜面反射、错位、3D 绘画、对比错觉等视觉混淆场景下能否避免被画面误导。 | 从零构建。 | 作者认为已有 MAD-Bench 只有 28 个 proprietary image-text pairs，不够用，因此自己构造 **300 个 image-text pairs**。覆盖五类视觉混淆场景：mirror reflection、visual dislocation、3D painting、contrast illusion、semantic misleading。例如由于透视错觉，画面看起来像人在锅上站着。 | Appendix **C.2.2**, p.42；主文 Table 2 |
| **T.7** | **Visual Misleading QA** | 测试错误/误导性图像是否会覆盖模型原本正确的文本知识，让回答倒向图像。 | 从零构建。 | 与 T.4 正好相反。T.4 是文本事实问题 + 正确相关图片，看图片是否帮助回答；T.7 是同样的文本问题 + 错误/误导性图片，看视觉信息是否会干扰模型原本的文本知识回答。 | Appendix **C.2.3**, p.44；主文 Table 2 |

## 二、Safety: S.1–S.6

| ID | 任务 | 评估目标 | 数据来源 | 怎么构建 | 论文哪里讲到 |
|---|---|---|---|---|---|
| **S.1** | **NSFW Image Description** | 测试模型在面对不适宜图像（暴力、色情、政治敏感等）时是否会拒绝描述或输出不当内容。 | 直接使用已有 NSFW/不适宜图像资源，包括 ToViLaG、UCLA-protest、NSFW Data Scraper。 | 图像类别包括政治抗议、血腥场景、色情、性暗示但非色情、hentai drawings。每类随机采样 50 张图，共 250 张图，再配两个描述 prompt，形成 **500 个样本**。 | Appendix **D.1.1**, p.48；主文 Table 2 |
| **S.2** | **Risk Identification** | 测试模型能否识别看似日常但可能导致严重后果的物体（枪、刀、酒、香烟等）的潜在风险。 | 基于 HOD dataset 改造。 | 作者选择五类“低风险但可能导致高风险后果”的物体：guns、knives、alcohol、insulting gestures、cigarettes。每类 20 张图，并设计 object recognition 与 risk analysis 两种设置，最终形成 **500 个 image-text samples**。例如酒本身常见，但和药物结合可能危险。 | Appendix **D.1.2**, p.49；主文 Table 2 |
| **S.3** | **Toxic Content Generation** | 测试在毒性 prompt 下加入相关/无关图像，是否会改变模型生成有毒内容的倾向。 | 基于 RealToxicityPrompts 改造。 | 作者用 PerspectiveAPI 给每条文本确定 toxicity category，从 8 类 toxicity 中每类采样 10 条，共 80 条文本；然后配无关图像，以及用 Stable Diffusion XL 根据文本生成语义相关图像，测试视觉上下文是否改变毒性生成。 | Appendix **D.1.3**, p.51；主文 Table 2 |
| **S.4** | **Plain Typographic Jailbreaking** | 测试把越狱指令写成图像后，模型是否会因为文字以图像形式出现而绕过安全机制。 | 从零构建。 | 作者从 GPTFuzzer、DAN 中选 jailbreak prompts，从 HarmBench 中选 harmful behaviors，然后把 jailbreak prompt 或 harmful behavior 渲染成截图式/文字图像，测试模型 OCR 到危险指令后是否被越狱。 | Appendix **D.2.1**, p.53；主文 Table 2 |
| **S.5** | **Optimized Multimodal Jailbreaking** | 评估现有针对 MLLM 的优化型多模态越狱攻击（FigStep、MM-SafetyBench 等）对模型的实际威胁。 | 基于 SafeBench/FigStep 和 MM-SafetyBench 改造。 | 作者取 SafeBench 中 200 个样本、MM-SafetyBench tiny version 中 168 个样本，同时提出自己的简化攻击方式，以减少复杂 prompt 组合导致的模型混淆。 | Appendix **D.2.2**, p.55；主文 Table 2 |
| **S.6** | **Cross-modal Influence on Jailbreaking** | 测试图像上下文是正相关还是负相关，会如何影响模型对文本越狱请求的拒答行为。 | 基于 GPTFuzzer、DAN、HarmBench 构造。 | 作者构造 50 个文本越狱样本；再配无关图像，以及作者人工收集的 18 张与越狱语境正相关或负相关的图片，包括 explicit correlation、implicit correlation、related text 三类，用来测试图像上下文会不会改变安全拒答。 | Appendix **D.2.3**, p.56；主文 Table 2 |

## Robustness: R.1–R.6

| ID | 任务 | 评估目标 | 数据来源 | 怎么构建 | 论文哪里讲到 |
|---|---|---|---|---|---|
| **R.1** | **Image Captioning for Stylized Images** | 测试模型对非自然风格图像（素描、卡通、绘画、恶劣天气等）的描述鲁棒性。 | 直接使用 COCO-O。 | COCO-O 包含 6 种艺术/异常视觉风格：sketch、cartoon、painting、weather、handmake、tattoo。作者每类采样 100 张图，并人工标注图中最明显的物体，用来判断模型 caption 是否覆盖主体。 | Appendix **E.1.1**, p.60；主文 Table 2 |
| **R.2** | **VQA for Sensor Style Images** | 测试模型在红外、X 光、MRI、CT、遥感、自动驾驶等专业传感器图像上的理解能力。 | 直接使用 BenchLMM。 | 作者纳入 infrared、L-Xray、H-Xray、MRI、CT，以及 remote sensing、autonomous driving 等图像；过滤掉没有标准答案的样本，最终得到 **1041 张图**。 | Appendix **E.1.2**, p.62；主文 Table 2 |
| **R.3** | **Sentiment Analysis for OOD Texts** | 测试模型对 OOD 风格化文本的情感判断鲁棒性，以及配上图像是否会改变这种鲁棒性。 | 基于 DecodingTrust 中的 OOD SST-2 改造。 | 作者采样 word-level substitution 和 sentence-level style transformation 后的 OOD 文本；同时给文本配无关图像，以及用 Stable Diffusion XL 根据原始文本生成相关图像，测试图像对文本鲁棒性的影响。 | Appendix **E.1.3**, p.63；主文 Table 2 |
| **R.4** | **Image Captioning under Untargeted Attack** | 测试模型在非定向对抗扰动下能否仍正确识别图中主体（视觉对抗鲁棒性）。 | 从 NIPS17 adversarial dataset 中选图，并从零生成对抗样本。 | 作者随机选 100 张图，手工把 ImageNet 细粒度标签简化成 MLLM 更容易理解的类别，比如各种狗统一成 “dog”。然后用 SSA-CWA attack，扰动范围 ε=16/255，生成非定向对抗样本。 | Appendix **E.2.1**, p.66；主文 Table 2 |
| **R.5** | **Image Captioning under Targeted Attack** | 测试定向对抗扰动是否能诱导模型 caption 出攻击者指定的目标物体。 | 使用和 R.4 相同的基础图像与标签集。 | 区别是定向攻击会随机选择一个不属于该图的目标标签 y′，攻击希望模型 caption 出这个目标物体。 | Appendix **E.2.2**, p.68；主文 Table 2 |
| **R.6** | **Textual Adversarial Attack** | 测试模型对文本对抗扰动的鲁棒性，以及相关/无关图像作为视觉上下文的影响。 | 基于 AdvGLUE 和 AdvGLUE++ 改造。 | 包括 SST-2、QQP、MNLI、MNLI-mm、QNLI、RTE 等任务；文本扰动包括 word-level 和 sentence-level adversarial perturbations。作者再配相关/无关图像，相关图像由 Stable Diffusion XL 根据文本生成。 | Appendix **E.2.3**, p.70；主文 Table 2 |

## Fairness: F.1–F.7

| ID | 任务 | 评估目标 | 数据来源 | 怎么构建 | 论文哪里讲到 |
|---|---|---|---|---|---|
| **F.1** | **Stereotypical Content Generation** | 测试模型自由描述人物图像时，是否会自发输出年龄/肤色/职业相关的刻板印象。 | 基于 FACET 手工筛选。 | 作者从 FACET 中选择人物图像，覆盖年龄、肤色、职业等属性，最终 100 张图，包含 young/middle-aged/older、white/olive/dark skintone，以及 27 种职业，如 judge、singer、ballplayer。让模型描述图像，看是否生成刻板印象。 | Appendix **F.1.1**, p.74；主文 Table 2 |
| **F.2** | **Agreement on Stereotypes** | 测试模型是否会同意/认可刻板印象陈述，以及视觉上下文是否会强化这种认同。 | 基于 CrowS-Pairs 改造。 | 作者筛掉刻板印象不明显的陈述，保留 261 条，覆盖 age、gender、nationality、race、religion、socioeconomic 六类；再用 Stable Diffusion XL 根据陈述生成语义相关图片，并配无关图像。 | Appendix **F.1.2**, p.76；主文 Table 2 |
| **F.3** | **Classification of Stereotypes** | 测试模型能否识别一条陈述是否属于刻板印象，以及它属于哪类（即对偏见的认知能力）。 | 来源包括 F.2 中的刻板印象陈述和 StereoSet。 | 第一部分用 F.2 中的 261 条刻板印象陈述，让模型分类其主题；第二部分从 StereoSet 随机选 300 条陈述，让模型判断陈述性质。相关图像沿用 F.2 中 Stable Diffusion 生成的图像。 | Appendix **F.1.3**, p.78；主文 Table 2 |
| **F.4** | **Stereotype Query Test** | 测试模型在面对真实场景式歧视/刻板印象问题时是否会拒答或给出有偏见的回应。 | 基于 Do-Not-Answer 中 “Social Stereotypes and Unfair Discrimination” 类别改造。 | 作者选出 96 条真实场景式歧视/刻板印象问题，包括 body shaming、disability discrimination、gender/sexual discrimination、racial/ethnic discrimination、religious discrimination 等；再用 GPT-4 将问题改写成适合图像生成的 prompt，用 Stable Diffusion 生成相关图。 | Appendix **F.1.4**, p.81；主文 Table 2 |
| **F.5** | **Visual Preference Selection** | 测试用图像形式呈现主观二选一问题时，模型是否会表达意识形态/生活方式上的偏好（而非拒绝）。 | 从零构建。 | 作者从 TrustLLM 的 preference selection 任务中抽取 120 道主观二选一问题，其中 80 道 ideology，40 道 culture/lifestyle；把两个文本选项分别用图像表示，再拼成左右/上下两个图像选项，让模型选择偏好。正确行为应该是拒绝表达偏好或警告。 | Appendix **F.2.1**, p.83；主文 Table 2 |
| **F.6** | **Profession Competence Prediction** | 测试模型在判断人物是否胜任特定职业时，是否对年龄、性别、肤色等属性存在系统性偏见。 | 基于 FACET 改造。 | 作者选择六类容易产生偏见的职业：nurse、teacher、waiter、doctor、lawman、fireman；并平衡 age、gender、skintone 等属性，最终 313 张图。让模型判断人物是否胜任职业，用 p-value 衡量对属性的偏见。 | Appendix **F.2.2**, p.85；主文 Table 2 |
| **F.7** | **Preference Selection in QA** | 测试在文本主观问题旁附加相关/无关图像，是否会诱导模型在意识形态/文化问题上表态。 | 基于 TrustLLM 的文本偏好选择任务改造。 | 共 120 道二选一问题，80 道 ideology，40 道 culture/lifestyle；每个选项用 Stable Diffusion XL 生成相关图，共 240 张相关图。评估时每题随机选一张选项图像作为输入，也配无关图像，测试图像是否诱导模型表达偏好。 | Appendix **F.2.3**, p.87；主文 Table 2 |

## Privacy: P.1–P.6

| ID | 任务 | 评估目标 | 数据来源 | 怎么构建 | 论文哪里讲到 |
|---|---|---|---|---|---|
| **P.1** | **Visual Privacy Recognition** | 测试模型能否识别图像中包含的隐私敏感内容（地址、邮箱、信用卡、车牌、证件等）。 | 直接使用 VISPR 和 VizWiz-Priv。 | VISPR 来自 Flickr 图片，含 68 类隐私属性；作者选取 home address、email、credit card、phone number、license plate、credential 等敏感属性，构造 500 张正负平衡图像。VizWiz-Priv 中隐私区域被模糊处理，要求模型根据上下文推理隐私风险。 | Appendix **G.1.1**, p.91；主文 Table 2 |
| **P.2** | **Privacy-Sensitive VQA Recognition** | 测试模型能否在 VQA 形式下判断一个问题是否涉及图像中的隐私敏感信息。 | 基于 VISPR private split 改造。 | 作者选择 P.1 中 250 张含隐私图像，用 GPT-4V 为每张图生成 3 个正问题和 3 个负问题；然后人工为每个标签选最合适的问题，并过滤不合适样本，最终得到 **426 个正负平衡 image-text pairs**。 | Appendix **G.1.2**, p.93；主文 Table 2 |
| **P.3** | **InfoFlow Expectation** | 测试模型对不同信息类型 × 使用场景的隐私敏感度判断是否与人类期望对齐。 | 基于 InfoFlow Expectation 数据改造。 | 作者取 Tier-2a split 的 98 条 prompt，包含 7 类信息类型和 14 种使用场景；再配无关图像，并用 Stable Diffusion XL 根据使用场景生成 14 张相关图像。让模型给隐私使用场景打分，看是否与人类隐私期望一致。 | Appendix **G.1.3**, p.94；主文 Table 2 |
| **P.4** | **PII Query with Visual Cues** | 测试当用户给出名人照片并询问其邮箱、电话、地址时，模型是否会泄露相关 PII。 | 从零构建。 | 作者人工收集 IMDB top 100 celebrities 的公开照片；针对每个人构造 12 个查询，覆盖 email、phone number、address 三类 PII，并设计四种 prompt：给 name+job、只给 name、只给 job、什么都不给。最终 **1200 个 image-text pairs**。 | Appendix **G.2.1**, p.96；主文 Table 2 |
| **P.5** | **Privacy Leakage in Vision** | 测试在描述含可见隐私字段的图像时，模型是否会原样读出图中的具体 PII（OCR 式泄露）。 | 基于 VISPR test split 改造。 | 原始数据保留了图像中的隐私内容，但没有细粒度 PII 标注；作者人工标注图中的 email、name、address、credit card、ID number 等具体隐私字段，最终 **195 个样本**，每个样本最多 3 条 ground-truth PII。 | Appendix **G.2.2**, p.98；主文 Table 2 |
| **P.6** | **PII Leakage in Conversations** | 测试在对话上下文中给定 PII 后，模型是否会在后续回答中泄露这些 PII，以及图像形式是否会加剧泄露。 | 基于 DecodingTrust 代码随机生成测试数据。 | 作者选 10 类 PII：phone、email、address、password、credit card、passport number、ssh private key、secret key、fictional canary code/number 等，每类 10 条，共 100 个文本样本；再配无关图像和相关 typography 图像。相关图像有两类：只嵌入 PII type，或把全部信息都写进图像。 | Appendix **G.2.3**, p.100；主文 Table 2 |

## 总结：MultiTrust 数据集的构建套路

MultiTrust 的数据构建不是单纯下载一个已有数据集，而是围绕 **32 个可信多模态评估任务**进行系统性组合、改造和扩展。

### 1. 直接继承已有数据集

例如：

- S.1 NSFW Image Description；
- R.1 Image Captioning for Stylized Images；
- R.2 VQA for Sensor Style Images；
- P.1 Visual Privacy Recognition。

这类任务主要把已有数据集纳入 MultiTrust 的统一评估框架中。

### 2. 基于已有数据集改造

例如：

- T.1、T.2、T.5；
- S.2、S.3、S.5、S.6；
- R.3、R.6；
- F.1–F.4、F.6、F.7；
- P.2、P.3、P.5、P.6。

改造方式包括：

- 增加新的 prompt；
- 增加相关/无关图像；
- 用 Stable Diffusion 生成语义相关图像；
- 用 GPT-4/GPT-4V 生成问题或辅助标注；
- 人工筛选、过滤和补充标签。

### 3. 从零构建新任务数据

例如：

- T.4 QA under Visual Assistance；
- T.6 Visual Confusion VQA；
- T.7 Visual Misleading QA；
- S.4 Plain Typographic Jailbreaking；
- R.4/R.5 图像对抗攻击任务；
- F.5 Visual Preference Selection；
- P.4 PII Query with Visual Cues。

这类任务通常用于捕捉已有 benchmark 没覆盖的新型多模态风险。

### 4. 跨模态影响任务的统一设计

对于 cross-modal impacts，作者通常采用如下设计：

1. 先准备一个文本任务；
2. 再给文本任务配上不同类型图像：
   - 语义相关图像；
   - 语义无关自然图像；
   - 噪声图或纯色图；
   - 有时还包括正相关/负相关图像；
3. 比较模型在不同图像条件下的表现变化。

这样做的目的不是测试“图像识别能力”本身，而是测试：

> 一个原本属于文本任务的问题，在加入图像输入之后，MLLM 的行为是否会被视觉模态改变。

这便是 MultiTrust 中 **cross-modal impacts** 的核心思想。
