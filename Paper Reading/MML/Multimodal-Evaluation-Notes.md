# 多模态评估笔记

> 关于多模态大模型（MLLM）评估方向的学习笔记。按主题组织，逐步累积。

---

## 1. 多模态评估的类别

多模态评估可以从「**考什么（评估对象）**」与「**怎么考（评估方法论）**」两个维度切分。

### 1.1 考什么：评估对象的分类

#### A. 基础能力（Foundation Capabilities）

衡量模型通用感知与认知能力的基线评估。

| 子类 | 评估目标 | 代表基准 |
|---|---|---|
| 综合评估 | 感知 + 认知的综合表现 | MME, MMBench, SEED-Bench(-2), MMT-Bench, MM-Vet, MMStar, MME-RealWorld, BLINK, RealWorldQA |
| OCR | 自然/场景/手写文字识别 | TextVQA, OCR-VQA, OCRBench, VCR, InfoVQA, WebSRC |
| 图表与文档 | 结构化、信息密集型数据理解 | ChartQA, DocVQA, InfoVQA, CharXiv, MMLongBench-Doc, DocGenome |
| 数学推理 | 视觉数学问题求解 | MathVista, MathVerse, We-Math |
| 多学科知识 | 学科专业知识 | ScienceQA, MMMU, CMMU, CMMMU |
| 多语言 | 非英语语境下的能力 | MTVQA, M3Exam, CMMMU, Video-MME |
| 指令遵循 | 复杂分层指令服从性 | MIA-Bench |
| 多轮问答 | 长上下文对话能力 | ConvBench, MMDU |
| 多图理解 | 跨图聚合/比较 | NLVR2, MIRB, MuirBench, Mementos, SparklesEval |
| 交错图文 | 图文混排输入 | MMMU, VEGA |
| 高分辨率 | 大图/小目标识别 | V*Bench, MME-RealWorld |
| 视觉定位 | 短语→区域定位 | RefCOCO/+/g, Ref-L4 |
| 细粒度感知 | 子类/属性级识别 | FOCI, MMVP, LLVisionQA |
| 视频理解 | 时序/长视频/第一人称 | Video-MME, MVBench, MMBench-Video, MLVU, LVBench, EgoSchema, TempCompass |
| 组合性推理 | 属性绑定、关系组合 | Winoground, ARO, SugarCrepe |
| 跨模态检索 | 图文匹配/检索 | COCO/Flickr Retrieval, Urban-1k, DCI |
| 音频/语音 | 听觉模态理解 | AIR-Bench, MMAU, Dynamic-SUPERB |
| 3D / 点云 | 三维空间理解 | ScanQA, SQA3D, 3D-Bench |
| 物理常识 | 直觉物理 / 世界模型 | Physion, PhysBench |

#### B. 模型自分析（Model Self-Analysis）

不评"会不会做"，而评"做得是否可靠"。

| 子类 | 含义 | 代表基准 |
|---|---|---|
| 幻觉 | 输出与视觉内容不一致 | POPE, M-HalDetect, AMBER, VideoHallucer, HallusionBench, Bingo, VLind-Bench；合成图像类 PhD, MHaluBench, VHTest, OpenCHAIR |
| 偏见 | 社会/地区/虚假相关偏见 | VLBiasBench, Bingo, MM-SpuBench |
| 安全性 | OOD / 对抗 / 越狱 / 过度敏感 | OODCV-VQA, Sketchy-VQA, MultiTrust, AttackVLM, AdvDiffVLM, VLLM-safety-benchmark, MOSSBench |
| 因果推理 | 因果关系理解 | CELLO |
| 文化 / 价值观对齐 | 跨文化常识与价值 | CulturalVQA, GlobalBench, CVQA |
| 数据污染 / 泄漏 | 训练集与评测集重叠 | MMStar, contamination probing 系列 |

#### C. 扩展应用（Extended Applications）

垂直领域知识 + 实用任务。

| 子领域 | 代表基准 |
|---|---|
| 医学图像 | VQA-RAD, PathVQA, SLAKE, PMC-VQA, RadBench, GMAI-MMBench, OmniMedVQA |
| 情感分析 | EmoBench, FABA-Bench |
| 遥感 | RSVQA, RSIEval, VRSBench |
| 智能体 Agent | AppAgent, Mobile-Eval, GPT4Tools |
| 代码生成 | ChartMimic, WCGB |
| GUI | RefExp, Widget Captioning, Screen2Words, ScreenQA, Rico-semantics |
| 迁移能力 | VLAA, BenchLMM, MMCBench |
| 知识编辑 | MMEdit, VLKEB |
| 具身智能 | EQA, EPIC-KITCHENS, Ego4D, EMQA, SQA3D, MoTIF, EgoTaskQA, EmbodiedScan, RH20T-P |
| 自动驾驶 | BDD-X, HAD, Talk2Car, Rank2Tell, DRAMA, NuScenes-QA, DriveLM, Reason2Drive |
| 工具使用 / 函数调用 | 通用工具/API 调用 | MMTool-Bench 类 |
| 生成质量 | 统一理解-生成模型的图像/视频生成 | GenAI-Bench, T2I-CompBench, VBench, EvalCrafter |

#### D. 工程与系统级评估

| 子类 | 关注点 |
|---|---|
| 效率 / 部署成本 | 延迟、吞吐、显存、TTFT、长上下文成本 |
| 流式 / 实时交互 | 直播视频问答、实时语音对话（StreamingBench, OmniBench-Streaming） |

---

### 1.2 怎么考：评估方法论的维度

| 维度 | 选项 |
|---|---|
| 数据构建 | 人工标注 / 半自动 / 模型合成（如 DALL-E 生成） |
| 题型 | 是非题 / 多选题 / 开放式生成 |
| 裁判 | 人工评估 / LLM-as-Judge / 脚本规则匹配 |
| 指标 | 确定性指标（Acc, F1, BLEU, CIDEr, IoU…） / 非确定性指标（GPT 打分, Elo, Arena 对决） |
| 评估粒度 | 最终答案级 / 过程链级（推理步骤评分，PRM 风格） |
| 相对 vs 绝对 | 绝对分数 / 盲对决排名（WildVision-Arena, MM-Arena） |
| 工具包 | VLMEvalKit, LMMs-Eval, MultiMedEval, AgentStudio |

---

### 1.3 速查图谱

```
多模态评估
├── 考什么
│   ├── 基础能力（综合 / OCR / 图表 / 数学 / 多学科 / 多语言 /
│   │   指令 / 多轮 / 多图 / 交错 / 高分辨率 / 定位 / 细粒度 /
│   │   视频 / 组合性 / 检索 / 音频 / 3D / 物理）
│   ├── 模型自分析（幻觉 / 偏见 / 安全 / 因果 / 文化 / 数据污染）
│   ├── 扩展应用（医疗 / 情感 / 遥感 / Agent / 代码 / GUI /
│   │   迁移 / 知识编辑 / 具身 / 自动驾驶 / 工具调用 / 生成）
│   └── 工程系统（效率 / 流式实时）
└── 怎么考
    ├── 数据：人工 / 半自动 / 合成
    ├── 题型：是非 / 多选 / 开放
    ├── 裁判：人 / LLM / 脚本
    ├── 指标：确定性 / 非确定性
    ├── 粒度：答案级 / 过程级
    ├── 形式：绝对分 / Arena 对决
    └── 工具：VLMEvalKit / LMMs-Eval / ...
```

---

### 1.4 概念辨析

> 对类别中容易混淆的概念做深入说明。

#### 1.4.1 什么是「模型自分析（Model Self-Analysis）」

「Model Self-Analysis」字面是"模型自我分析"，但准确的理解是 **"对模型自身（内在属性）的分析性评估"**——评估对象不是模型外在的"任务能力"，而是模型自身的**性质、倾向、缺陷**。

**与"基础能力评估"的关键区别：**

| 对比维度 | 基础能力评估 | 模型自分析 |
|---|---|---|
| 回答的问题 | 模型**会不会做**这个任务？ | 模型**做得是否可靠**？ |
| 评估视角 | 任务导向（task-oriented） | 模型导向（model-oriented） |
| 关心的指标 | 准确率、能力上限 | 一致性、可信度、鲁棒性、公平性 |
| 典型问题 | "能不能识别这张图里的字？" | "会不会编造图里没有的内容？有没有性别偏见？被攻击会不会失败？" |
| 正面 vs 负面 | 测"做对了多少" | 测"会在什么情况下出问题" |

**直观比喻：**
- **基础能力评估** = 给学生出考卷，看他**能考多少分**。
- **模型自分析** = 心理学家研究这个学生：会不会**作弊（幻觉）**？有没有**刻板印象（偏见）**？**抗压能力（鲁棒性/越狱）**如何？**因果思维（因果推理）**成熟吗？

**为什么单独分一类：** 随着 MLLM 进入实际部署（医疗、自动驾驶、客服），"能力强"已经不够，必须知道模型会不会在关键场合出错、被骗、歧视用户。所以"模型自分析"是一类**面向可信部署**的评估，与"能力评估"互补。

---

## 2. 基准测试的构建

构建一个多模态评估基准通常涉及两条主线：**数据收集**（从哪儿来）与**标注**（QA 怎么造），并在过程中规避若干典型陷阱。

### 2.1 数据收集（Data Collection）

| 方式 | 做法 | 优点 | 缺点 / 风险 | 代表 |
|---|---|---|---|---|
| 复用已有数据集 | 直接从公开数据集中抽样 | 成本低、规模大 | **数据泄漏风险高**（与训练集重叠） | LVLM-eHub, MMT-Bench, SEED-Bench-2 |
| 修改已有数据 | 对原图加噪/遮挡/编辑 | 可针对性测试特定能力 | 修改流程劳动密集 | VCR（遮挡文字）、MMCBench（加噪）、HallusionBench（手工改图） |
| 互联网采集 | 从网页/媒体新爬 | 几乎无泄漏；时新性强 | 人工成本高、版权与隐私问题 | ScienceQA, MMBench, MME-RealWorld |

> 实践常**混合使用**：先互联网爬取保证新鲜度，再补充已有数据集中难度合适的样本。

### 2.2 标注方法（Annotation）

按"人 vs 模型"参与度分三类，**成本依次升高，质量也依次升高**：

| 方法 | 做法 | 适用场景 | 局限 |
|---|---|---|---|
| **自动构建** | 用模板/规则从已有标注重写 QA | 大规模、确定性任务 | 多样性差、易暴露模板痕迹 |
| **LLM / MLLM 辅助生成** | 让大模型出题 + 人工抽查 | 中等规模、开放题 | 受限于裁判模型能力，会引入噪声 |
| **纯人工标注** | 众包/专家手写 QA | 高质量、对抗性、细粒度任务 | 成本高、规模受限 |

代表性纯人工基准：VQA v2, VizWiz, TextVQA, MMBench, MME, Video-MME, MME-RealWorld（迄今最大纯人工标注，32 名标注员，29K QA）。

> 现代主流路线是 **"模型生成 + 人工审核"** 的混合范式——兼顾规模与质量。

### 2.3 常见挑战与陷阱

构建过程中必须规避的几个"坑"：

1. **多选题泄漏（MCQ Shortcut）**——选项本身可能泄露答案；模型不看图就能猜对。
   - 缓解：要求输出推理链；设计更具迷惑性的负选项；改用开放生成。
2. **数据污染 / 泄漏（Data Contamination）**——评测题在训练集中出现过。
   - 缓解：避开常见公开数据集；做污染探针（contamination probing）；对样本做扰动。
3. **缺乏视觉中心性（Vision-Centric）**——纯文本就能答对，违背"多模态评估"初衷。
   - 缓解：人工筛掉"看不看图都能答"的题；MMStar、MMEvalPro、CV-Bench、Video-MME 都做了这种过滤。
4. **多样性与规模不足**——题量小则统计不显著；任务单一则掩盖能力短板。
   - 缓解：扩规模；覆盖多领域、多粒度、多模态组合。

### 2.4 补充：综述未明确强调但同样重要的实践

5. **难度梯度设计**——按 easy/medium/hard 分层，避免"全难"或"全易"导致区分度差。
6. **答案分布平衡**——是非题中 yes/no 比例、选择题中各选项位置比例需打散，防止位置偏倚（POPE 即此类设计）。
7. **对照组 / 反事实样本**——配对的"看似相同但答案相反"题（HallusionBench、Winoground 风格），暴露模型走捷径行为。
8. **文化与语言覆盖**——避免英语/西方文化垄断，引入跨地区/跨语言样本。
9. **版权、隐私与伦理审查**——人脸、医疗影像等敏感数据需脱敏与合规审查。
10. **可重复性与版本化**——固定随机种子、记录数据版本（v1/v2）、提供官方评分脚本，便于横向对比。
11. **动态 / 防记忆机制**——题库定期轮换或私有保留测试集（held-out），防止刷榜过拟合。

### 2.5 速查图谱

```
基准测试构建
├── 数据收集
│   ├── 复用已有 → 泄漏风险
│   ├── 修改已有 → 针对性强
│   └── 互联网采集 → 新鲜但贵
├── 标注
│   ├── 自动模板（最便宜，最易模板化）
│   ├── LLM/MLLM 生成 + 人工审核（主流）
│   └── 纯人工（最贵，最高质量）
└── 必须规避
    ├── MCQ 泄漏
    ├── 数据污染
    ├── 非视觉中心
    ├── 规模/多样性不足
    ├── 难度无梯度
    ├── 答案分布失衡
    ├── 缺对照样本
    ├── 文化/语言偏窄
    ├── 版权/隐私
    └── 不可复现 / 易刷榜
```

---

## 3. 评估与裁判

构建好基准之后，下一步是**怎么打分**。这一环节包含三个互相耦合的问题：**谁来当裁判**、**用什么指标**、**用什么工具落地**。

### 3.1 谁来当裁判：三种评估范式

成本从高到低：**人工 > LLM/MLLM > 脚本**；可靠性与可扩展性是反向的权衡。

| 范式 | 做法 | 优点 | 缺点 | 代表 |
|---|---|---|---|---|
| **人工评估** | 众包 / 专家直接打分或投票 | 最贴近"为人服务"的最终目标；处理开放生成最准 | 慢、贵；少量评审员易引入个人偏好 | Screen2Words（MTurk）, Bingo, M-HalDetect, WV-Arena（人工投票 + Elo） |
| **LLM/MLLM 裁判** | 让大模型对响应打分或匹配 | 可扩展；适合开放式答案 | 系统性偏差（位置偏倚、长度偏倚、自偏）；受裁判模型能力上限制约 | 见下表两类参与度 |
| **脚本规则** | 正则匹配 / 精确匹配 | 快、便宜、可复现 | 依赖输出规范化；指令遵循差时失效 | MME 系列 |

**LLM/MLLM 裁判的两档参与度：**

| 档次 | 角色 | 例子 |
|---|---|---|
| 浅层参与（提取器） | 仅做"从模型自由文本里抽出选项 ABCD" | MMBench（GPT-4 提取选项）、BLINK（GPT-3.5-turbo） |
| 完全负责（评分者） | 对开放生成直接打分 / 对比参考答案 | MM-Vet, TouchStone, LLaVA-Bench（均用 GPT-4 评分） |

> **LLM 裁判的已知偏差**：①位置偏倚（更偏好 A 选项或先呈现的答案）；②长度偏倚（更长答案分更高）；③自偏（裁判模型偏爱自家家族输出）；④风格偏倚。缓解：随机打乱顺序、双向对比、多裁判投票、CoT 提示。

### 3.2 用什么指标：确定性 vs 非确定性

**确定性指标（Deterministic）**——结果唯一、可机器自动核对。

| 指标 | 计算 | 适用 |
|---|---|---|
| Exact Match (EM) | 完全字面匹配真值 | WebSRC, FakeBench, VCR |
| Accuracy | 选项/答案正确率 | MME, Video-MME, MTVQA |
| F1 | 精确率/召回率调和均值 | ScreenQA, ScreenAI |
| mAP / IoU | 定位质量 | LAMM, RefExp, RefCOCO |
| BLEU / CIDEr / ANLS | 生成文本与参考相似度 | VizWiz, Captioning |
| Log Probability | 取首 token 概率最高的选项 | 概率对齐评估 |
| CircularEval | 多次轮转选项位置后取一致性 | 抗位置偏倚 |
| ADRScore / Lingo-Judge | 训练分类器评估 CoT/正确性 | 推理链质量 |

**非确定性指标（Non-deterministic）**——结果不唯一，依赖人或大模型主观判定。

| 类别 | 做法 | 代表 |
|---|---|---|
| 评分（Scoring） | 按维度打 1–10 分（有用性、准确性、详尽度…） | LLaVA-Bench, MM-Vet |
| 对决比较（Pairwise） | 在两个模型输出间选优 | TouchStone（GPT-4 对比） |
| 胜率（Win-rate） | 多次对决中获胜比例 | Arena 风格 |
| Elo 排名 | 不只看输赢，按对手实力加权 | Chatbot Arena, WildVision-Arena |

> 选型经验：**封闭题（MCQ/是非）→ 确定性指标 + 脚本**；**开放生成 → 非确定性指标 + LLM/人工**；**多模型横向比较 → 优先 Arena 对决而非绝对分**。

### 3.3 评估工具包（Toolkits）

把"基准 + 模型 + 指标 + 裁判"打包，方便一键评测、横向对比、提交排行榜。

| 工具包 | 定位 | 关键特性 |
|---|---|---|
| **VLMEvalKit** | 通用 MLLM 评测主力 | 70+ 模型、20+ 基准；统一 `.generate()` 接口；公开 leaderboard；多 GPU + API 并行 |
| **LMMs-Eval** | 50+ 任务的统一框架 | 一键跑分；**LMMs-Eval lite** 通过 k-center 选子集省算力；**LiveBench** 月度更新防污染 |
| **MultiMedEval** | 医学专用 | 6 类医学任务、23 数据集、11 种医学模态；PyPI 安装 |
| **AgentStudio** | 多模态智能体 | 真实环境、跨 OS；函数调用 + 鼠键控制；77 个真实任务；三档难度 |

**未来缺口**：①跨模态格式标准（视频帧选取、音频对齐）尚不统一；②智能体评估仍重人工、场景多样性不足；③具身/医学类专用工具包仍稀缺。

### 3.4 实践选型决策树（补充）

```
任务类型？
├── 是非 / 多选
│   └── 脚本匹配 + Accuracy（注意答案位置打散）
├── 短答 / 抽取
│   └── 浅层 LLM 提取器 + EM/F1
├── 开放生成（描述、对话、写代码）
│   ├── 单模型质量评估 → LLM-as-Judge 评分（多维度 + CoT）
│   └── 多模型横向对比 → 配对对决 + Elo / Win-rate
├── 定位 / 检测
│   └── mAP / IoU
├── 推理链质量
│   └── 过程级评分（PRM 风格）+ ADRScore
└── 长期跑榜 / 防污染
    └── 动态题库（LiveBench 类） + 私有 held-out
```

### 3.5 三个原则（避坑）

1. **裁判一致性优先于裁判先进性**——同一基准应固定一种裁判与提示，否则跨论文不可比。
2. **多裁判交叉验证**——人工 vs LLM、不同 LLM 之间的相关性必须报告（Pearson/Spearman/Kendall）。
3. **报告分布而非点估计**——给出方差、置信区间或多次跑分均值，避免单次运行的随机性误导结论。

### 3.6 MLLM 评估的实际运作机制（FAQ）

> 一个常见困惑：MLLM 输出是自由文本，同一答案可有无数种表述方式（`"3"`、`"There are 3 cats."`、`"I see three cats"` 都对），那 benchmark 究竟怎么打分？是不是都靠 LLM-as-Judge？

**答案：不是。** ImageNet 那套"对就是对、不对就是不对"的范式确实失效了，但社区的解决思路**不是放任开放生成**，而是**反向把答案空间压回封闭集**，让脚本仍能打分。LLM-as-Judge 只在真正必须开放的场景才出场。

#### 实际有四条评估路线

| 路线 | 题型 | 评分方式 | 代表 |
|---|---|---|---|
| **A. 限定题型 + 脚本打分**（最主流） | MCQ / Yes-No / 单词短语 | 正则提取后做 EM / Accuracy | MME, MMBench, SEED-Bench, MMMU 几乎所有综合 benchmark |
| **B. 限定题型 + LLM 提取器**（兜底） | 同上，但模型输出啰嗦 | 让小 LLM 抽出答案 → 仍走脚本 EM | MMBench（GPT-4 抽选项）、BLINK |
| **C. 参考答案 + LLM 评分**（真开放） | 描述、对话、解释 | LLM-as-Judge 打 1–10 分 / 配对对决 | MM-Vet, TouchStone, LLaVA-Bench |
| **D. 程序化验证**（可执行） | 代码、定位、数学、Agent、检索 | 跑代码 / 算 IoU / 数值匹配 / 任务完成判定 | ChartMimic, RefCOCO, AgentStudio |

#### 关键技巧：用 Prompt 锁死答案空间

路线 A/B 之所以成立，是因为**题目 prompt 主动施加约束**，常见模板：

```
Answer with the option's letter from the given choices directly.
Please answer Yes or No.
Please answer with a single word or phrase.
```

模型即使生成 `"The answer is B."` 也无所谓，正则一抠就拿到 `B`。

#### 三个常见误解的澄清

1. **"benchmark 答案不唯一" → 错。** 多数 benchmark **故意**设计为唯一答案（MCQ 最便宜），是 prompt 在硬约束输出格式。
2. **"开放生成必然要 LLM-as-Judge" → 部分错。** 凡是能程序化验证的（代码、定位、检索、数学），尽量不用 Judge，因为 Judge 不稳定。
3. **"LLM-as-Judge 是主流" → 错。** 主流仍是路线 A。Judge 只在真正开放的对话/描述类任务上不可避免。

#### 一句话类比

| 范式 | 类比 |
|---|---|
| ImageNet | 选择题 + 答题卡（涂错就错） |
| MLLM 路线 A/B | **简答题但给了选项 ABCD**——学生答一段话也行，老师只看最终选了哪个 |
| MLLM 路线 C | 作文题——必须人工或 LLM 阅卷打分 |
| MLLM 路线 D | 编程题——直接跑测试用例 |

> **核心结论**：绝大多数 MLLM benchmark 不是"开放问答 + LLM 阅卷"，而是"**用 prompt 把答案空间压缩到封闭集，再用脚本打分**"。LLM-as-Judge 是兜底，不是主力。

