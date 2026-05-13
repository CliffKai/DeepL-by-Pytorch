**偏好数据 → 偏好建模 → Reward Model → RLHF/PPO → DPO → 其他偏好优化 → GRPO/RLVR**

---

# 一、偏好学习中应该掌握什么能力

1. **读懂 preference pair 数据**
   能理解一个样本通常由 `prompt / chosen / rejected` 构成，并知道它表达的是“相对偏好”，不是绝对真理。

2. **理解 chosen / rejected 的建模方式**
   知道 chosen 不一定完美，rejected 也不一定完全错误，它们只表示在某个标注标准下 chosen 被认为更好。

3. **掌握 Bradley-Terry 模型**
   能理解为什么偏好概率可以写成：

$$
P(y_{w} \succ y_{l} \mid x)
=\sigma\big(r(x,y_{w}) - r(x,y_{l})\big)
$$

4. **理解 Reward Model 与 Preference Model 的关系**
   Reward Model 给单个 response 打 scalar reward；Preference Model 预测两个 response 谁更好。二者可通过 reward difference 联系起来。

5. **理解 DPO、IPO、KTO、ORPO、SimPO 等方法的共同思想与差异**
   它们大多试图绕开 PPO-RLHF 的复杂性，直接用离线偏好数据优化 policy，但对 reward、reference model、pairwise 数据的依赖不同。

6. **从偏好学习自然过渡到 RLHF 和 PPO**
   知道 RLHF pipeline 是：
$$
\text{SFT} \rightarrow \text{Preference Data} \rightarrow \text{Reward Model} \rightarrow \text{PPO/RL}
$$

7. **自己实现最小版 DPO loss**

8. **训练一个简单 Reward Model**
   包括加 scalar head、计算 chosen/rejected reward、使用 pairwise ranking loss、评估 preference accuracy。

>（7、8 待定）

RLHF 的经典代表是 InstructGPT：先做监督微调，再收集人类排序数据训练 reward model，最后用 PPO 进一步优化模型；DPO 则提出用一个分类式目标直接从偏好数据优化 policy，绕过显式 reward model 和 PPO。([arXiv][1])

---

# 二、知识结构：从基础到进阶

## 1. 为什么 LLM 需要偏好学习？

### SFT 的局限

SFT 学的是：

$$
\max_\theta \log \pi_\theta(y_{\text{demo}} \mid x)
$$
也就是让模型模仿标注答案。SFT（Supervised Fine-Tuning）学的就是：给定输入 \(x\)，让模型尽可能生成示范答案 \(y_{\text{demo}}\)。更完整一点，如果数据集是：

$$
\mathcal{D}_{\text{SFT}}
=\{(x_i,y_i)\}_{i=1}^N
$$

那么 SFT 的目标是：

$$
\max_\theta
\sum_{i=1}^N
\log \pi_\theta(y_i \mid x_i)
$$

或者写成期望形式：

$$
\max_\theta
\mathbb{E}_{(x,y)\sim \mathcal{D}_{\text{SFT}}}
\left[
\log \pi_\theta(y \mid x)
\right]
$$

因为语言模型是逐 token 生成的，所以：

$$
\pi_\theta(y \mid x)
=\prod_{t=1}^{T}
\pi_\theta(y_t \mid x,y_{<t})
$$

取 log 后：

$$
\log \pi_\theta(y \mid x)
=\sum_{t=1}^{T}
\log \pi_\theta(y_t \mid x,y_{<t})
$$

所以 SFT 实际上是在最大化每个示范答案 token 的概率：

$$
\max_\theta
\sum_{t=1}^{T}
\log \pi_\theta(y_t \mid x,y_{<t})
$$

训练时通常写成最小化 loss，也就是负对数似然：

$$
\mathcal{L}_{\text{SFT}}
=-
\log \pi_\theta(y_{\text{demo}} \mid x)
$$

或展开成：

$$
\mathcal{L}_{\text{SFT}}
=-
\sum_{t=1}^{T}
\log \pi_\theta(y_t \mid x,y_{<t})
$$

也就是常说的 **cross-entropy loss**。

所以一句话：

> SFT 就是在优化模型参数 \(\theta\)，让模型在给定 prompt \(x\) 时，更高概率地输出人工示范答案 \(y_{\text{demo}}\)。

问题是：

| 问题           | 解释                    |
| ------------ | --------------------- |
| 只学“应该怎么说”    | 没直接学“哪个答案更好”          |
| 对多候选答案缺乏比较能力 | 同一个 prompt 下可能有多个合理回答 |
| 标注答案不是唯一最优   | SFT 会过度模仿风格、长度、模板     |
| 无法表达细粒度偏好    | 比如“更有帮助、更安全、更诚实、更简洁”  |

### “模仿好答案”不等于“知道哪个答案更好”

给定 prompt：

```text
请解释 Transformer 的 attention。
```

response A：准确、结构清楚、有公式。
response B：大体正确，但啰嗦、术语混乱。
response C：短小但不完整。

SFT 通常只看到一个标准答案，它不知道 A 比 B 好多少，也不知道 B 比 C 好在哪里。

偏好学习直接学习：

$$
A \succ B,\quad B \succ C
$$

这比单纯模仿一个答案更接近 alignment。

### 偏好学习在 alignment 中解决什么问题？

它把“人类更喜欢什么”转化成可优化信号：

$$
(x, y_w, y_l)
$$

其中 $(y_w)$ 是 winner/chosen，$(y_l)$ 是 loser/rejected。

这使模型不只是学会生成答案，而是学会区分：

* 更 helpful 的答案；
* 更 harmless 的答案；
* 更 honest 的答案；
* 更符合指令的答案；
* 更符合人类审美和沟通习惯的答案。

---

## 2. 偏好数据是什么？

### 标准 pairwise preference 数据

```json
{
  "prompt": "解释一下什么是 dropout。",
  "chosen": "Dropout 是一种正则化方法，训练时随机置零部分神经元输出，以减少过拟合。",
  "rejected": "Dropout 是一种让模型变大的方法。"
}
```

TRL 当前也支持标准格式和对话格式的 preference dataset，例如显式 prompt 格式 `{prompt, chosen, rejected}`，或隐式 prompt 格式 `{chosen, rejected}`。([GitHub][2])

>  TRL 指的是 “Training with Reinforcement Learning from Human Feedback”

### 常见偏好信号

| 类型                  | 例子                 | 特点                    |
| ------------------- | ------------------ | --------------------- |
| pairwise preference | A 比 B 好            | DPO/RM 最常见            |
| ranking preference  | A > B > C > D      | 可转成多个 pair            |
| binary feedback     | good / bad         | KTO 可用                |
| scalar reward       | 4.5 / 5            | 可用于 reward regression |
| human preference    | 人类标注               | 更贴近人类价值，但贵且有噪声        |
| AI preference       | GPT-4/Claude/RM 打分 | 便宜、可扩展，但有偏差           |

> KTO 是 “KL-Teacher Optimization” 

### chosen / rejected 的关键理解

不要把 chosen 理解成“完美答案”，也不要把 rejected 理解成“垃圾答案”。

更准确地说：

$$
y_w \succ y_l
$$

表示在某个标注规则下，$(y_w)$ 相对 $(y_l)$ 更受偏好。

---

## 3. 偏好建模基础

### Reward score

Reward Model 给一个 prompt-response 对打分：

$$
r_\phi(x,y) \in \mathbb{R}
$$

例如：

```text
r(prompt, chosen) = 3.2
r(prompt, rejected) = 1.1
```

### Reward margin

$$
\Delta r = r_\phi(x,y_w)-r_\phi(x,y_l)
$$

如果 $\Delta r > 0$，说明模型认为 chosen 更好。

### Bradley-Terry 模型

Bradley-Terry 假设两个候选的偏好概率由 reward 差决定：

$$
P(y_w \succ y_l \mid x)
=\frac{\exp(r(x,y_w))}
{\exp(r(x,y_w))+\exp(r(x,y_l))}
$$

等价于：

$$
P(y_w \succ y_l \mid x)
=\sigma(r(x,y_w)-r(x,y_l))
$$

其中：

$$
\sigma(z) = \frac{1}{1+\exp(-z)}
$$

直觉是：

* reward 差越大，chosen 被偏好的概率越接近 1；
* reward 差为 0，偏好概率为 0.5；
* reward 差为负，模型认为 rejected 更好。

---

## 4. Reward Model

### 输入输出

输入：

```text
[prompt] + [response]
```

输出：

$$
r_\phi(x,y)
$$

通常是一个 scalar。

模型结构：

```text
Transformer backbone
        ↓
last hidden state / EOS hidden state
        ↓
linear reward head
        ↓
scalar reward
```

### Reward Model pairwise loss

给定：

$$
r_w = r_\phi(x,y_w), \quad r_l = r_\phi(x,y_l)
$$

loss：

$$
\mathcal{L}_{RM}
=-\log \sigma(r_w - r_l)
$$

PyTorch 写法：

```python
import torch
import torch.nn.functional as F

def reward_model_loss(chosen_rewards, rejected_rewards):
    """
    chosen_rewards:   [batch]
    rejected_rewards: [batch]
    """
    diff = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(diff).mean()
    return loss
```

### Reward Model 和 classifier 的区别

| 维度             | Reward Model                    | Binary Classifier |
| -------------- | ------------------------------- | ----------------- |
| 输出             | scalar reward                   | 类别概率              |
| 训练目标           | chosen reward > rejected reward | 判断好/坏             |
| 可比较任意 response | 可以                              | 不一定               |
| 分数是否绝对可靠       | 不可靠                             | 概率也常不可靠           |
| 常用于 RL         | 是                               | 较少直接用于 PPO        |

Reward Model 学到的是一个相对偏好分数。它的绝对值没有天然意义，主要看差值：

$$
r(x,y_1)-r(x,y_2)
$$

### Reward overfitting

Reward Model 可能记住标注数据里的浅层模式，例如：

* 回答越长 reward 越高；
* 有条理的 markdown 总是高分；
* 礼貌语气过度加分；
* 安全拒答模板被过度奖励。

### Reward hacking

当 policy 被训练去最大化 reward model 时，它可能找到 reward model 的漏洞，而不是真正变好。

例如 reward model 偏好长答案，policy 就可能生成冗长但空洞的回答。



---

## 5. 从 Reward Model 到 RLHF

### RLHF pipeline

经典 RLHF 流程：

```text
Pretrained LM
    ↓
SFT on demonstrations
    ↓
Collect preference data
    ↓
Train Reward Model
    ↓
Use PPO/RL to optimize policy
    ↓
Aligned policy
```

InstructGPT 正是这种路线：先用 demonstration 做 SFT，再用 labeler 排序数据训练 reward model，最后用 PPO 优化策略。([arXiv][1])

### 四个模型分别是什么？

| 模型                                 | 作用                               |
| ---------------------------------- | -------------------------------- |
| policy model $(\pi_\theta)$          | 当前要训练的模型                         |
| reference model $(\pi_{\text{ref}})$ | 通常是 SFT 模型，用来约束 policy 不要漂移太远    |
| reward model $(r_\phi)$              | 给生成结果打 reward                    |
| value model $(V_\psi)$               | PPO 中估计 state/value，用于 advantage |

### KL penalty

RLHF 常优化：

$$
\max_\pi
\mathbb{E}_{x,y\sim \pi}
\left[
r_\phi(x,y)
-\beta
D_{KL}
\left(
\pi(\cdot|x)
\Vert
\pi_{\text{ref}}(\cdot|x)
\right)
\right]
$$

其中：

* $(r_\phi(x,y))$：reward model 分数；
* $(\pi)$：当前 policy；
* $(\pi_{\text{ref}})$：reference model；
* $(\beta)$：KL 惩罚强度。

### 为什么要限制偏离 reference model？

因为 reference model 通常是 SFT 后的稳定模型，它代表“语言质量、基本指令跟随、预训练能力”的基准。

KL penalty 的作用是：

```text
追求更高 reward
但不要为了 reward 牺牲语言质量和通用能力
```

PPO 本身是一类 policy gradient 方法，通过限制更新幅度提升训练稳定性；RLHF 中常用 PPO，是因为它可以在 reward model 提供的标量奖励下优化生成策略。([arXiv][3])

---

# 三、DPO

## 1. DPO 想解决 PPO-RLHF 的什么问题？

PPO-RLHF 复杂在于：

| 难点                | 说明                            |
| ----------------- | ----------------------------- |
| 需要 reward model   | 训练、调参、验证都不简单                  |
| 需要 value model    | PPO 需要 advantage/value        |
| 需要 online rollout | 训练时要采样生成                      |
| 训练不稳定             | reward hacking、KL 崩溃、长度偏置     |
| 工程复杂              | policy/ref/reward/value 多模型协同 |

> PPO 的 online 指的是：训练时会让当前 policy 针对 prompt 现场生成回答，并用 reward model 评分后更新 policy；更新后的 policy 又继续生成新的回答，如此循环。在大模型 RLHF 里的 PPO，通常是先让模型针对一批 prompts 完整生成回答，然后 reward model 给完整回答打分，再基于整段生成轨迹计算 loss，最后更新 policy。

DPO 的目标是：

```text
不用显式 reward model
不用 PPO
不用 online rollout
直接用 preference pairs 优化 policy
```

DPO 论文提出了一种 reward 与 optimal policy 的重参数化，使标准 KL-constrained RLHF 问题可以转成一个简单的二分类式 loss。([arXiv][4])

---

## 2. DPO 的核心思想

DPO 不显式学习：

$$
r_\phi(x,y)
$$

而是用 policy 和 reference model 的 logprob 比值定义一个 **implicit reward**：

$$
r_\theta(x,y)
=\beta
\log
\frac{\pi_\theta(y \mid x)}
{\pi_{\text{ref}}(y \mid x)}
+
\beta \log Z(x)
$$

因为在偏好比较里只看 reward difference，$(\log Z(x))$ 会抵消。

于是：

$$
r_\theta(x,y_w)-r_\theta(x,y_l)
=\beta
\left[
\log
\frac{\pi_\theta(y_w \mid x)}
{\pi_{\text{ref}}(y_w \mid x)}
-\log
\frac{\pi_\theta(y_l \mid x)}
{\pi_{\text{ref}}(y_l \mid x)}
\right]
$$

DPO 直接让 chosen 的 implicit reward 高于 rejected。

---
## 3. DPO loss

定义：

$$
\Delta_\theta
=\log \pi_\theta(y_w \mid x)
-\log \pi_\theta(y_l \mid x)
$$

$$
\Delta_{\text{ref}}
=\log \pi_{\text{ref}}(y_w \mid x)
-\log \pi_{\text{ref}}(y_l \mid x)
$$

DPO logits：

$$
z
=\beta
(\Delta_\theta-\Delta_{\text{ref}})
$$

loss：

$$
\mathcal{L}_{DPO}
=-\log \sigma(z)
$$

也就是：

$$
\mathcal{L}_{DPO}
=-\log
\sigma
\left(
\beta
\left[
\log
\frac{\pi_\theta(y_w \mid x)}
{\pi_\theta(y_l \mid x)}
-\log
\frac{\pi_{\text{ref}}(y_w \mid x)}
{\pi_{\text{ref}}(y_l \mid x)}
\right]
\right)
$$

### 直觉解释

DPO 不是单纯让 chosen 概率变高，而是让：

$$
\text{policy 对 chosen 的相对偏好}>\text{reference 对 chosen 的相对偏好}
$$

也就是：

```text
如果 reference 已经认为 chosen 比 rejected 好，
policy 不需要无限放大这个差距。

如果 reference 分不清 chosen 和 rejected，
policy 应该学会拉开差距。
```

---

## 4. beta 的作用

$$
z=\beta(\Delta_\theta-\Delta_{\text{ref}})
$$

| beta | 效果               |
| ---- | ---------------- |
| 较大   | 更激进地放大偏好差异       |
| 较小   | 更保守，接近 reference |
| 过大   | 可能过拟合偏好数据        |
| 过小   | 学习信号太弱           |

实践中常见范围：

```text
0.01 ~ 0.5
```

具体取决于模型、数据质量、batch size、response 长度。

---

## 5. DPO 和 SFT 的区别

| 维度            | SFT             | DPO                       |
| ------------- | --------------- | ------------------------- |
| 数据            | prompt → answer | prompt → chosen/rejected  |
| 目标            | 增大标准答案概率        | 增大 chosen 相对 rejected 的偏好 |
| 是否需要 rejected | 不需要             | 需要                        |
| 是否用 reference | 通常不用            | 通常需要                      |
| 优化对象          | imitation       | preference alignment      |

SFT loss：

$$
-\log \pi_\theta(y_w|x)
$$

DPO loss：

$$
-\log \sigma\left(
\beta
\left[
(\log \pi_\theta(y_w)-\log \pi_\theta(y_l))
-(\log \pi_{\text{ref}}(y_w)-\log \pi_{\text{ref}}(y_l))
\right]
\right)
$$

---

## 6. DPO 和 Reward Model + PPO 的区别

| 维度              | RM + PPO RLHF | DPO      |
| --------------- | ------------- | -------- |
| reward model    | 需要            | 不显式需要    |
| value model     | 需要            | 不需要      |
| reference model | 需要            | 需要       |
| online rollout  | 需要            | 不需要      |
| 训练稳定性           | 更难            | 更简单      |
| 可探索新答案          | 强             | 弱，依赖离线数据 |
| 工程成本            | 高             | 低        |
| 适合场景            | 大规模在线对齐       | 离线偏好微调   |

---
# 四、DPO 数学推导

## Step 1：KL-constrained RLHF objective

从 RLHF 目标开始：

$$
\max_{\pi}
\mathbb{E}_{x\sim D,\; y\sim\pi(\cdot|x)}
\left[
r(x,y)
\right]
-\beta
D_{KL}
\left(
\pi(\cdot|x)
\parallel
\pi_{\text{ref}}(\cdot|x)
\right)
$$

对单个 prompt $begin:math:text$x$end:math:text$，写成：

$$
\max_{\pi}
\sum_y \pi(y|x)\, r(x,y)
-\beta
\sum_y \pi(y|x)
\log
\frac{\pi(y|x)}
{\pi_{\text{ref}}(y|x)}
$$

直觉：

```text
第一项：希望生成高 reward 的 response。
第二项：惩罚 policy 偏离 reference。
```

---

## Step 2：optimal policy 与 reward 的关系

这个优化问题的最优解满足：

$$
\pi^*(y|x)
=\frac{1}{Z(x)}
\pi_{\text{ref}}(y|x)
\exp
\left(
\frac{1}{\beta}r(x,y)
\right)
$$

其中：

$$
Z(x)=\sum_y
\pi_{\text{ref}}(y|x)
\exp
\left(
\frac{1}{\beta}r(x,y)
\right)
$$

是归一化项。

直觉：

```text
最优 policy = reference policy × reward 指数加权
```

reward 越高，该 response 的概率越会被放大。

---

## Step 3：把 reward 反解出来

由：

$$
\pi^*(y|x)
=\frac{1}{Z(x)}
\pi_{\text{ref}}(y|x)
\exp
\left(
\frac{1}{\beta}r(x,y)
\right)
$$

得到：

$$
r(x,y)
=\beta
\log
\frac{\pi^*(y|x)}
{\pi_{\text{ref}}(y|x)}
+
\beta \log Z(x)
$$

DPO 用当前 policy $begin:math:text$\\pi\_\\theta$end:math:text$ 近似 $begin:math:text$\\pi\^\*$end:math:text$：

$$
r_\theta(x,y)
=\beta
\log
\frac{\pi_\theta(y|x)}
{\pi_{\text{ref}}(y|x)}
+
\beta \log Z(x)
$$

这就是 implicit reward。

---

## Step 4：preference probability

根据 Bradley-Terry：

$$
P(y_w \succ y_l|x)
=\sigma(r(x,y_w)-r(x,y_l))
$$

代入 implicit reward：

$$
r_\theta(x,y_w)-r_\theta(x,y_l)
=\beta
\log
\frac{\pi_\theta(y_w|x)}
{\pi_{\text{ref}}(y_w|x)}
-\beta
\log
\frac{\pi_\theta(y_l|x)}
{\pi_{\text{ref}}(y_l|x)}
$$

整理：

$$
\beta
\left[
\log
\frac{\pi_\theta(y_w|x)}
{\pi_\theta(y_l|x)}
-\log
\frac{\pi_{\text{ref}}(y_w|x)}
{\pi_{\text{ref}}(y_l|x)}
\right]
$$

注意 $begin:math:text$\\log Z\(x\)$end:math:text$ 消失了，因为 chosen 和 rejected 共享同一个 prompt。

---

## Step 5：最终 DPO loss

最大化 chosen 被偏好的概率：

$$
\max_\theta
\log
\sigma
\left(
\beta
\left[
\log
\frac{\pi_\theta(y_w|x)}
{\pi_\theta(y_l|x)}
-\log
\frac{\pi_{\text{ref}}(y_w|x)}
{\pi_{\text{ref}}(y_l|x)}
\right]
\right)
$$

最小化负 log likelihood：

$$
\mathcal{L}_{DPO}
=-\mathbb{E}_{(x,y_w,y_l)\sim D}
\left[
\log
\sigma
\left(
\beta
\left[
\log
\frac{\pi_\theta(y_w|x)}
{\pi_\theta(y_l|x)}
-\log
\frac{\pi_{\text{ref}}(y_w|x)}
{\pi_{\text{ref}}(y_l|x)}
\right]
\right)
\right]
$$

DPO 的核心理论贡献就是把 KL-constrained RLHF 目标转成了这个离线 pairwise classification loss。([arXiv][4])

---

# 五、最小 PyTorch DPO loss

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
):
    """
    policy_chosen_logps:   [batch] 当前 policy 对 chosen response 的 log p(y|x)
    policy_rejected_logps: [batch] 当前 policy 对 rejected response 的 log p(y|x)
    ref_chosen_logps:      [batch] reference model 对 chosen response 的 log p(y|x)
    ref_rejected_logps:    [batch] reference model 对 rejected response 的 log p(y|x)
    beta: DPO 温度 / KL 控制系数
    """

    # policy 对 chosen 相比 rejected 的偏好强度
    policy_logratios = policy_chosen_logps - policy_rejected_logps

    # reference model 原本对 chosen 相比 rejected 的偏好强度
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # DPO 关注的是 policy 相对 reference 增加了多少偏好
    logits = beta * (policy_logratios - ref_logratios)

    # 最大化 log sigmoid(logits)，等价于最小化负 log likelihood
    losses = -F.logsigmoid(logits)

    return losses.mean()
```

逐行解释：

| 代码                                            | 含义                                      |
| --------------------------------------------- | --------------------------------------- |
| `policy_chosen_logps - policy_rejected_logps` | 当前 policy 认为 chosen 比 rejected 好多少      |
| `ref_chosen_logps - ref_rejected_logps`       | reference 原本认为 chosen 比 rejected 好多少    |
| `policy_logratios - ref_logratios`            | policy 相对 reference 的偏好提升               |
| `beta * (...)`                                | 控制偏好优化强度                                |
| `-F.logsigmoid(logits)`                       | pairwise preference classification loss |

---

# 六、如何计算 response logprob

关键点：只计算 response 部分，不计算 prompt 部分。

```python
import torch
import torch.nn.functional as F

def compute_response_logps(model, input_ids, attention_mask, response_mask):
    """
    input_ids:      [batch, seq_len]
    attention_mask: [batch, seq_len]
    response_mask:  [batch, seq_len] response token 为 1，prompt/pad 为 0

    返回:
    sequence_logps: [batch]
    """

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits  # [batch, seq_len, vocab]

    # causal LM: 第 t 个 logits 预测第 t+1 个 token
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_response_mask = response_mask[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)

    token_logps = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    # 只保留 response 部分
    token_logps = token_logps * shift_response_mask

    # response sequence log probability
    sequence_logps = token_logps.sum(dim=-1)

    return sequence_logps
```

注意：

```text
不要把 prompt logprob 算进 DPO/RM response logprob，
否则模型可能学到“prompt 更容易预测”而不是“response 更受偏好”。
```

---

# 七、Reward Model 代码骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # 取每个样本最后一个非 padding token
        lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        last_token_hidden = last_hidden[batch_indices, lengths]

        rewards = self.reward_head(last_token_hidden).squeeze(-1)

        return rewards


def reward_loss(chosen_rewards, rejected_rewards):
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

训练时：

```python
chosen_rewards = reward_model(
    chosen_input_ids,
    chosen_attention_mask,
)

rejected_rewards = reward_model(
    rejected_input_ids,
    rejected_attention_mask,
)

loss = reward_loss(chosen_rewards, rejected_rewards)
```

评估指标：

```python
accuracy = (chosen_rewards > rejected_rewards).float().mean()
margin = (chosen_rewards - rejected_rewards).mean()
```

---

# 八、简化 DPO 训练 loop

```python
from copy import deepcopy
import torch

policy_model.train()
ref_model = deepcopy(policy_model)
ref_model.eval()

for p in ref_model.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6)

for batch in dataloader:
    chosen = batch["chosen"]
    rejected = batch["rejected"]

    policy_chosen_logps = compute_response_logps(
        policy_model,
        chosen["input_ids"],
        chosen["attention_mask"],
        chosen["response_mask"],
    )

    policy_rejected_logps = compute_response_logps(
        policy_model,
        rejected["input_ids"],
        rejected["attention_mask"],
        rejected["response_mask"],
    )

    with torch.no_grad():
        ref_chosen_logps = compute_response_logps(
            ref_model,
            chosen["input_ids"],
            chosen["attention_mask"],
            chosen["response_mask"],
        )

        ref_rejected_logps = compute_response_logps(
            ref_model,
            rejected["input_ids"],
            rejected["attention_mask"],
            rejected["response_mask"],
        )

    loss = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=0.1,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

你应该记录：

```text
loss
chosen_logps
rejected_logps
policy_logratios
ref_logratios
implicit_reward_margin
```

---

# 九、TRL DPOTrainer 示例

TRL 官方文档支持 DPOTrainer，并说明 DPO 可直接用 preference dataset 训练语言模型；RewardTrainer 也支持 preference dataset，用于训练 reward model。([Hugging Face][5])

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

dataset = load_dataset(
    "trl-lib/ultrafeedback_binarized",
    split="train[:5000]",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

training_args = DPOConfig(
    output_dir="./qwen-dpo-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=1,
    beta=0.1,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    max_length=1024,
    max_prompt_length=512,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()
```

常见日志含义：

| 指标                 | 含义                          |
| ------------------ | --------------------------- |
| `loss`             | DPO loss                    |
| `rewards/chosen`   | chosen 的 implicit reward    |
| `rewards/rejected` | rejected 的 implicit reward  |
| `rewards/margins`  | chosen - rejected           |
| `logps/chosen`     | policy 对 chosen 的 logprob   |
| `logps/rejected`   | policy 对 rejected 的 logprob |

---

# 十、其他偏好优化方法系统比较

| 方法      | 核心思想                                           | 需要 RM | 需要 reference |  需要 pairwise | 需要 online rollout | 优点                | 局限               | 适合场景                  |
| ------- | ---------------------------------------------- | ----: | -----------: | -----------: | ----------------: | ----------------- | ---------------- | --------------------- |
| DPO     | 用 policy/ref logprob 构造 implicit reward        |     否 |            是 |            是 |                 否 | 简单稳定，工程成本低        | 依赖 pair 数据；可能过拟合 | 通用离线偏好优化              |
| IPO     | 从 (\Psi)PO 理论出发，缓解 DPO 对 BT 假设和极端偏好的问题         |     否 |            是 |            是 |                 否 | 理论上更稳健            | 实践生态弱于 DPO       | 噪声偏好、过拟合分析            |
| KTO     | 用 desirable/undesirable 二元反馈优化人类效用             |     否 |          通常是 |            否 |                 否 | 不需要 pairwise 数据   | 对标签质量敏感          | 只有正负反馈时               |
| ORPO    | 把 SFT 与 odds-ratio 偏好惩罚合并                      |     否 |            否 |            是 |                 否 | 单阶段、无 reference   | 对数据格式和超参敏感       | 低资源对齐                 |
| SimPO   | 用平均 logprob 作为 reference-free reward，并加 margin |     否 |            否 |            是 |                 否 | 更省显存，避免 reference | 长度归一和 margin 需调  | 想降低 DPO 显存成本          |
| RRHF    | 用候选响应排名约束模型 likelihood 顺序                      |     否 |            否 | ranking/pair |                 否 | 类 SFT，简单          | 强依赖候选质量          | 多候选排序数据               |
| SLiC-HF | 用 sequence likelihood calibration 学人类偏好        |     否 |            否 |            是 |                 否 | 比 PPO 简单，离线可用     | 不如 DPO 普及        | summarization / 离线 HF |

KTO 论文强调它可以从 desirable/undesirable 这类二元信号学习，而不要求成对偏好；ORPO 明确提出 reference-model-free 的单阶段 odds-ratio 优化；SimPO 用平均 log probability 作为 reference-free reward，并引入目标 margin；RRHF 和 SLiC-HF 都是比 PPO 更简单的离线偏好对齐路线。([arXiv][6])

---

# 十一、偏好学习和强化学习的关系

## 偏好学习是不是强化学习？

不一定。

| 内容              | 是否 RL                                 |
| --------------- | ------------------------------------- |
| Reward Model 训练 | 不是，是监督学习                              |
| DPO             | 通常不是在线 RL，是离线 preference optimization |
| PPO-RLHF        | 是 RL                                  |
| GRPO/RLVR       | 是 RL 方向                               |
| KTO/ORPO/SimPO  | 通常是离线偏好优化                             |

## DPO 为什么常被称为 RLHF 的替代方法？

因为 DPO 从 RLHF 的 KL-constrained objective 推导而来，但不需要显式训练 reward model，也不需要 PPO rollout。它是在一定假设下对 RLHF 目标的离线重写。([arXiv][4])

## 离线偏好优化 vs 在线 RLHF

| 维度    | 离线偏好优化                 | 在线 RLHF                |
| ----- | ---------------------- | ---------------------- |
| 数据    | 固定 preference dataset  | policy 当前生成            |
| 是否探索  | 弱                      | 强                      |
| 工程复杂度 | 低                      | 高                      |
| 代表方法  | DPO/IPO/KTO/ORPO/SimPO | PPO/GRPO               |
| 风险    | 过拟合离线偏好                | reward hacking / KL 崩溃 |

## 三者关系

```text
Reward Modeling:
    学 r(x, y)

Preference Optimization:
    直接用偏好数据优化 policy

Policy Optimization:
    用 reward 或 preference signal 更新 policy
```

对应关系：

```text
RM + PPO:
    preference data → reward model → policy optimization

DPO:
    preference data → direct policy optimization

GRPO/RLVR:
    verifiable reward / rule reward → online RL policy optimization
```

DeepSeekMath 提出 GRPO，是 PPO 的一种变体，用组内相对分数估计 baseline，目标之一是提升数学推理并降低 PPO 的内存开销。([arXiv][7])

---

# 十二、推荐论文

## 1. Learning to summarize from human feedback

重点看：

* 如何收集 pairwise preference；
* 如何训练 reward model；
* 如何用 reward model 做 RL；
* 为什么 SFT 不够。

这篇工作收集人类比较数据，训练模型预测人类偏好的摘要，再把该模型作为 reward function 用于强化学习。([arXiv][8])

## 2. InstructGPT / RLHF

重点看：

* SFT → RM → PPO pipeline；
* labeler ranking 如何转成 reward model 数据；
* PPO 中 KL penalty；
* 为什么小模型经过 RLHF 可以比大模型更符合人类偏好。

## 3. DPO

重点看：

* KL-constrained objective；
* optimal policy；
* implicit reward；
* DPO loss；
* 和 PPO-RLHF 的工程差异。

## 4. IPO / (\Psi)PO

重点看：

* DPO 的理论假设；
* pairwise preference 与 pointwise reward 的关系；
* 为什么 DPO 可能过拟合极端偏好；
* Identity-PO 的目标。

(\Psi)PO 论文把 RLHF 和 DPO 放进统一理论框架，并提出 Identity-PO 作为一个特殊情况。([arXiv][9])

## 5. KTO

重点看：

* 为什么不需要 pairwise preference；
* desirable/undesirable 数据如何训练；
* prospect theory 的直觉；
* 数据不平衡时的优势。

## 6. ORPO

重点看：

* 为什么把 SFT 和 preference optimization 合并；
* odds ratio 是什么；
* 为什么不需要 reference model。

## 7. SimPO

重点看：

* average log probability；
* reference-free reward；
* target reward margin；
* 为什么它比 DPO 更省显存。

## 8. RRHF / SLiC-HF

重点看：

* 如何用 likelihood 排序；
* 为什么是 PPO 的简单替代；
* 与 DPO 的关系。

## 9. DeepSeekMath / GRPO

重点看：

* GRPO 如何省掉 value model；
* group relative advantage；
* RLVR 与 preference learning 的区别；
* 数学推理中 rule-based reward 的作用。

[1]: https://arxiv.org/abs/2203.02155?utm_source=chatgpt.com "Training language models to follow instructions with human feedback"
[2]: https://github.com/huggingface/trl/blob/main/docs/source/dataset_formats.md?utm_source=chatgpt.com "trl/docs/source/dataset_formats.md at main · huggingface/trl"
[3]: https://arxiv.org/abs/1707.06347?utm_source=chatgpt.com "Proximal Policy Optimization Algorithms"
[4]: https://arxiv.org/abs/2305.18290?utm_source=chatgpt.com "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
[5]: https://huggingface.co/docs/trl/dpo_trainer?utm_source=chatgpt.com "DPO Trainer · Hugging Face"
[6]: https://arxiv.org/abs/2402.01306?utm_source=chatgpt.com "KTO: Model Alignment as Prospect Theoretic Optimization"
[7]: https://arxiv.org/abs/2402.03300?utm_source=chatgpt.com "[2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning ..."
[8]: https://arxiv.org/abs/2009.01325?utm_source=chatgpt.com "[2009.01325] Learning to summarize from human feedback - arXiv.org"
[9]: https://arxiv.org/abs/2310.12036?utm_source=chatgpt.com "A General Theoretical Paradigm to Understand Learning from Human Preferences"
