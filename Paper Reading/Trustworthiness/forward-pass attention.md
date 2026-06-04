“**Transformer 机制 → LVLM 生成 → attention-based 幻觉检测 → 局限 → attention×gradient**”

## 0. Overview

**forward-pass attention 方法**的核心想法是：

> 模型在正常、视觉 grounded 的生成中，当前生成 token 应该在注意力上合理地回看图像 token、相关上下文 token、最近生成 token；
> 如果注意力过度集中到无关 token、sink token、纯语言先验 token，或者忽略图像证据，就可能说明模型正在脱离视觉输入，产生 hallucination。

但是它的问题也很明显：

> attention weight 只是前向传播里 softmax 后的“信息混合权重”，不等价于“这个 token 对最终预测有多重要”。
> 真正的影响还要看 value 向量内容、后续 MLP/残差/LayerNorm、logit 方向，以及输出对该 attention 的敏感性。
> 所以新一些的工作会引入 **attention × gradient saliency**，把“模型看哪里”与“输出真的依赖哪里”结合起来。

---

# 1. 什么是 forward-pass attention？

**forward-pass attention** 指的是：在模型做一次普通前向传播时，从 Transformer attention 层中直接取出的 attention weight / attention map。

也就是说，它不需要反向传播，不需要梯度，只是在模型正常生成下一个 token 的过程中，记录每一层、每个 attention head 的注意力分布。

在 Transformer 里，给定输入 hidden states：

$$
[
X = [x_1, x_2, \dots, x_n]
]
$$

每个 token 的 hidden state 会被线性投影成：

$$
[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
]
$$

然后 attention score 是 query 和 key 的相似度：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

其中：

* **Q / Query**：当前 token 想找什么信息；
* **K / Key**：每个历史 token 提供什么“索引”；
* **V / Value**：每个历史 token 真正被取走、混合进来的内容；
* **softmax attention weight**：当前 token 对历史 token 的归一化关注比例；
* **M**：causal mask，保证自回归生成时当前位置不能看未来 token。

所以 forward-pass attention map 本质上就是：

$$
[
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)
]
$$

其中 $A_{ij}$ 表示第 $i$ 个 token 在当前层、当前 head 中对第 $j$ 个 token 分配了多少 attention mass。


# 2. 在 Transformer / LVLM 自回归生成过程中，attention map 是如何得到的？

以 LVLM 自回归生成为例，输入通常可以抽象成：

$$
[
[\text{image tokens},\ \text{text prompt tokens},\ \text{generated tokens so far}]
]
$$

例如用户输入图像和问题：

> Image + “Describe the image.”

模型已经生成：

> “There is a room with”

现在要预测下一个 token，序列可能是：

$$
[
[v_1, v_2, ..., v_m,\ t_1, ..., t_p,\ y_1, ..., y_k]
]
$$

其中：

* $v_i$：图像 token，来自视觉编码器或投影后的视觉 patch token；
* $t_i$：文本 prompt token；
* $y_i$：已经生成的文本 token；
* 当前要预测 (y_{k+1})。

在每一层 Transformer 中，当前最后一个位置的 query 会和前面所有 token 的 key 做相似度：

$$
[
q_{\text{current}} \cdot k_j
]
$$

然后经过 softmax 得到 attention distribution：

$$
[
A_{\text{current}, j}
]
$$

这就告诉我们：**在预测下一个 token 时，模型当前层、当前 head 把注意力分配给了哪些历史 token。**

在 LVLM 中，attention map 可以进一步分成几类：

1. **image-token attention**
   当前文本 token 对图像 token 的 attention。
   例如生成 “dog” 时是否看到了图像中狗所在的 patch。

2. **text-token attention**
   当前 token 对 prompt 或历史生成文本的 attention。
   例如生成 “blue” 时是否看到了之前的 “wallpaper”。

3. **recent output attention**
   当前 token 是否关注最近生成的几个 token。
   这和局部连贯性、上下文记忆有关。

4. **sink-token attention**
   当前 token 是否大量关注 BOS、换行、标点、特殊 token、某些无语义但吸引注意力的 token。
   attention sink 一般指某些 token 会吸收过多 attention mass，即使它们不一定携带具体语义。近年的 LVLM/LLM 工作也把 attention sink 与 hallucination、context forgetting、visual grounding 异常联系起来讨论。([arXiv][1])

# 3. 为什么可以用 attention map 来分析模型是否产生幻觉？

因为在 VLM/LVLM 中，**幻觉的一个核心定义是：输出内容没有被图像证据支持。**

如果模型生成：

> “There is a blue wallpaper.”

但图像里根本没有蓝色墙纸，那么我们会怀疑模型不是基于视觉证据生成的，而是基于语言先验、上下文惯性、训练数据偏置生成的。

attention map 提供了一种可观察信号：

> 当模型生成某个视觉实体、属性、关系 token 时，它有没有关注图像中相关区域？

例如：

* 生成 **dog** 时，是否看向狗的视觉 patch？
* 生成 **red** 时，是否看向对应物体的颜色区域？
* 生成 **on the table** 时，是否看向物体和桌子的空间关系区域？
* 生成 **blue wallpaper** 时，是否关注墙面/背景区域，而不是只关注之前文本 token？

所以 attention-based hallucination detection 的基本假设是：

> 如果一个视觉描述 token 的生成没有足够依赖图像 token，或者 attention 主要集中到无关文本/sink token，那么它更可能是 hallucinated。

注意这里是“更可能”，不是必然。因为 attention map 只是一个 proxy signal。

# 4. 典型 attention-based hallucination detection / mitigation 方法怎么做？

可以按关注对象分成几类。

## 4.1 Image-token attention：看模型是否关注图像证据

这是最直观的一类。

当模型生成某个视觉相关 token，比如：

* object：dog, car, cup
* attribute：red, blue, wooden
* relation：on, under, beside
* scene：kitchen, street, bedroom

方法会检查当前 token 对图像 token 的 attention mass：

$$
[
S_{\text{image}}(y_t) = \sum_{j \in \text{image tokens}} A_{t,j}
]
$$

如果生成 “dog” 时，模型几乎没有看 image tokens，而主要看语言上下文，那么这个 token 可能是 hallucinated。

更细一点的方法会看：

$$
[
S_{\text{region}}(y_t) = \sum_{j \in \text{relevant image patches}} A_{t,j}
]
$$

比如生成 “blue” 时，它有没有看向对应物体区域，而不是看向图像边缘或无关背景。

## 4.2 Text-token attention：看模型是否过度依赖语言上下文

LVLM 里很多幻觉来自语言模型先验。

比如模型看到：

> “A man is holding a …”

它可能根据语言先验继续生成：

> “tennis racket”

即使图像里实际是 umbrella。

这时 attention 可能更多集中在 prompt 或已经生成的文本上，而不是图像 token 上。

所以有的方法会比较：

$$
[
\frac{\text{attention to image tokens}}{\text{attention to text tokens}}
]
$$

如果视觉相关 token 生成时：

* image attention 低；
* text attention 高；
* 特别是高 attention 给最近的语言模式或固定搭配；

那么可能说明模型在“语言补全”，而不是“视觉 grounding”。

## 4.3 Attention sink：看注意力是否被少数无关 token 吸走

**attention sink** 指某些 token 会吸收大量 attention mass。它们可能是：

* BOS token；
* system token；
* newline；
* punctuation；
* 特殊视觉 token；
* 某些固定位置 token；
* 在 LVLM 中也可能是某些图像 token 或视觉 patch token。

在纯 LLM 中，attention sink 常被认为和长上下文建模、注意力稳定性有关。但在 hallucination 检测里，问题在于：

> 如果模型预测视觉内容时，大量 attention 被 sink token 吸走，而不是分配给图像证据，那么模型可能正在从 grounded reasoning 转向 compressed prior / internal prior。

近期一些工作专门研究了 attention sink 与 hallucination 的联系，例如把 sink token 看作 hallucination detection 的内部信号，或者在 LVLM 中分析 visual attention sink 对跨模态 grounding 的影响。([arXiv][2])

## 4.4 Visual attention sink：图像 token 里也会有 sink

在 LVLM 中，attention sink 不只存在于文本 token，也可能存在于视觉 token。

所谓 **visual attention sink**，大致是指：

> 某些视觉 token 不一定对应关键语义区域，却长期吸引大量 attention。

例如图像中某个 patch token、CLS-like token、边界区域 token 或视觉编码器中特定位置的 token，可能在多层中持续吸收 attention。

这会带来两个可能后果：

1. **有利的一面**：
   它可能携带全局场景先验，帮助模型获得整体图像信息。

2. **有害的一面**：
   如果它过度主导注意力，模型可能忽略细粒度局部证据，比如物体颜色、数量、空间关系。
   有工作把 visual attention sink 与 LVLM 中的 irrelevant visual token、特定 hidden dimensions 的异常激活联系起来讨论。([arXiv][1])

## 4.5 Mitigation：不仅检测，还可以干预 decoding

attention-based 方法不只用于分析，也可以用于缓解 hallucination。

常见做法包括：

### 方法 A：降低不可靠 token 的概率

如果当前 candidate token 的 image attention 太低，就降低它的 logit。

例如：

$$
[
\tilde{z}(y_t) = z(y_t) - \lambda \cdot \mathbb{1}[S_{\text{image}}(y_t) < \tau]
]
$$

直觉是：

> 如果一个视觉词没有看图像，那就不要让它轻易被生成。

### 方法 B：增强视觉 token attention

有些方法会在浅层或特定 head 上增强 image-token attention，使模型更依赖视觉输入。例如 2024/2025 年的一些工作分析不同层和 head 上 image token attention 的分布，并尝试通过增强浅层视觉 attention sink 或校准 attention 来缓解 object hallucination。([arXiv][3])

### 方法 C：抑制 sink token

如果发现生成时 attention 过度集中到 sink token，可以降低 sink token 的 attention contribution，或者重新分配 attention mass 给图像 token / relevant text token。

近年的方法还尝试在 decoding 中动态监控 sink token，在出现 sink trigger 时评估 grounding reliability，再调整 self-attention 分布。([arXiv][4])


# 5. forward-pass attention 方法的直觉是什么？

它的直觉可以概括为三句话。

## 5.1 正常 grounding：视觉词应该看图像

如果模型生成：

> “a red apple”

那么生成 “red” 和 “apple” 时，attention 应该至少部分指向图像中对应物体区域。

如果它完全不看图像，却很自信生成视觉属性，就可能是幻觉。

## 5.2 幻觉可能来自语言先验主导

比如模型已经生成：

> “a man is riding a”

语言模型非常容易继续生成：

> “horse” 或 “bike”

即使图像里并没有 horse/bike。

这时模型可能更多关注：

* “man”
* “riding”
* “a”
* 训练数据中的常见搭配

而不是图像区域。

所以 attention-based 方法会认为：
**text attention 过强、image attention 过弱，可能是 hallucination signal。**

## 5.3 幻觉可能伴随 context forgetting / attention drift

在长生成中，模型可能逐渐忘记图像或前文约束。

例如开头看到图像里是白墙，后面却生成：

> “with a blue wallpaper”

这可能说明生成到 “blue” 时，模型不再关注图像中的墙面区域，也不再关注之前已建立的描述约束，而是被局部语言模式带偏。

一些近期 LVLM saliency 工作也把 hallucination 与“prior output tokens 对 next-token prediction 的 saliency 下降”联系起来，即模型没有有效利用最近上下文，出现 contextual memory failure。([OpenReview][5])

# 6. 为什么 forward-pass attention 可能不可靠？

这是重点。

**attention weight 高，不一定代表该 token 对最终预测真的重要。**

原因至少有六个。

## 6.1 Attention weight 只是信息混合系数，不是因果贡献

attention 输出是：
$$
[
o_i = \sum_j A_{ij} V_j
]
$$
即使 $A_{ij}$ 很高，也只说明 value $V_j$ 被较大比例混合进来。

但最终预测还要经过：

* 多个 attention head；
* output projection；
* residual connection；
* LayerNorm；
* MLP；
* 后续 Transformer layers；
* final LM head。

所以一个 token 被“看见”了，不等于它最终影响了 logits。

## 6.2 Value 向量可能没有携带相关信息

attention map 只看 $A_{ij}$，但真正进入 hidden state 的是：

$$
[
A_{ij} V_j
]
$$

如果某个 token attention weight 很高，但它的 $V_j$ 在相关语义方向上没什么信息，那么它对最终 “blue” 的预测可能影响很小。

反过来，一个 token attention weight 不高，但它的 value 向量恰好在决定性方向上很强，也可能对输出很重要。

## 6.3 多头 attention 会混合不同功能

一个 head 可能负责：

* 位置对齐；
* 复制前文；
* 语法结构；
* 图像 grounding；
* sink/stabilization；
* 长程记忆；
* 局部连贯性。

所以当看到某个 head 的 attention map，并不一定能直接解释最终预测。
有些 head 的高 attention 可能只是格式、位置、稳定性功能，而不是语义 grounding。

## 6.4 Softmax attention 是相对分布，不是绝对证据

softmax 后所有 attention weight 加起来为 1。

如果 image token 很多，单个 image token 的 attention 可能很低；如果 text token 很少，单个 text token 的 attention 可能显得很高。

因此简单看 “attention 高/低” 很容易误判。

例如：

* 图像 token 有 576 个；
* 文本 token 有 20 个；
* 图像总 attention 是 0.35。

单个图像 token 平均可能很低，但总视觉依赖并不一定低。

## 6.5 Attention 可以被替换但输出不变

经典论文 **Attention is not Explanation** 指出，attention weights 和 gradient-based feature importance 往往并不一致，而且可以找到不同的 attention distributions 产生相近预测，因此不能把 attention weight 直接当作解释。([ACL][6])

> 这篇论文的主要工作是系统检验：神经网络中的 attention weight 是否可以直接作为模型预测的解释。作者在多种 NLP 任务上，将 attention weight 与梯度显著性（gradient-based importance）以及留一法（leave-one-out，删除某个输入后观察预测变化）得到的特征重要性进行比较。实验发现，attention weight 与这些重要性指标往往相关性较弱。进一步地，作者还构造了与原始 attention 分布显著不同的替代分布，却发现模型仍然可以给出几乎相同的预测结果。这说明，同一个预测结果可能对应多种不同的 attention 分布，attention 并不能唯一揭示模型作出预测的依据。 

> 论文的核心观点是：attention weight 不应被直接等同于特征重要性，更不能在未经验证的情况下被视为模型决策过程的可靠解释。 Attention weight 反映的是模型在某一层前向传播中如何对输入表示进行加权聚合，但一个输入 token 对最终输出的实际影响，还取决于其 value 向量所携带的信息、后续网络层的变换、残差连接以及不同输入之间的信息冗余。作者并不是否认 attention 具有任何分析价值，而是强调：仅凭 attention heatmap 或 attention 数值，不能断言模型“因为关注了某个 token，所以作出了某个预测”。 

这对 hallucination 检测非常关键：

> 你看到模型“看了图像”，不代表这个图像证据真的决定了最终 token；
> 你看到模型“没怎么看某个 token”，也不代表它完全没影响输出。

## 6.6 Attention 不告诉你输出对它是否敏感

forward attention 只告诉你：

> 当前前向传播中，attention 是多少。

但它不告诉你：

> 如果这个 attention 变小一点，最终 token 概率会不会变？

这就是 gradient 的价值。

梯度回答的是：

$$
[
\frac{\partial \log p(y_t)}{\partial A_{ij}}
]
$$

也就是：

> 当前输出概率对这条 attention connection 有多敏感。

所以 attention × gradient 会比单纯 attention 更接近“影响力”。


# 7. 例子：生成 “a blue wallpaper” 时，如何从 attention map 判断可能 hallucinate？

假设图像里是一间房间，墙是白色，没有蓝色墙纸。模型生成：

> “The room has a blue wallpaper.”

我们观察生成每个关键 token 时的 attention。

## 7.1 生成 “wallpaper” 时

当前 token 是 “wallpaper”。我们看它对 image tokens 的 attention。

### 情况 A：比较 grounded

attention map 显示：

* 较多 attention 指向墙面区域；
* 相关 image patches 是背景墙；
* 也关注前文 “room”“has”“a”。

这时模型至少在视觉区域上有一定依据。

但注意：这仍不保证它是对的，因为墙面区域可能只是普通墙，不是 wallpaper。

### 情况 B：可能 hallucination

attention map 显示：

* image-token attention 很低；
* 大量 attention 给 “room has a”；
* 或大量 attention 给 BOS / punctuation / sink token；
* 生成 “wallpaper” 时并没有关注墙面 patch。

这说明 “wallpaper” 很可能来自语言先验：

> room → has → wallpaper

而不是来自图像证据。

## 7.2 生成 “blue” 时

这是更关键的属性词。

如果模型生成 “blue”，我们希望它看向有颜色证据的区域。

### 可能 grounded 的 pattern

* attention 指向墙面区域；
* 视觉区域确实偏蓝；
* 当前 token 对图像区域和 “wallpaper” 都有合理 attention。

### 可能 hallucinated 的 pattern

* “blue” 主要关注前文 “a” 或 “wallpaper”；
* image-token attention 低；
* 或只关注某个 visual sink token，而不是墙面相关 patches；
* “blue” 的 attention pattern 和其他颜色词类似，说明它可能只是语言补全。

这时可以怀疑：

> 模型不是因为看到了蓝色墙纸才生成 blue，而是因为语言模式中 “blue wallpaper” 是一个常见短语。

## 7.3 更细粒度判断：image attention 是否 spatially aligned

只看 image attention 总量还不够。

比如 “blue” 的 image attention 总量很高，但注意力集中在窗户、床、天花板，而不是墙面区域，也可能是错误 grounding。

所以更好的 attention-based 判断会看：

1. 当前 token 是否视觉相关；
2. 它对 image tokens 的总 attention 是否足够；
3. 它关注的 image patches 是否与该 token 语义区域一致；
4. 它是否过度关注 sink tokens；
5. 它是否依赖最近生成 token，保持局部语义一致。

# 8. forward-pass attention vs attention × gradient saliency

LVLMs-Saliency 的核心。

## 8.1 forward-pass attention 看到了什么？

forward attention 看的是：

$$
[
A_{ij}
]
$$

也就是：

> 在当前前向传播中，token $i$ 从 token $j$ 那里取了多少信息。

它回答的问题是：

> 模型“看向”哪里？

例如生成 “blue” 时：

* 看了多少 image tokens？
* 看了多少 “wallpaper”？
* 看了多少 BOS/sink token？
* 看了多少最近输出 token？

但是它不回答：

> 这些 attention connection 对最终 “blue” 的概率到底有没有影响？

## 8.2 attention × gradient saliency 看到了什么？

attention × gradient 通常会计算类似：

$$
[
\mathrm{Saliency}*{ij} = A*{ij} \cdot \left|\frac{\partial \log p(y_t)}{\partial A_{ij}}\right|
]
$$

或者对某个目标 logit / loss 做梯度：

$$
[
\mathrm{Saliency}*{ij} = A*{ij} \cdot \left|\frac{\partial z_{y_t}}{\partial A_{ij}}\right|
]
$$

它结合了两件事：

1. **Attention weight $A_{ij}$**
   当前模型实际分配了多少注意力。

2. **Gradient sensitivity $\frac{\partial z}{\partial A_{ij}}$**
   如果这条 attention connection 改变，目标 token 的 logit / 概率会变化多少。

所以它回答的问题更接近：

> 模型不仅看了哪里，而且最终预测真的依赖哪里？

## 8.3 为什么 attention × gradient 更能反映 token influence？

举个直观例子。

生成 “blue” 时，有两个历史 token：

* token A：图像中的墙面 patch；
* token B：前文文本 “wallpaper”。

假设 attention 是：

| 来源 token               | attention weight | gradient magnitude | attention × gradient |
| ---------------------- | ---------------: | -----------------: | -------------------: |
| 墙面 image patch         |             0.30 |               0.01 |                0.003 |
| “wallpaper” text token |             0.10 |               0.50 |                0.050 |

单看 attention，你会以为模型主要看图像，所以 “blue” 可能 grounded。

但 attention × gradient 显示：

> 输出 “blue” 对 image patch 不敏感，反而强烈依赖 “wallpaper” 这个文本 token。

这就说明模型虽然“看了图像”，但真正驱动输出的可能是语言上下文。

这正是 forward-pass attention 容易误判的地方。

## 8.4 从因果直觉上理解

forward attention：

> 模型的信息流路径上，有多少质量分配给了某个 token？

attention × gradient：

> 这条路径如果被扰动，会不会影响最终预测？

所以后者更接近 token influence。

LVLMs-Saliency 的结论：只用 forward-pass attention 不能稳定地区分 hallucinated 和 correct outputs，因为它忽略了 gradient-based signal；而 gradient 能揭示 token influence 如何通过网络传播。([OpenReview][5])


[1]: https://arxiv.org/abs/2503.03321?utm_source=chatgpt.com "Visual Attention Sink in Large Multimodal Models"
[2]: https://arxiv.org/abs/2604.10697?utm_source=chatgpt.com "Attention Sinks as Internal Signals for Hallucination Detection in Large Language Models"
[3]: https://arxiv.org/html/2411.09968v1?utm_source=chatgpt.com "Enhancing Attention Heads to Alleviate Hallucination in ..."
[4]: https://arxiv.org/abs/2603.27898?utm_source=chatgpt.com "SAGE: Sink-Aware Grounded Decoding for Multimodal Hallucination Mitigation"
[5]: https://openreview.net/forum?id=sjnErRHXf3&utm_source=chatgpt.com "Hallucination Begins Where Saliency Drops"
[6]: https://aclanthology.org/N19-1357/?utm_source=chatgpt.com "Attention is not Explanation"


# 9. 论文

该类方法的共同特点是：**LVLM 在自回归生成过程中，attention 可能异常集中到少数 token 上，导致模型忽略图像信息或丢失有效上下文，最终产生幻觉。**

这里的 **attention sink** 可以暂时理解为“注意力黑洞”：后续 token 在生成时反复将较高的 attention weight 分配给某些 token，使这些 token 持续吸引信息流。问题不在于某个 token 偶尔获得较高 attention，而在于模型对少数 token 形成了不合理的长期依赖。

## 9.1 总览

| 方法           | 论文                                                                                                                                               | 现象                                                     | 干预方式                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- | -------------------------------------------------------------- |
| **OPERA**    | *OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation*，CVPR 2024            | 模型过度依赖少数 summary tokens，忽略图像 token                       | 对过度聚集的 attention 施加惩罚；必要时回滚并重新选择 token                         |
| **DOPRA**    | *DOPRA: Decoding Over-accumulation Penalization and Re-allocation in Specific Weighting Layer*，ACM MM 2024                                       | attention sink 主要集中在特定层，尤其是某些中间层                         | 在特定层惩罚 attention 过度累积，并重新分配 attention                          |
| **PAI**      | *Paying More Attention to Image: A Training-Free Method for Alleviating Hallucination in LVLMs*，ECCV 2024                                        | 模型存在 text inertia：即使不给图像，也容易依靠语言惯性生成相似内容                 | 增强 image-token attention；使用有图像与无图像 logits 的差值削弱语言先验            |
| **FastV**    | *An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models*，ECCV 2024 Oral               | 深层网络中的大量 visual-token attention 贡献有限                     | 在浅层保留视觉 token，随后剪枝低 attention 的 visual tokens                  |
| **EAH**      | *Seeing Clearly by Layer Two: Enhancing Attention Heads to Alleviate Hallucination in LVLMs*；后续正式版本标题为 *Shallow Focus, Deep Fixes...*，EMNLP 2025 | 浅层中存在密集的 visual attention sinks；部分 head 更有利于视觉 grounding | 找到视觉 sink 较强的 head，将其 attention pattern 广播给其他 heads            |
| **TAME**     | *Intervening Anchor Token: Decoding Strategy in Alleviating Hallucinations for MLLMs*，ICLR 2025                                                  | anchor token 的信息传播过强，使模型忽略视觉信息                           | 调节 attention 的谱特性，抑制 anchor token 的过度传播                        |
| **FarSight** | *Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding*，CVPR 2025                                            | outlier tokens 干扰正常信息流；远距离视觉信息逐渐衰减                       | 修改 causal mask，引入 attention register，吸收被异常 token 劫持的 attention |

## 9.2 OPERA：惩罚对少数 summary tokens 的过度信任

### 论文

**OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation**
Huang et al., CVPR 2024。

### 核心思路

OPERA 观察到，MLLM 在生成文本时，并不会均匀使用此前所有上下文，而是可能反复依赖少数具有“总结性”的 token。论文将这些 token 称为 **summary tokens**。

例如，模型已经生成：

> The image shows a room with a table and a chair ...

后续生成新 token 时，模型可能没有充分读取图像 token，也没有综合利用此前所有描述，而是不断依赖某个局部 token，例如 `room`。这样容易使后续内容逐渐偏离图像，根据语言先验继续编造：

> ... and a large sofa near the window.

即使图像中并不存在 sofa。

OPERA 使用两个机制：

1. **Over-trust Penalty**：如果 attention 过度集中于少数 summary tokens，就在解码时对相应候选 token 的 logits 施加惩罚。
2. **Retrospection-Allocation**：如果检测到已经生成的序列可能受 summary token 误导，就回溯此前的生成过程，并重新分配 token 选择。

因此，OPERA 的主要思想是：

> **不要让生成过程长期依赖少数“总结性 token”；一旦发现依赖过度，就惩罚或回滚。**

这是典型的 attention-sink 幻觉缓解方法。([arXiv][1])

## 9.3 DOPRA：只在关键层处理 attention 过度累积

### 论文

**DOPRA: Decoding Over-accumulation Penalization and Re-allocation in Specific Weighting Layer**
Wei & Zhang, ACM Multimedia 2024。

### 核心思路

DOPRA 与 OPERA 非常接近，但进一步提出：attention sink 并不是在每一层都同样严重。

作者观察到，模型对于 summary tokens 的过度依赖往往集中在特定 self-attention layer 中。例如，在某些实验设置中，第 12 层的异常 accumulation 尤其明显。

因此，DOPRA 没有在所有层统一处理，而是聚焦于特定层：

1. 找到容易出现 attention over-accumulation 的关键层；
2. 惩罚这些层中对少数 token 的 attention 过度聚集；
3. 将部分 attention 重新分配给其他上下文 token；
4. 必要时重新检查此前生成的序列并调整 token 选择。

可以将 OPERA 和 DOPRA 的差异概括为：

| 方法    | 关注点                             |
| ----- | ------------------------------- |
| OPERA | 检测并惩罚对少数 summary tokens 的整体过度依赖 |
| DOPRA | 进一步强调异常依赖主要发生在特定层，只在关键层做精细化干预   |

DOPRA 同样属于典型的 **attention sink / attention over-accumulation** 修正方法。([ACM数字图书馆][2])

## 9.4 PAI：提高图像 token 的存在感，削弱 text inertia

### 论文

**Paying More Attention to Image: A Training-Free Method for Alleviating Hallucination in LVLMs**
Liu et al., ECCV 2024。

### 核心思路

PAI 关注的重点不是少数 summary tokens，而是 **视觉信息和语言先验之间的不平衡**。

作者发现：对于部分 LVLM，即使移除图像输入，模型仍然可能生成与原始回答较为相似的内容。这说明生成结果可能主要来自 LLM 已经学到的语言模式，而不是当前图像。

论文将这种现象称为：

$$
[
\text{text inertia}
]
$$

即“文本惯性”。

PAI 使用两种方式缓解这一问题：

1. **增强 image tokens 的 attention weights**：让后续文本 token 更重视视觉信息；
2. **削弱纯文本路径的 logits**：比较“带图像输入”和“纯文本输入”时的 logits，减少语言模型仅依靠语言先验产生的顽固输出。

可以将它理解为一种视觉增强版本的 contrastive decoding：

$$
\text{final logits}
\approx
\text{multimodal logits}
-
\lambda \cdot \text{text-only logits}
$$

PAI 的出发点是：

> **模型之所以幻觉，不只是因为注意力集中到了错误 token，也可能是因为图像 token 的影响力整体过弱。**

因此，PAI 与 attention sink 有关，但更准确地说，它是一种 **visual attention enhancement + language-prior suppression** 方法。([ECVA][3])

## 9.5 FastV：主要解决推理效率，不是专门解决幻觉

### 论文

**An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models**
Chen et al., ECCV 2024 Oral。

### 核心思路

FastV 的核心目标是降低 LVLM 的推理成本。

作者观察到，在 LLaVA-1.5、QwenVL-Chat 和 Video-LLaVA 等模型中，视觉 token 通常很多，但经过前几层后，深层网络对其中大量 visual tokens 的 attention 已经很低。这意味着继续让所有 visual tokens 参与后续计算，会造成冗余。

FastV 的做法是：

1. 在浅层保留完整视觉 token；
2. 根据浅层 attention score 判断 visual-token 重要性；
3. 在后续层剪掉贡献较低的 visual tokens；
4. 只让较重要的视觉 token 继续参与深层计算。

例如，一个模型原本输入 576 个 image tokens。FastV 可以在第 2 层后仅保留其中约一半，从而减少 self-attention 和 FFN 的计算量。

FastV 和幻觉研究的联系在于：它揭示了 **不同视觉 token 在不同层中的 attention 分布并不均匀**。但它并没有以缓解幻觉为主要目标，也没有直接修复 attention sink。

因此，将 FastV 写进这一段更合适的理解是：

> FastV 为研究 image-token attention dynamics 提供了重要观察，但它本身属于 visual-token pruning 方法，而不是典型的 anti-hallucination 方法。([arXiv][4])

## 9.6 EAH：找到浅层中的优质 visual attention heads，并强化它们

### 论文

这项工作最初以如下标题发布预印本：

**Seeing Clearly by Layer Two: Enhancing Attention Heads to Alleviate Hallucination in LVLMs**

后续正式发表版本的标题是：

**Shallow Focus, Deep Fixes: Enhancing Shallow Layers Vision Attention Sinks to Alleviate Hallucination in LVLMs**
Zhang et al., EMNLP 2025。

### 核心思路

EAH 将 attention sink 的分析进一步转向 **image tokens 内部**。

作者观察到：

* 浅层 self-attention 中，image-token attention sink 往往比较密集；
* 深层中，这类视觉 sink 会变得稀疏；
* 某些 attention heads 更擅长将注意力集中到有价值的视觉 token 上；
* 这些视觉 attention heads 对减少幻觉反而是有帮助的。

这点很容易和 OPERA 混淆：

| 工作          | 如何看待 attention sink？                      |
| ----------- | ----------------------------------------- |
| OPERA、DOPRA | 对少数 summary tokens 的过度依赖通常是有害的            |
| EAH         | 某些浅层 visual attention sinks 可能有助于模型保留视觉信息 |

EAH 的干预方法是：

1. 在浅层识别具有较强视觉聚集能力的 attention head；
2. 提取该 head 的 attention map；
3. 将这个 attention pattern 广播给同层其他 heads；
4. 让该层整体更加关注图像。

EAH 的思想可以概括为：

> **不是所有 attention sink 都应当消除。某些 shallow-layer visual sinks 能够保留图像信息，应该被强化。**

这是一种 head-level、training-free 的视觉注意力增强方法。([arXiv][5])

## 9.7 TAME：从 anchor token 的过度传播入手

### 论文

**Intervening Anchor Token: Decoding Strategy in Alleviating Hallucinations for MLLMs**
Tang et al., ICLR 2025。

### 核心思路

TAME 认为，此前的方法将问题简单归结为“模型过度关注某些 token”，还不够深入。真正需要关注的是：这些 token 的影响力如何在网络中持续传播。

TAME 将容易吸引大量 attention、并将影响扩散至后续 token 的局部 token 称为：

$$
[
\text{anchor tokens}
]
$$

作者将 attention localization degree 定义为一种 token propagation probability，用来描述 anchor token 的信息传播程度。

当 anchor tokens 传播过强时：

1. 后续 token 过度依赖 anchor tokens；
2. attention 分布发生极化；
3. 图像信息受到抑制；
4. 生成结果越来越依赖语言上下文；
5. 最终出现幻觉。

TAME 不直接删除 anchor tokens，也不必执行复杂的回滚，而是动态干预 attention 的谱特性，尤其是 eigenspectrum variance，以限制 anchor token 的过度传播。

直观上可以理解为：

> OPERA 关注“哪些 token 吸引了过多 attention”；
> TAME 进一步关注“这种异常 attention 如何持续向后传播”。

TAME 的方法名称来自：

$$
[
\text{Dynamic Token Propagation Mechanism}
]
$$

它是一种 plug-and-play decoding intervention。([OpenReview][6])

## 9.8 FarSight：通过 causal mask 修复被 outlier tokens 劫持的信息流

### 论文

**Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding**
Tang et al., CVPR 2025。

### 核心思路

FarSight 将幻觉分成两类：

1. **Initial hallucination**：模型一开始就生成了错误内容；
2. **Snowball hallucination**：前面出现一个错误 token，后续生成不断基于该错误继续扩展，形成滚雪球效应。

作者认为，一个重要原因是 outlier tokens 干扰了多模态 token 之间的信息传播。随着距离增加，较早视觉 token 的影响逐步衰减，模型越来越依赖少数异常 token。

FarSight 的关键方法是修改 causal mask：

1. 在 causal mask 的上三角区域中引入 **attention registers**；
2. 这些 registers 用于接收原本被 outlier tokens 吸引的 attention；
3. 通过重新组织 attention propagation，使模型减少对异常 token 的依赖；
4. 同时使用位置感知机制，让模型能够更好地利用距离较远的上下文。

需要注意，这并不是让模型违反因果生成规则。FarSight 设计的 attention registers 仍然保留 causal decoding 性质，不会提前读取未来 token。

FarSight 的核心思想是：

> **不要仅仅惩罚错误 token，而要修复 attention 在序列中的传播路径。**

它尤其适用于较长文本和视频序列，因为这些任务更容易出现远距离信息衰减。([arXiv][7])

## 9.9 这些方法之间的演进关系

将这些工作按照研究问题的发展顺序排列，可以看到一条比较清晰的路线。

### 第一阶段：发现模型过度依赖少数文本 token

代表方法：

* OPERA
* DOPRA

核心观点：

$$
[
\text{summary-token over-trust}
\rightarrow
\text{image neglect}
\rightarrow
\text{hallucination}
]
$$

解决方式是惩罚、回溯和重新分配 attention。

### 第二阶段：增强视觉信息，纠正语言先验过强

代表方法：

* PAI
* EAH

核心观点：

$$
[
\text{weak visual grounding}
+
\text{strong language prior}
\rightarrow
\text{hallucination}
]
$$

PAI 直接增强 image-token attention，并削弱纯文本 logits；EAH 则寻找浅层中表现较好的 visual attention heads，并强化其视觉 pattern。

### 第三阶段：研究 attention 异常的传播机制

代表方法：

* TAME
* FarSight

核心观点：

$$
[
\text{anchor/outlier token}
\rightarrow
\text{abnormal propagation}
\rightarrow
\text{context drift}
\rightarrow
\text{hallucination}
]
$$

TAME 通过谱特性干预 anchor-token propagation；FarSight 通过 causal mask 和 attention registers 修复信息传播路径。

### 旁支：视觉 token 剪枝与 attention 效率

代表方法：

* FastV

核心观点：

$$
[
\text{many visual tokens}
\neq
\text{all visual tokens are useful in deep layers}
]
$$

它主要用于 inference acceleration，但为后续分析 visual-token attention dynamics 提供了重要依据。


[1]: https://arxiv.org/abs/2311.17911?utm_source=chatgpt.com "OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation"
[2]: https://dl.acm.org/doi/10.1145/3664647.3681076?utm_source=chatgpt.com "DOPRA: Decoding Over-accumulation Penalization and ..."
[3]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10933.pdf?utm_source=chatgpt.com "A Training-Free Method for Alleviating Hallucination in ..."
[4]: https://arxiv.org/abs/2403.06764?utm_source=chatgpt.com "An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models"
[5]: https://arxiv.org/abs/2411.09968?utm_source=chatgpt.com "Seeing Clearly by Layer Two: Enhancing Attention Heads to Alleviate Hallucination in LVLMs"
[6]: https://openreview.net/forum?id=zGb4WgCW5i&utm_source=chatgpt.com "Intervening Anchor Token: Decoding Strategy in Alleviating..."
[7]: https://arxiv.org/abs/2505.16652?utm_source=chatgpt.com "Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding"
