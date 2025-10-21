# 如何理解注意力机制？

灵感来自：[注意力机制的本质|Self-Attention|Transformer|QKV矩阵](https://www.bilibili.com/video/BV1dt4y1J7ov/?share_source=copy_web&vd_source=608471d0e25c02d240b92470bd78f213)

## 1 什么是注意力机制？

假设我现在有一个 Key-Value 存储库，里面存了很多 Key-Value 对：

| Key | Value |
|-----|-------|
| k1:51  | v1:40    |
| k2:56  | v2:43    |
| k3:58  | v3:48    |

现在我有一个 Query q:57，我想通过已有的 Key-Value 对，推测出如果有一个 k 为 57，那么它对应的 v 可能是多少。

我该怎么做？

很容易能看到 57 介于 56 和 58 之间，所以它对应的 v 很可能介于 43 和 48 之间，同时还能注意到 57 距离 56 和 58 的距离是一样的，所以它对应的 v 很可能也距离 43 和 48 一样，所以可以推测出 k:57 对应的 v 可能是 45.5

计算公式如下：
$$
f(q) = \alpha(q, k_1)v_1 + \alpha(q, k_2)v_2 
= \sum_{i=1}^{2} \alpha(q, k_i)v_i
$$

其中 $\alpha(q, k_i)$ 是一个权重函数，表示 q 和 $k_i$ 的相似度，也就是我们前面俗称的“距离”，距离越近，相似度越高，权重也就越大。

很明显，这里的 $\alpha(q, k_i)$ 计算方式为（假设一共有 n 个 Key）：
$$
\alpha(q, k_i) = \frac{(|q - k_i|)}{\sum_{j=1}^n (|q - k_j|)}
$$

如此我们就能发现两个问题：
1. 在计算 $f(q)$ 时我们只注意到了 k2 和 k3，忽略了 k1，其实 k1 也能提供一些信息的，虽然它离 q 比较远，但是它的 v 也能对最终结果产生一些影响。
2. 我们计算权重的方式过于简单。

为了解决这两个问题，我们引入了**注意力机制**，它的核心思想是：**对所有的 Key-Value 对都进行考虑，并且用一个更复杂的函数来计算权重**。

## 2 Attention

现在我们再来计算 $f(q)$，假设我们有 n 个 Key-Value 对，我们对所有的 Key-Value 对都进行考虑，并且用一个更复杂的函数来计算权重。

$$
f(q) = \alpha(q, k_1)v_1 + \alpha(q, k_2)v_2 + \alpha(q, k_3)v_3 + ... + \alpha(q, k_n)v_n = \sum_{i=1}^{n} \alpha(q, k_i)v_i
$$

将上面的式子携程矩阵的形式：

$$
f(q) = [v_1, v_2, v_3, ..., v_n] \cdot [\alpha(q, k_1), \alpha(q, k_2), \alpha(q, k_3), ..., \alpha(q, k_n)]^T
$$

而如果我们把所有的 Key 和 Value 都放到一个矩阵中：

$$
K = [k_1, k_2, k_3, ..., k_n]^T
$$

$$
V = [v_1, v_2, v_3, ..., v_n]^T
$$

那么我们就可以把 $f(q)$ 的计算方式简化为：

$$
f(q) = V \cdot \alpha(q, K)
$$

我们再将 q 扩展到 n 个（说明我们有 n 个 Query）：

$$
F(Q) = V \cdot \alpha(Q, K)
$$

其中：

$$
Q = [q_1, q_2, q_3, ..., q_n]^T
$$

$$
F(Q) = [f(q_1), f(q_2), f(q_3), ..., f(q_n)]^T
$$

$$
\alpha(Q, K) =
\begin{bmatrix}
\alpha(q_1, k_1) & \alpha(q_1, k_2) & ... & \alpha(q_1, k_n) \\
\alpha(q_2, k_1) & \alpha(q_2, k_2) & ... & \alpha(q_2, k_n) \\
... & ... & ... & ... \\
\alpha(q_n, k_1) & \alpha(q_n, k_2) & ... & \alpha(q_n, k_n) \\
\end{bmatrix}
$$

这样就得到了注意力机制的计算方式。

## 3 Self-Attention

如果 Q、K、V 都是从同一个输入 X 通过不同的线性变换得到的：

$$
Q = XW_Q, K = XW_K, V = XW_V
$$

所以这就叫做 Self-Attention，自注意力机制。

## 4 Cross-Attention

若是 Q、K、V 不是都来自同一个输入 X，而是 Q 来自一个输入 X1，K 和 V 来自另一个输入 X2：

$$
Q = X_1W_Q, K = X_2W_K, V = X_2W_V
$$

所以这就叫做 Cross-Attention，交叉注意力机制。

## 5 Attention 中的 $ W_Q, W_K, W_V $

那么 $W_Q, W_K, W_V$ 是什么呢？

因为在之前的那个例子中 Key 和 Value 都是显示给出的，所以我们可以直接计算。但是在实际应用中，Key 和 Value 往往是隐含的，我们需要通过某种方式来生成它们。通常，这些矩阵是通过训练学习得到的参数，用于将输入特征映射到查询、键和值的空间中。
这也就是$W_Q, W_K, W_V$ 的由来。
而 Self-Attention 和 Cross-Attention 的区别就在于 Q、K、V 是不是来自同一个输入。

# 回到 Transformer

我从我自己探究 Transformer 的两个优势开始讲起：
1. 直接建模任意两 token 关系，长程依赖自然保留；
2. 信息路径长度为 $O(1)$，因为注意力直接连接所有位置。

## 1. 直接建模任意两 token 关系


我们先看一个句子：

> “猫坐在垫子上。”

我们假设模型现在在处理单词 “垫子”。

当计算 “垫子” 的表示时，它直接和整个句子的所有 token（包括“猫”）进行交互：

```
猫 ↔ 坐 ↔ 在 ↔ 垫子 ↔ 上
```

注意力机制通过计算：
$$
\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right)V
$$
其中 $QK^\top$ 实际上就是「所有 token 两两之间的相似度矩阵」。

> V 我们后面会细讲

所以在 Transformer 中不管两个词距离多远（哪怕隔 1000 个 token），**“垫子” 都能直接看见 “猫” 的信息**。

所以说：

> 自注意力可以「直接建模任意两 token 之间的关系」。

而不是像 RNN 那样，必须通过一长串中间节点传递。

## 2. 信息路径长度 $O(1)$

这里我们与 RNN 进行对比。

* **RNN：**
  信息从 A 传到 B 要过很多“中继站”；距离越远，信号越弱。

  ```
  A → B → C → D → E
  ```

  A 想传消息给 E，要经过 4 层传递。

* **Transformer：**
  注意力矩阵让每个 token 都能直接“拨电话”给任何人。

  ```
  A ↔ B ↔ C ↔ D ↔ E
  ```

  A 想告诉 E 什么内容，一次就能解决。

## 3. 数学角度解析

假设我们有输入序列：
$$
X = [X_1, X_2, X_3, \ldots, X_n], \quad X \in \mathbb{R}^{n \times d_{\text{model}}}
$$

每个 $X_i$ 是一个 token 的向量表示。

Transformer 首先通过三组线性变换得到：
$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$
可得：
$$
Q = [Q_1, Q_2, \ldots, Q_n], \quad
K = [K_1, K_2, \ldots, K_n], \quad
V = [V_1, V_2, \ldots, V_n]
$$

每个 token $X_i$ 都有自己对应的 $Q_i, K_i, V_i$。

接下来计算注意力分数：
$$
S = \frac{QK^\top}{\sqrt{d_k}}
$$

这里的 $S$ 是一个 $n \times n$ 的矩阵，
每个元素：
$$
S_{ij} = \frac{Q_i \cdot K_j}{\sqrt{d_k}}
$$
表示 **第 i 个 token 对第 j 个 token 的注意力“相似度”**。

这里有两个关键点：
* $S_{ij}$ 是 **直接计算的点积**，不需要依赖任何中间步骤；
* 也就是说：

  > “第 i 个 token”$Query$可以**直接看到所有 j 的 Key（整个序列）**。

所以，**任何两个 token（比如 $X₁$ 和 $Xₙ$）之间都有直接的相似度连接**,也就是说 Transformer 的自注意力可以“直接建模任意两 token 的关系”，并且只需要一次计算即可，没有任何前置条件，所以时间复杂度是$O(1)$。

接下来计算注意力权重（softmax）：
$$
A_i = \text{softmax}(S_{i,:})
$$
这是一个长度为 $n$ 的向量，表示第 $i$ 个 token 对整个序列的注意力分布。

然后加权求和：
$$
Z_i = \sum_{j=1}^{n} A_{ij} V_j
$$
也就是说：

> 第 $i$ 个 token 的输出 $Z_i$ 是 **所有 token 的 Value 的加权和**。

它能直接利用所有其他 token 的信息，哪怕这些 token 距离再远。

## 4. 训练阶段

输入：
$$
X = [X_1, X_2, ..., X_i]
$$
目标是预测下一个 token：
$$
X_{i+1}
$$

Decoder 会计算：
$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
\text{Attention}(Q_i, K_{\le i}, V_{\le i}) = \text{softmax}\left( \frac{Q_i K_{\le i}^\top}{\sqrt{d_k}} \right) V_{\le i}
$$

> 所以第 $i$ 个位置（即 $token Xᵢ$）会根据自己$Qᵢ$与所有前面$K₁…Kᵢ$的匹配程度，得到上下文表示 $Z_i$。

$$
\hat{X}_{i+1} = \text{Linear}(Z_i)
$$

> 也就是说：第 $i$ 个 token 的上下文表示 $Z_i$（汇聚了前面所有信息）用于预测第 $i+1$ 个 token。


在训练中我们会用一个 **mask 矩阵**（上三角为 $−∞$）来确保模型只能看见前面（因为直接做矩阵运算取 $j<=i$ 的索引过于麻烦）：
$$
S_{ij}=
\begin{cases}
\frac{Q_i\cdot K_j}{\sqrt{d_k}}, & j\leq i \\
-\infty, & j>i & & 
\end{cases}
$$

这样 softmax 后，第 $i$ 个位置就**只关注前面的 token**。


## 5. 推理阶段

推理时我们是一步一步生成的：

| 步骤     | 输入           | 预测输出    |
| ------ | ------------ | ------- |
| Step 1 | [X₁]         | → 预测 X₂ |
| Step 2 | [X₁, X₂]     | → 预测 X₃ |
| Step 3 | [X₁, X₂, X₃] | → 预测 X₄ |
| …      | …            | …       |

每次都执行：
$$
Z_i = \text{Attention}(Q_i, K_{\le i}, V_{\le i})
$$
然后再根据 $Z_i$ 预测 $X_{i+1}$。

## 6. 注意力机制的核心逻辑

首先思考：**我们要做什么？**

我们希望得到第 $i$ 个 token 的新的表示 $Z_i$。

这个表示应该能：

* 总结它**自己**的信息；
* 同时融合 **其他 token（上下文）** 的信息；
* 并且知道该“信任谁”更多。

于是，Transformer 就让第 $i$ 个 token 主动“询问”整个序列中哪些 token 跟它有关、该关注谁，这也就是 Query 的由来。

$$
\text{score}_{ij} = Q_i \cdot K_j^\top
$$

含义：

* $Q_i$：第 i 个 token 的“问题” Query 向量，表示“我现在想知道什么”
* $K_j$：第 j 个 token 的“钥匙” Key 向量，表示“我包含了什么信息” 或 “我包含的信息有多重要”

还是前面那个例子：
| Token | 含义      |
| ----- | ------- |
| “猫”   | 内容：动物主体 |
| “坐”   | 动作      |
| “垫子”  | 场所      |
| “上”   | 位置关系    |

当模型计算 “垫子” 这个词时（即 $i=3$）：

* 它的 Query $Q₃$代表“我想知道与我相关的动作或位置”；
* 每个前面词都有自己的 Key $K₁,K₂,K₃$；
* 点积 $Q_3·K_j^\top$ 表示 $i = 3$ 处的词和每个词的“相关性”。

> $ Q_i \cdot K_j^\top $ 衡量的是「第 i 个 token 对第 j 个 token 的注意力强度」
> 也就是 —— 它“有多想看第 j 个 token”。

把所有这些相似度（score）通过 softmax 转成权重：
$$
a_{ij} = \text{softmax}\left(\frac{Q_i K_j^\top}{\sqrt{d_k}}\right)
$$
使得：
$$
\sum_j a_{ij} = 1
$$

> 这一步的作用：把“相关性分数”变成“注意力概率分布”。

再然后计算注意力得分：
$$
Z_i = \sum_{j=1}^{n} a_{ij} V_j
$$

也就是
* $V_j$：是第 j 个 token 里真正携带的“内容信息”（Value）；
* $a_{ij}$：告诉我们第 i 个 token 应该从第 j 个 token 中“取多少信息” 或 “取什么样？取哪些信息”。

总结一下：
| 符号             | 语义角色  | 数学意义        | 直觉意义           |
| -------------- | ----- | ----------- | -------------- |
| $Q_i$          | Query | “我想知道什么”    | 当前 token 提的问题  |
| $K_j$          | Key   | “我能提供什么”    | 每个 token 的信息标签 |
| $Q_i·K_j^\top$ | 相似度   | “我们匹配吗？”    | 决定注意力权重        |
| $V_j$          | Value | “我的实际内容”    | 被引用的信息         |
| $a_{ij}V_j$    | 加权信息  | “我从他那学到的内容” | 最终聚合的结果        |

## 7. $W_Q, W_K, W_V$ 是什么、为什么要它们

我们知道：
$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

其中：

* $X \in \mathbb{R}^{n \times d_{\text{model}}}$：输入序列；
* $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}$：三个不同的可学习矩阵；
* $Q,K,V \in \mathbb{R}^{n \times d_k}$。

直观理解：
| 名称    | 全称               | 含义（类比）         | 作用          |
| ----- | ---------------- | -------------- | ----------- |
| $W_Q$ | Query Projection | “我想问的问题的投影方式”  | 提取出用于提问的特征  |
| $W_K$ | Key Projection   | “我能回答的问题的投影方式” | 提取出用于匹配的特征  |
| $W_V$ | Value Projection | “我真正携带的内容”     | 提取出最终要传递的信息 |

Transformer 的输入向量 $X_i$ 需要：

* 担任“提问者”$Query$；
* 被别人“提问”$Key$；
* 提供“知识内容”$Value$。

但是：

> 这三种任务对信息的需求是不一样的。

还是这个例子：
> “猫 坐 在 垫子 上”

每个词经过 embedding 后变成向量 $Xᵢ$。
但是当模型处理“在”时：

* 它要提问：「我应该和谁关联？→ 动作相关的词」
  → 所以要从 $Xᵢ$ 中抽取出一种“提问特征”（通过 $W_Q$ 投影）

* 每个词要回答别人：「我是什么类型的信息？→ 我是一个动词/名词/位置词（这是一种具象的理解，模型处理的时候肯定是抽象的矩阵）」
  → 所以要从 $Xᵢ$ 中抽取“回答特征”（通过 $W_K$ 投影）

* 同时它还携带自身内容（词义、上下文 embedding）
  → 所以要再抽取“语义内容”（通过 $W_V$ 投影）

因此：$W_Q, W_K, W_V$ 是三个不同的「线性投影头」，
它们把相同的输入 $X_i$ 映射到不同的**语义空间**中。

从数学角度看,每个 token 向量 $X_i \in \mathbb{R}^{d_{\text{model}}}$，
经过这三组线性变换：

$
Q_i = X_i W_Q, \quad K_i = X_i W_K, \quad V_i = X_i W_V
$

这三个变换：

* 改变了向量的“语义方向”；
* 降维（或升维）到 $d_k$；
* 并且是**可学习的参数**，通过反向传播自动调整。

这里可以类比一下：
| 角色    | 代表      | 作用              | 类比              |
| ----- | ------- | --------------- | --------------- |
| $W_Q$ | 问问题的方式  | 把自己的想法转成“我要问什么” | 每个人整理自己想问的问题清单  |
| $W_K$ | 提供答案的方式 | 决定别人怎么看待我       | 每个人写下自己能回答的问题类别 |
| $W_V$ | 内容      | 真实要分享的知识        | 每个人准备的发言内容      |

最后：

* 每个人 $Query$ 根据问题清单 $Q_i$ 去问所有人 $Key$；
* 根据匹配程度 $Q·Kᵀ$ 决定听谁多；
* 然后用对应的 Value（发言内容）加权求和。

## 8. 交叉注意力机制

**交叉注意力**是：

> 一个序列的 Query 去“看”另一个序列的 Key、Value。

公式：
$$
Q = X_{\text{target}} W_Q, \quad K = X_{\text{source}} W_K, \quad V = X_{\text{source}} W_V
$$
$$
Z = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

区别：

* Q 来自 **目标序列**（target，比如 decoder 当前状态）；
* K,V 来自 **源序列**（source，比如 encoder 输出）。

举个翻译例子：

输入英文：

> “The cat sits on the mat.”

目标是输出中文：

> “猫 坐 在 垫子 上。”

在 Decoder 的生成过程中：

* Decoder 自己有一层「自注意力」：看自己生成到目前为止的 token；
* 还会有一层「交叉注意力」：用当前生成状态（Q）去看 Encoder 的输出（K,V）。

比如：

> 当 Decoder 正在生成 “垫子” 时，它的 Query 会去关注 Encoder 输出中对应 “mat” 的位置。

> 所以交叉注意力让“翻译端”去“关注输入端”的信息。

1. 自注意力（Self-Attention）

   * Q,K,V 都来自输入序列；
   * 让模型理解输入内部的结构。

2. Masked 自注意力（Self-Attention）

   * 只看自己和之前的 token；
   * 用于生成时的语言建模。

3. 交叉注意力（Cross-Attention）

   * Q 来自当前状态；
   * K,V 来自旁边输出；
   * 用于让生成端结合源信息。

## 9. 交叉注意力不会丢失信息吗？

这个问题的完整思考是这样的：

以 ALBEF 为例：
![ALBEF_Figure_1](../images/ALBEF_Figure_1.png)

> 三个encoder，一个image encoder、一个text encoder和一个multimodal encoder，然后text encoder后做Cross-Attention用text encoder中的Q和image encoder中的K和V，那这样不是就丢失text encoder中的K和V的信息了吗？

问题核心：

> “Cross-Attention 是不是覆盖了 text encoder 的信息？”

ALBEF 的架构可以拆成三部分：

| 模块                     | 输入         | 功能                                       |
| ---------------------- | ---------- | ---------------------------------------- |
| **Image Encoder**      | 图像输入       | 提取视觉特征（Self-Attention）                   |
| **Text Encoder**       | 文本输入       | 提取语言特征（Self-Attention）                   |
| **Multimodal Encoder** | image+text | 做跨模态交互（Cross-Attention + Self-Attention） |

> 即：图像编码、文本编码各自先建模内部结构（intra-modal），
> 然后再通过 multimodal encoder 进行跨模态对齐（inter-modal）。

Cross-Attention 计算过程为：

在 multimodal encoder 的 Cross-Attention 层中：
$$
Q = X_{\text{text}} W_Q, \quad K = X_{\text{image}} W_K, \quad V = X_{\text{image}} W_V
$$
然后：
$$
Z_{\text{text}} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这意味着：

* 每个文本 token 会“去看”图像中的所有视觉 token；
* 这一步计算得到的输出 $Z_{\text{text}}$，是融合了视觉信息的文本特征。

这里有个关键点：

> Cross-Attention 的输出不是直接替换掉文本的原表示，而是**在原文本特征的基础上融合图像信息**。

**(1) multimodal encoder 的输入是经过 text encoder 的输出**

也就是说：
$$
X_{\text{text}}^{(0)} = \text{TextEncoder}(text)
$$
这个已经是包含语言语义的高层表示。
Cross-Attention 只是再加一层「看图」的机制，不会覆盖掉原来的文本语义。

**(2) Cross-Attention 的输出会通过残差连接（residual connection）**

Transformer 每一层都有结构：
$$
\text{output} = \text{LayerNorm}(X + \text{Attention}(X))
$$

在 Cross-Attention 层中：
$$
\text{output} = \text{LayerNorm}(X_{\text{text}} + Z_{\text{text}})
$$

所以：

* $X_{\text{text}}$：保留原始文本自身语义；
* $Z_{\text{text}}$：来自图像的跨模态补充；
* 两者相加再归一化。

> 因此信息不是被“丢掉”，而是被**融合**进了新的 multimodal 表征中。

**(3) 在 multimodal encoder 内还有 Self-Attention**

* Cross-Attention 先注入跨模态信息；
* 再用 Self-Attention 让这些融合后的 token 相互建模；
* 所以最终输出包含了**文本上下文 + 图像语义**的综合表示。
