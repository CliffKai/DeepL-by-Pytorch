方法：研读相关论文，分析其中的变与不变，找出关键的设计要素，让 Transformer 发挥最佳性能——**实践出真知**

# Transformer 架构

![the‘original'transformer](../images/CS336/CS336_Lecture_3_Figure_1.png)

![varient transformer](../images/CS336/CS336_Lecture_3_Figure_2.png)

![the architecture](../images/CS336/CS336_Lecture_3_Figure_3.png)

![architecture variations](../images/CS336/CS336_Lecture_3_Figure_4.png)

# 归一化层

## Pre-vs-post norm

![Pre-vs-post norm](../images/CS336/CS336_Lecture_3_Figure_5.png)

Pre-Norm 的核心优势：保持 residual path 的数值稳定性，让梯度能平滑地穿过上百层网络。

## LayerNorm vs RMSNorm

![LayerNorm vs RMSNorm](../images/CS336/CS336_Lecture_3_Figure_6.png)

![LayerNorm vs RMSNorm](../images/CS336/CS336_Lecture_3_Figure_7.png)

1. **计算更少**：LN 需要计算均值和方差，RMSNorm 只要**平方和一次归约**即可；同时相比于 LN,RMSNorm 还少加了一个偏置项，虽然加法等在计算中占比很少，但是其内存调转引起的时延占比非常非常高（见上图图7）；
2. **反向传播简单**：没有取均值与方差项，梯度表达式更简单，数值更稳定（尤其混合精度、低比特场景）。
3. **能够保留均值（偏移）信息**：LN 强制零均值，RMSNorm 不会抹去均值方向上的信息；实践上，在**预归一化（Pre-Norm）Transformer**的残差通路里，这常常配合得很好（许多现代解码式 LLM——如 LLaMA 系列、Mistral 等——采用 RMSNorm）。
详细讲解可看[assignment1各模块实现讲解](https://github.com/CliffKai/assignment1-basics/blob/main/Note/assignment1%E5%90%84%E6%A8%A1%E5%9D%97%E5%AE%9E%E7%8E%B0%E8%AE%B2%E8%A7%A3.ipynb) 9.4部分。

## dropping bias terms

![LayerNorm vs RMSNorm](../images/CS336/CS336_Lecture_3_Figure_8.png)

可以总结成以下三条：

1. **归一化已抵消偏移作用**：
　在 Pre-Norm + RMSNorm 架构中，输入已被标准化，bias 只会重新引入不必要的偏移，破坏数值平衡。

2. **提升训练稳定性**：
　去掉 bias 后，层间激活均值更稳定、梯度分布更平滑，尤其在混合精度训练中更容易收敛。

3. **减少内存与计算负担**：
　少一组参数、少一次广播加法，对大模型可节省显存并提高计算效率。

# 激活函数

![](../images/CS336/CS336_Lecture_3_Figure_9.png)

# 位置编码

RoPE
![](../images/CS336/CS336_Lecture_3_Figure_10.png)

# 超参数

## FFN

传统 GELU-MLP 用 $d_{\text{ff}}=4d$ 且只有两次投影（$d\to4d\to d$），参数量约 $8d^2$。
SwiGLU 有 **三** 次投影（两入一出）：参数量 $\approx 3 d\cdot d_{\text{ff}}$。
为了和传统 MLP 的参数/算力预算相当，令 $3 d\cdot d_{\text{ff}}\approx 8d^2\Rightarrow d_{\text{ff}}\approx \frac{8}{3}d$。

## 模型维度

$(head_{dim} * head_{num}) : d_{model}$ 一般取 $1:1$。

$d_{model} : n_{layer}$ 一般取 $128$ 左右。

# 权重衰减

现在在 LLM 中使用 weight decay 并不是为了防止过拟合（regularization），而是为了优化训练动态，让模型训练得更稳定、更有效。

这一部分课程中没有讲原因，我个人理解如下：

在现代大模型（比如 70 层、100 层 Transformer）里，一个输入 token 的信号会：
$$
x_0 \xrightarrow{W_1} x_1 \xrightarrow{W_2} x_2 \xrightarrow{W_3} \dots \xrightarrow{W_L} x_L
$$
也就是说它会经过成百上千次线性变换 + 非线性激活 + 残差连接。

如果其中某一层的参数 $W_i$ 过大，比如 scale 变成原来的 1.2 倍，
它会在信号流的每一层都放大一点点：
$$
|x_L| \propto \prod_{i=1}^L |W_i|
$$
即便每层只放大 1.01 倍，经过 100 层也变成：
$$
1.01^{100} \approx 2.7
$$
这会造成激活分布漂移、梯度爆炸、正则化不再有效。

最重要的是：**Adam 优化器不会“自然发现”这个问题**

因为 Adam 对梯度进行了逐维归一化，它看不到参数的绝对大小，它只管“梯度相对变化”，不会自动去约束参数的范数。

所以当某些层的参数变大时：

* Adam 不会惩罚它；
* 正则化会“部分抵消”表面上的放大；
* 但**反向传播的梯度分布却会一点一点漂移**；
* 结果模型整体 scale 不断上升（即所谓的 “norm inflation”）。

Weight Decay 在这里起到的作用就像：“在每一层都安装一个微弱的阻尼器，防止参数的尺度慢慢积累到危险值”。

它确保：
$$
|W_i| \text{ 在整个训练过程中保持在合理范围内。}
$$

这就让深层堆叠不会发生指数放大，从而使得学习率调度仍然有效。

# 模型输出前的 softmax + z-loss

我废了好大的劲才看明白这个问题，**才看明白这个问题！！！（心累😩）**

## 1. 为什么 softmax 会出现不稳定的问题

要弄懂这个问题，就要从交叉熵函数的优化开始。

### 1）. **交叉熵损失函数的计算**

交叉熵损失函数通常用于分类任务，它的定义是：
$$
\mathcal{L} = -\log(p_y)
$$
其中 $ p_y $ 是模型对目标类别 $ y $ 的预测概率，通常通过 softmax 函数计算出来。具体来说，softmax 对每个类别 $ k $ 的概率计算为：
$$
p_k = \frac{e^{u_k}}{\sum_j e^{u_j}}
$$
其中，$ u_k $ 是类别 $ k $ 的 logit（即模型的原始输出），$ p_k $ 是 softmax 计算出的概率。

## 2. **softmax 对全局偏移的“平移不敏感性”**

**关键概念**：softmax 函数对所有 logits 加上一个常数 $ c $ 后，输出的概率分布**不变**。换句话说：
$$
p_k = \frac{e^{u_k + c}}{\sum_j e^{u_j + c}} = \frac{e^{u_k}}{\sum_j e^{u_j}}
$$
也就是说，如果你对 logits $ u_k $ 中的每个值都加上一个常数（例如，给每个 logit 都加上一个常量 $ c $），softmax 输出的概率分布不会受到影响。

**为什么没有影响？**

* softmax 计算的是每个类别的概率**相对其他类别**的概率。因此，所有的 logits 都加上同一个常数 $ c $，这些常数对概率分布的相对关系没有影响。

### 3）. **全局偏置的“无目的漂移”**（我知道前面的应该是大家都明白的道理，前面只是铺垫，这个是关键）

* 当我们训练神经网络时，**我们并不直接控制 logits 的数值**，而是通过梯度下降来优化损失函数。假设我们在训练过程中并没有约束 logits 的整体数值尺度，那么在某些情况下，**所有 logits 可能会“无目的地漂移”**。
* 具体来说，**如果你没有对 logits 做限制，优化过程中可能会出现这样的情况**：logits 可能会随着训练不断增大或减小，但这并不会影响 softmax 输出的概率分布，因为 softmax 本身对所有 logits 加同一个常数不敏感。
* 也就是说，我全都增大到特别大也没关系，只要保证正确标签的那个值输出的更大即可。
* 但这样的“漂移”可能会引发数值不稳定性，比如：

  * **数值溢出（overflow）**：如果 logits 的值变得非常大，指数函数 $ e^{u_k} $ 可能会导致数值溢出（这个问题相对来说很好解决，只需要所有值都减去 $max(u_i)$ 即可，也就是取 $ {\sum_j e^{{u_j}-{max(u_j)}}} $，但是，这解决的只是计算时的问题，解决不了训来年过程中梯度消失或爆炸的问题。
  * **梯度消失或爆炸**：在深层网络中，logits 的尺度如果不加以控制，梯度在反向传播时可能会变得非常小（梯度消失）或非常大（梯度爆炸），影响模型训练的稳定性。

> 这里很简单，计算一下 $ u_1 = 1000, u_2 = 1001, u_3 = 2000 $ 情况下的 $ p_k $ 就明白了。

### 4）. **为什么需要 z-loss 约束这种漂移？**

z-loss 的作用就是通过惩罚项约束 logits 的整体尺度，**避免 logits 之间的差距过大**，从而**减少这种漂移**，让训练过程更加稳定。具体来说，z-loss 惩罚的是 $ \log Z(x) ,  的平方，强制 logits 的总体尺度保持在一个合适的范围内，从而避免了上述的数值不稳定和梯度问题。

## 2. **z-loss 的原理是什么？**

### **目标：控制 logits 的整体尺度**

z-loss 的设计目的是确保 **softmax 的归一化因子**（即 $Z(x)$）保持在一个稳定的范围内，特别是希望它接近于 1，避免出现数值溢出或梯度爆炸的问题。z-loss 的公式如下：
$$
\mathcal{L}_{\text{aux}} = -\alpha (\log Z(x))^2
$$
其中，$ Z(x) = \sum_j e^{u_j(x)} $ 是 logits 归一化因子，$\alpha$ 是一个很小的常数，控制 z-loss 的惩罚强度。

### **z-loss 惩罚的是 $\log Z(x)$**

* **$ Z(x) $** 是 softmax 归一化因子，即所有 logits 指数的总和：
  $$
  Z(x) = \sum_{r=1}^{|V|} e^{u_r(x)}
  $$
  它决定了 softmax 输出的概率的大小。

* **$\log Z(x)$** 是 $Z(x)$ 的对数，它是衡量 logits “整体尺度”的一个重要指标。**如果 $\log Z(x)$ 很大或很小**，说明 logits 的数值范围不平衡（这里可以自己手动推导一下，因为地方写不下我就直接给结论：$\log Z(x) ≈ logits 的“平均数值” + 一个常数\log|V|$），可能导致数值不稳定、梯度消失/爆炸等问题。

* **惩罚项 $(\log Z(x))^2$**：z-loss 惩罚的是 $\log Z(x)$ 的平方。**当 $\log Z(x)$ 离 0 越远时，z-loss 的惩罚越大**，这样就会通过梯度下降的方式，强迫模型调整 logits 的尺度，使得 $\log Z(x)$ 接近 0。理论上，**当 $\log Z(x)$ 趋近 0 时，模型的 logits 就会保持在一个合适的范围内**，从而避免了数值不稳定和梯度问题。

## 3. **为什么 z-loss 能稳定 softmax 的输出？**

### **1) 避免 logits 差距过大**

logits 中不同类别之间的差距过大，会导致 **softmax 输出过于极端**，比如概率分布几乎是 one-hot（即一个类别的概率接近 1，其他类别的概率接近 0）。这种极端的输出会使得梯度变得非常小（因为 softmax 的梯度和概率之间存在反比关系）。使用 z-loss 后，**$\log Z(x)$ 趋向 0**，意味着 logits 的整体尺度得到了限制，类别之间的差距不会太大，**softmax 输出变得更加平滑**，从而避免了梯度消失的问题。

例如，假设在没有 z-loss 的情况下，logits 为：
$$
u_1 = 1000, \quad u_2 = 1000, \quad u_3 = -1000
$$
softmax 输出可能会是：
$$
p_1 \approx p_2 \approx 0.5, \quad p_3 \approx 0
$$
此时，由于 logits 之间差距过大，**softmax 输出极端化**，导致反向传播时梯度几乎为 0，不利于训练。加上 z-loss 后，模型会被迫让 $\log Z(x)$ 接近 0，保持 logits 之间的差距较小，从而保证 softmax 输出的平滑性。

### **2) 使得 logits 规模更可控**

z-loss 的一个重要作用是**控制 logits 的整体规模**。在训练中，如果 logits 的值太大或太小，可能导致 **数值溢出/下溢**，或者 **梯度爆炸/消失**。z-loss 通过惩罚 $\log Z(x)$ 的平方，**限制了 logits 规模的过大**，从而避免了指数计算时的溢出或过小的数值导致下溢。

例如，在没有 z-loss 的情况下，如果 logits 非常大：
$$
u_1 = 10000, \quad u_2 = 10000, \quad u_3 = 10000
$$
softmax 输出将会出现溢出，导致数值不稳定。使用 z-loss 后，通过限制 $\log Z(x)$ 使其接近 0，模型会减小 logits 的规模，从而避免了这种不稳定。

### **3) 平滑梯度，避免梯度爆炸**

通过约束 logits 的尺度，z-loss **平滑了梯度的计算**，使得训练过程中梯度不会变得过大或过小。当 logits 的差距过大时，softmax 的梯度会变得非常小（如果某个类别的概率接近 1，其他类别接近 0，梯度几乎为 0）。这种情况下，模型更新缓慢，甚至会导致训练停滞。通过 z-loss，**模型强制 logits 之间的差距适中**，梯度变得更均匀，有利于模型稳定收敛。

### **4) 自归一化特性**

z-loss 通过强制 $\log Z(x)$ 接近 0，使得 logits 的输出变得自归一化（self-normalizing）。**这种自归一化特性**使得模型的 softmax 输出在数值上变得更加稳定，从而避免了由于 logits 极端化所导致的训练问题。

例如，假设在没有 z-loss 的情况下，logits 输出的差距非常大，可能导致 softmax 输出非常不稳定。而加上 z-loss 后，模型会控制 logits 的整体尺度，使得输出概率保持在合适范围内，进而提高训练的稳定性。

# Attention 中的 softmax + QK norm

