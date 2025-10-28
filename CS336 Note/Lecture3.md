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

# 模型训练稳定角度的 softmax

## 输出层的 softmax

## attention 中的 softmax
