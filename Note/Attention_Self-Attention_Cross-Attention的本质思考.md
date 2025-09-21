灵感来自：[注意力机制的本质|Self-Attention|Transformer|QKV矩阵](https://www.bilibili.com/video/BV1dt4y1J7ov/?share_source=copy_web&vd_source=608471d0e25c02d240b92470bd78f213)

# 1 什么是注意力机制？

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

# 2 Attention

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

# 3 Self-Attention

如果 Q、K、V 都是从同一个输入 X 通过不同的线性变换得到的：

$$
Q = XW_Q, K = XW_K, V = XW_V
$$

所以这就叫做 Self-Attention，自注意力机制。

# 4 Cross-Attention

若是 Q、K、V 不是都来自同一个输入 X，而是 Q 来自一个输入 X1，K 和 V 来自另一个输入 X2：

$$
Q = X_1W_Q, K = X_2W_K, V = X_2W_V
$$

所以这就叫做 Cross-Attention，交叉注意力机制。

# 5 Attention 中的 $ W_Q, W_K, W_V $

那么 $W_Q, W_K, W_V$ 是什么呢？

因为在之前的那个例子中 Key 和 Value 都是显示给出的，所以我们可以直接计算。但是在实际应用中，Key 和 Value 往往是隐含的，我们需要通过某种方式来生成它们。通常，这些矩阵是通过训练学习得到的参数，用于将输入特征映射到查询、键和值的空间中。
这也就是$W_Q, W_K, W_V$ 的由来。
而 Self-Attention 和 Cross-Attention 的区别就在于 Q、K、V 是不是来自同一个输入。
