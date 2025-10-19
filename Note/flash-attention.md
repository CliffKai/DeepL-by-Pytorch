# 思想

1. **Attention** 计算方法：
$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$
如果序列长度是 $n$，那么 $QK^\top$ 的大小就是 $n^2$。所以一般 Attention 计算方法的 **时间复杂度** 是 $O(n^2 d)$，**显存复杂度** 也是 $O(n^2)$,这在长序列上非常吃显存。

2. **FlashAttention 的核心思想**：
不去显式地存储整个 $QK^\top$ 矩阵，而是 **分块（tiling）计算 +  Online softmax**。      
因为最后我们只需要 $softmax(QK^T)V$ 的结果，而不需要中间完整的 $QK^T$。         
也就是说我们可以：
   - 把 $QK^\top$ 拆成一小块一小块计算（比如 128×128 的块），
   - 在每一块上算完后立即更新 softmax 的中间结果，
   - 最后得到整体输出。            

这样做的结果是：
   - 显存复杂度从 (O(n^2)) 降到了 (O(n))；
   - 而且可以更好地利用 GPU 的高速缓存（比如 shared memory）。

# Online softmax

首先我们来看一般的 softmax：

$$
y_i = \frac{e^{x_i - \max_{k=1}^V x_k}}{\sum_{j=1}^V e^{x_j - \max_{k=1}^V x_k}}
$$

1.  $m₀ ← −∞$
2.  $for k ← 1, V do$
3.  &emsp;&emsp;$mₖ ← max(mₖ₋₁, xₖ)$
4.  $end for$
5.  $d₀ ← 0$
6.  $for j ← 1, V do$
7.  &emsp;&emsp;$dⱼ ← dⱼ₋₁ + e^{xⱼ - m_V}$
8.  $end for$
9.  $for i ← 1, V do$
10. &emsp;&emsp;$yᵢ ← e^{xᵢ - m_V} / d_V$
11. $end for$

我们需要等所有要计算的值都得到后计算出 $max(x_i)$ 然后才能进行计算。

再来看 Online softmax：

1.  $m₀ ← −∞$
2.  $d₀ ← 0$
3.  $for j ← 1, V do$
4.  &emsp;&emsp;$mⱼ ← max(mⱼ₋₁, xⱼ)$
5.  &emsp;&emsp;$dⱼ ← dⱼ₋₁ × e^{mⱼ₋₁ - mⱼ} + e^{xⱼ - mⱼ}$
6.  $end for$
7.  $for i ← 1, V do$
8.  &emsp;&emsp;$yᵢ ← e^{xᵢ - m_V} / d_V$
9.  $end for$

# 分块计算

