1. How long would it take to train a 70B parameter model on 15T tokens on 1024 H100s?
- total_flops = 6 * 70e9 * 15e12
- h100_flop_per_sec = 1979e12 / 2 （这里的除 2 是因为 1979e12 的是针对稀疏矩阵而言的，对于 dense 的来说计算能力是要再削弱一些的，并不是后面的 mfu）
- mfu = 0/5
- flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24
- days = total_flops / flops_per_day



2. What's the largest model that can you train on 8 H100s using AdamW(naively)?
- h100_bytes = 80e9
- bytes_per_parameter = 4 + 4 + (4 + 4)
- num_parameters = (h100_bytes * 8) / bytes_per_parameter

**注**：4 + 4 + (4 + 4)的意思
- 4 (parameters): 假设模型参数本身以 FP32 (4字节) 存储。
- 4 (gradients): 假设梯度也以 FP32 (4字节) 存储。
- (4 + 4) (optimizer state): 这是 AdamW 的优化器状态，即一阶矩 (m) 和二阶矩 (v)，它们都是 FP32 (各4字节)。

# 存储

**深度学习中常见的有三种数据存储方式：**
1. FP32：1-8-23
2. FP16：1-5-10
3. BF16：1-8-7

一般计算时常用 BF16，但为了保证训练稳定，存储优化器状态和参数时还是需要 FP32。

**什么是混合精度训练？**

混合精度训练一般有这样的的一个思想：“用低精度（FP16/BF16）加速计算，用高精度（FP32）保持稳定性”，具体是怎么做的呢？
- 前向传播与反向传播时使用 BF16 进行计算，模型权重与优化器状态（无论是存储还是更新）使用 FP32 来进行更新，以此平衡计算效率和模型稳定性；
- 同时在维持这个思想的基础上，也会对不同的计算模块进行区分，在不同算子之间使用不同的精度策略，所以一些方法中可能会在注意力机制计算的时候使用 FP32，在其他矩阵计算时使用 BF16。

总结：我们一般希望长久保留或是很重要的数值与计算使用 FP32，但是对于一般计算或是临时的一些东西就是用 BF16 等低精度的方式处理。

# MFU

MFU（Model FLOPs Utilization，模型FLOP利用率）是指模型实际处理的吞吐量（如每秒处理的tokens数量）与硬件理论最大FLOP（每秒浮点运算次数）吞吐量之间的比率。
这个指标用于衡量模型在训练或推理过程中，是否充分利用了硬件的计算能力。

一般认为 MFU 大于0.5就是好的。

# 计算

对于一个参数量为 P 的模型，每前向处理 T 的数据，要计算的浮点次数为 $2 * P * T$，每反向一次要计算的浮点次数为 $4 * P * T$。

所以一般估算的时候，总的浮点计算量就是**6倍**的模型参数量和训练数据量的乘积。