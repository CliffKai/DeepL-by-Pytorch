# Part 1 recap

1. 新的计算单元 —— 数据中心 (New unit of compute – the datacenter)
2. 多机扩展的理想目标 (What we want from multi-machine scaling) :
   - 线性内存扩展 (Linear memory scaling)
   - 线性算力扩展 (Linear compute scaling)
3. 简单的集合通信原语 (Simple collective comms primitives)

这三条便是构建现代 AI 基础设施的三大基石：
1. 宏观架构：把数据中心当成一台电脑用。
2. 性能目标：加多少卡，就得涨多少显存和速度（拒绝边际效应递减）。
3. 通信基础：需要标准化的通信方式来连接这些卡。

> 课程中，在评估一个分布式系统的性能时，通过计算**集合通信原语（Collective Communications Primitives）**的数量来进行推理，不会在这里深入探讨网速、带宽、延迟毫秒数等，也不去探讨通信算法的底层实现细节

## 1.1 basics about collective communication

首先来认识一下并行计算中五种最核心的 “集体通信” (Collective Communication) 操作。
![Figure_1](../images/CS336/CS336_Lecture_7_Figure_2.png)

这张图详细解释了并行计算中五种最核心的 **“集体通信” (Collective Communication)** 操作。

"集体通信"是指在一个并行计算集群中（例如，由多个 GPU 或多个服务器节点组成），所有参与的进程（图中的 "rank" 可以理解为一个个独立的计算单元，比如一个 GPU）**同时参与**的数据交换操作。

这些操作是实现大规模分布式计算的基石。

以下是图中五种操作的详细讲解：

### 1） All reduce (全归约)

* **操作前:** 每个 "rank"（rank 0, 1, 2, 3）都持有一份自己的输入数据（`in0`, `in1`, `in2`, `in3`）。
* **操作过程:**
    1.  **Reduce (归约):** 系统对所有 "rank" 的输入数据执行一个归约操作（例如 `sum` 求和）。`result = in0 + in1 + in2 + in3`。
    2.  **All (全部):** 将这个最终的计算结果（`out`）分发给**所有**的 "rank"。
* **操作后:** **所有 "rank"** 都拥有了相同的、归约后的最终结果 `out`。
* **常见用途:** 在数据并行训练中，每个GPU计算出自己的梯度（`in0`...`in3`），`All reduce` 用来计算所有梯度的总和，并将这个总和（`out`）分发回给每个GPU，以便它们能同步更新模型。

* **通信量：** 一次 all reduce 的通信量大概是本次 all reducing 的数据的两倍。

### 2） Broadcast (广播)

* **操作前:** 只有一个 "rank"（称为 "root"，例如图的 `rank 2`）持有数据（`in`）。
* **操作过程:** "root" 进程将其数据（`in`）发送给组内所有其他的进程。
* **操作后:** **所有 "rank"** (包括 "root" 自己) 都拥有了 `rank 2` 原始数据的副本（`out`）。
* **常见用途:** 在训练开始时，由 "root" 进程（rank 0 或 2）加载模型权重，然后 `Broadcast` 给所有其他GPU，确保大家从同一个起点开始。

### 3） Reduce (归约)

* **操作前:** 和 `All reduce` 一样，每个 "rank" 都持有一份自己的输入数据（`in0`, `in1`, `in2`, `in3`）。
* **操作过程:** 系统对所有 "rank" 的输入数据执行一个归约操作（例如 `sum` 求和）。
* **操作后:** **只有 "root" 进程**（图中指定了 `rank 2`）接收并存储了这个最终的归约结果（`out`）。其他 "rank" 不会收到这个结果。
* **区别 (Reduce vs All reduce):** `Reduce` 只把结果给 "root"；`All reduce` 把结果给所有人。

### 4） All Gather (全收集)

* **操作前:** 每个 "rank" 都持有**一部分**数据（`in0`, `in1`, `in2`, `in3`）。
* **操作过程:** 系统从所有 "rank" 收集它们各自的数据，并将这些数据**按 "rank" 的顺序拼接**（concatenate）成一个完整的大数据集。然后，系统将这个完整的数据集分发给所有 "rank"。
* **操作后:** **所有 "rank"** 都拥有了完整的、拼接后的数据集（`out`），这个 `out` 包含了 `in0`, `in1`, `in2`, `in3` 的所有内容。
* **常见用途:** 在模型并行或张量并行中，如果一个张量被切分到不同GPU上，`All Gather` 可以将这些分片收集起来，在每个GPU上还原出完整的张量。

### 5） Reduce Scatter (归约-分散)

* **操作前:** 每个 "rank" 都持有一份**完整的输入数据**。请注意图中每个 "rank" 的输入数据都被分成了四块（对应4个 "rank"）。
* **操作过程:** 可以先将每一块上的数据看作被分为了四块，`in0` 变成了 `in00,in01,in02,in03`，`in1,in2,in3`同理，然后每块 `rank` 分别负责自己对应的那一部分，`rank0` 负责 `out0 = in00 + in10 + in20 + in30`，`rank1,rank2,rank3` 同理。
* **操作后:** 每个 "rank" 只接收到**对应于自己 "rank" 编号的那一块**数据的归约结果。
* **常见用途:** 这是 `All reduce` 的一种高效变体。在某些算法中，每个进程在归约后只需要自己负责的那部分结果，使用 `Reduce Scatter` 可以减少不必要的数据传输。

> 注意：这里只是简单的讲述了一下数据的起始和最终的分布情况，具体到实际计算中会有很多各种各样针对硬件设备的优化，会有着很多很多的方法。

## 1.2 Important detail

![Figure_2](../images/CS336/CS336_Lecture_7_Figure_3.png)

这两者在结果和性能能耗等各方面就是一样的，这里的意思就是 all reduce 可以看作是 reduce-scatter + all-gather 两次操作之和。

还有一个角度可以更清晰的理解为什么等式两边等价，那就是：为什么一次 All-Reduce 操作中，**每张 GPU** 需要发送和接收的数据量大约是数据量的 2 倍？
> 这里指 **Ring All-Reduce**（这是目前深度学习训练中最常用的算法，如 NVIDIA NCCL 库所采用）


如果待传输数据大小（比如梯度）为 $M$ 字节，GPU 数量为 $N$，那么每张 GPU 需要传输的总数据量是：
$$\text{Traffic per GPU} = 2 \cdot \frac{N-1}{N} \cdot M$$
当 GPU 数量 $N$ 较大时（比如 $N \ge 8$），$\frac{N-1}{N}$ 趋近于 1，因此我们通常直接估算为：**$2M$**。
Ring All-Reduce 详细推导过程:
为了理解这个 $2M$ 是怎么来的，我们需要看 Ring All-Reduce 算法的两个阶段。假设有 $N$ 个 GPU，围成一个环：
#### 第一阶段：Scatter-Reduce（分散归约）
* **目标**：把巨大的梯度向量切成 $N$ 块，让每个 GPU 最终负责计算其中 1 块的总和。
* **过程**：每个 GPU 将自己的数据传给下一个 GPU，经过 $N-1$ 步传输后，每个 GPU 都拥有了它负责的那一小块数据的“全剧终总和”。
* **通信量**：每个 GPU 发送了 $\frac{N-1}{N} \times M$ 的数据。
#### 第二阶段：All-Gather（全收集）
* **目标**：现在每个 GPU 只知道自己那一小块的和，它需要把这个和分享给其他所有 GPU，同时从其他 GPU 拿回剩下的部分。
* **过程**：每个 GPU 把它手里那块计算好的总和传给下一个 GPU，再经过 $N-1$ 步。
* **通信量**：每个 GPU 又发送了 $\frac{N-1}{N} \times M$ 的数据。
#### 总计
$$\text{Total} = \text{Scatter-Reduce} + \text{All-Gather} = \frac{N-1}{N} M + \frac{N-1}{N} M = 2 \frac{N-1}{N} M$$

**基本就是：梯度多大，通信量就是它的 2 倍。**

> 这是详细的理解过程：
> #### **1. 准备阶段**
> * **硬件：** $N$ 个 GPU，标记为 $GPU_0, GPU_1, \dots, GPU_{N-1}$。
> * **数据：** 模型梯度总大小为 $M$（字节）。
> * **切分：** 将每个 GPU 上的梯度向量逻辑切分为 $N$ 个小块（Chunks）。
>     * **单块大小：** $\frac{M}{N}$。
>     * **标记：** $GPU_i$ 持有的数据第 $j$ 块数据记为 $G_{i,j}$。
> #### **2. 计算阶段 (Local Computation)**
> * 各 GPU 根据分配给自己的那部分 Batch 数据，独立进行反向传播。
> * **结果：**
>     * $GPU_0$ 算出自己的梯度向量 $G_0$（包含 $G_{0,0}, \dots, G_{0,N-1}$）。
>     * $GPU_1$ 算出自己的梯度向量 $G_1$（包含 $G_{1,0}, \dots, G_{1,N-1}$）。
>     * ... 以此类推。
>     * *此时，所有 GPU 的梯度数值都不一样。*
> #### **3. 第一阶段：Reduce-Scatter (分散求和)**
> * **目标：** 让每个 GPU 负责计算**其中某一块**梯度的全局总和。
>     * 让 $GPU_0$ 拥有第 0 块的全局和：$\sum_{i=0}^{N-1} G_{i,0}$。
>     * 让 $GPU_1$ 拥有第 1 块的全局和：$\sum_{i=0}^{N-1} G_{i,1}$。
>     * ...
>     * 让 $GPU_k$ 拥有第 $k$ 块的全局和。
> * **过程：**
>     * 为了凑齐某一块的全局和，需要所有其他 $N-1$ 个 GPU 把它们手里的那块对应数据传出来。
>     * 每个 GPU 都需要把自己手里**除了自己负责的那一块以外**的所有数据块发送出去。
> * **通信量计算 (每张卡)：**
>     * 总共 $N$ 块，自己留 1 块，发送 $N-1$ 块。
>     * 发送量 = $(N-1) \times \text{单块大小}$
>     * $$\text{Traffic}_{scatter} = (N-1) \times \frac{M}{N} = \frac{N-1}{N} M$$
> #### **4. 第二阶段：All-Gather (全收集/广播)**
> * **当前状态：** Reduce-Scatter 结束后，$GPU_k$ 手里只有第 $k$ 块的完美梯度的和，其他 $N-1$ 块它是缺失的（或者说是旧的）。
> * **目标：** 让每个 GPU 都拥有**完整的**、包含所有 $N$ 块的全局梯度和。
> * **过程：**
>     * 每个 GPU 把自己刚刚算好的、负责的那**唯一一块**完整梯度和，分享给其他所有的 $N-1$ 个 GPU。
>     * （在 Ring 算法中，这意味着这块数据要在环里转 $N-1$ 次才能到达所有人手中）。
> * **通信量计算 (每张卡)：**
>     * 每个 GPU 需要发送自己负责的那 1 块数据给其他 $N-1$ 个节点（逻辑上）。
>     * 发送量 = $(N-1) \times \text{单块大小}$
>     * $$\text{Traffic}_{gather} = (N-1) \times \frac{M}{N} = \frac{N-1}{N} M$$
> #### **5. 总计 (Total Communication Cost)**
> 将两个阶段的通信量相加：
> $$
> \text{Total Traffic} = \text{Reduce-Scatter} + \text{All-Gather}
> $$
> $$
> = \left( \frac{N-1}{N} M \right) + \left( \frac{N-1}{N} M \right)
> $$
> $$
> = 2 \cdot \frac{N-1}{N} \cdot M
> $$
> #### **结论：**
> 当 $N$（GPU 数量）很大时，比如 $N=8, 64, 1024$：
> $$\frac{N-1}{N} \approx 1$$
> **所以，单次 All-Reduce 操作，每张 GPU 的总通信量近似为：**
> $$\mathbf{2M}$$

### 1） TPU 与 GPU的设计对比

![Figure_3](../images/CS336/CS336_Lecture_7_Figure_4.png)

| 维度 | **Google TPU** | **NVIDIA GPU** |
| :--- | :--- | :--- |
| **网络拓扑** | **2D/3D Toroidal Mesh (环形网格)** | **Hierarchical All-to-All (层级化全互联)** |
| **连接方式** | **芯片直连 (Direct Connect)**。<br>没有中心交换机，芯片只和邻居（上下左右）相连。 | **交换机网络 (Switch Fabric)**。<br>通过 NVSwitch/InfiniBand 交换机，实现任意节点间的高速跳转。 |
| **设计哲学** | **“专车专用，够用就好”**。<br>针对深度学习中最确定的通信模式（如矩阵乘法、All-Reduce）进行极致精简和优化。 | **“大力出奇迹，灵活第一”**。<br>用极其昂贵的网络带宽和交换设备，解决所有可能的通信问题，不让通信成为瓶颈。 |

### 2） Dense Training 计算场景

*代表模型：Llama 2, BERT, 早期 GPT-3*

在这类模型中，每一层的计算都需要所有参数参与，通信模式主要是 **All-Reduce**（同步梯度）。

  * **TPU 的优势：极致的性价比与能效**

      * **原因：** 最高效的 `All-Reduce` 算法通常是 **Ring-based（环状）** 的。这种算法只要求节点和“邻居”通信。
      * **结论：** TPU 的网格结构（Mesh）**天生匹配** Ring 算法。硬件上没有浪费任何一根连线，也没有昂贵的交换机闲置。这就是为什么课程中讲的“如果只考虑集体通信，TPU 的设计更合理”。它用最少的硬件实现了同等的效果。

  * **GPU 的表现：**

      * **原因：** GPU 的全互联网络当然也能跑 Ring 算法，但就像“开着法拉利去送快递”，虽然送得很快，但有些大材小用。NVIDIA 必须通过软件（NCCL）来模拟环状路径，虽然性能很强，但硬件成本（BOM Cost）极高。

### 3） Mixture-of-Experts 计算场景

*代表模型：GPT-4, Mixtral 8x7B, DeepSeek-V3*

这类模型引入了“稀疏性”。对于每一个 token，网络会动态选择几个“专家”（Experts）来处理，而这些专家可能分布在不同的显卡上。

  * **GPU 的优势：统治级的灵活性**

      * **原因：** MoE 的通信模式是 **All-to-All (Shuffle/Dispatch)**。
          * 比如：GPU 1 上的数据突然想找 GPU 56 上的专家，下一秒 GPU 1 上的数据又要找 GPU 200 上的专家。
          * 这种通信是**随机的、跳跃的**。
          * NVIDIA 的 **NVSwitch** 允许 GPU 1 直接跳到 GPU 56，延迟极低。
      * **结论：** 在处理这种乱序、长距离通信时，GPU 的全互联架构展现了碾压级的优势。

  * **TPU 的劣势：多跳带来的延迟 (Multi-hop Latency)**

      * **原因：** 在 Mesh 网络中，如果 TPU 1 想发数据给 TPU 56，数据可能需要经过 2-\>3-\>...-\>55 这么多跳。
          * 这不仅增加了**延迟**（Latency）。
          * 更糟糕的是会造成**数据拥塞**（Congestion）。中间的芯片不仅要算自己的数，还得帮别人传数据，带宽被挤占了。
      * **结论：** 虽然 TPU 也可以通过软件优化 MoE，但在大规模 MoE 训练上，TPU 的网格结构会面临严重的通信瓶颈，效率通常不如同档次的 GPU 集群。

# Part 2 Standard LLM parallelization primitives

这一部分讲述的是标准 LLM 的并行化原语。

并行化训练 LLM 的三大核心技术路线：
1. Data parallelism (数据并行，分割数据)：
   - Naïve data parallel
   - ZeRO levels 1-3
2. Model parallelism (模型并行，分割模型，训练大模型的方法)：
   - Pipeline parallel (流水线并行)
   - Tensor parallel (张量并行)
3. Activation parallelism (激活值并行，处理长序列)：
   - Sequence parallel (序列并行)

## 2.1 Naïve data parallel

### 1）.**“朴素数据并行”（Naïve Data Parallelism）** 的基本原理及其优缺点。

它是分布式训练中最基础的形式，理解了它，就能理解为什么后来会出现 ZeRO、FSDP 这些更高级的技术。

1. 随机梯度下降（SGD）的标准公式：
$$\theta_{t+1} = \theta_t - \eta \sum_{i=1}^B \nabla f(x_i)$$
* **含义：** 为了更新模型参数 $\theta$，我们需要计算一个批次（Batch Size = $B$）中所有数据的梯度总和。
* **挑战：** 当 $B$ 很大时，单张卡算不过来，或者存不下这么多数据。

2. Naïve parallelism 的做法：
* **核心策略：** 将这个大小为 $B$ 的大批次数据，切分成 $M$ 份（假设有 $M$ 台机器/GPU）。
* **分工：** 每台机器只拿到一小部分数据（$B/M$），算出这一小部分的梯度。
* **同步：** 算完后，所有机器之间进行通信（Exchange gradients），把大家的梯度加起来平均，确保每个人更新后的模型参数是一模一样的。

3. How does this do?
这部分是这张图的精华，它从三个维度评估了这种简单粗暴的方法：

* **计算扩展性 (Compute scaling) —— 好**
    * "each GPU gets B/M examples."
    * 任务被完美平分了。GPU 越多，每张卡处理的数据越少，计算速度越快。

* **通信开销 (Communication overhead) —— 一般**
    * "transmits 2x # params every batch."
    * 这正好验证了你刚才问的问题！每跑完一个 Batch，都需要进行一次 All-Reduce，通信量是模型参数量的 **2 倍**。
    * 如果 Batch 很大（计算时间很长），这部分通信时间可以被掩盖；如果模型太大或 Batch 太小，通信就会成为瓶颈。

* **内存扩展性 (Memory scaling) —— 极差**
    * "none. Every GPU needs # params at least"
    * **这是朴素数据并行的最大死穴。**
    * 即使你有 1000 张 GPU，**每张 GPU 都必须完整地存储一份模型参数**。
    * **结果：** 如果模型是 80GB，而 GPU 只有 40GB 显存，哪怕有 1 万张卡也跑不起来，因为单卡连模型都装不下。
