首先，我强烈推荐还是看一下原课程比较好，课程中的思维和各种方法真的都非常值得学习。

下面总结一下我们一般使用 GPU 进行计算的五种方法以及各自的优劣：
1. **manual**：
   - 自己手动实现，这种方式实现起来是最简单的，也就是我们再作业 a1-basics 中的各种实现方法
   - 优点：简单，易读
   - 缺点：GPU 利用率最低、计算速度最慢的方法
2. **Pytorch function**：
   - Pytorch 默认提供的方法
   - 优点：调用简单，易读，部分实现性能很好或是较好（基本全都优于 manual 方法，部分优于 CUDA、Triton、Torch compile）
   - 缺点：有部分实现性能较为一般，不如 CUDA、Triton、Torch compile
3. **CUDA**：
   - 直接写 CUDA 代码
   - 优点：性能普遍占优（但不是绝对，理论上这是绝对占优的方法，但是优多少取决于你的代码实力）
   - 缺点：难写、且不如前两种方法易懂，小部分方法不一定优于 CUDA、Triton、Torch compile、Torch function
4. **Triton**：
   - 使用 Triton 库进行编写，Triton 会自动将代码编译为底层代码
   - 优点：简单易上手且性能基本占优，小部分方法不一定优于 CUDA、Triton、Torch compile、Torch function
   - 缺点：不适合太复杂的 kernel、
5. **Pyorch compile**：
   - 使用 Torch 中的 Compile 对 manual 方法进行编译，提升其性能（本质上来讲就是针对 manual 代码生成 Triton 代码）
   - 优点：优点：简单易上手且性能基本占优，kernel fusion，
   - 缺点：没有明显缺点，小部分方法不一定优于 CUDA、Triton、Torch compile、Torch function

性能优劣主要取决于：

1. 算子是否 fused（最关键）
2. 是否走 cuBLAS/cuDNN 等高度优化的库
3. 内存访问是否 coalesced
4. 张量形状是否匹配 tensor core
5. op 是否频繁调用（kernel launch overhead）
6. 数据读写 vs 计算是否平衡（memory bound / compute bound）

这些原则决定了：

* PyTorch function 为何通常最优
* manual 为何最慢
* Triton/CUDA 为何有时能超过 PyTorch（因为做了 fusion）