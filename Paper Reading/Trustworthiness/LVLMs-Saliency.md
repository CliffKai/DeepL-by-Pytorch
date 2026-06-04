HALLUCINATION BEGINS WHERE SALIENCY DROPS

# 0 Abstract

现有的基于 attention map 的 attention sinks 方法仅仅依赖前向传播的信号，忽略了基于梯度的信号，而这些梯度信号却能够揭示 token 影响力是如何在模型中传播的，本篇论文提出了**LVLMs-Saliency**。

# 1 Introduction

当前缓解 LVLMs 幻觉的方法:
-  incorporating external knowledge
-  retraining with additional data
-  training-free methods(see the summary in [training-free methods.md](training-free%20methods.md))

但这些方法都缺乏可解释性。
近期有一些关于 attention sinks 的方法(see the summary in [forward-pass attention.md](forward-pass%20attention.md#L651))尝试对幻觉问题进行解释，一般性质的结论是：当某个 token 在后续 token 中持续吸引较高的注意力权重时，这种过度依赖可能导致模型输出中的幻觉。

