import torch
import math
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Bool
from einops import einsum

from ..nn_utils import softmax

def attention(
    q: Float[Tensor, "... quaries d_k"],
    k: Float[Tensor, "... key d_k"],
    v: Float[Tensor, "... value d_k"],
    mask: Bool[Tensor, "... quaries value"] | None=None,
) -> Float[Tensor, "... quaries d_k"]:
    d_k = q.size(-1)
    scores = einsum(q, k, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    aw = softmax(scores, dim=-1)
    output = einsum(aw, v, "... q k, ... k d -> ... q d")
    return output














# attention test code
'''
import torch
from torch import Tensor
from einops import einsum

from src.model.attention import attention

# 1️⃣ 固定随机种子
torch.manual_seed(0)

# 2️⃣ 基本超参数
batch_size = 2
seq_len_q = 3
seq_len_kv = 4
d_k = 8

# 3️⃣ 构造 Q, K, V（可以带 batch 维度）
q = torch.randn(batch_size, seq_len_q, d_k, requires_grad=True)
k = torch.randn(batch_size, seq_len_kv, d_k, requires_grad=True)
v = torch.randn(batch_size, seq_len_kv, d_k, requires_grad=True)

# 4️⃣ 构造 mask：允许前两个 query 看到所有 key，最后一个 query 只看前 2 个 key
mask = torch.ones(batch_size, seq_len_q, seq_len_kv, dtype=torch.bool)
mask[:, -1, 2:] = 0  # 模拟下三角或局部注意力的情况

print("Q shape:", q.shape)
print("K shape:", k.shape)
print("V shape:", v.shape)
print("Mask shape:", mask.shape)

# 5️⃣ 前向传播
output = attention(q, k, v, mask)
print("Output shape:", output.shape)
print("Output sample:\n", output[0, 0, :5])

# 6️⃣ 手动实现对比（验证 softmax & einsum 逻辑）
with torch.no_grad():
    d_k_sqrt = d_k ** 0.5
    scores_ref = einsum(q, k, "... q d_k, ... k d_k -> ... q k") / d_k_sqrt
    scores_ref = scores_ref.masked_fill(mask == False, float("-inf"))
    aw_ref = torch.softmax(scores_ref, dim=-1)
    output_ref = einsum(aw_ref, v, "... q k, ... k d_k -> ... q d_k")

print("\nAll close to manual reference?", torch.allclose(output, output_ref, atol=1e-6))

# 7️⃣ 检查输出形状
assert output.shape == (batch_size, seq_len_q, d_k), "❌ 输出形状不匹配！"
print("✅ 形状检查通过。")

# 8️⃣ 零输入稳定性（应输出全 0）
with torch.no_grad():
    q_zero = torch.zeros_like(q)
    k_zero = torch.zeros_like(k)
    v_zero = torch.zeros_like(v)
    mask_all = torch.ones_like(mask, dtype=torch.bool)
    out_zero = attention(q_zero, k_zero, v_zero, mask_all)
    assert torch.count_nonzero(out_zero) == 0, "❌ 零输入时输出应为全 0"
print("✅ 零输入稳定性通过。")

# 9️⃣ 反向传播检查
loss = output.sum()
loss.backward()
assert q.grad is not None, "❌ Q 没有梯度"
assert k.grad is not None, "❌ K 没有梯度"
assert v.grad is not None, "❌ V 没有梯度"
print("✅ 反向传播检查通过。")

# 1️⃣0️⃣ CUDA 半精度测试（可选）
if torch.cuda.is_available():
    print("\n🚀 在 CUDA + float16 下测试：")
    device = torch.device("cuda")
    q_half = q.detach().to(device).half().requires_grad_(True)
    k_half = k.detach().to(device).half().requires_grad_(True)
    v_half = v.detach().to(device).half().requires_grad_(True)
    mask_half = mask.to(device)
    out_half = attention(q_half, k_half, v_half, mask_half)
    assert out_half.shape == (batch_size, seq_len_q, d_k)
    print("✅ CUDA float16 形状检查通过。")
'''