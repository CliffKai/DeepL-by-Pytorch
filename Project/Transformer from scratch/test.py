import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Int
from src.model.Embedding import RotaryPositionalEmbedding

# ======= 工具函数：用“纯实数版”手工旋转，作为参考实现 =======
def manual_rope_rotate_real(
    x: Tensor,                       # (..., S, D)
    token_positions: Tensor,         # (..., S) int
    theta: float = 10000.0,
) -> Tensor:
    """
    不依赖复数运算的参考实现：将最后一维 D 拆成两两一组，
    用 cos/sin 做二维旋转，再拼回去，作为对照。
    """
    assert x.size(-1) % 2 == 0, "D must be even"
    *prefix, S, D = x.shape
    d2 = D // 2

    # 频率构造：freqs[i] = 1 / theta**( (2*i)/D ) 与实现等价
    # 这里用 arange(0, D, 2)/D 的写法
    idx = torch.arange(0, D, 2, device=x.device, dtype=torch.float32)  # [d2]
    freqs = 1.0 / (theta ** (idx / D))                                 # [d2]

    # 计算每个 token 的角度：theta_mat[..., s, i] = pos[..., s] * freqs[i]
    # 先把 token_positions 转成 float32 便于广播
    pos = token_positions.to(dtype=torch.float32)
    # 目标 shape: (..., S, d2)
    theta_mat = pos.unsqueeze(-1) * freqs.view(*([1] * len(prefix)), 1, d2)

    cos = torch.cos(theta_mat)  # (..., S, d2)
    sin = torch.sin(theta_mat)  # (..., S, d2)

    # 将 x 拆成两两一组
    x32 = x.to(torch.float32)
    x_pair = x32.view(*prefix, S, d2, 2)
    x_even = x_pair[..., 0]  # (..., S, d2)
    x_odd  = x_pair[..., 1]  # (..., S, d2)

    # 旋转：(a,b) -> (a*cos - b*sin, a*sin + b*cos)
    out_even = x_even * cos - x_odd * sin
    out_odd  = x_even * sin + x_odd * cos

    out = torch.stack([out_even, out_odd], dim=-1).reshape(*prefix, S, D)
    return out.to(dtype=x.dtype)


# ======= 1) 基本参数 =======
torch.manual_seed(0)
batch_size = 2
heads = 3
seq_len = 5
d_k = 8
theta = 10000.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= 2) 构造 RoPE 层 =======
rope = RotaryPositionalEmbedding(d_k=d_k, max_seq_len=1024, theta=theta, device=device).to(device)

# ======= 3) 3D 输入测试：形状 (B, S, D) =======
print("\n=== 3D input test (B, S, D) ===")
x3 = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
# 每个 batch 的位置可以不同，构造一个有差异的 token_positions
pos3 = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
# 让第二个 batch 故意平移 1
pos3[1] += 1

y3 = rope(x3, pos3)
print("x3 shape:", x3.shape)
print("y3 shape:", y3.shape)
assert y3.shape == x3.shape, "❌ 3D 输出形状不匹配"

# 与“手工旋转”对齐
with torch.no_grad():
    y3_ref = manual_rope_rotate_real(x3, pos3, theta=theta)
print("All close to manual (3D)?", torch.allclose(y3, y3_ref, atol=1e-6))

# 反向传播检查
loss3 = y3.sum()
loss3.backward()
assert x3.grad is not None, "❌ 3D 输入没有梯度"
print("✅ 3D backward ok")


# ======= 4) 4D 输入测试：形状 (B, H, S, D) =======
print("\n=== 4D input test (B, H, S, D) ===")
x4 = torch.randn(batch_size, heads, seq_len, d_k, device=device, requires_grad=True)
pos4 = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
# 让第二个 batch 平移 2
pos4[1] += 2

y4 = rope(x4, pos4)  # 注意：实现会在 head 维自动广播
print("x4 shape:", x4.shape)
print("pos4 shape:", pos4.shape)
print("y4 shape:", y4.shape)
assert y4.shape == x4.shape, "❌ 4D 输出形状不匹配"

# 与“手工旋转”对齐（手工实现同样不区分 head 维，天然广播）
with torch.no_grad():
    y4_ref = manual_rope_rotate_real(x4, pos4.unsqueeze(1).expand(-1, heads, -1), theta=theta)
print("All close to manual (4D)?", torch.allclose(y4, y4_ref, atol=1e-6))

# 反向传播检查
loss4 = y4.sum()
loss4.backward()
assert x4.grad is not None, "❌ 4D 输入没有梯度"
print("✅ 4D backward ok")


# ======= 5) 零输入稳定性（应输出全 0） =======
print("\n=== Zero input stability ===")
with torch.no_grad():
    x_zero3 = torch.zeros_like(x3)
    y_zero3 = rope(x_zero3, pos3)
    assert torch.count_nonzero(y_zero3) == 0, "❌ 3D: 零输入输出应全 0"

    x_zero4 = torch.zeros_like(x4)
    y_zero4 = rope(x_zero4, pos4)
    assert torch.count_nonzero(y_zero4) == 0, "❌ 4D: 零输入输出应全 0"
print("✅ Zero input stability ok")


# ======= 6) CUDA 半精度（可选） =======
if torch.cuda.is_available():
    print("\n=== CUDA float16 test ===")
    x4_half = x4.detach().to(device).half().requires_grad_(True)
    pos4_half = pos4  # 位置仍是 int
    y4_half = rope(x4_half, pos4_half)
    print("Half output shape:", y4_half.shape)
    assert y4_half.dtype == torch.float16, "❌ 半精度输出 dtype 不对"
    assert y4_half.shape == x4_half.shape, "❌ 半精度输出形状不匹配"

    # 与手工参考（手工里内部会转到 float32 做三角函数，然后回到半精度）
    with torch.no_grad():
        y4_half_ref = manual_rope_rotate_real(x4_half, pos4_half.unsqueeze(1).expand(-1, heads, -1), theta=theta)
    print("All close (half)?", torch.allclose(y4_half, y4_half_ref, atol=3e-3))  # 半精度放松阈值
    y4_half.sum().backward()
    assert x4_half.grad is not None, "❌ 半精度输入没有梯度"
    print("✅ CUDA float16 ok")
else:
    print("\n(跳过 CUDA float16 测试：CUDA 不可用)")


# ======= 7) 异常用例：奇数 d_k 应报错 =======
print("\n=== Error case: odd d_k should raise ===")
try:
    _ = RotaryPositionalEmbedding(d_k=7, max_seq_len=16, theta=theta, device=device)
    print("❌ 预期应报错，但未报错")
except ValueError as e:
    print("✅ 捕获到预期错误:", str(e))


# ======= 8) 异常用例：位置越界应触发索引错误 =======
print("\n=== Error case: position out of range ===")
try:
    rope_short = RotaryPositionalEmbedding(d_k=d_k, max_seq_len=8, theta=theta, device=device)
    x_tmp = torch.randn(1, 4, d_k, device=device)
    pos_tmp = torch.tensor([[0, 1, 7, 8]], device=device)  # 8 越界（max_seq_len=8 合法索引 0..7）
    _ = rope_short(x_tmp, pos_tmp)
    print("❌ 预期应报错，但未报错")
except (IndexError, RuntimeError) as e:
    print("✅ 捕获到预期错误:", str(e))


print("\n🎉 All RoPE tests finished.")