
import torch
import torch.nn as nn

from jaxtyping import Float
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None=None,
        dtype: torch.dtype | None=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device":device, "dtype":dtype}
        self.weight = nn.Parameter(torch.ones((d_model), **factory_kwargs))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = x * rrms * self.weight
        return output.to(in_dtype)

# test code
'''
import torch
from src.model.RMSNorm import RMSNorm  # 假设你的文件路径为 model/RMSNorm.py

# 1️⃣ 固定随机种子，便于复现
torch.manual_seed(0)

# 2️⃣ 基本超参数
batch_size = 2
seq_len = 3
d_model = 8
eps = 1e-5

# 3️⃣ 构造层与输入（可在 CUDA 上测试半精度）
layer = RMSNorm(d_model=d_model, eps=eps)

x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
print("Input x shape:", x.shape)

# 🧪 可选：在 GPU 上测试 float16（若可用）
if torch.cuda.is_available():
    device = torch.device("cuda")
    layer = layer.to(device)
    x = x.to(device).half().detach().requires_grad_(True)
    print("Running on CUDA with float16")

# 4️⃣ 前向计算
y = layer(x)
print("Output y shape:", y.shape)

# 5️⃣ 与“手工”RMS归一化对比（不依赖任何官方实现）
#    rrms = 1 / sqrt(mean(x^2) + eps)
#    y_ref = x * rrms * weight
with torch.no_grad():
    x32 = x.float()
    rrms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    y_ref32 = x32 * rrms * layer.weight.float()
    y_ref = y_ref32.to(y.dtype)

print("All close to manual reference?", torch.allclose(y, y_ref, atol=1e-6 if y.dtype.is_floating_point else 1e-3))

# 6️⃣ 形状断言
assert y.shape == (batch_size, seq_len, d_model), "❌ 输出形状不匹配！"
print("✅ 形状检查通过。")

# 7️⃣ 零输入数值稳定性（应输出全 0）
with torch.no_grad():
    x_zero = torch.zeros_like(x)
    y_zero = layer(x_zero)
    assert torch.count_nonzero(y_zero) == 0, "❌ 零输入时输出应全 0"
print("✅ 零输入稳定性通过。")

# 8️⃣ 反向传播检查（权重与输入均应有梯度）
loss = y.sum()
loss.backward()
assert layer.weight.grad is not None, "❌ 权重没有梯度"
assert x.grad is not None, "❌ 输入没有梯度"
print("✅ 反向传播检查通过。")
'''
