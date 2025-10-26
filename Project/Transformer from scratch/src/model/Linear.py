from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

import math

from jaxtyping import Bool, Float, Int
from einops import einsum

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device":device, "dtype":dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        w = self.weight.to(x.dtype)
        return einsum(x, w, "... d_in, d_out d_in -> ... d_out")
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"
    
# test code
'''
import torch
from einops import einsum

from src.model.Linear import Linear 

# 1️⃣ 创建一个简单的线性层实例
in_features = 4
out_features = 3
layer = Linear(in_features, out_features)

# 2️⃣ 构造一个输入张量（batch_size=2）
x = torch.randn(2, in_features)
print("Input x shape:", x.shape)

# 3️⃣ 前向计算
y = layer(x)
print("Output y shape:", y.shape)
print("Output y:\n", y)

# 4️⃣ 验证 shape 是否正确
assert y.shape == (2, out_features), "❌ 输出形状不正确"

# 5️⃣ 对比 PyTorch 官方 nn.Linear（只为验证逻辑）
ref = torch.nn.Linear(in_features, out_features, bias=False)

with torch.no_grad():
    ref.weight.copy_(layer.weight)

y_ref = ref(x)
print("\nReference output from nn.Linear:\n", y_ref)

# 6️⃣ 比较两者是否一致
print("\nAll close?", torch.allclose(y, y_ref, atol=1e-6))
'''