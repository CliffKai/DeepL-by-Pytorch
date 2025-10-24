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
