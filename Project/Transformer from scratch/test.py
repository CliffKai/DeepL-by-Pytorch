import torch
from src.model.FFN import SwiGLU   # 假设文件路径为 model/SwiGLU.py
from src.nn_utils import silu   # 若 nn_utils.py 在同级目录下

# 1️⃣ 固定随机种子，确保结果可复现
torch.manual_seed(0)

# 2️⃣ 超参数定义
batch_size = 2
seq_len = 4
d_model = 8
d_ff = None   # 测试自动推导分支

# 3️⃣ 构造层与输入（可选测试 CUDA + float16）
layer = SwiGLU(d_model=d_model, d_ff=d_ff)

x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
print("Input x shape:", x.shape)

# 🧪 GPU 测试（若可用）
if torch.cuda.is_available():
    device = torch.device("cuda")
    layer = layer.to(device)
    x = x.to(device).half().detach().requires_grad_(True)
    print("Running on CUDA with float16")

# 4️⃣ 前向计算
y = layer(x)
print("Output y shape:", y.shape)

# 5️⃣ 手工验证逻辑：
# gate = w1(x)
# hidden = silu(w2(x))
# y_ref = w3(gate * hidden)
with torch.no_grad():
    x32 = x.float()
    gate = layer.w1(x32)
    hidden = silu(layer.w2(x32))
    y_ref32 = layer.w3(gate * hidden)
    y_ref = y_ref32.to(y.dtype)

print("All close to manual reference?",
      torch.allclose(y, y_ref, atol=1e-5 if y.dtype.is_floating_point else 1e-3))

# 6️⃣ 形状断言
assert y.shape == (batch_size, seq_len, d_model), "❌ 输出形状不匹配！"
print("✅ 形状检查通过。")

# 7️⃣ 反向传播检查
loss = y.sum()
loss.backward()

assert x.grad is not None, "❌ 输入没有梯度"
assert layer.w1.weight.grad is not None, "❌ w1 权重没有梯度"
assert layer.w2.weight.grad is not None, "❌ w2 权重没有梯度"
assert layer.w3.weight.grad is not None, "❌ w3 权重没有梯度"
print("✅ 反向传播检查通过。")

# 8️⃣ 数值稳定性：零输入应输出零（因为 gate*hidden=0）
with torch.no_grad():
    x_zero = torch.zeros_like(x)
    y_zero = layer(x_zero)
    assert torch.count_nonzero(y_zero) == 0, "❌ 零输入时输出应全 0"
print("✅ 零输入稳定性通过。")

# 9️⃣ 自动推导的 d_ff 检查
expected_d_ff = int((8/3) * d_model)
expected_d_ff = (expected_d_ff + 63) // 64 * 64
print(f"Expected d_ff={expected_d_ff}, actual w1.out_features={layer.w1.out_features}")
assert layer.w1.out_features == expected_d_ff, "❌ d_ff 自动推导不正确"
print("✅ 自动推导检查通过。")