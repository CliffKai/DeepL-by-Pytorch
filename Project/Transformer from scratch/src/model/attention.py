import torch
import math
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Bool, Int
from einops import einsum, rearrange

from ..nn_utils import softmax
from .Embedding import RotaryPositionalEmbedding
from .Linear import Linear

def scaled_dot_product_attention(
    q: Float[Tensor, "... quaries d_k"],
    k: Float[Tensor, "... key d_k"],
    v: Float[Tensor, "... value d_k"],
    mask: Bool[Tensor, "... quaries value"] | None=None,
) -> Float[Tensor, "... quaries d_k"]:
    d_k = q.size(-1)
    scores = einsum(q, k, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    output = einsum(attn_weights, v, "... q k, ... k d -> ... q d")
    return output

# attention test code
'''
import torch
from torch import Tensor
from einops import einsum

from src.model.attention import scaled_dot_product_attention

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
output = scaled_dot_product_attention(q, k, v, mask)
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
    out_zero = scaled_dot_product_attention(q_zero, k_zero, v_zero, mask_all)
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
    out_half = scaled_dot_product_attention(q_half, k_half, v_half, mask_half)
    assert out_half.shape == (batch_size, seq_len_q, d_k)
    print("✅ CUDA float16 形状检查通过。")
'''

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding,
        device: torch.device | str | None=None,
        dtype: torch.dtype | None=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope
        factory_kwargs = {"device": device, "dtype": dtype}

        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **factory_kwargs)

        self.register_buffer("causal_mask", None, persistent=False)
    
    def get_causal_mask(
        self,
        seq_len: int,
        device: torch.device | str | None=None,
    ) -> Bool[Tensor, "seq_len seq_len"]:
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            self.causal_mask = ~mask
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        token_positions: Int[Tensor, "batch seq_len"],
    ) -> Float[Tensor, "batch seq_len d_model"]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "b s (h d) -> b h s d", h = self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h = self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h = self.num_heads)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        causal_mask = self.get_causal_mask(seq_len=x.size(1), device=x.device)
        attn_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")
        return self.output_proj(attn_output)
    

# MHA test code
'''
import math
import torch
import torch.nn as nn
from einops import rearrange

from src.model.attention import MultiHeadSelfAttention, scaled_dot_product_attention
from src.model.Embedding import RotaryPositionalEmbedding, Embedding

torch.manual_seed(0)

def build_inputs(B=2, S=6, d_model=16, num_heads=4, device="cpu"):
    x = torch.randn(B, S, d_model, device=device, requires_grad=True)
    # 第二个 batch 故意平移位置，测试 RoPE 的广播
    pos = torch.arange(S, device=device).unsqueeze(0).repeat(B, 1)
    pos[1] += 2
    return x, pos

def test_mha_forward_and_shapes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, S, d_model, H = 2, 6, 16, 4
    d_head = d_model // H

    # --- 构造 RoPE（d_k = d_head） ---
    rope = RotaryPositionalEmbedding(d_k=d_head, max_seq_len=1024, device=device)

    # --- 构造 MHA ---
    mha = MultiHeadSelfAttention(
        d_model=d_model, num_heads=H, rope=rope, device=device
    ).to(device)

    assert mha.d_head == d_head

    # --- 输入 ---
    x, token_positions = build_inputs(B, S, d_model, H, device)

    # --- 前向 ---
    y = mha(x, token_positions)
    print("Output shape:", y.shape)

    # 形状检查
    assert y.shape == (B, S, d_model), "❌ 输出形状不匹配！"

    # 反向传播（检查参数与输入均有梯度）
    loss = y.sum()
    loss.backward()

    for name, p in mha.named_parameters():
        assert p.grad is not None, f"❌ 参数 {name} 没有梯度"
    assert x.grad is not None, "❌ 输入 x 没有梯度"

    print("✅ 形状与反向传播检查通过。")

def test_causal_mask_effect():
    """
    掩码有效性：验证注意力在位置 t 不会关注到未来位置 (>t)。
    做法：
      1) 手动用与 MHA 一致的 Q/K/V 计算 scores；
      2) 分别用 “无掩码” 与 “因果掩码” 得到注意力 aw；
      3) 断言在 masked 情况下 aw[..., t, t+1:] ≈ 0。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, S, d_model, H = 2, 6, 16, 4
    d_head = d_model // H

    rope = RotaryPositionalEmbedding(d_k=d_head, max_seq_len=1024, device=device)
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=H, rope=rope, device=device).to(device)

    x, token_positions = build_inputs(B, S, d_model, H, device)
    x = x.detach().requires_grad_(True)

    # --- 与 forward 中一致的投影与重排 ---
    q = mha.q_proj(x)  # (B,S,d_model)
    k = mha.k_proj(x)
    v = mha.v_proj(x)

    q = rearrange(q, "b s (h d) -> b h s d", h=H)
    k = rearrange(k, "b s (h d) -> b h s d", h=H)
    v = rearrange(v, "b s (h d) -> b h s d", h=H)

    # RoPE 只作用 Q/K
    q = mha.rope(q, token_positions)
    k = mha.rope(k, token_positions)

    # --- 计算 unmasked 注意力权重 ---
    # 复制你实现里的逻辑：scores = (q @ k^T) / sqrt(d_k)；softmax(-1)
    scores_un = torch.einsum("... q d, ... k d -> ... q k", q, k) / math.sqrt(d_head)
    aw_un = torch.softmax(scores_un, dim=-1)  # (B,H,S,S)

    # --- 构造因果掩码（下三角 True=可见），并应用 ---
    causal_mask = mha.get_causal_mask(seq_len=S, device=device)   # (S,S) bool
    # 你的 scaled_dot_product_attention 定义里：mask==False 的位置会被设为 -inf
    scores_ma = scores_un.masked_fill(causal_mask == False, float("-inf"))
    aw_ma = torch.softmax(scores_ma, dim=-1)

    # --- 断言未来注意力为 ~0 ---
    # 随机抽几行检查：对每个 t，aw_ma[..., t, t+1:] 应接近 0
    with torch.no_grad():
        upper = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)  # 上三角 True=未来
        leaked = aw_ma.masked_select(upper.expand(B, H, -1, -1)).abs().max().item()
        print("Max attention on future positions (should be ~0):", leaked)
        assert leaked < 1e-6, "❌ 因果掩码失效：仍然关注了未来位置"

    print("✅ 因果掩码有效性通过。")

def test_cuda_half_optional():
    """
    可选：在 CUDA 上检查 float16。若无 CUDA 则跳过。
    """
    if not torch.cuda.is_available():
        print("(跳过 CUDA/float16 测试：CUDA 不可用)")
        return

    device = torch.device("cuda")
    B, S, d_model, H = 2, 8, 32, 4
    d_head = d_model // H

    rope = RotaryPositionalEmbedding(d_k=d_head, max_seq_len=1024, device=device).to(device)
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=H, rope=rope, device=device).to(device)

    x = torch.randn(B, S, d_model, device=device, dtype=torch.float16, requires_grad=True)
    pos = torch.arange(S, device=device).unsqueeze(0).repeat(B, 1)

    y = mha(x, pos)
    assert y.shape == (B, S, d_model)
    # 允许半精度有更松的 allclose 标准，这里只检查是否能跑通与反向
    y.sum().backward()
    assert x.grad is not None
    print("✅ CUDA float16 前向/反向通过。")

if __name__ == "__main__":
    test_mha_forward_and_shapes()
    test_causal_mask_effect()
    test_cuda_half_optional()
    print("\n🎉 All MultiHeadSelfAttention tests finished.")
'''