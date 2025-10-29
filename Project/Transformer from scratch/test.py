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