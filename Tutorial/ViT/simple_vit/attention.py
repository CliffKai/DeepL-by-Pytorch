import torch
from torch import nn
from einops import rearrange

# 这是一个继承自 torch.nn.Module 的 PyTorch 模块类。
# 实现一个简单的多头自注意力机制。
# dim：输入特征的维度。
# heads：注意力头的数量（默认是8）。
# dim_head：每个注意力头的维度（默认是64）。
# 注意力机制的核心思想是通过计算查询（query）、键（key）和值（value）之间的关系来捕捉输入序列中不同位置之间的依赖关系。
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)          #

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# 这是一个继承自 torch.nn.Module 的 PyTorch 模块类。
# 实现一个简单的前馈神经网络，包含两个线性层和一个 GELU 激活函数。
# dim：输入和输出的特征维度，通常对应于 token 的嵌入维度（embedding dim）。
# hidden_dim：隐藏层的维度。
class FeedForward(nn.Module):
    # 搭建前馈神经网络的结构
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    # 前向传播
    def forward(self, x):
        return self.net(x)
