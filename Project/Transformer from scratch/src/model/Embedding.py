import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Int, Float

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None=None,
        dtype: torch.dtype | None=None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {"device":device, "dtype":dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.weight[token_ids]
    
# test code
'''
import torch
from model.Embedding import Embedding

# 1️⃣ 基本超参数
num_embeddings = 10    # 词表大小
embedding_dim = 4      # 每个 token 的向量维度
batch_size = 2
seq_len = 5

# 2️⃣ 构造自定义 embedding 层
embedding = Embedding(num_embeddings, embedding_dim)
print("Embedding weight shape:", embedding.weight.shape)  # [10, 4]

# 3️⃣ 构造一个 token 索引输入
token_ids = torch.randint(0, num_embeddings, (batch_size, seq_len))
print("Input token_ids:\n", token_ids)
print("Input shape:", token_ids.shape)  # [2, 5]

# 4️⃣ 前向传播（查表）
output = embedding(token_ids)
print("Output embeddings:\n", output)
print("Output shape:", output.shape)  # [2, 5, 4]

# 5️⃣ 验证结果是否合理
assert output.shape == (batch_size, seq_len, embedding_dim), "❌ 输出形状不匹配！"
print("✅ 测试通过：embedding 输出维度正确。")

# 6️⃣ 可选：检查梯度是否可反向传播
output.sum().backward()
print("Gradient shape:", embedding.weight.grad.shape)
'''