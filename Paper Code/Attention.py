import torch
import torch.nn.functional as F

Wq = torch.tensor([[1.0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1]])
Wk = torch.tensor([[0, 0, 1.0], [1, 1, 0], [0, 1, 0], [1, 1, 0]])
Wv = torch.tensor([[0, 2.0, 0], [0, 3, 0], [1, 0, 3], [1, 1, 0]])

A = torch.tensor([1.0, 0, 1, 0])
B = torch.tensor([0, 2, 0, 2])
C = torch.tensor([1, 1, 1, 1])
X = torch.stack([A, B, C], dim=0)  # shape: (3, 4)

Q = X @ Wq  # (3, 3)
K = X @ Wk  # (3, 3)
V = X @ Wv  # (3, 3)

d_k = Q.size(-1)
scores = Q @ K.T  # (3, 3)
scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

attn_weights = F.softmax(scores, dim=-1)  # (3, 3)

output = attn_weights @ V  # (3, 3)

print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)
print("Attention Scores (scaled):\n", scores)
print("Attention Weights:\n", attn_weights)
print("Final Output:\n", output)
