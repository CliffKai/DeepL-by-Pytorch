import torch

x = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.]])
print(x.max(dim=1, keepdim=True))
shifted = x - x.max(dim=-1, keepdim=True).values
exps = torch.exp(shifted)
print(exps)
print(exps / exps.sum(dim=-1, keepdim=True))