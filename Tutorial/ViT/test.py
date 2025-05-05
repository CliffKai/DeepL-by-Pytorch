import torch

h = 4
w = 3
x, y = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

print(x)
print(y)