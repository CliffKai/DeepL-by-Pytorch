import torch
from simple_vit import SimpleViT  # 来自 simple_vit 包

h = 4
w = 3
torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")