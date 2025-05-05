import torch
from simple_vit import SimpleViT  # 来自 simple_vit 包

model = SimpleViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 1024,
    channels = 3
)

img = torch.randn(1, 3, 224, 224)
logits = model(img)
print(logits.shape)  # torch.Size([1, 10])