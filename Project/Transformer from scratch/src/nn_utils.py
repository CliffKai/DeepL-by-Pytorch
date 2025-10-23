import torch

from torch import Tensor
from jaxtyping import Float

def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exps = torch.exps(shifted)
    return exps / exps.sum(dim=dim, keepdim=True)

def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return torch.sigmoid(x)