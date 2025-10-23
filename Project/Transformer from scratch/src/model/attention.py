import torch
import math
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Bool
from einops import einsum

from ..nn_utils import softmax

def attention(
    q: Float[Tensor, "... quaries d_k"],
    k: Float[Tensor, "... key d_k"],
    v: Float[Tensor, "... value d_k"],
    mask: Bool[Tensor, "... quaries value"] | None=None,
) -> Float[Tensor, "... quaries d_k"]:
    d_k = q.size(-1)
    scores = einsum(q, k, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    aw = softmax(scores, dim=-1)
    output = einsum(aw, v, "... q k, ... k d -> ... q d")
    return output

