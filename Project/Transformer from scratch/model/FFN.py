import torch
import torch.nn as nn

from torch import Tensor

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None=None,
        device: torch.device | None=None,
        dtype: torch.dtype | None=None,
    ):
        super().__init__()
        factory_kwargs = {"device":device, "dtype": dtype}
        if d_ff is None:
            d_ff = int((8/3) * d_model)
            