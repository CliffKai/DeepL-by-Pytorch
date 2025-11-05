from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

def get_batch(
    *,
    dataset: npt.NDArray,
    batch_size: int,
    context_length,
    device: torch.device | str | None=None,
    pin_memory: bool=False,
) -> tuple[torch.Tensor, torch.Tensor]:
    is_cuda = str(device).startswith("cuda")
    if pin_memory and not is_cuda:
        raise ValueError("pin_memory=True only makes sense when device is CUDA")
    
    if pin_memory:
        toks = torch.as_tensor(dataset, dtype=torch.long, device="cpu").pin_memory()
        idx_device = "cpu"
    else:
        toks = torch.as_tensor(dataset, dtype=torch.long, device=device)
        idx_device = device

    n = toks.numel()
    if n < context_length + 1:
        raise ValueError(f"need at least {context_length+1}, got {n}")
    
    max_start = n - context_length - 1
    starts = torch.randint(0, max_start + 1, (batch_size,), device=idx_device)
    ar = torch.arange(context_length, device=idx_device)
    idx = starts[:, None] + ar[None, :]
    X = toks[idx]
    Y = toks[idx + 1]

    if pin_memory:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

    return X, Y