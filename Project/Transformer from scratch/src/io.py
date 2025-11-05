from __future__ import annotations

import os
import torch

from typing import BinaryIO, IO

def _is_pathlike(x) -> bool:
    return isinstance(x, (str, bytes, os.PathLike))

def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    if _is_pathlike(out):
        with open(out, "wb") as f:
            torch.save(payload, f)
    else:
        torch.save(payload, f)

def load_checkpoint(
    *,
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    if _is_pathlike(src):
        with open(src, "rb") as f:
            payload = torch.load(f, map_location="cpu")
    else:
        payload = torch.load(src, map_location="cpu")
    
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload["iteration"])