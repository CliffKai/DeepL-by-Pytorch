from __future__ import annotations

import math
import torch
import torch.nn as nn

from typing import Iterable, Optional
from torch.optim import Optimizer

class AdamWCustom(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def forward(self,closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("AdamWCustom dose not support sparse gradients")
                
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                state["step"] += 1
                step: int = state["step"]

                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                step_size = lr / bias_correction1

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
    
def get_adamw_cls():
    return AdamWCustom

def get_lr_cosine_schedule(
    *,
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosint_cycle_iters: int,
) -> float:
    if it <= warmup_iters:
        if warmup_iters <= 0:
            return float(max_learning_rate)
        return float(max_learning_rate) * (it / float(warmup_iters))
    
    span = max(1, cosint_cycle_iters - warmup_iters)
    progress = (it - warmup_iters) / float(span)
    if progress >= 1.0:
        return float(min_learning_rate)
    
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = min_learning_rate + (max_learning_rate - min_learning_rate) * cosine

    if lr < min_learning_rate:
        lr = min_learning_rate
    if lr > max_learning_rate:
        lr =max_learning_rate
    return float(lr)
    



        