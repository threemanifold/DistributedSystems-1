"""Shared model and synthetic data.

All training-related tasks (5-11) start from this model so that step times,
memory numbers and agreement checks refer to the same thing.

The model is a stack of identical Linear+GELU blocks. Width and depth control
the parameter count for size sweeps, and each block is one "layer" for tasks
that hook per-layer gradients (task 6).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    in_dim: int = 1024
    hidden: int = 1024
    depth: int = 8
    out_dim: int = 1024


class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.linear(x))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.inp = nn.Linear(cfg.in_dim, cfg.hidden)
        self.blocks = nn.Sequential(*[Block(cfg.hidden) for _ in range(cfg.depth)])
        self.out = nn.Linear(cfg.hidden, cfg.out_dim)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def synthetic_batch(
    step: int,
    global_batch_size: int,
    cfg: ModelConfig,
    device,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    dtype=torch.float32,
):
    """Deterministic synthetic (x, y) for a given step.

    The *global* batch depends only on (seed, step). Each rank receives its
    contiguous slice, so a single-GPU run with global_batch_size=B and a
    world_size=W run with the same B see exactly the same data. That is what
    makes the "equivalent global batch" agreement checks in tasks 5, 8, 10
    and 11 well defined.
    """
    assert global_batch_size % world_size == 0
    g = torch.Generator(device=device)
    g.manual_seed(seed * 1_000_003 + step)
    x = torch.randn(global_batch_size, cfg.in_dim, device=device, dtype=dtype, generator=g)
    y = torch.randn(global_batch_size, cfg.out_dim, device=device, dtype=dtype, generator=g)
    local = global_batch_size // world_size
    sl = slice(rank * local, (rank + 1) * local)
    return x[sl], y[sl]
