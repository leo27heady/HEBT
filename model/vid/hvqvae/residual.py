"""
Residual blocks for HVQVAE encoder/decoder.

Adapted from vqvae-reference/models/residual.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualLayer(nn.Module):
    """Single residual block: ReLU → Conv3×3 → ReLU → Conv1×1 + skip."""

    def __init__(self, in_dim: int, h_dim: int, res_h_dim: int):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_block(x)


class ResidualStack(nn.Module):
    """Stack of residual layers followed by a final ReLU."""

    def __init__(self, in_dim: int, h_dim: int, res_h_dim: int, n_res_layers: int):
        super().__init__()
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim) for _ in range(n_res_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)
