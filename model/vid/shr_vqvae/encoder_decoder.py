"""Convolutional Encoder and Decoder for S-HR-VQVAE.

Encoder: (B, 3, H, W) → (B, embedding_dim, H/4, W/4)
Decoder: (B, embedding_dim, H/4, W/4) → (B, 3, H, W)

Both use a lightweight ResBlock architecture.
The decoder applies Sigmoid so its output matches [0, 1] ToTensor inputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import SHRVQVAEConfig


class ResBlock(nn.Module):
    """Pre-activation residual block."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        groups = min(8, ch)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConvEncoder(nn.Module):
    """Encode a single frame to a latent grid (4× spatial downsampling).

    Input  : (B, 3, H, W)
    Output : (B, embedding_dim, H/4, W/4)
    """

    def __init__(self, cfg: SHRVQVAEConfig) -> None:
        super().__init__()
        C = cfg.base_channels
        E = cfg.embedding_dim
        self.net = nn.Sequential(
            # H → H/2
            nn.Conv2d(cfg.image_c, C, 3, stride=2, padding=1, bias=False),
            nn.SiLU(),
            ResBlock(C),
            # H/2 → H/4
            nn.Conv2d(C, C * 2, 3, stride=2, padding=1, bias=False),
            nn.SiLU(),
            ResBlock(C * 2),
            # project to embedding dim
            nn.Conv2d(C * 2, E, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """Decode a latent grid back to image space (4× upsampling).

    Input  : (B, embedding_dim, H/4, W/4)
    Output : (B, 3, H, W)  — values in [0, 1] via Sigmoid
    """

    def __init__(self, cfg: SHRVQVAEConfig) -> None:
        super().__init__()
        C = cfg.base_channels
        E = cfg.embedding_dim
        self.net = nn.Sequential(
            nn.Conv2d(E, C * 2, 1, bias=False),
            ResBlock(C * 2),
            # H/4 → H/2
            nn.ConvTranspose2d(C * 2, C, 4, stride=2, padding=1, bias=False),
            nn.SiLU(),
            ResBlock(C),
            # H/2 → H
            nn.ConvTranspose2d(C, C // 2, 4, stride=2, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(C // 2, cfg.image_c, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
