"""Decoder and Upscaler modules for Fresh HVQVAE."""

import torch
import torch.nn as nn

from .encoder import ResBlock
from .config import FreshHVQVAEConfig


class Decoder(nn.Module):
    """
    Shared decoder: (B, C_bot, 16, 16) → (B, 3, 64, 64).
    Receives input from bot direct, or mid/top upscaled to 16x16.
    """

    def __init__(self, C_bot: int = 128):
        super().__init__()
        self.decode = nn.Sequential(
            ResBlock(C_bot, C_bot),
            nn.ConvTranspose2d(C_bot, 64, 4, stride=2, padding=1),  # 16→32
            nn.SiLU(),
            ResBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     # 32→64
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x)


class UpscaleMid(nn.Module):
    """Upscale mid features: (B, C_mid, 4, 4) → (B, C_bot, 16, 16)."""

    def __init__(self, C_mid: int = 192, C_bot: int = 128):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(C_mid, C_bot, 4, stride=2, padding=1),  # 4→8
            nn.SiLU(),
            nn.ConvTranspose2d(C_bot, C_bot, 4, stride=2, padding=1),  # 8→16
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UpscaleTop(nn.Module):
    """Upscale top features: (B, C_top, 1, 1) → (B, C_bot, 16, 16)."""

    def __init__(self, C_top: int = 256, C_bot: int = 128):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(C_top, C_bot, 4, stride=4, padding=0),  # 1→4
            nn.SiLU(),
            nn.ConvTranspose2d(C_bot, C_bot, 4, stride=4, padding=0),  # 4→16
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
