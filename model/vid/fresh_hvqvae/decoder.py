"""Per-stage decoders for Fresh HVQVAE.

Each stage has its own decoder with complexity proportional to the upscaling needed:
  - DecoderBot: 16x16 → 64x64 (shallow, 2 upsample steps)
  - DecoderMid:  4x4  → 64x64 (medium, 4 upsample steps with ResBlocks)
  - DecoderTop:  1x1  → 64x64 (deep, project + 4 upsample steps with ResBlocks)
"""

import torch
import torch.nn as nn

from .encoder import ResBlock


class DecoderBot(nn.Module):
    """Decode bot features: (B, C_bot, 16, 16) → (B, 3, 64, 64)."""

    def __init__(self, C_bot: int = 64):
        super().__init__()
        self.decode = nn.Sequential(
            ResBlock(C_bot, C_bot),
            nn.ConvTranspose2d(C_bot, 64, 4, stride=2, padding=1),   # 16→32
            nn.SiLU(),
            ResBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),      # 32→64
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x)


class DecoderMid(nn.Module):
    """Decode mid features: (B, C_mid, 4, 4) → (B, 3, 64, 64).

    4 upsample steps with ResBlocks for medium complexity.
    """

    def __init__(self, C_mid: int = 128):
        super().__init__()
        self.decode = nn.Sequential(
            ResBlock(C_mid, 256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),    # 4→8
            nn.SiLU(),
            ResBlock(128, 128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),     # 8→16
            nn.SiLU(),
            ResBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),      # 16→32
            nn.SiLU(),
            ResBlock(32, 32),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),      # 32→64
            nn.SiLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x)


class DecoderTop(nn.Module):
    """Decode top features: (B, C_top, 1, 1) → (B, 3, 64, 64).

    Deep decoder: project 1x1 vector → 4x4 spatial, then 4 upsample steps
    with ResBlocks. Enough capacity to reconstruct from a single vector.
    """

    def __init__(self, C_top: int = 256):
        super().__init__()
        self.project = nn.Sequential(
            nn.Flatten(),                         # (B, C_top)
            nn.Linear(C_top, 512 * 4 * 4),
            nn.SiLU(),
        )
        self.decode = nn.Sequential(
            ResBlock(512, 512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),    # 4→8
            nn.SiLU(),
            ResBlock(256, 256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),    # 8→16
            nn.SiLU(),
            ResBlock(128, 128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),     # 16→32
            nn.SiLU(),
            ResBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),      # 32→64
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.project(x).reshape(B, 512, 4, 4)
        return self.decode(x)
