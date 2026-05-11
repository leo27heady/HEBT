"""
Multi-stage convolutional encoder for HVQVAE.

Produces features at three spatial resolutions from 64×64 input:
    s1:  (B, 64,  8, 8)   — finest
    s2:  (B, 128, 4, 4)
    s3:  (B, 256, 2, 2)   — coarsest

Architecture follows the reference VQ-VAE encoder style (strided convs +
residual stacks) but extended to produce multi-scale outputs via progressive
downsampling. Each output has a 1×1 pre-quantization conv to project to the
stage's embedding dimension.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from model.vid.hvqvae.residual import ResidualStack


class MultiStageEncoder(nn.Module):
    """Reference-style encoder producing features at 3 resolutions.

    For 64×64 input:
        stem:    Conv(3→64, k=4, s=2)  → 32×32
        down1:   Conv(64→128, k=4, s=2) → 16×16
        block1:  Conv(128→128, k=3, s=1) + ResStack → 16×16
        down2:   Conv(128→128, k=4, s=2) + ResStack → 8×8
                 pre_quant_s1: Conv(128→64, k=1) → s1: (64, 8, 8)
        down3:   Conv(128→256, k=4, s=2) + ResStack → 4×4
                 pre_quant_s2: Conv(256→128, k=1) → s2: (128, 4, 4)
        down4:   Conv(256→256, k=4, s=2) + ResStack → 2×2
                 pre_quant_s3: Conv(256→256, k=1) → s3: (256, 2, 2)

    Parameters
    ----------
    h_dim : base hidden dimension (default 128, matching reference).
    res_h_dim : residual block hidden dimension.
    n_res_layers : number of residual layers per stack.
    stage_channels : dict mapping stage name to output channels.
    """

    def __init__(
        self,
        h_dim: int = 128,
        res_h_dim: int = 32,
        n_res_layers: int = 2,
        stage_channels: Dict[str, int] = None,
    ):
        super().__init__()
        if stage_channels is None:
            stage_channels = {"s1": 64, "s2": 128, "s3": 256}

        # Stem: 64×64 → 32×32
        self.stem = nn.Sequential(
            nn.Conv2d(3, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Down1: 32×32 → 16×16
        self.down1 = nn.Sequential(
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Block at 16×16 (reference-style: conv + residual stack)
        self.block_16 = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )

        # Down2: 16×16 → 8×8 (s1 tap point)
        self.down2 = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )
        self.pre_quant_s1 = nn.Conv2d(h_dim, stage_channels["s1"], kernel_size=1)

        # Down3: 8×8 → 4×4 (s2 tap point)
        self.down3 = nn.Sequential(
            nn.Conv2d(h_dim, h_dim * 2, kernel_size=4, stride=2, padding=1),
            ResidualStack(h_dim * 2, h_dim * 2, res_h_dim, n_res_layers),
        )
        self.pre_quant_s2 = nn.Conv2d(h_dim * 2, stage_channels["s2"], kernel_size=1)

        # Down4: 4×4 → 2×2 (s3 tap point)
        self.down4 = nn.Sequential(
            nn.Conv2d(h_dim * 2, h_dim * 2, kernel_size=4, stride=2, padding=1),
            ResidualStack(h_dim * 2, h_dim * 2, res_h_dim, n_res_layers),
        )
        self.pre_quant_s3 = nn.Conv2d(h_dim * 2, stage_channels["s3"], kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode a batch of images.

        Args:
            x: (B, 3, H, W) images in [0, 1] or normalized.

        Returns:
            dict with keys "s1", "s2", "s3" mapping to feature tensors.
        """
        x = self.stem(x)       # (B, 64, 32, 32)
        x = self.down1(x)      # (B, 128, 16, 16)
        x = self.block_16(x)   # (B, 128, 16, 16)

        x_8 = self.down2(x)    # (B, 128, 8, 8)
        s1 = self.pre_quant_s1(x_8)   # (B, 64, 8, 8)

        x_4 = self.down3(x_8)  # (B, 256, 4, 4)
        s2 = self.pre_quant_s2(x_4)   # (B, 128, 4, 4)

        x_2 = self.down4(x_4)  # (B, 256, 2, 2)
        s3 = self.pre_quant_s3(x_2)   # (B, 256, 2, 2)

        return {"s1": s1, "s2": s2, "s3": s3}
