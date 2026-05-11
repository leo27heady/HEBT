"""
Pixel decoder for HVQVAE.

Takes the finest stage's (s1) quantized features and reconstructs to RGB.
Reference-style architecture: ConvTranspose stack mirroring the encoder.

For s1 input (64ch, 8×8) → output (3ch, 64×64):
    ConvTranspose(64→128, k=3, s=1)  + ResStack → (128, 8, 8)
    ConvTranspose(128→64,  k=4, s=2) + ReLU     → (64, 16, 16)
    ConvTranspose(64→32,   k=4, s=2) + ReLU     → (32, 32, 32)
    ConvTranspose(32→3,    k=4, s=2)             → (3, 64, 64)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.vid.hvqvae.residual import ResidualStack


class Decoder(nn.Module):
    """Reference-style VQ-VAE decoder: quantized features → RGB pixels.

    Parameters
    ----------
    in_dim : input channels (finest stage embedding dim, e.g. 64).
    h_dim : hidden dimension for transpose convs (e.g. 128).
    n_res_layers : number of residual layers.
    res_h_dim : residual block hidden dimension.
    """

    def __init__(
        self,
        in_dim: int = 64,
        h_dim: int = 128,
        n_res_layers: int = 2,
        res_h_dim: int = 32,
    ):
        super().__init__()
        # 8×8 → 8×8 (expand channels + refine with residuals)
        # 8×8 → 16×16 → 32×32 → 64×64
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h_dim // 2, h_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h_dim // 4, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode quantized features to RGB.

        Args:
            x: (B, C, H, W) quantized features.

        Returns:
            (B, 3, H_out, W_out) reconstructed RGB.
        """
        return self.net(x)
