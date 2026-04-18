"""
Optional pixel decoder for HVEBT.

Takes the **finest** stage's final MCMC prediction (in CLIP feature space at
H_low x W_low spatial resolution) and decodes to (3, out_size, out_size) RGB
in [0, 1] via a stack of stride-2 transpose-convs. This decoder is trained
**independently** of the energy hierarchy: callers must `.detach()` the
feature input before passing it in, so gradient from pixel loss does NOT flow
into HVEBT parameters. The decoder exists purely as a debugging / visualization
tool to confirm the predicted features are semantically meaningful.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _pow2_factor(target: int, base: int) -> int:
    if target % base != 0:
        raise ValueError(f"out_size {target} is not a multiple of in spatial {base}")
    ratio = target // base
    if ratio < 1 or (ratio & (ratio - 1)) != 0:
        raise ValueError(f"out_size/in_spatial ratio must be a power of 2, got {ratio}")
    return int(math.log2(ratio))


class PixelDecoder(nn.Module):
    """
    in:  (B, T, C, Hin, Win)  or (N, C, Hin, Win)
    out: (B, T, 3, S,   S)    or (N, 3, S,   S)   in [0, 1]

    `out_size` must be a power-of-2 multiple of `in_HW[0]==in_HW[1]`.
    """

    def __init__(
        self,
        in_channels: int,
        in_HW: Tuple[int, int],
        out_size: int = 256,
        base_channels: int = 128,
        min_channels: int = 32,
    ):
        super().__init__()
        Hin, Win = in_HW
        if Hin != Win:
            raise ValueError(f"PixelDecoder expects square input, got {in_HW}")
        n_up = _pow2_factor(out_size, Hin)
        self.in_channels = in_channels
        self.in_HW = (Hin, Win)
        self.out_size = out_size

        layers: list[nn.Module] = []
        ch = in_channels
        # Optional 1x1 conv to a friendlier channel count first.
        layers += [
            nn.Conv2d(ch, base_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
        ]
        ch = base_channels
        for _ in range(n_up):
            out_ch = max(ch // 2, min_channels)
            layers += [
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
            ]
            ch = out_ch
        layers += [nn.Conv2d(ch, 3, kernel_size=3, padding=1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            y = self.net(x.reshape(B * T, C, H, W))
            return y.reshape(B, T, 3, self.out_size, self.out_size)
        if x.dim() == 4:
            return self.net(x)
        raise ValueError(f"PixelDecoder expects 4D or 5D, got {x.shape}")


def save_recon_grid(
    real_rgb: torch.Tensor,         # (B, T, 3, S, S) in [0,1]
    pred_rgb: torch.Tensor,         # (B, T, 3, S, S) in [0,1]
    path: str,
    max_clips: int = 2,
) -> None:
    """
    Save a side-by-side comparison grid (real | pred) for the first
    `max_clips` clips at every time step. Falls back silently if torchvision
    is unavailable (no exception raised — the rest of training continues).
    """
    try:
        from torchvision.utils import make_grid, save_image
    except ImportError:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    B, T = real_rgb.shape[:2]
    n = min(B, max_clips)
    # Interleave real and pred per time step: rows = clips * 2, cols = T
    rows = []
    for b in range(n):
        rows.append(real_rgb[b])           # (T, 3, S, S)
        rows.append(pred_rgb[b].clamp(0, 1))
    stacked = torch.cat(rows, dim=0)       # (n*2*T, 3, S, S)
    grid = make_grid(stacked, nrow=T, padding=2)
    save_image(grid, path)
