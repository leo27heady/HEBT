"""Decoder: progressive upsampling from quantized bottleneck only (no skips)."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .blocks import ResBlock
from .config import LFQVAEConfig, _log2_int


def _build_upsample_stage(
    in_ch: int,
    out_ch: int,
    from_spatial: int,
    to_spatial: int,
) -> nn.Sequential:
    """ResBlock + ConvTranspose2d x log2(to/from) to upscale spatially."""
    factor = to_spatial // from_spatial
    n_steps = _log2_int(factor)
    layers: List[nn.Module] = []
    ch = in_ch
    for i in range(n_steps):
        next_ch = out_ch if i == n_steps - 1 else max(ch, out_ch)
        layers.append(ResBlock(ch, next_ch))
        layers.append(nn.ConvTranspose2d(next_ch, next_ch, 4, stride=2, padding=1, bias=False))
        layers.append(nn.SiLU())
        ch = next_ch
    return nn.Sequential(*layers)


class LFQDecoder(nn.Module):
    """
    Upsample from quantized bottleneck features to full image.

    Input:  (B, C_top, 1, 1)
    Output: (B, 3, H, W) in [0, 1]
    """

    def __init__(self, cfg: LFQVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        ascending = list(cfg.spatial_sizes_ascending())
        # Channels at each spatial level when decoding upward (bottleneck -> coarse)
        ch_at_level = list(reversed(cfg.stage_channels))

        self.stages = nn.ModuleList()
        for i in range(len(ascending) - 1):
            from_s, to_s = ascending[i], ascending[i + 1]
            in_ch = ch_at_level[i]
            if i + 1 < len(ch_at_level):
                out_ch = ch_at_level[i + 1]
            else:
                out_ch = cfg.stage_channels[0]
            self.stages.append(_build_upsample_stage(in_ch, out_ch, from_s, to_s))

        head_in = cfg.stage_channels[0]
        self.head = nn.Sequential(
            nn.Conv2d(head_in, head_in // 2, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(head_in // 2, cfg.image_c, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, quant_feat: torch.Tensor) -> torch.Tensor:
        h = quant_feat
        for stage in self.stages:
            h = stage(h)
        return self.head(h)
