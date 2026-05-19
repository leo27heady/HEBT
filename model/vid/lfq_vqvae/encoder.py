"""Encoder: continuous multi-stage downsampling + LFQ at 1x1 bottleneck."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from vector_quantize_pytorch import LFQ

from .blocks import ResBlock
from .config import LFQVAEConfig, _log2_int


def _build_downsample_stage(
    in_ch: int,
    out_ch: int,
    from_spatial: int,
    to_spatial: int,
) -> nn.Sequential:
    """Stack of stride-2 ResBlocks reducing spatial size from_spatial -> to_spatial."""
    factor = from_spatial // to_spatial
    n_blocks = _log2_int(factor)
    blocks: List[nn.Module] = []
    ch = in_ch
    for i in range(n_blocks):
        if i == n_blocks - 1:
            blocks.append(ResBlock(ch, out_ch, stride=2))
        else:
            mid = 64 if (in_ch == 3 and i == 0) else max(ch, out_ch)
            blocks.append(ResBlock(ch, mid, stride=2))
            ch = mid
    return nn.Sequential(*blocks)


class LFQEncoder(nn.Module):
    """
    Bottom-up encoder with continuous intermediate stages and LFQ only at the end.

    Input:  (B, 3, H, H)
    Output: dict with feat_pre_vq, quant_feat, indices, vq_loss, z_pre_vq
    """

    def __init__(self, cfg: LFQVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.stages = nn.ModuleList()
        prev_ch = cfg.image_c
        sizes = cfg.spatial_sizes_descending()
        for i, (out_ch, out_spatial) in enumerate(zip(cfg.stage_channels, cfg.stage_sizes)):
            from_spatial = sizes[i]
            self.stages.append(
                _build_downsample_stage(prev_ch, out_ch, from_spatial, out_spatial)
            )
            prev_ch = out_ch

        c_top = cfg.bottleneck_channels
        self.top_to_vq = nn.Conv2d(c_top, cfg.lfq_dim, 1, bias=False)
        self.top_pre_vq_norm = nn.BatchNorm2d(cfg.lfq_dim)
        self.top_from_vq = nn.Conv2d(cfg.lfq_dim, c_top, 1, bias=False)
        self.vq = LFQ(
            codebook_size=cfg.codebook_size,
            dim=cfg.lfq_dim,
            entropy_loss_weight=cfg.entropy_loss_weight,
            diversity_gamma=cfg.diversity_gamma,
            channel_first=True,
        )

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous bottleneck features before VQ projection."""
        h = x
        for stage in self.stages:
            h = stage(h)
        return h

    def forward(self, x: torch.Tensor) -> dict:
        feat_pre_vq = self.encode_features(x)
        z = self.top_to_vq(feat_pre_vq)
        z = self.top_pre_vq_norm(z)
        quant, indices, vq_loss = self.vq(z)
        quant_feat = self.top_from_vq(quant)
        return {
            "feat_pre_vq": feat_pre_vq,
            "quant_feat": quant_feat,
            "indices": indices,
            "vq_loss": vq_loss,
            "z_pre_vq": z,
        }
