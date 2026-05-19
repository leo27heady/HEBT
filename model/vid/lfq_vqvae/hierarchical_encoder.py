"""Hierarchical encoder: LFQ at every stage (bot / mid / top)."""

from __future__ import annotations

import torch
import torch.nn as nn
from vector_quantize_pytorch import LFQ

from .config import LFQVAEConfig
from .encoder import _build_downsample_stage


class _LFQStage(nn.Module):
    """Project -> (optional BN at top) -> LFQ -> project back."""

    def __init__(
        self,
        channels: int,
        lfq_dim: int,
        codebook_size: int,
        cfg: LFQVAEConfig,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        self.to_vq = nn.Conv2d(channels, lfq_dim, 1, bias=False)
        self.pre_vq_norm = nn.BatchNorm2d(lfq_dim) if use_bn else nn.Identity()
        self.from_vq = nn.Conv2d(lfq_dim, channels, 1, bias=False)
        self.vq = LFQ(
            codebook_size=codebook_size,
            dim=lfq_dim,
            entropy_loss_weight=cfg.entropy_loss_weight,
            diversity_gamma=cfg.diversity_gamma,
            channel_first=True,
        )

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.pre_vq_norm(self.to_vq(feat))
        quant, indices, vq_loss = self.vq(z)
        quant_feat = self.from_vq(quant)
        return quant_feat, indices, vq_loss


class LFQHierarchicalEncoder(nn.Module):
    """
    Bottom-up encoder with LFQ at each resolution.

    Default: 64 -> 16 (bot) -> 4 (mid) -> 1 (top), K = 32 / 512 / 4096.
    """

    def __init__(self, cfg: LFQVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n = len(cfg.stage_sizes)
        if n != 3:
            raise ValueError(
                f"Hierarchical encoder currently expects 3 stages, got {n}"
            )

        sizes = cfg.spatial_sizes_descending()
        c0, c1, c2 = cfg.stage_channels
        s0, s1, s2 = cfg.stage_sizes

        self.enc_to_bot = _build_downsample_stage(cfg.image_c, c0, sizes[0], s0)
        self.enc_bot_to_mid = _build_downsample_stage(c0, c1, s0, s1)
        self.enc_mid_to_top = _build_downsample_stage(c1, c2, s1, s2)

        k0, k1, k2 = cfg.stage_codebook_sizes
        d0, d1, d2 = cfg.stage_lfq_dims
        self.vq_bot = _LFQStage(c0, d0, k0, cfg, use_bn=False)
        self.vq_mid = _LFQStage(c1, d1, k1, cfg, use_bn=False)
        self.vq_top = _LFQStage(c2, d2, k2, cfg, use_bn=True)

    def forward(self, x: torch.Tensor) -> dict:
        feat_bot = self.enc_to_bot(x)
        quant_bot, idx_bot, vq_loss_bot = self.vq_bot(feat_bot)

        feat_mid = self.enc_bot_to_mid(feat_bot)
        quant_mid, idx_mid, vq_loss_mid = self.vq_mid(feat_mid)

        feat_top = self.enc_mid_to_top(feat_mid)
        quant_top, idx_top, vq_loss_top = self.vq_top(feat_top)

        vq_loss = vq_loss_bot + vq_loss_mid + vq_loss_top

        return {
            "feat_bot": feat_bot,
            "feat_mid": feat_mid,
            "feat_top": feat_top,
            "quant_bot": quant_bot,
            "quant_mid": quant_mid,
            "quant_top": quant_top,
            "quant_feat": quant_top,
            "idx_bot": idx_bot,
            "idx_mid": idx_mid,
            "idx_top": idx_top,
            "indices": idx_top,
            "vq_loss_bot": vq_loss_bot,
            "vq_loss_mid": vq_loss_mid,
            "vq_loss_top": vq_loss_top,
            "vq_loss": vq_loss,
        }
