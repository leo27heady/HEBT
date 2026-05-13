"""Hierarchical Encoder with LFQ quantization at 3 resolutions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import LFQ

from .config import FreshHVQVAEConfig


class ResBlock(nn.Module):
    """Residual block with optional stride for downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride)
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class HierarchicalEncoder(nn.Module):
    """
    Bottom-up convolutional encoder with VQ taps at 3 resolutions.
    Input:  (B, 3, 64, 64)
    Output: dict with quantized features, indices, and entropy losses per stage.
    """

    def __init__(self, cfg: FreshHVQVAEConfig):
        super().__init__()
        C_bot, C_mid, C_top = cfg.C_bot, cfg.C_mid, cfg.C_top

        # Stage 1: 64x64 → 16x16
        self.enc_to_bot = nn.Sequential(
            ResBlock(3, 64, stride=2),       # 64→32
            ResBlock(64, C_bot, stride=2),   # 32→16
        )
        self.bot_to_vq = nn.Conv2d(C_bot, cfg.lfq_dim_bot, 1)
        self.bot_from_vq = nn.Conv2d(cfg.lfq_dim_bot, C_bot, 1)
        self.vq_bot = LFQ(
            codebook_size=cfg.K_bot,
            dim=cfg.lfq_dim_bot,
            entropy_loss_weight=cfg.entropy_loss_weight,
            diversity_gamma=cfg.diversity_gamma,
            channel_first=True,
        )

        # Stage 2: 16x16 → 4x4
        self.enc_bot_to_mid = nn.Sequential(
            ResBlock(C_bot, C_mid, stride=2),  # 16→8
            ResBlock(C_mid, C_mid, stride=2),  # 8→4
        )
        self.mid_to_vq = nn.Conv2d(C_mid, cfg.lfq_dim_mid, 1)
        self.mid_from_vq = nn.Conv2d(cfg.lfq_dim_mid, C_mid, 1)
        self.vq_mid = LFQ(
            codebook_size=cfg.K_mid,
            dim=cfg.lfq_dim_mid,
            entropy_loss_weight=cfg.entropy_loss_weight,
            diversity_gamma=cfg.diversity_gamma,
            channel_first=True,
        )

        # Stage 3: 4x4 → 1x1
        self.enc_mid_to_top = nn.Sequential(
            ResBlock(C_mid, C_top, stride=2),  # 4→2
            ResBlock(C_top, C_top, stride=2),  # 2→1
        )
        self.top_to_vq = nn.Conv2d(C_top, cfg.lfq_dim_top, 1)
        self.top_from_vq = nn.Conv2d(cfg.lfq_dim_top, C_top, 1)
        self.vq_top = LFQ(
            codebook_size=cfg.K_top,
            dim=cfg.lfq_dim_top,
            entropy_loss_weight=cfg.entropy_loss_weight,
            diversity_gamma=cfg.diversity_gamma,
            channel_first=True,
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, 3, 64, 64)
        Returns dict with keys:
            quant_bot, idx_bot, loss_bot,
            quant_mid, idx_mid, loss_mid,
            quant_top, idx_top, loss_top
        """
        # Bot stage
        feat_bot = self.enc_to_bot(x)                    # (B, C_bot, 16, 16)
        z_bot = self.bot_to_vq(feat_bot)                 # (B, lfq_dim_bot, 16, 16)
        quant_bot, idx_bot, loss_bot = self.vq_bot(z_bot)  # channel_first handles (B,C,H,W)
        quant_bot_feat = self.bot_from_vq(quant_bot)     # (B, C_bot, 16, 16)

        # Mid stage (takes pre-VQ features for gradient flow)
        feat_mid = self.enc_bot_to_mid(feat_bot)         # (B, C_mid, 4, 4)
        z_mid = self.mid_to_vq(feat_mid)                 # (B, lfq_dim_mid, 4, 4)
        quant_mid, idx_mid, loss_mid = self.vq_mid(z_mid)
        quant_mid_feat = self.mid_from_vq(quant_mid)

        # Top stage
        feat_top = self.enc_mid_to_top(feat_mid)         # (B, C_top, 1, 1)
        z_top = self.top_to_vq(feat_top)                 # (B, lfq_dim_top, 1, 1)
        quant_top, idx_top, loss_top = self.vq_top(z_top)
        quant_top_feat = self.top_from_vq(quant_top)

        return {
            'quant_bot': quant_bot_feat, 'idx_bot': idx_bot, 'loss_bot': loss_bot,
            'quant_mid': quant_mid_feat, 'idx_mid': idx_mid, 'loss_mid': loss_mid,
            'quant_top': quant_top_feat, 'idx_top': idx_top, 'loss_top': loss_top,
        }
