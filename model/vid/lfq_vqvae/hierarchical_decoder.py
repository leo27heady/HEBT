"""Hierarchical decoder: top-down upsample, CE prior, configurable skip fusion."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LFQVAEConfig
from .decoder import _build_upsample_stage


class SkipFusion(nn.Module):
    """Fuse coarse decoder state with quantized encoder features at this scale."""

    def __init__(
        self,
        fusion: str,
        h_channels: int,
        skip_channels: int,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.proj_skip = nn.Conv2d(skip_channels, h_channels, 1, bias=False)

        if fusion == "concat":
            self.merge = nn.Sequential(
                nn.Conv2d(h_channels * 2, h_channels, 1, bias=False),
                nn.SiLU(),
            )
        elif fusion == "conv":
            self.merge = None
        elif fusion == "gamma":
            self.gamma = nn.Parameter(torch.zeros(1))
            self.merge = None
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

    def forward(self, h: torch.Tensor, quant_skip: torch.Tensor) -> torch.Tensor:
        skip = self.proj_skip(quant_skip)
        if self.fusion == "concat":
            return self.merge(torch.cat([h, skip], dim=1))
        if self.fusion == "conv":
            return h + skip
        return h + self.gamma * skip


class HierarchicalDecoderBlock(nn.Module):
    """Upsample coarse state, CE prior on target indices, fuse quantized skip."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        skip_ch: int,
        codebook_size: int,
        fusion: str,
        from_spatial: int,
        to_spatial: int,
    ) -> None:
        super().__init__()
        self.up = _build_upsample_stage(in_ch, out_ch, from_spatial, to_spatial)
        self.to_prior = nn.Conv2d(out_ch, codebook_size, 1)
        self.fusion = SkipFusion(fusion, out_ch, skip_ch)

    def forward(
        self,
        h: torch.Tensor,
        quant_skip: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.up(h)
        logits = self.to_prior(h)
        prior_ce = F.cross_entropy(logits, target_indices.long(), reduction="mean")
        h = self.fusion(h, quant_skip)
        return h, prior_ce

    def forward_no_ce(self, h: torch.Tensor, quant_skip: torch.Tensor) -> torch.Tensor:
        h = self.up(h)
        return self.fusion(h, quant_skip)


class DecoderStem(nn.Module):
    """Top level: start from quant_top; CE from learned prior vs idx_top."""

    def __init__(self, channels: int, codebook_size: int) -> None:
        super().__init__()
        self.prior = nn.Parameter(torch.full((1, channels, 1, 1), 0.01))
        self.to_prior = nn.Conv2d(channels, codebook_size, 1)

    def forward(
        self,
        quant_top: torch.Tensor,
        idx_top: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = quant_top.shape[0]
        prior = self.prior.expand(b, -1, -1, -1)
        logits = self.to_prior(prior)
        prior_ce = F.cross_entropy(logits, idx_top.long(), reduction="mean")
        return quant_top, prior_ce

    def forward_no_ce(self, quant_top: torch.Tensor) -> torch.Tensor:
        return quant_top


class DecoderImageTail(nn.Module):
    """Final image-space tail after hierarchical stage fusion."""

    def __init__(self, up_to_image: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.up_to_image = up_to_image
        self.head = head

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.up_to_image(h))


class DecoderTopOnlyPath(nn.Module):
    """Bottleneck-style decode from top quantized stage only."""

    def __init__(self, channels_top: int, channels_out: int, image_c: int, image_size: int) -> None:
        super().__init__()
        self.up = _build_upsample_stage(channels_top, channels_out, 1, image_size)
        self.head = nn.Sequential(
            nn.Conv2d(channels_out, channels_out // 2, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels_out // 2, image_c, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, top: torch.Tensor) -> torch.Tensor:
        return self.head(self.up(top))


class DecoderMidOnlyPath(nn.Module):
    """Decode from top->mid progressive path (without bot fusion)."""

    def __init__(self, channels_mid: int, channels_out: int, from_spatial: int, image_c: int, image_size: int) -> None:
        super().__init__()
        self.up = _build_upsample_stage(channels_mid, channels_out, from_spatial, image_size)
        self.head = nn.Sequential(
            nn.Conv2d(channels_out, channels_out // 2, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels_out // 2, image_c, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, mid: torch.Tensor) -> torch.Tensor:
        return self.head(self.up(mid))


class LFQHierarchicalDecoder(nn.Module):
    """
    Top-down decoder with notebook-style CE priors and fusion at each scale.

    Flow: quant_top -> [stem CE] -> up+fuse mid -> up+fuse bot -> up -> RGB.
    """

    def __init__(self, cfg: LFQVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if len(cfg.stage_sizes) != 3:
            raise ValueError(
                f"Hierarchical decoder expects 3 stages, got {len(cfg.stage_sizes)}"
            )

        c_bot, c_mid, c_top = cfg.stage_channels
        k_bot, k_mid, k_top = cfg.stage_codebook_sizes
        s_bot, s_mid, _s_top = cfg.stage_sizes

        self.stem = DecoderStem(c_top, k_top)
        self.block_mid = HierarchicalDecoderBlock(
            c_top, c_mid, c_mid, k_mid, cfg.fusion, from_spatial=1, to_spatial=s_mid
        )
        self.block_bot = HierarchicalDecoderBlock(
            c_mid, c_bot, c_bot, k_bot, cfg.fusion, from_spatial=s_mid, to_spatial=s_bot
        )
        self.up_to_image = _build_upsample_stage(
            c_bot, c_bot, s_bot, cfg.image_size
        )
        self.head = nn.Sequential(
            nn.Conv2d(c_bot, c_bot // 2, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(c_bot // 2, cfg.image_c, 3, padding=1),
            nn.Sigmoid(),
        )
        self.image_tail = DecoderImageTail(self.up_to_image, self.head)
        self.top_only_tail = DecoderTopOnlyPath(
            channels_top=c_top,
            channels_out=c_bot,
            image_c=cfg.image_c,
            image_size=cfg.image_size,
        )
        self.mid_only_tail = DecoderMidOnlyPath(
            channels_mid=c_mid,
            channels_out=c_bot,
            from_spatial=s_mid,
            image_c=cfg.image_c,
            image_size=cfg.image_size,
        )

    def forward(
        self,
        enc: dict,
        depth: int = 3,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if depth < 1 or depth > 3:
            raise ValueError(f"depth must be in [1,3], got {depth}")
        prior_ces: List[torch.Tensor] = []

        h, ce = self.stem(enc["quant_top"], enc["idx_top"])
        prior_ces.append(ce)
        if depth == 1:
            return self.top_only_tail(h), prior_ces

        h, ce = self.block_mid(h, enc["quant_mid"], enc["idx_mid"])
        prior_ces.append(ce)
        if depth == 2:
            return self.mid_only_tail(h), prior_ces

        h, ce = self.block_bot(h, enc["quant_bot"], enc["idx_bot"])
        prior_ces.append(ce)

        x_hat = self.image_tail(h)
        return x_hat, prior_ces

    def decode_from_stages(
        self,
        quant_top: torch.Tensor,
        quant_mid: torch.Tensor,
        quant_bot: torch.Tensor,
    ) -> torch.Tensor:
        """Decode from provided stage features without prior CE terms."""
        h = self.stem.forward_no_ce(quant_top)
        h = self.block_mid.forward_no_ce(h, quant_mid)
        h = self.block_bot.forward_no_ce(h, quant_bot)
        return self.image_tail(h)

    def gamma_values(self) -> List[float]:
        """Collect learnable gamma scalars when fusion='gamma'."""
        gammas: List[float] = []
        for mod in self.modules():
            if isinstance(mod, SkipFusion) and mod.fusion == "gamma":
                gammas.append(float(mod.gamma.detach().item()))
        return gammas
