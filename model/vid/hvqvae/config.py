"""
Configuration dataclasses for Hierarchical VQ-VAE (HVQVAE).

Minimal hierarchical VQ-VAE for video frame prediction.
No MCMC, no EBT, no energy functions — direct logit prediction.

Hierarchy (three stages, coarsest first in prediction order):
    s3:  256 channels, 2×2 spatial   (coarsest / apex)
    s2:  128 channels, 4×4 spatial
    s1:   64 channels, 8×8 spatial   (finest / base)

Spatial sizes assume 64×64 input frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HVQVAEStageConfig:
    """Configuration for one hierarchy stage."""

    stage_name: str = "s1"
    channels: int = 64          # encoder output / embedding dim at this stage
    H: int = 8                  # spatial height
    W: int = 8                  # spatial width
    num_codes: int = 64         # codebook size K
    transformer_dim: int = 64   # predictor internal dim D (= channels, no bottleneck)
    n_heads: int = 2            # attention heads
    n_layers: int = 2           # transformer blocks
    temporal_window: Optional[int] = None   # None = full causal
    spatial_window: Optional[int] = None    # None = full spatial


@dataclass
class HVQVAEConfig:
    """Top-level configuration for HVQVAE."""

    stages: List[HVQVAEStageConfig] = field(default_factory=lambda: _default_stages())
    beta: float = 0.25                  # VQ commitment cost
    encoder_h_dim: int = 128            # encoder hidden dim
    encoder_res_h_dim: int = 32         # residual block hidden dim
    encoder_n_res_layers: int = 2       # residual layers per stage
    decoder_out_size: int = 64          # decoder output pixel size
    image_size: int = 64                # input image size


def _default_stages() -> List[HVQVAEStageConfig]:
    """Three-stage hierarchy for 64×64 images: s3 (coarsest) → s2 → s1 (finest).

    Codebook K is proportional to compression: coarse stages represent more
    content per token, so they need a richer vocabulary.
    """
    s3 = HVQVAEStageConfig(
        stage_name="s3",
        channels=256, H=2, W=2,
        num_codes=512,
        transformer_dim=256, n_heads=4, n_layers=2,
        temporal_window=None,   # full causal — coarsest needs full temporal context
        spatial_window=None,    # 2×2 is already tiny
    )
    s2 = HVQVAEStageConfig(
        stage_name="s2",
        channels=128, H=4, W=4,
        num_codes=256,
        transformer_dim=128, n_heads=4, n_layers=2,
        temporal_window=2,
        spatial_window=None,
    )
    s1 = HVQVAEStageConfig(
        stage_name="s1",
        channels=64, H=8, W=8,
        num_codes=64,
        transformer_dim=64, n_heads=2, n_layers=2,
        temporal_window=1,      # self-frame only — finest focuses on spatial detail
        spatial_window=None,
    )
    return [s3, s2, s1]
