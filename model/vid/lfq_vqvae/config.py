"""Configuration for LFQ VQ-VAE (reconstruction + optional video predictor)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _log2_int(n: int) -> int:
    if not _is_power_of_two(n):
        raise ValueError(f"{n} is not a power of 2")
    return int(math.log2(n))


@dataclass
class LFQVAEConfig:
    """VQ-VAE with LFQ (bottleneck or hierarchical), plus optional video predictor."""

    image_size: int = 64
    image_c: int = 3
    quantization_mode: Literal["bottleneck", "hierarchical"] = "bottleneck"

    # Spatial sizes after each encoder stage (bot -> mid -> top)
    stage_sizes: tuple[int, ...] = (16, 4, 1)
    stage_channels: tuple[int, ...] = (64, 128, 256)

    # Bottleneck mode: single LFQ at final stage
    codebook_size: int = 2**12
    lfq_dim: int = 12

    # Hierarchical mode: per-stage LFQ (bot, mid, top)
    stage_codebook_sizes: tuple[int, ...] = (8, 32, 128)
    stage_lfq_dims: tuple[int, ...] = (3, 5, 7)

    # LFQ regularization
    entropy_loss_weight: float = 0.1
    diversity_gamma: float = 1.0

    # Hierarchical decoder (reconstruction prior CE)
    fusion: Literal["concat", "conv", "gamma"] = "conv"
    lambda_prior_ce: float = 1.0
    prior_ce_weights: Literal["spatial", "uniform"] = "spatial"
    gamma_l2: float = 0.0

    # Video predictor
    enable_video_predictor: bool = False
    predictor_mode: Literal["vanilla"] = "vanilla"
    train_mode: Literal["recon_only", "disjoint", "joint", "progressive"] = "recon_only"
    progressive_stage_steps: tuple[int, ...] = (8000, 8000, 8000)
    progressive_steps_per_stage: int | None = None
    progressive_freeze_parents: bool = True
    progressive_prior_ce: bool = False
    pred_n_heads: int = 8
    pred_n_layers: int = 4
    pred_dim_top: int = 256
    pred_dim_mid: int = 128
    pred_dim_bot: int = 64
    window_top: int = -1
    window_mid: int = 2
    window_bot: int = 1
    soft_lookup_temperature: float = 1.0
    use_gumbel_softmax: bool = False
    gumbel_tau: float = 1.0
    max_T: int = 16
    detach_encoder_for_predictor: bool = True
    detach_parent_features: bool = True

    # Video losses
    lambda_ce: float = 1.0
    lambda_pred_mse: float = 0.0

    # Loss
    recon_loss: str = "mse"  # "mse" | "l1"
    vq_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.progressive_steps_per_stage is not None:
            if self.progressive_steps_per_stage < 1:
                raise ValueError(
                    "progressive_steps_per_stage must be >= 1, got "
                    f"{self.progressive_steps_per_stage}"
                )
            if self.progressive_stage_steps == (8000, 8000, 8000):
                steps = self.progressive_steps_per_stage
                self.progressive_stage_steps = (steps, steps, steps)
        self.validate()

    def validate(self) -> None:
        if self.quantization_mode not in ("bottleneck", "hierarchical"):
            raise ValueError(
                f"quantization_mode must be 'bottleneck' or 'hierarchical', got {self.quantization_mode!r}"
            )
        if self.fusion not in ("concat", "conv", "gamma"):
            raise ValueError(f"fusion must be 'concat', 'conv', or 'gamma', got {self.fusion!r}")
        if self.prior_ce_weights not in ("spatial", "uniform"):
            raise ValueError(
                f"prior_ce_weights must be 'spatial' or 'uniform', got {self.prior_ce_weights!r}"
            )
        if self.predictor_mode != "vanilla":
            raise ValueError(f"predictor_mode must be 'vanilla', got {self.predictor_mode!r}")
        if self.train_mode not in ("recon_only", "disjoint", "joint", "progressive"):
            raise ValueError(
                f"train_mode must be one of recon_only/disjoint/joint/progressive, got {self.train_mode!r}"
            )
        if len(self.progressive_stage_steps) != len(self.stage_sizes):
            raise ValueError(
                "progressive_stage_steps must match stage_sizes length "
                f"({len(self.progressive_stage_steps)} vs {len(self.stage_sizes)})"
            )
        for i, steps in enumerate(self.progressive_stage_steps):
            if steps < 1:
                raise ValueError(f"progressive stage {i} steps must be >= 1, got {steps}")

        if len(self.stage_sizes) != len(self.stage_channels):
            raise ValueError(
                "stage_sizes and stage_channels must have the same length "
                f"({len(self.stage_sizes)} vs {len(self.stage_channels)})"
            )
        if len(self.stage_sizes) < 1:
            raise ValueError("At least one encoder stage is required")
        if self.stage_sizes[-1] != 1:
            raise ValueError(f"Final stage spatial size must be 1, got {self.stage_sizes[-1]}")

        sizes = (self.image_size,) + tuple(self.stage_sizes)
        for i in range(len(sizes) - 1):
            from_s, to_s = sizes[i], sizes[i + 1]
            if from_s <= to_s:
                raise ValueError(f"Stage sizes must strictly decrease: {from_s} -> {to_s}")
            factor = from_s // to_s
            if factor * to_s != from_s or not _is_power_of_two(factor):
                raise ValueError(f"Downsample factor {from_s}/{to_s} must be a power of 2")

        if self.quantization_mode == "hierarchical":
            if len(self.stage_codebook_sizes) != len(self.stage_sizes):
                raise ValueError(
                    "stage_codebook_sizes must match stage_sizes length "
                    f"({len(self.stage_codebook_sizes)} vs {len(self.stage_sizes)})"
                )
            if len(self.stage_lfq_dims) != len(self.stage_sizes):
                raise ValueError(
                    "stage_lfq_dims must match stage_sizes length "
                    f"({len(self.stage_lfq_dims)} vs {len(self.stage_sizes)})"
                )
            for i, (k, d) in enumerate(zip(self.stage_codebook_sizes, self.stage_lfq_dims)):
                if not _is_power_of_two(k):
                    raise ValueError(f"stage {i} codebook_size must be power of 2, got {k}")
                if 2**d != k:
                    raise ValueError(f"stage {i}: lfq_dim={d} requires codebook_size={2**d}, got {k}")
        else:
            if not _is_power_of_two(self.codebook_size):
                raise ValueError(f"codebook_size must be a power of 2, got {self.codebook_size}")
            if 2**self.lfq_dim != self.codebook_size:
                raise ValueError(
                    f"lfq_dim={self.lfq_dim} requires codebook_size={2**self.lfq_dim}, got {self.codebook_size}"
                )

        if self.recon_loss not in ("mse", "l1"):
            raise ValueError(f"recon_loss must be 'mse' or 'l1', got {self.recon_loss!r}")
        if self.max_T < 1:
            raise ValueError(f"max_T must be >= 1, got {self.max_T}")

        if self.enable_video_predictor:
            if self.quantization_mode != "hierarchical":
                raise ValueError("Video predictor requires quantization_mode='hierarchical'")
            if len(self.stage_sizes) != 3:
                raise ValueError("Video predictor v1 expects exactly 3 stages (16, 4, 1)")
            if tuple(self.stage_sizes) != (16, 4, 1):
                raise ValueError(
                    f"Video predictor v1 expects stage_sizes=(16, 4, 1), got {self.stage_sizes}"
                )
            if self.pred_dim_top != self.stage_channels[2]:
                raise ValueError(
                    "pred_dim_top must match top stage channel size "
                    f"({self.pred_dim_top} vs {self.stage_channels[2]})"
                )
            if self.pred_dim_mid != self.stage_channels[1]:
                raise ValueError(
                    "pred_dim_mid must match mid stage channel size "
                    f"({self.pred_dim_mid} vs {self.stage_channels[1]})"
                )
            if self.pred_dim_bot != self.stage_channels[0]:
                raise ValueError(
                    "pred_dim_bot must match bot stage channel size "
                    f"({self.pred_dim_bot} vs {self.stage_channels[0]})"
                )

        if self.train_mode == "progressive" and self.quantization_mode != "hierarchical":
            raise ValueError("train_mode='progressive' requires quantization_mode='hierarchical'")

    @property
    def bottleneck_channels(self) -> int:
        return self.stage_channels[-1]

    def stage_codebook_size(self, stage_idx: int) -> int:
        if self.quantization_mode == "hierarchical":
            return self.stage_codebook_sizes[stage_idx]
        return self.codebook_size

    def stage_lfq_dim(self, stage_idx: int) -> int:
        if self.quantization_mode == "hierarchical":
            return self.stage_lfq_dims[stage_idx]
        return self.lfq_dim

    def prior_ce_weight(self, spatial_size: int) -> float:
        """Notebook-style weight: n_latent / n_image_pixels."""
        if self.prior_ce_weights == "uniform":
            return 1.0
        n_latent = spatial_size * spatial_size
        n_image = self.image_size * self.image_size * self.image_c
        return n_latent / n_image

    def spatial_sizes_descending(self) -> tuple[int, ...]:
        """All spatial sizes from image down to bottleneck: (64, 16, 4, 1)."""
        return (self.image_size,) + tuple(self.stage_sizes)

    def spatial_sizes_ascending(self) -> tuple[int, ...]:
        """Bottleneck up to image: (1, 4, 16, 64)."""
        return tuple(reversed(self.spatial_sizes_descending()))
