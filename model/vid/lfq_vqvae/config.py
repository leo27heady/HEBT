"""Configuration for simple LFQ VQ-VAE (encode-decode only)."""

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
    """VQ-VAE with LFQ (bottleneck or hierarchical multi-stage quantization)."""

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
    stage_codebook_sizes: tuple[int, ...] = (64, 512, 4096)
    stage_lfq_dims: tuple[int, ...] = (6, 9, 12)

    entropy_loss_weight: float = 0.1
    diversity_gamma: float = 1.0

    # Hierarchical decoder
    fusion: Literal["concat", "conv", "gamma"] = "conv"
    lambda_prior_ce: float = 1.0
    prior_ce_weights: Literal["spatial", "uniform"] = "spatial"
    gamma_l2: float = 0.0

    # Loss
    recon_loss: str = "mse"  # "mse" | "l1"
    vq_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.quantization_mode not in ("bottleneck", "hierarchical"):
            raise ValueError(
                f"quantization_mode must be 'bottleneck' or 'hierarchical', "
                f"got {self.quantization_mode!r}"
            )
        if self.fusion not in ("concat", "conv", "gamma"):
            raise ValueError(f"fusion must be 'concat', 'conv', or 'gamma', got {self.fusion!r}")
        if self.prior_ce_weights not in ("spatial", "uniform"):
            raise ValueError(
                f"prior_ce_weights must be 'spatial' or 'uniform', "
                f"got {self.prior_ce_weights!r}"
            )

        if len(self.stage_sizes) != len(self.stage_channels):
            raise ValueError(
                "stage_sizes and stage_channels must have the same length "
                f"({len(self.stage_sizes)} vs {len(self.stage_channels)})"
            )
        if len(self.stage_sizes) < 1:
            raise ValueError("At least one encoder stage is required")

        if self.stage_sizes[-1] != 1:
            raise ValueError(
                f"Final stage spatial size must be 1, got {self.stage_sizes[-1]}"
            )

        sizes = (self.image_size,) + tuple(self.stage_sizes)
        for i in range(len(sizes) - 1):
            from_s, to_s = sizes[i], sizes[i + 1]
            if from_s <= to_s:
                raise ValueError(
                    f"Stage sizes must strictly decrease: {from_s} -> {to_s}"
                )
            factor = from_s // to_s
            if factor * to_s != from_s or not _is_power_of_two(factor):
                raise ValueError(
                    f"Downsample factor {from_s}/{to_s} must be a power of 2"
                )

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
            for i, (k, d) in enumerate(
                zip(self.stage_codebook_sizes, self.stage_lfq_dims)
            ):
                if not _is_power_of_two(k):
                    raise ValueError(f"stage {i} codebook_size must be power of 2, got {k}")
                if 2**d != k:
                    raise ValueError(
                        f"stage {i}: lfq_dim={d} requires codebook_size={2**d}, got {k}"
                    )
        else:
            if not _is_power_of_two(self.codebook_size):
                raise ValueError(
                    f"codebook_size must be a power of 2, got {self.codebook_size}"
                )
            if 2**self.lfq_dim != self.codebook_size:
                raise ValueError(
                    f"lfq_dim={self.lfq_dim} requires codebook_size={2**self.lfq_dim}, "
                    f"got {self.codebook_size}"
                )

        if self.recon_loss not in ("mse", "l1"):
            raise ValueError(f"recon_loss must be 'mse' or 'l1', got {self.recon_loss!r}")

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
