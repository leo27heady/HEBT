"""Configuration for simple LFQ VQ-VAE (encode-decode only)."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _log2_int(n: int) -> int:
    if not _is_power_of_two(n):
        raise ValueError(f"{n} is not a power of 2")
    return int(math.log2(n))


@dataclass
class LFQVAEConfig:
    """VQ-VAE with LFQ at the final spatial bottleneck only."""

    image_size: int = 64
    image_c: int = 3

    # Spatial sizes after each encoder stage (coarse-to-fine order in the list)
    stage_sizes: tuple[int, ...] = (16, 4, 1)
    stage_channels: tuple[int, ...] = (64, 128, 256)

    # LFQ at final stage only
    codebook_size: int = 2**12
    lfq_dim: int = 12
    entropy_loss_weight: float = 0.1
    diversity_gamma: float = 1.0

    # Loss
    recon_loss: str = "mse"  # "mse" | "l1"
    vq_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
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

        if not _is_power_of_two(self.codebook_size):
            raise ValueError(f"codebook_size must be a power of 2, got {self.codebook_size}")
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

    def downsample_factor(self, stage_idx: int) -> int:
        """Integer factor from previous spatial size to stage_sizes[stage_idx]."""
        prev = self.image_size if stage_idx == 0 else self.stage_sizes[stage_idx - 1]
        return prev // self.stage_sizes[stage_idx]

    def upsample_factor(self, stage_idx: int) -> int:
        """Integer factor from stage_sizes[stage_idx] to next spatial size."""
        if stage_idx == len(self.stage_sizes) - 1:
            return self.image_size // self.stage_sizes[-1]
        return self.stage_sizes[stage_idx] // self.stage_sizes[stage_idx + 1]

    def spatial_sizes_descending(self) -> tuple[int, ...]:
        """All spatial sizes from image down to bottleneck: (64, 16, 4, 1)."""
        return (self.image_size,) + tuple(self.stage_sizes)

    def spatial_sizes_ascending(self) -> tuple[int, ...]:
        """Bottleneck up to image: (1, 4, 16, 64)."""
        return tuple(reversed(self.spatial_sizes_descending()))
