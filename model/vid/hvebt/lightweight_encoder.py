"""
Lightweight 3-stage CNN encoder for Hierarchical Video EBT.

For 64x64 RGB input [0, 1] the encoder produces:
    16x16 : (B,  64, 16, 16)
    4x4   : (B, 128,  4,  4)
    1x1   : (B, 256,  1,  1)
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn

# Default stage geometry for 64x64 input.
DEFAULT_STAGES: Tuple[str, ...] = ("16x16", "4x4", "1x1")
DEFAULT_CHANNELS: Tuple[int, int, int] = (64, 128, 256)
INPUT_SIZE: int = 64


class LightweightMultiStageEncoder(nn.Module):
    """
    Trainable CNN pyramid: 64x64 -> 16x16 -> 4x4 -> 1x1.
    Input is expected in [0, 1] RGB (unnormalized).
    """

    ALL_STAGES: Tuple[str, ...] = DEFAULT_STAGES

    def __init__(
        self,
        return_stages: Optional[Iterable[str]] = None,
        channels: Tuple[int, int, int] = DEFAULT_CHANNELS,
        input_size: int = INPUT_SIZE,
        trainable: bool = True,
    ):
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"channels must be length-3 (16x16, 4x4, 1x1), got {channels}")
        self.return_stages = tuple(return_stages) if return_stages else self.ALL_STAGES
        for s in self.return_stages:
            if s not in self.ALL_STAGES:
                raise ValueError(f"Unknown stage '{s}'. Valid: {self.ALL_STAGES}")
        self.channels = channels
        self.input_size = input_size
        c0, c1, c2 = channels

        self.conv0 = nn.Conv2d(3, c0, kernel_size=4, stride=4, padding=0)
        self.norm0 = nn.GroupNorm(min(8, c0), c0)
        self.conv1 = nn.Conv2d(c0, c1, kernel_size=4, stride=4, padding=0)
        self.norm1 = nn.GroupNorm(min(8, c1), c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=4, stride=4, padding=0)
        self.norm2 = nn.GroupNorm(min(8, c2), c2)

        self._trainable = trainable
        if not trainable:
            for p in self.parameters():
                p.requires_grad = False

    @staticmethod
    def stage_shapes(input_size: int = INPUT_SIZE) -> Dict[str, Tuple[int, int, int]]:
        """Return (C, H, W) per stage for a square input."""
        if input_size != 64:
            raise ValueError(f"Only input_size=64 is supported, got {input_size}")
        return {
            "16x16": (DEFAULT_CHANNELS[0], 16, 16),
            "4x4": (DEFAULT_CHANNELS[1], 4, 4),
            "1x1": (DEFAULT_CHANNELS[2], 1, 1),
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) in [0, 1] RGB.
        Returns:
            dict mapping stage name -> (B, C, Hs, Ws).
        """
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected (B,3,H,W), got {tuple(x.shape)}")
        _, _, h, w = x.shape
        if h != self.input_size or w != self.input_size:
            raise ValueError(
                f"Expected {self.input_size}x{self.input_size} input, got {h}x{w}"
            )

        feats: Dict[str, torch.Tensor] = {}

        h0 = torch.nn.functional.gelu(self.norm0(self.conv0(x)))
        if "16x16" in self.return_stages:
            feats["16x16"] = h0

        h1 = torch.nn.functional.gelu(self.norm1(self.conv1(h0)))
        if "4x4" in self.return_stages:
            feats["4x4"] = h1

        h2 = torch.nn.functional.gelu(self.norm2(self.conv2(h1)))
        if "1x1" in self.return_stages:
            feats["1x1"] = h2

        return feats

    def encode_video(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, 3, H, W) in [0, 1].
        Returns:
            dict mapping stage -> (B, T, C, Hs, Ws).
        """
        if x.dim() != 5 or x.shape[2] != 3:
            raise ValueError(f"Expected (B,T,3,H,W), got {tuple(x.shape)}")
        B, T = x.shape[:2]
        flat = x.reshape(B * T, *x.shape[2:])
        feats = self.forward(flat)
        out: Dict[str, torch.Tensor] = {}
        for k, v in feats.items():
            out[k] = v.reshape(B, T, *v.shape[1:])
        return out
