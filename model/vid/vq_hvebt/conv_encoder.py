"""
Lightweight multi-stage convolutional encoder for VQ-HVEBT (Option B).

Replaces the large pretrained CLIP backbone with a small (~2M param)
ConvNet trained from scratch. This eliminates the 10⁸-magnitude gradient
explosion that fine-tuning a pretrained 50M-param vision transformer causes.

Architecture
------------
Input: (B, 3, 256, 256) RGB in [0, 1]

Five stride-2 stages downsample 256 → 128 → 64 → 32 → 16 → 8.
Each stage is a ConvBlock: Conv(3×3, s=2) → GroupNorm → GELU → Conv(3×3) → GroupNorm → GELU.

Output stages (matching CLIP naming for backward compat):
    s1 : (B, 128, 32, 32)   — after stage 2
    s2 : (B, 256, 16, 16)   — after stage 3
    s3 : (B, 512,  8,  8)   — after stage 4

EMA target encoder
------------------
A frozen copy of this encoder is maintained via exponential moving average
(decay ~0.999). The EMA copy produces *target* codes for the CE loss, so
targets are stable even when the live encoder moves. This is the same trick
as BYOL/DINO/I-JEPA.
"""
from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #


class ConvBlock(nn.Module):
    """Two-conv residual block with optional spatial downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.GELU()

        # Shortcut for channel/spatial mismatch.
        self.shortcut: Optional[nn.Module] = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(min(32, out_ch), out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + identity)


# --------------------------------------------------------------------------- #
#  Multi-stage encoder
# --------------------------------------------------------------------------- #

# Stage name → (output_channels, spatial_size for 256×256 input)
_STAGE_SPEC: Dict[str, Tuple[int, int]] = {
    "s1": (128, 32),
    "s2": (256, 16),
    "s3": (512,  8),
}


class MultiStageConvEncoder(nn.Module):
    """Lightweight ConvNet producing multi-scale feature maps.

    For 256×256 input:
        stem   : 3 → 64 ch, 256 → 128
        stage1 : 64 → 128 ch, 128 → 64
        stage2 : 128 → 128 ch, 64 → 32   → s1 output
        stage3 : 128 → 256 ch, 32 → 16   → s2 output
        stage4 : 256 → 512 ch, 16 → 8    → s3 output

    Parameters
    ----------
    return_stages : which stages to return (subset of {"s1", "s2", "s3"}).
    base_channels : channel width of the stem.
    """

    def __init__(
        self,
        return_stages: Iterable[str] = ("s1", "s2", "s3"),
        base_channels: int = 64,
    ):
        super().__init__()
        self.return_stages = tuple(return_stages)
        for s in self.return_stages:
            if s not in _STAGE_SPEC:
                raise ValueError(f"Unknown stage '{s}'. Valid: {list(_STAGE_SPEC)}")

        C = base_channels
        # stem: 3 → C, 256 → 128
        self.stem = ConvBlock(3, C, stride=2)
        # stage1: C → 2C, 128 → 64
        self.stage1 = ConvBlock(C, C * 2, stride=2)
        # stage2: 2C → 2C, 64 → 32  (s1 output: 128 ch, 32×32)
        self.stage2 = ConvBlock(C * 2, C * 2, stride=2)
        # stage3: 2C → 4C, 32 → 16  (s2 output: 256 ch, 16×16)
        self.stage3 = ConvBlock(C * 2, C * 4, stride=2)
        # stage4: 4C → 8C, 16 → 8   (s3 output: 512 ch, 8×8)
        self.stage4 = ConvBlock(C * 4, C * 8, stride=2)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode a batch of images.

        Args:
            x: (B, 3, H, W) in [0, 1] RGB.

        Returns:
            dict mapping stage name → (B, C, Hs, Ws) feature map.
        """
        out: Dict[str, torch.Tensor] = {}

        h = self.stem(x)      # (B, 64, 128, 128)
        h = self.stage1(h)    # (B, 128, 64, 64)
        h = self.stage2(h)    # (B, 128, 32, 32)
        if "s1" in self.return_stages:
            out["s1"] = h

        h = self.stage3(h)    # (B, 256, 16, 16)
        if "s2" in self.return_stages:
            out["s2"] = h

        h = self.stage4(h)    # (B, 512, 8, 8)
        if "s3" in self.return_stages:
            out["s3"] = h

        return out


# --------------------------------------------------------------------------- #
#  EMA wrapper (BYOL/DINO-style target network)
# --------------------------------------------------------------------------- #


class EMAEncoder(nn.Module):
    """Exponential moving average copy of an encoder.

    The EMA encoder is never trained by gradient. Its weights track the live
    encoder's weights via ``update(live_encoder)``. It produces stable target
    features/codes so that the predictor's CE loss targets don't flip every
    step (the root cause of VQ training instability).

    Usage:
        ema = EMAEncoder(live_encoder, decay=0.999)
        ...
        loss.backward()
        optimizer.step()
        ema.update(live_encoder)  # after each optimizer step
    """

    def __init__(self, live_encoder: MultiStageConvEncoder, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        # Deep copy the encoder — completely separate parameters.
        self.encoder = copy.deepcopy(live_encoder)
        # Freeze all params.
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, live_encoder: MultiStageConvEncoder) -> None:
        """EMA update: ema_θ ← decay * ema_θ + (1 - decay) * live_θ."""
        for ema_p, live_p in zip(self.encoder.parameters(), live_encoder.parameters()):
            ema_p.data.mul_(self.decay).add_(live_p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode with the frozen EMA copy (always no_grad)."""
        return self.encoder(x)

    def encode_video(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode a video batch (B, T, 3, H, W) → {stage: (B, T, C, Hs, Ws)}.

        Folds T into batch, encodes, unfolds.
        """
        B, T, _, H, W = video.shape
        flat = video.reshape(B * T, 3, H, W)
        with torch.no_grad():
            feats_flat = self.encoder(flat)
        result: Dict[str, torch.Tensor] = {}
        for name, feat in feats_flat.items():
            _, C, Hs, Ws = feat.shape
            result[name] = feat.reshape(B, T, C, Hs, Ws)
        return result


# --------------------------------------------------------------------------- #
#  Video encoding wrapper (for the live encoder)
# --------------------------------------------------------------------------- #


class ConvEncoderWrapper(nn.Module):
    """Wraps MultiStageConvEncoder with video encoding and optimizer helpers.

    Drop-in replacement for VQClipBackbone in VQHVEBTModel.

    Parameters
    ----------
    return_stages : which stages to return.
    base_channels : stem channel width (default 64 → ~2M params).
    lr_scale      : LR multiplier for optimizer param groups.
    ema_decay     : EMA decay for the target encoder.
    """

    def __init__(
        self,
        return_stages: Iterable[str] = ("s1", "s2", "s3"),
        base_channels: int = 64,
        lr_scale: float = 1.0,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.return_stages = tuple(return_stages)
        self.lr_scale = lr_scale

        self.live = MultiStageConvEncoder(return_stages, base_channels)
        self.ema = EMAEncoder(self.live, decay=ema_decay)

    def encode_video(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode with the LIVE encoder (has gradient).

        Args:
            video: (B, T, 3, H, W) in [0, 1].

        Returns:
            {stage_name: (B, T, C, Hs, Ws)}.
        """
        B, T, _, H, W = video.shape
        flat = video.reshape(B * T, 3, H, W)
        feats_flat = self.live(flat)
        result: Dict[str, torch.Tensor] = {}
        for name, feat in feats_flat.items():
            _, C, Hs, Ws = feat.shape
            result[name] = feat.reshape(B, T, C, Hs, Ws)
        return result

    def encode_video_ema(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode with the frozen EMA encoder (no gradient, for targets)."""
        return self.ema.encode_video(video)

    @torch.no_grad()
    def update_ema(self) -> None:
        """Update the EMA encoder from the live encoder. Call after each step."""
        self.ema.update(self.live)

    def parameter_groups(self, base_lr: float) -> List[Dict]:
        """Return optimizer param groups (live encoder only; EMA is frozen)."""
        return [{"params": list(self.live.parameters()), "lr": base_lr * self.lr_scale}]
