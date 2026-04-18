"""
MobileCLIP2-S0 multi-stage feature extractor for Hierarchical Video EBT.

Exposes per-stage spatial feature maps from the frozen FastVit visual trunk.
Used as conditioning source and as prediction target (per-stage, detached).

For 256x256 input the produced feature maps are:
    stem   : (B,   64, 64, 64)
    s0     : (B,   64, 64, 64)
    s1     : (B,  128, 32, 32)
    s2     : (B,  256, 16, 16)
    s3     : (B,  512,  8,  8)
    final  : (B, 1024,  8,  8)   # final_conv output
    pooled : (B, 512)            # trunk head fc output
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import nn


# Default OpenAI CLIP normalization (MobileCLIP uses same stats).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class MobileClipMultiStageEncoder(nn.Module):
    """
    Wraps open_clip MobileCLIP2-S0 visual trunk. All parameters are frozen and
    the module is locked in eval mode. Input is expected in [0, 1] RGB
    (unnormalized); CLIP normalization is applied internally.
    """

    ALL_STAGES: tuple[str, ...] = ("stem", "s0", "s1", "s2", "s3", "final", "pooled")

    def __init__(
        self,
        weights_path: str = "clip/MobileCLIP2-S0/mobileclip2_s0.pt",
        model_name: str = "MobileCLIP2-S0",
        return_stages: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        import open_clip  # lazy

        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=weights_path
        )
        self.visual = model.visual  # TimmModel
        self.return_stages = tuple(return_stages) if return_stages else self.ALL_STAGES
        for s in self.return_stages:
            if s not in self.ALL_STAGES:
                raise ValueError(f"Unknown stage '{s}'. Valid: {self.ALL_STAGES}")

        self.register_buffer(
            "clip_mean", torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "clip_std", torch.tensor(_CLIP_STD).view(1, 3, 1, 1), persistent=False
        )

        # Freeze everything.
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    # Keep encoder permanently in eval mode (disables dropout + BN running-stats update).
    def train(self, mode: bool = True):  # type: ignore[override]
        return super().train(False)

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.clip_mean) / self.clip_std

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) in [0, 1] RGB.
        Returns:
            dict mapping stage name -> feature tensor.
        """
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected (B,3,H,W), got {tuple(x.shape)}")
        x = self._apply_norm(x)

        trunk = self.visual.trunk
        feats: Dict[str, torch.Tensor] = {}

        h = trunk.stem(x)
        if "stem" in self.return_stages:
            feats["stem"] = h

        for i, stage in enumerate(trunk.stages):
            h = stage(h)
            name = f"s{i}"
            if name in self.return_stages:
                feats[name] = h

        h = trunk.final_conv(h)
        if "final" in self.return_stages:
            feats["final"] = h

        if "pooled" in self.return_stages:
            # trunk.head = ClassifierHead(global_pool + drop + fc + flatten=Identity)
            # Bypass flatten (which is Identity anyway) and call the sub-ops manually
            # to keep shape predictable across timm versions.
            head = trunk.head
            pooled = head.global_pool(h)  # (B, 1024)
            pooled = head.drop(pooled)
            pooled = head.fc(pooled)  # (B, 512)
            feats["pooled"] = pooled

        return feats

    def encode_video(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, 3, H, W) in [0, 1].
        Returns:
            dict mapping stage -> (B, T, C, Hs, Ws) or (B, T, C) for 'pooled'.
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
