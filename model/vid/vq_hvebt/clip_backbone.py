"""
Trainable MobileCLIP multi-stage feature extractor for VQ-HVEBT.

Unlike the frozen encoder used in the original HVEBT, this wrapper allows
gradient flow through the CLIP visual trunk so the encoder can adapt jointly
with the per-stage vector quantizers and EBT predictors.

Key design choices
------------------
1. `trainable=True` is the default. When True:
   - Parameters have requires_grad=True.
   - The model still runs in EVAL mode (via the existing train() override
     in MobileClipMultiStageEncoder). This keeps BatchNorm running-stats
     frozen so the features are stable during training. Dropout (if any
     in the trunk) is also disabled in eval mode.
   - This is the standard practice for fine-tuning vision transformers.

2. `encode_video(video)` flattens the temporal dimension into batch,
   runs CLIP, and unfolds it back. This is memory-efficient compared to
   passing all frames simultaneously as separate inputs.

3. The LR multiplier `lr_scale` is used by the VQHVEBTModel when building
   optimizer parameter groups so the encoder trains at a slower rate than
   the codebooks and predictors.

Stage output shapes (256x256 input):
    s1 : (B, 128, 32, 32)
    s2 : (B, 256, 16, 16)
    s3 : (B, 512,  8,  8)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch import nn

from model.vid.hvebt.clip_encoder import MobileClipMultiStageEncoder


class VQClipBackbone(nn.Module):
    """Trainable CLIP backbone returning per-stage spatial feature maps.

    Parameters
    ----------
    weights_path : str
        Path to MobileCLIP2-S0 pretrained checkpoint.
    return_stages : iterable of str
        Which CLIP stages to return (e.g. ("s1", "s2", "s3")).
    trainable : bool
        If False, the encoder is frozen (no gradient). If True (default),
        parameters are learnable and the model runs in eval BN mode.
    lr_scale : float
        Relative LR multiplier for optimizer parameter groups. This is
        metadata only; callers must read it when building the optimizer.
    """

    def __init__(
        self,
        weights_path: str = "clip/MobileCLIP2-S0/mobileclip2_s0.pt",
        return_stages: Iterable[str] = ("s1", "s2", "s3"),
        trainable: bool = True,
        lr_scale: float = 0.1,
    ):
        super().__init__()
        self.return_stages = tuple(return_stages)
        self.lr_scale = lr_scale
        self._trainable = trainable

        self._encoder = MobileClipMultiStageEncoder(
            weights_path=weights_path,
            return_stages=self.return_stages,
            normalize_features=True,
            trainable=trainable,
        )

    # ------------------------------------------------------------------ #
    #  Single-frame encoding
    # ------------------------------------------------------------------ #

    def encode_frame(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode a single batch of frames.

        Args:
            x: (B, 3, H, W) in [0, 1] RGB.

        Returns:
            dict mapping stage name → (B, C, Hs, Ws) feature map.
        """
        return self._encoder(x)

    # ------------------------------------------------------------------ #
    #  Video encoding (T frames per clip)
    # ------------------------------------------------------------------ #

    def encode_video(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode a batch of video clips.

        The temporal dimension is folded into the batch dimension for the
        CLIP forward pass, then unfolded.

        Args:
            video: (B, T, 3, H, W) in [0, 1] RGB.

        Returns:
            dict mapping stage name → (B, T, C, Hs, Ws) feature tensor.
        """
        B, T, _, H, W = video.shape
        # Fold T into B: (B*T, 3, H, W)
        flat = video.reshape(B * T, 3, H, W)
        feats_flat = self._encoder(flat)   # {stage: (B*T, C, Hs, Ws)}

        # Unfold back: (B, T, C, Hs, Ws)
        result: Dict[str, torch.Tensor] = {}
        for name, feat in feats_flat.items():
            _, C, Hs, Ws = feat.shape
            result[name] = feat.reshape(B, T, C, Hs, Ws)
        return result

    # ------------------------------------------------------------------ #
    #  Optimizer parameter group helper
    # ------------------------------------------------------------------ #

    def parameter_groups(self, base_lr: float) -> List[Dict]:
        """Return optimizer parameter group dicts for the encoder.

        Args:
            base_lr: the base learning rate used for non-encoder params.

        Returns:
            List with one dict: {"params": [...], "lr": base_lr * lr_scale}.
        """
        return [{"params": list(self.parameters()), "lr": base_lr * self.lr_scale}]
