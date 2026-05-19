"""LFQ VQ-VAE: encode, quantize at 1x1 bottleneck, decode (reconstruction only)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LFQVAEConfig
from .decoder import LFQDecoder
from .encoder import LFQEncoder


class LFQVAE(nn.Module):
    """Simple VQ-VAE with LFQ at the final spatial stage only."""

    def __init__(self, cfg: LFQVAEConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.encoder = LFQEncoder(cfg)
        self.decoder = LFQDecoder(cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous bottleneck features before VQ (B, C_top, 1, 1)."""
        return self.encoder.encode_features(x)

    def quantize(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize bottleneck features. Returns (quant_feat, indices, vq_loss)."""
        z = self.encoder.top_to_vq(feat)
        z = self.encoder.top_pre_vq_norm(z)
        quant, indices, vq_loss = self.encoder.vq(z)
        quant_feat = self.encoder.top_from_vq(quant)
        return quant_feat, indices, vq_loss

    def decode(self, quant_feat: torch.Tensor) -> torch.Tensor:
        """(B, C_top, 1, 1) -> (B, 3, H, W) in [0, 1]."""
        return self.decoder(quant_feat)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x)
        x_hat = self.decode(enc["quant_feat"])
        return {
            "x_hat": x_hat,
            "indices": enc["indices"],
            "vq_loss": enc["vq_loss"],
            "quant_feat": enc["quant_feat"],
            "feat_pre_vq": enc["feat_pre_vq"],
        }

    def _recon_loss(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.recon_loss == "l1":
            return F.l1_loss(x_hat, x)
        return F.mse_loss(x_hat, x)

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(x)
        recon = self._recon_loss(out["x_hat"], x)
        vq = out["vq_loss"]
        total = recon + self.cfg.vq_loss_weight * vq
        return total, {
            "loss": total,
            "recon": recon,
            "vq": vq,
        }

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        return enc["indices"]

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from discrete indices at the 1x1 bottleneck."""
        codes = self.encoder.vq.indices_to_codes(indices)
        quant_feat = self.encoder.top_from_vq(codes)
        return self.decode(quant_feat)
