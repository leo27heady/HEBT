"""LFQ VQ-VAE: bottleneck or hierarchical multi-stage quantization."""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LFQVAEConfig
from .decoder import LFQDecoder
from .encoder import LFQEncoder
from .hierarchical_decoder import LFQHierarchicalDecoder
from .hierarchical_encoder import LFQHierarchicalEncoder


class LFQVAE(nn.Module):
    """VQ-VAE with LFQ (bottleneck or hierarchical)."""

    def __init__(self, cfg: LFQVAEConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        if cfg.quantization_mode == "bottleneck":
            self.encoder = LFQEncoder(cfg)
            self.decoder = LFQDecoder(cfg)
        else:
            self.encoder = LFQHierarchicalEncoder(cfg)
            self.decoder = LFQHierarchicalDecoder(cfg)

    @property
    def is_hierarchical(self) -> bool:
        return self.cfg.quantization_mode == "hierarchical"

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous bottleneck features (bottleneck) or quant_top (hierarchical)."""
        if self.is_hierarchical:
            return self.encoder(x)["quant_top"]
        return self.encoder.encode_features(x)

    def quantize(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_hierarchical:
            raise RuntimeError("Use forward() for hierarchical mode; multi-stage VQ is internal.")
        z = self.encoder.top_to_vq(feat)
        z = self.encoder.top_pre_vq_norm(z)
        quant, indices, vq_loss = self.encoder.vq(z)
        quant_feat = self.encoder.top_from_vq(quant)
        return quant_feat, indices, vq_loss

    def decode(self, quant_feat: torch.Tensor) -> torch.Tensor:
        if self.is_hierarchical:
            raise RuntimeError("Hierarchical decode requires full encoder dict; use forward().")
        return self.decoder(quant_feat)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x)
        if self.is_hierarchical:
            x_hat, prior_ces = self.decoder(enc)
            out: Dict[str, torch.Tensor] = {
                "x_hat": x_hat,
                "indices": enc["indices"],
                "idx_bot": enc["idx_bot"],
                "idx_mid": enc["idx_mid"],
                "idx_top": enc["idx_top"],
                "vq_loss": enc["vq_loss"],
                "vq_loss_bot": enc["vq_loss_bot"],
                "vq_loss_mid": enc["vq_loss_mid"],
                "vq_loss_top": enc["vq_loss_top"],
                "quant_feat": enc["quant_top"],
                "quant_bot": enc["quant_bot"],
                "quant_mid": enc["quant_mid"],
                "quant_top": enc["quant_top"],
                "prior_ces": prior_ces,
            }
            return out

        x_hat = self.decoder(enc["quant_feat"])
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

    def _weighted_prior_ce(self, prior_ces: List[torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        ascending = cfg.spatial_sizes_ascending()
        total = prior_ces[0].new_zeros(())
        for ce, spatial in zip(prior_ces, ascending[: len(prior_ces)]):
            w = cfg.prior_ce_weight(spatial)
            total = total + w * ce
        return total

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(x)
        recon = self._recon_loss(out["x_hat"], x)
        vq = out["vq_loss"]
        total = recon + self.cfg.vq_loss_weight * vq

        metrics: Dict[str, Union[torch.Tensor, float]] = {
            "loss": total,
            "recon": recon,
            "vq": vq,
        }

        if self.is_hierarchical:
            prior_ce = self._weighted_prior_ce(out["prior_ces"])
            total = total + self.cfg.lambda_prior_ce * prior_ce
            metrics["loss"] = total
            metrics["prior_ce"] = prior_ce
            metrics["vq_bot"] = out["vq_loss_bot"]
            metrics["vq_mid"] = out["vq_loss_mid"]
            metrics["vq_top"] = out["vq_loss_top"]
            if self.cfg.fusion == "gamma":
                gammas = self.decoder.gamma_values()
                if gammas:
                    gamma_abs = sum(abs(g) for g in gammas) / len(gammas)
                    metrics["gamma_abs"] = gamma_abs
                    if self.cfg.gamma_l2 > 0:
                        gamma_penalty = sum(g * g for g in gammas) / len(gammas)
                        total = total + vq.new_tensor(gamma_penalty * self.cfg.gamma_l2)

        metrics["loss"] = total
        return total, metrics  # type: ignore[return-value]

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        enc = self.encoder(x)
        if self.is_hierarchical:
            return {
                "idx_bot": enc["idx_bot"],
                "idx_mid": enc["idx_mid"],
                "idx_top": enc["idx_top"],
            }
        return enc["indices"]

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if self.is_hierarchical:
            raise RuntimeError("Hierarchical decode_indices not supported; use forward from codes.")
        codes = self.encoder.vq.indices_to_codes(indices)
        quant_feat = self.encoder.top_from_vq(codes)
        return self.decoder(quant_feat)
