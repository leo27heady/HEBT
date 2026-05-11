"""
HVQVAE: Hierarchical VQ-VAE for video frame prediction.

Top-level model that owns:
  - Multi-stage convolutional encoder.
  - Per-stage VQ codebooks (gradient-trained, reference-style).
  - Per-stage transformer predictors (direct logit prediction, no MCMC).
  - Pixel decoder on the finest stage.

Forward pass:
  1. Encode all T+1 frames → multi-scale features.
  2. Quantize at each stage → embedding loss per stage + straight-through z_q.
  3. Predict coarse→fine: each predictor takes context z_q, produces soft
     pred_embed via softmax(logits) @ codebook. Parent pred_embed is passed
     as cross-attention context to finer stages (gradient flows through by
     default; configurable via detach_parent_kv).
  4. Decode finest-stage pred_embed → predicted future RGB.
  5. (Optional) Decode finest-stage z_q → reconstructed RGB (autoencoder path).
  6. Loss = pred_loss + sum(embedding_losses) [+ recon_loss if enabled].

Loss structure:
  - pred_loss:       MSE(decode(pred_s1_future), future_frames) / data_var
  - embedding_loss:  Σ_stage [||z_q.detach()-z_e||² + β*||z_q-z_e.detach()||²]
  - recon_loss:      MSE(decode(z_q_s1_all_frames), video) / data_var  [optional]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vid.hvqvae.config import HVQVAEConfig, HVQVAEStageConfig
from model.vid.hvqvae.encoder import MultiStageEncoder
from model.vid.hvqvae.decoder import Decoder
from model.vid.hvqvae.quantizer import VectorQuantizer
from model.vid.hvqvae.predictor import StagePredictor


# --------------------------------------------------------------------------- #
#  Output container
# --------------------------------------------------------------------------- #


@dataclass
class HVQVAEOutput:
    """Full model output from one forward pass."""
    total_loss: torch.Tensor
    recon_loss: Optional[torch.Tensor]       # None if recon disabled
    pred_loss: torch.Tensor
    embedding_loss: torch.Tensor
    perplexities: Dict[str, float]
    embedding_losses_per_stage: Dict[str, float]
    metrics: Dict[str, float]                # flat dict for logging
    recon_rgb: Optional[torch.Tensor] = None   # (B, T+1, 3, H, W)
    pred_rgb: Optional[torch.Tensor] = None    # (B, T, 3, H, W)


# --------------------------------------------------------------------------- #
#  HVQVAEModel
# --------------------------------------------------------------------------- #


class HVQVAEModel(nn.Module):
    """Hierarchical VQ-VAE for video prediction.

    Parameters
    ----------
    cfg : HVQVAEConfig — full model configuration.
    """

    def __init__(self, cfg: HVQVAEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder
        stage_channels = {s.stage_name: s.channels for s in cfg.stages}
        self.encoder = MultiStageEncoder(
            h_dim=cfg.encoder_h_dim,
            res_h_dim=cfg.encoder_res_h_dim,
            n_res_layers=cfg.encoder_n_res_layers,
            stage_channels=stage_channels,
        )

        # Per-stage quantizers
        self.quantizers = nn.ModuleDict()
        for s in cfg.stages:
            self.quantizers[s.stage_name] = VectorQuantizer(
                n_e=s.num_codes,
                e_dim=s.channels,
                beta=cfg.beta,
            )

        # Per-stage predictors (coarsest first, cross-attn from parent)
        self.predictors = nn.ModuleDict()
        for idx, s in enumerate(cfg.stages):
            parent_cfg = cfg.stages[idx - 1] if idx > 0 else None
            self.predictors[s.stage_name] = StagePredictor(
                cfg=s,
                parent_cfg=parent_cfg,
            )

        # Decoder (operates on finest stage = last in list)
        finest = cfg.stages[-1]
        self.decoder = Decoder(
            in_dim=finest.channels,
            h_dim=cfg.encoder_h_dim,
            n_res_layers=cfg.encoder_n_res_layers,
            res_h_dim=cfg.encoder_res_h_dim,
        )

        # Data variance for loss normalization (set via set_data_variance)
        self.register_buffer("data_variance", torch.tensor(1.0))

    def set_data_variance(self, variance: float) -> None:
        """Set data variance for recon loss normalization (as in reference)."""
        self.data_variance.fill_(variance)

    # ------------------------------------------------------------------ #
    #  Encoding helpers
    # ------------------------------------------------------------------ #

    def _encode_video(
        self, video: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode all frames through the shared encoder.

        Args:
            video: (B, T, 3, H, W) video frames.

        Returns:
            {stage_name: (B, T, C, Hs, Ws)} encoder features per stage.
        """
        B, T, _, H, W = video.shape
        flat = video.reshape(B * T, 3, H, W)
        feats_flat = self.encoder(flat)  # {name: (B*T, C, Hs, Ws)}

        result: Dict[str, torch.Tensor] = {}
        for name, feat in feats_flat.items():
            _, C, Hs, Ws = feat.shape
            result[name] = feat.reshape(B, T, C, Hs, Ws)
        return result

    # ------------------------------------------------------------------ #
    #  Forward (training)
    # ------------------------------------------------------------------ #

    def forward(
        self, video: torch.Tensor,
    ) -> HVQVAEOutput:
        """Compute training loss for a video batch.

        Args:
            video: (B, T+1, 3, H, W) video clip in [0, 1].
                   Frames 0..T-1 are context, frame T is the future target.

        Returns:
            HVQVAEOutput with losses and metrics.
        """
        B, T1, _, H_in, W_in = video.shape
        T = T1 - 1
        if T < 1:
            raise ValueError(f"Video must have at least 2 frames, got {T1}")

        # ---- 1. Encode all T+1 frames at all stages ----
        features = self._encode_video(video)  # {name: (B, T+1, C, Hs, Ws)}

        # ---- 2. Quantize at each stage ----
        total_embedding_loss = torch.tensor(0.0, device=video.device)
        quantized: Dict[str, torch.Tensor] = {}        # z_q straight-through
        perplexities: Dict[str, float] = {}
        embed_losses_per_stage: Dict[str, float] = {}

        for s in self.cfg.stages:
            name = s.stage_name
            z_e = features[name]  # (B, T+1, C, Hs, Ws)
            B_s, T1_s, C_s, Hs, Ws = z_e.shape

            # Flatten time into batch for quantizer (expects 4D conv format)
            z_e_flat = z_e.reshape(B_s * T1_s, C_s, Hs, Ws)
            vq_loss, z_q_flat, perplexity, _, _ = self.quantizers[name](z_e_flat)

            total_embedding_loss = total_embedding_loss + vq_loss
            quantized[name] = z_q_flat.reshape(B_s, T1_s, C_s, Hs, Ws)
            perplexities[name] = perplexity.item()
            embed_losses_per_stage[name] = vq_loss.item()

        # ---- 3. Optional autoencoder reconstruction (all frames, finest stage) ----
        finest_name = self.cfg.stages[-1].stage_name
        recon_loss: Optional[torch.Tensor] = None
        recon_rgb: Optional[torch.Tensor] = None

        if self.cfg.use_recon_loss:
            z_q_finest = quantized[finest_name]  # (B, T+1, C, Hs, Ws)
            recon_flat = self.decoder(
                z_q_finest.reshape(B * T1, -1, *z_q_finest.shape[3:])
            )
            recon_rgb = recon_flat.reshape(B, T1, 3, *recon_flat.shape[2:])

            gt_all = video
            if recon_rgb.shape[-2:] != gt_all.shape[-2:]:
                gt_flat = gt_all.reshape(B * T1, 3, H_in, W_in)
                gt_flat = F.interpolate(
                    gt_flat, size=recon_rgb.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
                gt_all = gt_flat.reshape(B, T1, 3, *recon_rgb.shape[-2:])

            recon_loss = F.mse_loss(recon_rgb, gt_all) / self.data_variance

        # ---- 4. Prediction: coarse → fine ----
        parent_pred: Optional[torch.Tensor] = None

        for idx, s in enumerate(self.cfg.stages):
            name = s.stage_name
            z_q_ctx = quantized[name][:, :T]  # (B, T, C, Hs, Ws) context frames

            predictor = self.predictors[name]
            logits, pred_embed = predictor(
                context=z_q_ctx,
                # Detach codebook weight so pred_loss gradient does NOT pull
                # on codebook entries. Codebook trains via VQ embedding loss
                # only (same as reference VQ-VAE).
                codebook_weight=self.quantizers[name].embedding.weight.detach(),
                parent_context=parent_pred,
            )

            # Configurable: detach parent context or let gradient flow through
            if self.cfg.detach_parent_kv:
                parent_pred = pred_embed.detach()
            else:
                parent_pred = pred_embed

        # pred_embed from finest stage: (B, T, C, Hs, Ws)
        # Decode to pixels
        pred_flat = self.decoder(
            pred_embed.reshape(B * T, -1, *pred_embed.shape[3:])
        )  # (B*T, 3, H_out, W_out)
        pred_rgb = pred_flat.reshape(B, T, 3, *pred_flat.shape[2:])

        # GT future frames
        gt_future = video[:, 1:]
        if pred_rgb.shape[-2:] != gt_future.shape[-2:]:
            gt_f_flat = gt_future.reshape(B * T, 3, H_in, W_in)
            gt_f_flat = F.interpolate(
                gt_f_flat, size=pred_rgb.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            gt_future = gt_f_flat.reshape(B, T, 3, *pred_rgb.shape[-2:])

        pred_loss = F.mse_loss(pred_rgb, gt_future) / self.data_variance

        # ---- 5. Total loss ----
        total_loss = pred_loss + total_embedding_loss
        if recon_loss is not None:
            total_loss = total_loss + recon_loss

        # ---- 6. Build metrics dict ----
        metrics: Dict[str, float] = {
            "pred_loss": pred_loss.item(),
            "embedding_loss": total_embedding_loss.item(),
        }
        if recon_loss is not None:
            metrics["recon_loss"] = recon_loss.item()
        for name in embed_losses_per_stage:
            metrics[f"{name}/embed_loss"] = embed_losses_per_stage[name]
            metrics[f"{name}/perplexity"] = perplexities[name]

        return HVQVAEOutput(
            total_loss=total_loss,
            recon_loss=recon_loss,
            pred_loss=pred_loss,
            embedding_loss=total_embedding_loss,
            perplexities=perplexities,
            embedding_losses_per_stage=embed_losses_per_stage,
            metrics=metrics,
            recon_rgb=recon_rgb,
            pred_rgb=pred_rgb,
        )
