"""Full Fresh HVQVAE model assembly."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FreshHVQVAEConfig
from .encoder import HierarchicalEncoder
from .decoder import Decoder, UpscaleMid, UpscaleTop
from .predictor import PredictorStage
from .soft_lookup import build_lfq_codebook_matrix


class FreshHVQVAE(nn.Module):
    """
    Fresh Hierarchical VQ-VAE for video prediction.
    3-stage hierarchy: bot (16x16), mid (4x4), top (1x1).
    Per-stage reconstruction + isolated per-stage prediction.
    """

    def __init__(self, cfg: FreshHVQVAEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder
        self.encoder = HierarchicalEncoder(cfg)

        # Decoder + Upscalers
        self.decoder = Decoder(cfg.C_bot)
        self.upscale_mid = UpscaleMid(cfg.C_mid, cfg.C_bot)
        self.upscale_top = UpscaleTop(cfg.C_top, cfg.C_bot)

        # Predictor stages
        self.predictor_top = PredictorStage(
            dim=cfg.pred_dim_top, n_heads=cfg.pred_n_heads, n_layers=cfg.pred_n_layers,
            codebook_size=cfg.K_top, spatial_size=1,
            temporal_window=cfg.window_top, has_parent=False,
            lfq_dim=cfg.lfq_dim_top, max_T=cfg.max_T,
        )
        self.predictor_mid = PredictorStage(
            dim=cfg.pred_dim_mid, n_heads=cfg.pred_n_heads, n_layers=cfg.pred_n_layers,
            codebook_size=cfg.K_mid, spatial_size=16,
            temporal_window=cfg.window_mid, has_parent=True,
            parent_dim=cfg.pred_dim_top, lfq_dim=cfg.lfq_dim_mid, max_T=cfg.max_T,
        )
        self.predictor_bot = PredictorStage(
            dim=cfg.pred_dim_bot, n_heads=cfg.pred_n_heads, n_layers=cfg.pred_n_layers,
            codebook_size=cfg.K_bot, spatial_size=256,
            temporal_window=cfg.window_bot, has_parent=True,
            parent_dim=cfg.pred_dim_mid, lfq_dim=cfg.lfq_dim_bot, max_T=cfg.max_T,
        )


    def encode(self, video: torch.Tensor) -> dict:
        """
        Encode all frames.
        video: (B, T+1, 3, H, W)
        Returns dict with temporal-reshaped features and indices.
        """
        B, Tp1 = video.shape[:2]
        flat = video.reshape(B * Tp1, 3, video.shape[3], video.shape[4])
        enc = self.encoder(flat)

        # Reshape to temporal: (B, T+1, ...)
        for key in ['quant_bot', 'quant_mid', 'quant_top']:
            C = enc[key].shape[1]
            spatial = enc[key].shape[2:]
            enc[key] = enc[key].reshape(B, Tp1, C, *spatial)
        for key in ['idx_bot', 'idx_mid', 'idx_top']:
            spatial = enc[key].shape[1:]
            enc[key] = enc[key].reshape(B, Tp1, *spatial)

        return enc

    def reconstruct(self, enc: dict, B: int, Tp1: int):
        """
        Reconstruct from all 3 stages (for per-stage MSE).
        Returns 3 reconstructions, all (B*(T+1), 3, 64, 64).
        """
        cfg = self.cfg

        # Bot: direct to decoder
        quant_bot = enc['quant_bot'].reshape(B * Tp1, cfg.C_bot, 16, 16)
        recon_bot = self.decoder(quant_bot)

        # Mid: upscale then decode
        quant_mid = enc['quant_mid'].reshape(B * Tp1, cfg.C_mid, 4, 4)
        recon_mid = self.decoder(self.upscale_mid(quant_mid))

        # Top: upscale then decode
        quant_top = enc['quant_top'].reshape(B * Tp1, cfg.C_top, 1, 1)
        recon_top = self.decoder(self.upscale_top(quant_top))

        return recon_bot, recon_mid, recon_top

    def predict(self, enc: dict, T: int) -> dict:
        """
        Run top-down predictor on first T frames (encoder outputs DETACHED).
        All parent features are detached for stage isolation.
        """
        B = enc['quant_top'].shape[0]
        cfg = self.cfg

        inp_top = enc['quant_top'][:, :T].detach().reshape(B, T * 1, -1)
        inp_mid = enc['quant_mid'][:, :T].detach().reshape(B, T * 16, -1)
        inp_bot = enc['quant_bot'][:, :T].detach().reshape(B, T * 256, -1)

        logits_top, feat_top = self.predictor_top(inp_top, T=T,
                                                   temperature=cfg.soft_lookup_temperature)
        logits_mid, feat_mid = self.predictor_mid(
            inp_mid, parent_features=feat_top.detach(), T=T,
            temperature=cfg.soft_lookup_temperature,
        )
        logits_bot, feat_bot = self.predictor_bot(
            inp_bot, parent_features=feat_mid.detach(), T=T,
            temperature=cfg.soft_lookup_temperature,
        )

        return {
            'logits_top': logits_top, 'logits_mid': logits_mid, 'logits_bot': logits_bot,
            'feat_top': feat_top, 'feat_mid': feat_mid, 'feat_bot': feat_bot,
        }

    def forward(self, video: torch.Tensor) -> dict:
        """
        Full training forward: compute all losses.
        video: (B, T+1, 3, 64, 64)
        Returns dict of individual losses (not summed — caller handles backward).
        """
        B, Tp1 = video.shape[:2]
        T = Tp1 - 1
        all_frames = video.reshape(B * Tp1, 3, video.shape[3], video.shape[4])

        # Encode WITH gradient (for reconstruction)
        enc = self.encode(video)

        # Per-stage reconstruction
        recon_bot, recon_mid, recon_top = self.reconstruct(enc, B, Tp1)
        mse_bot = F.mse_loss(recon_bot, all_frames)
        mse_mid = F.mse_loss(recon_mid, all_frames)
        mse_top = F.mse_loss(recon_top, all_frames)
        vq_loss = enc['loss_bot'] + enc['loss_mid'] + enc['loss_top']

        # Prediction CE (detached from encoder)
        pred = self.predict(enc, T)
        tgt_top = enc['idx_top'][:, 1:].detach().reshape(B * T * 1)
        tgt_mid = enc['idx_mid'][:, 1:].detach().reshape(B * T * 16)
        tgt_bot = enc['idx_bot'][:, 1:].detach().reshape(B * T * 256)

        ce_top = F.cross_entropy(
            pred['logits_top'].reshape(-1, self.cfg.K_top), tgt_top
        )
        ce_mid = F.cross_entropy(
            pred['logits_mid'].reshape(-1, self.cfg.K_mid), tgt_mid
        )
        ce_bot = F.cross_entropy(
            pred['logits_bot'].reshape(-1, self.cfg.K_bot), tgt_bot
        )

        return {
            'mse_bot': mse_bot, 'mse_mid': mse_mid, 'mse_top': mse_top,
            'vq_loss': vq_loss,
            'ce_top': ce_top, 'ce_mid': ce_mid, 'ce_bot': ce_bot,
        }

    def get_encoder_decoder_params(self):
        """Parameters updated by reconstruction loss."""
        return (
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.upscale_mid.parameters()) +
            list(self.upscale_top.parameters())
        )

    def get_predictor_top_params(self):
        return list(self.predictor_top.parameters())

    def get_predictor_mid_params(self):
        return list(self.predictor_mid.parameters())

    def get_predictor_bot_params(self):
        return list(self.predictor_bot.parameters())
