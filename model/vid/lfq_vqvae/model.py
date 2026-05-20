"""LFQ VQ-VAE: reconstruction model with optional hierarchical video predictor."""

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
from .predictor import PredictorStage


class LFQVAE(nn.Module):
    """VQ-VAE with LFQ (bottleneck or hierarchical) and optional video predictor."""

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

        self.predictor_top: PredictorStage | None = None
        self.predictor_mid: PredictorStage | None = None
        self.predictor_bot: PredictorStage | None = None
        self.pred_to_quant_top: nn.Module = nn.Identity()
        self.pred_to_quant_mid: nn.Module = nn.Identity()
        self.pred_to_quant_bot: nn.Module = nn.Identity()

        if cfg.enable_video_predictor:
            self._build_video_predictors()

    def _build_video_predictors(self) -> None:
        cfg = self.cfg
        c_bot, c_mid, c_top = cfg.stage_channels
        k_bot, k_mid, k_top = cfg.stage_codebook_sizes
        d_bot, d_mid, d_top = cfg.stage_lfq_dims
        s_bot, s_mid, s_top = cfg.stage_sizes

        self.predictor_top = PredictorStage(
            dim=cfg.pred_dim_top,
            n_heads=cfg.pred_n_heads,
            n_layers=cfg.pred_n_layers,
            codebook_size=k_top,
            spatial_size=s_top * s_top,
            temporal_window=cfg.window_top,
            has_parent=False,
            lfq_dim=d_top,
            max_T=cfg.max_T,
            use_gumbel_softmax=cfg.use_gumbel_softmax,
        )
        self.predictor_mid = PredictorStage(
            dim=cfg.pred_dim_mid,
            n_heads=cfg.pred_n_heads,
            n_layers=cfg.pred_n_layers,
            codebook_size=k_mid,
            spatial_size=s_mid * s_mid,
            temporal_window=cfg.window_mid,
            has_parent=True,
            parent_dim=cfg.pred_dim_top,
            lfq_dim=d_mid,
            max_T=cfg.max_T,
            use_gumbel_softmax=cfg.use_gumbel_softmax,
        )
        self.predictor_bot = PredictorStage(
            dim=cfg.pred_dim_bot,
            n_heads=cfg.pred_n_heads,
            n_layers=cfg.pred_n_layers,
            codebook_size=k_bot,
            spatial_size=s_bot * s_bot,
            temporal_window=cfg.window_bot,
            has_parent=True,
            parent_dim=cfg.pred_dim_mid,
            lfq_dim=d_bot,
            max_T=cfg.max_T,
            use_gumbel_softmax=cfg.use_gumbel_softmax,
        )

        self.pred_to_quant_top = nn.Linear(cfg.pred_dim_top, c_top)
        self.pred_to_quant_mid = nn.Linear(cfg.pred_dim_mid, c_mid)
        self.pred_to_quant_bot = nn.Linear(cfg.pred_dim_bot, c_bot)

    @property
    def is_hierarchical(self) -> bool:
        return self.cfg.quantization_mode == "hierarchical"

    @property
    def has_video_predictor(self) -> bool:
        return self.predictor_top is not None and self.predictor_mid is not None and self.predictor_bot is not None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous bottleneck features (bottleneck) or quant_top (hierarchical)."""
        if self.is_hierarchical:
            return self.encoder(x)["quant_top"]
        return self.encoder.encode_features(x)

    def quantize(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            return {
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

    def _weighted_stage_ce(
        self,
        stage_ces: List[torch.Tensor],
        spatial_sizes: List[int],
    ) -> torch.Tensor:
        """Sum per-stage CE with notebook spatial weights (same as decoder prior_ce)."""
        cfg = self.cfg
        total = stage_ces[0].new_zeros(())
        for ce, spatial in zip(stage_ces, spatial_sizes):
            total = total + cfg.prior_ce_weight(spatial) * ce
        return total

    def _weighted_prior_ce(self, prior_ces: List[torch.Tensor]) -> torch.Tensor:
        ascending = list(self.cfg.spatial_sizes_ascending())
        return self._weighted_stage_ce(prior_ces, ascending[: len(prior_ces)])

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

    def encode_video(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode all frames: (B, T+1, C, H, W) -> temporal stage tensors."""
        if not self.is_hierarchical:
            raise RuntimeError("Video encoding is supported only for hierarchical mode.")
        B, Tp1, C, H, W = video.shape
        flat = video.reshape(B * Tp1, C, H, W)
        enc = self.encoder(flat)
        out = dict(enc)
        for key in ("quant_bot", "quant_mid", "quant_top"):
            c = out[key].shape[1]
            hs, ws = out[key].shape[2:]
            out[key] = out[key].reshape(B, Tp1, c, hs, ws)
        for key in ("idx_bot", "idx_mid", "idx_top"):
            hs, ws = out[key].shape[1:]
            out[key] = out[key].reshape(B, Tp1, hs, ws)
        return out

    @staticmethod
    def _flatten_stage_tokens(x: torch.Tensor) -> torch.Tensor:
        """(B, T, C, S, S) -> (B, T*S*S, C), token-major in raster order."""
        b, t, c, hs, ws = x.shape
        return x.permute(0, 1, 3, 4, 2).reshape(b, t * hs * ws, c)

    @staticmethod
    def _unflatten_stage_tokens(x: torch.Tensor, B: int, T: int, S: int, C: int) -> torch.Tensor:
        """(B, T*S*S, C) -> (B*T, C, S, S), inverse of _flatten_stage_tokens."""
        return x.reshape(B, T, S, S, C).permute(0, 1, 4, 2, 3).reshape(B * T, C, S, S)

    def predict_video(
        self,
        enc_video: Dict[str, torch.Tensor],
        T: int,
        use_gumbel: bool | None = None,
        detach_inputs: bool | None = None,
        detach_parents: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Top-down temporal prediction on frames [0..T-1] for targets [1..T]."""
        del use_gumbel  # kept for API compatibility
        if not self.has_video_predictor:
            raise RuntimeError("Video predictor is not enabled.")
        assert self.predictor_top and self.predictor_mid and self.predictor_bot
        if T > self.cfg.max_T:
            raise ValueError(f"T={T} exceeds max_T={self.cfg.max_T}")

        B = enc_video["quant_top"].shape[0]
        cfg = self.cfg
        s_bot, s_mid, s_top = cfg.stage_sizes
        temp = cfg.soft_lookup_temperature
        gumbel_tau = cfg.gumbel_tau
        if detach_inputs is None:
            detach_inputs = cfg.detach_encoder_for_predictor
        if detach_parents is None:
            detach_parents = cfg.detach_parent_features

        q_top = enc_video["quant_top"][:, :T]
        q_mid = enc_video["quant_mid"][:, :T]
        q_bot = enc_video["quant_bot"][:, :T]
        if detach_inputs:
            q_top = q_top.detach()
            q_mid = q_mid.detach()
            q_bot = q_bot.detach()
        inp_top = self._flatten_stage_tokens(q_top)
        inp_mid = self._flatten_stage_tokens(q_mid)
        inp_bot = self._flatten_stage_tokens(q_bot)

        logits_top, feat_top = self.predictor_top(inp_top, T=T, temperature=temp, gumbel_tau=gumbel_tau)
        parent_mid = feat_top.detach() if detach_parents else feat_top
        logits_mid, feat_mid = self.predictor_mid(
            inp_mid,
            parent_features=parent_mid,
            T=T,
            temperature=temp,
            gumbel_tau=gumbel_tau,
        )
        parent_bot = feat_mid.detach() if detach_parents else feat_mid
        logits_bot, feat_bot = self.predictor_bot(
            inp_bot,
            parent_features=parent_bot,
            T=T,
            temperature=temp,
            gumbel_tau=gumbel_tau,
        )
        return {
            "logits_top": logits_top,
            "logits_mid": logits_mid,
            "logits_bot": logits_bot,
            "feat_top": feat_top,
            "feat_mid": feat_mid,
            "feat_bot": feat_bot,
        }

    def decode_predicted(self, pred: Dict[str, torch.Tensor], B: int, T: int) -> torch.Tensor:
        """Decode predicted stage features into RGB for target frames [1..T]."""
        if not self.is_hierarchical:
            raise RuntimeError("Predicted decode is supported only in hierarchical mode.")
        s_bot, s_mid, s_top = self.cfg.stage_sizes
        c_bot, c_mid, c_top = self.cfg.stage_channels

        top_tokens = self.pred_to_quant_top(pred["feat_top"])
        mid_tokens = self.pred_to_quant_mid(pred["feat_mid"])
        bot_tokens = self.pred_to_quant_bot(pred["feat_bot"])

        top = self._unflatten_stage_tokens(top_tokens, B=B, T=T, S=s_top, C=c_top)
        mid = self._unflatten_stage_tokens(mid_tokens, B=B, T=T, S=s_mid, C=c_mid)
        bot = self._unflatten_stage_tokens(bot_tokens, B=B, T=T, S=s_bot, C=c_bot)
        return self.decoder.decode_from_stages(top, mid, bot)

    def _flatten_encoded_video(
        self, enc_video: Dict[str, torch.Tensor], B: int, Tp1: int
    ) -> Dict[str, torch.Tensor]:
        """Flatten encoded video dict back to per-frame tensors for decoder/loss."""
        out = dict(enc_video)
        for key in ("quant_bot", "quant_mid", "quant_top"):
            _, _, c, hs, ws = out[key].shape
            out[key] = out[key].reshape(B * Tp1, c, hs, ws)
        for key in ("idx_bot", "idx_mid", "idx_top"):
            _, _, hs, ws = out[key].shape
            out[key] = out[key].reshape(B * Tp1, hs, ws)
        return out

    def video_loss(self, video: torch.Tensor, train_mode: str = "joint") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Video training objective.

        - recon_only: image VQ-VAE objective on all frames
        - disjoint: spatially weighted predictor CE (+ optional pred MSE)
        - joint: recon objective + spatially weighted predictor CE (+ optional pred MSE)
        """
        if not self.is_hierarchical:
            raise RuntimeError("video_loss requires hierarchical mode.")
        if train_mode not in ("recon_only", "disjoint", "joint"):
            raise ValueError(f"Unsupported train_mode: {train_mode}")

        B, Tp1, C, H, W = video.shape
        T = Tp1 - 1
        all_frames = video.reshape(B * Tp1, C, H, W)
        target_frames = video[:, 1:].reshape(B * T, C, H, W)

        if train_mode == "recon_only":
            return self.loss(all_frames)

        if train_mode == "disjoint":
            with torch.no_grad():
                enc_video = self.encode_video(video)
        else:
            enc_video = self.encode_video(video)
        pred = self.predict_video(enc_video, T)

        k_bot, k_mid, k_top = self.cfg.stage_codebook_sizes
        s_bot, s_mid, s_top = self.cfg.stage_sizes
        tgt_top = enc_video["idx_top"][:, 1:].reshape(B * T * (s_top * s_top)).long()
        tgt_mid = enc_video["idx_mid"][:, 1:].reshape(B * T * (s_mid * s_mid)).long()
        tgt_bot = enc_video["idx_bot"][:, 1:].reshape(B * T * (s_bot * s_bot)).long()

        ce_top = F.cross_entropy(pred["logits_top"].reshape(-1, k_top), tgt_top)
        ce_mid = F.cross_entropy(pred["logits_mid"].reshape(-1, k_mid), tgt_mid)
        ce_bot = F.cross_entropy(pred["logits_bot"].reshape(-1, k_bot), tgt_bot)
        ce_weighted = self._weighted_stage_ce([ce_top, ce_mid, ce_bot], [s_top, s_mid, s_bot])

        if self.cfg.lambda_pred_mse > 0:
            pred_rgb = self.decode_predicted(pred, B, T)
            pred_mse = self._recon_loss(pred_rgb, target_frames)
        else:
            pred_mse = ce_weighted.new_zeros(())

        total = ce_weighted * self.cfg.lambda_ce + pred_mse * self.cfg.lambda_pred_mse
        metrics: Dict[str, torch.Tensor] = {
            "loss": total,
            "ce_top": ce_top,
            "ce_mid": ce_mid,
            "ce_bot": ce_bot,
            "ce": ce_weighted,
            "pred_mse": pred_mse,
        }

        if train_mode == "joint":
            enc_flat = self._flatten_encoded_video(enc_video, B=B, Tp1=Tp1)
            x_hat, prior_ces = self.decoder(enc_flat)
            recon = self._recon_loss(x_hat, all_frames)
            vq = enc_flat["vq_loss"]
            recon_total = recon + self.cfg.vq_loss_weight * vq
            recon_metrics: Dict[str, Union[torch.Tensor, float]] = {
                "recon": recon,
                "vq": vq,
                "vq_bot": enc_flat["vq_loss_bot"],
                "vq_mid": enc_flat["vq_loss_mid"],
                "vq_top": enc_flat["vq_loss_top"],
            }
            prior_ce = self._weighted_prior_ce(prior_ces)
            recon_total = recon_total + self.cfg.lambda_prior_ce * prior_ce
            recon_metrics["prior_ce"] = prior_ce
            if self.cfg.fusion == "gamma":
                gammas = self.decoder.gamma_values()
                if gammas:
                    gamma_abs = sum(abs(g) for g in gammas) / len(gammas)
                    recon_metrics["gamma_abs"] = gamma_abs
                    if self.cfg.gamma_l2 > 0:
                        gamma_penalty = sum(g * g for g in gammas) / len(gammas)
                        recon_total = recon_total + vq.new_tensor(gamma_penalty * self.cfg.gamma_l2)
            total = total + recon_total
            metrics["loss"] = total
            for key, value in recon_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = value
                else:
                    metrics[key] = total.new_tensor(value)
        return total, metrics

    def encoder_params(self) -> List[nn.Parameter]:
        return list(self.encoder.parameters())

    def decoder_prior_params(self) -> List[nn.Parameter]:
        if not self.is_hierarchical:
            return list(self.decoder.parameters())
        return (
            list(self.decoder.stem.parameters())
            + list(self.decoder.block_mid.parameters())
            + list(self.decoder.block_bot.parameters())
        )

    def decoder_image_params(self) -> List[nn.Parameter]:
        if not self.is_hierarchical:
            return list(self.decoder.parameters())
        return list(self.decoder.up_to_image.parameters()) + list(self.decoder.head.parameters())

    def vqvae_params(self) -> List[nn.Parameter]:
        """Parameters for encoder/decoder VQ-VAE stack."""
        return self.encoder_params() + list(self.decoder.parameters())

    def predictor_params(self) -> List[nn.Parameter]:
        """Parameters for predictor stack only."""
        if not self.has_video_predictor:
            return []
        assert self.predictor_top and self.predictor_mid and self.predictor_bot
        return (
            list(self.predictor_top.parameters())
            + list(self.predictor_mid.parameters())
            + list(self.predictor_bot.parameters())
            + list(self.pred_to_quant_top.parameters())
            + list(self.pred_to_quant_mid.parameters())
            + list(self.pred_to_quant_bot.parameters())
        )

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        enc = self.encoder(x)
        if self.is_hierarchical:
            return {"idx_bot": enc["idx_bot"], "idx_mid": enc["idx_mid"], "idx_top": enc["idx_top"]}
        return enc["indices"]

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if self.is_hierarchical:
            raise RuntimeError("Hierarchical decode_indices not supported; use forward from codes.")
        codes = self.encoder.vq.indices_to_codes(indices)
        quant_feat = self.encoder.top_from_vq(codes)
        return self.decoder(quant_feat)
