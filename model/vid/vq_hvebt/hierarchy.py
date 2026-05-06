"""
VQ-HVEBT: top-level hierarchical model.

This module owns:
  - The trainable CLIP encoder (VQClipBackbone).
  - One VectorQuantizer per stage.
  - One VQHVEBTStage (EBT predictor) per stage.
  - An optional PixelDecoder on the finest stage.

It orchestrates the complete forward pass for training and inference.

Forward pass overview
---------------------

Given a video clip of shape (B, T+1, 3, H, W):
  1. Encode ALL T+1 frames through CLIP → per-stage feature tensors.
  2. For each stage k, quantize all frames:
         z_e^k, z_q^k, z_q_st^k, indices^k
     and collect per-stage VQ losses (cb_loss^k, commit_loss^k).
  3. Build EBT training pairs (shifted by 1):
         real_ctx^k   = z_q_st^k[:, :-1]   (frames 0..T-1, with grad)
         target^k     = z_q^k[:, 1:]       (frames 1..T,   DETACHED)
  4. Predict from coarsest stage to finest (top-down):
     For each stage k (ordered coarse→fine):
       a. Run MCMC from random/zero init in logit space.
       b. Get final predicted embedding z_pred^k.
       c. Compute prediction loss: mse(z_pred^k, target^k).
       d. Pass (detached) z_pred^k as parent_context to the next finer stage.
  5. Optional: decode finest-stage prediction to RGB and compute pixel loss.
  6. Return total loss and a metric dict.

Gradient flow summary
---------------------

  encoder ←── commitment_loss^k  (for each stage)
  encoder ←── straight-through from downstream loss via z_q_st
  codebook ←── codebook_loss^k  (for each stage, z_e is detached)
  codebook ←── prediction_loss (via softmax @ E, unless detach_codebook_in_decode)
  predictor ←── prediction_loss (via MCMC unroll with create_graph=True)
  decoder ←── decoder_loss (only if use_decoder=True; does NOT flow to predictor)

Target leakage prevention
--------------------------
  target^k = z_q^k[:, 1:].detach().clone()
  Both `.detach()` AND `.clone()` are required:
    - `.detach()` stops gradients crossing the target branch.
    - `.clone()` makes a fresh tensor so in-place ops on z_q don't corrupt it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vid.hvebt.decoder import PixelDecoder
from model.vid.vq_hvebt.clip_backbone import VQClipBackbone
from model.vid.vq_hvebt.config import VQHVEBTConfig, VQStageConfig
from model.vid.vq_hvebt.losses import (
    aggregate_stage_losses,
    decoder_loss_l1,
    prediction_loss,
)
from model.vid.vq_hvebt.quantizer import VectorQuantizer, QuantizerOutput
from model.vid.vq_hvebt.stage_predictor import VQHVEBTStage


# --------------------------------------------------------------------------- #
#  Output containers
# --------------------------------------------------------------------------- #


@dataclass
class StageForwardResult:
    """Intermediate results for one stage during the forward pass."""
    stage_name: str
    z_e: torch.Tensor           # (B, T+1, C, H, W) encoder output
    z_q: torch.Tensor           # (B, T+1, C, H, W) quantized (hard)
    z_q_st: torch.Tensor        # (B, T+1, C, H, W) straight-through quantized
    indices: torch.Tensor       # (B, T+1, H*W) long — code indices per token
    cb_loss: torch.Tensor       # scalar
    commit_loss: torch.Tensor   # scalar
    pred_embed: torch.Tensor    # (B, T, C, H, W) MCMC output for frames 1..T
    pred_loss: torch.Tensor     # scalar
    energy_trace: List[float]   # MCMC energy at each step


@dataclass
class VQHVEBTOutput:
    """Full model output from one forward pass."""
    total_loss: torch.Tensor
    stage_results: Dict[str, StageForwardResult]   # keyed by clip_stage_name
    decoder_loss: Optional[torch.Tensor]           # None if no decoder
    pred_rgb: Optional[torch.Tensor]               # (B, T, 3, H_out, W_out) or None
    metrics: Dict[str, float]                      # flat metric dict for logging


# --------------------------------------------------------------------------- #
#  VQHVEBTModel
# --------------------------------------------------------------------------- #


class VQHVEBTModel(nn.Module):
    """Full VQ-HVEBT hierarchical model.

    Parameters
    ----------
    cfg : VQHVEBTConfig
        Complete model configuration. See config.py for field docs.
    """

    def __init__(self, cfg: VQHVEBTConfig):
        super().__init__()
        self.cfg = cfg

        # ---- CLIP encoder ---------------------------------------------------
        stage_names = [s.clip_stage_name for s in cfg.stages]
        self.encoder = VQClipBackbone(
            weights_path=cfg.weights_path,
            return_stages=stage_names,
            trainable=cfg.train_encoder,
            lr_scale=cfg.encoder_lr_scale,
        )

        # ---- Per-stage quantizers and predictors ----------------------------
        # stages in cfg are ordered coarsest → finest.
        # self.quantizers and self.predictors are in the same order.
        self.quantizers = nn.ModuleDict()
        self.predictors = nn.ModuleDict()

        for idx, stage_cfg in enumerate(cfg.stages):
            name = stage_cfg.clip_stage_name
            self.quantizers[name] = VectorQuantizer(stage_cfg.codebook)

            # Parent config: the previous entry in stages list (one coarser).
            parent_cfg: Optional[VQStageConfig] = cfg.stages[idx - 1] if idx > 0 else None
            self.predictors[name] = VQHVEBTStage(
                cfg=stage_cfg,
                quantizer=self.quantizers[name],
                parent_cfg=parent_cfg,
            )

        # ---- Optional pixel decoder -----------------------------------------
        self.decoder: Optional[PixelDecoder] = None
        if cfg.use_decoder:
            finest = cfg.stages[-1]  # finest stage is last
            self.decoder = PixelDecoder(
                in_channels=finest.clip_channels,
                in_HW=(finest.H, finest.W),
                out_size=cfg.decoder_out_size,
            )

        # Track whether codebooks have been initialized from data.
        self._codebook_initialized: Dict[str, bool] = {
            s.clip_stage_name: (s.codebook.init_mode == "random")
            for s in cfg.stages
        }

    # ------------------------------------------------------------------ #
    #  Codebook initialization
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def maybe_initialize_codebooks(self, video: torch.Tensor) -> bool:
        """Initialize codebooks from a data batch if not done yet.

        Call this once after constructing the model and before the first
        optimizer step. It runs the encoder once (no_grad) and uses the
        resulting features to seed each codebook.

        Args:
            video: (B, T+1, 3, H, W) a real data batch.

        Returns:
            True if any codebook was freshly initialized.
        """
        all_done = all(self._codebook_initialized.values())
        if all_done:
            return False

        # Encode a subset of frames to get enough tokens.
        with torch.no_grad():
            feats = self.encoder.encode_video(video)  # {name: (B, T+1, C, H, W)}

        any_init = False
        for stage_cfg in self.cfg.stages:
            name = stage_cfg.clip_stage_name
            if self._codebook_initialized[name]:
                continue
            z_e = feats[name]                   # (B, T+1, C, H, W)
            B, T1, C, H, W = z_e.shape
            z_flat = z_e.permute(0, 1, 3, 4, 2).reshape(-1, C)  # (B*(T+1)*H*W, C)
            self.quantizers[name].initialize_from_data(z_flat)
            self._codebook_initialized[name] = True
            any_init = True

        return any_init

    # ------------------------------------------------------------------ #
    #  Forward: encode + quantize all stages
    # ------------------------------------------------------------------ #

    def _encode_and_quantize(
        self,
        video: torch.Tensor,   # (B, T+1, 3, H, W)
    ) -> Dict[str, Tuple[torch.Tensor, QuantizerOutput]]:
        """Run encoder and quantizer for all stages.

        Returns:
            dict mapping stage_name → (z_e_5d, quant_output) where:
                z_e_5d : (B, T+1, C, H, W)
                quant_output.z_q_st : (B, T+1, C, H, W)  straight-through
                quant_output.z_q    : (B, T+1, C, H, W)  hard quantized
        """
        B, T1, _, H, W = video.shape
        feats = self.encoder.encode_video(video)  # {name: (B, T+1, C, H, W)}

        results: Dict[str, Tuple[torch.Tensor, QuantizerOutput]] = {}
        for stage_cfg in self.cfg.stages:
            name = stage_cfg.clip_stage_name
            z_e_5d = feats[name]                   # (B, T+1, C, Hs, Ws)
            Bs, T1s, C, Hs, Ws = z_e_5d.shape
            N = T1s * Hs * Ws

            # Flatten spatiotemporal dims for the quantizer.
            z_e_flat = z_e_5d.permute(0, 1, 3, 4, 2).reshape(Bs, N, C)  # (B, N, C)
            qout = self.quantizers[name].encode(z_e_flat)                 # QuantizerOutput

            # Reshape outputs back to 5D.
            def unflatten(t: torch.Tensor) -> torch.Tensor:
                return t.reshape(Bs, T1s, Hs, Ws, C).permute(0, 1, 4, 2, 3).contiguous()

            qout_5d = QuantizerOutput(
                z_q_st=unflatten(qout.z_q_st),
                z_q=unflatten(qout.z_q),
                indices=qout.indices.reshape(Bs, T1s, Hs * Ws),
                cb_loss=qout.cb_loss,
                commit_loss=qout.commit_loss,
            )
            results[name] = (z_e_5d, qout_5d)

        return results

    # ------------------------------------------------------------------ #
    #  Full forward pass (training)
    # ------------------------------------------------------------------ #

    def forward_loss(
        self,
        video: torch.Tensor,   # (B, T+1, 3, H, W) in [0, 1]
    ) -> VQHVEBTOutput:
        """Compute training loss for a video batch.

        Args:
            video: (B, T+1, 3, H, W) normalised RGB. The model predicts
                   frame T+1 given frames 0..T-1 at every temporal position,
                   i.e. it sees T context frames and predicts T future frames
                   via the shifted-by-1 pair (frames 0..T-1 → 1..T).

        Returns:
            VQHVEBTOutput with total_loss, per-stage breakdowns, and metrics.
        """
        B, T1, _, H, W = video.shape
        T = T1 - 1   # number of context / target pairs

        if T < 1:
            raise ValueError(f"Video must have at least 2 frames, got {T1}")

        # ---- 1. Encode + quantize all stages --------------------------------
        enc_quant = self._encode_and_quantize(video)
        # enc_quant: {name: (z_e_5d, qout_5d)}

        # ---- 2. Collect per-stage results (top-down: coarse → fine) ---------
        stage_results: Dict[str, StageForwardResult] = {}
        pred_loss_map: Dict[str, torch.Tensor] = {}
        cb_loss_map: Dict[str, torch.Tensor] = {}
        commit_loss_map: Dict[str, torch.Tensor] = {}
        pred_weight_map: Dict[str, float] = {}
        cb_weight_map: Dict[str, float] = {}
        commit_weight_map: Dict[str, float] = {}

        # Parent predicted embedding (coarser stage output → finer stage input).
        parent_pred: Optional[torch.Tensor] = None  # (B, T, Cp, Hp, Wp) or None

        for idx, stage_cfg in enumerate(self.cfg.stages):
            name = stage_cfg.clip_stage_name
            z_e_5d, qout = enc_quant[name]

            # Context uses straight-through (encoder gets grad via commitment + ST).
            # Slice: frames 0..T-1.
            real_ctx = qout.z_q_st[:, :T]        # (B, T, C, H, W)

            # Target: quantized future frames 1..T, DETACHED and CLONED.
            # - detach: stop encoder from moving the target to reduce loss.
            # - clone: ensure in-place ops on z_q don't corrupt target.
            target = qout.z_q[:, 1:].detach().clone()   # (B, T, C, H, W)

            # Parent context for cross-attention (already detached below).
            par_ctx = None
            if parent_pred is not None:
                par_ctx = parent_pred   # already detached (see below)

            # MCMC prediction.
            predictor: VQHVEBTStage = self.predictors[name]
            _, pred_embed, energy_trace = predictor.run_mcmc(
                real_ctx=real_ctx,
                init_logits=None,
                parent_context=par_ctx,
                learning=self.training,
            )
            # pred_embed: (B, T, C, H, W)

            # Prediction loss.
            # Flatten pred and target to (B, N, C) for loss calculation.
            Bs, Tc, C, Hs, Ws = pred_embed.shape
            N = Tc * Hs * Ws
            pred_flat = pred_embed.permute(0, 1, 3, 4, 2).reshape(Bs, N, C)
            tgt_flat = target.permute(0, 1, 3, 4, 2).reshape(Bs, N, C)
            l_pred = prediction_loss(pred_flat, tgt_flat, kind=stage_cfg.pred_loss)

            # Prepare detached parent embedding for next finer stage.
            if self.cfg.detach_parent_kv:
                parent_pred = pred_embed.detach()
            else:
                parent_pred = pred_embed  # live (experimental; risks instability)

            # Collect.
            sr = StageForwardResult(
                stage_name=name,
                z_e=z_e_5d,
                z_q=qout.z_q,
                z_q_st=qout.z_q_st,
                indices=qout.indices,
                cb_loss=qout.cb_loss,
                commit_loss=qout.commit_loss,
                pred_embed=pred_embed,
                pred_loss=l_pred,
                energy_trace=energy_trace,
            )
            stage_results[name] = sr

            pred_loss_map[name] = l_pred
            cb_loss_map[name] = qout.cb_loss
            commit_loss_map[name] = qout.commit_loss
            pred_weight_map[name] = stage_cfg.pred_loss_weight
            cb_weight_map[name] = stage_cfg.cb_loss_weight
            commit_weight_map[name] = stage_cfg.commit_loss_weight

        # ---- 3. Aggregate stage losses --------------------------------------
        total = aggregate_stage_losses(
            pred_losses=pred_loss_map,
            cb_losses=cb_loss_map,
            commit_losses=commit_loss_map,
            pred_weights=pred_weight_map,
            cb_weights=cb_weight_map,
            commit_weights=commit_weight_map,
        )

        # ---- 4. Optional pixel decoder on finest stage ----------------------
        dec_loss: Optional[torch.Tensor] = None
        pred_rgb: Optional[torch.Tensor] = None
        if self.decoder is not None:
            finest_name = self.cfg.stages[-1].clip_stage_name
            finest_pred = stage_results[finest_name].pred_embed   # (B, T, C, H, W)
            # Decoder is a diagnostic tool: detach so decoder gradient does
            # NOT back-propagate into the quantizer or predictor.
            finest_pred_det = finest_pred.detach()

            # Flatten (B, T, C, H, W) → (B*T, C, H, W) for conv decoder.
            BT = B * T
            dec_in = finest_pred_det.reshape(BT, *finest_pred_det.shape[2:])
            pred_rgb_flat = self.decoder(dec_in)   # (B*T, 3, H_out, W_out)
            pred_rgb = pred_rgb_flat.reshape(B, T, 3, *pred_rgb_flat.shape[2:])

            # Ground-truth: frames 1..T resized to decoder output size.
            gt_future = video[:, 1:]   # (B, T, 3, H_in, W_in)
            if pred_rgb.shape[-2:] != gt_future.shape[-2:]:
                gt_future = F.interpolate(
                    gt_future.reshape(BT, 3, *gt_future.shape[3:]),
                    size=pred_rgb.shape[-2:], mode="bilinear", align_corners=False,
                ).reshape(B, T, 3, *pred_rgb.shape[-2:])

            dec_loss = decoder_loss_l1(pred_rgb, gt_future)
            total = total + self.cfg.decoder_loss_weight * dec_loss

        # ---- 5. Build metrics dict ------------------------------------------
        metrics = self._build_metrics(stage_results, dec_loss)

        return VQHVEBTOutput(
            total_loss=total,
            stage_results=stage_results,
            decoder_loss=dec_loss,
            pred_rgb=pred_rgb,
            metrics=metrics,
        )

    # ------------------------------------------------------------------ #
    #  Inference: predict next-frame latent for one step
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_next(
        self,
        context_video: torch.Tensor,  # (B, T, 3, H, W) past frames
    ) -> Dict[str, torch.Tensor]:
        """Predict next-frame quantized latents for all stages.

        Args:
            context_video: (B, T, 3, H, W) in [0, 1].

        Returns:
            dict mapping stage_name → (B, C, H, W) predicted embedding for
            the next frame (frame T), at that stage's spatial resolution.
        """
        B, T, _, H, W = context_video.shape
        feats = self.encoder.encode_video(context_video)

        results: Dict[str, torch.Tensor] = {}
        parent_pred: Optional[torch.Tensor] = None

        for stage_cfg in self.cfg.stages:
            name = stage_cfg.clip_stage_name
            z_e_5d = feats[name]   # (B, T, C, Hs, Ws)
            Bs, Tc, C, Hs, Ws = z_e_5d.shape
            N = Tc * Hs * Ws

            z_e_flat = z_e_5d.permute(0, 1, 3, 4, 2).reshape(Bs, N, C)
            qout = self.quantizers[name].encode(z_e_flat)

            # Reshape to 5D.
            z_q_st_5d = qout.z_q_st.reshape(Bs, Tc, Hs, Ws, C).permute(0, 1, 4, 2, 3).contiguous()

            predictor: VQHVEBTStage = self.predictors[name]
            _, pred_embed, _ = predictor.run_mcmc(
                real_ctx=z_q_st_5d,
                init_logits=None,
                parent_context=parent_pred,
                learning=False,
            )
            # Return only the last predicted frame.
            results[name] = pred_embed[:, -1]   # (B, C, Hs, Ws)
            parent_pred = pred_embed.detach()

        return results

    # ------------------------------------------------------------------ #
    #  Optimizer parameter groups (encoder at reduced LR)
    # ------------------------------------------------------------------ #

    def parameter_groups(self, base_lr: float):
        """Build optimizer parameter groups with encoder at reduced LR.

        Args:
            base_lr: learning rate for quantizers and predictors.

        Returns:
            List of dicts suitable for torch.optim.AdamW or similar.
        """
        enc_params = list(self.encoder.parameters())
        enc_ids = {id(p) for p in enc_params}
        other_params = [p for p in self.parameters() if id(p) not in enc_ids]

        groups = [
            {"params": other_params, "lr": base_lr},
            {"params": enc_params, "lr": base_lr * self.cfg.encoder_lr_scale},
        ]
        return groups

    # ------------------------------------------------------------------ #
    #  Metrics
    # ------------------------------------------------------------------ #

    def _build_metrics(
        self,
        stage_results: Dict[str, StageForwardResult],
        dec_loss: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for name, sr in stage_results.items():
            metrics[f"{name}/loss_pred"] = sr.pred_loss.item()
            metrics[f"{name}/loss_cb"] = sr.cb_loss.item()
            metrics[f"{name}/loss_commit"] = sr.commit_loss.item()
            # Codebook usage and perplexity (no grad needed).
            with torch.no_grad():
                q = self.quantizers[name]
                idx = sr.indices.reshape(-1)
                metrics[f"{name}/codebook_usage"] = q.codebook_usage(idx).item()
                metrics[f"{name}/codebook_perplexity"] = q.perplexity(idx).item()
            # Energy trace: report first and last step energy.
            if sr.energy_trace:
                metrics[f"{name}/energy_step0"] = sr.energy_trace[0]
                metrics[f"{name}/energy_final"] = sr.energy_trace[-1]
        if dec_loss is not None:
            metrics["decoder/loss"] = dec_loss.item()
        return metrics
