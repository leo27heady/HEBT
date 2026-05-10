"""
VQ-HVEBT: top-level hierarchical model.

This module owns:
  - The encoder (either pretrained CLIP or custom ConvEncoder).
  - An EMA copy of the encoder that produces stable target codes (BYOL/DINO).
  - One VectorQuantizer per stage (EMA-updated, no gradient).
  - One VQHVEBTStage (EBT predictor) per stage.
  - An optional PixelDecoder on the finest stage.

Forward pass overview (Option B — custom encoder + EMA targets)
---------------------------------------------------------------

Given a video clip of shape (B, T+1, 3, H, W):
  1. Encode ALL T+1 frames through the LIVE encoder → per-stage features.
  2. Encode ALL T+1 frames through the frozen EMA encoder → per-stage features
     for computing TARGET code indices only (stable, no gradient).
  3. For each stage k:
     a. Quantize LIVE features → z_q_st (straight-through for context).
     b. Quantize EMA features → target_indices (stable codes for CE loss).
     c. EMA codebook is updated from LIVE features.
  4. Predict from coarsest stage to finest (top-down):
     For each stage k (ordered coarse→fine):
       a. Run MCMC from pred_head warm-start in logit space.
       b. Per-token gradient normalization makes MCMC scale-invariant (F4).
       c. Energy is bounded via tanh (F1).
       d. Compute CE loss: pred_logits vs EMA target_indices.
       e. Pass (detached) z_pred^k as parent_context to next finer stage.
  5. Optional: decode finest-stage prediction to RGB and compute pixel loss.
  6. Return total loss and a metric dict.

Gradient flow summary
---------------------

  live_encoder ←── prediction_loss (via straight-through z_q_st context)
  live_encoder ←── prediction_loss (via straight-through z_q_st when detach_pred_context=False)
  ema_encoder  : no gradient (updated by EMA after optimizer.step())
  codebook     : no gradient (updated by EMA from live encoder features)
  predictor    ←── prediction_loss (via MCMC unroll with create_graph=True)
  pred_head    ←── prediction_loss (direct CE on pred_head logits)
  decoder      ←── decoder_loss (does NOT flow to predictor)

Target stability
-----------------
  target_indices come from EMA encoder → EMA quantizer lookup.
  The EMA encoder changes slowly (decay 0.999 → 1000-step half-life),
  so targets drift smoothly rather than flipping 98.8% per step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vid.hvebt.decoder import PixelDecoder
from model.vid.vq_hvebt.clip_backbone import VQClipBackbone
from model.vid.vq_hvebt.config import VQHVEBTConfig, VQStageConfig
from model.vid.vq_hvebt.conv_encoder import ConvEncoderWrapper
from model.vid.vq_hvebt.losses import (
    aggregate_stage_losses,
    ce_prediction_loss,
    decoder_loss_l1,
    prediction_loss,
    soft_ce_prediction_loss,
)
from model.vid.vq_hvebt.quantizer import VectorQuantizer, QuantizerOutput
from model.vid.vq_hvebt.stage_predictor import VQHVEBTStage


# --------------------------------------------------------------------------- #
#  Gradient scaling for bottom-up flow
# --------------------------------------------------------------------------- #


class _GradScale(torch.autograd.Function):
    """Scale gradient by a constant during backward pass (identity forward)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale gradient magnitude without affecting forward pass."""
    if scale == 1.0:
        return x
    return _GradScale.apply(x, scale)


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
    final_logits: Optional[torch.Tensor] = None   # (B, T*H*W, K) final MCMC logits
    final_energy: Optional[torch.Tensor] = None   # (B, T*H*W) per-token energy
    mcmc_steps_taken: Optional[int] = None         # adaptive MCMC: actual steps
    mcmc_stop_reason: Optional[str] = None         # "converged" | "max_steps" | "nan_grad"


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
        self._step_counter = 0  # tracks training steps for encoder warmup

        # ---- Encoder (CLIP or custom ConvEncoder) ---------------------------
        stage_names = [s.clip_stage_name for s in cfg.stages]

        if cfg.use_custom_encoder:
            # Option B: Custom ConvEncoder + EMA target encoder.
            self.encoder = ConvEncoderWrapper(
                return_stages=stage_names,
                base_channels=cfg.encoder_base_channels,
                lr_scale=cfg.encoder_lr_scale,
                ema_decay=cfg.ema_target_decay,
            )
            self._use_custom_encoder = True
        else:
            # Legacy: pretrained CLIP backbone.
            self.encoder = VQClipBackbone(
                weights_path=cfg.weights_path,
                return_stages=stage_names,
                trainable=cfg.train_encoder,
                lr_scale=cfg.encoder_lr_scale,
            )
            self._use_custom_encoder = False

        # ---- Per-stage quantizers and predictors ----------------------------
        # stages in cfg are ordered coarsest → finest.
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
        detach_encoder: bool = False,
    ) -> Dict[str, Tuple[torch.Tensor, QuantizerOutput]]:
        """Run encoder and quantizer for all stages (LIVE encoder).

        Args:
            video: input video batch.
            detach_encoder: if True, detach encoder outputs before quantization.

        Returns:
            dict mapping stage_name → (z_e_5d, quant_output).
        """
        B, T1, _, H, W = video.shape
        feats = self.encoder.encode_video(video)  # {name: (B, T+1, C, H, W)}

        results: Dict[str, Tuple[torch.Tensor, QuantizerOutput]] = {}
        for stage_cfg in self.cfg.stages:
            name = stage_cfg.clip_stage_name
            z_e_5d = feats[name]

            if detach_encoder:
                z_e_5d = z_e_5d.detach()

            Bs, T1s, C, Hs, Ws = z_e_5d.shape
            N = T1s * Hs * Ws

            z_e_flat = z_e_5d.permute(0, 1, 3, 4, 2).reshape(Bs, N, C)
            qout = self.quantizers[name].encode(z_e_flat)

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

    def _encode_and_quantize_ema(
        self,
        video: torch.Tensor,   # (B, T+1, 3, H, W)
    ) -> Dict[str, torch.Tensor]:
        """Run EMA encoder and quantize to get STABLE target indices.

        Uses the frozen EMA encoder so targets don't flip every step.
        Only returns indices (no straight-through needed for targets).

        Args:
            video: input video batch.

        Returns:
            dict mapping stage_name → (B, T+1, H*W) long target indices.
        """
        if not self._use_custom_encoder:
            # CLIP mode: no EMA encoder, return None to fall back to live indices.
            return {}

        with torch.no_grad():
            feats = self.encoder.encode_video_ema(video)  # {name: (B, T+1, C, H, W)}

        target_indices: Dict[str, torch.Tensor] = {}
        for stage_cfg in self.cfg.stages:
            name = stage_cfg.clip_stage_name
            z_e_5d = feats[name]
            Bs, T1s, C, Hs, Ws = z_e_5d.shape
            N = T1s * Hs * Ws

            z_e_flat = z_e_5d.permute(0, 1, 3, 4, 2).reshape(Bs, N, C)
            # Nearest-neighbor lookup only (no EMA update — that's done by live encoder).
            E = self.quantizers[name].codebook_weight.detach()
            z_sq = (z_e_flat.reshape(-1, C) ** 2).sum(dim=1, keepdim=True)
            e_sq = (E ** 2).sum(dim=1, keepdim=True).T
            dot = z_e_flat.reshape(-1, C) @ E.T
            dist = z_sq + e_sq - 2 * dot
            indices_flat = dist.argmin(dim=1)
            target_indices[name] = indices_flat.reshape(Bs, T1s, Hs * Ws)

        return target_indices

    # ------------------------------------------------------------------ #
    #  Full forward pass (training)
    # ------------------------------------------------------------------ #

    def forward_loss(
        self,
        video: torch.Tensor,   # (B, T+1, 3, H, W) in [0, 1]
    ) -> VQHVEBTOutput:
        """Compute training loss for a video batch.

        Args:
            video: (B, T+1, 3, H, W) normalised RGB.

        Returns:
            VQHVEBTOutput with total_loss, per-stage breakdowns, and metrics.
        """
        B, T1, _, H, W = video.shape
        T = T1 - 1

        if T < 1:
            raise ValueError(f"Video must have at least 2 frames, got {T1}")

        if self.training:
            self._step_counter += 1

        # Encoder warmup: linearly ramp encoder features from detached to live
        # over the warmup window. At step 0 features are fully detached (no encoder
        # gradient). At step == encoder_warmup_steps they are fully live.
        # This avoids the hard unfreeze discontinuity that causes loss spikes.
        warmup_steps = self.cfg.encoder_warmup_steps
        if warmup_steps > 0 and self._step_counter <= warmup_steps:
            warmup_alpha = self._step_counter / warmup_steps  # 0→1 linearly
        else:
            warmup_alpha = 1.0  # fully live

        # ---- 1. Encode + quantize all stages (LIVE encoder) -----------------
        if warmup_alpha == 0.0:
            enc_quant = self._encode_and_quantize(video, detach_encoder=True)
        elif warmup_alpha < 1.0:
            enc_quant = self._encode_and_quantize(video, detach_encoder=False)
            # Blend: z = detached + alpha * (live - detached) = (1-alpha)*detached + alpha*live
            # This gives a smooth gradient scale-up from 0 to full.
            for name in enc_quant:
                z_e_5d, qout = enc_quant[name]
                z_e_blended = z_e_5d.detach() + warmup_alpha * (z_e_5d - z_e_5d.detach())
                enc_quant[name] = (z_e_blended, qout)
        else:
            enc_quant = self._encode_and_quantize(video, detach_encoder=False)

        # ---- 1b. Get STABLE target indices from EMA encoder -----------------
        ema_target_map = self._encode_and_quantize_ema(video)
        # ema_target_map: {name: (B, T+1, H*W)} or {} if CLIP mode

        # ---- 2. Collect per-stage results (top-down: coarse → fine) ---------
        stage_results: Dict[str, StageForwardResult] = {}
        pred_loss_map: Dict[str, torch.Tensor] = {}
        cb_loss_map: Dict[str, torch.Tensor] = {}
        commit_loss_map: Dict[str, torch.Tensor] = {}
        pred_weight_map: Dict[str, float] = {}
        cb_weight_map: Dict[str, float] = {}
        commit_weight_map: Dict[str, float] = {}

        parent_pred: Optional[torch.Tensor] = None

        for idx, stage_cfg in enumerate(self.cfg.stages):
            name = stage_cfg.clip_stage_name
            z_e_5d, qout = enc_quant[name]

            real_ctx = qout.z_q_st[:, :T]

            if self.cfg.detach_pred_context:
                real_ctx = real_ctx.detach()

            # Target code indices: use EMA encoder if available (stable targets).
            if name in ema_target_map:
                target_indices = ema_target_map[name][:, 1:]  # (B, T, H*W)
            else:
                target_indices = qout.indices[:, 1:]  # (B, T, H*W)

            par_ctx = None
            if parent_pred is not None:
                par_ctx = parent_pred

            # MCMC prediction.
            # In decoder_only_loss mode: use_linear_decode + mcmc_no_detach
            # are set automatically, so MCMC uses learned linear projection
            # (no softmax saturation) and keeps full computation graph
            # (decoder loss gradient flows through all MCMC steps).
            predictor: VQHVEBTStage = self.predictors[name]
            mcmc_steps_taken: Optional[int] = None
            mcmc_stop_reason: Optional[str] = None
            if stage_cfg.adaptive_mcmc:
                all_step_logits, pred_embed, energy_trace, mcmc_steps_taken, mcmc_stop_reason = (
                    predictor.run_mcmc_adaptive(
                        real_ctx=real_ctx,
                        init_logits=None,
                        parent_context=par_ctx,
                        learning=self.training,
                    )
                )
            else:
                all_step_logits, pred_embed, energy_trace = predictor.run_mcmc(
                    real_ctx=real_ctx,
                    init_logits=None,
                    parent_context=par_ctx,
                    learning=self.training,
                )

            Bs, Tc, C, Hs, Ws = pred_embed.shape
            N = Tc * Hs * Ws
            K = stage_cfg.codebook.num_codes
            tgt_idx_flat = target_indices.reshape(Bs, N)

            # ---- Per-stage prediction loss -------------------------------- #
            # In decoder_only_loss mode, skip per-stage CE/energy/contrastive.
            # The only training signal comes from the decoder pixel loss.
            if self.cfg.decoder_only_loss:
                l_pred = torch.tensor(0.0, device=pred_embed.device)
            else:
                use_soft = stage_cfg.soft_target_tau > 0

                if use_soft:
                    z_e_future = z_e_5d[:, 1:]
                    z_e_future_flat = z_e_future.permute(0, 1, 3, 4, 2).reshape(Bs, N, C)
                    cb_weight = self.quantizers[name].codebook.weight
                    z_e_det = z_e_future_flat.detach()
                    cb_det = cb_weight.detach()

                # Compute CE loss only on logits with live computation graphs.
                if stage_cfg.truncate_mcmc:
                    has_pred_head = predictor.pred_head is not None and len(all_step_logits) > 1
                    ce_logits = []
                    if has_pred_head:
                        ce_logits.append(all_step_logits[0])   # pred_head output
                    ce_logits.append(all_step_logits[-1])      # last MCMC step
                else:
                    ce_logits = all_step_logits

                l_pred = torch.tensor(0.0, device=pred_embed.device)
                for step_logits in ce_logits:
                    if use_soft:
                        l_pred = l_pred + soft_ce_prediction_loss(
                            step_logits, z_e_det, cb_det,
                            tau=stage_cfg.soft_target_tau,
                        )
                    else:
                        l_pred = l_pred + ce_prediction_loss(step_logits, tgt_idx_flat)
                l_pred = l_pred / max(len(ce_logits), 1)

                # F1: Energy regularization — penalize large energy magnitudes.
                if stage_cfg.energy_reg_weight > 0 and energy_trace:
                    z_pred_last_flat = self.quantizers[name].decode_logits(all_step_logits[-1])
                    z_pred_last = z_pred_last_flat.reshape(Bs, Tc, Hs, Ws, C).permute(0, 1, 4, 2, 3).contiguous()
                    energy_for_reg = predictor.forward_energy(real_ctx, z_pred_last.detach(), par_ctx)
                    energy_reg = stage_cfg.energy_reg_weight * energy_for_reg.pow(2).mean()
                    l_pred = l_pred + energy_reg

                # Contrastive energy loss.
                if self.cfg.contrastive_loss_weight > 0 and self.training:
                    true_embed = qout.z_q[:, 1:].detach()
                    true_energy = predictor.forward_energy(real_ctx, true_embed, par_ctx)
                    pred_energy = predictor.forward_energy(
                        real_ctx, pred_embed.detach(), par_ctx
                    )
                    energy_stack = torch.stack([true_energy.sum(-1), pred_energy.sum(-1)], dim=-1)
                    energy_targets = torch.zeros(Bs, dtype=torch.long, device=pred_embed.device)
                    contrastive_l = F.cross_entropy(-energy_stack, energy_targets)
                    l_pred = l_pred + self.cfg.contrastive_loss_weight * contrastive_l

                # Adaptive MCMC step penalty (discourages long runs).
                if stage_cfg.adaptive_mcmc and stage_cfg.adaptive_mcmc_step_penalty > 0 and mcmc_steps_taken is not None:
                    step_pen = stage_cfg.adaptive_mcmc_step_penalty * mcmc_steps_taken
                    l_pred = l_pred + step_pen

            if self.cfg.detach_parent_kv:
                parent_pred = pred_embed.detach()
            else:
                parent_pred = pred_embed  # grad_scale(pred_embed, self.cfg.bottom_up_grad_scale)

            # Compute per-token energy for diagnostics (energy maps).
            with torch.no_grad():
                final_energy = predictor.forward_energy(real_ctx, pred_embed.detach(), par_ctx)
                # final_energy: (B, T*H*W) per-token energy

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
                final_logits=all_step_logits[-1].detach(),
                final_energy=final_energy.detach(),
                mcmc_steps_taken=mcmc_steps_taken,
                mcmc_stop_reason=mcmc_stop_reason,
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
            # Detach by default so decoder gradient does NOT back-propagate
            # into the predictor. Set decoder_detach=False to enable flow.
            if self.cfg.decoder_detach:
                finest_pred = finest_pred.detach()

            # Flatten (B, T, C, H, W) → (B*T, C, H, W) for conv decoder.
            BT = B * T
            dec_in = finest_pred.reshape(BT, *finest_pred.shape[2:])
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

            # ---- Context reconstruction loss (autoencoder on input frames) ---
            # Trains the encoder-decoder to faithfully reconstruct context
            # frames, creating a meaningful feature space for MCMC prediction.
            if self.cfg.context_recon_weight > 0:
                finest_ctx = stage_results[finest_name].z_q_st[:, :T]  # (B, T, C, H, W)
                ctx_in = finest_ctx.reshape(BT, *finest_ctx.shape[2:])
                ctx_rgb_flat = self.decoder(ctx_in)  # (B*T, 3, H_out, W_out)
                ctx_rgb = ctx_rgb_flat.reshape(B, T, 3, *ctx_rgb_flat.shape[2:])

                gt_ctx = video[:, :T]  # (B, T, 3, H_in, W_in)
                if ctx_rgb.shape[-2:] != gt_ctx.shape[-2:]:
                    gt_ctx = F.interpolate(
                        gt_ctx.reshape(BT, 3, *gt_ctx.shape[3:]),
                        size=ctx_rgb.shape[-2:], mode="bilinear", align_corners=False,
                    ).reshape(B, T, 3, *ctx_rgb.shape[-2:])

                ctx_recon_loss = decoder_loss_l1(ctx_rgb, gt_ctx)
                total = total + self.cfg.context_recon_weight * ctx_recon_loss
            else:
                ctx_recon_loss = None
        else:
            ctx_recon_loss = None

        # ---- 5. Build metrics dict ------------------------------------------
        metrics = self._build_metrics(stage_results, dec_loss, ctx_recon_loss)

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
            # pred_embed is the final decoded embedding (5D).
            # Return only the last predicted frame.
            results[name] = pred_embed[:, -1]   # (B, C, Hs, Ws)
            parent_pred = pred_embed.detach()

        return results

    # ------------------------------------------------------------------ #
    #  Optimizer parameter groups (encoder at reduced LR)
    # ------------------------------------------------------------------ #

    def parameter_groups(self, base_lr: float):
        """Build optimizer parameter groups with encoder at reduced LR.

        Groups:
          1. Predictor params (full LR): transformers, energy heads, pred_head, step_size.
          2. Encoder params (encoder LR): live encoder — learned from scratch or fine-tuned.

        Note: Codebook is EMA-updated (no gradient), so no codebook params here.
              EMA encoder is frozen (no gradient).

        Args:
            base_lr: learning rate for predictors.

        Returns:
            List of dicts suitable for torch.optim.AdamW or similar.
        """
        if self._use_custom_encoder:
            enc_params = list(self.encoder.live.parameters())
        else:
            enc_params = list(self.encoder.parameters())

        enc_ids = {id(p) for p in enc_params}

        pred_params = [
            p for p in self.parameters()
            if id(p) not in enc_ids and p.requires_grad
        ]

        groups = [
            {"params": pred_params, "lr": base_lr},
            {"params": enc_params, "lr": base_lr * self.cfg.encoder_lr_scale},
        ]
        return groups

    def encoder_params(self):
        """Return live encoder parameters."""
        if self._use_custom_encoder:
            return list(self.encoder.live.parameters())
        return list(self.encoder.parameters())

    def non_encoder_params(self):
        """Return all learnable parameters except encoder."""
        enc_ids = {id(p) for p in self.encoder_params()}
        return [p for p in self.parameters() if id(p) not in enc_ids and p.requires_grad]

    def update_ema_encoder(self) -> None:
        """Update the EMA target encoder from the live encoder.

        Call this AFTER each optimizer step. Only relevant for custom encoder mode.
        """
        if self._use_custom_encoder:
            self.encoder.update_ema()

    # ------------------------------------------------------------------ #
    #  Metrics
    # ------------------------------------------------------------------ #

    def _build_metrics(
        self,
        stage_results: Dict[str, StageForwardResult],
        dec_loss: Optional[torch.Tensor],
        ctx_recon_loss: Optional[torch.Tensor] = None,
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

                # Average entropy of predicted distribution (bits).
                # Low entropy = confident predictions; high = uncertain.
                if sr.final_logits is not None:
                    probs = F.softmax(sr.final_logits, dim=-1)  # (B, N, K)
                    log_probs = F.log_softmax(sr.final_logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1)  # (B, N) nats
                    entropy_bits = entropy / math.log(2)
                    metrics[f"{name}/entropy_avg"] = entropy_bits.mean().item()
                    metrics[f"{name}/entropy_min"] = entropy_bits.min().item()
                    metrics[f"{name}/entropy_max"] = entropy_bits.max().item()

                # Energy statistics.
                if sr.final_energy is not None:
                    metrics[f"{name}/energy_mean"] = sr.final_energy.mean().item()
                    metrics[f"{name}/energy_std"] = sr.final_energy.std().item()

            # Energy trace: report first and last step energy.
            if sr.energy_trace:
                metrics[f"{name}/energy_step0"] = sr.energy_trace[0]
                metrics[f"{name}/energy_final"] = sr.energy_trace[-1]

            # Adaptive MCMC step count and stop reason.
            if sr.mcmc_steps_taken is not None:
                metrics[f"{name}/mcmc_steps"] = float(sr.mcmc_steps_taken)
                scfg = next((s for s in self.cfg.stages if s.clip_stage_name == name), None)
                if scfg is not None:
                    metrics[f"{name}/mcmc_converged"] = 1.0 if sr.mcmc_steps_taken < scfg.adaptive_mcmc_max_steps else 0.0
            if sr.mcmc_stop_reason is not None:
                # Encode as: converged=1, max_steps=0, nan_grad=-1.
                reason_map = {"converged": 1.0, "max_steps": 0.0, "nan_grad": -1.0}
                metrics[f"{name}/mcmc_stop_reason"] = reason_map.get(sr.mcmc_stop_reason, 0.0)

            # Energy decrease: first energy minus last (positive = energy went down).
            if len(sr.energy_trace) >= 2:
                metrics[f"{name}/energy_decrease"] = sr.energy_trace[0] - sr.energy_trace[-1]
        if dec_loss is not None:
            metrics["decoder/loss"] = dec_loss.item()
        if ctx_recon_loss is not None:
            metrics["decoder/ctx_recon_loss"] = ctx_recon_loss.item()
        return metrics
