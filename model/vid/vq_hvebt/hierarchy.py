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
         target^k     = indices^k[:, 1:]    (frames 1..T,   code indices)
  4. Predict from coarsest stage to finest (top-down):
     For each stage k (ordered coarse→fine):
       a. Run MCMC from zero init in logit space (NLP EBT style).
       b. Get final predicted logits (B, N, K) and decoded embedding z_pred^k.
       c. Compute prediction loss: CE(pred_logits, target_indices).
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
    ce_prediction_loss,
    decoder_loss_l1,
    prediction_loss,
    soft_ce_prediction_loss,
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
        self._step_counter = 0  # tracks training steps for encoder warmup

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
            self.quantizers[name] = VectorQuantizer(
                stage_cfg.codebook,
                detach_codebook_in_decode=cfg.detach_pred_context,
            )

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
        """Run encoder and quantizer for all stages.

        Args:
            video: input video batch.
            detach_encoder: if True, detach encoder outputs before quantization.
                This freezes the encoder gradient (used during warmup to
                stabilize target codes before letting the encoder train).

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

            # During warmup, detach encoder output to prevent encoder gradient.
            # This keeps target codes stable so predictor+codebook can train.
            if detach_encoder:
                z_e_5d = z_e_5d.detach()

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

        # Track training steps for encoder warmup.
        if self.training:
            self._step_counter += 1
        in_warmup = (
            self.cfg.encoder_warmup_steps > 0
            and self._step_counter <= self.cfg.encoder_warmup_steps
        )

        # ---- 1. Encode + quantize all stages --------------------------------
        enc_quant = self._encode_and_quantize(video, detach_encoder=in_warmup)
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

            # Detach context from encoder to prevent pred_loss from destabilizing
            # encoder training. When detached, encoder trains ONLY via commitment
            # loss (which stabilizes VQ assignments by pushing z_e toward codes).
            # The pred_loss gradient can still reach the codebook (through
            # softmax @ E in the decode step) and the predictor.
            if self.cfg.detach_pred_context:
                real_ctx = real_ctx.detach()

            # Target code indices: future frames 1..T.
            # These are the ground-truth codes the predictor should output.
            target_indices = qout.indices[:, 1:]  # (B, T, H*W) long

            # Parent context for cross-attention (already detached below).
            par_ctx = None
            if parent_pred is not None:
                par_ctx = parent_pred   # already detached (see below)

            # MCMC prediction in logit space (NLP EBT style).
            predictor: VQHVEBTStage = self.predictors[name]
            all_step_logits, pred_embed, energy_trace = predictor.run_mcmc(
                real_ctx=real_ctx,
                init_logits=None,
                parent_context=par_ctx,
                learning=self.training,
            )
            # all_step_logits: list of (B, T*H*W, K) — logits after each MCMC step
            # pred_embed:      (B, T, C, H, W) — decoded embedding from final logits

            # Prediction loss: CE on logits (NLP EBT pattern).
            # - truncate_mcmc=True (default): CE only on FINAL step logits.
            #   This is the practical NLP EBT setting — the final logits after
            #   full MCMC have moved far from uniform and represent a meaningful
            #   prediction. Earlier steps are noise that dilutes the signal.
            # - truncate_mcmc=False: CE on ALL steps (averaged).
            #   More gradient signal but noisier; requires smaller step_size.
            Bs, Tc, C, Hs, Ws = pred_embed.shape
            N = Tc * Hs * Ws
            K = stage_cfg.codebook.num_codes
            tgt_idx_flat = target_indices.reshape(Bs, N)   # (B, T*H*W)

            # Soft vs hard targets:
            # soft_target_tau > 0: use smooth distance-based distribution
            #   (eliminates moving-target discontinuity when encoder is trainable)
            # soft_target_tau == 0: hard one-hot CE (original, only stable
            #   with frozen encoder)
            use_soft = stage_cfg.soft_target_tau > 0

            if use_soft:
                # z_e of future frames for distance computation.
                z_e_future = z_e_5d[:, 1:]  # (B, T, C, H, W)
                z_e_future_flat = z_e_future.permute(0, 1, 3, 4, 2).reshape(Bs, N, C)
                cb_weight = self.quantizers[name].codebook.weight  # (K, C)

                # Detach both z_e and codebook — targets should not backprop
                # to encoder or codebook (those train via commit/cb loss).
                z_e_det = z_e_future_flat.detach()
                cb_det = cb_weight.detach()

            if stage_cfg.truncate_mcmc:
                # Only final step (NLP EBT truncate_mcmc=True default).
                if use_soft:
                    l_pred = soft_ce_prediction_loss(
                        all_step_logits[-1], z_e_det, cb_det,
                        tau=stage_cfg.soft_target_tau,
                    )
                else:
                    l_pred = ce_prediction_loss(all_step_logits[-1], tgt_idx_flat)
            else:
                # All steps averaged (NLP EBT truncate_mcmc=False).
                l_pred = torch.tensor(0.0, device=pred_embed.device)
                for step_logits in all_step_logits:
                    if use_soft:
                        l_pred = l_pred + soft_ce_prediction_loss(
                            step_logits, z_e_det, cb_det,
                            tau=stage_cfg.soft_target_tau,
                        )
                    else:
                        l_pred = l_pred + ce_prediction_loss(step_logits, tgt_idx_flat)
                l_pred = l_pred / len(all_step_logits)

            # Contrastive energy loss: energy(true) should be < energy(predicted).
            # This gives a DIRECT first-order signal to the energy function,
            # telling it what low-energy states look like (without going through
            # the second-order MCMC Hessian). Matches NLP EBT's contrastive_loss.
            if self.cfg.contrastive_loss_weight > 0 and self.training:
                # True embedding: the actual future quantized features.
                true_embed = qout.z_q[:, 1:].detach()  # (B, T, C, H, W)
                true_energy = predictor.forward_energy(real_ctx, true_embed, par_ctx)  # (B, N)
                # Predicted embedding energy (from final MCMC state, detached).
                pred_energy = predictor.forward_energy(
                    real_ctx, pred_embed.detach(), par_ctx
                )  # (B, N)
                # Stack: column 0 = true energy, column 1 = predicted energy.
                # Target: 0 (true should have lower energy → be selected by argmin).
                energy_stack = torch.stack([true_energy.sum(-1), pred_energy.sum(-1)], dim=-1)  # (B, 2)
                energy_targets = torch.zeros(Bs, dtype=torch.long, device=pred_embed.device)
                contrastive_l = F.cross_entropy(-energy_stack, energy_targets)
                l_pred = l_pred + self.cfg.contrastive_loss_weight * contrastive_l

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

        Three groups:
          1. Predictor params (full LR): transformers, energy heads, step_size.
          2. Codebook params (reduced LR): codebook embeddings — need low LR
             because AdamW's scale-invariance moves entries by O(LR) per step
             regardless of gradient magnitude, which can exceed inter-code
             distance and destabilize VQ assignments.
          3. Encoder params (tiny LR): pretrained CLIP — change slowly.

        Args:
            base_lr: learning rate for predictors.

        Returns:
            List of dicts suitable for torch.optim.AdamW or similar.
        """
        enc_params = list(self.encoder.parameters())
        enc_ids = {id(p) for p in enc_params}

        cb_params = []
        for q in self.quantizers.values():
            cb_params.extend(list(q.parameters()))
        cb_ids = {id(p) for p in cb_params}

        pred_params = [
            p for p in self.parameters()
            if id(p) not in enc_ids and id(p) not in cb_ids
        ]

        groups = [
            {"params": pred_params, "lr": base_lr},
            {"params": cb_params, "lr": base_lr * self.cfg.codebook_lr_scale},
            {"params": enc_params, "lr": base_lr * self.cfg.encoder_lr_scale},
        ]
        return groups

    def encoder_params(self):
        """Return encoder parameters (for separate SGD optimizer)."""
        return list(self.encoder.parameters())

    def non_encoder_params(self):
        """Return all parameters except encoder (for AdamW optimizer)."""
        enc_ids = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in enc_ids]

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
