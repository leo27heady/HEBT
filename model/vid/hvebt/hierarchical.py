"""
Hierarchical Video EBT (Phase 2 + Phase 3).

Stacks N HVEBTStage modules ordered finest -> coarsest (index 0 = finest).
The **prediction** (MCMC) tower runs **top-down**: from the apex (coarsest)
stage to the base (finest) stage. Each stage:
  - Has its own MCMC over its own predicted features at its own CLIP feature
    space (e.g. s1 32x32, s2 16x16, s3 8x8).
  - Beyond the apex, cross-attends to the previous (coarser) stage's final
    MCMC prediction with **detached** KV. This means:
        * The upper stage's params get NO gradient from the lower stage's loss.
        * The upper stage's prediction quality DOES condition the lower stage.
  - Strict child→parent mask + same-time constraint on the cross-attention
    (see `cross_attention.py`).

Optional pixel decoder is attached to the **finest** stage and trained
independently on detached features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vid.hvebt.clip_encoder import MobileClipMultiStageEncoder
from model.vid.hvebt.decoder import PixelDecoder
from model.vid.hvebt.hvebt import HVEBTStage, HVEBTStageConfig


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #


def default_3stage_configs() -> List[HVEBTStageConfig]:
    """
    Default Phase-3 stack matching MobileCLIP2-S0:
        Stage 0 (finest)  : s1 (128 ch, 32x32)
        Stage 1           : s2 (256 ch, 16x16)
        Stage 2 (apex)    : s3 (512 ch, 8x8)
    """
    return [
        HVEBTStageConfig(clip_stage_name="s1", clip_channels=128, H=32, W=32,
                         embed_dim=128, n_heads=4, n_layers=2),
        HVEBTStageConfig(clip_stage_name="s2", clip_channels=256, H=16, W=16,
                         embed_dim=192, n_heads=4, n_layers=2),
        HVEBTStageConfig(clip_stage_name="s3", clip_channels=512, H=8,  W=8,
                         embed_dim=256, n_heads=4, n_layers=2),
    ]


@dataclass
class HierarchicalHVEBTConfig:
    stages: List[HVEBTStageConfig] = field(default_factory=default_3stage_configs)
    mcmc_num_steps: int = 2
    mcmc_step_size: float = 1000.0
    mcmc_step_size_learnable: bool = True
    denoising_init: str = "zeros"          # "zeros" | "random_noise" | "real_current"
    truncate_mcmc: bool = False
    weights_path: str = "clip/MobileCLIP2-S0/mobileclip2_s0.pt"
    train_encoder: bool = False            # If True, CLIP encoder is unfrozen and trained
                                           # jointly with the EBT stages. Incompatible with
                                           # preprocessed features (features change each step).
    # Adaptive MCMC ----------------------------------------------------------- #
    adaptive_mcmc: bool = False            # If True, run MCMC until convergence instead of
                                           # fixed K steps. Overrides mcmc_num_steps as max_steps.
    adaptive_mcmc_max_steps: int = 50      # Hard upper bound on MCMC iterations.
    adaptive_mcmc_tol: float = 1e-3        # Relative energy-change threshold for convergence:
                                           # |E_new - E_old| / (|E_old| + eps) < tol => stop.
    adaptive_mcmc_patience: int = 3        # How many consecutive energy increases (overshoots)
                                           # before halving the step size for this sample.
    adaptive_mcmc_alpha_decay: float = 0.5 # Factor to decay alpha on overshoot patience exceeded.
    adaptive_mcmc_step_penalty: float = 0.0  # If > 0, adds penalty * (num_steps / max_steps) to
                                           # the loss to encourage the model to converge faster.
    # Ablation ---------------------------------------------------------------- #
    disable_cross_attn: bool = False       # If True, no parent KV conditioning between stages
    detach_kv: bool = True                 # If True (default), parent KV is detached so upper
                                           # stage gets no gradient from lower stage's loss.
                                           # If False, gradients flow through cross-attn KV.
    # Bottom-up loss ---------------------------------------------------------- #
    bottom_up_loss: bool = False           # If True: only the finest active stage has loss;
                                           # gradient flows upward through non-detached KV.
                                           # Upper stages become learned latents (no own loss).
                                           # Implies detach_kv=False, truncate_mcmc=True.
                                           # When decoder_enabled: decoder pixel loss is the
                                           # SOLE objective (no feature-space loss at all).
    # Progressive training ---------------------------------------------------- #
    progressive: bool = False              # If True, stages are activated one by one top→down.
    progressive_steps_per_stage: int = 500 # Steps to train each stage before activating the next.
    # Decoder ---------------------------------------------------------------- #
    decoder_enabled: bool = False
    decoder_out_size: int = 256
    decoder_loss_weight: float = 1.0


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #


class HierarchicalHVEBT(nn.Module):
    def __init__(self, cfg: HierarchicalHVEBTConfig):
        super().__init__()
        if len(cfg.stages) == 0:
            raise ValueError("Need at least one stage")
        self.cfg = cfg
        # Progressive: start with apex only; otherwise all active.
        self._num_active_stages: int = 1 if cfg.progressive else len(cfg.stages)

        # Validate parent-child geometry: each finer stage must have
        # spatial dims >= the coarser stage above it.
        for i in range(1, len(cfg.stages)):
            child = cfg.stages[i - 1]
            parent = cfg.stages[i]
            if child.H < parent.H or child.W < parent.W:
                raise ValueError(
                    f"Stage {i-1} ({child.H}x{child.W}) must be >= "
                    f"parent stage {i} ({parent.H}x{parent.W})"
                )

        stage_names = tuple(s.clip_stage_name for s in cfg.stages)
        if cfg.weights_path:
            self.encoder = MobileClipMultiStageEncoder(
                weights_path=cfg.weights_path,
                return_stages=stage_names,
                trainable=cfg.train_encoder,
            )
        else:
            self.encoder = None  # preprocessed features mode

        stages: List[HVEBTStage] = []
        for i, sc in enumerate(cfg.stages):
            if cfg.disable_cross_attn or i == len(cfg.stages) - 1:
                # Apex (coarsest) stage or ablation mode: no cross-attention
                stages.append(HVEBTStage(sc))
            else:
                # Every non-apex stage cross-attends to the coarser stage above
                parent_sc = cfg.stages[i + 1]
                stages.append(HVEBTStage(
                    sc,
                    parent_channels=parent_sc.clip_channels,
                    parent_HW=(parent_sc.H, parent_sc.W),
                ))
        self.stages = nn.ModuleList(stages)

        # One learnable alpha per stage.
        self.alphas = nn.ParameterList([
            nn.Parameter(
                torch.tensor(float(cfg.mcmc_step_size)),
                requires_grad=cfg.mcmc_step_size_learnable,
            )
            for _ in cfg.stages
        ])

        # Optional decoder on the finest stage's prediction.
        self.decoder: Optional[PixelDecoder] = None
        if cfg.decoder_enabled:
            base_sc = cfg.stages[0]
            self.decoder = PixelDecoder(
                in_channels=base_sc.clip_channels,
                in_HW=(base_sc.H, base_sc.W),
                out_size=cfg.decoder_out_size,
            )

    # ------------------------------------------------------------------ #
    # progressive training
    # ------------------------------------------------------------------ #

    @property
    def num_active_stages(self) -> int:
        return self._num_active_stages

    def set_active_stages(self, n: int) -> None:
        """Set how many stages are active (1 = apex only, len = all).
        Stages activate top-down: apex is always active."""
        n = max(1, min(n, len(self.stages)))
        self._num_active_stages = n

    def active_stage_indices(self) -> List[int]:
        """Return indices of currently active stages (top-down order).
        With N total stages and K active:
          active = [N-1, N-2, ..., N-K]  (apex first, then finer)
        """
        N = len(self.stages)
        K = self._num_active_stages
        return list(reversed(range(N - K, N)))

    def update_progressive(self, step: int) -> Optional[int]:
        """Call each step when cfg.progressive=True.
        Returns the newly activated stage index, or None."""
        if not self.cfg.progressive:
            return None
        N = len(self.stages)
        # After 0 steps: 1 active (apex). After progressive_steps_per_stage: 2, etc.
        desired = min(N, 1 + step // self.cfg.progressive_steps_per_stage)
        if desired > self._num_active_stages:
            self._num_active_stages = desired
            # Return the newly activated stage index (finest of active set)
            return N - desired
        return None

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #

    def encode(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args: video (B, T+1, 3, Hi, Wi) in [0, 1].
        Returns dict of stage_name -> (B, T+1, C, H, W).
        """
        if self.encoder is None:
            raise RuntimeError(
                "Encoder not loaded (weights_path was empty). "
                "Pass precomputed features via forward_loss(video, features=...)."
            )
        feats = self.encoder.encode_video(video)
        return {k: self._ensure_5d(v).float() for k, v in feats.items()}

    @staticmethod
    def _ensure_5d(x: torch.Tensor) -> torch.Tensor:
        """Normalize feature shape: (B, T, C) -> (B, T, C, 1, 1) for pooled vectors."""
        if x.dim() == 3:
            return x.unsqueeze(-1).unsqueeze(-1)
        return x

    # ------------------------------------------------------------------ #
    # MCMC for a single stage
    # ------------------------------------------------------------------ #

    def _init_pred(self, real_next: torch.Tensor) -> torch.Tensor:
        if self.cfg.denoising_init == "zeros":
            return torch.zeros_like(real_next)
        if self.cfg.denoising_init == "random_noise":
            return torch.randn_like(real_next)
        if self.cfg.denoising_init == "real_current":
            return real_next.clone().detach()
        raise ValueError(self.cfg.denoising_init)

    def _mcmc_for_stage(
        self,
        stage: HVEBTStage,
        alpha_param: torch.Tensor,
        real_ctx: torch.Tensor,
        init_pred: torch.Tensor,
        parent_ctx: Optional[torch.Tensor],
        learning: bool,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        preds: List[torch.Tensor] = []
        energies: List[torch.Tensor] = []
        alpha = torch.clamp(alpha_param, min=1e-4)
        K = self.cfg.mcmc_num_steps
        pred = init_pred
        with torch.set_grad_enabled(True):
            for step in range(K):
                pred = pred.detach().requires_grad_(True)
                energy = stage(real_ctx, pred, parent_context=parent_ctx)
                energies.append(energy)
                create_graph = learning and (
                    not self.cfg.truncate_mcmc or step == K - 1
                )
                grad = torch.autograd.grad(
                    [energy.sum()], [pred], create_graph=create_graph
                )[0]
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    raise RuntimeError(f"NaN/Inf MCMC grad in stage with H={stage.cfg.H}")
                pred = pred - alpha * grad
                preds.append(pred)
        return preds, energies

    def _mcmc_adaptive_for_stage(
        self,
        stage: HVEBTStage,
        alpha_param: torch.Tensor,
        real_ctx: torch.Tensor,
        init_pred: torch.Tensor,
        parent_ctx: Optional[torch.Tensor],
        learning: bool,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """
        Adaptive MCMC: run until energy converges or max_steps is reached.

        Strategy:
          1. Convergence phase (no grad graph): iterate until relative energy
             change < tolerance, or energy consistently overshoots (increases).
          2. Final step (with grad graph): one last step from the converged
             point to build the computation graph needed for training.

        Returns (preds, energies, num_steps) where:
          - preds has 2 elements: [first_step_pred, final_pred] for metrics
          - energies has 2 elements: [initial_energy, final_energy]
          - num_steps is total iterations used (convergence + final)
        """
        max_steps = self.cfg.adaptive_mcmc_max_steps
        tol = self.cfg.adaptive_mcmc_tol
        patience = self.cfg.adaptive_mcmc_patience
        alpha_decay = self.cfg.adaptive_mcmc_alpha_decay

        alpha = torch.clamp(alpha_param, min=1e-4)
        pred = init_pred
        prev_energy_val: Optional[float] = None
        overshoot_count = 0
        converge_steps = 0
        first_pred: Optional[torch.Tensor] = None
        first_energy: Optional[torch.Tensor] = None

        # Phase 1: converge without building the full grad graph.
        with torch.set_grad_enabled(True):
            for step in range(max_steps - 1):
                pred = pred.detach().requires_grad_(True)
                energy = stage(real_ctx, pred, parent_context=parent_ctx)
                energy_val = energy.sum().item()

                # Save first step for metrics
                if step == 0:
                    first_energy = energy

                # Convergence check (after first step)
                if prev_energy_val is not None:
                    rel_change = abs(energy_val - prev_energy_val) / (
                        abs(prev_energy_val) + 1e-8
                    )
                    if rel_change < tol:
                        converge_steps = step
                        break
                    # Overshoot: energy increased
                    if energy_val > prev_energy_val:
                        overshoot_count += 1
                        if overshoot_count >= patience:
                            alpha = alpha * alpha_decay
                            overshoot_count = 0
                    else:
                        overshoot_count = 0

                prev_energy_val = energy_val

                grad = torch.autograd.grad(
                    [energy.sum()], [pred], create_graph=False
                )[0]
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    converge_steps = step
                    break
                pred = (pred - alpha * grad).detach()

                # Save first step prediction for metrics
                if step == 0:
                    first_pred = pred.clone()
            else:
                converge_steps = max_steps - 1

        # Phase 2: final step WITH grad graph for training.
        pred = pred.detach().requires_grad_(True)
        energy = stage(real_ctx, pred, parent_context=parent_ctx)

        create_graph = learning  # always build graph on final step
        grad = torch.autograd.grad(
            [energy.sum()], [pred], create_graph=create_graph
        )[0]
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            final_pred = pred  # stay put
        else:
            final_pred = pred - alpha * grad

        # Build return lists: [first_step, final] for consistent metrics
        if first_pred is None:
            first_pred = final_pred  # only 1 step was taken
        if first_energy is None:
            first_energy = energy

        preds = [first_pred, final_pred]
        energies = [first_energy, energy]
        num_steps = converge_steps + 1  # convergence steps + 1 final step
        return preds, energies, num_steps

    # ------------------------------------------------------------------ #
    # full forward / loss
    # ------------------------------------------------------------------ #

    def forward_loss(
        self,
        video: torch.Tensor,                                # (B, T+1, 3, Hi, Wi) in [0,1]
        features: Optional[Dict[str, torch.Tensor]] = None, # precomputed {stage_name: (B, T+1, C, H, W)}
        learning: bool = True,
    ) -> Dict[str, object]:
        """
        Args:
            video:    Raw video tensor — always required (used as decoder target
                      and as encoder input when features are not provided).
            features: Optional precomputed CLIP features. If None, extracted via
                      the CLIP encoder from `video`. Supports 3D (B,T,C) for
                      pooled stages (auto-unsqueezed to 5D).
            learning: If True, create_graph for MCMC unroll (training mode).
        """
        if features is not None:
            feats_dict = {k: self._ensure_5d(v) for k, v in features.items()}
        else:
            feats_dict = self.encode(video)
        return self._forward_loss_impl(feats_dict, video=video, learning=learning)

    def _forward_loss_impl(
        self,
        feats_dict: Dict[str, torch.Tensor],
        video: torch.Tensor,
        learning: bool = True,
    ) -> Dict[str, object]:

        per_stage, total_loss = self._mcmc_sequential(feats_dict, learning)

        out: Dict[str, object] = {
            "loss_energy": total_loss,
            "per_stage": per_stage,
        }

        # Find finest active stage index
        finest_active = min(i for i, s in enumerate(per_stage) if s is not None)

        # Decoder branch (decoder is built for stage 0; skip if s0 not yet active)
        decoder_can_fire = (self.decoder is not None
                            and finest_active == 0
                            and per_stage[0] is not None)
        if decoder_can_fire:
            if self.cfg.bottom_up_loss:
                # Bottom-up: decoder is THE loss. Input is NOT detached so
                # gradient flows through MCMC pred → cross-attn KV → all stages.
                base_pred = per_stage[0]["final_pred_live"]
            else:
                # Standard: decoder trained independently on detached features.
                base_pred = per_stage[0]["final_pred"]
            decoded = self.decoder(base_pred)                      # (B, T, 3, S, S)
            target_rgb = video[:, 1:]
            decoder_loss = F.l1_loss(decoded, target_rgb)
            out["loss_decoder"] = decoder_loss
            out["decoded_rgb"] = decoded.detach()
            out["target_rgb"] = target_rgb.detach()
            if self.cfg.bottom_up_loss:
                # Decoder pixel loss is the sole objective.
                out["loss_total"] = decoder_loss
            else:
                out["loss_total"] = total_loss + self.cfg.decoder_loss_weight * decoder_loss
        else:
            out["loss_total"] = total_loss

        return out

    # ------------------------------------------------------------------ #
    # MCMC schedules
    # ------------------------------------------------------------------ #

    def _mcmc_sequential(
        self,
        feats_dict: Dict[str, torch.Tensor],
        learning: bool,
    ) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Sequential schedule: each stage runs ALL K MCMC steps to convergence
        before passing its fully-relaxed prediction as KV to the next finer stage.
        Respects progressive training (only active stages participate).

        When bottom_up_loss=True:
          - KV is never detached (gradient flows upward).
          - Only last MCMC step contributes (truncated).
          - Only the finest active stage computes feature loss (or none if
            decoder will provide the loss).
        """
        N = len(self.stages)
        per_stage: List[Optional[Dict[str, torch.Tensor]]] = [None] * N
        prev_pred: Optional[torch.Tensor] = None
        total_loss = next(iter(feats_dict.values())).new_zeros(())
        active = self.active_stage_indices()  # top-down order

        bu = self.cfg.bottom_up_loss
        # In bottom-up: finest active stage index (last in the top-down iteration)
        finest_active_idx = active[-1] if active else -1
        # In bottom-up + decoder: skip feature losses only when the decoder
        # can actually fire (stage 0 must be active). During progressive
        # warmup stage 0 isn't active yet, so the finest active stage's
        # feature loss serves as fallback.
        bu_skip_all_loss = bu and self.cfg.decoder_enabled and (0 in active)

        for i in active:
            stage = self.stages[i]
            sc = self.cfg.stages[i]
            feats = feats_dict[sc.clip_stage_name]
            real_ctx = feats[:, :-1]
            real_gt = feats[:, 1:]
            init_pred = self._init_pred(real_gt)

            parent_ctx = None
            if stage.use_cross_attn and prev_pred is not None:
                parent_ctx = prev_pred.detach() if self.cfg.detach_kv else prev_pred

            if self.cfg.adaptive_mcmc:
                preds, energies, num_steps = self._mcmc_adaptive_for_stage(
                    stage, self.alphas[i], real_ctx, init_pred,
                    parent_ctx=parent_ctx, learning=learning,
                )
            else:
                preds, energies = self._mcmc_for_stage(
                    stage, self.alphas[i], real_ctx, init_pred,
                    parent_ctx=parent_ctx, learning=learning,
                )
                num_steps = len(preds)

            # Loss computation
            K = len(preds)
            compute_loss = True
            if bu:
                # Bottom-up: only finest active stage gets feature loss
                # (and even that is skipped when decoder provides the loss)
                compute_loss = (i == finest_active_idx) and (not bu_skip_all_loss)

            if compute_loss:
                if self.cfg.truncate_mcmc or self.cfg.adaptive_mcmc:
                    stage_loss = F.smooth_l1_loss(preds[-1], real_gt)
                else:
                    stage_loss = sum(F.smooth_l1_loss(p, real_gt) for p in preds) / K
                total_loss = total_loss + stage_loss
            else:
                stage_loss = preds[-1].new_zeros(())

            # Step penalty: encourage model to converge in fewer steps
            if self.cfg.adaptive_mcmc and self.cfg.adaptive_mcmc_step_penalty > 0:
                step_ratio = num_steps / self.cfg.adaptive_mcmc_max_steps
                total_loss = total_loss + self.cfg.adaptive_mcmc_step_penalty * step_ratio

            with torch.no_grad():
                init_recon = F.smooth_l1_loss(preds[0].detach(), real_gt)
                final_recon = F.smooth_l1_loss(preds[-1].detach(), real_gt)
                init_e = energies[0].mean()
                final_e = energies[-1].mean()
                baseline = F.smooth_l1_loss(real_ctx, real_gt)
            per_stage[i] = {
                "loss": stage_loss.detach(),
                "init_recon": init_recon,
                "final_recon": final_recon,
                "init_energy": init_e,
                "final_energy": final_e,
                "energy_gap": init_e - final_e,
                "alpha": self.alphas[i].detach(),
                "baseline_copy_last": baseline,
                "final_pred": preds[-1].detach(),
                "final_pred_live": preds[-1],  # keeps grad graph for bottom-up decoder
                "real_gt": real_gt.detach(),
                "mcmc_steps_used": num_steps,
            }
            prev_pred = preds[-1]

        return per_stage, total_loss
