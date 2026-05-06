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

import os
import warnings
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

    # ----- VQ mode (Plan V2) ------------------------------------------------ #
    vq_mode: bool = False                  # Master switch for VQ classification mode.
                                           # Per-stage codebook sizes/paths must be set on
                                           # each HVEBTStageConfig (or via vq_codebook_dir).
    vq_codebook_dir: str = ""              # Directory with codebook_<stage>.pt files. When
                                           # set, populates per-stage vq_codebook_size and
                                           # vq_codebook_path automatically (auto-sizing).
    vq_use_precomputed_targets: bool = False  # If True, forward_loss expects target indices
                                           # in the batch (one tensor per stage). Skips CLIP.
    vq_no_features: bool = False           # If True, real CLIP features are NOT used: real_ctx
                                           # comes from quantized lookup of precomputed indices.
                                           # Implies vq_use_precomputed_targets.
    vq_soft_targets: bool = False          # Soft CE on cosine-sim distribution to codebook.
    vq_soft_temperature: float = 0.1
    vq_target_recompute_every: int = 0     # 0 = never. >0 = re-quantize on-the-fly every N steps
                                           # (only when CLIP features are present).
    allow_stale_targets: bool = False      # Permit train_encoder + precomputed targets without
                                           # any recomputation/EMA. Off-by-default safety guard.
    # ----- VQ maintenance (Phase 5) ----------------------------------------- #
    vq_dead_code_threshold: float = 1e-4
    vq_dead_code_check_every: int = 0      # 0 disables dead-code reset
    vq_merge_sim_threshold: float = 0.0    # 0 disables merging
    vq_merge_check_every: int = 0
    vq_usage_decay: float = 0.99           # Decay for usage EMA when track_usage is on


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

        # ----- VQ auto-wiring (Plan V2 §C.2/C.3) ------------------------- #
        if cfg.vq_mode:
            self._populate_vq_per_stage_from_dir()
            self._validate_vq_config()

        stage_names = tuple(s.clip_stage_name for s in cfg.stages)
        # CLIP encoder is loaded only when actually needed (i.e. when CLIP
        # features will be requested at training time). In CLIP-free VQ mode
        # (vq_no_features) we skip the encoder entirely even if a weights
        # path was provided.
        load_encoder = bool(cfg.weights_path) and not cfg.vq_no_features
        if load_encoder:
            self.encoder = MobileClipMultiStageEncoder(
                weights_path=cfg.weights_path,
                return_stages=stage_names,
                trainable=cfg.train_encoder,
            )
        else:
            self.encoder = None  # preprocessed-features mode or CLIP-free VQ mode

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

        # Global step counter for VQ maintenance schedules.
        self.register_buffer("_global_step_buf", torch.zeros((), dtype=torch.long), persistent=False)

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
    # VQ wiring (Plan V2)
    # ------------------------------------------------------------------ #

    def _populate_vq_per_stage_from_dir(self) -> None:
        """If `vq_codebook_dir` is set, fill per-stage `vq_codebook_path` and
        infer `vq_codebook_size` from the saved tensor shapes (auto-sizing)."""
        d = self.cfg.vq_codebook_dir
        if not d:
            return
        if not os.path.isdir(d):
            raise FileNotFoundError(f"vq_codebook_dir not found: {d}")
        for sc in self.cfg.stages:
            cb_path = os.path.join(d, f"codebook_{sc.clip_stage_name}.pt")
            if not os.path.isfile(cb_path):
                # User may have set vq_codebook_path manually; allow that.
                if not sc.vq_codebook_path:
                    raise FileNotFoundError(
                        f"Codebook for stage {sc.clip_stage_name} missing: {cb_path}"
                    )
                continue
            sc.vq_codebook_path = cb_path
            cb = torch.load(cb_path, map_location="cpu", weights_only=True)
            K, C = cb.shape
            if C != sc.clip_channels:
                raise ValueError(
                    f"Codebook channel mismatch for stage {sc.clip_stage_name}: "
                    f"file C={C}, stage clip_channels={sc.clip_channels}"
                )
            if sc.vq_codebook_size and sc.vq_codebook_size != K:
                raise ValueError(
                    f"Stage {sc.clip_stage_name}: vq_codebook_size={sc.vq_codebook_size} "
                    f"contradicts codebook file K={K}"
                )
            sc.vq_codebook_size = K

    def _validate_vq_config(self) -> None:
        cfg = self.cfg
        for sc in cfg.stages:
            if sc.vq_codebook_size <= 0:
                raise ValueError(
                    f"vq_mode requires vq_codebook_size > 0 on every stage; "
                    f"stage {sc.clip_stage_name} has 0. "
                    f"Set --vq_codebook_dir or per-stage paths."
                )
        if cfg.vq_no_features and not cfg.vq_use_precomputed_targets:
            raise ValueError("vq_no_features=True requires vq_use_precomputed_targets=True")
        if cfg.vq_use_precomputed_targets and cfg.train_encoder \
                and cfg.vq_target_recompute_every == 0 \
                and not cfg.allow_stale_targets:
            raise ValueError(
                "Precomputed VQ targets become stale when training the encoder. "
                "Either set vq_target_recompute_every > 0, enable per-stage "
                "vq_ema_decay > 0, or pass allow_stale_targets=True."
            )
        if cfg.train_encoder and all(sc.vq_ema_decay == 0.0 for sc in cfg.stages):
            warnings.warn(
                "train_encoder=True with frozen codebook (vq_ema_decay=0). "
                "Codebook will not follow encoder drift; consider EMA updates.",
                stacklevel=2,
            )

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
    # VQ MCMC (Plan V2 §C.4)
    # ------------------------------------------------------------------ #

    def _init_logits_vq(
        self,
        stage_idx: int,
        B: int,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
        prev_targets: Optional[torch.Tensor] = None,
        target_for_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Initialize VQ logits according to cfg.denoising_init.

        Args:
            prev_targets:     (B, T, H, W) long, previous-frame indices
                              (used by 'real_current').
            target_for_init:  (B, T, H, W) long, *current* target indices,
                              used only by debug 'real_target' init.
        Returns:
            (B, T*H*W, K) logits.
        """
        sc = self.cfg.stages[stage_idx]
        K = sc.vq_codebook_size
        N = T * sc.H * sc.W
        init = self.cfg.denoising_init
        if init == "zeros":
            return torch.zeros(B, N, K, device=device, dtype=dtype)
        if init == "random_noise":
            return torch.randn(B, N, K, device=device, dtype=dtype)
        if init == "real_current":
            if prev_targets is None:
                # Fall back to zeros silently for first stage / no targets.
                return torch.zeros(B, N, K, device=device, dtype=dtype)
            scale = 5.0  # gives softmax peak ~99% on chosen index
            base = torch.zeros(B, N, K, device=device, dtype=dtype)
            flat = prev_targets.reshape(B, N).to(device).long()
            base.scatter_(2, flat.unsqueeze(-1), scale)
            return base
        if init == "real_target":  # debug only
            if target_for_init is None:
                raise ValueError("denoising_init='real_target' requires target_for_init")
            scale = 5.0
            base = torch.zeros(B, N, K, device=device, dtype=dtype)
            flat = target_for_init.reshape(B, N).to(device).long()
            base.scatter_(2, flat.unsqueeze(-1), scale)
            return base
        raise ValueError(f"Unknown denoising_init for VQ: {init!r}")

    def _mcmc_for_stage_vq(
        self,
        stage: HVEBTStage,
        alpha_param: torch.Tensor,
        real_ctx: torch.Tensor,
        init_logits: torch.Tensor,
        parent_ctx: Optional[torch.Tensor],
        learning: bool,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        VQ MCMC: optimize in logit space.

        Returns:
            preds_for_metrics: list of (B,T,C,H,W) detached features per step
                               (last entry kept for parent-context handoff).
            energies:          list of (B, N) energies per step (last retains graph).
            final_logits:      (B, N, K) final logits with grad-graph for CE loss.
        """
        K = self.cfg.mcmc_num_steps
        alpha = torch.clamp(alpha_param, min=1e-4)
        T = real_ctx.shape[1]
        clamp_v = stage.cfg.vq_logit_clamp

        preds_for_metrics: List[torch.Tensor] = []
        energies: List[torch.Tensor] = []
        logits = init_logits

        with torch.set_grad_enabled(True):
            for step in range(K):
                logits = logits.detach().requires_grad_(True)
                pred_feats = stage.features_from_logits(logits, T=T)
                energy = stage(real_ctx, pred_feats, parent_context=parent_ctx)
                energies.append(energy)
                preds_for_metrics.append(pred_feats.detach())

                create_graph = learning and (
                    not self.cfg.truncate_mcmc or step == K - 1
                )
                grad = torch.autograd.grad(
                    [energy.sum()], [logits], create_graph=create_graph
                )[0]
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    raise RuntimeError(f"NaN/Inf VQ MCMC grad in stage H={stage.cfg.H}")
                logits = logits - alpha * grad
                if clamp_v > 0:
                    logits = logits.clamp(-clamp_v, clamp_v)

        return preds_for_metrics, energies, logits

    def _mcmc_adaptive_for_stage_vq(
        self,
        stage: HVEBTStage,
        alpha_param: torch.Tensor,
        real_ctx: torch.Tensor,
        init_logits: torch.Tensor,
        parent_ctx: Optional[torch.Tensor],
        learning: bool,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, int]:
        """
        Adaptive VQ MCMC: same convergence detector as continuous mode but
        the optimization variable is logits (B, N, K).
        Returns (preds_for_metrics, energies, final_logits, num_steps).
        """
        max_steps = self.cfg.adaptive_mcmc_max_steps
        tol = self.cfg.adaptive_mcmc_tol
        patience = self.cfg.adaptive_mcmc_patience
        alpha_decay = self.cfg.adaptive_mcmc_alpha_decay
        clamp_v = stage.cfg.vq_logit_clamp
        T = real_ctx.shape[1]

        alpha = torch.clamp(alpha_param, min=1e-4)
        logits = init_logits
        prev_energy_val: Optional[float] = None
        overshoot_count = 0
        converge_steps = 0
        first_pred: Optional[torch.Tensor] = None
        first_energy: Optional[torch.Tensor] = None

        with torch.set_grad_enabled(True):
            for step in range(max_steps - 1):
                logits = logits.detach().requires_grad_(True)
                pred_feats = stage.features_from_logits(logits, T=T)
                energy = stage(real_ctx, pred_feats, parent_context=parent_ctx)
                energy_val = energy.sum().item()
                if step == 0:
                    first_energy = energy
                if prev_energy_val is not None:
                    rel = abs(energy_val - prev_energy_val) / (abs(prev_energy_val) + 1e-8)
                    if rel < tol:
                        converge_steps = step
                        break
                    if energy_val > prev_energy_val:
                        overshoot_count += 1
                        if overshoot_count >= patience:
                            alpha = alpha * alpha_decay
                            overshoot_count = 0
                    else:
                        overshoot_count = 0
                prev_energy_val = energy_val
                grad = torch.autograd.grad([energy.sum()], [logits], create_graph=False)[0]
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    converge_steps = step
                    break
                logits = (logits - alpha * grad).detach()
                if clamp_v > 0:
                    logits = logits.clamp(-clamp_v, clamp_v)
                if step == 0:
                    first_pred = pred_feats.detach()
            else:
                converge_steps = max_steps - 1

        # Final step WITH graph
        logits = logits.detach().requires_grad_(True)
        pred_feats = stage.features_from_logits(logits, T=T)
        energy = stage(real_ctx, pred_feats, parent_context=parent_ctx)
        create_graph = learning
        grad = torch.autograd.grad([energy.sum()], [logits], create_graph=create_graph)[0]
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            final_logits = logits
        else:
            final_logits = logits - alpha * grad
            if clamp_v > 0:
                final_logits = final_logits.clamp(-clamp_v, clamp_v)
        final_pred = stage.features_from_logits(final_logits, T=T).detach()

        if first_pred is None:
            first_pred = final_pred
        if first_energy is None:
            first_energy = energy

        preds = [first_pred, final_pred]
        energies = [first_energy, energy]
        num_steps = converge_steps + 1
        return preds, energies, final_logits, num_steps

    # ------------------------------------------------------------------ #
    # full forward / loss
    # ------------------------------------------------------------------ #

    def forward_loss(
        self,
        video: torch.Tensor,                                # (B, T+1, 3, Hi, Wi) in [0,1]
        features: Optional[Dict[str, torch.Tensor]] = None, # precomputed {stage_name: (B, T+1, C, H, W)}
        vq_targets: Optional[Dict[str, torch.Tensor]] = None,  # {stage_name: (B, T+1, H, W) long}
        learning: bool = True,
    ) -> Dict[str, object]:
        """
        Args:
            video:      Raw video tensor — used as decoder target and as encoder
                        input when neither features nor vq targets are provided.
            features:   Optional precomputed CLIP features. If None and CLIP-free
                        VQ mode is not enabled, extracted via the CLIP encoder.
            vq_targets: Optional precomputed per-stage codebook indices. Required
                        when cfg.vq_use_precomputed_targets=True.
            learning:   If True, create_graph for MCMC unroll (training mode).
        """
        # ---- Precomputed VQ targets path (CLIP optional) ----------------- #
        if self.cfg.vq_mode and self.cfg.vq_use_precomputed_targets:
            if vq_targets is None:
                raise ValueError(
                    "vq_use_precomputed_targets=True but no vq_targets dict provided."
                )
            feats_dict: Optional[Dict[str, torch.Tensor]] = None
            if features is not None:
                feats_dict = {k: self._ensure_5d(v) for k, v in features.items()}
            elif not self.cfg.vq_no_features and self.encoder is not None:
                feats_dict = self.encode(video)
            # else: vq_no_features → real_ctx from quantized lookup of targets.
            return self._forward_loss_impl(
                feats_dict, video=video, learning=learning, vq_targets=vq_targets,
            )
        # ---- Standard / VQ-on-the-fly path ------------------------------- #
        if features is not None:
            feats_dict = {k: self._ensure_5d(v) for k, v in features.items()}
        else:
            feats_dict = self.encode(video)
        return self._forward_loss_impl(
            feats_dict, video=video, learning=learning, vq_targets=None,
        )

    def _forward_loss_impl(
        self,
        feats_dict: Optional[Dict[str, torch.Tensor]],
        video: torch.Tensor,
        learning: bool = True,
        vq_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, object]:

        per_stage, total_loss = self._mcmc_sequential(
            feats_dict, learning, vq_targets=vq_targets,
        )

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
                # Bottom-up: decoder pixel loss is the primary objective and its
                # gradient flows through live MCMC preds → cross-attn KV → all
                # stages. In VQ mode, per-stage CE is also kept (see VQ-mode
                # invariant in _mcmc_sequential) because pixel L1 alone cannot
                # train the codebook classification. In continuous mode,
                # total_loss is 0 here (per-stage feature losses are suppressed)
                # so this reduces to "decoder is sole loss".
                out["loss_total"] = total_loss + self.cfg.decoder_loss_weight * decoder_loss
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
        feats_dict: Optional[Dict[str, torch.Tensor]],
        learning: bool,
        vq_targets: Optional[Dict[str, torch.Tensor]] = None,
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

        When cfg.vq_mode=True: each stage runs MCMC in logit space (B, N, K)
        and uses cross-entropy loss against codebook indices (provided in
        vq_targets if precomputed, else computed on-the-fly from feats).
        """
        N = len(self.stages)
        per_stage: List[Optional[Dict[str, torch.Tensor]]] = [None] * N
        prev_pred: Optional[torch.Tensor] = None

        # zero-tensor anchor for total_loss
        anchor = None
        if feats_dict is not None and len(feats_dict) > 0:
            anchor = next(iter(feats_dict.values()))
        elif vq_targets is not None and len(vq_targets) > 0:
            # any tensor on the right device for new_zeros — float on the alpha device
            anchor = self.alphas[0]
        else:
            raise RuntimeError("No feats_dict and no vq_targets provided")
        total_loss = anchor.new_zeros((), dtype=torch.float32) if anchor.dtype != torch.float32 \
            else anchor.new_zeros(())

        active = self.active_stage_indices()  # top-down order
        bu = self.cfg.bottom_up_loss
        finest_active_idx = active[-1] if active else -1
        bu_skip_all_loss = bu and self.cfg.decoder_enabled and (0 in active)

        vq_mode = self.cfg.vq_mode

        for i in active:
            stage = self.stages[i]
            sc = self.cfg.stages[i]

            # ---- Resolve real_ctx / real_gt / targets per mode ----------- #
            real_ctx: torch.Tensor
            real_gt: Optional[torch.Tensor] = None
            targets_full: Optional[torch.Tensor] = None  # (B, T+1, H, W) long, VQ only

            if vq_mode and vq_targets is not None and sc.clip_stage_name in vq_targets:
                tgt_all = vq_targets[sc.clip_stage_name].long()  # (B, T+1, H, W)
                targets_full = tgt_all
                if feats_dict is not None and sc.clip_stage_name in feats_dict:
                    feats = feats_dict[sc.clip_stage_name]
                    real_ctx = feats[:, :-1]
                    real_gt = feats[:, 1:]
                else:
                    # CLIP-free: build real_ctx from quantized indices.
                    real_ctx = stage.features_from_indices(tgt_all[:, :-1])
                    # Real ground-truth features are unavailable; we still
                    # need a numerical proxy for init_recon/baseline metrics.
                    real_gt = stage.features_from_indices(tgt_all[:, 1:])
            else:
                if feats_dict is None or sc.clip_stage_name not in feats_dict:
                    raise RuntimeError(
                        f"No features for stage {sc.clip_stage_name} (and no precomputed targets)."
                    )
                feats = feats_dict[sc.clip_stage_name]
                real_ctx = feats[:, :-1]
                real_gt = feats[:, 1:]

            parent_ctx = None
            if stage.use_cross_attn and prev_pred is not None:
                parent_ctx = prev_pred.detach() if self.cfg.detach_kv else prev_pred

            # ================================================================
            # VQ MODE
            # ================================================================
            if vq_mode and stage.vq is not None:
                B, T = real_ctx.shape[0], real_ctx.shape[1]
                # Build target indices for the prediction window (T target frames).
                if targets_full is not None:
                    target_idx = targets_full[:, 1:].to(real_ctx.device)  # (B, T, H, W)
                    prev_idx = targets_full[:, :-1].to(real_ctx.device)
                else:
                    # On-the-fly: quantize real_gt CLIP features.
                    target_idx = stage.quantize_features(real_gt)
                    prev_idx = stage.quantize_features(real_ctx)

                init_logits = self._init_logits_vq(
                    i, B=B, T=T, device=real_ctx.device, dtype=real_ctx.dtype,
                    prev_targets=prev_idx, target_for_init=target_idx,
                )

                if self.cfg.adaptive_mcmc:
                    preds, energies, final_logits, num_steps = \
                        self._mcmc_adaptive_for_stage_vq(
                            stage, self.alphas[i], real_ctx, init_logits,
                            parent_ctx=parent_ctx, learning=learning,
                        )
                else:
                    preds, energies, final_logits = self._mcmc_for_stage_vq(
                        stage, self.alphas[i], real_ctx, init_logits,
                        parent_ctx=parent_ctx, learning=learning,
                    )
                    num_steps = len(preds)

                K_logits = final_logits.shape[-1]
                target_flat = target_idx.reshape(-1)

                # ---- Loss --------------------------------------------------
                # VQ-MODE INVARIANT: per-stage CE supervision is REQUIRED for the
                # codebook classification to train at all. Decoder pixel L1 alone
                # cannot drive a categorical distribution through softmax @ frozen
                # codebook (the gradient is far too weak / 1/K-suppressed). So in
                # VQ mode we ALWAYS compute CE on every active stage, regardless
                # of bottom_up_loss / decoder. The decoder loss (when enabled)
                # is added ON TOP via _forward_loss_impl.
                compute_loss = True

                if compute_loss:
                    if self.cfg.vq_soft_targets:
                        # Soft CE: cosine similarity between real features and codebook.
                        with torch.no_grad():
                            B2, T2, C2, H2, W2 = real_gt.shape
                            flat_gt = real_gt.permute(0, 1, 3, 4, 2).reshape(-1, C2)
                            sim = F.cosine_similarity(
                                flat_gt.unsqueeze(1),
                                stage.vq.weight.unsqueeze(0),
                                dim=-1,
                            )  # (N, K)
                            soft_t = F.softmax(sim / self.cfg.vq_soft_temperature, dim=-1)
                        log_p = F.log_softmax(final_logits.reshape(-1, K_logits), dim=-1)
                        stage_loss = -(soft_t * log_p).sum(dim=-1).mean()
                    else:
                        stage_loss = F.cross_entropy(
                            final_logits.reshape(-1, K_logits), target_flat,
                        )
                    total_loss = total_loss + stage_loss
                else:
                    stage_loss = final_logits.new_zeros(())

                if self.cfg.adaptive_mcmc and self.cfg.adaptive_mcmc_step_penalty > 0:
                    step_ratio = num_steps / self.cfg.adaptive_mcmc_max_steps
                    total_loss = total_loss + self.cfg.adaptive_mcmc_step_penalty * step_ratio

                # ---- Metrics ----------------------------------------------
                with torch.no_grad():
                    init_recon = F.smooth_l1_loss(preds[0].detach(), real_gt)
                    final_recon = F.smooth_l1_loss(preds[-1].detach(), real_gt)
                    init_e = energies[0].mean()
                    final_e = energies[-1].mean()
                    baseline = F.smooth_l1_loss(real_ctx, real_gt)
                    entropy = stage.vq.entropy(final_logits.detach())   # (B, N)
                    pred_idx = final_logits.detach().argmax(dim=-1)     # (B, N)
                    top1_acc = (pred_idx == target_flat.reshape(pred_idx.shape)).float().mean()
                    cb_usage = float(target_flat.unique().numel()) / float(K_logits)

                    # Maintenance: usage EMA + dead-code reset / merge
                    if stage.vq.track_usage:
                        stage.vq.update_usage(target_flat, decay=self.cfg.vq_usage_decay)
                    if not stage.vq._frozen and real_gt is not None:
                        # EMA codebook update against real (or quantized-real) features
                        flat_feats = real_gt.permute(0, 1, 3, 4, 2).reshape(-1, sc.clip_channels)
                        stage.vq.update_ema(target_flat, flat_feats)
                    if (self.cfg.vq_dead_code_check_every > 0 and
                            int(self._global_step_buf.item()) % self.cfg.vq_dead_code_check_every == 0):
                        cand = real_gt.permute(0, 1, 3, 4, 2).reshape(-1, sc.clip_channels)
                        n_reset = stage.vq.reset_dead_codes(cand, self.cfg.vq_dead_code_threshold)
                        if n_reset > 0:
                            warnings.warn(
                                f"[VQ stage {sc.clip_stage_name}] reset {n_reset} dead codes",
                                stacklevel=2,
                            )
                    if (self.cfg.vq_merge_check_every > 0 and
                            int(self._global_step_buf.item()) % self.cfg.vq_merge_check_every == 0):
                        n_merge = stage.vq.merge_similar(self.cfg.vq_merge_sim_threshold)
                        if n_merge > 0:
                            warnings.warn(
                                f"[VQ stage {sc.clip_stage_name}] merged {n_merge} codes",
                                stacklevel=2,
                            )

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
                    "final_pred_live": preds[-1] if preds[-1].requires_grad else preds[-1].detach(),
                    "real_gt": real_gt.detach() if real_gt is not None else None,
                    "mcmc_steps_used": num_steps,
                    # VQ-specific
                    "ce_loss": stage_loss.detach(),
                    "entropy_mean": entropy.mean(),
                    "entropy_std": entropy.std(),
                    "top1_accuracy": top1_acc,
                    "codebook_usage": torch.tensor(cb_usage),
                    "final_logits": final_logits.detach(),
                }
                # In bottom-up VQ mode, the live final_pred carries the grad
                # graph from CE/decoder loss back through softmax→logits→energy.
                # When KV is not detached, this propagates upward.
                if self.cfg.bottom_up_loss:
                    # Re-derive a live pred_feats with grad for parent / decoder
                    live_pred = stage.features_from_logits(final_logits, T=T)
                    per_stage[i]["final_pred_live"] = live_pred
                    prev_pred = live_pred
                else:
                    prev_pred = preds[-1].detach()
                continue
            # ================================================================
            # CONTINUOUS MODE (unchanged)
            # ================================================================
            init_pred = self._init_pred(real_gt)

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

            K = len(preds)
            compute_loss = True
            if bu:
                compute_loss = (i == finest_active_idx) and (not bu_skip_all_loss)

            if compute_loss:
                if self.cfg.truncate_mcmc or self.cfg.adaptive_mcmc:
                    stage_loss = F.smooth_l1_loss(preds[-1], real_gt)
                else:
                    stage_loss = sum(F.smooth_l1_loss(p, real_gt) for p in preds) / K
                total_loss = total_loss + stage_loss
            else:
                stage_loss = preds[-1].new_zeros(())

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
                "final_pred_live": preds[-1],
                "real_gt": real_gt.detach(),
                "mcmc_steps_used": num_steps,
            }
            prev_pred = preds[-1]

        # Tick global step counter (used by maintenance schedules)
        self._global_step_buf += 1

        return per_stage, total_loss
