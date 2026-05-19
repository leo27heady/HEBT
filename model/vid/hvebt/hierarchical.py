"""
Hierarchical Video EBT (Phase 2 + Phase 3).

Stacks N HVEBTStage modules ordered finest -> coarsest (index 0 = finest).
The **prediction** (MCMC) tower runs **top-down**: from the apex (coarsest)
stage to the base (finest) stage. Each stage:
  - Has its own MCMC over its own predicted features at its encoder feature
    space (16x16, 4x4, 1x1 for the default 64x64 input).
  - Beyond the apex, cross-attends to the previous (coarser) stage's final
    MCMC prediction with **detached** KV.
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

from model.vid.hvebt.decoder import PixelDecoder
from model.vid.hvebt.hvebt import HVEBTStage, HVEBTStageConfig
from model.vid.hvebt.lightweight_encoder import (
    DEFAULT_CHANNELS,
    INPUT_SIZE,
    LightweightMultiStageEncoder,
)


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #


def default_3stage_configs() -> List[HVEBTStageConfig]:
    """
    Default 3-stage stack for 64x64 input:
        Stage 0 (finest)  : 16x16 (64 ch)
        Stage 1           : 4x4   (128 ch)
        Stage 2 (apex)    : 1x1   (256 ch)
    """
    c0, c1, c2 = DEFAULT_CHANNELS
    return [
        HVEBTStageConfig(stage_name="16x16", channels=c0, H=16, W=16,
                         embed_dim=128, n_heads=4, n_layers=2),
        HVEBTStageConfig(stage_name="4x4", channels=c1, H=4, W=4,
                         embed_dim=192, n_heads=4, n_layers=2),
        HVEBTStageConfig(stage_name="1x1", channels=c2, H=1, W=1,
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
    input_size: int = INPUT_SIZE
    # Adaptive MCMC ----------------------------------------------------------- #
    adaptive_mcmc: bool = False
    adaptive_mcmc_max_steps: int = 50
    adaptive_mcmc_tol: float = 1e-3
    adaptive_mcmc_patience: int = 3
    adaptive_mcmc_alpha_decay: float = 0.5
    adaptive_mcmc_step_penalty: float = 0.0
    # Ablation ---------------------------------------------------------------- #
    disable_cross_attn: bool = False
    detach_kv: bool = True
    # Bottom-up loss ---------------------------------------------------------- #
    bottom_up_loss: bool = False
    # Progressive training ---------------------------------------------------- #
    progressive: bool = False
    progressive_steps_per_stage: int = 500
    # Decoder ---------------------------------------------------------------- #
    decoder_enabled: bool = False
    decoder_out_size: int = 64
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
        self._num_active_stages: int = 1 if cfg.progressive else len(cfg.stages)

        expected = LightweightMultiStageEncoder.stage_shapes(cfg.input_size)
        for sc in cfg.stages:
            if sc.stage_name not in expected:
                raise ValueError(
                    f"Unknown stage_name '{sc.stage_name}'; "
                    f"expected one of {tuple(expected)}"
                )
            ec, eh, ew = expected[sc.stage_name]
            if (sc.channels, sc.H, sc.W) != (ec, eh, ew):
                raise ValueError(
                    f"Stage {sc.stage_name}: config ({sc.channels},{sc.H},{sc.W}) "
                    f"!= encoder ({ec},{eh},{ew})"
                )

        for i in range(1, len(cfg.stages)):
            child = cfg.stages[i - 1]
            parent = cfg.stages[i]
            if child.H < parent.H or child.W < parent.W:
                raise ValueError(
                    f"Stage {i-1} ({child.H}x{child.W}) must be >= "
                    f"parent stage {i} ({parent.H}x{parent.W})"
                )

        stage_names = tuple(s.stage_name for s in cfg.stages)
        self.encoder = LightweightMultiStageEncoder(
            return_stages=stage_names,
            input_size=cfg.input_size,
        )

        stages: List[HVEBTStage] = []
        for i, sc in enumerate(cfg.stages):
            if cfg.disable_cross_attn or i == len(cfg.stages) - 1:
                stages.append(HVEBTStage(sc))
            else:
                parent_sc = cfg.stages[i + 1]
                stages.append(HVEBTStage(
                    sc,
                    parent_channels=parent_sc.channels,
                    parent_HW=(parent_sc.H, parent_sc.W),
                ))
        self.stages = nn.ModuleList(stages)

        self.alphas = nn.ParameterList([
            nn.Parameter(
                torch.tensor(float(cfg.mcmc_step_size)),
                requires_grad=cfg.mcmc_step_size_learnable,
            )
            for _ in cfg.stages
        ])

        self.decoder: Optional[PixelDecoder] = None
        if cfg.decoder_enabled:
            base_sc = cfg.stages[0]
            self.decoder = PixelDecoder(
                in_channels=base_sc.channels,
                in_HW=(base_sc.H, base_sc.W),
                out_size=cfg.decoder_out_size,
            )

    @property
    def num_active_stages(self) -> int:
        return self._num_active_stages

    def set_active_stages(self, n: int) -> None:
        n = max(1, min(n, len(self.stages)))
        self._num_active_stages = n

    def active_stage_indices(self) -> List[int]:
        N = len(self.stages)
        K = self._num_active_stages
        return list(reversed(range(N - K, N)))

    def update_progressive(self, step: int) -> Optional[int]:
        if not self.cfg.progressive:
            return None
        N = len(self.stages)
        desired = min(N, 1 + step // self.cfg.progressive_steps_per_stage)
        if desired > self._num_active_stages:
            self._num_active_stages = desired
            return N - desired
        return None

    def encode(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args: video (B, T+1, 3, 64, 64) in [0, 1].
        Returns dict of stage_name -> (B, T+1, C, H, W).
        """
        feats = self.encoder.encode_video(video)
        return {k: v.float() for k, v in feats.items()}

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

        with torch.set_grad_enabled(True):
            for step in range(max_steps - 1):
                pred = pred.detach().requires_grad_(True)
                energy = stage(real_ctx, pred, parent_context=parent_ctx)
                energy_val = energy.sum().item()

                if step == 0:
                    first_energy = energy

                if prev_energy_val is not None:
                    rel_change = abs(energy_val - prev_energy_val) / (
                        abs(prev_energy_val) + 1e-8
                    )
                    if rel_change < tol:
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

                grad = torch.autograd.grad(
                    [energy.sum()], [pred], create_graph=False
                )[0]
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    converge_steps = step
                    break
                pred = (pred - alpha * grad).detach()

                if step == 0:
                    first_pred = pred.clone()
            else:
                converge_steps = max_steps - 1

        pred = pred.detach().requires_grad_(True)
        energy = stage(real_ctx, pred, parent_context=parent_ctx)

        create_graph = learning
        grad = torch.autograd.grad(
            [energy.sum()], [pred], create_graph=create_graph
        )[0]
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            final_pred = pred
        else:
            final_pred = pred - alpha * grad

        if first_pred is None:
            first_pred = final_pred
        if first_energy is None:
            first_energy = energy

        preds = [first_pred, final_pred]
        energies = [first_energy, energy]
        num_steps = converge_steps + 1
        return preds, energies, num_steps

    def forward_loss(
        self,
        video: torch.Tensor,
        features: Optional[Dict[str, torch.Tensor]] = None,
        learning: bool = True,
    ) -> Dict[str, object]:
        """
        Args:
            video:    (B, T+1, 3, 64, 64) in [0, 1] — decoder target and encoder input.
            features: Optional precomputed encoder features for tests.
            learning: If True, create_graph for MCMC unroll (training mode).
        """
        if features is not None:
            feats_dict = features
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

        finest_active = min(i for i, s in enumerate(per_stage) if s is not None)

        decoder_can_fire = (self.decoder is not None
                            and finest_active == 0
                            and per_stage[0] is not None)
        if decoder_can_fire:
            if self.cfg.bottom_up_loss:
                base_pred = per_stage[0]["final_pred_live"]
            else:
                base_pred = per_stage[0]["final_pred"]
            decoded = self.decoder(base_pred)
            target_rgb = video[:, 1:]
            decoder_loss = F.l1_loss(decoded, target_rgb)
            out["loss_decoder"] = decoder_loss
            out["decoded_rgb"] = decoded.detach()
            out["target_rgb"] = target_rgb.detach()
            if self.cfg.bottom_up_loss:
                out["loss_total"] = decoder_loss
            else:
                out["loss_total"] = total_loss + self.cfg.decoder_loss_weight * decoder_loss
        else:
            out["loss_total"] = total_loss

        return out

    def _mcmc_sequential(
        self,
        feats_dict: Dict[str, torch.Tensor],
        learning: bool,
    ) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
        N = len(self.stages)
        per_stage: List[Optional[Dict[str, torch.Tensor]]] = [None] * N
        prev_pred: Optional[torch.Tensor] = None
        total_loss = next(iter(feats_dict.values())).new_zeros(())
        active = self.active_stage_indices()

        bu = self.cfg.bottom_up_loss
        finest_active_idx = active[-1] if active else -1
        bu_skip_all_loss = bu and self.cfg.decoder_enabled and (0 in active)

        for i in active:
            stage = self.stages[i]
            sc = self.cfg.stages[i]
            feats = feats_dict[sc.stage_name]
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

        return per_stage, total_loss
