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
    # Ablation ---------------------------------------------------------------- #
    disable_cross_attn: bool = False       # If True, no parent KV conditioning between stages
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

        # Validate parent-child geometry: each stage above stage 0 must be
        # exactly half the spatial size of the stage below.
        for i in range(1, len(cfg.stages)):
            child = cfg.stages[i - 1]
            parent = cfg.stages[i]
            if child.H != 2 * parent.H or child.W != 2 * parent.W:
                raise ValueError(
                    f"Stage {i} parent {parent.H}x{parent.W} must be exactly "
                    f"half of child stage {i-1} {child.H}x{child.W}"
                )

        stage_names = tuple(s.clip_stage_name for s in cfg.stages)
        if cfg.weights_path:
            self.encoder = MobileClipMultiStageEncoder(
                weights_path=cfg.weights_path,
                return_stages=stage_names,
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
    # encoding
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args: video (B, T+1, 3, Hi, Wi) in [0, 1].
        Returns dict of stage_name -> (B, T+1, C, H, W).
        """
        if self.encoder is None:
            raise RuntimeError(
                "Encoder not loaded (weights_path was empty). "
                "Use forward_loss_from_features() with precomputed features."
            )
        feats = self.encoder.encode_video(video)
        return {k: v.float() for k, v in feats.items()}

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

    # ------------------------------------------------------------------ #
    # full forward / loss
    # ------------------------------------------------------------------ #

    def forward_loss(
        self,
        video: torch.Tensor,        # (B, T+1, 3, Hi, Wi) in [0,1]
        learning: bool = True,
    ) -> Dict[str, object]:
        feats_dict = self.encode(video)
        return self._forward_loss_impl(feats_dict, video=video, learning=learning)

    def forward_loss_from_features(
        self,
        feats_dict: Dict[str, torch.Tensor],  # {stage_name: (B, T+1, C, H, W)}
        video: Optional[torch.Tensor] = None,  # only needed if decoder is enabled
        learning: bool = True,
    ) -> Dict[str, object]:
        """Forward pass using precomputed CLIP features (skips the encoder)."""
        return self._forward_loss_impl(feats_dict, video=video, learning=learning)

    def _forward_loss_impl(
        self,
        feats_dict: Dict[str, torch.Tensor],
        video: Optional[torch.Tensor] = None,
        learning: bool = True,
    ) -> Dict[str, object]:

        per_stage: List[Dict[str, torch.Tensor]] = [None] * len(self.stages)
        prev_pred_detached: Optional[torch.Tensor] = None
        total_loss = next(iter(feats_dict.values())).new_zeros(())

        # Top-down: process from apex (coarsest, last index) to base (finest, index 0)
        for i in reversed(range(len(self.stages))):
            stage = self.stages[i]
            sc = self.cfg.stages[i]
            feats = feats_dict[sc.clip_stage_name]                 # (B, T+1, C, H, W)
            real_ctx = feats[:, :-1]
            real_gt = feats[:, 1:]
            init_pred = self._init_pred(real_gt)

            preds, energies = self._mcmc_for_stage(
                stage,
                self.alphas[i],
                real_ctx,
                init_pred,
                parent_ctx=prev_pred_detached if stage.use_cross_attn else None,
                learning=learning,
            )

            K = len(preds)
            if self.cfg.truncate_mcmc:
                stage_loss = F.smooth_l1_loss(preds[-1], real_gt)
            else:
                stage_loss = sum(F.smooth_l1_loss(p, real_gt) for p in preds) / K
            total_loss = total_loss + stage_loss

            with torch.no_grad():
                init_recon = F.smooth_l1_loss(preds[0].detach(), real_gt)
                final_recon = F.smooth_l1_loss(preds[-1].detach(), real_gt)
                init_e = energies[0].mean()
                final_e = energies[-1].mean()
                # Copy-last-frame baseline in this stage's feature space
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
                "real_gt": real_gt.detach(),
            }

            # IMPORTANT: detach for the next (finer) stage's cross-attention.
            # This severs the gradient path from stage i-1's loss back into
            # stage i's parameters via the cross-attention KV.
            prev_pred_detached = preds[-1].detach()

        out: Dict[str, object] = {
            "loss_energy": total_loss,
            "per_stage": per_stage,
        }

        # Decoder (independent training; uses detached input).
        if self.decoder is not None:
            if video is None:
                raise ValueError("Decoder requires `video` tensor for target RGB")
            base_pred = per_stage[0]["final_pred"]                # already .detach()'d above
            decoded = self.decoder(base_pred)                      # (B, T, 3, S, S)
            target_rgb = video[:, 1:, :, : self.cfg.decoder_out_size, : self.cfg.decoder_out_size]
            if target_rgb.shape != decoded.shape:
                # video resolution may differ; resize target to match decoder out_size
                Bv, Tv = target_rgb.shape[:2]
                target_rgb = F.interpolate(
                    target_rgb.reshape(Bv * Tv, 3, *target_rgb.shape[-2:]),
                    size=self.cfg.decoder_out_size, mode="bilinear", align_corners=False,
                ).reshape(Bv, Tv, 3, self.cfg.decoder_out_size, self.cfg.decoder_out_size)
            decoder_loss = F.l1_loss(decoded, target_rgb)
            out["loss_decoder"] = decoder_loss
            out["decoded_rgb"] = decoded.detach()
            out["target_rgb"] = target_rgb.detach()
            out["loss_total"] = total_loss + self.cfg.decoder_loss_weight * decoder_loss
        else:
            out["loss_total"] = total_loss

        return out
