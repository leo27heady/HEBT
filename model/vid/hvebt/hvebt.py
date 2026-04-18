"""
Hierarchical Video EBT (HVEBT) - single-stage Phase 1 implementation.

Design (Phase 1, single stage at a fixed spatial resolution H x W):
  Input per clip (B, T+1, 3, Hi, Wi) is encoded by frozen MobileCLIP stage features
  of shape (B, T+1, C_clip, H, W). We use frames [0..T-1] as "real context" and
  frames [1..T] as prediction targets. At each time step t, the transformer sees
  a token with channel-wise concat of (real_t, predicted_{t+1}) projected to D.

  Tokens are arranged (B, T*H*W, D). Attention is:
    - block-causal across T (token at frame tq may attend tokens at frame tk iff tk <= tq)
    - full within a frame (spatial tokens of same frame attend each other)
  Positional encoding: 3D RoPE over (t, y, x) with normalized coordinates.

  Per-token scalar energy -> sum -> autograd wrt x_pred (NOT wrt the transformer
  weights during MCMC) -> alpha * step -> repeat for K MCMC steps.

  Loss = reconstruction(pred_final, clip_features_{1..T}) at this stage's feature
  space. Gradient flows through MCMC unroll (create_graph during train).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vid.hvebt.clip_encoder import MobileClipMultiStageEncoder
from model.vid.hvebt.positional import RoPE3DCache, apply_rope3d, build_rope3d


# --------------------------------------------------------------------------- #
#  Stage config
# --------------------------------------------------------------------------- #


@dataclass
class HVEBTStageConfig:
    clip_stage_name: str = "final"      # which MobileCLIP stage to condition on / target
    clip_channels: int = 1024            # C of that stage
    H: int = 8                           # spatial H at this stage
    W: int = 8                           # spatial W
    embed_dim: int = 256                 # transformer hidden D (must be divisible by n_heads and yield even head_dim)
    n_heads: int = 4
    n_layers: int = 4
    ffn_mult: float = 4.0
    dropout: float = 0.0
    attn_bias: bool = False
    init_std: float = 0.02


# --------------------------------------------------------------------------- #
#  Attention / Block
# --------------------------------------------------------------------------- #


def build_block_causal_mask(T: int, HW: int, device: torch.device) -> torch.Tensor:
    """
    Returns additive attention mask of shape (T*HW, T*HW): 0 where allowed, -inf where not.
    Token at frame tq can attend to tokens at frames tk <= tq (no restriction within a frame).
    """
    # frame index of each token position
    idx = torch.arange(T * HW, device=device) // HW  # (T*HW,)
    allowed = idx[:, None] >= idx[None, :]  # (N, N) bool, True = allowed
    mask = torch.zeros(T * HW, T * HW, device=device, dtype=torch.float32)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


class SelfAttention3DRoPE(nn.Module):
    def __init__(self, dim: int, n_heads: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,              # (B, N, D)
        rope: RoPE3DCache,
        attn_mask: torch.Tensor,      # (N, N) additive
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)  # each (B, H, N, head_dim)
        # Apply 3D RoPE to q and k
        q = apply_rope3d(q, rope)
        k = apply_rope3d(k, rope)
        # Manual scaled dot-product attention (portable; supports double backward for MCMC).
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale          # (B, H, N, N)
        scores = scores + attn_mask                                      # (N, N) broadcast
        attn = torch.softmax(scores, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.matmul(attn, v)                                      # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mult)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: HVEBTStageConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = SelfAttention3DRoPE(cfg.embed_dim, cfg.n_heads, cfg.attn_bias, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.ff = FeedForward(cfg.embed_dim, cfg.ffn_mult, cfg.dropout)

    def forward(self, x, rope, attn_mask):
        x = x + self.attn(self.norm1(x), rope, attn_mask)
        x = x + self.ff(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
#  HVEBTStage: one stage EBT (no cross-attention yet; added in Phase 2)
# --------------------------------------------------------------------------- #


class HVEBTStage(nn.Module):
    """
    Single stage of the hierarchical EBT. Given real clip features and current
    predicted features (both same shape), outputs a per-token scalar energy.
    """

    def __init__(self, cfg: HVEBTStageConfig):
        super().__init__()
        self.cfg = cfg
        # channel-wise concat of (real_t, pred_{t+1}) -> D
        self.input_proj = nn.Linear(2 * cfg.clip_channels, cfg.embed_dim, bias=True)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = nn.LayerNorm(cfg.embed_dim)
        self.energy_head = nn.Linear(cfg.embed_dim, 1)
        self._rope_cache: Dict[Tuple[int, torch.device], RoPE3DCache] = {}
        self._mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.cfg.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # zero-init final energy head for stable initial energies
        nn.init.zeros_(self.energy_head.weight)
        nn.init.zeros_(self.energy_head.bias)

    def _get_rope(self, T: int, device: torch.device, dtype: torch.dtype) -> RoPE3DCache:
        key = (T, device)
        cache = self._rope_cache.get(key)
        if cache is None or cache.cos.dtype != dtype:
            cache = build_rope3d(T, self.cfg.H, self.cfg.W, self.cfg.embed_dim // self.cfg.n_heads, device, dtype)
            self._rope_cache[key] = cache
        return cache

    def _get_mask(self, T: int, device: torch.device) -> torch.Tensor:
        key = (T, device)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = build_block_causal_mask(T, self.cfg.H * self.cfg.W, device)
            self._mask_cache[key] = mask
        return mask

    def forward(
        self,
        real_feats: torch.Tensor,      # (B, T, C, H, W)
        pred_feats: torch.Tensor,      # (B, T, C, H, W)
    ) -> torch.Tensor:
        """
        Returns per-token scalar energy of shape (B, T*H*W).
        """
        B, T, C, H, W = real_feats.shape
        if pred_feats.shape != real_feats.shape:
            raise ValueError("real_feats and pred_feats must have same shape")
        if (C, H, W) != (self.cfg.clip_channels, self.cfg.H, self.cfg.W):
            raise ValueError(
                f"Expected features (C={self.cfg.clip_channels}, H={self.cfg.H}, W={self.cfg.W}), "
                f"got ({C}, {H}, {W})"
            )
        # (B, T, C, H, W) -> (B, T, H, W, C) -> (B, T*H*W, C)
        r = real_feats.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)
        p = pred_feats.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)
        tokens = torch.cat([r, p], dim=-1)           # (B, N, 2C)
        x = self.input_proj(tokens)                   # (B, N, D)
        rope = self._get_rope(T, x.device, x.dtype)
        mask = self._get_mask(T, x.device)
        for blk in self.blocks:
            x = blk(x, rope, mask)
        x = self.norm_out(x)
        energy = self.energy_head(x).squeeze(-1)      # (B, N)
        return energy


# --------------------------------------------------------------------------- #
#  HVEBT wrapper with MCMC
# --------------------------------------------------------------------------- #


@dataclass
class HVEBTConfig:
    stage: HVEBTStageConfig = field(default_factory=HVEBTStageConfig)
    mcmc_num_steps: int = 2
    mcmc_step_size: float = 1000.0
    mcmc_step_size_learnable: bool = True
    langevin_noise: float = 0.0
    denoising_init: str = "zeros"        # "zeros" | "random_noise" | "real_current"
    truncate_mcmc: bool = False
    clamp_grad_max: float = 0.0          # 0 -> no clamp
    weights_path: str = "clip/MobileCLIP2-S0/mobileclip2_s0.pt"


class HVEBT(nn.Module):
    """Single-stage HVEBT for Phase 1. Wraps frozen encoder + one HVEBTStage."""

    def __init__(self, cfg: HVEBTConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MobileClipMultiStageEncoder(
            weights_path=cfg.weights_path,
            return_stages=(cfg.stage.clip_stage_name,),
        )
        self.stage = HVEBTStage(cfg.stage)
        self.alpha = nn.Parameter(
            torch.tensor(float(cfg.mcmc_step_size)),
            requires_grad=cfg.mcmc_step_size_learnable,
        )
        self.langevin_std = float(cfg.langevin_noise)

    # ---- feature extraction ------------------------------------------------ #

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args: video (B, T+1, 3, Hi, Wi) in [0, 1].
        Returns stage feats (B, T+1, C, H, W).
        """
        feats = self.encoder.encode_video(video)[self.cfg.stage.clip_stage_name]
        return feats.float()

    # ---- initial prediction ------------------------------------------------ #

    def _init_pred(self, real_next: torch.Tensor) -> torch.Tensor:
        if self.cfg.denoising_init == "zeros":
            return torch.zeros_like(real_next)
        if self.cfg.denoising_init == "random_noise":
            return torch.randn_like(real_next)
        if self.cfg.denoising_init == "real_current":
            # naive baseline: use same as context
            return real_next.clone().detach()
        raise ValueError(self.cfg.denoising_init)

    # ---- MCMC ---------------------------------------------------------------#

    def mcmc(
        self,
        real_ctx: torch.Tensor,       # (B, T, C, H, W)
        init_pred: torch.Tensor,      # (B, T, C, H, W)
        learning: bool,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Run K MCMC steps. Returns (pred_list, energy_list) of length K.
        Each pred in the list has gradient linkage to the prior step when `learning`.
        """
        preds: List[torch.Tensor] = []
        energies: List[torch.Tensor] = []
        alpha = torch.clamp(self.alpha, min=1e-4)

        pred = init_pred
        K = self.cfg.mcmc_num_steps
        with torch.set_grad_enabled(True):
            for step in range(K):
                pred = pred.detach().requires_grad_(True)
                inp = pred
                if self.langevin_std > 0.0:
                    inp = inp + torch.randn_like(inp) * self.langevin_std
                energy = self.stage(real_ctx, inp)       # (B, N)
                energies.append(energy)

                create_graph = learning and (
                    not self.cfg.truncate_mcmc or step == K - 1
                )
                grad = torch.autograd.grad(
                    [energy.sum()], [pred], create_graph=create_graph
                )[0]

                if self.cfg.clamp_grad_max > 0.0:
                    lim = self.cfg.clamp_grad_max / alpha.detach()
                    grad = torch.clamp(grad, -lim, lim)

                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    raise RuntimeError("NaN/Inf in MCMC gradient")

                pred = pred - alpha * grad
                preds.append(pred)
        return preds, energies

    # ---- full forward/loss --------------------------------------------------#

    def forward_loss(
        self,
        video: torch.Tensor,          # (B, T+1, 3, Hi, Wi) in [0,1]
        learning: bool = True,
    ) -> Dict[str, torch.Tensor]:
        feats = self.encode(video)                             # (B, T+1, C, H, W)
        real_ctx = feats[:, :-1]                               # (B, T, ...)
        real_gt = feats[:, 1:]                                 # (B, T, ...)
        init_pred = self._init_pred(real_gt)
        preds, energies = self.mcmc(real_ctx, init_pred, learning=learning)

        total = 0.0
        K = len(preds)
        for i, p in enumerate(preds):
            loss_i = F.smooth_l1_loss(p, real_gt)
            if self.cfg.truncate_mcmc:
                if i == K - 1:
                    total = loss_i
            else:
                total = total + loss_i / K

        with torch.no_grad():
            init_energy = energies[0].mean()
            final_energy = energies[-1].mean()
            init_recon = F.smooth_l1_loss(preds[0].detach(), real_gt)
            final_recon = F.smooth_l1_loss(preds[-1].detach(), real_gt)

        return {
            "loss": total,
            "init_energy": init_energy,
            "final_energy": final_energy,
            "energy_gap": init_energy - final_energy,
            "init_recon": init_recon,
            "final_recon": final_recon,
            "alpha": self.alpha.detach(),
        }
