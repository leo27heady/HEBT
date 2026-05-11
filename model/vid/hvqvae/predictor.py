"""
Stage predictor for HVQVAE.

Direct logit prediction (no MCMC, no energy function). For each stage:
  1. Takes context frames' quantized features as input tokens.
  2. Runs through a transformer with causal self-attention + optional
     cross-attention from the parent (coarser) stage.
  3. Outputs logits over codebook entries.
  4. Soft codebook lookup: softmax(logits) @ codebook.weight → predicted embedding.

The predicted embedding can be fed to the decoder for pixel reconstruction,
or passed as parent context to finer stages.

Reuses positional encoding (RoPE3D), causal masks, and cross-attention
from the existing HVEBT implementation.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vid.hvebt.cross_attention import CrossAttention3DRoPE, build_cross_attn_mask
from model.vid.hvebt.positional import RoPE3DCache, apply_rope3d, build_rope3d
from model.vid.hvebt.hvebt import (
    SelfAttention3DRoPE,
    FeedForward,
    build_block_causal_mask,
)
from model.vid.hvqvae.config import HVQVAEStageConfig


# --------------------------------------------------------------------------- #
#  Transformer block
# --------------------------------------------------------------------------- #


class _Block(nn.Module):
    """Transformer block: LayerNorm → SelfAttn → (optional CrossAttn) → FF."""

    def __init__(self, cfg: HVQVAEStageConfig, cross_attn_kv_dim: Optional[int] = None):
        super().__init__()
        D = cfg.transformer_dim
        self.norm1 = nn.LayerNorm(D)
        self.attn = SelfAttention3DRoPE(D, cfg.n_heads, bias=False, dropout=0.0)

        self.use_cross_attn = cross_attn_kv_dim is not None
        if self.use_cross_attn:
            self.norm_cross = nn.LayerNorm(D)
            self.cross_attn = CrossAttention3DRoPE(
                dim_q=D, dim_kv=D,
                n_heads=cfg.n_heads, bias=False, dropout=0.0,
            )

        self.norm2 = nn.LayerNorm(D)
        self.ff = FeedForward(D, mult=4.0, dropout=0.0)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE3DCache,
        attn_mask: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_ctx: Optional[RoPE3DCache] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope, attn_mask)
        if self.use_cross_attn and context is not None:
            x = x + self.cross_attn(
                self.norm_cross(x), context, rope, rope_ctx, cross_mask
            )
        x = x + self.ff(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
#  Stage predictor
# --------------------------------------------------------------------------- #


class StagePredictor(nn.Module):
    """Direct logit predictor for one hierarchy stage.

    Takes quantized context tokens, predicts logits over the codebook,
    and produces soft embeddings via softmax @ codebook.weight.

    Parameters
    ----------
    cfg : HVQVAEStageConfig for this stage.
    parent_cfg : config of the coarser parent stage (None for apex stage).
    """

    def __init__(
        self,
        cfg: HVQVAEStageConfig,
        parent_cfg: Optional[HVQVAEStageConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.use_cross_attn = parent_cfg is not None

        D = cfg.transformer_dim
        C = cfg.channels
        K = cfg.num_codes

        # Input projection: C → D
        self.input_proj = nn.Linear(C, D)

        # Parent projection for cross-attention
        if self.use_cross_attn:
            self.Hp = parent_cfg.H
            self.Wp = parent_cfg.W
            self.parent_proj = nn.Linear(parent_cfg.channels, D)
            self.parent_norm = nn.LayerNorm(D)

        # Transformer blocks
        cross_kv_dim = D if self.use_cross_attn else None
        self.blocks = nn.ModuleList([
            _Block(cfg, cross_attn_kv_dim=cross_kv_dim)
            for _ in range(cfg.n_layers)
        ])

        # Output head: predict codebook logits
        self.norm_out = nn.LayerNorm(D)
        self.logit_head = nn.Linear(D, K)

        # Cache masks and RoPE (keyed by T, device)
        self._rope_cache: Dict[Tuple[int, torch.device], RoPE3DCache] = {}
        self._mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._parent_rope_cache: Dict[Tuple[int, torch.device], RoPE3DCache] = {}
        self._cross_mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Logit head: small init so initial predictions are near-uniform
        nn.init.normal_(self.logit_head.weight, std=0.02)
        nn.init.zeros_(self.logit_head.bias)

    # ------------------------------------------------------------------ #
    #  Cache helpers
    # ------------------------------------------------------------------ #

    def _get_rope(self, T: int, device: torch.device, dtype: torch.dtype) -> RoPE3DCache:
        key = (T, device)
        c = self._rope_cache.get(key)
        if c is None or c.cos.dtype != dtype:
            head_dim = self.cfg.transformer_dim // self.cfg.n_heads
            c = build_rope3d(T, self.cfg.H, self.cfg.W, head_dim, device, dtype)
            self._rope_cache[key] = c
        return c

    def _get_mask(self, T: int, device: torch.device) -> torch.Tensor:
        key = (T, device)
        m = self._mask_cache.get(key)
        if m is None:
            m = build_block_causal_mask(
                T, self.cfg.H, device,
                temporal_window=self.cfg.temporal_window,
                W=self.cfg.W,
                spatial_window=self.cfg.spatial_window,
            )
            self._mask_cache[key] = m
        return m

    def _get_parent_rope(self, T: int, device: torch.device, dtype: torch.dtype) -> RoPE3DCache:
        key = (T, device)
        c = self._parent_rope_cache.get(key)
        if c is None or c.cos.dtype != dtype:
            head_dim = self.cfg.transformer_dim // self.cfg.n_heads
            c = build_rope3d(T, self.Hp, self.Wp, head_dim, device, dtype)
            self._parent_rope_cache[key] = c
        return c

    def _get_cross_mask(self, T: int, device: torch.device) -> torch.Tensor:
        key = (T, device)
        m = self._cross_mask_cache.get(key)
        if m is None:
            m = build_cross_attn_mask(
                T, self.cfg.H, self.cfg.W, self.Hp, self.Wp, device
            )
            self._cross_mask_cache[key] = m
        return m

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        context: torch.Tensor,                         # (B, T, C, H, W)
        codebook_weight: torch.Tensor,                  # (K, C)
        parent_context: Optional[torch.Tensor] = None,  # (B, T, Cp, Hp, Wp)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict future frame codes from context.

        Args:
            context: quantized features for context frames (straight-through).
            codebook_weight: this stage's codebook embedding matrix.
            parent_context: predicted embedding from coarser parent stage.

        Returns:
            logits: (B, T*H*W, K) raw logits over codebook entries.
            pred_embed: (B, T, C, H, W) soft codebook-weighted embedding.
        """
        B, T, C, H, W = context.shape

        # Flatten to tokens: (B, T*H*W, C)
        tokens = context.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)
        x = self.input_proj(tokens)  # (B, N, D)

        rope = self._get_rope(T, x.device, x.dtype)
        mask = self._get_mask(T, x.device)

        # Prepare parent context for cross-attention
        ctx_proj: Optional[torch.Tensor] = None
        rope_ctx: Optional[RoPE3DCache] = None
        cross_mask: Optional[torch.Tensor] = None

        if self.use_cross_attn and parent_context is not None:
            Bp, Tp, Cp, Hp, Wp = parent_context.shape
            par_tokens = parent_context.permute(0, 1, 3, 4, 2).reshape(B, T * Hp * Wp, Cp)
            ctx_proj = self.parent_norm(self.parent_proj(par_tokens))
            rope_ctx = self._get_parent_rope(T, x.device, x.dtype)
            cross_mask = self._get_cross_mask(T, x.device)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, rope, mask, ctx_proj, rope_ctx, cross_mask)

        # Output logits
        logits = self.logit_head(self.norm_out(x))  # (B, N, K)

        # Soft codebook lookup: softmax(logits) @ codebook → predicted embedding
        probs = F.softmax(logits, dim=-1)                      # (B, N, K)
        pred_flat = torch.matmul(probs, codebook_weight)       # (B, N, C)

        # Reshape to spatial: (B, T, H, W, C) → (B, T, C, H, W)
        pred_embed = pred_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

        return logits, pred_embed
