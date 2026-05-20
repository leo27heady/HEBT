"""Transformer-based predictor stages for LFQ-VQVAE video mode."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import (
    build_cross_attn_mask_mid_to_bot,
    build_cross_attn_mask_top_to_mid,
    build_temporal_window_mask,
)
from .soft_lookup import build_lfq_codebook_matrix


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and optional cross-attention."""

    def __init__(self, dim: int, n_heads: int, has_cross_attn: bool = False, parent_dim: int | None = None):
        super().__init__()
        self.has_cross_attn = has_cross_attn

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

        if has_cross_attn:
            self.norm_cross = nn.LayerNorm(dim)
            kv_dim = parent_dim if parent_dim is not None else dim
            self.cross_attn_q = nn.Linear(dim, dim)
            self.cross_attn_k = nn.Linear(kv_dim, dim)
            self.cross_attn_v = nn.Linear(kv_dim, dim)
            self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_kv: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_mask = ~self_attn_mask if self_attn_mask is not None else None
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = residual + x_attn

        if self.has_cross_attn and cross_kv is not None:
            residual = x
            x_norm = self.norm_cross(x)
            q = self.cross_attn_q(x_norm)
            k = self.cross_attn_k(cross_kv)
            v = self.cross_attn_v(cross_kv)
            cross_mask = ~cross_attn_mask if cross_attn_mask is not None else None
            x_cross, _ = self.cross_attn(q, k, v, attn_mask=cross_mask)
            x = residual + x_cross

        return x + self.ffn(self.norm2(x))


class PredictorStage(nn.Module):
    """
    Single predictor stage in top-down hierarchy.

    Input:
        quant_input: (B, T*S, dim)
        parent_features: (B, T*S_parent, parent_dim) or None
    Output:
        logits: (B, T*S, codebook_size)
        pred_features: (B, T*S, dim)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_layers: int,
        codebook_size: int,
        spatial_size: int,
        temporal_window: int,
        has_parent: bool = False,
        parent_dim: int | None = None,
        lfq_dim: int | None = None,
        max_T: int = 16,
        use_gumbel_softmax: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.spatial_size = spatial_size
        self.temporal_window = temporal_window
        self.codebook_size = codebook_size
        self.use_gumbel_softmax = use_gumbel_softmax
        self.max_T = max_T

        self.input_proj = nn.Linear(dim, dim)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, spatial_size, dim) * 0.02)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, max_T, dim) * 0.02)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim=dim, n_heads=n_heads, has_cross_attn=has_parent, parent_dim=parent_dim)
                for _ in range(n_layers)
            ]
        )

        self.output_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, codebook_size))

        if lfq_dim is None:
            raise ValueError("lfq_dim is required for LFQ soft lookup")
        self.soft_lookup_proj = nn.Linear(lfq_dim, dim)
        self.register_buffer("codebook_weights", build_lfq_codebook_matrix(lfq_dim))

    def _get_pos_encoding(self, T: int, S: int) -> torch.Tensor:
        if T > self.max_T:
            raise ValueError(f"T={T} exceeds max_T={self.max_T}")
        spatial = self.spatial_pos_embed[:, :S, :]
        temporal = self.temporal_pos_embed[:, :T, :]
        temporal_expanded = temporal.repeat_interleave(S, dim=1)
        spatial_expanded = spatial.repeat(1, T, 1)
        return temporal_expanded + spatial_expanded

    def _soft_lookup(self, logits: torch.Tensor, temperature: float = 1.0, gumbel_tau: float = 1.0) -> torch.Tensor:
        if self.use_gumbel_softmax:
            probs = F.gumbel_softmax(logits / max(temperature, 1e-6), tau=gumbel_tau, hard=False, dim=-1)
        else:
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        soft_codes = probs @ self.codebook_weights  # (B, T*S, lfq_dim)
        return self.soft_lookup_proj(soft_codes)

    def forward(
        self,
        quant_input: torch.Tensor,
        parent_features: torch.Tensor | None = None,
        T: int | None = None,
        temperature: float = 1.0,
        gumbel_tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if T is None:
            raise ValueError("T must be provided")
        S = self.spatial_size
        device = quant_input.device

        x = self.input_proj(quant_input)
        x = x + self._get_pos_encoding(T, S)

        self_attn_mask = build_temporal_window_mask(T, S, self.temporal_window, device)

        cross_attn_mask = None
        if parent_features is not None:
            if S == 16:
                cross_attn_mask = build_cross_attn_mask_top_to_mid(T, device)
            elif S == 256:
                cross_attn_mask = build_cross_attn_mask_mid_to_bot(T, device)
            else:
                raise ValueError(f"Unsupported child token count for cross-attention mask: S={S}")

        for layer in self.layers:
            x = layer(x, self_attn_mask=self_attn_mask, cross_kv=parent_features, cross_attn_mask=cross_attn_mask)

        logits = self.output_head(x)
        pred_features = self._soft_lookup(logits, temperature=temperature, gumbel_tau=gumbel_tau)
        return logits, pred_features
