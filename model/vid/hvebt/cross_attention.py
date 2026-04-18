"""
Cross-attention for HVEBT hierarchy.

Each upper (coarser) stage cross-attends to the lower (finer) stage's detached
predicted features with two restrictions:

  1. Same time step:    a parent at frame tp may only attend keys at frame tp.
  2. 2x2 parent-child:  a parent at spatial position (yp, xp) may only attend
                         the 4 children at (2*yp + dy, 2*xp + dx) for dy,dx in {0,1}.

This keeps the hierarchy strictly local in space and time, mirrors the spatial
downsampling factor of the underlying CLIP backbone (each MobileCLIP stage is
1/2 the previous stage's spatial size), and gives a finite, predictable
receptive field for each parent token.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vid.hvebt.positional import RoPE3DCache, apply_rope3d


def build_parent_child_2x2_mask(
    T: int, Hp: int, Wp: int, device: torch.device
) -> torch.Tensor:
    """
    Additive cross-attention mask of shape (T*Hp*Wp, T*Hc*Wc) with Hc=2*Hp, Wc=2*Wp.

    Token order is t-major then y-major then x-major (matches `build_rope3d`).
    Returns 0 where attention is allowed and -inf elsewhere.
    """
    if Hp <= 0 or Wp <= 0 or T <= 0:
        raise ValueError("T, Hp, Wp must be positive")
    Hc, Wc = 2 * Hp, 2 * Wp
    Np = T * Hp * Wp
    Nc = T * Hc * Wc

    p_idx = torch.arange(Np, device=device)
    p_t = p_idx // (Hp * Wp)
    p_yx = p_idx % (Hp * Wp)
    p_y = p_yx // Wp
    p_x = p_yx % Wp

    c_idx = torch.arange(Nc, device=device)
    c_t = c_idx // (Hc * Wc)
    c_yx = c_idx % (Hc * Wc)
    c_y = c_yx // Wc
    c_x = c_yx % Wc

    same_t = p_t[:, None] == c_t[None, :]                # (Np, Nc)
    y_match = (c_y[None, :] // 2) == p_y[:, None]        # (Np, Nc)
    x_match = (c_x[None, :] // 2) == p_x[:, None]        # (Np, Nc)
    allowed = same_t & y_match & x_match                 # exactly 4 trues per row

    mask = torch.zeros(Np, Nc, device=device, dtype=torch.float32)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


class CrossAttention3DRoPE(nn.Module):
    """
    Cross-attention with 3D RoPE on Q (parent grid) and K (child grid).

    Projections operate in `dim_q`. Child KV input has channel `dim_kv` and is
    projected to `dim_q`. Manual scaled dot-product attention is used (instead
    of `F.scaled_dot_product_attention`) to support double-backward through the
    parent's MCMC unroll on CPU.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        n_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim_q % n_heads != 0:
            raise ValueError(f"dim_q {dim_q} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim_q // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        self.q_proj = nn.Linear(dim_q, dim_q, bias=bias)
        self.k_proj = nn.Linear(dim_kv, dim_q, bias=bias)
        self.v_proj = nn.Linear(dim_kv, dim_q, bias=bias)
        self.out_proj = nn.Linear(dim_q, dim_q, bias=bias)
        self.dropout = dropout

    def forward(
        self,
        q_tokens: torch.Tensor,        # (B, Np, dim_q)
        kv_tokens: torch.Tensor,       # (B, Nc, dim_kv)
        rope_q: RoPE3DCache,           # parent grid (T, Hp, Wp)
        rope_kv: RoPE3DCache,          # child  grid (T, Hc, Wc)
        attn_mask: torch.Tensor,       # (Np, Nc) additive
    ) -> torch.Tensor:
        B, Np, _ = q_tokens.shape
        Nc = kv_tokens.shape[1]

        q = (
            self.q_proj(q_tokens)
            .reshape(B, Np, self.n_heads, self.head_dim)
            .transpose(1, 2)                              # (B, H, Np, dh)
        )
        k = (
            self.k_proj(kv_tokens)
            .reshape(B, Nc, self.n_heads, self.head_dim)
            .transpose(1, 2)                              # (B, H, Nc, dh)
        )
        v = (
            self.v_proj(kv_tokens)
            .reshape(B, Nc, self.n_heads, self.head_dim)
            .transpose(1, 2)                              # (B, H, Nc, dh)
        )

        q = apply_rope3d(q, rope_q)
        k = apply_rope3d(k, rope_kv)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, Np, Nc)
        scores = scores + attn_mask                              # broadcast
        attn = torch.softmax(scores, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.matmul(attn, v)                              # (B, H, Np, dh)
        out = out.transpose(1, 2).reshape(B, Np, self.n_heads * self.head_dim)
        return self.out_proj(out)
