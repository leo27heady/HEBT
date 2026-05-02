"""
Cross-attention for HVEBT hierarchy (top-down prediction tower).

The prediction tower processes stages from coarsest (apex) to finest. Each
finer stage cross-attends to the coarser stage's **detached** predicted
features (parent → child conditioning) with two restrictions:

  1. Same time step:    a child at frame tc may only attend keys at frame tc.
  2. Spatial parent:    a child at spatial position (yc, xc) may only attend
                        the parent at (yc // 2, xc // 2).

This gives each child token exactly 1 parent key (many-to-one: 4 children
share the same parent). The parent provides abstract context that the child
specializes into finer detail.

The CLIP encoder is bottom-up (fine→coarse), the prediction tower is top-down
(coarse→fine).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vid.hvebt.positional import RoPE3DCache, apply_rope3d


def build_child_to_parent_mask(
    T: int, Hp: int, Wp: int, device: torch.device
) -> torch.Tensor:
    """
    Additive cross-attention mask of shape (T*Hc*Wc, T*Hp*Wp) with Hc=2*Hp, Wc=2*Wp.

    Queries are from the child (finer) grid, keys are from the parent (coarser) grid.
    A child at (yc, xc, tc) attends to the parent at (yc//2, xc//2, tc).
    Each child row has exactly 1 allowed entry.

    Token order is t-major then y-major then x-major (matches `build_rope3d`).
    Returns 0 where attention is allowed and -inf elsewhere.
    """
    if Hp <= 0 or Wp <= 0 or T <= 0:
        raise ValueError("T, Hp, Wp must be positive")
    Hc, Wc = 2 * Hp, 2 * Wp
    Nc = T * Hc * Wc   # child (query) positions
    Np = T * Hp * Wp   # parent (key) positions

    c_idx = torch.arange(Nc, device=device)
    c_t = c_idx // (Hc * Wc)
    c_yx = c_idx % (Hc * Wc)
    c_y = c_yx // Wc
    c_x = c_yx % Wc

    p_idx = torch.arange(Np, device=device)
    p_t = p_idx // (Hp * Wp)
    p_yx = p_idx % (Hp * Wp)
    p_y = p_yx // Wp
    p_x = p_yx % Wp

    same_t = c_t[:, None] == p_t[None, :]                    # (Nc, Np)
    y_match = (c_y[:, None] // 2) == p_y[None, :]            # (Nc, Np)
    x_match = (c_x[:, None] // 2) == p_x[None, :]            # (Nc, Np)
    allowed = same_t & y_match & x_match                     # exactly 1 true per row

    mask = torch.zeros(Nc, Np, device=device, dtype=torch.float32)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


def build_cross_attn_mask(
    T: int, Hc: int, Wc: int, Hp: int, Wp: int, device: torch.device
) -> torch.Tensor:
    """
    Generalized cross-attention mask: child (Hc, Wc) queries attend to
    parent (Hp, Wp) keys at the same time step.

    Three cases:
      1. Parent is a vector (Hp=1, Wp=1): every child token at time t attends
         to the single parent token at time t (broadcast).
      2. Child is exactly 2x parent (Hc=2*Hp, Wc=2*Wp): standard spatial
         parent mapping, child at (yc,xc) attends to parent at (yc//2, xc//2).
      3. General case: child at (yc,xc) attends to the nearest parent via
         floor division: parent_y = yc * Hp // Hc, parent_x = xc * Wp // Wc.

    Token order: t-major, then y-major, then x-major.
    Returns additive mask (0 = allowed, -inf = blocked).
    """
    if T <= 0 or Hc <= 0 or Wc <= 0 or Hp <= 0 or Wp <= 0:
        raise ValueError("All dimensions must be positive")

    Nc = T * Hc * Wc
    Np = T * Hp * Wp

    c_idx = torch.arange(Nc, device=device)
    c_t = c_idx // (Hc * Wc)
    c_yx = c_idx % (Hc * Wc)
    c_y = c_yx // Wc
    c_x = c_yx % Wc

    p_idx = torch.arange(Np, device=device)
    p_t = p_idx // (Hp * Wp)
    p_yx = p_idx % (Hp * Wp)
    p_y = p_yx // Wp
    p_x = p_yx % Wp

    same_t = c_t[:, None] == p_t[None, :]

    if Hp == 1 and Wp == 1:
        # Vector parent: every child at time t attends to the 1 parent at time t.
        allowed = same_t
    else:
        # Map child spatial to parent spatial via floor division.
        mapped_py = c_y * Hp // Hc  # (Nc,)
        mapped_px = c_x * Wp // Wc  # (Nc,)
        y_match = mapped_py[:, None] == p_y[None, :]
        x_match = mapped_px[:, None] == p_x[None, :]
        allowed = same_t & y_match & x_match

    mask = torch.zeros(Nc, Np, device=device, dtype=torch.float32)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


class CrossAttention3DRoPE(nn.Module):
    """
    Cross-attention with 3D RoPE on Q (child/finer grid) and K (parent/coarser grid).

    In the top-down prediction tower, the child (finer, query) cross-attends to
    the parent (coarser, key/value) to receive abstract conditioning.

    Projections operate in `dim_q`. Parent KV input has channel `dim_kv` and is
    projected to `dim_q`. Manual scaled dot-product attention is used (instead
    of `F.scaled_dot_product_attention`) to support double-backward through the
    MCMC unroll on CPU.
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
        q_tokens: torch.Tensor,        # (B, Nq, dim_q) - child (finer) queries
        kv_tokens: torch.Tensor,       # (B, Nkv, dim_kv) - parent (coarser) keys/values
        rope_q: RoPE3DCache,           # child grid (T, Hc, Wc)
        rope_kv: RoPE3DCache,          # parent grid (T, Hp, Wp)
        attn_mask: torch.Tensor,       # (Nq, Nkv) additive
    ) -> torch.Tensor:
        B, Nq, _ = q_tokens.shape
        Nkv = kv_tokens.shape[1]

        q = (
            self.q_proj(q_tokens)
            .reshape(B, Nq, self.n_heads, self.head_dim)
            .transpose(1, 2)                              # (B, H, Nq, dh)
        )
        k = (
            self.k_proj(kv_tokens)
            .reshape(B, Nkv, self.n_heads, self.head_dim)
            .transpose(1, 2)                              # (B, H, Nkv, dh)
        )
        v = (
            self.v_proj(kv_tokens)
            .reshape(B, Nkv, self.n_heads, self.head_dim)
            .transpose(1, 2)                              # (B, H, Nkv, dh)
        )

        q = apply_rope3d(q, rope_q)
        k = apply_rope3d(k, rope_kv)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, Nq, Nkv)
        scores = scores + attn_mask                              # broadcast
        attn = torch.softmax(scores, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.matmul(attn, v)                              # (B, H, Nq, dh)
        out = out.transpose(1, 2).reshape(B, Nq, self.n_heads * self.head_dim)
        return self.out_proj(out)
