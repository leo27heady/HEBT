"""
3D Rotary Position Embedding (RoPE) over (t, y, x).

The head dimension is split into three contiguous chunks, each of even size:
    head_dim = d_t + d_y + d_x
RoPE is applied independently to each chunk using that axis's coordinate.
Positions are normalized to [0, 1] per axis so that different stages (with
different H, W, or T) share the same frequency schedule and so cross-stage
attention stays geometrically meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


def split_head_dim(head_dim: int) -> Tuple[int, int, int]:
    """Split `head_dim` into three even chunks (d_t, d_y, d_x) as evenly as possible."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    # Each chunk must be even. Start with equal thirds rounded down to even.
    base = (head_dim // 3) // 2 * 2
    leftover = head_dim - 3 * base
    # distribute leftover in pairs (each +=2) to x first, then y, then t
    extras = [0, 0, 0]
    i = 2
    while leftover > 0:
        extras[i] += 2
        leftover -= 2
        i = (i - 1) % 3
    d_t = base + extras[0]
    d_y = base + extras[1]
    d_x = base + extras[2]
    assert d_t + d_y + d_x == head_dim
    assert d_t % 2 == 0 and d_y % 2 == 0 and d_x % 2 == 0
    return d_t, d_y, d_x


def _rope_cos_sin(dim: int, positions: torch.Tensor, base: float = 10000.0):
    """
    Args:
        dim: even head-sub-dim.
        positions: (N,) float tensor.
    Returns:
        cos, sin: each of shape (N, dim).
    """
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=positions.device).float() / half))
    freqs = torch.outer(positions.float(), inv_freq)  # (N, half)
    # Use the [a, a] duplication scheme matching `apply_rope` below.
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (N, dim)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def _apply_rope_chunk(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to last dim of x using the interleaved pair scheme.
    x: (..., N, dim); cos/sin: (N, dim) -> broadcast over leading dims.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # Stack rotated pairs: (-x2, x1) interleaved -> same layout as cos/sin repeat_interleave(2)
    rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos + rot * sin


@dataclass
class RoPE3DCache:
    cos: torch.Tensor  # (N, head_dim)
    sin: torch.Tensor  # (N, head_dim)
    splits: Tuple[int, int, int]


def build_rope3d(
    T: int,
    H: int,
    W: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> RoPE3DCache:
    """
    Build per-token cos/sin caches for 3D RoPE over a (T, H, W) grid, with positions
    normalized to [0, 1] per axis. Token order is t-major, then y-major, then x:
        idx = t * (H*W) + y * W + x.
    """
    d_t, d_y, d_x = split_head_dim(head_dim)

    t_pos = torch.arange(T, device=device).float() / max(T - 1, 1) if T > 1 else torch.zeros(T, device=device)
    y_pos = torch.arange(H, device=device).float() / max(H - 1, 1) if H > 1 else torch.zeros(H, device=device)
    x_pos = torch.arange(W, device=device).float() / max(W - 1, 1) if W > 1 else torch.zeros(W, device=device)
    # Scale up so different normalized positions produce distinguishable frequencies.
    # Multiplying by grid size keeps classic integer-like behavior within each axis.
    t_scale = float(T)
    y_scale = float(H)
    x_scale = float(W)

    ct, st = _rope_cos_sin(d_t, t_pos * t_scale)  # (T, d_t)
    cy, sy = _rope_cos_sin(d_y, y_pos * y_scale)  # (H, d_y)
    cx, sx = _rope_cos_sin(d_x, x_pos * x_scale)  # (W, d_x)

    # Broadcast over the (T, H, W) grid and concat along last dim.
    # Shapes: ct -> (T,1,1,d_t), cy -> (1,H,1,d_y), cx -> (1,1,W,d_x)
    ct_b = ct.view(T, 1, 1, -1).expand(T, H, W, -1)
    st_b = st.view(T, 1, 1, -1).expand(T, H, W, -1)
    cy_b = cy.view(1, H, 1, -1).expand(T, H, W, -1)
    sy_b = sy.view(1, H, 1, -1).expand(T, H, W, -1)
    cx_b = cx.view(1, 1, W, -1).expand(T, H, W, -1)
    sx_b = sx.view(1, 1, W, -1).expand(T, H, W, -1)

    cos = torch.cat([ct_b, cy_b, cx_b], dim=-1).reshape(T * H * W, head_dim).to(dtype)
    sin = torch.cat([st_b, sy_b, sx_b], dim=-1).reshape(T * H * W, head_dim).to(dtype)
    return RoPE3DCache(cos=cos, sin=sin, splits=(d_t, d_y, d_x))


def apply_rope3d(x: torch.Tensor, cache: RoPE3DCache) -> torch.Tensor:
    """
    x: (..., N, head_dim) where N == T*H*W matching the cache.
    Returns x with 3D RoPE applied.
    """
    if x.shape[-1] != cache.cos.shape[-1]:
        raise ValueError(f"head_dim mismatch: x={x.shape[-1]} vs cache={cache.cos.shape[-1]}")
    if x.shape[-2] != cache.cos.shape[-2]:
        raise ValueError(f"seq len mismatch: x={x.shape[-2]} vs cache={cache.cos.shape[-2]}")
    cos = cache.cos
    sin = cache.sin
    # Broadcast cos/sin (N, d) over leading dims of x.
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return _apply_rope_chunk(x, cos, sin)
