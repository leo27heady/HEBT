"""Autoregressive Spatiotemporal Predictive Model (AST-PM).

Implements Equations 6, 7 from S-HR-VQVAE (arXiv 2307.06701).

Architecture (per VQ layer):
  - Learnable token embedding for discrete local indices (0..M-1).
  - Linear projection to hidden channels.
  - N causal blocks, each containing:
      • CausalConv3d  (PixelCNN-style 3-D masked conv, kernel 2×2×2, padding 1)
        First block uses mask 'A' (current position invisible),
        subsequent blocks use mask 'B' (current position visible).
      • Causal multi-head self-attention over the flattened (T,H,W) sequence.
        An upper-triangular bool mask prevents attending to future tokens in
        raster-scan order (time → height → width).
  - Output Conv3d: hidden → M logits per position.

For an HR-VQVAE with n layers, the full model instantiates n independent
AST_PM modules (one per VQ layer), each predicting that layer's local index.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv3d(nn.Conv3d):
    """Autoregressive 3-D convolution with raster-scan causal mask.

    Raster-scan order: time → height → width.

    Mask types
    ----------
    'A' — current position is **not** visible (used for the first block so
          the model cannot trivially copy the input token).
    'B' — current position **is** visible (used for all subsequent blocks so
          the model can refine its estimate using its own hidden state).
    """

    def __init__(
        self,
        mask_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        **kwargs,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        assert mask_type in ("A", "B"), f"Unknown mask type: {mask_type!r}"

        self.register_buffer("mask", torch.zeros_like(self.weight))
        _, _, kT, kH, kW = self.weight.shape

        # 1. All past time-steps: fully visible
        if kT // 2 > 0:
            self.mask[:, :, : kT // 2, :, :] = 1

        # 2. Current time-step, past rows: fully visible
        if kH // 2 > 0:
            self.mask[:, :, kT // 2, : kH // 2, :] = 1

        # 3. Current time-step, current row: past columns visible.
        #    Type 'B' additionally includes the current column.
        kW_allow = kW // 2 if mask_type == "A" else (kW // 2) + 1
        self.mask[:, :, kT // 2, kH // 2, :kW_allow] = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.conv3d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CausalBlock(nn.Module):
    """One causal-conv + causal-attention residual block.

    Spatial padding of 1 makes the convolution output one element *larger*
    than the input in every dimension; the extra border is cropped off
    (``[:-1, :-1, :-1]``) to restore the original (T, H, W) shape.
    """

    def __init__(self, mask_type: str, channels: int, num_heads: int) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            mask_type,
            channels,
            channels,
            kernel_size=2,
            padding=1,
            bias=True,
        )
        self.norm_conv = nn.GroupNorm(min(8, channels), channels)
        self.elu = nn.ELU(inplace=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape

        # ---- Causal conv residual ---- #
        h = self.conv(x)
        h = h[:, :, :-1, :-1, :-1]        # crop extra border from padding
        h = self.elu(self.norm_conv(h))
        x = x + h

        # ---- Causal self-attention residual ---- #
        N = T * H * W
        x_seq = x.reshape(B, C, N).permute(0, 2, 1)           # [B, N, C]
        # Upper-triangular bool mask: True = position to *ignore* (future token)
        attn_mask = torch.triu(
            torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
        )
        out, _ = self.attn(x_seq, x_seq, x_seq, attn_mask=attn_mask)
        x_seq = self.norm_attn(x_seq + out)
        x = x_seq.permute(0, 2, 1).reshape(B, C, T, H, W)

        return x


class AST_PM(nn.Module):
    """Autoregressive Spatiotemporal Predictive Model for one VQ layer.

    Predicts the *local* index (0..M-1) at each spatiotemporal position
    given all causally preceding positions in raster-scan order.

    Args:
        M            : codebook branching factor (number of output classes).
        in_embed_dim : embedding dimension for the input discrete tokens.
        hidden       : internal feature-channel width.
        num_heads    : number of attention heads.
        num_blocks   : total number of CausalBlock layers (≥ 1).
    """

    def __init__(
        self,
        M: int,
        in_embed_dim: int,
        hidden: int = 256,
        num_heads: int = 4,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.M = M

        # Learnable embedding for the M local code classes
        self.embedding = nn.Embedding(M, in_embed_dim)

        # Project embedding dim → hidden dim
        self.input_proj = nn.Conv3d(in_embed_dim, hidden, kernel_size=1, bias=True)

        # Causal blocks: first uses mask 'A', rest use mask 'B'
        self.blocks = nn.ModuleList(
            [CausalBlock("A", hidden, num_heads)]
            + [CausalBlock("B", hidden, num_heads) for _ in range(num_blocks - 1)]
        )

        self.out_norm = nn.GroupNorm(min(8, hidden), hidden)
        self.out_proj = nn.Conv3d(hidden, M, kernel_size=1, bias=True)

    def forward(self, q_indices: torch.Tensor) -> torch.Tensor:
        """Predict per-position logits over M candidate indices.

        Args:
            q_indices : (B, T, H, W) integer tensor of local indices (0..M-1).

        Returns:
            logits : (B, M, T, H, W)
        """
        # Embed discrete tokens
        x = self.embedding(q_indices)        # [B, T, H, W, embed_dim]
        x = x.permute(0, 4, 1, 2, 3)        # [B, embed_dim, T, H, W]
        x = self.input_proj(x)               # [B, hidden, T, H, W]

        for block in self.blocks:
            x = block(x)

        return self.out_proj(self.out_norm(x))  # [B, M, T, H, W]
