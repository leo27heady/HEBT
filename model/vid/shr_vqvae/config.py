"""Configuration dataclass for S-HR-VQVAE.

S-HR-VQVAE combines:
  - HR-VQVAE: encoder + conditional-tree hierarchical residual VQ + decoder.
  - AST-PM: one autoregressive spatiotemporal predictive model per VQ layer.

Three-stage training pipeline:
  Stage 1  –  disjoint HR-VQVAE training (reconstruction + VQ losses).
  Stage 2  –  disjoint AST-PM training (cross-entropy on frozen codes).
  Stage 3  –  joint fine-tuning (CE + pixel recon via Gumbel-Softmax, Eq 9).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SHRVQVAEConfig:
    # ------------------------------------------------------------------ #
    # Image
    # ------------------------------------------------------------------ #
    image_h: int = 64          # input frame height
    image_w: int = 64          # input frame width
    image_c: int = 3           # RGB channels

    # ------------------------------------------------------------------ #
    # Encoder / Decoder  (4× spatial downsampling: H → H/4)
    # ------------------------------------------------------------------ #
    base_channels: int = 64    # first conv channel count
    embedding_dim: int = 128   # latent / codebook vector dimension

    # ------------------------------------------------------------------ #
    # HR-VQ Quantizer  (Conditional Tree, Eq 3–5)
    # ------------------------------------------------------------------ #
    num_vq_layers: int = 3     # n: tree depth; layer i has M^(i+1) codewords
    M: int = 16                # branching factor (use 8–32 for GPU feasibility)
    vq_beta: float = 0.25      # commitment loss coefficient β

    # ------------------------------------------------------------------ #
    # AST-PM  (Autoregressive Spatiotemporal Predictive Model, Eq 6–7)
    # ------------------------------------------------------------------ #
    astpm_hidden: int = 256    # internal channel width
    astpm_heads: int = 4       # multi-head attention heads
    astpm_blocks: int = 2      # number of CausalConv+Attention blocks

    # ------------------------------------------------------------------ #
    # Video sequence lengths
    # ------------------------------------------------------------------ #
    T: int = 10                # observed context frames
    S: int = 10                # future frames to predict

    # ------------------------------------------------------------------ #
    # Stage 3 joint training  (Eq 9)
    # ------------------------------------------------------------------ #
    lambda_joint: float = 0.11  # λ: weight on pixel-reconstruction term
    gumbel_tau: float = 1.0     # Gumbel-Softmax temperature

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #
    @property
    def latent_h(self) -> int:
        return self.image_h // 4

    @property
    def latent_w(self) -> int:
        return self.image_w // 4
