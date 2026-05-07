"""
VQ-VAE Vector Quantizer with straight-through estimator.

Mathematical specification
--------------------------
Encoder output:      z_e  ∈ R^{B, N, C}      (N = T*H*W tokens)
Codebook:            E    ∈ R^{K, C}          (K discrete codes)

Nearest-neighbor quantization:
    k* = argmin_j ||z_e - E_j||²
    z_q = E[k*]                                (hard, discrete)

Straight-through copy (gradient bypass):
    z_q_st = z_e + sg(z_q - z_e)
    Forward:  z_q_st == z_q   (uses quantized code)
    Backward: ∂loss/∂z_q_st is copied unchanged to ∂loss/∂z_e
              (as if z_q_st were an identity function of z_e)

This means:
  - The ENCODER is trained by TWO gradient paths:
      1. Commitment loss:   β * ||z_e - sg(z_q)||²  (encoder → codebook)
      2. Straight-through:  gradients from downstream (predictor loss, decoder)
         pass through z_q_st back to z_e as if no quantization happened.
  - The CODEBOOK is trained by ONE explicit gradient path:
      Codebook loss:  ||sg(z_e) - z_q||²  (codebook → encoder outputs)

Prediction decode (logit → embedding):
    p = softmax(logits)                        logits ∈ R^{B, N, K}
    z_pred = p @ E                             z_pred ∈ R^{B, N, C}

This differentiably maps predicted code distributions back to the embedding
space. At convergence a one-hot distribution at code k gives exactly E[k],
matching the quantizer output. The gradient flows from loss through z_pred
back through softmax to logits, and through p @ E to the codebook entries.

NOTE on codebook grad flow during prediction:
    If we do NOT want the codebook to be trained by the prediction loss (only
    by the explicit codebook loss), we can detach E before the matmul. However
    the plan keeps the codebook trainable through BOTH paths by default; the
    codebook loss term already provides the dominant training signal, and the
    prediction loss contribution is small. A `detach_codebook_in_decode` flag
    is provided if needed for ablation.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vid.vq_hvebt.config import VQCodebookConfig


# --------------------------------------------------------------------------- #
#  Output container
# --------------------------------------------------------------------------- #


class QuantizerOutput(NamedTuple):
    """All outputs from one VectorQuantizer.encode() call.

    z_q_st  : (B, N, C)   straight-through quantized (use this downstream).
    z_q     : (B, N, C)   hard-quantized (no grad to encoder; for targets).
    indices : (B, N)      long tensor of nearest-code indices.
    cb_loss : ()          codebook loss  ||sg(z_e) - z_q||²
    commit_loss: ()       commitment loss β||z_e - sg(z_q)||²
    """
    z_q_st: torch.Tensor
    z_q: torch.Tensor
    indices: torch.Tensor
    cb_loss: torch.Tensor
    commit_loss: torch.Tensor


# --------------------------------------------------------------------------- #
#  VectorQuantizer
# --------------------------------------------------------------------------- #


class VectorQuantizer(nn.Module):
    """Online VQ-VAE quantizer for a single hierarchy stage.

    Parameters
    ----------
    cfg : VQCodebookConfig
        Codebook size K, dimension C, init mode, and commitment β.
    detach_codebook_in_decode : bool
        If True, the codebook is detached when decoding logits to embeddings.
        Default False (codebook trains through prediction too).
    """

    def __init__(
        self,
        cfg: VQCodebookConfig,
        detach_codebook_in_decode: bool = False,
    ):
        super().__init__()
        self.K = cfg.num_codes
        self.C = cfg.code_dim
        self.beta = cfg.commitment_beta
        self.detach_codebook_in_decode = detach_codebook_in_decode
        self._initialized = cfg.init_mode == "random"

        # Codebook as nn.Embedding — weight is a learnable nn.Parameter (K, C).
        self.codebook = nn.Embedding(self.K, self.C)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0 / (self.C ** 0.5))

    # ------------------------------------------------------------------ #
    #  Initialization helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def initialize_from_data(self, z_e: torch.Tensor) -> None:
        """Replace codebook entries with randomly-sampled encoder outputs.

        Should be called once after the first forward pass when
        ``cfg.init_mode == "data_first_batch"``.  If z_e has fewer than K
        tokens, entries cycle through the available samples with random jitter.

        Args:
            z_e: (M, C) flat encoder outputs (M >= 1).
        """
        M, C = z_e.shape
        if C != self.C:
            raise ValueError(f"z_e dim {C} != codebook dim {self.C}")
        perm = torch.randperm(M, device=z_e.device)
        sampled = z_e[perm]  # (M, C)
        # Repeat to fill K slots.
        repeats = (self.K + M - 1) // M
        sampled = sampled.repeat(repeats, 1)[:self.K]  # (K, C)
        # Add jitter proportional to feature spread so codes are well-separated.
        # Too small jitter (0.01) creates near-identical codes with narrow Voronoi
        # regions that flip assignment with tiny encoder/codebook changes.
        feat_std = z_e.std().item()
        jitter_scale = max(0.1 * feat_std, 0.02)
        jitter = torch.randn_like(sampled) * jitter_scale
        self.codebook.weight.data.copy_(sampled + jitter)
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------ #
    #  Core: encode (quantize)
    # ------------------------------------------------------------------ #

    def encode(self, z_e: torch.Tensor) -> QuantizerOutput:
        """Quantize encoder outputs with straight-through estimator.

        Args:
            z_e: (B, N, C) encoder outputs; N = T*H*W.

        Returns:
            QuantizerOutput with z_q_st, z_q, indices, cb_loss, commit_loss.
        """
        B, N, C = z_e.shape
        if C != self.C:
            raise ValueError(f"z_e channels {C} != codebook dim {self.C}")

        # ---- nearest-neighbor lookup --------------------------------------- #
        # Expand to (B*N, C) for batched distance calculation.
        z_flat = z_e.reshape(B * N, C)                         # (M, C)
        E = self.codebook.weight                               # (K, C)

        # ||z - e||² = ||z||² + ||e||² - 2 <z, e>
        # Shapes: z_sq (M,1), e_sq (1,K), dot (M,K) -> dist (M,K)
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)          # (M, 1)
        e_sq = (E ** 2).sum(dim=1, keepdim=True).T             # (1, K)
        dot = z_flat @ E.T                                     # (M, K)
        dist = z_sq + e_sq - 2 * dot                           # (M, K)

        indices_flat = dist.argmin(dim=1)                      # (M,)
        indices = indices_flat.reshape(B, N)                   # (B, N)

        # ---- quantized embedding ------------------------------------------ #
        # z_q: hard nearest-neighbor embedding (no grad to z_e).
        z_q_flat = F.embedding(indices_flat, E)                # (M, C)
        z_q = z_q_flat.reshape(B, N, C)                       # (B, N, C)

        # ---- straight-through copy ---------------------------------------- #
        # Forward: z_q_st == z_q
        # Backward: gradient of z_q_st w.r.t. z_e is identity (copy of ∂/∂z_q_st)
        z_q_st = z_e + (z_q - z_e).detach()                   # (B, N, C)

        # ---- VQ-VAE losses ------------------------------------------------- #
        # Codebook loss: moves codebook entries toward encoder outputs.
        # sg(z_e) means z_e is detached — only codebook receives gradient here.
        cb_loss = F.mse_loss(z_e.detach(), z_q)

        # Commitment loss: keeps encoder outputs near chosen codebook entries.
        # sg(z_q) means z_q is detached — only encoder receives gradient here.
        commit_loss = self.beta * F.mse_loss(z_e, z_q.detach())

        return QuantizerOutput(
            z_q_st=z_q_st,
            z_q=z_q,
            indices=indices,
            cb_loss=cb_loss,
            commit_loss=commit_loss,
        )

    # ------------------------------------------------------------------ #
    #  Decode: logits → embedding (for predictor)
    # ------------------------------------------------------------------ #

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert predicted code-distribution logits to embedding vectors.

        Computes z_pred = softmax(logits) @ E, which is a differentiable
        weighted combination of codebook entries. This keeps predictions on
        the convex hull of the codebook and makes the MSE prediction loss
        informative from the first step.

        Args:
            logits: (B, N, K) raw logits over the K codebook entries.

        Returns:
            (B, N, C) predicted embedding.
        """
        probs = F.softmax(logits, dim=-1)                      # (B, N, K)
        E = self.codebook.weight                               # (K, C)
        if self.detach_codebook_in_decode:
            E = E.detach()
        return probs @ E                                       # (B, N, C)

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """Fraction of codebook entries used in a batch of indices.

        Args:
            indices: (...) long tensor of code indices in [0, K).

        Returns:
            Scalar float in [0, 1].
        """
        unique = indices.unique().numel()
        return torch.tensor(unique / self.K, dtype=torch.float32)

    @torch.no_grad()
    def perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """Codebook perplexity = exp(H) where H is index entropy.

        A perplexity of K means uniform usage; 1 means total collapse.

        Args:
            indices: (...) long tensor of code indices in [0, K).

        Returns:
            Scalar float.
        """
        counts = torch.bincount(indices.flatten(), minlength=self.K).float()
        probs = counts / counts.sum().clamp(min=1)
        entropy = -(probs * (probs + 1e-10).log()).sum()
        return entropy.exp()
