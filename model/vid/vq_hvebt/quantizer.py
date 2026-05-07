"""
VQ-VAE Vector Quantizer with EMA codebook and straight-through estimator.

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

Gradient flow:
  - The ENCODER is trained ONLY via straight-through: gradients from
    downstream (prediction loss) pass through z_q_st back to z_e.
  - The CODEBOOK is trained via EMA updates (no gradient). Each entry
    tracks the exponential moving average of encoder outputs assigned to it.
  - NO commitment loss, NO codebook loss. The EMA update replaces both.

EMA codebook update (VQ-VAE-2 style):
    For each code j, track:
        ema_count_j  = decay * ema_count_j + (1-decay) * count_j
        ema_sum_j    = decay * ema_sum_j   + (1-decay) * sum_of_assigned_z_e_j
        E_j          = ema_sum_j / ema_count_j

    This makes the codebook track encoder outputs without ANY gradient
    interaction, eliminating the 10⁹-magnitude gradient explosion that
    commitment loss through LayerNorm produces.

Prediction decode (logit → embedding):
    p = softmax(logits)                        logits ∈ R^{B, N, K}
    z_pred = p @ E                             z_pred ∈ R^{B, N, C}

    The codebook is DETACHED during decode because the codebook is not
    trained by gradient (it's EMA-only). Prediction loss flows through
    softmax → logits only.
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
    cb_loss : ()          scalar zero (kept for API compat; codebook trains via EMA).
    commit_loss: ()       scalar zero (no commitment loss in EMA mode).
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
    """EMA-updated VQ-VAE quantizer for a single hierarchy stage.

    The codebook is NOT trained by gradient. Instead, it tracks encoder
    outputs via exponential moving average (VQ-VAE-2 style). This eliminates
    the gradient explosion that commitment loss through normalized features
    produces.

    Parameters
    ----------
    cfg : VQCodebookConfig
        Codebook size K, dimension C, init mode, EMA decay.
    detach_codebook_in_decode : bool
        If True, the codebook is detached when decoding logits to embeddings.
        Default True for EMA mode (codebook has no grad anyway).
    """

    def __init__(
        self,
        cfg: VQCodebookConfig,
        detach_codebook_in_decode: bool = True,
    ):
        super().__init__()
        self.K = cfg.num_codes
        self.C = cfg.code_dim
        self.ema_decay = cfg.ema_decay
        self.dead_code_reset = cfg.dead_code_reset
        self.detach_codebook_in_decode = detach_codebook_in_decode
        self._initialized = cfg.init_mode == "random"

        # Codebook as a buffer (NOT a parameter — no gradient).
        self.register_buffer("codebook_weight", torch.randn(self.K, self.C) * (1.0 / (self.C ** 0.5)))
        # EMA tracking buffers.
        self.register_buffer("ema_count", torch.ones(self.K))
        self.register_buffer("ema_sum", self.codebook_weight.clone())

        # Backward-compat property so decode_logits and other code can still
        # access `self.codebook.weight`.
        self.codebook = _CodebookView(self)

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
        feat_std = z_e.std().item()
        jitter_scale = max(0.1 * feat_std, 0.02)
        jitter = torch.randn_like(sampled) * jitter_scale
        init_data = sampled + jitter
        self.codebook_weight.copy_(init_data)
        self.ema_sum.copy_(init_data)
        self.ema_count.fill_(1.0)
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------ #
    #  Core: encode (quantize) + EMA update
    # ------------------------------------------------------------------ #

    def encode(self, z_e: torch.Tensor) -> QuantizerOutput:
        """Quantize encoder outputs with straight-through estimator.

        The codebook is updated via EMA during training (when self.training).
        No gradient flows to or from the codebook.

        Args:
            z_e: (B, N, C) encoder outputs; N = T*H*W.

        Returns:
            QuantizerOutput with z_q_st, z_q, indices, cb_loss=0, commit_loss=0.
        """
        B, N, C = z_e.shape
        if C != self.C:
            raise ValueError(f"z_e channels {C} != codebook dim {self.C}")

        # ---- nearest-neighbor lookup --------------------------------------- #
        z_flat = z_e.reshape(B * N, C)                         # (M, C)
        E = self.codebook_weight                               # (K, C)

        # ||z - e||² = ||z||² + ||e||² - 2 <z, e>
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)          # (M, 1)
        e_sq = (E ** 2).sum(dim=1, keepdim=True).T             # (1, K)
        dot = z_flat @ E.T                                     # (M, K)
        dist = z_sq + e_sq - 2 * dot                           # (M, K)

        indices_flat = dist.argmin(dim=1)                      # (M,)
        indices = indices_flat.reshape(B, N)                   # (B, N)

        # ---- quantized embedding ------------------------------------------ #
        z_q_flat = E[indices_flat]                              # (M, C)
        z_q = z_q_flat.reshape(B, N, C)                       # (B, N, C)

        # ---- straight-through copy ---------------------------------------- #
        z_q_st = z_e + (z_q - z_e).detach()                   # (B, N, C)

        # ---- EMA codebook update (training only) -------------------------- #
        if self.training:
            self._ema_update(z_flat.detach(), indices_flat)

        # No explicit losses — codebook trains via EMA, encoder via straight-through.
        zero = torch.tensor(0.0, device=z_e.device)
        return QuantizerOutput(
            z_q_st=z_q_st,
            z_q=z_q,
            indices=indices,
            cb_loss=zero,
            commit_loss=zero,
        )

    @torch.no_grad()
    def _ema_update(self, z_flat: torch.Tensor, indices_flat: torch.Tensor) -> None:
        """Update codebook entries via exponential moving average.

        Also resets dead codes (codes with very low assignment count) by
        replacing them with randomly-selected encoder outputs + jitter.

        Args:
            z_flat: (M, C) detached encoder outputs.
            indices_flat: (M,) assigned code indices.
        """
        M = z_flat.shape[0]
        decay = self.ema_decay

        # One-hot assignments: (M, K)
        one_hot = F.one_hot(indices_flat, self.K).float()      # (M, K)

        # Count how many tokens assigned to each code.
        batch_count = one_hot.sum(dim=0)                       # (K,)
        # Sum of assigned encoder features per code.
        batch_sum = one_hot.T @ z_flat                         # (K, C)

        # EMA update.
        self.ema_count.mul_(decay).add_(batch_count, alpha=1 - decay)
        self.ema_sum.mul_(decay).add_(batch_sum, alpha=1 - decay)

        # Laplace smoothing to avoid division by zero for unused codes.
        n = self.ema_count.sum()
        count_smoothed = (self.ema_count + 1e-5) / (n + self.K * 1e-5) * n

        # Update codebook weights.
        self.codebook_weight.copy_(self.ema_sum / count_smoothed.unsqueeze(1))

        # ---- Dead code reset ---------------------------------------------- #
        # Codes with very low EMA count are effectively unused. Replace them
        # with randomly-chosen encoder features + small jitter. This prevents
        # permanent codebook collapse.
        if not self.dead_code_reset:
            return
        dead_mask = self.ema_count < 1.0  # threshold: count < 1 means nearly dead
        n_dead = dead_mask.sum().item()
        if n_dead > 0 and M > 0:
            # Sample random encoder features as replacements.
            replace_idx = torch.randint(0, M, (n_dead,), device=z_flat.device)
            new_codes = z_flat[replace_idx]
            # Add small jitter to avoid exact duplicates.
            jitter = torch.randn_like(new_codes) * 0.01 * z_flat.std()
            new_codes = new_codes + jitter
            # Reset the dead entries.
            self.codebook_weight[dead_mask] = new_codes
            self.ema_sum[dead_mask] = new_codes
            self.ema_count[dead_mask] = 1.0

    # ------------------------------------------------------------------ #
    #  Decode: logits → embedding (for predictor)
    # ------------------------------------------------------------------ #

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert predicted code-distribution logits to embedding vectors.

        Computes z_pred = softmax(logits) @ E, which is a differentiable
        weighted combination of codebook entries. Since the codebook is
        EMA-updated (no gradient), E is always detached here.

        Args:
            logits: (B, N, K) raw logits over the K codebook entries.

        Returns:
            (B, N, C) predicted embedding.
        """
        probs = F.softmax(logits, dim=-1)                      # (B, N, K)
        E = self.codebook_weight.detach()                      # (K, C) — always detach for EMA
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


# --------------------------------------------------------------------------- #
#  Backward-compat helper: allows `quantizer.codebook.weight` access
# --------------------------------------------------------------------------- #


class _CodebookView:
    """Provides `.weight` attribute pointing to the parent's buffer."""
    def __init__(self, parent: VectorQuantizer):
        self._parent = parent

    @property
    def weight(self) -> torch.Tensor:
        return self._parent.codebook_weight
