"""
VQ-VAE Vector Quantizer with EMA or gradient-trained codebook.

Supports two modes:
  1. EMA codebook (use_ema=True): codebook is a buffer updated via exponential
     moving average. No gradient, no codebook/commitment loss.
  2. Gradient codebook (use_ema=False): codebook is an nn.Parameter trained via
     standard VQ-VAE losses:
       - Codebook loss:   ||sg(z_e) - e_k||²  — pulls codes toward encoder features
       - Commitment loss:  β·||z_e - sg(e_k)||² — anchors encoder to codebook entries

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

Prediction decode (logit → embedding):
    p = softmax(logits)                        logits ∈ R^{B, N, K}
    z_pred = p @ E                             z_pred ∈ R^{B, N, C}
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
    cb_loss : ()          codebook loss ||sg(z_e) - e_k||² (0 in EMA mode).
    commit_loss: ()       commitment loss β·||z_e - sg(e_k)||² (0 in EMA mode).
    diversity_loss: ()    mean pairwise cosine sim of encoder features.
    """
    z_q_st: torch.Tensor
    z_q: torch.Tensor
    indices: torch.Tensor
    cb_loss: torch.Tensor
    commit_loss: torch.Tensor
    diversity_loss: torch.Tensor


# --------------------------------------------------------------------------- #
#  VectorQuantizer
# --------------------------------------------------------------------------- #


class VectorQuantizer(nn.Module):
    """VQ-VAE quantizer for a single hierarchy stage.

    Two modes:
      - EMA (use_ema=True): codebook is a buffer, updated via exponential moving
        average. No gradient to codebook. cb_loss=0, commit_loss=0.
      - Gradient (use_ema=False): codebook is an nn.Parameter trained via
        codebook loss + commitment loss. Standard VQ-VAE.

    Parameters
    ----------
    cfg : VQCodebookConfig
        Codebook size K, dimension C, mode, etc.
    """

    def __init__(self, cfg: VQCodebookConfig):
        super().__init__()
        self.K = cfg.num_codes
        self.C = cfg.code_dim
        self.use_ema = cfg.use_ema
        self.ema_decay = cfg.ema_decay
        self.dead_code_reset = cfg.dead_code_reset
        self.commitment_beta = cfg.commitment_beta
        self._initialized = cfg.init_mode == "random"

        init_weight = torch.randn(self.K, self.C) * (1.0 / (self.C ** 0.5))

        if self.use_ema:
            # EMA mode: codebook is a buffer (no gradient).
            self.register_buffer("codebook_weight", init_weight)
            self.register_buffer("ema_count", torch.ones(self.K))
            self.register_buffer("ema_sum", init_weight.clone())
        else:
            # Gradient mode: codebook is a parameter (trained via optimizer).
            self.codebook_weight = nn.Parameter(init_weight)

        # Backward-compat: `self.codebook.weight` → `self.codebook_weight`.
        self.codebook = _CodebookView(self)

    # ------------------------------------------------------------------ #
    #  Initialization helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def initialize_from_data(self, z_e: torch.Tensor) -> None:
        """Replace codebook entries with randomly-sampled encoder outputs.

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
        if self.use_ema:
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

        In gradient mode: computes codebook loss and commitment loss.
        In EMA mode: updates codebook via EMA, losses are zero.

        Args:
            z_e: (B, N, C) encoder outputs; N = T*H*W.

        Returns:
            QuantizerOutput with z_q_st, z_q, indices, cb_loss, commit_loss.
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

        # ---- Losses -------------------------------------------------------- #
        zero = torch.tensor(0.0, device=z_e.device)

        if self.use_ema:
            # EMA mode: update codebook from detached features, no losses.
            if self.training:
                self._ema_update(z_flat.detach(), indices_flat)
            cb_loss = zero
            commit_loss = zero
        else:
            # Gradient mode: standard VQ-VAE losses.
            # Codebook loss: pull codebook entries toward encoder features.
            cb_loss = F.mse_loss(z_q, z_e.detach().reshape(B, N, C))
            # Commitment loss: anchor encoder features to codebook entries.
            commit_loss = self.commitment_beta * F.mse_loss(
                z_e, z_q.detach().reshape(B, N, C)
            )

        # ---- Diversity loss: penalise encoder feature collapse ------------ #
        M = z_flat.shape[0]
        if self.training and M > 1:
            max_sample = 128
            if M > max_sample:
                idx = torch.randperm(M, device=z_flat.device)[:max_sample]
                z_sample = z_flat[idx]
            else:
                z_sample = z_flat
            z_norm = F.normalize(z_sample, dim=1)
            cos_sim = z_norm @ z_norm.T
            S = z_norm.shape[0]
            mask = ~torch.eye(S, dtype=torch.bool, device=z_flat.device)
            div_loss = cos_sim[mask].mean()
        else:
            div_loss = zero

        return QuantizerOutput(
            z_q_st=z_q_st,
            z_q=z_q,
            indices=indices,
            cb_loss=cb_loss,
            commit_loss=commit_loss,
            diversity_loss=div_loss,
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
        # with randomly-chosen encoder features + small jitter.
        #
        # The threshold scales with M/K (expected tokens per code per batch).
        # Equilibrium ema_count ≈ M/K when assignments are uniform.
        # A code is "dead" if its count falls below half the equilibrium,
        # meaning it gets far less than its fair share of assignments.
        #
        # Examples (B=4, T+1=5):
        #   s_pool K=2048, M=20:  eq=0.0098, threshold=0.0049
        #   s3     K=512,  M=160: eq=0.312,  threshold=0.156
        #   s1     K=16,   M=2560: eq=160,   threshold=1.0 (capped)
        if not self.dead_code_reset:
            return
        tokens_per_code = M / max(self.K, 1)
        dead_threshold = min(1.0, tokens_per_code * 0.5)
        dead_mask = self.ema_count < dead_threshold
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
            # Reset count to equilibrium level, not a fixed 1.0.
            # Setting to 1.0 when equilibrium is 0.01 causes immediate
            # decay back below threshold → perpetual reset cycle.
            reset_count = max(tokens_per_code, dead_threshold * 2)
            self.ema_count[dead_mask] = reset_count

    # ------------------------------------------------------------------ #
    #  Decode: logits → embedding (for predictor)
    # ------------------------------------------------------------------ #

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert predicted code-distribution logits to embedding vectors.

        Computes z_pred = softmax(logits) @ E.

        In EMA mode:     E is detached (codebook has no gradient anyway).
        In gradient mode: E keeps gradient so prediction loss trains the
                          codebook alongside the predictor.

        Args:
            logits: (B, N, K) raw logits over the K codebook entries.

        Returns:
            (B, N, C) predicted embedding.
        """
        probs = F.softmax(logits, dim=-1)                      # (B, N, K)
        if self.use_ema:
            E = self.codebook_weight.detach()                  # EMA: no grad
        else:
            E = self.codebook_weight                           # Gradient: keep grad
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
