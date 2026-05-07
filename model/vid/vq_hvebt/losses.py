"""
Explicit loss functions for VQ-HVEBT.

Having loss formulas in one place makes the math auditable and easy to swap
during ablations. Every function returns a scalar tensor that participates in
the computation graph (no in-place ops, no detach here — detaches are the
caller's responsibility as specified).

Loss taxonomy per stage k
-------------------------

1. Prediction loss (trains the EBT predictor):
       L_pred = CE(pred_logits, target_indices)
   where pred_logits are the final MCMC logits (B, N, K) and target_indices
   are the code indices of the quantized future frame (B, N).
   This is the NLP EBT analog: cross-entropy directly on logits gives a
   clean gradient signal with no softmax dilution.
   (Legacy MSE variant on decoded embeddings also available.)

2. Codebook loss (trains the codebook):
       L_cb = ||sg(z_e) - z_q||²
   Moves codebook entries toward encoder outputs. z_e must be detached by
   the caller (or the VectorQuantizer.encode() call already handles this).

3. Commitment loss (trains the encoder):
       L_commit = β * ||z_e - sg(z_q)||²
   Keeps encoder outputs near the chosen codebook entry. z_q must be
   detached by the caller (or VectorQuantizer.encode() handles this).

4. Decoder loss (optional, trains the pixel decoder):
       L_dec = ||D(z_pred_finest) - x_future||₁
   MSE variant also provided. The input z_pred_finest should be detached
   IF the intent is to NOT let decoder gradients flow into the predictor
   (decoder is then a pure reconstruction probe). When joint training is
   wanted, pass the live tensor.

Total loss:
    L = Σ_k (λ_pred * L_pred^k + λ_cb * L_cb^k + λ_commit * L_commit^k)
        + λ_dec * L_dec
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Per-stage losses
# --------------------------------------------------------------------------- #


def prediction_loss(
    z_pred: torch.Tensor,
    z_q_future_detached: torch.Tensor,
    kind: str = "mse",
) -> torch.Tensor:
    """Loss that trains the EBT predictor.

    Args:
        z_pred: (B, N, C) predicted embedding from MCMC (live graph).
        z_q_future_detached: (B, N, C) quantized future target — caller MUST
            pass `z_q_future.detach().clone()` here. No assertion is made
            (assertions have runtime cost and would block autograd in traced
            modes), but correctness requires it.
        kind: "mse" or "smooth_l1".

    Returns:
        Scalar loss.
    """
    if kind == "mse":
        return F.mse_loss(z_pred, z_q_future_detached)
    elif kind == "smooth_l1":
        return F.smooth_l1_loss(z_pred, z_q_future_detached)
    else:
        raise ValueError(f"Unknown prediction loss kind: {kind!r}. Use 'mse' or 'smooth_l1'.")


def ce_prediction_loss(
    pred_logits: torch.Tensor,
    target_indices: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss on MCMC logits vs target code indices.

    This is the NLP EBT analog: the training loss operates DIRECTLY on the
    logits (the MCMC optimization variable), giving a clean gradient signal
    with no softmax dilution.

    Args:
        pred_logits: (B, N, K) raw logits over K codebook entries — live graph
            from MCMC's final step.
        target_indices: (B, N) long tensor of ground-truth code indices from
            the quantizer (e.g. qout.indices[:, 1:].reshape(B, N)).

    Returns:
        Scalar loss (mean over all tokens in the batch).
    """
    B, N, K = pred_logits.shape
    # F.cross_entropy expects (*, C) logits and (*,) targets.
    return F.cross_entropy(
        pred_logits.reshape(-1, K),
        target_indices.reshape(-1),
    )


def soft_ce_prediction_loss(
    pred_logits: torch.Tensor,
    z_e_future: torch.Tensor,
    codebook_weight: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Soft cross-entropy loss using distance-based target distribution.

    Instead of hard one-hot targets (which flip discontinuously when VQ
    assignments change), this uses a smooth softmax distribution over
    codes based on L2 distance.  Small encoder/codebook changes produce
    small target changes — eliminating the "moving target" discontinuity.

    Args:
        pred_logits: (B, N, K) raw logits from MCMC final step.
        z_e_future: (B, N, C) encoder features of future frames (DETACHED).
        codebook_weight: (K, C) codebook entries (DETACHED).
        tau: temperature for soft targets.  Larger = softer. Typically 0.1–1.0.
            Scaled internally by mean nearest-code distance for robustness.

    Returns:
        Scalar loss (mean over all tokens in the batch).
    """
    B, N, C = z_e_future.shape
    K = codebook_weight.shape[0]

    # Squared L2 distances: ||z_e - E_j||² for all j
    # z_e: (B*N, C), E: (K, C) → dist: (B*N, K)
    z_flat = z_e_future.reshape(B * N, C)
    z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)        # (B*N, 1)
    e_sq = (codebook_weight ** 2).sum(dim=1, keepdim=True).T  # (1, K)
    dot = z_flat @ codebook_weight.T                     # (B*N, K)
    dist_sq = z_sq + e_sq - 2 * dot                      # (B*N, K)

    # Adaptive temperature: scale tau by mean nearest-code distance.
    # This makes the hyperparameter tau independent of feature magnitude.
    min_dist = dist_sq.min(dim=-1)[0]                    # (B*N,)
    adaptive_tau = tau * (min_dist.mean().clamp(min=1e-6))

    # Soft target distribution (detached — no encoder gradient through targets).
    soft_targets = F.softmax(-dist_sq / adaptive_tau, dim=-1)  # (B*N, K)

    # Soft cross-entropy: -sum(target * log_softmax(pred))
    log_probs = F.log_softmax(pred_logits.reshape(B * N, K), dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1).mean()
    return loss


def codebook_loss(
    z_e_detached: torch.Tensor,
    z_q: torch.Tensor,
) -> torch.Tensor:
    """Loss that trains the codebook entries.

    Moves each code entry e_{k*} toward the encoder output z_e that chose it.

    Args:
        z_e_detached: (B, N, C) encoder output — caller MUST detach this
            (or equivalently pass `z_e.detach()`). The VectorQuantizer.encode()
            method already computes and returns this loss with the correct detach.
        z_q: (B, N, C) quantized embedding (live, codebook gets gradient here).

    Returns:
        Scalar loss.
    """
    return F.mse_loss(z_e_detached, z_q)


def commitment_loss(
    z_e: torch.Tensor,
    z_q_detached: torch.Tensor,
    beta: float = 0.25,
) -> torch.Tensor:
    """Loss that trains the encoder to stay near its chosen codebook entry.

    Args:
        z_e: (B, N, C) encoder output (live, encoder gets gradient here).
        z_q_detached: (B, N, C) quantized embedding — caller MUST detach.
            VectorQuantizer.encode() already computes and returns this.
        beta: commitment weight (typically 0.25).

    Returns:
        Scalar loss.
    """
    return beta * F.mse_loss(z_e, z_q_detached)


# --------------------------------------------------------------------------- #
#  Decoder loss
# --------------------------------------------------------------------------- #


def decoder_loss_l1(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
) -> torch.Tensor:
    """L1 pixel reconstruction loss (typical for image decoders).

    Args:
        pred_rgb: (B, T, 3, H, W) or (B, 3, H, W) in [0, 1].
        gt_rgb: same shape as pred_rgb.

    Returns:
        Scalar loss.
    """
    return F.l1_loss(pred_rgb, gt_rgb)


def decoder_loss_mse(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
) -> torch.Tensor:
    """MSE pixel reconstruction loss.

    Args:
        pred_rgb: (B, T, 3, H, W) or (B, 3, H, W) in [0, 1].
        gt_rgb: same shape as pred_rgb.

    Returns:
        Scalar loss.
    """
    return F.mse_loss(pred_rgb, gt_rgb)


# --------------------------------------------------------------------------- #
#  Aggregation helper
# --------------------------------------------------------------------------- #


def aggregate_stage_losses(
    pred_losses: Dict[str, torch.Tensor],
    cb_losses: Dict[str, torch.Tensor],
    commit_losses: Dict[str, torch.Tensor],
    pred_weights: Dict[str, float],
    cb_weights: Dict[str, float],
    commit_weights: Dict[str, float],
) -> torch.Tensor:
    """Weighted sum of all per-stage losses.

    Args:
        pred_losses    : {stage_name: scalar loss} for each stage.
        cb_losses      : same structure.
        commit_losses  : same structure.
        pred_weights   : {stage_name: λ_pred}.
        cb_weights     : {stage_name: λ_cb}.
        commit_weights : {stage_name: λ_commit}.

    Returns:
        Scalar total loss (sum over all stages and terms).
    """
    total = torch.tensor(0.0, device=next(iter(pred_losses.values())).device)
    for name in pred_losses:
        total = total + pred_weights[name] * pred_losses[name]
        total = total + cb_weights[name] * cb_losses[name]
        total = total + commit_weights[name] * commit_losses[name]
    return total
