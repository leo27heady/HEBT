"""
Explicit loss functions for VQ-HVEBT.

Having loss formulas in one place makes the math auditable and easy to swap
during ablations. Every function returns a scalar tensor that participates in
the computation graph (no in-place ops, no detach here — detaches are the
caller's responsibility as specified).

Loss taxonomy per stage k
-------------------------

1. Prediction loss (trains the EBT predictor):
       L_pred = ||z_pred - sg(z_q_future)||²
   where z_pred is the MCMC output (decoded from logits via softmax @ E)
   and z_q_future is the quantized CLIP feature of the true next frame.
   The target MUST be detached by the caller.

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
