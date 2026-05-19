"""Reconstruction visualizations for LFQ VQ-VAE training."""

from __future__ import annotations

import os

import torch
from torchvision.utils import save_image

from model.vid.lfq_vqvae import LFQVAE


@torch.no_grad()
def save_recon_panel(
    model: LFQVAE,
    batch: torch.Tensor,
    step: int,
    log_dir: str,
    device: torch.device,
    max_samples: int = 4,
    tag: str = "recon",
) -> str:
    """
    Save a grid: top row inputs, bottom row reconstructions.

    batch: (B, T, C, H, W) or (B, C, H, W)
  """
    model.eval()
    if batch.dim() == 5:
        B, T, C, H, W = batch.shape
        frames = batch.reshape(B * T, C, H, W)
    else:
        frames = batch

    n = min(max_samples, frames.shape[0])
    x = frames[:n].to(device)
    out = model(x)
    x_hat = out["x_hat"].clamp(0.0, 1.0)

    grid = torch.cat([x, x_hat], dim=0)
    frames_dir = os.path.join(log_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    out_path = os.path.join(frames_dir, f"{tag}_step_{step:06d}.png")
    save_image(grid, out_path, nrow=n, padding=2)
    model.train()
    return out_path
