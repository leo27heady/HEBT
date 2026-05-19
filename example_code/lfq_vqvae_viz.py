"""Reconstruction visualizations for LFQ VQ-VAE training."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
from torchvision.utils import save_image

from model.vid.lfq_vqvae import LFQVAE


def _code_usage(indices: torch.Tensor, codebook_size: int) -> Tuple[int, int]:
    unique = indices.detach().unique().numel()
    return unique, codebook_size


@torch.no_grad()
def _hierarchical_usage(out: dict, cfg) -> Dict[str, Tuple[int, int]]:
    sizes = cfg.stage_codebook_sizes
    return {
        "bot": _code_usage(out["idx_bot"], sizes[0]),
        "mid": _code_usage(out["idx_mid"], sizes[1]),
        "top": _code_usage(out["idx_top"], sizes[2]),
    }


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

    if model.is_hierarchical:
        usage = _hierarchical_usage(out, model.cfg)
        parts = [f"{k}={u}/{k_tot}" for k, (u, k_tot) in usage.items()]
        print(f"  [viz] code usage @ step {step}: " + "  ".join(parts))

    model.train()
    return out_path
