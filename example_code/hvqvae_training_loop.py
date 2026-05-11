"""
HVQVAE Training Loop.

Trains the minimal hierarchical VQ-VAE on synthetic 2D shape videos.

Usage
-----
# Quick single-batch overfit test:
python example_code/hvqvae_training_loop.py --overfit_single_batch --steps 500

# Full training run:
python example_code/hvqvae_training_loop.py --steps 10000 --log_every 100

# Multi-stage with logging:
python example_code/hvqvae_training_loop.py \\
    --stages s3 s2 s1 \\
    --batch_size 8 \\
    --T 4 \\
    --steps 10000 \\
    --log_dir logs/hvqvae_test \\
    --save_images_every 500
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Path fix so script can be run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset
from model.vid.hvqvae.config import HVQVAEConfig, HVQVAEStageConfig
from model.vid.hvqvae.model import HVQVAEModel


# --------------------------------------------------------------------------- #
#  Dataset helpers
# --------------------------------------------------------------------------- #


def build_dataset(args: argparse.Namespace) -> VIDShapeSyntheticDataset:
    """Build shape dataset from args."""
    hparams = SimpleNamespace(
        context_length=args.T + 1,
        image_dims=[args.image_size, args.image_size],
        shape_scene_type="DIM_2",
        shape_min_cubes=2,
        shape_max_cubes=6,
        shape_angle_min=15,
        shape_angle_max=45,
        shape_temporal_patterns=[],
        shape_pattern_combining=False,
        shape_accel_min=3,
        shape_accel_max=6,
        shape_oscillation_period_min=1,
        shape_oscillation_period_max=4,
        shape_interruption_period_min=1,
        shape_interruption_period_max=4,
        shape_cache_dir=args.data_dir if args.data_dir else "data/vid/shape_cache",
    )
    return VIDShapeSyntheticDataset(hparams, size=args.dataset_size)


def make_shape_batch(
    B: int, T: int, H: int, W: int, device: torch.device,
) -> torch.Tensor:
    """Return (B, T+1, 3, H, W) of rotating 2D shapes."""
    hparams = SimpleNamespace(
        context_length=T + 1,
        image_dims=[H, W],
        shape_scene_type="DIM_2",
        shape_min_cubes=2,
        shape_max_cubes=6,
        shape_angle_min=15,
        shape_angle_max=45,
        shape_temporal_patterns=[],
        shape_pattern_combining=False,
        shape_accel_min=3,
        shape_accel_max=6,
        shape_oscillation_period_min=1,
        shape_oscillation_period_max=4,
        shape_interruption_period_min=1,
        shape_interruption_period_max=4,
        shape_cache_dir="data/vid/shape_cache",
    )
    ds = VIDShapeSyntheticDataset(hparams, size=B)
    frames = torch.stack([ds[i] for i in range(B)], dim=0)  # (B, T+1, 3, H, W)
    return frames.to(device)


# --------------------------------------------------------------------------- #
#  Model construction
# --------------------------------------------------------------------------- #


def build_model(args: argparse.Namespace, device: torch.device) -> HVQVAEModel:
    """Build HVQVAE model from CLI args."""
    STAGE_INFO = {
        "s1": (64,  8, 8),
        "s2": (128, 4, 4),
        "s3": (256, 2, 2),
    }

    STAGE_K = {
        "s3": 512,
        "s2": 256,
        "s1": 64,
    }

    # Windowing: coarsest=full, middle=2, finest=1
    STAGE_TW = {
        "s3": None,
        "s2": 2,
        "s1": 1,
    }

    stage_cfgs = []
    for stage_name in args.stages:
        C, H, W = STAGE_INFO[stage_name]
        n_heads = max(2, C // 64)
        stage_cfgs.append(HVQVAEStageConfig(
            stage_name=stage_name,
            channels=C,
            H=H, W=W,
            num_codes=STAGE_K[stage_name],
            transformer_dim=C,
            n_heads=n_heads,
            n_layers=args.n_layers,
            temporal_window=STAGE_TW[stage_name],
            spatial_window=None,
        ))

    cfg = HVQVAEConfig(
        stages=stage_cfgs,
        beta=args.beta,
        encoder_h_dim=128,
        encoder_res_h_dim=32,
        encoder_n_res_layers=2,
        decoder_out_size=args.image_size,
        image_size=args.image_size,
    )

    model = HVQVAEModel(cfg).to(device)
    return model


def compute_data_variance(dataset, n_samples: int = 100) -> float:
    """Estimate data variance from a subset of the dataset."""
    samples = []
    n = min(n_samples, len(dataset))
    for i in range(n):
        samples.append(dataset[i])
    data = torch.stack(samples)  # (n, T+1, 3, H, W)
    return data.var().item()


# --------------------------------------------------------------------------- #
#  Image saving
# --------------------------------------------------------------------------- #


def save_comparison_images(
    pred_rgb: torch.Tensor,
    gt_batch: torch.Tensor,
    recon_rgb: Optional[torch.Tensor],
    step: int,
    save_dir: Path,
) -> None:
    """Save predicted, ground-truth, and (optionally) reconstructed frames."""
    save_dir.mkdir(parents=True, exist_ok=True)
    B, T = pred_rgb.shape[:2]

    pred_frames = pred_rgb[0].clamp(0, 1)       # (T, 3, H, W)
    gt_frames = gt_batch[0, 1:T+1].clamp(0, 1)  # (T, 3, H, W)

    if gt_frames.shape[-2:] != pred_frames.shape[-2:]:
        gt_frames = F.interpolate(
            gt_frames, size=pred_frames.shape[-2:],
            mode="bilinear", align_corners=False,
        )

    rows = [pred_frames, gt_frames]
    if recon_rgb is not None:
        recon_frames = recon_rgb[0, 1:T+1].clamp(0, 1)
        if recon_frames.shape[-2:] != pred_frames.shape[-2:]:
            recon_frames = F.interpolate(
                recon_frames, size=pred_frames.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        rows.append(recon_frames)

    grid = torch.cat(rows, dim=0)
    save_image(grid, save_dir / f"step_{step:06d}.png", nrow=T)


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[HVQVAE] Building model on {device} (stages: {args.stages}) ...")
    model = build_model(args, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[HVQVAE] Total params: {total_params:,}  Trainable: {trainable_params:,}")

    # Single Adam optimizer (as in reference)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, amsgrad=True
    )

    # Logging setup
    log_dir = Path(args.log_dir) if args.log_dir else None
    csv_file = None
    csv_writer = None
    img_dir = None

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_file = open(log_dir / "train_log.csv", "w", newline="")

    if args.save_images_every > 0:
        img_dir = (log_dir or Path("logs/_images")) / "predictions"
        img_dir.mkdir(parents=True, exist_ok=True)
        print(f"[HVQVAE] Saving images every {args.save_images_every} steps to {img_dir}")

    # ---- Single-batch overfit test ---------------------------------------- #
    if args.overfit_single_batch:
        print("[HVQVAE] Overfitting single batch ...")
        fixed_batch = make_shape_batch(
            args.batch_size, args.T, args.image_size, args.image_size, device
        )

        # Estimate data variance from this batch
        data_var = fixed_batch.var().item()
        model.set_data_variance(data_var)
        print(f"[HVQVAE] Data variance: {data_var:.4f}")

        model.train()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad()
            out = model(fixed_batch)
            out.total_loss.backward()
            optimizer.step()

            if step % args.log_every == 0 or step == 1:
                perp_str = "  ".join(
                    f"{k}: {v:.1f}" for k, v in out.perplexities.items()
                )
                print(
                    f"  step {step:5d} | "
                    f"loss {out.total_loss.item():.4f} | "
                    f"recon {out.recon_loss.item():.4f} | "
                    f"pred {out.pred_loss.item():.4f} | "
                    f"embed {out.embedding_loss.item():.4f} | "
                    f"perplexity [{perp_str}]"
                )

                if csv_file:
                    row = {
                        "step": step,
                        "total_loss": out.total_loss.item(),
                        "recon_loss": out.recon_loss.item(),
                        "pred_loss": out.pred_loss.item(),
                        "embedding_loss": out.embedding_loss.item(),
                    }
                    for k, v in out.perplexities.items():
                        row[f"perplexity_{k}"] = v
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                    csv_file.flush()

            if img_dir and args.save_images_every > 0 and step % args.save_images_every == 0:
                with torch.no_grad():
                    out_eval = model(fixed_batch)
                save_comparison_images(
                    out_eval.pred_rgb, fixed_batch, out_eval.recon_rgb,
                    step, img_dir,
                )

        if csv_file:
            csv_file.close()
        print("[HVQVAE] Overfit test complete.")
        return

    # ---- Full training loop ----------------------------------------------- #
    print("[HVQVAE] Building dataset ...")
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    print(f"[HVQVAE] Dataset: {len(dataset)} clips, {len(loader)} batches/epoch")

    # Estimate data variance
    data_var = compute_data_variance(dataset)
    model.set_data_variance(data_var)
    print(f"[HVQVAE] Data variance: {data_var:.4f}")

    print("[HVQVAE] Starting training ...")
    model.train()
    step = 0
    epoch = 0

    while step < args.steps:
        epoch += 1
        for batch in loader:
            step += 1
            if step > args.steps:
                break

            batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(batch)
            out.total_loss.backward()
            optimizer.step()

            if step % args.log_every == 0 or step == 1:
                perp_str = "  ".join(
                    f"{k}: {v:.1f}" for k, v in out.perplexities.items()
                )
                print(
                    f"  epoch {epoch} step {step:5d} | "
                    f"loss {out.total_loss.item():.4f} | "
                    f"recon {out.recon_loss.item():.4f} | "
                    f"pred {out.pred_loss.item():.4f} | "
                    f"embed {out.embedding_loss.item():.4f} | "
                    f"perplexity [{perp_str}]"
                )

                if csv_file:
                    row = {
                        "step": step,
                        "total_loss": out.total_loss.item(),
                        "recon_loss": out.recon_loss.item(),
                        "pred_loss": out.pred_loss.item(),
                        "embedding_loss": out.embedding_loss.item(),
                    }
                    for k, v in out.perplexities.items():
                        row[f"perplexity_{k}"] = v
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                    csv_file.flush()

            if img_dir and args.save_images_every > 0 and step % args.save_images_every == 0:
                with torch.no_grad():
                    out_eval = model(batch)
                save_comparison_images(
                    out_eval.pred_rgb, batch, out_eval.recon_rgb,
                    step, img_dir,
                )

    if csv_file:
        csv_file.close()
    print("[HVQVAE] Training complete.")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HVQVAE training loop")
    p.add_argument("--stages", nargs="+", default=["s3", "s2", "s1"],
                   choices=["s1", "s2", "s3"],
                   help="Stages to train (coarsest first)")
    p.add_argument("--beta", type=float, default=0.25,
                   help="VQ commitment cost β")
    p.add_argument("--n_layers", type=int, default=2,
                   help="Transformer layers per stage predictor")
    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--T", type=int, default=4,
                   help="Number of context frames; model predicts T future frames")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Data
    p.add_argument("--dataset_size", type=int, default=1000)
    p.add_argument("--data_dir", type=str, default=None)
    # Modes
    p.add_argument("--overfit_single_batch", action="store_true",
                   help="Overfit on a single fixed batch for debugging")
    p.add_argument("--log_dir", type=str, default=None,
                   help="Directory for CSV logs and images")
    p.add_argument("--save_images_every", type=int, default=0,
                   help="Save comparison images every N steps (0=disabled)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
