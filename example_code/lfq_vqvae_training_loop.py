"""LFQ VQ-VAE training loop on VIDShapeSyntheticDataset.

Reconstruction-only VQ-VAE with LFQ at the 1x1 bottleneck.

Usage
-----
Activate the project environment first, then install the VQ dependency if needed:

    .\\venv\\Scripts\\Activate.ps1
    pip install vector-quantize-pytorch

Quick smoke test:

    python example_code/lfq_vqvae_training_loop.py ^
        --dataset_size 200 --batch_size 8 --num_workers 0 ^
        --epochs 3 --log_every 10 --viz_every 50 ^
        --log_dir logs/lfq_vqvae_smoke
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset  # noqa: E402
from model.vid.lfq_vqvae import LFQVAE, LFQVAEConfig  # noqa: E402
from lfq_vqvae_viz import save_recon_panel  # noqa: E402


class CSVLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        self._header_written = os.path.isfile(path)

    def log(self, row: Dict) -> None:
        write_header = not self._header_written
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()}
            )


def make_dataloader(
    image_size: int,
    context_length: int,
    dataset_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    hparams = SimpleNamespace(
        context_length=context_length,
        image_dims=[image_size, image_size],
        shape_scene_type="DIM_2",
        shape_min_cubes=2,
        shape_max_cubes=6,
        shape_angle_min=5,
        shape_angle_max=20,
        shape_temporal_patterns=[],
        shape_pattern_combining=False,
        shape_accel_min=3,
        shape_accel_max=6,
        shape_oscillation_period_min=1,
        shape_oscillation_period_max=4,
        shape_interruption_period_min=1,
        shape_interruption_period_max=4,
        shape_cache_dir="data/vid/shape_cache_lfq_vqvae",
        shape_no_imagenet_norm=True,
    )
    dataset = VIDShapeSyntheticDataset(hparams, size=dataset_size, cache=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=(num_workers > 0),
        drop_last=True,
    )


def save_checkpoint(
    model: LFQVAE,
    epoch: int,
    step: int,
    log_dir: str,
    suffix: str = "",
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"epoch_{epoch:03d}_step_{step:06d}{suffix}.pt")
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state": model.state_dict(),
            "cfg": model.cfg,
        },
        path,
    )
    print(f"  [ckpt] saved -> {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LFQ VQ-VAE training")

    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--context_length", type=int, default=1,
                   help="Frames per sample; flattened to images when >1.")
    p.add_argument("--stage_sizes", type=int, nargs="+", default=[16, 4, 1])
    p.add_argument("--stage_channels", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--codebook_size", type=int, default=2**12)
    p.add_argument("--lfq_dim", type=int, default=12)
    p.add_argument("--entropy_loss_weight", type=float, default=0.1)
    p.add_argument("--diversity_gamma", type=float, default=1.0)
    p.add_argument("--vq_loss_weight", type=float, default=1.0)
    p.add_argument("--recon_loss", type=str, default="mse", choices=["mse", "l1"])

    p.add_argument(
        "--quantization_mode",
        type=str,
        default="bottleneck",
        choices=["bottleneck", "hierarchical"],
    )
    p.add_argument(
        "--stage_codebook_sizes",
        type=int,
        nargs="+",
        default=[64, 512, 4096],
        help="Hierarchical: bot, mid, top codebook sizes.",
    )
    p.add_argument(
        "--stage_lfq_dims",
        type=int,
        nargs="+",
        default=None,
        help="Hierarchical LFQ dims (default: log2 of stage_codebook_sizes).",
    )
    p.add_argument(
        "--fusion",
        type=str,
        default="conv",
        choices=["concat", "conv", "gamma"],
        help="Hierarchical skip fusion mode.",
    )
    p.add_argument("--lambda_prior_ce", type=float, default=1.0)
    p.add_argument(
        "--prior_ce_weights",
        type=str,
        default="spatial",
        choices=["spatial", "uniform"],
    )
    p.add_argument("--gamma_l2", type=float, default=0.0)

    p.add_argument("--dataset_size", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=0,
                   help="Stop after N steps (0 = run all epochs).")

    p.add_argument("--log_dir", type=str, default="logs/lfq_vqvae")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--viz_every", type=int, default=100,
                   help="Save recon panels every N steps (0 = disabled).")
    p.add_argument("--save_every", type=int, default=5,
                   help="Save checkpoint every N epochs.")
    p.add_argument("--load_ckpt", type=str, default="")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    stage_lfq_dims = args.stage_lfq_dims
    if stage_lfq_dims is None:
        import math
        stage_lfq_dims = [
            int(math.log2(k)) for k in args.stage_codebook_sizes
        ]

    cfg = LFQVAEConfig(
        image_size=args.image_size,
        quantization_mode=args.quantization_mode,
        stage_sizes=tuple(args.stage_sizes),
        stage_channels=tuple(args.stage_channels),
        codebook_size=args.codebook_size,
        lfq_dim=args.lfq_dim,
        stage_codebook_sizes=tuple(args.stage_codebook_sizes),
        stage_lfq_dims=tuple(stage_lfq_dims),
        entropy_loss_weight=args.entropy_loss_weight,
        diversity_gamma=args.diversity_gamma,
        vq_loss_weight=args.vq_loss_weight,
        recon_loss=args.recon_loss,
        fusion=args.fusion,
        lambda_prior_ce=args.lambda_prior_ce,
        prior_ce_weights=args.prior_ce_weights,
        gamma_l2=args.gamma_l2,
    )
    model = LFQVAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Mode: {cfg.quantization_mode}  fusion: {cfg.fusion}")
    print(f"Stages: {cfg.spatial_sizes_descending()}  channels: {cfg.stage_channels}")
    if cfg.quantization_mode == "hierarchical":
        print(f"Codebooks: {cfg.stage_codebook_sizes}  lfq_dims: {cfg.stage_lfq_dims}")
    else:
        print(f"Codebook: K={cfg.codebook_size}  lfq_dim={cfg.lfq_dim}")

    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {args.load_ckpt}")

    loader = make_dataloader(
        image_size=args.image_size,
        context_length=args.context_length,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Dataset: {args.dataset_size} samples, {len(loader)} batches/epoch "
        f"(bs={args.batch_size}, context={args.context_length})"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * len(loader)), eta_min=1e-5
    )
    logger = CSVLogger(os.path.join(args.log_dir, "train.csv"))

    global_step = 0
    last_batch = None
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        for batch in loader:
            last_batch = batch
            if batch.dim() == 5 and batch.shape[1] > 1:
                B, TS, C, H, W = batch.shape
                frames = batch.reshape(B * TS, C, H, W)
            elif batch.dim() == 5:
                frames = batch[:, 0]
            else:
                frames = batch
            frames = frames.to(device)

            optimizer.zero_grad()
            loss, metrics = model.loss(frames)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % args.log_every == 0:
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": metrics["loss"].item(),
                    "recon": metrics["recon"].item(),
                    "vq": metrics["vq"].item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if "prior_ce" in metrics:
                    row["prior_ce"] = metrics["prior_ce"].item()
                    row["vq_bot"] = metrics["vq_bot"].item()
                    row["vq_mid"] = metrics["vq_mid"].item()
                    row["vq_top"] = metrics["vq_top"].item()
                if "gamma_abs" in metrics:
                    row["gamma_abs"] = float(metrics["gamma_abs"])
                logger.log(row)
                msg = (
                    f"  step {global_step:5d} | loss {row['loss']:.4f}  "
                    f"recon {row['recon']:.4f}  vq {row['vq']:.4f}"
                )
                if "prior_ce" in row:
                    msg += f"  prior_ce {row['prior_ce']:.4f}"
                if "gamma_abs" in row:
                    msg += f"  gamma {row['gamma_abs']:.4f}"
                print(msg)

            if args.viz_every > 0 and global_step % args.viz_every == 0:
                save_recon_panel(
                    model, batch, global_step, args.log_dir, device, tag="recon"
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        print(f"  epoch {epoch}/{args.epochs}  ({time.time() - epoch_t0:.1f}s)")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                model,
                epoch=epoch,
                step=global_step,
                log_dir=os.path.join(args.log_dir, "checkpoints"),
                suffix="_final" if epoch == args.epochs else "",
            )

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if last_batch is not None:
        path = save_recon_panel(
            model,
            last_batch,
            global_step,
            args.log_dir,
            device,
            tag="recon_final",
        )
        print(f"  [viz] final panel -> {path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
