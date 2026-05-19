"""S-HR-VQVAE Training Loop.

Three sequential stages on a synthetic rotating-shapes dataset:
  Stage 1  —  HR-VQVAE reconstruction (encoder + quantizer + decoder).
  Stage 2  —  AST-PM prediction        (frozen encoder/quantizer, cross-entropy).
  Stage 3  —  Joint fine-tuning        (AST-PM + decoder, Gumbel-Softmax, Eq 9).

Usage
-----
# Full 3-stage run (recommended first test):
python example_code/shr_vqvae_training_loop.py

# Quick smoke-test with tiny model:
python example_code/shr_vqvae_training_loop.py \
    --image_size 32 --M 4 --embedding_dim 32 --base_channels 16 \
    --T 3 --S 3 \
    --stage1_epochs 2 --stage2_epochs 2 --stage3_epochs 2 \
    --batch_size 4 --num_workers 0 --dataset_size 50 \
    --log_dir logs/shr_vqvae_test

# Resume from a specific stage checkpoint:
python example_code/shr_vqvae_training_loop.py \
    --start_stage 2 --load_ckpt logs/shr_vqvae/stage1_final.pt
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

# ---- project root on path ------------------------------------------------- #
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset  # noqa: E402
from model.vid.shr_vqvae import SHRVQVAEConfig, SHRVQVAEModel              # noqa: E402
from shr_vqvae_viz import (                                               # noqa: E402
    save_labeled_prediction_panel,
    save_labeled_reconstruction_panel,
)


# =========================================================================== #
#  Helpers
# =========================================================================== #

def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(True)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def make_dataloader(
    cfg: SHRVQVAEConfig,
    dataset_size: int,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> DataLoader:
    hparams = SimpleNamespace(
        context_length=cfg.T + cfg.S,
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
        shape_cache_dir="data/vid/shape_cache",
        shape_no_imagenet_norm=True,   # returns [0,1] tensors — matches Sigmoid decoder
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


class CSVLogger:
    """Append-mode CSV log for per-step metrics."""

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
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})


def save_checkpoint(
    model: SHRVQVAEModel,
    stage: int,
    epoch: int,
    log_dir: str,
    suffix: str = "",
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    name = f"stage{stage}_epoch{epoch:03d}{suffix}.pt"
    path = os.path.join(log_dir, name)
    torch.save(
        {
            "stage": stage,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "cfg": model.cfg,
        },
        path,
    )
    print(f"  [ckpt] saved -> {path}")


def load_checkpoint(model: SHRVQVAEModel, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  [ckpt] loaded <- {path}  (stage {ckpt.get('stage')}, epoch {ckpt.get('epoch')})")


@torch.no_grad()
def resolve_viz_num_future(args: argparse.Namespace, cfg: SHRVQVAEConfig) -> int:
    if args.viz_num_future <= 0:
        return cfg.S
    return min(args.viz_num_future, cfg.S)


# =========================================================================== #
#  Training stages
# =========================================================================== #

def train_stage1(
    model: SHRVQVAEModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    log_dir: str,
) -> None:
    """Stage 1: Disjoint HR-VQVAE training (reconstruction + VQ losses)."""
    print("\n" + "=" * 60)
    print("STAGE 1  —  HR-VQVAE reconstruction")
    print("=" * 60)

    unfreeze(model.encoder)
    unfreeze(model.quantizer)
    unfreeze(model.decoder)
    freeze(model.ast_pms)

    optimizer = torch.optim.AdamW(
        model.vqvae_params(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage1_epochs * len(loader), eta_min=1e-5
    )
    logger = CSVLogger(os.path.join(log_dir, "stage1.csv"))

    global_step = 0
    last_batch: torch.Tensor | None = None
    for epoch in range(1, args.stage1_epochs + 1):
        epoch_t0 = time.time()
        for batch in loader:
            last_batch = batch
            # Flatten all frames in the sequence into a single batch of images
            B, TS, C, H, W = batch.shape
            frames = batch.reshape(B * TS, C, H, W).to(device)

            optimizer.zero_grad()
            loss, metrics = model.stage1_loss(frames)
            loss.backward()
            nn.utils.clip_grad_norm_(model.vqvae_params(), 1.0)
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
                logger.log(row)
                print(
                    f"  s1 step {global_step:5d} | "
                    f"loss {row['loss']:.4f}  recon {row['recon']:.4f}  vq {row['vq']:.4f}"
                )

            if args.viz_every > 0 and global_step % args.viz_every == 0:
                save_labeled_reconstruction_panel(
                    model, batch, global_step, log_dir, device, tag="recon_panel"
                )

        ep_time = time.time() - epoch_t0
        print(f"  epoch {epoch}/{args.stage1_epochs}  ({ep_time:.1f}s)")

        if epoch % args.save_every == 0 or epoch == args.stage1_epochs:
            save_checkpoint(model, stage=1, epoch=epoch, log_dir=log_dir,
                            suffix="_final" if epoch == args.stage1_epochs else "")

        if (
            args.viz_end_of_stage
            and epoch == args.stage1_epochs
            and last_batch is not None
        ):
            save_labeled_reconstruction_panel(
                model,
                last_batch,
                global_step,
                log_dir,
                device,
                tag="recon_panel_stage1_final",
            )


def train_stage2(
    model: SHRVQVAEModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    log_dir: str,
) -> None:
    """Stage 2: Disjoint AST-PM training (CE on frozen VQ codes)."""
    print("\n" + "=" * 60)
    print("STAGE 2  —  AST-PM cross-entropy (frozen encoder/quantizer)")
    print("=" * 60)

    freeze(model.encoder)
    freeze(model.quantizer)
    freeze(model.decoder)
    unfreeze(model.ast_pms)

    optimizer = torch.optim.AdamW(
        model.astpm_params(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage2_epochs * len(loader), eta_min=1e-5
    )
    logger = CSVLogger(os.path.join(log_dir, "stage2.csv"))

    global_step = 0
    last_batch: torch.Tensor | None = None
    for epoch in range(1, args.stage2_epochs + 1):
        epoch_t0 = time.time()
        for batch in loader:
            last_batch = batch
            x_seq = batch.to(device)   # (B, T+S, 3, H, W)

            optimizer.zero_grad()
            loss, metrics = model.stage2_loss(x_seq)
            loss.backward()
            nn.utils.clip_grad_norm_(model.astpm_params(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % args.log_every == 0:
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": metrics["loss"].item(),
                    "ce": metrics["ce"].item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                logger.log(row)
                print(
                    f"  s2 step {global_step:5d} | "
                    f"loss {row['loss']:.4f}  ce {row['ce']:.4f}"
                )

        ep_time = time.time() - epoch_t0
        print(f"  epoch {epoch}/{args.stage2_epochs}  ({ep_time:.1f}s)")

        if epoch % args.save_every == 0 or epoch == args.stage2_epochs:
            save_checkpoint(model, stage=2, epoch=epoch, log_dir=log_dir,
                            suffix="_final" if epoch == args.stage2_epochs else "")

        if (
            args.viz_end_of_stage
            and epoch == args.stage2_epochs
            and last_batch is not None
        ):
            save_labeled_prediction_panel(
                model=model,
                batch=last_batch,
                step=global_step,
                log_dir=log_dir,
                device=device,
                num_future=resolve_viz_num_future(args, model.cfg),
                show_indices=args.viz_show_indices,
                tag="pred_panel_stage2_final",
            )


def train_stage3(
    model: SHRVQVAEModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    log_dir: str,
) -> None:
    """Stage 3: Joint fine-tuning (AST-PM + Decoder, Eq 9)."""
    print("\n" + "=" * 60)
    print("STAGE 3  —  Joint fine-tuning (Gumbel-Softmax, Eq 9)")
    print("=" * 60)

    freeze(model.encoder)
    freeze(model.quantizer)
    unfreeze(model.ast_pms)
    unfreeze(model.decoder)

    joint_params = model.astpm_params() + model.decoder_params()
    optimizer = torch.optim.AdamW(joint_params, lr=args.lr * 0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage3_epochs * len(loader), eta_min=1e-6
    )
    logger = CSVLogger(os.path.join(log_dir, "stage3.csv"))

    global_step = 0
    last_batch: torch.Tensor | None = None
    for epoch in range(1, args.stage3_epochs + 1):
        epoch_t0 = time.time()
        for batch in loader:
            last_batch = batch
            x_seq = batch.to(device)   # (B, T+S, 3, H, W)

            optimizer.zero_grad()
            loss, metrics = model.stage3_loss(x_seq)
            loss.backward()
            nn.utils.clip_grad_norm_(joint_params, 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % args.log_every == 0:
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": metrics["loss"].item(),
                    "ce": metrics["ce"].item(),
                    "recon": metrics["recon"].item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                logger.log(row)
                print(
                    f"  s3 step {global_step:5d} | "
                    f"loss {row['loss']:.4f}  ce {row['ce']:.4f}  recon {row['recon']:.4f}"
                )

            if args.viz_every > 0 and global_step % args.viz_every == 0:
                try:
                    save_labeled_prediction_panel(
                        model=model,
                        batch=batch,
                        step=global_step,
                        log_dir=log_dir,
                        device=device,
                        num_future=resolve_viz_num_future(args, model.cfg),
                        show_indices=args.viz_show_indices,
                        tag="pred_panel",
                    )
                except (RuntimeError, ValueError, OSError) as e:
                    print(f"  [viz] prediction grid skipped: {e}")

        ep_time = time.time() - epoch_t0
        print(f"  epoch {epoch}/{args.stage3_epochs}  ({ep_time:.1f}s)")

        if epoch % args.save_every == 0 or epoch == args.stage3_epochs:
            save_checkpoint(model, stage=3, epoch=epoch, log_dir=log_dir,
                            suffix="_final" if epoch == args.stage3_epochs else "")

        if (
            args.viz_end_of_stage
            and epoch == args.stage3_epochs
            and last_batch is not None
        ):
            save_labeled_prediction_panel(
                model=model,
                batch=last_batch,
                step=global_step,
                log_dir=log_dir,
                device=device,
                num_future=resolve_viz_num_future(args, model.cfg),
                show_indices=args.viz_show_indices,
                tag="pred_panel_stage3_final",
            )


# =========================================================================== #
#  Entry point
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S-HR-VQVAE training")

    # ---- Model ---- #
    p.add_argument("--image_size",    type=int, default=64)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--num_vq_layers", type=int, default=3,
                   help="Tree depth n.  Memory grows as M^n; use n≤3.")
    p.add_argument("--M",             type=int, default=16,
                   help="Branching factor.  Layer n-1 has M^n codewords.")
    p.add_argument("--vq_beta",       type=float, default=0.25)
    p.add_argument("--astpm_hidden",  type=int, default=256)
    p.add_argument("--astpm_heads",   type=int, default=4)
    p.add_argument("--astpm_blocks",  type=int, default=2)

    # ---- Sequence ---- #
    p.add_argument("--T", type=int, default=5,  help="Context frames")
    p.add_argument("--S", type=int, default=5,  help="Future frames")

    # ---- Joint training ---- #
    p.add_argument("--lambda_joint", type=float, default=0.11)
    p.add_argument("--gumbel_tau",   type=float, default=1.0)

    # ---- Dataset ---- #
    p.add_argument("--dataset_size", type=int, default=2000)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--num_workers",  type=int, default=4)

    # ---- Optimiser ---- #
    p.add_argument("--lr", type=float, default=3e-4)

    # ---- Stages ---- #
    p.add_argument("--stage1_epochs", type=int, default=5)
    p.add_argument("--stage2_epochs", type=int, default=5)
    p.add_argument("--stage3_epochs", type=int, default=5)
    p.add_argument("--start_stage",   type=int, default=1,
                   choices=[1, 2, 3], help="Resume from this stage.")
    p.add_argument("--load_ckpt",     type=str, default="",
                   help="Path to checkpoint to load before training.")

    # ---- Logging ---- #
    p.add_argument("--log_dir",   type=str, default="logs/shr_vqvae")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--viz_every", type=int, default=100,
                   help="Save visualisations every N steps (0 = disabled).")
    p.add_argument(
        "--viz_end_of_stage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one labeled visualization panel at the end of each stage.",
    )
    p.add_argument(
        "--viz_num_future",
        type=int,
        default=-1,
        help="Future frames to visualize (-1 means full S).",
    )
    p.add_argument(
        "--viz_show_indices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include per-layer latent-index heatmaps in prediction panels.",
    )
    p.add_argument("--save_every", type=int, default=10,
                   help="Save checkpoint every N epochs.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ---- Build config & model ---- #
    cfg = SHRVQVAEConfig(
        image_h=args.image_size,
        image_w=args.image_size,
        base_channels=args.base_channels,
        embedding_dim=args.embedding_dim,
        num_vq_layers=args.num_vq_layers,
        M=args.M,
        vq_beta=args.vq_beta,
        astpm_hidden=args.astpm_hidden,
        astpm_heads=args.astpm_heads,
        astpm_blocks=args.astpm_blocks,
        T=args.T,
        S=args.S,
        lambda_joint=args.lambda_joint,
        gumbel_tau=args.gumbel_tau,
    )
    model = SHRVQVAEModel(cfg).to(device)

    total = count_params(model)
    enc_q = count_params(model.encoder) + count_params(model.quantizer)
    dec   = count_params(model.decoder)
    astpm = count_params(model.ast_pms)
    print(f"Parameters: total={total:,}  enc+q={enc_q:,}  dec={dec:,}  ast-pm={astpm:,}")
    print(f"Latent grid: {cfg.latent_h}×{cfg.latent_w}  |  "
          f"VQ layers: {cfg.num_vq_layers}  M={cfg.M}  "
          f"(codebook sizes: {[cfg.M**(i+1) for i in range(cfg.num_vq_layers)]})")
    print(f"Sequence: T={cfg.T} context + S={cfg.S} future = {cfg.T+cfg.S} total frames")

    # ---- Optional checkpoint load ---- #
    if args.load_ckpt:
        load_checkpoint(model, args.load_ckpt)

    # ---- Dataset / DataLoader ---- #
    loader = make_dataloader(
        cfg=cfg,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    print(f"Dataset : {args.dataset_size} samples, "
          f"{len(loader)} batches/epoch (bs={args.batch_size})")

    # ---- Run stages ---- #
    if args.start_stage <= 1:
        train_stage1(model, loader, device, args, args.log_dir)

    if args.start_stage <= 2:
        train_stage2(model, loader, device, args, args.log_dir)

    if args.start_stage <= 3:
        train_stage3(model, loader, device, args, args.log_dir)

    print("\nAll stages complete.")


if __name__ == "__main__":
    main()
