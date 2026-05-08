"""
VQ-HVEBT Training Loop Example.

Trains VQ-HVEBT (single or multi-stage) to predict quantized video latents
one step into the future.

Usage
-----
python example_code/vq_hvebt_training_loop.py \
    --weights_path clip/MobileCLIP2-S0/mobileclip2_s0.pt \
    --stages s3 s2 s1 \
    --num_codes 512 \
    --batch_size 4 \
    --T 4 \
    --lr 3e-4 \
    --encoder_lr_scale 0.1 \
    --steps 10000 \
    --log_every 100

For a quick sanity check on a single batch (overfitting test with 2D shapes):
python example_code/vq_hvebt_training_loop.py \
    --weights_path clip/MobileCLIP2-S0/mobileclip2_s0.pt \
    --overfit_single_batch \
    --steps 500
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# ---- path fix so script can be run from project root ---------------------- #
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset  # noqa: E402
from model.vid.vq_hvebt.config import (
    VQCodebookConfig,
    VQHVEBTConfig,
    VQStageConfig,
    _default_stages,
)
from model.vid.vq_hvebt.hierarchy import VQHVEBTModel


# --------------------------------------------------------------------------- #
#  Synthetic video dataset for testing
# --------------------------------------------------------------------------- #


def make_synthetic_batch(
    B: int,
    T: int,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """Return (B, T+1, 3, H, W) random video in [0, 1]."""
    return torch.rand(B, T + 1, 3, H, W, device=device)


def make_shape_batch(
    B: int,
    T: int,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """Return (B, T+1, 3, H, W) of rotating 2D shapes (ImageNet-normalised).

    Uses VIDShapeSyntheticDataset (disk-cached) so repeated calls are fast.
    """
    hparams = SimpleNamespace(
        context_length=T + 1,
        image_dims=[H, W],
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
    )
    ds = VIDShapeSyntheticDataset(hparams, size=B)
    frames = torch.stack([ds[i] for i in range(B)], dim=0)  # (B, T+1, 3, H, W)
    return frames.to(device)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #


def build_dataset(args: argparse.Namespace) -> VIDShapeSyntheticDataset:
    """Build shape dataset from args (generates or loads from cache)."""
    hparams = SimpleNamespace(
        context_length=args.T + 1,
        image_dims=[args.image_size, args.image_size],
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
        shape_cache_dir=args.data_dir if args.data_dir else "data/vid/shape_cache",
    )
    return VIDShapeSyntheticDataset(hparams, size=args.dataset_size)


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #


def build_model(args: argparse.Namespace, device: torch.device) -> VQHVEBTModel:
    """Construct VQ-HVEBT model with selected stages."""
    STAGE_INFO = {
        "s1": (128, 32, 32),
        "s2": (256, 16, 16),
        "s3": (512,  8,  8),
    }

    stage_cfgs: List[VQStageConfig] = []
    for stage_name in args.stages:
        C, H, W = STAGE_INFO[stage_name]
        stage_cfgs.append(VQStageConfig(
            clip_stage_name=stage_name,
            clip_channels=C,
            H=H, W=W,
            transformer_dim=args.transformer_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            mcmc_steps=args.mcmc_steps,
            mcmc_step_size=args.mcmc_step_size,
            soft_target_tau=args.soft_target_tau,
            codebook=VQCodebookConfig(
                num_codes=args.num_codes,
                code_dim=C,
                init_mode="data_first_batch",
                ema_decay=args.ema_decay,
                dead_code_reset=args.dead_code_reset,
            ),
            pred_loss_weight=1.0,
            cb_loss_weight=0.0,
            commit_loss_weight=0.0,
        ))

    cfg = VQHVEBTConfig(
        stages=stage_cfgs,
        train_encoder=not args.freeze_encoder,
        encoder_lr_scale=args.encoder_lr_scale,
        weights_path=args.weights_path,
        use_decoder=args.use_decoder,
        contrastive_loss_weight=args.contrastive_loss_weight,
        encoder_warmup_steps=args.encoder_warmup_steps,
    )
    model = VQHVEBTModel(cfg).to(device)
    return model


def build_optimizer(model: VQHVEBTModel, base_lr: float) -> torch.optim.Optimizer:
    """Build a single AdamW optimizer with encoder at reduced LR.

    With EMA codebook there is no codebook gradient, no commitment loss,
    and no 10⁹-scale gradient explosion. A single AdamW with per-group
    LR is sufficient. No separate SGD needed.
    """
    return torch.optim.AdamW(
        model.parameter_groups(base_lr),
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )


def save_pred_images(pred_rgb: torch.Tensor, gt_batch: torch.Tensor, step: int, save_dir: Path) -> None:
    """Save predicted and ground-truth frames as image grids.

    Args:
        pred_rgb: (B, T, 3, H_out, W_out) predicted frames in [0, 1].
        gt_batch: (B, T+1, 3, H, W) full batch (context+future) in [0, 1].
        step: Current training step number.
        save_dir: Directory to write images into.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    B, T = pred_rgb.shape[:2]
    # Take first sample in batch, save each predicted frame + corresponding GT
    pred_frames = pred_rgb[0].clamp(0, 1)       # (T, 3, H, W)
    gt_frames = gt_batch[0, 1:T+1].clamp(0, 1)  # (T, 3, H, W) future frames

    # Resize GT to match pred resolution if needed
    if gt_frames.shape[-2:] != pred_frames.shape[-2:]:
        gt_frames = nn.functional.interpolate(
            gt_frames, size=pred_frames.shape[-2:], mode="bilinear", align_corners=False
        )

    # Stack pred and GT vertically: top=pred, bottom=GT for each frame
    grid = torch.cat([pred_frames, gt_frames], dim=0)  # (2*T, 3, H, W)
    save_image(grid, save_dir / f"step_{step:06d}.png", nrow=T)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[VQ-HVEBT] Building model on {device} (stages: {args.stages}) ...")
    model = build_model(args, device)

    # Verify cross-attention is active for multi-stage configs.
    if len(args.stages) > 1:
        n_cross = sum(1 for p in model.predictors.values() if p.use_cross_attn)
        print(f"[VQ-HVEBT] Multi-stage: {n_cross}/{len(args.stages)-1} finer stages have cross-attention from parent.")
        if n_cross == 0:
            raise RuntimeError(
                "Multiple stages requested but no cross-attention found! "
                "Check that stages are ordered coarsest-first (e.g. --stages s3 s2 s1)."
            )

    optimizers = build_optimizer(model, args.lr)
    # Per-group param lists for separate grad clipping.
    # Encoder grads are huge (raw CLIP backbone) and would starve predictor
    # if clipped jointly.
    enc_params = model.encoder_params()
    pred_params = model.non_encoder_params()

    # ---- logging setup ---------------------------------------------------- #
    log_dir = Path(args.log_dir) if args.log_dir else None
    csv_file = None
    csv_writer = None
    img_dir = None

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_file = open(log_dir / "train_log.csv", "w", newline="")
        csv_writer = None  # header written on first step (dynamic columns)

    if args.save_images_every > 0:
        if not args.use_decoder:
            print("[WARNING] --save_images_every requires --use_decoder; image saving disabled.")
            args.save_images_every = 0
        else:
            img_dir = (log_dir or Path("logs/_images")) / "predictions"
            img_dir.mkdir(parents=True, exist_ok=True)
            print(f"[VQ-HVEBT] Saving predicted images every {args.save_images_every} steps to {img_dir}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[VQ-HVEBT] Total params: {total_params:,}  Trainable: {trainable_params:,}")

    # ---- single-batch overfit test ---------------------------------------- #
    if args.overfit_single_batch:
        print("[VQ-HVEBT] Overfitting single batch ...")
        fixed_batch = make_shape_batch(args.batch_size, args.T, args.image_size, args.image_size, device)
        print(f"[VQ-HVEBT] Initializing codebooks from data ...")
        model.maybe_initialize_codebooks(fixed_batch)

        model.train()
        for step in range(1, args.steps + 1):
            optimizers.zero_grad()
            out = model.forward_loss(fixed_batch)
            out.total_loss.backward()
            nn.utils.clip_grad_norm_(enc_params, max_norm=1.0)
            nn.utils.clip_grad_norm_(pred_params, max_norm=1.0)
            optimizers.step()

            if step % args.log_every == 0 or step == 1:
                m = out.metrics
                pred_l = next(v for k, v in m.items() if "loss_pred" in k)
                cb_l = next(v for k, v in m.items() if "loss_cb" in k)
                commit_l = next(v for k, v in m.items() if "loss_commit" in k)
                usage = next(v for k, v in m.items() if "codebook_usage" in k)
                perpl = next(v for k, v in m.items() if "codebook_perplexity" in k)
                print(
                    f"  step {step:>5}  total={out.total_loss.item():.4f}"
                    f"  pred={pred_l:.4f}  cb={cb_l:.4f}  commit={commit_l:.4f}"
                    f"  usage={usage:.3f}  perplexity={perpl:.1f}"
                )
                # CSV logging
                if csv_file:
                    row = {"step": step, "total_loss": out.total_loss.item(), **m}
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                    csv_file.flush()

            # Image saving
            if args.save_images_every > 0 and step % args.save_images_every == 0:
                if out.pred_rgb is not None:
                    save_pred_images(out.pred_rgb.detach(), fixed_batch.detach(), step, img_dir)

        if csv_file:
            csv_file.close()
        return

    # ---- real training loop ----------------------------------------------- #
    print("[VQ-HVEBT] Building dataset ...")
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    print(f"[VQ-HVEBT] Dataset: {len(dataset)} clips, {len(loader)} batches/epoch")

    print("[VQ-HVEBT] Starting training ...")
    model.train()
    codebook_init_done = False
    step = 0
    epoch = 0

    while step < args.steps:
        epoch += 1
        for batch in loader:
            step += 1
            if step > args.steps:
                break
            batch = batch.to(device, non_blocking=True)

            # Codebook initialization from first real batch.
            if not codebook_init_done:
                did_init = model.maybe_initialize_codebooks(batch)
                codebook_init_done = True
                if did_init:
                    print("[VQ-HVEBT] Codebooks initialized from first data batch.")

            t0 = time.perf_counter()
            optimizers.zero_grad()
            out = model.forward_loss(batch)
            out.total_loss.backward()
            nn.utils.clip_grad_norm_(enc_params, max_norm=1.0)
            nn.utils.clip_grad_norm_(pred_params, max_norm=1.0)
            optimizers.step()
            dt = time.perf_counter() - t0

            if step % args.log_every == 0 or step == 1:
                m = out.metrics
                parts = [f"step {step:>6}  total={out.total_loss.item():.4f}  dt={dt*1000:.0f}ms"]
                for key, val in sorted(m.items()):
                    parts.append(f"{key}={val:.4f}")
                print("  ".join(parts))

                # CSV logging
                if csv_file:
                    row = {"step": step, "total_loss": out.total_loss.item(), "dt_ms": dt * 1000, **m}
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                    csv_file.flush()

            # Image saving
            if args.save_images_every > 0 and step % args.save_images_every == 0:
                if out.pred_rgb is not None:
                    save_pred_images(out.pred_rgb.detach(), batch.detach(), step, img_dir)

    if csv_file:
        csv_file.close()
    print("[VQ-HVEBT] Training complete.")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VQ-HVEBT training loop")
    p.add_argument("--weights_path", default="clip/MobileCLIP2-S0/mobileclip2_s0.pt")
    p.add_argument("--stages", nargs="+", default=["s1"],
                   choices=["s1", "s2", "s3"],
                   help="Stages to train (coarsest first), e.g. --stages s3 s2 s1")
    p.add_argument("--use_decoder", action="store_true",
                   help="Attach pixel decoder on the finest stage")
    p.add_argument("--num_codes", type=int, default=512)
    p.add_argument("--ema_decay", type=float, default=0.99,
                   help="EMA decay for codebook updates (0.99-0.999)")
    p.add_argument("--dead_code_reset", action="store_true", default=True,
                   help="Reset dead codebook entries with encoder samples (default: enabled)")
    p.add_argument("--no_dead_code_reset", dest="dead_code_reset", action="store_false",
                   help="Disable dead code reset")
    p.add_argument("--transformer_dim", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--mcmc_steps", type=int, default=20)
    p.add_argument("--mcmc_step_size", type=float, default=10.0)
    p.add_argument("--soft_target_tau", type=float, default=0.0,
                   help="Soft target temperature (>0: smooth distance-based targets, 0: hard one-hot)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--T", type=int, default=4,
                   help="Number of context frames; model predicts T future frames")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder_lr_scale", type=float, default=0.1)
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze CLIP encoder (useful for debugging predictor in isolation)")
    p.add_argument("--encoder_warmup_steps", type=int, default=0,
                   help="Freeze encoder gradient for first N steps (stabilizes target codes)")
    p.add_argument("--contrastive_loss_weight", type=float, default=0.0,
                   help="Weight for contrastive energy loss (E(true) < E(predicted))")
    p.add_argument("--dataset_size", type=int, default=1000,
                   help="Number of shape clips to generate (ignored if --data_dir points to existing cache)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to pre-created shape cache folder. If None, generates to data/vid/shape_cache/")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overfit_single_batch", action="store_true",
                   help="Run overfitting test on a single fixed batch (2D rotating shapes)")
    p.add_argument("--log_dir", type=str, default=None,
                   help="Directory to save training logs (CSV). If None, no logs saved.")
    p.add_argument("--save_images_every", type=int, default=0,
                   help="Save predicted images every N steps (requires --use_decoder, 0=disabled)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
