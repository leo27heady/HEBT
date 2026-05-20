"""LFQ VQ-VAE training loop on VIDShapeSyntheticDataset.

Supports bottleneck or hierarchical VQ-VAE, plus optional video predictor
(recon_only / disjoint / joint).

Usage
-----
Activate the project environment first, then install the VQ dependency if needed:

    .\\venv\\Scripts\\Activate.ps1
    pip install vector-quantize-pytorch

1) Hierarchical reconstruction only (GT + Recon panels):

    python example_code/lfq_vqvae_training_loop.py ^
        --quantization_mode hierarchical ^
        --dataset_size 2000 --batch_size 8 --context_length 1 ^
        --epochs 20 --viz_every 100 --log_dir logs/lfq_vqvae_recon

2) Joint video training (full panel: GT, Recon, Pred, entropy maps):

    python example_code/lfq_vqvae_training_loop.py ^
        --quantization_mode hierarchical --enable_video_predictor ^
        --train_mode joint --context_length 5 ^
        --dataset_size 500 --batch_size 4 --num_workers 0 ^
        --epochs 10 --viz_every 50 --log_dir logs/lfq_vqvae_joint

3) Disjoint predictor training (load recon checkpoint, freeze VQ stack):

    python example_code/lfq_vqvae_training_loop.py ^
        --quantization_mode hierarchical --enable_video_predictor ^
        --train_mode disjoint --context_length 5 ^
        --load_ckpt logs/lfq_vqvae_recon/checkpoints/epoch_020_step_XXXXXX_final.pt ^
        --dataset_size 500 --batch_size 4 --num_workers 0 ^
        --epochs 10 --viz_every 50 --log_dir logs/lfq_vqvae_disjoint

4) Bottleneck mode smoke test:

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
from lfq_vqvae_viz import save_recon_panel, save_video_panel  # noqa: E402


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
                   help="Frames per sample. recon_only flattens to images; disjoint/joint use full video (T+1).")
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
        default=[32, 512, 4096],
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
    p.add_argument("--enable_video_predictor", action="store_true")
    p.add_argument(
        "--train_mode",
        type=str,
        default="recon_only",
        choices=["recon_only", "disjoint", "joint"],
        help="recon_only: VQ-VAE only, disjoint: predictor only, joint: all losses end-to-end.",
    )
    p.add_argument("--pred_n_heads", type=int, default=8)
    p.add_argument("--pred_n_layers", type=int, default=4)
    p.add_argument("--pred_dim_top", type=int, default=256)
    p.add_argument("--pred_dim_mid", type=int, default=128)
    p.add_argument("--pred_dim_bot", type=int, default=64)
    p.add_argument("--window_top", type=int, default=-1)
    p.add_argument("--window_mid", type=int, default=2)
    p.add_argument("--window_bot", type=int, default=1)
    p.add_argument("--soft_lookup_temperature", type=float, default=1.0)
    p.add_argument("--use_gumbel_softmax", action="store_true")
    p.add_argument("--gumbel_tau", type=float, default=1.0)
    p.add_argument("--max_T", type=int, default=0,
                   help="Predictor positional horizon. 0 = auto(context_length-1).")
    p.add_argument(
        "--disjoint_target",
        type=str,
        default="predictor_and_tail",
        choices=["predictor_only", "decoder_tail_only", "predictor_and_tail"],
        help="Which modules to train in disjoint mode.",
    )
    p.add_argument(
        "--detach_encoder_for_predictor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach encoder quantized features before predictor input.",
    )
    p.add_argument(
        "--detach_parent_features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach top->mid and mid->bot parent features between predictor stages.",
    )
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--lambda_pred_mse", type=float, default=0.0)

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

    cfg_max_t = args.max_T if args.max_T > 0 else max(1, args.context_length - 1)

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
        enable_video_predictor=args.enable_video_predictor,
        predictor_mode="vanilla",
        train_mode=args.train_mode,
        pred_n_heads=args.pred_n_heads,
        pred_n_layers=args.pred_n_layers,
        pred_dim_top=args.pred_dim_top,
        pred_dim_mid=args.pred_dim_mid,
        pred_dim_bot=args.pred_dim_bot,
        window_top=args.window_top,
        window_mid=args.window_mid,
        window_bot=args.window_bot,
        soft_lookup_temperature=args.soft_lookup_temperature,
        use_gumbel_softmax=args.use_gumbel_softmax,
        gumbel_tau=args.gumbel_tau,
        max_T=cfg_max_t,
        detach_encoder_for_predictor=args.detach_encoder_for_predictor,
        detach_parent_features=args.detach_parent_features,
        lambda_ce=args.lambda_ce,
        lambda_pred_mse=args.lambda_pred_mse,
    )
    model = LFQVAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Mode: {cfg.quantization_mode}  fusion: {cfg.fusion}  train_mode: {cfg.train_mode}")
    print(f"Stages: {cfg.spatial_sizes_descending()}  channels: {cfg.stage_channels}")
    if cfg.quantization_mode == "hierarchical":
        print(f"Codebooks: {cfg.stage_codebook_sizes}  lfq_dims: {cfg.stage_lfq_dims}")
    else:
        print(f"Codebook: K={cfg.codebook_size}  lfq_dim={cfg.lfq_dim}")

    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        strict = not (args.enable_video_predictor and args.train_mode == "disjoint")
        model.load_state_dict(ckpt["model_state"], strict=strict)
        print(f"Loaded checkpoint: {args.load_ckpt}")

    if args.train_mode in ("disjoint", "joint") and not cfg.enable_video_predictor:
        raise ValueError("train_mode disjoint/joint requires --enable_video_predictor")
    if args.train_mode in ("disjoint", "joint") and args.context_length < 2:
        raise ValueError("Video training requires --context_length >= 2 (T+1 frames).")
    if args.train_mode in ("disjoint", "joint") and (args.context_length - 1) > cfg.max_T:
        raise ValueError(
            f"context_length-1 ({args.context_length - 1}) exceeds max_T ({cfg.max_T}). "
            "Increase --max_T or reduce --context_length."
        )
    if args.train_mode == "recon_only" and cfg.enable_video_predictor:
        print("Warning: recon_only does not train predictor losses; predictor modules stay unused.")

    trainable: list[torch.nn.Parameter]
    if args.train_mode == "disjoint":
        for p in model.parameters():
            p.requires_grad_(False)
        if args.disjoint_target == "predictor_only":
            trainable = model.predictor_params()
            for p in trainable:
                p.requires_grad_(True)
            model.encoder.eval()
            model.decoder.eval()
            if model.predictor_top is not None:
                model.predictor_top.train()
            if model.predictor_mid is not None:
                model.predictor_mid.train()
            if model.predictor_bot is not None:
                model.predictor_bot.train()
            model.pred_to_quant_top.train()
            model.pred_to_quant_mid.train()
            model.pred_to_quant_bot.train()
        elif args.disjoint_target == "decoder_tail_only":
            trainable = model.decoder_image_params()
            for p in trainable:
                p.requires_grad_(True)
            model.encoder.eval()
            model.decoder.eval()
            model.decoder.up_to_image.train()
            model.decoder.head.train()
        else:
            pred_trainable = model.predictor_params()
            tail_trainable = model.decoder_image_params()
            trainable = pred_trainable + tail_trainable
            for p in trainable:
                p.requires_grad_(True)
            model.encoder.eval()
            model.decoder.eval()
            model.decoder.up_to_image.train()
            model.decoder.head.train()
            if model.predictor_top is not None:
                model.predictor_top.train()
            if model.predictor_mid is not None:
                model.predictor_mid.train()
            if model.predictor_bot is not None:
                model.predictor_bot.train()
            model.pred_to_quant_top.train()
            model.pred_to_quant_mid.train()
            model.pred_to_quant_bot.train()
    elif args.train_mode == "joint":
        for p in model.parameters():
            p.requires_grad_(True)
        trainable = list(model.parameters())
        model.train()
    else:
        trainable = list(model.parameters())
        model.train()

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

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
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
            batch_video = batch.to(device) if batch.dim() == 5 else None

            optimizer.zero_grad()
            if args.train_mode == "recon_only":
                loss, metrics = model.loss(frames)
            else:
                assert batch_video is not None
                loss, metrics = model.video_loss(batch_video, train_mode=args.train_mode)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % args.log_every == 0:
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": metrics["loss"].item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if "recon" in metrics:
                    row["recon"] = metrics["recon"].item()
                if "vq" in metrics:
                    row["vq"] = metrics["vq"].item()
                if "prior_ce" in metrics:
                    row["prior_ce"] = metrics["prior_ce"].item()
                    row["vq_bot"] = metrics["vq_bot"].item()
                    row["vq_mid"] = metrics["vq_mid"].item()
                    row["vq_top"] = metrics["vq_top"].item()
                if "ce" in metrics:
                    row["ce"] = metrics["ce"].item()
                    row["ce_top"] = metrics["ce_top"].item()
                    row["ce_mid"] = metrics["ce_mid"].item()
                    row["ce_bot"] = metrics["ce_bot"].item()
                    row["pred_mse"] = metrics["pred_mse"].item()
                if "gamma_abs" in metrics:
                    row["gamma_abs"] = float(metrics["gamma_abs"])
                logger.log(row)
                msg = f"  step {global_step:5d} | loss {row['loss']:.4f}"
                if "recon" in row:
                    msg += f"  recon {row['recon']:.4f}"
                if "vq" in row:
                    msg += f"  vq {row['vq']:.4f}"
                if "prior_ce" in row:
                    msg += f"  prior_ce {row['prior_ce']:.4f}"
                if "ce" in row:
                    msg += f"  ce {row['ce']:.4f}  pred_mse {row['pred_mse']:.4f}"
                if "gamma_abs" in row:
                    msg += f"  gamma {row['gamma_abs']:.4f}"
                print(msg)

            if args.viz_every > 0 and global_step % args.viz_every == 0:
                if (
                    batch.dim() == 5
                    and batch.shape[1] > 1
                    and cfg.quantization_mode == "hierarchical"
                ):
                    save_video_panel(
                        model,
                        batch.to(device),
                        global_step,
                        args.log_dir,
                        device,
                        tag="panel",
                    )
                else:
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
        if (
            last_batch.dim() == 5
            and last_batch.shape[1] > 1
            and cfg.quantization_mode == "hierarchical"
        ):
            path = save_video_panel(
                model,
                last_batch.to(device),
                global_step,
                args.log_dir,
                device,
                tag="panel_final",
            )
        else:
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
