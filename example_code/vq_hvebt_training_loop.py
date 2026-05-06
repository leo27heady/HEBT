"""
VQ-HVEBT Single-Stage Training Loop Example.

This is the minimal training loop to validate Phase 1 of the VQ-HVEBT
implementation (single stage, no hierarchy, no decoder). It trains the model
to predict quantized video latents one step into the future.

Usage
-----
python example_code/vq_hvebt_training_loop.py \
    --weights_path clip/MobileCLIP2-S0/mobileclip2_s0.pt \
    --stage s1 \
    --num_codes 512 \
    --batch_size 4 \
    --T 4 \
    --lr 3e-4 \
    --encoder_lr_scale 0.1 \
    --steps 10000 \
    --log_every 100

For a quick sanity check on a single batch (overfitting test):
python example_code/vq_hvebt_training_loop.py \
    --weights_path clip/MobileCLIP2-S0/mobileclip2_s0.pt \
    --overfit_single_batch \
    --steps 500
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Optional

import torch
import torch.nn as nn

# ---- path fix so script can be run from project root ---------------------- #
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vid.vq_hvebt.config import (
    VQCodebookConfig,
    VQHVEBTConfig,
    VQStageConfig,
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


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #


def build_model(args: argparse.Namespace, device: torch.device) -> VQHVEBTModel:
    """Construct a single-stage VQ-HVEBT model."""
    STAGE_INFO = {
        "s1": (128, 32, 32),
        "s2": (256, 16, 16),
        "s3": (512,  8,  8),
    }
    C, H, W = STAGE_INFO[args.stage]

    stage_cfg = VQStageConfig(
        clip_stage_name=args.stage,
        clip_channels=C,
        H=H, W=W,
        transformer_dim=args.transformer_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mcmc_steps=args.mcmc_steps,
        mcmc_step_size=args.mcmc_step_size,
        codebook=VQCodebookConfig(
            num_codes=args.num_codes,
            code_dim=C,
            init_mode="data_first_batch",
            commitment_beta=args.commitment_beta,
        ),
        pred_loss_weight=1.0,
        cb_loss_weight=1.0,
        commit_loss_weight=args.commitment_beta,
    )

    cfg = VQHVEBTConfig(
        stages=[stage_cfg],
        train_encoder=not args.freeze_encoder,
        encoder_lr_scale=args.encoder_lr_scale,
        weights_path=args.weights_path,
        use_decoder=False,
    )
    model = VQHVEBTModel(cfg).to(device)
    return model


def build_optimizer(model: VQHVEBTModel, base_lr: float) -> torch.optim.Optimizer:
    """AdamW with separate LR for encoder vs rest."""
    groups = model.parameter_groups(base_lr)
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=1e-4)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[VQ-HVEBT] Building model on {device} ...")
    model = build_model(args, device)
    opt = build_optimizer(model, args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[VQ-HVEBT] Total params: {total_params:,}  Trainable: {trainable_params:,}")

    # ---- single-batch overfit test ---------------------------------------- #
    if args.overfit_single_batch:
        print("[VQ-HVEBT] Overfitting single batch ...")
        fixed_batch = make_synthetic_batch(args.batch_size, args.T, args.image_size, args.image_size, device)
        print(f"[VQ-HVEBT] Initializing codebooks from data ...")
        model.maybe_initialize_codebooks(fixed_batch)

        model.train()
        for step in range(1, args.steps + 1):
            opt.zero_grad()
            out = model.forward_loss(fixed_batch)
            out.total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

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
        return

    # ---- real training loop ----------------------------------------------- #
    print("[VQ-HVEBT] Starting training ...")
    model.train()
    codebook_init_done = False

    for step in range(1, args.steps + 1):
        # Replace with real dataloader in production.
        batch = make_synthetic_batch(args.batch_size, args.T, args.image_size, args.image_size, device)

        # Codebook initialization from first real batch.
        if not codebook_init_done:
            did_init = model.maybe_initialize_codebooks(batch)
            codebook_init_done = True
            if did_init:
                print("[VQ-HVEBT] Codebooks initialized from first data batch.")

        t0 = time.perf_counter()
        opt.zero_grad()
        out = model.forward_loss(batch)
        out.total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        dt = time.perf_counter() - t0

        if step % args.log_every == 0 or step == 1:
            m = out.metrics
            parts = [f"step {step:>6}  total={out.total_loss.item():.4f}  dt={dt*1000:.0f}ms"]
            for key, val in sorted(m.items()):
                parts.append(f"{key}={val:.4f}")
            print("  ".join(parts))

    print("[VQ-HVEBT] Training complete.")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VQ-HVEBT single-stage training loop")
    p.add_argument("--weights_path", default="clip/MobileCLIP2-S0/mobileclip2_s0.pt")
    p.add_argument("--stage", default="s1", choices=["s1", "s2", "s3"])
    p.add_argument("--num_codes", type=int, default=512)
    p.add_argument("--commitment_beta", type=float, default=0.25)
    p.add_argument("--transformer_dim", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--mcmc_steps", type=int, default=3)
    p.add_argument("--mcmc_step_size", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--T", type=int, default=4,
                   help="Number of context frames; model predicts T future frames")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder_lr_scale", type=float, default=0.1)
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze CLIP encoder (useful for debugging predictor in isolation)")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overfit_single_batch", action="store_true",
                   help="Run overfitting test on a single fixed batch")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
