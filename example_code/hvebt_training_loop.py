"""
HVEBT Phase 1 — minimal training loop on VIDShapeSyntheticDataset at 64x64.

Trains the single-stage HVEBT (16x16 encoder features) on rotating
2D triangles. Logs per-step loss, energies, reconstruction, gradient norm, and
(every N steps) running averages + simple stability checks.

Run:
    .\venv\Scripts\Activate.ps1
    python example_code\hvebt_training_loop.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset  # noqa: E402
from model.vid.hvebt import HVEBT, HVEBTStageConfig  # noqa: E402
from model.vid.hvebt.hvebt import HVEBTConfig  # noqa: E402


# ImageNet stats used by the dataset's transform.
_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def denormalize_imnet(x: torch.Tensor) -> torch.Tensor:
    """(B, T, 3, H, W) ImageNet-normalized -> [0, 1] RGB."""
    mean = _IMNET_MEAN.to(x.device, x.dtype)
    std = _IMNET_STD.to(x.device, x.dtype)
    return (x * std + mean).clamp(0.0, 1.0)


def make_hparams(args) -> SimpleNamespace:
    return SimpleNamespace(
        context_length=args.context_length,
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
        shape_cache_dir="data/vid/shape_cache",
    )


def make_model(args, device: torch.device) -> HVEBT:
    stage_cfg = HVEBTStageConfig(
        stage_name="16x16",
        channels=64,
        H=16, W=16,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_mult=4.0,
        dropout=0.0,
    )
    cfg = HVEBTConfig(
        stage=stage_cfg,
        mcmc_num_steps=args.mcmc_steps,
        mcmc_step_size=args.mcmc_step_size,
        mcmc_step_size_learnable=True,
        langevin_noise=0.0,
        denoising_init=args.denoising_init,
        truncate_mcmc=False,
        clamp_grad_max=args.clamp_grad_max,
    )
    model = HVEBT(cfg).to(device)
    return model


def format_scalar(v):
    if isinstance(v, torch.Tensor):
        v = v.item()
    return f"{v:+.4e}"


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[hvebt] device={device}")
    print(f"[hvebt] building dataset (size={args.dataset_size}, T={args.context_length}, HxW={args.image_size})")
    hparams = make_hparams(args)
    dataset = VIDShapeSyntheticDataset(hparams, size=args.dataset_size)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,          # Windows + pyvista cache already on disk, loading is cheap
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    print(f"[hvebt] building model (embed={args.embed_dim}, layers={args.n_layers}, heads={args.n_heads}, mcmc_K={args.mcmc_steps})")
    model = make_model(args, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[hvebt] trainable params: {n_trainable/1e6:.2f}M")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    log_path = os.path.join(args.log_dir, "train_log.csv")
    os.makedirs(args.log_dir, exist_ok=True)
    log_f = open(log_path, "w")
    log_f.write("step,loss,init_recon,final_recon,init_energy,final_energy,energy_gap,grad_norm,alpha,baseline_copy_last,secs\n")

    # Also keep a rolling-window baseline for sanity checks.
    recent_losses: List[float] = []
    window = 20
    step = 0
    t_start = time.time()
    done = False

    print(f"[hvebt] starting training: max_steps={args.max_steps}")
    while not done:
        for batch in loader:
            t0 = time.time()
            batch = batch.to(device, non_blocking=True)      # (B, T, 3, H, W), ImageNet-normalized
            video01 = denormalize_imnet(batch)               # (B, T, 3, H, W) in [0,1]

            out: Dict[str, torch.Tensor] = model.forward_loss(video01, learning=True)
            loss = out["loss"]

            # Copy-last-frame baseline in feature space: predict frame t+1 features
            # as frame t features. Model should eventually beat this.
            with torch.no_grad():
                all_feats = model.encode(video01)            # (B, T, C, H, W)
                baseline_copy = F.smooth_l1_loss(all_feats[:, :-1], all_feats[:, 1:]).item()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip).item()

            # Catch NaN/Inf early and abort with diagnostic info.
            if not math.isfinite(loss.item()) or not math.isfinite(grad_norm):
                print(f"[hvebt][ABORT] non-finite at step {step}: loss={loss.item()}, grad_norm={grad_norm}")
                return

            opt.step()

            dt = time.time() - t0
            recent_losses.append(loss.item())
            if len(recent_losses) > window:
                recent_losses.pop(0)

            log_f.write(
                f"{step},{loss.item():.6f},{out['init_recon'].item():.6f},{out['final_recon'].item():.6f},"
                f"{out['init_energy'].item():.6f},{out['final_energy'].item():.6f},"
                f"{out['energy_gap'].item():.6f},{grad_norm:.6f},{out['alpha'].item():.6f},"
                f"{baseline_copy:.6f},{dt:.3f}\n"
            )
            log_f.flush()

            if step % args.log_every == 0:
                running = sum(recent_losses) / len(recent_losses)
                ir = out['init_recon'].item()
                fr = out['final_recon'].item()
                improve = (ir - fr) / max(ir, 1e-8)
                vs_base = (baseline_copy - fr) / max(baseline_copy, 1e-8)
                print(
                    f"[step {step:5d}] loss={loss.item():.3f} run{window}={running:.3f} "
                    f"recon(i/f)={ir:.3f}/{fr:.3f} d%={improve*100:+.2f} "
                    f"base={baseline_copy:.3f} vs_base={vs_base*100:+.1f}% "
                    f"E(i/f)={out['init_energy'].item():+.2e}/{out['final_energy'].item():+.2e} "
                    f"Egap={out['energy_gap'].item():+.2e} gnorm={grad_norm:.2e} "
                    f"alpha={out['alpha'].item():.2f} dt={dt:.2f}s"
                )

            step += 1
            if step >= args.max_steps:
                done = True
                break

    log_f.close()
    total = time.time() - t_start
    print(f"[hvebt] finished: {step} steps in {total:.1f}s ({step/total:.2f} steps/s)")
    print(f"[hvebt] log saved to {log_path}")

    if len(recent_losses) >= window:
        early_window = args.max_steps // 10
        import csv
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
        early = [float(r["loss"]) for r in rows[:max(early_window, 1)]]
        late = [float(r["loss"]) for r in rows[-window:]]
        print(f"[hvebt] sanity: mean loss first {len(early)} = {sum(early)/len(early):.4f}, "
              f"last {len(late)} = {sum(late)/len(late):.4f}")


def parse_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--context_length", type=int, default=8)
    ap.add_argument("--dataset_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    # model
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    # mcmc
    ap.add_argument("--mcmc_steps", type=int, default=2)
    ap.add_argument("--mcmc_step_size", type=float, default=1000.0)
    ap.add_argument("--denoising_init", type=str, default="zeros",
                    choices=["zeros", "random_noise", "real_current"])
    ap.add_argument("--clamp_grad_max", type=float, default=0.0)
    # optimization
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_dir", type=str, default="logs/hvebt_phase1")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
