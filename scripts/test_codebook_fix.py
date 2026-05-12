"""
Test EMA + dead-code reset effect on codebook utilization.

Runs two short training sessions:
  A) OLD: gradient-based VQ, no dead code reset (use_ema=False, dead_code_threshold=0)
  B) NEW: EMA + dead code reset (use_ema=True, dead_code_threshold=100)

Compares per-stage perplexity over training. The new version should show
dramatically higher perplexity (more codes used) at s2 and s3.

Usage:
  python scripts/test_codebook_fix.py [--steps 5000]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset
from model.vid.hvqvae.config import HVQVAEConfig, HVQVAEStageConfig
from model.vid.hvqvae.model import HVQVAEModel


def build_dataset(size: int, T: int) -> VIDShapeSyntheticDataset:
    hparams = SimpleNamespace(
        context_length=T + 1,
        image_dims=[64, 64],
        shape_scene_type="DIM_2",
        shape_min_cubes=2, shape_max_cubes=6,
        shape_angle_min=15, shape_angle_max=45,
        shape_temporal_patterns=[], shape_pattern_combining=False,
        shape_accel_min=3, shape_accel_max=6,
        shape_oscillation_period_min=1, shape_oscillation_period_max=4,
        shape_interruption_period_min=1, shape_interruption_period_max=4,
        shape_cache_dir="data/vid/shape_cache_exp_500",
    )
    return VIDShapeSyntheticDataset(hparams, size=size)


def build_model(device, use_ema: bool, dead_code_threshold: int, entropy_weight: float) -> HVQVAEModel:
    stages = [
        HVQVAEStageConfig(stage_name="s3", channels=256, H=2, W=2, num_codes=512,
                          transformer_dim=256, n_heads=4, n_layers=2, temporal_window=None),
        HVQVAEStageConfig(stage_name="s2", channels=128, H=4, W=4, num_codes=256,
                          transformer_dim=128, n_heads=2, n_layers=2, temporal_window=2),
        HVQVAEStageConfig(stage_name="s1", channels=64, H=8, W=8, num_codes=64,
                          transformer_dim=64, n_heads=2, n_layers=2, temporal_window=1),
    ]
    cfg = HVQVAEConfig(
        stages=stages,
        use_ema=use_ema,
        ema_decay=0.99,
        dead_code_threshold=dead_code_threshold,
        entropy_weight=entropy_weight,
    )
    return HVQVAEModel(cfg).to(device)


def train_and_log(
    label: str, model: HVQVAEModel, dataset, device,
    steps: int, batch_size: int, log_every: int,
) -> list:
    """Train and return list of (step, {metrics}) tuples."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True)

    samples = torch.stack([dataset[i] for i in range(min(100, len(dataset)))])
    model.set_data_variance(samples.var().item())

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, amsgrad=True)
    model.train()

    history = []
    step = 0
    epoch = 0
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    while step < steps:
        epoch += 1
        for batch in loader:
            step += 1
            if step > steps:
                break

            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(batch)
            out.total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % log_every == 0 or step == 1:
                m = out.metrics
                elapsed = time.time() - t0
                perps = {s: m.get(f"{s}/perplexity", 0) for s in ["s3", "s2", "s1"]}
                history.append((step, {**m, **{f"{s}_perp": v for s, v in perps.items()}}))

                perp_str = "  ".join(f"{s}:{v:.1f}/{k}" for s, v, k in
                                     [("s3", perps["s3"], 512), ("s2", perps["s2"], 256), ("s1", perps["s1"], 64)])
                print(f"  [{label}] step {step:>5} ({elapsed:.0f}s)  "
                      f"loss={out.total_loss.item():.4f}  pred={m['pred_loss']:.4f}  "
                      f"perplexity: {perp_str}")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Building dataset...")
    dataset = build_dataset(args.dataset_size, T=4)

    # ---- Run A: OLD (gradient VQ, no reset) ----
    torch.manual_seed(42)
    model_old = build_model(device, use_ema=False, dead_code_threshold=0, entropy_weight=0.0)
    hist_old = train_and_log("OLD (gradient VQ)", model_old, dataset, device,
                             args.steps, args.batch_size, args.log_every)

    # ---- Run B: NEW (EMA + dead code reset) ----
    torch.manual_seed(42)
    model_new = build_model(device, use_ema=True, dead_code_threshold=100, entropy_weight=0.0)
    hist_new = train_and_log("NEW (EMA + reset)", model_new, dataset, device,
                             args.steps, args.batch_size, args.log_every)

    # ---- Comparison ----
    print("\n" + "=" * 70)
    print("COMPARISON: Final perplexity (higher = more codes used)")
    print("=" * 70)

    old_final = hist_old[-1][1] if hist_old else {}
    new_final = hist_new[-1][1] if hist_new else {}

    print(f"\n{'Stage':<8} | {'OLD (gradient)':<20} | {'NEW (EMA+reset)':<20} | {'K (codebook size)':<18}")
    print("-" * 70)
    for stage, K in [("s3", 512), ("s2", 256), ("s1", 64)]:
        old_p = old_final.get(f"{stage}_perp", 0)
        new_p = new_final.get(f"{stage}_perp", 0)
        improvement = f"{new_p/old_p:.1f}x" if old_p > 0.01 else "N/A"
        print(f"  {stage:<6} | {old_p:<20.1f} | {new_p:<20.1f} | {K:<18} | {improvement} improvement")

    print(f"\n{'Metric':<20} | {'OLD':<14} | {'NEW':<14}")
    print("-" * 50)
    for metric in ["pred_loss", "embedding_loss"]:
        old_v = old_final.get(metric, 0)
        new_v = new_final.get(metric, 0)
        print(f"  {metric:<18} | {old_v:<14.4f} | {new_v:<14.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
