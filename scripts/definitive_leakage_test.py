"""
Definitive leakage vs memorization test for HVQVAE.
Run on GPU: python scripts/definitive_leakage_test.py [--steps 5000]

Tests:
  1. Per-position MSE on TRAIN data vs UNSEEN data
     → If train pos0 << test pos0: MEMORIZATION (not leakage)
     → If train pos0 ≈ test pos0 and both low: model learned something real
  2. Causal intervention: corrupt each input frame, check which outputs change
     → If future frame corruption affects earlier positions: LEAKAGE
  3. Saves visualization grids for both train and test data

Expected result if no leakage:
  - Position 0 predictions on UNSEEN data will be poor/blurry
    (because rotation angle/direction is random and unknowable from frame 0)
  - Position 0 predictions on TRAINING data may look OK (memorization of 1000 examples)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from types import SimpleNamespace

from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset
from model.vid.hvqvae.config import HVQVAEConfig, HVQVAEStageConfig
from model.vid.hvqvae.model import HVQVAEModel

try:
    from torchvision.utils import save_image
    HAS_SAVE = True
except ImportError:
    HAS_SAVE = False


def build_dataset(size: int, T: int, seed: int) -> VIDShapeSyntheticDataset:
    """Build dataset with a specific cache seed (different seed → different data)."""
    hparams = SimpleNamespace(
        context_length=T + 1,
        image_dims=[64, 64],
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
        shape_cache_dir=f"data/vid/shape_cache_exp_{seed}",
    )
    return VIDShapeSyntheticDataset(hparams, size=size)


def build_model(device: torch.device) -> HVQVAEModel:
    """Default 3-stage model matching training loop defaults."""
    stages = [
        HVQVAEStageConfig(
            stage_name="s3", channels=256, H=2, W=2, num_codes=512,
            transformer_dim=256, n_heads=4, n_layers=2, temporal_window=None),
        HVQVAEStageConfig(
            stage_name="s2", channels=128, H=4, W=4, num_codes=256,
            transformer_dim=128, n_heads=2, n_layers=2, temporal_window=2),
        HVQVAEStageConfig(
            stage_name="s1", channels=64, H=8, W=8, num_codes=64,
            transformer_dim=64, n_heads=2, n_layers=2, temporal_window=1),
    ]
    cfg = HVQVAEConfig(stages=stages)
    return HVQVAEModel(cfg).to(device)


def compute_per_position_mse(
    model: HVQVAEModel, dataset, device: torch.device,
    n_batches: int = 10, batch_size: int = 8,
) -> dict:
    """Compute MSE per prediction position over multiple batches.
    Returns dict with per-position MSE and overall MSE.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=True)
    model.eval()

    pos_mse_sum = None
    n_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch = batch.to(device, non_blocking=True)
            out = model(batch)

            pred = out.pred_rgb  # (B, T, 3, H, W)
            gt = batch[:, 1:]    # (B, T, 3, H, W)

            if pred.shape[-2:] != gt.shape[-2:]:
                B_t, T_t = gt.shape[:2]
                gt = F.interpolate(
                    gt.reshape(B_t * T_t, 3, gt.shape[-2], gt.shape[-1]),
                    size=pred.shape[-2:], mode="bilinear", align_corners=False
                ).reshape(B_t, T_t, 3, *pred.shape[-2:])

            # Per-position MSE: average over B, C, H, W
            per_pos = ((pred - gt) ** 2).mean(dim=(0, 2, 3, 4))  # (T,)

            if pos_mse_sum is None:
                pos_mse_sum = per_pos.clone()
            else:
                pos_mse_sum += per_pos
            n_samples += 1

    pos_mse = pos_mse_sum / n_samples
    return {f"pos_{i}": pos_mse[i].item() for i in range(len(pos_mse))}


def causal_intervention_test(model: HVQVAEModel, video: torch.Tensor) -> None:
    """Corrupt each input frame, check which prediction positions change."""
    model.eval()
    with torch.no_grad():
        base_out = model(video)
        base_pred = base_out.pred_rgb.clone()

    B, T1 = video.shape[:2]
    T = T1 - 1

    print("\n=== CAUSAL INTERVENTION TEST ===")
    header = f"{'Corrupt Frame':<15}"
    for p in range(T):
        header += f" | {'Pos '+str(p)+' (→f'+str(p+1)+')':<16}"
    print(header)
    print("-" * (16 + 19 * T))

    any_leak = False
    for corrupt_t in range(T1):
        v2 = video.clone()
        v2[:, corrupt_t] = torch.randn_like(v2[:, corrupt_t])
        with torch.no_grad():
            out2 = model(v2)

        row = f"  frame {corrupt_t:<8}"
        for pos in range(T):
            diff = (base_pred[:, pos] - out2.pred_rgb[:, pos]).abs().max().item()
            # Position pos predicts frame pos+1. It should only change when
            # corrupting frames 0..pos (those are the CONTEXT frames it sees).
            is_leak = (corrupt_t > pos) and (diff > 1e-5)
            if is_leak:
                any_leak = True
                marker = "LEAK!"
            elif diff > 1e-5:
                marker = "changed"
            else:
                marker = "unchanged"
            row += f" | {diff:.2e} ({marker})"
        print(row)

    print(f"\nVerdict: {'LEAKAGE DETECTED!' if any_leak else 'No leakage — causal structure is correct.'}")
    return any_leak


def save_per_position_viz(
    model: HVQVAEModel, batch: torch.Tensor,
    label: str, save_dir: Path,
) -> None:
    """Save per-position prediction vs GT visualization."""
    if not HAS_SAVE:
        print(f"  [skip visualization — torchvision.utils.save_image not available]")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        out = model(batch)

    pred = out.pred_rgb[0].clamp(0, 1)   # (T, 3, H, W) — first sample
    gt = batch[0, 1:].clamp(0, 1)        # (T, 3, H, W) — GT future frames
    ctx = batch[0, :1].clamp(0, 1)       # (1, 3, H, W) — context frame 0

    if pred.shape[-2:] != gt.shape[-2:]:
        gt = F.interpolate(gt, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    if ctx.shape[-2:] != pred.shape[-2:]:
        ctx = F.interpolate(ctx, size=pred.shape[-2:], mode="bilinear", align_corners=False)

    # Row 1: context frame 0 + predictions at each position
    # Row 2: context frame 0 + GT at each position
    row1 = torch.cat([ctx, pred], dim=0)     # (1+T, 3, H, W)
    row2 = torch.cat([ctx, gt], dim=0)       # (1+T, 3, H, W)
    grid = torch.cat([row1, row2], dim=0)    # (2*(1+T), 3, H, W)

    fname = save_dir / f"{label}.png"
    save_image(grid, str(fname), nrow=1 + pred.shape[0])
    print(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default="logs/leakage_test")
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    # ---- Build datasets ----
    print("Building TRAIN dataset (seed=500, 1000 examples)...")
    train_ds = build_dataset(args.dataset_size, args.T, seed=500)

    print("Building TEST dataset (seed=999, 1000 different examples)...")
    test_ds = build_dataset(args.dataset_size, args.T, seed=999)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True)

    # ---- Build model ----
    print("Building model...")
    model = build_model(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_p:,}")

    # Data variance
    samples = torch.stack([train_ds[i] for i in range(min(100, len(train_ds)))])
    model.set_data_variance(samples.var().item())
    print(f"  Data variance: {samples.var().item():.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    # ---- Train ----
    print(f"\nTraining for {args.steps} steps on {args.dataset_size} examples...")
    model.train()
    step = 0
    epoch = 0
    t0 = time.time()

    while step < args.steps:
        epoch += 1
        for batch in train_loader:
            step += 1
            if step > args.steps:
                break

            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(batch)
            out.total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                m = out.metrics
                perps = " ".join(f"{s}:{m.get(f'{s}/perplexity', 0):.1f}" for s in ['s3', 's2', 's1'])
                print(f"  step {step:>6} ({elapsed:.0f}s)  loss={out.total_loss.item():.4f}  "
                      f"pred={m['pred_loss']:.4f}  perplexity: {perps}")

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s ({args.steps / elapsed:.1f} steps/s)\n")

    # ---- TEST 1: Per-position MSE on train vs test ----
    print("=" * 60)
    print("TEST 1: Per-position MSE — TRAIN data vs UNSEEN data")
    print("=" * 60)

    train_mse = compute_per_position_mse(model, train_ds, device, n_batches=20)
    test_mse = compute_per_position_mse(model, test_ds, device, n_batches=20)

    print(f"\n{'Position':<12} | {'Train MSE':<14} | {'Test MSE':<14} | {'Ratio test/train':<18} | Interpretation")
    print("-" * 80)
    for key in sorted(train_mse.keys()):
        tr = train_mse[key]
        te = test_mse[key]
        ratio = te / tr if tr > 1e-8 else float('inf')
        if ratio > 3.0:
            interp = "MEMORIZATION (test >> train)"
        elif ratio > 1.5:
            interp = "Partial memorization"
        else:
            interp = "Generalizes well"
        print(f"  {key:<10} | {tr:<14.6f} | {te:<14.6f} | {ratio:<18.2f} | {interp}")

    # Check position 0 specifically
    tr0 = train_mse.get("pos_0", 0)
    te0 = test_mse.get("pos_0", 0)
    print(f"\n  Position 0 predicts frame 1 from frame 0 ONLY (random rotation).")
    if te0 > tr0 * 2:
        print(f"  → Test MSE {te0:.4f} >> Train MSE {tr0:.4f}")
        print(f"  → CONCLUSION: Model MEMORIZED training data, NOT leaking.")
        print(f"  → On unseen data, position 0 can't predict the random rotation.")
    else:
        print(f"  → Test MSE {te0:.4f} ≈ Train MSE {tr0:.4f}")
        print(f"  → CONCLUSION: Model generalizes at pos 0 — investigate further.")

    # ---- TEST 2: Causal intervention ----
    print("\n" + "=" * 60)
    print("TEST 2: Causal intervention (forward-pass leakage check)")
    print("=" * 60)
    test_batch = torch.stack([test_ds[i] for i in range(4)]).to(device)
    has_leak = causal_intervention_test(model, test_batch)

    # ---- TEST 3: Visualizations ----
    print("\n" + "=" * 60)
    print("TEST 3: Visualizations (train vs test)")
    print("=" * 60)

    viz_train = torch.stack([train_ds[i] for i in range(4)]).to(device)
    viz_test = torch.stack([test_ds[i] for i in range(4)]).to(device)

    for sample_idx in range(min(4, args.batch_size)):
        single_train = viz_train[sample_idx:sample_idx+1]
        single_test = viz_test[sample_idx:sample_idx+1]
        save_per_position_viz(model, single_train, f"train_sample_{sample_idx}", save_dir / "viz")
        save_per_position_viz(model, single_test, f"test_sample_{sample_idx}", save_dir / "viz")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Causal intervention: {'LEAKAGE FOUND' if has_leak else 'No leakage'}")
    print(f"  Position 0 train MSE: {tr0:.6f}")
    print(f"  Position 0 test MSE:  {te0:.6f}")
    print(f"  Ratio (test/train):   {te0/tr0 if tr0 > 1e-8 else float('inf'):.2f}x")
    if not has_leak and te0 > tr0 * 2:
        print(f"\n  FINAL VERDICT: No leakage. Model is memorizing training data.")
        print(f"  Position 0 cannot generalize because rotation is unpredictable.")
    elif has_leak:
        print(f"\n  FINAL VERDICT: LEAKAGE DETECTED — see causal intervention results above.")
    else:
        print(f"\n  FINAL VERDICT: No clear leakage, but pos 0 generalizes unexpectedly.")
        print(f"  May need further investigation.")

    print(f"\n  All results saved to: {save_dir}/")
    print(f"  Visualizations: {save_dir}/viz/")


if __name__ == "__main__":
    main()
