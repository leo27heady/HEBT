"""
Unified End-to-End VQ-VAE Training Loop.

Single optimizer, single backward pass through entire model.
Trains encoder + predictors + decoder head jointly.

Usage:
    python example_code/unified_vqvae_training_loop.py --log_dir logs/unified_test --steps 2000
    python example_code/unified_vqvae_training_loop.py --data_source shapes --steps 5000 --T 4
"""

import sys
import os
import argparse
import time
import csv
import json

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torchvision.utils import save_image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vid.fresh_hvqvae.unified_experiment import UnifiedModel, UnifiedConfig


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_synthetic_batch(B: int, T: int, H: int = 64, W: int = 64, device: str = 'cpu'):
    """Simple translating colored square + static circle."""
    frames = []
    for t in range(T + 1):
        frame = torch.zeros(B, 3, H, W, device=device)
        cx = int(H * 0.3 + H * 0.4 * (t / T))
        cy = int(W * 0.3 + W * 0.4 * (t / T))
        size = 8
        r_start, r_end = max(0, cx - size), min(H, cx + size)
        c_start, c_end = max(0, cy - size), min(W, cy + size)
        frame[:, 0, r_start:r_end, c_start:c_end] = 1.0
        frame[:, 1, r_start:r_end, c_start:c_end] = 0.5
        for r in range(H):
            for c in range(W):
                if (r - H // 2) ** 2 + (c - W // 4) ** 2 < 100:
                    frame[:, 2, r, c] = 0.7
        frames.append(frame)
    return torch.stack(frames, dim=1) * 2 - 1


def generate_synthetic_shapes_batch(B: int, T: int, H: int = 64, W: int = 64, device: str = 'cpu'):
    """Rotating 2D shapes via VIDShapeSyntheticDataset."""
    from types import SimpleNamespace
    from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset

    hparams = SimpleNamespace(
        context_length=T + 1, image_dims=[H, W],
        shape_scene_type="DIM_2", shape_min_cubes=2, shape_max_cubes=6,
        shape_angle_min=15, shape_angle_max=45,
        shape_temporal_patterns=[], shape_pattern_combining=False,
        shape_accel_min=3, shape_accel_max=6,
        shape_oscillation_period_min=1, shape_oscillation_period_max=4,
        shape_interruption_period_min=1, shape_interruption_period_max=4,
        shape_cache_dir="data/vid/shape_cache",
        shape_no_imagenet_norm=True,
    )
    ds = VIDShapeSyntheticDataset(hparams, size=B, cache=False)
    frames = torch.stack([ds[i] for i in range(B)], dim=0)
    return (frames.to(device) * 2 - 1)


def make_batch(data_source: str, B: int, T: int, H: int = 64, W: int = 64, device: str = 'cpu'):
    if data_source == 'simple':
        return generate_synthetic_batch(B, T, H, W, device)
    elif data_source == 'shapes':
        return generate_synthetic_shapes_batch(B, T, H, W, device)
    else:
        raise ValueError(f"Unknown data_source: {data_source}")


# ---------------------------------------------------------------------------
# Dataset caching
# ---------------------------------------------------------------------------

def cache_dataset(data_source: str, num_batches: int, B: int, T: int, cache_dir: str):
    """Pre-generate batches and save to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, 'meta.json')

    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if (meta.get('num_batches') == num_batches and
                meta.get('batch_size') == B and
                meta.get('T') == T and
                meta.get('data_source') == data_source):
            print(f"Using existing cache at {cache_dir} ({num_batches} batches)")
            return
        print(f"Cache config mismatch, regenerating...")

    print(f"Pre-generating {num_batches} batches to {cache_dir}...")
    t0 = time.time()
    for i in range(num_batches):
        batch = make_batch(data_source, B, T, device='cpu')
        torch.save(batch, os.path.join(cache_dir, f'batch_{i:06d}.pt'))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{num_batches}]")
    with open(meta_path, 'w') as f:
        json.dump({'num_batches': num_batches, 'batch_size': B, 'T': T,
                   'data_source': data_source}, f)
    print(f"Done in {time.time()-t0:.1f}s")


def load_cached_batch(cache_dir: str, step: int, num_batches: int, device: str):
    idx = (step - 1) % num_batches
    return torch.load(os.path.join(cache_dir, f'batch_{idx:06d}.pt'),
                      map_location=device, weights_only=True)


# ---------------------------------------------------------------------------
# Codebook usage tracking
# ---------------------------------------------------------------------------

def compute_codebook_usage(model, video):
    """Compute unique codes used per stage."""
    with torch.no_grad():
        enc = model.encode(video)
    usage = {}
    for name, key in [('bot', 'idx_bot'), ('mid', 'idx_mid'), ('top', 'idx_top')]:
        indices = enc[key].reshape(-1)
        unique = indices.unique().numel()
        total = getattr(model.cfg, f'K_{name}')
        usage[f'usage_{name}'] = unique
        usage[f'usage_{name}_pct'] = 100.0 * unique / total
    return usage


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = args.device
    os.makedirs(args.log_dir, exist_ok=True)

    # Config
    cfg = UnifiedConfig(
        max_T=args.T + 1,
        lr=args.lr,
        lambda_ce=args.lambda_ce,
        lambda_vq=args.lambda_vq,
    )

    # Model
    model = UnifiedModel(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"  Encoder: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  Predictor Top: {sum(p.numel() for p in model.predictor_top.parameters()):,}")
    print(f"  Predictor Mid: {sum(p.numel() for p in model.predictor_mid.parameters()):,}")
    print(f"  Predictor Bot: {sum(p.numel() for p in model.predictor_bot.parameters()):,}")
    print(f"  Decoder Head: {sum(p.numel() for p in model.decoder_head.parameters()):,}")

    # Optimizer (single for everything)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    # Dataset caching
    cache_dir = os.path.join(args.log_dir, 'batch_cache')
    num_batches = max(args.steps, 200)
    cache_dataset(args.data_source, num_batches, args.batch_size, args.T, cache_dir)

    # CSV logger
    csv_path = os.path.join(args.log_dir, 'metrics.csv')
    csv_fields = ['step', 'loss', 'mse', 'ce_top', 'ce_mid', 'ce_bot', 'ce_total',
                  'vq_bot', 'vq_mid', 'vq_top', 'vq_total',
                  'usage_bot', 'usage_mid', 'usage_top', 'time_per_step']
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Save config
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump(vars(args) | vars(cfg), f, indent=2, default=str)

    print(f"\nStarting training for {args.steps} steps...")
    print(f"  data_source={args.data_source}, B={args.batch_size}, T={args.T}")
    print(f"  lr={cfg.lr}, λ_ce={cfg.lambda_ce}, λ_vq={cfg.lambda_vq}")
    print(f"  log_dir={args.log_dir}\n")

    model.train()
    t0 = time.time()

    for step in range(1, args.steps + 1):
        step_start = time.time()

        # Load batch
        video = load_cached_batch(cache_dir, step, num_batches, device)  # (B, T+1, 3, 64, 64)

        # Forward
        out = model(video)

        # Backward
        optimizer.zero_grad()
        out['loss'].backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        step_time = time.time() - step_start

        # Logging
        if step % args.log_every == 0 or step == 1:
            usage = compute_codebook_usage(model, video)
            row = {
                'step': step,
                'loss': f"{out['loss'].item():.5f}",
                'mse': f"{out['mse'].item():.5f}",
                'ce_top': f"{out['ce_top'].item():.4f}",
                'ce_mid': f"{out['ce_mid'].item():.4f}",
                'ce_bot': f"{out['ce_bot'].item():.4f}",
                'ce_total': f"{out['ce_total'].item():.4f}",
                'vq_bot': f"{out['vq_bot'].item():.5f}",
                'vq_mid': f"{out['vq_mid'].item():.5f}",
                'vq_top': f"{out['vq_top'].item():.5f}",
                'vq_total': f"{out['vq_total'].item():.5f}",
                'usage_bot': usage['usage_bot'],
                'usage_mid': usage['usage_mid'],
                'usage_top': usage['usage_top'],
                'time_per_step': f"{step_time:.3f}",
            }
            csv_writer.writerow(row)
            csv_file.flush()

            print(f"[{step:5d}/{args.steps}] "
                  f"loss={out['loss'].item():.4f} "
                  f"mse={out['mse'].item():.4f} "
                  f"ce={out['ce_total'].item():.3f}(t{out['ce_top'].item():.2f}/m{out['ce_mid'].item():.2f}/b{out['ce_bot'].item():.2f}) "
                  f"vq={out['vq_total'].item():.4f} "
                  f"usage={usage['usage_bot']}/{usage['usage_mid']}/{usage['usage_top']} "
                  f"({step_time:.2f}s)")

        # Visualization
        if step % args.vis_every == 0 or step == 1:
            model.eval()
            vis = model.build_visualization(video)
            vis_path = os.path.join(args.log_dir, f'vis_step_{step:06d}.png')
            save_image(vis, vis_path)
            model.train()

        # Checkpoint
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.log_dir, f'checkpoint_step_{step:06d}.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': out['loss'].item(),
            }, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    # Final save
    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s ({total_time/args.steps:.2f}s/step avg)")

    final_path = os.path.join(args.log_dir, 'final_model.pt')
    torch.save({
        'step': args.steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(cfg),
    }, final_path)
    print(f"Saved final model → {final_path}")

    csv_file.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified VQ-VAE Training")
    parser.add_argument('--log_dir', type=str, default='logs/unified_test')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--data_source', type=str, default='shapes', choices=['simple', 'shapes'])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lambda_ce', type=float, default=1.0)
    parser.add_argument('--lambda_vq', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--vis_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=500)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
