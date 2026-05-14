"""
Fresh HVQVAE Training Loop.

Supports 3 training modes:
  encoder_only   — train encoder + per-stage decoders (no predictors)
  predictor_only — load pretrained encoder+decoders (frozen), train predictors only
  full           — train everything (encoder+decoders first, then predictors)

Features:
  - Checkpoint saving every N steps with top-K best kept
  - Dataset caching to disk for reproducibility
  - Per-stage isolated backward passes
"""

import sys
import os
import argparse
import time
import csv
import glob
import json

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vid.fresh_hvqvae import FreshHVQVAE, FreshHVQVAEConfig


def generate_synthetic_batch(B: int, T: int, H: int = 64, W: int = 64, device: str = 'cpu'):
    """Simple translating colored square + static circle on black background."""
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
    return torch.stack(frames, dim=1)


def generate_synthetic_shapes_batch(B: int, T: int, H: int = 64, W: int = 64, device: str = 'cpu'):
    """Rotating 2D shapes via VIDShapeSyntheticDataset. Returns [0,1] range."""
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
    return frames.to(device)


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

def cache_dataset(data_source: str, num_batches: int, B: int, T: int, cache_dir: str, device: str):
    """Pre-generate num_batches batches and save to disk as .pt files."""
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, 'meta.json')

    # Check if compatible cache already exists
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if (meta.get('num_batches') == num_batches and
                meta.get('batch_size') == B and
                meta.get('T') == T and
                meta.get('data_source') == data_source):
            print(f"Using existing batch cache at {cache_dir} ({num_batches} batches)")
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
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model, step, losses, log_dir, metric_key='mse_bot'):
    """Save checkpoint and return the path."""
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    metric_val = losses.get(metric_key, 0.0)
    path = os.path.join(ckpt_dir, f'step_{step:06d}_metric_{metric_val:.6f}.pt')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'losses': losses,
    }, path)
    print(f"  Saved checkpoint -> {path}")
    return path


def prune_checkpoints(log_dir, keep_top_k):
    """Keep only the top-K best checkpoints (lowest metric value in filename)."""
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'step_*.pt')))
    if len(ckpts) <= keep_top_k:
        return
    # Parse metric from filename: step_NNNNNN_metric_X.XXXXXX.pt
    def get_metric(p):
        name = os.path.basename(p)
        try:
            return float(name.split('_metric_')[1].replace('.pt', ''))
        except (IndexError, ValueError):
            return float('inf')
    ckpts.sort(key=get_metric)
    for p in ckpts[keep_top_k:]:
        os.remove(p)


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def train_step_full(model, batch, opt_enc_dec, opt_pred_top, opt_pred_mid, opt_pred_bot, cfg):
    """Full training step: encoder+decoders + all 3 predictors."""
    losses = model(batch)
    recon_total = (
        cfg.weight_mse_bot * losses['mse_bot'] +
        cfg.weight_mse_mid * losses['mse_mid'] +
        cfg.weight_mse_top * losses['mse_top'] +
        losses['vq_loss']
    )
    opt_enc_dec.zero_grad()
    recon_total.backward()
    torch.nn.utils.clip_grad_norm_(model.get_encoder_decoder_params(), max_norm=cfg.max_grad_norm)
    opt_enc_dec.step()

    opt_pred_top.zero_grad()
    losses['ce_top'].backward()
    opt_pred_top.step()

    opt_pred_mid.zero_grad()
    losses['ce_mid'].backward()
    opt_pred_mid.step()

    opt_pred_bot.zero_grad()
    losses['ce_bot'].backward()
    opt_pred_bot.step()

    return {k: v.item() for k, v in losses.items()}


def train_step_encoder_only(model, batch, opt_enc_dec, cfg):
    """Encoder-only: reconstruction loss, no predictors."""
    B, Tp1 = batch.shape[:2]
    all_frames = batch.reshape(B * Tp1, 3, batch.shape[3], batch.shape[4])
    enc = model.encode(batch)
    recon_bot, recon_mid, recon_top = model.reconstruct(enc, B, Tp1)

    mse_bot = F.mse_loss(recon_bot, all_frames)
    mse_mid = F.mse_loss(recon_mid, all_frames)
    mse_top = F.mse_loss(recon_top, all_frames)
    vq_loss = enc['loss_bot'] + enc['loss_mid'] + enc['loss_top']

    recon_total = (
        cfg.weight_mse_bot * mse_bot +
        cfg.weight_mse_mid * mse_mid +
        cfg.weight_mse_top * mse_top +
        vq_loss
    )
    opt_enc_dec.zero_grad()
    recon_total.backward()
    torch.nn.utils.clip_grad_norm_(model.get_encoder_decoder_params(), max_norm=cfg.max_grad_norm)
    opt_enc_dec.step()

    return {'mse_bot': mse_bot.item(), 'mse_mid': mse_mid.item(),
            'mse_top': mse_top.item(), 'vq_loss': vq_loss.item()}


def train_step_predictor_only(model, batch, opt_pred_top, opt_pred_mid, opt_pred_bot, cfg):
    """Predictor-only: encoder frozen, CE losses only."""
    B, Tp1 = batch.shape[:2]
    T = Tp1 - 1

    with torch.no_grad():
        enc = model.encode(batch)

    pred = model.predict(enc, T)
    tgt_top = enc['idx_top'][:, 1:].detach().reshape(B * T * 1)
    tgt_mid = enc['idx_mid'][:, 1:].detach().reshape(B * T * 16)
    tgt_bot = enc['idx_bot'][:, 1:].detach().reshape(B * T * 256)

    ce_top = F.cross_entropy(pred['logits_top'].reshape(-1, cfg.K_top), tgt_top)
    ce_mid = F.cross_entropy(pred['logits_mid'].reshape(-1, cfg.K_mid), tgt_mid)
    ce_bot = F.cross_entropy(pred['logits_bot'].reshape(-1, cfg.K_bot), tgt_bot)

    opt_pred_top.zero_grad()
    ce_top.backward()
    opt_pred_top.step()

    opt_pred_mid.zero_grad()
    ce_mid.backward()
    opt_pred_mid.step()

    opt_pred_bot.zero_grad()
    ce_bot.backward()
    opt_pred_bot.step()

    return {'ce_top': ce_top.item(), 'ce_mid': ce_mid.item(), 'ce_bot': ce_bot.item()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Fresh HVQVAE Training')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--T', type=int, default=4, help='Number of prediction frames')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overfit_single_batch', action='store_true',
                        help='Overfit on a single batch (sanity check)')
    parser.add_argument('--log_dir', type=str, default='logs/fresh_hvqvae',
                        help='Directory for CSV logs, images, and checkpoints')
    parser.add_argument('--save_images_every', type=int, default=0,
                        help='Save visualization grid every N steps (0 = disabled)')
    parser.add_argument('--data_source', type=str, default='simple',
                        choices=['simple', 'shapes'],
                        help="'simple' = translating square; 'shapes' = rotating 2D shapes")

    # Training mode
    parser.add_argument('--mode', type=str, default='full',
                        choices=['encoder_only', 'predictor_only', 'full'],
                        help='Training mode')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint (required for predictor_only)')

    # Checkpointing
    parser.add_argument('--save_every', type=int, default=0,
                        help='Save checkpoint every N steps (0 = disabled)')
    parser.add_argument('--keep_top_k', type=int, default=3,
                        help='Keep only K best checkpoints by primary metric')

    # Dataset caching
    parser.add_argument('--cache_batches', type=int, default=0,
                        help='Pre-generate N batches to disk (0 = generate on-the-fly)')
    args = parser.parse_args()

    # Validation
    if args.mode == 'predictor_only' and not args.pretrained:
        parser.error("--pretrained is required for predictor_only mode")

    print(f"Device: {args.device}")
    print(f"Mode: {args.mode}")
    print(f"Batch size: {args.batch_size}, T: {args.T}, Steps: {args.steps}")
    print(f"Data source: {args.data_source}")
    print(f"Log dir: {args.log_dir}")
    if args.save_images_every > 0:
        print(f"Saving images every {args.save_images_every} steps")

    os.makedirs(args.log_dir, exist_ok=True)

    # Config
    cfg = FreshHVQVAEConfig(max_T=args.T + 1)
    skip_predictors = (args.mode == 'encoder_only')

    # Model
    model = FreshHVQVAE(cfg, skip_predictors=skip_predictors).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Load pretrained weights for predictor_only mode
    if args.mode == 'predictor_only':
        print(f"Loading pretrained weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=args.device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        # Freeze encoder + decoders
        for p in model.get_encoder_decoder_params():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters (predictors only): {trainable:,}")

    # Optimizers
    opt_enc_dec = opt_pred_top = opt_pred_mid = opt_pred_bot = None
    if args.mode in ('encoder_only', 'full'):
        opt_enc_dec = Adam(model.get_encoder_decoder_params(), lr=cfg.lr_encoder_decoder)
    if args.mode in ('predictor_only', 'full'):
        opt_pred_top = Adam(model.get_predictor_top_params(), lr=cfg.lr_predictor_top)
        opt_pred_mid = Adam(model.get_predictor_mid_params(), lr=cfg.lr_predictor_mid)
        opt_pred_bot = Adam(model.get_predictor_bot_params(), lr=cfg.lr_predictor_bot)

    # Dataset caching
    batch_cache_dir = os.path.join(args.log_dir, 'batch_cache')
    if args.cache_batches > 0:
        cache_dataset(args.data_source, args.cache_batches, args.batch_size,
                       args.T, batch_cache_dir, args.device)

    # Single batch for overfitting
    if args.overfit_single_batch:
        if args.cache_batches > 0:
            batch = load_cached_batch(batch_cache_dir, 1, args.cache_batches, args.device)
        else:
            batch = make_batch(args.data_source, args.batch_size, args.T, device=args.device)
        print(f"Overfitting single batch: {batch.shape}")
    else:
        batch = None

    # CSV logging setup
    csv_path = os.path.join(args.log_dir, 'training_log.csv')
    if args.mode == 'encoder_only':
        csv_fields = ['step', 'mse_bot', 'mse_mid', 'mse_top', 'vq_loss', 'time_s']
        metric_key = 'mse_bot'
    elif args.mode == 'predictor_only':
        csv_fields = ['step', 'ce_top', 'ce_mid', 'ce_bot', 'time_s']
        metric_key = 'ce_top'
    else:
        csv_fields = ['step', 'mse_bot', 'mse_mid', 'mse_top', 'vq_loss',
                      'ce_top', 'ce_mid', 'ce_bot', 'time_s']
        metric_key = 'mse_bot'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Training loop
    model.train()
    for step in range(1, args.steps + 1):
        # Get batch
        if not args.overfit_single_batch:
            if args.cache_batches > 0:
                batch = load_cached_batch(batch_cache_dir, step, args.cache_batches, args.device)
            else:
                batch = make_batch(args.data_source, args.batch_size, args.T, device=args.device)

        t0 = time.time()
        if args.mode == 'encoder_only':
            losses = train_step_encoder_only(model, batch, opt_enc_dec, cfg)
        elif args.mode == 'predictor_only':
            losses = train_step_predictor_only(model, batch, opt_pred_top, opt_pred_mid, opt_pred_bot, cfg)
        else:
            losses = train_step_full(model, batch, opt_enc_dec, opt_pred_top, opt_pred_mid, opt_pred_bot, cfg)
        dt = time.time() - t0

        # CSV logging
        csv_row = {'step': step, 'time_s': f'{dt:.4f}'}
        for k, v in losses.items():
            csv_row[k] = f'{v:.6f}'
        csv_writer.writerow(csv_row)
        csv_file.flush()

        # Console logging
        if step % 10 == 0 or step == 1:
            parts = [f"Step {step:4d}"]
            if 'mse_bot' in losses:
                parts.append(f"mse_bot={losses['mse_bot']:.4f} mse_mid={losses['mse_mid']:.4f} "
                             f"mse_top={losses['mse_top']:.4f} vq={losses['vq_loss']:.4f}")
            if 'ce_top' in losses:
                parts.append(f"ce_top={losses['ce_top']:.3f} ce_mid={losses['ce_mid']:.3f} "
                             f"ce_bot={losses['ce_bot']:.3f}")
            parts.append(f"{dt:.2f}s")
            print(" | ".join(parts))

        # Image saving
        if args.save_images_every > 0 and step % args.save_images_every == 0:
            model.eval()
            with torch.no_grad():
                vis_batch = batch if args.overfit_single_batch else \
                    make_batch(args.data_source, args.batch_size, args.T, device=args.device)
                grid = model.build_visualization(vis_batch)
                img_path = os.path.join(args.log_dir, f'vis_step_{step:06d}.png')
                save_image(grid, img_path)
                print(f"  Saved visualization -> {img_path}")
            model.train()

        # Checkpointing
        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(model, step, losses, args.log_dir, metric_key=metric_key)
            prune_checkpoints(args.log_dir, args.keep_top_k)

    csv_file.close()
    print(f"\nTraining complete. Logs saved to {csv_path}")

    # Save final checkpoint
    if args.save_every > 0:
        save_checkpoint(model, args.steps, losses, args.log_dir, metric_key=metric_key)
        prune_checkpoints(args.log_dir, args.keep_top_k)

    # Final sanity checks
    if args.overfit_single_batch:
        print("\n--- Overfit Sanity Check ---")
        if 'mse_bot' in losses:
            print(f"  mse_bot: {losses['mse_bot']:.6f} (should approach 0)")
            print(f"  mse_mid: {losses['mse_mid']:.6f} (should be small)")
            print(f"  mse_top: {losses['mse_top']:.6f} (stays relatively high — expected)")
        if 'ce_top' in losses:
            print(f"  ce_top:  {losses['ce_top']:.4f} (should approach 0)")
            print(f"  ce_mid:  {losses['ce_mid']:.4f} (should approach 0)")
            print(f"  ce_bot:  {losses['ce_bot']:.4f} (should approach 0)")


if __name__ == '__main__':
    main()
