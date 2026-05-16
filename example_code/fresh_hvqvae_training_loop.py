"""
Fresh HVQVAE Training Loop.

Supports 4 training modes:
  sequential     — (DEFAULT) train stages one at a time:
                   enc_bot -> enc_mid -> enc_top -> pred_top -> pred_mid -> pred_bot
  encoder_only   — train all encoder stages simultaneously (legacy)
  predictor_only — load pretrained encoder (frozen), train all predictors
  full           — train everything simultaneously (legacy)

Features:
  - Sequential per-stage training with automatic phase transitions
  - Checkpoint saving every N steps with top-K best kept
  - Auto-checkpoint at phase transitions
  - Phase resume via --start_phase
  - Dataset caching to disk
  - Per-stage logging: MSE, VQ loss, codebook usage, perplexity
"""

import sys
import os
import argparse
import time
import csv
import glob
import json
import math

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torchvision.utils import save_image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vid.fresh_hvqvae import FreshHVQVAE, FreshHVQVAEConfig


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_synthetic_batch(B: int, T: int, H: int = 64, W: int = 64, device: str = 'cpu'):
    """Simple translating colored square + static circle on black background. Returns [-1, 1] range."""
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
    # Scale from [0, 1] to [-1, 1]
    return torch.stack(frames, dim=1) * 2 - 1


def generate_synthetic_shapes_batch(B: int, T: int, H: int = 64, W: int = 64, device: str = 'cpu'):
    """Rotating 2D shapes via VIDShapeSyntheticDataset. Returns [-1, 1] range."""
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
    # Scale from [0, 1] to [-1, 1]
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

def cache_dataset(data_source: str, num_batches: int, B: int, T: int, cache_dir: str, device: str):
    """Pre-generate num_batches batches and save to disk as .pt files."""
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, 'meta.json')

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

def save_checkpoint(model, step, phase_name, losses, log_dir, metric_key='mse_bot'):
    """Save checkpoint with phase info."""
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    metric_val = losses.get(metric_key, 0.0)
    path = os.path.join(ckpt_dir, f'{phase_name}_step_{step:06d}_metric_{metric_val:.6f}.pt')
    torch.save({
        'step': step,
        'phase': phase_name,
        'model_state_dict': model.state_dict(),
        'losses': losses,
    }, path)
    print(f"  Saved checkpoint -> {path}")
    return path


def prune_checkpoints(log_dir, keep_top_k, phase_prefix=None):
    """Keep only the top-K best checkpoints (lowest metric value in filename)."""
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return
    pattern = f'{phase_prefix}_step_*.pt' if phase_prefix else '*_step_*.pt'
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
    if len(ckpts) <= keep_top_k:
        return
    def get_metric(p):
        name = os.path.basename(p)
        try:
            return float(name.split('_metric_')[1].replace('.pt', ''))
        except (IndexError, ValueError):
            return float('inf')
    ckpts.sort(key=get_metric)
    for p in ckpts[keep_top_k:]:
        os.remove(p)


def save_phase_checkpoint(model, phase_name, log_dir):
    """Save a checkpoint at phase transition (always kept, not pruned)."""
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'phase_complete_{phase_name}.pt')
    torch.save({
        'phase': phase_name,
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"  Phase checkpoint -> {path}")
    return path


# ---------------------------------------------------------------------------
# Codebook statistics
# ---------------------------------------------------------------------------

class CodebookUsageTracker:
    """Track codebook utilization over a sliding window via EMA."""

    def __init__(self, codebook_size: int, decay: float = 0.99):
        self.codebook_size = codebook_size
        self.decay = decay
        self.ema_counts = torch.zeros(codebook_size)
        self.total_updates = 0

    def update(self, indices: torch.Tensor):
        """Update with new batch of indices."""
        flat = indices.reshape(-1).cpu()
        batch_counts = torch.bincount(flat, minlength=self.codebook_size).float()
        batch_counts = batch_counts / batch_counts.sum()  # normalize to prob dist

        if self.total_updates == 0:
            self.ema_counts = batch_counts
        else:
            self.ema_counts = self.decay * self.ema_counts + (1 - self.decay) * batch_counts
        self.total_updates += 1

    def get_stats(self) -> dict:
        """Get current utilization statistics."""
        if self.total_updates == 0:
            return {'active_codes': 0, 'dead_codes': self.codebook_size,
                    'ema_usage_pct': 0.0, 'ema_perplexity': 0.0, 'max_prob': 0.0}
        probs = self.ema_counts
        active_mask = probs > 1e-8
        active_codes = active_mask.sum().item()
        dead_codes = self.codebook_size - active_codes

        probs_active = probs[active_mask]
        entropy = -(probs_active * probs_active.log()).sum()
        perplexity = entropy.exp().item()

        return {
            'active_codes': int(active_codes),
            'dead_codes': int(dead_codes),
            'ema_usage_pct': 100.0 * active_codes / self.codebook_size,
            'ema_perplexity': perplexity,
            'max_prob': probs.max().item(),
        }


def compute_codebook_stats(indices, codebook_size):
    """Compute per-batch codebook usage statistics from index tensor."""
    flat = indices.reshape(-1)
    total = flat.numel()
    if total == 0:
        return {'unique_codes': 0, 'usage_pct': 0.0, 'perplexity': 0.0}

    unique = flat.unique().numel()
    counts = torch.bincount(flat, minlength=codebook_size).float()
    probs = counts / total
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum()
    perplexity = entropy.exp().item()

    return {
        'unique_codes': unique,
        'usage_pct': 100.0 * unique / codebook_size,
        'perplexity': perplexity,
    }


# ---------------------------------------------------------------------------
# Per-stage encoder training step
# ---------------------------------------------------------------------------

def train_step_encoder_stage(model, batch, optimizer, stage, cfg):
    """Train a single encoder stage. Other stages run forward but are frozen."""
    B, Tp1 = batch.shape[:2]
    all_frames = batch.reshape(B * Tp1, 3, batch.shape[3], batch.shape[4])

    enc = model.encoder(all_frames)

    if stage == 'bot':
        recon = model.decoder_bot(enc['quant_bot'])
        mse = F.mse_loss(recon, all_frames)
        vq_loss = enc['loss_bot']
        idx = enc['idx_bot']
        codebook_size = cfg.K_bot
    elif stage == 'mid':
        recon = model.decoder_mid(enc['quant_mid'])
        mse = F.mse_loss(recon, all_frames)
        vq_loss = enc['loss_mid']
        idx = enc['idx_mid']
        codebook_size = cfg.K_mid
    else:
        recon = model.decoder_top(enc['quant_top'])
        mse = F.mse_loss(recon, all_frames)
        vq_loss = enc['loss_top']
        idx = enc['idx_top']
        codebook_size = cfg.K_top

    # Scale MSE by 0.25 to compensate for [-1,1] range (4x larger than [0,1] for same error)
    loss = 0.25 * mse + vq_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'],
                                    max_norm=cfg.max_grad_norm)
    optimizer.step()

    cb_stats = compute_codebook_stats(idx.detach(), codebook_size)

    result = {
        f'mse_{stage}': mse.item(),
        f'vq_{stage}': vq_loss.item(),
        f'unique_{stage}': cb_stats['unique_codes'],
        f'ppl_{stage}': cb_stats['perplexity'],
        f'_idx_{stage}': idx.detach(),  # for EMA tracker (avoid redundant forward)
    }

    # Add pre-VQ diagnostic stats for top stage
    if stage == 'top' and 'z_top' in enc:
        z = enc['z_top'].detach()
        result['z_top_mean'] = z.mean().item()
        result['z_top_std'] = z.std().item()
        # Sign balance: fraction of positive values (should be ~0.5)
        result['z_top_sign_balance'] = (z > 0).float().mean().item()

    return result


# ---------------------------------------------------------------------------
# Per-stage predictor training step
# ---------------------------------------------------------------------------

def train_step_predictor_stage(model, batch, optimizer, stage, cfg):
    """Train a single predictor stage. Encoder is fully frozen."""
    B, Tp1 = batch.shape[:2]
    T = Tp1 - 1

    with torch.no_grad():
        enc = model.encode(batch)

    pred = model.predict(enc, T)

    if stage == 'top':
        tgt = enc['idx_top'][:, 1:].detach().reshape(B * T * 1)
        ce = F.cross_entropy(pred['logits_top'].reshape(-1, cfg.K_top), tgt)
    elif stage == 'mid':
        tgt = enc['idx_mid'][:, 1:].detach().reshape(B * T * 16)
        ce = F.cross_entropy(pred['logits_mid'].reshape(-1, cfg.K_mid), tgt)
    else:
        tgt = enc['idx_bot'][:, 1:].detach().reshape(B * T * 256)
        ce = F.cross_entropy(pred['logits_bot'].reshape(-1, cfg.K_bot), tgt)

    optimizer.zero_grad()
    ce.backward()
    optimizer.step()

    return {f'ce_{stage}': ce.item()}


# ---------------------------------------------------------------------------
# Legacy simultaneous training steps
# ---------------------------------------------------------------------------

def train_step_encoder_only(model, batch, opt_enc_dec, cfg):
    """Train all encoder stages simultaneously (legacy mode)."""
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

    cb_bot = compute_codebook_stats(enc['idx_bot'].detach(), cfg.K_bot)
    cb_mid = compute_codebook_stats(enc['idx_mid'].detach(), cfg.K_mid)
    cb_top = compute_codebook_stats(enc['idx_top'].detach(), cfg.K_top)

    return {
        'mse_bot': mse_bot.item(), 'mse_mid': mse_mid.item(), 'mse_top': mse_top.item(),
        'vq_bot': enc['loss_bot'].item(), 'vq_mid': enc['loss_mid'].item(),
        'vq_top': enc['loss_top'].item(),
        'unique_bot': cb_bot['unique_codes'], 'unique_mid': cb_mid['unique_codes'],
        'unique_top': cb_top['unique_codes'],
        'ppl_bot': cb_bot['perplexity'], 'ppl_mid': cb_mid['perplexity'],
        'ppl_top': cb_top['perplexity'],
    }


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


def train_step_predictor_only(model, batch, opt_pred_top, opt_pred_mid, opt_pred_bot, cfg):
    """Train all predictors simultaneously (legacy mode)."""
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
# Logging
# ---------------------------------------------------------------------------

def format_log_line(phase_name, step, total_steps, losses, dt):
    """Format a console log line with phase-aware formatting."""
    parts = [f"[{phase_name}] {step:5d}/{total_steps}"]

    for stage in ('bot', 'mid', 'top'):
        mse_k = f'mse_{stage}'
        vq_k = f'vq_{stage}'
        ppl_k = f'ppl_{stage}'
        uniq_k = f'unique_{stage}'
        ema_active_k = f'ema_active_{stage}'
        ema_ppl_k = f'ema_ppl_{stage}'
        if mse_k in losses:
            s = f"mse={losses[mse_k]:.4f} vq={losses[vq_k]:.3f}"
            if ema_active_k in losses:
                s += f" ema_codes={losses[ema_active_k]}"
            elif uniq_k in losses:
                s += f" codes={losses[uniq_k]}"
            if ema_ppl_k in losses:
                s += f" ema_ppl={losses[ema_ppl_k]:.0f}"
            elif ppl_k in losses:
                s += f" ppl={losses[ppl_k]:.0f}"
            parts.append(f"{stage}[{s}]")

    for stage in ('top', 'mid', 'bot'):
        ce_k = f'ce_{stage}'
        if ce_k in losses:
            parts.append(f"{stage}[ce={losses[ce_k]:.3f}]")

    parts.append(f"{dt:.2f}s")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Sequential training
# ---------------------------------------------------------------------------

PHASE_DEFS = [
    ('enc_bot', 'encoder', 'bot', 'mse_bot'),
    ('enc_mid', 'encoder', 'mid', 'mse_mid'),
    ('enc_top', 'encoder', 'top', 'mse_top'),
    ('pred_top', 'predictor', 'top', 'ce_top'),
    ('pred_mid', 'predictor', 'mid', 'ce_mid'),
    ('pred_bot', 'predictor', 'bot', 'ce_bot'),
]


def get_phase_steps(args):
    """Return ordered list of (phase_name, phase_type, stage, steps, metric_key)."""
    step_map = {
        'enc_bot': args.steps_enc_bot,
        'enc_mid': args.steps_enc_mid,
        'enc_top': args.steps_enc_top,
        'pred_top': args.steps_pred_top,
        'pred_mid': args.steps_pred_mid,
        'pred_bot': args.steps_pred_bot,
    }
    return [(name, ptype, stage, step_map[name], metric)
            for name, ptype, stage, metric in PHASE_DEFS
            if step_map[name] > 0]


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_params(params):
    for p in params:
        p.requires_grad = True


def get_stage_params(model, phase_type, stage):
    if phase_type == 'encoder':
        if stage == 'bot':
            return model.get_bot_stage_params()
        elif stage == 'mid':
            return model.get_mid_stage_params()
        else:
            return model.get_top_stage_params()
    else:
        if stage == 'top':
            return model.get_predictor_top_params()
        elif stage == 'mid':
            return model.get_predictor_mid_params()
        else:
            return model.get_predictor_bot_params()


def get_lr(phase_type, stage, cfg):
    if phase_type == 'encoder':
        return cfg.lr_encoder_decoder
    lr_map = {'top': cfg.lr_predictor_top, 'mid': cfg.lr_predictor_mid,
               'bot': cfg.lr_predictor_bot}
    return lr_map[stage]


def run_sequential(model, cfg, args):
    """Run sequential per-stage training."""
    phases = get_phase_steps(args)
    if not phases:
        print("No phases to run (all step counts are 0).")
        return

    start_idx = 0
    if args.start_phase:
        for i, (name, *_) in enumerate(phases):
            if name == args.start_phase:
                start_idx = i
                break
        else:
            print(f"Warning: --start_phase '{args.start_phase}' not found. Starting from beginning.")

    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=args.device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    batch_cache_dir = os.path.join(args.log_dir, 'batch_cache')
    if args.cache_batches > 0:
        cache_dataset(args.data_source, args.cache_batches, args.batch_size,
                       args.T, batch_cache_dir, args.device)

    fixed_batch = None
    if args.overfit_single_batch:
        fixed_batch = make_batch(args.data_source, args.batch_size, args.T, device=args.device)
        print(f"Overfitting single batch: {fixed_batch.shape}")

    global_step = 0

    print("\n" + "=" * 70)
    print("SEQUENTIAL TRAINING PLAN")
    print("=" * 70)
    total_steps = 0
    for i, (name, ptype, stage, steps, metric) in enumerate(phases):
        marker = ">>>" if i == start_idx else "   "
        skip = "(skip)" if i < start_idx else ""
        print(f"  {marker} Phase {i+1}: {name:10s} | {steps:6d} steps | metric: {metric} {skip}")
        if i >= start_idx:
            total_steps += steps
    print(f"  Total steps to run: {total_steps:,}")
    print("=" * 70 + "\n")

    for phase_idx in range(start_idx, len(phases)):
        phase_name, phase_type, stage, num_steps, metric_key = phases[phase_idx]
        if num_steps <= 0:
            continue

        print(f"\n{'='*70}")
        print(f"PHASE: {phase_name} ({phase_type} stage={stage}, {num_steps} steps)")
        print(f"{'='*70}")

        freeze_all(model)
        active_params = get_stage_params(model, phase_type, stage)
        unfreeze_params(active_params)

        trainable = sum(p.numel() for p in active_params if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters()) - trainable
        print(f"  Trainable: {trainable:,} | Frozen: {frozen:,}")

        lr = get_lr(phase_type, stage, cfg)
        optimizer = Adam(active_params, lr=lr)
        print(f"  LR: {lr}")

        if phase_type == 'encoder':
            csv_fields = ['step', 'global_step', f'mse_{stage}', f'vq_{stage}',
                          f'unique_{stage}', f'ppl_{stage}',
                          f'ema_active_{stage}', f'ema_ppl_{stage}', 'time_s']
            if stage == 'top':
                csv_fields.insert(-1, 'z_top_mean')
                csv_fields.insert(-1, 'z_top_std')
                csv_fields.insert(-1, 'z_top_sign_balance')
        else:
            csv_fields = ['step', 'global_step', f'ce_{stage}', 'time_s']

        csv_path = os.path.join(args.log_dir, f'log_{phase_name}.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()

        # EMA codebook tracker for encoder phases
        cb_tracker = None
        if phase_type == 'encoder':
            codebook_size = {'bot': cfg.K_bot, 'mid': cfg.K_mid, 'top': cfg.K_top}[stage]
            cb_tracker = CodebookUsageTracker(codebook_size, decay=0.99)

        model.train()
        phase_t0 = time.time()

        for step in range(1, num_steps + 1):
            global_step += 1

            if fixed_batch is not None:
                batch = fixed_batch
            elif args.cache_batches > 0:
                batch = load_cached_batch(batch_cache_dir, global_step,
                                           args.cache_batches, args.device)
            else:
                batch = make_batch(args.data_source, args.batch_size, args.T,
                                    device=args.device)

            t0 = time.time()
            if phase_type == 'encoder':
                losses = train_step_encoder_stage(model, batch, optimizer, stage, cfg)
                # Update EMA tracker using indices from training step (no redundant forward)
                if cb_tracker is not None:
                    cb_tracker.update(losses.pop(f'_idx_{stage}'))
                    ema_stats = cb_tracker.get_stats()
                    losses[f'ema_active_{stage}'] = ema_stats['active_codes']
                    losses[f'ema_ppl_{stage}'] = ema_stats['ema_perplexity']
            else:
                losses = train_step_predictor_stage(model, batch, optimizer, stage, cfg)
            dt = time.time() - t0

            csv_row = {'step': step, 'global_step': global_step, 'time_s': f'{dt:.4f}'}
            for k, v in losses.items():
                csv_row[k] = f'{v:.6f}' if isinstance(v, float) else str(v)
            csv_writer.writerow(csv_row)
            if step % 10 == 0:
                csv_file.flush()

            if step % 10 == 0 or step == 1:
                print(format_log_line(phase_name, step, num_steps, losses, dt))

            if args.save_images_every > 0 and step % args.save_images_every == 0:
                model.eval()
                with torch.no_grad():
                    vis_batch = fixed_batch if fixed_batch is not None else \
                        make_batch(args.data_source, args.batch_size, args.T,
                                    device=args.device)
                    grid = model.build_visualization(vis_batch)
                    img_path = os.path.join(args.log_dir,
                                             f'vis_{phase_name}_step_{step:06d}.png')
                    save_image(grid, img_path)
                    print(f"  Saved visualization -> {img_path}")
                model.train()

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(model, step, phase_name, losses,
                                args.log_dir, metric_key=metric_key)
                prune_checkpoints(args.log_dir, args.keep_top_k,
                                   phase_prefix=phase_name)

        csv_file.close()
        phase_elapsed = time.time() - phase_t0
        print(f"\n  Phase {phase_name} complete in {phase_elapsed:.1f}s")
        print(f"  Log saved to {csv_path}")

        save_phase_checkpoint(model, phase_name, args.log_dir)

    print(f"\nAll phases complete. Total global steps: {global_step}")


# ---------------------------------------------------------------------------
# Legacy modes
# ---------------------------------------------------------------------------

def run_legacy(model, cfg, args):
    """Run legacy training modes (encoder_only, predictor_only, full)."""
    if args.mode == 'predictor_only':
        if not args.pretrained:
            print("ERROR: --pretrained is required for predictor_only mode")
            return
        print(f"Loading pretrained weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=args.device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        for p in model.get_encoder_decoder_params():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters (predictors only): {trainable:,}")

    opt_enc_dec = opt_pred_top = opt_pred_mid = opt_pred_bot = None
    if args.mode in ('encoder_only', 'full'):
        opt_enc_dec = Adam(model.get_encoder_decoder_params(), lr=cfg.lr_encoder_decoder)
    if args.mode in ('predictor_only', 'full'):
        opt_pred_top = Adam(model.get_predictor_top_params(), lr=cfg.lr_predictor_top)
        opt_pred_mid = Adam(model.get_predictor_mid_params(), lr=cfg.lr_predictor_mid)
        opt_pred_bot = Adam(model.get_predictor_bot_params(), lr=cfg.lr_predictor_bot)

    batch_cache_dir = os.path.join(args.log_dir, 'batch_cache')
    if args.cache_batches > 0:
        cache_dataset(args.data_source, args.cache_batches, args.batch_size,
                       args.T, batch_cache_dir, args.device)

    fixed_batch = None
    if args.overfit_single_batch:
        fixed_batch = make_batch(args.data_source, args.batch_size, args.T, device=args.device)
        print(f"Overfitting single batch: {fixed_batch.shape}")

    if args.mode == 'encoder_only':
        csv_fields = ['step', 'mse_bot', 'mse_mid', 'mse_top',
                      'vq_bot', 'vq_mid', 'vq_top',
                      'unique_bot', 'unique_mid', 'unique_top',
                      'ppl_bot', 'ppl_mid', 'ppl_top', 'time_s']
        metric_key = 'mse_bot'
    elif args.mode == 'predictor_only':
        csv_fields = ['step', 'ce_top', 'ce_mid', 'ce_bot', 'time_s']
        metric_key = 'ce_top'
    else:
        csv_fields = ['step', 'mse_bot', 'mse_mid', 'mse_top', 'vq_loss',
                      'ce_top', 'ce_mid', 'ce_bot', 'time_s']
        metric_key = 'mse_bot'

    csv_path = os.path.join(args.log_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    model.train()
    for step in range(1, args.steps + 1):
        if fixed_batch is not None:
            batch = fixed_batch
        elif args.cache_batches > 0:
            batch = load_cached_batch(batch_cache_dir, step, args.cache_batches, args.device)
        else:
            batch = make_batch(args.data_source, args.batch_size, args.T, device=args.device)

        t0 = time.time()
        if args.mode == 'encoder_only':
            losses = train_step_encoder_only(model, batch, opt_enc_dec, cfg)
        elif args.mode == 'predictor_only':
            losses = train_step_predictor_only(model, batch, opt_pred_top, opt_pred_mid,
                                                opt_pred_bot, cfg)
        else:
            losses = train_step_full(model, batch, opt_enc_dec, opt_pred_top,
                                      opt_pred_mid, opt_pred_bot, cfg)
        dt = time.time() - t0

        csv_row = {'step': step, 'time_s': f'{dt:.4f}'}
        for k, v in losses.items():
            csv_row[k] = f'{v:.6f}' if isinstance(v, float) else str(v)
        csv_writer.writerow(csv_row)
        csv_file.flush()

        if step % 10 == 0 or step == 1:
            print(format_log_line(args.mode, step, args.steps, losses, dt))

        if args.save_images_every > 0 and step % args.save_images_every == 0:
            model.eval()
            with torch.no_grad():
                vis_batch = fixed_batch if fixed_batch is not None else \
                    make_batch(args.data_source, args.batch_size, args.T, device=args.device)
                grid = model.build_visualization(vis_batch)
                img_path = os.path.join(args.log_dir, f'vis_step_{step:06d}.png')
                save_image(grid, img_path)
                print(f"  Saved visualization -> {img_path}")
            model.train()

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(model, step, args.mode, losses, args.log_dir,
                            metric_key=metric_key)
            prune_checkpoints(args.log_dir, args.keep_top_k)

    csv_file.close()
    print(f"\nTraining complete. Logs saved to {csv_path}")

    if args.save_every > 0:
        save_checkpoint(model, args.steps, args.mode, losses, args.log_dir,
                        metric_key=metric_key)
        prune_checkpoints(args.log_dir, args.keep_top_k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Fresh HVQVAE Training')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--T', type=int, default=4, help='Number of prediction frames')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overfit_single_batch', action='store_true')
    parser.add_argument('--log_dir', type=str, default='logs/fresh_hvqvae')
    parser.add_argument('--save_images_every', type=int, default=0,
                        help='Save visualization every N steps (0 = disabled)')
    parser.add_argument('--data_source', type=str, default='shapes',
                        choices=['simple', 'shapes'])

    # Training mode
    parser.add_argument('--mode', type=str, default='sequential',
                        choices=['sequential', 'encoder_only', 'predictor_only', 'full'])

    # Sequential phase step counts
    parser.add_argument('--steps_enc_bot', type=int, default=5000)
    parser.add_argument('--steps_enc_mid', type=int, default=10000)
    parser.add_argument('--steps_enc_top', type=int, default=15000)
    parser.add_argument('--steps_pred_top', type=int, default=15000)
    parser.add_argument('--steps_pred_mid', type=int, default=10000)
    parser.add_argument('--steps_pred_bot', type=int, default=5000)
    parser.add_argument('--start_phase', type=str, default=None,
                        choices=['enc_bot', 'enc_mid', 'enc_top',
                                 'pred_top', 'pred_mid', 'pred_bot'],
                        help='Skip phases before this one (for resume)')

    # Legacy mode step count
    parser.add_argument('--steps', type=int, default=5000,
                        help='Total steps (for legacy modes: encoder_only/predictor_only/full)')

    # Pretrained weights
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint (for resume or predictor_only)')

    # Checkpointing
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--keep_top_k', type=int, default=3)

    # Dataset caching
    parser.add_argument('--cache_batches', type=int, default=0,
                        help='Pre-generate N batches to disk (0 = on-the-fly)')

    # Predictor mode
    parser.add_argument('--predictor_mode', type=str, default='vanilla',
                        choices=['vanilla', 'ebt'],
                        help='Predictor type: vanilla (one-shot) or ebt (MCMC refinement)')
    parser.add_argument('--ebt_mcmc_steps', type=int, default=5,
                        help='Number of MCMC refinement steps for EBT predictor')

    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Mode: {args.mode}")
    print(f"Batch size: {args.batch_size}, T: {args.T}")
    print(f"Data source: {args.data_source}")
    print(f"Log dir: {args.log_dir}")
    os.makedirs(args.log_dir, exist_ok=True)

    cfg = FreshHVQVAEConfig(max_T=args.T + 1,
                            predictor_mode=args.predictor_mode,
                            ebt_mcmc_num_steps=args.ebt_mcmc_steps)

    skip_predictors = args.mode in ('encoder_only', 'sequential')
    if args.mode == 'sequential':
        has_pred = (args.steps_pred_top > 0 or args.steps_pred_mid > 0 or
                    args.steps_pred_bot > 0)
        skip_predictors = not has_pred

    model = FreshHVQVAE(cfg, skip_predictors=skip_predictors).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    if args.mode == 'sequential':
        run_sequential(model, cfg, args)
    else:
        run_legacy(model, cfg, args)


if __name__ == '__main__':
    main()
