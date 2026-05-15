"""
Diagnostic script: Why is the top stage VQ collapsing?

Hypotheses:
1. Codebook collapse — the encoder always produces the same top-level code
2. weight_mse_top=0.1 is too low — gradients from top recon are 10x weaker than bot
3. The 256→14 bit bottleneck at 1x1 spatial is too narrow (only 14 binary decisions)
4. The entropy/VQ loss is fighting against reconstruction improvement

This script loads the trained model and checks:
- How many unique top codes are produced across a batch of diverse inputs
- The distribution of pre-VQ activations (z_top before sign())
- Gradient magnitudes through the top path
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from model.vid.fresh_hvqvae import FreshHVQVAE, FreshHVQVAEConfig


def load_model(ckpt_dir, device='cpu'):
    """Load latest checkpoint or train fresh if none found."""
    import glob
    ckpt_path = os.path.join(ckpt_dir, 'checkpoints')
    ckpts = sorted(glob.glob(os.path.join(ckpt_path, 'step_*.pt'))) if os.path.isdir(ckpt_path) else []
    
    cfg = FreshHVQVAEConfig()
    model = FreshHVQVAE(cfg, skip_predictors=True)
    
    if ckpts:
        ckpts.sort(key=lambda p: int(os.path.basename(p).split('_')[1]))
        path = ckpts[-1]
        print(f"Loading: {path}")
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state['model_state_dict'], strict=False)
    else:
        print("No checkpoint found. Training 200 steps to reproduce collapse...")
        model.train()
        opt = torch.optim.Adam(model.get_encoder_decoder_params(), lr=cfg.lr_encoder_decoder)
        for step in range(200):
            batch = generate_diverse_batch(B=4, T=4)
            B, Tp1, C, H, W = batch.shape
            all_frames = batch.reshape(B * Tp1, C, H, W)
            enc = model.encoder(all_frames)
            recon_bot = model.decoder_bot(enc['quant_bot'])
            recon_mid = model.decoder_mid(enc['quant_mid'])
            recon_top = model.decoder_top(enc['quant_top'])
            mse_bot = F.mse_loss(recon_bot, all_frames)
            mse_mid = F.mse_loss(recon_mid, all_frames)
            mse_top = F.mse_loss(recon_top, all_frames)
            vq_loss = enc['loss_bot'] + enc['loss_mid'] + enc['loss_top']
            loss = cfg.weight_mse_bot * mse_bot + cfg.weight_mse_mid * mse_mid + cfg.weight_mse_top * mse_top + vq_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_encoder_decoder_params(), max_norm=cfg.max_grad_norm)
            opt.step()
            if (step+1) % 50 == 0:
                print(f"  step {step+1}: mse_bot={mse_bot.item():.4f} mse_mid={mse_mid.item():.4f} mse_top={mse_top.item():.4f}")
        print("Done training.")
    
    model.eval()
    return model, cfg


def generate_diverse_batch(B=32, T=4, H=64, W=64, device='cpu'):
    """Generate diverse shapes using the shape dataset."""
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
        shape_cache_dir="data/vid/shape_cache_diag",
        shape_no_imagenet_norm=True,
    )
    ds = VIDShapeSyntheticDataset(hparams, size=B, cache=False)
    frames = torch.stack([ds[i] for i in range(B)], dim=0)
    return frames.to(device)


def diagnose_codebook_usage(model, batch):
    """Check how many unique codes the top stage produces."""
    B, Tp1, C, H, W = batch.shape
    all_frames = batch.reshape(B * Tp1, C, H, W)
    
    with torch.no_grad():
        # Run encoder manually to inspect intermediate states
        enc = model.encoder
        
        feat_bot = enc.enc_to_bot(all_frames)
        feat_mid = enc.enc_bot_to_mid(feat_bot)
        feat_top = enc.enc_mid_to_top(feat_mid)
        
        # Pre-VQ projection
        z_top = enc.top_to_vq(feat_top)  # (N, 14, 1, 1)
        z_mid = enc.mid_to_vq(feat_mid)  # (N, 10, 4, 4)
        z_bot = enc.bot_to_vq(feat_bot)  # (N, 6, 16, 16)
        
        # Quantize
        quant_top, idx_top, _ = enc.vq_top(z_top)
        quant_mid, idx_mid, _ = enc.vq_mid(z_mid)
        quant_bot, idx_bot, _ = enc.vq_bot(z_bot)
    
    N = all_frames.shape[0]
    
    print("=" * 60)
    print("CODEBOOK USAGE ANALYSIS")
    print("=" * 60)
    
    # Top stage
    idx_top_flat = idx_top.reshape(-1).numpy()
    unique_top = np.unique(idx_top_flat)
    print(f"\n[TOP] (1x1 spatial, dim={z_top.shape[1]}, codebook=2^14=16384)")
    print(f"  Total tokens: {len(idx_top_flat)}")
    print(f"  Unique codes used: {len(unique_top)} / 16384")
    print(f"  Code distribution: {np.bincount(idx_top_flat, minlength=16384).max()} max count")
    if len(unique_top) <= 20:
        print(f"  Codes: {unique_top}")
    
    # Mid stage
    idx_mid_flat = idx_mid.reshape(-1).numpy()
    unique_mid = np.unique(idx_mid_flat)
    print(f"\n[MID] (4x4 spatial, dim={z_mid.shape[1]}, codebook=2^10=1024)")
    print(f"  Total tokens: {len(idx_mid_flat)}")
    print(f"  Unique codes used: {len(unique_mid)} / 1024")
    
    # Bot stage
    idx_bot_flat = idx_bot.reshape(-1).numpy()
    unique_bot = np.unique(idx_bot_flat)
    print(f"\n[BOT] (16x16 spatial, dim={z_bot.shape[1]}, codebook=2^6=64)")
    print(f"  Total tokens: {len(idx_bot_flat)}")
    print(f"  Unique codes used: {len(unique_bot)} / 64")
    
    # Analyze pre-VQ activations for top
    print("\n" + "=" * 60)
    print("PRE-VQ ACTIVATIONS (z_top before sign quantization)")
    print("=" * 60)
    z_top_flat = z_top.reshape(N, -1).numpy()  # (N, 14)
    print(f"\n  Shape: {z_top.shape} → flattened to ({N}, 14)")
    print(f"  Mean per dim: {z_top_flat.mean(axis=0)}")
    print(f"  Std per dim:  {z_top_flat.std(axis=0)}")
    print(f"  Min per dim:  {z_top_flat.min(axis=0)}")
    print(f"  Max per dim:  {z_top_flat.max(axis=0)}")
    
    # Key insight: if std is very small or mean is far from 0, 
    # sign() will always give the same result → codebook collapse
    abs_mean = np.abs(z_top_flat.mean(axis=0))
    print(f"\n  |Mean| per dim: {abs_mean}")
    print(f"  Dims where |mean| > 2*std (likely collapsed):")
    stds = z_top_flat.std(axis=0)
    for d in range(14):
        if abs_mean[d] > 2 * stds[d]:
            sign = "+" if z_top_flat.mean(axis=0)[d] > 0 else "-"
            print(f"    dim {d}: mean={z_top_flat.mean(axis=0)[d]:.4f}, std={stds[d]:.4f} → always {sign}")
    
    # How many dims are actually varying?
    varying = np.sum(stds > 0.1)
    print(f"\n  Dims with std > 0.1: {varying}/14")
    print(f"  Effective bits: ~{varying} (only these dims contribute to code diversity)")
    
    # Compare with mid 
    z_mid_flat = z_mid.reshape(N, z_mid.shape[1], -1).permute(0, 2, 1).reshape(-1, z_mid.shape[1]).numpy()
    print(f"\n  [MID] z_mid std per dim: {z_mid_flat.std(axis=0)}")


def diagnose_gradient_flow(model, batch, cfg):
    """Check gradient magnitudes through the top path."""
    model.train()
    B, Tp1, C, H, W = batch.shape
    all_frames = batch.reshape(B * Tp1, C, H, W)
    
    # Forward
    enc_dict = model.encoder(all_frames)
    recon_top = model.decoder_top(enc_dict['quant_top'])
    recon_bot = model.decoder_bot(enc_dict['quant_bot'])
    
    mse_top = F.mse_loss(recon_top, all_frames)
    mse_bot = F.mse_loss(recon_bot, all_frames)
    
    # Backward for top only
    model.zero_grad()
    (cfg.weight_mse_top * mse_top).backward(retain_graph=True)
    
    print("\n" + "=" * 60)
    print("GRADIENT ANALYSIS (top path only)")
    print("=" * 60)
    
    # Check gradients on key layers
    layers_to_check = [
        ("encoder.top_to_vq.weight", model.encoder.top_to_vq),
        ("encoder.enc_mid_to_top.0.conv1.weight", model.encoder.enc_mid_to_top[0].conv1),
        ("decoder_top.project.1.weight", model.decoder_top.project[1]),
    ]
    for name, layer in layers_to_check:
        if layer.weight.grad is not None:
            g = layer.weight.grad
            print(f"  {name}: grad_norm={g.norm():.6f}, grad_mean={g.abs().mean():.8f}")
        else:
            print(f"  {name}: NO GRADIENT")
    
    # Now compare with bot path
    model.zero_grad()
    (cfg.weight_mse_bot * mse_bot).backward()
    
    print(f"\n  --- Compare with bot path (weight_mse_bot={cfg.weight_mse_bot}) ---")
    bot_layers = [
        ("encoder.bot_to_vq.weight", model.encoder.bot_to_vq),
        ("encoder.enc_to_bot.0.conv1.weight", model.encoder.enc_to_bot[0].conv1),
        ("decoder_bot.decode.0.conv1.weight", model.decoder_bot.decode[0].conv1),
    ]
    for name, layer in bot_layers:
        if layer.weight.grad is not None:
            g = layer.weight.grad
            print(f"  {name}: grad_norm={g.norm():.6f}, grad_mean={g.abs().mean():.8f}")


def diagnose_loss_landscape(model, batch, cfg):
    """Check what MSE we'd get from mean image vs actual codes."""
    B, Tp1, C, H, W = batch.shape
    all_frames = batch.reshape(B * Tp1, C, H, W)
    
    mean_image = all_frames.mean(dim=0, keepdim=True).expand_as(all_frames)
    mse_mean = F.mse_loss(mean_image, all_frames).item()
    
    with torch.no_grad():
        enc_dict = model.encoder(all_frames)
        recon_top = model.decoder_top(enc_dict['quant_top'])
    mse_top_actual = F.mse_loss(recon_top, all_frames).item()
    
    print("\n" + "=" * 60)
    print("LOSS LANDSCAPE")
    print("=" * 60)
    print(f"  MSE of mean image:     {mse_mean:.6f}")
    print(f"  MSE of top decoder:    {mse_top_actual:.6f}")
    print(f"  Ratio (decoder/mean):  {mse_top_actual/mse_mean:.4f}")
    print(f"  → If ratio ≈ 1.0, decoder is just outputting the mean!")
    
    # Check decoder output diversity
    unique_outputs = recon_top.reshape(recon_top.shape[0], -1)
    pairwise_diffs = torch.cdist(unique_outputs[:8], unique_outputs[:8])
    print(f"\n  Pairwise L2 distance between first 8 decoder outputs:")
    print(f"  Mean: {pairwise_diffs.mean():.6f}, Max: {pairwise_diffs.max():.6f}")
    print(f"  → If ≈ 0, all outputs are identical (collapsed)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='logs/test_ecn_only2')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    print("Loading model...")
    model, cfg = load_model(args.log_dir)
    
    print("Generating diverse batch...")
    batch = generate_diverse_batch(B=args.batch_size)
    
    diagnose_codebook_usage(model, batch)
    diagnose_gradient_flow(model, batch, cfg)
    diagnose_loss_landscape(model, batch, cfg)
    
    print("\n" + "=" * 60)
    print("SUMMARY OF LIKELY CAUSES")
    print("=" * 60)
    print("""
    1. weight_mse_top = 0.1 is 10x weaker than bot, but task is 100x harder
    2. LFQ with dim=14 at 1x1: the sign() quantization + straight-through 
       might not provide enough gradient signal to push encoder outputs 
       away from their initial collapsed state
    3. The entropy loss (which IS negative, suggesting it pushes diversity)
       may not be strong enough to counteract the "safe" strategy of just
       outputting the mean
    """)
