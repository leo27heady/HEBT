"""Quick test: verify the BatchNorm fix prevents top-stage codebook collapse."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from model.vid.fresh_hvqvae import FreshHVQVAE, FreshHVQVAEConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_batch(B=4, T=4, H=64, W=64):
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
        shape_cache_dir="data/vid/shape_cache_test",
        shape_no_imagenet_norm=True,
    )
    ds = VIDShapeSyntheticDataset(hparams, size=B, cache=False)
    return torch.stack([ds[i] for i in range(B)], dim=0).to(DEVICE)


def main():
    print(f"Device: {DEVICE}")
    cfg = FreshHVQVAEConfig()
    model = FreshHVQVAE(cfg, skip_predictors=True).to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.get_encoder_decoder_params(), lr=cfg.lr_encoder_decoder)

    print(f"Config: weight_mse_top={cfg.weight_mse_top}, weight_mse_mid={cfg.weight_mse_mid}")
    print(f"Training 300 steps...")
    
    for step in range(300):
        batch = generate_batch(B=4)
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
            print(f"  step {step+1}: mse_bot={mse_bot.item():.4f} mse_mid={mse_mid.item():.4f} mse_top={mse_top.item():.4f} vq={vq_loss.item():.4f}")

    # Now check codebook usage
    print("\n--- Checking codebook usage after 300 steps ---")
    model.eval()
    eval_batch = generate_batch(B=16)
    B, Tp1, C, H, W = eval_batch.shape
    all_frames = eval_batch.reshape(B * Tp1, C, H, W)
    
    with torch.no_grad():
        enc = model.encoder(all_frames)
        
        # Check z_top stats
        feat_top = model.encoder.enc_mid_to_top(
            model.encoder.enc_bot_to_mid(model.encoder.enc_to_bot(all_frames))
        )
        z_top_raw = model.encoder.top_to_vq(feat_top)
        z_top_normed = model.encoder.top_pre_vq_norm(z_top_raw)
        
    idx_top = enc['idx_top'].reshape(-1).cpu().numpy()
    idx_mid = enc['idx_mid'].reshape(-1).cpu().numpy()
    idx_bot = enc['idx_bot'].reshape(-1).cpu().numpy()
    
    unique_top = len(np.unique(idx_top))
    unique_mid = len(np.unique(idx_mid))
    unique_bot = len(np.unique(idx_bot))
    
    print(f"\n  [TOP] Unique codes: {unique_top} / 16384 (from {len(idx_top)} tokens)")
    print(f"  [MID] Unique codes: {unique_mid} / 1024 (from {len(idx_mid)} tokens)")
    print(f"  [BOT] Unique codes: {unique_bot} / 64 (from {len(idx_bot)} tokens)")
    
    z_flat = z_top_normed.reshape(all_frames.shape[0], -1).cpu().numpy()
    print(f"\n  z_top (after BatchNorm) stats:")
    print(f"  Mean per dim: {z_flat.mean(axis=0)}")
    print(f"  Std per dim:  {z_flat.std(axis=0)}")
    print(f"  BN running_mean (first 5): {model.encoder.top_pre_vq_norm.running_mean[:5].tolist()}")
    
    # Check reconstruction quality
    with torch.no_grad():
        recon_top = model.decoder_top(enc['quant_top'])
    mse_top = F.mse_loss(recon_top, all_frames).item()
    mean_img = all_frames.mean(0, keepdim=True).expand_as(all_frames)
    mse_mean = F.mse_loss(mean_img, all_frames).item()
    print(f"\n  MSE top decoder: {mse_top:.6f}")
    print(f"  MSE mean image:  {mse_mean:.6f}")
    print(f"  Ratio:           {mse_top/mse_mean:.4f} (< 1.0 = better than mean)")
    
    # Pairwise diversity
    outputs = recon_top.reshape(recon_top.shape[0], -1)
    dists = torch.cdist(outputs[:8], outputs[:8])
    print(f"  Output diversity (pairwise L2): mean={dists.mean():.4f}, max={dists.max():.4f}")
    
    if unique_top > 10:
        print("\n  SUCCESS: Top stage is using diverse codes!")
    elif unique_top > 2:
        print("\n  PARTIAL: Some diversity, may need more training")
    else:
        print("\n  STILL COLLAPSED: Fix insufficient")


if __name__ == "__main__":
    main()
