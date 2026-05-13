"""
Fresh HVQVAE Training Loop.

Per-stage reconstruction (encoder+decoder) + isolated per-stage CE (predictors).
Single forward pass, 4 isolated backward passes (no retain_graph).
"""

import sys
import os
import argparse
import time

import torch
import torch.nn.functional as F
from torch.optim import Adam

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vid.fresh_hvqvae import FreshHVQVAE, FreshHVQVAEConfig


def generate_synthetic_batch(B: int, T: int, H: int = 64, W: int = 64, device: str = 'cpu'):
    """
    Generate a synthetic video batch for testing.
    Simple rotating/translating colored shapes on black background.
    """
    frames = []
    for t in range(T + 1):
        frame = torch.zeros(B, 3, H, W, device=device)
        # Moving colored square
        cx = int(H * 0.3 + H * 0.4 * (t / T))
        cy = int(W * 0.3 + W * 0.4 * (t / T))
        size = 8
        r_start = max(0, cx - size)
        r_end = min(H, cx + size)
        c_start = max(0, cy - size)
        c_end = min(W, cy + size)
        frame[:, 0, r_start:r_end, c_start:c_end] = 1.0  # Red channel
        frame[:, 1, r_start:r_end, c_start:c_end] = 0.5  # Green channel

        # Add a static circle-ish region
        for r in range(H):
            for c in range(W):
                if (r - H // 2) ** 2 + (c - W // 4) ** 2 < 100:
                    frame[:, 2, r, c] = 0.7

        frames.append(frame)

    return torch.stack(frames, dim=1)  # (B, T+1, 3, H, W)


def train_step(model: FreshHVQVAE, batch: torch.Tensor, opt_enc_dec, opt_pred_top, opt_pred_mid, opt_pred_bot, cfg: FreshHVQVAEConfig):
    """
    Single training step with isolated backward passes.
    batch: (B, T+1, 3, 64, 64)
    """
    # Forward: compute all losses
    losses = model(batch)

    # Combined reconstruction loss
    recon_total = (
        cfg.weight_mse_bot * losses['mse_bot'] +
        cfg.weight_mse_mid * losses['mse_mid'] +
        cfg.weight_mse_top * losses['mse_top'] +
        losses['vq_loss']
    )

    # Backward 1: Encoder + Decoder + Upscalers
    opt_enc_dec.zero_grad()
    recon_total.backward()
    torch.nn.utils.clip_grad_norm_(model.get_encoder_decoder_params(), max_norm=cfg.max_grad_norm)
    opt_enc_dec.step()

    # Backward 2: Predictor Top
    opt_pred_top.zero_grad()
    losses['ce_top'].backward()
    opt_pred_top.step()

    # Backward 3: Predictor Mid
    opt_pred_mid.zero_grad()
    losses['ce_mid'].backward()
    opt_pred_mid.step()

    # Backward 4: Predictor Bot
    opt_pred_bot.zero_grad()
    losses['ce_bot'].backward()
    opt_pred_bot.step()

    return {k: v.item() for k, v in losses.items()}


def main():
    parser = argparse.ArgumentParser(description='Fresh HVQVAE Training')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--T', type=int, default=4, help='Number of prediction frames')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overfit_single_batch', action='store_true',
                        help='Overfit on a single batch (sanity check)')
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}, T: {args.T}, Steps: {args.steps}")

    # Config
    cfg = FreshHVQVAEConfig(max_T=args.T + 1)

    # Model
    model = FreshHVQVAE(cfg).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizers (strictly separated)
    opt_enc_dec = Adam(model.get_encoder_decoder_params(), lr=cfg.lr_encoder_decoder)
    opt_pred_top = Adam(model.get_predictor_top_params(), lr=cfg.lr_predictor_top)
    opt_pred_mid = Adam(model.get_predictor_mid_params(), lr=cfg.lr_predictor_mid)
    opt_pred_bot = Adam(model.get_predictor_bot_params(), lr=cfg.lr_predictor_bot)

    # Generate synthetic data
    if args.overfit_single_batch:
        batch = generate_synthetic_batch(args.batch_size, args.T, device=args.device)
        print(f"Overfitting single batch: {batch.shape}")
    else:
        batch = None

    # Training loop
    model.train()
    for step in range(1, args.steps + 1):
        if not args.overfit_single_batch:
            batch = generate_synthetic_batch(args.batch_size, args.T, device=args.device)

        t0 = time.time()
        losses = train_step(model, batch, opt_enc_dec, opt_pred_top, opt_pred_mid, opt_pred_bot, cfg)
        dt = time.time() - t0

        if step % 10 == 0 or step == 1:
            print(f"Step {step:4d} | "
                  f"mse_bot={losses['mse_bot']:.4f} mse_mid={losses['mse_mid']:.4f} "
                  f"mse_top={losses['mse_top']:.4f} vq={losses['vq_loss']:.4f} | "
                  f"ce_top={losses['ce_top']:.3f} ce_mid={losses['ce_mid']:.3f} "
                  f"ce_bot={losses['ce_bot']:.3f} | {dt:.2f}s")

    print("\nTraining complete.")

    # Final sanity checks
    if args.overfit_single_batch:
        print("\n--- Overfit Sanity Check ---")
        print(f"  mse_bot: {losses['mse_bot']:.6f} (should approach 0)")
        print(f"  mse_mid: {losses['mse_mid']:.6f} (should be small)")
        print(f"  mse_top: {losses['mse_top']:.6f} (stays relatively high — expected)")
        print(f"  ce_top:  {losses['ce_top']:.4f} (should approach 0)")
        print(f"  ce_mid:  {losses['ce_mid']:.4f} (should approach 0)")
        print(f"  ce_bot:  {losses['ce_bot']:.4f} (should approach 0)")


if __name__ == '__main__':
    main()
