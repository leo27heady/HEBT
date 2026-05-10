"""
VQ-HVEBT Training Loop Example.

Trains VQ-HVEBT (single or multi-stage) to predict quantized video latents
one step into the future.

Usage
-----
python example_code/vq_hvebt_training_loop.py \
    --weights_path clip/MobileCLIP2-S0/mobileclip2_s0.pt \
    --stages s3 s2 s1 \
    --num_codes 512 \
    --batch_size 4 \
    --T 4 \
    --lr 3e-4 \
    --encoder_lr_scale 0.1 \
    --steps 10000 \
    --log_every 100

For a quick sanity check on a single batch (overfitting test with 2D shapes):
python example_code/vq_hvebt_training_loop.py \
    --weights_path clip/MobileCLIP2-S0/mobileclip2_s0.pt \
    --overfit_single_batch \
    --steps 500
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# ---- path fix so script can be run from project root ---------------------- #
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset  # noqa: E402
from model.vid.vq_hvebt.config import (
    VQCodebookConfig,
    VQHVEBTConfig,
    VQStageConfig,
    _default_stages,
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


def make_shape_batch(
    B: int,
    T: int,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """Return (B, T+1, 3, H, W) of rotating 2D shapes (ImageNet-normalised).

    Uses VIDShapeSyntheticDataset (disk-cached) so repeated calls are fast.
    """
    hparams = SimpleNamespace(
        context_length=T + 1,
        image_dims=[H, W],
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
        shape_cache_dir="data/vid/shape_cache",
    )
    ds = VIDShapeSyntheticDataset(hparams, size=B)
    frames = torch.stack([ds[i] for i in range(B)], dim=0)  # (B, T+1, 3, H, W)
    return frames.to(device)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #


def build_dataset(args: argparse.Namespace) -> VIDShapeSyntheticDataset:
    """Build shape dataset from args (generates or loads from cache)."""
    hparams = SimpleNamespace(
        context_length=args.T + 1,
        image_dims=[args.image_size, args.image_size],
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
        shape_cache_dir=args.data_dir if args.data_dir else "data/vid/shape_cache",
    )
    return VIDShapeSyntheticDataset(hparams, size=args.dataset_size)


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #


def build_model(args: argparse.Namespace, device: torch.device) -> VQHVEBTModel:
    """Construct VQ-HVEBT model with selected stages."""
    # Spatial sizes for 64×64 input with base_channels=32 encoder.
    STAGE_INFO = {
        "s1": (64,   8,  8),
        "s2": (128,  4,  4),
        "s3": (256,  2,  2),
        "s_pool": (256, 1, 1),
    }

    stage_cfgs: List[VQStageConfig] = []
    num_stages = len(args.stages)
    for idx, stage_name in enumerate(args.stages):
        C, H, W = STAGE_INFO[stage_name]

        # Per-stage windowing: auto_windowing assigns exponential temporal
        # windows (finest=1, each coarser stage doubles) and spatial_window=8
        # for stages with H>8. Explicit --temporal_window/--spatial_window
        # override auto values for all stages.
        if args.temporal_window is not None:
            tw = args.temporal_window
        elif args.auto_windowing and num_stages > 1:
            # Stages are coarsest-first: idx=0 is coarsest, idx=num_stages-1 is finest.
            # depth_from_finest: finest=0, next=1, coarsest=num_stages-1.
            depth_from_finest = num_stages - 1 - idx
            # Formula: w_t = min(2^depth, T). Finest gets 1, coarsest gets 2^(N-1).
            tw = min(2 ** depth_from_finest, args.T)
        else:
            tw = None

        if args.spatial_window is not None:
            sw = args.spatial_window
        elif args.auto_windowing and num_stages > 1 and H > 8:
            sw = 8
        else:
            sw = None

        # Scale codebook size inversely with spatial resolution.
        # Coarse stages (few tokens) must encode the whole scene → large K.
        # Fine stages (many tokens) describe single patches → small K.
        # Mapping: 1×1 → num_codes, 2×2 → num_codes, 4×4 → num_codes/8, 8×8 → 16
        STAGE_K = {
            "s_pool": args.num_codes,            # 1×1: entire scene in 1 token
            "s3":     args.num_codes,            # 2×2: scene in 4 tokens
            "s2":     max(16, args.num_codes // 8),  # 4×4: 16 tokens
            "s1":     16,                        # 8×8: 64 tokens, each is a small patch
        }
        stage_K = STAGE_K[stage_name]

        stage_cfgs.append(VQStageConfig(
            clip_stage_name=stage_name,
            clip_channels=C,
            H=H, W=W,
            transformer_dim=args.transformer_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            mcmc_steps=args.mcmc_steps,
            mcmc_step_size=args.mcmc_step_size,
            mcmc_per_token_norm=args.mcmc_per_token_norm,
            soft_target_tau=args.soft_target_tau,
            pred_head=args.pred_head,
            energy_bound=args.energy_bound,
            energy_reg_weight=args.energy_reg_weight,
            temporal_window=tw,
            spatial_window=sw,
            adaptive_mcmc=args.adaptive_mcmc,
            adaptive_mcmc_max_steps=args.adaptive_mcmc_max_steps,
            adaptive_mcmc_tol=args.adaptive_mcmc_tol,
            adaptive_mcmc_patience=args.adaptive_mcmc_patience,
            adaptive_mcmc_step_penalty=args.adaptive_mcmc_step_penalty,
            codebook=VQCodebookConfig(
                num_codes=stage_K,
                code_dim=C,
                init_mode="data_first_batch",
                use_ema=args.use_ema_codebook,
                ema_decay=args.ema_decay,
                commitment_beta=args.commitment_beta,
                dead_code_reset=args.dead_code_reset,
            ),
            pred_loss_weight=1.0,
            cb_loss_weight=1.0 if not args.use_ema_codebook else 0.0,
            commit_loss_weight=args.commitment_beta if not args.use_ema_codebook else 0.0,
        ))

    cfg = VQHVEBTConfig(
        stages=stage_cfgs,
        train_encoder=not args.freeze_encoder,
        encoder_lr_scale=args.encoder_lr_scale,
        weights_path=args.weights_path,
        use_custom_encoder=args.use_custom_encoder,
        encoder_base_channels=args.encoder_base_channels,
        ema_target_decay=args.ema_target_decay,
        use_decoder=args.use_decoder,
        decoder_out_size=args.image_size,
        contrastive_loss_weight=args.contrastive_loss_weight,
        encoder_warmup_steps=args.encoder_warmup_steps,
        detach_parent_kv=args.detach_parent_kv,
        bottom_up_grad_scale=args.bottom_up_grad_scale,
        decoder_detach=args.decoder_detach,
        decoder_only_loss=args.decoder_only_loss,
        context_recon_weight=args.context_recon_weight,
        codebook_diversity_weight=args.codebook_diversity_weight,
    )
    model = VQHVEBTModel(cfg).to(device)
    return model


def build_optimizer(model: VQHVEBTModel, base_lr: float) -> torch.optim.Optimizer:
    """Build a single AdamW optimizer with encoder at reduced LR.

    With EMA codebook there is no codebook gradient, no commitment loss,
    and no 10⁹-scale gradient explosion. A single AdamW with per-group
    LR is sufficient. No separate SGD needed.
    """
    return torch.optim.AdamW(
        model.parameter_groups(base_lr),
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )


def save_pred_images(pred_rgb: torch.Tensor, gt_batch: torch.Tensor,
                     stage_results: dict, step: int, save_dir: Path) -> None:
    """Save predicted/GT frames and per-stage energy maps.

    Args:
        pred_rgb: (B, T, 3, H_out, W_out) predicted frames in [0, 1].
        gt_batch: (B, T+1, 3, H, W) full batch (context+future) in [0, 1].
        stage_results: dict of StageForwardResult from model output.
        step: Current training step number.
        save_dir: Directory to write images into.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    B, T = pred_rgb.shape[:2]
    # Take first sample in batch, save each predicted frame + corresponding GT
    pred_frames = pred_rgb[0].clamp(0, 1)       # (T, 3, H, W)
    gt_frames = gt_batch[0, 1:T+1].clamp(0, 1)  # (T, 3, H, W) future frames

    # Resize GT to match pred resolution if needed
    if gt_frames.shape[-2:] != pred_frames.shape[-2:]:
        gt_frames = nn.functional.interpolate(
            gt_frames, size=pred_frames.shape[-2:], mode="bilinear", align_corners=False
        )

    # Stack pred and GT vertically: top=pred, bottom=GT for each frame
    grid = torch.cat([pred_frames, gt_frames], dim=0)  # (2*T, 3, H, W)
    save_image(grid, save_dir / f"step_{step:06d}.png", nrow=T)

    # Save per-stage energy maps.
    for name, sr in stage_results.items():
        if sr.final_energy is None:
            continue
        H_s, W_s = sr.pred_embed.shape[3], sr.pred_embed.shape[4]
        if H_s < 2 or W_s < 2:
            continue  # skip s_pool (1x1 has no spatial map)
        # sr.final_energy: (B, T*H*W) → reshape to (T, 1, H, W) for first sample
        energy_map = sr.final_energy[0].reshape(T, H_s, W_s).unsqueeze(1)  # (T, 1, H, W)
        # Normalize to [0, 1] for visualization
        e_min = energy_map.min()
        e_max = energy_map.max()
        if e_max > e_min:
            energy_map = (energy_map - e_min) / (e_max - e_min)
        else:
            energy_map = energy_map * 0
        # Upscale to match pred frame size for easy comparison
        energy_map = nn.functional.interpolate(
            energy_map, size=pred_frames.shape[-2:], mode="nearest"
        )
        # Save as grayscale (repeat to 3ch for save_image)
        energy_rgb = energy_map.repeat(1, 3, 1, 1)  # (T, 3, H, W)
        save_image(energy_rgb, save_dir / f"step_{step:06d}_energy_{name}.png", nrow=T)


def save_entropy_maps(
    stage_results: dict,
    stage_configs: Dict[str, VQStageConfig],
    step: int,
    save_dir: Path,
    pred_frame_size: Tuple[int, int],
) -> None:
    """Save per-stage entropy maps as grayscale images.

    White = low entropy (certain), Black = high entropy (uncertain).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, sr in stage_results.items():
        if sr.final_logits is None:
            continue
        cfg = stage_configs[name]
        K = cfg.codebook.num_codes
        H_s, W_s = cfg.H, cfg.W
        if H_s < 2 or W_s < 2:
            continue

        T = sr.final_logits.shape[1] // (H_s * W_s)

        # Shannon entropy in bits.
        probs = torch.softmax(sr.final_logits[0], dim=-1)            # (N, K)
        log_probs = torch.log2(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)                   # (N,)
        max_entropy = math.log2(K)

        # 1 = certain (white), 0 = uncertain (black).
        entropy_norm = (1.0 - (entropy / max_entropy).clamp(0, 1))

        entropy_map = entropy_norm.reshape(T, H_s, W_s).unsqueeze(1)  # (T, 1, H, W)
        entropy_map = nn.functional.interpolate(entropy_map, size=pred_frame_size, mode="nearest")
        entropy_rgb = entropy_map.repeat(1, 3, 1, 1)
        save_image(entropy_rgb, save_dir / f"step_{step:06d}_entropy_{name}.png", nrow=T)


def save_energy_maps(
    stage_results: dict,
    stage_configs: Dict[str, VQStageConfig],
    step: int,
    save_dir: Path,
    pred_frame_size: Tuple[int, int],
) -> None:
    """Save per-stage energy maps with principled normalization.

    With bounded energy: black = -B, gray = 0, white = +B.
    With unbounded energy: per-batch [min, max] normalization.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, sr in stage_results.items():
        if sr.final_energy is None:
            continue
        cfg = stage_configs[name]
        H_s, W_s = cfg.H, cfg.W
        if H_s < 2 or W_s < 2:
            continue

        T = sr.final_energy.shape[1] // (H_s * W_s)
        energy = sr.final_energy[0]

        bound = cfg.energy_bound
        if bound > 0:
            energy_norm = (energy + bound) / (2 * bound)
        else:
            e_min, e_max = energy.min(), energy.max()
            energy_norm = (energy - e_min) / (e_max - e_min + 1e-8)

        energy_norm = energy_norm.clamp(0, 1)
        energy_map = energy_norm.reshape(T, H_s, W_s).unsqueeze(1)
        energy_map = nn.functional.interpolate(energy_map, size=pred_frame_size, mode="nearest")
        energy_rgb = energy_map.repeat(1, 3, 1, 1)
        save_image(energy_rgb, save_dir / f"step_{step:06d}_energy_bounded_{name}.png", nrow=T)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # ---- decoder_only_loss implies several flags -------------------------- #
    if args.decoder_only_loss:
        args.use_decoder = True
        args.decoder_detach = False
        args.detach_parent_kv = False
        # Per-stage flags are set by VQHVEBTConfig.__post_init__:
        #   mcmc_no_detach=True, truncate_mcmc=False
        # CE loss + decoder loss combine: CE gives energy function direct
        # supervision, decoder provides pixel-level feedback.
        print("[VQ-HVEBT] decoder_only_loss: CE + decoder loss, no detach, "
              f"full gradient flow through MCMC chain")

    print(f"[VQ-HVEBT] Building model on {device} (stages: {args.stages}) ...")
    model = build_model(args, device)

    # Verify cross-attention is active for multi-stage configs.
    if len(args.stages) > 1:
        n_cross = sum(1 for p in model.predictors.values() if p.use_cross_attn)
        print(f"[VQ-HVEBT] Multi-stage: {n_cross}/{len(args.stages)-1} finer stages have cross-attention from parent.")
        if n_cross == 0:
            raise RuntimeError(
                "Multiple stages requested but no cross-attention found! "
                "Check that stages are ordered coarsest-first (e.g. --stages s3 s2 s1)."
            )

    optimizers = build_optimizer(model, args.lr)
    # Per-group param lists for separate grad clipping.
    # Encoder grads are huge (raw CLIP backbone) and would starve predictor
    # if clipped jointly.
    enc_params = model.encoder_params()
    pred_params = model.non_encoder_params()

    # ---- logging setup ---------------------------------------------------- #
    log_dir = Path(args.log_dir) if args.log_dir else None
    csv_file = None
    csv_writer = None
    img_dir = None

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_file = open(log_dir / "train_log.csv", "w", newline="")
        csv_writer = None  # header written on first step (dynamic columns)

    if args.save_images_every > 0:
        if not args.use_decoder:
            print("[WARNING] --save_images_every requires --use_decoder; image saving disabled.")
            args.save_images_every = 0
        else:
            img_dir = (log_dir or Path("logs/_images")) / "predictions"
            img_dir.mkdir(parents=True, exist_ok=True)
            print(f"[VQ-HVEBT] Saving predicted images every {args.save_images_every} steps to {img_dir}")

    # Build stage name → config map for entropy/energy map functions.
    stage_configs_map: Dict[str, VQStageConfig] = {
        sc.clip_stage_name: sc for sc in model.cfg.stages
    }
    # Predicted frame size (for upscaling maps) — use decoder output size or image_size.
    pred_frame_size = (args.image_size, args.image_size)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[VQ-HVEBT] Total params: {total_params:,}  Trainable: {trainable_params:,}")

    # ---- single-batch overfit test ---------------------------------------- #
    if args.overfit_single_batch:
        print("[VQ-HVEBT] Overfitting single batch ...")
        fixed_batch = make_shape_batch(args.batch_size, args.T, args.image_size, args.image_size, device)
        print(f"[VQ-HVEBT] Initializing codebooks from data ...")
        model.maybe_initialize_codebooks(fixed_batch)

        model.train()
        for step in range(1, args.steps + 1):
            optimizers.zero_grad()
            out = model.forward_loss(fixed_batch)
            out.total_loss.backward()
            nn.utils.clip_grad_norm_(enc_params, max_norm=1.0)
            nn.utils.clip_grad_norm_(pred_params, max_norm=1.0)
            optimizers.step()
            model.update_ema_encoder()  # EMA target encoder tracks live encoder

            if step % args.log_every == 0 or step == 1:
                m = out.metrics
                # --- General summary line ---
                parts = [f"  step {step:>5}  total={out.total_loss.item():.4f}"]
                if "decoder/loss" in m:
                    parts.append(f"  dec={m['decoder/loss']:.4f}")
                if "decoder/ctx_recon_loss" in m:
                    parts.append(f"  ctx_recon={m['decoder/ctx_recon_loss']:.4f}")
                print("".join(parts))
                # --- Per-stage details ---
                for sname in args.stages:
                    pred_l = m.get(f"{sname}/loss_pred", 0)
                    usage = m.get(f"{sname}/codebook_usage", 0)
                    perpl = m.get(f"{sname}/codebook_perplexity", 0)
                    ent = m.get(f"{sname}/entropy_avg", 0)
                    e_mean = m.get(f"{sname}/energy_mean", 0)
                    e_std = m.get(f"{sname}/energy_std", 0)
                    print(
                        f"    {sname:>6}: pred={pred_l:.4f}"
                        f"  usage={usage:.3f}  perpl={perpl:.1f}"
                        f"  entropy={ent:.2f}b"
                        f"  energy={e_mean:.2f}\u00b1{e_std:.2f}"
                    )
                # CSV logging
                if csv_file:
                    row = {"step": step, "total_loss": out.total_loss.item(), **m}
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                    csv_file.flush()

            # Image saving (overfit loop)
            if args.save_images_every > 0 and step % args.save_images_every == 0:
                if out.pred_rgb is not None:
                    save_pred_images(out.pred_rgb.detach(), fixed_batch.detach(),
                                     out.stage_results, step, img_dir)
                save_entropy_maps(out.stage_results, stage_configs_map, step, img_dir, pred_frame_size)
                save_energy_maps(out.stage_results, stage_configs_map, step, img_dir, pred_frame_size)

        if csv_file:
            csv_file.close()
        return

    # ---- real training loop ----------------------------------------------- #
    print("[VQ-HVEBT] Building dataset ...")
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    print(f"[VQ-HVEBT] Dataset: {len(dataset)} clips, {len(loader)} batches/epoch")

    print("[VQ-HVEBT] Starting training ...")
    model.train()
    codebook_init_done = False
    step = 0
    epoch = 0

    while step < args.steps:
        epoch += 1
        for batch in loader:
            step += 1
            if step > args.steps:
                break
            batch = batch.to(device, non_blocking=True)

            # Codebook initialization from first real batch.
            if not codebook_init_done:
                did_init = model.maybe_initialize_codebooks(batch)
                codebook_init_done = True
                if did_init:
                    print("[VQ-HVEBT] Codebooks initialized from first data batch.")

            t0 = time.perf_counter()
            optimizers.zero_grad()
            out = model.forward_loss(batch)
            out.total_loss.backward()
            nn.utils.clip_grad_norm_(enc_params, max_norm=1.0)
            nn.utils.clip_grad_norm_(pred_params, max_norm=1.0)
            optimizers.step()
            model.update_ema_encoder()  # EMA target encoder tracks live encoder
            dt = time.perf_counter() - t0

            if step % args.log_every == 0 or step == 1:
                m = out.metrics
                # --- General summary line ---
                parts = [f"  step {step:>6}  total={out.total_loss.item():.4f}  dt={dt*1000:.0f}ms"]
                if "decoder/loss" in m:
                    parts.append(f"  dec={m['decoder/loss']:.4f}")
                if "decoder/ctx_recon_loss" in m:
                    parts.append(f"  ctx_recon={m['decoder/ctx_recon_loss']:.4f}")
                print("".join(parts))
                # --- Per-stage details ---
                for sname in args.stages:
                    pred_l = m.get(f"{sname}/loss_pred", 0)
                    usage = m.get(f"{sname}/codebook_usage", 0)
                    perpl = m.get(f"{sname}/codebook_perplexity", 0)
                    ent = m.get(f"{sname}/entropy_avg", 0)
                    e_mean = m.get(f"{sname}/energy_mean", 0)
                    e_std = m.get(f"{sname}/energy_std", 0)
                    print(
                        f"    {sname:>6}: pred={pred_l:.4f}"
                        f"  usage={usage:.3f}  perpl={perpl:.1f}"
                        f"  entropy={ent:.2f}b"
                        f"  energy={e_mean:.2f}\u00b1{e_std:.2f}"
                    )

                # CSV logging
                if csv_file:
                    row = {"step": step, "total_loss": out.total_loss.item(), "dt_ms": dt * 1000, **m}
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                    csv_file.flush()

            # Image saving
            if args.save_images_every > 0 and step % args.save_images_every == 0:
                if out.pred_rgb is not None:
                    save_pred_images(out.pred_rgb.detach(), batch.detach(),
                                     out.stage_results, step, img_dir)
                save_entropy_maps(out.stage_results, stage_configs_map, step, img_dir, pred_frame_size)
                save_energy_maps(out.stage_results, stage_configs_map, step, img_dir, pred_frame_size)

    if csv_file:
        csv_file.close()
    print("[VQ-HVEBT] Training complete.")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VQ-HVEBT training loop")
    p.add_argument("--weights_path", default="clip/MobileCLIP2-S0/mobileclip2_s0.pt")
    p.add_argument("--stages", nargs="+", default=["s1"],
                   choices=["s1", "s2", "s3", "s_pool"],
                   help="Stages to train (coarsest first), e.g. --stages s_pool s3 s2 s1")
    p.add_argument("--use_decoder", action="store_true",
                   help="Attach pixel decoder on the finest stage")
    # ---- Encoder architecture ----
    p.add_argument("--use_custom_encoder", action="store_true", default=True,
                   help="Use custom ConvEncoder instead of CLIP (Option B, default: enabled)")
    p.add_argument("--use_clip_encoder", dest="use_custom_encoder", action="store_false",
                   help="Use pretrained CLIP backbone instead of custom ConvEncoder")
    p.add_argument("--encoder_base_channels", type=int, default=32,
                   help="Stem width for custom ConvEncoder (32 → ~1.3M params)")
    p.add_argument("--ema_target_decay", type=float, default=0.999,
                   help="EMA decay for target encoder (BYOL/DINO style, 0.999 → 1000-step half-life)")
    # ---- VQ codebook ----
    p.add_argument("--num_codes", type=int, default=256)
    p.add_argument("--use_ema_codebook", action="store_true", default=False,
                   help="Use EMA-updated codebook (no gradient). Default: gradient-trained.")
    p.add_argument("--ema_decay", type=float, default=0.99,
                   help="EMA decay for codebook updates (only with --use_ema_codebook)")
    p.add_argument("--commitment_beta", type=float, default=0.25,
                   help="Commitment loss weight β (gradient codebook mode)")
    p.add_argument("--dead_code_reset", action="store_true", default=True,
                   help="Reset dead codebook entries with encoder samples (EMA mode only)")
    p.add_argument("--no_dead_code_reset", dest="dead_code_reset", action="store_false",
                   help="Disable dead code reset")
    # ---- Transformer / predictor ----
    p.add_argument("--transformer_dim", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--n_layers", type=int, default=2)
    # ---- MCMC ----
    p.add_argument("--mcmc_steps", type=int, default=10)
    p.add_argument("--mcmc_step_size", type=float, default=1.0,
                   help="MCMC step size α (per-token after normalization)")
    p.add_argument("--mcmc_per_token_norm", action="store_true", default=True,
                   help="F4: Per-token gradient normalization in MCMC (default: enabled)")
    p.add_argument("--no_mcmc_per_token_norm", dest="mcmc_per_token_norm", action="store_false")
    p.add_argument("--soft_target_tau", type=float, default=0.0,
                   help="Soft target temperature (>0: smooth distance-based targets, 0: hard one-hot)")
    # ---- Adaptive MCMC (Improvement 2) ----
    p.add_argument("--adaptive_mcmc", action="store_true",
                   help="Enable adaptive MCMC convergence (run until energy converges)")
    p.add_argument("--adaptive_mcmc_max_steps", type=int, default=50,
                   help="Hard upper bound on adaptive MCMC iterations")
    p.add_argument("--adaptive_mcmc_tol", type=float, default=1e-3,
                   help="Relative energy-change threshold for convergence")
    p.add_argument("--adaptive_mcmc_patience", type=int, default=3,
                   help="Consecutive overshoots before halving step size")
    p.add_argument("--adaptive_mcmc_step_penalty", type=float, default=0.0,
                   help="Weight for step-count regularizer (0=disabled)")
    # ---- Attention windowing (Improvement 1) ----
    p.add_argument("--auto_windowing", action="store_true",
                   help="Enable hierarchical per-stage windowing (exponential scaling: "
                        "finest stage=temporal_window=1, each coarser stage doubles). "
                        "spatial_window=8 for stages with H>8.")
    p.add_argument("--temporal_window", type=int, default=None,
                   help="Override: apply same temporal window to ALL stages (None=use auto or full)")
    p.add_argument("--spatial_window", type=int, default=None,
                   help="Override: apply same spatial window to ALL stages (None=use auto or full)")
    # ---- Prediction head (F2/F3) ----
    p.add_argument("--pred_head", action="store_true", default=False,
                   help="F2/F3: Learned prediction head for MCMC warm-start (default: enabled)")
    p.add_argument("--no_pred_head", dest="pred_head", action="store_false")
    # ---- Energy bounding (F1) ----
    p.add_argument("--energy_bound", type=float, default=10.0,
                   help="F1: Bound energy via tanh to [-bound, +bound]. 0=unbounded.")
    p.add_argument("--energy_reg_weight", type=float, default=0.01,
                   help="F1: Energy regularization weight (λ * energy².mean()). 0=disabled.")
    # ---- Bottom-up gradient flow (Improvement 3) ----
    p.add_argument("--no_detach_parent_kv", dest="detach_parent_kv", action="store_false",
                   default=True, help="Enable bottom-up gradient flow through parent KV")
    p.add_argument("--bottom_up_grad_scale", type=float, default=0.1,
                   help="Gradient scale for bottom-up flow (only when --no_detach_parent_kv)")
    p.add_argument("--no_decoder_detach", dest="decoder_detach", action="store_false",
                   default=True, help="Let decoder loss gradient flow into predictor")
    p.add_argument("--decoder_only_loss", action="store_true",
                   help="Train only via decoder pixel loss (disables per-stage CE, "
                        "enables --use_decoder, --no_decoder_detach, --no_detach_parent_kv)")
    p.add_argument("--context_recon_weight", type=float, default=0.0,
                   help="Weight for context-frame reconstruction loss (VQ-VAE autoencoder "
                        "objective). Trains encoder-decoder feature space. "
                        "Auto-set to 1.0 when --decoder_only_loss is used.")
    p.add_argument("--codebook_diversity_weight", type=float, default=1.0,
                   help="Weight for codebook diversity loss. Penalises high pairwise "
                        "cosine similarity among encoder features, preventing codebook "
                        "collapse at coarse stages.")
    # ---- Training ----
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--T", type=int, default=4,
                   help="Number of context frames; model predicts T future frames")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder_lr_scale", type=float, default=1.0,
                   help="Encoder LR = lr * scale (1.0 for custom encoder, 0.01 for CLIP)")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze encoder (useful for debugging predictor in isolation)")
    p.add_argument("--encoder_warmup_steps", type=int, default=200,
                   help="Freeze encoder gradient for first N steps (stabilizes codebook + predictor)")
    p.add_argument("--contrastive_loss_weight", type=float, default=0.0,
                   help="Weight for contrastive energy loss (E(true) < E(predicted))")
    p.add_argument("--dataset_size", type=int, default=1000,
                   help="Number of shape clips to generate (ignored if --data_dir points to existing cache)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to pre-created shape cache folder. If None, generates to data/vid/shape_cache/")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overfit_single_batch", action="store_true",
                   help="Run overfitting test on a single fixed batch (2D rotating shapes)")
    p.add_argument("--log_dir", type=str, default=None,
                   help="Directory to save training logs (CSV). If None, no logs saved.")
    p.add_argument("--save_images_every", type=int, default=0,
                   help="Save predicted images every N steps (requires --use_decoder, 0=disabled)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
