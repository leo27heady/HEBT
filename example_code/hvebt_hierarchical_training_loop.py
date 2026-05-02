"""
HVEBT Phase 2/3 + decoder — hierarchical training loop on
VIDShapeSyntheticDataset at 256x256.

Trains a 2- or 3-stage HVEBT (CLIP s1 -> s2 -> s3) with strict 2x2 parent-child
cross-attention and detached KV between stages. Optionally trains a pixel
decoder on top of the finest stage's prediction (detached) and dumps
side-by-side comparison grids of (real | decoded) frames every N steps.

Logs per-stage init/final reconstruction, per-stage energy gap, per-stage
gradient norm, alpha values, and a copy-last-frame baseline at every stage.

Run examples:
    .\\venv\\Scripts\\Activate.ps1
    # 2-stage smoke (s1 + s2), no decoder:
    python example_code\\hvebt_hierarchical_training_loop.py \\
        --stages s1 s2 --max_steps 30 --dataset_size 32 --context_length 4

    # 3-stage with decoder + frame dumps:
    python example_code\\hvebt_hierarchical_training_loop.py \\
        --stages s1 s2 s3 --max_steps 60 --dataset_size 32 --context_length 4 \\
        --decoder --decoder_save_every 10
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vid.preprocessed_clip_dataset import PreprocessedCLIPDataset  # noqa: E402
from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset  # noqa: E402
from model.vid.hvebt import (  # noqa: E402
    HierarchicalHVEBT,
    HierarchicalHVEBTConfig,
    HVEBTStageConfig,
    save_recon_grid,
)


_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


# Mapping from clip stage name -> (channels, H, W) at 256x256 input.
_STAGE_INFO: Dict[str, Tuple[int, int, int]] = {
    "s1": (128, 32, 32),
    "s2": (256, 16, 16),
    "s3": (512, 8,  8),
}


def denormalize_imnet(x: torch.Tensor) -> torch.Tensor:
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


def build_stage_configs(args) -> List[HVEBTStageConfig]:
    cfgs: List[HVEBTStageConfig] = []
    for i, name in enumerate(args.stages):
        if name not in _STAGE_INFO:
            raise ValueError(f"Unknown stage '{name}'; choose from {list(_STAGE_INFO)}")
        c, h, w = _STAGE_INFO[name]
        # Per-stage embed_dim: --embed_dim_per_stage list, else default args.embed_dim.
        if args.embed_dim_per_stage and len(args.embed_dim_per_stage) == len(args.stages):
            d = args.embed_dim_per_stage[i]
        else:
            d = args.embed_dim
        cfgs.append(HVEBTStageConfig(
            clip_stage_name=name, clip_channels=c, H=h, W=w,
            embed_dim=d, n_heads=args.n_heads, n_layers=args.n_layers,
        ))
    return cfgs


def make_model(args, device: torch.device) -> HierarchicalHVEBT:
    stage_cfgs = build_stage_configs(args)
    weights = "" if args.preprocessed_dir else "clip/MobileCLIP2-S0/mobileclip2_s0.pt"
    cfg = HierarchicalHVEBTConfig(
        stages=stage_cfgs,
        mcmc_num_steps=args.mcmc_steps,
        mcmc_step_size=args.mcmc_step_size,
        mcmc_step_size_learnable=True,
        denoising_init=args.denoising_init,
        disable_cross_attn=args.disable_cross_attn,
        interleaved_mcmc=args.interleaved_mcmc,
        detach_kv=not args.no_detach_kv,
        progressive=args.progressive,
        progressive_steps_per_stage=args.progressive_steps,
        decoder_enabled=args.decoder,
        decoder_out_size=args.image_size,
        decoder_loss_weight=args.decoder_loss_weight,
        weights_path=weights,
    )
    return HierarchicalHVEBT(cfg).to(device)


def _per_stage_grad_norms(model: HierarchicalHVEBT) -> List[float]:
    norms = []
    for stage in model.stages:
        gs = [p.grad for p in stage.parameters() if p.grad is not None]
        if not gs:
            norms.append(0.0); continue
        norms.append(math.sqrt(sum((g.detach() ** 2).sum().item() for g in gs)))
    return norms


def _decoder_grad_norm(model: HierarchicalHVEBT) -> float:
    if model.decoder is None:
        return 0.0
    gs = [p.grad for p in model.decoder.parameters() if p.grad is not None]
    if not gs:
        return 0.0
    return math.sqrt(sum((g.detach() ** 2).sum().item() for g in gs))


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[hvebt-h] device={device}  stages={args.stages}  decoder={args.decoder}")
    if args.disable_cross_attn:
        print("[hvebt-h] *** ABLATION MODE: cross-attention DISABLED ***")
    if args.interleaved_mcmc:
        print("[hvebt-h] MCMC schedule: INTERLEAVED (1 step per stage, cycle K times)")
    else:
        print("[hvebt-h] MCMC schedule: SEQUENTIAL (all K steps per stage before passing KV)")
    if args.no_detach_kv:
        print("[hvebt-h] KV detach: OFF (gradients flow through cross-attn KV)")
    if args.progressive:
        print(f"[hvebt-h] Progressive training: {args.progressive_steps} steps per stage")
    use_preprocessed = args.preprocessed_dir is not None

    if use_preprocessed:
        dataset = PreprocessedCLIPDataset(args.preprocessed_dir, stages=args.stages)
        print(f"[hvebt-h] Using preprocessed features from: {args.preprocessed_dir}")
    else:
        hparams = make_hparams(args)
        dataset = VIDShapeSyntheticDataset(hparams, size=args.dataset_size)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True,
    )

    model = make_model(args, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[hvebt-h] trainable params: {n_trainable/1e6:.2f}M")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "train_log.csv")
    log_f = open(log_path, "w")
    header_cols = ["step", "loss_total", "loss_energy"]
    if args.decoder:
        header_cols += ["loss_decoder"]
    for i, name in enumerate(args.stages):
        header_cols += [
            f"s{i}_{name}_init_recon", f"s{i}_{name}_final_recon",
            f"s{i}_{name}_init_energy", f"s{i}_{name}_final_energy",
            f"s{i}_{name}_energy_gap",
            f"s{i}_{name}_baseline_copy",
            f"s{i}_{name}_alpha", f"s{i}_{name}_grad_norm",
        ]
    if args.decoder:
        header_cols += ["dec_grad_norm"]
    header_cols += ["secs"]
    log_f.write(",".join(header_cols) + "\n")

    print(f"[hvebt-h] starting: max_steps={args.max_steps}")

    step = 0
    t_start = time.time()
    done = False
    while not done:
        for batch in loader:
            t0 = time.time()

            # Progressive training: activate stages one by one.
            newly_activated = model.update_progressive(step)
            if newly_activated is not None:
                stage_name = args.stages[newly_activated]
                print(f"[hvebt-h] step {step}: activated stage {newly_activated} "
                      f"({stage_name}) — now {model.num_active_stages}/{len(args.stages)} active")

            if use_preprocessed:
                # batch is dict {stage_name: (B, T, C, H, W)}
                feats_dict = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                out = model.forward_loss_from_features(feats_dict, learning=True)
            else:
                batch = batch.to(device, non_blocking=True)
                video01 = denormalize_imnet(batch)
                out = model.forward_loss(video01, learning=True)
            loss_total = out["loss_total"]

            opt.zero_grad(set_to_none=True)
            loss_total.backward()

            if not torch.isfinite(loss_total):
                print(f"[hvebt-h][ABORT] non-finite loss at step {step}: {loss_total.item()}")
                return

            stage_norms = _per_stage_grad_norms(model)
            dec_norm = _decoder_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
            opt.step()

            dt = time.time() - t0

            # CSV
            row: List[str] = [str(step), f"{loss_total.item():.6f}", f"{out['loss_energy'].item():.6f}"]
            if args.decoder:
                dl = out.get('loss_decoder')
                row += [f"{dl.item():.6f}" if dl is not None else ""]
            for i, s in enumerate(out["per_stage"]):
                if s is not None:
                    row += [
                        f"{s['init_recon'].item():.6f}", f"{s['final_recon'].item():.6f}",
                        f"{s['init_energy'].item():.6f}", f"{s['final_energy'].item():.6f}",
                        f"{s['energy_gap'].item():.6f}",
                        f"{s['baseline_copy_last'].item():.6f}",
                        f"{s['alpha'].item():.4f}", f"{stage_norms[i]:.6f}",
                    ]
                else:
                    row += [""] * 8  # inactive stage
            if args.decoder:
                row += [f"{dec_norm:.6f}"]
            row += [f"{dt:.3f}"]
            log_f.write(",".join(row) + "\n")
            log_f.flush()

            # Console
            if step % args.log_every == 0:
                pieces = [f"[step {step:5d}] L={loss_total.item():.3f}"]
                if args.decoder and 'loss_decoder' in out:
                    pieces += [f"Le={out['loss_energy'].item():.3f}",
                               f"Ld={out['loss_decoder'].item():.3f}"]
                for i, s in enumerate(out["per_stage"]):
                    if s is None:
                        continue  # inactive stage
                    name = args.stages[i]
                    fr = s['final_recon'].item()
                    base = s['baseline_copy_last'].item()
                    vs_base = (base - fr) / max(base, 1e-8)
                    pieces.append(
                        f"[{name}] r{fr:.3f}/b{base:.3f} ({vs_base*100:+.0f}%) "
                        f"Eg{s['energy_gap'].item():+.2e} g{stage_norms[i]:.2f}"
                    )
                if args.decoder:
                    pieces.append(f"dg{dec_norm:.2f}")
                pieces.append(f"dt={dt:.2f}s")
                print(" ".join(pieces))

            # Decoder image dumps
            if args.decoder and 'decoded_rgb' in out and args.decoder_save_every > 0 and step % args.decoder_save_every == 0:
                # `decoded_rgb` and `target_rgb` come from forward_loss when decoder is enabled.
                save_recon_grid(
                    real_rgb=out["target_rgb"].cpu(),
                    pred_rgb=out["decoded_rgb"].cpu(),
                    path=os.path.join(args.log_dir, "frames", f"step_{step:06d}.png"),
                    max_clips=2,
                )

            step += 1
            if step >= args.max_steps:
                done = True
                break

    log_f.close()
    elapsed = time.time() - t_start
    print(f"[hvebt-h] finished: {step} steps in {elapsed:.1f}s ({step/max(elapsed,1e-6):.2f} steps/s)")
    print(f"[hvebt-h] log: {log_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--context_length", type=int, default=8)
    ap.add_argument("--dataset_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--preprocessed_dir", type=str, default=None,
                    help="Path to preprocessed CLIP features (from preprocess_clip_features.py). "
                         "If set, skips CLIP encoder during training.")
    # stages
    ap.add_argument("--stages", nargs="+", default=["s1", "s2"],
                    help="Ordered list of MobileCLIP stages from finest to coarsest, e.g. 's1 s2 s3'.")
    ap.add_argument("--embed_dim", type=int, default=128, help="Default per-stage embed_dim.")
    ap.add_argument("--embed_dim_per_stage", type=int, nargs="+", default=None,
                    help="Optional per-stage override (must match #stages).")
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    # mcmc
    ap.add_argument("--mcmc_steps", type=int, default=2)
    ap.add_argument("--mcmc_step_size", type=float, default=1000.0)
    ap.add_argument("--denoising_init", type=str, default="zeros",
                    choices=["zeros", "random_noise", "real_current"])
    # ablation
    ap.add_argument("--disable_cross_attn", action="store_true",
                    help="Ablation: disable parent KV conditioning between stages. "
                         "Each stage runs independent MCMC without top-down signal.")
    ap.add_argument("--interleaved_mcmc", action="store_true",
                    help="Interleaved MCMC: 1 step per stage, cycle K times. "
                         "Default is sequential (all K steps per stage before passing KV).")
    ap.add_argument("--no_detach_kv", action="store_true",
                    help="Allow gradients to flow through cross-attn KV "
                         "(default: KV is detached).")
    ap.add_argument("--progressive", action="store_true",
                    help="Progressive training: start with apex stage only, "
                         "activate finer stages one by one (StyleGAN-like).")
    ap.add_argument("--progressive_steps", type=int, default=500,
                    help="Steps per stage before activating the next finer stage.")
    # decoder
    ap.add_argument("--decoder", action="store_true")
    ap.add_argument("--decoder_loss_weight", type=float, default=1.0)
    ap.add_argument("--decoder_save_every", type=int, default=20)
    # optimization
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_dir", type=str, default="logs/hvebt_hier")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
