"""One-off labeled prediction visualization for S-HR-VQVAE checkpoints."""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

# ---- project root on path ------------------------------------------------- #
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset  # noqa: E402
from model.vid.shr_vqvae import SHRVQVAEConfig, SHRVQVAEModel              # noqa: E402
from shr_vqvae_viz import save_labeled_prediction_panel                     # noqa: E402


def make_dataloader(
    cfg: SHRVQVAEConfig,
    dataset_size: int,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> DataLoader:
    hparams = SimpleNamespace(
        context_length=cfg.T + cfg.S,
        image_dims=[image_size, image_size],
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
        shape_no_imagenet_norm=True,
    )
    dataset = VIDShapeSyntheticDataset(hparams, size=dataset_size, cache=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=(num_workers > 0),
        drop_last=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate one labeled prediction panel")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    p.add_argument("--dataset_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--num_future", type=int, default=-1, help="-1 means full S")
    p.add_argument("--show_indices", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log_dir", type=str, default="logs/shr_vqvae_predict_viz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", SHRVQVAEConfig(image_h=args.image_size, image_w=args.image_size))
    model = SHRVQVAEModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {args.ckpt}")

    loader = make_dataloader(
        cfg=cfg,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    batch = next(iter(loader))
    num_future = cfg.S if args.num_future <= 0 else min(args.num_future, cfg.S)

    save_labeled_prediction_panel(
        model=model,
        batch=batch,
        step=0,
        log_dir=args.log_dir,
        device=device,
        num_future=num_future,
        show_indices=args.show_indices,
        tag="pred_panel_oneoff",
    )
    print(f"Saved panel to {os.path.join(args.log_dir, 'viz')}")


if __name__ == "__main__":
    main()
