"""Visualizations for LFQ VQ-VAE training (reconstruction and video prediction)."""

from __future__ import annotations

import math
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from model.vid.lfq_vqvae import LFQVAE


def flatten_video_batch(batch: torch.Tensor) -> torch.Tensor:
    """
    Collapse video clips into a flat image batch for per-frame VQ-VAE training/viz.

    (B, T, C, H, W) -> (B*T, C, H, W). Already-flat (B, C, H, W) batches pass through.
    """
    if batch.dim() == 5:
        b, t, c, h, w = batch.shape
        return batch.reshape(b * t, c, h, w)
    if batch.dim() == 4:
        return batch
    raise ValueError(f"Expected batch dim 4 or 5, got shape {tuple(batch.shape)}")


_PAD = 4
_LEFT_LABEL_W = 100
_HEADER_H = 22
_BLANK_GRAY = 0.15


def _to_uint8_chw(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [0, 1] -> uint8 HWC."""
    arr = img.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0).astype(np.uint8)


def _blank_rgb(h: int, w: int, value: float = _BLANK_GRAY) -> np.ndarray:
    v = int(max(0, min(255, round(value * 255))))
    return np.full((h, w, 3), v, dtype=np.uint8)


def _entropy_rgb_from_logits(
    logits: torch.Tensor,
    spatial_hw: int,
    image_h: int,
    image_w: int,
    codebook_size: int,
) -> np.ndarray:
    """
    Per-frame entropy map from predictor logits.

    logits: (S, K) tokens for one target frame at one stage.
    Returns uint8 HWC: black = high entropy, white = low entropy.
    """
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    ent = -(probs * log_probs).sum(dim=-1)  # (S,)
    max_ent = math.log(codebook_size)
    ent_norm = 1.0 - (ent / max(max_ent, 1e-8)).clamp(0.0, 1.0)  # invert to get black = high entropy, white = low entropy
    ent_map = ent_norm.view(spatial_hw, spatial_hw)
    ent_4d = ent_map.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
    up = F.interpolate(ent_4d, size=(image_h, image_w), mode="nearest")
    gray = up[0, 0].cpu().numpy()  # (H, W) in [0, 1]
    rgb = np.stack([gray, gray, gray], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def _draw_labeled_grid(
    rows: List[List[np.ndarray]],
    row_labels: List[str],
    col_labels: List[str],
) -> Image.Image:
    """Assemble labeled grid: rows of HWC uint8 tiles, column headers, row labels."""
    n_rows = len(rows)
    n_cols = len(col_labels)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("Empty grid")

    tile_h, tile_w = rows[0][0].shape[:2]
    grid_w = n_cols * (tile_w + _PAD) + _PAD
    grid_h = n_rows * (tile_h + _PAD) + _PAD
    canvas = Image.new("RGB", (_LEFT_LABEL_W + grid_w, _HEADER_H + grid_h), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 11)
        font_sm = ImageFont.truetype("arial.ttf", 9)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    for c, label in enumerate(col_labels):
        x = _LEFT_LABEL_W + _PAD + c * (tile_w + _PAD) + tile_w // 2
        draw.text((x, 4), str(label), fill=(240, 240, 240), font=font_sm, anchor="mm")

    for r, (row_label, row_tiles) in enumerate(zip(row_labels, rows)):
        y = _HEADER_H + _PAD + r * (tile_h + _PAD) + tile_h // 2
        draw.text((6, y), row_label, fill=(240, 240, 240), font=font, anchor="lm")
        for c, tile in enumerate(row_tiles):
            x = _LEFT_LABEL_W + _PAD + c * (tile_w + _PAD)
            y0 = _HEADER_H + _PAD + r * (tile_h + _PAD)
            canvas.paste(Image.fromarray(tile), (x, y0))

    return canvas


@torch.no_grad()
def save_recon_panel(
    model: LFQVAE,
    batch: torch.Tensor,
    step: int,
    log_dir: str,
    device: torch.device,
    max_samples: int = 4,
    tag: str = "recon",
) -> str:
    """Simple grid: inputs on top row, reconstructions on bottom row."""
    model.eval()
    frames = flatten_video_batch(batch)

    n = min(max_samples, frames.shape[0])
    x = frames[:n].to(device)
    out = model(x)
    x_hat = out["x_hat"].clamp(0.0, 1.0)

    grid = torch.cat([x, x_hat], dim=0)
    frames_dir = os.path.join(log_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    out_path = os.path.join(frames_dir, f"{tag}_step_{step:06d}.png")
    from torchvision.utils import save_image

    save_image(grid, out_path, nrow=n, padding=2)
    model.train()
    return out_path


@torch.no_grad()
def save_video_panel(
    model: LFQVAE,
    video: torch.Tensor,
    step: int,
    log_dir: str,
    device: torch.device,
    sample_idx: int = 0,
    tag: str = "video_panel",
) -> str:
    """
    Full video visualization for one sequence (B, T+1, C, H, W).

    Columns: frame indices 0 .. T (T = T+1 - 1).
    Rows: GT, Recon, Pred (col 0 blank), Entropy Top/Mid/Bot (col 0 blank).

    Prediction and entropy columns 1..T correspond to target frames 1..T.
    """
    if video.dim() != 5:
        raise ValueError("save_video_panel expects video shape (B, T+1, C, H, W)")

    model.eval()
    cfg = model.cfg
    B, Tp1, C, H, W = video.shape
    T = Tp1 - 1
    b = min(sample_idx, B - 1)
    vid = video[b : b + 1].to(device)  # (1, T+1, C, H, W)

    col_labels = [str(t) for t in range(Tp1)]
    row_labels: List[str] = []
    row_tiles: List[List[np.ndarray]] = []

    # --- GT row ---
    row_labels.append("GT")
    row_tiles.append([_to_uint8_chw(vid[0, t]) for t in range(Tp1)])

    # --- Reconstruction row (all frames) ---
    flat = vid.reshape(Tp1, C, H, W)
    # Use model.forward so progressive depth matches training (not always depth=3).
    recon_flat = model(flat)["x_hat"].clamp(0.0, 1.0)
    row_labels.append("Recon")
    row_tiles.append([_to_uint8_chw(recon_flat[t]) for t in range(Tp1)])

    has_predictor = model.has_video_predictor and cfg.enable_video_predictor

    if has_predictor and T >= 1:
        enc = model.encode_video(vid)
        pred = model.predict_video(enc, T)
        pred_rgb = model.decode_predicted(pred, 1, T).clamp(0.0, 1.0)  # (T, C, H, W)

        # Pred row: blank at 0, predicted RGB at 1..T
        row_labels.append("Pred")
        pred_row: List[np.ndarray] = [_blank_rgb(H, W)]
        for t in range(1, Tp1):
            pred_row.append(_to_uint8_chw(pred_rgb[t - 1]))
        row_tiles.append(pred_row)

        k_bot, k_mid, k_top = cfg.stage_codebook_sizes
        stage_specs: List[Tuple[str, int, int]] = [
            ("Entropy Top", 1, k_top),
            ("Entropy Mid", 4, k_mid),
            ("Entropy Bot", 16, k_bot),
        ]
        logits_map = {
            "Entropy Top": pred["logits_top"][0],
            "Entropy Mid": pred["logits_mid"][0],
            "Entropy Bot": pred["logits_bot"][0],
        }
        for row_name, spatial, k in stage_specs:
            s_tokens = spatial * spatial
            row_labels.append(row_name)
            ent_row: List[np.ndarray] = [_blank_rgb(H, W)]
            for t in range(1, Tp1):
                tok_logits = logits_map[row_name][(t - 1) * s_tokens : t * s_tokens]
                ent_row.append(
                    _entropy_rgb_from_logits(tok_logits, spatial, H, W, k)
                )
            row_tiles.append(ent_row)

    canvas = _draw_labeled_grid(row_tiles, row_labels, col_labels)
    frames_dir = os.path.join(log_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    out_path = os.path.join(frames_dir, f"{tag}_step_{step:06d}.png")
    canvas.save(out_path)

    if has_predictor:
        usage = {
            "bot": int(enc["idx_bot"][0, 1:].unique().numel()),
            "mid": int(enc["idx_mid"][0, 1:].unique().numel()),
            "top": int(enc["idx_top"][0, 1:].unique().numel()),
        }
        print(
            f"  [viz] {tag} step {step} | usage bot={usage['bot']}/{k_bot} "
            f"mid={usage['mid']}/{k_mid} top={usage['top']}/{k_top}"
        )
    else:
        depth_note = ""
        if model.is_hierarchical and cfg.train_mode == "progressive":
            depth_note = f" depth={model.progressive_depth}"
        print(f"  [viz] {tag} step {step}{depth_note} -> {out_path}")

    model.train()
    return out_path
