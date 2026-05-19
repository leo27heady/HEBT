"""Labeled visualizations for S-HR-VQVAE training and inference."""
from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import colormaps

from model.vid.shr_vqvae import SHRVQVAEModel


_PAD = 6
_LEFT_LABEL_W = 220
_HEADER_H = 28


def _to_uint8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    arr = img_chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0).astype(np.uint8)


def _blank_tile(height: int, width: int) -> Image.Image:
    return Image.new("RGB", (width, height), color=(20, 20, 20))


def _indices_to_heatmap_tile(
    idx_hw: torch.Tensor, out_h: int, out_w: int, m: int
) -> Image.Image:
    cmap = colormaps["turbo"]
    norm = idx_hw.detach().float().cpu().numpy() / max(float(m - 1), 1.0)
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    im = Image.fromarray(rgb, mode="RGB")
    if im.size != (out_w, out_h):
        im = im.resize((out_w, out_h), resample=Image.Resampling.NEAREST)
    return im


def _draw_panel(
    rows: List[List[Image.Image]],
    row_labels: List[str],
    col_labels: List[str],
    out_path: str,
) -> None:
    n_rows = len(rows)
    n_cols = len(col_labels)
    tile_w, tile_h = rows[0][0].size
    width = _LEFT_LABEL_W + _PAD + n_cols * (tile_w + _PAD)
    height = _HEADER_H + _PAD + n_rows * (tile_h + _PAD)
    canvas = Image.new("RGB", (width, height), color=(10, 10, 10))
    draw = ImageDraw.Draw(canvas)

    for col, label in enumerate(col_labels):
        x = _LEFT_LABEL_W + _PAD + col * (tile_w + _PAD)
        draw.text((x + 4, 6), label, fill=(235, 235, 235))

    for row_i, (row_label, row_tiles) in enumerate(zip(row_labels, rows)):
        y = _HEADER_H + _PAD + row_i * (tile_h + _PAD)
        draw.text((8, y + max(2, tile_h // 2 - 8)), row_label, fill=(235, 235, 235))
        for col_i, tile in enumerate(row_tiles):
            x = _LEFT_LABEL_W + _PAD + col_i * (tile_w + _PAD)
            canvas.paste(tile, (x, y))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


@torch.no_grad()
def save_labeled_reconstruction_panel(
    model: SHRVQVAEModel,
    batch: torch.Tensor,
    step: int,
    log_dir: str,
    device: torch.device,
    tag: str = "recon_panel",
) -> None:
    """Save labeled input/reconstruction panel with frame indices."""
    model.eval()
    sample = batch[0:1].to(device)  # (1, T+S, C, H, W)
    _, ts, c, h, w = sample.shape
    frames = sample.reshape(ts, c, h, w)
    z = model.encode(frames)
    e_c_st, _, _ = model.quantize(z)
    recon = model.decode(e_c_st)

    in_tiles = [Image.fromarray(_to_uint8_hwc(frames[t])) for t in range(ts)]
    out_tiles = [Image.fromarray(_to_uint8_hwc(recon[t])) for t in range(ts)]
    col_labels = [f"t={t}" for t in range(ts)]

    out_dir = os.path.join(log_dir, "viz")
    out_path = os.path.join(out_dir, f"{tag}_step{step:06d}.png")
    _draw_panel(
        rows=[in_tiles, out_tiles],
        row_labels=["Input (GT)", "Reconstruction"],
        col_labels=col_labels,
        out_path=out_path,
    )
    model.train()


@torch.no_grad()
def save_labeled_prediction_panel(
    model: SHRVQVAEModel,
    batch: torch.Tensor,
    step: int,
    log_dir: str,
    device: torch.device,
    num_future: int,
    show_indices: bool = True,
    tag: str = "pred_panel",
) -> None:
    """Save labeled context/GT/prediction panel plus optional index heatmaps."""
    model.eval()
    cfg = model.cfg
    sample = batch[0:1].to(device)  # (1, T+S, C, H, W)
    _, ts, c, h, w = sample.shape
    t_ctx = cfg.T
    num_future = min(num_future, cfg.S)
    t_total = t_ctx + num_future

    context = sample[:, :t_ctx]
    gt_future = sample[:, t_ctx : t_ctx + num_future]

    pred_frames_out = model.generate(
        context, num_future=num_future, temperature=1.0, return_indices=show_indices
    )
    if show_indices:
        pred_future, pred_indices = pred_frames_out
    else:
        pred_future = pred_frames_out
        pred_indices = []

    context_frames = context[0]  # (T, C, H, W)
    gt_future_frames = gt_future[0]  # (S, C, H, W)
    pred_future_frames = pred_future[0]  # (S, C, H, W)

    blank = _blank_tile(h, w)
    rows: List[List[Image.Image]] = []
    row_labels: List[str] = []

    row_ctx = [Image.fromarray(_to_uint8_hwc(context_frames[t])) for t in range(t_ctx)]
    row_ctx += [blank.copy() for _ in range(num_future)]
    rows.append(row_ctx)
    row_labels.append("Context (given)")

    row_gt = [blank.copy() for _ in range(t_ctx)]
    row_gt += [Image.fromarray(_to_uint8_hwc(gt_future_frames[t])) for t in range(num_future)]
    rows.append(row_gt)
    row_labels.append("GT future")

    row_pred = [blank.copy() for _ in range(t_ctx)]
    row_pred += [Image.fromarray(_to_uint8_hwc(pred_future_frames[t])) for t in range(num_future)]
    rows.append(row_pred)
    row_labels.append("Predicted future")

    if show_indices:
        gt_indices = model.extract_indices(sample[:, :t_total])
        for layer_i, idx_layer in enumerate(gt_indices):
            row = []
            for t in range(t_ctx):
                row.append(_indices_to_heatmap_tile(idx_layer[0, t], h, w, cfg.M))
            for t in range(num_future):
                row.append(_indices_to_heatmap_tile(idx_layer[0, t_ctx + t], h, w, cfg.M))
            rows.append(row)
            row_labels.append(f"L{layer_i} index (GT)")

        for layer_i, pred_idx_layer in enumerate(pred_indices):
            row = [blank.copy() for _ in range(t_ctx)]
            for t in range(num_future):
                row.append(_indices_to_heatmap_tile(pred_idx_layer[0, t], h, w, cfg.M))
            rows.append(row)
            row_labels.append(f"L{layer_i} index (Pred)")

    col_labels = [f"t={t}" for t in range(t_ctx)] + [f"t={t_ctx + t}" for t in range(num_future)]
    out_dir = os.path.join(log_dir, "viz")
    out_path = os.path.join(out_dir, f"{tag}_step{step:06d}.png")
    _draw_panel(rows=rows, row_labels=row_labels, col_labels=col_labels, out_path=out_path)
    model.train()
