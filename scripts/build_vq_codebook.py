"""
Build per-stage VQ codebooks (and optionally precomputed target indices) from
preprocessed CLIP features.

Pipeline:
  1) Read preprocessed per-stage CLIP features from disk (output of
     `scripts/preprocess_clip_features.py`).
  2) Subsample spatial tokens per stage and fit MiniBatchKMeans → codebook
     centroids (K, C).
  3) Optional K sweep with elbow logging.
  4) Optional precompute of hard target indices for every sample, every stage:
     targets are nearest-codebook-entry per spatial token.

Outputs (under --out_dir):
    codebook_<stage>.pt          (K, C) float32
    targets_<stage>/<idx>.pt     (T, H, W) int16/int32
    vq_meta.json                 metadata (K per stage, sklearn version, ...)

Typical usage:
    # Sweep K to choose elbow per stage, no targets yet
    python scripts/build_vq_codebook.py \
        --features_dir data/vid/clip_features/2d_1k \
        --out_dir       data/vid/vq/2d_1k \
        --stages s1 s2 s3 \
        --sweep "64,128,256,512,1024,2048" \
        --no_targets

    # Build final codebook + precomputed targets
    python scripts/build_vq_codebook.py \
        --features_dir data/vid/clip_features/2d_1k \
        --out_dir       data/vid/vq/2d_1k \
        --stages s1 s2 s3 \
        --K 1024 512 256

After this script you can train CLIP-free with:
    python example_code/hvebt_hierarchical_training_loop.py \
        --vq_mode --no_features \
        --preprocessed_dir data/vid/clip_features/2d_1k \
        --vq_dir            data/vid/vq/2d_1k \
        --stages s1 s2 s3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_feature_files(features_dir: str, stage: str) -> List[str]:
    stage_dir = os.path.join(features_dir, stage)
    if not os.path.isdir(stage_dir):
        raise FileNotFoundError(f"Missing stage dir: {stage_dir}")
    files = [f for f in os.listdir(stage_dir) if f.endswith(".pt")]
    files.sort(key=lambda x: int(x.replace(".pt", "")))
    return [os.path.join(stage_dir, f) for f in files]


def _stream_token_sample(
    files: List[str],
    max_tokens: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stream-load files, randomly subsample tokens to bound memory.
    Returns float32 array (N <= max_tokens, C).
    """
    pieces: List[np.ndarray] = []
    total = 0
    for f in files:
        feat = torch.load(f, weights_only=True)  # (T, C, H, W)
        if feat.ndim == 3:  # (T, C) pooled
            feat = feat.unsqueeze(-1).unsqueeze(-1)
        T, C, H, W = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (T*H*W, C)
        flat_np = flat.float().cpu().numpy()
        pieces.append(flat_np)
        total += flat_np.shape[0]
        if total >= max_tokens * 4:  # gather more than needed, then subsample once
            break
    big = np.concatenate(pieces, axis=0)
    if big.shape[0] > max_tokens:
        idx = rng.choice(big.shape[0], size=max_tokens, replace=False)
        big = big[idx]
    return big.astype(np.float32)


def _fit_kmeans(features: np.ndarray, K: int, seed: int, batch: int = 4096):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(
        n_clusters=K,
        batch_size=batch,
        n_init=3,
        max_iter=200,
        random_state=seed,
        verbose=0,
    )
    km.fit(features)
    return km


def _sweep_k(features: np.ndarray, K_list: List[int], seed: int) -> Dict[int, float]:
    inertias: Dict[int, float] = {}
    for K in K_list:
        t0 = time.time()
        km = _fit_kmeans(features, K, seed=seed)
        inertias[K] = float(km.inertia_)
        print(f"    K={K:5d}  inertia={inertias[K]:.4e}  ({time.time()-t0:.1f}s)")
    return inertias


def _precompute_targets(
    files: List[str],
    centroids: torch.Tensor,         # (K, C) float32
    out_dir: str,
    device: torch.device,
    chunk: int = 65536,
):
    """
    For every sample/file, compute nearest-codebook index per spatial token.
    Saves (T, H, W) int16 (or int32 if K > 32767) into out_dir/<idx>.pt.
    """
    os.makedirs(out_dir, exist_ok=True)
    K, C = centroids.shape
    centroids_dev = centroids.to(device)
    use_int16 = K <= 32767
    dtype_save = torch.int16 if use_int16 else torch.int32

    n = len(files)
    t0 = time.time()
    for i, f in enumerate(files):
        feat = torch.load(f, weights_only=True)  # (T, C, H, W) or (T, C)
        if feat.ndim == 3:
            feat = feat.unsqueeze(-1).unsqueeze(-1)
        T, C_f, H, W = feat.shape
        if C_f != C:
            raise ValueError(f"Codebook C={C} != feature C={C_f} for {f}")
        flat = feat.permute(0, 2, 3, 1).reshape(-1, C).float().to(device)
        # Chunked argmin to bound memory
        out_idx = torch.empty(flat.shape[0], dtype=torch.long, device=device)
        for s in range(0, flat.shape[0], chunk):
            e = min(s + chunk, flat.shape[0])
            d = torch.cdist(flat[s:e], centroids_dev)  # (chunk, K)
            out_idx[s:e] = d.argmin(dim=-1)
        targets = out_idx.reshape(T, H, W).to(dtype_save).cpu()
        idx_str = os.path.basename(f).replace(".pt", "")
        torch.save(targets, os.path.join(out_dir, f"{idx_str}.pt"))
        if (i + 1) % 50 == 0 or (i + 1) == n:
            rate = (i + 1) / (time.time() - t0 + 1e-9)
            print(f"    targets [{i+1}/{n}]  {rate:.1f} samples/s")


def main():
    parser = argparse.ArgumentParser(description="Build VQ codebooks (and targets) from preprocessed CLIP features")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Directory produced by scripts/preprocess_clip_features.py")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Where to write codebooks and target indices")
    parser.add_argument("--stages", nargs="+", default=["s1", "s2", "s3"],
                        help="Stages to build codebooks for")
    parser.add_argument("--K", type=int, nargs="+", default=None,
                        help="Codebook size per stage (must match --stages length). "
                             "Mutually exclusive with --sweep.")
    parser.add_argument("--sweep", type=str, default="",
                        help="Comma-separated K values to sweep (e.g. '64,128,256,512,1024'). "
                             "Logs inertia and exits without saving codebooks.")
    parser.add_argument("--max_tokens", type=int, default=2_000_000,
                        help="Max number of feature vectors used for KMeans fitting per stage. "
                             "Subsampled uniformly across files.")
    parser.add_argument("--no_targets", action="store_true",
                        help="Skip target precomputation (only save codebooks)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for target precomputation (KMeans always on CPU/sklearn)")
    args = parser.parse_args()

    if args.sweep and args.K:
        raise SystemExit("--K and --sweep are mutually exclusive")
    if args.K and len(args.K) != len(args.stages):
        raise SystemExit(f"--K must have {len(args.stages)} values to match --stages")

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    sweep_log: Dict[str, Dict[int, float]] = {}
    K_per_stage: Dict[str, int] = {}
    centroid_meta: Dict[str, Dict] = {}

    print(f"=== VQ Codebook Build ===")
    print(f"  Features: {args.features_dir}")
    print(f"  Out:      {args.out_dir}")
    print(f"  Stages:   {args.stages}")
    print(f"  Device:   {device}  (sklearn KMeans always uses CPU)")
    print()

    for si, stage in enumerate(args.stages):
        print(f"-- Stage {stage} --")
        files = _load_feature_files(args.features_dir, stage)
        print(f"   {len(files)} feature files found")

        print(f"   sampling up to {args.max_tokens} tokens...")
        sample = _stream_token_sample(files, args.max_tokens, rng)
        print(f"   sample shape: {sample.shape} dtype={sample.dtype}")

        if args.sweep:
            K_list = [int(x) for x in args.sweep.split(",") if x.strip()]
            print(f"   sweeping K: {K_list}")
            inertias = _sweep_k(sample, K_list, seed=args.seed)
            sweep_log[stage] = inertias
            continue

        K = args.K[si]
        print(f"   fitting MiniBatchKMeans K={K}...")
        t0 = time.time()
        km = _fit_kmeans(sample, K, seed=args.seed)
        print(f"   inertia={km.inertia_:.4e}  ({time.time()-t0:.1f}s)")

        centroids = torch.from_numpy(km.cluster_centers_.astype(np.float32))  # (K, C)
        cb_path = os.path.join(args.out_dir, f"codebook_{stage}.pt")
        torch.save(centroids, cb_path)
        print(f"   wrote {cb_path}  shape={tuple(centroids.shape)}")
        K_per_stage[stage] = K
        centroid_meta[stage] = {
            "K": K,
            "C": int(centroids.shape[1]),
            "inertia": float(km.inertia_),
        }

        if not args.no_targets:
            tgt_dir = os.path.join(args.out_dir, f"targets_{stage}")
            print(f"   precomputing targets → {tgt_dir}")
            _precompute_targets(files, centroids, tgt_dir, device=device)

    if args.sweep:
        sweep_path = os.path.join(args.out_dir, "vq_sweep.json")
        with open(sweep_path, "w") as f:
            json.dump(sweep_log, f, indent=2)
        print(f"\nSweep results → {sweep_path}")
        print("Pick K at the elbow per stage, then re-run without --sweep with --K K1 K2 K3")
        return

    # Source feature meta (to record num_samples / shapes)
    src_meta_path = os.path.join(args.features_dir, "meta.json")
    src_meta: Dict = {}
    if os.path.isfile(src_meta_path):
        with open(src_meta_path, "r") as f:
            src_meta = json.load(f)

    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = "unknown"

    meta = {
        "stages": args.stages,
        "K_per_stage": K_per_stage,
        "centroid_meta": centroid_meta,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "sklearn_version": sklearn_version,
        "no_targets": args.no_targets,
        "features_dir": args.features_dir,
        "num_samples": src_meta.get("num_samples"),
        "image_size": src_meta.get("image_size"),
    }
    with open(os.path.join(args.out_dir, "vq_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
