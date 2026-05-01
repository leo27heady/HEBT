"""
Preprocess a video dataset through frozen MobileCLIP2-S0 and save per-stage
features to disk. This avoids running CLIP during training, saving GPU time.

Input: either a directory of .npy files (from generate_dataset.py) or a
       recognized dataset name (e.g. 'taxibj', 'kth', 'mmnist').

Output: for each sample idx, saves:
    {output_dir}/s1/{idx}.pt   -> (T, 128, 32, 32) float16
    {output_dir}/s2/{idx}.pt   -> (T, 256, 16, 16) float16
    {output_dir}/s3/{idx}.pt   -> (T, 512,  8,  8) float16
    {output_dir}/meta.json     -> metadata (num_samples, stages, frame_size, etc.)

Usage:
    # From synthetic .npy dataset
    python scripts/preprocess_clip_features.py \\
        --input_dir data/vid/shape_cache/2d_1k/<hash> \\
        --output_dir data/vid/clip_features/2d_1k \\
        --image_size 256 --batch_size 16

    # Specify which stages to extract
    python scripts/preprocess_clip_features.py \\
        --input_dir data/vid/shape_cache/3d_1k/<hash> \\
        --output_dir data/vid/clip_features/3d_1k \\
        --stages s1 s2 s3 --image_size 256
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vid.hvebt.clip_encoder import MobileClipMultiStageEncoder


def load_npy_sample(path: str, image_size: int) -> torch.Tensor:
    """Load a single .npy sample (T, H, W, 3 uint8) and return (T, 3, H, W) in [0,1]."""
    frames_uint8 = np.load(path)  # (T, H, W, 3)
    T = frames_uint8.shape[0]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # -> [0, 1]
    ])
    tensors = []
    for i in range(T):
        pil = Image.fromarray(frames_uint8[i])
        tensors.append(transform(pil))
    return torch.stack(tensors)  # (T, 3, H, W)


def find_npy_files(input_dir: str) -> list:
    """Find all .npy files sorted numerically."""
    files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    # Sort numerically (0.npy, 1.npy, ..., 999.npy)
    files.sort(key=lambda x: int(x.replace(".npy", "")))
    return [os.path.join(input_dir, f) for f in files]


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset through CLIP")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing .npy files (from generate_dataset.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for preprocessed features")
    parser.add_argument("--stages", nargs="+", default=["s1", "s2", "s3"],
                        help="Which CLIP stages to extract")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Resize images to this before CLIP")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for CLIP processing (in samples)")
    parser.add_argument("--weights_path", type=str,
                        default="clip/MobileCLIP2-S0/mobileclip2_s0.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Save features in float16 to save disk space")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== CLIP Feature Preprocessing ===")
    print(f"  Input: {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Stages: {args.stages}")
    print(f"  Device: {device}")
    print()

    # Find input files
    npy_files = find_npy_files(args.input_dir)
    if not npy_files:
        print(f"ERROR: No .npy files found in {args.input_dir}")
        sys.exit(1)
    print(f"  Found {len(npy_files)} samples")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    for stage in args.stages:
        os.makedirs(os.path.join(args.output_dir, stage), exist_ok=True)

    # Load CLIP encoder
    print(f"  Loading MobileCLIP2-S0 from {args.weights_path}...")
    encoder = MobileClipMultiStageEncoder(
        weights_path=args.weights_path,
        return_stages=tuple(args.stages),
    ).to(device)
    print(f"  Encoder loaded.")

    # Process in batches
    t0 = time.time()
    num_samples = len(npy_files)
    batch_size = args.batch_size

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_files = npy_files[batch_start:batch_end]

        # Load batch: list of (T, 3, H, W) tensors
        batch_tensors = []
        for f in batch_files:
            sample = load_npy_sample(f, args.image_size)
            batch_tensors.append(sample)

        # Stack into (B, T, 3, H, W) - all samples must have same T
        video_batch = torch.stack(batch_tensors).to(device)  # (B, T, 3, H, W)

        # Process through CLIP
        with torch.no_grad():
            feats_dict = encoder.encode_video(video_batch)

        # Save per-sample per-stage
        for i, idx in enumerate(range(batch_start, batch_end)):
            for stage in args.stages:
                feat = feats_dict[stage][i]  # (T, C, H, W)
                if args.fp16:
                    feat = feat.half()
                save_path = os.path.join(args.output_dir, stage, f"{idx}.pt")
                torch.save(feat.cpu(), save_path)

        # Progress
        elapsed = time.time() - t0
        rate = batch_end / elapsed
        remaining = (num_samples - batch_end) / max(rate, 1e-6)
        if (batch_end) % max(1, batch_size * 5) == 0 or batch_end == num_samples:
            print(f"  [{batch_end}/{num_samples}] {rate:.1f} samples/s, ~{remaining:.0f}s remaining")

    # Save metadata
    meta = {
        "num_samples": num_samples,
        "stages": args.stages,
        "image_size": args.image_size,
        "fp16": args.fp16,
        "input_dir": args.input_dir,
    }
    # Get shape info from first sample
    for stage in args.stages:
        sample = torch.load(os.path.join(args.output_dir, stage, "0.pt"), weights_only=True)
        meta[f"{stage}_shape"] = list(sample.shape)

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone! Features saved to: {args.output_dir}")
    print(f"  Time: {elapsed:.1f}s ({num_samples / elapsed:.1f} samples/s)")
    for stage in args.stages:
        print(f"  {stage}: shape {meta[f'{stage}_shape']}")


if __name__ == "__main__":
    main()
