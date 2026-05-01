"""
Standalone script for generating synthetic video datasets (2D shapes / 3D polycubes)
without being tied to any training pipeline.

Usage:
    # Generate 2D dataset (1000 samples, 8 frames each, 256x256)
    python scripts/generate_dataset.py --type 2d --size 1000 --context_length 8 --image_size 256 --output_dir data/vid/shape_cache/2d_1k

    # Generate 3D polycubes
    python scripts/generate_dataset.py --type 3d --size 1000 --context_length 8 --image_size 256 --output_dir data/vid/shape_cache/3d_1k

    # With temporal patterns (acceleration, oscillation, etc.)
    python scripts/generate_dataset.py --type 3d --size 500 --patterns acceleration oscillation --output_dir data/vid/shape_cache/3d_patterns
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic video datasets")
    parser.add_argument("--type", choices=["2d", "3d"], default="2d", help="2D triangles or 3D polycubes")
    parser.add_argument("--size", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--context_length", type=int, default=8, help="Number of frames per sample")
    parser.add_argument("--image_size", type=int, default=256, help="Image resolution (square)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides default cache)")
    # Shape params
    parser.add_argument("--min_cubes", type=int, default=2)
    parser.add_argument("--max_cubes", type=int, default=6)
    parser.add_argument("--angle_min", type=int, default=5)
    parser.add_argument("--angle_max", type=int, default=20)
    # Temporal patterns
    parser.add_argument("--patterns", nargs="*", default=[],
                        choices=["acceleration", "deceleration", "oscillation", "interruption"])
    parser.add_argument("--pattern_combining", action="store_true")
    args = parser.parse_args()

    scene_type = "DIM_2" if args.type == "2d" else "DIM_3"

    # Set output dir: either user-specified or auto-generated
    if args.output_dir:
        cache_dir = args.output_dir
    else:
        cache_dir = f"data/vid/shape_cache/{args.type}_{args.size}"

    hparams = SimpleNamespace(
        context_length=args.context_length,
        image_dims=[args.image_size, args.image_size],
        shape_scene_type=scene_type,
        shape_min_cubes=args.min_cubes,
        shape_max_cubes=args.max_cubes,
        shape_angle_min=args.angle_min,
        shape_angle_max=args.angle_max,
        shape_temporal_patterns=args.patterns,
        shape_pattern_combining=args.pattern_combining,
        shape_accel_min=3,
        shape_accel_max=6,
        shape_oscillation_period_min=1,
        shape_oscillation_period_max=4,
        shape_interruption_period_min=1,
        shape_interruption_period_max=4,
        shape_cache_dir=cache_dir,
    )

    print(f"=== Generating {args.type.upper()} dataset ===")
    print(f"  Samples: {args.size}")
    print(f"  Frames/sample: {args.context_length}")
    print(f"  Resolution: {args.image_size}x{args.image_size}")
    print(f"  Patterns: {args.patterns or 'none (constant rotation)'}")
    print(f"  Output: {cache_dir}")
    print()

    # Force generation by using the parent as cache_dir directly
    # The dataset class uses a hash-based subdir, so we override it
    dataset = VIDShapeSyntheticDataset(hparams, size=args.size)

    print(f"\nDone! Dataset stored at: {dataset.cache_dir}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Sample shape: {dataset[0].shape}")


if __name__ == "__main__":
    main()
