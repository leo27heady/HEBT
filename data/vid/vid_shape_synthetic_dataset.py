import os
import json
import hashlib
import random
from enum import Enum

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from shapekit import Scene, SceneType, Random2DShapeCreator, Random3DShapeCreator


class TemporalPattern(Enum):
    ACCELERATION = "acceleration"
    DECELERATION = "deceleration"
    OSCILLATION = "oscillation"
    INTERRUPTION = "interruption"


class VIDShapeSyntheticDataset(Dataset):
    """
    Pre-rendered rotating 2D/3D shape sequences for the EBT video pipeline.
    On first use (or when config changes) renders all samples to disk as .npy uint8 files.
    Subsequent runs load from cache, enabling multi-worker DataLoader.
    Returns tensors of shape (context_length, 3, H, W).
    """

    SCALE_FACTOR = 1000

    def __init__(self, hparams, size=10000, cache=True):
        self.cache = cache
        self.context_length = hparams.context_length
        self.image_dims = hparams.image_dims

        # shape config
        self.scene_type = SceneType[hparams.shape_scene_type]
        self.min_cubes = getattr(hparams, "shape_min_cubes", 2)
        self.max_cubes = getattr(hparams, "shape_max_cubes", 6)
        self.angle_min = getattr(hparams, "shape_angle_min", 5)
        self.angle_max = getattr(hparams, "shape_angle_max", 20)

        pattern_names = getattr(hparams, "shape_temporal_patterns", [])
        self.temporal_patterns = [TemporalPattern(p) for p in pattern_names]
        self.pattern_combining = getattr(hparams, "shape_pattern_combining", False)
        self.accel_min = getattr(hparams, "shape_accel_min", 3)
        self.accel_max = getattr(hparams, "shape_accel_max", 6)
        self.oscillation_period_min = getattr(hparams, "shape_oscillation_period_min", 1)
        self.oscillation_period_max = getattr(hparams, "shape_oscillation_period_max", 4)
        self.interruption_period_min = getattr(hparams, "shape_interruption_period_min", 1)
        self.interruption_period_max = getattr(hparams, "shape_interruption_period_max", 4)

        self.render_size = max(self.image_dims)

        # Normalization (optional ImageNet normalization, or plain [0,1] for VQ-VAE)
        no_norm = getattr(hparams, "shape_no_imagenet_norm", False)
        if no_norm:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_dims[0], self.image_dims[1])),
                transforms.ToTensor(),  # [0,1] range only
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_dims[0], self.image_dims[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Cache directory
        cache_root = getattr(hparams, "shape_cache_dir", "data/vid/shape_cache")
        config_hash = self._config_hash(size)
        self.cache_dir = os.path.join(cache_root, config_hash)
        self.size = size

        if not self.cache:
            self._samples = self._generate_in_memory()
        elif self._cache_valid():
            print(f"[VIDShapeSyntheticDataset] Using cached dataset at {self.cache_dir} ({self.size} samples)")
        else:
            self._generate_and_save()

    def _config_hash(self, size):
        """Deterministic hash of all config that affects the generated data."""
        cfg = dict(
            size=size,
            context_length=self.context_length,
            image_dims=self.image_dims,
            scene_type=self.scene_type.name,
            min_cubes=self.min_cubes,
            max_cubes=self.max_cubes,
            angle_min=self.angle_min,
            angle_max=self.angle_max,
            patterns=[p.value for p in self.temporal_patterns],
            pattern_combining=self.pattern_combining,
            accel_min=self.accel_min,
            accel_max=self.accel_max,
            osc_min=self.oscillation_period_min,
            osc_max=self.oscillation_period_max,
            int_min=self.interruption_period_min,
            int_max=self.interruption_period_max,
            render_size=self.render_size,
        )
        raw = json.dumps(cfg, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _cache_valid(self):
        meta_path = os.path.join(self.cache_dir, "meta.json")
        if not os.path.isfile(meta_path):
            return False
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta.get("size") == self.size and meta.get("done", False)

    def _generate_and_save(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"[VIDShapeSyntheticDataset] Generating {self.size} samples to {self.cache_dir} ...")

        is_2d = self.scene_type == SceneType.DIM_2
        creator_2d = Random2DShapeCreator()
        creator_3d = None
        if not is_2d:
            creator_3d = Random3DShapeCreator(self.max_cubes, include_reflections=False)

        # Create initial figure + scene once, then reuse via prepare_scene
        if is_2d:
            init_fig = creator_2d.create_equilateral_triangle()
        else:
            init_fig, _ = creator_3d.create_connected_cubes(self.min_cubes)

        scene = Scene(
            init_fig, self.scene_type, self.render_size,
            bg_color="white",
            mesh_color="black" if is_2d else "gray",
            show_edges=False,
            lighting=not is_2d,
            line_width=4.0,
            distance_factor=1.0 if is_2d else 2.5,
            fixed_camera_distance=2.7 if is_2d else None,
            axis="z",
        )
        scene.plotter.render()

        import time
        t0 = time.time()

        for idx in range(self.size):
            frames = self._render_with_scene(scene, creator_2d, creator_3d, is_2d)
            np.save(os.path.join(self.cache_dir, f"{idx}.npy"), frames)

            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                remaining = (self.size - idx - 1) / rate
                print(f"  [{idx + 1}/{self.size}] {rate:.1f} samples/s, ~{remaining:.0f}s remaining")

        scene.plotter.close()

        # Write metadata
        with open(os.path.join(self.cache_dir, "meta.json"), "w") as f:
            json.dump({"size": self.size, "done": True}, f)

        elapsed = time.time() - t0
        print(f"[VIDShapeSyntheticDataset] Done in {elapsed:.1f}s ({self.size / elapsed:.1f} samples/s)")

    def __len__(self):
        return self.size

    def _generate_in_memory(self):
        """Generate all samples into RAM (no disk I/O). Used when cache=False."""
        is_2d = self.scene_type == SceneType.DIM_2
        creator_2d = Random2DShapeCreator()
        creator_3d = None if is_2d else Random3DShapeCreator(self.max_cubes, include_reflections=False)

        if is_2d:
            init_fig = creator_2d.create_equilateral_triangle()
        else:
            init_fig, _ = creator_3d.create_connected_cubes(self.min_cubes)

        scene = Scene(
            init_fig, self.scene_type, self.render_size,
            bg_color="white",
            mesh_color="black" if is_2d else "gray",
            show_edges=False,
            lighting=not is_2d,
            line_width=4.0,
            distance_factor=1.0 if is_2d else 2.5,
            fixed_camera_distance=2.7 if is_2d else None,
            axis="z",
        )
        scene.plotter.render()

        samples = [self._render_with_scene(scene, creator_2d, creator_3d, is_2d)
                    for _ in range(self.size)]
        scene.plotter.close()
        return samples

    def _render_with_scene(self, scene, creator_2d, creator_3d, is_2d):
        """Render a single sample using an existing scene (shared between cache and in-memory paths).

        Returns (context_length, render_size, render_size, 3) uint8 numpy array.
        """
        if is_2d:
            base = random.randint(self._scale(0.5), self._scale(1)) / self.SCALE_FACTOR
            shift = random.randint(self._scale(-0.2), self._scale(base + 0.2)) / self.SCALE_FACTOR
            height = random.randint(self._scale(0.5), self._scale(1)) / self.SCALE_FACTOR
            figure = creator_2d.create_triangle(base, shift, height)
        else:
            num_blocks = random.randint(self.min_cubes, self.max_cubes)
            figure, _ = creator_3d.create_connected_cubes(num_blocks)

        scene.prepare_scene(
            figure,
            bg_color="white",
            mesh_color="black" if is_2d else "gray",
            lighting=not is_2d,
            show_edges=False,
            line_width=4.0,
            distance_factor=1.0 if is_2d else 2.5,
            fixed_camera_distance=2.7 if is_2d else None,
            axis="z",
        )
        scene.plotter.render()

        step = np.array([self._angle_gen(), self._angle_gen(), self._angle_gen()])
        if is_2d:
            step[0] = 0.0
            step[1] = 0.0

        selected_patterns = self._select_patterns()
        acceleration = 1.0
        oscillation_period = 0
        interruption_period = 0
        step_swap = np.array([0.0, 0.0, 0.0])

        if TemporalPattern.OSCILLATION in selected_patterns:
            oscillation_period = random.randint(self.oscillation_period_min, self.oscillation_period_max)
        elif TemporalPattern.INTERRUPTION in selected_patterns:
            interruption_period = random.randint(self.interruption_period_min, self.interruption_period_max)

        if TemporalPattern.ACCELERATION in selected_patterns:
            step /= 1.5
            acceleration = 1.0 + random.randint(
                self._scale(self.accel_min / 2), self._scale(self.accel_max / 2)
            ) / (self.SCALE_FACTOR * 100)
        elif TemporalPattern.DECELERATION in selected_patterns:
            step *= 1.5
            dec = 1.0 + random.randint(
                self._scale(self.accel_min * 2), self._scale(self.accel_max * 2)
            ) / (self.SCALE_FACTOR * 100)
            acceleration = 1.0 / dec

        frames = np.empty((self.context_length, self.render_size, self.render_size, 3), dtype=np.uint8)
        for i in range(self.context_length):
            if not is_2d:
                figure.rotate_x(step[0], point=scene.center_of_mass, inplace=True)
                figure.rotate_y(step[1], point=scene.center_of_mass, inplace=True)
            figure.rotate_z(step[2], point=scene.center_of_mass, inplace=True)
            scene.plotter.render()
            frames[i] = np.array(scene.plotter.screenshot())

            if TemporalPattern.OSCILLATION in selected_patterns and oscillation_period > 0:
                if (i + 1) % oscillation_period == 0:
                    step = step * -1.0
            elif TemporalPattern.INTERRUPTION in selected_patterns and interruption_period > 0:
                if (i + 1) % interruption_period == 0:
                    step, step_swap = step_swap, step

            if TemporalPattern.ACCELERATION in selected_patterns or TemporalPattern.DECELERATION in selected_patterns:
                step = step * acceleration

        return frames

    def __getitem__(self, idx):
        frames_uint8 = self._samples[idx] if not self.cache else \
            np.load(os.path.join(self.cache_dir, f"{idx}.npy"))  # (T, H, W, 3)
        frame_tensors = []
        for i in range(frames_uint8.shape[0]):
            pil_image = Image.fromarray(frames_uint8[i])
            frame_tensors.append(self.transform(pil_image))
        return torch.stack(frame_tensors)  # (context_length, 3, H, W)

    def _scale(self, v):
        return int(v * self.SCALE_FACTOR)

    def _angle_gen(self):
        return random.choice((-1, 1)) * random.randint(
            self._scale(self.angle_min), self._scale(self.angle_max)
        ) / self.SCALE_FACTOR

    def _select_patterns(self):
        if not self.temporal_patterns:
            return []
        selected = self.temporal_patterns.copy()
        if TemporalPattern.ACCELERATION in selected and TemporalPattern.DECELERATION in selected:
            selected.remove(random.choice((TemporalPattern.ACCELERATION, TemporalPattern.DECELERATION)))
        if TemporalPattern.OSCILLATION in selected and TemporalPattern.INTERRUPTION in selected:
            selected.remove(random.choice((TemporalPattern.OSCILLATION, TemporalPattern.INTERRUPTION)))
        if selected:
            k = random.randint(1, len(selected) if self.pattern_combining else 1)
            selected = list(np.random.choice(selected, size=k, replace=False))
        return selected
