"""
Dataset that loads precomputed CLIP features from disk.
Works with output from scripts/preprocess_clip_features.py.

Returns a dict of {stage_name: (T, C, H, W)} tensors per sample.
"""
import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class PreprocessedCLIPDataset(Dataset):
    """
    Loads precomputed per-stage CLIP features.

    Directory structure expected:
        features_dir/
            meta.json
            s1/0.pt, s1/1.pt, ...
            s2/0.pt, s2/1.pt, ...
            s3/0.pt, s3/1.pt, ...
    """

    def __init__(self, features_dir: str, stages: Optional[List[str]] = None):
        self.features_dir = features_dir
        meta_path = os.path.join(features_dir, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"No meta.json found in {features_dir}")

        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.num_samples = self.meta["num_samples"]
        self.available_stages = self.meta["stages"]

        if stages is not None:
            for s in stages:
                if s not in self.available_stages:
                    raise ValueError(f"Stage '{s}' not available. Have: {self.available_stages}")
            self.stages = stages
        else:
            self.stages = self.available_stages

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = {}
        for stage in self.stages:
            path = os.path.join(self.features_dir, stage, f"{idx}.pt")
            feat = torch.load(path, weights_only=True)
            result[stage] = feat.float()  # convert from fp16 if needed
        # Load video frames (T, 3, H, W) in [0,1]
        video_path = os.path.join(self.features_dir, "video", f"{idx}.pt")
        if os.path.isfile(video_path):
            result["video"] = torch.load(video_path, weights_only=True).float()
        return result
