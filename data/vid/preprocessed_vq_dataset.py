"""
Dataset that loads precomputed CLIP features AND/OR precomputed VQ target
indices. Designed to feed `HierarchicalHVEBT.forward_loss(..., vq_targets=...)`.

Directory structure:

    features_dir/
        meta.json
        video/<idx>.pt           # (T, 3, H, W) float — needed for decoder loss
        s1/<idx>.pt              # (T, C, H, W) — optional, can be skipped
        s2/<idx>.pt
        s3/<idx>.pt

    vq_dir/
        vq_meta.json
        codebook_<stage>.pt
        targets_<stage>/<idx>.pt # (T, H, W) int16/int32

Returned dict (collated by default DataLoader):
    {
      "video":             (T, 3, H, W) float
      "<stage>":           (T, C, H, W) float       (only when load_features=True)
      "target_<stage>":    (T, H, W)    long
    }

Use with the trainer's `vq_no_features` flag for the CLIP-free fast path.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class PreprocessedCLIPVQDataset(Dataset):
    def __init__(
        self,
        features_dir: str,
        vq_dir: str,
        stages: Optional[List[str]] = None,
        load_features: bool = False,
        require_video: bool = True,
    ) -> None:
        self.features_dir = features_dir
        self.vq_dir = vq_dir
        self.load_features = load_features
        self.require_video = require_video

        # ---- features meta (for num_samples / available stages) ---------- #
        feat_meta_path = os.path.join(features_dir, "meta.json")
        if not os.path.isfile(feat_meta_path):
            raise FileNotFoundError(f"No meta.json in {features_dir}")
        with open(feat_meta_path) as f:
            self.feat_meta = json.load(f)
        self.num_samples = int(self.feat_meta["num_samples"])
        available_stages = self.feat_meta["stages"]

        # ---- vq meta ----------------------------------------------------- #
        vq_meta_path = os.path.join(vq_dir, "vq_meta.json")
        if not os.path.isfile(vq_meta_path):
            raise FileNotFoundError(f"No vq_meta.json in {vq_dir}")
        with open(vq_meta_path) as f:
            self.vq_meta = json.load(f)
        vq_stages = self.vq_meta["stages"]

        # Resolve requested stages
        if stages is None:
            stages = vq_stages
        for s in stages:
            if s not in vq_stages:
                raise ValueError(f"Stage '{s}' missing in {vq_dir} (have {vq_stages})")
            if load_features and s not in available_stages:
                raise ValueError(
                    f"Stage '{s}' missing CLIP features in {features_dir} (have {available_stages})"
                )
            tgt_dir = os.path.join(vq_dir, f"targets_{s}")
            if not os.path.isdir(tgt_dir):
                raise FileNotFoundError(
                    f"Missing target dir: {tgt_dir}. Re-run build_vq_codebook.py without --no_targets."
                )
        self.stages = stages

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        # Video (always loaded if available; decoder needs it)
        video_path = os.path.join(self.features_dir, "video", f"{idx}.pt")
        if os.path.isfile(video_path):
            out["video"] = torch.load(video_path, weights_only=True).float()
        elif self.require_video:
            raise FileNotFoundError(
                f"video file not found: {video_path} "
                f"(set require_video=False if you don't need decoder targets)"
            )

        for s in self.stages:
            t_path = os.path.join(self.vq_dir, f"targets_{s}", f"{idx}.pt")
            tgt = torch.load(t_path, weights_only=True).long()
            out[f"target_{s}"] = tgt
            if self.load_features:
                f_path = os.path.join(self.features_dir, s, f"{idx}.pt")
                feat = torch.load(f_path, weights_only=True).float()
                out[s] = feat
        return out
