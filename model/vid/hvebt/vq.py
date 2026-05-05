"""
VQ codebook module shared by all VQ-mode HVEBT stages.

The module owns a single (K, C) weight tensor that serves three roles:
  1. Lookup of "real" features by codebook index   → context embeddings.
  2. Decode of predicted logits via softmax @ W    → continuous features
     for the energy function (the convex hull of the codebook).
  3. Quantize raw features by nearest-neighbor    → CE targets.

Maintenance ops (dead-code reset, similarity merging, EMA updates) are exposed
as no-op-friendly methods so that calling code can enable them via config flags
without restructuring.

This realizes the "shared parameter for context-embed and prediction-decode"
design: there is exactly ONE matrix; at MCMC convergence (softmax → one-hot),
the predicted features equal `lookup(target)` by construction.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQModule(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        dim: int,
        ema_decay: float = 0.0,
        track_usage: bool = True,
    ) -> None:
        super().__init__()
        if codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {codebook_size}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.K = codebook_size
        self.C = dim
        self.ema_decay = float(ema_decay)
        self.track_usage = bool(track_usage)
        self._frozen = ema_decay <= 0.0  # frozen ↔ no EMA codebook update

        # When EMA is enabled, codebook is a buffer (updated only via EMA).
        # Otherwise it is a buffer too (frozen) — gradients NEVER flow into the
        # codebook weights directly; this matches the V2 plan (Part B.1).
        self.register_buffer("weight", torch.empty(codebook_size, dim))
        nn.init.normal_(self.weight, std=0.02)

        if self.track_usage:
            self.register_buffer("_usage_ema", torch.full((codebook_size,), 1.0 / codebook_size))
        if not self._frozen:
            self.register_buffer("_ema_count", torch.zeros(codebook_size))
            self.register_buffer("_ema_sum",   torch.zeros(codebook_size, dim))

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def load_from_kmeans(self, path: str) -> None:
        """Load codebook centroids from a .pt file produced by build_vq_codebook.py."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"VQ codebook file not found: {path}")
        cb = torch.load(path, map_location="cpu", weights_only=True).float()
        if cb.shape != (self.K, self.C):
            raise ValueError(
                f"VQ codebook shape mismatch: file {tuple(cb.shape)} vs expected ({self.K}, {self.C})"
            )
        self.weight.copy_(cb.to(self.weight.device, self.weight.dtype))
        if not self._frozen:
            # Seed EMA accumulators consistent with current weight.
            self._ema_count.fill_(1.0)
            self._ema_sum.copy_(self.weight)

    # ------------------------------------------------------------------ #
    # Core ops
    # ------------------------------------------------------------------ #

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Embed code indices into continuous features.

        Args:
            indices: long tensor of any shape (..,) with values in [0, K).
        Returns:
            features: same leading shape with trailing dim C.
        """
        if indices.dtype not in (torch.long, torch.int32, torch.int16, torch.int64):
            raise TypeError(f"indices must be integer dtype, got {indices.dtype}")
        idx = indices.long()
        return F.embedding(idx, self.weight)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits → continuous features via softmax @ codebook.

        Args:
            logits: (B, N, K) raw logits per spatial token.
        Returns:
            features: (B, N, C) continuous features (live grad through softmax).
        """
        if logits.shape[-1] != self.K:
            raise ValueError(f"logits last dim {logits.shape[-1]} != K={self.K}")
        probs = F.softmax(logits, dim=-1)
        return torch.matmul(probs, self.weight)

    def quantize(self, features: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
        """
        Nearest-codebook-index per token.

        Args:
            features: (..., C)  float
        Returns:
            indices: (...,) long
        """
        if features.shape[-1] != self.C:
            raise ValueError(f"features last dim {features.shape[-1]} != C={self.C}")
        flat = features.reshape(-1, self.C)
        out = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
        for s in range(0, flat.shape[0], chunk):
            e = min(s + chunk, flat.shape[0])
            d = torch.cdist(flat[s:e].float(), self.weight.float())
            out[s:e] = d.argmin(dim=-1)
        return out.reshape(features.shape[:-1])

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Shannon entropy H = -sum p log p of the predicted distribution.
        Returns (B, N) per-token entropy.
        """
        if logits.shape[-1] != self.K:
            raise ValueError(f"logits last dim {logits.shape[-1]} != K={self.K}")
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)

    # ------------------------------------------------------------------ #
    # Maintenance (Phase 5; safe to call when track_usage=True / ema_decay>0)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def update_usage(self, indices: torch.Tensor, decay: float = 0.99) -> None:
        """EMA of selection frequency (per-step)."""
        if not self.track_usage:
            return
        idx = indices.reshape(-1).long()
        K = self.K
        onehot = F.one_hot(idx, num_classes=K).float().mean(dim=0)  # (K,)
        self._usage_ema.mul_(decay).add_(onehot, alpha=1.0 - decay)

    @torch.no_grad()
    def update_ema(self, indices: torch.Tensor, features: torch.Tensor) -> None:
        """
        VQ-VAE-style EMA update of the codebook weights.

        Args:
            indices:  (..,) long  (assignments produced by `quantize`)
            features: (.., C) float — the encoder-side features that were assigned.
        """
        if self._frozen:
            return
        idx = indices.reshape(-1).long()
        feats = features.reshape(-1, self.C).to(self.weight.dtype)
        d = self.ema_decay
        K = self.K

        onehot = F.one_hot(idx, num_classes=K).to(self.weight.dtype)  # (N, K)
        count_new = onehot.sum(dim=0)                                 # (K,)
        sum_new = onehot.t() @ feats                                  # (K, C)
        self._ema_count.mul_(d).add_(count_new, alpha=1.0 - d)
        self._ema_sum.mul_(d).add_(sum_new, alpha=1.0 - d)

        n = self._ema_count.unsqueeze(1).clamp(min=1e-5)
        self.weight.copy_(self._ema_sum / n)

    @torch.no_grad()
    def reset_dead_codes(
        self,
        candidate_features: torch.Tensor,
        usage_threshold: float,
    ) -> int:
        """
        Reinitialize codes whose usage EMA is below threshold from random
        encoder features in the current batch.

        Returns number of codes reset.
        """
        if not self.track_usage:
            return 0
        dead = (self._usage_ema < usage_threshold).nonzero(as_tuple=False).flatten()
        n_dead = int(dead.numel())
        if n_dead == 0:
            return 0
        cand = candidate_features.reshape(-1, self.C)
        if cand.shape[0] < n_dead:
            warnings.warn(f"reset_dead_codes: {n_dead} dead but only {cand.shape[0]} candidates")
            n_dead = min(n_dead, cand.shape[0])
            dead = dead[:n_dead]
        pick = torch.randperm(cand.shape[0], device=cand.device)[:n_dead]
        self.weight[dead] = cand[pick].to(self.weight.dtype)
        # Reset usage so they don't immediately re-die.
        self._usage_ema[dead] = 1.0 / self.K
        if not self._frozen:
            self._ema_count[dead] = 1.0
            self._ema_sum[dead] = self.weight[dead]
        return n_dead

    @torch.no_grad()
    def merge_similar(self, sim_threshold: float) -> int:
        """
        Merge code pairs (i, j) with cosine similarity above sim_threshold.
        Replaces the merged-out slot's weight with a small random perturbation
        of an existing high-usage code (cheap revival).

        Returns number of codes merged.
        Caveat: if precomputed targets exist on disk they will become invalid
        for the merged-out indices. Caller is responsible for periodic
        codebook rebuilds when using on-disk targets.
        """
        if sim_threshold <= 0.0:
            return 0
        W = F.normalize(self.weight.float(), dim=-1)  # (K, C)
        sim = W @ W.t()                                # (K, K)
        sim.fill_diagonal_(-1.0)
        # Greedy merge
        merged = 0
        merged_set: set = set()
        # Iterate over upper triangle by descending sim
        flat_idx = torch.argsort(sim.view(-1), descending=True)
        K = self.K
        for f in flat_idx.tolist():
            i, j = divmod(f, K)
            if i >= j:
                continue
            if sim[i, j].item() < sim_threshold:
                break
            if i in merged_set or j in merged_set:
                continue
            # Replace slot j with a perturbation of a randomly chosen surviving slot
            survivor = torch.randint(0, K, (1,), device=self.weight.device).item()
            while survivor in merged_set or survivor == j:
                survivor = torch.randint(0, K, (1,), device=self.weight.device).item()
            self.weight[j] = self.weight[survivor] + 1e-3 * torch.randn_like(self.weight[survivor])
            if self.track_usage:
                self._usage_ema[j] = 1.0 / K
            merged_set.add(j)
            merged += 1
        return merged
