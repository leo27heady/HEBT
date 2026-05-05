"""
Tests for VQ-mode HVEBT.

Run only on a machine with torch installed:
    pytest tests/test_hvebt_vq.py -x

These tests exercise:
  * VQModule: shape contracts and softmax/lookup identity at one-hot.
  * HVEBTStage: features_from_logits / features_from_indices / quantize_features.
  * HierarchicalHVEBT: VQ MCMC + CE loss + entropy/top1/cb_usage metrics.
  * Precomputed-target path: vq_no_features=True with synthetic targets.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vid.hvebt import (  # noqa: E402
    HierarchicalHVEBT,
    HierarchicalHVEBTConfig,
    HVEBTStage,
    HVEBTStageConfig,
)
from model.vid.hvebt.vq import VQModule  # noqa: E402


# --------------------------------------------------------------------------- #
# VQModule
# --------------------------------------------------------------------------- #


def test_vq_module_decode_one_hot_returns_codebook_row():
    torch.manual_seed(0)
    K, C = 8, 4
    vq = VQModule(codebook_size=K, dim=C)
    # Replace weights with known values
    vq.weight.copy_(torch.arange(K * C, dtype=torch.float32).reshape(K, C))
    # Build "near one-hot" logits at index 3, very large peak
    logits = torch.full((1, 1, K), -1e6)
    logits[..., 3] = 1e6
    feats = vq.decode(logits)
    assert feats.shape == (1, 1, C)
    assert torch.allclose(feats[0, 0], vq.weight[3], atol=1e-4)


def test_vq_module_lookup_identity_after_quantize():
    torch.manual_seed(0)
    K, C = 16, 8
    vq = VQModule(codebook_size=K, dim=C)
    # Use codebook itself as features ⇒ quantize must return identity arange
    indices = vq.quantize(vq.weight.unsqueeze(0))  # (1, K)
    assert indices.shape == (1, K)
    assert torch.equal(indices.squeeze(0), torch.arange(K))


def test_vq_module_entropy_uniform_equals_log_K():
    K = 32
    vq = VQModule(codebook_size=K, dim=4)
    logits = torch.zeros(2, 5, K)  # uniform softmax
    h = vq.entropy(logits)
    assert h.shape == (2, 5)
    assert torch.allclose(h, torch.full_like(h, math.log(K)), atol=1e-5)


def test_vq_module_load_from_kmeans_roundtrip(tmp_path):
    K, C = 6, 3
    cb = torch.randn(K, C)
    p = tmp_path / "cb.pt"
    torch.save(cb, p)
    vq = VQModule(codebook_size=K, dim=C)
    vq.load_from_kmeans(str(p))
    assert torch.allclose(vq.weight, cb)


# --------------------------------------------------------------------------- #
# HVEBTStage VQ helpers
# --------------------------------------------------------------------------- #


def _make_vq_stage(K=16, C=8, H=4, W=4, embed_dim=16):
    sc = HVEBTStageConfig(
        clip_stage_name="s_test", clip_channels=C, H=H, W=W,
        embed_dim=embed_dim, n_heads=2, n_layers=1,
        vq_codebook_size=K,
    )
    return HVEBTStage(sc)


def test_hvebtstage_features_from_indices_then_quantize_is_identity():
    stage = _make_vq_stage()
    K, C, H, W = 16, 8, 4, 4
    B, T = 2, 3
    idx = torch.randint(0, K, (B, T, H, W))
    feats = stage.features_from_indices(idx)               # (B, T, C, H, W)
    assert feats.shape == (B, T, C, H, W)
    idx_back = stage.quantize_features(feats)
    assert torch.equal(idx_back, idx)


def test_hvebtstage_features_from_logits_shape_and_grad():
    stage = _make_vq_stage()
    K, C, H, W = 16, 8, 4, 4
    B, T = 2, 3
    logits = torch.randn(B, T * H * W, K, requires_grad=True)
    feats = stage.features_from_logits(logits, T=T)
    assert feats.shape == (B, T, C, H, W)
    feats.sum().backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


# --------------------------------------------------------------------------- #
# HierarchicalHVEBT VQ end-to-end (single stage, on-the-fly targets)
# --------------------------------------------------------------------------- #


def _make_single_stage_vq_model(K=8, C=6, H=4, W=4):
    sc = HVEBTStageConfig(
        clip_stage_name="s_test", clip_channels=C, H=H, W=W,
        embed_dim=16, n_heads=2, n_layers=1,
        vq_codebook_size=K,
    )
    cfg = HierarchicalHVEBTConfig(
        stages=[sc],
        mcmc_num_steps=2,
        mcmc_step_size=10.0,
        weights_path="",     # no encoder
        vq_mode=True,
        vq_use_precomputed_targets=False,  # on-the-fly
    )
    return HierarchicalHVEBT(cfg)


def test_vq_forward_loss_runs_and_returns_metrics():
    torch.manual_seed(0)
    model = _make_single_stage_vq_model()
    B, T_plus_1, C, H, W = 1, 3, 6, 4, 4
    feats = torch.randn(B, T_plus_1, C, H, W)
    video = torch.zeros(B, T_plus_1, 3, 32, 32)  # unused (no decoder)
    out = model.forward_loss(video, features={"s_test": feats}, learning=True)
    assert "loss_total" in out
    assert "per_stage" in out
    s = out["per_stage"][0]
    for k in ("ce_loss", "entropy_mean", "top1_accuracy",
              "codebook_usage", "init_recon", "final_recon",
              "energy_gap", "alpha"):
        assert k in s, f"missing metric {k}"
    # Backprop sanity
    out["loss_total"].backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad


def test_vq_forward_loss_precomputed_targets_no_features():
    torch.manual_seed(0)
    K, C, H, W = 8, 6, 4, 4
    sc = HVEBTStageConfig(
        clip_stage_name="s_test", clip_channels=C, H=H, W=W,
        embed_dim=16, n_heads=2, n_layers=1,
        vq_codebook_size=K,
    )
    cfg = HierarchicalHVEBTConfig(
        stages=[sc],
        mcmc_num_steps=2,
        mcmc_step_size=10.0,
        weights_path="",
        vq_mode=True,
        vq_use_precomputed_targets=True,
        vq_no_features=True,
    )
    model = HierarchicalHVEBT(cfg)
    B, T_plus_1 = 2, 3
    targets = torch.randint(0, K, (B, T_plus_1, H, W))
    video = torch.zeros(B, T_plus_1, 3, 32, 32)
    out = model.forward_loss(
        video, vq_targets={"s_test": targets}, learning=True,
    )
    assert torch.isfinite(out["loss_total"])
    s = out["per_stage"][0]
    assert s is not None
    out["loss_total"].backward()


def test_vq_validation_rejects_train_encoder_with_stale_targets():
    sc = HVEBTStageConfig(
        clip_stage_name="s_test", clip_channels=4, H=2, W=2,
        embed_dim=8, n_heads=2, n_layers=1,
        vq_codebook_size=4,
    )
    cfg = HierarchicalHVEBTConfig(
        stages=[sc], weights_path="",
        vq_mode=True, vq_use_precomputed_targets=True,
        vq_no_features=True,
        train_encoder=True,
        allow_stale_targets=False,
    )
    with pytest.raises(ValueError):
        HierarchicalHVEBT(cfg)


def test_vq_loss_decreases_on_overfit():
    """Tiny overfit sanity test: 1 stage, fixed target, optimizer should drive CE down."""
    torch.manual_seed(0)
    K, C, H, W = 4, 6, 3, 3
    sc = HVEBTStageConfig(
        clip_stage_name="s_test", clip_channels=C, H=H, W=W,
        embed_dim=16, n_heads=2, n_layers=1,
        vq_codebook_size=K,
    )
    cfg = HierarchicalHVEBTConfig(
        stages=[sc], mcmc_num_steps=2, mcmc_step_size=5.0,
        weights_path="",
        vq_mode=True, vq_use_precomputed_targets=True, vq_no_features=True,
    )
    model = HierarchicalHVEBT(cfg)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    B, T_plus_1 = 1, 2
    targets = torch.randint(0, K, (B, T_plus_1, H, W))
    video = torch.zeros(B, T_plus_1, 3, 16, 16)
    losses = []
    for _ in range(15):
        opt.zero_grad()
        out = model.forward_loss(video, vq_targets={"s_test": targets}, learning=True)
        out["loss_total"].backward()
        opt.step()
        losses.append(out["loss_total"].item())
    assert losses[-1] < losses[0], f"VQ loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
