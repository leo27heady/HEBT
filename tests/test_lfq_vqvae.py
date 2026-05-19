"""Tests for simple LFQ VQ-VAE module."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vid.lfq_vqvae import LFQVAE, LFQVAEConfig


def test_config_defaults():
    cfg = LFQVAEConfig()
    assert cfg.stage_sizes == (16, 4, 1)
    assert cfg.stage_channels == (64, 128, 256)
    assert cfg.codebook_size == 2**12
    assert cfg.lfq_dim == 12


def test_config_rejects_bad_stage_sizes():
    with pytest.raises(ValueError, match="Final stage spatial size must be 1"):
        LFQVAEConfig(stage_sizes=(16, 4, 2))


def test_config_rejects_codebook_mismatch():
    with pytest.raises(ValueError, match="lfq_dim"):
        LFQVAEConfig(codebook_size=512, lfq_dim=12)


def test_forward_shapes():
    cfg = LFQVAEConfig(
        stage_channels=(32, 64, 128),
        codebook_size=2**8,
        lfq_dim=8,
    )
    model = LFQVAE(cfg)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)

    assert out["x_hat"].shape == (2, 3, 64, 64)
    assert out["indices"].shape == (2, 1, 1)
    assert out["quant_feat"].shape == (2, 128, 1, 1)
    assert out["vq_loss"].dim() == 0


def test_loss_backward():
    model = LFQVAE(LFQVAEConfig(stage_channels=(32, 64, 128), codebook_size=2**8, lfq_dim=8))
    x = torch.randn(2, 3, 64, 64)
    loss, metrics = model.loss(x)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["recon"])
    assert torch.isfinite(metrics["vq"])
    assert any(p.grad is not None for p in model.parameters())


def test_encode_decode_roundtrip():
    model = LFQVAE(LFQVAEConfig(stage_channels=(32, 64, 128), codebook_size=2**8, lfq_dim=8))
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    x_hat2 = model.decode(out["quant_feat"])
    assert x_hat2.shape == x.shape
    assert torch.isfinite(x_hat2).all()


def test_decode_indices():
    model = LFQVAE(LFQVAEConfig(stage_channels=(32, 64, 128), codebook_size=2**8, lfq_dim=8))
    x = torch.randn(2, 3, 64, 64)
    indices = model.encode_indices(x)
    x_hat = model.decode_indices(indices)
    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()
