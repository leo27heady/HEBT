"""Tests for simple LFQ VQ-VAE module."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from example_code.lfq_vqvae_viz import flatten_video_batch
from model.vid.lfq_vqvae import (
    LFQVAE,
    LFQVAEConfig,
    build_cross_attn_mask_mid_to_bot,
    build_cross_attn_mask_top_to_mid,
    build_temporal_window_mask,
)


def _small_bottleneck_cfg() -> LFQVAEConfig:
    return LFQVAEConfig(
        stage_channels=(32, 64, 128),
        codebook_size=2**8,
        lfq_dim=8,
    )


def _small_hierarchical_cfg(fusion: str = "conv") -> LFQVAEConfig:
    return LFQVAEConfig(
        quantization_mode="hierarchical",
        stage_channels=(32, 64, 128),
        stage_codebook_sizes=(8, 32, 128),
        stage_lfq_dims=(3, 5, 7),
        fusion=fusion,  # type: ignore[arg-type]
    )


def _small_progressive_cfg(fusion: str = "conv") -> LFQVAEConfig:
    return LFQVAEConfig(
        quantization_mode="hierarchical",
        train_mode="progressive",
        stage_channels=(32, 64, 128),
        stage_codebook_sizes=(8, 32, 128),
        stage_lfq_dims=(3, 5, 7),
        fusion=fusion,  # type: ignore[arg-type]
        progressive_stage_steps=(2, 3, 4),
        progressive_freeze_parents=True,
        progressive_prior_ce=False,
    )


def _small_video_cfg(train_mode: str = "joint") -> LFQVAEConfig:
    return LFQVAEConfig(
        quantization_mode="hierarchical",
        stage_channels=(32, 64, 128),
        stage_codebook_sizes=(8, 32, 128),
        stage_lfq_dims=(3, 5, 7),
        enable_video_predictor=True,
        train_mode=train_mode,  # type: ignore[arg-type]
        pred_n_heads=4,
        pred_n_layers=2,
        pred_dim_top=128,
        pred_dim_mid=64,
        pred_dim_bot=32,
        max_T=8,
    )


def test_config_defaults():
    cfg = LFQVAEConfig()
    assert cfg.stage_sizes == (16, 4, 1)
    assert cfg.stage_channels == (64, 128, 256)
    assert cfg.codebook_size == 2**12
    assert cfg.lfq_dim == 12
    assert cfg.quantization_mode == "bottleneck"


def test_config_rejects_bad_stage_sizes():
    with pytest.raises(ValueError, match="Final stage spatial size must be 1"):
        LFQVAEConfig(stage_sizes=(16, 4, 2))


def test_config_rejects_codebook_mismatch():
    with pytest.raises(ValueError, match="lfq_dim"):
        LFQVAEConfig(codebook_size=512, lfq_dim=12)


def test_config_hierarchical_rejects_mismatched_codebooks():
    with pytest.raises(ValueError, match="stage_codebook_sizes"):
        LFQVAEConfig(
            quantization_mode="hierarchical",
            stage_codebook_sizes=(32, 512),
            stage_lfq_dims=(5, 9, 12),
        )


def test_config_rejects_video_predictor_on_bottleneck():
    with pytest.raises(ValueError, match="requires quantization_mode='hierarchical'"):
        LFQVAEConfig(enable_video_predictor=True, quantization_mode="bottleneck")


def test_config_rejects_progressive_on_bottleneck():
    with pytest.raises(ValueError, match="progressive"):
        LFQVAEConfig(train_mode="progressive", quantization_mode="bottleneck")


def test_config_rejects_bad_progressive_stage_steps_length():
    with pytest.raises(ValueError, match="progressive_stage_steps"):
        LFQVAEConfig(
            quantization_mode="hierarchical",
            train_mode="progressive",
            progressive_stage_steps=(2, 3),
        )


def test_config_rejects_predictor_dim_mismatch():
    with pytest.raises(ValueError, match="pred_dim_top"):
        LFQVAEConfig(
            quantization_mode="hierarchical",
            enable_video_predictor=True,
            pred_dim_top=64,
            stage_channels=(32, 64, 128),
        )


def test_forward_shapes_bottleneck():
    model = LFQVAE(_small_bottleneck_cfg())
    x = torch.randn(2, 3, 64, 64)
    out = model(x)

    assert out["x_hat"].shape == (2, 3, 64, 64)
    assert out["indices"].shape == (2, 1, 1)
    assert out["quant_feat"].shape == (2, 128, 1, 1)
    assert out["vq_loss"].dim() == 0


def test_forward_shapes_hierarchical():
    model = LFQVAE(_small_hierarchical_cfg())
    x = torch.randn(2, 3, 64, 64)
    out = model(x)

    assert out["x_hat"].shape == (2, 3, 64, 64)
    assert out["idx_bot"].shape == (2, 16, 16)
    assert out["idx_mid"].shape == (2, 4, 4)
    assert out["idx_top"].shape == (2, 1, 1)
    assert out["vq_loss"].dim() == 0


def test_progressive_forward_shapes():
    model = LFQVAE(_small_progressive_cfg())
    x = torch.randn(2, 3, 64, 64)
    for depth in (1, 2, 3):
        model.set_progressive_depth(depth)
        out = model(x)
        assert out["x_hat"].shape == (2, 3, 64, 64)


@pytest.mark.parametrize("fusion", ["concat", "conv", "gamma"])
def test_hierarchical_loss_backward(fusion: str):
    model = LFQVAE(_small_hierarchical_cfg(fusion))
    x = torch.randn(2, 3, 64, 64)
    loss, metrics = model.loss(x)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["recon"])
    assert torch.isfinite(metrics["vq"])
    assert torch.isfinite(metrics["prior_ce"])
    assert any(p.grad is not None for p in model.parameters())


def test_loss_backward_bottleneck():
    model = LFQVAE(_small_bottleneck_cfg())
    x = torch.randn(2, 3, 64, 64)
    loss, metrics = model.loss(x)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["recon"])
    assert torch.isfinite(metrics["vq"])
    assert any(p.grad is not None for p in model.parameters())


def test_encode_decode_roundtrip():
    model = LFQVAE(_small_bottleneck_cfg())
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    x_hat2 = model.decode(out["quant_feat"])
    assert x_hat2.shape == x.shape
    assert torch.isfinite(x_hat2).all()


def test_decode_indices():
    model = LFQVAE(_small_bottleneck_cfg())
    x = torch.randn(2, 3, 64, 64)
    indices = model.encode_indices(x)
    x_hat = model.decode_indices(indices)
    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()


def test_hierarchical_encode_indices():
    model = LFQVAE(_small_hierarchical_cfg())
    x = torch.randn(2, 3, 64, 64)
    indices = model.encode_indices(x)
    assert isinstance(indices, dict)
    assert indices["idx_top"].shape == (2, 1, 1)


def test_temporal_window_masks():
    m_top = build_temporal_window_mask(T=3, S=1, temporal_window=-1, device=torch.device("cpu"))
    m_mid = build_temporal_window_mask(T=3, S=16, temporal_window=2, device=torch.device("cpu"))
    m_bot = build_temporal_window_mask(T=3, S=256, temporal_window=1, device=torch.device("cpu"))
    assert m_top.shape == (3, 3)
    assert m_mid.shape == (48, 48)
    assert m_bot.shape == (768, 768)
    # bot window=1 should block previous frame
    assert not bool(m_bot[256, 0])
    # but allow same-frame token
    assert bool(m_bot[256, 256])


def test_cross_attention_masks():
    t = 4
    m_tm = build_cross_attn_mask_top_to_mid(t, torch.device("cpu"))
    m_mb = build_cross_attn_mask_mid_to_bot(t, torch.device("cpu"))
    assert m_tm.shape == (t * 16, t * 1)
    assert m_mb.shape == (t * 256, t * 16)


def test_encode_video_shapes():
    model = LFQVAE(_small_video_cfg(train_mode="joint"))
    video = torch.randn(2, 4, 3, 64, 64)
    enc = model.encode_video(video)
    assert enc["quant_bot"].shape == (2, 4, 32, 16, 16)
    assert enc["quant_mid"].shape == (2, 4, 64, 4, 4)
    assert enc["quant_top"].shape == (2, 4, 128, 1, 1)
    assert enc["idx_bot"].shape == (2, 4, 16, 16)


def test_predict_video_shapes():
    model = LFQVAE(_small_video_cfg(train_mode="joint"))
    video = torch.randn(2, 4, 3, 64, 64)
    enc = model.encode_video(video)
    pred = model.predict_video(enc, T=3)
    assert pred["logits_top"].shape == (2, 3, 128)
    assert pred["logits_mid"].shape == (2, 48, 32)
    assert pred["logits_bot"].shape == (2, 768, 8)


def test_predict_video_rejects_t_exceeding_max_t():
    model = LFQVAE(_small_video_cfg(train_mode="joint"))
    video = torch.randn(2, 11, 3, 64, 64)
    enc = model.encode_video(video)
    with pytest.raises(ValueError, match="exceeds max_T"):
        model.predict_video(enc, T=10)


def test_flatten_video_batch_shape():
    batch = torch.randn(4, 5, 3, 64, 64)
    flat = flatten_video_batch(batch)
    assert flat.shape == (20, 3, 64, 64)


def test_progressive_recon_depth_matches_forward_not_full_decoder():
    model = LFQVAE(_small_progressive_cfg())
    model.set_progressive_depth(1)
    x = torch.randn(5, 3, 64, 64)
    via_forward = model(x)["x_hat"]
    enc = model.encoder(x)
    via_full_decoder, _ = model.decoder(enc, depth=3)
    assert via_forward.shape == via_full_decoder.shape
    assert not torch.allclose(via_forward, via_full_decoder)


def test_stage_token_flatten_roundtrip_order():
    x = torch.arange(2 * 3 * 4 * 2 * 2, dtype=torch.float32).reshape(2, 3, 4, 2, 2)
    flat = LFQVAE._flatten_stage_tokens(x)
    rec = LFQVAE._unflatten_stage_tokens(flat, B=2, T=3, S=2, C=4).reshape(2, 3, 4, 2, 2)
    assert torch.equal(x, rec)


def test_video_loss_joint_backward():
    model = LFQVAE(_small_video_cfg(train_mode="joint"))
    video = torch.randn(2, 4, 3, 64, 64)
    loss, metrics = model.video_loss(video, train_mode="joint")
    loss.backward()
    assert torch.isfinite(loss)
    assert "recon" in metrics and "ce" in metrics


def test_predictor_ce_uses_spatial_weights():
    """Weighted predictor CE should match prior_ce scale, not raw stage sum."""
    model = LFQVAE(_small_video_cfg(train_mode="joint"))
    video = torch.randn(2, 4, 3, 64, 64)
    _, metrics = model.video_loss(video, train_mode="joint")
    ce_sum = metrics["ce_top"] + metrics["ce_mid"] + metrics["ce_bot"]
    assert metrics["ce"] < ce_sum
    assert metrics["ce"].item() < ce_sum.item() * 0.1


def test_video_loss_disjoint_backward():
    model = LFQVAE(_small_video_cfg(train_mode="disjoint"))
    # mimic disjoint mode: freeze VQ-VAE stack
    for p in model.vqvae_params():
        p.requires_grad_(False)
    video = torch.randn(2, 4, 3, 64, 64)
    loss, metrics = model.video_loss(video, train_mode="disjoint")
    loss.backward()
    assert torch.isfinite(loss)
    assert "ce" in metrics and "pred_mse" in metrics


def test_progressive_depth_schedule():
    model = LFQVAE(_small_progressive_cfg())
    assert model.progressive_depth == 1
    assert model.update_progressive(0) is None
    assert model.progressive_depth == 1
    assert model.update_progressive(2) == 2
    assert model.progressive_depth == 2
    assert model.update_progressive(5) == 3
    assert model.progressive_depth == 3


def test_progressive_freeze_parents():
    model = LFQVAE(_small_progressive_cfg())
    model.set_progressive_depth(2)
    trainable = model.apply_progressive_freeze(True)
    top_group = model.progressive_param_groups()["top"]
    assert all(not p.requires_grad for p in top_group)
    assert all(p.requires_grad for p in trainable)


def test_progressive_loss_only_active_vq():
    model = LFQVAE(_small_progressive_cfg())
    model.set_progressive_depth(1)
    x = torch.randn(2, 3, 64, 64)
    loss, metrics = model.loss(x)
    loss.backward()

    top_grads = [p.grad for p in model.encoder.vq_top.parameters()]
    mid_grads = [p.grad for p in model.encoder.vq_mid.parameters()]
    bot_grads = [p.grad for p in model.encoder.vq_bot.parameters()]
    assert any(g is not None for g in top_grads)
    assert all(g is None for g in mid_grads)
    assert all(g is None for g in bot_grads)


def test_disjoint_decoder_tail_only_gradients():
    cfg = _small_video_cfg(train_mode="disjoint")
    cfg.lambda_pred_mse = 1.0
    model = LFQVAE(cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.decoder_image_params():
        p.requires_grad_(True)

    video = torch.randn(2, 4, 3, 64, 64)
    loss, _ = model.video_loss(video, train_mode="disjoint")
    loss.backward()

    tail_grads = [p.grad for p in model.decoder_image_params()]
    assert any(g is not None for g in tail_grads)

    frozen_groups = model.encoder_params() + model.decoder_prior_params() + model.predictor_params()
    assert all(p.grad is None for p in frozen_groups if not p.requires_grad)


@pytest.mark.parametrize("detach_parents", [False, True])
def test_decode_predicted_bot_only_shape_and_detach(detach_parents: bool):
    cfg = _small_video_cfg(train_mode="joint")
    cfg.detach_parent_features = detach_parents
    model = LFQVAE(cfg)
    B, T = 2, 3
    feat_bot = torch.randn(B, T * 16 * 16, cfg.pred_dim_bot, requires_grad=True)
    pred = {
        "feat_bot": feat_bot,
        "feat_mid": torch.randn(B, T * 4 * 4, cfg.pred_dim_mid),
        "feat_top": torch.randn(B, T, cfg.pred_dim_top),
    }
    pred_rgb = model.decode_predicted(pred, B=B, T=T)
    assert pred_rgb.shape == (B * T, 3, 64, 64)
    pred_rgb.mean().backward()
    if detach_parents:
        assert feat_bot.grad is None
    else:
        assert feat_bot.grad is not None
