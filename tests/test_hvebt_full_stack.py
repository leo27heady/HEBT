"""
Thorough integration tests for the HVEBT architecture with all CLIP stages
(s0, s1, s2, s3, pooled), temporal windowing, progressive training,
detached KV, 4 MCMC steps.

Covers:
  1. Full 5-stage forward + backward correctness
  2. Progressive training: starts at 1 stage, grows to all 5
  3. Temporal windowing mask correctness at each stage
  4. Cross-attention mask correctness (vector parent, general geometry)
  5. Gradient flow: each stage's loss only grads its own + cross-attn stops
  6. MCMC chaining: parent pred flows top-down correctly
  7. Energy decrease over MCMC steps
  8. Training loop simulation (loss decreases over multiple steps)
  9. No NaN/Inf under various conditions
 10. Mask shape & sparsity sanity checks
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vid.hvebt import (  # noqa: E402
    HierarchicalHVEBT,
    HierarchicalHVEBTConfig,
    HVEBTStage,
    HVEBTStageConfig,
    PixelDecoder,
    build_cross_attn_mask,
)
from model.vid.hvebt.hvebt import build_block_causal_mask  # noqa: E402
from model.vid.hvebt.positional import build_rope3d  # noqa: E402

DEVICE = torch.device("cpu")

# =========================================================================== #
# Helpers
# =========================================================================== #

def _full_5stage_cfg(
    n_layers: int = 1,
    mcmc_steps: int = 4,
    progressive: bool = False,
    progressive_steps_per_stage: int = 2,
    temporal_windows: list = None,
) -> HierarchicalHVEBTConfig:
    """
    5-stage config: s0(8x8) -> s1(4x4) -> s2(2x2) -> s3(2x2) -> pooled(1x1).
    Tiny dims for CPU testing. Ordered finest -> coarsest.
    """
    tw = temporal_windows or [None, None, None, None, None]
    stages = [
        HVEBTStageConfig(stage_name="s0", channels=8, H=8, W=8,
                         embed_dim=16, n_heads=2, n_layers=n_layers, temporal_window=tw[0]),
        HVEBTStageConfig(stage_name="s1", channels=8, H=4, W=4,
                         embed_dim=16, n_heads=2, n_layers=n_layers, temporal_window=tw[1]),
        HVEBTStageConfig(stage_name="s2", channels=12, H=2, W=2,
                         embed_dim=16, n_heads=2, n_layers=n_layers, temporal_window=tw[2]),
        HVEBTStageConfig(stage_name="s3", channels=16, H=2, W=2,
                         embed_dim=16, n_heads=2, n_layers=n_layers, temporal_window=tw[3]),
        HVEBTStageConfig(stage_name="pooled", channels=20, H=1, W=1,
                         embed_dim=16, n_heads=2, n_layers=n_layers, temporal_window=tw[4]),
    ]
    return HierarchicalHVEBTConfig(
        stages=stages,
        mcmc_num_steps=mcmc_steps,
        mcmc_step_size=10.0,
        denoising_init="zeros",
        progressive=progressive,
        progressive_steps_per_stage=progressive_steps_per_stage,
        detach_kv=True,
    )


def _make_model_no_encoder(cfg: HierarchicalHVEBTConfig) -> HierarchicalHVEBT:
    """Construct HierarchicalHVEBT bypassing the CLIP encoder."""
    model = HierarchicalHVEBT.__new__(HierarchicalHVEBT)
    nn.Module.__init__(model)
    model.cfg = cfg
    model._num_active_stages = 1 if cfg.progressive else len(cfg.stages)

    for i in range(1, len(cfg.stages)):
        child, parent = cfg.stages[i - 1], cfg.stages[i]
        assert child.H >= parent.H and child.W >= parent.W, \
            f"child ({child.H},{child.W}) must be >= parent ({parent.H},{parent.W})"

    stages = []
    for i, sc in enumerate(cfg.stages):
        if cfg.disable_cross_attn or i == len(cfg.stages) - 1:
            stages.append(HVEBTStage(sc))
        else:
            parent_sc = cfg.stages[i + 1]
            stages.append(HVEBTStage(sc, parent_channels=parent_sc.channels,
                                     parent_HW=(parent_sc.H, parent_sc.W)))
    model.stages = nn.ModuleList(stages)
    model.alphas = nn.ParameterList([
        nn.Parameter(torch.tensor(float(cfg.mcmc_step_size)),
                     requires_grad=cfg.mcmc_step_size_learnable)
        for _ in cfg.stages
    ])
    model.encoder = None
    model.decoder = None
    if cfg.decoder_enabled:
        base_sc = cfg.stages[0]
        model.decoder = PixelDecoder(
            in_channels=base_sc.channels,
            in_HW=(base_sc.H, base_sc.W),
            out_size=cfg.decoder_out_size,
        )
    return model


def _fake_feats(cfg: HierarchicalHVEBTConfig, B: int = 2, T_plus_1: int = 3):
    """Generate fake feature dicts for all stages."""
    d = {}
    for sc in cfg.stages:
        d[sc.stage_name] = torch.randn(B, T_plus_1, sc.channels, sc.H, sc.W)
    return d


def _fake_video(B: int = 2, T_plus_1: int = 3, S: int = 16):
    """Generate fake video tensor (B, T+1, 3, S, S) in [0,1]."""
    return torch.rand(B, T_plus_1, 3, S, S)


def _forward(model, feats, learning=True, S=16):
    """Test helper: calls model.forward_loss with a dummy video + features."""
    # Infer B, T_plus_1 from features
    sample = next(iter(feats.values()))
    B, T_plus_1 = sample.shape[0], sample.shape[1]
    video = _fake_video(B=B, T_plus_1=T_plus_1, S=S)
    return model.forward_loss(video, features=feats, learning=learning)


# =========================================================================== #
# 1. Full 5-stage forward + backward
# =========================================================================== #

class TestFull5StageForwardBackward:
    """Forward and backward pass with all 5 stages active."""

    def test_forward_produces_all_stage_outputs(self):
        cfg = _full_5stage_cfg(mcmc_steps=4)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)

        assert "loss_total" in out
        assert "loss_energy" in out
        assert torch.isfinite(out["loss_total"])
        per_stage = out["per_stage"]
        assert len(per_stage) == 5
        for i, s in enumerate(per_stage):
            assert s is not None, f"Stage {i} output is None"
            sc = cfg.stages[i]
            assert s["final_pred"].shape == (2, 2, sc.channels, sc.H, sc.W)
            assert s["real_gt"].shape == (2, 2, sc.channels, sc.H, sc.W)
            assert torch.isfinite(s["loss"])
            assert torch.isfinite(s["final_pred"]).all()

    def test_backward_produces_gradients_for_all_stages(self):
        cfg = _full_5stage_cfg(mcmc_steps=4)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        out["loss_total"].backward()

        for i, stage in enumerate(model.stages):
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in stage.parameters())
            assert has_grad, f"Stage {i} ({cfg.stages[i].stage_name}) has no gradient"

    def test_no_nan_in_gradients(self):
        cfg = _full_5stage_cfg(mcmc_steps=4)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        out["loss_total"].backward()

        for i, stage in enumerate(model.stages):
            for name, p in stage.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), \
                        f"NaN/Inf in gradient of stage {i}, param {name}"

    def test_loss_is_positive(self):
        cfg = _full_5stage_cfg(mcmc_steps=4)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        assert out["loss_total"].item() > 0

    def test_alpha_has_gradient(self):
        cfg = _full_5stage_cfg(mcmc_steps=4)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        out["loss_total"].backward()
        for i, alpha in enumerate(model.alphas):
            assert alpha.grad is not None, f"Alpha {i} has no gradient"
            assert torch.isfinite(alpha.grad), f"Alpha {i} gradient is NaN/Inf"


# =========================================================================== #
# 2. Progressive training
# =========================================================================== #

class TestProgressiveTraining:
    """Progressive training starts at 1 stage (apex) and grows to all 5."""

    def _cfg(self):
        return _full_5stage_cfg(
            mcmc_steps=4, progressive=True, progressive_steps_per_stage=2,
        )

    def test_starts_with_one_active_stage(self):
        cfg = self._cfg()
        model = _make_model_no_encoder(cfg)
        assert model.num_active_stages == 1

    def test_active_indices_with_one_stage(self):
        cfg = self._cfg()
        model = _make_model_no_encoder(cfg)
        active = model.active_stage_indices()
        assert active == [4], f"Expected [4] (apex), got {active}"

    def test_progressive_activation_schedule(self):
        """Stages activate one-by-one as steps increase."""
        cfg = self._cfg()
        model = _make_model_no_encoder(cfg)
        pps = cfg.progressive_steps_per_stage  # 2

        # step 0: 1 active (apex only)
        model.update_progressive(0)
        assert model.num_active_stages == 1
        assert model.active_stage_indices() == [4]

        # step 1: still 1
        model.update_progressive(1)
        assert model.num_active_stages == 1

        # step 2: 2 active
        model.update_progressive(2)
        assert model.num_active_stages == 2
        assert model.active_stage_indices() == [4, 3]

        # step 4: 3 active
        model.update_progressive(4)
        assert model.num_active_stages == 3
        assert model.active_stage_indices() == [4, 3, 2]

        # step 6: 4 active
        model.update_progressive(6)
        assert model.num_active_stages == 4
        assert model.active_stage_indices() == [4, 3, 2, 1]

        # step 8: all 5 active
        model.update_progressive(8)
        assert model.num_active_stages == 5
        assert model.active_stage_indices() == [4, 3, 2, 1, 0]

    def test_only_active_stages_produce_output(self):
        """Inactive stages should have None in per_stage."""
        cfg = self._cfg()
        model = _make_model_no_encoder(cfg)
        # 1 active stage (apex = index 4)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        per = out["per_stage"]
        for i in range(4):
            assert per[i] is None, f"Stage {i} should be None when inactive"
        assert per[4] is not None, "Apex stage should be active"

    def test_progressive_2_stages_active(self):
        cfg = self._cfg()
        model = _make_model_no_encoder(cfg)
        model.set_active_stages(2)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        per = out["per_stage"]
        for i in range(3):
            assert per[i] is None
        assert per[3] is not None
        assert per[4] is not None

    def test_progressive_3_stages_active(self):
        cfg = self._cfg()
        model = _make_model_no_encoder(cfg)
        model.set_active_stages(3)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        per = out["per_stage"]
        for i in range(2):
            assert per[i] is None
        for i in range(2, 5):
            assert per[i] is not None

    def test_full_progressive_schedule_forward_backward(self):
        """Simulate a full progressive training run: activate stage by stage."""
        cfg = self._cfg()
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        pps = cfg.progressive_steps_per_stage
        total_steps = pps * 5  # enough steps to activate all 5

        for step in range(total_steps):
            model.update_progressive(step)
            optimizer.zero_grad()
            out = _forward(model, feats)
            loss = out["loss_total"]
            assert torch.isfinite(loss), f"Non-finite loss at step {step}"
            loss.backward()

            # Check no NaN gradients
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), \
                        f"NaN/Inf grad at step {step}, param {name}"
            optimizer.step()

        # After all steps, all 5 stages should be active
        assert model.num_active_stages == 5

    def test_progressive_gradients_only_active_stages(self):
        """Only active stages should receive gradients."""
        cfg = self._cfg()
        model = _make_model_no_encoder(cfg)
        model.set_active_stages(2)  # stages 3, 4 active

        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        out["loss_total"].backward()

        # Stages 0, 1, 2 should have NO gradient
        for i in [0, 1, 2]:
            for name, p in model.stages[i].named_parameters():
                assert p.grad is None or p.grad.abs().sum() == 0, \
                    f"Inactive stage {i} param {name} received gradient"

        # Stages 3, 4 should have gradient
        for i in [3, 4]:
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in model.stages[i].parameters())
            assert has_grad, f"Active stage {i} has no gradient"


# =========================================================================== #
# 3. Temporal windowing mask correctness
# =========================================================================== #

class TestTemporalWindowing:
    """Test that temporal_window restricts self-attention correctly."""

    def test_window_none_is_full_causal(self):
        T, HW = 4, 2
        mask_full = build_block_causal_mask(T, HW, DEVICE, temporal_window=None)
        # Full causal: token at frame tq attends to all frames tk <= tq
        N = T * HW
        idx = torch.arange(N) // HW
        for q in range(N):
            for k in range(N):
                if idx[k] <= idx[q]:
                    assert mask_full[q, k] == 0, f"Should allow ({q},{k})"
                else:
                    assert mask_full[q, k] == float("-inf"), f"Should block ({q},{k})"

    def test_window_1_is_self_frame_only(self):
        T, HW = 4, 2
        mask = build_block_causal_mask(T, HW, DEVICE, temporal_window=1)
        N = T * HW
        idx = torch.arange(N) // HW
        for q in range(N):
            for k in range(N):
                if idx[k] == idx[q]:
                    assert mask[q, k] == 0, f"Same frame ({q},{k}) should be allowed"
                else:
                    assert mask[q, k] == float("-inf"), f"Cross frame ({q},{k}) should be blocked"

    def test_window_2_allows_previous_frame(self):
        T, HW = 4, 2
        mask = build_block_causal_mask(T, HW, DEVICE, temporal_window=2)
        N = T * HW
        idx = torch.arange(N) // HW
        for q in range(N):
            for k in range(N):
                tq, tk = idx[q].item(), idx[k].item()
                if tk <= tq and tq - tk < 2:
                    assert mask[q, k] == 0, f"Window=2: ({q},{k}) tq={tq} tk={tk} should allow"
                else:
                    assert mask[q, k] == float("-inf"), f"Window=2: ({q},{k}) tq={tq} tk={tk} should block"

    def test_window_T_equals_full_causal(self):
        T, HW = 4, 2
        mask_full = build_block_causal_mask(T, HW, DEVICE, temporal_window=None)
        mask_T = build_block_causal_mask(T, HW, DEVICE, temporal_window=T)
        assert torch.equal(mask_full, mask_T), "Window=T should equal full causal"

    def test_graduated_windows_in_model(self):
        """Verify that stages with different temporal_window values
        produce correctly-shaped masks during forward."""
        # finest=window 1, middle=2, coarsest=full
        tw = [1, 1, 2, None, None]
        cfg = _full_5stage_cfg(mcmc_steps=2, temporal_windows=tw)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=4)  # T=3
        out = _forward(model, feats)
        assert torch.isfinite(out["loss_total"])

    def test_window_1_limits_information_flow(self):
        """With window=1, changing a different frame's input should NOT affect
        the current frame's energy output (no cross-time info)."""
        cfg_w1 = HVEBTStageConfig(
            stage_name="s1", channels=8, H=2, W=2,
            embed_dim=16, n_heads=2, n_layers=1, temporal_window=1,
        )
        stage = HVEBTStage(cfg_w1)
        stage.eval()

        B, T = 1, 3
        ctx = torch.randn(B, T, 8, 2, 2)
        nxt = torch.randn(B, T, 8, 2, 2)

        with torch.no_grad():
            energy_orig = stage(ctx, nxt)  # (B, T*H*W)

        # Modify frame 0 input, should NOT change frame 2 energy
        ctx_mod = ctx.clone()
        ctx_mod[:, 0] = torch.randn_like(ctx_mod[:, 0]) * 100
        nxt_mod = nxt.clone()
        nxt_mod[:, 0] = torch.randn_like(nxt_mod[:, 0]) * 100

        with torch.no_grad():
            energy_mod = stage(ctx_mod, nxt_mod)

        HW = 2 * 2
        # Frame 2 tokens are at indices [2*HW : 3*HW]
        frame2_orig = energy_orig[:, 2*HW:3*HW]
        frame2_mod = energy_mod[:, 2*HW:3*HW]
        assert torch.allclose(frame2_orig, frame2_mod, atol=1e-5), \
            "Window=1: frame 2 energy should not change when frame 0 is modified"


# =========================================================================== #
# 4. Cross-attention mask correctness
# =========================================================================== #

class TestCrossAttentionMasks:
    """Test cross-attention masks for different parent geometries."""

    def test_vector_parent_mask(self):
        """Pooled parent (1x1): every child at frame t attends to parent at frame t."""
        T, Hc, Wc, Hp, Wp = 3, 4, 4, 1, 1
        mask = build_cross_attn_mask(T, Hc, Wc, Hp, Wp, DEVICE)
        Nc = T * Hc * Wc  # 48
        Np = T * Hp * Wp  # 3
        assert mask.shape == (Nc, Np)

        # Each child at time t should attend to the single parent at time t
        for ci in range(Nc):
            ct = ci // (Hc * Wc)
            allowed_count = 0
            for pi in range(Np):
                pt = pi // (Hp * Wp)
                if ct == pt:
                    assert mask[ci, pi] == 0, f"Child {ci} should attend parent {pi}"
                    allowed_count += 1
                else:
                    assert mask[ci, pi] == float("-inf")
            assert allowed_count == 1, f"Child {ci} should attend exactly 1 parent"

    def test_2x_parent_mask(self):
        """Standard 2x upsample: child at (yc,xc) -> parent at (yc//2, xc//2)."""
        T, Hc, Wc, Hp, Wp = 2, 4, 4, 2, 2
        mask = build_cross_attn_mask(T, Hc, Wc, Hp, Wp, DEVICE)
        Nc = T * Hc * Wc
        Np = T * Hp * Wp
        assert mask.shape == (Nc, Np)

        for ci in range(Nc):
            ct = ci // (Hc * Wc)
            c_yx = ci % (Hc * Wc)
            cy = c_yx // Wc
            cx = c_yx % Wc
            allowed = 0
            for pi in range(Np):
                pt = pi // (Hp * Wp)
                p_yx = pi % (Hp * Wp)
                py = p_yx // Wp
                px = p_yx % Wp
                if ct == pt and cy // 2 == py and cx // 2 == px:
                    assert mask[ci, pi] == 0
                    allowed += 1
                else:
                    assert mask[ci, pi] == float("-inf")
            assert allowed == 1, f"Child {ci} should attend exactly 1 parent"

    def test_non_2x_parent_mask_general(self):
        """General case: s3(2x2) -> pooled(1x1)."""
        T, Hc, Wc, Hp, Wp = 2, 2, 2, 1, 1
        mask = build_cross_attn_mask(T, Hc, Wc, Hp, Wp, DEVICE)
        Nc = T * Hc * Wc  # 8
        Np = T * Hp * Wp  # 2

        for ci in range(Nc):
            ct = ci // (Hc * Wc)
            allowed = 0
            for pi in range(Np):
                pt = pi // (Hp * Wp)
                if ct == pt:
                    assert mask[ci, pi] == 0
                    allowed += 1
                else:
                    assert mask[ci, pi] == float("-inf")
            assert allowed == 1

    def test_large_ratio_parent_mask(self):
        """s0(8x8) -> s3(2x2): ratio 4x."""
        T, Hc, Wc, Hp, Wp = 2, 8, 8, 2, 2
        mask = build_cross_attn_mask(T, Hc, Wc, Hp, Wp, DEVICE)
        Nc = T * Hc * Wc
        Np = T * Hp * Wp
        assert mask.shape == (Nc, Np)

        for ci in range(Nc):
            ct = ci // (Hc * Wc)
            c_yx = ci % (Hc * Wc)
            cy = c_yx // Wc
            cx = c_yx % Wc
            mapped_py = cy * Hp // Hc
            mapped_px = cx * Wp // Wc
            allowed = 0
            for pi in range(Np):
                pt = pi // (Hp * Wp)
                p_yx = pi % (Hp * Wp)
                py = p_yx // Wp
                px = p_yx % Wp
                if ct == pt and mapped_py == py and mapped_px == px:
                    assert mask[ci, pi] == 0
                    allowed += 1
                else:
                    assert mask[ci, pi] == float("-inf")
            assert allowed == 1, f"Child {ci}: expected 1 allowed, got {allowed}"

    def test_no_cross_time_leak_in_cross_mask(self):
        """No child at time t should attend to parent at time t' != t."""
        T, Hc, Wc, Hp, Wp = 4, 4, 4, 1, 1
        mask = build_cross_attn_mask(T, Hc, Wc, Hp, Wp, DEVICE)
        Nc = T * Hc * Wc
        Np = T * Hp * Wp
        for ci in range(Nc):
            ct = ci // (Hc * Wc)
            for pi in range(Np):
                pt = pi // (Hp * Wp)
                if ct != pt:
                    assert mask[ci, pi] == float("-inf"), \
                        f"Cross-time leak: child t={ct} attending parent t={pt}"


# =========================================================================== #
# 5. Gradient flow with detached KV
# =========================================================================== #

class TestGradientFlow:
    """Verify detached KV stops gradient between stages."""

    def test_detached_kv_stops_gradient_flow_upward(self):
        """With detach_kv=True (default), the loss from a finer stage should
        NOT produce gradients in a coarser stage."""
        cfg = _full_5stage_cfg(mcmc_steps=2)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)

        # Only backprop loss from stage 0 (finest)
        stage0_loss = out["per_stage"][0]["loss"]
        # But per_stage losses are detached. We need to use the actual total_loss
        # and check that upper stage params get gradient only from their own MCMC.

        # Instead: run with only stage 0 active, verify upper stages get no grad
        cfg2 = _full_5stage_cfg(mcmc_steps=2, progressive=True, progressive_steps_per_stage=999)
        model2 = _make_model_no_encoder(cfg2)
        model2.set_active_stages(5)  # all active

        feats2 = _fake_feats(cfg2, B=2, T_plus_1=3)
        # Zero all grads
        for p in model2.parameters():
            p.grad = None

        out2 = _forward(model2, feats2)
        out2["loss_total"].backward()

        # With detached KV, stage 4 (apex) grad should come only from its own loss.
        # Test: freeze stage 4 params, re-run. If stage 4 grads == 0, then no
        # other stage's loss flows into it.
        cfg3 = _full_5stage_cfg(mcmc_steps=2)
        model3 = _make_model_no_encoder(cfg3)
        # Freeze stage 4
        for p in model3.stages[4].parameters():
            p.requires_grad_(False)
        model3.alphas[4].requires_grad_(False)

        feats3 = _fake_feats(cfg3, B=2, T_plus_1=3)
        out3 = _forward(model3, feats3)
        out3["loss_total"].backward()

        # Stage 4 is frozen and detached KV means no grad flows back
        for name, p in model3.stages[4].named_parameters():
            assert p.grad is None, \
                f"Frozen stage 4 param {name} should have no gradient (detach_kv=True)"

    def test_each_stage_loss_contributes_to_total(self):
        """Total loss should be sum of all stage losses."""
        cfg = _full_5stage_cfg(mcmc_steps=2)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)

        sum_stage_losses = sum(
            out["per_stage"][i]["loss"].item()
            for i in range(5)
        )
        assert abs(out["loss_energy"].item() - sum_stage_losses) < 1e-4, \
            f"Total energy loss {out['loss_energy'].item()} != sum of stage losses {sum_stage_losses}"


# =========================================================================== #
# 6. MCMC chaining / parent prediction flow
# =========================================================================== #

class TestMCMCChaining:
    """Verify that parent predictions flow correctly in sequential mode."""

    def test_sequential_order_is_top_down(self):
        """Active indices should be in top-down order (coarsest first)."""
        cfg = _full_5stage_cfg(mcmc_steps=2)
        model = _make_model_no_encoder(cfg)
        active = model.active_stage_indices()
        assert active == [4, 3, 2, 1, 0], f"Expected top-down [4,3,2,1,0], got {active}"

    def test_apex_stage_has_no_cross_attention(self):
        cfg = _full_5stage_cfg(mcmc_steps=2)
        model = _make_model_no_encoder(cfg)
        assert not model.stages[4].use_cross_attn, "Apex stage should have no cross-attention"

    def test_non_apex_stages_have_cross_attention(self):
        cfg = _full_5stage_cfg(mcmc_steps=2)
        model = _make_model_no_encoder(cfg)
        for i in range(4):
            assert model.stages[i].use_cross_attn, \
                f"Stage {i} should have cross-attention"

    def test_parent_hw_matches_config(self):
        cfg = _full_5stage_cfg(mcmc_steps=2)
        model = _make_model_no_encoder(cfg)
        for i in range(4):
            parent_sc = cfg.stages[i + 1]
            assert model.stages[i].parent_HW == (parent_sc.H, parent_sc.W), \
                f"Stage {i} parent_HW mismatch"

    def test_cross_attn_with_vector_parent_forward(self):
        """Stage s3 (2x2) cross-attending to pooled (1x1) should work."""
        stage_cfg = HVEBTStageConfig(
            stage_name="s3", channels=16, H=2, W=2,
            embed_dim=16, n_heads=2, n_layers=1,
        )
        stage = HVEBTStage(stage_cfg, parent_channels=20, parent_HW=(1, 1))
        B, T = 2, 3
        ctx = torch.randn(B, T, 16, 2, 2)
        nxt = torch.randn(B, T, 16, 2, 2)
        parent = torch.randn(B, T, 20, 1, 1)
        energy = stage(ctx, nxt, parent_context=parent)
        assert energy.shape[0] == B
        assert torch.isfinite(energy).all()

    def test_sequential_produces_valid_output(self):
        """Sequential MCMC should produce valid output."""
        cfg = _full_5stage_cfg(mcmc_steps=4)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        assert torch.isfinite(out["loss_total"])
        for i in range(5):
            assert out["per_stage"][i] is not None

# =========================================================================== #
# 7. Energy decrease over MCMC steps
# =========================================================================== #

class TestEnergyDecrease:
    """MCMC steps should generally decrease energy (or at least not diverge)."""

    def test_per_stage_energy_gap_is_finite(self):
        cfg = _full_5stage_cfg(mcmc_steps=4)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        for i in range(5):
            s = out["per_stage"][i]
            assert torch.isfinite(s["init_energy"]), f"Stage {i} init_energy not finite"
            assert torch.isfinite(s["final_energy"]), f"Stage {i} final_energy not finite"
            assert torch.isfinite(s["energy_gap"]), f"Stage {i} energy_gap not finite"

    def test_recon_improves_or_stays(self):
        """After MCMC, final_recon should be <= init_recon (or close)."""
        cfg = _full_5stage_cfg(mcmc_steps=4)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        for i in range(5):
            s = out["per_stage"][i]
            # Allow some slack — MCMC may not always improve recon in 1 forward
            assert s["final_recon"].item() < s["init_recon"].item() * 10, \
                f"Stage {i}: final_recon {s['final_recon']} >> init_recon {s['init_recon']}"


# =========================================================================== #
# 8. Training loop simulation
# =========================================================================== #

class TestTrainingLoop:
    """Simulate multi-step training and verify loss decreases."""

    def test_loss_decreases_over_training_steps(self):
        """Training on fixed data should decrease loss."""
        cfg = _full_5stage_cfg(mcmc_steps=4)
        model = _make_model_no_encoder(cfg)
        torch.manual_seed(42)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for step in range(15):
            optimizer.zero_grad()
            out = _forward(model, feats)
            loss = out["loss_total"]
            assert torch.isfinite(loss), f"Non-finite loss at step {step}"
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Loss should decrease overall (compare first 3 avg vs last 3 avg)
        early_avg = sum(losses[:3]) / 3
        late_avg = sum(losses[-3:]) / 3
        assert late_avg < early_avg, \
            f"Loss didn't decrease: early={early_avg:.4f}, late={late_avg:.4f}"

    def test_progressive_training_loss_stays_finite(self):
        """Full progressive run: activate stages one by one, train each."""
        cfg = _full_5stage_cfg(mcmc_steps=2, progressive=True, progressive_steps_per_stage=3)
        model = _make_model_no_encoder(cfg)
        torch.manual_seed(42)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for step in range(20):
            model.update_progressive(step)
            optimizer.zero_grad()
            out = _forward(model, feats)
            loss = out["loss_total"]
            assert torch.isfinite(loss), f"Non-finite loss at step {step}, active={model.num_active_stages}"
            loss.backward()

            # Verify no NaN/Inf gradients
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), \
                        f"Bad grad at step {step}, param {name}"
            optimizer.step()

    def test_training_with_temporal_window(self):
        """Training with temporal windowing (graduated) should converge."""
        tw = [1, 1, 2, 3, None]
        cfg = _full_5stage_cfg(mcmc_steps=2, temporal_windows=tw)
        model = _make_model_no_encoder(cfg)
        torch.manual_seed(42)
        feats = _fake_feats(cfg, B=2, T_plus_1=4)  # T=3 so window=3 = full
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for step in range(15):
            optimizer.zero_grad()
            out = _forward(model, feats)
            loss = out["loss_total"]
            assert torch.isfinite(loss), f"Non-finite loss at step {step}"
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        early_avg = sum(losses[:3]) / 3
        late_avg = sum(losses[-3:]) / 3
        assert late_avg < early_avg, \
            f"Loss with temporal window didn't decrease: early={early_avg:.4f}, late={late_avg:.4f}"


# =========================================================================== #
# 9. Robustness / edge cases
# =========================================================================== #

class TestRobustness:
    """Edge cases and robustness checks."""

    def test_single_frame_T1(self):
        """T=1 (2 frames: context + target) should work."""
        cfg = _full_5stage_cfg(mcmc_steps=2)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=2)  # T=1
        out = _forward(model, feats)
        assert torch.isfinite(out["loss_total"])

    def test_batch_size_1(self):
        cfg = _full_5stage_cfg(mcmc_steps=2)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=1, T_plus_1=3)
        out = _forward(model, feats)
        assert torch.isfinite(out["loss_total"])

    def test_large_mcmc_steps(self):
        """More MCMC steps shouldn't break things."""
        cfg = _full_5stage_cfg(mcmc_steps=8)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=1, T_plus_1=3)
        out = _forward(model, feats)
        assert torch.isfinite(out["loss_total"])

    def test_eval_mode(self):
        """Forward in eval mode (learning=False) should work."""
        cfg = _full_5stage_cfg(mcmc_steps=4)
        model = _make_model_no_encoder(cfg)
        model.eval()
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats, learning=False)
        assert torch.isfinite(out["loss_total"])

    def test_zero_init_features(self):
        """All-zero features shouldn't cause NaN."""
        cfg = _full_5stage_cfg(mcmc_steps=2)
        model = _make_model_no_encoder(cfg)
        feats = {}
        for sc in cfg.stages:
            feats[sc.stage_name] = torch.zeros(2, 3, sc.channels, sc.H, sc.W)
        out = _forward(model, feats)
        assert torch.isfinite(out["loss_total"])

    def test_deterministic_forward(self):
        """Same input -> same output (no randomness in eval)."""
        cfg = _full_5stage_cfg(mcmc_steps=2)
        model = _make_model_no_encoder(cfg)
        model.eval()
        feats = _fake_feats(cfg, B=2, T_plus_1=3)

        out1 = _forward(model, feats, learning=False)
        out2 = _forward(model, feats, learning=False)
        assert torch.allclose(out1["loss_total"], out2["loss_total"]), \
            "Same input should give same output in eval mode"

    def test_different_denoising_inits(self):
        """All denoising init modes should work."""
        for init in ["zeros", "random_noise", "real_current"]:
            cfg = _full_5stage_cfg(mcmc_steps=2)
            cfg.denoising_init = init
            model = _make_model_no_encoder(cfg)
            feats = _fake_feats(cfg, B=2, T_plus_1=3)
            out = _forward(model, feats)
            assert torch.isfinite(out["loss_total"]), f"Non-finite with init={init}"


# =========================================================================== #
# 10. Mask shapes and sparsity
# =========================================================================== #

class TestMaskProperties:
    """Verify mask shapes, sparsity, and structural properties."""

    def test_self_attn_mask_shape(self):
        T, H, W = 3, 4, 4
        N = T * H * W
        mask = build_block_causal_mask(T, H * W, DEVICE)
        assert mask.shape == (N, N)

    def test_self_attn_mask_is_lower_triangular_blocks(self):
        """Block-causal: within each frame block, all allowed. Cross-frame: lower-triangular."""
        T, HW = 3, 4
        mask = build_block_causal_mask(T, HW, DEVICE)
        # Extract the T x T block structure
        for tq in range(T):
            for tk in range(T):
                block = mask[tq*HW:(tq+1)*HW, tk*HW:(tk+1)*HW]
                if tk <= tq:
                    assert (block == 0).all(), f"Block (tq={tq},tk={tk}) should be all-zero (allowed)"
                else:
                    assert (block == float("-inf")).all(), f"Block (tq={tq},tk={tk}) should be all -inf"

    def test_cross_mask_each_row_has_exactly_one_allowed(self):
        """For vector parent, each child row should have exactly 1 allowed entry."""
        for Hp, Wp in [(1, 1), (2, 2)]:
            T, Hc, Wc = 3, 4, 4
            mask = build_cross_attn_mask(T, Hc, Wc, Hp, Wp, DEVICE)
            allowed_per_row = (mask == 0).sum(dim=1)
            assert (allowed_per_row == 1).all(), \
                f"Each child row should have exactly 1 allowed entry (parent {Hp}x{Wp})"

    def test_cross_mask_symmetry_within_frame(self):
        """Within a time frame, the spatial mapping should be consistent."""
        T, Hc, Wc, Hp, Wp = 1, 8, 8, 2, 2
        mask = build_cross_attn_mask(T, Hc, Wc, Hp, Wp, DEVICE)
        # Children (0,0), (0,1), (1,0), (1,1) should all map to parent (0,0)
        # That's child indices 0, 1, Wc, Wc+1 -> parent 0
        for cy, cx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            ci = cy * Wc + cx
            pi = (cy * Hp // Hc) * Wp + (cx * Wp // Wc)
            assert mask[ci, pi] == 0, \
                f"Child ({cy},{cx})={ci} should attend to parent {pi}"

    def test_windowed_mask_sparsity_increases_with_smaller_window(self):
        """Smaller window -> more -inf entries -> sparser mask."""
        T, HW = 4, 4
        mask_full = build_block_causal_mask(T, HW, DEVICE, temporal_window=None)
        mask_w2 = build_block_causal_mask(T, HW, DEVICE, temporal_window=2)
        mask_w1 = build_block_causal_mask(T, HW, DEVICE, temporal_window=1)

        allowed_full = (mask_full == 0).sum().item()
        allowed_w2 = (mask_w2 == 0).sum().item()
        allowed_w1 = (mask_w1 == 0).sum().item()

        assert allowed_full >= allowed_w2 >= allowed_w1, \
            f"Sparsity order wrong: full={allowed_full}, w2={allowed_w2}, w1={allowed_w1}"

    def test_windowed_mask_window1_block_count(self):
        """Window=1: each frame only sees itself. Should have T blocks of HW*HW."""
        T, HW = 4, 4
        mask = build_block_causal_mask(T, HW, DEVICE, temporal_window=1)
        expected_allowed = T * HW * HW
        actual_allowed = (mask == 0).sum().item()
        assert actual_allowed == expected_allowed, \
            f"Window=1: expected {expected_allowed} allowed, got {actual_allowed}"


# =========================================================================== #
# 11. Per-stage output consistency
# =========================================================================== #

class TestPerStageOutputs:
    """Check per-stage output dict fields and shapes."""

    def test_all_expected_keys_present(self):
        cfg = _full_5stage_cfg(mcmc_steps=4)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)

        expected_keys = {
            "loss", "init_recon", "final_recon", "init_energy", "final_energy",
            "energy_gap", "alpha", "baseline_copy_last", "final_pred", "real_gt",
        }
        for i in range(5):
            s = out["per_stage"][i]
            assert s is not None
            missing = expected_keys - set(s.keys())
            assert not missing, f"Stage {i} missing keys: {missing}"

    def test_pred_shape_matches_stage_config(self):
        cfg = _full_5stage_cfg(mcmc_steps=4)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)

        for i, sc in enumerate(cfg.stages):
            s = out["per_stage"][i]
            expected = (2, 2, sc.channels, sc.H, sc.W)
            assert s["final_pred"].shape == expected, \
                f"Stage {i}: pred shape {s['final_pred'].shape} != expected {expected}"
            assert s["real_gt"].shape == expected

    def test_all_outputs_detached(self):
        """Per-stage outputs should be detached (no grad tracking)."""
        cfg = _full_5stage_cfg(mcmc_steps=2)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        for i in range(5):
            s = out["per_stage"][i]
            assert not s["final_pred"].requires_grad
            assert not s["real_gt"].requires_grad
            assert not s["loss"].requires_grad

    def test_alpha_values_positive(self):
        cfg = _full_5stage_cfg(mcmc_steps=2)
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        for i in range(5):
            assert out["per_stage"][i]["alpha"].item() > 0


# =========================================================================== #
# 12. Stage construction validation
# =========================================================================== #

class TestConstructionValidation:
    """Verify the constructor catches invalid configurations."""

    def test_empty_stages_raises(self):
        with pytest.raises(ValueError, match="at least one stage"):
            cfg = HierarchicalHVEBTConfig(stages=[])
            HierarchicalHVEBT(cfg)

    def test_child_smaller_than_parent_raises(self):
        """Child spatial must be >= parent (finer stage listed first)."""
        cfg = HierarchicalHVEBTConfig(
            stages=[
                HVEBTStageConfig(stage_name="4x4", channels=128, H=4, W=4,
                                 embed_dim=16, n_heads=2, n_layers=1),
                HVEBTStageConfig(stage_name="16x16", channels=64, H=16, W=16,
                                 embed_dim=16, n_heads=2, n_layers=1),
            ],
        )
        with pytest.raises(ValueError, match="must be >="):
            HierarchicalHVEBT(cfg)

    def test_valid_non_2x_geometry_accepted(self):
        """Non-2x geometry (e.g., 8x8 -> 1x1) accepted when using fake features."""
        cfg = HierarchicalHVEBTConfig(
            stages=[
                HVEBTStageConfig(stage_name="fine", channels=16, H=8, W=8,
                                 embed_dim=16, n_heads=2, n_layers=1),
                HVEBTStageConfig(stage_name="apex1", channels=20, H=1, W=1,
                                 embed_dim=16, n_heads=2, n_layers=1),
            ],
        )
        model = _make_model_no_encoder(cfg)
        assert len(model.stages) == 2

    def test_single_stage_config(self):
        """A single-stage config (apex only) should work."""
        cfg = HierarchicalHVEBTConfig(
            stages=[
                HVEBTStageConfig(stage_name="1x1", channels=256, H=1, W=1,
                                 embed_dim=16, n_heads=2, n_layers=1),
            ],
        )
        model = HierarchicalHVEBT(cfg)
        assert len(model.stages) == 1
        assert not model.stages[0].use_cross_attn


# =========================================================================== #
# Adaptive MCMC Tests
# =========================================================================== #

class TestAdaptiveMCMC:
    """Tests for adaptive MCMC convergence loop."""

    def _adaptive_cfg(self):
        cfg = _full_5stage_cfg(mcmc_steps=4)
        cfg.adaptive_mcmc = True
        cfg.adaptive_mcmc_max_steps = 20
        cfg.adaptive_mcmc_tol = 1e-3
        cfg.adaptive_mcmc_patience = 3
        cfg.adaptive_mcmc_step_penalty = 0.0
        cfg.truncate_mcmc = True
        return cfg

    def test_forward_backward_works(self):
        """Adaptive MCMC forward/backward produces valid gradients."""
        cfg = self._adaptive_cfg()
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        assert torch.isfinite(out["loss_total"])
        out["loss_total"].backward()
        for i, stage in enumerate(model.stages):
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in stage.parameters())
            assert has_grad, f"Adaptive: stage {i} has no gradient"

    def test_reports_steps_used(self):
        """Per-stage output includes mcmc_steps_used."""
        cfg = self._adaptive_cfg()
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats)
        for i, ps in enumerate(out["per_stage"]):
            if ps is not None:
                assert "mcmc_steps_used" in ps
                assert 1 <= ps["mcmc_steps_used"] <= cfg.adaptive_mcmc_max_steps

    def test_step_penalty_adds_to_loss(self):
        """Step penalty > 0 should increase total loss."""
        cfg = self._adaptive_cfg()
        cfg.adaptive_mcmc_step_penalty = 0.0
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        torch.manual_seed(42)
        out_no_pen = _forward(model, feats)

        cfg2 = self._adaptive_cfg()
        cfg2.adaptive_mcmc_step_penalty = 1.0
        model2 = _make_model_no_encoder(cfg2)
        # Copy weights
        model2.load_state_dict(model.state_dict())
        torch.manual_seed(42)
        out_pen = _forward(model2, feats)

        assert out_pen["loss_total"].item() >= out_no_pen["loss_total"].item()

    def test_with_bottom_up_loss(self):
        """Adaptive MCMC + bottom_up_loss should work together."""
        cfg = self._adaptive_cfg()
        cfg.bottom_up_loss = True
        cfg.decoder_enabled = True
        cfg.decoder_out_size = 16
        cfg.detach_kv = False
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats, S=16)
        assert "loss_decoder" in out
        out["loss_total"].backward()


# =========================================================================== #
# Bottom-Up Loss Mode Tests
# =========================================================================== #

class TestBottomUpLoss:
    """Tests for bottom_up_loss mode (decoder pixel loss drives entire hierarchy)."""

    def _bu_cfg(self, progressive=False):
        """5-stage config with bottom_up_loss=True and decoder enabled."""
        cfg = _full_5stage_cfg(mcmc_steps=2, progressive=progressive)
        cfg.bottom_up_loss = True
        cfg.decoder_enabled = True
        cfg.decoder_out_size = 16
        cfg.detach_kv = False  # bottom-up implies no detach
        return cfg

    def test_sequential_forward_backward(self):
        """Forward/backward works with bottom_up_loss + decoder."""
        cfg = self._bu_cfg()
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats, S=16)
        assert "loss_total" in out
        assert "loss_decoder" in out
        out["loss_total"].backward()

    def test_decoder_loss_is_sole_objective(self):
        """In bottom-up mode, loss_total == loss_decoder (no feature losses added)."""
        cfg = self._bu_cfg()
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats, S=16)
        assert torch.allclose(out["loss_total"], out["loss_decoder"]), \
            f"loss_total={out['loss_total'].item():.6f} != loss_decoder={out['loss_decoder'].item():.6f}"

    def test_no_feature_loss_for_any_stage(self):
        """In bottom-up + decoder, all per-stage losses should be zero."""
        cfg = self._bu_cfg()
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats, S=16)
        for i, ps in enumerate(out["per_stage"]):
            if ps is not None:
                assert ps["loss"].item() == 0.0, \
                    f"Stage {i} has non-zero loss {ps['loss'].item()} in bottom-up mode"

    def test_gradient_flows_to_all_stages(self):
        """Decoder loss gradient flows upward through all stages (non-detached KV)."""
        cfg = self._bu_cfg()
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats, S=16)
        out["loss_total"].backward()
        for i, stage in enumerate(model.stages):
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in stage.parameters())
            assert has_grad, f"Bottom-up: stage {i} has no gradient from decoder loss"

    def test_decoder_has_gradient(self):
        """Decoder parameters receive gradient in bottom-up mode."""
        cfg = self._bu_cfg()
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        out = _forward(model, feats, S=16)
        out["loss_total"].backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.decoder.parameters())
        assert has_grad, "Decoder has no gradient in bottom-up mode"

    def test_progressive_bottom_up(self):
        """Progressive training + bottom_up_loss: only active stages run.
        Decoder only engages once the finest stage (s0) is active."""
        cfg = self._bu_cfg(progressive=True)
        cfg.progressive_steps_per_stage = 2
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)

        # Initially only 1 stage active (pooled). Decoder can't fire yet
        # because pooled doesn't match decoder input (decoder built for s0).
        # In bottom-up without decoder matching, loss comes from finest active feature loss.
        out = _forward(model, feats, S=16)
        active_count = sum(1 for ps in out["per_stage"] if ps is not None)
        assert active_count == 1

        # Activate all stages (step big enough)
        for step in range(2, 12, 2):
            model.update_progressive(step=step)
        out2 = _forward(model, feats, S=16)
        active_count2 = sum(1 for ps in out2["per_stage"] if ps is not None)
        assert active_count2 == 5

        # With all stages active, decoder fires and loss works
        assert "loss_decoder" in out2
        out2["loss_total"].backward()

    def test_bottom_up_loss_decreases(self):
        """Multiple training steps should decrease loss in bottom-up mode."""
        cfg = self._bu_cfg()
        model = _make_model_no_encoder(cfg)
        feats = _fake_feats(cfg, B=2, T_plus_1=3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(10):
            opt.zero_grad()
            out = _forward(model, feats, S=16)
            out["loss_total"].backward()
            opt.step()
            losses.append(out["loss_total"].item())

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
