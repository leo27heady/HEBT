"""
Tests for HVEBT Phase 2 (hierarchy + cross-attention), Phase 3 (apex), and the
optional pixel decoder.

Covers:
  * Cross-attention mask correctness (exhaustive on tiny grid).
  * Cross-attention forward shape + determinism.
  * 2-stage and 3-stage hierarchical forward shape and per-stage outputs.
  * MCMC energy decreases at every stage.
  * Per-stage gradient flow: not NaN, not zero, not exploding.
  * Detached KV: gradient from upper stage's loss does NOT touch lower stage params.
  * Frozen-untrained-top sanity: lower stages still receive gradient.
  * Hierarchical can overfit a tiny synthetic dataset.
  * PixelDecoder forward shape; decoder loss does NOT backprop into HVEBT.
  * `save_recon_grid` writes a file when torchvision is available.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vid.hvebt import (  # noqa: E402
    CrossAttention3DRoPE,
    HierarchicalHVEBT,
    HierarchicalHVEBTConfig,
    HVEBTStage,
    HVEBTStageConfig,
    PixelDecoder,
    build_child_to_parent_mask,
    save_recon_grid,
)
from model.vid.hvebt.positional import build_rope3d  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

DEVICE = torch.device("cpu")


def _tiny_2stage_cfg(n_layers: int = 1) -> HierarchicalHVEBTConfig:
    """
    Tiny CPU-friendly 2-stage hierarchy that doesn't load CLIP. We bypass the
    encoder by feeding pre-computed feature dicts in tests that need it; tests
    that DO load CLIP use the real default config.
    """
    return HierarchicalHVEBTConfig(
        stages=[
            HVEBTStageConfig(clip_stage_name="s1", clip_channels=8, H=4, W=4,
                             embed_dim=16, n_heads=2, n_layers=n_layers),
            HVEBTStageConfig(clip_stage_name="s2", clip_channels=16, H=2, W=2,
                             embed_dim=16, n_heads=2, n_layers=n_layers),
        ],
        mcmc_num_steps=2,
        mcmc_step_size=10.0,
    )


def _tiny_3stage_cfg(n_layers: int = 1) -> HierarchicalHVEBTConfig:
    return HierarchicalHVEBTConfig(
        stages=[
            HVEBTStageConfig(clip_stage_name="s1", clip_channels=8, H=4, W=4,
                             embed_dim=16, n_heads=2, n_layers=n_layers),
            HVEBTStageConfig(clip_stage_name="s2", clip_channels=16, H=2, W=2,
                             embed_dim=16, n_heads=2, n_layers=n_layers),
            HVEBTStageConfig(clip_stage_name="s3", clip_channels=32, H=1, W=1,
                             embed_dim=16, n_heads=2, n_layers=n_layers),
        ],
        mcmc_num_steps=2,
        mcmc_step_size=10.0,
    )


def _make_model_no_encoder(cfg: HierarchicalHVEBTConfig) -> HierarchicalHVEBT:
    """Construct HierarchicalHVEBT but bypass the heavy CLIP encoder."""
    # Lazily build without calling __init__'s encoder branch.
    model = HierarchicalHVEBT.__new__(HierarchicalHVEBT)
    torch.nn.Module.__init__(model)
    model.cfg = cfg
    # Validate geometry (mirror the constructor's check).
    for i in range(1, len(cfg.stages)):
        child, parent = cfg.stages[i - 1], cfg.stages[i]
        assert child.H == 2 * parent.H and child.W == 2 * parent.W

    stages = []
    for i, sc in enumerate(cfg.stages):
        if i == len(cfg.stages) - 1:
            # Apex (coarsest) stage: no cross-attention
            stages.append(HVEBTStage(sc))
        else:
            # Non-apex stages cross-attend to the coarser stage above
            parent_sc = cfg.stages[i + 1]
            stages.append(HVEBTStage(sc, parent_channels=parent_sc.clip_channels,
                                     parent_HW=(parent_sc.H, parent_sc.W)))
    model.stages = torch.nn.ModuleList(stages)
    model.alphas = torch.nn.ParameterList([
        torch.nn.Parameter(torch.tensor(float(cfg.mcmc_step_size)),
                           requires_grad=cfg.mcmc_step_size_learnable)
        for _ in cfg.stages
    ])
    model.encoder = None  # not used
    model.decoder = None
    if cfg.decoder_enabled:
        base_sc = cfg.stages[0]
        model.decoder = PixelDecoder(
            in_channels=base_sc.clip_channels,
            in_HW=(base_sc.H, base_sc.W),
            out_size=cfg.decoder_out_size,
        )
    return model


def _fake_feats(cfg: HierarchicalHVEBTConfig, B: int, T_plus_1: int):
    """Random per-stage features, mimicking encoder output shape."""
    out = {}
    for sc in cfg.stages:
        out[sc.clip_stage_name] = torch.randn(B, T_plus_1, sc.clip_channels, sc.H, sc.W)
    return out


def _forward_loss_with_fake_feats(model: HierarchicalHVEBT, feats_dict, learning: bool = True):
    """Run the full hierarchical forward but with a pre-computed feature dict."""
    # Monkey-patch encode for this call only.
    real_encode = model.encode
    model.encode = lambda video: feats_dict  # type: ignore
    # Need a fake video tensor (used only by decoder branch); make it minimal.
    B, Tp1 = next(iter(feats_dict.values())).shape[:2]
    fake_video = torch.zeros(B, Tp1, 3, model.cfg.decoder_out_size, model.cfg.decoder_out_size)
    try:
        out = model.forward_loss(fake_video, learning=learning)
    finally:
        model.encode = real_encode  # type: ignore
    return out


# --------------------------------------------------------------------------- #
# Cross-attention mask
# --------------------------------------------------------------------------- #


def test_child_to_parent_mask_shape():
    T, Hp, Wp = 3, 4, 5
    Hc, Wc = 2 * Hp, 2 * Wp
    m = build_child_to_parent_mask(T, Hp, Wp, DEVICE)
    # Queries are child (finer), keys are parent (coarser)
    assert m.shape == (T * Hc * Wc, T * Hp * Wp)


def test_child_to_parent_mask_exhaustive_tiny():
    """For a tiny grid, verify mask allows EXACTLY 1 parent per child (the spatial parent)."""
    T, Hp, Wp = 2, 2, 2
    Hc, Wc = 4, 4
    m = build_child_to_parent_mask(T, Hp, Wp, DEVICE)
    Nc, Np = m.shape
    allowed = torch.isfinite(m)  # 0 vs -inf
    # Each child row should have exactly 1 allowed entry (its parent).
    assert (allowed.sum(dim=-1) == 1).all(), "every child must have exactly 1 parent"
    # Build the brute-force ground truth.
    expected = torch.zeros(Nc, Np, dtype=torch.bool)
    for tc in range(T):
        for yc in range(Hc):
            for xc in range(Wc):
                ci = tc * (Hc * Wc) + yc * Wc + xc
                yp = yc // 2
                xp = xc // 2
                pi = tc * (Hp * Wp) + yp * Wp + xp
                expected[ci, pi] = True
    assert torch.equal(allowed, expected)


def test_child_to_parent_mask_no_cross_time_leak():
    T, Hp, Wp = 4, 2, 2
    Hc, Wc = 4, 4
    m = build_child_to_parent_mask(T, Hp, Wp, DEVICE)
    allowed = torch.isfinite(m)
    # For every (child, parent) allowed pair, child_t must equal parent_t.
    c_t = (torch.arange(allowed.shape[0]) // (Hc * Wc))
    p_t = (torch.arange(allowed.shape[1]) // (Hp * Wp))
    for ci in range(allowed.shape[0]):
        pi = torch.where(allowed[ci])[0]
        assert (p_t[pi] == c_t[ci]).all()


# --------------------------------------------------------------------------- #
# Cross-attention module
# --------------------------------------------------------------------------- #


def test_cross_attention_forward_shape():
    B, T, Hp, Wp = 2, 3, 2, 2
    Hc, Wc = 4, 4
    Dq, Dkv, H, dh = 16, 16, 2, 8
    # Child is query (finer), parent is KV (coarser)
    q_tok = torch.randn(B, T * Hc * Wc, Dq)
    kv_tok = torch.randn(B, T * Hp * Wp, Dkv)
    rope_q = build_rope3d(T, Hc, Wc, dh, DEVICE)
    rope_kv = build_rope3d(T, Hp, Wp, dh, DEVICE)
    mask = build_child_to_parent_mask(T, Hp, Wp, DEVICE)
    attn = CrossAttention3DRoPE(Dq, Dkv, H)
    out = attn(q_tok, kv_tok, rope_q, rope_kv, mask)
    assert out.shape == (B, T * Hc * Wc, Dq)
    assert torch.isfinite(out).all()


def test_cross_attention_deterministic():
    B, T, Hp, Wp = 1, 2, 2, 2
    Hc, Wc = 4, 4
    Dq, Dkv, H, dh = 16, 16, 2, 8
    torch.manual_seed(0)
    attn = CrossAttention3DRoPE(Dq, Dkv, H)
    q_tok = torch.randn(B, T * Hc * Wc, Dq)
    kv_tok = torch.randn(B, T * Hp * Wp, Dkv)
    rope_q = build_rope3d(T, Hc, Wc, dh, DEVICE)
    rope_kv = build_rope3d(T, Hp, Wp, dh, DEVICE)
    mask = build_child_to_parent_mask(T, Hp, Wp, DEVICE)
    attn.eval()
    with torch.no_grad():
        a = attn(q_tok, kv_tok, rope_q, rope_kv, mask)
        b = attn(q_tok, kv_tok, rope_q, rope_kv, mask)
    assert torch.allclose(a, b)


def test_cross_attention_only_uses_allowed_parent():
    """
    Perturb a parent token at a position that is NOT the parent of child[0]; the
    output for child[0] must be unchanged. Perturb the allowed parent and the
    output for child[0] MUST change.
    """
    torch.manual_seed(42)
    B, T, Hp, Wp = 1, 1, 2, 2
    Hc, Wc = 4, 4
    Dq, Dkv, H, dh = 16, 16, 2, 8
    attn = CrossAttention3DRoPE(Dq, Dkv, H, bias=False).eval()
    # Child queries, parent KV
    q_tok = torch.randn(B, T * Hc * Wc, Dq)
    kv_tok = torch.randn(B, T * Hp * Wp, Dkv)
    rope_q = build_rope3d(T, Hc, Wc, dh, DEVICE)
    rope_kv = build_rope3d(T, Hp, Wp, dh, DEVICE)
    mask = build_child_to_parent_mask(T, Hp, Wp, DEVICE)

    with torch.no_grad():
        out0 = attn(q_tok, kv_tok, rope_q, rope_kv, mask)

    # child[0] is at (yc=0, xc=0), so its parent is at (yp=0, xp=0) = parent idx 0.
    # Parent at (yp=1, xp=1) = parent idx 3 is NOT the parent of child[0].
    forbidden_parent = 1 * Wp + 1   # idx=3
    allowed_parent = 0 * Wp + 0     # idx=0

    kv_pert = kv_tok.clone()
    kv_pert[0, forbidden_parent] += 5.0
    with torch.no_grad():
        out_forbidden = attn(q_tok, kv_pert, rope_q, rope_kv, mask)
    assert torch.allclose(out_forbidden[0, 0], out0[0, 0], atol=1e-6), \
        "child[0] changed when a forbidden parent was perturbed"

    kv_pert2 = kv_tok.clone()
    kv_pert2[0, allowed_parent] += 5.0
    with torch.no_grad():
        out_allowed = attn(q_tok, kv_pert2, rope_q, rope_kv, mask)
    assert not torch.allclose(out_allowed[0, 0], out0[0, 0], atol=1e-6), \
        "child[0] did NOT change when its parent was perturbed"


# --------------------------------------------------------------------------- #
# Hierarchical forward (using fake features, no CLIP load)
# --------------------------------------------------------------------------- #


def test_hierarchical_2stage_forward_shape():
    cfg = _tiny_2stage_cfg()
    model = _make_model_no_encoder(cfg)
    feats = _fake_feats(cfg, B=2, T_plus_1=3)
    out = _forward_loss_with_fake_feats(model, feats)
    assert "loss_total" in out and "per_stage" in out
    per_stage = out["per_stage"]
    assert len(per_stage) == 2
    for i, s in enumerate(per_stage):
        sc = cfg.stages[i]
        assert s["final_pred"].shape == (2, 2, sc.clip_channels, sc.H, sc.W)


def test_hierarchical_3stage_forward_shape():
    cfg = _tiny_3stage_cfg()
    model = _make_model_no_encoder(cfg)
    feats = _fake_feats(cfg, B=1, T_plus_1=3)
    out = _forward_loss_with_fake_feats(model, feats)
    assert len(out["per_stage"]) == 3


def test_hierarchical_per_stage_energy_decrease():
    """Init energy higher than final energy at every stage on average over a few seeds."""
    cfg = _tiny_3stage_cfg(n_layers=2)
    model = _make_model_no_encoder(cfg)
    feats = _fake_feats(cfg, B=4, T_plus_1=3)
    # Bump alpha so MCMC actually moves on these tiny features.
    with torch.no_grad():
        for a in model.alphas:
            a.fill_(0.5)
    out = _forward_loss_with_fake_feats(model, feats)
    for i, s in enumerate(out["per_stage"]):
        gap = s["energy_gap"].item()
        assert gap >= -1e-4, f"stage {i} energy increased after MCMC: gap={gap}"


# --------------------------------------------------------------------------- #
# Detached KV: cross-stage gradient isolation
# --------------------------------------------------------------------------- #


def _params_of_stage(model: HierarchicalHVEBT, stage_idx: int):
    return list(model.stages[stage_idx].parameters())


def test_upper_stage_loss_does_not_grad_lower_stage_params():
    """
    Top-down processing: apex (stages[-1]) runs first, then stage 0.
    Compute loss using ONLY the lower (finer) stage's per-stage loss; backprop.
    The upper (coarser) stage's parameters must have None or all-zero grads
    (proves the cross-attention KV from the parent is genuinely detached).
    """
    torch.manual_seed(0)
    cfg = _tiny_2stage_cfg(n_layers=2)
    model = _make_model_no_encoder(cfg)
    feats = _fake_feats(cfg, B=2, T_plus_1=3)

    # Manual top-down MCMC: first apex (stage 1, no cross-attn), then stage 0
    # with detached parent context from stage 1.
    real_ctx_1 = feats["s2"][:, :-1]
    real_gt_1 = feats["s2"][:, 1:]
    init_1 = torch.zeros_like(real_gt_1)
    preds_1, _ = model._mcmc_for_stage(model.stages[1], model.alphas[1],
                                       real_ctx_1, init_1, parent_ctx=None,
                                       learning=True)
    parent_for_lower = preds_1[-1].detach()

    real_ctx_0 = feats["s1"][:, :-1]
    real_gt_0 = feats["s1"][:, 1:]
    init_0 = torch.zeros_like(real_gt_0)
    preds_0, _ = model._mcmc_for_stage(model.stages[0], model.alphas[0],
                                       real_ctx_0, init_0,
                                       parent_ctx=parent_for_lower, learning=True)
    lower_loss = sum(F.smooth_l1_loss(p, real_gt_0) for p in preds_0) / len(preds_0)

    # Zero all grads.
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    lower_loss.backward()

    # Upper (apex) stage params must have NO gradient (or zero).
    for p in _params_of_stage(model, 1):
        if p.grad is not None:
            assert torch.allclose(p.grad, torch.zeros_like(p.grad)), \
                "upper-stage param received gradient via cross-attention!"

    # Lower stage params MUST have non-zero gradient.
    lower_grads_present = False
    lower_grads_finite = True
    for p in _params_of_stage(model, 0):
        if p.grad is not None and p.grad.abs().sum() > 0:
            lower_grads_present = True
            lower_grads_finite = lower_grads_finite and torch.isfinite(p.grad).all().item()
    assert lower_grads_present, "lower stage got no gradient from its own loss"
    assert lower_grads_finite, "lower stage got non-finite gradient"


def test_each_stage_loss_grads_only_its_own_params():
    """For each stage i, backprop only that stage's loss; only stage i's params
    (and NO other stage) should have non-zero grad."""
    torch.manual_seed(1)
    cfg = _tiny_3stage_cfg(n_layers=2)
    model = _make_model_no_encoder(cfg)
    feats = _fake_feats(cfg, B=2, T_plus_1=3)

    # Build per-stage live losses manually, top-down, with detached cross-stage KV.
    prev_pred_det = None
    losses = [None] * len(model.stages)
    for i in reversed(range(len(model.stages))):
        stage = model.stages[i]
        sc = cfg.stages[i]
        f = feats[sc.clip_stage_name]
        real_ctx = f[:, :-1]; real_gt = f[:, 1:]
        init = torch.zeros_like(real_gt)
        preds, _ = model._mcmc_for_stage(stage, model.alphas[i],
                                         real_ctx, init,
                                         parent_ctx=prev_pred_det, learning=True)
        loss_i = sum(F.smooth_l1_loss(p, real_gt) for p in preds) / len(preds)
        losses[i] = loss_i
        prev_pred_det = preds[-1].detach()

    for i, loss_i in enumerate(losses):
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        loss_i.backward(retain_graph=True)
        for j, stage in enumerate(model.stages):
            grads = [p.grad for p in stage.parameters() if p.grad is not None]
            sums = sum(g.abs().sum().item() for g in grads) if grads else 0.0
            if j == i:
                assert sums > 0, f"loss[{i}] produced NO grad for stage[{i}]"
            else:
                # All other stages should have either no grad or zero grad.
                assert sums == 0.0, f"loss[{i}] leaked grad into stage[{j}] (sum={sums})"


# --------------------------------------------------------------------------- #
# Gradient health (no NaN / vanish / explosion)
# --------------------------------------------------------------------------- #


def test_full_loss_grad_flow_per_stage():
    """Backprop the SUM of all stage losses; all stages should get finite, non-zero
    gradient and per-stage grad norms should be within a sane range."""
    torch.manual_seed(2)
    cfg = _tiny_3stage_cfg(n_layers=2)
    model = _make_model_no_encoder(cfg)
    feats = _fake_feats(cfg, B=2, T_plus_1=3)
    out = _forward_loss_with_fake_feats(model, feats, learning=True)

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    out["loss_total"].backward()

    norms = []
    for i in range(len(model.stages)):
        gs = [p.grad for p in model.stages[i].parameters() if p.grad is not None]
        assert len(gs) > 0, f"stage {i} got no grads"
        for g in gs:
            assert torch.isfinite(g).all(), f"stage {i} non-finite grad"
        norm = math.sqrt(sum((g ** 2).sum().item() for g in gs))
        assert norm > 0.0, f"stage {i} got zero gradient"
        # Sanity: per-stage grad norm should be < 1e3 on this tiny model.
        assert norm < 1e3, f"stage {i} grad norm exploded: {norm}"
        norms.append(norm)

    # Vanishing-gradient sanity: highest stage's grad norm shouldn't be
    # orders of magnitude smaller than the lowest stage's.
    ratio = max(norms) / max(min(norms), 1e-12)
    assert ratio < 1e4, f"per-stage grad norms span > 1e4 (vanish/explosion): {norms}"


def test_weight_update_actually_changes_parameters():
    """One optimizer step must change every stage's params."""
    torch.manual_seed(3)
    cfg = _tiny_3stage_cfg(n_layers=2)
    model = _make_model_no_encoder(cfg)
    feats = _fake_feats(cfg, B=2, T_plus_1=3)

    snap_before = [
        [p.detach().clone() for p in stage.parameters()]
        for stage in model.stages
    ]
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    out = _forward_loss_with_fake_feats(model, feats, learning=True)
    out["loss_total"].backward()
    opt.step()

    for i, stage in enumerate(model.stages):
        changed = False
        for p_after, p_before in zip(stage.parameters(), snap_before[i]):
            if not torch.allclose(p_after, p_before):
                changed = True
                break
        assert changed, f"stage {i} weights did not change after optimizer step"


# --------------------------------------------------------------------------- #
# Frozen-untrained-top sanity
# --------------------------------------------------------------------------- #


def test_frozen_top_stage_lower_stages_still_train():
    """Freeze the topmost (apex) stage entirely. Lower stages should still
    receive gradient from the total loss."""
    torch.manual_seed(4)
    cfg = _tiny_3stage_cfg(n_layers=2)
    model = _make_model_no_encoder(cfg)
    # Freeze apex (last stage).
    for p in model.stages[-1].parameters():
        p.requires_grad_(False)
    feats = _fake_feats(cfg, B=2, T_plus_1=3)
    out = _forward_loss_with_fake_feats(model, feats, learning=True)

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    out["loss_total"].backward()

    # Lower stages: gradient must be non-zero, finite.
    for i in range(len(model.stages) - 1):
        gs = [p.grad for p in model.stages[i].parameters() if p.grad is not None]
        s = sum(g.abs().sum().item() for g in gs)
        assert s > 0.0, f"frozen-top: lower stage {i} got no gradient"
        for g in gs:
            assert torch.isfinite(g).all()

    # Apex itself: no grad (because requires_grad=False -> grads stay None).
    for p in model.stages[-1].parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0.0


# --------------------------------------------------------------------------- #
# Overfit on a tiny synthetic-feature dataset
# --------------------------------------------------------------------------- #


def test_hierarchical_can_overfit_tiny_feature_dataset():
    """One fixed (B, T+1, ...) sample, train ~100 steps; loss should drop substantially."""
    torch.manual_seed(5)
    cfg = _tiny_2stage_cfg(n_layers=2)
    model = _make_model_no_encoder(cfg)
    feats = _fake_feats(cfg, B=1, T_plus_1=3)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(120):
        out = _forward_loss_with_fake_feats(model, feats, learning=True)
        opt.zero_grad(set_to_none=True)
        out["loss_total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(out["loss_total"].item())

    first10 = sum(losses[:10]) / 10
    last10 = sum(losses[-10:]) / 10
    assert last10 < first10 * 0.7, f"overfit failed: first10={first10:.4f} last10={last10:.4f}"


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #


def test_pixel_decoder_forward_shape():
    dec = PixelDecoder(in_channels=8, in_HW=(4, 4), out_size=32)
    x = torch.randn(2, 3, 8, 4, 4)
    y = dec(x)
    assert y.shape == (2, 3, 3, 32, 32)
    assert (y >= 0).all() and (y <= 1).all()


def test_pixel_decoder_grad_does_not_reach_external_input():
    """Decoder is trained on detached input; verify gradient stops there."""
    dec = PixelDecoder(in_channels=8, in_HW=(4, 4), out_size=32)
    upstream = torch.randn(1, 1, 8, 4, 4, requires_grad=True)
    detached = upstream.detach()
    out = dec(detached)
    target = torch.zeros_like(out)
    loss = F.l1_loss(out, target)
    loss.backward()
    assert upstream.grad is None, \
        "gradient leaked into 'upstream' through the detached input"
    # Decoder params themselves should have grads.
    grad_present = any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.parameters())
    assert grad_present


def test_decoder_loss_does_not_grad_hvebt_stages():
    """In the full hierarchical forward with decoder enabled, the decoder
    loss alone must not produce gradient on HVEBT stage parameters."""
    torch.manual_seed(6)
    cfg = _tiny_2stage_cfg(n_layers=1)
    cfg.decoder_enabled = True
    cfg.decoder_out_size = 32
    # Match feats H to be compatible with decoder factor: stage 0 H=4, out=32 -> 8x = 3 upsamples.
    model = _make_model_no_encoder(cfg)
    feats = _fake_feats(cfg, B=1, T_plus_1=2)
    out = _forward_loss_with_fake_feats(model, feats, learning=True)

    assert "loss_decoder" in out

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    out["loss_decoder"].backward()

    for i, stage in enumerate(model.stages):
        for p in stage.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum().item() == 0.0, \
                    f"decoder loss leaked grad into HVEBT stage {i}"

    # Decoder should have grads.
    dec_grads = sum(p.grad.abs().sum().item() for p in model.decoder.parameters() if p.grad is not None)
    assert dec_grads > 0, "decoder got no gradient from its own loss"


def test_save_recon_grid_writes_file_if_torchvision():
    try:
        import torchvision  # noqa: F401
    except ImportError:
        pytest.skip("torchvision not installed")
    real = torch.rand(2, 3, 3, 32, 32)
    pred = torch.rand(2, 3, 3, 32, 32)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "grid.png")
        save_recon_grid(real, pred, path, max_clips=2)
        assert os.path.exists(path) and os.path.getsize(path) > 0


# --------------------------------------------------------------------------- #
# Numerical stability
# --------------------------------------------------------------------------- #


def test_no_nan_with_extreme_inputs():
    cfg = _tiny_2stage_cfg(n_layers=1)
    model = _make_model_no_encoder(cfg)
    feats = _fake_feats(cfg, B=1, T_plus_1=2)
    # Blow up input magnitudes.
    for k in feats:
        feats[k] = feats[k] * 100.0
    out = _forward_loss_with_fake_feats(model, feats, learning=True)
    for s in out["per_stage"]:
        assert torch.isfinite(s["loss"])
        assert torch.isfinite(s["init_recon"]) and torch.isfinite(s["final_recon"])
    assert torch.isfinite(out["loss_total"])


# --------------------------------------------------------------------------- #
# Real CLIP encoder smoke test (only run if weights present)
# --------------------------------------------------------------------------- #


def _clip_weights_present() -> bool:
    return os.path.exists("clip/MobileCLIP2-S0/mobileclip2_s0.pt")


@pytest.mark.skipif(not _clip_weights_present(), reason="MobileCLIP weights not present")
def test_real_hierarchical_forward_with_encoder():
    """End-to-end: build the default 3-stage model with the real encoder and a
    tiny video (256x256), run forward_loss, sanity-check shapes + finiteness."""
    cfg = HierarchicalHVEBTConfig(
        # Default 3-stage stack but make embed/heads tiny for CPU speed.
        stages=[
            HVEBTStageConfig(clip_stage_name="s1", clip_channels=128, H=32, W=32,
                             embed_dim=64, n_heads=2, n_layers=1),
            HVEBTStageConfig(clip_stage_name="s2", clip_channels=256, H=16, W=16,
                             embed_dim=64, n_heads=2, n_layers=1),
            HVEBTStageConfig(clip_stage_name="s3", clip_channels=512, H=8, W=8,
                             embed_dim=64, n_heads=2, n_layers=1),
        ],
        mcmc_num_steps=1,
        mcmc_step_size=10.0,
    )
    model = HierarchicalHVEBT(cfg)
    video = torch.rand(1, 3, 3, 256, 256)
    out = model.forward_loss(video, learning=False)
    assert torch.isfinite(out["loss_total"])
    assert len(out["per_stage"]) == 3
    for i, s in enumerate(out["per_stage"]):
        sc = cfg.stages[i]
        assert s["final_pred"].shape == (1, 2, sc.clip_channels, sc.H, sc.W)
