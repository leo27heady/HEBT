"""
Tests for Hierarchical Video EBT Phase 0 (CLIP encoder) + Phase 1 (single-stage EBT).

Run with venv activated:
    python tests/test_hvebt.py

All tests run on CPU with tiny shapes for speed, except the encoder shape test
which uses 256x256 (matching planned training resolution).
"""
from __future__ import annotations

import math
import os
import sys
import traceback

import torch

# Make project importable when run as a script from repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.vid.hvebt.clip_encoder import MobileClipMultiStageEncoder  # noqa: E402
from model.vid.hvebt.hvebt import (  # noqa: E402
    HVEBT,
    HVEBTConfig,
    HVEBTStage,
    HVEBTStageConfig,
    build_block_causal_mask,
)
from model.vid.hvebt.positional import (  # noqa: E402
    apply_rope3d,
    build_rope3d,
    split_head_dim,
)


_PASS = []
_FAIL = []


def _case(name):
    def deco(fn):
        def wrapped():
            try:
                fn()
                print(f"[PASS] {name}")
                _PASS.append(name)
            except AssertionError as e:
                print(f"[FAIL] {name}: {e}")
                _FAIL.append((name, str(e)))
                traceback.print_exc()
            except Exception as e:
                print(f"[ERROR] {name}: {type(e).__name__}: {e}")
                _FAIL.append((name, f"{type(e).__name__}: {e}"))
                traceback.print_exc()
        wrapped.__name__ = fn.__name__
        return wrapped
    return deco


# --------------------------------------------------------------------------- #
#  Positional / RoPE
# --------------------------------------------------------------------------- #


@_case("positional.split_head_dim sums and is even per axis")
def test_split_head_dim():
    for d in (48, 64, 96, 192, 384):
        dt, dy, dx = split_head_dim(d)
        assert dt + dy + dx == d
        assert dt % 2 == 0 and dy % 2 == 0 and dx % 2 == 0
        assert min(dt, dy, dx) >= 2


@_case("positional.build_rope3d shapes")
def test_rope_shapes():
    cache = build_rope3d(T=4, H=8, W=8, head_dim=64, device=torch.device("cpu"))
    assert cache.cos.shape == (4 * 8 * 8, 64)
    assert cache.sin.shape == (4 * 8 * 8, 64)


@_case("positional.apply_rope3d preserves norms")
def test_rope_preserves_norm():
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 32, 64
    cache = build_rope3d(T=2, H=4, W=4, head_dim=D, device=torch.device("cpu"))
    x = torch.randn(B, H, N, D)
    y = apply_rope3d(x, cache)
    # RoPE is an orthogonal rotation per pair; norms (per vector) must be preserved.
    nx = x.norm(dim=-1)
    ny = y.norm(dim=-1)
    assert torch.allclose(nx, ny, atol=1e-5), f"norm drift max={(nx - ny).abs().max()}"


@_case("positional.apply_rope3d produces different outputs at different positions")
def test_rope_position_sensitive():
    torch.manual_seed(0)
    D = 48
    cache = build_rope3d(T=2, H=2, W=2, head_dim=D, device=torch.device("cpu"))
    # Put the SAME vector at every position. RoPE should produce different outputs.
    x = torch.randn(1, 1, 1, D).expand(1, 1, 2 * 2 * 2, D).clone()
    y = apply_rope3d(x, cache)
    # Row 0 is position (0,0,0); most frequencies produce 0 phase -> identical. Compare
    # row 0 to row 7 (last position) - must differ.
    diff = (y[0, 0, 0] - y[0, 0, -1]).abs().max()
    assert diff > 1e-3, f"RoPE did not vary across positions (max diff={diff})"


# --------------------------------------------------------------------------- #
#  Attention mask
# --------------------------------------------------------------------------- #


@_case("block-causal mask is correct")
def test_block_causal_mask():
    T, HW = 3, 4
    m = build_block_causal_mask(T, HW, device=torch.device("cpu"))
    N = T * HW
    assert m.shape == (N, N)
    # Derive frame index
    f = torch.arange(N) // HW
    for q in range(N):
        for k in range(N):
            allowed = f[k].item() <= f[q].item()
            if allowed:
                assert m[q, k].item() == 0.0, f"({q},{k}) should be allowed"
            else:
                assert math.isinf(m[q, k].item()) and m[q, k].item() < 0, f"({q},{k}) should be -inf"


# --------------------------------------------------------------------------- #
#  HVEBTStage forward
# --------------------------------------------------------------------------- #


def _tiny_stage_cfg():
    return HVEBTStageConfig(
        clip_stage_name="final",
        clip_channels=16,
        H=4, W=4,
        embed_dim=48,    # head_dim=12, splits (4,4,4) all even
        n_heads=4,
        n_layers=2,
        ffn_mult=2.0,
    )


@_case("HVEBTStage forward shape")
def test_stage_forward_shape():
    torch.manual_seed(0)
    cfg = _tiny_stage_cfg()
    stage = HVEBTStage(cfg)
    B, T = 2, 3
    real = torch.randn(B, T, cfg.clip_channels, cfg.H, cfg.W)
    pred = torch.randn(B, T, cfg.clip_channels, cfg.H, cfg.W)
    e = stage(real, pred)
    assert e.shape == (B, T * cfg.H * cfg.W), e.shape


@_case("HVEBTStage energy depends on pred (zero-init head -> non-zero after tiny perturb of head weights)")
def test_stage_energy_nonzero_after_perturb():
    torch.manual_seed(0)
    cfg = _tiny_stage_cfg()
    stage = HVEBTStage(cfg)
    # With zero-init energy head, energies start at 0. Tiny random init is more realistic.
    with torch.no_grad():
        stage.energy_head.weight.normal_(0, 0.02)
        stage.energy_head.bias.normal_(0, 0.02)
    B, T = 2, 3
    real = torch.randn(B, T, cfg.clip_channels, cfg.H, cfg.W)
    p1 = torch.randn(B, T, cfg.clip_channels, cfg.H, cfg.W)
    p2 = torch.randn(B, T, cfg.clip_channels, cfg.H, cfg.W)
    e1 = stage(real, p1)
    e2 = stage(real, p2)
    assert not torch.allclose(e1, e2), "energy should depend on pred"


@_case("HVEBTStage respects block-causal: frame t tokens unaffected by frame >t predictions")
def test_stage_causality():
    torch.manual_seed(0)
    cfg = _tiny_stage_cfg()
    # Randomize energy head weights for a nontrivial dependency
    stage = HVEBTStage(cfg)
    with torch.no_grad():
        stage.energy_head.weight.normal_(0, 0.1)
    B, T = 1, 4
    real = torch.randn(B, T, cfg.clip_channels, cfg.H, cfg.W)
    pred_a = torch.randn(B, T, cfg.clip_channels, cfg.H, cfg.W)
    pred_b = pred_a.clone()
    # Change ONLY the last frame's predicted features
    pred_b[:, -1] = torch.randn(B, cfg.clip_channels, cfg.H, cfg.W)

    e_a = stage(real, pred_a).reshape(B, T, cfg.H * cfg.W)
    e_b = stage(real, pred_b).reshape(B, T, cfg.H * cfg.W)
    # Frames 0..T-2 must be identical (causal): they cannot see frame T-1.
    diff_past = (e_a[:, :-1] - e_b[:, :-1]).abs().max().item()
    diff_last = (e_a[:, -1] - e_b[:, -1]).abs().max().item()
    assert diff_past < 1e-6, f"causality violated: past energies changed by {diff_past}"
    assert diff_last > 1e-6, f"last-frame energy should change (got diff {diff_last})"


@_case("HVEBTStage: changing real_ctx at frame t changes energy at frames >= t only")
def test_stage_context_causality():
    torch.manual_seed(0)
    cfg = _tiny_stage_cfg()
    stage = HVEBTStage(cfg)
    with torch.no_grad():
        stage.energy_head.weight.normal_(0, 0.1)
        # Also bump input_proj so there's a real dependency on real_ctx
        stage.input_proj.weight.normal_(0, 0.1)
    B, T = 1, 4
    real_a = torch.randn(B, T, cfg.clip_channels, cfg.H, cfg.W)
    real_b = real_a.clone()
    pred = torch.randn(B, T, cfg.clip_channels, cfg.H, cfg.W)
    # perturb real at frame 2 only
    real_b[:, 2] = torch.randn(B, cfg.clip_channels, cfg.H, cfg.W)
    e_a = stage(real_a, pred).reshape(B, T, cfg.H * cfg.W)
    e_b = stage(real_b, pred).reshape(B, T, cfg.H * cfg.W)
    # Frames 0,1 must be unaffected. Frames 2,3 may differ.
    before = (e_a[:, :2] - e_b[:, :2]).abs().max().item()
    after = (e_a[:, 2:] - e_b[:, 2:]).abs().max().item()
    assert before < 1e-6, f"context causality violated: past changed by {before}"
    assert after > 1e-6, f"future energies should change (got {after})"


# --------------------------------------------------------------------------- #
#  MCMC / full model (no encoder - to keep tests fast)
# --------------------------------------------------------------------------- #


class _HVEBTTiny(HVEBT):
    """Bypasses MobileCLIP encoder load for fast tests."""

    def __init__(self, cfg: HVEBTConfig):
        # Skip parent __init__; build minimal state.
        torch.nn.Module.__init__(self)
        self.cfg = cfg
        self.encoder = None
        self.stage = HVEBTStage(cfg.stage)
        self.alpha = torch.nn.Parameter(
            torch.tensor(float(cfg.mcmc_step_size)),
            requires_grad=cfg.mcmc_step_size_learnable,
        )
        self.langevin_std = float(cfg.langevin_noise)

    def encode(self, video):  # pragma: no cover
        raise NotImplementedError("tiny model has no encoder")


def _tiny_hvebt_cfg():
    return HVEBTConfig(
        stage=_tiny_stage_cfg(),
        mcmc_num_steps=3,
        mcmc_step_size=1e-2,
        mcmc_step_size_learnable=True,
        langevin_noise=0.0,
        denoising_init="zeros",
        truncate_mcmc=False,
    )


@_case("MCMC: energy decreases on average during inner loop on random batch (after a few warmup updates)")
def test_mcmc_energy_decreases():
    torch.manual_seed(0)
    cfg = _tiny_hvebt_cfg()
    cfg.mcmc_num_steps = 8
    cfg.mcmc_step_size = 1.0
    model = _HVEBTTiny(cfg)
    # Break zero-init of energy head so gradients are nonzero
    with torch.no_grad():
        model.stage.energy_head.weight.normal_(0, 0.1)
        model.stage.energy_head.bias.normal_(0, 0.1)

    B, T = 2, 3
    scfg = cfg.stage
    real = torch.randn(B, T, scfg.clip_channels, scfg.H, scfg.W)
    init = torch.randn_like(real)
    preds, energies = model.mcmc(real, init, learning=False)
    e = torch.stack([en.mean() for en in energies]).detach()
    # Energy at final step should be <= energy at first step (gradient descent on energy).
    assert e[-1].item() <= e[0].item() + 1e-4, f"energy increased: {e.tolist()}"


@_case("MCMC: preds gradient links back to alpha (learnable step size)")
def test_mcmc_alpha_grad():
    torch.manual_seed(0)
    cfg = _tiny_hvebt_cfg()
    model = _HVEBTTiny(cfg)
    with torch.no_grad():
        model.stage.energy_head.weight.normal_(0, 0.1)
    B, T = 1, 2
    scfg = cfg.stage
    real = torch.randn(B, T, scfg.clip_channels, scfg.H, scfg.W)
    init = torch.randn_like(real)
    preds, _ = model.mcmc(real, init, learning=True)
    loss = preds[-1].pow(2).mean()
    loss.backward()
    assert model.alpha.grad is not None and torch.isfinite(model.alpha.grad).all(), "alpha should receive grad"
    # At least one stage weight must receive grad too
    any_grad = any(
        (p.grad is not None and p.grad.abs().sum().item() > 0)
        for p in model.stage.parameters()
    )
    assert any_grad, "no stage parameter received gradient"


@_case("Training: loss decreases after a few optimizer steps on fixed random batch")
def test_overfit_tiny():
    torch.manual_seed(0)
    cfg = _tiny_hvebt_cfg()
    cfg.mcmc_num_steps = 3
    cfg.mcmc_step_size = 10.0
    cfg.mcmc_step_size_learnable = False
    model = _HVEBTTiny(cfg)

    B, T = 2, 3
    scfg = cfg.stage
    real_ctx = torch.randn(B, T, scfg.clip_channels, scfg.H, scfg.W)
    real_gt = torch.randn(B, T, scfg.clip_channels, scfg.H, scfg.W)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step():
        opt.zero_grad()
        init_pred = torch.zeros_like(real_gt)
        preds, _ = model.mcmc(real_ctx, init_pred, learning=True)
        loss = 0.0
        K = len(preds)
        for i, p in enumerate(preds):
            loss = loss + torch.nn.functional.smooth_l1_loss(p, real_gt) / K
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        return loss.item(), float(gn)

    losses = [step()[0] for _ in range(40)]
    assert math.isfinite(losses[-1]), f"non-finite loss: {losses}"
    assert losses[-1] < losses[0], f"loss did not decrease: start={losses[0]:.4f} end={losses[-1]:.4f}"
    # Rough improvement
    assert losses[-1] < 0.9 * losses[0], f"loss barely improved: {losses[0]:.4f} -> {losses[-1]:.4f}"


@_case("Gradient flow: per-layer grad norms are finite and non-vanishing")
def test_grad_flow_layers():
    torch.manual_seed(0)
    cfg = _tiny_hvebt_cfg()
    cfg.mcmc_num_steps = 2
    model = _HVEBTTiny(cfg)
    with torch.no_grad():
        model.stage.energy_head.weight.normal_(0, 0.1)

    B, T = 2, 3
    scfg = cfg.stage
    real_ctx = torch.randn(B, T, scfg.clip_channels, scfg.H, scfg.W)
    real_gt = torch.randn(B, T, scfg.clip_channels, scfg.H, scfg.W)
    init_pred = torch.zeros_like(real_gt)
    preds, _ = model.mcmc(real_ctx, init_pred, learning=True)
    loss = sum(torch.nn.functional.smooth_l1_loss(p, real_gt) for p in preds) / len(preds)
    loss.backward()

    per_block = []
    for i, blk in enumerate(model.stage.blocks):
        gsum = 0.0
        gabs = 0.0
        for p in blk.parameters():
            if p.grad is not None:
                gsum += p.grad.norm(2).item() ** 2
                gabs += p.grad.abs().sum().item()
        per_block.append((i, gsum ** 0.5, gabs))
    print("   per-block grad L2 norms:", [(i, f"{n:.3e}") for i, n, _ in per_block])
    for i, n, gabs in per_block:
        assert math.isfinite(n), f"block {i} grad norm non-finite"
        assert gabs > 0.0, f"block {i} received zero gradient"
    # no explosion
    max_norm = max(n for _, n, _ in per_block)
    assert max_norm < 1e3, f"grad norm explosion: {max_norm}"


@_case("Frozen encoder: parameters have requires_grad=False")
def test_encoder_frozen():
    import open_clip  # noqa: F401  - ensures available
    try:
        enc = MobileClipMultiStageEncoder()
    except FileNotFoundError as e:
        raise AssertionError(f"weights missing: {e}")
    n_total = sum(1 for _ in enc.parameters())
    n_frozen = sum(1 for p in enc.parameters() if not p.requires_grad)
    assert n_total == n_frozen, f"{n_total - n_frozen} params still trainable"
    # eval mode sticky
    enc.train(True)
    assert not enc.training, "encoder must stay in eval mode"


@_case("Encoder produces expected stage shapes at 256x256")
def test_encoder_shapes_256():
    enc = MobileClipMultiStageEncoder()
    x = torch.rand(1, 3, 256, 256)
    feats = enc(x)
    expected = {
        "stem": (1, 64, 64, 64),
        "s0":   (1, 64, 64, 64),
        "s1":   (1, 128, 32, 32),
        "s2":   (1, 256, 16, 16),
        "s3":   (1, 512, 8, 8),
        "final": (1, 1024, 8, 8),
        "pooled": (1, 512),
    }
    for k, shape in expected.items():
        assert k in feats, f"missing {k}"
        assert tuple(feats[k].shape) == shape, f"{k}: got {tuple(feats[k].shape)}, expected {shape}"


@_case("Encoder is deterministic (same input -> same output)")
def test_encoder_deterministic():
    enc = MobileClipMultiStageEncoder(return_stages=("final",))
    x = torch.rand(1, 3, 256, 256)
    a = enc(x)["final"]
    b = enc(x)["final"]
    assert torch.allclose(a, b), "encoder not deterministic"


@_case("End-to-end: HVEBT.forward_loss runs on fake 64x64 video (small resolution)")
def test_full_model_forward_small():
    # Use 64x64 video so the encoder runs faster. Real training would use 256.
    cfg = HVEBTConfig(
        stage=HVEBTStageConfig(
            clip_stage_name="final",
            clip_channels=1024,
            H=2, W=2,    # stride 32 on 64x64 input -> 2x2
            embed_dim=64,
            n_heads=4,
            n_layers=2,
        ),
        mcmc_num_steps=2,
        mcmc_step_size=1.0,
    )
    model = HVEBT(cfg)
    B, T1 = 1, 3
    video = torch.rand(B, T1, 3, 64, 64)
    out = model.forward_loss(video, learning=True)
    loss = out["loss"]
    assert torch.isfinite(loss), "loss not finite"
    loss.backward()
    any_grad = any(
        (p.grad is not None and p.grad.abs().sum().item() > 0)
        for p in model.stage.parameters()
    )
    assert any_grad, "stage received no gradient"
    # Encoder must have zero grads
    for p in model.encoder.parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0.0, "encoder got grad"


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #


def main():
    tests = [
        # positional
        test_split_head_dim,
        test_rope_shapes,
        test_rope_preserves_norm,
        test_rope_position_sensitive,
        # mask
        test_block_causal_mask,
        # stage
        test_stage_forward_shape,
        test_stage_energy_nonzero_after_perturb,
        test_stage_causality,
        test_stage_context_causality,
        # mcmc
        test_mcmc_energy_decreases,
        test_mcmc_alpha_grad,
        test_overfit_tiny,
        test_grad_flow_layers,
        # encoder (requires weights)
        test_encoder_frozen,
        test_encoder_shapes_256,
        test_encoder_deterministic,
        test_full_model_forward_small,
    ]
    for t in tests:
        t()
    print("\n=== RESULTS ===")
    print(f"passed: {len(_PASS)}")
    print(f"failed: {len(_FAIL)}")
    for n, msg in _FAIL:
        print(f"  [FAIL] {n}: {msg}")
    sys.exit(0 if not _FAIL else 1)


if __name__ == "__main__":
    main()
