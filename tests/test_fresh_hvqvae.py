"""
Comprehensive verification tests for Fresh HVQVAE implementation.
Tests alignment with the FRESH_HVQVAE_PLAN.md specification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.optim import Adam

from model.vid.fresh_hvqvae import (
    FreshHVQVAE, FreshHVQVAEConfig,
    HierarchicalEncoder, Decoder, UpscaleMid, UpscaleTop,
    PredictorStage, build_temporal_window_mask,
    build_cross_attn_mask_top_to_mid, build_cross_attn_mask_mid_to_bot,
    build_lfq_codebook_matrix,
)


def test_config():
    """Test config defaults match plan."""
    cfg = FreshHVQVAEConfig()
    assert cfg.C_bot == 128
    assert cfg.C_mid == 192
    assert cfg.C_top == 256
    assert cfg.K_bot == 4096
    assert cfg.K_mid == 1024
    assert cfg.K_top == 512
    assert cfg.lfq_dim_bot == 12
    assert cfg.lfq_dim_mid == 10
    assert cfg.lfq_dim_top == 9
    assert cfg.window_top == -1  # full causal
    assert cfg.window_mid == 2
    assert cfg.window_bot == 1
    print("[PASS] test_config")


def test_encoder_shapes():
    """Test encoder outputs correct spatial shapes per stage."""
    cfg = FreshHVQVAEConfig()
    encoder = HierarchicalEncoder(cfg)
    x = torch.randn(2, 3, 64, 64)
    out = encoder(x)

    assert out['quant_bot'].shape == (2, cfg.C_bot, 16, 16), f"Bot: {out['quant_bot'].shape}"
    assert out['quant_mid'].shape == (2, cfg.C_mid, 4, 4), f"Mid: {out['quant_mid'].shape}"
    assert out['quant_top'].shape == (2, cfg.C_top, 1, 1), f"Top: {out['quant_top'].shape}"
    assert out['idx_bot'].shape == (2, 16, 16), f"Idx bot: {out['idx_bot'].shape}"
    assert out['idx_mid'].shape == (2, 4, 4), f"Idx mid: {out['idx_mid'].shape}"
    assert out['idx_top'].shape == (2, 1, 1), f"Idx top: {out['idx_top'].shape}"

    # VQ losses should be scalars
    assert out['loss_bot'].dim() == 0
    assert out['loss_mid'].dim() == 0
    assert out['loss_top'].dim() == 0
    print("[PASS] test_encoder_shapes")


def test_decoder_shapes():
    """Test decoder and upscaler output shapes."""
    cfg = FreshHVQVAEConfig()
    decoder = Decoder(cfg.C_bot)
    upscale_mid = UpscaleMid(cfg.C_mid, cfg.C_bot)
    upscale_top = UpscaleTop(cfg.C_top, cfg.C_bot)

    # Decoder: (B, C_bot, 16, 16) → (B, 3, 64, 64)
    x_bot = torch.randn(2, cfg.C_bot, 16, 16)
    out = decoder(x_bot)
    assert out.shape == (2, 3, 64, 64), f"Decoder output: {out.shape}"

    # UpscaleMid: (B, C_mid, 4, 4) → (B, C_bot, 16, 16)
    x_mid = torch.randn(2, cfg.C_mid, 4, 4)
    out_mid = upscale_mid(x_mid)
    assert out_mid.shape == (2, cfg.C_bot, 16, 16), f"UpscaleMid: {out_mid.shape}"

    # UpscaleTop: (B, C_top, 1, 1) → (B, C_bot, 16, 16)
    x_top = torch.randn(2, cfg.C_top, 1, 1)
    out_top = upscale_top(x_top)
    assert out_top.shape == (2, cfg.C_bot, 16, 16), f"UpscaleTop: {out_top.shape}"

    # Full path: mid → upscale → decode
    recon_mid = decoder(upscale_mid(x_mid))
    assert recon_mid.shape == (2, 3, 64, 64)

    # Full path: top → upscale → decode
    recon_top = decoder(upscale_top(x_top))
    assert recon_top.shape == (2, 3, 64, 64)
    print("[PASS] test_decoder_shapes")


def test_temporal_window_mask():
    """Test self-attention masks match plan specification."""
    device = torch.device('cpu')

    # Top: S=1, T=4, window=-1 (full causal) → lower triangular
    mask_top = build_temporal_window_mask(4, 1, -1, device)
    expected_top = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ], dtype=torch.bool)
    assert mask_top.equal(expected_top), f"Top mask:\n{mask_top}"

    # Bot: S=256, T=4, window=1 → block diagonal (each frame only sees itself)
    mask_bot = build_temporal_window_mask(4, 256, 1, device)
    assert mask_bot.shape == (1024, 1024)
    # Check frame 2 tokens can attend to frame 2 but NOT frame 1 or 3
    assert mask_bot[512, 512] == True   # frame 2, pos 0 → frame 2, pos 0
    assert mask_bot[512, 256] == False  # frame 2 → frame 1 (blocked by window=1)
    assert mask_bot[512, 768] == False  # frame 2 → frame 3 (causal blocks)

    # Mid: S=16, T=4, window=2
    mask_mid = build_temporal_window_mask(4, 16, 2, device)
    assert mask_mid.shape == (64, 64)
    # Frame 2 sees frames 1,2 but not frame 0
    assert mask_mid[32, 16] == True    # frame 2 → frame 1
    assert mask_mid[32, 32] == True    # frame 2 → frame 2
    assert mask_mid[32, 0] == False    # frame 2 → frame 0 (outside window)
    assert mask_mid[32, 48] == False   # frame 2 → frame 3 (causal)
    print("[PASS] test_temporal_window_mask")


def test_cross_attn_masks():
    """Test cross-attention mask shapes and structure."""
    device = torch.device('cpu')
    T = 4

    # Top → Mid: (T*16, T*1) — mid queries, top keys
    mask_tm = build_cross_attn_mask_top_to_mid(T, device)
    assert mask_tm.shape == (64, 4), f"Top→Mid mask shape: {mask_tm.shape}"
    # All 16 mid tokens at frame 2 attend to top token at frame 2
    assert mask_tm[32:48, 2].all()
    # No cross-frame attention
    assert not mask_tm[32:48, 0].any()
    assert not mask_tm[32:48, 1].any()
    assert not mask_tm[32:48, 3].any()

    # Mid → Bot: (T*256, T*16) — bot queries, mid keys
    mask_mb = build_cross_attn_mask_mid_to_bot(T, device)
    assert mask_mb.shape == (1024, 64), f"Mid→Bot mask shape: {mask_mb.shape}"
    # Bot token at (r=0, c=0) of frame 0 → mid at (0//4, 0//4)=(0,0) = flat idx 0 of frame 0
    assert mask_mb[0, 0] == True
    # Bot token at (r=7, c=7) of frame 0 → mid at (1, 1) = flat idx 5 of frame 0
    bot_idx = 7 * 16 + 7
    parent_idx = (7 // 4) * 4 + (7 // 4)  # (1, 1) → 1*4 + 1 = 5
    assert mask_mb[bot_idx, parent_idx] == True
    # No cross-frame attention
    assert not mask_mb[0:256, 16:].any()  # frame 0 bot → frame 1+ mid
    print("[PASS] test_cross_attn_masks")


def test_lfq_codebook_matrix():
    """Test LFQ codebook matrix structure."""
    # dim=9 → 512 codes
    cb = build_lfq_codebook_matrix(9)
    assert cb.shape == (512, 9)
    # All values should be +1 or -1
    assert ((cb == 1.0) | (cb == -1.0)).all()
    # All codes should be unique
    assert cb.unique(dim=0).shape[0] == 512

    # dim=12 → 4096 codes
    cb12 = build_lfq_codebook_matrix(12)
    assert cb12.shape == (4096, 12)
    assert cb12.unique(dim=0).shape[0] == 4096
    print("[PASS] test_lfq_codebook_matrix")


def test_model_forward_shapes():
    """Test full model forward returns correct loss dict."""
    cfg = FreshHVQVAEConfig(max_T=5)
    model = FreshHVQVAE(cfg)
    batch = torch.randn(2, 5, 3, 64, 64).clamp(0, 1)

    losses = model(batch)

    expected_keys = {'mse_bot', 'mse_mid', 'mse_top', 'vq_loss', 'ce_top', 'ce_mid', 'ce_bot'}
    assert set(losses.keys()) == expected_keys, f"Keys: {losses.keys()}"

    for k, v in losses.items():
        assert v.dim() == 0, f"{k} should be scalar, got shape {v.shape}"
        assert not torch.isnan(v), f"{k} is NaN"
        assert not torch.isinf(v), f"{k} is Inf"
    print("[PASS] test_model_forward_shapes")


def test_gradient_isolation():
    """
    Test that predictor backward does NOT create gradients on encoder params,
    and reconstruction backward does NOT create gradients on predictor params.
    """
    cfg = FreshHVQVAEConfig(max_T=4)
    model = FreshHVQVAE(cfg)
    batch = torch.randn(1, 4, 3, 64, 64).clamp(0, 1)

    losses = model(batch)

    # Test 1: CE losses should not touch encoder
    model.zero_grad()
    losses['ce_top'].backward(retain_graph=True)
    for name, p in model.encoder.named_parameters():
        if p.grad is not None:
            assert (p.grad == 0).all(), f"Encoder param {name} has gradient from ce_top!"
    print("  ce_top does not affect encoder ✓")

    model.zero_grad()
    losses['ce_mid'].backward(retain_graph=True)
    for name, p in model.encoder.named_parameters():
        if p.grad is not None:
            assert (p.grad == 0).all(), f"Encoder param {name} has gradient from ce_mid!"
    print("  ce_mid does not affect encoder ✓")

    model.zero_grad()
    losses['ce_bot'].backward(retain_graph=True)
    for name, p in model.encoder.named_parameters():
        if p.grad is not None:
            assert (p.grad == 0).all(), f"Encoder param {name} has gradient from ce_bot!"
    print("  ce_bot does not affect encoder ✓")

    # Test 2: Recon loss should not touch predictors
    model.zero_grad()
    recon = cfg.weight_mse_bot * losses['mse_bot'] + cfg.weight_mse_mid * losses['mse_mid'] + \
            cfg.weight_mse_top * losses['mse_top'] + losses['vq_loss']
    recon.backward()
    for name, p in model.predictor_top.named_parameters():
        if p.grad is not None:
            assert (p.grad == 0).all(), f"Predictor top param {name} has gradient from recon!"
    for name, p in model.predictor_mid.named_parameters():
        if p.grad is not None:
            assert (p.grad == 0).all(), f"Predictor mid param {name} has gradient from recon!"
    for name, p in model.predictor_bot.named_parameters():
        if p.grad is not None:
            assert (p.grad == 0).all(), f"Predictor bot param {name} has gradient from recon!"
    print("  recon loss does not affect predictors ✓")

    # Test 3: Predictor stages are isolated from each other
    model.zero_grad()
    losses2 = model(batch)
    losses2['ce_mid'].backward(retain_graph=True)
    for name, p in model.predictor_top.named_parameters():
        if p.grad is not None:
            assert (p.grad == 0).all(), f"Predictor top param {name} has gradient from ce_mid!"
    for name, p in model.predictor_bot.named_parameters():
        if p.grad is not None:
            assert (p.grad == 0).all(), f"Predictor bot param {name} has gradient from ce_mid!"
    print("  ce_mid does not affect top/bot predictors ✓")
    print("[PASS] test_gradient_isolation")


def test_no_retain_graph_needed():
    """Test that all 4 backward passes work without retain_graph."""
    cfg = FreshHVQVAEConfig(max_T=4)
    model = FreshHVQVAE(cfg)
    batch = torch.randn(1, 4, 3, 64, 64).clamp(0, 1)

    losses = model(batch)
    recon = cfg.weight_mse_bot * losses['mse_bot'] + cfg.weight_mse_mid * losses['mse_mid'] + \
            cfg.weight_mse_top * losses['mse_top'] + losses['vq_loss']

    # All 4 backwards without retain_graph
    model.zero_grad()
    recon.backward()

    model.zero_grad()
    losses['ce_top'].backward()

    model.zero_grad()
    losses['ce_mid'].backward()

    model.zero_grad()
    losses['ce_bot'].backward()
    print("[PASS] test_no_retain_graph_needed")


def test_encoder_gradient_flow():
    """Test that reconstruction loss reaches encoder parameters via STE."""
    cfg = FreshHVQVAEConfig(max_T=4)
    model = FreshHVQVAE(cfg)
    batch = torch.randn(1, 4, 3, 64, 64).clamp(0, 1)

    losses = model(batch)
    recon = losses['mse_bot']
    model.zero_grad()
    recon.backward()

    # bot encoder should have gradient from mse_bot
    has_grad = False
    for p in model.encoder.enc_to_bot.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "enc_to_bot should receive gradient from mse_bot"
    print("  mse_bot → enc_to_bot gradient ✓")

    # Test mid loss reaches bot encoder (through feedforward chain)
    losses2 = model(batch)
    model.zero_grad()
    losses2['mse_mid'].backward()
    has_grad = False
    for p in model.encoder.enc_to_bot.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "enc_to_bot should receive gradient from mse_mid (through mid encoder)"
    print("  mse_mid → enc_to_bot gradient (via chain) ✓")
    print("[PASS] test_encoder_gradient_flow")


def test_overfit_single_batch():
    """Test that model can overfit a single batch (sanity check)."""
    cfg = FreshHVQVAEConfig(max_T=4)
    model = FreshHVQVAE(cfg)

    # Simple deterministic batch
    torch.manual_seed(42)
    batch = torch.randn(1, 4, 3, 64, 64).clamp(0, 1)

    opt_enc_dec = Adam(model.get_encoder_decoder_params(), lr=3e-3)
    opt_pred_top = Adam(model.get_predictor_top_params(), lr=3e-3)
    opt_pred_mid = Adam(model.get_predictor_mid_params(), lr=3e-3)
    opt_pred_bot = Adam(model.get_predictor_bot_params(), lr=3e-3)

    initial_mse = None
    final_mse = None

    for step in range(50):
        losses = model(batch)
        recon = cfg.weight_mse_bot * losses['mse_bot'] + cfg.weight_mse_mid * losses['mse_mid'] + \
                cfg.weight_mse_top * losses['mse_top'] + losses['vq_loss']

        opt_enc_dec.zero_grad()
        recon.backward()
        opt_enc_dec.step()

        opt_pred_top.zero_grad()
        losses['ce_top'].backward()
        opt_pred_top.step()

        opt_pred_mid.zero_grad()
        losses['ce_mid'].backward()
        opt_pred_mid.step()

        opt_pred_bot.zero_grad()
        losses['ce_bot'].backward()
        opt_pred_bot.step()

        if step == 0:
            initial_mse = losses['mse_bot'].item()
        if step == 49:
            final_mse = losses['mse_bot'].item()

    assert final_mse < initial_mse * 0.85, \
        f"MSE should decrease by >15%: {initial_mse:.4f} → {final_mse:.4f} ({(1-final_mse/initial_mse)*100:.0f}%)"
    print(f"  MSE: {initial_mse:.4f} → {final_mse:.4f} (decreased by {(1-final_mse/initial_mse)*100:.0f}%)")
    print("[PASS] test_overfit_single_batch")


def test_codebook_buffers_on_model():
    """Test that codebook matrices are registered buffers and shared correctly."""
    cfg = FreshHVQVAEConfig()
    model = FreshHVQVAE(cfg)

    # Check buffers exist
    assert 'codebook_top' in dict(model.named_buffers())
    assert 'codebook_mid' in dict(model.named_buffers())
    assert 'codebook_bot' in dict(model.named_buffers())

    # Check shapes
    assert model.codebook_top.shape == (512, 9)
    assert model.codebook_mid.shape == (1024, 10)
    assert model.codebook_bot.shape == (4096, 12)

    # Check shared with predictors
    assert model.predictor_top.codebook_weights is model.codebook_top
    assert model.predictor_mid.codebook_weights is model.codebook_mid
    assert model.predictor_bot.codebook_weights is model.codebook_bot

    # Check not in parameters (should be buffers, not learnable)
    param_names = [n for n, _ in model.named_parameters()]
    assert 'codebook_top' not in param_names
    assert 'codebook_mid' not in param_names
    assert 'codebook_bot' not in param_names
    print("[PASS] test_codebook_buffers_on_model")


def test_predictor_ce_targets():
    """Test that prediction targets are next-frame indices (shift-by-1)."""
    cfg = FreshHVQVAEConfig(max_T=5)
    model = FreshHVQVAE(cfg)
    B, T = 2, 4

    batch = torch.randn(B, T + 1, 3, 64, 64).clamp(0, 1)
    enc = model.encode(batch)

    # Target indices are from frames [1:T+1] (next frames)
    tgt_top = enc['idx_top'][:, 1:].detach()  # (B, T, 1, 1)
    tgt_mid = enc['idx_mid'][:, 1:].detach()  # (B, T, 4, 4)
    tgt_bot = enc['idx_bot'][:, 1:].detach()  # (B, T, 16, 16)

    assert tgt_top.shape == (B, T, 1, 1)
    assert tgt_mid.shape == (B, T, 4, 4)
    assert tgt_bot.shape == (B, T, 16, 16)

    # Input to predictors is from frames [0:T] (current frames)
    inp_top = enc['quant_top'][:, :T]  # (B, T, C_top, 1, 1)
    assert inp_top.shape == (B, T, cfg.C_top, 1, 1)
    print("[PASS] test_predictor_ce_targets")


def test_reconstruction_output_range():
    """Test decoder outputs are in [0, 1] (sigmoid activation)."""
    cfg = FreshHVQVAEConfig(max_T=4)
    model = FreshHVQVAE(cfg)
    batch = torch.randn(1, 4, 3, 64, 64).clamp(0, 1)

    enc = model.encode(batch)
    recon_bot, recon_mid, recon_top = model.reconstruct(enc, 1, 4)

    assert recon_bot.min() >= 0.0 and recon_bot.max() <= 1.0, \
        f"recon_bot range: [{recon_bot.min():.4f}, {recon_bot.max():.4f}]"
    assert recon_mid.min() >= 0.0 and recon_mid.max() <= 1.0
    assert recon_top.min() >= 0.0 and recon_top.max() <= 1.0
    print("[PASS] test_reconstruction_output_range")


if __name__ == '__main__':
    print("=" * 60)
    print("Fresh HVQVAE Verification Tests")
    print("=" * 60)

    test_config()
    test_encoder_shapes()
    test_decoder_shapes()
    test_temporal_window_mask()
    test_cross_attn_masks()
    test_lfq_codebook_matrix()
    test_model_forward_shapes()
    test_codebook_buffers_on_model()
    test_predictor_ce_targets()
    test_reconstruction_output_range()
    test_gradient_isolation()
    test_no_retain_graph_needed()
    test_encoder_gradient_flow()
    test_overfit_single_batch()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
