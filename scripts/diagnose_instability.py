"""
Diagnostic script: investigates VQ-HVEBT training instability.

This script instruments a single forward+backward pass to measure:
1. Feature magnitudes at each stage of the pipeline
2. Gradient magnitudes for each component (encoder, codebook, predictor)
3. MCMC dynamics (energy scale, gradient scale per step)
4. The second-order gradient amplification through MCMC unrolling
5. Interaction between commitment loss gradients and prediction loss gradients
6. Float16 vs float32 behavior

Run:
    python scripts/diagnose_instability.py
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from model.vid.vq_hvebt.config import VQCodebookConfig, VQHVEBTConfig, VQStageConfig
from model.vid.vq_hvebt.hierarchy import VQHVEBTModel
from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset


def make_shape_batch(B, T, H, W, device):
    hparams = SimpleNamespace(
        context_length=T + 1,
        image_dims=[H, W],
        shape_scene_type="DIM_2",
        shape_min_cubes=2, shape_max_cubes=6,
        shape_angle_min=5, shape_angle_max=20,
        shape_temporal_patterns=[],
        shape_pattern_combining=False,
        shape_accel_min=3, shape_accel_max=6,
        shape_oscillation_period_min=1, shape_oscillation_period_max=4,
        shape_interruption_period_min=1, shape_interruption_period_max=4,
        shape_cache_dir="data/vid/shape_cache",
    )
    ds = VIDShapeSyntheticDataset(hparams, size=B)
    frames = torch.stack([ds[i] for i in range(B)], dim=0)
    return frames.to(device)


def build_model(device, freeze_encoder=False):
    stage_cfg = VQStageConfig(
        clip_stage_name="s3",
        clip_channels=512,
        H=8, W=8,
        transformer_dim=256,
        n_heads=4,
        n_layers=4,
        mcmc_steps=5,
        mcmc_step_size=100.0,
        codebook=VQCodebookConfig(
            num_codes=512,
            code_dim=512,
            init_mode="data_first_batch",
            commitment_beta=0.25,
        ),
        pred_loss_weight=1.0,
        cb_loss_weight=1.0,
        commit_loss_weight=0.25,
    )
    cfg = VQHVEBTConfig(
        stages=[stage_cfg],
        train_encoder=not freeze_encoder,
        encoder_lr_scale=0.001,
        weights_path="clip/MobileCLIP2-S0/mobileclip2_s0.pt",
        use_decoder=False,
        contrastive_loss_weight=0.0,
        encoder_warmup_steps=0,
    )
    model = VQHVEBTModel(cfg).to(device)
    return model


def stat(name, tensor):
    """Print statistics for a tensor."""
    if tensor is None:
        print(f"  {name}: None")
        return
    t = tensor.detach().float()
    print(f"  {name}: shape={list(t.shape)}, "
          f"mean={t.mean().item():.6f}, std={t.std().item():.6f}, "
          f"min={t.min().item():.6f}, max={t.max().item():.6f}, "
          f"abs_mean={t.abs().mean().item():.6f}")


def check_for_nan(name, tensor):
    if tensor is None:
        return False
    has_nan = tensor.isnan().any().item()
    has_inf = tensor.isinf().any().item()
    if has_nan or has_inf:
        print(f"  *** {name}: NaN={has_nan}, Inf={has_inf} ***")
        return True
    return False


# ============================================================================
#  TEST 1: Feature magnitudes and normalization
# ============================================================================

def test_feature_magnitudes(model, batch, device):
    print("\n" + "="*80)
    print("TEST 1: Feature Magnitudes Through Pipeline")
    print("="*80)

    with torch.no_grad():
        feats = model.encoder.encode_video(batch)
        for name, feat in feats.items():
            stat(f"CLIP stage {name} output", feat)
            # Check per-token norm
            B, T, C, H, W = feat.shape
            tokens = feat.permute(0, 1, 3, 4, 2).reshape(-1, C)
            norms = tokens.norm(dim=-1)
            print(f"    per-token L2 norm: mean={norms.mean().item():.4f}, "
                  f"std={norms.std().item():.4f}, "
                  f"min={norms.min().item():.4f}, max={norms.max().item():.4f}")

        # Quantize and check
        for stage_cfg in model.cfg.stages:
            name = stage_cfg.clip_stage_name
            z_e = feats[name]
            B, T1, C, Hs, Ws = z_e.shape
            N = T1 * Hs * Ws
            z_flat = z_e.permute(0, 1, 3, 4, 2).reshape(B, N, C)
            qout = model.quantizers[name].encode(z_flat)

            stat(f"  z_e (encoder out)", z_flat)
            stat(f"  z_q (quantized)", qout.z_q)
            stat(f"  z_q_st (straight-through)", qout.z_q_st)

            # Quantization error
            quant_err = (z_flat - qout.z_q).norm(dim=-1)
            print(f"    quantization error L2: mean={quant_err.mean().item():.4f}, "
                  f"max={quant_err.max().item():.4f}")

            # Codebook statistics
            E = model.quantizers[name].codebook.weight
            stat(f"  codebook weights", E)
            code_norms = E.norm(dim=-1)
            print(f"    codebook entry norms: mean={code_norms.mean().item():.4f}, "
                  f"std={code_norms.std().item():.4f}")

            # Inter-code distances
            with torch.no_grad():
                sample_idx = torch.randperm(E.shape[0])[:50]
                E_sample = E[sample_idx]
                dists = torch.cdist(E_sample, E_sample)
                # Mask diagonal
                mask = ~torch.eye(len(sample_idx), dtype=torch.bool, device=device)
                print(f"    inter-code L2 dist (sample): mean={dists[mask].mean().item():.4f}, "
                      f"min={dists[mask].min().item():.4f}")


# ============================================================================
#  TEST 2: MCMC dynamics
# ============================================================================

def test_mcmc_dynamics(model, batch, device):
    print("\n" + "="*80)
    print("TEST 2: MCMC Dynamics")
    print("="*80)

    # Do a forward pass manually to see what happens at each MCMC step
    with torch.no_grad():
        feats = model.encoder.encode_video(batch)

    name = model.cfg.stages[0].clip_stage_name
    z_e = feats[name]
    B, T1, C, Hs, Ws = z_e.shape
    N = T1 * Hs * Ws
    z_flat = z_e.permute(0, 1, 3, 4, 2).reshape(B, N, C)
    qout = model.quantizers[name].encode(z_flat)

    T = T1 - 1
    z_q_st_5d = qout.z_q_st.reshape(B, T1, Hs, Ws, C).permute(0, 1, 4, 2, 3).contiguous()
    real_ctx = z_q_st_5d[:, :T].detach()  # Detach like the real training does

    predictor = model.predictors[name]
    K = model.cfg.stages[0].codebook.num_codes

    # Manual MCMC loop with instrumentation
    pred_logits = torch.zeros(B, T * Hs * Ws, K, device=device, dtype=real_ctx.dtype)
    alpha = torch.clamp(predictor.alpha, min=1e-6)
    print(f"\n  MCMC step_size (alpha): {alpha.item():.4f}")
    print(f"  K={K}, N={T*Hs*Ws}, C={C}")

    for step in range(model.cfg.stages[0].mcmc_steps):
        pred_logits = pred_logits.detach().requires_grad_(True)

        # Decode logits → embedding
        z_pred_flat = model.quantizers[name].decode_logits(pred_logits)
        z_pred = z_pred_flat.reshape(B, T, Hs, Ws, C).permute(0, 1, 4, 2, 3).contiguous()

        # Compute energy
        energy = predictor.forward_energy(real_ctx, z_pred, None)

        # Gradient
        grad = torch.autograd.grad([energy.sum()], [pred_logits], create_graph=False)[0]

        print(f"\n  Step {step}:")
        stat(f"    pred_logits", pred_logits)
        stat(f"    softmax(logits)", F.softmax(pred_logits, dim=-1))
        stat(f"    z_pred (decoded)", z_pred_flat)
        stat(f"    energy", energy)
        stat(f"    gradient (d_energy/d_logits)", grad)

        # Effective update magnitude
        update = alpha * grad
        stat(f"    update (alpha*grad)", update)
        print(f"    update/logit ratio: {(update.abs() / (pred_logits.abs() + 1e-8)).mean().item():.4f}")

        # Check if gradient is dominated by specific directions
        grad_per_token = grad.norm(dim=-1)  # (B, N)
        print(f"    grad norm per token: mean={grad_per_token.mean().item():.6f}, "
              f"max={grad_per_token.max().item():.6f}")

        pred_logits = pred_logits - alpha * grad


# ============================================================================
#  TEST 3: Gradient magnitudes per component during backward
# ============================================================================

def test_gradient_flow(model, batch, device):
    print("\n" + "="*80)
    print("TEST 3: Gradient Flow (Full Forward+Backward)")
    print("="*80)

    model.train()
    model.zero_grad()
    model.maybe_initialize_codebooks(batch)

    # Hook to capture gradients at various points
    grad_records = {}

    def make_hook(name):
        def hook(grad):
            grad_records[name] = grad.detach().clone()
        return hook

    # Forward pass
    out = model.forward_loss(batch)
    print(f"\n  Total loss: {out.total_loss.item():.6f}")
    for k, v in out.metrics.items():
        print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

    # Check for NaN in output
    if check_for_nan("total_loss", out.total_loss):
        print("  *** NaN in loss - cannot analyze gradients ***")
        return

    # Backward
    out.total_loss.backward()

    # Encoder gradients
    print("\n  --- Encoder gradients ---")
    enc_grad_norms = []
    enc_param_norms = []
    for pname, p in model.encoder.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            pn = p.data.norm().item()
            enc_grad_norms.append(gn)
            enc_param_norms.append(pn)
            if gn > 10:
                print(f"    LARGE: {pname}: grad_norm={gn:.4f}, param_norm={pn:.4f}, "
                      f"ratio={gn/max(pn,1e-8):.4f}")

    if enc_grad_norms:
        print(f"    Encoder grad norms: mean={np.mean(enc_grad_norms):.6f}, "
              f"max={np.max(enc_grad_norms):.6f}, "
              f"min={np.min(enc_grad_norms):.6f}")
        print(f"    Encoder param norms: mean={np.mean(enc_param_norms):.6f}")
        print(f"    Grad/Param ratio: mean={np.mean(np.array(enc_grad_norms)/(np.array(enc_param_norms)+1e-8)):.6f}")

    # Codebook gradients
    print("\n  --- Codebook gradients ---")
    for name, q in model.quantizers.items():
        if q.codebook.weight.grad is not None:
            stat(f"    {name} codebook grad", q.codebook.weight.grad)
            stat(f"    {name} codebook weight", q.codebook.weight.data)

    # Predictor gradients
    print("\n  --- Predictor gradients ---")
    for sname, pred in model.predictors.items():
        pred_grads = []
        for pname, p in pred.named_parameters():
            if p.grad is not None:
                pred_grads.append(p.grad.norm().item())
                if p.grad.norm().item() > 10:
                    print(f"    LARGE: {sname}/{pname}: grad_norm={p.grad.norm().item():.4f}")
        if pred_grads:
            print(f"    {sname} predictor grad norms: "
                  f"mean={np.mean(pred_grads):.6f}, max={np.max(pred_grads):.6f}")

        # Check alpha (MCMC step size)
        if pred.alpha.grad is not None:
            print(f"    {sname} alpha (mcmc_step_size): value={pred.alpha.item():.4f}, "
                  f"grad={pred.alpha.grad.item():.6f}")


# ============================================================================
#  TEST 4: Second-order effects through MCMC unrolling
# ============================================================================

def test_mcmc_second_order(model, batch, device):
    print("\n" + "="*80)
    print("TEST 4: Second-Order Effects Through MCMC")
    print("="*80)
    print("  (Comparing gradient magnitude with 1 step vs 5 steps)")

    model.maybe_initialize_codebooks(batch)

    # Test with 1 MCMC step
    orig_steps = model.cfg.stages[0].mcmc_steps
    model.cfg.stages[0].mcmc_steps = 1
    model.predictors["s3"].cfg.mcmc_steps = 1
    model.zero_grad()
    out1 = model.forward_loss(batch)
    out1.total_loss.backward()

    grads_1step = {}
    for pname, p in model.named_parameters():
        if p.grad is not None:
            grads_1step[pname] = p.grad.norm().item()

    # Test with 5 MCMC steps
    model.cfg.stages[0].mcmc_steps = orig_steps
    model.predictors["s3"].cfg.mcmc_steps = orig_steps
    model.zero_grad()
    out5 = model.forward_loss(batch)
    out5.total_loss.backward()

    grads_5step = {}
    for pname, p in model.named_parameters():
        if p.grad is not None:
            grads_5step[pname] = p.grad.norm().item()

    print(f"\n  Loss (1 step): {out1.total_loss.item():.4f}")
    print(f"  Loss (5 steps): {out5.total_loss.item():.4f}")

    # Compare key parameters
    print("\n  Gradient amplification (5-step / 1-step):")
    amplifications = []
    for pname in sorted(grads_1step.keys()):
        if pname in grads_5step and grads_1step[pname] > 1e-10:
            ratio = grads_5step[pname] / (grads_1step[pname] + 1e-10)
            amplifications.append(ratio)
            if ratio > 5 or ratio < 0.2:
                print(f"    {pname}: 1step={grads_1step[pname]:.6f}, "
                      f"5step={grads_5step[pname]:.6f}, ratio={ratio:.2f}")

    if amplifications:
        print(f"\n  Mean amplification: {np.mean(amplifications):.2f}")
        print(f"  Max amplification: {np.max(amplifications):.2f}")
        print(f"  Params with >5x amplification: "
              f"{sum(1 for a in amplifications if a > 5)}/{len(amplifications)}")


# ============================================================================
#  TEST 5: The fundamental issue - gradient conflict
# ============================================================================

def test_gradient_conflict(model, batch, device):
    print("\n" + "="*80)
    print("TEST 5: Gradient Conflict Analysis")
    print("="*80)
    print("  Comparing encoder gradients from commitment_loss vs prediction_loss")

    model.maybe_initialize_codebooks(batch)

    # Get encoder features with grad
    feats = model.encoder.encode_video(batch)
    name = "s3"
    z_e = feats[name]
    B, T1, C, Hs, Ws = z_e.shape
    T = T1 - 1
    N = T1 * Hs * Ws
    z_flat = z_e.permute(0, 1, 3, 4, 2).reshape(B, N, C)

    # Quantize
    qout = model.quantizers[name].encode(z_flat)

    # Commitment loss gradient on encoder
    commit_loss = qout.commit_loss
    if not commit_loss.requires_grad:
        print("\n  Commitment loss has no gradient (EMA mode) — no gradient conflict possible.")
        print("  This confirms the fix: encoder is only driven by prediction loss now.")
        return

    model.zero_grad()
    commit_loss.backward(retain_graph=True)

    commit_grads = {}
    for pname, p in model.encoder.named_parameters():
        if p.grad is not None:
            commit_grads[pname] = p.grad.clone()

    # Now get the prediction loss gradient on encoder (via straight-through)
    # First, run MCMC with the straight-through quantized features
    z_q_st_5d = qout.z_q_st.reshape(B, T1, Hs, Ws, C).permute(0, 1, 4, 2, 3).contiguous()
    real_ctx = z_q_st_5d[:, :T]  # DON'T detach - we want to see the encoder grad

    predictor = model.predictors[name]
    _, pred_embed, _ = predictor.run_mcmc(
        real_ctx=real_ctx,
        init_logits=None,
        parent_context=None,
        learning=True,
    )

    # Prediction loss (CE on final logits)
    target_indices = qout.indices.reshape(B, T1, Hs * Ws)[:, 1:]
    tgt_flat = target_indices.reshape(B, T * Hs * Ws)

    # Get final logits for CE loss
    K = model.cfg.stages[0].codebook.num_codes
    pred_logits = torch.zeros(B, T * Hs * Ws, K, device=device, dtype=z_e.dtype)
    alpha = torch.clamp(predictor.alpha, min=1e-6)

    # Quick single-step to get a pred_logits with grad
    pred_logits = pred_logits.detach().requires_grad_(True)
    z_pred_flat = model.quantizers[name].decode_logits(pred_logits)
    z_pred = z_pred_flat.reshape(B, T, Hs, Ws, C).permute(0, 1, 4, 2, 3).contiguous()
    energy = predictor.forward_energy(real_ctx, z_pred, None)
    grad_e = torch.autograd.grad([energy.sum()], [pred_logits], create_graph=True)[0]
    final_logits = pred_logits - alpha * grad_e

    pred_loss = F.cross_entropy(
        final_logits.reshape(-1, K),
        tgt_flat.reshape(-1),
    )

    model.zero_grad()
    pred_loss.backward()

    # Compare
    print("\n  Gradient magnitudes per encoder parameter:")
    conflicts = 0
    total = 0
    for pname, p in model.encoder.named_parameters():
        if p.grad is not None and pname in commit_grads:
            pred_grad = p.grad
            comm_grad = commit_grads[pname]

            # Cosine similarity between the two gradient signals
            cos_sim = F.cosine_similarity(
                pred_grad.flatten().unsqueeze(0),
                comm_grad.flatten().unsqueeze(0)
            ).item()

            pred_norm = pred_grad.norm().item()
            comm_norm = comm_grad.norm().item()

            total += 1
            if cos_sim < 0:
                conflicts += 1

            if pred_norm > 0.01 or comm_norm > 0.01:
                print(f"    {pname}: "
                      f"pred_grad_norm={pred_norm:.6f}, "
                      f"commit_grad_norm={comm_norm:.6f}, "
                      f"cos_sim={cos_sim:.4f}, "
                      f"ratio={pred_norm/(comm_norm+1e-8):.2f}")

    print(f"\n  Gradient conflict rate: {conflicts}/{total} params have opposing gradients")
    print(f"  (Negative cosine similarity = commitment and prediction pulling encoder in opposite directions)")


# ============================================================================
#  TEST 6: Numerical precision (float16 simulation)
# ============================================================================

def test_numerical_precision(model, batch, device):
    print("\n" + "="*80)
    print("TEST 6: Numerical Precision Check")
    print("="*80)

    model.maybe_initialize_codebooks(batch)

    # Get features and check softmax precision
    with torch.no_grad():
        feats = model.encoder.encode_video(batch)
        name = "s3"
        z_e = feats[name]
        B, T1, C, Hs, Ws = z_e.shape
        N = T1 * Hs * Ws
        z_flat = z_e.permute(0, 1, 3, 4, 2).reshape(B, N, C)
        qout = model.quantizers[name].encode(z_flat)

    K = model.cfg.stages[0].codebook.num_codes
    T = T1 - 1

    # After one MCMC step, check logit magnitudes
    z_q_st_5d = qout.z_q_st.reshape(B, T1, Hs, Ws, C).permute(0, 1, 4, 2, 3).contiguous()
    real_ctx = z_q_st_5d[:, :T].detach()

    predictor = model.predictors[name]
    pred_logits = torch.zeros(B, T * Hs * Ws, K, device=device)
    alpha = torch.clamp(predictor.alpha, min=1e-6)

    print(f"\n  MCMC step_size (alpha) = {alpha.item():.2f}")
    print(f"  Codebook size K = {K}")
    print(f"  Feature dim C = {C}")

    for step in range(5):
        pred_logits = pred_logits.detach().requires_grad_(True)
        z_pred_flat = model.quantizers[name].decode_logits(pred_logits)
        z_pred = z_pred_flat.reshape(B, T, Hs, Ws, C).permute(0, 1, 4, 2, 3).contiguous()
        energy = predictor.forward_energy(real_ctx, z_pred, None)
        grad = torch.autograd.grad([energy.sum()], [pred_logits], create_graph=False)[0]

        # Check what happens in float16
        logits_f16 = pred_logits.half()
        grad_f16 = grad.half()
        update_f16 = (alpha * grad_f16)
        new_logits_f16 = logits_f16 - update_f16

        # Check for overflow/underflow
        print(f"\n  Step {step}:")
        print(f"    logits: max={pred_logits.abs().max().item():.4f}")
        print(f"    grad: max={grad.abs().max().item():.6f}")
        print(f"    alpha*grad: max={update_f16.abs().max().item():.4f}")
        print(f"    new_logits (f16): max={new_logits_f16.abs().max().item():.4f}, "
              f"has_inf={new_logits_f16.isinf().any().item()}, "
              f"has_nan={new_logits_f16.isnan().any().item()}")

        # Softmax of large logits
        probs_f32 = F.softmax(pred_logits, dim=-1)
        probs_f16 = F.softmax(pred_logits.half(), dim=-1)
        print(f"    softmax (f32): min={probs_f32.min().item():.8f}, max={probs_f32.max().item():.8f}")
        print(f"    softmax (f16): min={probs_f16.min().item():.4f}, max={probs_f16.max().item():.4f}")
        print(f"    softmax diff (f32 vs f16): max={((probs_f32 - probs_f16.float()).abs().max().item()):.6f}")

        pred_logits = pred_logits - alpha * grad


# ============================================================================
#  TEST 7: The core instability - encoder output scale vs codebook scale
# ============================================================================

def test_scale_mismatch(model, batch, device):
    print("\n" + "="*80)
    print("TEST 7: Scale Mismatch Analysis")
    print("="*80)

    model.maybe_initialize_codebooks(batch)

    # Do TWO forward passes with a gradient step in between
    # to see how much encoder features shift
    model.train()

    out1 = model.forward_loss(batch)
    print(f"  Step 0 loss: {out1.total_loss.item():.4f}")

    # Simulate one optimizer step
    # Use the same LR setup as training loop
    enc_params = model.encoder_params()
    non_enc_params = model.non_encoder_params()

    opt_main = torch.optim.AdamW(non_enc_params, lr=3e-4)
    opt_enc = torch.optim.SGD(enc_params, lr=3e-4 * 0.001, momentum=0.0)

    out1.total_loss.backward()
    nn.utils.clip_grad_norm_(non_enc_params, max_norm=1.0)
    nn.utils.clip_grad_norm_(enc_params, max_norm=1.0)

    # Record pre-step encoder features
    with torch.no_grad():
        feats_before = model.encoder.encode_video(batch)
        z_before = feats_before["s3"].clone()

    opt_main.step()
    opt_enc.step()

    # Record post-step encoder features
    with torch.no_grad():
        feats_after = model.encoder.encode_video(batch)
        z_after = feats_after["s3"]

    # How much did the encoder features change?
    z_diff = (z_after - z_before)
    print(f"\n  Encoder feature change after 1 step:")
    stat(f"    z_diff", z_diff)
    print(f"    relative change: {z_diff.norm().item() / z_before.norm().item():.6f}")

    # How does this compare to inter-code distance?
    E = model.quantizers["s3"].codebook.weight.detach()
    sample_dists = torch.cdist(E[:50], E[:50])
    mask = ~torch.eye(50, dtype=torch.bool, device=device)
    min_inter_code = sample_dists[mask].min().item()
    mean_inter_code = sample_dists[mask].mean().item()
    print(f"    min inter-code distance: {min_inter_code:.6f}")
    print(f"    mean inter-code distance: {mean_inter_code:.6f}")
    print(f"    feature shift / min_inter_code: "
          f"{z_diff.abs().mean().item() / max(min_inter_code, 1e-8):.4f}")
    print(f"    -> If this ratio > 0.1, a single step can flip many VQ assignments!")

    # Check how many assignments actually flip
    B, T1, C, Hs, Ws = z_before.shape
    N = T1 * Hs * Ws
    z_flat_before = z_before.permute(0, 1, 3, 4, 2).reshape(B, N, C)
    z_flat_after = z_after.permute(0, 1, 3, 4, 2).reshape(B, N, C)

    with torch.no_grad():
        idx_before = model.quantizers["s3"].encode(z_flat_before).indices
        idx_after = model.quantizers["s3"].encode(z_flat_after).indices

    flip_rate = (idx_before != idx_after).float().mean().item()
    print(f"    VQ assignment flip rate: {flip_rate:.4f} ({flip_rate*100:.1f}%)")
    print(f"    -> High flip rate = moving targets = unstable training!")


# ============================================================================
#  TEST 8: Energy scale analysis
# ============================================================================

def test_energy_scale(model, batch, device):
    print("\n" + "="*80)
    print("TEST 8: Energy Function Scale Analysis")
    print("="*80)

    model.maybe_initialize_codebooks(batch)
    model.train()
    model.zero_grad()
    out = model.forward_loss(batch)

    for name, sr in out.stage_results.items():
        print(f"\n  Stage {name}:")
        print(f"    Energy trace (per MCMC step): {sr.energy_trace}")
        if len(sr.energy_trace) > 1:
            # Energy change rate
            diffs = [sr.energy_trace[i+1] - sr.energy_trace[i]
                     for i in range(len(sr.energy_trace)-1)]
            print(f"    Energy changes per step: {[f'{d:.2f}' for d in diffs]}")
            print(f"    -> Large absolute energy = large gradients = instability")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Default dtype: {torch.get_default_dtype()}")

    print("\nBuilding model...")
    model = build_model(device)

    print("Generating batch...")
    batch = make_shape_batch(4, 4, 256, 256, device)
    print(f"Batch shape: {list(batch.shape)}")

    # Initialize codebooks
    model.maybe_initialize_codebooks(batch)

    test_feature_magnitudes(model, batch, device)
    test_mcmc_dynamics(model, batch, device)
    test_gradient_flow(model, batch, device)
    test_mcmc_second_order(model, batch, device)
    test_gradient_conflict(model, batch, device)
    test_numerical_precision(model, batch, device)
    test_scale_mismatch(model, batch, device)
    test_energy_scale(model, batch, device)

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
