"""
Deep architecture diagnostic for VQ-HVEBT.

Tests:
  1. Encoder feature diversity per stage (are features already collapsed at encoder output?)
  2. L2 normalization impact (does norm=1 kill variance needed for VQ?)
  3. Codebook loss gradient flow (does cb_loss actually move the codebook?)
  4. Commitment loss gradient flow (does commit_loss push encoder features?)
  5. CE prediction loss gradient flow through straight-through
  6. Quantizer encode → decode round-trip fidelity
  7. Per-stage capacity analysis (are transformer dims large enough?)
  8. MCMC dynamics per stage (does energy decrease? does it converge?)
  9. Feature magnitude and codebook magnitude match
  10. Diversity loss vs CE loss magnitude comparison
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace

from model.vid.vq_hvebt.config import VQCodebookConfig, VQStageConfig, VQHVEBTConfig
from model.vid.vq_hvebt.hierarchy import VQHVEBTModel
from model.vid.vq_hvebt.quantizer import VectorQuantizer
from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset

torch.manual_seed(42)
device = torch.device("cpu")

# ============================================================
# Helper: build model matching training loop defaults
# ============================================================

def build_diagnostic_model(stages=("s3", "s2", "s1"), num_codes=256,
                            transformer_dim=64, n_heads=2, n_layers=2):
    STAGE_INFO = {
        "s1": (64, 8, 8),
        "s2": (128, 4, 4),
        "s3": (256, 2, 2),
        "s_pool": (256, 1, 1),
    }
    STAGE_K = {
        "s_pool": num_codes,
        "s3": num_codes,
        "s2": max(16, num_codes // 8),
        "s1": 16,
    }
    stage_cfgs = []
    for sn in stages:
        C, H, W = STAGE_INFO[sn]
        stage_cfgs.append(VQStageConfig(
            clip_stage_name=sn,
            clip_channels=C,
            H=H, W=W,
            transformer_dim=transformer_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            mcmc_steps=10,
            mcmc_step_size=1.0,
            mcmc_per_token_norm=True,
            pred_head=False,
            energy_bound=10.0,
            energy_reg_weight=0.01,
            codebook=VQCodebookConfig(
                num_codes=STAGE_K[sn],
                code_dim=C,
                init_mode="data_first_batch",
                use_ema=False,
                commitment_beta=0.25,
            ),
            pred_loss_weight=1.0,
            cb_loss_weight=1.0,
            commit_loss_weight=0.25,
        ))

    cfg = VQHVEBTConfig(
        stages=stage_cfgs,
        train_encoder=True,
        encoder_lr_scale=1.0,
        use_custom_encoder=True,
        encoder_base_channels=32,
        ema_target_decay=0.999,
        use_decoder=True,
        decoder_out_size=64,
        encoder_warmup_steps=0,
        codebook_diversity_weight=1.0,
    )
    model = VQHVEBTModel(cfg).to(device)
    return model


def make_batch(B=4, T=4, H=64, W=64):
    hparams = SimpleNamespace(
        context_length=T + 1,
        image_dims=[H, W],
        shape_scene_type="DIM_2",
        shape_min_cubes=2, shape_max_cubes=6,
        shape_angle_min=15, shape_angle_max=45,
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


# ============================================================
print("=" * 80)
print("DEEP ARCHITECTURE DIAGNOSTIC FOR VQ-HVEBT")
print("=" * 80)

model = build_diagnostic_model()
video = make_batch()
print(f"\nVideo batch: {video.shape}")  # (B, T+1, 3, 64, 64)

# Initialize codebooks
model.maybe_initialize_codebooks(video)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}  Trainable: {trainable_params:,}")

# ============================================================
# TEST 1: Encoder feature diversity per stage
# ============================================================
print("\n" + "=" * 80)
print("TEST 1: Encoder Feature Diversity Per Stage")
print("=" * 80)

with torch.no_grad():
    feats = model.encoder.encode_video(video)

for name, feat in feats.items():
    B, T1, C, H, W = feat.shape
    # Flatten to (B*T1*H*W, C)
    flat = feat.permute(0, 1, 3, 4, 2).reshape(-1, C)
    M = flat.shape[0]
    
    # L2 norms
    norms = flat.norm(dim=1)
    
    # Pairwise cosine similarity (subsample to 128)
    if M > 128:
        idx = torch.randperm(M)[:128]
        sample = flat[idx]
    else:
        sample = flat
    z_norm = F.normalize(sample, dim=1)
    cos_sim = z_norm @ z_norm.T
    S = z_norm.shape[0]
    mask = ~torch.eye(S, dtype=torch.bool)
    mean_cos = cos_sim[mask].mean().item()
    max_cos = cos_sim[mask].max().item()
    min_cos = cos_sim[mask].min().item()
    
    # Per-channel variance
    channel_var = flat.var(dim=0)
    mean_var = channel_var.mean().item()
    min_var = channel_var.min().item()
    max_var = channel_var.max().item()
    dead_channels = (channel_var < 1e-6).sum().item()
    
    # Unique nearest codebook codes
    E = model.quantizers[name].codebook_weight.detach()
    z_sq = (flat ** 2).sum(dim=1, keepdim=True)
    e_sq = (E ** 2).sum(dim=1, keepdim=True).T
    dist = z_sq + e_sq - 2 * flat @ E.T
    indices = dist.argmin(dim=1)
    unique_codes = indices.unique().numel()
    K = E.shape[0]
    
    print(f"\n  {name}: shape=({B},{T1},{C},{H},{W})  tokens={M}  codebook K={K}")
    print(f"    L2 norm:  mean={norms.mean():.4f}  std={norms.std():.4f}  min={norms.min():.4f}  max={norms.max():.4f}")
    print(f"    Cosine sim: mean={mean_cos:.4f}  min={min_cos:.4f}  max={max_cos:.4f}")
    print(f"    Channel var: mean={mean_var:.6f}  min={min_var:.6f}  max={max_var:.6f}  dead={dead_channels}/{C}")
    print(f"    Codebook usage: {unique_codes}/{K} = {unique_codes/K:.3f}")
    
    # CRITICAL CHECK: are features already collapsed before VQ?
    if mean_cos > 0.9:
        print(f"    *** WARNING: Features are highly similar (cos_sim={mean_cos:.4f}). "
              f"Encoder is collapsing features at this stage!")
    if unique_codes <= 2:
        print(f"    *** WARNING: Only {unique_codes} unique codes used. "
              f"Features map to very few codebook entries!")

# ============================================================
# TEST 2: L2 Normalization Impact
# ============================================================
print("\n" + "=" * 80)
print("TEST 2: L2 Normalization Impact (target_norm=1.0)")
print("=" * 80)

# Check what happens to feature variance after L2 norm
with torch.no_grad():
    # Get pre-norm features by running encoder without norm
    encoder = model.encoder.live
    old_norm = encoder.target_norm
    encoder.target_norm = 0  # disable norm
    feats_raw = model.encoder.encode_video(video)
    encoder.target_norm = old_norm  # restore
    feats_normed = model.encoder.encode_video(video)

for name in feats.keys():
    raw = feats_raw[name]
    normed = feats_normed[name]
    B, T1, C, H, W = raw.shape
    raw_flat = raw.permute(0, 1, 3, 4, 2).reshape(-1, C)
    normed_flat = normed.permute(0, 1, 3, 4, 2).reshape(-1, C)
    
    raw_norms = raw_flat.norm(dim=1)
    normed_norms = normed_flat.norm(dim=1)
    
    # Information content: how much variance is preserved?
    raw_var = raw_flat.var(dim=0).mean().item()
    normed_var = normed_flat.var(dim=0).mean().item()
    
    # Direction diversity (cosine sim should be same)
    if raw_flat.shape[0] > 128:
        idx = torch.randperm(raw_flat.shape[0])[:128]
        raw_s, normed_s = raw_flat[idx], normed_flat[idx]
    else:
        raw_s, normed_s = raw_flat, normed_flat
    
    raw_cos = (F.normalize(raw_s, dim=1) @ F.normalize(raw_s, dim=1).T)
    normed_cos = (F.normalize(normed_s, dim=1) @ F.normalize(normed_s, dim=1).T)
    S = raw_s.shape[0]
    mask = ~torch.eye(S, dtype=torch.bool)
    
    print(f"\n  {name}:")
    print(f"    Raw norms:    mean={raw_norms.mean():.4f}  std={raw_norms.std():.4f}")
    print(f"    Normed norms: mean={normed_norms.mean():.4f}  std={normed_norms.std():.4f}")
    print(f"    Raw variance: {raw_var:.6f}  Normed variance: {normed_var:.6f}  ratio={normed_var/(raw_var+1e-10):.4f}")
    print(f"    Raw cos_sim:  mean={raw_cos[mask].mean():.4f}")
    print(f"    Norm cos_sim: mean={normed_cos[mask].mean():.4f}")
    
    # KEY INSIGHT: With target_norm=1.0 and C=256 channels:
    # Each channel has magnitude ~1/sqrt(C). For C=256, that's ~0.0625.
    # The VQ distance is ||z - e||² where both have norm 1.
    # For C=256, the per-channel signal is tiny. Codebook entries on the
    # unit sphere may be too close in L2 for VQ to discriminate.
    expected_per_channel = 1.0 / (C ** 0.5)
    print(f"    Expected per-channel magnitude at norm=1: {expected_per_channel:.4f}")
    print(f"    Actual per-channel std: {normed_flat.std(dim=0).mean():.6f}")


# ============================================================
# TEST 3: Codebook geometry after initialization
# ============================================================
print("\n" + "=" * 80)
print("TEST 3: Codebook Geometry After Data Init")
print("=" * 80)

for name, q in model.quantizers.items():
    E = q.codebook_weight.detach()
    K, C = E.shape
    
    # Codebook entry norms
    cb_norms = E.norm(dim=1)
    
    # Pairwise distances between codebook entries
    dist_matrix = torch.cdist(E.unsqueeze(0), E.unsqueeze(0)).squeeze(0)
    mask_diag = ~torch.eye(K, dtype=torch.bool)
    pairwise_dists = dist_matrix[mask_diag]
    
    # Cosine similarity between codebook entries
    E_norm = F.normalize(E, dim=1)
    cos_cb = E_norm @ E_norm.T
    cos_off_diag = cos_cb[mask_diag]
    
    # Compare codebook norms to encoder feature norms
    feat = feats[name]
    feat_flat = feat.permute(0, 1, 3, 4, 2).reshape(-1, C)
    feat_norms = feat_flat.norm(dim=1)
    
    print(f"\n  {name}: K={K}, C={C}")
    print(f"    CB norms:  mean={cb_norms.mean():.4f}  std={cb_norms.std():.4f}  min={cb_norms.min():.4f}  max={cb_norms.max():.4f}")
    print(f"    Feat norms: mean={feat_norms.mean():.4f}  std={feat_norms.std():.4f}")
    print(f"    Norm mismatch: CB={cb_norms.mean():.4f} vs Feat={feat_norms.mean():.4f}")
    print(f"    CB pairwise L2: mean={pairwise_dists.mean():.4f}  min={pairwise_dists.min():.4f}")
    print(f"    CB pairwise cos: mean={cos_off_diag.mean():.4f}  min={cos_off_diag.min():.4f}  max={cos_off_diag.max():.4f}")
    
    if cb_norms.mean().item() > 2 * feat_norms.mean().item():
        print(f"    *** WARNING: Codebook norms >> feature norms. Distance is dominated by norm mismatch!")
    if cb_norms.mean().item() < 0.5 * feat_norms.mean().item():
        print(f"    *** WARNING: Codebook norms << feature norms. Distance is dominated by norm mismatch!")


# ============================================================
# TEST 4: Gradient Flow Analysis
# ============================================================
print("\n" + "=" * 80)
print("TEST 4: Gradient Flow Analysis (1 step)")
print("=" * 80)

model.train()
model.zero_grad()
out = model.forward_loss(video)
out.total_loss.backward()

# Check gradients for each component
print("\n  [Encoder gradients]")
enc_grads = []
for n, p in model.encoder.live.named_parameters():
    if p.grad is not None:
        g = p.grad.abs().mean().item()
        enc_grads.append((n, g, p.numel()))
enc_grads.sort(key=lambda x: -x[1])
for n, g, numel in enc_grads[:5]:
    print(f"    {n}: grad_mean={g:.6f}  params={numel}")
if not enc_grads:
    print(f"    *** NO ENCODER GRADIENTS!")

print("\n  [Codebook gradients (gradient mode)]")
for name, q in model.quantizers.items():
    if q.codebook_weight.grad is not None:
        g = q.codebook_weight.grad.abs().mean().item()
        g_max = q.codebook_weight.grad.abs().max().item()
        print(f"    {name}: grad_mean={g:.6f}  grad_max={g_max:.6f}")
    else:
        print(f"    {name}: *** NO GRADIENT on codebook!")

print("\n  [Predictor gradients]")
for stage_name, predictor in model.predictors.items():
    grad_norms = []
    for n, p in predictor.named_parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.abs().mean().item())
    if grad_norms:
        print(f"    {stage_name}: mean={np.mean(grad_norms):.6f}  max={np.max(grad_norms):.6f}")
    else:
        print(f"    {stage_name}: *** NO GRADIENTS!")

print("\n  [Loss breakdown]")
m = out.metrics
for sname in ["s3", "s2", "s1"]:
    pred_l = m.get(f"{sname}/loss_pred", "N/A")
    cb_l = m.get(f"{sname}/loss_cb", "N/A")
    commit_l = m.get(f"{sname}/loss_commit", "N/A")
    usage = m.get(f"{sname}/codebook_usage", "N/A")
    perpl = m.get(f"{sname}/codebook_perplexity", "N/A")
    ent = m.get(f"{sname}/entropy_avg", "N/A")
    print(f"    {sname}: pred={pred_l}  cb={cb_l}  commit={commit_l}  usage={usage}  perpl={perpl}  entropy={ent}")
print(f"    total_loss: {out.total_loss.item():.4f}")
if "decoder/loss" in m:
    print(f"    decoder_loss: {m['decoder/loss']:.4f}")


# ============================================================
# TEST 5: Multi-step training stability
# ============================================================
print("\n" + "=" * 80)
print("TEST 5: Multi-step Training (20 steps)")
print("=" * 80)

model2 = build_diagnostic_model()
model2.maybe_initialize_codebooks(video)
model2.train()
opt = torch.optim.AdamW(model2.parameter_groups(3e-4), weight_decay=1e-4)

for step in range(1, 21):
    opt.zero_grad()
    out = model2.forward_loss(video)
    out.total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
    opt.step()
    model2.update_ema_encoder()
    
    if step in [1, 5, 10, 15, 20]:
        m = out.metrics
        parts = [f"  step {step:>3}: total={out.total_loss.item():.4f}"]
        for sname in ["s3", "s2", "s1"]:
            usage = m.get(f"{sname}/codebook_usage", 0)
            perpl = m.get(f"{sname}/codebook_perplexity", 0)
            cb_l = m.get(f"{sname}/loss_cb", 0)
            commit_l = m.get(f"{sname}/loss_commit", 0)
            pred_l = m.get(f"{sname}/loss_pred", 0)
            ent = m.get(f"{sname}/entropy_avg", 0)
            parts.append(f"\n    {sname}: pred={pred_l:.4f} cb={cb_l:.4f} commit={commit_l:.4f} "
                        f"usage={usage:.3f} perpl={perpl:.1f} ent={ent:.2f}b")
        print("".join(parts))


# ============================================================
# TEST 6: Straight-through gradient magnitude comparison
# ============================================================
print("\n" + "=" * 80)
print("TEST 6: Straight-Through Gradient Comparison")
print("=" * 80)
print("  Checking if CE loss gradient through ST actually reaches encoder...")

model3 = build_diagnostic_model()
model3.maybe_initialize_codebooks(video)
model3.train()

# Forward once
out = model3.forward_loss(video)

# Isolate CE loss gradient
model3.zero_grad()
# Just the prediction losses
pred_total = sum(
    sr.pred_loss * model3.cfg.stages[i].pred_loss_weight
    for i, (name, sr) in enumerate(out.stage_results.items())
)
pred_total.backward(retain_graph=True)

print("\n  [Encoder grads from CE prediction loss only]")
enc_grad_from_ce = {}
for n, p in model3.encoder.live.named_parameters():
    if p.grad is not None:
        enc_grad_from_ce[n] = p.grad.abs().mean().item()
if enc_grad_from_ce:
    max_grad_name = max(enc_grad_from_ce, key=enc_grad_from_ce.get)
    print(f"    Max encoder grad: {max_grad_name} = {enc_grad_from_ce[max_grad_name]:.8f}")
    print(f"    Mean encoder grad: {np.mean(list(enc_grad_from_ce.values())):.8f}")
else:
    print("    *** NO ENCODER GRADIENTS from CE loss!")
    print("    This means straight-through is not working or context is detached.")

# Now check VQ losses
model3.zero_grad()
vq_total = sum(
    sr.cb_loss * model3.cfg.stages[i].cb_loss_weight +
    sr.commit_loss * model3.cfg.stages[i].commit_loss_weight
    for i, (name, sr) in enumerate(out.stage_results.items())
)
vq_total.backward(retain_graph=True)

print("\n  [Encoder grads from VQ losses (cb + commit) only]")
enc_grad_from_vq = {}
for n, p in model3.encoder.live.named_parameters():
    if p.grad is not None:
        enc_grad_from_vq[n] = p.grad.abs().mean().item()
if enc_grad_from_vq:
    max_grad_name = max(enc_grad_from_vq, key=enc_grad_from_vq.get)
    print(f"    Max encoder grad: {max_grad_name} = {enc_grad_from_vq[max_grad_name]:.8f}")
    print(f"    Mean encoder grad: {np.mean(list(enc_grad_from_vq.values())):.8f}")
else:
    print("    *** NO ENCODER GRADIENTS from VQ losses!")

# Compare magnitudes
if enc_grad_from_ce and enc_grad_from_vq:
    ce_mean = np.mean(list(enc_grad_from_ce.values()))
    vq_mean = np.mean(list(enc_grad_from_vq.values()))
    ratio = vq_mean / (ce_mean + 1e-10)
    print(f"\n  Ratio VQ_grad / CE_grad on encoder: {ratio:.2f}x")
    if ratio > 10:
        print("    *** VQ losses dominate encoder gradient! "
              "Commitment loss may overwhelm prediction signal.")
    elif ratio < 0.1:
        print("    *** CE loss dominates encoder gradient. "
              "VQ losses may be too weak to prevent feature collapse.")


# ============================================================
# TEST 7: Per-stage capacity analysis 
# ============================================================
print("\n" + "=" * 80)
print("TEST 7: Per-Stage Capacity Analysis")
print("=" * 80)

for name, predictor in model.predictors.items():
    cfg = predictor.cfg
    C = cfg.clip_channels
    D = cfg.transformer_dim
    K = cfg.codebook.num_codes
    H, W = cfg.H, cfg.W
    T = 4  # from our batch
    N = T * H * W  # tokens per batch element
    
    pred_params = sum(p.numel() for p in predictor.parameters())
    
    # Information capacity: how many bits can this stage represent?
    bits_per_token = np.log2(K) if K > 1 else 0
    total_bits = bits_per_token * H * W  # per frame
    
    print(f"\n  {name}: C={C}, D={D}, K={K}, H×W={H}×{W}")
    print(f"    Tokens per frame: {H*W}")
    print(f"    Tokens per sequence: {N} (T={T})")
    print(f"    Bits per token: {bits_per_token:.1f}")
    print(f"    Bits per frame: {total_bits:.1f}")
    print(f"    Predictor params: {pred_params:,}")
    print(f"    D / C ratio: {D/C:.2f}")
    
    if D < C:
        print(f"    *** WARNING: transformer_dim ({D}) < clip_channels ({C}). "
              f"The predictor may be too narrow to process features!")
    if D < 2 * C:
        print(f"    *** NOTE: transformer_dim ({D}) < 2*clip_channels ({2*C}). "
              f"Input projection from 2C→D is compressive.")
    
    # Energy head capacity: 1 scalar from D features
    # Check if energy head has enough capacity
    print(f"    Input projection: 2C={2*C} → D={D} "
          f"({'COMPRESSIVE' if D < 2*C else 'EXPANSIVE'})")


# ============================================================
# TEST 8: MCMC Energy Dynamics
# ============================================================
print("\n" + "=" * 80)
print("TEST 8: MCMC Energy Dynamics Per Stage")
print("=" * 80)

model4 = build_diagnostic_model()
model4.maybe_initialize_codebooks(video)
model4.train()
out4 = model4.forward_loss(video)

for name, sr in out4.stage_results.items():
    trace = sr.energy_trace
    if trace:
        print(f"\n  {name}: {len(trace)} MCMC steps")
        print(f"    Energy: start={trace[0]:.4f}  end={trace[-1]:.4f}  "
              f"decrease={trace[0]-trace[-1]:.4f}")
        if len(trace) > 1:
            diffs = [trace[i]-trace[i+1] for i in range(len(trace)-1)]
            print(f"    Per-step decrease: mean={np.mean(diffs):.4f}  "
                  f"min={np.min(diffs):.4f}  max={np.max(diffs):.4f}")
            if np.min(diffs) < 0:
                print(f"    *** WARNING: Energy INCREASED on some steps!")
    else:
        print(f"\n  {name}: No energy trace available")


# ============================================================
# TEST 9: VQ distance analysis
# ============================================================
print("\n" + "=" * 80)
print("TEST 9: VQ Assignment Quality")
print("=" * 80)

with torch.no_grad():
    feats = model.encoder.encode_video(video)

for name in model.quantizers:
    q = model.quantizers[name]
    feat = feats[name]
    B, T1, C, H, W = feat.shape
    flat = feat.permute(0, 1, 3, 4, 2).reshape(-1, C)
    M = flat.shape[0]
    
    E = q.codebook_weight.detach()
    K = E.shape[0]
    
    # Compute all distances
    z_sq = (flat ** 2).sum(dim=1, keepdim=True)
    e_sq = (E ** 2).sum(dim=1, keepdim=True).T
    dist = z_sq + e_sq - 2 * flat @ E.T  # (M, K)
    
    # Nearest distance
    min_dist, min_idx = dist.min(dim=1)
    
    # Second nearest distance (margin)
    dist_sorted, _ = dist.sort(dim=1)
    second_dist = dist_sorted[:, 1] if K > 1 else dist_sorted[:, 0]
    margin = second_dist - min_dist
    
    # How many tokens assigned to each code
    counts = torch.bincount(min_idx, minlength=K)
    active = (counts > 0).sum().item()
    max_count = counts.max().item()
    
    print(f"\n  {name}: M={M} tokens, K={K} codes")
    print(f"    Nearest dist: mean={min_dist.mean():.4f}  std={min_dist.std():.4f}")
    print(f"    Margin (2nd-1st): mean={margin.mean():.4f}  std={margin.std():.4f}")
    print(f"    Active codes: {active}/{K}")
    print(f"    Max code count: {max_count}/{M} ({max_count/M*100:.1f}%)")
    
    if margin.mean().item() < 0.01:
        print(f"    *** WARNING: Very small margin ({margin.mean():.4f}). "
              f"Codes are too close together or features are uniform.")
    if max_count > 0.5 * M:
        print(f"    *** WARNING: One code has >50% of tokens. Severe codebook collapse!")


# ============================================================
# TEST 10: Feature Space Dimensionality Analysis
# ============================================================
print("\n" + "=" * 80)
print("TEST 10: Feature Space Effective Dimensionality (PCA)")
print("=" * 80)

with torch.no_grad():
    feats = model.encoder.encode_video(video)

for name, feat in feats.items():
    B, T1, C, H, W = feat.shape
    flat = feat.permute(0, 1, 3, 4, 2).reshape(-1, C)
    
    # Center the data
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    
    # SVD for PCA
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    
    # Fraction of variance explained
    var_explained = (S ** 2) / (S ** 2).sum()
    cum_var = var_explained.cumsum(0)
    
    # Effective dimensionality: how many components for 95% variance?
    dim_95 = (cum_var < 0.95).sum().item() + 1
    dim_99 = (cum_var < 0.99).sum().item() + 1
    
    # Participation ratio: (sum(λ))² / sum(λ²)
    eigenvalues = S ** 2
    participation_ratio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    
    print(f"\n  {name}: C={C}")
    print(f"    Top 5 singular values: {S[:5].tolist()}")
    print(f"    Dims for 95% var: {dim_95}/{C}")
    print(f"    Dims for 99% var: {dim_99}/{C}")
    print(f"    Participation ratio: {participation_ratio:.1f}")
    
    if dim_95 < 5:
        print(f"    *** WARNING: Only {dim_95} dimensions explain 95% of variance. "
              f"Features are extremely low-rank!")
    if participation_ratio < C * 0.1:
        print(f"    *** WARNING: Participation ratio {participation_ratio:.1f} << {C}. "
              f"Features are concentrated in few dimensions.")


# ============================================================
# TEST 11: Decoder sanity check
# ============================================================
print("\n" + "=" * 80)
print("TEST 11: Decoder Architecture Check")
print("=" * 80)

if model.decoder is not None:
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"  Decoder params: {dec_params:,}")
    
    # Test with random input at finest stage resolution
    finest = model.cfg.stages[-1]
    test_in = torch.randn(1, finest.clip_channels, finest.H, finest.W)
    test_out = model.decoder(test_in)
    print(f"  Input shape: {test_in.shape}")
    print(f"  Output shape: {test_out.shape}")
    print(f"  Output range: [{test_out.min():.4f}, {test_out.max():.4f}]")
    
    # Test with actual quantized features
    with torch.no_grad():
        feats = model.encoder.encode_video(video)
        feat = feats[finest.clip_stage_name]
        z_q = model.quantizers[finest.clip_stage_name].encode(
            feat.permute(0, 1, 3, 4, 2).reshape(feat.shape[0], -1, feat.shape[2])
        ).z_q.reshape(feat.shape[0], feat.shape[1], finest.H, finest.W, finest.clip_channels)
        z_q = z_q.permute(0, 1, 4, 2, 3)  # (B, T+1, C, H, W)
        # Decode first frame
        dec_out = model.decoder(z_q[:, 0])  # (B, 3, H_out, W_out)
        print(f"  Decoded from real features: range=[{dec_out.min():.4f}, {dec_out.max():.4f}]")
else:
    print("  No decoder in model.")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)

print("""
Key things to check in the output above:

1. ENCODER FEATURE DIVERSITY (Test 1):
   - If mean cosine similarity > 0.8 at s3/s2, features are too similar.
   - L2 norm=1 + high C means per-channel signal is tiny (~1/√C).

2. L2 NORMALIZATION (Test 2):
   - All features forced to unit norm → VQ operates on unit sphere.
   - With C=256, entries on the sphere are very close in L2.
   - This makes VQ discrimination hard at coarse stages.

3. GRADIENT BALANCE (Tests 4, 6):
   - If VQ losses >> CE loss gradients on encoder, commitment loss
     dominates and may fight prediction.
   - If CE loss has zero encoder gradient, straight-through is broken.

4. CAPACITY (Test 7):
   - transformer_dim=64 with clip_channels=256 at s3 means
     input_proj compresses 512→64 (8× compression!).
   - The predictor may lack capacity for coarse stages.

5. VQ ASSIGNMENT QUALITY (Test 9):
   - Small margin = codes are too close or features are uniform.
   - High concentration on 1-2 codes = codebook collapse.

6. EFFECTIVE DIMENSIONALITY (Test 10):
   - If features are very low-rank, VQ with many codes is wasteful.
   - Low participation ratio = most information in few directions.
""")
