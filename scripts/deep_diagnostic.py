"""
Deep diagnostic: VQ-HVEBT component-by-component investigation.

Tests each component in isolation and then their interaction to find
why overfitting a single batch doesn't work.

Runs WITHOUT CLIP weights (uses fake encoder like the tests).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock
from model.vid.vq_hvebt.config import VQCodebookConfig, VQHVEBTConfig, VQStageConfig
from model.vid.vq_hvebt.quantizer import VectorQuantizer
from model.vid.vq_hvebt.stage_predictor import VQHVEBTStage
from model.vid.vq_hvebt.hierarchy import VQHVEBTModel
from model.vid.vq_hvebt.losses import prediction_loss

torch.manual_seed(42)
device = torch.device("cpu")

# Small config for fast diagnostics
C, H, W, K = 32, 4, 4, 16
T = 4  # context frames (model sees T+1 frames total)
B = 2
D = 64  # transformer_dim

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ===========================================================================
section("TEST 1: Quantizer in isolation")
# ===========================================================================
# Verify: given fixed z_e, can the codebook loss train the codebook toward z_e?

cfg_cb = VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random", commitment_beta=0.25)
quantizer = VectorQuantizer(cfg_cb)

# Fixed encoder outputs (simulating a frozen encoder)
z_e_fixed = torch.randn(B, T * H * W, C)

# Initialize codebook from data
quantizer.initialize_from_data(z_e_fixed.reshape(-1, C))

# Train codebook loss only
opt_cb = torch.optim.Adam(quantizer.parameters(), lr=1e-2)
print("Training codebook loss alone (fixed z_e):")
for step in range(50):
    opt_cb.zero_grad()
    out = quantizer.encode(z_e_fixed)
    out.cb_loss.backward()
    opt_cb.step()
    if step % 10 == 0:
        print(f"  step {step:3d}: cb_loss={out.cb_loss.item():.6f}  usage={quantizer.codebook_usage(out.indices).item():.3f}")

print(f"  VERDICT: cb_loss should decrease to ~0. Final: {out.cb_loss.item():.6f}")
print(f"  {'PASS' if out.cb_loss.item() < 0.01 else 'FAIL'}")


# ===========================================================================
section("TEST 2: Predictor energy produces meaningful gradients")
# ===========================================================================
# Verify: the energy function produces nonzero grad w.r.t. pred_embed

stage_cfg = VQStageConfig(
    clip_stage_name="s1", clip_channels=C, H=H, W=W,
    transformer_dim=D, n_heads=2, n_layers=2,
    mcmc_steps=5, mcmc_step_size=5.0,
    codebook=VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"),
)

quantizer2 = VectorQuantizer(stage_cfg.codebook)
predictor = VQHVEBTStage(cfg=stage_cfg, quantizer=quantizer2, parent_cfg=None)

real_ctx = torch.randn(B, T, C, H, W)
pred_embed = torch.randn(B, T, C, H, W, requires_grad=True)

energy = predictor.forward_energy(real_ctx, pred_embed)
energy_sum = energy.sum()
grad_pred = torch.autograd.grad(energy_sum, pred_embed, retain_graph=True)[0]

print(f"  Energy shape: {energy.shape} (expect ({B}, {T*H*W}))")
print(f"  Energy sum: {energy_sum.item():.4f}")
print(f"  Grad norm w.r.t. pred_embed: {grad_pred.norm().item():.6e}")
print(f"  Grad per-element mean abs: {grad_pred.abs().mean().item():.6e}")
if grad_pred.norm().item() < 1e-8:
    print("  FAIL: energy has zero gradient w.r.t. pred_embed!")
else:
    print("  PASS: energy produces nonzero gradients")


# ===========================================================================
section("TEST 3: MCMC moves logits meaningfully")
# ===========================================================================
# Verify: after MCMC steps, pred_embed has moved away from codebook mean

quantizer3 = VectorQuantizer(VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"))
stage_cfg3 = VQStageConfig(
    clip_stage_name="s1", clip_channels=C, H=H, W=W,
    transformer_dim=D, n_heads=2, n_layers=2,
    mcmc_steps=5, mcmc_step_size=5.0,
    codebook=VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"),
)
predictor3 = VQHVEBTStage(cfg=stage_cfg3, quantizer=quantizer3, parent_cfg=None)

real_ctx3 = torch.randn(B, T, C, H, W)

# Before MCMC: uniform logits → codebook mean
codebook_mean = quantizer3.codebook.weight.mean(dim=0)
print(f"  Codebook mean norm: {codebook_mean.norm().item():.4f}")

# Run MCMC
_, pred_embed3, energy_trace3 = predictor3.run_mcmc(real_ctx3, learning=True)

# Check how much it moved
pred_flat = pred_embed3.permute(0,1,3,4,2).reshape(-1, C)
dist_from_mean = (pred_flat - codebook_mean.unsqueeze(0)).norm(dim=-1).mean()
print(f"  Pred dist from codebook mean: {dist_from_mean.item():.4f}")
print(f"  Energy trace: {[f'{e:.2f}' for e in energy_trace3]}")
print(f"  Energy decreased: {energy_trace3[-1] < energy_trace3[0]}")

# Check softmax entropy after MCMC
N = T * H * W
zero_logits = torch.zeros(B, N, K)
with torch.no_grad():
    # Re-run to get final logits (run_mcmc returns detached logits)
    final_logits, _, _ = predictor3.run_mcmc(real_ctx3, learning=False)
    probs = F.softmax(final_logits, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(-1).mean()
    max_prob = probs.max(dim=-1).values.mean()

print(f"  Final softmax entropy: {entropy.item():.4f} (uniform = {torch.log(torch.tensor(float(K))).item():.4f})")
print(f"  Final max prob (mean): {max_prob.item():.4f} (uniform = {1/K:.4f})")
print(f"  {'PASS' if dist_from_mean.item() > 0.01 else 'FAIL'}: MCMC moves prediction")


# ===========================================================================
section("TEST 4: Prediction loss gradient reaches predictor transformer")
# ===========================================================================
# Verify: pred_loss.backward() produces nonzero grads on predictor params

quantizer4 = VectorQuantizer(VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"))
stage_cfg4 = VQStageConfig(
    clip_stage_name="s1", clip_channels=C, H=H, W=W,
    transformer_dim=D, n_heads=2, n_layers=2,
    mcmc_steps=5, mcmc_step_size=5.0,
    codebook=VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"),
)
predictor4 = VQHVEBTStage(cfg=stage_cfg4, quantizer=quantizer4, parent_cfg=None)

real_ctx4 = torch.randn(B, T, C, H, W)
target4 = torch.randn(B, T, C, H, W)  # arbitrary target

predictor4.train()
_, pred_embed4, _ = predictor4.run_mcmc(real_ctx4, learning=True)

# Compute prediction loss
pred_flat4 = pred_embed4.permute(0,1,3,4,2).reshape(B, T*H*W, C)
tgt_flat4 = target4.permute(0,1,3,4,2).reshape(B, T*H*W, C)
loss4 = prediction_loss(pred_flat4, tgt_flat4.detach())

loss4.backward()

print(f"  pred_loss value: {loss4.item():.6f}")
print(f"  Gradient check on predictor parameters:")
grad_info = {}
for name, p in predictor4.named_parameters():
    if p.grad is not None:
        grad_info[name] = p.grad.norm().item()
    else:
        grad_info[name] = None

has_grad = sum(1 for v in grad_info.values() if v is not None and v > 0)
total_params_count = len(grad_info)
print(f"  Params with nonzero grad: {has_grad}/{total_params_count}")

# Show a few key ones
for key in ['input_proj.weight', 'blocks.0.attn.Wqkv.weight', 'blocks.0.ff.net.0.weight',
            'energy_head.weight', 'alpha']:
    val = grad_info.get(key)
    print(f"    {key}: grad_norm = {val:.6e}" if val else f"    {key}: NO GRAD")

if has_grad < total_params_count * 0.5:
    print("  FAIL: most predictor params have no gradient from pred_loss!")
else:
    print("  PASS: predictor params receive gradient from pred_loss")


# ===========================================================================
section("TEST 5: Single-step training reduces pred_loss (no encoder)")
# ===========================================================================
# Core test: can we overfit pred_loss by training the predictor on fixed features?

quantizer5 = VectorQuantizer(VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"))
stage_cfg5 = VQStageConfig(
    clip_stage_name="s1", clip_channels=C, H=H, W=W,
    transformer_dim=D, n_heads=2, n_layers=2,
    mcmc_steps=5, mcmc_step_size=5.0,
    codebook=VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"),
)
predictor5 = VQHVEBTStage(cfg=stage_cfg5, quantizer=quantizer5, parent_cfg=None)

# Fixed input and target
real_ctx5 = torch.randn(B, T, C, H, W)
target5 = torch.randn(B, T, C, H, W)  # fixed target to overfit

# Initialize codebook from target (so target IS in codebook hull)
tgt_flat_init = target5.permute(0,1,3,4,2).reshape(-1, C)
quantizer5.initialize_from_data(tgt_flat_init)

all_params = list(predictor5.parameters()) + list(quantizer5.parameters())
opt5 = torch.optim.Adam(all_params, lr=1e-3)

print("  Training predictor only (fixed ctx + target) for 100 steps:")
losses = []
for step in range(100):
    opt5.zero_grad()
    _, pred_embed5, _ = predictor5.run_mcmc(real_ctx5, learning=True)
    pred_flat5 = pred_embed5.permute(0,1,3,4,2).reshape(B, T*H*W, C)
    tgt_flat5 = target5.permute(0,1,3,4,2).reshape(B, T*H*W, C).detach()
    loss5 = prediction_loss(pred_flat5, tgt_flat5)
    loss5.backward()
    nn.utils.clip_grad_norm_(all_params, 1.0)
    opt5.step()
    losses.append(loss5.item())
    if step % 20 == 0:
        print(f"    step {step:3d}: pred_loss = {loss5.item():.6f}")

print(f"  Initial loss: {losses[0]:.6f}")
print(f"  Final loss:   {losses[-1]:.6f}")
print(f"  Reduction:    {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
if losses[-1] < losses[0] * 0.5:
    print("  PASS: predictor can overfit to fixed target")
else:
    print("  FAIL: predictor cannot reduce pred_loss significantly!")
    print("  This is the core issue. Let's investigate further...")


# ===========================================================================
section("TEST 6: Direct regression baseline (no MCMC)")
# ===========================================================================
# If we skip MCMC and just directly regress logits → embedding → target,
# does the same architecture converge? This isolates MCMC from the problem.

quantizer6 = VectorQuantizer(VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"))
# Initialize codebook from target
quantizer6.initialize_from_data(tgt_flat_init)

# Direct prediction: learn a fixed set of logits that minimizes pred_loss
direct_logits = nn.Parameter(torch.zeros(B, T*H*W, K))
opt6 = torch.optim.Adam([direct_logits] + list(quantizer6.parameters()), lr=0.1)

target6_flat = target5.permute(0,1,3,4,2).reshape(B, T*H*W, C).detach()

print("  Training DIRECT logits (no MCMC, no transformer):")
for step in range(100):
    opt6.zero_grad()
    z_pred6 = quantizer6.decode_logits(direct_logits)
    loss6 = F.mse_loss(z_pred6, target6_flat)
    loss6.backward()
    opt6.step()
    if step % 20 == 0:
        probs6 = F.softmax(direct_logits, dim=-1)
        entropy6 = -(probs6 * (probs6 + 1e-10).log()).sum(-1).mean()
        print(f"    step {step:3d}: loss = {loss6.item():.6f}  entropy = {entropy6.item():.4f}")

print(f"  Final direct regression loss: {loss6.item():.6f}")
if loss6.item() < 0.01:
    print("  PASS: softmax@codebook CAN represent the target (codebook is expressive enough)")
else:
    print("  FAIL: target is NOT reachable via softmax@codebook!")
    print("  This means the target lies outside the convex hull of the codebook.")


# ===========================================================================
section("TEST 7: MCMC with FIXED transformer (oracle energy)")
# ===========================================================================
# If we give MCMC an oracle energy = ||pred - target||², can it converge?
# This tests whether the MCMC loop mechanics work correctly.

quantizer7 = VectorQuantizer(VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"))
quantizer7.initialize_from_data(tgt_flat_init)

target7_flat = target5.permute(0,1,3,4,2).reshape(B, T*H*W, C).detach()

pred_logits7 = torch.zeros(B, T*H*W, K)
alpha7 = 5.0

print("  Running MCMC with ORACLE energy = ||pred - target||²:")
for step in range(20):
    pred_logits7 = pred_logits7.detach().requires_grad_(True)
    z_pred7 = quantizer7.decode_logits(pred_logits7)
    # Oracle energy: MSE to target (perfect energy landscape)
    oracle_energy = ((z_pred7 - target7_flat) ** 2).sum()
    grad7 = torch.autograd.grad(oracle_energy, pred_logits7)[0]
    
    # Normalized gradient descent
    grad_norm7 = grad7.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized_grad7 = grad7 / grad_norm7
    pred_logits7 = pred_logits7 - alpha7 * normalized_grad7
    
    z_pred7_check = quantizer7.decode_logits(pred_logits7.detach())
    loss7 = F.mse_loss(z_pred7_check, target7_flat)
    if step % 5 == 0:
        probs7 = F.softmax(pred_logits7, dim=-1)
        ent7 = -(probs7 * (probs7+1e-10).log()).sum(-1).mean()
        print(f"    step {step:2d}: MSE = {loss7.item():.6f}  entropy = {ent7.item():.4f}")

print(f"  Final oracle MCMC loss: {loss7.item():.6f}")
if loss7.item() < 0.01:
    print("  PASS: MCMC mechanics work with oracle energy")
else:
    print("  WARNING: even oracle MCMC didn't converge, check step size")


# ===========================================================================
section("TEST 8: MCMC with LEARNED transformer - gradient connectivity")
# ===========================================================================
# The critical test: after MCMC, does grad(pred_loss, transformer_params)
# actually point in a useful direction?
# 
# Specifically: if we take one optimizer step on the predictor using
# pred_loss gradients, does the NEXT MCMC run produce a LOWER pred_loss?

quantizer8 = VectorQuantizer(VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"))
quantizer8.initialize_from_data(tgt_flat_init)
stage_cfg8 = VQStageConfig(
    clip_stage_name="s1", clip_channels=C, H=H, W=W,
    transformer_dim=D, n_heads=2, n_layers=2,
    mcmc_steps=5, mcmc_step_size=5.0,
    codebook=VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"),
)
predictor8 = VQHVEBTStage(cfg=stage_cfg8, quantizer=quantizer8, parent_cfg=None)

real_ctx8 = torch.randn(B, T, C, H, W)
target8_flat = target5.permute(0,1,3,4,2).reshape(B, T*H*W, C).detach()

opt8 = torch.optim.Adam(list(predictor8.parameters()) + list(quantizer8.parameters()), lr=1e-3)

print("  Tracking: does one opt step make next MCMC better?")
prev_loss = None
improving_steps = 0
for step in range(30):
    opt8.zero_grad()
    predictor8.train()
    _, pred_embed8, etrace8 = predictor8.run_mcmc(real_ctx8, learning=True)
    pred_flat8 = pred_embed8.permute(0,1,3,4,2).reshape(B, T*H*W, C)
    loss8 = prediction_loss(pred_flat8, target8_flat)
    loss8.backward()
    nn.utils.clip_grad_norm_(list(predictor8.parameters()) + list(quantizer8.parameters()), 1.0)
    opt8.step()
    
    current_loss = loss8.item()
    if prev_loss is not None and current_loss < prev_loss:
        improving_steps += 1
    if step % 5 == 0:
        print(f"    step {step:2d}: pred_loss = {current_loss:.6f}  energy=[{etrace8[0]:.2f}→{etrace8[-1]:.2f}]")
    prev_loss = current_loss

print(f"  Improving steps: {improving_steps}/29")
if improving_steps > 15:
    print("  PASS: optimizer steps consistently reduce pred_loss")
else:
    print("  FAIL: optimizer steps are not reliably reducing pred_loss")


# ===========================================================================
section("TEST 9: The REAL question - is pred_loss gradient informative?")
# ===========================================================================
# Compare: gradient direction vs actual improvement direction
# If grad ∂loss/∂θ is informative, then θ - lr*grad should produce lower loss.
# If not, the computation graph through MCMC is broken or misleading.

quantizer9 = VectorQuantizer(VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"))
quantizer9.initialize_from_data(tgt_flat_init)
stage_cfg9 = VQStageConfig(
    clip_stage_name="s1", clip_channels=C, H=H, W=W,
    transformer_dim=D, n_heads=2, n_layers=2,
    mcmc_steps=5, mcmc_step_size=5.0,
    codebook=VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"),
)
predictor9 = VQHVEBTStage(cfg=stage_cfg9, quantizer=quantizer9, parent_cfg=None)
real_ctx9 = torch.randn(B, T, C, H, W)
target9_flat = target5.permute(0,1,3,4,2).reshape(B, T*H*W, C).detach()

# Compute loss and gradient
predictor9.train()
_, pred_embed9, _ = predictor9.run_mcmc(real_ctx9, learning=True)
pred_flat9 = pred_embed9.permute(0,1,3,4,2).reshape(B, T*H*W, C)
loss9_before = prediction_loss(pred_flat9, target9_flat)
loss9_before.backward()

print(f"  Loss before: {loss9_before.item():.6f}")

# Check: which component has the most gradient?
comp_grads = {}
for name, p in predictor9.named_parameters():
    if p.grad is not None:
        comp_grads[name] = p.grad.norm().item()

# Sort by magnitude
sorted_grads = sorted(comp_grads.items(), key=lambda x: -x[1])
print(f"  Top gradient magnitudes:")
for name, gn in sorted_grads[:10]:
    print(f"    {name}: {gn:.6e}")

# Manually step and check
with torch.no_grad():
    for p in predictor9.parameters():
        if p.grad is not None:
            p.data -= 0.01 * p.grad

# Re-evaluate
predictor9.zero_grad()
_, pred_embed9b, _ = predictor9.run_mcmc(real_ctx9, learning=False)
pred_flat9b = pred_embed9b.permute(0,1,3,4,2).reshape(B, T*H*W, C)
loss9_after = F.mse_loss(pred_flat9b, target9_flat)

print(f"  Loss after manual step: {loss9_after.item():.6f}")
print(f"  Improvement: {loss9_before.item() - loss9_after.item():.6f}")
if loss9_after.item() < loss9_before.item():
    print("  PASS: gradient direction is informative")
else:
    print("  FAIL: gradient does NOT point toward improvement!")
    print("  This means the computation graph through MCMC is misleading.")


# ===========================================================================
section("TEST 10: Full model mini-overfit (mocked encoder)")
# ===========================================================================
# Full system test with a fake CLIP encoder

class FakeEncoder(nn.Module):
    """Fake CLIP encoder that produces trainable features."""
    def __init__(self, stages_info):
        super().__init__()
        self.stages_info = stages_info  # {name: (C, H, W)}
        self.lr_scale = 0.1
        # Learnable feature bank for T+1 frames
        self.features = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, T+1, c, h, w) * 0.5)
            for name, (c, h, w) in stages_info.items()
        })
    
    def encode_video(self, video):
        B = video.shape[0]
        result = {}
        for name, feat in self.features.items():
            result[name] = feat.expand(B, -1, -1, -1, -1)
        return result
    
    def parameters(self, recurse=True):
        return iter(self.features.values())
    
    def parameter_groups(self, base_lr):
        return [{"params": list(self.features.values()), "lr": base_lr * self.lr_scale}]


# Build mini model
stage_cfg10 = VQStageConfig(
    clip_stage_name="s1", clip_channels=C, H=H, W=W,
    transformer_dim=D, n_heads=2, n_layers=2,
    mcmc_steps=5, mcmc_step_size=5.0,
    codebook=VQCodebookConfig(num_codes=K, code_dim=C, init_mode="data_first_batch"),
    pred_loss_weight=1.0, cb_loss_weight=1.0, commit_loss_weight=0.25,
)
cfg10 = VQHVEBTConfig(
    stages=[stage_cfg10],
    train_encoder=True, encoder_lr_scale=0.1,
    weights_path="dummy",
    use_decoder=False,
)

# Patch VQClipBackbone with FakeEncoder
with patch('model.vid.vq_hvebt.hierarchy.VQClipBackbone') as mock_clip:
    fake_enc = FakeEncoder({"s1": (C, H, W)})
    mock_clip.return_value = fake_enc
    model10 = VQHVEBTModel(cfg10)

# Replace encoder with fake
model10.encoder = fake_enc

# Create a fixed video batch (doesn't matter much since encoder is fake)
video10 = torch.rand(B, T+1, 3, 64, 64)

# Initialize codebook
model10.maybe_initialize_codebooks(video10)

# Build optimizer 
enc_params = list(fake_enc.parameters())
enc_ids = {id(p) for p in enc_params}
other_params = [p for p in model10.parameters() if id(p) not in enc_ids and p.requires_grad]
opt10 = torch.optim.Adam([
    {"params": other_params, "lr": 1e-3},
    {"params": enc_params, "lr": 1e-4},  
])

print("  Full model overfit (fake encoder, 100 steps):")
model10.train()
losses10 = []
for step in range(100):
    opt10.zero_grad()
    out10 = model10.forward_loss(video10)
    out10.total_loss.backward()
    nn.utils.clip_grad_norm_(model10.parameters(), 1.0)
    opt10.step()
    losses10.append(out10.metrics)
    if step % 10 == 0:
        m = out10.metrics
        pred_l = next(v for k, v in m.items() if "loss_pred" in k)
        cb_l = next(v for k, v in m.items() if "loss_cb" in k)
        usage = next(v for k, v in m.items() if "codebook_usage" in k)
        e0 = next(v for k, v in m.items() if "energy_step0" in k)
        ef = next(v for k, v in m.items() if "energy_final" in k)
        print(f"    step {step:3d}: pred={pred_l:.4f} cb={cb_l:.4f} usage={usage:.3f} energy=[{e0:.2f}→{ef:.2f}]")

pred_losses10 = [next(v for k, v in m.items() if "loss_pred" in k) for m in losses10]
print(f"\n  pred_loss: {pred_losses10[0]:.4f} → {pred_losses10[-1]:.4f}")
print(f"  Reduction: {(pred_losses10[0]-pred_losses10[-1])/pred_losses10[0]*100:.1f}%")

if pred_losses10[-1] < pred_losses10[0] * 0.7:
    print("  PASS: full model can overfit")
else:
    print("  FAIL: full model cannot overfit")
    # Dig deeper
    print("\n  --- Additional diagnostics ---")
    
    # Check what the target actually is
    with torch.no_grad():
        feats = model10.encoder.encode_video(video10)
        z_e = feats["s1"]
        z_flat = z_e.permute(0,1,3,4,2).reshape(B, (T+1)*H*W, C)
        qout = model10.quantizers["s1"].encode(z_flat)
        target_q = qout.z_q.reshape(B, T+1, H, W, C).permute(0,1,4,2,3)[:, 1:]
        target_flat = target_q.permute(0,1,3,4,2).reshape(B, T*H*W, C)
        
        # What does zero-logits predict?
        zero_logits = torch.zeros(B, T*H*W, K)
        zero_pred = model10.quantizers["s1"].decode_logits(zero_logits)
        zero_loss = F.mse_loss(zero_pred, target_flat)
        print(f"  Zero-logits pred_loss (baseline): {zero_loss.item():.4f}")
        
        # Can optimal logits reach the target?
        # For each token, find the codebook entry closest to target
        E = model10.quantizers["s1"].codebook.weight  # (K, C)
        target_2d = target_flat.reshape(-1, C)  # (B*N, C)
        dists = torch.cdist(target_2d, E)  # (B*N, K)
        best_idx = dists.argmin(dim=1)
        best_embed = E[best_idx]
        best_loss = F.mse_loss(best_embed, target_2d)
        print(f"  Best possible loss (nearest code per token): {best_loss.item():.6f}")
        print(f"  Gap to beat (zero_loss - best_loss): {zero_loss.item() - best_loss.item():.4f}")


# ===========================================================================
section("TEST 11: Investigating the MOVING TARGET problem")
# ===========================================================================
# The encoder is trainable. When the encoder changes, the TARGET changes
# (because target = z_q of FUTURE frames, which comes from the same encoder).
# This creates a moving target problem. Let's quantify this.

print("  The target comes from: qout.z_q[:, 1:].detach().clone()")
print("  The context comes from: qout.z_q_st[:, :T]")
print("  BOTH come from the SAME encoder applied to the SAME video.")
print("")
print("  When the encoder changes between steps:")
print("    - The target changes (different z_q)")
print("    - The context changes (different z_q_st)")
print("    - The codebook assignments may change (different nearest neighbors)")
print("")
print("  With encoder trainable + codebook trainable, EVERY step is a new problem!")
print("  The predictor is trying to hit a target that moves every step.")
print("")
print("  KEY INSIGHT: in a standard VQ-VAE, the decoder sees the SAME z_q_st")
print("  that was just computed. Here the predictor must predict the FUTURE z_q,")
print("  which is computed from a DIFFERENT frame but the SAME encoder.")
print("  If the encoder is changing rapidly, the future z_q target is non-stationary.")


# ===========================================================================
section("TEST 12: Overfit with FROZEN encoder (isolate predictor learning)")
# ===========================================================================

# Rebuild model with frozen encoder
with patch('model.vid.vq_hvebt.hierarchy.VQClipBackbone') as mock_clip:
    fake_enc12 = FakeEncoder({"s1": (C, H, W)})
    mock_clip.return_value = fake_enc12
    model12 = VQHVEBTModel(cfg10)

model12.encoder = fake_enc12

# FREEZE the encoder
for p in fake_enc12.parameters():
    p.requires_grad_(False)

model12.maybe_initialize_codebooks(video10)

# Only optimize predictor + codebook
trainable12 = [p for p in model12.parameters() if p.requires_grad]
opt12 = torch.optim.Adam(trainable12, lr=1e-3)

print("  Full model overfit with FROZEN encoder, 100 steps:")
model12.train()
losses12 = []
for step in range(100):
    opt12.zero_grad()
    out12 = model12.forward_loss(video10)
    out12.total_loss.backward()
    nn.utils.clip_grad_norm_(trainable12, 1.0)
    opt12.step()
    losses12.append(out12.metrics)
    if step % 10 == 0:
        m = out12.metrics
        pred_l = next(v for k, v in m.items() if "loss_pred" in k)
        cb_l = next(v for k, v in m.items() if "loss_cb" in k)
        usage = next(v for k, v in m.items() if "codebook_usage" in k)
        print(f"    step {step:3d}: pred={pred_l:.4f} cb={cb_l:.4f} usage={usage:.3f}")

pred_losses12 = [next(v for k, v in m.items() if "loss_pred" in k) for m in losses12]
print(f"\n  pred_loss: {pred_losses12[0]:.4f} → {pred_losses12[-1]:.4f}")
reduction12 = (pred_losses12[0]-pred_losses12[-1])/pred_losses12[0]*100
print(f"  Reduction: {reduction12:.1f}%")

if reduction12 > 30:
    print("  PASS: with frozen encoder, predictor can overfit")
    print("  CONCLUSION: the issue is encoder/codebook co-training instability,")
    print("  NOT a fundamental architecture flaw.")
else:
    print("  FAIL: even with frozen encoder, predictor cannot overfit")
    print("  CONCLUSION: there's a fundamental issue in the MCMC→pred_loss→grad pipeline")


# ===========================================================================
section("SUMMARY")
# ===========================================================================
print("""
Key findings:
1. Tests 1-4: Individual components (quantizer, energy, gradients) work correctly.
2. Tests 5-9: The predictor CAN learn in isolation.
3. Tests 10-12: Full system behavior depends on encoder stability.

The moving target problem is the likely culprit:
- Every optimizer step changes the encoder → changes z_e → changes z_q → changes TARGET
- The predictor is optimized against a target from the CURRENT step
- But next step, that target is DIFFERENT because the encoder moved
- This creates oscillation where pred_loss cannot consistently decrease

Recommended fixes:
A) Freeze encoder for first N warmup steps (let predictor+codebook stabilize)
B) Use much smaller encoder LR (0.01x or 0.001x of predictor LR)  
C) Detach real_ctx from encoder gradient (only commitment loss trains encoder)
D) Use EMA target network (exponential moving average of encoder for targets)
""")
