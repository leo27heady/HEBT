"""
Quick verification: embedding-space MCMC works correctly.
Tests the oracle MCMC and predictor overfitting that failed before.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vid.vq_hvebt.config import VQCodebookConfig, VQStageConfig
from model.vid.vq_hvebt.quantizer import VectorQuantizer
from model.vid.vq_hvebt.stage_predictor import VQHVEBTStage
from model.vid.vq_hvebt.losses import prediction_loss

torch.manual_seed(42)
C, H, W, K = 32, 4, 4, 16
T, B, D = 4, 2, 64

def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


section("TEST A: Oracle MCMC in embedding space")
# The oracle energy = ||pred - target||^2, should converge to target directly.

target = torch.randn(B, T*H*W, C)
pred_embed = torch.zeros(B, T*H*W, C)
alpha = 100.0

print("  Running MCMC with oracle energy = ||pred - target||²:")
for step in range(10):
    pred_embed = pred_embed.detach().requires_grad_(True)
    oracle_energy = ((pred_embed - target) ** 2).sum()
    grad = torch.autograd.grad(oracle_energy, pred_embed)[0]
    pred_embed = pred_embed - alpha * grad
    
    mse = F.mse_loss(pred_embed.detach(), target)
    if step % 2 == 0:
        print(f"    step {step:2d}: MSE = {mse.item():.6f}  grad_norm = {grad.norm().item():.4f}")

final_mse = F.mse_loss(pred_embed.detach(), target)
print(f"  Final MSE: {final_mse.item():.6f}")
print(f"  {'PASS' if final_mse.item() < 0.01 else 'FAIL'}")


section("TEST B: Predictor overfit (fixed ctx + target, 100 steps)")

stage_cfg = VQStageConfig(
    clip_stage_name="s1", clip_channels=C, H=H, W=W,
    transformer_dim=D, n_heads=2, n_layers=2,
    mcmc_steps=5, mcmc_step_size=100.0,
    codebook=VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"),
)
quantizer = VectorQuantizer(stage_cfg.codebook)
predictor = VQHVEBTStage(cfg=stage_cfg, quantizer=quantizer, parent_cfg=None)

real_ctx = torch.randn(B, T, C, H, W)
target_5d = torch.randn(B, T, C, H, W)
target_flat = target_5d.permute(0,1,3,4,2).reshape(B, T*H*W, C).detach()

all_params = list(predictor.parameters()) + list(quantizer.parameters())
opt = torch.optim.Adam(all_params, lr=1e-3)

print("  Training predictor (embedding-space MCMC):")
losses = []
for step in range(100):
    opt.zero_grad()
    predictor.train()
    _, pred_embed_5d, etrace = predictor.run_mcmc(real_ctx, learning=True)
    pred_flat = pred_embed_5d.permute(0,1,3,4,2).reshape(B, T*H*W, C)
    loss = prediction_loss(pred_flat, target_flat)
    loss.backward()
    nn.utils.clip_grad_norm_(all_params, 1.0)
    opt.step()
    losses.append(loss.item())
    if step % 10 == 0:
        print(f"    step {step:3d}: pred_loss = {loss.item():.6f}  energy=[{etrace[0]:.1f}→{etrace[-1]:.1f}]")

print(f"\n  pred_loss: {losses[0]:.4f} → {losses[-1]:.4f}")
reduction = (losses[0]-losses[-1])/losses[0]*100
print(f"  Reduction: {reduction:.1f}%")
print(f"  {'PASS' if reduction > 50 else 'FAIL'}")


section("TEST C: Gradient informativeness (one step should improve)")

stage_cfg2 = VQStageConfig(
    clip_stage_name="s1", clip_channels=C, H=H, W=W,
    transformer_dim=D, n_heads=2, n_layers=2,
    mcmc_steps=5, mcmc_step_size=100.0,
    codebook=VQCodebookConfig(num_codes=K, code_dim=C, init_mode="random"),
)
quantizer2 = VectorQuantizer(stage_cfg2.codebook)
predictor2 = VQHVEBTStage(cfg=stage_cfg2, quantizer=quantizer2, parent_cfg=None)

real_ctx2 = torch.randn(B, T, C, H, W)
target2_flat = torch.randn(B, T*H*W, C)

predictor2.train()
_, pred_5d, _ = predictor2.run_mcmc(real_ctx2, learning=True)
pred_flat2 = pred_5d.permute(0,1,3,4,2).reshape(B, T*H*W, C)
loss_before = prediction_loss(pred_flat2, target2_flat.detach())
loss_before.backward()

print(f"  Loss before: {loss_before.item():.6f}")

# Manual gradient step
with torch.no_grad():
    for p in predictor2.parameters():
        if p.grad is not None:
            p.data -= 0.001 * p.grad

# Re-evaluate
predictor2.zero_grad()
_, pred_5d_after, _ = predictor2.run_mcmc(real_ctx2, learning=False)
pred_flat_after = pred_5d_after.permute(0,1,3,4,2).reshape(B, T*H*W, C)
loss_after = F.mse_loss(pred_flat_after, target2_flat.detach())

print(f"  Loss after step: {loss_after.item():.6f}")
print(f"  Improvement: {loss_before.item() - loss_after.item():.6f}")
print(f"  {'PASS' if loss_after.item() < loss_before.item() else 'FAIL'}")


section("TEST D: Full model with larger K=512, C=512 (like real s3)")
# This simulates the actual failing scenario

from unittest.mock import patch
from model.vid.vq_hvebt.config import VQHVEBTConfig
from model.vid.vq_hvebt.hierarchy import VQHVEBTModel

C_big, H_big, W_big, K_big = 64, 4, 4, 64  # scaled version (not full 512 for speed)
T_big = 4
B_big = 2

class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr_scale = 0.01
        self.features = nn.Parameter(torch.randn(1, T_big+1, C_big, H_big, W_big) * 0.5)
    def encode_video(self, video):
        B = video.shape[0]
        return {"s1": self.features.expand(B, -1, -1, -1, -1)}
    def parameter_groups(self, base_lr):
        return [{"params": [self.features], "lr": base_lr * self.lr_scale}]

stage_cfg_big = VQStageConfig(
    clip_stage_name="s1", clip_channels=C_big, H=H_big, W=W_big,
    transformer_dim=128, n_heads=4, n_layers=3,
    mcmc_steps=5, mcmc_step_size=100.0,
    codebook=VQCodebookConfig(num_codes=K_big, code_dim=C_big, init_mode="data_first_batch"),
    pred_loss_weight=1.0, cb_loss_weight=1.0, commit_loss_weight=0.25,
)
cfg_big = VQHVEBTConfig(
    stages=[stage_cfg_big], train_encoder=True, encoder_lr_scale=0.01,
    weights_path="dummy", use_decoder=False,
)

with patch('model.vid.vq_hvebt.hierarchy.VQClipBackbone') as mock:
    fake_enc = FakeEncoder()
    mock.return_value = fake_enc
    model = VQHVEBTModel(cfg_big)
model.encoder = fake_enc

video = torch.rand(B_big, T_big+1, 3, 64, 64)
model.maybe_initialize_codebooks(video)

enc_params = list(fake_enc.parameters())
enc_ids = {id(p) for p in enc_params}
other_params = [p for p in model.parameters() if id(p) not in enc_ids and p.requires_grad]
opt = torch.optim.Adam([
    {"params": other_params, "lr": 3e-4},
    {"params": enc_params, "lr": 3e-6},
])

print(f"  Training full model (C={C_big}, K={K_big}, mcmc_step_size=100):")
model.train()
pred_losses = []
for step in range(100):
    opt.zero_grad()
    out = model.forward_loss(video)
    out.total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    m = out.metrics
    pred_l = next(v for k, v in m.items() if "loss_pred" in k)
    pred_losses.append(pred_l)
    if step % 10 == 0:
        cb_l = next(v for k, v in m.items() if "loss_cb" in k)
        usage = next(v for k, v in m.items() if "codebook_usage" in k)
        e0 = next(v for k, v in m.items() if "energy_step0" in k)
        ef = next(v for k, v in m.items() if "energy_final" in k)
        print(f"    step {step:3d}: pred={pred_l:.4f} cb={cb_l:.4f} usage={usage:.3f} energy=[{e0:.1f}→{ef:.1f}]")

print(f"\n  pred_loss: {pred_losses[0]:.4f} → {pred_losses[-1]:.4f}")
reduction_big = (pred_losses[0]-pred_losses[-1])/pred_losses[0]*100
print(f"  Reduction: {reduction_big:.1f}%")
print(f"  {'PASS' if reduction_big > 30 else 'FAIL'}")


section("SUMMARY")
print(f"""
Results:
  A. Oracle MCMC in embedding space: {'PASS' if final_mse.item() < 0.01 else 'FAIL'}
  B. Predictor overfit (small):      {'PASS' if reduction > 50 else 'FAIL'} ({reduction:.0f}% reduction)
  C. Gradient informativeness:       {'PASS' if loss_after.item() < loss_before.item() else 'FAIL'}
  D. Full model (larger scale):      {'PASS' if reduction_big > 30 else 'FAIL'} ({reduction_big:.0f}% reduction)
""")
