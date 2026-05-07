"""Diagnostic script to identify why VQ-HVEBT overfitting fails."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from model.vid.vq_hvebt.config import VQCodebookConfig, VQHVEBTConfig, VQStageConfig
from model.vid.vq_hvebt.hierarchy import VQHVEBTModel

device = torch.device("cpu")
torch.manual_seed(42)

# Build model (same as training script with --stage s3)
stage_cfg = VQStageConfig(
    clip_stage_name="s3", clip_channels=512, H=8, W=8,
    transformer_dim=256, n_heads=4, n_layers=4,
    mcmc_steps=3, mcmc_step_size=0.1,
    codebook=VQCodebookConfig(num_codes=512, code_dim=512, init_mode="data_first_batch", commitment_beta=0.25),
    pred_loss_weight=1.0, cb_loss_weight=1.0, commit_loss_weight=0.25,
)
cfg = VQHVEBTConfig(stages=[stage_cfg], train_encoder=True, encoder_lr_scale=0.1,
                     weights_path="clip/MobileCLIP2-S0/mobileclip2_s0.pt", use_decoder=False)
model = VQHVEBTModel(cfg).to(device)

# Create fake batch
B, T, H_img, W_img = 2, 4, 256, 256
video = torch.rand(B, T+1, 3, H_img, W_img, device=device)
model.maybe_initialize_codebooks(video)

# Forward pass
model.train()
out = model.forward_loss(video)

print("=" * 60)
print("DIAGNOSTIC: VQ-HVEBT Training Issues")
print("=" * 60)

# 1. Check loss magnitude
print(f"\n--- Loss values ---")
for k, v in sorted(out.metrics.items()):
    print(f"  {k}: {v:.6f}")

# 2. Check energy trace (does MCMC decrease energy?)
sr = out.stage_results["s3"]
print(f"\n--- MCMC Energy Trace ---")
for i, e in enumerate(sr.energy_trace):
    print(f"  Step {i}: {e:.4f}")
print(f"  Energy decreased? {sr.energy_trace[-1] < sr.energy_trace[0] if len(sr.energy_trace) > 1 else 'N/A'}")

# 3. Check pred_embed vs target distance
target = sr.z_q[:, 1:].detach()
pred = sr.pred_embed
B2, T2, C, Hs, Ws = pred.shape
pred_flat = pred.permute(0,1,3,4,2).reshape(-1, C)
tgt_flat = target.permute(0,1,3,4,2).reshape(-1, C)
token_dist = (pred_flat - tgt_flat).norm(dim=-1)
print(f"\n--- Prediction Distance ---")
print(f"  Mean L2 dist (pred vs target per token): {token_dist.mean().item():.4f}")
print(f"  Target token norm: {tgt_flat.norm(dim=-1).mean().item():.4f}")
print(f"  Pred token norm: {pred_flat.norm(dim=-1).mean().item():.4f}")

# 4. Check how much MCMC moves the prediction
# Compare pred_embed to what you'd get from uniform logits (codebook mean)
codebook_mean = model.quantizers["s3"].codebook.weight.mean(dim=0)
mean_dist = (pred_flat - codebook_mean.unsqueeze(0)).norm(dim=-1).mean()
uniform_dist = (tgt_flat - codebook_mean.unsqueeze(0)).norm(dim=-1).mean()
print(f"  Pred dist from codebook mean: {mean_dist.item():.4f}")
print(f"  Target dist from codebook mean: {uniform_dist.item():.4f}")
print(f"  Ratio (pred_moved / target_dist): {mean_dist.item() / uniform_dist.item():.4f}")

# 5. Check gradient magnitudes
out.total_loss.backward()
print(f"\n--- Gradient Magnitudes ---")

# Energy head
eh = model.predictors["s3"].energy_head
print(f"  energy_head.weight grad norm: {eh.weight.grad.norm().item():.6e}")
print(f"  energy_head.weight value norm: {eh.weight.data.norm().item():.6e}")
print(f"  energy_head.bias grad norm: {eh.bias.grad.norm().item() if eh.bias.grad is not None else 0:.6e}")

# MCMC step size
alpha = model.predictors["s3"].alpha
print(f"  alpha (step_size) value: {alpha.data.item():.6f}")
if alpha.grad is not None:
    print(f"  alpha grad: {alpha.grad.item():.6e}")
else:
    print(f"  alpha grad: None")

# Input projection
ip = model.predictors["s3"].input_proj
print(f"  input_proj.weight grad norm: {ip.weight.grad.norm().item():.6e}")

# Transformer blocks
for i, blk in enumerate(model.predictors["s3"].blocks):
    attn_w = getattr(blk.attn, 'qkv', getattr(blk.attn, 'Wqkv', None))
    attn_grad = attn_w.weight.grad.norm().item() if (attn_w is not None and attn_w.weight.grad is not None) else 0
    ff_layers = [m for m in blk.ff.modules() if isinstance(m, nn.Linear)]
    ff_grad = ff_layers[0].weight.grad.norm().item() if (ff_layers and ff_layers[0].weight.grad is not None) else 0
    print(f"  block[{i}] attn grad norm: {attn_grad:.6e}, ff grad norm: {ff_grad:.6e}")

# Codebook
cb = model.quantizers["s3"].codebook
print(f"  codebook.weight grad norm: {cb.weight.grad.norm().item():.6e}")
print(f"  codebook.weight value norm: {cb.weight.data.norm().item():.6e}")

# Encoder (sample a few layers)
enc_params = list(model.encoder.parameters())
enc_grads = [p.grad.norm().item() for p in enc_params if p.grad is not None]
if enc_grads:
    print(f"  encoder grad norms: min={min(enc_grads):.6e}, max={max(enc_grads):.6e}, mean={sum(enc_grads)/len(enc_grads):.6e}")
else:
    print(f"  encoder: NO gradients!")

# 6. Check softmax distribution after MCMC
print(f"\n--- Softmax Analysis (after MCMC) ---")
# We need to redo MCMC to capture intermediate logits
predictor = model.predictors["s3"]
with torch.no_grad():
    feats = model.encoder.encode_video(video)
    z_e = feats["s3"]
    z_flat = z_e.permute(0,1,3,4,2).reshape(B, (T+1)*64, 512)
    qout = model.quantizers["s3"].encode(z_flat)
    from model.vid.vq_hvebt.quantizer import QuantizerOutput
    z_q_st_5d = qout.z_q_st.reshape(B, T+1, 8, 8, 512).permute(0,1,4,2,3)
    real_ctx = z_q_st_5d[:, :T]
    
    # Run MCMC manually to inspect logits
    N = T * 8 * 8
    K = 512
    pred_logits = torch.zeros(B, N, K)
    alpha_val = predictor.alpha.item()
    
    for step in range(3):
        pred_logits.requires_grad_(True)
        z_pred_flat = model.quantizers["s3"].decode_logits(pred_logits)
        z_pred = z_pred_flat.reshape(B, T, 8, 8, 512).permute(0,1,4,2,3)
        energy = predictor.forward_energy(real_ctx, z_pred)
        grad = torch.autograd.grad([energy.sum()], [pred_logits])[0]
        
        print(f"  MCMC step {step}:")
        print(f"    energy sum: {energy.sum().item():.4f}")
        print(f"    grad norm (per token avg): {grad.norm(dim=-1).mean().item():.6e}")
        print(f"    grad abs max: {grad.abs().max().item():.6e}")
        print(f"    logit change (alpha*grad): {(alpha_val * grad).norm(dim=-1).mean().item():.6e}")
        
        pred_logits = (pred_logits - alpha_val * grad).detach()
    
    # Check final softmax entropy
    final_probs = torch.softmax(pred_logits, dim=-1)
    entropy = -(final_probs * (final_probs + 1e-10).log()).sum(-1).mean()
    max_prob = final_probs.max(dim=-1).values.mean()
    print(f"\n  Final logits: mean={pred_logits.mean():.6e}, std={pred_logits.std():.6e}")
    print(f"  Final softmax entropy: {entropy.item():.4f} (uniform={torch.log(torch.tensor(512.0)).item():.4f})")
    print(f"  Final max prob (mean): {max_prob.item():.4f} (uniform={1/512:.4f})")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
If the MCMC logit changes are tiny (< 0.01), the prediction is stuck
at the codebook mean and the model cannot learn. This would indicate:
1. mcmc_step_size is too small for logit space (softmax divides grad by ~K)
2. energy_head weights are too small to produce meaningful gradients
3. The combination means MCMC effectively does nothing.
""")
