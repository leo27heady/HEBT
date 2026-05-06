# VQ-HVEBT: Architecture and Usage Guide

This document explains how VQ-HVEBT works, what each component does, and how
to use it. Read this first before looking at the code.

---

## What problem does this solve?

Given a video of T+1 frames, VQ-HVEBT learns to **predict future frames in a
discrete latent space**. The latent space is learned jointly with the encoder
and defined by a codebook of K discrete vectors.

The key novelty is combining two ideas:

1. **VQ-VAE** — encode visual features into a discrete codebook, train it with
   a straight-through gradient trick so the whole system is end-to-end trainable.
2. **Energy-Based Model (EBT)** — instead of predicting the future with a
   single-pass decoder, use iterative gradient descent (MCMC) to search for the
   lowest-energy (most likely) future latent state.

---

## Conceptual walkthrough

Consider a 5-frame video clip. The model:

1. **Encodes** all 5 frames with a trainable CLIP visual encoder. Each frame
   becomes a spatial grid of features, e.g. 32×32 at 128 channels for stage s1.

2. **Quantizes** those features. For each spatial token, find the nearest
   vector in a codebook of K=512 entries. This converts continuous features
   into discrete codes.

3. **Predicts**: for each of frames 1, 2, 3, 4 (i.e. the future), run MCMC
   to produce a low-energy predicted latent embedding:
   - Start with zero logits over all K codes (= uniform distribution).
   - Decode logits → weighted combination of codebook entries (predicted embed).
   - Concatenate real past frame features with predicted future embed.
   - Pass through a transformer → per-token scalar energy.
   - Compute gradient of total energy with respect to the logits.
   - Update logits: `logits ← logits − α × ∂energy/∂logits`.
   - Repeat 3 times → final predicted embedding.

4. **Computes loss**: compare predicted embedding to the true quantized future
   latent (detached, so the encoder cannot cheat by moving the target).

The model trains the encoder, codebook, and predictor transformer jointly.

---

## Architecture diagram

```
video (B, T+1, 3, H, W)
         │
         ▼
┌─────────────────────────┐
│   VQClipBackbone        │  trainable MobileCLIP2-S0
│   (always trainable)    │  returns feature maps per stage
└────────┬────────────────┘
         │  stage features: {s1: (B,T+1,128,32,32), s2: (B,T+1,256,16,16), s3: (B,T+1,512,8,8)}
         ▼
┌───────────────────────────────────────────────────────────────────┐
│  Per-stage VQ-VAE Quantizer (VectorQuantizer)                     │
│                                                                   │
│  z_e ──► nearest neighbor in codebook ──► z_q  (hard, discrete)  │
│  z_q_st = z_e + sg(z_q - z_e)           (straight-through)       │
│                                                                   │
│  cb_loss     = ||sg(z_e) - z_q||²        (trains codebook)       │
│  commit_loss = β||z_e - sg(z_q)||²       (trains encoder)        │
└───────────────────────────────────────────────────────────────────┘
         │  z_q_st[:, :-1] → real_ctx (frames 0..T-1, has grad)
         │  z_q[:, 1:].detach().clone() → target (frames 1..T, no grad)
         ▼
┌───────────────────────────────────────────────────────────────────┐
│  EBT Predictor (VQHVEBTStage) - one per spatial stage             │
│                                                                   │
│  MCMC in logit space:                                             │
│    init: pred_logits = zeros (B, T*H*W, K)                       │
│    for k = 1..3:                                                  │
│      z_pred = softmax(pred_logits) @ codebook.weight             │
│      energy = transformer(concat(real_ctx, z_pred))              │
│      grad   = ∂energy/∂pred_logits                               │
│      pred_logits ← pred_logits − α × grad                        │
│                                                                   │
│  pred_loss = ||final_z_pred - target||²  (trains predictor+enc)  │
└───────────────────────────────────────────────────────────────────┘
         │  coarser stage provides parent context to finer stage
         ▼ (top-down: s3 → s2 → s1)
  (optional) PixelDecoder: finest z_pred → RGB image
```

---

## Gradient flow

Understanding what each loss trains is crucial.

| Loss term | Gradient reaches | Gradient does NOT reach |
|-----------|-----------------|------------------------|
| `cb_loss = \|\|sg(z_e) - z_q\|\|²` | Codebook entries | Encoder (`sg(z_e)` blocks it) |
| `commit_loss = β\|\|z_e - sg(z_q)\|\|²` | Encoder | Codebook (`sg(z_q)` blocks it) |
| `pred_loss = \|\|z_pred - sg(target)\|\|²` via straight-through | Predictor transformer, Encoder (via z_q_st), Codebook (via softmax@E) | Target features (`sg(target)` blocks it) |

The encoder receives gradient from two sources:
1. Commitment loss (direct term).
2. Prediction loss, via the straight-through path through `z_q_st` and then
   through the predictor's MCMC unroll.

---

## Key mathematical identities

**Straight-through estimator:**
```
z_q_st = z_e + sg(z_q - z_e)

Forward:  z_q_st == z_q   (uses the discrete code)
Backward: ∂L/∂z_e = ∂L/∂z_q_st  (identity copy, ignores quantization)
```

**Prediction decode:**
```
p = softmax(pred_logits)       ∈ R^{B×N×K}
z_pred = p @ codebook.weight   ∈ R^{B×N×C}
```
At convergence, if the predictor is confident about code k:
- `pred_logits` will be large at index k and small elsewhere.
- `z_pred` will be approximately `codebook[k]`.
- This exactly matches what the quantizer would produce for that token.

---

## Module structure

```
model/vid/vq_hvebt/
├── config.py            Dataclasses: VQCodebookConfig, VQStageConfig, VQHVEBTConfig
├── quantizer.py         VectorQuantizer: encode, decode_logits, usage metrics
├── losses.py            Explicit loss functions (math auditable)
├── stage_predictor.py   VQHVEBTStage: forward_energy, run_mcmc
├── clip_backbone.py     VQClipBackbone: trainable CLIP wrapper
├── hierarchy.py         VQHVEBTModel: full model, forward_loss, predict_next
└── __init__.py          Public API exports
```

---

## VQCodebookConfig reference

```python
@dataclass
class VQCodebookConfig:
    num_codes: int = 512       # K — vocabulary size
    code_dim: int = 256        # C — must match clip_channels of the stage
    init_mode: str = "data_first_batch"  # or "random"
    commitment_beta: float = 0.25        # β in β||z_e - sg(z_q)||²
```

### init_mode

| Mode | Behavior |
|------|----------|
| `"random"` | Standard Gaussian init for `nn.Embedding`. Often leads to dead codes early. |
| `"data_first_batch"` | Call `model.maybe_initialize_codebooks(batch)` once before training starts. Seeds codebook entries from real encoder outputs. Recommended. |

---

## VQStageConfig reference

```python
@dataclass
class VQStageConfig:
    clip_stage_name: str     # "s1", "s2", or "s3"
    clip_channels: int       # must match CLIP output: s1=128, s2=256, s3=512
    H: int; W: int           # spatial grid size: s1=32x32, s2=16x16, s3=8x8
    transformer_dim: int = 256
    n_heads: int = 4
    n_layers: int = 4
    mcmc_steps: int = 3
    mcmc_step_size: float = 0.1    # α for logit-space gradient descent
    codebook: VQCodebookConfig = ...
    pred_loss: str = "mse"          # "mse" or "smooth_l1"
    pred_loss_weight: float = 1.0   # λ_pred
    cb_loss_weight: float = 1.0     # λ_cb
    commit_loss_weight: float = 0.25  # λ_commit
```

### mcmc_step_size

Step size `α` is in **logit space**, not feature space. Logit gradients are
much larger than feature-space gradients (because they go through softmax).
Use small values like 0.05–0.2. If MCMC energy explodes, halve the step size.

The step size is a learnable `nn.Parameter` by default (`mcmc_step_learnable=True`).

---

## VQHVEBTConfig reference

```python
@dataclass
class VQHVEBTConfig:
    stages: list[VQStageConfig]      # coarsest → finest: [s3_cfg, s2_cfg, s1_cfg]
    train_encoder: bool = True       # True = CLIP adapts; False = frozen
    encoder_lr_scale: float = 0.1   # encoder LR = base_lr * this
    weights_path: str = "..."        # path to MobileCLIP2-S0 checkpoint
    use_decoder: bool = False
    decoder_loss_weight: float = 1.0
    decoder_out_size: int = 256      # pixel decoder output size (power of 2)
    detach_parent_kv: bool = True    # if True, parent context is detached
```

### Stage ordering

`stages` must be **coarsest first, finest last**:
- `stages[0]` = s3 (8×8, 512ch) — predicted first
- `stages[1]` = s2 (16×16, 256ch)
- `stages[2]` = s1 (32×32, 128ch) — predicted last

Each stage receives the previous stage's prediction as parent context for
cross-attention. This propagates coarse structure information top-down.

---

## Usage examples

### Minimal single-stage setup

```python
from model.vid.vq_hvebt import (
    VQCodebookConfig, VQStageConfig, VQHVEBTConfig, VQHVEBTModel
)

# Single stage on s1 (finest).
stage_cfg = VQStageConfig(
    clip_stage_name="s1",
    clip_channels=128,
    H=32, W=32,
    transformer_dim=256, n_heads=4, n_layers=4,
    mcmc_steps=3, mcmc_step_size=0.1,
    codebook=VQCodebookConfig(num_codes=512, code_dim=128),
)

cfg = VQHVEBTConfig(
    stages=[stage_cfg],
    train_encoder=True,
    encoder_lr_scale=0.1,
    weights_path="clip/MobileCLIP2-S0/mobileclip2_s0.pt",
)

model = VQHVEBTModel(cfg).cuda()
```

### Full three-stage hierarchy

```python
from model.vid.vq_hvebt import VQHVEBTConfig, VQHVEBTModel, _default_stages

cfg = VQHVEBTConfig(
    stages=_default_stages(),   # [s3, s2, s1] with sensible defaults
    train_encoder=True,
    weights_path="clip/MobileCLIP2-S0/mobileclip2_s0.pt",
)
model = VQHVEBTModel(cfg).cuda()
```

### Training loop

```python
import torch
import torch.nn as nn

# Build model and optimizer.
model = VQHVEBTModel(cfg).cuda()

# IMPORTANT: initialize codebooks from real data before first optimizer step.
first_batch = next(iter(train_loader))          # (B, T+1, 3, H, W)
model.maybe_initialize_codebooks(first_batch.cuda())

opt = torch.optim.AdamW(model.parameter_groups(base_lr=3e-4), weight_decay=1e-4)

# Training loop.
model.train()
for step, batch in enumerate(train_loader):
    video = batch.cuda()  # (B, T+1, 3, H, W) in [0, 1]

    opt.zero_grad()
    out = model.forward_loss(video)
    out.total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()

    if step % 100 == 0:
        for k, v in sorted(out.metrics.items()):
            print(f"  {k}: {v:.4f}")
```

### Inference (predict next frame latent)

```python
model.eval()
context = video[:, :4]  # use first 4 frames as context
with torch.no_grad():
    predicted_latents = model.predict_next(context)
    # predicted_latents["s1"]: (B, 128, 32, 32) — finest stage prediction
    # predicted_latents["s3"]: (B, 512, 8, 8)   — coarsest stage prediction
```

---

## Logging and diagnostics

`VQHVEBTOutput.metrics` is a flat dict of floats ready to pass to any logging
framework (wandb, tensorboard, etc.):

| Key | Meaning | Healthy range |
|-----|---------|--------------|
| `s1/loss_pred` | Prediction MSE at s1 | Decreasing |
| `s1/loss_cb` | Codebook MSE at s1 | Small positive |
| `s1/loss_commit` | Commitment loss at s1 | Small positive |
| `s1/codebook_usage` | Fraction of K codes used | > 0.5 (avoid collapse) |
| `s1/codebook_perplexity` | Entropy-based diversity | Close to K means uniform |
| `s1/energy_step0` | Energy before MCMC | Baseline |
| `s1/energy_final` | Energy after MCMC | Should be ≤ energy_step0 |
| `decoder/loss` | Pixel reconstruction L1 | Only if `use_decoder=True` |

**Watch out for**:
- `codebook_usage` near 0: codebook collapse. Try reducing learning rate or
  enabling `data_first_batch` initialization.
- `energy_final > energy_step0`: MCMC is diverging. Reduce `mcmc_step_size`.
- `loss_pred` not decreasing after 1000 steps: check gradient flow with the
  tests in `tests/test_vq_hvebt.py`.

---

## Running the tests

```bash
# All tests (no GPU required, uses fake encoder).
pytest tests/test_vq_hvebt.py -v

# Specific test group.
pytest tests/test_vq_hvebt.py -v -k "TestVectorQuantizer"
pytest tests/test_vq_hvebt.py -v -k "TestNumericalSanity"
pytest tests/test_vq_hvebt.py -v -k "TestOverfitting"

# Fast smoke test (skip overfitting which takes ~3 seconds).
pytest tests/test_vq_hvebt.py -v -k "not TestOverfitting"
```

**The tests do not require CLIP weights.** They patch the CLIP backbone with a
fast fake encoder.

---

## Running the training loop example

```bash
# Quick overfitting smoke test (synthetic data, no real video needed).
python example_code/vq_hvebt_training_loop.py \
    --weights_path clip/MobileCLIP2-S0/mobileclip2_s0.pt \
    --overfit_single_batch \
    --steps 200 \
    --log_every 20

# Real training on stage s1 with frozen encoder (faster to debug predictor).
python example_code/vq_hvebt_training_loop.py \
    --weights_path clip/MobileCLIP2-S0/mobileclip2_s0.pt \
    --stage s1 \
    --freeze_encoder \
    --num_codes 512 \
    --batch_size 4 \
    --T 4 \
    --steps 5000

# Full joint training (encoder + codebook + predictor).
python example_code/vq_hvebt_training_loop.py \
    --weights_path clip/MobileCLIP2-S0/mobileclip2_s0.pt \
    --stage s1 \
    --num_codes 512 \
    --batch_size 4 \
    --T 4 \
    --lr 3e-4 \
    --encoder_lr_scale 0.1 \
    --steps 20000
```

---

## Recommended training phases

Follow the implementation plan phases for stability:

### Phase 1: Single-stage, no decoder

Goal: verify codebook doesn't collapse, prediction loss decreases.

```python
cfg = VQHVEBTConfig(
    stages=[VQStageConfig(clip_stage_name="s3", clip_channels=512, H=8, W=8, ...)],
    train_encoder=True,
    use_decoder=False,
)
```

Start with s3 (coarsest, 8×8 = 64 tokens per frame) — the smallest problem.
Then try s2, then s1.

### Phase 2: Add pixel decoder

```python
cfg = VQHVEBTConfig(
    ...,
    use_decoder=True,
    decoder_out_size=256,
    decoder_loss_weight=0.1,   # start small
)
```

### Phase 3: Full three-stage hierarchy

```python
cfg = VQHVEBTConfig(stages=_default_stages(), ...)
```

The cross-attention from finer to coarser stages provides top-down structural
conditioning. With `detach_parent_kv=True` (default), gradient does NOT flow
from the s1 predictor back into the s3 predictor via cross-attention KV.

---

## Common mistakes to avoid

1. **Not initializing codebooks before training.**
   Always call `model.maybe_initialize_codebooks(first_batch)` before the first
   optimizer step. Otherwise early training is dominated by random codebook noise.

2. **Too large mcmc_step_size.**
   Logit-space MCMC uses step size ~0.1, not ~1000 like feature-space MCMC.
   If energy increases after MCMC steps, halve the step size.

3. **Forgetting to detach the target.**
   The code does this automatically (`z_q[:, 1:].detach().clone()`), but if you
   write a custom loss using `z_q` directly, always detach before using it as
   a target. Otherwise the encoder can reduce loss by shifting the target rather
   than improving the predictor.

4. **Using the old HVEBT code as a base.**
   The `model/vid/hvebt/` directory contains the old continuous + offline-VQ
   implementation. It is NOT the base for this module. Only the utility files
   (`positional.py`, `cross_attention.py`, `decoder.py`) are reused.

5. **Checking gradients without `model.train()`.**
   In eval mode, some autograd paths may be disabled. Always call `model.train()`
   before gradient checks.

---

## File index

| File | Purpose |
|------|---------|
| [model/vid/vq_hvebt/config.py](model/vid/vq_hvebt/config.py) | All configuration dataclasses |
| [model/vid/vq_hvebt/quantizer.py](model/vid/vq_hvebt/quantizer.py) | VQ-VAE quantizer, straight-through, metrics |
| [model/vid/vq_hvebt/losses.py](model/vid/vq_hvebt/losses.py) | Explicit loss functions |
| [model/vid/vq_hvebt/stage_predictor.py](model/vid/vq_hvebt/stage_predictor.py) | EBT predictor per stage |
| [model/vid/vq_hvebt/clip_backbone.py](model/vid/vq_hvebt/clip_backbone.py) | Trainable CLIP wrapper |
| [model/vid/vq_hvebt/hierarchy.py](model/vid/vq_hvebt/hierarchy.py) | Full model, forward_loss, predict_next |
| [model/vid/vq_hvebt/__init__.py](model/vid/vq_hvebt/__init__.py) | Public API |
| [example_code/vq_hvebt_training_loop.py](example_code/vq_hvebt_training_loop.py) | Training loop example |
| [tests/test_vq_hvebt.py](tests/test_vq_hvebt.py) | Comprehensive test suite |
