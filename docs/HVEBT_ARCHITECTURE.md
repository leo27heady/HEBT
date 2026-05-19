# HVEBT Architecture Documentation

**Hierarchical Video Energy-Based Transformer (HVEBT)**

## Overview

HVEBT is a multi-stage hierarchical generative model for video prediction. It uses a lightweight trainable CNN encoder (64×64 RGB → 16×16 / 4×4 / 1×1 features), MCMC Langevin dynamics per stage, and cross-attention between hierarchy levels for top-down conditioning.

## Architecture Diagram

```
Input Video (B, T+1, 3, 64, 64)  [0, 1] RGB
         │
         ▼
┌─────────────────────────────────────────┐
│  LightweightMultiStageEncoder           │
│  (per-frame, trainable CNN)             │
│  Bottom-up: fine → coarse               │
│                                         │
│  16x16: (64,  16, 16)  ◄── Stage 0     │
│  4x4:   (128,  4,  4)  ◄── Stage 1     │
│  1x1:   (256,  1,  1)  ◄── Stage 2     │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Hierarchical MCMC Prediction           │
│  TOP-DOWN: coarse → fine                │
│                                         │
│  Stage 2 (apex): 1×1 space              │
│    256ch, 1×1, embed=256                │
│    Self-attention + 3D RoPE             │
│    NO cross-attention                   │
│         │ (detach)                      │
│         ▼                               │
│  Stage 1: 4×4 space                     │
│    128ch, 4×4, embed=192                │
│    Cross-attn to 1×1 parent (broadcast) │
│         │ (detach)                      │
│         ▼                               │
│  Stage 0 (finest): 16×16 space          │
│    64ch, 16×16, embed=128               │
│    Cross-attn to 4×4 parent (4× down)   │
└─────────────────────────────────────────┘
         │
         ▼ (optional, detached)
┌─────────────────────────────────────────┐
│  Pixel Decoder                          │
│  TransposeConv: 16×16 → 64×64           │
└─────────────────────────────────────────┘
```

## Core Components

### 1. Lightweight Encoder (`model/vid/hvebt/lightweight_encoder.py`)

- **Input**: [0, 1] RGB, 64×64
- **Stages**: three stride-4 conv blocks with GroupNorm + GELU
- **Trainable**: always (jointly with EBT stages)
- **Video handling**: Reshapes (B, T, 3, H, W) → (B*T, 3, H, W), encodes, reshapes back

### 2. Single Stage (`model/vid/hvebt/hvebt.py`)

Each `HVEBTStage` contains:
- **Input projection**: Linear(2 * channels → embed_dim)
- **Transformer blocks**: Self-attention with 3D RoPE + optional cross-attention + FFN
- **Energy head**: per-token scalar, summed for MCMC

**Cross-attention mask** (`build_cross_attn_mask`):
- Same time step only
- Spatial mapping via floor division (e.g. 16×16 child → 4×4 parent at `(yc//4, xc//4)`)
- 4×4 → 1×1: all spatial children at time `t` attend the single apex token

### 3. MCMC Langevin Dynamics

```
pred = init(real_gt)  # zeros / noise / copy (training only for copy)
for step in range(K):
    energy = stage(real_ctx, pred, parent_context=...)
    pred = pred - alpha * grad(energy.sum(), pred)
```

- **real_ctx** = frames 0..T-1; **real_gt** = frames 1..T (no future-frame leakage in context)
- **Loss**: smooth_l1(pred_final, real_gt)

### 4. Data leakage (audit summary)

| Mechanism | Status |
|-----------|--------|
| Per-frame encoder | Safe — no temporal mixing |
| Block-causal self-attention | Safe |
| Cross-attn same-time + spatial mask | Safe |
| `denoising_init=real_current` | Teacher forcing at MCMC step 0 only — do not use at inference |

## Training

```bash
python example_code/hvebt_hierarchical_training_loop.py \
    --stages 16x16 4x4 1x1 --image_size 64 --max_steps 1000 --decoder
```

## File Structure

```
model/vid/hvebt/
├── lightweight_encoder.py   # 64→16→4→1 CNN
├── cross_attention.py
├── decoder.py
├── hierarchical.py
├── hvebt.py
└── positional.py
```
