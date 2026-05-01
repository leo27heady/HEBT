# HVEBT Architecture Documentation

**Hierarchical Video Energy-Based Transformer (HVEBT)**

## Overview

HVEBT is a multi-stage hierarchical generative model for video prediction. It operates in CLIP feature space using MCMC Langevin dynamics to refine predictions, with cross-attention between hierarchy levels providing top-down conditioning.

## Architecture Diagram

```
Input Video (B, T+1, 3, 256, 256)
         │
         ▼
┌─────────────────────────────────────────┐
│  Frozen MobileCLIP2-S0 Encoder          │
│  (per-frame feature extraction)         │
│  Bottom-up: fine → coarse               │
│                                         │
│  stem: (64, 64, 64)                     │
│  s0:   (64, 64, 64)                     │
│  s1:  (128, 32, 32)  ◄── Stage 0       │
│  s2:  (256, 16, 16)  ◄── Stage 1       │
│  s3:  (512,  8,  8)  ◄── Stage 2       │
│  final:(1024, 8, 8)                     │
│  pooled: (512)                          │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Hierarchical MCMC Prediction           │
│  TOP-DOWN: coarse → fine                │
│                                         │
│  Stage 2 (apex): s3 space               │
│    512ch, 8×8, embed=256                │
│    Self-attention + 3D RoPE             │
│    NO cross-attention (starts the chain)│
│    MCMC Langevin refinement             │
│         │ (detach)                      │
│         ▼                               │
│  Stage 1: s2 space                      │
│    256ch, 16×16, embed=192              │
│    Self-attention + Cross-attention     │
│    (child queries attend to Stage 2     │
│     parent KV via child→parent mask)    │
│    MCMC Langevin refinement             │
│         │ (detach)                      │
│         ▼                               │
│  Stage 0 (finest): s1 space             │
│    128ch, 32×32, embed=128              │
│    Self-attention + Cross-attention     │
│    (child queries attend to Stage 1     │
│     parent KV via child→parent mask)    │
│    MCMC Langevin refinement             │
└─────────────────────────────────────────┘
         │
         ▼ (optional, detached)
┌─────────────────────────────────────────┐
│  Pixel Decoder                          │
│  TransposeConv stack: 32×32 → 256×256   │
│  Input: Stage 0 final prediction        │
│  Output: RGB (B, T, 3, 256, 256)        │
└─────────────────────────────────────────┘
```

## Core Components

### 1. CLIP Encoder (`model/vid/hvebt/clip_encoder.py`)

- **Model**: MobileCLIP2-S0 (FastVit visual trunk)
- **Input**: [0, 1] RGB, 256×256
- **Normalization**: CLIP stats applied internally, channel-wise LayerNorm on outputs
- **Mode**: Always frozen (eval mode, no grad)
- **Video handling**: Reshapes (B, T, 3, H, W) → (B*T, 3, H, W), encodes, reshapes back

### 2. Single Stage (`model/vid/hvebt/hvebt.py`)

Each `HVEBTStage` contains:
- **Input projection**: Linear(2 * clip_channels → embed_dim) — concatenation of real context and predicted features
- **Transformer blocks**: Self-attention with 3D RoPE + FFN
- **Energy head**: Linear(embed_dim → 1) per token, summed for scalar energy
- **Cross-attention** (non-apex stages): Child tokens (finer) attend to parent stage's (coarser) detached prediction via child→parent mask

**Token layout**: (B, T×H×W, D) — time-major, then y-major, then x-major.

**Attention mask**: Block-causal across time (frame t sees frames ≤ t), full within each frame.

### 3. MCMC Langevin Dynamics

```
pred = zeros(B, T, C, H, W)  # or noise / copy-last
for step in range(K):
    pred.requires_grad_(True)
    energy = stage(real_ctx, pred, parent_context=...)
    grad = autograd.grad(energy.sum(), pred, create_graph=learning)
    pred = pred - alpha * grad  # alpha is learnable per-stage
```

- **K** (mcmc_num_steps): typically 2-4
- **alpha** (mcmc_step_size): learnable, initialized to 1000.0, clamped ≥ 1e-4
- **Loss**: smooth_l1(pred_final, real_gt) averaged over all MCMC steps (or last step if truncated)
- **Gradient flow**: Through full MCMC unroll (create_graph=True) unless `truncate_mcmc=True`

### 4. Cross-Attention (`model/vid/hvebt/cross_attention.py`)

- **Mask**: Same time-step only + spatial child→parent mapping (each child at (yc,xc) attends to parent at (yc//2, xc//2))
- **KV source**: Previous (coarser) stage's final MCMC prediction, **detached**
- **Direction**: Child (finer) queries attend to parent (coarser) keys/values
- **Effect**: No gradient flows from finer stages back to coarser stages

### 5. 3D RoPE (`model/vid/hvebt/positional.py`)

- Coordinates: (t, y, x) normalized to [0, 1]
- Applied to Q and K in both self-attention and cross-attention
- Head dimension split: 1/3 for t, 1/3 for y, 1/3 for x (padded if not divisible)

### 6. Pixel Decoder (`model/vid/hvebt/decoder.py`)

- Stack of stride-2 TransposeConv2d layers
- Input: (B, T, 128, 32, 32) from Stage 0
- Output: (B, T, 3, 256, 256) in [0, 1]
- Trained independently (detached input), no gradient to HVEBT stages
- Purpose: visualization/debugging tool to verify feature quality

## Data Pipeline

### Synthetic Data (`data/vid/vid_shape_synthetic_dataset.py`)

- Uses `shapekit` library for procedural generation
- **2D mode**: Rotating triangles with various temporal patterns
- **3D mode**: Rotating polycubes (2-6 cubes)
- **Temporal patterns**: acceleration, deceleration, oscillation, interruption
- **Caching**: Renders saved as .npy files with hash-based directory structure
- **Output**: (T+1, 3, H, W) with ImageNet normalization

### Preprocessed Features (`data/vid/preprocessed_clip_dataset.py`)

- Pre-extracted per-stage CLIP features stored as .pt files
- Skips CLIP encoder at training time → significant speedup
- Structure: `{dir}/s1/0.pt`, `{dir}/s2/0.pt`, etc.

## Scripts

### Dataset Generation

```bash
python scripts/generate_dataset.py --type 2d --size 1000 --context_length 8 --output_dir data/vid/shape_cache/2d_1k
```

### CLIP Preprocessing

```bash
python scripts/preprocess_clip_features.py \
    --input_dir data/vid/shape_cache/2d_1k/<hash> \
    --output_dir data/vid/clip_features/2d_1k \
    --stages s1 s2 s3 --image_size 256 --batch_size 16
```

### Training

```bash
# Standard (CLIP in-loop):
python example_code/hvebt_hierarchical_training_loop.py \
    --stages s1 s2 s3 --max_steps 1000 --dataset_size 128 --decoder

# With preprocessed features (faster):
python example_code/hvebt_hierarchical_training_loop.py \
    --stages s1 s2 s3 --max_steps 1000 \
    --preprocessed_dir data/vid/clip_features/2d_1k --decoder
```

## File Structure

```
model/vid/hvebt/
├── __init__.py              # Public exports
├── clip_encoder.py          # Frozen MobileCLIP2-S0 feature extractor
├── cross_attention.py       # Cross-attention with child→parent mask (top-down)
├── decoder.py               # Pixel decoder (TransposeConv stack)
├── hierarchical.py          # Multi-stage model (HierarchicalHVEBT)
├── hvebt.py                 # Single-stage implementation (HVEBTStage)
└── positional.py            # 3D RoPE (t, y, x)

data/vid/
├── vid_shape_synthetic_dataset.py   # Procedural synthetic dataset
└── preprocessed_clip_dataset.py     # Loads precomputed CLIP features

scripts/
├── generate_dataset.py              # Standalone dataset generation
└── preprocess_clip_features.py      # CLIP feature extraction to disk

example_code/
└── hvebt_hierarchical_training_loop.py  # Full training loop
```

## Key Design Decisions

1. **Detached cross-attention**: Finer stages cannot push gradients into coarser stages. Each stage is trained to minimize its own reconstruction loss only. This prevents gradient interference between scales.

2. **MCMC in feature space (not pixel space)**: Operating on compact CLIP features (128-512 channels at 8-32px) is far more efficient than pixel-space MCMC. The energy function learns to score feature-space proposals.

3. **Top-down prediction (coarse→fine)**: The prediction tower processes from the most abstract (coarsest/apex) stage first, providing increasingly refined context as conditioning flows downward. Each child token attends to exactly 1 parent token (its spatial ancestor), keeping cross-attention cost O(N) rather than O(N²).

4. **Learnable step size (alpha)**: Per-stage alpha allows each scale to adapt its MCMC dynamics independently. Initialized large (1000.0) to ensure meaningful initial gradients.

5. **Block-causal temporal mask**: Enables autoregressive video generation while maintaining full spatial attention within each frame.

## Training Metrics

The training loop logs per-stage:
- `init_recon` / `final_recon`: Reconstruction loss at first / last MCMC step
- `energy_gap`: Initial energy − final energy (should be positive = energy decreases)
- `baseline_copy_last`: Loss if we simply copy the previous frame's features (sanity check)
- `alpha`: Current learnable step size
- `grad_norm`: Per-stage gradient norm

A healthy training run shows:
- `final_recon < init_recon` (MCMC improves predictions)
- `final_recon < baseline_copy_last` (model beats trivial baseline)
- `energy_gap > 0` (energy function assigns lower energy to better predictions)
