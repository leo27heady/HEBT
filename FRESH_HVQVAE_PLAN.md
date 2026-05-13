# Fresh Hierarchical VQ-VAE Video Prediction — Implementation Plan

> **Goal**: Clean-room implementation of hierarchical VQ-VAE with hierarchical video prediction.  
> **No reuse of existing model components.** External libs only: PyTorch, `vector-quantize-pytorch`, `einops`.  
> **No MCMC / EBT steps.** Simple transformer prediction with cross-entropy loss.

---

## 1. Architecture Overview

```
                        ENCODER (bottom-up)
                        ==================
  Input: (B, T+1, 3, 64, 64)
         ↓ reshape to (B*(T+1), 3, 64, 64)
         ↓ ConvBlocks (stride-2 convs + residual)
  ┌──────┴──────────────────────────────────────┐
  │  feat_bot: (B*(T+1), C_bot, 16, 16)        │ ──→ LFQ_bot → indices_bot, quant_bot
  │         ↓ ConvBlocks                        │
  │  feat_mid: (B*(T+1), C_mid, 4, 4)          │ ──→ LFQ_mid → indices_mid, quant_mid
  │         ↓ ConvBlocks                        │
  │  feat_top: (B*(T+1), C_top, 1, 1)          │ ──→ LFQ_top → indices_top, quant_top
  └─────────────────────────────────────────────┘

                UPSCALERS (stage → unified 16×16)
                ==================================
  upscale_mid: (B, C_mid, 4, 4)  → (B, C_bot, 16, 16)   [ConvTranspose ×2]
  upscale_top: (B, C_top, 1, 1)  → (B, C_bot, 16, 16)   [ConvTranspose ×2]
  (bot needs no upscaling — already 16×16)

                    DECODER (shared, 16×16 → RGB)
                    ==============================
  Input: (B, C_bot, 16, 16) — from ANY stage (bot direct, mid/top upscaled)
         ↓ ConvTranspose (parametrized upsampling)
  Output: (B, 3, 64, 64)
         → per-stage MSE with SAME input frames (trains encoder+upscalers+decoder)

                        PREDICTOR (top-down)
                        ====================
  ┌─────────────────────────────────────────────┐
  │  Top predictor: Self-attn (causal, window=T)│
  │    Input: quant_top[:, 0:T]                 │
  │    Output: logits_top (B, T, K_top)         │
  │         ↓ soft-lookup + proj → feat_top     │
  │                                             │
  │  Mid predictor: Self-attn (causal, window=2)│
  │    + Cross-attn from feat_top               │
  │    Input: quant_mid[:, 0:T]                 │
  │    Output: logits_mid (B, T*16, K_mid)      │
  │         ↓ soft-lookup + proj → feat_mid     │
  │                                             │
  │  Bot predictor: Self-attn (window=1, spatial)│
  │    + Cross-attn from feat_mid               │
  │    Input: quant_bot[:, 0:T]                 │
  │    Output: logits_bot (B, T*256, K_bot)     │
  └─────────────────────────────────────────────┘

  Predictor trained by per-stage CE ONLY (isolated, encoder frozen).
  No prediction-MSE. No gradient between predictor and encoder.
```

---

## 2. VQ Choice: LFQ (Lookup-Free Quantization)

### Why LFQ over SimVQ

| Criterion | LFQ | SimVQ |
|---|---|---|
| Native video support | Yes `(B, C, T, H, W)` | No (sequence only) |
| Codebook collapse prevention | Built-in entropy regularization | Frozen codebook (mixed results per author) |
| Auxiliary losses | Entropy loss only (no commitment loss) | Commitment loss needed |
| Proven for video | MagViT-v2 (SOTA) | Not tested on video |
| Discrete targets for CE | Clean integer indices | Clean integer indices |
| Maturity | Well-tested, published | "Hearing mixed results" (README note) |

### LFQ Configuration Per Stage

| Stage | Spatial | codebook_size | LFQ dim | Notes |
|---|---|---|---|---|
| Top | 1×1 | 512 (2^9) | 9 | Small spatial = smaller codebook sufficient |
| Mid | 4×4 | 1024 (2^10) | 10 | Medium spatial complexity |
| Bot | 16×16 | 4096 (2^12) | 12 | Fine spatial details need more codes |

### LFQ API Usage

```python
from vector_quantize_pytorch import LFQ

vq_bot = LFQ(
    codebook_size=4096,       # 2^12
    dim=12,                   # log2(codebook_size)
    entropy_loss_weight=0.1,
    diversity_gamma=1.0,
    accept_image_fmap=True,   # REQUIRED: encoder outputs (B, C, H, W) format
)

# Input: (B*(T+1), 12, 16, 16) — channel-first spatial tensor
# Output: quantized (B*(T+1), 12, 16, 16), indices (B*(T+1), 16, 16), entropy_loss ()
```

**Important**: `accept_image_fmap=True` is required because our encoder outputs are `(B, C, H, W)` format (standard Conv2d output). Without this flag, LFQ expects channels-last `(B, ..., C)`.

The encoder's final projection per stage maps `C_stage → lfq_dim` before quantization. After VQ, a projection maps `lfq_dim → C_stage` back for predictor input and decoder input.

---

## 3. Encoder Architecture

### 3.1 Design

Single bottom-up convolutional encoder with VQ taps at each resolution.

```python
class HierarchicalEncoder(nn.Module):
    """
    Input:  (B, 3, 64, 64)
    Output: features at 3 resolutions + VQ outputs at each
    """
    def __init__(self, C_bot=128, C_mid=192, C_top=256, K_bot=4096, K_mid=1024, K_top=512):
        # Stage 1: 64x64 → 16x16 (stride 4 total: two stride-2 conv blocks)
        self.enc_to_bot = nn.Sequential(
            ResBlock(3, 64, stride=2),      # 64→32
            ResBlock(64, C_bot, stride=2),  # 32→16
        )
        self.bot_to_vq = nn.Conv2d(C_bot, 12, 1)  # project to LFQ dim
        self.bot_from_vq = nn.Conv2d(12, C_bot, 1)  # project back
        self.vq_bot = LFQ(codebook_size=K_bot, dim=12, entropy_loss_weight=0.1,
                          accept_image_fmap=True)
        
        # Stage 2: 16x16 → 4x4 (stride 4 total: two stride-2 blocks)
        self.enc_bot_to_mid = nn.Sequential(
            ResBlock(C_bot, C_mid, stride=2),  # 16→8
            ResBlock(C_mid, C_mid, stride=2),  # 8→4
        )
        self.mid_to_vq = nn.Conv2d(C_mid, 10, 1)
        self.mid_from_vq = nn.Conv2d(10, C_mid, 1)
        self.vq_mid = LFQ(codebook_size=K_mid, dim=10, entropy_loss_weight=0.1,
                          accept_image_fmap=True)
        
        # Stage 3: 4x4 → 1x1 (adaptive avg pool or stride-4 conv)
        self.enc_mid_to_top = nn.Sequential(
            ResBlock(C_mid, C_top, stride=2),  # 4→2
            ResBlock(C_top, C_top, stride=2),  # 2→1
        )
        self.top_to_vq = nn.Conv2d(C_top, 9, 1)
        self.top_from_vq = nn.Conv2d(9, C_top, 1)
        self.vq_top = LFQ(codebook_size=K_top, dim=9, entropy_loss_weight=0.1,
                          accept_image_fmap=True)
    
    def forward(self, x):
        """x: (B, 3, 64, 64) → dict of quant features and indices per stage"""
        # Bot
        feat_bot = self.enc_to_bot(x)               # (B, C_bot, 16, 16)
        z_bot = self.bot_to_vq(feat_bot)            # (B, 12, 16, 16)
        quant_bot, idx_bot, loss_bot = self.vq_bot(z_bot)
        quant_bot_feat = self.bot_from_vq(quant_bot) # (B, C_bot, 16, 16)
        
        # Mid (takes features BEFORE VQ to preserve gradient for encoder training)
        feat_mid = self.enc_bot_to_mid(feat_bot)    # (B, C_mid, 4, 4)
        z_mid = self.mid_to_vq(feat_mid)
        quant_mid, idx_mid, loss_mid = self.vq_mid(z_mid)
        quant_mid_feat = self.mid_from_vq(quant_mid)
        
        # Top
        feat_top = self.enc_mid_to_top(feat_mid)    # (B, C_top, 1, 1)
        z_top = self.top_to_vq(feat_top)
        quant_top, idx_top, loss_top = self.vq_top(z_top)
        quant_top_feat = self.top_from_vq(quant_top)
        
        return {
            'quant_bot': quant_bot_feat, 'idx_bot': idx_bot, 'loss_bot': loss_bot,
            'quant_mid': quant_mid_feat, 'idx_mid': idx_mid, 'loss_mid': loss_mid,
            'quant_top': quant_top_feat, 'idx_top': idx_top, 'loss_top': loss_top,
        }
```

### 3.2 ResBlock Design

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride) if (in_ch != out_ch or stride != 1) else nn.Identity()
    
    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))
```

---

## 4. Decoder & Upscaler Architecture

### 4.1 Shared Decoder (16×16 → 64×64)

Single decoder module, always receives `(B, C_bot, 16, 16)` regardless of which stage produced it.

```python
class Decoder(nn.Module):
    """
    Input:  (B, C_bot, 16, 16) — from bot direct, or mid/top upscaled
    Output: (B, 3, 64, 64)     — reconstructed RGB
    """
    def __init__(self, C_bot=128):
        self.decode = nn.Sequential(
            ResBlock(C_bot, C_bot),
            nn.ConvTranspose2d(C_bot, 64, 4, stride=2, padding=1),  # 16→32
            nn.SiLU(),
            ResBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     # 32→64
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),  # output in [0,1]
        )
    
    def forward(self, x):
        return self.decode(x)
```

### 4.2 Upscalers (Mid/Top → 16×16 unified input for decoder)

Shallow ConvTranspose modules that upscale mid/top quantized features to match bot's 16×16 spatial size and C_bot channels.

```python
class UpscaleMid(nn.Module):
    """(B, C_mid, 4, 4) → (B, C_bot, 16, 16)"""
    def __init__(self, C_mid=192, C_bot=128):
        self.up = nn.Sequential(
            nn.ConvTranspose2d(C_mid, C_bot, 4, stride=2, padding=1),  # 4→8
            nn.SiLU(),
            nn.ConvTranspose2d(C_bot, C_bot, 4, stride=2, padding=1),  # 8→16
            nn.SiLU(),
        )
    
    def forward(self, x):
        return self.up(x)


class UpscaleTop(nn.Module):
    """(B, C_top, 1, 1) → (B, C_bot, 16, 16)"""
    def __init__(self, C_top=256, C_bot=128):
        self.up = nn.Sequential(
            nn.ConvTranspose2d(C_top, C_bot, 4, stride=4, padding=0),  # 1→4
            nn.SiLU(),
            nn.ConvTranspose2d(C_bot, C_bot, 4, stride=4, padding=0),  # 4→16
            nn.SiLU(),
        )
    
    def forward(self, x):
        return self.up(x)
```

**Output size verification**:
- `UpscaleMid`: (4-1)*2 - 2*1 + 4 = 8, then (8-1)*2 - 2*1 + 4 = 16 ✓
- `UpscaleTop`: (1-1)*4 - 0 + 4 = 4, then (4-1)*4 - 0 + 4 = 16 ✓

### 4.3 Reconstruction Data Flow

```
Bot path:  quant_bot → bot_from_vq → (B, C_bot, 16, 16) → Decoder → RGB
Mid path:  quant_mid → mid_from_vq → (B, C_mid, 4, 4) → UpscaleMid → (B, C_bot, 16, 16) → Decoder → RGB
Top path:  quant_top → top_from_vq → (B, C_top, 1, 1) → UpscaleTop → (B, C_bot, 16, 16) → Decoder → RGB
```

All three paths produce `(B, 3, 64, 64)` and are compared with the SAME input frame via MSE.

---

## 5. Predictor Architecture

### 5.1 Overview

Three transformer-based predictor stages, running top-down. Each stage:
1. Takes quantized features from encoder as input sequence
2. Self-attention with stage-specific temporal window (causal)
3. Cross-attention from parent stage's predicted features (except top)
4. Outputs logits over that stage's codebook

### 5.2 Predictor Stage Module

```python
class PredictorStage(nn.Module):
    def __init__(self, dim, n_heads, n_layers, codebook_size, spatial_size, 
                 temporal_window, has_parent=False, parent_dim=None):
        """
        Args:
            dim: internal transformer dim (= C_stage)
            n_heads: attention heads
            n_layers: transformer blocks
            codebook_size: output logit size
            spatial_size: H*W for this stage (1, 16, or 256)
            temporal_window: how many frames each position can attend back
            has_parent: whether to include cross-attention layers
            parent_dim: dimension of parent features (for cross-attn KV projection)
        """
        self.spatial_size = spatial_size
        self.temporal_window = temporal_window
        
        # Input projection (from quantized features)
        self.input_proj = nn.Linear(dim, dim)
        
        # Positional encodings
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, spatial_size, dim) * 0.02)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, MAX_T, dim) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(TransformerBlock(
                dim=dim, n_heads=n_heads, has_cross_attn=has_parent
            ))
        
        # Output head: predict code distribution
        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, codebook_size),
        )
        
        # For soft-lookup to produce cross-attn features for child stage
        # (codebook weights are shared from the VQ layer)
        self.codebook_weights = None  # set externally from VQ
    
    def forward(self, quant_input, parent_features=None, T=None):
        """
        Args:
            quant_input: (B, T*S, dim) — quantized features flattened (S=spatial_size)
            parent_features: (B, T*S_parent, dim_parent) or None
            T: number of frames
        Returns:
            logits: (B, T*S, codebook_size)
            pred_features: (B, T*S, dim) — soft-decoded for child cross-attn
        """
        B = quant_input.shape[0]
        S = self.spatial_size
        
        # Add positional encoding
        x = self.input_proj(quant_input)
        x = x + self._get_pos_encoding(T, S, x.device)
        
        # Build self-attention mask
        self_attn_mask = self._build_temporal_window_mask(T, S, x.device)
        
        # Build cross-attention mask (if parent)
        cross_attn_mask = None
        if parent_features is not None:
            cross_attn_mask = self._build_cross_attn_mask(T, S, parent_features, x.device)
        
        # Transformer forward
        for layer in self.layers:
            x = layer(x, self_attn_mask=self_attn_mask, 
                      cross_kv=parent_features, cross_attn_mask=cross_attn_mask)
        
        # Output logits
        logits = self.output_head(x)  # (B, T*S, codebook_size)
        
        # Soft-lookup for child stage cross-attention
        pred_features = self._soft_lookup(logits)  # (B, T*S, dim)
        
        return logits, pred_features
```

### 5.3 Soft Lookup (Differentiable Code-to-Feature Mapping)

Used ONLY for cross-attention between predictor stages. Not used for reconstruction or final decoding.

```python
def _soft_lookup(self, logits, temperature=1.0):
    """
    Convert logits to continuous feature vectors via soft codebook lookup + projection.
    logits: (B, T*S, codebook_size)
    Returns: (B, T*S, dim)  ← projected to C_stage dimension for cross-attn
    """
    probs = F.softmax(logits / temperature, dim=-1)  # (B, T*S, K)
    # codebook_weights: (K, lfq_dim) — registered buffer, all binary code vectors
    soft_codes = probs @ self.codebook_weights        # (B, T*S, lfq_dim)
    # Project from lfq_dim → dim (C_stage) for cross-attention compatibility
    pred_features = self.soft_lookup_proj(soft_codes)  # (B, T*S, dim)
    return pred_features
```

**Dimension fix**: Raw soft-lookup produces `lfq_dim` (9/10/12). The child predictor's cross-attention expects `parent_dim = C_stage` (256/192/128). The `soft_lookup_proj = nn.Linear(lfq_dim, dim)` bridges this gap. Each predictor stage has its own projection.

```python
# In PredictorStage.__init__:
self.soft_lookup_proj = nn.Linear(lfq_dim, dim)  # lfq_dim → C_stage
```

**Note on LFQ**: LFQ doesn't have a traditional codebook matrix. Its codes are binary vectors of dim `d`, producing `2^d` possible codes. For soft lookup, we precompute all `2^d` binary code vectors and use them as the "codebook weight matrix". Since our dims are 9-12, this means matrices of size 512×9, 1024×10, 4096×12 — all perfectly manageable.

```python
# Precompute LFQ codebook matrix for soft lookup
def build_lfq_codebook_matrix(dim):
    """Build all 2^dim binary code vectors as a matrix."""
    K = 2 ** dim
    codes = torch.zeros(K, dim)
    for i in range(K):
        for bit in range(dim):
            codes[i, bit] = 1.0 if (i >> bit) & 1 else -1.0  # LFQ uses {-1, +1}
    return codes  # (K, dim)
```

**IMPORTANT**: These matrices must be registered as buffers (not plain attributes) so they move to GPU with `.cuda()`:
```python
# In model __init__:
self.register_buffer('codebook_top', build_lfq_codebook_matrix(cfg.lfq_dim_top))
self.register_buffer('codebook_mid', build_lfq_codebook_matrix(cfg.lfq_dim_mid))
self.register_buffer('codebook_bot', build_lfq_codebook_matrix(cfg.lfq_dim_bot))
# Then assign to predictor stages:
self.predictor_top.codebook_weights = self.codebook_top  # shares same tensor
```

---

## 6. Attention Masking — Detailed Specification

### 6.1 Self-Attention Mask (Temporal Windowing)

The sequence for each predictor stage is `(B, T * S, dim)` where `S` = spatial positions per frame.

Token at position `i` belongs to:
- Frame `t = i // S`
- Spatial position `s = i % S`

**Mask rule**: Token at position `i` (frame `t_i`) can attend to token at position `j` (frame `t_j`) if and only if:
1. `t_j <= t_i` (causal: can't attend to future frames)
2. `t_i - t_j < temporal_window` (within the window)

**No spatial restriction** — within allowed frames, all spatial positions attend to all spatial positions.

```python
def _build_temporal_window_mask(self, T, S, device):
    """
    Build boolean attention mask: (T*S, T*S)
    True = ALLOWED to attend, False = BLOCKED.
    """
    seq_len = T * S
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    
    for i in range(seq_len):
        t_i = i // S
        for j in range(seq_len):
            t_j = j // S
            # Causal + window
            if t_j <= t_i and (t_i - t_j) < self.temporal_window:
                mask[i, j] = True
    
    return mask  # True = attend, False = mask out
```

**Efficient implementation** (actual code should use this):
```python
def _build_temporal_window_mask(self, T, S, device):
    """Efficient vectorized mask construction."""
    frame_indices = torch.arange(T, device=device)
    # Frame-level mask: (T, T) — causal + window
    frame_mask = (frame_indices.unsqueeze(0) - frame_indices.unsqueeze(1))  # (T, T)
    frame_mask = (frame_mask >= 0) & (frame_mask < self.temporal_window)    # (T, T)
    
    # Expand to token level: each frame has S tokens, all with same temporal mask
    # (T, T) → (T*S, T*S) via kronecker-style expansion
    token_mask = frame_mask.repeat_interleave(S, dim=0).repeat_interleave(S, dim=1)
    return token_mask
```

### 6.2 Concrete Mask Examples

**Top stage** (S=1, T=4, window=4):
```
Frame: 0 1 2 3
    0 [1 0 0 0]   Frame 0 sees: [0]
    1 [1 1 0 0]   Frame 1 sees: [0,1]
    2 [1 1 1 0]   Frame 2 sees: [0,1,2]
    3 [1 1 1 1]   Frame 3 sees: [0,1,2,3]
```
→ Standard lower-triangular causal mask.

**Mid stage** (S=16, T=4, window=2):
```
Frame: 0 1 2 3
    0 [1 0 0 0]   Frame 0 sees: [0]
    1 [1 1 0 0]   Frame 1 sees: [0,1]
    2 [0 1 1 0]   Frame 2 sees: [1,2]
    3 [0 0 1 1]   Frame 3 sees: [2,3]
```
Then each `1` in this 4×4 frame mask becomes a 16×16 block of `True` in the full (64×64) mask.

**Bot stage** (S=256, T=4, window=1):
```
Frame: 0 1 2 3
    0 [1 0 0 0]   Frame 0 sees: [0]
    1 [0 1 0 0]   Frame 1 sees: [1]
    2 [0 0 1 0]   Frame 2 sees: [2]
    3 [0 0 0 1]   Frame 3 sees: [3]
```
→ Block-diagonal mask. Each frame's 256 tokens only attend to each other. **No temporal leakage at bot level.** All temporal information comes from parent cross-attention.

### 6.3 Cross-Attention Mask (Spatial Hierarchy)

Cross-attention: child tokens QUERY, parent tokens are KEY/VALUE.

**Cross-attn mask shape**: `(T*S_child, T*S_parent)`

**Mask rules**:
1. **Temporal alignment**: child at frame `t` attends to parent at frame `t` only (same time step, as both predict frame `t+1`)
2. **Spatial hierarchy**: child spatial position `(r, c)` attends to its parent at `(r // ratio, c // ratio)`

**Top → Mid** (S_parent=1, S_child=16, ratio=4):
- Each mid token attends to the single top token at the same time step
- Mask: block-diagonal in time, all-ones within each time block (since there's only 1 parent per frame)

```python
def _build_cross_attn_mask_top_to_mid(self, T, device):
    """(T*16, T*1) mask — mid queries, top keys."""
    mask = torch.zeros(T * 16, T * 1, dtype=torch.bool, device=device)
    for t in range(T):
        # All 16 mid tokens at frame t attend to the 1 top token at frame t
        mask[t*16:(t+1)*16, t:t+1] = True
    return mask
```

**Mid → Bot** (S_parent=16, S_child=256, ratio=4):
- Bot at spatial `(r, c)` attends to mid at `(r//4, c//4)` at the same frame
- Within frame `t`: bot token `s_child = r*16 + c` attends to mid token `s_parent = (r//4)*4 + (c//4)`

```python
def _build_cross_attn_mask_mid_to_bot(self, T, device):
    """(T*256, T*16) mask — bot queries, mid keys."""
    mask = torch.zeros(T * 256, T * 16, dtype=torch.bool, device=device)
    for t in range(T):
        for r in range(16):
            for c in range(16):
                child_idx = t * 256 + r * 16 + c
                parent_r, parent_c = r // 4, c // 4
                parent_idx = t * 16 + parent_r * 4 + parent_c
                mask[child_idx, parent_idx] = True
    return mask
```

**Efficient version**:
```python
def _build_cross_attn_mask_mid_to_bot(self, T, device):
    """Vectorized construction."""
    # For each bot spatial position, compute parent index
    r = torch.arange(16, device=device)
    c = torch.arange(16, device=device)
    grid_r, grid_c = torch.meshgrid(r, c, indexing='ij')  # (16, 16) each
    parent_r = grid_r // 4  # (16, 16)
    parent_c = grid_c // 4
    parent_idx = parent_r * 4 + parent_c  # (16, 16) — flat index into 4x4 parent grid
    parent_idx_flat = parent_idx.reshape(256)  # (256,)
    
    # Build per-frame mask: (256, 16)
    frame_mask = torch.zeros(256, 16, dtype=torch.bool, device=device)
    frame_mask[torch.arange(256, device=device), parent_idx_flat] = True
    
    # Expand to full sequence: block-diagonal in time
    # Full mask: (T*256, T*16)
    full_mask = torch.zeros(T * 256, T * 16, dtype=torch.bool, device=device)
    for t in range(T):
        full_mask[t*256:(t+1)*256, t*16:(t+1)*16] = frame_mask
    return full_mask
```

### 6.4 Key Difference from EBT Masking

In EBT, the sequence is `[context_0, ..., context_T, pred_0, ..., pred_T]` (length 2T). The mask must handle cross-group visibility.

**In THIS implementation**: the sequence is simply `[frame_0, frame_1, ..., frame_{T-1}]`. Standard causal masking with windowing. The prediction is implicit: at position `t`, the output logits are the prediction for frame `t+1`. **No combined context-prediction sequence. No 2T trick.**

---

## 7. Training Strategy — Per-Stage Reconstruction + Isolated Prediction

### 7.1 Design Principle

Complete separation of concerns:
- **Reconstruction losses** (per-stage MSE): train encoder + upscalers + decoder to faithfully encode/decode at each resolution
- **Prediction losses** (per-stage CE): train each predictor stage to predict next-frame codes

No loss type ever touches the other's parameters. No prediction-MSE through the predictor. No "graph-live frozen params" complexity.

### 7.2 Loss Inventory

| Loss | Formula | Updates | Gradient reaches |
|---|---|---|---|
| `mse_bot` | MSE(decoder(bot_from_vq(quant_bot)), input_frame) | encoder bot, bot_from_vq, decoder | enc_to_bot, bot_to_vq, bot_from_vq, decoder |
| `mse_mid` | MSE(decoder(upscale_mid(mid_from_vq(quant_mid))), input_frame) | encoder bot+mid, mid_from_vq, upscale_mid, decoder | enc_to_bot, enc_bot_to_mid, mid_to_vq, mid_from_vq, upscale_mid, decoder |
| `mse_top` | MSE(decoder(upscale_top(top_from_vq(quant_top))), input_frame) | entire encoder, top_from_vq, upscale_top, decoder | ALL encoder conv blocks + VQ projections, upscale_top, decoder |
| `vq_entropy` | LFQ internal entropy losses (sum of 3 stages) | encoder VQ projections (*_to_vq) | *_to_vq layers via STE |
| `ce_top` | CE(logits_top, next_frame_idx_top) | pred_top ONLY | pred_top params only |
| `ce_mid` | CE(logits_mid, next_frame_idx_mid) | pred_mid ONLY | pred_mid params only |
| `ce_bot` | CE(logits_bot, next_frame_idx_bot) | pred_bot ONLY | pred_bot params only |

### 7.3 Single Forward, Isolated Backwards

All losses computed in ONE forward pass. Each loss has a disjoint computation graph thanks to `.detach()`. No `retain_graph` needed.

```python
# Optimizers — strictly separated parameter groups
opt_enc_dec = Adam(
    list(encoder.parameters()) + list(decoder.parameters()) +
    list(upscale_mid.parameters()) + list(upscale_top.parameters()),
    lr=3e-4
)
opt_pred_top = Adam(predictor_top.parameters(), lr=3e-4)
opt_pred_mid = Adam(predictor_mid.parameters(), lr=3e-4)
opt_pred_bot = Adam(predictor_bot.parameters(), lr=1e-4)  # more tokens → lower LR
```

### 7.4 Training Step (Complete)

```python
def training_step(batch):
    """
    batch: (B, T+1, 3, 64, 64)
    Single forward pass, 4 isolated backward passes.
    """
    B, Tp1, C, H, W = batch.shape
    T = Tp1 - 1
    all_frames = batch.reshape(B * Tp1, C, H, W)  # reconstruction targets (same frame)
    
    # ═══════════════════════════════════════════════════════
    # FORWARD: Encoder (WITH gradient for reconstruction)
    # ═══════════════════════════════════════════════════════
    enc = model.encode(batch)  # returns dict with quant features, indices, vq losses
    
    # ═══════════════════════════════════════════════════════
    # RECONSTRUCTION LOSSES (encoder + upscalers + decoder)
    # Target: SAME input frames (standard VQ-VAE autoencoder)
    # ═══════════════════════════════════════════════════════
    # Bot: already 16×16, direct to decoder
    quant_bot_spatial = enc['quant_bot'].reshape(B * Tp1, cfg.C_bot, 16, 16)
    recon_bot = decoder(quant_bot_spatial)
    mse_bot = F.mse_loss(recon_bot, all_frames)
    
    # Mid: 4×4 → upscale → 16×16 → decoder
    quant_mid_spatial = enc['quant_mid'].reshape(B * Tp1, cfg.C_mid, 4, 4)
    recon_mid = decoder(upscale_mid(quant_mid_spatial))
    mse_mid = F.mse_loss(recon_mid, all_frames)
    
    # Top: 1×1 → upscale → 16×16 → decoder
    quant_top_spatial = enc['quant_top'].reshape(B * Tp1, cfg.C_top, 1, 1)
    recon_top = decoder(upscale_top(quant_top_spatial))
    mse_top = F.mse_loss(recon_top, all_frames)
    
    # VQ entropy losses
    vq_loss = enc['loss_bot'] + enc['loss_mid'] + enc['loss_top']
    
    # Combined reconstruction loss (weighted)
    recon_total = 1.0 * mse_bot + 0.5 * mse_mid + 0.1 * mse_top + vq_loss
    
    # ═══════════════════════════════════════════════════════
    # PREDICTION CE LOSSES (predictor stages, isolated)
    # Target: NEXT frame codes (shifted by 1)
    # Encoder output DETACHED — no gradient to encoder.
    # ═══════════════════════════════════════════════════════
    # Detach encoder outputs for predictor (disjoint graph)
    inp_top = enc['quant_top'][:, :T].detach().reshape(B, T * 1, -1)
    inp_mid = enc['quant_mid'][:, :T].detach().reshape(B, T * 16, -1)
    inp_bot = enc['quant_bot'][:, :T].detach().reshape(B, T * 256, -1)
    
    # Target: next-frame code indices
    tgt_top = enc['idx_top'][:, 1:].detach().reshape(B, T * 1)
    tgt_mid = enc['idx_mid'][:, 1:].detach().reshape(B, T * 16)
    tgt_bot = enc['idx_bot'][:, 1:].detach().reshape(B, T * 256)
    
    # Top predictor — fully independent
    logits_top, feat_top = predictor_top(inp_top, T=T)
    ce_top = F.cross_entropy(logits_top.reshape(-1, K_top), tgt_top.reshape(-1))
    
    # Mid predictor — parent features DETACHED (no gradient to top)
    logits_mid, feat_mid = predictor_mid(
        inp_mid, parent_features=feat_top.detach(), T=T
    )
    ce_mid = F.cross_entropy(logits_mid.reshape(-1, K_mid), tgt_mid.reshape(-1))
    
    # Bot predictor — parent features DETACHED (no gradient to mid)
    logits_bot, _ = predictor_bot(
        inp_bot, parent_features=feat_mid.detach(), T=T
    )
    ce_bot = F.cross_entropy(logits_bot.reshape(-1, K_bot), tgt_bot.reshape(-1))
    
    # ═══════════════════════════════════════════════════════
    # ISOLATED BACKWARD + STEP (4 disjoint graphs)
    # No retain_graph needed — each loss has independent graph.
    # ═══════════════════════════════════════════════════════
    opt_enc_dec.zero_grad()
    recon_total.backward()
    torch.nn.utils.clip_grad_norm_(encoder_decoder_params, max_norm=1.0)
    opt_enc_dec.step()
    
    opt_pred_top.zero_grad()
    ce_top.backward()
    opt_pred_top.step()
    
    opt_pred_mid.zero_grad()
    ce_mid.backward()
    opt_pred_mid.step()
    
    opt_pred_bot.zero_grad()
    ce_bot.backward()
    opt_pred_bot.step()
    
    return {
        'mse_bot': mse_bot.item(), 'mse_mid': mse_mid.item(), 'mse_top': mse_top.item(),
        'vq_loss': vq_loss.item(),
        'ce_top': ce_top.item(), 'ce_mid': ce_mid.item(), 'ce_bot': ce_bot.item(),
    }
```

### 7.5 Why This Works

1. **No conflicting gradients**: Each parameter group receives gradient from exactly one loss type.
   - Encoder/decoder/upscalers ← reconstruction MSE only
   - Each predictor stage ← its own CE only
   - No cross-contamination between reconstruction and prediction

2. **Per-stage reconstruction ensures encoder quality**: 
   - `mse_bot` forces fine-detail encoding at 16×16
   - `mse_mid` forces medium-structure encoding at 4×4
   - `mse_top` forces global-content encoding at 1×1
   - Each VQ stage MUST carry useful information — can't cheat by collapsing to unused codes

3. **Gradient flow through encoder (reconstruction)**:
   - `mse_bot` → decoder → bot_from_vq → STE → bot_to_vq → enc_to_bot
   - `mse_mid` → decoder → upscale_mid → mid_from_vq → STE → mid_to_vq → enc_bot_to_mid → enc_to_bot
   - `mse_top` → decoder → upscale_top → top_from_vq → STE → top_to_vq → enc_mid_to_top → enc_bot_to_mid → enc_to_bot
   
   Note: mid/top losses reach bot encoder layers because the encoder is feedforward (each stage takes pre-VQ features from previous stage).

4. **No chicken-and-egg problem**: Encoder/decoder learn independently of predictor from step 1. Predictor receives meaningful codes from step 1 because reconstruction loss immediately forces the encoder to produce useful VQ codes.

5. **No distribution shift at inference**: Decoder always receives exact VQ code vectors (via `*_from_vq`), whether from encoding or from predicted argmax codes. Identical format.

### 7.6 Reconstruction Loss Weighting

Bot encoder layers receive gradient from ALL THREE MSE losses. To avoid over-weighting:

```python
recon_total = 1.0 * mse_bot + 0.5 * mse_mid + 0.1 * mse_top + vq_loss
```

Rationale:
- `mse_bot` has lowest inherent MSE (16×16 is near-lossless) — weight 1.0
- `mse_mid` has medium MSE (4×4 loses detail) — weight 0.5
- `mse_top` has highest MSE (1×1 = 512 possible images, always blurry) — weight 0.1

These weights are tuning knobs. Start here, adjust if one stage dominates gradient.

### 7.7 Per-Stage CE (Prediction) — Isolation via Detach

Each predictor stage is fully isolated:
- **Inputs** from encoder: `.detach()` → no gradient to encoder
- **Parent features** from higher stage: `.detach()` → no gradient leaks between stages
- **Separate optimizer**: each stage has its own Adam instance

This means:
- Bot's 256× more tokens don't overwhelm top/mid
- Each stage can have its own learning rate
- A badly-trained bot predictor can't corrupt top/mid

### 7.8 Verification Metric (Hard Decode — Not Used for Training)

Periodically decode **hard** predicted codes (argmax, no soft-lookup) to measure true inference-time quality:

```python
@torch.no_grad()
def compute_hard_prediction_mse(batch, model):
    """Verification: do argmax-predicted codes decode to correct frames?"""
    B, Tp1, C, H, W = batch.shape
    T = Tp1 - 1
    enc = model.encode(batch)
    
    # Run predictor
    inp_bot = enc['quant_bot'][:, :T].reshape(B, T * 256, -1)
    # ... (run full predictor hierarchy) ...
    logits_bot = ...
    
    # Hard decode
    pred_indices = logits_bot.argmax(dim=-1)  # (B, T*256)
    # Use LFQ's indices_to_codes to get binary vectors
    pred_codes = model.encoder.vq_bot.indices_to_codes(pred_indices)  # (B, T*256, 12)
    pred_spatial = pred_codes.reshape(B*T, 16, 16, 12).permute(0, 3, 1, 2)
    pred_rgb = model.decoder(model.encoder.bot_from_vq(pred_spatial))
    
    target_rgb = batch[:, 1:].reshape(B*T, 3, 64, 64)
    return F.mse_loss(pred_rgb, target_rgb).item()
```

This measures how well predicted codes decode to actual future frames — the true end-to-end quality at inference time.

---

## 8. Codebook Collapse Prevention

### 8.1 Why LFQ Mitigates Collapse

LFQ has **no learnable codebook** — codes are fixed binary vectors `{-1, +1}^d`. The "codebook" is the set of all `2^d` corners of the hypercube. This eliminates:
- Dead codes from EMA drift
- Mode collapse to subset of codes
- Codebook requires no maintenance

The entropy regularization in LFQ encourages uniform usage:
```python
# Built into LFQ:
# entropy_loss = -H(avg_code_probs) + diversity_gamma * H(per_sample_probs)
# This pushes for diverse code usage across batch while confident per-sample
```

### 8.2 Additional Measures

1. **Per-stage reconstruction loss**: Each stage has its own MSE forcing it to carry useful information. Unlike pure prediction-only training (where encoder could cheat by collapsing codes), reconstruction requires distinct codes to represent distinct frames.

2. **Per-stage CE as additional signal**: The predictor must predict future codes — if codes are collapsed (all frames map to same code), prediction is trivial but reconstruction fails. The two signals are complementary.

3. **Monitor codebook utilization**: Track unique codes used per batch. LFQ should naturally use most codes due to entropy loss. Alert if utilization drops below 50%.

4. **Diversity in targets**: Since all T+1 frames of a video are encoded, temporal variation naturally produces diverse codes.

---

## 9. File Structure

```
model/vid/fresh_hvqvae/
├── __init__.py
├── config.py          # Dataclass configs
├── encoder.py         # HierarchicalEncoder + ResBlocks
├── decoder.py         # Decoder + UpscaleMid + UpscaleTop
├── predictor.py       # PredictorStage + TransformerBlock
├── masks.py           # All mask construction functions
├── model.py           # FreshHVQVAE (ties everything together)
└── soft_lookup.py     # LFQ codebook matrix + soft lookup utils

example_code/
└── fresh_hvqvae_training_loop.py  # Training script
```

---

## 10. Config

```python
@dataclass
class FreshHVQVAEConfig:
    # Image
    image_size: int = 64
    
    # Encoder channels
    C_bot: int = 128
    C_mid: int = 192
    C_top: int = 256
    
    # Codebook sizes (must be powers of 2 for LFQ)
    K_bot: int = 4096   # 2^12
    K_mid: int = 1024   # 2^10
    K_top: int = 512    # 2^9
    
    # LFQ dims (= log2(K))
    lfq_dim_bot: int = 12
    lfq_dim_mid: int = 10
    lfq_dim_top: int = 9
    
    # LFQ entropy regularization
    entropy_loss_weight: float = 0.1
    diversity_gamma: float = 1.0
    
    # Predictor
    pred_n_heads: int = 8
    pred_n_layers: int = 4
    pred_dim_top: int = 256    # = C_top
    pred_dim_mid: int = 192    # = C_mid
    pred_dim_bot: int = 128    # = C_bot
    
    # Temporal windows
    window_top: int = -1       # -1 = full context (causal)
    window_mid: int = 2
    window_bot: int = 1
    
    # Cross-attention
    soft_lookup_temperature: float = 1.0
    
    # Reconstruction loss weights
    weight_mse_bot: float = 1.0
    weight_mse_mid: float = 0.5
    weight_mse_top: float = 0.1
    
    # Training
    lr_encoder_decoder: float = 3e-4
    lr_predictor_top: float = 3e-4
    lr_predictor_mid: float = 3e-4
    lr_predictor_bot: float = 1e-4
    max_grad_norm: float = 1.0
```

---

## 11. Model Forward Pass (Complete)

```python
class FreshHVQVAE(nn.Module):
    def __init__(self, cfg: FreshHVQVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = HierarchicalEncoder(cfg)
        self.decoder = Decoder(cfg.C_bot)
        self.upscale_mid = UpscaleMid(cfg.C_mid, cfg.C_bot)
        self.upscale_top = UpscaleTop(cfg.C_top, cfg.C_bot)
        
        self.predictor_top = PredictorStage(
            dim=cfg.pred_dim_top, n_heads=cfg.pred_n_heads, n_layers=cfg.pred_n_layers,
            codebook_size=cfg.K_top, spatial_size=1,
            temporal_window=cfg.window_top, has_parent=False,
            lfq_dim=cfg.lfq_dim_top,
        )
        self.predictor_mid = PredictorStage(
            dim=cfg.pred_dim_mid, n_heads=cfg.pred_n_heads, n_layers=cfg.pred_n_layers,
            codebook_size=cfg.K_mid, spatial_size=16,
            temporal_window=cfg.window_mid, has_parent=True,
            parent_dim=cfg.pred_dim_top, lfq_dim=cfg.lfq_dim_mid,
        )
        self.predictor_bot = PredictorStage(
            dim=cfg.pred_dim_bot, n_heads=cfg.pred_n_heads, n_layers=cfg.pred_n_layers,
            codebook_size=cfg.K_bot, spatial_size=256,
            temporal_window=cfg.window_bot, has_parent=True,
            parent_dim=cfg.pred_dim_mid, lfq_dim=cfg.lfq_dim_bot,
        )
        
        # Codebook matrices as registered buffers (move to GPU with model)
        self.register_buffer('codebook_top', build_lfq_codebook_matrix(cfg.lfq_dim_top))
        self.register_buffer('codebook_mid', build_lfq_codebook_matrix(cfg.lfq_dim_mid))
        self.register_buffer('codebook_bot', build_lfq_codebook_matrix(cfg.lfq_dim_bot))
        # Share references with predictor stages
        self.predictor_top.codebook_weights = self.codebook_top
        self.predictor_mid.codebook_weights = self.codebook_mid
        self.predictor_bot.codebook_weights = self.codebook_bot
        
    def encode(self, video):
        """Encode all frames. video: (B, T+1, 3, H, W)"""
        B, Tp1, _, H, W = video.shape
        flat = video.reshape(B * Tp1, 3, H, W)
        enc = self.encoder(flat)
        # Reshape to temporal: (B, T+1, C, H_s, W_s)
        for key in ['quant_bot', 'quant_mid', 'quant_top']:
            C = enc[key].shape[1]
            spatial = enc[key].shape[2:]
            enc[key] = enc[key].reshape(B, Tp1, C, *spatial)
        for key in ['idx_bot', 'idx_mid', 'idx_top']:
            spatial = enc[key].shape[1:]
            enc[key] = enc[key].reshape(B, Tp1, *spatial)
        return enc
    
    def reconstruct(self, enc, B, Tp1):
        """Reconstruct from all 3 stages (for per-stage MSE).
        Returns 3 reconstructions, all (B*(T+1), 3, 64, 64).
        """
        # Bot: direct to decoder
        quant_bot = enc['quant_bot'].reshape(B * Tp1, self.cfg.C_bot, 16, 16)
        recon_bot = self.decoder(quant_bot)
        
        # Mid: upscale then decode
        quant_mid = enc['quant_mid'].reshape(B * Tp1, self.cfg.C_mid, 4, 4)
        recon_mid = self.decoder(self.upscale_mid(quant_mid))
        
        # Top: upscale then decode
        quant_top = enc['quant_top'].reshape(B * Tp1, self.cfg.C_top, 1, 1)
        recon_top = self.decoder(self.upscale_top(quant_top))
        
        return recon_bot, recon_mid, recon_top
    
    def predict(self, enc, T):
        """Run top-down predictor on first T frames (encoder outputs DETACHED).
        All parent features are detached for stage isolation.
        """
        B = enc['quant_top'].shape[0]
        
        inp_top = enc['quant_top'][:, :T].detach().reshape(B, T * 1, -1)
        inp_mid = enc['quant_mid'][:, :T].detach().reshape(B, T * 16, -1)
        inp_bot = enc['quant_bot'][:, :T].detach().reshape(B, T * 256, -1)
        
        logits_top, feat_top = self.predictor_top(inp_top, T=T)
        logits_mid, feat_mid = self.predictor_mid(
            inp_mid, parent_features=feat_top.detach(), T=T
        )
        logits_bot, feat_bot = self.predictor_bot(
            inp_bot, parent_features=feat_mid.detach(), T=T
        )
        
        return {
            'logits_top': logits_top, 'logits_mid': logits_mid, 'logits_bot': logits_bot,
            'feat_top': feat_top, 'feat_mid': feat_mid, 'feat_bot': feat_bot,
        }
    
    def forward(self, video):
        """Full training forward: compute all losses.
        Returns dict of individual losses (not summed — caller handles backward).
        """
        B, Tp1 = video.shape[:2]
        T = Tp1 - 1
        all_frames = video.reshape(B * Tp1, 3, video.shape[3], video.shape[4])
        
        # Encode WITH gradient (for reconstruction)
        enc = self.encode(video)
        
        # Per-stage reconstruction
        recon_bot, recon_mid, recon_top = self.reconstruct(enc, B, Tp1)
        mse_bot = F.mse_loss(recon_bot, all_frames)
        mse_mid = F.mse_loss(recon_mid, all_frames)
        mse_top = F.mse_loss(recon_top, all_frames)
        vq_loss = enc['loss_bot'] + enc['loss_mid'] + enc['loss_top']
        
        # Prediction CE (detached from encoder)
        pred = self.predict(enc, T)
        tgt_top = enc['idx_top'][:, 1:].detach().reshape(B, T * 1)
        tgt_mid = enc['idx_mid'][:, 1:].detach().reshape(B, T * 16)
        tgt_bot = enc['idx_bot'][:, 1:].detach().reshape(B, T * 256)
        
        ce_top = F.cross_entropy(pred['logits_top'].reshape(-1, self.cfg.K_top), tgt_top.reshape(-1))
        ce_mid = F.cross_entropy(pred['logits_mid'].reshape(-1, self.cfg.K_mid), tgt_mid.reshape(-1))
        ce_bot = F.cross_entropy(pred['logits_bot'].reshape(-1, self.cfg.K_bot), tgt_bot.reshape(-1))
        
        return {
            'mse_bot': mse_bot, 'mse_mid': mse_mid, 'mse_top': mse_top,
            'vq_loss': vq_loss,
            'ce_top': ce_top, 'ce_mid': ce_mid, 'ce_bot': ce_bot,
        }
```

---

## 12. Expected Behavior & Sanity Checks

### 12.1 First Frame Prediction Should Be Fuzzy

With the temporal masking:
- **Bot (window=1)**: frame 0 only sees itself. It cannot know the rotation direction/speed. Its prediction for frame 1 should be uncertain (high entropy logits, spread distribution).
- **Mid (window=2)**: frame 0 only sees itself. Same uncertainty.
- **Top (window=T)**: frame 0 only sees itself. Same uncertainty.

**For frame 0 → predicting frame 1**: ALL stages only see frame 0. No temporal history. The model CANNOT know the motion. Predictions should be high-entropy / blurry when decoded.

**For frame 1 → predicting frame 2** (T≥3):
- Top sees [0,1] → can infer motion direction
- Mid sees [0,1] → some temporal info
- Bot sees [1] only → relies on parent cross-attention for temporal signal

This progressively sharpens predictions as more context accumulates. **This is the correct behavior.**

### 12.2 Sanity Check: Overfit Single Batch

- `mse_bot` → near 0 (encoder bot + decoder learn to reconstruct fine detail)
- `mse_mid` → small but nonzero (4×4 can't capture all fine detail)
- `mse_top` → stays relatively high (1×1 is very lossy — this is expected)
- `ce_*` → near 0 (predictor memorizes codes for this batch)
- First-frame CE should remain higher than later frames (inherent uncertainty)

### 12.3 Expected Loss Magnitudes

- `ce_top`: log2(512) = 9 bits max → starts ~9, should decrease to <3
- `ce_mid`: log2(1024) = 10 bits max → starts ~10, should decrease to <4
- `ce_bot`: log2(4096) = 12 bits max → starts ~12, should decrease to <6
- `mse_bot`: MSE on [0,1] range → starts ~0.1, should decrease to <0.01
- `mse_mid`: MSE on [0,1] range → starts ~0.15, should decrease to <0.03
- `mse_top`: MSE on [0,1] range → starts ~0.2, converges to ~0.05-0.10 (inherent information loss)

---

## 13. Implementation Order

| Phase | Task | Files | Dependency |
|---|---|---|---|
| 1 | Config dataclass | `config.py` | None |
| 2 | Encoder + ResBlocks | `encoder.py` | Config |
| 3 | Decoder + Upscalers | `decoder.py` | Config |
| 4 | Mask utilities | `masks.py` | None |
| 5 | Transformer block + Predictor stage | `predictor.py` | Masks |
| 6 | Soft lookup / LFQ utils | `soft_lookup.py` | None |
| 7 | Full model assembly | `model.py` | All above |
| 8 | Training loop | `fresh_hvqvae_training_loop.py` | Model |
| 9 | Single-batch overfit test | (in training loop) | All |
| 10 | Full training + eval | (in training loop) | All |

---

## 14. Dependencies

```
torch>=2.0
vector-quantize-pytorch>=1.14
einops>=0.7
torchvision  # for save_image
```

Install: `pip install vector-quantize-pytorch einops`

---

## 15. Key Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| VQ method | LFQ | MagViT-v2 proven for video; no codebook collapse; native video support |
| Gradient isolation | 4 optimizers (enc+dec+upscale, top, mid, bot) | No conflicting gradients; each component trained by one loss type |
| Reconstruction | Per-stage (bot/mid/top) with shared decoder | Forces each VQ stage to carry meaningful information; no chicken-and-egg |
| Upscalers | Shallow ConvTranspose (mid 4→16, top 1→16) | Reuses single decoder; cheap; produces unified 16×16 format |
| Predictor input | Encoder output DETACHED | Predictor never sends gradient to encoder — complete isolation |
| Predictor cross-attn parents | Always detached | Each stage trained by its own CE only; no cross-stage gradient bleed |
| Prediction target | Next-frame codes (shifted by 1) | Standard autoregressive prediction via CE |
| Temporal window | top=T, mid=2, bot=1 | Forces hierarchy: temporal reasoning at coarse scale |
| Spatial self-attn | Full (no windowing) | Keeps implementation simple; bot stage is only 256 tokens |
| Cross-attn spatial | 4×4 parent-child sectors | Natural hierarchical decomposition 1→4→16 |
| Soft lookup | softmax(logits) @ codebook → proj | Smooth cross-attn features; proj fixes dim mismatch (lfq_dim → C_stage) |
| Codebook matrices | register_buffer | Moves to GPU with model; not a parameter (no gradient) |
| Recon loss weights | 1.0 bot, 0.5 mid, 0.1 top | Compensates for gradient accumulation imbalance on bot encoder |
| No prediction-MSE | Discarded | Per-stage recon is sufficient; eliminates "graph-live frozen predictor" complexity |

---

## 16. Potential Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LFQ entropy loss dominates | Tune `entropy_loss_weight` (start 0.1, reduce if needed) |
| Bot predictor too large (T*256 tokens) | Use efficient attention (e.g., `F.scaled_dot_product_attention`), or reduce to 8×8 bot |
| Top predictor trivial (only T tokens) | Fine — it should be simple. The hierarchy provides expressiveness |
| Soft lookup temperature too hot/cold | Start at 1.0, anneal to 0.5 during training if needed |
| Bot encoder gets 3× gradient (from all MSE losses) | Use reconstruction weights: 1.0/0.5/0.1 to rebalance |
| `mse_top` never converges well (1×1 too lossy) | Expected — weight it down (0.1); provides coarse-structure signal only |
| VQ STE doesn't signal quantization error | LFQ entropy loss is the corrective; monitor codebook utilization |
| Upscaler too simple (introduces artifacts) | Add a ResBlock after ConvTranspose if needed; keep shallow initially |
| Per-stage CE magnitudes differ wildly | Normal (different codebook sizes); they're isolated so no interference |

### 16.1 Architectural Note: Non-Residual Hierarchy

The encoder uses pre-VQ features as input to the next stage (`enc_bot_to_mid(feat_bot)`, not `enc_bot_to_mid(quant_bot_feat)`). This means each higher stage encodes from the continuous representation, not the quantized one. The hierarchy is NOT "residual" (each stage does NOT encode what the previous stage missed). Instead, each stage independently encodes the same input at different spatial resolutions.

This is intentional:
- Preserves full gradient flow through the encoder chain
- Avoids the "residual VQ-VAE" complexity where later stages depend on earlier quantization quality
- Each stage has its own VQ + reconstruction loss to force it to capture useful information

---

## 17. Future Extensions (Not Implemented Now)

1. **Larger image sizes**: add a 4th stage (64×64 "finest") or use progressive growing
2. **Longer sequences**: increase T with gradient checkpointing
3. **EBT integration**: replace predictor forward pass with MCMC refinement in logit space
4. **Conditional generation**: add class/text conditioning to top predictor
5. **Multi-scale decoding at inference**: combine bot+mid+top predictions for richer output
6. **Residual hierarchy**: optionally encode quantization residuals at each stage
