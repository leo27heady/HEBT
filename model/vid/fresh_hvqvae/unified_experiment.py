"""
Unified End-to-End Hierarchical VQ-VAE Experiment.

Key differences from FreshHVQVAE:
- No detach between encoder stages (full end-to-end gradient flow)
- Supports both standard VQ and LFQ (selectable via config)
- No per-stage decoders; single decoder head from bot predictions → RGB
- Predictor and encoder trained jointly (single backward pass)
- Losses: VQ per stage + CE per stage + MSE on final RGB prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from .predictor import TransformerBlock
from .masks import (
    build_temporal_window_mask,
    build_cross_attn_mask_top_to_mid,
    build_cross_attn_mask_mid_to_bot,
)
from .soft_lookup import build_lfq_codebook_matrix


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class UnifiedConfig:
    # Spatial resolutions: bot=16x16, mid=4x4, top=1x1
    # Encoder channels
    C_bot: int = 64
    C_mid: int = 128
    C_top: int = 256

    # VQ type: 'standard' or 'lfq'
    vq_type: str = 'lfq'

    # VQ embedding dims (used for standard VQ)
    D_bot: int = 32
    D_mid: int = 64
    D_top: int = 128

    # LFQ dims (= log2(K), used when vq_type='lfq')
    lfq_dim_bot: int = 9
    lfq_dim_mid: int = 9
    lfq_dim_top: int = 9

    # Codebook sizes (for standard VQ; for LFQ this is 2^lfq_dim)
    K_bot: int = 512
    K_mid: int = 512
    K_top: int = 512

    # Standard VQ
    commitment_cost: float = 0.25

    # LFQ
    entropy_loss_weight: float = 0.1
    diversity_gamma: float = 1.0

    # Predictor
    pred_n_heads: int = 8
    pred_n_layers: int = 4
    pred_dim_top: int = 256
    pred_dim_mid: int = 128
    pred_dim_bot: int = 64

    # Temporal windows for predictor attention
    window_top: int = -1   # full causal
    window_mid: int = 2
    window_bot: int = 1

    # Soft lookup temperature
    temperature: float = 1.0

    # Sequence
    max_T: int = 16

    # Loss weights
    lambda_ce: float = 1.0
    lambda_vq: float = 1.0
    # MSE weight is implicitly 1.0

    # Training
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    def get_lfq_codebook_size(self, stage: str) -> int:
        """Get effective codebook size for a stage (2^lfq_dim)."""
        dim = getattr(self, f'lfq_dim_{stage}')
        return 2 ** dim


# ---------------------------------------------------------------------------
# VectorQuantizer (standard, like notebook)
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Standard VQ with commitment loss and straight-through estimator."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, D, H, W) — continuous latent features (channel-first)

        Returns:
            quantized: (B, D, H, W) — quantized with STE
            indices: (B, H, W) — codebook indices
            vq_loss: scalar — codebook + commitment loss
        """
        # z: (B, D, H, W) → (B, H, W, D) for distance computation
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
        flat_z = z_perm.reshape(-1, self.embedding_dim)  # (N, D)

        # L2 distances to codebook: (N, K)
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z@e^T
        dist = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * flat_z @ self.embedding.weight.T
        )

        # Nearest codebook entry
        indices = torch.argmin(dist, dim=-1)  # (N,)
        quantized_flat = self.embedding(indices)  # (N, D)

        # Reshape back
        quantized = quantized_flat.reshape(z_perm.shape)  # (B, H, W, D)

        # Losses
        codebook_loss = F.mse_loss(quantized, z_perm.detach())
        commitment_loss = F.mse_loss(z_perm, quantized.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = z_perm + (quantized - z_perm).detach()

        # Back to (B, D, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        indices = indices.reshape(z_perm.shape[0], z_perm.shape[1], z_perm.shape[2])  # (B, H, W)

        return quantized, indices, vq_loss


# ---------------------------------------------------------------------------
# LFQ Wrapper (consistent interface with VectorQuantizer)
# ---------------------------------------------------------------------------

class LFQWrapper(nn.Module):
    """Wraps vector_quantize_pytorch.LFQ with same interface as VectorQuantizer."""

    def __init__(self, codebook_size: int, dim: int, entropy_loss_weight: float = 0.1,
                 diversity_gamma: float = 1.0):
        super().__init__()
        from vector_quantize_pytorch import LFQ
        self.codebook_size = codebook_size
        self.dim = dim  # = log2(codebook_size)
        self.lfq = LFQ(
            codebook_size=codebook_size,
            dim=dim,
            entropy_loss_weight=entropy_loss_weight,
            diversity_gamma=diversity_gamma,
            channel_first=True,
        )
        # Pre-build the codebook matrix for soft-lookup
        self.register_buffer('codebook_weights', build_lfq_codebook_matrix(dim))

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, D, H, W) — continuous latent features (channel-first)

        Returns:
            quantized: (B, D, H, W) — quantized with STE
            indices: (B, H, W) — codebook indices
            loss: scalar — entropy loss
        """
        quantized, indices, loss = self.lfq(z)
        # LFQ indices shape: (B, H, W) already
        return quantized, indices, loss


# ---------------------------------------------------------------------------
# Encoder (bottom-up, NO detach between stages)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with GroupNorm."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride)
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class UnifiedEncoder(nn.Module):
    """
    Bottom-up encoder with VQ taps at 3 resolutions.
    NO DETACH between stages — full gradient flow.
    Supports both standard VQ and LFQ.

    Input: (B, 3, 64, 64)
    Outputs: quantized features + indices + losses at each stage.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg

        # Determine VQ dims based on type
        if cfg.vq_type == 'lfq':
            vq_dim_bot = cfg.lfq_dim_bot
            vq_dim_mid = cfg.lfq_dim_mid
            vq_dim_top = cfg.lfq_dim_top
        else:
            vq_dim_bot = cfg.D_bot
            vq_dim_mid = cfg.D_mid
            vq_dim_top = cfg.D_top

        # Stage 1: 64→16
        self.enc_to_bot = nn.Sequential(
            ResBlock(3, 64, stride=2),       # 64→32
            ResBlock(64, cfg.C_bot, stride=2),  # 32→16
        )
        self.bot_to_vq = nn.Conv2d(cfg.C_bot, vq_dim_bot, 1)
        self.bot_from_vq = nn.Conv2d(vq_dim_bot, cfg.C_bot, 1)

        # Stage 2: 16→4 (NO DETACH from bot)
        self.enc_bot_to_mid = nn.Sequential(
            ResBlock(cfg.C_bot, cfg.C_mid, stride=2),  # 16→8
            ResBlock(cfg.C_mid, cfg.C_mid, stride=2),  # 8→4
        )
        self.mid_to_vq = nn.Conv2d(cfg.C_mid, vq_dim_mid, 1)
        self.mid_from_vq = nn.Conv2d(vq_dim_mid, cfg.C_mid, 1)

        # Stage 3: 4→1 (NO DETACH from mid)
        self.enc_mid_to_top = nn.Sequential(
            ResBlock(cfg.C_mid, cfg.C_top, stride=2),  # 4→2
            ResBlock(cfg.C_top, cfg.C_top, stride=2),  # 2→1
        )
        self.top_to_vq = nn.Conv2d(cfg.C_top, vq_dim_top, 1)
        self.top_from_vq = nn.Conv2d(vq_dim_top, cfg.C_top, 1)

        # Create VQ modules
        if cfg.vq_type == 'lfq':
            self.vq_bot = LFQWrapper(cfg.get_lfq_codebook_size('bot'), vq_dim_bot,
                                     cfg.entropy_loss_weight, cfg.diversity_gamma)
            self.vq_mid = LFQWrapper(cfg.get_lfq_codebook_size('mid'), vq_dim_mid,
                                     cfg.entropy_loss_weight, cfg.diversity_gamma)
            self.vq_top = LFQWrapper(cfg.get_lfq_codebook_size('top'), vq_dim_top,
                                     cfg.entropy_loss_weight, cfg.diversity_gamma)
        else:
            self.vq_bot = VectorQuantizer(cfg.K_bot, vq_dim_bot, cfg.commitment_cost)
            self.vq_mid = VectorQuantizer(cfg.K_mid, vq_dim_mid, cfg.commitment_cost)
            self.vq_top = VectorQuantizer(cfg.K_top, vq_dim_top, cfg.commitment_cost)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, 3, 64, 64)
        Returns dict with quantized features, indices, and VQ losses.
        """
        # Bot stage
        feat_bot = self.enc_to_bot(x)                     # (B, C_bot, 16, 16)
        z_bot = self.bot_to_vq(feat_bot)                  # (B, D_bot, 16, 16)
        quant_bot, idx_bot, vq_loss_bot = self.vq_bot(z_bot)
        quant_bot_feat = self.bot_from_vq(quant_bot)      # (B, C_bot, 16, 16)

        # Mid stage — NO DETACH! Gradients flow through feat_bot → enc_bot_to_mid
        feat_mid = self.enc_bot_to_mid(feat_bot)          # (B, C_mid, 4, 4)
        z_mid = self.mid_to_vq(feat_mid)                  # (B, D_mid, 4, 4)
        quant_mid, idx_mid, vq_loss_mid = self.vq_mid(z_mid)
        quant_mid_feat = self.mid_from_vq(quant_mid)      # (B, C_mid, 4, 4)

        # Top stage — NO DETACH! Gradients flow through feat_mid → enc_mid_to_top
        feat_top = self.enc_mid_to_top(feat_mid)          # (B, C_top, 1, 1)
        z_top = self.top_to_vq(feat_top)                  # (B, D_top, 1, 1)
        quant_top, idx_top, vq_loss_top = self.vq_top(z_top)
        quant_top_feat = self.top_from_vq(quant_top)      # (B, C_top, 1, 1)

        return {
            'quant_bot': quant_bot_feat, 'idx_bot': idx_bot, 'vq_loss_bot': vq_loss_bot,
            'quant_mid': quant_mid_feat, 'idx_mid': idx_mid, 'vq_loss_mid': vq_loss_mid,
            'quant_top': quant_top_feat, 'idx_top': idx_top, 'vq_loss_top': vq_loss_top,
        }


# ---------------------------------------------------------------------------
# Decoder Head (bot-level features → RGB)
# ---------------------------------------------------------------------------

class DecoderHead(nn.Module):
    """Lightweight decoder: (B, C_bot, 16, 16) → (B, 3, 64, 64)."""

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.decode = nn.Sequential(
            ResBlock(in_channels, in_channels),
            nn.ConvTranspose2d(in_channels, 64, 4, stride=2, padding=1),  # 16→32
            nn.SiLU(),
            ResBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),           # 32→64
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x)


# ---------------------------------------------------------------------------
# Predictor Stage (adapted for standard VQ soft-lookup)
# ---------------------------------------------------------------------------

class UnifiedPredictorStage(nn.Module):
    """
    Predictor stage using standard VQ codebook for soft-lookup.
    Same transformer architecture as PredictorStage but with VQ embedding weights.
    """

    def __init__(self, dim: int, n_heads: int, n_layers: int, codebook_size: int,
                 embedding_dim: int, spatial_size: int, temporal_window: int,
                 has_parent: bool = False, parent_dim: int = None, max_T: int = 16):
        super().__init__()
        self.dim = dim
        self.spatial_size = spatial_size
        self.temporal_window = temporal_window
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        # Input projection
        self.input_proj = nn.Linear(dim, dim)

        # Positional encodings
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, spatial_size, dim) * 0.02)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, max_T, dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim=dim, n_heads=n_heads, has_cross_attn=has_parent, parent_dim=parent_dim)
            for _ in range(n_layers)
        ])

        # Output head → logits over codebook
        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, codebook_size),
        )

        # Soft-lookup: project from embedding_dim → dim
        self.soft_lookup_proj = nn.Linear(embedding_dim, dim)

    def _get_pos_encoding(self, T: int, S: int, device: torch.device) -> torch.Tensor:
        spatial = self.spatial_pos_embed[:, :S, :]
        temporal = self.temporal_pos_embed[:, :T, :]
        temporal_expanded = temporal.repeat_interleave(S, dim=1)
        spatial_expanded = spatial.repeat(1, T, 1)
        return temporal_expanded + spatial_expanded

    def soft_lookup(self, logits: torch.Tensor, codebook_weights: torch.Tensor,
                    temperature: float = 1.0) -> torch.Tensor:
        """
        Differentiable soft-lookup using VQ codebook weights.
        logits: (B, T*S, K)
        codebook_weights: (K, D_emb) — from vq.embedding.weight
        Returns: (B, T*S, dim) — projected features for child cross-attention
        """
        probs = F.softmax(logits / temperature, dim=-1)    # (B, T*S, K)
        soft_codes = probs @ codebook_weights              # (B, T*S, D_emb)
        return self.soft_lookup_proj(soft_codes)            # (B, T*S, dim)

    def forward(self, quant_input: torch.Tensor, codebook_weights: torch.Tensor,
                parent_features: torch.Tensor = None, T: int = None,
                temperature: float = 1.0):
        """
        Args:
            quant_input: (B, T*S, dim) — encoder quantized features as context
            codebook_weights: (K, D_emb) — VQ embedding weights for soft-lookup
            parent_features: (B, T*S_parent, parent_dim) or None
            T: number of frames
            temperature: softmax temperature

        Returns:
            logits: (B, T*S, K) — predicted code logits
            pred_features: (B, T*S, dim) — soft-decoded for child cross-attention
        """
        B = quant_input.shape[0]
        S = self.spatial_size
        device = quant_input.device

        # Input projection + positional encoding
        x = self.input_proj(quant_input)
        x = x + self._get_pos_encoding(T, S, device)

        # Self-attention mask (causal + temporal window)
        self_attn_mask = build_temporal_window_mask(T, S, self.temporal_window, device)

        # Cross-attention mask
        cross_attn_mask = None
        if parent_features is not None:
            if S == 16:
                cross_attn_mask = build_cross_attn_mask_top_to_mid(T, device)
            elif S == 256:
                cross_attn_mask = build_cross_attn_mask_mid_to_bot(T, device)

        # Transformer
        for layer in self.layers:
            x = layer(x, self_attn_mask=self_attn_mask,
                      cross_kv=parent_features, cross_attn_mask=cross_attn_mask)

        # Output logits
        logits = self.output_head(x)  # (B, T*S, K)

        # Soft-lookup for child cross-attention
        pred_features = self.soft_lookup(logits, codebook_weights, temperature)

        return logits, pred_features


# ---------------------------------------------------------------------------
# Full Unified Model
# ---------------------------------------------------------------------------

class UnifiedModel(nn.Module):
    """
    End-to-end hierarchical VQ-VAE with joint encoder-predictor training.

    Forward pass:
        1. Encode all T+1 frames → quant features + indices at 3 levels
        2. Predict frames 1..T from context 0..T-1 (top-down)
        3. Decode bot-level predictions to RGB
        4. Compute all losses (VQ + CE + MSE) in single backward
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder
        self.encoder = UnifiedEncoder(cfg)

        # Decoder head (bot features → RGB)
        self.decoder_head = DecoderHead(cfg.C_bot)

        # Determine codebook sizes and embedding dims for predictors
        if cfg.vq_type == 'lfq':
            K_top = cfg.get_lfq_codebook_size('top')
            K_mid = cfg.get_lfq_codebook_size('mid')
            K_bot = cfg.get_lfq_codebook_size('bot')
            emb_dim_top = cfg.lfq_dim_top
            emb_dim_mid = cfg.lfq_dim_mid
            emb_dim_bot = cfg.lfq_dim_bot
        else:
            K_top, K_mid, K_bot = cfg.K_top, cfg.K_mid, cfg.K_bot
            emb_dim_top = cfg.D_top
            emb_dim_mid = cfg.D_mid
            emb_dim_bot = cfg.D_bot

        # Predictors (top-down)
        self.predictor_top = UnifiedPredictorStage(
            dim=cfg.pred_dim_top, n_heads=cfg.pred_n_heads, n_layers=cfg.pred_n_layers,
            codebook_size=K_top, embedding_dim=emb_dim_top,
            spatial_size=1, temporal_window=cfg.window_top,
            has_parent=False, max_T=cfg.max_T,
        )
        self.predictor_mid = UnifiedPredictorStage(
            dim=cfg.pred_dim_mid, n_heads=cfg.pred_n_heads, n_layers=cfg.pred_n_layers,
            codebook_size=K_mid, embedding_dim=emb_dim_mid,
            spatial_size=16, temporal_window=cfg.window_mid,
            has_parent=True, parent_dim=cfg.pred_dim_top, max_T=cfg.max_T,
        )
        self.predictor_bot = UnifiedPredictorStage(
            dim=cfg.pred_dim_bot, n_heads=cfg.pred_n_heads, n_layers=cfg.pred_n_layers,
            codebook_size=K_bot, embedding_dim=emb_dim_bot,
            spatial_size=256, temporal_window=cfg.window_bot,
            has_parent=True, parent_dim=cfg.pred_dim_mid, max_T=cfg.max_T,
        )

    def _get_codebook_weights(self, vq_module) -> torch.Tensor:
        """Get codebook weight matrix from either VectorQuantizer or LFQWrapper."""
        if isinstance(vq_module, LFQWrapper):
            return vq_module.codebook_weights  # (K, lfq_dim) — fixed binary codes
        else:
            return vq_module.embedding.weight  # (K, D_emb) — learned embeddings

    def _get_vq_dim(self, stage: str) -> int:
        """Get the VQ embedding dim for a stage."""
        if self.cfg.vq_type == 'lfq':
            return getattr(self.cfg, f'lfq_dim_{stage}')
        else:
            return getattr(self.cfg, f'D_{stage}')

    def encode(self, video: torch.Tensor) -> dict:
        """
        Encode video frames.
        video: (B, T+1, 3, 64, 64)
        Returns dict with temporally-reshaped quant features and indices.
        """
        B, Tp1 = video.shape[:2]
        flat = video.reshape(B * Tp1, 3, 64, 64)
        enc = self.encoder(flat)

        # Reshape to temporal: (B, T+1, ...)
        # quant_bot: (B*Tp1, C_bot, 16, 16) → (B, Tp1, C_bot, 16, 16)
        enc['quant_bot'] = enc['quant_bot'].reshape(B, Tp1, self.cfg.C_bot, 16, 16)
        enc['quant_mid'] = enc['quant_mid'].reshape(B, Tp1, self.cfg.C_mid, 4, 4)
        enc['quant_top'] = enc['quant_top'].reshape(B, Tp1, self.cfg.C_top, 1, 1)

        # idx: (B*Tp1, H, W) → (B, Tp1, H, W)
        enc['idx_bot'] = enc['idx_bot'].reshape(B, Tp1, 16, 16)
        enc['idx_mid'] = enc['idx_mid'].reshape(B, Tp1, 4, 4)
        enc['idx_top'] = enc['idx_top'].reshape(B, Tp1, 1, 1)

        return enc

    def predict(self, enc: dict, T: int) -> dict:
        """
        Predict frames 1..T from context frames 0..T-1.
        Context = encoder's quantized features (NO detach — gradients flow back to encoder).
        """
        B = enc['quant_top'].shape[0]
        cfg = self.cfg

        # Flatten spatial dims for transformer input: (B, T, C, H, W) → (B, T*S, C)
        # Use frames 0..T-1 as context
        ctx_top = enc['quant_top'][:, :T].reshape(B, T * 1, -1)    # (B, T*1, C_top)
        ctx_mid = enc['quant_mid'][:, :T].reshape(B, T * 16, -1)   # (B, T*16, C_mid)
        ctx_bot = enc['quant_bot'][:, :T].reshape(B, T * 256, -1)  # (B, T*256, C_bot)

        # Top predictor (no parent)
        logits_top, feat_top = self.predictor_top(
            ctx_top, codebook_weights=self._get_codebook_weights(self.encoder.vq_top),
            T=T, temperature=cfg.temperature,
        )

        # Mid predictor (parent = top predictions)
        logits_mid, feat_mid = self.predictor_mid(
            ctx_mid, codebook_weights=self._get_codebook_weights(self.encoder.vq_mid),
            parent_features=feat_top, T=T, temperature=cfg.temperature,
        )

        # Bot predictor (parent = mid predictions)
        logits_bot, feat_bot = self.predictor_bot(
            ctx_bot, codebook_weights=self._get_codebook_weights(self.encoder.vq_bot),
            parent_features=feat_mid, T=T, temperature=cfg.temperature,
        )

        return {
            'logits_top': logits_top, 'logits_mid': logits_mid, 'logits_bot': logits_bot,
            'feat_top': feat_top, 'feat_mid': feat_mid, 'feat_bot': feat_bot,
        }

    def decode_prediction(self, pred: dict, T: int) -> torch.Tensor:
        """
        Decode bot-level predicted features to RGB.
        Uses soft-lookup of logits_bot → continuous features → decoder head.

        Returns: (B, T, 3, 64, 64) — predicted frames in [-1, 1]
        """
        B = pred['logits_bot'].shape[0]
        cfg = self.cfg

        # Soft-lookup: logits → continuous bot features
        logits_bot = pred['logits_bot']  # (B, T*256, K_bot)
        probs = F.softmax(logits_bot / cfg.temperature, dim=-1)   # (B, T*256, K_bot)
        # Lookup in codebook: (B, T*256, D_bot)
        codebook_weights = self._get_codebook_weights(self.encoder.vq_bot)
        soft_codes = probs @ codebook_weights  # (B, T*256, vq_dim_bot)

        # Reshape to spatial and project to channel space
        vq_dim_bot = self._get_vq_dim('bot')
        soft_codes_spatial = soft_codes.reshape(B * T, 16, 16, vq_dim_bot)
        soft_codes_spatial = soft_codes_spatial.permute(0, 3, 1, 2)  # (B*T, vq_dim_bot, 16, 16)

        # Project to channel space and decode
        features = self.encoder.bot_from_vq(soft_codes_spatial)  # (B*T, C_bot, 16, 16)
        rgb = self.decoder_head(features)                        # (B*T, 3, 64, 64)

        return rgb.reshape(B, T, 3, 64, 64)

    def forward(self, video: torch.Tensor) -> dict:
        """
        Full forward pass with all losses.
        video: (B, T+1, 3, 64, 64) in [-1, 1]

        Returns dict with individual losses and total loss.
        """
        B, Tp1 = video.shape[:2]
        T = Tp1 - 1
        cfg = self.cfg

        # 1. Encode all frames
        enc = self.encode(video)

        # 2. Predict frames 1..T
        pred = self.predict(enc, T)

        # 3. Decode bot predictions to RGB
        pred_rgb = self.decode_prediction(pred, T)  # (B, T, 3, 64, 64)

        # 4. Compute losses
        # VQ losses (already computed during encoding, summed over all frames)
        vq_loss = enc['vq_loss_bot'] + enc['vq_loss_mid'] + enc['vq_loss_top']

        # CE losses (predicted logits vs encoder indices for frames 1..T)
        # Target indices for frames 1..T
        tgt_top = enc['idx_top'][:, 1:].reshape(B * T * 1)       # (B*T*1,)
        tgt_mid = enc['idx_mid'][:, 1:].reshape(B * T * 16)      # (B*T*16,)
        tgt_bot = enc['idx_bot'][:, 1:].reshape(B * T * 256)     # (B*T*256,)

        K_top = pred['logits_top'].shape[-1]
        K_mid = pred['logits_mid'].shape[-1]
        K_bot = pred['logits_bot'].shape[-1]

        ce_top = F.cross_entropy(pred['logits_top'].reshape(-1, K_top), tgt_top)
        ce_mid = F.cross_entropy(pred['logits_mid'].reshape(-1, K_mid), tgt_mid)
        ce_bot = F.cross_entropy(pred['logits_bot'].reshape(-1, K_bot), tgt_bot)
        ce_loss = ce_top + ce_mid + ce_bot

        # MSE loss (predicted RGB vs ground truth frames 1..T)
        target_rgb = video[:, 1:]  # (B, T, 3, 64, 64)
        mse_loss = F.mse_loss(pred_rgb, target_rgb)

        # Total loss
        total_loss = mse_loss + cfg.lambda_ce * ce_loss + cfg.lambda_vq * vq_loss

        return {
            'loss': total_loss,
            'mse': mse_loss,
            'ce_top': ce_top, 'ce_mid': ce_mid, 'ce_bot': ce_bot,
            'ce_total': ce_loss,
            'vq_bot': enc['vq_loss_bot'], 'vq_mid': enc['vq_loss_mid'], 'vq_top': enc['vq_loss_top'],
            'vq_total': vq_loss,
        }

    @torch.no_grad()
    def build_visualization(self, video: torch.Tensor) -> torch.Tensor:
        """
        Build visualization grid.
        Rows: GT, Predicted RGB, Enc Bot Recon (via decoder_head)
        """
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        B, Tp1 = video.shape[:2]
        T = Tp1 - 1
        H, W = 64, 64
        device = video.device

        enc = self.encode(video)
        pred = self.predict(enc, T)
        pred_rgb = self.decode_prediction(pred, T)  # (B, T, 3, 64, 64)

        # Encoder reconstruction via decoder_head (all frames)
        quant_bot_all = enc['quant_bot'][0]  # (T+1, C_bot, 16, 16)
        enc_recon = self.decoder_head(quant_bot_all)  # (T+1, 3, 64, 64)

        # Ground truth
        gt = video[0]  # (T+1, 3, 64, 64)

        # Predicted (blank first frame + T predicted)
        blank = torch.full((1, 3, H, W), -1.0, device=device)
        pred_row = torch.cat([blank, pred_rgb[0]], dim=0)  # (T+1, 3, 64, 64)

        row_data = [gt, enc_recon, pred_row]
        row_labels = ['GT', 'Enc Recon', 'Predicted']

        # Assemble grid
        n_rows = len(row_data)
        n_cols = Tp1
        row_images = []
        for frames in row_data:
            row_img = torch.cat([frames[t] for t in range(Tp1)], dim=2)
            row_images.append(row_img)
        raw_grid = torch.cat(row_images, dim=1)
        raw_grid = (raw_grid + 1) / 2  # [-1,1] → [0,1]
        raw_grid = raw_grid.clamp(0, 1)

        # Add labels
        grid_np = (raw_grid.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        pil_img = Image.fromarray(grid_np)

        label_left_w = 80
        label_top_h = 18
        canvas = Image.new('RGB', (label_left_w + n_cols * W, label_top_h + n_rows * H), (0, 0, 0))
        canvas.paste(pil_img, (label_left_w, label_top_h))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for c in range(n_cols):
            x = label_left_w + c * W + W // 2 - 5
            draw.text((x, 2), str(c), fill=(255, 255, 255), font=font)
        for r, label in enumerate(row_labels):
            y = label_top_h + r * H + H // 2 - 6
            draw.text((3, y), label, fill=(255, 255, 255), font=font)

        result = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
        return result
