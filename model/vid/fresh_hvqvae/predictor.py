"""Transformer-based predictor stages for Fresh HVQVAE."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import build_temporal_window_mask, build_cross_attn_mask_top_to_mid, build_cross_attn_mask_mid_to_bot


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and optional cross-attention."""

    def __init__(self, dim: int, n_heads: int, has_cross_attn: bool = False, parent_dim: int = None):
        super().__init__()
        self.has_cross_attn = has_cross_attn

        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

        # Cross-attention (optional)
        if has_cross_attn:
            self.norm_cross = nn.LayerNorm(dim)
            # Project parent_dim → dim for KV if dimensions differ
            kv_dim = parent_dim if parent_dim is not None else dim
            self.cross_attn_q = nn.Linear(dim, dim)
            self.cross_attn_k = nn.Linear(kv_dim, dim)
            self.cross_attn_v = nn.Linear(kv_dim, dim)
            self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

        # FFN
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, self_attn_mask: torch.Tensor = None,
                cross_kv: torch.Tensor = None, cross_attn_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        residual = x
        x_norm = self.norm1(x)
        # PyTorch MHA uses additive mask or key_padding_mask.
        # For boolean mask: True = IGNORE. Our mask convention: True = ATTEND.
        # Convert: ~mask (invert)
        if self_attn_mask is not None:
            # (seq, seq) → need to be (seq, seq) with True=ignore
            attn_mask = ~self_attn_mask  # True = blocked
        else:
            attn_mask = None
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = residual + x_attn

        # Cross-attention
        if self.has_cross_attn and cross_kv is not None:
            residual = x
            x_norm = self.norm_cross(x)
            q = self.cross_attn_q(x_norm)
            k = self.cross_attn_k(cross_kv)
            v = self.cross_attn_v(cross_kv)
            if cross_attn_mask is not None:
                cross_mask = ~cross_attn_mask  # invert for PyTorch convention
            else:
                cross_mask = None
            x_cross, _ = self.cross_attn(q, k, v, attn_mask=cross_mask)
            x = residual + x_cross

        # FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x


class PredictorStage(nn.Module):
    """
    Single predictor stage in the top-down hierarchy.
    Self-attention with temporal windowing + optional cross-attention from parent.
    """

    def __init__(self, dim: int, n_heads: int, n_layers: int, codebook_size: int,
                 spatial_size: int, temporal_window: int, has_parent: bool = False,
                 parent_dim: int = None, lfq_dim: int = None, max_T: int = 16):
        super().__init__()
        self.dim = dim
        self.spatial_size = spatial_size
        self.temporal_window = temporal_window
        self.codebook_size = codebook_size
        self.lfq_dim = lfq_dim

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

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, codebook_size),
        )

        # Soft-lookup projection: lfq_dim → dim
        if lfq_dim is not None:
            self.soft_lookup_proj = nn.Linear(lfq_dim, dim)

        # Codebook weights buffer (set externally)
        self.codebook_weights = None

    def _get_pos_encoding(self, T: int, S: int, device: torch.device) -> torch.Tensor:
        """Combine spatial and temporal positional encodings."""
        # spatial: (1, S, dim) → (1, T*S, dim) by repeating T times
        # temporal: (1, T, dim) → (1, T*S, dim) by repeating S times per frame
        spatial = self.spatial_pos_embed[:, :S, :]  # (1, S, dim)
        temporal = self.temporal_pos_embed[:, :T, :]  # (1, T, dim)

        # Expand: each frame t has S spatial positions
        temporal_expanded = temporal.repeat_interleave(S, dim=1)  # (1, T*S, dim)
        spatial_expanded = spatial.repeat(1, T, 1)  # (1, T*S, dim)

        return temporal_expanded + spatial_expanded  # (1, T*S, dim)

    def _soft_lookup(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Convert logits to continuous feature vectors via soft codebook lookup + projection.
        logits: (B, T*S, codebook_size)
        Returns: (B, T*S, dim)
        """
        probs = F.softmax(logits / temperature, dim=-1)  # (B, T*S, K)
        # codebook_weights: (K, lfq_dim)
        soft_codes = probs @ self.codebook_weights  # (B, T*S, lfq_dim)
        # Project to dim for cross-attention compatibility
        pred_features = self.soft_lookup_proj(soft_codes)  # (B, T*S, dim)
        return pred_features

    def forward(self, quant_input: torch.Tensor, parent_features: torch.Tensor = None,
                T: int = None, temperature: float = 1.0):
        """
        Args:
            quant_input: (B, T*S, dim) — quantized features flattened
            parent_features: (B, T*S_parent, parent_dim) or None
            T: number of frames

        Returns:
            logits: (B, T*S, codebook_size)
            pred_features: (B, T*S, dim) — soft-decoded for child cross-attn
        """
        B = quant_input.shape[0]
        S = self.spatial_size
        device = quant_input.device

        # Input projection + positional encoding
        x = self.input_proj(quant_input)
        x = x + self._get_pos_encoding(T, S, device)

        # Build self-attention mask
        self_attn_mask = build_temporal_window_mask(T, S, self.temporal_window, device)

        # Build cross-attention mask
        cross_attn_mask = None
        if parent_features is not None:
            if S == 16:  # mid stage, parent is top (S_parent=1)
                cross_attn_mask = build_cross_attn_mask_top_to_mid(T, device)
            elif S == 256:  # bot stage, parent is mid (S_parent=16)
                cross_attn_mask = build_cross_attn_mask_mid_to_bot(T, device)

        # Transformer forward
        for layer in self.layers:
            x = layer(x, self_attn_mask=self_attn_mask,
                      cross_kv=parent_features, cross_attn_mask=cross_attn_mask)

        # Output logits
        logits = self.output_head(x)  # (B, T*S, codebook_size)

        # Soft-lookup for child stage cross-attention
        pred_features = self._soft_lookup(logits, temperature=temperature)

        return logits, pred_features
