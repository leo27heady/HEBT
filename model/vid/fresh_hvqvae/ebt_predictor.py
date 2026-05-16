"""EBT-MCMC Predictor stages for Fresh HVQVAE.

Energy-Based Training predictor that uses iterative MCMC refinement
of codebook logits instead of one-shot transformer prediction.
Same interface as PredictorStage: forward(quant_input, parent_features, T) → logits, pred_features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import build_ebt_self_attn_mask, build_ebt_cross_attn_mask_top_to_mid, build_ebt_cross_attn_mask_mid_to_bot
from .soft_lookup import build_lfq_codebook_matrix


class EBTTransformerBlock(nn.Module):
    """Transformer block for energy computation with self-attention and optional cross-attention."""

    def __init__(self, dim: int, n_heads: int, has_cross_attn: bool = False, parent_dim: int = None):
        super().__init__()
        self.has_cross_attn = has_cross_attn

        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

        # Cross-attention (optional)
        if has_cross_attn:
            self.norm_cross = nn.LayerNorm(dim)
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
        if self_attn_mask is not None:
            attn_mask = ~self_attn_mask  # True = blocked (PyTorch convention)
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
                cross_mask = ~cross_attn_mask
            else:
                cross_mask = None
            x_cross, _ = self.cross_attn(q, k, v, attn_mask=cross_mask)
            x = residual + x_cross

        # FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x


class EBTPredictorStage(nn.Module):
    """
    EBT-MCMC predictor stage. Uses iterative energy-based refinement
    instead of one-shot transformer prediction.

    Same interface as PredictorStage:
        forward(quant_input, parent_features, T) → logits, pred_features
    """

    def __init__(self, dim: int, n_heads: int, n_layers: int, codebook_size: int,
                 spatial_size: int, temporal_window: int, has_parent: bool = False,
                 parent_dim: int = None, lfq_dim: int = None, max_T: int = 16,
                 mcmc_num_steps: int = 5, mcmc_step_size: float = 0.1,
                 langevin_noise: float = 0.01, truncate_mcmc: bool = True,
                 clamp_grad_max: float = 10.0, mcmc_step_size_learnable: bool = True,
                 initial_condition: str = 'zeros'):
        super().__init__()
        self.dim = dim
        self.spatial_size = spatial_size
        self.temporal_window = temporal_window
        self.codebook_size = codebook_size
        self.lfq_dim = lfq_dim
        self.mcmc_num_steps = mcmc_num_steps
        self.langevin_noise = langevin_noise
        self.truncate_mcmc = truncate_mcmc
        self.clamp_grad_max = clamp_grad_max
        self.initial_condition = initial_condition

        # Learnable MCMC step size
        self.alpha = nn.Parameter(
            torch.tensor(mcmc_step_size),
            requires_grad=mcmc_step_size_learnable
        )

        # Input projection for real context
        self.input_proj = nn.Linear(dim, dim)

        # Positional encodings
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, spatial_size, dim) * 0.02)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, max_T, dim) * 0.02)

        # Soft-lookup: codebook buffer + projection (for converting logits → features)
        self.soft_lookup_proj = nn.Linear(lfq_dim, dim)
        self.register_buffer('codebook_weights', build_lfq_codebook_matrix(lfq_dim))

        # Energy transformer (separate from vanilla predictor)
        self.energy_layers = nn.ModuleList([
            EBTTransformerBlock(dim=dim, n_heads=n_heads, has_cross_attn=has_parent, parent_dim=parent_dim)
            for _ in range(n_layers)
        ])
        self.energy_norm = nn.LayerNorm(dim)
        self.energy_head = nn.Linear(dim, 1, bias=False)  # scalar energy per position

    def _get_pos_encoding(self, T: int, S: int, device: torch.device) -> torch.Tensor:
        """Combine spatial and temporal positional encodings."""
        spatial = self.spatial_pos_embed[:, :S, :]  # (1, S, dim)
        temporal = self.temporal_pos_embed[:, :T, :]  # (1, T, dim)
        temporal_expanded = temporal.repeat_interleave(S, dim=1)  # (1, T*S, dim)
        spatial_expanded = spatial.repeat(1, T, 1)  # (1, T*S, dim)
        return temporal_expanded + spatial_expanded  # (1, T*S, dim)

    def _soft_lookup(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Convert logits to continuous feature vectors via soft codebook lookup."""
        probs = F.softmax(logits / temperature, dim=-1)  # (B, T*S, K)
        soft_codes = probs @ self.codebook_weights  # (B, T*S, lfq_dim)
        pred_features = self.soft_lookup_proj(soft_codes)  # (B, T*S, dim)
        return pred_features

    def _compute_energy(self, combined: torch.Tensor, parent_features: torch.Tensor,
                        self_attn_mask: torch.Tensor, cross_attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute per-position energy on the predicted half of combined context.

        Args:
            combined: (B, 2*T*S, dim) — [real_context | pred_features]
            parent_features: (B, T*S_parent, parent_dim) or None
            self_attn_mask: (2*T*S, 2*T*S) boolean mask
            cross_attn_mask: (2*T*S, T*S_parent) boolean mask or None

        Returns:
            energy: (B, T*S, 1) — scalar energy per predicted position
        """
        seq_len = combined.shape[1] // 2  # T*S

        x = combined
        for layer in self.energy_layers:
            x = layer(x, self_attn_mask=self_attn_mask,
                      cross_kv=parent_features, cross_attn_mask=cross_attn_mask)

        # Extract predicted half only
        pred_half = x[:, seq_len:]  # (B, T*S, dim)
        pred_half = self.energy_norm(pred_half)
        energy = self.energy_head(pred_half)  # (B, T*S, 1)
        return energy

    def forward(self, quant_input: torch.Tensor, parent_features: torch.Tensor = None,
                T: int = None, temperature: float = 1.0):
        """
        Args:
            quant_input: (B, T*S, dim) — quantized encoder features (frozen/detached)
            parent_features: (B, T*S_parent, parent_dim) or None
            T: number of frames

        Returns:
            logits: (B, T*S, codebook_size) — refined logits
            pred_features: (B, T*S, dim) — soft-decoded for child cross-attn
        """
        B = quant_input.shape[0]
        S = self.spatial_size
        K = self.codebook_size
        device = quant_input.device

        # Real context with positional encoding
        pos_enc = self._get_pos_encoding(T, S, device)
        real_context = self.input_proj(quant_input) + pos_enc  # (B, T*S, dim)

        # Build self-attention mask for combined [real | predicted] context
        self_attn_mask = build_ebt_self_attn_mask(T, S, self.temporal_window, device)

        # Build cross-attention mask (for child stages with parent)
        # Must be (2*T*S, T*S_parent): both halves get same-frame alignment
        # (real half doesn't strictly need it, but blocking causes NaN from empty softmax)
        cross_attn_mask = None
        if parent_features is not None:
            if S == 16:  # mid stage, parent is top (S_parent=1)
                base_cross_mask = build_ebt_cross_attn_mask_top_to_mid(T, device)
            elif S == 256:  # bot stage, parent is mid (S_parent=16)
                base_cross_mask = build_ebt_cross_attn_mask_mid_to_bot(T, device)
            else:
                base_cross_mask = None

            if base_cross_mask is not None:
                # Same mask for both halves: (T*S, S_parent) → (2*T*S, S_parent)
                cross_attn_mask = torch.cat([base_cross_mask, base_cross_mask], dim=0)

        # Initialize predicted logits (the MCMC variable)
        if self.initial_condition == 'zeros':
            logits = torch.zeros(B, T * S, K, device=device)
        else:  # random_noise
            logits = torch.randn(B, T * S, K, device=device) * 0.1

        # Store energies for logging
        energies = []
        energies_tensor = None

        # MCMC refinement loop
        alpha = torch.clamp(self.alpha, min=1e-4)

        for step in range(self.mcmc_num_steps):
            logits = logits.detach().requires_grad_()

            # Langevin noise
            if self.training and self.langevin_noise > 0:
                logits = logits + self.langevin_noise * torch.randn_like(logits)

            # Soft lookup: logits → features
            pred_features = self._soft_lookup(logits, temperature)  # (B, T*S, dim)
            # Add positional encoding to predicted features
            pred_features = pred_features + pos_enc

            # Combined context: [real | predicted]
            combined = torch.cat([real_context, pred_features], dim=1)  # (B, 2*T*S, dim)

            # Compute energy
            energy = self._compute_energy(combined, parent_features,
                                          self_attn_mask, cross_attn_mask)  # (B, T*S, 1)
            energies.append(energy.detach().mean().item())
            energies_tensor = energy.detach()  # keep last step's full tensor

            # MCMC gradient
            if self.truncate_mcmc:
                create_graph = (step == self.mcmc_num_steps - 1)
            else:
                create_graph = True

            grad = torch.autograd.grad(energy.sum(), logits, create_graph=create_graph)[0]

            # Gradient clamping
            grad = torch.clamp(grad, -self.clamp_grad_max, self.clamp_grad_max)

            # Update logits (gradient descent on energy)
            logits = logits - alpha * grad

        # Final soft-lookup for child cross-attention
        pred_features = self._soft_lookup(logits, temperature)

        # Store last energy map for visualization (B, T*S, 1)
        self._last_energy = energies_tensor if energies_tensor is not None else None

        return logits, pred_features
