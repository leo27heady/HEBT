"""EBT-MCMC predictor stages for LFQ-VQVAE video mode."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import (
    build_cross_attn_mask_mid_to_bot,
    build_cross_attn_mask_top_to_mid,
    build_ebt_self_attn_mask,
)
from .soft_lookup import build_lfq_codebook_matrix


class EBTTransformerBlock(nn.Module):
    """Transformer block for EBT energy prediction."""

    def __init__(self, dim: int, n_heads: int, has_cross_attn: bool = False, parent_dim: int | None = None):
        super().__init__()
        self.has_cross_attn = has_cross_attn

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

        if has_cross_attn:
            self.norm_cross = nn.LayerNorm(dim)
            kv_dim = parent_dim if parent_dim is not None else dim
            self.cross_attn_q = nn.Linear(dim, dim)
            self.cross_attn_k = nn.Linear(kv_dim, dim)
            self.cross_attn_v = nn.Linear(kv_dim, dim)
            self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_kv: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_mask = ~self_attn_mask if self_attn_mask is not None else None
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = residual + x_attn

        if self.has_cross_attn and cross_kv is not None:
            residual = x
            x_norm = self.norm_cross(x)
            q = self.cross_attn_q(x_norm)
            k = self.cross_attn_k(cross_kv)
            v = self.cross_attn_v(cross_kv)
            cross_mask = ~cross_attn_mask if cross_attn_mask is not None else None
            x_cross, _ = self.cross_attn(q, k, v, attn_mask=cross_mask)
            x = residual + x_cross

        return x + self.ffn(self.norm2(x))


class EBTPredictorStage(nn.Module):
    """
    Energy-based stage predictor with MCMC refinement in codebook-logit space.

    API matches PredictorStage:
      forward(quant_input, parent_features, T, temperature, gumbel_tau)
      -> (logits, pred_features)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_layers: int,
        codebook_size: int,
        spatial_size: int,
        temporal_window: int,
        has_parent: bool = False,
        parent_dim: int | None = None,
        lfq_dim: int | None = None,
        max_T: int = 16,
        mcmc_num_steps: int = 5,
        mcmc_step_size: float = 0.1,
        initial_condition: str = "random",
    ) -> None:
        super().__init__()
        if lfq_dim is None:
            raise ValueError("lfq_dim is required for LFQ soft lookup")
        if initial_condition not in ("random", "zero"):
            raise ValueError(f"Unsupported initial_condition: {initial_condition}")

        self.dim = dim
        self.spatial_size = spatial_size
        self.temporal_window = temporal_window
        self.codebook_size = codebook_size
        self.max_T = max_T
        self.mcmc_num_steps = mcmc_num_steps
        self.initial_condition = initial_condition

        self.alpha = nn.Parameter(torch.tensor(mcmc_step_size), requires_grad=True)

        self.input_proj = nn.Linear(dim, dim)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, spatial_size, dim) * 0.02)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, max_T, dim) * 0.02)

        self.energy_layers = nn.ModuleList(
            [
                EBTTransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    has_cross_attn=has_parent,
                    parent_dim=parent_dim,
                )
                for _ in range(n_layers)
            ]
        )
        self.energy_norm = nn.LayerNorm(dim)
        self.energy_head = nn.Linear(dim, 1, bias=False)

        self.soft_lookup_proj = nn.Linear(lfq_dim, dim)
        self.register_buffer("codebook_weights", build_lfq_codebook_matrix(lfq_dim))

        self._last_energy: torch.Tensor | None = None

    def _get_pos_encoding(self, T: int, S: int) -> torch.Tensor:
        if T > self.max_T:
            raise ValueError(f"T={T} exceeds max_T={self.max_T}")
        spatial = self.spatial_pos_embed[:, :S, :]
        temporal = self.temporal_pos_embed[:, :T, :]
        temporal_expanded = temporal.repeat_interleave(S, dim=1)
        spatial_expanded = spatial.repeat(1, T, 1)
        return temporal_expanded + spatial_expanded

    def _soft_lookup(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        soft_codes = probs @ self.codebook_weights
        return self.soft_lookup_proj(soft_codes)

    def _build_cross_mask(self, T: int, S: int, device: torch.device) -> torch.Tensor | None:
        if S == 16:
            base = build_cross_attn_mask_top_to_mid(T, device)
        elif S == 256:
            base = build_cross_attn_mask_mid_to_bot(T, device)
        else:
            return None
        return torch.cat([base, base], dim=0)

    def _compute_energy(
        self,
        combined: torch.Tensor,
        parent_features: torch.Tensor | None,
        self_attn_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        seq_len = combined.shape[1] // 2
        x = combined
        for layer in self.energy_layers:
            x = layer(
                x,
                self_attn_mask=self_attn_mask,
                cross_kv=parent_features,
                cross_attn_mask=cross_attn_mask,
            )
        pred_half = self.energy_norm(x[:, seq_len:])
        return self.energy_head(pred_half)

    def forward(
        self,
        quant_input: torch.Tensor,
        parent_features: torch.Tensor | None = None,
        T: int | None = None,
        temperature: float = 1.0,
        gumbel_tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del gumbel_tau  # kept for API compatibility with PredictorStage
        if T is None:
            raise ValueError("T must be provided")
        S = self.spatial_size
        B = quant_input.shape[0]
        device = quant_input.device

        pos_enc = self._get_pos_encoding(T, S)
        real_context = self.input_proj(quant_input) + pos_enc
        self_attn_mask = build_ebt_self_attn_mask(T, S, self.temporal_window, device)
        cross_attn_mask = None
        if parent_features is not None:
            cross_attn_mask = self._build_cross_mask(T, S, device)

        if self.initial_condition == "zero":
            logits = torch.zeros(B, T * S, self.codebook_size, device=device)
        else:
            logits = torch.randn(B, T * S, self.codebook_size, device=device) * 0.1

        alpha = torch.clamp(self.alpha, min=1e-6)
        last_energy = None
        with torch.enable_grad():
            for step in range(self.mcmc_num_steps):
                logits = logits.detach().requires_grad_()
                pred_features = self._soft_lookup(logits, temperature=temperature)
                pred_features_with_pos = pred_features + pos_enc
                combined = torch.cat([real_context, pred_features_with_pos], dim=1)
                energy = self._compute_energy(
                    combined,
                    parent_features=parent_features,
                    self_attn_mask=self_attn_mask,
                    cross_attn_mask=cross_attn_mask,
                )
                create_graph = self.training and step == (self.mcmc_num_steps - 1)
                grad = torch.autograd.grad(energy.sum(), logits, create_graph=create_graph)[0]
                grad = torch.nan_to_num(grad)
                logits = logits - alpha * grad
                last_energy = energy

        self._last_energy = None if last_energy is None else last_energy.detach()
        final_features = self._soft_lookup(logits, temperature=temperature)
        return logits, final_features
