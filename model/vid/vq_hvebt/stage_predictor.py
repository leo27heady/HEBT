"""
VQ-HVEBT Stage Predictor.

One stage predictor models future quantized latent states via an
Energy-Based Model (EBT) with MCMC inference.

Architecture for a single stage k
-----------------------------------

The predictor takes two aligned feature grids at the same spatial resolution:

    real_ctx  : (B, T, C, H, W) — quantized context (past frames).
                These are straight-through quantized encoder features z_q_st.
    pred_embed: (B, T, C, H, W) — predicted future embedding from MCMC.
                Decoded from predicted logits: softmax(l) @ E.

Both grids are flattened to tokens of shape (B, T*H*W, C), concatenated
channel-wise to (B, T*H*W, 2C), then projected to transformer dim D.  The
transformer computes a per-token scalar energy; summing gives total energy.

MCMC (logit-space gradient descent, NLP EBT style)
----------------------------------------------------

The optimization variable is ``pred_logits ∈ R^{B, T*H*W, K}`` — raw logits
over the K codebook entries at each spatiotemporal token.

At each MCMC step:
    1. Decode: z_pred = softmax(pred_logits) @ E   → (B, T*H*W, C)
    2. Reshape z_pred to (B, T, C, H, W).
    3. Compute energy = stage(real_ctx, z_pred_reshaped, parent_ctx).
    4. Gradient: g = ∂energy/∂pred_logits  (via autograd through softmax+matmul).
    5. Update:   pred_logits = pred_logits - α * g

The training loss is **cross-entropy** directly on the final pred_logits
against target code indices (not MSE on embeddings). This gives the same
direct gradient from loss → logits as NLP EBT, avoiding the softmax dilution
problem that plagues MSE-on-embedding loss.

Shannon entropy of the predicted distribution is naturally available via
softmax(pred_logits).

The computation graph through MCMC is controlled by `truncate_mcmc`:
  - False (default): all steps keep create_graph=True, full gradient chain.
  - True: only the last step keeps the graph (saves memory).
Following NLP EBT, logits are DETACHED between steps by default.

Cross-attention from parent stage (hierarchical conditioning)
-------------------------------------------------------------

Each finer stage optionally receives a parent context from the stage above
(coarser spatial resolution). The cross-attention mask restricts each child
token at (y, x, t) to attend only to the parent token at (y//2, x//2, t),
matching the CLIP bottom-up spatial hierarchy.

The parent context is expected to be DETACHED by the caller before being
passed here (so gradients do NOT flow from this stage back into the parent
stage's parameters via cross-attention KV).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vid.hvebt.cross_attention import CrossAttention3DRoPE, build_cross_attn_mask
from model.vid.hvebt.positional import RoPE3DCache, apply_rope3d, build_rope3d
from model.vid.hvebt.hvebt import (
    SelfAttention3DRoPE,
    FeedForward,
    build_block_causal_mask,
)
from model.vid.vq_hvebt.config import VQStageConfig
from model.vid.vq_hvebt.quantizer import VectorQuantizer


# --------------------------------------------------------------------------- #
#  Transformer block (with optional cross-attention to parent)
# --------------------------------------------------------------------------- #


class _Block(nn.Module):
    """One transformer block: LayerNorm → SelfAttn → (optional CrossAttn) → FF."""

    def __init__(self, cfg: VQStageConfig, cross_attn_kv_dim: Optional[int] = None):
        super().__init__()
        D = cfg.transformer_dim
        self.norm1 = nn.LayerNorm(D)
        self.attn = SelfAttention3DRoPE(D, cfg.n_heads, cfg.attn_bias, cfg.dropout)
        self.use_cross_attn = cross_attn_kv_dim is not None
        if self.use_cross_attn:
            self.norm_cross = nn.LayerNorm(D)
            self.cross_attn = CrossAttention3DRoPE(
                dim_q=D,
                dim_kv=D,  # parent has been pre-projected to D
                n_heads=cfg.n_heads,
                bias=cfg.attn_bias,
                dropout=cfg.dropout,
            )
        self.norm2 = nn.LayerNorm(D)
        self.ff = FeedForward(D, cfg.ffn_mult, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,                          # (B, N, D)
        rope: RoPE3DCache,
        attn_mask: torch.Tensor,                  # (N, N) additive
        context: Optional[torch.Tensor] = None,   # (B, Np, D)
        rope_ctx: Optional[RoPE3DCache] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope, attn_mask)
        if self.use_cross_attn:
            if context is None or rope_ctx is None or cross_mask is None:
                raise ValueError("Block cross-attn is enabled but context/rope_ctx/cross_mask is None")
            x = x + self.cross_attn(self.norm_cross(x), context, rope, rope_ctx, cross_mask)
        x = x + self.ff(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
#  VQHVEBTStage: single-stage EBT predictor
# --------------------------------------------------------------------------- #


class VQHVEBTStage(nn.Module):
    """Energy-based predictor for one spatial resolution stage.

    Parameters
    ----------
    cfg : VQStageConfig
        Stage configuration (channels, spatial dims, transformer hypers, MCMC).
    quantizer : VectorQuantizer
        The codebook for this stage. Used inside run_mcmc to decode logits.
        Not owned here — owned by the VQHVEBTModel.
    parent_cfg : VQStageConfig or None
        Config of the coarser parent stage. When not None, cross-attention
        is added in every transformer block. Parent context must then be
        provided in forward_energy().
    """

    def __init__(
        self,
        cfg: VQStageConfig,
        quantizer: VectorQuantizer,
        parent_cfg: Optional[VQStageConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.quantizer = quantizer  # reference only (parameters owned by model)
        self.use_cross_attn = parent_cfg is not None
        if self.use_cross_attn:
            self.Hp = parent_cfg.H
            self.Wp = parent_cfg.W
            self.parent_proj = nn.Linear(parent_cfg.clip_channels, cfg.transformer_dim, bias=True)
            self.parent_norm = nn.LayerNorm(cfg.transformer_dim)

        D = cfg.transformer_dim
        C = cfg.clip_channels

        # Channel-concat of (real_ctx, pred_embed) → project to D.
        self.input_proj = nn.Linear(2 * C, D, bias=True)

        cross_kv_dim: Optional[int] = D if self.use_cross_attn else None
        self.blocks = nn.ModuleList([
            _Block(cfg, cross_attn_kv_dim=cross_kv_dim)
            for _ in range(cfg.n_layers)
        ])
        self.norm_out = nn.LayerNorm(D)
        self.energy_head = nn.Linear(D, 1, bias=True)

        # F1: Energy bounding — prevent unbounded energy (range [-234, +1037])
        # from causing MCMC blow-up. When > 0, output = bound * tanh(raw/bound).
        self.energy_bound = cfg.energy_bound

        # F2/F3: Learned prediction head for MCMC warm-start.
        # Produces initial logits from context features so MCMC doesn't start
        # from all-zeros (which gives uniform softmax → flat energy landscape).
        K = cfg.codebook.num_codes
        self.pred_head: Optional[nn.Module] = None
        if cfg.pred_head:
            self.pred_head = nn.Sequential(
                nn.Linear(D, D, bias=True),
                nn.GELU(),
                nn.Linear(D, K, bias=True),
            )

        # Learnable MCMC step size per stage.
        self.alpha = nn.Parameter(
            torch.tensor(float(cfg.mcmc_step_size)),
            requires_grad=cfg.mcmc_step_learnable,
        )

        # Cache attention masks and RoPE tables (keyed by (T, device)).
        self._rope_cache: Dict[Tuple[int, torch.device], RoPE3DCache] = {}
        self._mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._parent_rope_cache: Dict[Tuple[int, torch.device], RoPE3DCache] = {}
        self._cross_mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Weight init
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.cfg.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Energy head: small init so initial energy is near zero.
        nn.init.normal_(self.energy_head.weight, std=self.cfg.init_std * 0.1)
        nn.init.zeros_(self.energy_head.bias)
        # Prediction head: small init so initial logits are near zero (near-uniform).
        if self.pred_head is not None:
            for m in self.pred_head:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=self.cfg.init_std)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    #  Cache helpers
    # ------------------------------------------------------------------ #

    def _get_rope(self, T: int, device: torch.device, dtype: torch.dtype) -> RoPE3DCache:
        key = (T, device)
        c = self._rope_cache.get(key)
        if c is None or c.cos.dtype != dtype:
            head_dim = self.cfg.transformer_dim // self.cfg.n_heads
            c = build_rope3d(T, self.cfg.H, self.cfg.W, head_dim, device, dtype)
            self._rope_cache[key] = c
        return c

    def _get_mask(self, T: int, device: torch.device) -> torch.Tensor:
        key = (T, device)
        m = self._mask_cache.get(key)
        if m is None:
            m = build_block_causal_mask(
                T, self.cfg.H, device,
                temporal_window=self.cfg.temporal_window,
                W=self.cfg.W,
                spatial_window=self.cfg.spatial_window,
            )
            self._mask_cache[key] = m
        return m

    def _get_parent_rope(self, T: int, device: torch.device, dtype: torch.dtype) -> RoPE3DCache:
        key = (T, device)
        c = self._parent_rope_cache.get(key)
        if c is None or c.cos.dtype != dtype:
            head_dim = self.cfg.transformer_dim // self.cfg.n_heads
            c = build_rope3d(T, self.Hp, self.Wp, head_dim, device, dtype)
            self._parent_rope_cache[key] = c
        return c

    def _get_cross_mask(self, T: int, device: torch.device) -> torch.Tensor:
        key = (T, device)
        m = self._cross_mask_cache.get(key)
        if m is None:
            m = build_cross_attn_mask(
                T, self.cfg.H, self.cfg.W, self.Hp, self.Wp, device
            )
            self._cross_mask_cache[key] = m
        return m

    # ------------------------------------------------------------------ #
    #  Energy computation
    # ------------------------------------------------------------------ #

    def forward_energy(
        self,
        real_ctx: torch.Tensor,                          # (B, T, C, H, W)
        pred_embed: torch.Tensor,                        # (B, T, C, H, W)
        parent_context: Optional[torch.Tensor] = None,  # (B, T, Cp, Hp, Wp) — DETACHED
    ) -> torch.Tensor:
        """Compute per-token scalar energy.

        Args:
            real_ctx    : (B, T, C, H, W) straight-through quantized context.
            pred_embed  : (B, T, C, H, W) current predicted future embedding.
            parent_context : (B, T, Cp, Hp, Wp) from coarser stage (detached
                             by caller). Required iff stage has cross-attention.

        Returns:
            (B, T*H*W) per-token energy.
        """
        B, T, C, H, W = real_ctx.shape
        cfg = self.cfg
        if (C, H, W) != (cfg.clip_channels, cfg.H, cfg.W):
            raise ValueError(
                f"real_ctx shape mismatch: expected (B,T,{cfg.clip_channels},{cfg.H},{cfg.W}), "
                f"got (B,T,{C},{H},{W})"
            )
        if pred_embed.shape != real_ctx.shape:
            raise ValueError("pred_embed must have same shape as real_ctx")
        if self.use_cross_attn and parent_context is None:
            raise ValueError("parent_context is required for this stage (cross-attn enabled)")
        if parent_context is not None and not self.use_cross_attn:
            raise ValueError("parent_context was given but cross-attn is disabled for this stage")

        # Flatten spatial dims: (B, T, C, H, W) → (B, T*H*W, C)
        r = real_ctx.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)
        p = pred_embed.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)
        tokens = torch.cat([r, p], dim=-1)   # (B, N, 2C)
        x = self.input_proj(tokens)           # (B, N, D)

        rope = self._get_rope(T, x.device, x.dtype)
        mask = self._get_mask(T, x.device)

        # Optionally prepare parent context for cross-attention.
        ctx_proj: Optional[torch.Tensor] = None
        rope_ctx: Optional[RoPE3DCache] = None
        cross_mask: Optional[torch.Tensor] = None
        if self.use_cross_attn and parent_context is not None:
            Bp, Tp, Cp, Hp, Wp = parent_context.shape
            if Bp != B or Tp != T:
                raise ValueError(f"parent_context batch/time mismatch: child (B,T)=({B},{T}), parent=({Bp},{Tp})")
            # (B, T, Cp, Hp, Wp) → (B, T*Hp*Wp, Cp)
            ctx = parent_context.permute(0, 1, 3, 4, 2).reshape(B, T * Hp * Wp, Cp)
            ctx = self.parent_proj(ctx)        # → (B, T*Hp*Wp, D)
            ctx_proj = self.parent_norm(ctx)
            rope_ctx = self._get_parent_rope(T, x.device, x.dtype)
            cross_mask = self._get_cross_mask(T, x.device)

        for blk in self.blocks:
            x = blk(x, rope, mask, context=ctx_proj, rope_ctx=rope_ctx, cross_mask=cross_mask)

        x = self.norm_out(x)                  # (B, N, D)
        energy = self.energy_head(x).squeeze(-1)  # (B, N)

        # F1: Bound energy to [-bound, +bound] via tanh scaling.
        # This prevents unbounded energy (range [-234, +1037]) from causing
        # astronomical MCMC gradients and fp16 overflow.
        if self.energy_bound > 0:
            energy = self.energy_bound * torch.tanh(energy / self.energy_bound)

        return energy

    # ------------------------------------------------------------------ #
    #  MCMC inference in logit space (NLP EBT style)
    # ------------------------------------------------------------------ #

    def _compute_init_logits(
        self,
        real_ctx: torch.Tensor,
        parent_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute warm-start logits from the learned prediction head.

        Uses the same transformer trunk as forward_energy but feeds it
        (real_ctx, zero_embed) and applies the prediction head on the
        intermediate representation to produce (B, N, K) logits.

        This gives MCMC a meaningful starting point instead of all-zeros
        (which produce uniform softmax → flat energy landscape).
        """
        B, T, C, H, W = real_ctx.shape
        N = T * H * W
        cfg = self.cfg
        D = cfg.transformer_dim

        # Use zero embedding as the "query" (we don't have a prediction yet).
        zero_embed = torch.zeros_like(real_ctx)
        r = real_ctx.permute(0, 1, 3, 4, 2).reshape(B, N, C)
        p = zero_embed.permute(0, 1, 3, 4, 2).reshape(B, N, C)
        tokens = torch.cat([r, p], dim=-1)
        x = self.input_proj(tokens)

        rope = self._get_rope(T, x.device, x.dtype)
        mask = self._get_mask(T, x.device)

        ctx_proj = None
        rope_ctx = None
        cross_mask = None
        if self.use_cross_attn and parent_context is not None:
            Bp, Tp, Cp, Hp, Wp = parent_context.shape
            ctx = parent_context.permute(0, 1, 3, 4, 2).reshape(B, T * Hp * Wp, Cp)
            ctx = self.parent_proj(ctx)
            ctx_proj = self.parent_norm(ctx)
            rope_ctx = self._get_parent_rope(T, x.device, x.dtype)
            cross_mask = self._get_cross_mask(T, x.device)

        for blk in self.blocks:
            x = blk(x, rope, mask, context=ctx_proj, rope_ctx=rope_ctx, cross_mask=cross_mask)

        x = self.norm_out(x)  # (B, N, D)
        init_logits = self.pred_head(x)  # (B, N, K)
        return init_logits

    def run_mcmc(
        self,
        real_ctx: torch.Tensor,                          # (B, T, C, H, W)
        init_logits: Optional[torch.Tensor] = None,      # (B, T*H*W, K) or None
        parent_context: Optional[torch.Tensor] = None,
        learning: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[float]]:
        """Run MCMC to find low-energy logits in code distribution space.

        Changes from the original:
        - F2/F3: If pred_head is enabled, compute warm-start logits from
          the learned prediction head instead of zeros. The prediction head
          also gets a direct CE loss (returned as the first entry in
          all_step_logits).
        - F4: Per-token gradient normalization. Each token's gradient is
          normalized to unit norm before the step, making the effective
          step size invariant to the energy function's absolute scale.
          The step_size α then directly controls how far in logit-space
          each token moves per step (in L2 norm).

        Args:
            real_ctx      : (B, T, C, H, W) quantized context (z_q_st).
            init_logits   : initial logits; if None, uses prediction head or zeros.
            parent_context: (B, T, Cp, Hp, Wp) from coarser stage (detached).
            learning      : if True, create_graph=True for training.

        Returns:
            all_step_logits : List of (B, T*H*W, K) logits after each MCMC step.
                              If pred_head is enabled, the FIRST entry is the
                              pred_head output (before any MCMC steps).
            final_embed     : (B, T, C, H, W) — decoded embedding from final logits.
            energy_trace    : list of per-step summed energies (for diagnostics).
        """
        B, T, C, H, W = real_ctx.shape
        N = T * H * W
        K = self.cfg.codebook.num_codes
        device = real_ctx.device

        all_step_logits: List[torch.Tensor] = []

        # ---- Initialise logits -------------------------------------------- #
        if init_logits is not None:
            if init_logits.shape != (B, N, K):
                raise ValueError(
                    f"init_logits shape {tuple(init_logits.shape)} != expected ({B},{N},{K})"
                )
            pred_logits = init_logits.clone()
        elif self.pred_head is not None:
            # F2/F3: Warm-start from learned prediction head.
            pred_logits = self._compute_init_logits(real_ctx, parent_context)
            # Include pred_head logits in loss (direct CE supervision on the head).
            all_step_logits.append(pred_logits)
        else:
            pred_logits = torch.zeros(B, N, K, device=device, dtype=real_ctx.dtype)

        # Clamp step size to be numerically safe.
        alpha = torch.clamp(self.alpha, min=1e-6)

        num_steps = self.cfg.mcmc_steps
        energy_trace: List[float] = []
        use_per_token_norm = self.cfg.mcmc_per_token_norm

        for step in range(num_steps):
            create_graph = learning and (
                not self.cfg.truncate_mcmc or step == num_steps - 1
            )

            # Detach logits between steps (NLP EBT default).
            pred_logits = pred_logits.detach().requires_grad_(True)

            with torch.enable_grad():
                z_pred_flat = self.quantizer.decode_logits(pred_logits)  # (B, N, C)
                z_pred = z_pred_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

                energy = self.forward_energy(real_ctx, z_pred, parent_context)  # (B, N)
                energy_trace.append(energy.detach().sum().item())

                grad = torch.autograd.grad(
                    [energy.sum()], [pred_logits],
                    create_graph=create_graph,
                    retain_graph=create_graph,
                )[0]

            # F4: Per-token gradient normalization.
            # Normalize each token's K-dim gradient to unit L2 norm.
            # This makes the effective step size independent of the energy
            # function's absolute scale (which can vary by 1000×).
            if use_per_token_norm:
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                grad = grad / grad_norm
            elif self.cfg.mcmc_grad_clamp > 0:
                # Legacy: global clamp (only when per-token norm is off).
                grad = torch.clamp(grad, min=-self.cfg.mcmc_grad_clamp, max=self.cfg.mcmc_grad_clamp)

            pred_logits = pred_logits - alpha * grad
            all_step_logits.append(pred_logits)

        # Decode final logits → final embedding.
        z_pred_flat = self.quantizer.decode_logits(pred_logits)   # (B, N, C)
        z_pred_final = z_pred_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

        return all_step_logits, z_pred_final, energy_trace

    # ------------------------------------------------------------------ #
    #  Adaptive MCMC (run until convergence)
    # ------------------------------------------------------------------ #

    def run_mcmc_adaptive(
        self,
        real_ctx: torch.Tensor,                          # (B, T, C, H, W)
        init_logits: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
        learning: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[float], int, str]:
        """Adaptive MCMC: iterate until energy converges, then one final step with graph.

        Phase 1: run without computation graph until convergence (relative energy
                  change < tol) or max_steps reached. On consecutive overshoots,
                  halve step size.
        Phase 2: one step with create_graph=True to produce differentiable logits.

        Returns:
            all_step_logits : [pred_head_logits (if pred_head), final_logits].
            final_embed     : (B, T, C, H, W).
            energy_trace    : per-step summed energy.
            num_steps       : total steps taken (for metrics / step penalty).
            stop_reason     : "converged" | "max_steps" | "nan_grad".
        """
        B, T, C, H, W = real_ctx.shape
        N = T * H * W
        K = self.cfg.codebook.num_codes
        device = real_ctx.device
        cfg = self.cfg

        all_step_logits: List[torch.Tensor] = []

        # ---- Initialise logits -------------------------------------------- #
        if init_logits is not None:
            pred_logits = init_logits.clone()
        elif self.pred_head is not None:
            pred_logits = self._compute_init_logits(real_ctx, parent_context)
            all_step_logits.append(pred_logits)
        else:
            pred_logits = torch.zeros(B, N, K, device=device, dtype=real_ctx.dtype)

        alpha = torch.clamp(self.alpha, min=1e-6)
        use_per_token_norm = cfg.mcmc_per_token_norm
        max_steps = cfg.adaptive_mcmc_max_steps
        tol = cfg.adaptive_mcmc_tol
        patience = cfg.adaptive_mcmc_patience
        alpha_decay = cfg.adaptive_mcmc_alpha_decay

        energy_trace: List[float] = []
        prev_energy_val: Optional[float] = None
        overshoot_count = 0
        converge_step = 0
        stop_reason = "max_steps"   # default if loop exhausts all iterations

        # ---- Phase 1: iterate without graph until convergence ------------- #
        with torch.set_grad_enabled(True):
            for step in range(max_steps - 1):
                pred_logits = pred_logits.detach().requires_grad_(True)

                z_pred_flat = self.quantizer.decode_logits(pred_logits)
                z_pred = z_pred_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()
                energy = self.forward_energy(real_ctx, z_pred, parent_context)
                energy_val = energy.detach().sum().item()
                energy_trace.append(energy_val)

                # Convergence check.
                if prev_energy_val is not None:
                    rel = abs(energy_val - prev_energy_val) / (abs(prev_energy_val) + 1e-8)
                    if rel < tol:
                        converge_step = step
                        stop_reason = "converged"
                        break
                    # Overshoot detection.
                    if energy_val > prev_energy_val:
                        overshoot_count += 1
                        if overshoot_count >= patience:
                            alpha = alpha * alpha_decay
                            overshoot_count = 0
                    else:
                        overshoot_count = 0
                prev_energy_val = energy_val

                grad = torch.autograd.grad(
                    [energy.sum()], [pred_logits], create_graph=False
                )[0]
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    converge_step = step
                    stop_reason = "nan_grad"
                    break

                if use_per_token_norm:
                    grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    grad = grad / grad_norm
                elif cfg.mcmc_grad_clamp > 0:
                    grad = torch.clamp(grad, min=-cfg.mcmc_grad_clamp, max=cfg.mcmc_grad_clamp)

                pred_logits = (pred_logits - alpha * grad).detach()
            else:
                converge_step = max_steps - 1

        # ---- Phase 2: one final step WITH graph --------------------------- #
        pred_logits = pred_logits.detach().requires_grad_(True)

        with torch.enable_grad():
            z_pred_flat = self.quantizer.decode_logits(pred_logits)
            z_pred = z_pred_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()
            energy = self.forward_energy(real_ctx, z_pred, parent_context)
            energy_trace.append(energy.detach().sum().item())

            create_graph = learning
            grad = torch.autograd.grad(
                [energy.sum()], [pred_logits],
                create_graph=create_graph, retain_graph=create_graph,
            )[0]

            if use_per_token_norm:
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                grad = grad / grad_norm
            elif cfg.mcmc_grad_clamp > 0:
                grad = torch.clamp(grad, min=-cfg.mcmc_grad_clamp, max=cfg.mcmc_grad_clamp)

            final_logits = pred_logits - alpha * grad
            all_step_logits.append(final_logits)

        # Decode final logits → final embedding.
        z_pred_flat = self.quantizer.decode_logits(final_logits)
        z_pred_final = z_pred_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

        num_steps = converge_step + 1  # +1 for the final graph step
        return all_step_logits, z_pred_final, energy_trace, num_steps, stop_reason
