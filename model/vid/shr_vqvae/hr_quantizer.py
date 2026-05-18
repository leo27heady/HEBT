"""Hierarchical Residual Vector Quantizer with a Conditional Tree codebook.

Implements Equations 3, 4, 5 from S-HR-VQVAE (arXiv 2307.06701).

Conditional tree structure
--------------------------
Layer i contains M^(i+1) codewords in a single flat Embedding table.
During forward, only the M children of the parent chosen at layer i-1 are
considered as candidates (Eq 3).

Notation from the paper
-----------------------
  xi^0        — encoder output z, flattened to [N, C].
  xi^i        — residual after subtracting layers 0..i-1.
  e^i         — nearest codeword at layer i.
  e_C         — combined representation: sum of all e^i.
  sg[·]       — stop-gradient (detach).
  β           — commitment loss weight.

Loss (Eq 4 + 5)
---------------
  L_vq_layer_i = ||sg[xi^{i-1}] - e^i||² + β·||sg[e^i] - xi^{i-1}||²
  L_vq_global  = ||sg[z] - e_C||²         + β·||sg[e_C] - z||²
  L_vq         = sum_i(L_vq_layer_i) + L_vq_global

NOTE on the residual update
---------------------------
The plan uses `xi = xi - e_i` (without detaching e_i) so that the next
layer's commitment loss can back-propagate through the chain
  xi_{i+1} = xi_i - e_i → xi_0 = z.
This is faithful to the plan. To keep layers' losses truly independent,
change to `xi = xi - e_i.detach()`.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HR_Quantizer(nn.Module):
    """Conditional-tree hierarchical residual VQ codebook.

    Args:
        num_layers    : tree depth  n  (paper default: 3).
        M             : branching factor; layer i has M^(i+1) codewords.
        embedding_dim : codeword dimension  C.
        beta          : commitment loss coefficient  β.
    """

    def __init__(
        self,
        num_layers: int = 3,
        M: int = 16,
        embedding_dim: int = 128,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.M = M
        self.beta = beta
        self.embedding_dim = embedding_dim

        # Layer i: M^(i+1) entries  →  M root-branches at L0, M children per
        # node at each subsequent level.
        self.codebooks = nn.ModuleList(
            [nn.Embedding(M ** (i + 1), embedding_dim) for i in range(num_layers)]
        )
        for cb in self.codebooks:
            cb.weight.data.uniform_(-1.0 / M, 1.0 / M)

    # ------------------------------------------------------------------ #
    # Forward pass (Eq 3 + 5)
    # ------------------------------------------------------------------ #

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Encode-and-quantize a spatial feature map.

        Args:
            z : (B, C, H, W) — encoder output.

        Returns:
            e_C_st      : (B, C, H, W) with straight-through gradient.
            indices_list: list of ``num_layers`` tensors, each (B, H, W),
                          holding the *local* (0..M-1) index per position.
            loss_vq     : scalar VQ loss (Eq 4 + 5 combined).
        """
        B, C, H, W = z.shape
        N = B * H * W
        z_flat = z.permute(0, 2, 3, 1).reshape(N, C)  # [N, C]

        xi = z_flat           # running residual
        quantized: List[torch.Tensor] = []
        indices_list: List[torch.Tensor] = []
        loss_vq: torch.Tensor = z.new_zeros(())

        # Every token starts at the virtual root: its "parent global index" is 0,
        # meaning it will look at codebook-0 entries [0 .. M-1].
        prefix = torch.zeros(N, dtype=torch.long, device=z.device)

        for i in range(self.num_layers):
            # ---- 1. Determine the M valid codeword indices (Eq 3) ---- #
            start = prefix * self.M                              # [N]
            offsets = torch.arange(self.M, device=z.device)     # [M]
            valid_idx = start.unsqueeze(1) + offsets.unsqueeze(0)  # [N, M]

            # ---- 2. Nearest-neighbour search ---- #
            cw = self.codebooks[i](valid_idx)                   # [N, M, C]
            dist = ((xi.unsqueeze(1) - cw) ** 2).sum(dim=-1)   # [N, M]
            local_choice = dist.argmin(dim=1)                   # [N]
            global_choice = start + local_choice                # [N]

            indices_list.append(local_choice.view(B, H, W))

            # ---- 3. Retrieve quantized vector ---- #
            e_i = self.codebooks[i](global_choice)              # [N, C]
            quantized.append(e_i)

            # ---- 4. Per-layer VQ loss (Eq 5) ---- #
            # Codebook term: move e^i toward sg[xi^{i-1}]
            # Commitment term: move encoder output toward sg[e^i]
            loss_vq = loss_vq + (
                F.mse_loss(e_i, xi.detach())
                + self.beta * F.mse_loss(xi, e_i.detach())
            )

            # ---- 5. Residual update + advance tree prefix ---- #
            xi = xi - e_i          # NOTE: e_i not detached — see module docstring
            prefix = global_choice

        # ---- 6. Global combined loss (Eq 4) ---- #
        e_C_flat = torch.stack(quantized, dim=0).sum(dim=0)    # [N, C]
        loss_vq = loss_vq + (
            F.mse_loss(e_C_flat, z_flat.detach())
            + self.beta * F.mse_loss(z_flat, e_C_flat.detach())
        )

        # ---- 7. Reshape + Straight-Through Estimator ---- #
        e_C = e_C_flat.view(B, H, W, C).permute(0, 3, 1, 2)   # [B, C, H, W]
        e_C_st = z + (e_C - z).detach()                        # STE

        return e_C_st, indices_list, loss_vq

    # ------------------------------------------------------------------ #
    # Decode helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def decode_indices(self, indices_list: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct e_C from per-layer local index grids (no gradient).

        Args:
            indices_list: list of (B, H, W) local-index tensors (0..M-1).

        Returns:
            (B, C, H, W) combined quantized tensor.
        """
        B, H, W = indices_list[0].shape
        C = self.embedding_dim
        N = B * H * W

        prefix = torch.zeros(N, dtype=torch.long, device=indices_list[0].device)
        quantized: List[torch.Tensor] = []

        for i, local_idx in enumerate(indices_list):
            local_flat = local_idx.reshape(N)
            start = prefix * self.M
            global_choice = start + local_flat
            e_i = self.codebooks[i](global_choice)              # [N, C]
            quantized.append(e_i)
            prefix = global_choice

        e_C = torch.stack(quantized, dim=0).sum(dim=0)         # [N, C]
        return e_C.view(B, H, W, C).permute(0, 3, 1, 2)        # [B, C, H, W]

    def gumbel_decode(
        self,
        logits_list: List[torch.Tensor],
        tau: float = 1.0,
    ) -> torch.Tensor:
        """Differentiable decode of per-layer logits via Gumbel-Softmax.

        Used in Stage 3 (joint training) to bridge the discrete sampling step
        so that gradients flow back through the Decoder.

        The tree is traversed greedily (hard=True Gumbel at each layer so the
        next layer's valid M entries are well-defined).  Gradients flow through
        each layer's own Gumbel-Softmax.

        Args:
            logits_list: list of (B, M, *spatial_dims) tensors, one per layer.
                         ``spatial_dims`` may be e.g. (S, h, w).
            tau        : Gumbel-Softmax temperature.

        Returns:
            (B, C, N) where N = prod(spatial_dims).
        """
        B = logits_list[0].shape[0]
        C = self.embedding_dim

        # Infer the flat spatial size from the first logits tensor
        spatial_shape = logits_list[0].shape[2:]   # e.g. (S, h, w)
        N = 1
        for s in spatial_shape:
            N *= s

        prefix: torch.Tensor | None = None          # global parent indices [B, N]
        quantized: List[torch.Tensor] = []

        for i, logits in enumerate(logits_list):
            # logits: [B, M, *spatial] → [B, N, M]
            logits_flat = logits.reshape(B, self.M, N).permute(0, 2, 1)

            # Hard Gumbel-Softmax: forward = one-hot, backward = soft gradient
            one_hot = F.gumbel_softmax(logits_flat, tau=tau, hard=True, dim=-1)  # [B, N, M]
            hard_local = one_hot.detach().argmax(dim=-1)                          # [B, N]

            if prefix is None:
                # Layer 0: valid entries are global indices 0..M-1 for all tokens
                cw = self.codebooks[i].weight[:self.M]                  # [M, C]
                # e_i[b, n, c] = sum_m one_hot[b, n, m] * cw[m, c]
                e_i = one_hot @ cw                                       # [B, N, C]
                prefix = hard_local                                      # [B, N]
            else:
                # Layers 1+: valid entries depend on the parent's choice
                start = prefix * self.M                                  # [B, N]
                offsets = torch.arange(self.M, device=logits.device)    # [M]
                valid_global = start.unsqueeze(-1) + offsets             # [B, N, M]
                cw = self.codebooks[i](valid_global)                    # [B, N, M, C]
                # weighted sum via one-hot
                e_i = (one_hot.unsqueeze(-1) * cw).sum(dim=2)          # [B, N, C]
                prefix = start + hard_local                              # [B, N]

            quantized.append(e_i)

        e_C = torch.stack(quantized, dim=0).sum(dim=0)                 # [B, N, C]
        return e_C.permute(0, 2, 1)                                     # [B, C, N]
