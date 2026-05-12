"""
Vector Quantizer for HVQVAE.

Adapted from vqvae-reference/models/quantizer.py with additions:
  - EMA codebook updates (optional, default ON): more stable than gradient-
    based training, especially when token-to-code ratio is low.
  - Dead code reinitialization: replaces codes unused for N steps with
    perturbed copies of active encoder outputs.
  - Entropy regularization loss (optional): encourages uniform code usage.

When EMA is enabled the codebook embedding is NOT trained by gradient descent;
instead it tracks encoder outputs via exponential moving average (decay ~0.99).
The only gradient-based loss is the commitment term β * ||z_q - z_e.detach()||².
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """VQ-VAE discretization bottleneck with EMA + dead-code reset.

    Parameters
    ----------
    n_e : number of embedding vectors (codebook size K).
    e_dim : dimension of each embedding vector.
    beta : commitment cost weight.
    use_ema : if True, update codebook via EMA instead of gradients.
    ema_decay : EMA decay factor (higher = slower update).
    dead_code_threshold : reset codes unused for this many forward calls.
                          0 = disable reset.
    entropy_weight : weight for entropy regularization loss (0 = disabled).
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 100,
        entropy_weight: float = 0.0,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        self.entropy_weight = entropy_weight

        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        if use_ema:
            # EMA codebook: embedding.weight is NOT trained by optimizer
            self.embedding.weight.requires_grad = False
            # Running sum of assigned encoder vectors
            self.register_buffer("_ema_cluster_size", torch.zeros(n_e))
            self.register_buffer("_ema_embed_sum", self.embedding.weight.clone())

        if dead_code_threshold > 0:
            # Track how many forward calls since each code was last used
            self.register_buffer("_usage_count", torch.zeros(n_e, dtype=torch.long))
            self.register_buffer("_forward_count", torch.tensor(0, dtype=torch.long))

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoder output.

        Args:
            z: (B, C, H, W) continuous encoder features.

        Returns:
            loss: scalar embedding loss.
            z_q: (B, C, H, W) quantized features (straight-through).
            perplexity: scalar codebook utilization metric.
            min_encodings: (B*H*W, K) one-hot assignment matrix.
            min_encoding_indices: (B*H*W, 1) nearest code indices.
        """
        # (B, C, H, W) → (B, H, W, C) → (B*H*W, C)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # Compute distances: ||z - e||² = ||z||² + ||e||² - 2⟨z, e⟩
        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Find nearest codebook entries
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, device=z.device
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Quantized vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # ---- Loss computation ----
        if self.use_ema:
            # EMA: only commitment loss is backpropagated
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2)

            # Update codebook via EMA (only during training)
            if self.training:
                self._ema_update(z_flattened, min_encodings)
        else:
            # Gradient-based: codebook loss + commitment loss
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Perplexity (codebook utilization metric)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # ---- Entropy regularization (optional) ----
        if self.entropy_weight > 0:
            # Encourage uniform code usage: maximize entropy of avg assignment
            # entropy_loss is NEGATIVE because we want to MAXIMIZE entropy
            entropy = -torch.sum(e_mean * torch.log(e_mean + 1e-10))
            max_entropy = torch.log(torch.tensor(self.n_e, dtype=z.dtype, device=z.device))
            # Loss = weight * (1 - entropy/max_entropy), so 0 when perfectly uniform
            loss = loss + self.entropy_weight * (1.0 - entropy / max_entropy)

        # ---- Dead code reinitialization (during training) ----
        if self.training and self.dead_code_threshold > 0:
            self._track_and_reset_dead_codes(z_flattened, min_encoding_indices)

        # Back to (B, C, H, W)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

    @torch.no_grad()
    def _ema_update(
        self, z_flattened: torch.Tensor, min_encodings: torch.Tensor
    ) -> None:
        """Update codebook via exponential moving average of encoder outputs."""
        # Count how many tokens are assigned to each code
        cluster_size = min_encodings.sum(dim=0)  # (K,)
        # Sum of encoder vectors assigned to each code
        embed_sum = min_encodings.t() @ z_flattened  # (K, e_dim)

        self._ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )
        self._ema_embed_sum.mul_(self.ema_decay).add_(
            embed_sum, alpha=1 - self.ema_decay
        )

        # Laplace smoothing to avoid division by zero
        n = self._ema_cluster_size.sum()
        cluster_size_smooth = (
            (self._ema_cluster_size + 1e-5) / (n + self.n_e * 1e-5) * n
        )

        # Update codebook embeddings
        self.embedding.weight.data.copy_(
            self._ema_embed_sum / cluster_size_smooth.unsqueeze(1)
        )

    @torch.no_grad()
    def _track_and_reset_dead_codes(
        self, z_flattened: torch.Tensor, min_encoding_indices: torch.Tensor
    ) -> None:
        """Replace dead codes with perturbed copies of active encoder outputs."""
        self._forward_count += 1

        # Mark used codes
        used_codes = min_encoding_indices.squeeze(1).unique()
        self._usage_count[used_codes] = self._forward_count

        # Find dead codes (not used for dead_code_threshold steps)
        steps_since_use = self._forward_count - self._usage_count
        dead_mask = steps_since_use >= self.dead_code_threshold
        n_dead = dead_mask.sum().item()

        if n_dead == 0:
            return

        # Sample replacement vectors from current batch encoder outputs
        n_tokens = z_flattened.shape[0]
        if n_tokens == 0:
            return

        # Random indices from the current batch
        replace_indices = torch.randint(0, n_tokens, (n_dead,), device=z_flattened.device)
        new_vectors = z_flattened[replace_indices]

        # Add small noise to break symmetry
        noise = torch.randn_like(new_vectors) * 0.01
        new_vectors = new_vectors + noise

        # Replace dead codes
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        self.embedding.weight.data[dead_indices] = new_vectors
        self._usage_count[dead_indices] = self._forward_count

        # Also reset EMA state for these codes
        if self.use_ema:
            self._ema_cluster_size[dead_indices] = 1e-5
            self._ema_embed_sum[dead_indices] = new_vectors
