"""
Vector Quantizer for HVQVAE.

Directly adapted from vqvae-reference/models/quantizer.py.
Gradient-trained codebook with straight-through estimator.
No EMA, no normalization tricks, no diversity loss.

Loss = ||z_q.detach() - z_e||² + β * ||z_q - z_e.detach()||²
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """VQ-VAE discretization bottleneck.

    Parameters
    ----------
    n_e : number of embedding vectors (codebook size K).
    e_dim : dimension of each embedding vector.
    beta : commitment cost weight.
    """

    def __init__(self, n_e: int, e_dim: int, beta: float):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoder output.

        Args:
            z: (B, C, H, W) continuous encoder features.

        Returns:
            loss: scalar embedding loss (codebook + commitment).
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

        # VQ loss: codebook loss + commitment loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Perplexity (codebook utilization metric)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # Back to (B, C, H, W)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
