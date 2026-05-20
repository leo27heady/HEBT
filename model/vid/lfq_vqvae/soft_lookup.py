"""Soft lookup utilities for LFQ codebook matrices."""

from __future__ import annotations

import torch


def build_lfq_codebook_matrix(dim: int) -> torch.Tensor:
    """
    Build all 2^dim binary code vectors for LFQ.

    Returns:
        Tensor of shape (2^dim, dim) with values in {-1, +1}.
    """
    k = 2 ** dim
    codes = torch.zeros(k, dim)
    for i in range(k):
        for bit in range(dim):
            codes[i, bit] = 1.0 if (i >> bit) & 1 else -1.0
    return codes
