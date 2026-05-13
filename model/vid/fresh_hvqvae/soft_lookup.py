"""Soft lookup utilities for LFQ codebook matrices."""

import torch


def build_lfq_codebook_matrix(dim: int) -> torch.Tensor:
    """
    Build all 2^dim binary code vectors as a matrix.
    LFQ uses {-1, +1} values.

    Args:
        dim: LFQ dimension (log2 of codebook size)

    Returns:
        (2^dim, dim) tensor of all possible code vectors.
    """
    K = 2 ** dim
    codes = torch.zeros(K, dim)
    for i in range(K):
        for bit in range(dim):
            codes[i, bit] = 1.0 if (i >> bit) & 1 else -1.0
    return codes
