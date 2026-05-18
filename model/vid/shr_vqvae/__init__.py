"""S-HR-VQVAE: Sequential Hierarchical Residual Learning VQ-VAE for video prediction.

Reference: arXiv 2307.06701
"""
from .config import SHRVQVAEConfig
from .model import SHRVQVAEModel

__all__ = ["SHRVQVAEConfig", "SHRVQVAEModel"]
