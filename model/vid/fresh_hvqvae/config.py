"""Configuration for Fresh Hierarchical VQ-VAE."""

from dataclasses import dataclass


@dataclass
class FreshHVQVAEConfig:
    # Image
    image_size: int = 64

    # Encoder channels
    C_bot: int = 64
    C_mid: int = 128
    C_top: int = 256

    # Codebook sizes (must be powers of 2 for LFQ)
    K_bot: int = 2**6
    K_mid: int = 2**10
    K_top: int = 2**14

    # LFQ dims (= log2(K))
    lfq_dim_bot: int = 6
    lfq_dim_mid: int = 10
    lfq_dim_top: int = 14

    # LFQ entropy regularization
    entropy_loss_weight: float = 0.1
    diversity_gamma: float = 1.0

    # Predictor
    pred_n_heads: int = 8
    pred_n_layers: int = 4
    pred_dim_top: int = 256    # = C_top
    pred_dim_mid: int = 128    # = C_mid
    pred_dim_bot: int = 64     # = C_bot

    # Temporal windows
    window_top: int = -1       # -1 = full context (causal)
    window_mid: int = 2
    window_bot: int = 1

    # Cross-attention
    soft_lookup_temperature: float = 1.0

    # Reconstruction loss weights
    weight_mse_bot: float = 1.0
    weight_mse_mid: float = 1.0
    weight_mse_top: float = 1.0

    # Training
    lr_encoder_decoder: float = 3e-4
    lr_predictor_top: float = 3e-4
    lr_predictor_mid: float = 3e-4
    lr_predictor_bot: float = 1e-4
    max_grad_norm: float = 1.0

    # Sequence
    max_T: int = 16

    # Predictor mode
    predictor_mode: str = 'vanilla'  # 'vanilla' | 'ebt'

    # EBT-MCMC predictor settings
    ebt_mcmc_num_steps: int = 5
    ebt_mcmc_step_size: float = 0.1
    ebt_langevin_noise: float = 0.01
    ebt_truncate_mcmc: bool = True
    ebt_clamp_grad_max: float = 10.0
    ebt_mcmc_step_size_learnable: bool = True
    ebt_initial_condition: str = 'zeros'  # 'zeros' | 'random_noise'
    ebt_n_layers: int = 2  # energy transformer layers (can be shallower)
