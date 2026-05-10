"""
Configuration dataclasses for VQ-HVEBT.

VQ-HVEBT combines a trainable MobileCLIP encoder with per-stage online
vector quantizers (VQ-VAE style) and EBT predictors that model future
quantized latent states via MCMC.

Hierarchy (three stages, coarsest first in prediction order):
    s3:  256 channels, 2x2 spatial   (coarsest / apex)
    s2:  128 channels, 4x4 spatial
    s1:   64 channels, 8x8 spatial   (finest / base)

Spatial sizes assume 64x64 input frames with base_channels=32 encoder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# --------------------------------------------------------------------------- #
#  Codebook sub-config
# --------------------------------------------------------------------------- #


@dataclass
class VQCodebookConfig:
    """Configuration for one VQ codebook.

    num_codes : size K — number of distinct code vectors (vocabulary size).
    code_dim  : dimension C of each code vector; must match clip_channels of
                the parent VQStageConfig.
    init_mode : how to initialize the codebook.
                "random"          — standard Gaussian init.
                "data_first_batch"— replace entries with random-sampled encoder
                                    outputs on the first forward pass. Call
                                    VectorQuantizer.initialize_from_data(z_e)
                                    manually after the first encoding.
    ema_decay : EMA decay rate for codebook updates. Higher = more stable
                codebook (slower adaptation). Typical: 0.99–0.999.
    commitment_beta : UNUSED (kept for backward compat). No commitment loss
                      in EMA mode.
    """
    num_codes: int = 512
    code_dim: int = 256
    init_mode: str = "data_first_batch"   # "random" | "data_first_batch"
    ema_decay: float = 0.99
    commitment_beta: float = 0.0  # unused in EMA mode
    dead_code_reset: bool = True  # replace dead codes with encoder samples


# --------------------------------------------------------------------------- #
#  Per-stage config
# --------------------------------------------------------------------------- #


@dataclass
class VQStageConfig:
    """Configuration for one VQ-HVEBT stage (one spatial resolution).

    Fields
    ------
    clip_stage_name : which MobileCLIP output to consume ("s1", "s2", "s3").
    clip_channels   : C_clip of that stage.
    H, W            : spatial grid size at this stage.
    transformer_dim : internal dimension D of the predictor transformer.
    n_heads         : number of attention heads (must divide transformer_dim).
    n_layers        : number of transformer blocks.
    ffn_mult        : hidden-to-D multiplier for feed-forward layers.
    dropout         : attention and FF dropout rate.
    attn_bias       : whether QKV projections use bias.
    init_std        : weight initialisation std.
    temporal_window : None = full causal self-attention across the T dimension.
                      1 = self-frame only. k > 1 = last k frames.
    codebook        : VQCodebookConfig for this stage's quantizer.
    mcmc_steps      : number of gradient-descent MCMC steps.
    mcmc_step_size  : initial step size α for logit-space MCMC.
    mcmc_step_learnable : whether α is an nn.Parameter (recommended True).
    truncate_mcmc   : if True, only the last MCMC step keeps the computation
                      graph (saves memory; sufficient for most training).
    pred_loss       : "mse" or "smooth_l1" for the prediction loss term.
    pred_loss_weight: λ_pred in the total loss formula.
    cb_loss_weight  : λ_cb for the codebook loss term.
    commit_loss_weight: λ_commit for the commitment loss term (usually == β).
    """
    clip_stage_name: str = "s1"
    clip_channels: int = 128
    H: int = 32
    W: int = 32
    transformer_dim: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ffn_mult: float = 4.0
    dropout: float = 0.0
    attn_bias: bool = False
    init_std: float = 0.02
    temporal_window: Optional[int] = None
    spatial_window: Optional[int] = None  # None = full spatial attention.
                                          # Integer w: each token attends within
                                          # a w×w neighborhood centered on itself.
    codebook: VQCodebookConfig = field(default_factory=VQCodebookConfig)
    mcmc_steps: int = 20
    mcmc_step_size: float = 10.0
    mcmc_step_learnable: bool = True
    mcmc_grad_clamp: float = 10.0
    mcmc_per_token_norm: bool = True    # F4: normalize MCMC gradient per token
                                        # to unit norm. Makes step size invariant
                                        # to energy function's absolute scale.
    truncate_mcmc: bool = True
    mcmc_no_detach: bool = False     # Like NLP EBT's no_mcmc_detach: do NOT detach
                                    # logits between MCMC steps, keeping the full
                                    # computation graph.  Required for decoder_only_loss
                                    # so pixel loss gradients flow through all steps.
    use_linear_decode: bool = False  # Replace softmax(logits)@codebook with a learned
                                    # nn.Linear(K, C) for MCMC decode (like NLP EBT's
                                    # vocab_to_embed).  Eliminates softmax saturation
                                    # that kills gradient after MCMC refinement.
    # Adaptive MCMC convergence (set adaptive_mcmc=True to enable).
    adaptive_mcmc: bool = False             # False = fixed mcmc_steps.
    adaptive_mcmc_max_steps: int = 50       # hard upper bound on iterations.
    adaptive_mcmc_tol: float = 1e-3         # relative energy-change threshold.
    adaptive_mcmc_patience: int = 3         # consecutive overshoots before halving α.
    adaptive_mcmc_alpha_decay: float = 0.5  # α multiplier on overshoot.
    adaptive_mcmc_step_penalty: float = 0.0 # weight for step-count regularizer.
    soft_target_tau: float = 0.0    # >0: use soft distance-based targets (smooth)
                                    # 0: use hard one-hot targets (default, works
                                    # when indices are stable via detach_pred_context)
    pred_head: bool = True          # F2/F3: learned prediction head that produces
                                    # initial logits for MCMC warm-start.
                                    # Eliminates the zero-init → uniform problem.
    energy_bound: float = 10.0      # F1: bound energy output via tanh scaling.
                                    # energy_head output is scaled to [-bound, +bound].
                                    # 0 = unbounded (legacy). Prevents MCMC blow-up.
    energy_reg_weight: float = 0.01 # F1: regularizer λ * energy².mean() to keep
                                    # energy magnitudes small. 0 = disabled.
    pred_loss: str = "mse"              # "mse" | "smooth_l1"
    pred_loss_weight: float = 1.0
    cb_loss_weight: float = 0.0     # unused (EMA codebook, no gradient loss)
    commit_loss_weight: float = 0.0  # unused (no commitment loss)

    def __post_init__(self):
        # Auto-fill code_dim from clip_channels so caller doesn't have to repeat it.
        if self.codebook.code_dim != self.clip_channels:
            self.codebook = VQCodebookConfig(
                num_codes=self.codebook.num_codes,
                code_dim=self.clip_channels,
                init_mode=self.codebook.init_mode,
                ema_decay=self.codebook.ema_decay,
                commitment_beta=self.codebook.commitment_beta,
            )


# --------------------------------------------------------------------------- #
#  Top-level model config
# --------------------------------------------------------------------------- #


@dataclass
class VQHVEBTConfig:
    """Top-level configuration for the VQ-HVEBT model.

    stages           : list of VQStageConfig, ordered coarsest → finest.
                       The default is a three-stage hierarchy [s3, s2, s1].
    train_encoder    : if False the CLIP encoder is frozen (gradient blocked).
                       Default True (both encoder and codebook train jointly).
    encoder_lr_scale : scale factor applied to encoder param LR relative to
                       the rest of the model (set < 1 to slow down encoder).
    weights_path     : path to MobileCLIP2-S0 pretrained weights.
    use_decoder      : if True attach a pixel decoder on the finest stage.
    decoder_loss_weight : λ_dec in total loss.
    decoder_out_size : output spatial size for the pixel decoder (e.g. 256).
    detach_parent_kv : if True the parent context fed to a finer stage is
                       detached (default True; prevents gradient leakage).
    detach_pred_context : if True the quantized context fed to the predictor
                       is detached from the encoder graph. Encoder trains only
                       via straight-through from the prediction loss on the
                       predicted future tokens.
    """
    stages: List[VQStageConfig] = field(default_factory=lambda: _default_stages())
    train_encoder: bool = True
    encoder_lr_scale: float = 1.0       # encoder LR relative to predictor
    weights_path: str = "clip/MobileCLIP2-S0/mobileclip2_s0.pt"
    use_custom_encoder: bool = True     # Option B: use small ConvEncoder instead of CLIP
    encoder_base_channels: int = 32     # stem width for ConvEncoder (32 → ~1.3M params)
    ema_target_decay: float = 0.999     # EMA decay for target encoder (BYOL/DINO style)
    use_decoder: bool = False
    decoder_loss_weight: float = 1.0
    decoder_out_size: int = 64          # output pixel size of the decoder
    decoder_only_loss: bool = False     # If True, enable decoder pixel loss alongside CE.
                                        # Forces decoder_detach=False and
                                        # detach_parent_kv=False so gradient flows
                                        # from decoder through entire hierarchy.
                                        # Also sets per-stage: mcmc_no_detach=True,
                                        # truncate_mcmc=False.
    context_recon_weight: float = 0.0   # Weight for context-frame reconstruction loss.
                                        # Trains encoder-decoder to faithfully reconstruct
                                        # input frames, creating a meaningful feature space.
    detach_parent_kv: bool = True
    bottom_up_grad_scale: float = 0.1   # Scale factor for gradient flowing from
                                        # child through parent KV. Only active when
                                        # detach_parent_kv=False. Prevents top stages
                                        # from being overwhelmed by cascaded errors.
    decoder_detach: bool = True         # If False, decoder loss gradient flows into
                                        # the predictor (and up through hierarchy if
                                        # detach_parent_kv=False).
    contrastive_loss_weight: float = 0.0
    encoder_warmup_steps: int = 200     # Freeze encoder for first N steps so
                                        # codebook + predictor converge to a stable
                                        # baseline before encoder features start shifting.
    detach_pred_context: bool = False   # With EMA codebook there is no commitment
                                        # loss gradient bomb, so we can let prediction
                                        # loss flow into the encoder via context too.
    codebook_diversity_weight: float = 1.0  # Weight for per-stage codebook diversity
                                            # loss.  Penalises high pairwise cosine
                                            # similarity among encoder features, which
                                            # directly counteracts feature collapse at
                                            # coarse stages (where few spatial tokens
                                            # make EMA winner-take-all extreme).

    def __post_init__(self):
        if self.decoder_only_loss:
            self.use_decoder = True
            self.decoder_detach = False
            self.detach_parent_kv = False
            for s in self.stages:
                # s.use_linear_decode = True
                s.mcmc_no_detach = True
                s.truncate_mcmc = False


def _default_stages() -> List[VQStageConfig]:
    """Three-stage hierarchy for 64×64 images: s3 (coarsest) → s2 → s1 (finest).

    With base_channels=32 encoder on 64×64 input:
      s3: 256 ch, 2×2, temporal_window=4 (full for T≤4)
      s2: 128 ch, 4×4, temporal_window=2
      s1:  64 ch, 8×8, temporal_window=1 (self-frame only)

    Codebook K is INVERSELY proportional to spatial size:
      - Coarser stages (fewer tokens) need MORE codes because each token
        must describe the entire scene compressed into 1-4 spatial positions.
      - Finer stages (many tokens) need FEWER codes because each token
        only describes a single small patch; neighbouring pixels provide
        context.
    """
    s3 = VQStageConfig(
        clip_stage_name="s3",
        clip_channels=256,
        H=2, W=2,
        transformer_dim=64, n_heads=2, n_layers=2,
        temporal_window=4,
        spatial_window=None,   # 2×2 grid → full spatial always
        codebook=VQCodebookConfig(num_codes=512, code_dim=256, ema_decay=0.99),
    )
    s2 = VQStageConfig(
        clip_stage_name="s2",
        clip_channels=128,
        H=4, W=4,
        transformer_dim=64, n_heads=2, n_layers=2,
        temporal_window=2,
        spatial_window=None,   # 4×4 grid → spatial window not needed
        codebook=VQCodebookConfig(num_codes=64, code_dim=128, ema_decay=0.99),
    )
    s1 = VQStageConfig(
        clip_stage_name="s1",
        clip_channels=64,
        H=8, W=8,
        transformer_dim=64, n_heads=2, n_layers=2,
        temporal_window=1,
        spatial_window=None,   # 8×8 grid → full spatial ok at this size
        codebook=VQCodebookConfig(num_codes=16, code_dim=64, ema_decay=0.99),
    )
    return [s3, s2, s1]
