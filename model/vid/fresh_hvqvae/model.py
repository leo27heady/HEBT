"""Full Fresh HVQVAE model assembly."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FreshHVQVAEConfig
from .encoder import HierarchicalEncoder
from .decoder import DecoderBot, DecoderMid, DecoderTop
from .predictor import PredictorStage
from .ebt_predictor import EBTPredictorStage
from .soft_lookup import build_lfq_codebook_matrix


class FreshHVQVAE(nn.Module):
    """
    Fresh Hierarchical VQ-VAE for video prediction.
    3-stage hierarchy: bot (16x16), mid (4x4), top (1x1).
    Per-stage reconstruction with per-stage decoders + isolated per-stage prediction.
    """

    def __init__(self, cfg: FreshHVQVAEConfig, skip_predictors: bool = False):
        super().__init__()
        self.cfg = cfg
        self.skip_predictors = skip_predictors

        # Encoder
        self.encoder = HierarchicalEncoder(cfg)

        # Per-stage decoders
        self.decoder_bot = DecoderBot(cfg.C_bot)
        self.decoder_mid = DecoderMid(cfg.C_mid)
        self.decoder_top = DecoderTop(cfg.C_top)

        # Predictor stages (optional)
        if not skip_predictors:
            if cfg.predictor_mode == 'ebt':
                ebt_kwargs = dict(
                    mcmc_num_steps=cfg.ebt_mcmc_num_steps,
                    mcmc_step_size=cfg.ebt_mcmc_step_size,
                    langevin_noise=cfg.ebt_langevin_noise,
                    truncate_mcmc=cfg.ebt_truncate_mcmc,
                    clamp_grad_max=cfg.ebt_clamp_grad_max,
                    mcmc_step_size_learnable=cfg.ebt_mcmc_step_size_learnable,
                    initial_condition=cfg.ebt_initial_condition,
                )
                self.predictor_top = EBTPredictorStage(
                    dim=cfg.pred_dim_top, n_heads=cfg.pred_n_heads, n_layers=cfg.ebt_n_layers,
                    codebook_size=cfg.K_top, spatial_size=1,
                    temporal_window=cfg.window_top, has_parent=False,
                    lfq_dim=cfg.lfq_dim_top, max_T=cfg.max_T, **ebt_kwargs,
                )
                self.predictor_mid = EBTPredictorStage(
                    dim=cfg.pred_dim_mid, n_heads=cfg.pred_n_heads, n_layers=cfg.ebt_n_layers,
                    codebook_size=cfg.K_mid, spatial_size=16,
                    temporal_window=cfg.window_mid, has_parent=True,
                    parent_dim=cfg.pred_dim_top, lfq_dim=cfg.lfq_dim_mid, max_T=cfg.max_T,
                    **ebt_kwargs,
                )
                self.predictor_bot = EBTPredictorStage(
                    dim=cfg.pred_dim_bot, n_heads=cfg.pred_n_heads, n_layers=cfg.ebt_n_layers,
                    codebook_size=cfg.K_bot, spatial_size=256,
                    temporal_window=cfg.window_bot, has_parent=True,
                    parent_dim=cfg.pred_dim_mid, lfq_dim=cfg.lfq_dim_bot, max_T=cfg.max_T,
                    **ebt_kwargs,
                )
            else:
                self.predictor_top = PredictorStage(
                    dim=cfg.pred_dim_top, n_heads=cfg.pred_n_heads, n_layers=cfg.pred_n_layers,
                    codebook_size=cfg.K_top, spatial_size=1,
                    temporal_window=cfg.window_top, has_parent=False,
                    lfq_dim=cfg.lfq_dim_top, max_T=cfg.max_T,
                )
                self.predictor_mid = PredictorStage(
                    dim=cfg.pred_dim_mid, n_heads=cfg.pred_n_heads, n_layers=cfg.pred_n_layers,
                    codebook_size=cfg.K_mid, spatial_size=16,
                    temporal_window=cfg.window_mid, has_parent=True,
                    parent_dim=cfg.pred_dim_top, lfq_dim=cfg.lfq_dim_mid, max_T=cfg.max_T,
                )
                self.predictor_bot = PredictorStage(
                    dim=cfg.pred_dim_bot, n_heads=cfg.pred_n_heads, n_layers=cfg.pred_n_layers,
                    codebook_size=cfg.K_bot, spatial_size=256,
                    temporal_window=cfg.window_bot, has_parent=True,
                    parent_dim=cfg.pred_dim_mid, lfq_dim=cfg.lfq_dim_bot, max_T=cfg.max_T,
                )


    def encode(self, video: torch.Tensor) -> dict:
        """
        Encode all frames.
        video: (B, T+1, 3, H, W)
        Returns dict with temporal-reshaped features and indices.
        """
        B, Tp1 = video.shape[:2]
        flat = video.reshape(B * Tp1, 3, video.shape[3], video.shape[4])
        enc = self.encoder(flat)

        # Reshape to temporal: (B, T+1, ...)
        for key in ['quant_bot', 'quant_mid', 'quant_top']:
            C = enc[key].shape[1]
            spatial = enc[key].shape[2:]
            enc[key] = enc[key].reshape(B, Tp1, C, *spatial)
        for key in ['idx_bot', 'idx_mid', 'idx_top']:
            spatial = enc[key].shape[1:]
            enc[key] = enc[key].reshape(B, Tp1, *spatial)

        return enc

    def reconstruct(self, enc: dict, B: int, Tp1: int):
        """
        Reconstruct from all 3 stages (for per-stage MSE).
        Returns 3 reconstructions, all (B*(T+1), 3, 64, 64).
        """
        cfg = self.cfg

        # Bot: direct to decoder_bot
        quant_bot = enc['quant_bot'].reshape(B * Tp1, cfg.C_bot, 16, 16)
        recon_bot = self.decoder_bot(quant_bot)

        # Mid: direct to decoder_mid
        quant_mid = enc['quant_mid'].reshape(B * Tp1, cfg.C_mid, 4, 4)
        recon_mid = self.decoder_mid(quant_mid)

        # Top: direct to decoder_top
        quant_top = enc['quant_top'].reshape(B * Tp1, cfg.C_top, 1, 1)
        recon_top = self.decoder_top(quant_top)

        return recon_bot, recon_mid, recon_top

    def predict(self, enc: dict, T: int) -> dict:
        """
        Run top-down predictor on first T frames (encoder outputs DETACHED).
        All parent features are detached for stage isolation.
        """
        B = enc['quant_top'].shape[0]
        cfg = self.cfg

        inp_top = enc['quant_top'][:, :T].detach().reshape(B, T * 1, -1)
        inp_mid = enc['quant_mid'][:, :T].detach().reshape(B, T * 16, -1)
        inp_bot = enc['quant_bot'][:, :T].detach().reshape(B, T * 256, -1)

        logits_top, feat_top = self.predictor_top(inp_top, T=T,
                                                   temperature=cfg.soft_lookup_temperature)
        logits_mid, feat_mid = self.predictor_mid(
            inp_mid, parent_features=feat_top.detach(), T=T,
            temperature=cfg.soft_lookup_temperature,
        )
        logits_bot, feat_bot = self.predictor_bot(
            inp_bot, parent_features=feat_mid.detach(), T=T,
            temperature=cfg.soft_lookup_temperature,
        )

        result = {
            'logits_top': logits_top, 'logits_mid': logits_mid, 'logits_bot': logits_bot,
            'feat_top': feat_top, 'feat_mid': feat_mid, 'feat_bot': feat_bot,
        }

        # Include EBT energy maps if available
        if hasattr(self.predictor_top, '_last_energy') and self.predictor_top._last_energy is not None:
            result['energy_top'] = self.predictor_top._last_energy
        if hasattr(self.predictor_mid, '_last_energy') and self.predictor_mid._last_energy is not None:
            result['energy_mid'] = self.predictor_mid._last_energy
        if hasattr(self.predictor_bot, '_last_energy') and self.predictor_bot._last_energy is not None:
            result['energy_bot'] = self.predictor_bot._last_energy

        return result

    def forward(self, video: torch.Tensor) -> dict:
        """
        Full training forward: compute all losses.
        video: (B, T+1, 3, 64, 64)
        Returns dict of individual losses (not summed — caller handles backward).
        """
        B, Tp1 = video.shape[:2]
        T = Tp1 - 1
        all_frames = video.reshape(B * Tp1, 3, video.shape[3], video.shape[4])

        # Encode WITH gradient (for reconstruction)
        enc = self.encode(video)

        # Per-stage reconstruction
        recon_bot, recon_mid, recon_top = self.reconstruct(enc, B, Tp1)
        mse_bot = F.mse_loss(recon_bot, all_frames)
        mse_mid = F.mse_loss(recon_mid, all_frames)
        mse_top = F.mse_loss(recon_top, all_frames)
        vq_loss = enc['loss_bot'] + enc['loss_mid'] + enc['loss_top']

        # Prediction CE (detached from encoder)
        pred = self.predict(enc, T)
        tgt_top = enc['idx_top'][:, 1:].detach().reshape(B * T * 1)
        tgt_mid = enc['idx_mid'][:, 1:].detach().reshape(B * T * 16)
        tgt_bot = enc['idx_bot'][:, 1:].detach().reshape(B * T * 256)

        ce_top = F.cross_entropy(
            pred['logits_top'].reshape(-1, self.cfg.K_top), tgt_top
        )
        ce_mid = F.cross_entropy(
            pred['logits_mid'].reshape(-1, self.cfg.K_mid), tgt_mid
        )
        ce_bot = F.cross_entropy(
            pred['logits_bot'].reshape(-1, self.cfg.K_bot), tgt_bot
        )

        return {
            'mse_bot': mse_bot, 'mse_mid': mse_mid, 'mse_top': mse_top,
            'vq_loss': vq_loss,
            'ce_top': ce_top, 'ce_mid': ce_mid, 'ce_bot': ce_bot,
        }

    def get_encoder_decoder_params(self):
        """Parameters updated by reconstruction loss."""
        return (
            list(self.encoder.parameters()) +
            list(self.decoder_bot.parameters()) +
            list(self.decoder_mid.parameters()) +
            list(self.decoder_top.parameters())
        )

    def get_bot_stage_params(self):
        """Bot encoder block + VQ + decoder."""
        return (
            list(self.encoder.enc_to_bot.parameters()) +
            list(self.encoder.bot_to_vq.parameters()) +
            list(self.encoder.bot_from_vq.parameters()) +
            list(self.encoder.vq_bot.parameters()) +
            list(self.decoder_bot.parameters())
        )

    def get_mid_stage_params(self):
        """Mid encoder block + VQ + decoder."""
        return (
            list(self.encoder.enc_bot_to_mid.parameters()) +
            list(self.encoder.mid_to_vq.parameters()) +
            list(self.encoder.mid_from_vq.parameters()) +
            list(self.encoder.vq_mid.parameters()) +
            list(self.decoder_mid.parameters())
        )

    def get_top_stage_params(self):
        """Top encoder block + VQ + BN + decoder."""
        return (
            list(self.encoder.enc_mid_to_top.parameters()) +
            list(self.encoder.top_to_vq.parameters()) +
            list(self.encoder.top_pre_vq_norm.parameters()) +
            list(self.encoder.top_from_vq.parameters()) +
            list(self.encoder.vq_top.parameters()) +
            list(self.decoder_top.parameters())
        )

    def get_predictor_top_params(self):
        return list(self.predictor_top.parameters())

    def get_predictor_mid_params(self):
        return list(self.predictor_mid.parameters())

    def get_predictor_bot_params(self):
        return list(self.predictor_bot.parameters())

    @torch.no_grad()
    def build_visualization(self, video: torch.Tensor) -> torch.Tensor:
        """
        Build a visualization grid for the first sample in the batch.
        video: (B, T+1, 3, 64, 64)
        Returns: (3, H_grid, W_grid) image tensor in [0,1].

        Grid layout:
          Encoder-only: 4 rows (GT, Enc Bot, Enc Mid, Enc Top)
          With predictors: adds per-stage Pred + Entropy rows (+ Energy rows in EBT mode)
        """
        from PIL import Image, ImageDraw, ImageFont
        import math

        cfg = self.cfg
        B, Tp1 = video.shape[:2]
        T = Tp1 - 1
        H, W = video.shape[3], video.shape[4]
        device = video.device

        # Encode all frames
        enc = self.encode(video)

        # --- Encoder reconstructions (ALL frames 0..T) ---
        quant_bot_all = enc['quant_bot'][:1].reshape(Tp1, cfg.C_bot, 16, 16)
        recon_enc_bot = self.decoder_bot(quant_bot_all)

        quant_mid_all = enc['quant_mid'][:1].reshape(Tp1, cfg.C_mid, 4, 4)
        recon_enc_mid = self.decoder_mid(quant_mid_all)

        quant_top_all = enc['quant_top'][:1].reshape(Tp1, cfg.C_top, 1, 1)
        recon_enc_top = self.decoder_top(quant_top_all)

        # Ground truth all frames
        gt_all = video[0]  # (T+1, 3, 64, 64)

        row_data = [gt_all, recon_enc_bot, recon_enc_mid, recon_enc_top]
        row_labels = ['GT', 'Enc Bot', 'Enc Mid', 'Enc Top']

        # --- Predictor decoded (predicted frames 1..T) ---
        if not self.skip_predictors:
            # EBT predictors need gradients for MCMC, so re-enable inside no_grad block
            with torch.enable_grad():
                pred = self.predict(enc, T)
            # Blank frames: black in [-1,1] space = -1, gray maps in [-1,1] space
            blank = torch.full((1, 3, H, W), -1.0, device=device)
            blank_gray = torch.full((1, 3, H, W), -1.0, device=device)

            stages = [
                ('bot', pred['logits_bot'], self.encoder.vq_bot, self.encoder.bot_from_vq,
                 self.decoder_bot, cfg.lfq_dim_bot, cfg.K_bot, 16),
                ('mid', pred['logits_mid'], self.encoder.vq_mid, self.encoder.mid_from_vq,
                 self.decoder_mid, cfg.lfq_dim_mid, cfg.K_mid, 4),
                ('top', pred['logits_top'], self.encoder.vq_top, self.encoder.top_from_vq,
                 self.decoder_top, cfg.lfq_dim_top, cfg.K_top, 1),
            ]

            for stage_name, logits, vq_mod, from_vq, decoder, lfq_dim, K, spatial_hw in stages:
                # Decode predicted frames
                pred_idx = logits[0].argmax(dim=-1)  # (T*S,)
                pred_codes = vq_mod.indices_to_codes(pred_idx)
                S = spatial_hw * spatial_hw
                pred_codes = pred_codes.reshape(T, spatial_hw, spatial_hw, lfq_dim).permute(0, 3, 1, 2)
                pred_recon = decoder(from_vq(pred_codes))  # (T, 3, 64, 64)

                # Prediction row (blank first frame + T predicted)
                row_data.append(torch.cat([blank, pred_recon], dim=0))
                row_labels.append(f'Pred {stage_name.capitalize()}')

                # --- Entropy map ---
                # logits shape: (B, T*S, K) — take first sample
                stage_logits = logits[0]  # (T*S, K)
                probs = torch.softmax(stage_logits, dim=-1)  # (T*S, K)
                log_probs = torch.log(probs + 1e-10)
                entropy_per_pos = -(probs * log_probs).sum(dim=-1)  # (T*S,)
                max_entropy = math.log(K)
                # Normalize: 0 entropy → black (0), max entropy → white (1)
                entropy_norm = (entropy_per_pos / max_entropy).clamp(0, 1)  # (T*S,)
                entropy_map = entropy_norm.reshape(T, 1, spatial_hw, spatial_hw)  # (T, 1, h, w)
                # Interpolate to 64x64
                entropy_map_up = F.interpolate(entropy_map, size=(H, W), mode='nearest')  # (T, 1, 64, 64)
                # Convert to [-1, 1] so global (x+1)/2 rescaling maps back to [0, 1]
                entropy_map_up = entropy_map_up * 2 - 1
                entropy_rgb = entropy_map_up.expand(T, 3, H, W)  # grayscale → RGB
                row_data.append(torch.cat([blank_gray, entropy_rgb], dim=0))
                row_labels.append(f'Entropy {stage_name.capitalize()}')

            # --- Energy maps (EBT mode only) ---
            has_energy = any(f'energy_{s}' in pred for s in ('bot', 'mid', 'top'))
            if has_energy:
                for stage_name, spatial_hw in [('bot', 16), ('mid', 4), ('top', 1)]:
                    energy_key = f'energy_{stage_name}'
                    if energy_key not in pred:
                        continue
                    S = spatial_hw * spatial_hw
                    energy = pred[energy_key][0]  # (T*S, 1)
                    energy_flat = energy.squeeze(-1)  # (T*S,)

                    # Robust normalization: use percentile-based min/max to handle outliers
                    e_min = energy_flat.quantile(0.02)
                    e_max = energy_flat.quantile(0.98)
                    if e_max - e_min < 1e-6:
                        energy_norm = torch.zeros_like(energy_flat)
                    else:
                        energy_norm = ((energy_flat - e_min) / (e_max - e_min)).clamp(0, 1)

                    energy_map = energy_norm.reshape(T, 1, spatial_hw, spatial_hw)
                    energy_map_up = F.interpolate(energy_map, size=(H, W), mode='nearest')
                    # Convert to [-1, 1] so global (x+1)/2 rescaling maps back to [0, 1]
                    energy_map_up = energy_map_up * 2 - 1
                    energy_rgb = energy_map_up.expand(T, 3, H, W)
                    row_data.append(torch.cat([blank_gray, energy_rgb], dim=0))
                    row_labels.append(f'Energy {stage_name.capitalize()}')

        # Assemble raw grid without labels
        n_rows = len(row_data)
        n_cols = Tp1
        row_images = []
        for frames in row_data:
            row_img = torch.cat([frames[t] for t in range(Tp1)], dim=2)
            row_images.append(row_img)
        raw_grid = torch.cat(row_images, dim=1)
        # Rescale from [-1, 1] to [0, 1] for visualization
        # Entropy/energy rows are stored in [-1, 1] too, so this is uniform
        raw_grid = (raw_grid + 1) / 2
        raw_grid = raw_grid.clamp(0, 1)

        # Convert to PIL for text rendering
        grid_np = (raw_grid.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        pil_img = Image.fromarray(grid_np)

        # Add margins for labels
        label_left_w = 90
        label_top_h = 18

        canvas = Image.new('RGB', (label_left_w + n_cols * W, label_top_h + n_rows * H), (0, 0, 0))
        canvas.paste(pil_img, (label_left_w, label_top_h))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Column headers (frame indices)
        for c in range(n_cols):
            x = label_left_w + c * W + W // 2 - 5
            draw.text((x, 2), str(c), fill=(255, 255, 255), font=font)

        # Row labels
        for r, label in enumerate(row_labels):
            y = label_top_h + r * H + H // 2 - 6
            draw.text((3, y), label, fill=(255, 255, 255), font=font)

        import numpy as np
        result = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
        return result
