"""S-HR-VQVAE: Full model combining HR-VQVAE and AST-PM.

Three-stage training API
------------------------
  model.stage1_loss(x)       — Stage 1: disjoint HR-VQVAE (reconstruction).
  model.stage2_loss(x_seq)   — Stage 2: disjoint AST-PM (CE on frozen codes).
  model.stage3_loss(x_seq)   — Stage 3: joint fine-tuning (Eq 9).

Inference
---------
  model.generate(context, num_future)  — autoregressive future-frame generation.

Parameter groups for Stage-specific optimizers
-----------------------------------------------
  model.vqvae_params()       — encoder + quantizer + decoder.
  model.astpm_params()       — all AST-PM modules.
  model.decoder_params()     — decoder only (unfrozen in Stage 3).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ast_pm import AST_PM
from .config import SHRVQVAEConfig
from .encoder_decoder import ConvDecoder, ConvEncoder
from .hr_quantizer import HR_Quantizer


class SHRVQVAEModel(nn.Module):
    """Sequential Hierarchical Residual Learning VQ-VAE (arXiv 2307.06701)."""

    def __init__(self, cfg: SHRVQVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = ConvEncoder(cfg)
        self.quantizer = HR_Quantizer(
            num_layers=cfg.num_vq_layers,
            M=cfg.M,
            embedding_dim=cfg.embedding_dim,
            beta=cfg.vq_beta,
        )
        self.decoder = ConvDecoder(cfg)

        # One AST-PM per VQ layer; each predicts local indices (0..M-1)
        self.ast_pms = nn.ModuleList(
            [
                AST_PM(
                    M=cfg.M,
                    in_embed_dim=cfg.embedding_dim,
                    hidden=cfg.astpm_hidden,
                    num_heads=cfg.astpm_heads,
                    num_blocks=cfg.astpm_blocks,
                )
                for _ in range(cfg.num_vq_layers)
            ]
        )

    # ------------------------------------------------------------------ #
    # Parameter-group helpers
    # ------------------------------------------------------------------ #

    def vqvae_params(self):
        return (
            list(self.encoder.parameters())
            + list(self.quantizer.parameters())
            + list(self.decoder.parameters())
        )

    def astpm_params(self):
        return list(self.ast_pms.parameters())

    def decoder_params(self):
        return list(self.decoder.parameters())

    # ------------------------------------------------------------------ #
    # Primitive operations
    # ------------------------------------------------------------------ #

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → latent z (B, E, h, w)."""
        return self.encoder(x)

    def quantize(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Quantize z.  Returns (e_C_st, indices_list, loss_vq)."""
        return self.quantizer(z)

    def decode(self, e_C: torch.Tensor) -> torch.Tensor:
        """(B, E, h, w) → reconstructed image (B, 3, H, W)."""
        return self.decoder(e_C)

    # ------------------------------------------------------------------ #
    # Stage 1: Disjoint HR-VQVAE Training
    # ------------------------------------------------------------------ #

    def stage1_loss(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Reconstruction + VQ loss on a single-frame batch.

        Args:
            x : (B, 3, H, W) in [0, 1].

        Returns:
            total_loss, metrics dict with keys ``loss``, ``recon``, ``vq``.
        """
        z = self.encode(x)
        e_C_st, _indices, loss_vq = self.quantize(z)
        x_hat = self.decode(e_C_st)
        loss_recon = F.mse_loss(x_hat, x)
        total = loss_recon + loss_vq
        return total, {"loss": total, "recon": loss_recon, "vq": loss_vq}

    # ------------------------------------------------------------------ #
    # Stage 2: Disjoint AST-PM Training
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _extract_indices(
        self, x_seq: torch.Tensor
    ) -> List[torch.Tensor]:
        """Encode a sequence and extract per-layer local VQ indices.

        Args:
            x_seq : (B, T_total, 3, H, W).

        Returns:
            list of ``num_vq_layers`` tensors, each (B, T_total, h, w).
        """
        B, T_total, C, H, W = x_seq.shape
        x_flat = x_seq.reshape(B * T_total, C, H, W)
        z = self.encode(x_flat)
        _e, indices_list, _loss = self.quantize(z)
        h, w = indices_list[0].shape[1:]
        return [idx.view(B, T_total, h, w) for idx in indices_list]

    def stage2_loss(
        self, x_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Per-layer cross-entropy loss for AST-PM with frozen encoder/quantizer.

        PixelCNN-style: input == target.  The Type-A causal mask in the first
        CausalBlock prevents the model from seeing the current position, so the
        network must predict each token from strictly prior positions in
        raster-scan order (time → height → width), matching Eq 6 in the paper.

        Args:
            x_seq : (B, T+S, 3, H, W).

        Returns:
            total_loss (mean over layers), metrics dict.
        """
        indices = self._extract_indices(x_seq)  # list of [B, T+S, h, w]

        total_ce: torch.Tensor = x_seq.new_zeros(())
        for ast_pm, idx in zip(self.ast_pms, indices):
            # PixelCNN-style: input = target = full index sequence.
            # The mask-A first block ensures no self-information leakage.
            logits = ast_pm(idx)    # (B, M, T+S, h, w)
            total_ce = total_ce + F.cross_entropy(logits, idx)

        total_ce = total_ce / self.cfg.num_vq_layers
        return total_ce, {"loss": total_ce, "ce": total_ce}

    # ------------------------------------------------------------------ #
    # Stage 3: Joint Training  (Eq 9)
    # ------------------------------------------------------------------ #

    def stage3_loss(
        self, x_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Joint CE + pixel-reconstruction loss via Gumbel-Softmax decode (Eq 9).

        Frozen: encoder, quantizer.
        Unfrozen: ast_pms, decoder.

        PixelCNN-style: input == target (mask-A prevents self-leakage).
        CE is computed over future frames only (Eq 7).
        Future-frame logits are decoded via Gumbel-Softmax so pixel gradients
        flow back to AST-PM parameters.

        Args:
            x_seq : (B, T+S, 3, H, W).

        Returns:
            total_loss, metrics dict with keys ``loss``, ``ce``, ``recon``.
        """
        cfg = self.cfg
        B, TS, C, H, W = x_seq.shape
        T, S = cfg.T, cfg.S
        assert TS == T + S, f"Expected {T + S} frames, got {TS}."

        indices = self._extract_indices(x_seq)  # list of [B, T+S, h, w]
        h, w = indices[0].shape[2:]

        ce_loss: torch.Tensor = x_seq.new_zeros(())
        logits_future: List[torch.Tensor] = []

        for ast_pm, idx in zip(self.ast_pms, indices):
            # PixelCNN-style: input = target = full sequence
            logits = ast_pm(idx)                    # (B, M, T+S, h, w)

            # CE over future frames only (paper Eq 7)
            logits_fut = logits[:, :, T:, :, :]     # (B, M, S, h, w)
            tgt_fut = idx[:, T:, :, :]              # (B, S, h, w)
            ce_loss = ce_loss + F.cross_entropy(logits_fut, tgt_fut)

            # Collect future logits for Gumbel-Softmax decoding
            logits_future.append(logits_fut)

        ce_loss = ce_loss / cfg.num_vq_layers

        # ---- Gumbel-Softmax → continuous future embeddings ---- #
        # gumbel_decode returns (B, C, N) where N = S*h*w
        e_C = self.quantizer.gumbel_decode(logits_future, tau=cfg.gumbel_tau)
        # e_C is (B, C, S*h*w) — unflatten spatial dims correctly
        e_C = e_C.view(B, cfg.embedding_dim, S, h, w)        # (B, C, S, h, w)
        e_C = e_C.permute(0, 2, 1, 3, 4).contiguous()        # (B, S, C, h, w)
        e_C = e_C.reshape(B * S, cfg.embedding_dim, h, w)    # (B*S, C, h, w)

        x_hat_future = self.decode(e_C)                    # (B*S, 3, H, W)
        x_future = x_seq[:, T:].reshape(B * S, C, H, W)
        recon_loss = F.mse_loss(x_hat_future, x_future)

        total = ce_loss + cfg.lambda_joint * recon_loss
        return total, {"loss": total, "ce": ce_loss, "recon": recon_loss}

    # ------------------------------------------------------------------ #
    # Inference: autoregressive generation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        num_future: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively generate future frames from context.

        Generation is pixel-by-pixel in raster-scan order (time → h → w).
        Each layer's AST-PM is run independently (the conditional tree
        dependency is handled by HR_Quantizer.decode_indices).

        Args:
            context    : (B, T_ctx, 3, H, W) observed frames in [0, 1].
            num_future : number of future frames to generate.
            temperature: sampling temperature (0 = argmax / greedy).

        Returns:
            (B, num_future, 3, H, W) predicted frames in [0, 1].
        """
        cfg = self.cfg
        B, T_ctx, C, H, W = context.shape

        # Encode context frames
        ctx_flat = context.reshape(B * T_ctx, C, H, W)
        z = self.encode(ctx_flat)
        h, w = z.shape[2:]

        _e, ctx_indices, _loss = self.quantize(z)
        # idx_buffers[layer]: (B, T_ctx, h, w)
        idx_buffers: List[torch.Tensor] = [
            idx.view(B, T_ctx, h, w) for idx in ctx_indices
        ]

        generated_frames: List[torch.Tensor] = []

        for _step in range(num_future):
            # new_frame_idx[layer]: (B, h, w) — will be filled position by position
            new_frame_idx: List[torch.Tensor] = [
                torch.zeros(B, h, w, dtype=torch.long, device=context.device)
                for _ in range(cfg.num_vq_layers)
            ]

            # Raster-scan spatial autoregression (independently per layer)
            for layer_i, ast_pm in enumerate(self.ast_pms):
                # Construct input: known buffer + new (partially-filled) frame
                # Shape: (B, T_cur + 1, h, w)
                cur_buf = torch.cat(
                    [idx_buffers[layer_i], new_frame_idx[layer_i].unsqueeze(1)],
                    dim=1,
                )

                for hi in range(h):
                    for wi in range(w):
                        logits = ast_pm(cur_buf)        # (B, M, T_cur+1, h, w)
                        # Logit for the last time-step at (hi, wi)
                        pos_logits = logits[:, :, -1, hi, wi]  # (B, M)

                        if temperature == 0:
                            chosen = pos_logits.argmax(dim=-1)
                        else:
                            probs = torch.softmax(pos_logits / temperature, dim=-1)
                            chosen = torch.multinomial(probs, 1).squeeze(1)

                        # Fill in this position and update the buffer in-place
                        cur_buf[:, -1, hi, wi] = chosen

                # The last frame in cur_buf now holds the fully generated frame
                new_frame_idx[layer_i] = cur_buf[:, -1, :, :]   # (B, h, w)

            # Decode the newly generated index grid → image
            e_C = self.quantizer.decode_indices(new_frame_idx)   # (B, E, h, w)
            frame = self.decode(e_C)                              # (B, 3, H, W)
            generated_frames.append(frame)

            # Append generated frame indices to buffers for the next step
            for layer_i in range(cfg.num_vq_layers):
                idx_buffers[layer_i] = torch.cat(
                    [idx_buffers[layer_i], new_frame_idx[layer_i].unsqueeze(1)],
                    dim=1,
                )

        return torch.stack(generated_frames, dim=1)  # (B, num_future, 3, H, W)
