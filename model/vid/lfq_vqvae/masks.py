"""Attention mask construction for LFQ-VQVAE video predictor stages."""

from __future__ import annotations

import torch


def build_temporal_window_mask(T: int, S: int, temporal_window: int, device: torch.device) -> torch.Tensor:
    """
    Build boolean self-attention mask with causal temporal windowing.

    Token i (frame i//S) can attend token j (frame j//S) if:
      1) j_frame <= i_frame (causal)
      2) i_frame - j_frame < temporal_window

    Returns:
        (T*S, T*S) boolean mask, True means allowed.
    """
    if temporal_window == -1:
        temporal_window = T

    frame_indices = torch.arange(T, device=device)
    diff = frame_indices.unsqueeze(1) - frame_indices.unsqueeze(0)  # (T, T)
    frame_mask = (diff >= 0) & (diff < temporal_window)  # (T, T)
    token_mask = frame_mask.repeat_interleave(S, dim=0).repeat_interleave(S, dim=1)
    return token_mask


def build_cross_attn_mask_top_to_mid(T: int, device: torch.device) -> torch.Tensor:
    """
    Cross-attention mask: mid queries (T*16) -> top keys (T*1), same frame only.
    """
    s_child = 16
    s_parent = 1
    mask = torch.zeros(T * s_child, T * s_parent, dtype=torch.bool, device=device)
    for t in range(T):
        mask[t * s_child:(t + 1) * s_child, t:t + 1] = True
    return mask


def build_cross_attn_mask_mid_to_bot(T: int, device: torch.device) -> torch.Tensor:
    """
    Cross-attention mask: bot queries (T*256) -> mid keys (T*16), same frame only.
    Bot (r,c) attends mid (r//4, c//4).
    """
    s_child = 256
    s_parent = 16

    r = torch.arange(16, device=device)
    c = torch.arange(16, device=device)
    grid_r, grid_c = torch.meshgrid(r, c, indexing="ij")
    parent_r = grid_r // 4
    parent_c = grid_c // 4
    parent_idx = (parent_r * 4 + parent_c).reshape(s_child)

    frame_mask = torch.zeros(s_child, s_parent, dtype=torch.bool, device=device)
    frame_mask[torch.arange(s_child, device=device), parent_idx] = True

    full = torch.zeros(T * s_child, T * s_parent, dtype=torch.bool, device=device)
    for t in range(T):
        full[t * s_child:(t + 1) * s_child, t * s_parent:(t + 1) * s_parent] = frame_mask
    return full
