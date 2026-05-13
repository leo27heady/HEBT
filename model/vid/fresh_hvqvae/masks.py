"""Attention mask construction for Fresh HVQVAE predictor stages."""

import torch


def build_temporal_window_mask(T: int, S: int, temporal_window: int, device: torch.device) -> torch.Tensor:
    """
    Build boolean self-attention mask with causal temporal windowing.

    Token at position i (frame t_i = i // S) can attend to token at position j
    (frame t_j = j // S) iff:
      1. t_j <= t_i  (causal)
      2. t_i - t_j < temporal_window  (within window)

    Args:
        T: number of frames
        S: spatial positions per frame
        temporal_window: how many past frames (inclusive of current) each frame sees.
                         -1 means full causal (window = T).
        device: torch device

    Returns:
        (T*S, T*S) boolean mask. True = allowed to attend.
    """
    if temporal_window == -1:
        temporal_window = T

    frame_indices = torch.arange(T, device=device)
    # Frame-level difference: (T, T) where [i, j] = i - j
    diff = frame_indices.unsqueeze(1) - frame_indices.unsqueeze(0)  # (T, T)
    # Causal + window: can attend if diff >= 0 and diff < window
    frame_mask = (diff >= 0) & (diff < temporal_window)  # (T, T)

    # Expand to token level: each frame has S tokens
    token_mask = frame_mask.repeat_interleave(S, dim=0).repeat_interleave(S, dim=1)
    return token_mask  # (T*S, T*S)


def build_cross_attn_mask_top_to_mid(T: int, device: torch.device) -> torch.Tensor:
    """
    Cross-attention mask: mid queries (T*16) attending to top keys (T*1).
    Each mid token at frame t attends to the single top token at frame t.

    Returns: (T*16, T*1) boolean mask.
    """
    S_child = 16
    S_parent = 1
    mask = torch.zeros(T * S_child, T * S_parent, dtype=torch.bool, device=device)
    for t in range(T):
        mask[t * S_child:(t + 1) * S_child, t:t + 1] = True
    return mask


def build_cross_attn_mask_mid_to_bot(T: int, device: torch.device) -> torch.Tensor:
    """
    Cross-attention mask: bot queries (T*256) attending to mid keys (T*16).
    Bot at spatial (r, c) attends to mid at (r//4, c//4) at same frame.

    Returns: (T*256, T*16) boolean mask.
    """
    S_child = 256
    S_parent = 16

    # For each bot spatial position, compute parent index
    r = torch.arange(16, device=device)
    c = torch.arange(16, device=device)
    grid_r, grid_c = torch.meshgrid(r, c, indexing='ij')  # (16, 16) each
    parent_r = grid_r // 4  # (16, 16)
    parent_c = grid_c // 4
    parent_idx = parent_r * 4 + parent_c  # (16, 16) — flat index into 4x4 parent grid
    parent_idx_flat = parent_idx.reshape(S_child)  # (256,)

    # Build per-frame mask: (256, 16)
    frame_mask = torch.zeros(S_child, S_parent, dtype=torch.bool, device=device)
    frame_mask[torch.arange(S_child, device=device), parent_idx_flat] = True

    # Expand to full sequence: block-diagonal in time
    full_mask = torch.zeros(T * S_child, T * S_parent, dtype=torch.bool, device=device)
    for t in range(T):
        full_mask[t * S_child:(t + 1) * S_child, t * S_parent:(t + 1) * S_parent] = frame_mask

    return full_mask
