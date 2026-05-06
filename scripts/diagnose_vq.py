"""
VQ-HVEBT diagnostic harness.

Exercises the VQ training pipeline across modes and reports:
  - Per-stage gradient norms (energy params vs codebook lookup vs encoder)
  - CE loss / top-1 acc / entropy trajectory
  - Decoder-only (bottom_up + decoder) gradient flow audit
  - Adaptive-CLIP plausibility check (train_encoder + on-the-fly targets)

This is a SYNTHETIC test (random data, tiny model). It is *not* a benchmark;
the goal is to verify that each mode produces non-zero, well-shaped gradients
and that CE / entropy actually move when they are supposed to move.
"""
from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vid.hvebt import (  # noqa: E402
    HierarchicalHVEBT,
    HierarchicalHVEBTConfig,
    HVEBTStageConfig,
)


# ---------- helpers ---------- #

def grad_norm(params) -> float:
    s = 0.0
    n = 0
    for p in params:
        if p.grad is None:
            continue
        s += float((p.grad.detach() ** 2).sum().item())
        n += 1
    return math.sqrt(s) if n else 0.0


def stage_param_groups(model: HierarchicalHVEBT):
    out = []
    for i, st in enumerate(model.stages):
        groups = {
            "energy": [p for n, p in st.named_parameters() if "input_proj" in n or "blocks" in n or "norm_out" in n or "energy_head" in n],
            "parent_proj": [p for n, p in st.named_parameters() if n.startswith("parent_proj") or n.startswith("parent_norm")],
        }
        out.append(groups)
    return out


def build_stages(K_per: list[int], C_per: list[int], HW_per: list[tuple[int, int]], embed=32):
    cfgs = []
    for name, K, C, (H, W) in zip(["s2", "s3", "pooled"], K_per, C_per, HW_per):
        cfgs.append(HVEBTStageConfig(
            clip_stage_name=name, clip_channels=C, H=H, W=W,
            embed_dim=embed, n_heads=2, n_layers=1,
            vq_codebook_size=K,
        ))
    return cfgs


def build_model(
    *,
    bottom_up: bool,
    decoder: bool,
    train_encoder: bool = False,
    no_features: bool = True,
    use_precomputed: bool = True,
    target_recompute_every: int = 0,
    truncate_mcmc: bool = False,
    mcmc_steps: int = 2,
    step_size: float = 1000.0,
):
    K_per = [16, 8, 4]
    C_per = [16, 24, 32]
    HW_per = [(8, 8), (4, 4), (2, 2)]
    stages = build_stages(K_per, C_per, HW_per)
    cfg = HierarchicalHVEBTConfig(
        stages=stages,
        mcmc_num_steps=mcmc_steps,
        mcmc_step_size=step_size,
        truncate_mcmc=truncate_mcmc,
        weights_path="",  # no encoder when no_features
        bottom_up_loss=bottom_up,
        detach_kv=not bottom_up,
        decoder_enabled=decoder,
        decoder_out_size=32,
        train_encoder=train_encoder,
        vq_mode=True,
        vq_use_precomputed_targets=use_precomputed,
        vq_no_features=no_features and use_precomputed,
        vq_target_recompute_every=target_recompute_every,
        allow_stale_targets=True,
    )
    return HierarchicalHVEBT(cfg), K_per, C_per, HW_per


def make_batch(B, Tp1, K_per, C_per, HW_per, with_features: bool):
    targets = {}
    feats = {}
    for name, K, C, (H, W) in zip(["s2", "s3", "pooled"], K_per, C_per, HW_per):
        targets[name] = torch.randint(0, K, (B, Tp1, H, W))
        if with_features:
            feats[name] = torch.randn(B, Tp1, C, H, W)
    video = torch.rand(B, Tp1, 3, 32, 32)
    return video, targets, (feats if with_features else None)


# ---------- diagnostics ---------- #

def _run_mode(
    label,
    *,
    bottom_up,
    decoder,
    use_precomputed=True,
    no_features=True,
    train_encoder=False,
    target_recompute_every=0,
    n_steps=30,
    seed=0,
    print_every=5,
    step_size=1000.0,
):
    torch.manual_seed(seed)
    model, K_per, C_per, HW_per = build_model(
        bottom_up=bottom_up,
        decoder=decoder,
        train_encoder=train_encoder,
        no_features=no_features,
        use_precomputed=use_precomputed,
        target_recompute_every=target_recompute_every,
        step_size=step_size,
    )
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2,
    )

    print(f"\n========== {label} ==========")
    print(f"  bottom_up={bottom_up} decoder={decoder} use_precomputed={use_precomputed} "
          f"no_features={no_features} train_encoder={train_encoder}")

    # Fixed batch — overfit sanity (loss should fall)
    B, Tp1 = 2, 3
    video, targets, feats = make_batch(
        B, Tp1, K_per, C_per, HW_per, with_features=not no_features,
    )

    history = []
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        out = model.forward_loss(
            video,
            features=feats,
            vq_targets=targets if use_precomputed else None,
            learning=True,
        )
        loss = out["loss_total"]
        loss.backward()

        # Inspect grads BEFORE optimizer step
        per_stage_norms = [
            grad_norm([p for p in st.parameters() if p.requires_grad])
            for st in model.stages
        ]
        decoder_norm = (grad_norm(model.decoder.parameters())
                        if model.decoder is not None else 0.0)

        # Energy-only grad (parent_proj excluded)
        energy_norms = []
        for st in model.stages:
            ps = [p for n, p in st.named_parameters()
                  if any(k in n for k in ("input_proj", "blocks", "norm_out", "energy_head"))]
            energy_norms.append(grad_norm(ps))

        opt.step()

        # Metrics
        ps = out["per_stage"]
        snap = {
            "step": step,
            "L": float(loss.item()),
            "Le": float(out["loss_energy"].item()),
            "Ld": float(out.get("loss_decoder", torch.tensor(0.0)).item()),
            "stages": [],
            "stage_grad_norms": per_stage_norms,
            "energy_grad_norms": energy_norms,
            "decoder_grad_norm": decoder_norm,
        }
        for i, s in enumerate(ps):
            if s is None:
                snap["stages"].append(None); continue
            snap["stages"].append({
                "ce": float(s.get("ce_loss", torch.tensor(0.0)).item()),
                "H": float(s.get("entropy_mean", torch.tensor(0.0)).item()),
                "acc": float(s.get("top1_accuracy", torch.tensor(0.0)).item()),
                "cb": float(s.get("codebook_usage", torch.tensor(0.0)).item()),
                "Eg": float(s["energy_gap"].item()),
            })
        history.append(snap)

        if step % print_every == 0 or step == n_steps - 1:
            line = f"[{step:3d}] L={snap['L']:.3f} Le={snap['Le']:.3f} Ld={snap['Ld']:.3f}"
            for i, ss in enumerate(snap["stages"]):
                if ss is None: continue
                line += (f"  s{i}: ce={ss['ce']:.3f} H={ss['H']:.2f} "
                         f"acc={ss['acc']*100:.0f}% cb={ss['cb']*100:.0f}% "
                         f"|gE|={energy_norms[i]:.2e}")
            line += f"  |gDec|={decoder_norm:.2e}"
            print(line)

    # Trajectory summary: did entropy move? CE drop? acc rise?
    first = history[0]["stages"]
    last = history[-1]["stages"]
    print("  --- summary ---")
    for i, (a, b) in enumerate(zip(first, last)):
        if a is None or b is None: continue
        dh = b["H"] - a["H"]
        dce = b["ce"] - a["ce"]
        dacc = b["acc"] - a["acc"]
        print(f"  s{i}: dH={dh:+.2f}  dCE={dce:+.3f}  dacc={dacc:+.2f}  "
              f"final acc={b['acc']*100:.0f}%  final cb={b['cb']*100:.0f}%")
    # Is anyone training?
    avg_e = sum(history[-1]["energy_grad_norms"]) / max(len(history[-1]["energy_grad_norms"]), 1)
    print(f"  avg energy-grad-norm at last step: {avg_e:.2e}")
    return history


def run_all():
    # -------- Mode A: standard VQ + per-stage CE (no bottom_up) -------- #
    run_mode("A: VQ + per-stage CE (no bottom_up, no decoder), step=1000",
             bottom_up=False, decoder=False)

    # -------- Mode B: VQ + decoder, no bottom_up (CE + decoder both) -- #
    run_mode("B-large-step: VQ + decoder, step=1000 (expect blowup)",
             bottom_up=False, decoder=True)
    run_mode("B-small-step: VQ + decoder, step=10",
             bottom_up=False, decoder=True, step_size=10.0)

    # -------- Mode C: VQ + bottom_up + decoder (USER'S ACTUAL CASE) --- #
    run_mode("C-large-step: VQ + bottom_up + decoder, step=1000 (after fix)",
             bottom_up=True, decoder=True)
    run_mode("C-small-step: VQ + bottom_up + decoder, step=10 (recommended)",
             bottom_up=True, decoder=True, step_size=10.0)

    # -------- Mode D: VQ + bottom_up, NO decoder ---------------------- #
    run_mode("D: VQ + bottom_up, no decoder, step=10",
             bottom_up=True, decoder=False, step_size=10.0)

    # -------- Mode E: on-the-fly targets (sim of train_encoder) ------- #
    run_mode("E: features given (on-the-fly targets), step=10",
             bottom_up=False, decoder=False,
             use_precomputed=False, no_features=False, train_encoder=False,
             step_size=10.0)


def run_mode(*args, **kwargs):
    if "step_size" not in kwargs:
        kwargs["step_size"] = 1000.0
    return _run_mode(*args, **kwargs)


if __name__ == "__main__":
    run_all()
