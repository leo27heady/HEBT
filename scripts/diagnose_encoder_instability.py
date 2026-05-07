"""
Quick diagnostic: measure how many VQ code indices flip between training steps
when the encoder is trainable. Tests different mitigation strategies.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from model.vid.vq_hvebt.config import VQCodebookConfig, VQHVEBTConfig, VQStageConfig
from model.vid.vq_hvebt.hierarchy import VQHVEBTModel


def run_experiment(name: str, detach_pred_context: bool, soft_tau: float, 
                   codebook_lr_scale: float = 1.0, steps: int = 30):
    """Run a short training experiment and report index stability."""
    device = torch.device("cpu")
    torch.manual_seed(42)

    stage_cfg = VQStageConfig(
        clip_stage_name="s3",
        clip_channels=512, H=8, W=8,
        transformer_dim=256, n_heads=4, n_layers=4,
        mcmc_steps=5, mcmc_step_size=100.0,
        soft_target_tau=soft_tau,
        codebook=VQCodebookConfig(num_codes=512, code_dim=512,
                                   init_mode="data_first_batch", commitment_beta=0.25),
        pred_loss_weight=1.0, cb_loss_weight=1.0, commit_loss_weight=0.25,
    )
    cfg = VQHVEBTConfig(
        stages=[stage_cfg],
        train_encoder=True,
        encoder_lr_scale=0.01,
        codebook_lr_scale=codebook_lr_scale,
        weights_path="clip/MobileCLIP2-S0/mobileclip2_s0.pt",
        use_decoder=False,
        detach_pred_context=detach_pred_context,
    )
    model = VQHVEBTModel(cfg).to(device)
    groups = model.parameter_groups(3e-4)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=1e-4)

    video = torch.rand(2, 5, 3, 256, 256, device=device)
    model.maybe_initialize_codebooks(video)
    model.train()

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  detach_pred_context={detach_pred_context}, soft_tau={soft_tau}, cb_lr_scale={codebook_lr_scale}")
    print(f"{'='*70}")

    for step in range(1, steps + 1):
        # Get indices BEFORE step
        with torch.no_grad():
            feats = model.encoder.encode_video(video)
            z_e = feats["s3"]
            B, T1, C, H, W = z_e.shape
            z_flat = z_e.permute(0, 1, 3, 4, 2).reshape(B, T1*H*W, C)
            qout = model.quantizers["s3"].encode(z_flat)
            pre_indices = qout.indices.clone()

        opt.zero_grad()
        out = model.forward_loss(video)
        out.total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        # Get indices AFTER step
        with torch.no_grad():
            feats2 = model.encoder.encode_video(video)
            z_e2 = feats2["s3"]
            z_flat2 = z_e2.permute(0, 1, 3, 4, 2).reshape(B, T1*H*W, C)
            qout2 = model.quantizers["s3"].encode(z_flat2)
            post_indices = qout2.indices

        total_tokens = post_indices.numel()
        flipped = (post_indices != pre_indices).sum().item()
        flip_pct = 100.0 * flipped / total_tokens

        pred_l = next(v for k, v in out.metrics.items() if "loss_pred" in k)
        if step <= 5 or step % 5 == 0:
            print(f"  step {step:>3}  pred={pred_l:.4f}  flipped={flip_pct:.1f}%")


if __name__ == "__main__":
    device = torch.device("cpu")
    torch.manual_seed(42)

    # Build model with all fixes
    stage_cfg = VQStageConfig(
        clip_stage_name="s3",
        clip_channels=512, H=8, W=8,
        transformer_dim=256, n_heads=4, n_layers=4,
        mcmc_steps=5, mcmc_step_size=100.0,
        soft_target_tau=0.0,
        codebook=VQCodebookConfig(num_codes=512, code_dim=512,
                                   init_mode="data_first_batch", commitment_beta=0.25),
        pred_loss_weight=1.0, cb_loss_weight=1.0, commit_loss_weight=0.25,
    )
    cfg = VQHVEBTConfig(
        stages=[stage_cfg],
        train_encoder=True,
        encoder_lr_scale=0.01,
        codebook_lr_scale=0.1,
        weights_path="clip/MobileCLIP2-S0/mobileclip2_s0.pt",
        use_decoder=False,
        detach_pred_context=True,
    )
    model = VQHVEBTModel(cfg).to(device)

    # Two optimizers: AdamW for predictor+codebook, SGD for encoder
    enc_params = model.encoder_params()
    non_enc_params = model.non_encoder_params()
    opt_main = torch.optim.AdamW(
        [{"params": non_enc_params, "lr": 3e-4}],
        betas=(0.9, 0.999), weight_decay=1e-4,
    )
    opt_enc = torch.optim.SGD(enc_params, lr=3e-4 * 0.001, momentum=0.9)

    video = torch.rand(2, 5, 3, 256, 256, device=device)
    model.maybe_initialize_codebooks(video)
    model.train()

    print("SGD encoder + AdamW predictor/codebook, detach_pred_context=True:")
    print(f"{'step':>5} {'pred_loss':>10} {'z_e_shift':>12} {'cb_shift':>12} {'flip%':>8}")

    for step in range(1, 31):
        with torch.no_grad():
            feats = model.encoder.encode_video(video)
            z_e = feats["s3"]
            B, T1, C, H, W = z_e.shape
            z_flat = z_e.permute(0, 1, 3, 4, 2).reshape(B, T1*H*W, C)
            qout = model.quantizers["s3"].encode(z_flat)
            pre_indices = qout.indices.clone()
            pre_z_e = z_flat.clone()
            pre_cb = model.quantizers["s3"].codebook.weight.clone()

        opt_main.zero_grad()
        opt_enc.zero_grad()
        out = model.forward_loss(video)
        out.total_loss.backward()
        # Clip per-group: prevents grad explosion in either sub-network
        nn.utils.clip_grad_norm_(non_enc_params, max_norm=1.0)
        nn.utils.clip_grad_norm_(enc_params, max_norm=1.0)
        opt_main.step()
        opt_enc.step()

        with torch.no_grad():
            feats2 = model.encoder.encode_video(video)
            z_e2 = feats2["s3"]
            z_flat2 = z_e2.permute(0, 1, 3, 4, 2).reshape(B, T1*H*W, C)
            qout2 = model.quantizers["s3"].encode(z_flat2)
            post_indices = qout2.indices
            post_cb = model.quantizers["s3"].codebook.weight.clone()

            z_e_shift = (z_flat2 - pre_z_e).norm(dim=-1).mean().item()
            cb_shift = (post_cb - pre_cb).norm(dim=-1).mean().item()
            flip_pct = 100.0 * (post_indices != pre_indices).float().mean().item()

        pred_l = next(v for k, v in out.metrics.items() if "loss_pred" in k)
        if step <= 5 or step % 5 == 0:
            print(f"{step:>5} {pred_l:>10.4f} {z_e_shift:>12.6f} {cb_shift:>12.6f} {flip_pct:>7.1f}%")
