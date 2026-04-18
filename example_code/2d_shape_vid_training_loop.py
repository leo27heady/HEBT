import os
import sys
from typing import Dict, Any

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # append parent dir

from data.vid.vid_shape_synthetic_dataset import VIDShapeSyntheticDataset
from model.vid.baseline_transformer import Baseline_Transformer_VID
from model.vid.ebt import EBT_VID
from model.model_utils import get_encoded_images

########################################################################################################################

# NOTE IMPORTANT this code is not to reproduce results because it doesnt have all features (LR scheduler, correct wd on all params, etc); it is just a proof of concept for how things work. results are far from exact; it is recommended to use the whole codebase 

########################################################################################################################

class ModelWrapper(pl.LightningModule):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(hparams)

        model_cls = {
            "baseline_transformer": Baseline_Transformer_VID,
            "ebt": EBT_VID,
        }[self.hparams.model_name]

        self.model = model_cls(self.hparams)
        self.dataset = VIDShapeSyntheticDataset(self.hparams,
                                                 size=self.hparams.get("shape_dataset_size", 5000))

        # debug saving config
        self.debug_save_every_n = self.hparams.get("debug_save_every_n", 0)
        self.debug_save_dir = self.hparams.get("debug_save_dir", "logs/debug_pairs")

    def training_step(self, batch, batch_idx):
        metrics = self.model.forward_loss_wrapper(batch, "train")
        loss = metrics["loss"]
        self.log_dict({f"train_{k}": v for k, v in metrics.items()}, prog_bar=True)

        if self.debug_save_every_n > 0 and (batch_idx + 1) % self.debug_save_every_n == 0:
            self._save_debug_pairs(batch, batch_idx)

        return loss

    @torch.no_grad()
    def _save_debug_pairs(self, batch, batch_idx):
        """Save input context frames, GT next frames, and per-frame cosine similarity."""
        step_dir = os.path.join(self.debug_save_dir, f"step_{batch_idx + 1}")
        os.makedirs(step_dir, exist_ok=True)

        B, S, C, H, W = batch.shape
        encoder = self.model.image_encoder
        backbone_type = self.hparams.backbone_type

        # Encode all frames -> (B, S, D)
        flat = batch.reshape(-1, C, H, W)
        all_embeds = get_encoded_images(flat, backbone_type, encoder,
                                        sdxl_vae_standardization=self.hparams.sdxl_vae_standardization)
        all_embeds = all_embeds.reshape(B, S, -1)

        # Input context: frames 0..S-2, GT targets: frames 1..S-1
        input_embeds = all_embeds[:, :-1]  # B, S-1, D
        gt_embeds = all_embeds[:, 1:]       # B, S-1, D

        # Run model forward to get predicted embeddings (last MCMC step)
        if self.hparams.model_name == "ebt":
            down_input = self.model.encoder_down_projection(input_embeds)
            pred_list, _ = self.model(input_embeds, learning=False, no_randomness=True)
            pred_embeds = pred_list[-1]  # B, S-1, D (encoder space)
        else:
            pred_embeds = self.model(input_embeds)

        # Per-frame cosine similarity between predicted and GT
        cos_sim = F.cosine_similarity(pred_embeds, gt_embeds, dim=-1)  # B, S-1

        # Denormalize frames for saving (undo ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, 1, 3, 1, 1)
        frames_denorm = (batch * std + mean).clamp(0, 1)

        # Save first sample in batch
        b = 0
        ctx_frames = frames_denorm[b, :-1]  # S-1, C, H, W  (input context)
        gt_frames = frames_denorm[b, 1:]    # S-1, C, H, W  (ground truth next)
        sims = cos_sim[b]                    # S-1

        # Save context grid and GT grid side by side
        save_image(ctx_frames, os.path.join(step_dir, "context_frames.png"),
                   nrow=min(8, S - 1), padding=2)
        save_image(gt_frames, os.path.join(step_dir, "gt_next_frames.png"),
                   nrow=min(8, S - 1), padding=2)

        # Save similarity scores
        with open(os.path.join(step_dir, "cosine_similarity.txt"), "w") as f:
            f.write("frame_idx | cos_sim(predicted, gt)\n")
            f.write("-" * 35 + "\n")
            for i, s in enumerate(sims):
                f.write(f"  {i:3d}     | {s.item():.6f}\n")
            f.write(f"\n  mean    | {sims.mean().item():.6f}\n")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        workers = max(torch.cuda.device_count(), 1) * self.hparams.num_workers_per_gpu
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size_per_device,
            shuffle=True,
            num_workers=workers,
            persistent_workers=(workers > 0),
            collate_fn=None,
        )

def main():
    hparams = dict(
        # optimisation
        lr=1e-3,
        batch_size_per_device=2,
        num_workers_per_gpu=2,  # workers OK now — data is pre-rendered to disk (set 0 if faster on your machine)
        max_steps=100000,
        # data
        dataset_name="vid_shape_synthetic",
        context_length=16,
        image_dims=[224, 224],
        shape_dataset_size=5000,  # pre-rendered samples (~3min to generate, ~11GB)
        # shape generation config
        shape_scene_type="DIM_2",             # "DIM_2" or "DIM_3"
        shape_angle_min=5,
        shape_angle_max=20,
        shape_temporal_patterns=[],           # e.g. ["acceleration", "oscillation"]
        shape_pattern_combining=False,
        shape_accel_min=3,
        shape_accel_max=6,
        shape_oscillation_period_min=1,
        shape_oscillation_period_max=4,
        shape_interruption_period_min=1,
        shape_interruption_period_max=4,
        shape_min_cubes=2,
        shape_max_cubes=6,
        # model choice
        model_name="ebt",  # "baseline_transformer" or "ebt"
        # backbone
        backbone_type="dinov2",
        vit_backbone_size="small",
        # model size
        embedding_dim=384,
        num_transformer_blocks=6,
        multiheaded_attention_heads=6,
        ffn_dim_multiplier=1,
        weight_initialization_method="xavier",
        weight_initialization_gain=1.0,
        # misc
        execution_mode="pretrain",
        debug_unused_parameters=False,
        # debug visualization
        debug_save_every_n=50,    # save debug pairs every N steps (0 = disabled)
        debug_save_dir="logs/debug_pairs",
    )

    ebt_params = dict( #NOTE 
        mcmc_step_size=60.0,
        mcmc_step_size_lr_multiplier=1500.0, 
        mcmc_num_steps=2,
        ebt_type="time_embed",
        normalize_initial_condition=False,
        denoising_initial_condition="random_noise",
        mcmc_step_size_learnable=True,
        no_mcmc_detach=False,
        # only up to these first ones are actually used, and really only the first four of them are real hparams (the others can stay as they are so normalize, learnable, and condition dont need to be tuned). time_embed can also almost always stay and mcmc_num_steps = 2 is very safe. keeping mcmc_step_size_lr_multiplier = 3x mcmc_step_size is safe and what works well so the most important and arguably only really neccesary to tune hparam is mcmc_step_size

        # below are just to make existing code run well and happy :) they are not used at all. you can try them out if you ever want to add fancier hparams but are not recommended for getting started
        ebt_norm="rms",
        ebt_act_func="silu",
        dyt_alpha_init=0.5,
        mcmc_replay_buffer=False,
        gaussian_random_noise_scaling=1.0,
        normalize_initial_condition_only_first_step=False,
        randomize_mcmc_step_size_scale=1.0,
        randomize_mcmc_num_steps=0,
        randomize_mcmc_num_steps_min=0,
        randomize_mcmc_num_steps_final_landscape=False,
        langevin_dynamics_noise=0.0,
        langevin_dynamics_noise_learnable=False,
        vocab_to_embed_uses_prob_dist=False,
        num_modality_processing_mlp_layers=1,
        truncate_mcmc=False,
        clamp_futures_grad=False,
        clamp_futures_grad_max_change=9.0,
        absolute_clamp=0.0,
        clamp_max_after_warm_up=0.0,
        sharpen_predicted_distribution=0.0,
        reconstruction_coeff=1.0,
        contrastive_loss=False,
        contrastive_loss_coeff=0.0005,
        soften_target_prob_dist=0.0,
        
        energy_loss_fn="MSE",
        sdxl_vae_standardization=False,
        energy_loss_coeff=0.0,
        energy_loss_hinge=0.0,
        out_of_bounds_loss_coeff=0.0,
        embeddings_distance_fn="cosine",
        scale_cosine_sim_decay=7,
    )
    hparams.update(ebt_params)

    model = ModelWrapper(hparams)
    logger = WandbLogger(
        name="minimal_wrapper_run", project="vid_pretrain_minimal", entity=""
    )

    trainer = pl.Trainer(
        max_steps=hparams["max_steps"],
        # devices=-1,
        logger=logger,
        max_epochs=1,
        enable_model_summary=True,
        enable_checkpointing=False,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()