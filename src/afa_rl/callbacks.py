import lightning as pl
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

import wandb


class ImageLoggerCallback(pl.Callback):
    def __init__(self, num_images=8):
        self.num_images = num_images

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(trainer.datamodule.val_dataloader()))
        x, _ = val_batch
        x = x.to(pl_module.device)
        mask = (torch.rand_like(x) > 0.6).float()
        x_masked = x * mask
        x_hat, _, _ = pl_module(x, mask)

        imgs = []
        for i in range(self.num_images):
            original = x[i]
            masked = x_masked[i]
            reconstructed = x_hat[i]

            # Stack as a single 3-row image (C, H, 3*W)
            trio = torch.cat([original, masked, reconstructed], dim=-1)
            imgs.append(wandb.Image(trio, caption=f"Sample {i}"))

        trainer.logger.experiment.log(
            {"imputations": imgs, "global_step": trainer.global_step}
        )
