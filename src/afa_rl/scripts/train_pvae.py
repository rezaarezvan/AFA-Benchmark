import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms

import wandb
from afa_rl.datasets import DataModuleFromDatasets, MNISTDataModule
from afa_rl.models import (
    PartialVAE,
    PointNet1D,
    PointNet2D,
)
from common.datasets import CubeDataset


def main1D():
    torch.set_float32_matmul_precision("medium")

    dataset = CubeDataset(
        n_features=20,
        data_points=10000,
        sigma=0.01,
        seed=42,
    )
    dataset.generate_data()
    datamodule = DataModuleFromDatasets(
        dataset=dataset, batch_size=128, train_ratio=0.8, num_workers=1
    )

    model = PartialVAE(
        encoder=PointNet1D(
            element_encoder=nn.Sequential(
                nn.Linear(21, 50),
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.ReLU(),
                nn.Linear(50, 50),
            ),
        ),
        mu_net=nn.Sequential(
            nn.Linear(50, 30),
        ),
        logvar_net=nn.Sequential(
            nn.Linear(50, 30),
        ),
        decoder=nn.Sequential(
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
        ),
    )
    logger = WandbLogger(project="pvae", save_dir="logs/afa_rl")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=10,
        accelerator="auto",
        # callbacks=[ImageLoggerCallback()]
    )

    wandb.init(entity="valterschutz-chalmers-university-of-technology", project="pvae")

    trainer.fit(model, datamodule=datamodule)


def main2D():
    torch.set_float32_matmul_precision("medium")

    datamodule = MNISTDataModule(
        batch_size=128,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
        ),
    )

    model = PartialVAE(
        encoder=PointNet2D(
            element_encoder=nn.Sequential(
                nn.Linear(3, 50),
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.ReLU(),
                nn.Linear(50, 50),
            ),
            image_shape=(28, 28),
        ),
        mu_net=nn.Sequential(
            nn.Linear(50, 30),
        ),
        logvar_net=nn.Sequential(
            nn.Linear(50, 30),
        ),
        decoder=nn.Sequential(
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 784),
        ),
    )
    logger = WandbLogger(project="pvae", save_dir="logs/afa_rl")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=10,
        accelerator="auto",
        # callbacks=[ImageLoggerCallback()]
    )

    wandb.init(entity="valterschutz-chalmers-university-of-technology", project="pvae")

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main2D()
