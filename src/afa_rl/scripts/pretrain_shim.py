from argparse import ArgumentParser

import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger

from afa_rl.datasets import CubeDataset, DataModuleFromDataset
from afa_rl.models import (
    MLPClassifier,
    ReadProcessEncoder,
    ShimEmbedder,
    ShimEmbedderClassifier,
)


def main():
    torch.set_float32_matmul_precision("medium")

    parser = ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of epochs to train the ShimEmbedderClassifier",
    )
    # Does not seem to make a difference...
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        help="Number of workers for the DataLoaders",
        default=1,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Where to save the checkpoint",
    )
    args = parser.parse_args()

    # TODO: use 20 features
    # n_features = 20
    n_features = 11
    dataset = CubeDataset(
        n_features=n_features, data_points=100_000, sigma=0.01, seed=42
    )
    datamodule = DataModuleFromDataset(
        dataset=dataset, batch_size=128, train_ratio=0.8, num_workers=args.num_workers
    )

    encoder = ReadProcessEncoder(
        feature_size=n_features + 1,  # state contains one value and one index
        output_size=16,
        reading_block_cells=[32, 32],
        writing_block_cells=[32, 32],
        memory_size=16,
        processing_steps=5,
    )
    embedder = ShimEmbedder(encoder)
    classifier = MLPClassifier(16, 8, [32, 32])
    model = ShimEmbedderClassifier(embedder=embedder, classifier=classifier, lr=1e-4)

    logger = WandbLogger(project="pretrain-shim-embedder-classifier", save_dir="logs")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,  # Use only 1 GPU
        strategy="ddp"
    )
    trainer.fit(model, datamodule)

    # Move the best checkpoint to the desired location
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    torch.save(torch.load(best_checkpoint), args.checkpoint)


if __name__ == "__main__":
    main()
