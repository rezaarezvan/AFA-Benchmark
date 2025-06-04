import logging
from tempfile import NamedTemporaryFile
import hydra

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
from common.afa_methods import RandomClassificationAFAMethod
from common.classifiers import WrappedMaskedMLPClassifier
from common.config_classes import TrainMaskedMLPClassifierConfig
from afa_rl.datasets import DataModuleFromDatasets
from common.models import LitMaskedMLPClassifier
from common.utils import (
    get_class_probabilities,
    load_dataset_artifact,
    set_seed,
)
from pathlib import Path
from omegaconf import OmegaConf

from eval.metrics import eval_afa_method
from eval.utils import plot_metrics

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../../conf/classifiers/masked_mlp_classifier",
    config_name="tmp",
)
def main(cfg: TrainMaskedMLPClassifierConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        group="train_masked_mlp_classifier",
        job_type="train_classifier",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
    )

    # Load dataset artifact
    train_dataset, val_dataset, _, dataset_metadata = load_dataset_artifact(
        cfg.dataset_artifact_name
    )
    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=cfg.batch_size
    )

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(f"Class probabilities in training set: {train_class_probabilities}")
    lit_model = LitMaskedMLPClassifier(
        n_features=n_features,
        n_classes=n_classes,
        num_cells=tuple(cfg.num_cells),
        dropout=cfg.dropout,
        class_probabilities=train_class_probabilities,
        max_masking_probability=cfg.max_masking_probability,
        lr=cfg.lr,
    )

    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_full",
        save_top_k=1,
        mode="min",
    )

    logger = WandbLogger()
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        accelerator=cfg.device,
        devices=1,  # Use only 1 GPU
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )

    try:
        trainer.fit(lit_model, datamodule, ckpt_path=None)
    except KeyboardInterrupt:
        pass
    finally:
        # Convert lightning model to a classifier that implements the AFAClassifier interface
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        best_lit_model = LitMaskedMLPClassifier.load_from_checkpoint(
            best_checkpoint,
            n_features=n_features,
            n_classes=n_classes,
            num_cells=cfg.num_cells,
            dropout=cfg.dropout,
            class_probabilities=train_class_probabilities,
            max_masking_probability=cfg.max_masking_probability,
            lr=cfg.lr,
        )
        best_model = best_lit_model.classifier
        wrapped_classifier = WrappedMaskedMLPClassifier(
            module=best_model, device=torch.device(cfg.device)
        )
        if cfg.evaluate_final_performance:
            # Evaluate an AFA method that randomly selects features and uses this classifier
            afa_method = RandomClassificationAFAMethod(
                afa_classifier=wrapped_classifier, device=torch.device("cpu")
            )
            metrics = eval_afa_method(
                afa_select_fn=afa_method.select,
                dataset=val_dataset,
                budget=n_features,
                afa_predict_fn=afa_method.predict,
            )
            fig = plot_metrics(metrics)
            run.log({"metrics_plot": fig})
        # Save model in a temporary file
        with NamedTemporaryFile(delete=False) as tmp_file:
            save_path = Path(tmp_file.name)
            wrapped_classifier.save(save_path)
        # Save classifier as wandb artifact
        trained_classifier_artifact = wandb.Artifact(
            name=f"masked_mlp_classifier-{cfg.dataset_artifact_name.split(':')[0]}",
            type="trained_classifier",
            metadata={
                "dataset_type": dataset_metadata["dataset_type"],
                "seed": cfg.seed,
                "classifier_class_name": wrapped_classifier.__class__.__name__,
                "classifier_type": "MaskedMLPClassifier",
            },
        )
        trained_classifier_artifact.add_file(str(save_path), name="classifier.pt")
        run.log_artifact(
            trained_classifier_artifact, aliases=cfg.output_artifact_aliases
        )
        run.finish()


if __name__ == "__main__":
    main()
