import hydra
import torch
import wandb
import logging
import lightning as pl

from pathlib import Path
from omegaconf import DictConfig
from tempfile import TemporaryDirectory
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from afa_rl.datasets import DataModuleFromDatasets
from common.config_classes import TrainXGBoostClassifierConfig
from common.utils import (
    get_class_probabilities,
    load_dataset_artifact,
    set_seed,
)
from common.xgboost_classifier import (
    LitXGBoostAFAClassifier,
    WrappedXGBoostAFAClassifier,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/classifiers/xgboost_classifier",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Train XGBoost classifier."""
    config = TrainXGBoostClassifierConfig(**cfg)

    set_seed(config.seed)

    # Initialize wandb
    run = wandb.init(
        group="train_xgboost_classifier",
        job_type="train_classifier",
        config=cfg,
    )

    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    # Load dataset
    train_dataset, val_dataset, test_dataset, dataset_metadata = load_dataset_artifact(
        config.dataset_artifact_name
    )

    log.info(f"Loaded dataset: {dataset_metadata}")
    log.info(f"Training samples: {len(train_dataset)}")
    log.info(f"Validation samples: {len(val_dataset)}")
    log.info(f"Test samples: {len(test_dataset)}")

    # Dataset info
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.info(f"Class probabilities in training set: {
             train_class_probabilities}")

    # Create data module
    datamodule = DataModuleFromDatasets(
        train_dataset,
        val_dataset,
        batch_size=config.batch_size,
        num_workers=0,  # XGBoost doesn't need multiple workers
    )

    # Create XGBoost model
    xgb_params = {
        'n_estimators': config.n_estimators,
        'max_depth': config.max_depth,
        'learning_rate': config.learning_rate,
        'subsample': config.subsample,
        'colsample_bytree': config.colsample_bytree,
        'random_state': config.seed,
    }

    lit_model = LitXGBoostAFAClassifier(
        n_features=n_features,
        n_classes=n_classes,
        min_masking_probability=config.min_masking_probability,
        max_masking_probability=config.max_masking_probability,
        **xgb_params
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-checkpoint",
        every_n_epochs=1,
    )

    # Setup training (Lightning handles the fit/eval cycle)
    logger = WandbLogger()

    # Since XGBoost doesn't use gradient-based training, we just need 1 epoch
    # The actual training happens in setup()
    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        accelerator="cpu",  # XGBoost runs on CPU
        devices=1,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )

    try:
        # Fit the model (training happens in setup)
        trainer.fit(lit_model, datamodule)

        # Evaluate final performance if requested
        if config.evaluate_final_performance:
            log.info("Evaluating final performance...")
            trainer.test(lit_model, datamodule)

    except KeyboardInterrupt:
        log.info("Training interrupted by user")

    finally:
        # Wrap the classifier for the pipeline
        wrapped_classifier = WrappedXGBoostAFAClassifier(
            lit_model=lit_model,
            device=torch.device("cpu")
        )

        # Save as artifact
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wrapped_classifier.save(tmp_path / "classifier.pt")

            # Create wandb artifact
            artifact = wandb.Artifact(
                name=f"xgboost_classifier-{dataset_metadata['dataset_type']}_split_{
                    config.dataset_artifact_name.split('_split_')[1].split(':')[0]}",
                type="trained_classifier",
                metadata={
                    "classifier_class_name": "WrappedXGBoostAFAClassifier",
                    "dataset_artifact_name": config.dataset_artifact_name,
                    "dataset_type": dataset_metadata["dataset_type"],
                    "n_features": n_features,
                    "n_classes": n_classes,
                    "seed": config.seed,
                    "xgb_params": xgb_params,
                },
            )

            artifact.add_file(str(tmp_path / "classifier.pt"))
            run.log_artifact(artifact, aliases=config.output_artifact_aliases)

        log.info("Training completed successfully!")

    run.finish()


if __name__ == "__main__":
    main()
