import gc
import wandb
import torch
import hydra
import logging
import lightning as pl

from pathlib import Path
from typing import Any, cast
from omegaconf import OmegaConf
from tempfile import TemporaryDirectory
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from common.config_classes import ACOTrainConfig
from common.models import LitMaskedMLPClassifier
from common.classifiers import WrappedMaskedMLPClassifier
from afa_rl.datasets import DataModuleFromDatasets
from afa_oracle.aco_core import create_aco_oracle
from common.utils import get_class_probabilities, load_dataset_artifact, set_seed
from afa_oracle.afa_methods import (
    ACOAFAMethod,
    ACOBCAFAMethod,
    train_behavioral_cloning_policy,
)
from eval.metrics import eval_afa_method
from eval.utils import plot_metrics

log = logging.getLogger(__name__)


def train_or_load_classifier(
    config: ACOTrainConfig,
    train_dataset,
    val_dataset,
    n_features: int,
    n_classes: int,
    train_class_probabilities: torch.Tensor,
    device: torch.device,
) -> WrappedMaskedMLPClassifier:
    """
    Train a new classifier or load existing one.
    """

    if not config.train_classifier and config.classifier_artifact_name:
        # Load existing classifier
        log.info(f"Loading classifier from {config.classifier_artifact_name}")
        classifier_artifact = wandb.use_artifact(
            config.classifier_artifact_name, type="trained_classifier"
        )
        classifier_dir = Path(classifier_artifact.download())

        # Load the classifier
        return WrappedMaskedMLPClassifier.load(classifier_dir / "classifier.pt", device)
    else:
        # Train new classifier
        log.info("Training new masked MLP classifier...")

        # Create data module
        datamodule = DataModuleFromDatasets(
            train_dataset, val_dataset, batch_size=config.classifier_batch_size
        )

        # Create model
        lit_model = LitMaskedMLPClassifier(
            n_features=n_features,
            n_classes=n_classes,
            num_cells=tuple(config.classifier_num_cells),
            dropout=config.classifier_dropout,
            class_probabilities=train_class_probabilities,
            min_masking_probability=0.0,
            max_masking_probability=0.9,
            lr=config.classifier_lr,
        )

        # Training setup
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss_many_observations",
            save_top_k=1,
            mode="min",
        )

        logger = WandbLogger()
        trainer = pl.Trainer(
            max_epochs=config.classifier_epochs,
            logger=logger,
            accelerator=device.type,
            devices=1,
            callbacks=[checkpoint_callback],
        )

        # Train
        trainer.fit(lit_model, datamodule)

        # Load best model and wrap
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        best_lit_model = LitMaskedMLPClassifier.load_from_checkpoint(
            best_checkpoint,
            n_features=n_features,
            n_classes=n_classes,
            num_cells=config.classifier_num_cells,
            dropout=config.classifier_dropout,
            class_probabilities=train_class_probabilities,
            max_masking_probability=0.9,
            lr=config.classifier_lr,
        )

        wrapped_classifier = WrappedMaskedMLPClassifier(
            module=best_lit_model.classifier, device=device
        )

        log.info("Classifier training completed")
        return wrapped_classifier


@hydra.main(version_base=None, config_path="../../conf/train/aco", config_name="config")
def main(cfg: ACOTrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    run = wandb.init(
        group="train_aco",
        job_type="training",
        config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
    )

    # Load dataset
    train_dataset, val_dataset, _, dataset_metadata = load_dataset_artifact(
        cfg.dataset_artifact_name
    )

    # Get dataset info
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]
    train_class_probabilities = get_class_probabilities(train_dataset.labels)

    log.info(f"Dataset: {dataset_metadata['dataset_type']}")
    log.info(f"Features: {n_features}, Classes: {n_classes}")
    log.info(f"Training samples: {len(train_dataset)}")

    # Train or load classifier
    classifier = train_or_load_classifier(
        cfg,
        train_dataset,
        val_dataset,
        n_features,
        n_classes,
        train_class_probabilities,
        device,
    )

    log.info("Creating ACO oracle...")

    # Create ACO oracle
    aco_oracle = create_aco_oracle(
        k_neighbors=cfg.aco.k_neighbors,
        acquisition_cost=cfg.aco.acquisition_cost,
        exhaustive_threshold=cfg.aco.exhaustive_search_threshold,
        subset_search_size=cfg.aco.subset_search_size,
        max_subset_size=cfg.aco.max_subset_size,
        hide_val=cfg.aco.hide_val,
        distance_metric=cfg.aco.distance_metric,
        standardize_features=cfg.aco.standardize_features,
    )

    # Fit oracle on training data
    log.info("Fitting ACO oracle on training data...")
    aco_oracle.fit(train_dataset.features, train_dataset.labels.argmax(dim=-1))

    # Create ACO method
    aco_method = ACOAFAMethod(
        aco_oracle=aco_oracle, afa_classifier=classifier, _device=device
    )

    log.info("ACO method created and fitted")

    # Optionally train behavioral cloning version
    aco_bc_method = None
    if cfg.aco_bc is not None:
        log.info("Training behavioral cloning version...")

        bc_policy = train_behavioral_cloning_policy(
            aco_method=aco_method,
            dataset=train_dataset,
            n_features=n_features,
            bc_config=cfg.aco_bc,
            device=device,
        )

        aco_bc_method = ACOBCAFAMethod(
            policy_network=bc_policy, afa_classifier=classifier, _device=device
        )

        log.info("Behavioral cloning training completed")

    # Evaluate if requested
    if cfg.aco.evaluate_final_performance:
        log.info("Evaluating ACO method...")

        metrics = eval_afa_method(
            afa_select_fn=aco_method.select,
            dataset=val_dataset,
            budget=n_features,
            afa_predict_fn=aco_method.predict,
            only_n_samples=cfg.aco.eval_only_n_samples,
        )

        fig = plot_metrics(metrics)
        run.log({"aco_metrics_plot": fig})

        # Also evaluate BC version if available
        if aco_bc_method is not None:
            log.info("Evaluating ACO+BC method...")

            bc_metrics = eval_afa_method(
                afa_select_fn=aco_bc_method.select,
                dataset=val_dataset,
                budget=n_features,
                afa_predict_fn=aco_bc_method.predict,
                only_n_samples=cfg.aco.eval_only_n_samples,
            )

            bc_fig = plot_metrics(bc_metrics)
            run.log({"aco_bc_metrics_plot": bc_fig})

    try:
        # Save ACO method as artifact
        with TemporaryDirectory(delete=False) as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save main ACO method
            aco_method.save(tmp_path)

            aco_artifact = wandb.Artifact(
                name=f"train_aco-{dataset_metadata['dataset_type']}-seed_{cfg.seed}",
                type="trained_method",
                metadata={
                    "method_type": "aco",
                    "dataset_artifact_name": cfg.dataset_artifact_name,
                    "dataset_type": dataset_metadata["dataset_type"],
                    "budget": None,  # ACO doesn't have a fixed budget
                    "seed": cfg.seed,
                    "k_neighbors": cfg.aco.k_neighbors,
                    "acquisition_cost": cfg.aco.acquisition_cost,
                },
            )

            aco_artifact.add_dir(str(tmp_path))
            run.log_artifact(aco_artifact, aliases=cfg.output_artifact_aliases)

            log.info(f"ACO method saved as artifact")

        # Save BC method if available
        if aco_bc_method is not None:
            with TemporaryDirectory(delete=False) as tmp_dir_bc:
                tmp_path_bc = Path(tmp_dir_bc)

                aco_bc_method.save(tmp_path_bc)

                aco_bc_artifact = wandb.Artifact(
                    name=f"train_aco_bc-{dataset_metadata['dataset_type']}-seed_{
                        cfg.seed
                    }",
                    type="trained_method",
                    metadata={
                        "method_type": "aco_bc",
                        "dataset_artifact_name": cfg.dataset_artifact_name,
                        "dataset_type": dataset_metadata["dataset_type"],
                        "budget": None,
                        "seed": cfg.seed,
                        "k_neighbors": cfg.aco.k_neighbors,
                        "acquisition_cost": cfg.aco.acquisition_cost,
                        "bc_epochs": cfg.aco_bc.bc_epochs,
                    },
                )

                aco_bc_artifact.add_dir(str(tmp_path_bc))
                run.log_artifact(
                    aco_bc_artifact,
                    aliases=[f"{alias}_bc" for alias in cfg.output_artifact_aliases],
                )

                log.info(f"ACO+BC method saved as artifact")

    except Exception as e:
        log.error(f"Error saving artifacts: {e}")
        raise
    finally:
        run.finish()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
