import logging
from pathlib import Path

import hydra
import lightning as pl
import matplotlib.pyplot as plt
import torch
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from afabench.afa_rl.datasets import DataModuleFromDatasets
from afabench.common.afa_methods import RandomClassificationAFAMethod
from afabench.common.classifiers import WrappedMaskedMLPClassifier
from afabench.common.config_classes import TrainMaskedMLPClassifierConfig
from afabench.common.models import LitMaskedMLPClassifier
from afabench.common.utils import (
    get_class_frequencies,
    load_dataset_splits,
    set_seed,
)
from afabench.eval.eval import eval_afa_method
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.eval.utils import plot_metrics

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/classifiers/masked_mlp_classifier",
    config_name="config",
)
def main(cfg: TrainMaskedMLPClassifierConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    dataset_dir = Path(to_absolute_path(cfg.dataset_path))
    log.info(f"Loading dataset from: {dataset_dir}")

    # Load dataset
    train_dataset, val_dataset, _, dataset_metadata = load_dataset_splits(
        dataset_dir
    )
    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=cfg.batch_size
    )

    n_features = train_dataset.feature_shape[0]
    n_classes = train_dataset.label_shape[0]

    _, train_labels = train_dataset.get_all_data()
    train_class_probabilities = get_class_frequencies(train_labels)
    log.debug(f"Class probabilities: {train_class_probabilities}")

    lit_model = LitMaskedMLPClassifier(
        n_features=n_features,
        n_classes=n_classes,
        num_cells=tuple(cfg.num_cells),
        dropout=cfg.dropout,
        class_probabilities=train_class_probabilities,
        min_masking_probability=cfg.min_masking_probability,
        max_masking_probability=cfg.max_masking_probability,
        lr=cfg.lr,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_many_observations",
        save_top_k=1,
        mode="min",
    )

    log_dir = Path(cfg.save_path) / "logs"
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(
        save_dir=str(log_dir),
        name="masked_mlp_classifier",
    )
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        accelerator=cfg.device,
        devices=1,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )

    try:
        trainer.fit(lit_model, datamodule, ckpt_path=None)
    except KeyboardInterrupt:
        pass
    finally:
        # Load best checkpoint
        best_checkpoint = checkpoint_callback.best_model_path
        if not best_checkpoint:
            raise RuntimeError("No best checkpoint was saved.")
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
        initializer = get_afa_initializer_from_config(
            cfg.initializer
        )
        initializer.set_seed(cfg.seed)
        unmasker = get_afa_unmasker_from_config(
            cfg.unmasker
        )
        unmasker.set_seed(cfg.seed)

        if cfg.evaluate_final_performance:
            log.info("Evaluating final performance...")
            afa_method = RandomClassificationAFAMethod(
                afa_classifier=wrapped_classifier, device=torch.device("cpu")
            )
            metrics = eval_afa_method(
                afa_select_fn=afa_method.select,
                afa_unmask_fn=unmasker.unmask,
                n_selection_choices=unmasker.get_n_selections(
                    feature_shape=val_dataset.feature_shape
                ),
                afa_initialize_fn=initializer.initialize,
                dataset=val_dataset,
                builtin_afa_predict_fn=afa_method.predict,
                selection_budget=n_features,
                only_n_samples=cfg.eval_only_n_samples,
            )
            csv_path = Path(cfg.save_path) / "eval_data.csv"
            metrics.to_csv(csv_path, index=False)
            log.info(f"Saved evaluation data to CSV at: {csv_path}")
            # TODO test the visualization of saved evaluation results in the pipeline
            # fig = plot_metrics(metrics)
            # plots_dir = Path(cfg.save_path) / "plots"
            # plots_dir.mkdir(parents=True, exist_ok=True)
            # dataset_name = dataset_dir.parent.name
            # dataset_instance = dataset_dir.name
            # classifier_name = f"masked_mlp_classifier-{dataset_name}_instance_{dataset_instance}"
            # fig_path = plots_dir / f"{classifier_name}_metrics.png"
            # fig.savefig(fig_path, bbox_inches="tight")
            # plt.close(fig)
            # log.info(f"Saved metrics plot to: {fig_path}")

        # Save to local filesystem
        dataset_name = dataset_dir.parent.name
        dataset_instance = dataset_dir.name
        classifier_name = (
            f"masked_mlp_classifier-{dataset_name}_instance_{dataset_instance}"
        )
        classifier_dir = Path(cfg.save_path) / "classifiers" / classifier_name
        classifier_dir.mkdir(parents=True, exist_ok=True)

        # Save classifier
        wrapped_classifier.save(classifier_dir / "classifier.pt")

        # Save metadata
        import json

        metadata = {
            "classifier_class_name": wrapped_classifier.__class__.__name__,
            "dataset_path": str(dataset_dir),
            "dataset_type": dataset_metadata["class_name"],
            "seed": cfg.seed,
            "num_cells": list(cfg.num_cells),
            "dropout": cfg.dropout,
        }
        with open(classifier_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        log.info(f"Classifier saved to: {classifier_dir}")


if __name__ == "__main__":
    main()
