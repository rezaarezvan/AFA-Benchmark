import torch
import hydra
import wandb
import logging

from pathlib import Path
from typing import Any, cast
from omegaconf import OmegaConf
from eval.utils import plot_metrics
from tempfile import TemporaryDirectory
from eval.metrics import eval_afa_method
from afa_oracle import create_aaco_method
from common.config_classes import AACOTrainConfig
from common.utils import load_dataset_artifact, set_seed

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../conf/train/aco", config_name="config"
)
def main(cfg: AACOTrainConfig):
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    run = wandb.init(
        config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        job_type="training",
        tags=["aaco"],
    )

    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    train_dataset, val_dataset, _, dataset_metadata = load_dataset_artifact(
        cfg.dataset_artifact_name
    )

    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]
    dataset_type = dataset_metadata["dataset_type"]

    log.info(f"Dataset: {dataset_type}")
    log.info(f"Features: {n_features}, Classes: {n_classes}")
    log.info(f"Training samples: {len(train_dataset)}")

    aaco_method = create_aaco_method(
        k_neighbors=cfg.aco.k_neighbors,
        acquisition_cost=cfg.aco.acquisition_cost,
        hide_val=cfg.aco.hide_val,
        dataset_name=dataset_type.lower(),
        device=device,
    )

    log.info("Fitting AACO oracle on training data...")
    log.info(f"Using hyperparameters: k_neighbors={cfg.aco.k_neighbors}, acquisition_cost={
             cfg.aco.acquisition_cost}, hide_val={cfg.aco.hide_val}")

    X_train = train_dataset.features.to(device)
    y_train = train_dataset.labels.to(device)
    aaco_method.aaco_oracle.fit(X_train, y_train)

    log.info("AACO method created and fitted")

    run.log({
        "oracle_config/k_neighbors": cfg.aco.k_neighbors,
        "oracle_config/acquisition_cost": cfg.aco.acquisition_cost,
        "oracle_config/hide_val": cfg.aco.hide_val,
        "oracle_config/dataset_features": n_features,
        "oracle_config/dataset_classes": n_classes,
        "oracle_config/training_samples": len(train_dataset),
    })

    if cfg.aco.evaluate_final_performance:
        log.info("Evaluating AACO method...")

        metrics = eval_afa_method(
            afa_select_fn=aaco_method.select,
            dataset=val_dataset,
            # AACO does not have a fixed budget, but for feasibility we use a budget of 25 features (no reason to use more)
            budget=25,
            # batch_size=32,
            afa_predict_fn=aaco_method.predict,
            only_n_samples=cfg.aco.eval_only_n_samples,
            device=device,
        )

        fig = plot_metrics(metrics)
        run.log({"aaco_metrics_plot": fig})

        final_accuracy = metrics["accuracy_all"][-1].item()
        final_f1 = metrics["f1_all"][-1].item()
        final_bce = metrics["bce_all"][-1].item()

        run.log({
            "final_accuracy": final_accuracy,
            "final_f1_score": final_f1,
            "final_bce": final_bce,
            "avg_features_used": len(metrics["accuracy_all"]),
        })

        log.info(f"Final performance - Accuracy: {final_accuracy:.4f}, F1: {
                 final_f1:.4f}, BCE: {final_bce:.4f}")

        budget_metrics = {}
        for i, (acc, f1, bce) in enumerate(zip(metrics["accuracy_all"], metrics["f1_all"], metrics["bce_all"])):
            budget_metrics[f"accuracy_at_{i+1}_features"] = acc.item()
            budget_metrics[f"f1_at_{i+1}_features"] = f1.item()
            budget_metrics[f"bce_at_{i+1}_features"] = bce.item()

        run.log(budget_metrics)

    with TemporaryDirectory(delete=False) as tmp_dir:
        tmp_path = Path(tmp_dir)

        aaco_method.save(tmp_path)

        aaco_artifact = wandb.Artifact(
            name=f"train_aaco-{dataset_type}-seed_{cfg.seed}",
            type="trained_method",
            metadata={
                "method_type": "aaco",
                "dataset_artifact_name": cfg.dataset_artifact_name,
                "dataset_type": dataset_type,
                "budget": None,  # AACO doesn't have a fixed budget
                "seed": cfg.seed,
                "k_neighbors": cfg.aco.k_neighbors,
                "acquisition_cost": cfg.aco.acquisition_cost,
                "hide_val": cfg.aco.hide_val,
                "implementation": "aaco",
            },
        )
        aaco_artifact.add_dir(str(tmp_path))
        run.log_artifact(aaco_artifact, aliases=cfg.output_artifact_aliases)

    run.finish()


if __name__ == "__main__":
    main()
