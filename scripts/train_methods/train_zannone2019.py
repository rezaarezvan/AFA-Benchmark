import matplotlib
from matplotlib import pyplot as plt
import gc
import logging
from pathlib import Path
from typing import Any, cast
from tempfile import TemporaryDirectory

import hydra
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, check_env_specs, set_exploration_type
from afa_rl.agents import Agent
from tqdm import tqdm
from dacite import from_dict

import wandb
from afa_rl.afa_env import AFAEnv
from afa_rl.afa_methods import RLAFAMethod
from afa_rl.zannone2019.agents import Zannone2019Agent
from afa_rl.zannone2019.reward import get_zannone2019_reward_fn
from afa_rl.datasets import get_afa_dataset_fn
from afa_rl.zannone2019.models import (
    Zannone2019PretrainingModel,
    Zannone2019AFAClassifier,
    Zannone2019AFAPredictFn,
)
from afa_rl.zannone2019.utils import (
    get_zannone2019_model_from_config,
)
from afa_rl.utils import (
    get_eval_metrics,
)
from common.afa_methods import RandomDummyAFAMethod
from common.config_classes import Zannone2019PretrainConfig, Zannone2019TrainConfig
from common.custom_types import (
    AFADataset,
)
from common.utils import (
    dict_with_prefix,
    get_class_probabilities,
    load_dataset_artifact,
    set_seed,
)

from eval.metrics import eval_afa_method
from eval.utils import plot_metrics

# matplotlib.use("WebAgg")


def load_pretrained_model_artifacts(
    artifact_name: str,
) -> tuple[
    AFADataset,  # train dataset
    AFADataset,  # val dataset
    AFADataset,  # test dataset
    dict[str, Any],  # dataset metadata
    Zannone2019PretrainingModel,
    Zannone2019PretrainConfig,
]:
    """Load a pretrained model and the dataset it was trained on, from a WandB artifact."""
    pretrained_model_artifact = wandb.use_artifact(
        artifact_name, type="pretrained_model"
    )
    pretrained_model_artifact_dir = Path(pretrained_model_artifact.download())
    # The dataset dir should contain a file called model.pt
    artifact_filenames = [f.name for f in pretrained_model_artifact_dir.iterdir()]
    assert {"model.pt"}.issubset(artifact_filenames), (
        f"Dataset artifact must contain a model.pt file. Instead found: {artifact_filenames}"
    )

    # Access config of the run that produced this pretrained model
    pretraining_run = pretrained_model_artifact.logged_by()
    assert pretraining_run is not None, (
        "Pretrained model artifact must be logged by a run."
    )
    pretrained_model_config_dict = pretraining_run.config
    print(f"{pretrained_model_config_dict=}")
    pretrained_model_config: Zannone2019PretrainConfig = from_dict(
        data_class=Zannone2019PretrainConfig, data=pretrained_model_config_dict
    )

    # Load the dataset that the pretrained model was trained on
    train_dataset, val_dataset, test_dataset, dataset_metadata = load_dataset_artifact(
        pretrained_model_config.dataset_artifact_name
    )

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(f"Class probabilities in training set: {train_class_probabilities}")

    pretrained_model = get_zannone2019_model_from_config(
        pretrained_model_config,
        n_features,
        n_classes,
        train_class_probabilities,
    )
    pretrained_model_checkpoint = torch.load(
        pretrained_model_artifact_dir / "model.pt",
        weights_only=True,
        map_location=torch.device("cpu"),
    )
    pretrained_model.load_state_dict(pretrained_model_checkpoint["state_dict"])

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        dataset_metadata,
        pretrained_model,
        pretrained_model_config,
    )


def visualize_digits(
    features: torch.Tensor, labels: torch.Tensor, shuffle: bool = True
):
    """Visualize 9 MNIST digits"""
    fig, axs = plt.subplots(3, 3)
    if shuffle:
        indices = torch.randperm(len(features))[:9]
    else:
        indices = torch.arange(9)
    for i in range(9):
        row = i // 3
        col = i % 3
        idx = indices[i]
        axs[row, col].imshow(features[idx].numpy().reshape((28, 28)))
        axs[row, col].set_title(labels[idx].argmax().item())
    return fig, axs


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../conf/train/zannone2019", config_name="config"
)
def main(cfg: Zannone2019TrainConfig):
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    run = wandb.init(
        config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        job_type="training",
        tags=["zannone2019"],
    )

    # Log W&B run URL
    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    # Load pretrained model and dataset from artifacts
    (
        train_dataset,
        val_dataset,
        _,
        dataset_metadata,
        pretrained_model,
        pretrained_model_config,
    ) = load_pretrained_model_artifacts(cfg.pretrained_model_artifact_name)
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()
    pretrained_model.requires_grad_(False)  # zannone2019 does not train jointly
    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(f"Class probabilities in training set: {train_class_probabilities}")
    class_weights = F.softmax(1 / train_class_probabilities, dim=-1).to(device)
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    pretrained_model = pretrained_model.to(device)

    reward_fn = get_zannone2019_reward_fn(
        afa_predict_fn=Zannone2019AFAPredictFn(pretrained_model), weights=class_weights
    )

    # Use the pretrained model to generate new artificial data
    generated_features = torch.zeros(cfg.n_generated_samples, n_features)
    generated_labels = torch.zeros(cfg.n_generated_samples, n_classes)
    n_generation_batches = cfg.n_generated_samples // cfg.generation_batch_size
    for batch_idx in tqdm(
        range(n_generation_batches), desc="Generating artificial samples"
    ):
        generated_features_batch, generated_labels_batch = (
            pretrained_model.generate_data(
                latent_size=pretrained_model_config.partial_vae.latent_size,
                n_features=n_features,
                device=device,
                n_samples=cfg.generation_batch_size,
            )
        )
        generated_features[
            batch_idx * cfg.generation_batch_size : (batch_idx + 1)
            * cfg.generation_batch_size,
            :,
        ] = generated_features_batch.cpu()
        # Convert labels to one hot instead of continuous probabilities
        generated_labels[
            batch_idx * cfg.generation_batch_size : (batch_idx + 1)
            * cfg.generation_batch_size,
            :,
        ] = F.one_hot(
            generated_labels_batch.argmax(-1),
            num_classes=generated_labels_batch.shape[-1],
        ).cpu()

    # Also reconstruct some samples, both when providing a label and not
    # _, label_reconstructed_features = pretrained_model.fully_observed_reconstruction(
    #     features=train_dataset.features[:100].to(device),
    #     n_classes=n_classes,
    #     label=train_dataset.labels[:100].to(device),
    # )
    # _, reconstructed_features = pretrained_model.fully_observed_reconstruction(
    #     features=train_dataset.features[:100].to(device),
    #     n_classes=n_classes,
    #     label=None,
    # )
    # z = z.cpu()
    # label_reconstructed_features = label_reconstructed_features.cpu()
    # reconstructed_features = reconstructed_features.cpu()

    # Visualize MNIST digits
    # fig_real, _axs_real = visualize_digits(
    #     train_dataset.features, train_dataset.labels, shuffle=False
    # )
    # fig_real.suptitle("Real digits")
    # if cfg.n_generated_samples >= 9:
    #     fig_fake, _axs_fake = visualize_digits(
    #         generated_features, generated_labels, shuffle=False
    #     )
    #     fig_fake.suptitle("Generated digits")
    # fig_label_recon, _axs_label_recon = visualize_digits(
    #     label_reconstructed_features, train_dataset.labels, shuffle=False
    # )
    # fig_label_recon.suptitle("Reconstructed digits using label")
    # fig_recon, _axs_recon = visualize_digits(
    #     reconstructed_features, train_dataset.labels, shuffle=False
    # )
    # fig_recon.suptitle("Reconstructed digits without using label")
    #
    # plt.show()
    #
    # Plot distribution of latent variables
    # plt.figure(figsize=(12, 4))
    # for i in range(min(z.shape[1], 10)):
    #     plt.subplot(2, 5, i + 1)
    #     plt.hist(z[:, i], bins=30)
    #     plt.title(f"Latent dim {i}")
    # plt.tight_layout()
    # plt.show()

    combined_features = torch.cat([train_dataset.features, generated_features])
    combined_features = combined_features[torch.randperm(len(combined_features))]
    combined_labels = torch.cat([train_dataset.labels, generated_labels])
    combined_labels = combined_labels[torch.randperm(len(combined_labels))]

    # MDP expects special dataset functions
    train_dataset_fn = get_afa_dataset_fn(combined_features, combined_labels)
    val_dataset_fn = get_afa_dataset_fn(val_dataset.features, val_dataset.labels)

    train_env = AFAEnv(
        dataset_fn=train_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((cfg.n_agents,)),
        feature_size=n_features,
        n_classes=n_classes,
        hard_budget=cfg.hard_budget,
    )
    check_env_specs(train_env)

    # Manual debugging
    # td = train_env.reset()
    # td = train_env.rand_step(td)

    eval_env = AFAEnv(
        dataset_fn=val_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((1,)),
        feature_size=n_features,
        n_classes=n_classes,
        hard_budget=cfg.hard_budget,
    )

    agent: Agent = Zannone2019Agent(
        cfg=cfg.agent,
        pointnet=pretrained_model.partial_vae.pointnet,
        encoder=pretrained_model.partial_vae.encoder,
        action_spec=train_env.action_spec,
        latent_size=pretrained_model_config.partial_vae.latent_size,
        action_mask_key="action_mask",
        batch_size=cfg.batch_size,
        module_device=device,
        replay_buffer_device=device,
    )

    collector = SyncDataCollector(
        train_env,
        agent.get_policy(),
        frames_per_batch=cfg.batch_size,
        total_frames=cfg.n_batches * cfg.batch_size,
        # device=device,
    )

    # Training loop
    try:
        for batch_idx, tds in tqdm(
            enumerate(collector), total=cfg.n_batches, desc="Training agent..."
        ):
            collector.update_policy_weights_()

            # Collapse agent and batch dimensions
            td = tds.flatten(start_dim=0, end_dim=1)
            loss_info = agent.process_batch(td)

            # Log training info
            run.log(
                dict_with_prefix(
                    "train/",
                    loss_info
                    | dict_with_prefix("cheap_info.", agent.get_cheap_info())
                    | {
                        "reward": td["next", "reward"].mean().item(),
                    },
                )
            )

            if batch_idx != 0 and batch_idx % cfg.eval_every_n_batches == 0:
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    td_evals = [
                        eval_env.rollout(
                            cfg.eval_max_steps, agent.get_exploitative_policy()
                        ).squeeze(0)
                        for _ in tqdm(range(cfg.n_eval_episodes), desc="Evaluating")
                    ]
                metrics_eval = get_eval_metrics(
                    td_evals, Zannone2019AFAPredictFn(pretrained_model)
                )
                run.log(
                    dict_with_prefix(
                        "eval/",
                        metrics_eval
                        | dict_with_prefix(
                            "expensive_info.", agent.get_expensive_info()
                        ),
                    )
                )

    except KeyboardInterrupt:
        pass
    finally:
        # Convert the embedder+agent to an AFAMethod and save it as a temporary file
        pretrained_model = pretrained_model.to(torch.device("cpu"))
        afa_method = RLAFAMethod(
            agent.get_policy().to("cpu"),
            Zannone2019AFAClassifier(pretrained_model, device=torch.device("cpu")),
        )
        # Save the method to a temporary directory and load it again to ensure it is saved correctly
        with TemporaryDirectory(delete=False) as tmp_path_str:
            tmp_path = Path(tmp_path_str)
            afa_method.save(tmp_path)
            del afa_method
            afa_method = RLAFAMethod.load(
                tmp_path,
                device=torch.device("cpu"),
            )
            if cfg.evaluate_final_performance:
                # Check what the final performance of the method is. Costly for large datasets.
                metrics = eval_afa_method(
                    afa_method.select,
                    val_dataset,
                    cfg.hard_budget,
                    afa_method.predict,
                    only_n_samples=cfg.eval_only_n_samples,
                )
                fig_metrics = plot_metrics(metrics)
                # Also check performance of dummy method for comparison
                dummy_afa_method = RandomDummyAFAMethod(
                    device=torch.device("cpu"), n_classes=n_classes
                )
                dummy_metrics = eval_afa_method(
                    dummy_afa_method.select,
                    val_dataset,
                    cfg.hard_budget,
                    afa_method.predict,
                    only_n_samples=cfg.eval_only_n_samples,
                )
                fig_dummy_metrics = plot_metrics(dummy_metrics)
                run.log(
                    {
                        "final_performance_plot": fig_metrics,
                        "final_dummy_performance_plot": fig_dummy_metrics,
                    }
                )

            # Save the model as a WandB artifact
            # Save the name of the afa method class as metadata
            afa_method_artifact = wandb.Artifact(
                name=f"train_zannone2019-{pretrained_model_config.dataset_artifact_name.split(':')[0]}-budget_{cfg.hard_budget}-seed_{cfg.seed}",
                type="trained_method",
                metadata={
                    "method_type": "zannone2019",
                    "dataset_artifact_name": pretrained_model_config.dataset_artifact_name,
                    "dataset_type": dataset_metadata["dataset_type"],
                    "budget": cfg.hard_budget,
                    "seed": cfg.seed,
                },
            )

            afa_method_artifact.add_dir(str(tmp_path))
            run.log_artifact(afa_method_artifact, aliases=cfg.output_artifact_aliases)

        run.finish()

        gc.collect()  # Force Python GC
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
            torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
