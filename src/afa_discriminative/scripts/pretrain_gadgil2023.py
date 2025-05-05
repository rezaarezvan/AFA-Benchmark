import os
import torch
import yaml
import argparse
import torch.nn as nn
from pathlib import Path
from torchmetrics import Accuracy
from afa_discriminative.utils import MaskLayer
from afa_discriminative.models import MaskingPretrainer, fc_Net
from afa_discriminative.datasets import prepare_datasets
from common.utils import dict_to_namespace, set_seed
from common.custom_types import AFADataset
from common.registry import AFA_DATASET_REGISTRY


def main(pretrain_config_path: Path, dataset_type: str, train_dataset_path: Path, val_dataset_path: Path, pretrained_model_path: Path, seed: int):
    set_seed(seed)
    with open(pretrain_config_path, "r") as file:
        pretrain_config_dict: dict = yaml.safe_load(file)
    pretrain_config = dict_to_namespace(pretrain_config_dict)
    device = torch.device(pretrain_config.device)

    set_seed(seed)

    train_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        train_dataset_path
    )
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        val_dataset_path
    )
    train_loader, val_loader, d_in, d_out = prepare_datasets(train_dataset, val_dataset, pretrain_config.batch_size)
    predictor = fc_Net(
            input_dim=d_in * 2,
            output_dim=d_out,
            hidden_layer_num=len(pretrain_config.architecture.hidden_units),
            hidden_unit=pretrain_config.architecture.hidden_units,
            activations=pretrain_config.architecture.activations,
            drop_out_rate=pretrain_config.architecture.dropout,
            flag_drop_out=pretrain_config.architecture.flag_drop_out,
            flag_only_output_layer=pretrain_config.architecture.flag_only_output_layer
        )
    
    mask_layer = MaskLayer(append=True)
    print('Pretraining predictor')
    print('-'*8)
    pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
    pretrain.fit(
        train_loader,
        val_loader,
        lr=pretrain_config.lr,
        nepochs=pretrain_config.nepochs,
        loss_fn=nn.CrossEntropyLoss(),
        val_loss_fn=Accuracy(task='multiclass', num_classes=d_out),
        val_loss_mode='max',
        verbose=True)
    
    pretrained_model_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        'predictor_state_dict': pretrain.model.state_dict(),
        'architecture': {
            'd_in': d_in,
            'd_out': d_out,
            'predictor_hidden_layers': pretrain_config.architecture.hidden_units,
            'dropout': pretrain_config.architecture.dropout,
        }}, os.path.join(pretrained_model_path, f'model.pt'))

    with open(pretrained_model_path / "params.yml", "w") as file:
        yaml.dump({
            "dataset_type": dataset_type,
            "train_dataset_path": str(train_dataset_path),
            "val_dataset_path": str(val_dataset_path),
            "seed": seed,
        }, file)


if __name__ == "__main__":

    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_config_path", type=Path, required=True)
    parser.add_argument("--dataset_type", type=str, required=True, choices=AFA_DATASET_REGISTRY.keys())
    parser.add_argument("--train_dataset_path", type=Path, required=True)
    parser.add_argument("--val_dataset_path", type=Path, required=True)
    parser.add_argument("--pretrained_model_path", type=Path, required=True, help="Path to folder to save the pretrained model")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    main(
        pretrain_config_path=args.pretrain_config_path,
        dataset_type=args.dataset_type,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        pretrained_model_path=args.pretrained_model_path,
        seed=args.seed
    )
