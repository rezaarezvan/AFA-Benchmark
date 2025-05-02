import os
import torch
import yaml
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from afa_discriminative.utils import MaskLayer
from afa_discriminative.models import MaskingPretrainer, fc_Net
from afa_discriminative.afa_methods import GreedyDynamicSelection, CMIEstimator, Covert2023AFAMethod, Gadgil2023AFAMethod
from common.utils import dict_to_namespace, set_seed
from common.custom_types import AFADataset
from common.registry import AFA_DATASET_REGISTRY


def prepare_datasets(dataset_type: str, train_path: Path, val_path: Path, batch_size: int):
    train_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        train_path
    )
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        val_path
    )
    
    for ds in (train_dataset, val_dataset):
        ds.features = ds.features.float()
        ds.labels = ds.labels.argmax(dim=1).long()
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset,   batch_size=batch_size,
                              pin_memory=True)
    
    d_in = train_dataset.features.shape[-1]
    d_out = train_dataset.labels.shape[-1]
    return train_loader, val_loader, d_in, d_out

def pretrain_gdfs(
        pretrain_config_path: Path,
        dataset_type: str,
        train_dataset_path: Path,
        val_dataset_path: Path,
        pretrained_model_path: Path,
        seed: int
    ) -> None:
    with open(pretrain_config_path, 'r') as f:
        pretrain_config_dict = yaml.safe_load(f)
    pretrain_config = dict_to_namespace(pretrain_config_dict)
    device = torch.device(pretrain_config.device)

    set_seed(seed)
    print(train_dataset_path)
    train_loader, val_loader, d_in, d_out = prepare_datasets(dataset_type, train_dataset_path, val_dataset_path, pretrain_config.batch_size)
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
    pretrain = MaskingPretrainer(predictor, mask_layer).to(device)

    pretrain.fit(
        train_loader,
        val_loader,
        lr=pretrain_config.lr,
        nepochs=pretrain_config.nepochs,
        loss_fn=nn.CrossEntropyLoss(),
        patience=pretrain_config.patience,
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

def pretrain_dime(
        pretrain_config_path: Path,
        dataset_type: str,
        train_dataset_path: Path,
        val_dataset_path: Path,
        pretrained_model_path: Path,
        seed: int
    ) -> None:
    with open(pretrain_config_path, "r") as file:
        pretrain_config_dict: dict = yaml.safe_load(file)
    pretrain_config = dict_to_namespace(pretrain_config_dict)
    device = torch.device(pretrain_config.device)

    set_seed(seed)

    train_loader, val_loader, d_in, d_out = prepare_datasets(dataset_type, train_dataset_path, val_dataset_path, pretrain_config.batch_size)
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

def train_gdfs(
        pretrain_config_path: Path, 
        train_config_path: Path, 
        dataset_type: str, 
        train_dataset_path: Path, 
        val_dataset_path: Path, 
        pretrained_model_path: Path, 
        hard_budget: int, 
        seed: int, 
        afa_method_path: Path
    )-> None:
    set_seed(seed)
    with open(train_config_path, "r") as file:
        train_config_dict: dict = yaml.safe_load(file)
    train_config = dict_to_namespace(train_config_dict)

    with open(pretrain_config_path, "r") as file:
        pretrain_config_dict: dict = yaml.safe_load(file)
    pretrain_config = dict_to_namespace(pretrain_config_dict)

    device = torch.device(train_config.device)

    train_loader, val_loader, d_in, d_out = prepare_datasets(dataset_type, train_dataset_path, val_dataset_path, train_config.batch_size)

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
    
    predictor.load_state_dict(torch.load(pretrained_model_path  / "model.pt", map_location=device)["predictor_state_dict"])
    
    selector = fc_Net(
            input_dim=d_in * 2,
            output_dim=d_in,
            hidden_layer_num=len(train_config.architecture.hidden_units),
            hidden_unit=train_config.architecture.hidden_units,
            activations=train_config.architecture.activations,
            drop_out_rate=train_config.architecture.dropout,
            flag_drop_out=train_config.architecture.flag_drop_out,
            flag_only_output_layer=train_config.architecture.flag_only_output_layer
        )
    
    mask_layer = MaskLayer(append=True)
    gdfs = GreedyDynamicSelection(selector, predictor, mask_layer).to(device)
    gdfs.fit(
        train_loader,
        val_loader,
        lr=train_config.lr,
        nepochs=train_config.nepochs,
        max_features=hard_budget,
        loss_fn=nn.CrossEntropyLoss(),
        patience=train_config.patience,
        verbose=True)

    afa_method = Covert2023AFAMethod(gdfs.selector.cpu(), gdfs.predictor.cpu())
    afa_method_path.mkdir(parents=True, exist_ok=True)
    afa_method.save(afa_method_path / "model.pt")
    with open(afa_method_path / "params.yml", "w") as file:
        yaml.dump(
            {
                "hard_budget": hard_budget,
                "seed": seed,
                "dataset_type": dataset_type,
                "train_dataset_path": str(train_dataset_path),
                "val_dataset_path": str(val_dataset_path),
                "pretrained_model_path": str(pretrained_model_path),
            },
            file,
        )

    print(f"Covert2023AFAMethod saved to {afa_method_path}")

def train_dime(
        pretrain_config_path: Path, 
        train_config_path: Path, 
        dataset_type: str, 
        train_dataset_path: Path, 
        val_dataset_path: Path, 
        pretrained_model_path: Path, 
        hard_budget: int, 
        seed: int, 
        afa_method_path: Path
    )-> None:
    set_seed(seed)
    with open(train_config_path, "r") as file:
        train_config_dict: dict = yaml.safe_load(file)
    train_config = dict_to_namespace(train_config_dict)

    with open(pretrain_config_path, "r") as file:
        pretrain_config_dict: dict = yaml.safe_load(file)
    pretrain_config = dict_to_namespace(pretrain_config_dict)

    device = torch.device(train_config.device)

    train_loader, val_loader, d_in, d_out = prepare_datasets(dataset_type, train_dataset_path, val_dataset_path, train_config.batch_size)

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
    
    predictor.load_state_dict(torch.load(pretrained_model_path  / "model.pt", map_location=device)["predictor_state_dict"])
    
    value_network = fc_Net(
            input_dim=d_in * 2,
            output_dim=d_in,
            hidden_layer_num=len(train_config.architecture.hidden_units),
            hidden_unit=train_config.architecture.hidden_units,
            activations=train_config.architecture.activations,
            drop_out_rate=train_config.architecture.dropout,
            flag_drop_out=train_config.architecture.flag_drop_out,
            flag_only_output_layer=train_config.architecture.flag_only_output_layer
        )
    
    value_network.hidden[0] = predictor.hidden[0]
    value_network.hidden[1] = predictor.hidden[1]
    mask_layer = MaskLayer(append=True)
    
    greedy_cmi_estimator = CMIEstimator(value_network, predictor, mask_layer).to(device)
    greedy_cmi_estimator.fit(
        train_loader,
        val_loader,
        lr=train_config.lr,
        nepochs=train_config.nepochs,
        max_features=hard_budget,
        eps=train_config.eps,
        loss_fn=nn.CrossEntropyLoss(reduction='none'),
        val_loss_fn=Accuracy(task='multiclass', num_classes=d_out),
        val_loss_mode='max',
        eps_decay=train_config.eps_decay,
        eps_steps=train_config.eps_steps,
        patience=train_config.patience,
        feature_costs=None)

    afa_method = Gadgil2023AFAMethod(greedy_cmi_estimator.value_network.cpu(), greedy_cmi_estimator.predictor.cpu())
    afa_method_path.mkdir(parents=True, exist_ok=True)
    afa_method.save(afa_method_path / "model.pt")
    with open(afa_method_path / "params.yml", "w") as file:
        yaml.dump(
            {
                "hard_budget": hard_budget,
                "seed": seed,
                "dataset_type": dataset_type,
                "train_dataset_path": str(train_dataset_path),
                "val_dataset_path": str(val_dataset_path),
                "pretrained_model_path": str(pretrained_model_path),
            },
            file,
        )

    print(f"Gadgil2023AFAMethod saved to {afa_method_path}")
