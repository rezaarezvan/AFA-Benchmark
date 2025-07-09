import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchrl.modules import MLP
from afa_generative.afa_methods import EDDI, UniformSampler, IterativeSelector, Ma2018AFAMethod
from afa_generative.utils import MaskLayer
from afa_generative.models import PartialVAE, fc_Net
from afa_generative.datasets import prepare_datasets
from common.registry import AFA_DATASET_REGISTRY
from common.custom_types import AFADataset
from common.utils import set_seed, dict_to_namespace
from afa_rl.zannone2019.models import PointNet, PointNetType
from afa_rl.utils import get_1D_identity


def main(pretrain_config_path: Path, dataset_type: str, train_dataset_path: Path, 
         val_dataset_path: Path, pretrained_model_path: Path, seed: int):
    set_seed(seed)
    
    with open(pretrain_config_path, "r") as file:
        config_dict: dict = yaml.safe_load(file)

    config = dict_to_namespace(config_dict)

    device = torch.device(config.device)
    
    train_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        train_dataset_path
    )
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[dataset_type].load(
        val_dataset_path
    )

    train_loader, val_loader, d_in, d_out \
        = prepare_datasets(train_dataset, val_dataset, config.dataloader.batch_size)
    
    naive_identity_fn=get_1D_identity
    naive_identity_size = d_in

    # Train PVAE.
    pointnet = PointNet(
        naive_identity_fn=naive_identity_fn,
        identity_network=MLP(
            in_features=naive_identity_size,
            out_features=config.pointnet.identity_size,
            num_cells=config.pointnet.identity_network_num_cells,
            activation_class=nn.ReLU,
        ),
        feature_map_encoder=MLP(
            in_features=config.pointnet.identity_size,
            out_features=config.pointnet.output_size,
            num_cells=config.pointnet.feature_map_encoder_num_cells,
            activation_class=nn.ReLU,
        ),
        pointnet_type=PointNetType.POINTNETPLUS,
    )
    encoder = MLP(
        in_features=config.pointnet.output_size,
        out_features=2 * config.partial_vae.latent_size,
        num_cells=config.encoder.num_cells,
        activation_class=nn.ReLU,
    )
    pv = PartialVAE(
        pointnet=pointnet,
        encoder=encoder,
        decoder=nn.Sequential(
            MLP(
                in_features=config.partial_vae.latent_size,
                out_features=d_in,
                num_cells=config.partial_vae.decoder_num_cells,
                activation_class=nn.ReLU,
            ),
            nn.Identity(),
        ),
    )
    pv = pv.to(device)
    pv.fit(train_loader=train_loader, 
           val_loader=val_loader, 
           lr=config.lr, 
           nepochs=config.nepochs,
           p_max=config.max_masking_probability,
           patience=config.patience,
           kl_scaling_factor=config.kl_scaling_factor)
    
    # Train masked predictor.
    mask_layer = MaskLayer(append=True)
    model = fc_Net(
        input_dim=d_in * 2,
        output_dim=d_out,
        hidden_layer_num=2,
        hidden_unit=[128, 128],
        activations='ReLU',
        drop_out_rate=0.3,
        flag_drop_out=True,
        flag_only_output_layer=False
    )
    sampler = UniformSampler(train_dataset.features)
    iterative = IterativeSelector(model, mask_layer, sampler).to(device)
    iterative.fit(
        train_loader,
        val_loader,
        lr=1e-3,
        nepochs=100,
        loss_fn=nn.CrossEntropyLoss(),
        patience=5,
        verbose=True)

    eddi_selector = Ma2018AFAMethod(pv, model)
    eddi_selector.save(pretrained_model_path)

    with open(pretrained_model_path / "params.yml", "w") as file:
        yaml.dump({
            "dataset_type": dataset_type,
            "train_dataset_path": str(train_dataset_path),
            "val_dataset_path": str(val_dataset_path),
            "seed": seed,
        }, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_config_path", type=str,
                        default="configs/ma2018/pretrain_ma2018.yml")
    parser.add_argument("--dataset_type", type=str, required=True, choices=AFA_DATASET_REGISTRY.keys())
    parser.add_argument("--train_dataset_path", type=Path, required=True)
    parser.add_argument("--val_dataset_path", type=Path, required=True)
    parser.add_argument("--pretrained_model_path", type=Path, required=True)
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
