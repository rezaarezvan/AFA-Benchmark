import os
import torch
import argparse
import torch.nn as nn
from torchmetrics import Accuracy, AUROC
from torch.utils.data import DataLoader
from afa_generative.afa_methods import EDDI, UniformSampler, IterativeSelector
from afa_generative.utils import MaskLayer
from afa_generative.models import PVAE, fc_Net, PointNetPlusEncoder
from common.registry import AFA_DATASET_REGISTRY
from common.custom_types import AFADataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", type=str, required=True, choices=AFA_DATASET_REGISTRY.keys())
parser.add_argument("--dataset_train_path", type=str, required=True)
parser.add_argument("--dataset_val_path", type=str, required=True)
parser.add_argument("--dataset_test_path", type=str, required=True)


max_features_dict = {
    'spam': 35,
    'diabetes': 35,
    'miniboone': 35,
}


if __name__ == '__main__':
    # Parse args.
    args = parser.parse_args()
    device = torch.device('cuda')
    
    train_dataset: AFADataset = AFA_DATASET_REGISTRY[args.dataset_type].load(
        args.dataset_train_path
    )
    val_dataset: AFADataset = AFA_DATASET_REGISTRY[args.dataset_type].load(
        args.dataset_val_path
    )
    test_dataset: AFADataset = AFA_DATASET_REGISTRY[args.dataset_type].load(
        args.dataset_test_path
    )

    d_in = train_dataset.features.shape[-1]
    d_out = train_dataset.labels.shape[-1]

    for ds in (train_dataset, val_dataset, test_dataset):
        ds.features = ds.features.float()
        ds.labels = ds.labels.argmax(dim=1).long()

    train_loader = DataLoader(
        train_dataset, batch_size=128,
        shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=128, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=128, pin_memory=True
    )

    # results_dict = {
    #     'auroc': {},
    #     'acc': {},
    #     'features': {}
    # }
    # auroc_metric = lambda pred, y: AUROC(task='multiclass', num_classes=d_out)(pred.softmax(dim=1), y)
    # acc_metric = Accuracy(task='multiclass', num_classes=d_out)
    
    # Train PVAE.
    bottleneck = 16
    encoder = PointNetPlusEncoder(
        obs_dim=d_in,
        latent_dim=bottleneck
    ).to(device)
    decoder = fc_Net(
        input_dim=bottleneck,
        output_dim=d_in,
        hidden_layer_num=2,
        hidden_unit=[128, 128],
        activations='ReLU',
        drop_out_rate=0.3,
        flag_drop_out=True,
        flag_only_output_layer=False
    ).to(device)
    pv = PVAE(encoder, decoder).to(device)
    pv.fit(
        train_loader,
        val_loader,
        lr=1e-3,
        nepochs=250,
        verbose=True)
    
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
    sampler = UniformSampler(train_dataset.features)  # TODO don't actually need sampler
    iterative = IterativeSelector(model, mask_layer, sampler).to(device)
    iterative.fit(
        train_loader,
        val_loader,
        lr=1e-3,
        nepochs=100,
        loss_fn=nn.CrossEntropyLoss(),
        patience=5,
        verbose=True)
    
    # Set up EDDI feature selection object.
    eddi_selector = EDDI(pv, model, mask_layer, 'classification').to(device)
    
    # # Evaluate
    # num_features = list(range(1, d_in, 5))
    # metrics_dict = eddi_selector.evaluate_multiple(test_loader, num_features, (auroc_metric, acc_metric))
    # for num in num_features:
    #     auroc, acc = metrics_dict[num]
    #     results_dict['auroc'][num] = auroc
    #     results_dict['acc'][num] = acc
    #     print(f'Num = {num}, AUROC = {100*auroc:.2f}, Acc = {100*acc:.2f}')

