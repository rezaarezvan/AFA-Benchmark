import os
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AUROC
from afa_generative.afa_methods import EDDI, UniformSampler, IterativeSelector
from afa_generative.utils import MaskLayer
from afa_generative.models import PVAE, fc_Net
from afa_generative.datasets import load_spam, load_diabetes, load_miniboone, data_split, get_xy


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='spam',
                    choices=['spam', 'diabetes', 'miniboone', 'concrete'])
parser.add_argument('--method', type=str, default='eddi',
                    choices=['eddi'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_trials', type=int, default=1)
parser.add_argument('--num_restarts', type=int, default=1)


# Various configurations.
load_data_dict = {
    'spam': load_spam,
    'diabetes': load_diabetes,
    'miniboone': load_miniboone,
}
num_features_dict = {
    'spam': list(range(1, 11)) + list(range(15, 30, 5)),
    'diabetes': list(range(1, 11)),
    'miniboone': list(range(1, 11)) + list(range(15, 30, 5)),
}
max_features_dict = {
    'spam': 35,
    'diabetes': 35,
    'miniboone': 35,
}


if __name__ == '__main__':
    # Parse args.
    args = parser.parse_args()
    load_data = load_data_dict[args.dataset]
    num_features = num_features_dict[args.dataset]
    device = torch.device('cuda', args.gpu)
    
    # Load dataset.
    dataset = load_data()
    d_in = dataset.input_size
    d_out = dataset.output_size
    
    # Normalize and split dataset.
    mean = dataset.tensors[0].mean(dim=0)
    std = torch.clamp(dataset.tensors[0].std(dim=0), min=1e-3)
    if args.method == 'eddi':
        # PVAE generative model works better with standardized data.
        dataset.tensors = ((dataset.tensors[0] - mean) / std, dataset.tensors[1])
    else:
        dataset.tensors = (dataset.tensors[0] - mean, dataset.tensors[1])
    train_dataset, val_dataset, test_dataset = data_split(dataset)
    
    # Prepare dataloaders.
    train_loader = DataLoader(
        train_dataset, batch_size=128,
        shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, pin_memory=True)
    
    # Make results directory.
    if not os.path.exists('results'):
        os.makedirs('results')
    
    for trial in range(args.num_trials):
        # For saving results.
        results_dict = {
            'auroc': {},
            'acc': {},
            'features': {}
        }
        auroc_metric = lambda pred, y: AUROC(task='multiclass', num_classes=d_out)(pred.softmax(dim=1), y)
        acc_metric = Accuracy(task='multiclass', num_classes=d_out)
        
        if args.method == 'eddi':
            # Train PVAE.
            bottleneck = 16
            encoder = fc_Net(
                input_dim=d_in * 2,
                output_dim=bottleneck * 2,
                hidden_layer_num=2,
                hidden_unit=[128, 128],
                activations='ReLU',
                drop_out_rate=0.3,
                flag_drop_out=True,
                flag_only_output_layer=False
            )
            decoder = fc_Net(
                input_dim=bottleneck,
                output_dim=d_in,
                hidden_layer_num=2,
                hidden_unit=[128, 128],
                activations='ReLU',
                drop_out_rate=0.3,
                flag_drop_out=True,
                flag_only_output_layer=False
            )
            mask_layer = MaskLayer(append=True)
            pv = PVAE(encoder, decoder, mask_layer, 128, 'gaussian').to(device)
            pv.fit(
                train_loader,
                val_loader,
                lr=1e-3,
                nepochs=250,
                verbose=True)
            
            # Train masked predictor.
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
            sampler = UniformSampler(get_xy(train_dataset)[0])  # TODO don't actually need sampler
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
            
            # Evaluate
            metrics_dict = eddi_selector.evaluate_multiple(test_loader, num_features, (auroc_metric, acc_metric))
            for num in num_features:
                auroc, acc = metrics_dict[num]
                results_dict['auroc'][num] = auroc
                results_dict['acc'][num] = acc
                print(f'Num = {num}, AUROC = {100*auroc:.2f}, Acc = {100*acc:.2f}')
        
        # Save results.
        with open(f'results/{args.dataset}_{args.method}_{trial}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
