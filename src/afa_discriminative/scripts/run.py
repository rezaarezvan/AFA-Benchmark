import os
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchmetrics import Accuracy, AUROC
from afa_discriminative.utils import MaskLayer
from afa_discriminative.datasets import load_spam, load_diabetes, load_miniboone, data_split
from afa_discriminative.models import MaskingPretrainer, fc_Net
from afa_discriminative.afa_methods import GreedyDynamicSelection, CMIEstimator
from common.custom_types import AFADataset
from common.registry import AFA_DATASET_REGISTRY


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", type=str, required=True, choices=AFA_DATASET_REGISTRY.keys())
parser.add_argument("--dataset_train_path", type=str, required=True)
parser.add_argument("--dataset_val_path", type=str, required=True)
parser.add_argument("--dataset_test_path", type=str, required=True)
parser.add_argument('--method', type=str, choices=['GDFS', 'DIME'], required=True)
parser.add_argument('--num_trials', type=int, default=1)
parser.add_argument('--num_restarts', type=int, default=1)


# num_features_dict = {
#     'spam': list(range(1, 11)) + list(range(15, 30, 5)),
#     'diabetes': list(range(1, 11)),
#     'miniboone': list(range(1, 11)) + list(range(15, 30, 5)),
# }
max_features_dict = {
    'spam': 35,
    'diabetes': 35,
    'miniboone': 35,
    'MNIST': 50,
    'cube': 20,
    'physionet': 41
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

    train_dataset.features = train_dataset.features.float()
    val_dataset.features = val_dataset.features.float()
    test_dataset.features = test_dataset.features.float()
    train_dataset.labels = train_dataset.labels.argmax(dim=1).long()
    val_dataset.labels = val_dataset.labels.argmax(dim=1).long()
    test_dataset.labels = test_dataset.labels.argmax(dim=1).long()
    # num_features = list(range(d_in))

    if args.method == "GDFS":
        train_mean = train_dataset.features.mean(dim=0)
        train_std = torch.clamp(train_dataset.features.std(dim=0), min=1e-3)
        val_mean = val_dataset.features.mean(dim=0)
        val_std = torch.clamp(val_dataset.features.std(dim=0), min=1e-3)
        test_mean = train_dataset.features.mean(dim=0)
        test_std = torch.clamp(train_dataset.features.std(dim=0), min=1e-3)

        train_dataset.features = (train_dataset.features - train_mean) / train_std
        val_dataset.features = (val_dataset.features - val_mean) / val_std
        test_dataset.features = (test_dataset.features - test_mean) / test_std

    train_loader = DataLoader(
        train_dataset, batch_size=128,
        shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=128, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=128, pin_memory=True
    )

    # Make results directory.
    # if not os.path.exists('results'):
    #     os.makedirs('results')
    
    for trial in range(args.num_trials):
        # For saving results.
        results_dict = {
            'auroc': {},
            'acc': {},
            'features': {}
        }
        auroc_metric = lambda pred, y: AUROC(task='multiclass', num_classes=d_out)(pred.softmax(dim=1), y)
        acc_metric = Accuracy(task='multiclass', num_classes=d_out)
        
        if args.method == 'GDFS':
            # Prepare networks.
            predictor = fc_Net(
                input_dim=d_in * 2,
                output_dim=d_out,
                hidden_layer_num=2,
                hidden_unit=[128, 128],
                activations='ReLU',
                drop_out_rate=0.3,
                flag_drop_out=True,
                flag_only_output_layer=False
            )
            selector = fc_Net(
                input_dim=d_in * 2,
                output_dim=d_in,
                hidden_layer_num=2,
                hidden_unit=[128, 128],
                activations='ReLU',
                drop_out_rate=0.3,
                flag_drop_out=True,
                flag_only_output_layer=False
            )
            
            # Pretrain predictor.
            mask_layer = MaskLayer(append=True)
            pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
            pretrain.fit(
                train_loader,
                val_loader,
                lr=1e-3,
                nepochs=100,
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=True)
            
            # Train selector and predictor jointly.
            gdfs = GreedyDynamicSelection(selector, predictor, mask_layer).to(device)
            gdfs.fit(
                train_loader,
                val_loader,
                lr=1e-3,
                nepochs=250,
                # max_features=max_features_dict[args.dataset],
                max_features=max_features_dict[args.dataset_type],
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=True)
            
            # Evaluate.
            # for num in num_features:
            #     auroc, acc = gdfs.evaluate(test_loader, num, (auroc_metric, acc_metric))
            #     results_dict['auroc'][num] = auroc
            #     results_dict['acc'][num] = acc
            #     print(f'Num = {num}, AUROC = {100*auroc:.2f}, Acc = {100*acc:.2f}')
                
            # Save model
            gdfs.cpu()
            run_description = f"trial_{trial}"
            save_dir = os.path.join(f"models/afa_discriminative/{args.method}/{args.dataset_type}", run_description)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(gdfs, os.path.join(save_dir,f'best_val_loss_gdfs_model.pt'))
        
        elif args.method == "DIME":
            predictor = fc_Net(
                input_dim=d_in * 2,
                output_dim=d_out,
                hidden_layer_num=2,
                hidden_unit=[512, 512],
                activations='ReLU',
                drop_out_rate=0.3,
                flag_drop_out=True,
                flag_only_output_layer=False
            )
            # CMI Predictor
            value_network = fc_Net(
                input_dim=d_in * 2,
                output_dim=d_in,
                hidden_layer_num=2,
                hidden_unit=[512, 512],
                activations='ReLU',
                drop_out_rate=0.3,
                flag_drop_out=True,
                flag_only_output_layer=False
            )
            # Tie weights
            value_network.hidden[0] = predictor.hidden[0]
            value_network.hidden[1] = predictor.hidden[1]
            mask_layer = MaskLayer(mask_size=d_in, append=True)

            # Pretrain predictor.
            print('Pretraining predictor')
            print('-'*8)
            pretrain = MaskingPretrainer(predictor, mask_layer).to(device)

            pretrain.fit(
                train_loader,
                val_loader,
                lr=1e-3,
                nepochs=200,
                loss_fn=nn.CrossEntropyLoss(),
                val_loss_fn=Accuracy(task='multiclass', num_classes=d_out),
                val_loss_mode='max',
                verbose=True)
            
            run_description = f"trial_{trial}"
            save_dir = os.path.join(f"models/afa_discriminative/{args.method}/{args.dataset_type}/pretrain", run_description)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(pretrain.model.state_dict(), os.path.join(save_dir, 'best_val_perf_predictor.pth'))

            # Joint training.
            print('Training CMI estimator')
            print('-'*8)

            greedy_cmi_estimator = CMIEstimator(value_network, predictor, mask_layer)
            greedy_cmi_estimator.fit(
                train_loader,
                val_loader,
                lr=1e-3,
                nepochs=250,
                max_features=max_features_dict[args.dataset_type],
                eps=0.05,
                loss_fn=nn.CrossEntropyLoss(reduction='none'),
                val_loss_fn=Accuracy(task='multiclass', num_classes=d_out),
                val_loss_mode='max',
                eps_decay=0.2,
                eps_steps=10,
                patience=5,
                feature_costs=None) 
            
            run_description = f"trial_{trial}"
            save_dir = os.path.join(f"models/afa_discriminative/{args.method}/{args.dataset_type}/joint_training", run_description)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(greedy_cmi_estimator.cpu(),
                os.path.join(save_dir, 'best_val_perf_cmi_estimator.pth'))

        
        # # Save results.
        # with open(f'results/{args.dataset}_{args.method}_{trial}.pkl', 'wb') as f:
        #     pickle.dump(results_dict, f)
