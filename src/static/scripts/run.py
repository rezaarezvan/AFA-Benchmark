import os
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, AUROC
from static.models import get_network, BaseModel
from static.static_methods import DifferentiableSelector, ConcreteMask
from common.custom_types import AFADataset
from common.registry import AFA_DATASET_REGISTRY
from common.utils import set_seed


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", type=str, required=True, choices=AFA_DATASET_REGISTRY.keys())
parser.add_argument("--dataset_train_path", type=str, required=True)
parser.add_argument("--dataset_val_path", type=str, required=True)
parser.add_argument("--dataset_test_path", type=str, required=True)
parser.add_argument('--method', type=str, choices=['cae', 'permutation'], required=True)
parser.add_argument('--num_restarts', default=1, type=int)

max_features_dict = {
    'spam': 35,
    'diabetes': 35,
    'miniboone': 35,
    'MNIST': 50,
    'cube': 20,
    'physionet': 41,
    'AFAContext': 30,
}

def transform_dataset(dataset: AFADataset, selected_features):
    x = dataset.features
    y = dataset.labels
    x_selected = x[:, selected_features]
    return TensorDataset(x_selected, y)

if __name__ == '__main__':
    set_seed(42)
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

    d_in = train_dataset.features.shape[-1]
    d_out = train_dataset.labels.shape[-1]

    results_dict = {
        'acc': {},
        'features': {}
    }
    
    acc_metric = Accuracy(task='multiclass', num_classes=d_out)
    auroc_metric = lambda pred, y: AUROC(task='multiclass', num_classes=d_out)(pred.softmax(dim=1), y)

    num_features = list(range(1, d_in))
    if args.method == "cae":
        for num in num_features:
            # Train model with differentiable feature selection.
            model = get_network(d_in, d_out)
            selector_layer = ConcreteMask(d_in, num)
            diff_selector = DifferentiableSelector(model, selector_layer).to(device)
            diff_selector.fit(
                train_loader,
                val_loader,
                lr=1e-3,
                nepochs=250,
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=False)

            # Extract top features.
            logits = selector_layer.logits.cpu().data.numpy()
            selected_features = np.sort(logits.argmax(axis=1))
            if len(np.unique(selected_features)) != num:
                print(f'{len(np.unique(selected_features))} selected instead of {num}, appending extras')
                num_extras = num - len(np.unique(selected_features))
                remaining_features = np.setdiff1d(np.arange(d_in), selected_features)
                selected_features = np.sort(np.concatenate([np.unique(selected_features), remaining_features[:num_extras]]))

            train_subset = transform_dataset(train_dataset, selected_features)
            val_subset = transform_dataset(val_dataset, selected_features)
            test_subset = transform_dataset(test_dataset, selected_features)
            
            # Prepare subset dataloaders.
            train_subset_loader = DataLoader(train_subset, batch_size=128, shuffle=True, pin_memory=True, drop_last=True)
            val_subset_loader = DataLoader(val_subset, batch_size=1024, pin_memory=True)
            test_subset_loader = DataLoader(test_subset, batch_size=1024, pin_memory=True)
            
            best_loss = np.inf
            for _ in range(args.num_restarts):
                # Train model.
                model = get_network(num, d_out)
                basemodel = BaseModel(model).to(device)
                basemodel.fit(
                    train_subset_loader,
                    val_subset_loader,
                    lr=1e-3,
                    nepochs=250,
                    loss_fn=nn.CrossEntropyLoss(),
                    verbose=False)
                
                # Check if best.
                val_loss = basemodel.evaluate(val_subset_loader, nn.CrossEntropyLoss())
                if val_loss < best_loss:
                    best_model = basemodel
                    best_loss = val_loss

            acc = best_model.evaluate(test_subset_loader, (acc_metric))
            results_dict['acc'][num] = acc
            results_dict['features'][num] = selected_features
            print(f'Num = {num}, Acc = {100*acc:.2f}')
    elif args.method == "permutation":
        model = get_network(d_in, d_out)
        basemodel = BaseModel(model).to(device)
        basemodel.fit(
            train_loader,
            val_loader,
            lr=1e-3,
            nepochs=250,
            # loss_fn=nn.CrossEntropyLoss() ,
            loss_fn=nn.CrossEntropyLoss(),
            verbose=False)
        
        # Calculate feature importance scores.
        permutation_importance = np.zeros(d_in)
        x_train = train_dataset.features
        for i in tqdm(range(d_in)):
            x_val = val_dataset.features
            y_val = val_dataset.labels
            x_val[:, i] = x_train[np.random.choice(len(x_train), size=len(x_val)), i]
            with torch.no_grad():
                pred = model(x_val.to(device)).cpu()
                permutation_importance[i] = - auroc_metric(pred, y_val)
        ranked_features = np.argsort(permutation_importance)[::-1]

        for num in num_features:
            # Prepare top features and smaller version of dataset.
            selected_features = ranked_features[:num]
            train_subset = transform_dataset(train_dataset, selected_features.copy())
            val_subset = transform_dataset(val_dataset, selected_features.copy())
            test_subset = transform_dataset(test_dataset, selected_features.copy())
            
            # Prepare subset dataloaders.
            train_subset_loader = DataLoader(train_subset, batch_size=128, shuffle=True, pin_memory=True, drop_last=True)
            val_subset_loader = DataLoader(val_subset, batch_size=1024, pin_memory=True)
            test_subset_loader = DataLoader(test_subset, batch_size=1024, pin_memory=True)
        
            best_loss = np.inf
            for _ in range(args.num_restarts):
                # Train model.
                model = get_network(num, d_out)
                basemodel = BaseModel(model).to(device)
                basemodel.fit(
                    train_subset_loader,
                    val_subset_loader,
                    lr=1e-3,
                    nepochs=250,
                    loss_fn=nn.CrossEntropyLoss(),
                    verbose=False)
                
                # Check if best.
                val_loss = basemodel.evaluate(val_subset_loader, nn.CrossEntropyLoss())
                if val_loss < best_loss:
                    best_model = basemodel
                    best_loss = val_loss
            
            # Evaluate using best model.
            acc = best_model.evaluate(test_subset_loader, (acc_metric))
            results_dict['acc'][num] = acc
            results_dict['features'][num] = selected_features
            print(f'Num = {num}, Acc = {100*acc:.2f}')

