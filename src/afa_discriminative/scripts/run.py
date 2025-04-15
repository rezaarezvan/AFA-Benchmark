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
from afa_discriminative.models import MaskingPretrainer
from afa_discriminative.afa_methods import GreedyDynamicSelection, CMIEstimator


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='spam',
                    choices=['spam', 'diabetes', 'miniboone', 'MNIST'])
parser.add_argument('--method', type=str, default='GDFS',
                    choices=['GDFS', 'DIME'])
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


# Helper function for network architecture.
def get_network(d_in, d_out, hidden=128, dropout=0.3):
    model = nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_out))
    return model


if __name__ == '__main__':
    # Parse args.
    args = parser.parse_args()
    device = torch.device('cuda', args.gpu)
    if args.method == "GDFS":
        load_data = load_data_dict[args.dataset]
        num_features = num_features_dict[args.dataset]
        
        # Load dataset.
        dataset = load_data()
        d_in = dataset.input_size
        d_out = dataset.output_size
    
    
        # Normalize and split dataset.
        mean = dataset.tensors[0].mean(dim=0)
        std = torch.clamp(dataset.tensors[0].std(dim=0), min=1e-3)
        
        dataset.tensors = (dataset.tensors[0] - mean, dataset.tensors[1])
        train_dataset, val_dataset, test_dataset = data_split(dataset)
    
        # Prepare dataloaders.
        train_loader = DataLoader(
            train_dataset, batch_size=128,
            shuffle=True, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=1024, pin_memory=True)
    
    elif args.method == "DIME":
        if args.dataset == "MNIST":
            mnist_dataset = MNIST('/tmp/mnist/', download=True, train=True,
                          transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)]))
            np.random.seed(0)
            val_inds = np.sort(np.random.choice(len(mnist_dataset), size=10000, replace=False))
            train_inds = np.setdiff1d(np.arange(len(mnist_dataset)), val_inds)
            train_dataset = torch.utils.data.Subset(mnist_dataset, train_inds)
            val_dataset = torch.utils.data.Subset(mnist_dataset, val_inds)
            d_in = 784
            d_out = 10

            test_dataset = MNIST('/tmp/mnist/', download=True, train=False,
                         transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)]))

            train_loader = DataLoader(
                train_dataset, batch_size=128, shuffle=True, pin_memory=True,
                drop_last=True, num_workers=16)
            
            val_loader = DataLoader(
                val_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=16)
            
            test_loader = DataLoader(
                test_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=16)

    
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
            predictor = get_network(d_in * 2, d_out)
            selector = get_network(d_in * 2, d_in)
            
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
                max_features=max_features_dict[args.dataset],
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=True)
            
            # # Evaluate.
            # for num in num_features:
            #     auroc, acc = gdfs.evaluate(test_loader, num, (auroc_metric, acc_metric))
            #     results_dict['auroc'][num] = auroc
            #     results_dict['acc'][num] = acc
            #     print(f'Num = {num}, AUROC = {100*auroc:.2f}, Acc = {100*acc:.2f}')
                
            # Save model
            gdfs.cpu()
            run_description = f"trial_{trial}"
            save_dir = os.path.join(f"models/{args.method}/{args.dataset}", run_description)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(gdfs, os.path.join(save_dir,f'best_val_loss_gdfs_model.pt'))
        
        elif args.method == "DIME":
            if args.dataset == "MNIST":
                predictor = get_network(d_in * 2, d_out, hidden=512)
                # CMI Predictor
                value_network = get_network(d_in * 2, d_in, hidden=512)
                # Tie weights
                value_network[0] = predictor[0]
                value_network[3] = predictor[3]
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
                    val_loss_fn=Accuracy(task='multiclass', num_classes=10),
                    val_loss_mode='max',
                    verbose=True)
                
                run_description = f"trial_{trial}"
                save_dir = os.path.join("models/DIME/MNIST/pretrain", run_description)
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
                    max_features=50,
                    eps=0.05,
                    loss_fn=nn.CrossEntropyLoss(reduction='none'),
                    val_loss_fn=Accuracy(task='multiclass', num_classes=10),
                    val_loss_mode='max',
                    eps_decay=0.2,
                    eps_steps=10,
                    patience=5,
                    feature_costs=None) 
                
                run_description = f"trial_{trial}"
                save_dir = os.path.join("models/DIME/MNIST/joint_training", run_description)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(greedy_cmi_estimator,
                    os.path.join(save_dir, 'best_val_perf_cmi_estimator.pth'))

        
        # # Save results.
        # with open(f'results/{args.dataset}_{args.method}_{trial}.pkl', 'wb') as f:
        #     pickle.dump(results_dict, f)
