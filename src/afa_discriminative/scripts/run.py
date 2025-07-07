import os
import torch
import argparse
import yaml
import torch.nn as nn
import numpy as np
import random
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AUROC
from afa_discriminative.utils import MaskLayer
from afa_discriminative.models import MaskingPretrainer, fc_Net
from afa_discriminative.afa_methods import GreedyDynamicSelection, CMIEstimator
from common.custom_types import AFADataset
from common.registry import AFA_DATASET_REGISTRY


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_type", type=str, required=True, choices=AFA_DATASET_REGISTRY.keys()
)
parser.add_argument("--dataset_train_path", type=str, required=True)
parser.add_argument("--dataset_val_path", type=str, required=True)
parser.add_argument("--dataset_test_path", type=str, required=True)
parser.add_argument("--method", type=str, choices=["GDFS", "DIME"], required=True)


max_features_dict = {
    "spam": 35,
    "diabetes": 35,
    "miniboone": 35,
    "MNIST": 50,
    "cube": 20,
    "physionet": 41,
    "AFAContext": 30,
}


if __name__ == "__main__":
    random_seed = 42
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Parse args.
    args = parser.parse_args()
    device = torch.device("cuda")
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

    # if args.method == "GDFS":
    #     train_mean = train_dataset.features.mean(dim=0)
    #     train_std = torch.clamp(train_dataset.features.std(dim=0), min=1e-3)
    #     val_mean = val_dataset.features.mean(dim=0)
    #     val_std = torch.clamp(val_dataset.features.std(dim=0), min=1e-3)
    #     test_mean = train_dataset.features.mean(dim=0)
    #     test_std = torch.clamp(train_dataset.features.std(dim=0), min=1e-3)

    #     train_dataset.features = (train_dataset.features - train_mean)
    #     val_dataset.features = (val_dataset.features - val_mean)
    #     test_dataset.features = (test_dataset.features - test_mean)

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=128, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, pin_memory=True)

    # For saving results.
    # results_dict = {
    #     'auroc': {},
    #     'acc': {},
    #     'features': {}
    # }
    # auroc_metric = lambda pred, y: AUROC(task='multiclass', num_classes=d_out)(pred.softmax(dim=1), y)
    # acc_metric = Accuracy(task='multiclass', num_classes=d_out)

    if args.method == "GDFS":
        # Prepare networks.
        predictor = fc_Net(
            input_dim=d_in * 2,
            output_dim=d_out,
            hidden_layer_num=2,
            hidden_unit=[128, 128],
            activations="ReLU",
            drop_out_rate=0.3,
            flag_drop_out=True,
            flag_only_output_layer=False,
        )
        selector = fc_Net(
            input_dim=d_in * 2,
            output_dim=d_in,
            hidden_layer_num=2,
            hidden_unit=[128, 128],
            activations="ReLU",
            drop_out_rate=0.3,
            flag_drop_out=True,
            flag_only_output_layer=False,
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
            verbose=True,
        )

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
            verbose=True,
        )

        # Evaluate.
        # for num in num_features:
        #     auroc, acc = gdfs.evaluate(test_loader, num, (auroc_metric, acc_metric))
        #     results_dict['auroc'][num] = auroc
        #     results_dict['acc'][num] = acc
        #     print(f'Num = {num}, AUROC = {100*auroc:.2f}, Acc = {100*acc:.2f}')

        # Save model
        gdfs = gdfs.cpu()
        save_dir = os.path.join("models", f"{args.method}/{args.dataset_type}")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(
            {
                "selector_state_dict": gdfs.selector.state_dict(),
                "predictor_state_dict": gdfs.predictor.state_dict(),
                "architecture": {
                    "d_in": d_in,
                    "d_out": d_out,
                    "selector_hidden_layers": [128, 128],
                    "predictor_hidden_layers": [128, 128],
                    "dropout": 0.3,
                },
            },
            os.path.join(save_dir, f"model.pt"),
        )

        params = {
            "hard_budget": max_features_dict[args.dataset_type],
            "seed": random_seed,
            "dataset_type": args.dataset_type,
            "train_dataset_path": args.dataset_train_path,
            "val_dataset_path": args.dataset_val_path,
        }
        with open(os.path.join(save_dir, "params.yml"), "w") as f:
            yaml.dump(params, f)

    elif args.method == "DIME":
        predictor = fc_Net(
            input_dim=d_in * 2,
            output_dim=d_out,
            hidden_layer_num=2,
            hidden_unit=[128, 128],
            activations="ReLU",
            drop_out_rate=0.3,
            flag_drop_out=True,
            flag_only_output_layer=False,
        )
        # CMI Predictor
        value_network = fc_Net(
            input_dim=d_in * 2,
            output_dim=d_in,
            hidden_layer_num=2,
            hidden_unit=[128, 128],
            activations="ReLU",
            drop_out_rate=0.3,
            flag_drop_out=True,
            flag_only_output_layer=False,
        )
        # Tie weights
        value_network.hidden[0] = predictor.hidden[0]
        value_network.hidden[1] = predictor.hidden[1]
        mask_layer = MaskLayer(mask_size=d_in, append=True)

        # Pretrain predictor.
        print("Pretraining predictor")
        print("-" * 8)
        pretrain = MaskingPretrainer(predictor, mask_layer).to(device)

        pretrain.fit(
            train_loader,
            val_loader,
            lr=1e-3,
            nepochs=200,
            loss_fn=nn.CrossEntropyLoss(),
            val_loss_fn=Accuracy(task="multiclass", num_classes=d_out),
            val_loss_mode="max",
            verbose=True,
        )

        # Joint training.
        print("Training CMI estimator")
        print("-" * 8)

        greedy_cmi_estimator = CMIEstimator(value_network, predictor, mask_layer)
        greedy_cmi_estimator.fit(
            train_loader,
            val_loader,
            lr=1e-3,
            nepochs=250,
            max_features=max_features_dict[args.dataset_type],
            eps=0.05,
            loss_fn=nn.CrossEntropyLoss(reduction="none"),
            val_loss_fn=Accuracy(task="multiclass", num_classes=d_out),
            val_loss_mode="max",
            eps_decay=0.2,
            eps_steps=10,
            patience=5,
            feature_costs=None,
        )

        save_dir = os.path.join("models", f"{args.method}/{args.dataset_type}")
        os.makedirs(save_dir, exist_ok=True)
        greedy_cmi_estimator = greedy_cmi_estimator.cpu()
        torch.save(
            {
                "value_network_state_dict": greedy_cmi_estimator.value_network.state_dict(),
                "predictor_state_dict": greedy_cmi_estimator.predictor.state_dict(),
                "architecture": {
                    "d_in": d_in,
                    "d_out": d_out,
                    "value_network_hidden_layers": [128, 128],
                    "predictor_hidden_layers": [128, 128],
                    "dropout": 0.3,
                },
            },
            os.path.join(save_dir, f"model.pt"),
        )

        params = {
            "hard_budget": max_features_dict[args.dataset_type],
            "seed": random_seed,
            "dataset_type": args.dataset_type,
            "train_dataset_path": args.dataset_train_path,
            "val_dataset_path": args.dataset_val_path,
        }
        with open(os.path.join(save_dir, "params.yml"), "w") as f:
            yaml.dump(params, f)
