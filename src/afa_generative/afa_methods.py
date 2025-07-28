import math
import os
import copy
import collections
import numpy as np
from tqdm import trange
from pathlib import Path
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from scipy.stats import bernoulli
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from afa_generative.utils import *
from afa_generative.datasets import base_UCI_Dataset
from afa_generative.models import PartialVAE
from afa_rl.utils import mask_data
from common.custom_types import AFAMethod, AFASelection, FeatureMask, Label, MaskedFeatures, Features


def valid_probs(preds):
    '''Ensure valid probabilities.'''
    return torch.all((preds >= 0) & (preds <= 1))


def calculate_criterion(preds):
    '''Calculate feature selection criterion.'''

    if (len(preds.shape) == 1) or (preds.shape[1] == 1):
        # Binary classification.
        if not valid_probs(preds):
            preds = preds.sigmoid()
        if len(preds.shape) == 1:
            preds = preds.view(-1, 1)
        preds = torch.cat([1 - preds, preds])
    else:
        # Multiclass classification.
        if not valid_probs(preds):
            preds = preds.softmax(dim=1)
            
    # Calculate criterion: MI I(Y; X_j | x_s), KL divergence interpretation.
    mean = torch.mean(preds, dim=0, keepdim=True)
    kl = torch.sum(preds * torch.log(preds / (mean + 1e-6) + 1e-6), dim=1)
    return torch.mean(kl)


class Ma2018AFAMethod(AFAMethod):
    def __init__(self, sampler, predictor, num_classes, device=torch.device("cpu")):
        super().__init__()
        assert hasattr(sampler, 'impute')
        self.sampler: PartialVAE = sampler
        self.predictor = predictor
        self.num_classes = num_classes
        self._device: torch.device = device

    def predict(self, masked_features: MaskedFeatures, feature_mask: FeatureMask, features=None, label=None) -> Label:
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)
        B, F = masked_features.shape
        zeros_label = torch.zeros(B, self.num_classes, device=self._device)
        zeros_mask = torch.zeros(B, self.num_classes, device=self._device, dtype=feature_mask.dtype)
        augmented_masked_feature = torch.cat([masked_features, zeros_label], dim=-1).to(self._device)
        augmented_feature_mask = torch.cat([feature_mask, zeros_mask], dim=-1).to(self._device)
        
        with torch.no_grad():
            _, _, _, z, _ = self.sampler(augmented_masked_feature, augmented_feature_mask)
        
        return self.predictor(z).softmax(dim=-1)
    
    def select(self, masked_features: MaskedFeatures, feature_mask: FeatureMask, features=None, label=None) -> AFASelection:
        device = self._device
        B, F = masked_features.shape
        zeros_label = torch.zeros(B, self.num_classes, device=self._device)
        zeros_mask = torch.zeros(B, self.num_classes, device=self._device, dtype=feature_mask.dtype)
        augmented_masked_feature = torch.cat([masked_features, zeros_label], dim=-1).to(self._device)
        augmented_feature_mask = torch.cat([feature_mask, zeros_mask], dim=-1).to(self._device)
        x_full = self.sampler.impute(augmented_masked_feature, augmented_feature_mask).view(B, F+self.num_classes)
        next_feature_idx = []

        for i in range(B):
            m_i = augmented_feature_mask[i]
            x_i = x_full[i : i+1]
            best_j, best_score = None, -float('inf')

            for j in range(F):
                if m_i[j] == 1:
                    continue

                m_test = m_i.clone()
                m_test[j] = 1
                m_test = m_test.unsqueeze(0).repeat(x_i.size(0), 1)

                x_masked = x_i * m_test
                with torch.no_grad():
                    _, _, _, z, _ = self.sampler(x_masked, m_test)
                preds = self.predictor(z)

                score = calculate_criterion(preds)
                if score > best_score:
                    best_score = score
                    best_j = j
            
            next_feature_idx.append(best_j)
        
        next_feature_idx = torch.tensor(next_feature_idx, device=device)
        return next_feature_idx
        
    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(str(path / "model.pt"), map_location=device, weights_only=False)
        sampler = checkpoint["sampler"]
        predictor = checkpoint["predictor"]
        num_classes = checkpoint["num_classes"]
    
        predictor = predictor.to(device)
        sampler = sampler.to(device)
        return cls(sampler, predictor, num_classes, device)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "sampler": self.sampler,
            "predictor": self.predictor,
            "num_classes": self.num_classes
        }, str(path / "model.pt"))

    def to(self, device):
        self.sampler = self.sampler.to(device)
        self.predictor = self.predictor.to(device)
        self._device = device
        return self
    
    @property
    def device(self) -> torch.device:
        return self._device
 

class EDDI_Training(nn.Module):
    def __init__(self, 
                 classifier, 
                 partial_vae,
                 num_classes: int,
                 n_annealing_epochs: int,
                 start_kl_scaling_factor: float,
                 end_kl_scaling_factor: float,):
        super().__init__()
        self.classifier = classifier
        self.partial_vae = partial_vae
        self.num_classes = num_classes
        self.n_annealing_epochs = n_annealing_epochs
        self.start_kl_scaling_factor: float = start_kl_scaling_factor
        self.end_kl_scaling_factor: float = end_kl_scaling_factor

    def current_kl_weight(self, current_epoch) -> float:
        """Compute the current KL weight using linear annealing."""
        progress = min(1.0, current_epoch / self.n_annealing_epochs)
        return (
            self.start_kl_scaling_factor
            + (self.end_kl_scaling_factor - self.start_kl_scaling_factor) * progress
        )
    
    def fit(self,
            train_loader,
            val_loader,
            lr: float = 1e-3,
            classifier_loss_scaling_factor: float = 1,
            min_mask: float = 0.1,
            max_mask: float = 0.9,
            epochs: int = 100,
            device="cuda"):
        wandb.watch([self.classifier, self.partial_vae], log="all", log_freq=100)
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in trange(epochs, desc="Epochs", unit="epoch"):
            # print(f'{"-"*8}Epoch {epoch+1}{"-"*8}')
            self.train()
            train_results = {
                "loss_vae": 0.0, "loss_clf": 0.0, "loss_total": 0.0
            }
            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)
                labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
                labels_onehot = labels_onehot.to(device)
                optimizer.zero_grad()
                p = min_mask + torch.rand(1).item() * (max_mask - min_mask)
                augmented_features = torch.cat([features, labels_onehot], dim=-1)
                # masked_features, feature_mask, _ = mask_data(features, p)
                augmented_masked_features, augmented_feature_mask, _ \
                    = mask_data(augmented_features, p=p)
                encoding, mu, logvar, z, estimated_features = self.partial_vae(
                    augmented_masked_features, augmented_feature_mask
                )
                total_vae_loss, recon_loss, label_loss, kl_loss = self.partial_vae_loss(estimated_features, augmented_features, mu, logvar, epoch)
                logits = self.classifier(z)
                classifier_loss = F.cross_entropy(logits, labels)
                total = total_vae_loss + classifier_loss_scaling_factor * classifier_loss
                total.backward()
                optimizer.step()
                train_results['loss_vae'] += total_vae_loss.item()
                train_results['loss_clf'] += classifier_loss.item()
                train_results['loss_total'] += total.item()

            for k, v in train_results.items():
                train_results[k] = v / len(train_loader)
            
            # print(f"Train loss total: {train_results['loss_total']}")
            
            self.eval()
            eval_results = {
                "loss_vae_many": 0.0, "loss_clf_many": 0.0, "loss_total_many": 0.0, "acc_many": 0.0,
                "loss_vae_few": 0.0,  "loss_clf_few":  0.0,  "loss_total_few":  0.0,  "acc_few":  0.0,
            }
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
                    labels_onehot = labels_onehot.to(device)
                    augmented_features = torch.cat([features, labels_onehot], dim=-1)
                    for suffix, p in [("many", min_mask), ("few", max_mask)]:
                        augmented_feature_mask = torch.rand_like(augmented_features) > p
                        augmented_masked_features = augmented_features.clone()
                        augmented_masked_features[~augmented_feature_mask] = 0

                        _, mu, logvar, z, estimated_features = self.partial_vae(augmented_masked_features, augmented_feature_mask)
                        total_vae_loss, _, _, _ = self.partial_vae_loss(estimated_features, augmented_features, mu, logvar, epoch)
                        logits = self.classifier(z)
                        classifier_loss = F.cross_entropy(logits, labels)

                        preds = logits.argmax(dim=1)
                        acc = (preds == labels).float().mean()
                        eval_results[f"loss_vae_{suffix}"] += total_vae_loss.item()
                        eval_results[f"loss_clf_{suffix}"] += classifier_loss.item()
                        eval_results[f"loss_total_{suffix}"] += (total_vae_loss + classifier_loss).item()
                        eval_results[f"acc_{suffix}"] += acc.item()
                    
                for k, v in eval_results.items():
                    eval_results[k] = v / len(val_loader)
                
                # print(f"Val loss total (min mask): {eval_results[f"loss_total_many"]}, acc: {eval_results[f"acc_many"]}")
                # print(f"Val loss total (max mask): {eval_results[f"loss_total_few"]}, acc: {eval_results[f"acc_few"]}")
            
            wandb.log({
                "joint_training/train_loss": train_results['loss_total'],
                "joint_training/val_loss_many": eval_results[f"loss_total_many"],
                "joint_training/val_loss_few": eval_results[f"loss_total_few"],
                "joint_training/val_acc_many": eval_results[f"acc_many"],
                "joint_training/val_acc_few": eval_results[f"acc_few"]
            })

        wandb.unwatch([self.classifier, self.partial_vae])

    def partial_vae_loss(self, 
                         estimated_augmented_features, 
                         augmented_features, 
                         mu, 
                         logvar,
                         current_epoch):
        # recon_loss = ((x_hat - x_true) ** 2).sum(dim=1).mean()
        features, labels = (
            augmented_features[..., :-self.num_classes],
            augmented_features[..., -self.num_classes:],
        )
        estimated_features, estimated_label_logits = (
            estimated_augmented_features[..., :-self.num_classes],
            estimated_augmented_features[..., -self.num_classes:],
        )
        label_recon_loss = F.cross_entropy(
            estimated_label_logits, 
            labels, 
            reduction="mean"
        )
        feature_recon_loss = (
            ((estimated_features - features) ** 2).sum(dim=1).mean(dim=0)
        )
        kl_div_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean(
            dim=0
        )
        
        # kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean(dim=0)
        kl_div_loss = kl_div_loss * self.current_kl_weight(current_epoch)
        return (
            feature_recon_loss + label_recon_loss + kl_div_loss, 
            feature_recon_loss, 
            label_recon_loss,
            kl_div_loss,
        )

