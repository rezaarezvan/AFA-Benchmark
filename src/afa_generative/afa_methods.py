import math
import os
import copy
import collections
import numpy as np
from pathlib import Path
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
from common.custom_types import AFAMethod, AFASelection, FeatureMask, Label, MaskedFeatures


class EDDI(nn.Module):
    '''
    Efficient dynamic discovery of high-value information (EDDI): dynamic
    feature selection with missing features sampled from a conditional
    generative model.
    
    Args:
      sampler:
      predictor:
      task:
    '''
    def __init__(self, sampler, predictor, mask_layer, task='classification'):
        super().__init__()
        assert hasattr(sampler, 'impute')
        self.sampler = sampler
        self.model = predictor
        self.mask_layer = mask_layer
        assert task in ('regression', 'classification')
        self.task = task
        
        if isinstance(mask_layer, MaskLayerGrouped):
            self.data_imputer = Imputer(self.mask_layer.group_matrix.cpu().data.numpy())
        else:
            self.data_imputer = Imputer()
    
    def fit(self):
        raise NotImplementedError('models should be fit beforehand')
    
    def forward(self, x, max_features, verbose=False):
        '''
        Select features and make prediction.
        
        Args:
          x:
          max_features:
          num_samples:
          verbose:
        '''
        x_masked, _ = self.select_features(x, max_features, verbose)
        return self.model(x_masked)
    
    def forward_multiple(self, x, num_features_list, verbose=False):
        '''
        Select features and make prediction for multiple feature budgets.
        
        Args:
          x:
          max_features:
          num_samples:
          verbose:
        '''
        for num, x_masked, _ in self.select_features_multiple(x, num_features_list, verbose):
            yield num, self.model(x_masked)

    def select_features(self, x, max_features, verbose=False):
        '''
        Select features.

        Args:
          x:
          max_features:
          num_samples:
          verbose:
        '''
        # Set up model.
        model = self.model
        mask_layer = self.mask_layer
        sampler = self.sampler
        data_imputer = self.data_imputer
        device = next(model.parameters()).device
        
        # Set up mask.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            assert len(x.shape) == 2
            mask_size = x.shape[1]
        num_features = mask_size
        assert 0 < max_features < num_features
        m = torch.zeros((x.shape[0], mask_size), device=device)

        for i in tqdm(range(len(x))):
            # Get row.
            x_row = x[i:i+1]
            m_row = m[i:i+1]

            for k in range(max_features):
                # Setup.
                best_ind = None
                best_criterion = - np.inf
                
                # Sample values for all remaining features.
                x_sampled = sampler.impute(x_row, m_row)[0]
                num_samples = x_sampled.shape[0]
                m_expand = m_row.repeat(num_samples, 1)
                for j in range(num_features):
                    if m[i][j] == 1:
                        # TODO support for groups is not elegant
                        if isinstance(mask_layer, MaskLayerGrouped):
                            inds = torch.where(mask_layer.group_matrix[j])[0].cpu().data.numpy()
                            original = x_row[:, inds]
                        else:
                            original = x_row[:, j]
                        x_sampled = data_imputer.impute(x_sampled, original, j)

                for j in range(num_features):
                    # Check if already included.
                    if m[i][j] == 1:
                        continue
                    
                    # Adjust mask.
                    m_expand[:, j] = 1
                    x_expand_masked = mask_layer(x_sampled, m_expand)
                
                    # Make predictions.
                    with torch.no_grad():
                        preds = model(x_expand_masked)
                    
                    # Measure criterion.
                    criterion = calculate_criterion(preds, self.task)
                    if verbose:
                        print(f'Feature {j} criterion = {criterion:.4f}')
                    
                    # Check if best.
                    if criterion > best_criterion:
                        best_criterion = criterion
                        best_ind = j
                        
                    # Turn off entry.
                    m_expand[:, j] = 0
                    
                # Select new feature.
                if verbose:
                    print(f'Selecting feature {best_ind}')
                m[i][best_ind] = 1

        # Apply mask.
        x_masked = mask_layer(x, m)
        return x_masked, m
    
    def select_features_multiple(self, x, num_features_list, verbose=False):
        '''
        Select features for multiple budgets.

        Args:
          x:
          num_features_list:
          num_samples:
          verbose:
        '''

        # Set up model.
        model = self.model
        mask_layer = self.mask_layer
        sampler = self.sampler
        data_imputer = self.data_imputer
        device = next(model.parameters()).device
        
        # Set up mask.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            assert len(x.shape) == 2
            mask_size = x.shape[1]
        num_features = mask_size
        assert isinstance(num_features_list, (list, tuple, np.ndarray))
        # print(max(num_features_list), num_features)
        assert 0 < max(num_features_list) < num_features
        assert min(num_features_list) > 0
        max_features = max(num_features_list)
        m = torch.zeros((x.shape[0], mask_size), device=device)

        for k in range(max_features):
            for i in range(len(x)):
                # Get row.
                x_row = x[i:i+1]
                m_row = m[i:i+1]
                
                # Setup.
                best_ind = None
                best_criterion = - np.inf
                
                # Sample values for all remaining features.
                x_sampled = sampler.impute(x_row, m_row)[0]
                if x_sampled.dim() == 1:
                    x_sampled = x_sampled.unsqueeze(0)
                num_samples = x_sampled.shape[0]
                m_expand = m_row.repeat(num_samples, 1)
                for j in range(num_features):
                    if m[i][j] == 1:
                        # TODO support for groups is not elegant
                        if isinstance(mask_layer, MaskLayerGrouped):
                            inds = torch.where(mask_layer.group_matrix[j])[0].cpu().data.numpy()
                            original = x_row[:, inds]
                        else:
                            original = x_row[:, j]
                        x_sampled = data_imputer.impute(x_sampled, original, j)

                for j in range(num_features):
                    # Check if already included.
                    if m[i][j] == 1:
                        continue
                    
                    # Adjust mask.
                    m_expand[:, j] = 1
                    x_expand_masked = mask_layer(x_sampled, m_expand)
                
                    # Make predictions.
                    with torch.no_grad():
                        preds = model(x_expand_masked)
                    
                    # Measure criterion.
                    criterion = calculate_criterion(preds, self.task)
                    if verbose:
                        print(f'Feature {j} criterion = {criterion:.4f}')
                    
                    # Check if best.
                    if criterion > best_criterion:
                        best_criterion = criterion
                        best_ind = j
                        
                    # Turn off entry.
                    m_expand[:, j] = 0
                    
                # Select new feature.
                if verbose:
                    print(f'Selecting feature {best_ind}')
                m[i][best_ind] = 1

            # Yield current results if necessary.
            if (k + 1) in num_features_list:
                yield k + 1, mask_layer(x, m), m
    
    def evaluate(self,
                 loader,
                 max_features,
                 metric):
        '''
        Evaluate mean performance across a dataset.
        
        Args:
          loader:
          max_features:
          metric:
        '''
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # For calculating mean loss.
        pred_list = []
        label_list = []

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device)

                # Calculate loss.
                pred = self.forward(x, max_features)
                pred_list.append(pred.cpu())
                label_list.append(y.cpu())
        
            # Calculate metric(s).
            y = torch.cat(label_list, 0)
            pred = torch.cat(pred_list, 0)
            if isinstance(metric, (tuple, list)):
                score = [m(pred, y).item() for m in metric]
            elif isinstance(metric, dict):
                score = {name: m(pred, y).item() for name, m in metric.items()}
            else:
                score = metric(pred, y).item()
                
        return score
    
    def evaluate_multiple(self,
                          loader,
                          num_features_list,
                          metric):
        '''
        Evaluate mean performance across a dataset for multiple feature budgets.
        
        Args:
          loader:
          num_features_list:
          metric:
        '''
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # For calculating mean loss.
        pred_dict = {num: [] for num in num_features_list}
        score_dict = {num: None for num in num_features_list}
        label_list = []

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device)

                # Calculate loss.
                for num, pred in self.forward_multiple(x, num_features_list):
                    pred_dict[num].append(pred.cpu())
                label_list.append(y.cpu())
        
            # Calculate metric(s).
            y = torch.cat(label_list, 0)
            for num in num_features_list:
                pred = torch.cat(pred_dict[num], 0)
                if self.task == "regression":
                    pred = pred.squeeze(-1)
                if isinstance(metric, (tuple, list)):
                    score = [m(pred, y).item() for m in metric]
                elif isinstance(metric, dict):
                    score = {name: m(pred, y).item() for name, m in metric.items()}
                else:
                    score = metric(pred, y).item()
                score_dict[num] = score
                
        return score_dict


def valid_probs(preds):
    '''Ensure valid probabilities.'''
    return torch.all((preds >= 0) & (preds <= 1))


def calculate_criterion(preds, task):
    '''Calculate feature selection criterion.'''
    if task == 'regression':
        # Calculate criterion: prediction variance.
        return torch.var(preds)

    elif task == 'classification':
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
    else:
        raise ValueError(f'unsupported task: {task}. Must be classification or regression')


class Ma2018AFAMethod(AFAMethod):
    def __init__(self, sampler, predictor, task='classification'):
        super().__init__()
        assert hasattr(sampler, 'impute')
        self.sampler = sampler
        self.predictor = predictor
        assert task in ('regression', 'classification')
        self.task = task

    def predict(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> Label:
        x_masked = torch.cat([masked_features, feature_mask], dim=1)
        pred = self.predictor(x_masked)
        return pred
    
    def select(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> AFASelection:
        device = next(self.predictor.parameters()).device
        B, F = masked_features.shape
        x_full = self.sampler.impute(masked_features, feature_mask).view(B, F)
        next_feature_idx = []

        for i in range(B):
            m_i = feature_mask[i]
            x_i = x_full[i : i+1]
            best_j, best_score = None, -float('inf')

            for j in range(F):
                if m_i[j] == 1:
                    continue

                m_test = m_i.clone()
                m_test[j] = 1
                m_test = m_test.unsqueeze(0).repeat(x_i.size(0), 1)

                x_masked = x_i * m_test
                x_masked = torch.cat([x_masked, m_test], dim=1)
                preds = self.predictor(x_masked)

                score = calculate_criterion(preds, self.task)
                if score > best_score:
                    best_score = score
                    best_j = j
            
            next_feature_idx.append(best_j)
        
        next_feature_idx = torch.tensor(next_feature_idx, device=device)
        return next_feature_idx + 1
        
    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
        sampler = checkpoint["sampler"]
        predictor = checkpoint["predictor"]
        task = checkpoint["task"]
    
        predictor = predictor.to(device)
        sampler = sampler.to(device)
        return cls(sampler, predictor, task)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "sampler": self.sampler,
            "predictor": self.predictor,
            "task": self.task,
        }, str(path / "model.pt"))


class UniformSampler:
    '''
    Sample from rows of the dataset uniformly at random.
    
    Args:
      x:
      group_matrix:
      deterministic:
    '''
    def __init__(self, x, group_matrix=None, deterministic=True):
        self.x = x
        self.deterministic = deterministic
        self.group_matrix = group_matrix
        
    def sample(self, x_masked, m, ind, num_samples):
        '''
        Generate feature samples.
        
        Args:
          x_masked:
          m:
          ind:
          num_samples:
        '''
        if self.deterministic:
            rng = np.random.default_rng(0)
            rows = rng.choice(len(self.x), size=num_samples)
        else:
            rows = np.random.choice(len(self.x), size=num_samples)
        if self.group_matrix is not None:
            ind = np.where(self.group_matrix[ind] == 1)[0]
            return self.x[rows][:, ind]
        return self.x[rows, ind]


class Imputer:
    '''
    Impute entries in the data matrix.
    
    Args:
      group_matrix:
    '''
    def __init__(self, group_matrix=None):
        self.group_matrix = group_matrix
        
    def impute(self, x, x_ind, ind):
        '''
        Generate feature samples.
        
        Args:
          x:
          x_ind:
          ind:
        '''
        if self.group_matrix is not None:
            ind = np.where(self.group_matrix[ind] == 1)[0]
            x[:, ind] = x_ind
        else:
            x[:, ind] = x_ind
        return x
    

class IterativeSelector(nn.Module):
    '''
    Iteratively select features based on maximum prediction variability.
    
    Args:
      model:
      mask_layer:
      data_sampler:
      task:
    '''
    def __init__(self, model, mask_layer, data_sampler, task='classification'):
        super().__init__()
        self.model = model
        self.mask_layer = mask_layer
        self.data_sampler = data_sampler
        assert task in ('regression', 'classification')
        self.task = task
        if self.data_sampler.group_matrix is not None:
            self.data_imputer = Imputer(self.data_sampler.group_matrix)
        else:
            self.data_imputer = Imputer()
        
    def fit(self,
            train_loader,
            val_loader,
            lr,
            nepochs,
            loss_fn,
            val_loss_fn=None,
            val_loss_mode='min',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            early_stopping_epochs=None,
            verbose=True):
        '''
        Train model.
        
        Args:
          train_loader:
          val_loader:
          lr:
          nepochs:
          loss_fn:
          val_loss_fn:
          val_loss_mode:
          factor:
          patience:
          min_lr:
          early_stopping_epochs:
          verbose:
        '''
        # Verify arguments.
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = 'min'
        else:
            if val_loss_mode is None:
                raise ValueError('must specify val_loss_mode (min or max) when validation_loss_fn is specified')
        
        # Set up optimizer and lr scheduler.
        model = self.model
        mask_layer = self.mask_layer
        device = next(model.parameters()).device
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=val_loss_mode, factor=factor, patience=patience,
            min_lr=min_lr)
        
        # Determine mask size.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            x, y = next(iter(val_loader))
            assert len(x.shape) == 2
            mask_size = x.shape[1]

        # For tracking best model and early stopping.
        best_model = None
        num_bad_epochs = 0
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
            
        for epoch in range(nepochs):
            # Switch model to training mode.
            model.train()

            for x, y in train_loader:
                # Move to device.
                x = x.to(device)
                y = y.to(device)
                
                # Generate missingness.
                m = generate_uniform_mask(len(x), mask_size).to(device)

                # Calculate loss.
                x_masked = mask_layer(x, m)
                pred = model(x_masked)
                loss = loss_fn(pred, y)

                # Take gradient step.
                loss.backward()
                opt.step()
                model.zero_grad()
                
            # Calculate validation loss.
            model.eval()
            with torch.no_grad():
                # For mean loss.
                pred_list = []
                label_list = []

                for x, y in val_loader:
                    # Move to device.
                    x = x.to(device)
                    
                    # Generate missingness.
                    # TODO this should be precomputed and shared across epochs
                    m = generate_uniform_mask(len(x), mask_size).to(device)

                    # Calculate prediction.
                    x_masked = mask_layer(x, m)
                    pred = model(x_masked)
                    pred_list.append(pred.cpu())
                    label_list.append(y.cpu())
                    
                # Calculate loss.
                y = torch.cat(label_list, 0)
                pred = torch.cat(pred_list, 0)
                val_loss = val_loss_fn(pred, y).item()
                
            
            # Print progress.
            if verbose:
                print(f'{"-"*8}Epoch {epoch+1}{"-"*8}')
                print(f'Val loss = {val_loss:.4f}\n')
                
            # Update scheduler.
            scheduler.step(val_loss)

            # Check if best model.
            if val_loss == scheduler.best:
                best_model = copy.deepcopy(model)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                
            # Early stopping.
            if num_bad_epochs > early_stopping_epochs:
                if verbose:
                    print(f'Stopping early at epoch {epoch+1}')
                break

        # Copy parameters from best model.
        restore_parameters(model, best_model)
    
    def forward(self, x, max_features, num_samples=128, verbose=False):
        '''
        Select features and make prediction.
        
        Args:
          x:
          max_features:
          num_samples:
          verbose:
        '''
        x_masked, _ = self.select_features(x, max_features, num_samples, verbose)
        return self.model(x_masked)
    
    def forward_multiple(self, x, num_features_list, num_samples=128, verbose=False):
        '''
        Select features and make prediction for multiple feature budgets.
        
        Args:
          x:
          num_features_list:
          num_samples:
          verbose:
        '''
        for num, x_masked, _ in self.select_features_multiple(x, num_features_list, num_samples, verbose):
            yield num, self.model(x_masked)
    
    def select_features(self, x, max_features, num_samples=128, verbose=False):
        '''
        Select features.
        
        Args:
          x:
          max_features:
          num_samples:
          verbose:
        '''
        # Set up model.
        model = self.model
        mask_layer = self.mask_layer
        data_sampler = self.data_sampler
        data_imputer = self.data_imputer
        device = next(model.parameters()).device
        
        # Set up mask.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            assert len(x.shape) == 2
            mask_size = x.shape[1]
        num_features = mask_size
        assert 0 < max_features < num_features
        m = torch.zeros((x.shape[0], mask_size), device=device)
        
        for i in tqdm(range(len(x))):
            # Get row.
            x_row = x[i:i+1]

            for k in range(max_features):
                # Setup.
                best_ind = None
                best_criterion = - np.inf
                m_row = m[i:i+1]
                x_masked = mask_layer(x_row, m_row)

                for j in range(num_features):
                    # Check if already included.
                    if m[i][j] == 1:
                        continue
                    
                    # Generate feature samples.
                    x_j = data_sampler.sample(x_masked, m_row, j, num_samples)
                    x_j = x_j.to(device)
                    
                    # Perform imputation.
                    x_expand = x_row.repeat(num_samples, 1)
                    x_expand = data_imputer.impute(x_expand, x_j, j)
                    m_expand = m_row.repeat(num_samples, 1)
                    m_expand[:, j] = 1
                    x_expand_masked = mask_layer(x_expand, m_expand)
                
                    # Make predictions.
                    with torch.no_grad():
                        preds = model(x_expand_masked)
                    
                    # Measure criterion.
                    criterion = calculate_criterion(preds, self.task)
                    if verbose:
                        print(f'Feature {j} criterion = {criterion:.4f}')
                    
                    # Check if best.
                    if criterion > best_criterion:
                        best_criterion = criterion
                        best_ind = j
                    
                # Select new feature.
                if verbose:
                    print(f'Selecting feature {best_ind}')
                m[i][best_ind] = 1

        # Apply mask.
        x_masked = mask_layer(x, m)
        return x_masked, m
    
    def select_features_multiple(self, x, num_features_list, num_samples=128, verbose=False):
        '''
        Select features for multiple budgets.
        
        Args:
          x:
          num_features_list:
          num_samples:
          verbose:
        '''
        # Set up model.
        model = self.model
        mask_layer = self.mask_layer
        data_sampler = self.data_sampler
        data_imputer = self.data_imputer
        device = next(model.parameters()).device
        
        # Set up mask.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            assert len(x.shape) == 2
            mask_size = x.shape[1]
        num_features = mask_size
        assert isinstance(num_features_list, (list, tuple, np.ndarray))
        assert 0 < max(num_features_list) < num_features
        assert min(num_features_list) > 0
        max_features = max(num_features_list)
        m = torch.zeros((x.shape[0], mask_size), device=device)
        
        for k in range(max_features):
            for i in range(len(x)):
                # Get row.
                x_row = x[i:i+1]
                m_row = m[i:i+1]
                x_masked = mask_layer(x_row, m_row)

                # Setup.
                best_ind = None
                best_criterion = - np.inf

                for j in range(num_features):
                    # Check if already included.
                    if m[i][j] == 1:
                        continue
                    
                    # Generate feature samples.
                    x_j = data_sampler.sample(x_masked, m_row, j, num_samples)
                    x_j = x_j.to(device)
                    
                    # Perform imputation.
                    x_expand = x_row.repeat(num_samples, 1)
                    x_expand = data_imputer.impute(x_expand, x_j, j)
                    m_expand = m_row.repeat(num_samples, 1)
                    m_expand[:, j] = 1
                    x_expand_masked = mask_layer(x_expand, m_expand)
                
                    # Make predictions.
                    with torch.no_grad():
                        preds = model(x_expand_masked)
                    
                    # Measure criterion.
                    criterion = calculate_criterion(preds, self.task)
                    if verbose:
                        print(f'Feature {j} criterion = {criterion:.4f}')
                    
                    # Check if best.
                    if criterion > best_criterion:
                        best_criterion = criterion
                        best_ind = j
                    
                # Select new feature.
                if verbose:
                    print(f'Selecting feature {best_ind}')
                m[i][best_ind] = 1
                
            # Yield current results if necessary.
            if (k + 1) in num_features_list:
                yield k + 1, mask_layer(x, m), m
    
    def evaluate(self,
                 loader,
                 max_features,
                 metric,
                 num_samples=128):
        '''
        Evaluate mean performance across a dataset.
        
        Args:
          loader:
          max_features:
          metric:
          num_samples:
        '''
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # For calculating mean loss.
        pred_list = []
        label_list = []

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device)

                # Calculate loss.
                pred = self.forward(x, max_features, num_samples)
                pred_list.append(pred.cpu())
                label_list.append(y.cpu())
        
            # Calculate metric(s).
            y = torch.cat(label_list, 0)
            pred = torch.cat(pred_list, 0)
            if isinstance(metric, (tuple, list)):
                score = [m(pred, y).item() for m in metric]
            elif isinstance(metric, dict):
                score = {name: m(pred, y).item() for name, m in metric.items()}
            else:
                score = metric(pred, y).item()
                
        return score
    
    def evaluate_multiple(self,
                          loader,
                          num_features_list,
                          metric,
                          num_samples=128):
        '''
        Evaluate mean performance across a dataset.
        
        Args:
          loader:
          num_features_list:
          metric:
          num_samples:
        '''
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # For calculating mean loss.
        pred_dict = {num: [] for num in num_features_list}
        score_dict = {num: None for num in num_features_list}
        label_list = []

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device)

                # Calculate loss.
                for num, pred in self.forward_multiple(x, num_features_list, num_samples):
                    pred_dict[num].append(pred.cpu())
                label_list.append(y.cpu())
        
            # Calculate metric(s).
            y = torch.cat(label_list, 0)
            for num in num_features_list:
                pred = torch.cat(pred_dict[num], 0)
                if isinstance(metric, (tuple, list)):
                    score = [m(pred, y).item() for m in metric]
                elif isinstance(metric, dict):
                    score = {name: m(pred, y).item() for name, m in metric.items()}
                else:
                    score = metric(pred, y).item()
                score_dict[num] = score
                
        return score_dict


class base_Infer(object):
    def __init__(self,model):
        self.model=model
    def sample_pos_W(self,**kwargs):
        raise NotImplementedError('Must override this method!')
    def sample_pos_Z(self,**kwargs):
        raise NotImplementedError('Must override this method!')
    def sample_X(self,**kwargs):
        raise NotImplementedError('Must override this method!')
    def compute_pos_W(self,**kwargs):
        raise NotImplementedError('Must override this method for last layer Gaussian model!')


class SGHMC(base_Infer):
    def __init__(self,model,Infer_name='SGHMC'):
        super(SGHMC,self).__init__(model)
        self.Infer_name=Infer_name
        self.flag_only_output_layer=self.model.decoder.flag_only_output_layer
        self.V,self.M,self.r=None,None,None # This is reserved for train_SGHMC() Method
        self.flag_LV=model.flag_LV
    def _SGHMC_step(self,**kwargs):
        W_dict=kwargs['W_dict']
        G_dict=kwargs['G_dict']
        eps=kwargs['eps']
        eps2=kwargs['eps2']
        M=kwargs['M']
        V=kwargs['V']
        r=kwargs['r']
        flag_optimizer=kwargs['flag_optimizer']
        counter=kwargs['counter']
        flag_compensator=kwargs['flag_compensator']
        update_noise=kwargs['update_noise']
        if flag_compensator:
            M_old=kwargs['M_old']
            #W_old=kwargs['W_old']

        # gradient descent
        for key_r,value_r in r.items():
            G_=G_dict[key_r]
            M_=M[key_r] # equivalent to M^-1
            V_=V[key_r]
            W_=W_dict[key_r]
            C_=0.1/(math.sqrt(eps))*torch.sqrt(V_)
            B_=0.5*math.sqrt(eps)*V_
            noise=torch.randn(value_r.shape)
            Diff_CB = torch.clamp((C_ - B_), min=0.)

            if flag_optimizer:
                V_.data = 0.99 * V_.data + 0.01 * (G_.data ** 2)
                M_.data = 1. / torch.sqrt((torch.sqrt(V_.data)))
                C_ = 0.1 / math.sqrt(eps)
                B_ = 0.5 * math.sqrt(eps)
                Diff_CB = C_ - B_
                rescaled_noise = torch.sqrt(2 * Diff_CB * math.sqrt(eps) * eps * M_.data**2) * noise
                value_r.data = value_r.data - eps * (M_.data ** 2) * G_.data - 0.1 * value_r.data + update_noise*rescaled_noise
                W_.data = W_.data + value_r.data

            else:
                V_.data = 0.99 * V_.data + 0.01 * (G_.data ** 2)
                M_.data = 1. / torch.sqrt((torch.sqrt(V_.data)))
                C_=0.1/math.sqrt(eps2)
                B_ = 0.5 * math.sqrt(eps2)
                Diff_CB=C_-B_

                if flag_compensator:
                    M_old_=M_old[key_r]
                    Gamma=(M_.data-M_old_.data)/(torch.sign(value_r.data)*torch.clamp(torch.abs(value_r.data),min=1e-7))
                rescaled_noise=torch.sqrt(2*Diff_CB*math.sqrt(eps2)*eps2*M_.data**2)*noise
                if flag_compensator:
                    value_r.data=value_r.data-eps2*(M_.data**2)*G_.data-0.1*value_r.data+1*rescaled_noise
                else:
                    value_r.data = value_r.data - eps2 * (
                                M_.data ** 2) * G_.data - 0.1 * value_r.data + 1 * rescaled_noise

                W_.data = W_.data + value_r.data

        return W_dict,r,V,M
    
    def _SGHMC_burnin_step(self,**kwargs):
        # No longer used for burn-in
        W_dict = kwargs['W_dict']
        G_dict = kwargs['G_dict']
        eps = kwargs['eps']
        M = kwargs['M']
        V = kwargs['V']
        r = kwargs['r']
        tau=kwargs['tau']
        flag_init=kwargs['flag_init']
        flag_update_tau=kwargs['flag_update_tau']
        if flag_init:
            g = copy.deepcopy(G_dict)
            V=copy.deepcopy(square_dict(G_dict))

        else:
            g=kwargs['g']
        # Update hyperparameter
        for key_r, value_r in r.items():
            g_=g[key_r]
            tau_=tau[key_r]
            G_=G_dict[key_r]
            V_=V[key_r]
            # Update g and V
            g_.data=g_.data-1./tau_*g_.data+1./tau_*G_.data
            V_.data=torch.clamp(V_.data-1./tau_*V_.data+1./(tau_)*((G_.data)**2),min=0.0001)
            if flag_update_tau:
                # Update window
                tau_.data=tau_.data-((g_.data)**2)/((V_.data))*tau_.data+1
                tau_.data=torch.clamp(tau_.data,min=1.01) # set minimum window size
                tau_.data.fill_(1.01)

        # Update the model
        for key_r,value_r in r.items():
            G_=G_dict[key_r]
            V_=V[key_r]
            M_ = 1. / torch.sqrt(V_)  # equivalent to M^-1
            M[key_r] = M_
            W_=W_dict[key_r]
            C_=0.1/(eps)*torch.sqrt(V_)
            B_=0.5*eps*V_
            noise=torch.randn(value_r.shape)
            Diff_CB=torch.clamp((C_-B_),min=0.)
            rescaled_noise=0*torch.sqrt(torch.clamp(2*(eps**3)/(V_)*C_-eps**4,min=0.))*noise#1*torch.sqrt(2*eps*Diff_CB) *noise

            value_r.data=value_r.data-0.1*value_r.data-G_.data
            W_.data=W_.data+0.003*M_.data*value_r.data

        return W_dict,r,M,V,tau,g

    def _init_mat(self):
        # This is used to initialize the M and V and r
        W_dict=self.model.decoder._get_W_dict()
        V=collections.OrderedDict()
        M=collections.OrderedDict()
        r=collections.OrderedDict()
        tau=collections.OrderedDict()
        g=collections.OrderedDict()
        if not self.flag_only_output_layer:
            for layer_ind in range(self.model.decoder.hidden_layer_num):
                V['weight_layer_%s'%(layer_ind)]=torch.ones(W_dict['weight_layer_%s'%(layer_ind)].shape)
                V['bias_layer_%s'%(layer_ind)]=torch.ones(W_dict['bias_layer_%s'%(layer_ind)].shape)
                M['weight_layer_%s'%(layer_ind)] = 1. / torch.sqrt(V['weight_layer_%s'%(layer_ind)])
                M['bias_layer_%s'%(layer_ind)] = 1. / (torch.sqrt(V['bias_layer_%s'%(layer_ind)]))
                r['weight_layer_%s'%(layer_ind)]=0.01*torch.randn(W_dict['weight_layer_%s'%(layer_ind)].shape)
                r['bias_layer_%s'%(layer_ind)]=0.01*torch.randn(W_dict['bias_layer_%s'%(layer_ind)].shape)
                tau['weight_layer_%s' % (layer_ind)] = 1.01 * torch.ones(W_dict['weight_layer_%s' % (layer_ind)].shape)
                tau['bias_layer_%s' % (layer_ind)] = 1.01 * torch.ones(W_dict['bias_layer_%s' % (layer_ind)].shape)
        # Output layer
        V['weight_out'] = torch.ones(W_dict['weight_out'].shape)
        V['bias_out'] = torch.ones(W_dict['bias_out'].shape)
        M['weight_out'] = 1. / torch.sqrt(V['weight_out'])
        M['bias_out'] = 1. / (torch.sqrt(V['bias_out']))
        r['weight_out'] = 0.01 * torch.randn(W_dict['weight_out'].shape)
        r['bias_out'] = 0.01 * torch.randn(W_dict['bias_out'].shape)
        tau['weight_out'] = 1.01 * torch.ones(W_dict['weight_out'].shape)
        tau['bias_out'] = 1.01 * torch.ones(W_dict['bias_out'].shape)
        if self.flag_LV:
            V['weight_out_LV'] = torch.ones(W_dict['weight_out_LV'].shape)
            V['bias_out_LV'] = torch.ones(W_dict['bias_out_LV'].shape)
            M['weight_out_LV'] = 1. / torch.sqrt(V['weight_out_LV'])
            M['bias_out_LV'] = 1. / (torch.sqrt(V['bias_out_LV']))
            r['weight_out_LV'] = 0.01 * torch.randn(W_dict['weight_out_LV'].shape)
            r['bias_out_LV'] = 0.01 * torch.randn(W_dict['bias_out_LV'].shape)
            tau['weight_out_LV'] = 1.01 * torch.ones(W_dict['weight_out_LV'].shape)
            tau['bias_out_LV'] = 1.01 * torch.ones(W_dict['bias_out_LV'].shape)

        g=copy.deepcopy(tau)

        return V,M,r,tau,g

    def sample_pos_W(self,observed_train,eps=0.01,tot_iter=500,thinning=50,flag_burnin=False,flag_reset_r=False,flag_hybrid=False,W_dict_init=None,**kwargs):
        # this is to draw samples of W (no longer used now, sample W is drawn using train_SGHMC method), this is only used as hyperparameter initialization
        if flag_hybrid:
            conditional_coef=kwargs['conditional_coef']
            target_dim=kwargs['target_dim']
        if flag_burnin:
            if W_dict_init is not None:
                W_dict=W_dict_init
            else:
                W_dict=self.model.decoder._get_W_dict() # No graph attached
        else:
            W_dict=kwargs['W_dict']
        W_sample=collections.OrderedDict()
        Drop_p=kwargs['Drop_p']
        if flag_burnin==True:
            V,M,r,tau,g=self._init_mat()
        else:
            M = kwargs['M']
            V = kwargs['V']
            r = kwargs['r']

        if flag_reset_r:
            _,_,r,_,_=self._init_mat()

        observed_train_size=observed_train.shape[0] # Assume the observed_train is after removing 0 row
        batch_size=int(observed_train_size/2.)
        if batch_size>100:
            batch_size=100
       # Define Data loader
        train_dataset = base_UCI_Dataset(observed_train, transform=None, flag_GPU=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator('cuda'))
        sigma_out=kwargs['sigma_out']
        iter=0
        Acc_ELBO=0
        sample_counter=0
        update_noise=kwargs['update_noise']
        while iter<=tot_iter:
            for idx, data in enumerate(train_loader):
                mask = get_mask(data)
                Drop_p_var = np.minimum(np.random.rand(1), Drop_p)
                mask_drop = np.array([bernoulli.rvs(1 - Drop_p_var, size=data.shape[1])] * data.shape[0])
                mask_drop = torch.from_numpy(mask_drop).float().cuda()

                if flag_hybrid:
                    # mask_drop_hybrid = torch.tensor(mask_drop.data)
                    mask_drop_hybrid = mask_drop.clone().detach()
                    mask_drop_hybrid[:, target_dim] = 1.  # Reserve the target dim
                    # mask_drop_Xdy = torch.tensor(mask_drop.data)
                    mask_drop_Xdy = mask_drop.clone().detach()
                    mask_drop_Xdy[:, -1] = 0.
                    mask_target_ELBO = mask * mask_drop_hybrid
                    mask = mask * mask_drop
                else:
                    mask = mask * mask_drop

                if (iter)%thinning==0 and not flag_burnin and iter>0:
                    flag_record=True
                else:
                    flag_record=False
                # Compute ELBO
                if flag_hybrid:
                    ELBO,ELBO_mean=self._compute_target_ELBO(data,mask,mask_target_ELBO,observed_train_size,W_dict,size_Z=10,sigma_out=sigma_out,z_sigma_prior=1.,W_mean_prior=0.,W_sigma_prior=1.,record_memory=flag_record,target_dim=target_dim,conditional_coef=conditional_coef)
                else:
                    ELBO,ELBO_mean=self._compute_ELBO(data,mask,observed_train_size,W_dict,size_Z=10,sigma_out=sigma_out,z_sigma_prior=1.,W_mean_prior=0.,W_sigma_prior=1.,record_memory=flag_record)

                Acc_ELBO +=ELBO_mean.cpu().data.numpy()

                # Obtain grad of ELBO
                (ELBO).backward()
                G_dict=self.model.decoder._get_grad_W_dict() # No graph attached

                # SGHMC update
                if not flag_burnin:
                    W_dict,r=self._SGHMC_step(W_dict=W_dict,G_dict=G_dict,eps=eps,M=M,V=V,r=r,update_noise=update_noise)
                else:
                    W_dict, r, M, V, tau, g=self._SGHMC_burnin_step(W_dict=W_dict,G_dict=G_dict,eps=eps,M=M,V=V,r=r,tau=tau,g=g,flag_init=(iter==0),flag_update_tau=(iter>50))

                # Update iter
                iter+=1
                if (iter)%thinning==0:
                    sample_counter+=1
                    # Store samples
                    if not flag_burnin:
                        W_sample['sample_%s'%(sample_counter)]=copy.deepcopy(W_dict)
                    Acc_ELBO=0.
                if iter>tot_iter:
                    break

        if flag_burnin:
            r,M,V=copy.deepcopy(r),copy.deepcopy(M),copy.deepcopy(V)
            return W_dict,r,M,V
        else:
            return W_sample,W_dict

    def completion(self,X,mask,W_sample,size_Z=10,record_memory=False):
        # Imputing all missing values
        X = X.clone().detach()
        with torch.no_grad():
            z, _ = self.model.sample_latent_variable(X, mask, size=size_Z)
        if record_memory==False:
            z=z.clone().detach()
        if self.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            X=self.sample_X(z,W_sample) # N_w x N_z x N x obs_dim
            del z
            return X

    def sample_X(self,z,W_sample):
        # z is N_z x N x latent
        counter=0
        for key_W,value_W in W_sample.items():
            self.model.decoder._assign_weight(value_W)
            # Compute the log likelihood
            # Decode
            decode = self.model.decoder.forward(z) # N_z x N x obs_dim
            # Remove memory
            decode=decode.clone().detach()
            if self.flag_LV:
                raise RuntimeError('flag_LV must be False')

            else:
                if counter==0:

                    decode=torch.unsqueeze(decode,dim=0) # 1 x N_z x N x obs_dim
                    X=decode
                else:
                    decode = torch.unsqueeze(decode, dim=0)  # 1 x N_z x N x obs_dim
                    X = torch.cat((X,decode),dim=0) # N_W x N_z x N x obs_dim
                counter+=1

        return X

    def _compute_ELBO(self,X,mask,data_N,W_dict,size_Z=10,sigma_out=0.1,z_sigma_prior=1.,W_mean_prior=0.,W_sigma_prior=1.,record_memory=False):
        num_X=X.shape[0]
        obs_dim=X.shape[1]
        # Sample Z_i
        z,encoding=self.model.sample_latent_variable(X,mask,size=size_Z) # N_z x N x latent_dim

        q_mean = encoding[:, :self.model.latent_dim]

        # q_sigma=F.softplus(encoding[:,self.latent_dim:]) # N x latent_dim
        if self.model.flag_log_q == True:
            q_sigma = torch.sqrt(torch.exp(encoding[:, self.model.latent_dim:]))
        else:
            q_sigma = torch.clamp(torch.sqrt(torch.clamp((encoding[:, self.model.latent_dim:]) ** 2,min=1e-8)),min=1e-4,max=30)

        if not record_memory:
            # Remove memory
            z=z.clone().detach()
            q_mean,q_sigma=q_mean.clone().detach(),q_sigma.clone().detach()

        #  No need to compute the KL term because no gradient w.r.t W
        # Compute the log likelihood term

        # Assign W_dict
        self.model.decoder._assign_weight(W_dict)
        # Compute the log likelihood
        # Decode
        if self.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            decode_X=self.model.decoder.forward(z) # Nz x N x obs_dim

            # Log likelihood
            log_likelihood=self._scaled_gaussian_log_likelihood(X, mask, decode_X, sigma_out,sigma_out_target_scale=1.) # N_z x N
            ELBO=log_likelihood-1.*torch.unsqueeze(self.model._KL_encoder(q_mean,q_sigma,z_sigma_prior),dim=0) # N_z x N
            ELBO=torch.sum(torch.mean(ELBO,dim=0) )/num_X*data_N+1*self._compute_W_prior(W_mean_prior=W_mean_prior,W_sigma_prior=W_sigma_prior) # TODO: Need to check if we only have prior, does it have gradient? Yes!
            ELBO_mean=ELBO/data_N
        return ELBO,ELBO_mean
    
    def _compute_target_ELBO(self,X,mask,mask_target_ELBO,data_N,W_dict,size_Z=10,sigma_out=0.1,z_sigma_prior=1.,W_mean_prior=0.,W_sigma_prior=1.,record_memory=False,target_dim=-1,conditional_coef=0.5):
        # ASSUME THE X CONTAIN THE TARGET VARIABLE and MASK IS FOR XUY!!!!! mask is for reconstruction mask_target_ELBO is for target loss
        num_X = X.shape[0]
        obs_dim=X.shape[1]
        # XUy = torch.tensor(X.data)  # with target variable
        XUy = X.clone().detach()
        # Xdy = torch.tensor(X.data)
        Xdy = X.clone().detach()
        X=X.clone().detach()

        Xdy[:, target_dim] = 0.  # zero the target dim
        # mask_Xdy = torch.tensor(mask_target_ELBO.data)
        mask_Xdy = mask_target_ELBO.clone().detach()
        mask_Xdy[:, target_dim] = 0.

        z_XUy, encoding_XUy = self.model.sample_latent_variable(XUy, mask_target_ELBO, size=size_Z)  # N_z x N x latent_dim
        z_Xdy, encoding_Xdy = self.model.sample_latent_variable(Xdy, mask_Xdy, size=size_Z)  # N_z x N x latent_dim

        z_X,encoding_X = self.model.sample_latent_variable(X, mask, size=size_Z)  # N_z x N x latent_dim

        q_mean_XUy = encoding_XUy[:, :self.model.latent_dim]
        q_mean_Xdy = encoding_Xdy[:, :self.model.latent_dim]
        q_mean_X = encoding_X[:, :self.model.latent_dim]

        # q_sigma=F.softplus(encoding[:,self.latent_dim:]) # N x latent_dim
        if self.model.flag_log_q == True:
            q_sigma_XUy = torch.sqrt(torch.exp(encoding_XUy[:, self.model.latent_dim:]))
            q_sigma_Xdy = torch.sqrt(torch.exp(encoding_Xdy[:, self.model.latent_dim:]))
            q_sigma_X = torch.sqrt(torch.exp(encoding_X[:, self.model.latent_dim:]))

        else:
            q_sigma_XUy = torch.clamp(torch.sqrt((encoding_XUy[:, self.model.latent_dim:]) ** 2), min=1e-5, max=30.)
            q_sigma_Xdy = torch.clamp(torch.sqrt((encoding_Xdy[:, self.model.latent_dim:]) ** 2), min=1e-5, max=30.)
            q_sigma_X = torch.clamp(torch.sqrt((encoding_X[:, self.model.latent_dim:]) ** 2), min=1e-5, max=30.)

        if not record_memory:
            # Remove memory
            z_XUy=z_XUy.clone().detach()
            z_Xdy = z_Xdy.clone().detach()
            z_X=z_X.clone().detach()

            q_mean_XUy,q_sigma_XUy,q_sigma_X=q_mean_XUy.clone().detach(),q_sigma_XUy.clone().detach(),q_sigma_X.clone().detach()
            q_mean_Xdy, q_sigma_Xdy,q_mean_X = q_mean_Xdy.clone().detach(), q_sigma_Xdy.clone().detach(),q_mean_X.clone().detach()

        q_mean_Xdy=q_mean_Xdy.clone().detach()
        q_sigma_Xdy=q_sigma_Xdy.clone().detach()


        KL_z_target = self.model._KL_encoder_target_ELBO(q_mean_XUy, q_sigma_XUy, q_mean_Xdy, q_sigma_Xdy)  # N
        KL_z=self.model._KL_encoder(q_mean_X,q_sigma_X,z_sigma_prior) # N

        # Assign W_dict
        self.model.decoder._assign_weight(W_dict)

        # Reconstruction loss
        if self.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:

            decode_X = self.model.decoder.forward(z_X)  # Nz x N x obs_dim
            # Log likelihood
            log_likelihood_recon = self._scaled_gaussian_log_likelihood(X, mask, decode_X, sigma_out,flag_target=False)  # N_z x N
            ELBO_recon = 1*log_likelihood_recon - 1*torch.unsqueeze(KL_z,
                                                    dim=0)  # N_z x N
            ELBO_recon = torch.sum(torch.mean(ELBO_recon, dim=0))/ num_X * data_N  + 1*self._compute_W_prior(W_mean_prior=W_mean_prior,
                                                                                               W_sigma_prior=W_sigma_prior)  # TODO: Need to check if we only have prior, does it have gradient? Yes!
            ELBO_mean_recon = ELBO_recon / data_N

        # Target loss
        mask_y = torch.zeros(mask.shape)
        mask_y[:, target_dim] = 1.  # Only reserve the target dim and disable all other

        if self.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:

            decode_target = self.model.decoder.forward(z_XUy)  # Nz x N x obs_dim

            log_likelihood_target = self._scaled_gaussian_log_likelihood(XUy, mask_y, decode_target, sigma_out,flag_target=True)  # N_z x N
            ELBO_target = log_likelihood_target - 1.*torch.unsqueeze(KL_z_target,
                                                    dim=0)  # N_z x N
            ELBO_target = torch.sum(torch.mean(ELBO_target, dim=0)) / num_X * data_N + 1*self._compute_W_prior(W_mean_prior=W_mean_prior,
                                                                                               W_sigma_prior=W_sigma_prior)
            ELBO_mean_target = ELBO_target / data_N

        # Hybrid model
        ELBO=conditional_coef*ELBO_recon+(1-conditional_coef)*ELBO_target
        ELBO_mean=ELBO_target
        return ELBO, ELBO_mean
    
    def _scaled_gaussian_log_likelihood(self,X,mask,decode,sigma_out,sigma_out_target_scale=1.,flag_target=False,**kwargs):
        if self.flag_LV:
            decode_LV=kwargs['decode_X_LV']
        # decode is Nz x N x obs_dim
        active_dim_scale=torch.clamp(torch.sum(torch.abs(get_mask(X))>0.,dim=1).float(),min=1) # N
        X = X * mask
        decoding_size = len(decode.shape)
        obs_dim=X.shape[1]
        if decoding_size==3: # Nz x N x obs_dim
            X_expand = torch.unsqueeze(X, dim=0)
            mask_expand = torch.unsqueeze(mask, dim=0)  # 1 x N x obs_dim

            if flag_target==False:
                active_dim=torch.clamp(torch.sum(torch.abs(mask)>0.,dim=1).float(),min=1) # N
            else:
                active_dim=1.

            decoding = decode * mask_expand  # N_z x N x obs_dim
            if self.flag_LV:
                raise RuntimeError('flag_LV must be False')
            else:
                # Apart from target
                X_expand_X=X_expand[:,:,0:-1]
                X_expand_Y=torch.unsqueeze(X_expand[:,:,-1],dim=2) # N_z x N x 1
                decoding_X=decoding[:,:,0:-1]
                decoding_Y=torch.unsqueeze(decoding[:,:,-1],dim=2) # N_z x N x 1
                log_likelihood= -0.5 * torch.sum((X_expand_X - decoding_X) ** 2 / (sigma_out ** 2),
                                                  dim=2)-0.5*torch.sum((X_expand_Y-decoding_Y)**2/((sigma_out*sigma_out_target_scale)**2),dim=2) - 0.5 * torch.unsqueeze(torch.sum(mask, dim=1), dim=0) * (
                                             math.log(sigma_out ** 2) + math.log(2 * np.pi)) # N_z x N

            # Scale it with |D/D_0|
            if flag_target==False:
                active_dim_scale_expand = torch.unsqueeze(active_dim_scale, dim=0)
                active_dim_expand=torch.unsqueeze(active_dim,dim=0) # 1 x N

                log_likelihood=log_likelihood/active_dim_expand*active_dim_scale_expand # N_z x N
            else:
                active_dim_scale_expand = torch.unsqueeze(active_dim_scale, dim=0)
                log_likelihood = log_likelihood#*obs_dim/active_dim_scale_expand #TODO: This is to maintain the relative scale of conditional model and reconstruction model

        else:
            raise NotImplementedError
        return log_likelihood

    def _compute_W_prior(self,W_mean_prior=0.,W_sigma_prior=1.):
        flat_weight,out_wight,out_bias=self.model.decoder._flatten_stat() # 1 x D_weight,N_in x N_out and N_out
        if self.model.decoder.flag_only_output_layer==False:
            # Prior of flat_weight
            log_prior_shared=torch.sum(-0.5/(W_sigma_prior**2)*((flat_weight-W_mean_prior)**2)-0.5*math.log(W_sigma_prior**2)-0.5*math.log(2*np.pi))
            # Compute out_weight prior
            log_prior_out_weight=torch.sum(-0.5/(W_sigma_prior**2)*((out_wight-W_mean_prior)**2)-0.5*math.log(W_sigma_prior**2)-0.5*math.log(2*np.pi))
            log_prior_out_bias=torch.sum(-0.5/(W_sigma_prior**2)*((out_bias-W_mean_prior)**2)-0.5*math.log(W_sigma_prior**2)-0.5*math.log(2*np.pi))
            log_prior=log_prior_shared+log_prior_out_weight+log_prior_out_bias
        else:
            # Compute out_weight prior
            log_prior_out_weight = torch.sum(
                -0.5 / (W_sigma_prior ** 2) * ((out_wight - W_mean_prior) ** 2) - 0.5 * math.log(
                    W_sigma_prior ** 2) - 0.5 * math.log(2 * np.pi))
            log_prior_out_bias = torch.sum(
                -0.5 / (W_sigma_prior ** 2) * ((out_bias - W_mean_prior) ** 2) - 0.5 * math.log(
                    W_sigma_prior ** 2) - 0.5 * math.log(2 * np.pi))
            log_prior = log_prior_out_weight + log_prior_out_bias

        return log_prior
    
    def test_log_likelihood(self,X_in,X_test,W_sample,mask,sigma_out,size=10):
        if self.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            with torch.no_grad():
                complete=self.completion(X_in,mask,W_sample,size_Z=size,record_memory=False)
            complete=complete.clone().detach()
            test_mask=get_mask(X_test)

            X_expand = torch.unsqueeze(torch.unsqueeze(X_test, dim=0), dim=0)  # 1 x 1 x N x obs_dim
            mask_expand = torch.unsqueeze(torch.unsqueeze(test_mask, dim=0), dim=0)  # 1 x 1 x N x obs_dim
            decoding = complete * mask_expand  # N_w x N_z x N x obs_dim
            with torch.no_grad():
            # Proper NLL
                log_likelihood = -0.5 * torch.sum((X_expand - decoding) ** 2 / (sigma_out ** 2),
                                              dim=3) - 0.5 * torch.unsqueeze(
                torch.unsqueeze(torch.sum(test_mask, dim=1), dim=0),
                dim=0) * (math.log(
                sigma_out ** 2) + math.log(2 * np.pi))  # N_w x N_z x N

                log_likelihood = log_likelihood.view(-1, log_likelihood.shape[2])  # (N_w x N_z) x N
                pred_log_likelihood = torch.logsumexp(log_likelihood, dim=0) - math.log(float(log_likelihood.shape[0]))  # N
                mean_pred_log_likelihood = 1. / (torch.sum(test_mask)) * torch.sum(pred_log_likelihood)
                tot_pred_ll = torch.sum(pred_log_likelihood)
            del complete
            del W_sample
        # return torch.tensor(mean_pred_log_likelihood.data), torch.tensor(tot_pred_ll.data)  # Clear Memory
        return mean_pred_log_likelihood.clone().detach(), tot_pred_ll.clone().detach()


class base_Active_Learning_SGHMC(object):
    def __init__(self,model,Infer_model,overall_data,sigma_out,Optim_settings,Adam_encoder,Adam_embedding,flag_clear_target_train,flag_clear_target_test,rs=10,model_name='SGHMC Active'):
        self.model=model
        self.Infer_model=Infer_model
        self.overall_data=overall_data
        self.random_seed=rs
        self.flag_clear_target_train=flag_clear_target_train
        self.flag_clear_target_test=flag_clear_target_test
        self.model_name=model_name
        self.sigma_out=sigma_out
        self.Optim_settings=Optim_settings
        self.Adam_encoder=Adam_encoder
        self.Adam_embedding=Adam_embedding
    
    def _get_pretrain_data(self,target_dim=-1,**kwargs):
        rs=self.random_seed
        pretrain_data_number=kwargs['pretrain_number']
        # Shuffle the train data
        self.train_data=shuffle(self.train_data,random_state=rs)
        _, pretrain_data, _, _ = train_test_split(self.train_data, self.train_data, test_size=pretrain_data_number,
                                                       random_state=rs, shuffle=False)

        self.train_data[self.train_data.shape[0]-pretrain_data.shape[0]:,:]=0.
        self.init_observed_train=np.zeros(self.train_data.shape) # Original size
        self.init_observed_train[self.train_data.shape[0]-pretrain_data.shape[0]:,:]=pretrain_data

        self.train_data_tensor=torch.from_numpy(self.train_data).float().cuda()
        self.pretrain_data_tensor=torch.from_numpy(pretrain_data).float().cuda()
        self.init_observed_train_tensor=torch.from_numpy(self.init_observed_train).float().cuda()
        self.train_pool_tensor=self.train_data_tensor.clone().detach()
        if self.flag_clear_target_train:
            self.train_pool_tensor[:,target_dim]=0. # Remove the target dim
    
    def _data_preprocess(self,target_dim=-1,test_missing=0.2,**kwargs):
        rs = self.random_seed
        test_size = kwargs['test_size']
        missing_prop = kwargs['missing_prop']

        # Split user first, then mask by certain proportion aligned with train data. Train data are further split by proportion for mimicing variable size training dataset.
        train_data, test_data, _, _ = train_test_split(self.overall_data, self.overall_data, test_size=test_size,
                                                       random_state=rs, shuffle=True)
        # Split to get validation dataset
        train_data,valid_data,_,_ = train_test_split(train_data,train_data,test_size=0.1,random_state=rs,shuffle=True)

        # Mask the train data
        train_mask, train_mask_other = base_generate_mask_incomplete(train_data, mask_prop=missing_prop+1e-5)
        valid_mask, valid_mask_other =base_generate_mask_incomplete(valid_data,mask_prop=missing_prop+1e-5)

        if self.flag_clear_target_train==True: # Reserve the target variable
            train_mask[:,target_dim]=1.
            train_mask_other[:,target_dim]=0.
            valid_mask[:,target_dim]=1.
            valid_mask_other[:,target_dim]=0.
            valid_mask_input=np.ones(valid_mask.shape)
            valid_mask_input[:,target_dim]=0.
            valid_mask_target=np.zeros(valid_mask.shape)
            valid_mask_target[:,target_dim]=1.
        else:
            valid_mask_input=copy.deepcopy(valid_mask)
            valid_mask_target=copy.deepcopy(valid_mask_other)

        self.train_data = train_data * train_mask
        self.train_other = train_data * train_mask_other
        self.valid_data=valid_data*valid_mask
        self.valid_other=valid_data*valid_mask_other

        if self.flag_clear_target_test==True:
            test_mask = np.ones(test_data.shape)
            test_mask[:, target_dim] = 0
            target_mask = np.zeros(test_data.shape)
            target_mask[:, target_dim] = 1
        else:
            # NO CLEAR TARGET
            test_mask, target_mask = base_generate_mask_incomplete(test_data, mask_prop=missing_prop+test_missing)
            _,valid_mask_target=base_generate_mask_incomplete(valid_data, mask_prop=missing_prop+test_missing)

        #######################
        # transform to tensor
        self.valid_data_input=valid_data*valid_mask_input
        self.valid_data_target = valid_data * valid_mask_target
        self.valid_data_tensor=torch.from_numpy(self.valid_data).float().cuda()
        self.valid_data_target_tensor=torch.from_numpy(self.valid_data_target).float().cuda()
        self.valid_data_input_tensor = torch.from_numpy(self.valid_data_input).float().cuda()

        self.test_input = test_data * test_mask
        self.test_target = test_data * target_mask
        self.test_target_tensor = torch.from_numpy(self.test_target).float().cuda()
        self.test_input_tensor = torch.from_numpy(self.test_input).float().cuda()
        self.train_data_tensor=torch.from_numpy(self.train_data).float().cuda()
        self.init_observed_train=np.zeros(self.train_data_tensor.shape)
        self.init_observed_train_tensor=torch.from_numpy(self.init_observed_train).float().cuda()
        # self.train_pool_tensor=torch.tensor(self.train_data_tensor.data)
        self.train_pool_tensor = self.train_data_tensor.clone().detach()
        if self.flag_clear_target_train==True:
            self.train_pool_tensor[:,target_dim]=0. # Remove the target variable

    def train_BNN(self,observed_train,eps=0.01,max_sample_size=20,tot_epoch=100,thinning=50,hyper_param_update=200,sample_int=1,flag_hybrid=False,flag_reset_optim=True,flag_imputation=False,W_dict_init=None,**kwargs):
        if flag_reset_optim==True:
            # Redefine the optimizer
            Adam_encoder = torch.optim.Adam(
                list(self.model.encoder_before_agg.parameters()) + list(self.model.encoder_after_agg.parameters()),
                lr=self.Optim_settings['lr_sghmc'],
                betas=(self.Optim_settings['beta1'], self.Optim_settings['beta2']),
                weight_decay=self.Optim_settings['weight_decay'])
            Adam_embedding = torch.optim.Adam([self.model.encode_embedding, self.model.encode_bias], lr=self.Optim_settings['lr_sghmc'],
                                              betas=(self.Optim_settings['beta1'], self.Optim_settings['beta2']),
                                              weight_decay=self.Optim_settings['weight_decay'])
        else:
            Adam_encoder = self.Adam_encoder
            Adam_embedding = self.Adam_embedding

        kwargs['Adam_encoder']=Adam_encoder
        kwargs['Adam_embedding']=Adam_embedding

        # Train the sghmc
        W_sample,_,_,_= train_SGHMC(self.Infer_model,observed_train,eps=eps,max_sample_size=max_sample_size,tot_epoch=tot_epoch,thinning=thinning,hyper_param_update=hyper_param_update,
                    sample_int=sample_int,flag_hybrid=flag_hybrid,flag_results=False,W_dict_init=W_dict_init,flag_imputation=flag_imputation,**kwargs
                    )
        return W_sample
    
    def random_select_test(self,**kwargs):
        # Test time random selection
        test_input = kwargs['test_input']  # If this is the first time run, this should be all initialized at 0
        pool_data_tensor = kwargs['pool_data_tensor']
        total_user = pool_data_tensor.shape[0]
        # random select
        idx_array = []
        for user_i in range(total_user):
            non_zero_idx = (torch.abs(pool_data_tensor[user_i, :]) > 0.).nonzero()  # 1D idx tensor

            select_idx = non_zero_idx[torch.randperm(len(non_zero_idx))[0]]  # random select 1 index
            test_input[user_i, select_idx] = pool_data_tensor[user_i, select_idx]
            # remove the pool
            pool_data_tensor[user_i, select_idx] = 0.
            idx_array.append(select_idx)
        return test_input,idx_array,pool_data_tensor
    
    def generate_active_learn_input_z_target_test(self,test_input,pool_candidate,target_candidate,flag_only_i=False,flag_same_pool=False,**kwargs):
        # test_input is the observed test_input, pool_candidate is the NW x Nz x N x obs after applying pool_mask, target_candidate is N_w x Nz x N x obs after applying target_candidate
        # flag_same_pool is to indicate whether for each user i, the pool has the same number of candidates (for fully observed pool, it is true, but if pool has some missing data, then this may not be true)
        # This flag is for efficient computation
        # flag_only_i indicate whether only generate for row i/slice i
        # This is used as an component for EDDI selection at test time

        if flag_only_i==True:
            active_input_o_target=kwargs['active_input_o_target']
        if flag_same_pool==False:
            slice=kwargs['slice']

        if flag_only_i==False:
            # decouple test_input
            test_input=test_input.clone().detach() # N x obs
            # Store N and obs
            N,obs=target_candidate.shape[2],target_candidate.shape[3]
            # Generate X_phi, X_o
            target_candidate_reshape=target_candidate.view(-1,target_candidate.shape[2],target_candidate.shape[3]) # tot x N x obs
            target_candidate_reshape=target_candidate_reshape.view(target_candidate_reshape.shape[0],-1) # tot x (N x obs)

            # Non-zero idx
            non_zero_idx=(torch.abs(target_candidate_reshape)>0.).nonzero() # 2D array
            non_zero_idx=non_zero_idx[:,1]
            non_zero_idx=torch.unique(non_zero_idx) # 1D array with size d_target

            total_sample_number = target_candidate_reshape.shape[0]

            non_zero_idx_expand = torch.unsqueeze(non_zero_idx, dim=0).repeat(total_sample_number,
                                                                                               1)  # tot x d_pool
            test_input_reshape=test_input.view(1,-1).repeat(total_sample_number,1) # tot x (Nxobs)

            # select index
            target_candidate_reshape_selected = torch.index_select(target_candidate_reshape, dim=1,
                                                                   index=non_zero_idx)  #TODO: Check (total) x d_target (Checked)


            active_input_o_target=test_input_reshape.scatter_(dim=1,index=non_zero_idx_expand,src=target_candidate_reshape_selected) # tot x (Nxobs)
            active_input_o_target=active_input_o_target.view(total_sample_number,N,obs) # TODO: Check tot x N x obs and Check if it does the desired behavior (Checked)

        if flag_same_pool==True:
            # Generate X_phi,X_o,X_i
            total_sample_number_pool=pool_candidate.shape[0]*pool_candidate.shape[1]
            d_active=torch.sum(torch.abs(pool_candidate[0,0,0,:])>0.).int()
            pool_candidate_reshape=pool_candidate.view(-1,N,obs) # tot x N x obs
            #print('d_active:%s'%(d_active))
            active_input_o_target_rep=torch.unsqueeze(active_input_o_target,dim=2).repeat(1,1,d_active,1) #TODO: Check if tot x N x d x obs

            non_zero_idx_pool=(torch.abs(pool_candidate_reshape)>0.).nonzero() # 3 eleement matrix
            non_zero_idx_pool=non_zero_idx_pool[:,1:] # 2D with N,pool_dim
            non_zero_idx_pool_reshape=non_zero_idx_pool[0:N*d_active,1:2].view(N,d_active) # TODO: Check if N x d (Checked)
            non_zero_idx_pool_reshape_final=torch.unsqueeze(torch.unsqueeze(non_zero_idx_pool_reshape,dim=0),dim=3).repeat(total_sample_number_pool,1,1,1) # TODO: Check if tot x N x d x 1

            pool_candidate_reshape_reshape=pool_candidate_reshape.view(total_sample_number_pool,-1) # tot x (N x obs)
            non_zero_idx_tmp=(torch.abs(pool_candidate_reshape_reshape)>0.).nonzero()
            non_zero_idx_tmp=torch.unique(non_zero_idx_tmp[:,1]) # 1 D array

            pool_candidate_reshape_reshape=torch.index_select(pool_candidate_reshape_reshape,dim=1,index=non_zero_idx_tmp) #TODO: Check tot x (N x d)
            pool_candidate_reshape_reshape=pool_candidate_reshape_reshape.view(-1,N,d_active)
            pool_candidate_reshape_reshape=torch.unsqueeze(pool_candidate_reshape_reshape,dim=3) # TODO: Check if tot x N x d x 1(Checked)
            active_input_o_target_i=active_input_o_target_rep.scatter_(dim=3,index=non_zero_idx_pool_reshape_final,src=pool_candidate_reshape_reshape) # TODO Check if does desired and shape tot x N x d x obs
            return active_input_o_target,active_input_o_target_i,non_zero_idx_pool_reshape
        else:
            active_input_o_target_slice=active_input_o_target[:,slice,:]
            pool_candidate_slice = pool_candidate.view(-1, pool_candidate.shape[2],
                                                       pool_candidate.shape[3])  # (N_zxN_W) x N x obs_dim
            pool_candidate_slice = torch.unsqueeze(pool_candidate_slice[:, slice, :],
                                                   dim=2)  # (total_sample_num) x obs_dim x 1
            non_zero_idx = (torch.abs(pool_candidate_slice) > 0.).nonzero()  # 3D tensor
            non_zero_idx_dim = non_zero_idx[:, 1]
            non_zero_idx_dim = torch.unique(non_zero_idx_dim)
            total_pool_size = non_zero_idx_dim.shape[0]
            total_sample_number = pool_candidate_slice.shape[0]
            # Non zero pool
            pool_candidate_slice = torch.index_select(pool_candidate_slice, dim=1,
                                                      index=non_zero_idx_dim)  # (total) x d_pool x 1
            # index array
            non_zero_idx_array = torch.unsqueeze(
                torch.unsqueeze(non_zero_idx_dim, dim=0).repeat(total_sample_number, 1),
                dim=2)  # N x d_pool x 1

            # replicate the active_input_target
            active_input_pool_target = torch.unsqueeze(active_input_o_target_slice, dim=1).repeat(1, total_pool_size,
                                                                                          1)  # tot x d x obs_dim

            active_input_pool_target = active_input_pool_target.scatter_(dim=2, index=non_zero_idx_array,
                                                                         src=pool_candidate_slice)  # total x d x obs_dim
            return active_input_o_target,active_input_pool_target,non_zero_idx_dim

    def generate_active_learn_input_z_i_test(self,test_input,pool_candidate,flag_same_pool=False,**kwargs):
        # Also used as a component for EDDI computation in test time
        if flag_same_pool==False:
            slice=kwargs['slice']
        # Decouple test_input
        test_input=test_input.clone().detach()
        if flag_same_pool==True:
            N,obs=test_input.shape[0],test_input.shape[1]
            d_active=torch.sum(torch.abs(pool_candidate[0,0,0,:])>0.).int()
            pool_candidate_reshape=pool_candidate.view(-1,N,obs) # tot x N x obs
            sample_tot=pool_candidate_reshape.shape[0]
            test_input_reshape=torch.unsqueeze(test_input,dim=1).repeat(1,d_active,1) # N x d x obs
            test_input_reshape=torch.unsqueeze(test_input_reshape,dim=0).repeat(sample_tot,1,1,1) # tot x N x d x obs

            non_zero_idx=(torch.abs(pool_candidate_reshape)>0.).nonzero()
            non_zero_idx=non_zero_idx[:,1:] # 2 D array
            non_zero_idx_reshape=non_zero_idx[0:N*d_active,1:2].view(N,d_active) # N x d_active
            non_zero_idx_reshape_final=torch.unsqueeze(torch.unsqueeze(non_zero_idx_reshape,dim=2),dim=0).repeat(sample_tot,1,1,1) # tot x N x d x 1

            pool_candidate_flat=pool_candidate_reshape.view(sample_tot,-1)
            non_zero_idx_tmp=(torch.abs(pool_candidate_flat)>0.).nonzero()
            non_zero_idx_tmp=torch.unique(non_zero_idx_tmp[:,1]) # 1 d array

            pool_candidate_select=torch.index_select(pool_candidate_flat,dim=1,index=non_zero_idx_tmp)
            pool_candidate_select=torch.unsqueeze(pool_candidate_select.view(sample_tot,N,d_active),dim=3) # tot x N x d x 1

            active_input_o_i=test_input_reshape.scatter_(dim=3,index=non_zero_idx_reshape_final,src=pool_candidate_select)

            return active_input_o_i,non_zero_idx_reshape

        else:
            # Generate the input for computing the BALD, slice indicate the user row number for test_input
            pool_candidate_slice = pool_candidate.view(-1, pool_candidate.shape[2],
                                                       pool_candidate.shape[3])  # (N_zxN_W) x N x obs_dim
            pool_candidate_slice = torch.unsqueeze(pool_candidate_slice[:, slice, :],
                                                   dim=2)  # (total_sample_num) x obs_dim x 1
            non_zero_idx = (torch.abs(pool_candidate_slice) > 0.).nonzero()  # 3D tensor
            non_zero_idx_dim = non_zero_idx[:, 1]
            non_zero_idx_dim = torch.unique(non_zero_idx_dim)  # 1D Tensor
            total_pool_size = non_zero_idx_dim.shape[0]
            total_sample_number = pool_candidate_slice.shape[0]
            # Non zero pool
            pool_candidate_slice = torch.index_select(pool_candidate_slice, dim=1,
                                                      index=non_zero_idx_dim)  # (total) x d_pool x 1
            # index array
            non_zero_idx_array = torch.unsqueeze(
                torch.unsqueeze(non_zero_idx_dim, dim=0).repeat(total_sample_number, 1), dim=2)  # N x d_pool x 1
            # replicate the test_input
            test_input_slice = test_input[slice, :]  # obs_dim
            test_input_slice = torch.unsqueeze(torch.unsqueeze(test_input_slice, dim=0), dim=0).repeat(
                total_sample_number, total_pool_size, 1)

            active_input = test_input_slice.scatter_(dim=2, index=non_zero_idx_array,
                                                     src=pool_candidate_slice)  # total x d x obs_dim

            return active_input, non_zero_idx_dim

    def active_learn_target_test(self,**kwargs):
        # EDDI computation and selection
        flag_same_pool=kwargs['flag_same_pool']

        test_input = kwargs['test_input']
        pool_data_tensor = kwargs['pool_data_tensor']
        target_data_tensor = kwargs['target_data_tensor']
        size_z=kwargs['size_z']
        W_sample=kwargs['W_sample']
        split=kwargs['split']
        test_input_orig=test_input.clone().detach()
        if split>1 and split<test_input.shape[0]:
            batch_size = int(math.ceil((test_input.shape[0] / split)))
            pre_idx = 0
            counter_idx = 0
            for idx in range(split + 1):
                idx = min((idx + 1) * batch_size, test_input.shape[0])
                if pre_idx == idx:
                    break

                data_input = test_input[pre_idx:idx, :]
                data_pool=pool_data_tensor[pre_idx:idx, :]
                data_target=target_data_tensor[pre_idx:idx, :]

                data_input_mask = get_mask(data_input)

                # Sample X_phi and X_id
                z, _ = self.model.sample_latent_variable(data_input, data_input_mask, size=size_z)  # size_z x N x latent
                if self.Infer_model.flag_LV:
                    decode,_=self.Infer_model.sample_X(z,W_sample)
                else:
                    decode = self.Infer_model.sample_X(z,W_sample) # N_w x Nz x N x obs

                # Remove memory of decode and z
                decode = decode.clone().detach()
                z = z.clone().detach()

                target_mask = get_mask(data_target)  # N x obs_dim

                target_mask_expand = torch.unsqueeze(torch.unsqueeze(target_mask, dim=0), dim=0)  # 1 x 1 x N x obs_dim

                target_candidate = target_mask_expand * decode  # N_w x N_z x N x obs_dim

                pool_mask = get_mask(data_pool)
                pool_mask_expand = torch.unsqueeze(torch.unsqueeze(pool_mask, dim=0), dim=0)  # 1 x 1 x N x obs_dim
                pool_candidate = pool_mask_expand * decode  # N_w x N_z x N x obs_dim
                total_user = pool_candidate.shape[2]

                if flag_same_pool:
                    # Efficient computation, compute values in parallel for all users
                    # sample (X_o,X_phi),(X_o,X_target),(X_o,X_target,X_i)
                    active_input_o_target,active_input_o_i_target,non_zero_idx=self.generate_active_learn_input_z_target_test(data_input,pool_candidate,target_candidate,flag_only_i=False,flag_same_pool=True)
                    active_input_o_i,non_zero_idx=self.generate_active_learn_input_z_i_test(data_input,pool_candidate,flag_same_pool=True)

                    mask_active_target = get_mask(active_input_o_target)  # tot x N x obs
                    mask_active_pool_target = get_mask(active_input_o_i_target)  # tot x N x d x obs_dim
                    mask_active = get_mask(active_input_o_i) # tot x N x d x obs

                    encoding_target = self.model._encoding(active_input_o_target,
                                                           mask_active_target)  # total x N x 2*latent
                    encoding_pool_target = self.model._encoding(active_input_o_i_target,
                                                                mask_active_pool_target)  # tot x N x d x 2*latent
                    encoding_o_i = self.model._encoding(active_input_o_i, mask_active)  # total x N x d x 2*latent

                    encoding_o= self.model._encoding(data_input, data_input_mask) # N x 2*latent

                    # clear memory
                    encoding_target = encoding_target.clone().detach()
                    encoding_pool_target = encoding_pool_target.clone().detach()
                    encoding_o_i = encoding_o_i.clone().detach()
                    encoding_o=encoding_o.clone().detach()

                    encoding_target = torch.unsqueeze(encoding_target, dim=2).repeat(1, 1,encoding_pool_target.shape[2],
                                                                                     1)  # tot x N x d x 2*latent
                    KL_2 = self._gaussian_KL_all(encoding_pool_target, encoding_target)  # tot x N x d

                    encoding_o=torch.unsqueeze(encoding_o,dim=0).repeat(encoding_o_i.shape[0],1,1)
                    encoding_o=torch.unsqueeze(encoding_o,dim=2).repeat(1,1,encoding_o_i.shape[2],1) # tot  x N x d x 2*latent
                    KL_1=self._gaussian_KL_all(encoding_o_i, encoding_o)  # tot x N x d

                    loss=KL_1-KL_2
                    mean_loss = torch.mean(loss, dim=0)  # N x d

                    # Take the maximum value
                    _,idx_max=torch.max(mean_loss,dim=1)

                    # original index
                    idx_selected=non_zero_idx[np.arange(non_zero_idx.shape[0]),idx_max] # N
                    
                    # update data_input
                    test_input[torch.arange(pre_idx,idx), idx_selected]=pool_data_tensor[torch.arange(pre_idx,idx), idx_selected]
                    # update the pool by removing the selected ones
                    pool_data_tensor[torch.arange(pre_idx,idx), idx_selected] = 0.

                else:
                    # less efficient computation, compute EDDI for single row at a time
                    active_input_o_target, _,_= self.generate_active_learn_input_z_target_test(
                        data_input, pool_candidate, target_candidate, flag_only_i=False, flag_same_pool=False,slice=0) # what ever slice is fine

                    for user_i in range(total_user):
                        _, active_input_o_i_target, non_zero_idx = self.generate_active_learn_input_z_target_test(
                            data_input, pool_candidate, target_candidate, flag_only_i=True, flag_same_pool=False,active_input_o_target=active_input_o_target,slice=user_i)
                        active_input_o_i, non_zero_idx = self.generate_active_learn_input_z_i_test(data_input,
                                                                                                   pool_candidate,
                                                                                                   flag_same_pool=False,slice=user_i)
                        active_input_o_target_slice=active_input_o_target[:,user_i,:]

                        mask_active_target = get_mask(active_input_o_target_slice)  # tot  x obs
                        mask_active_pool_target = get_mask(active_input_o_i_target)  # tot x d x obs_dim
                        mask_active = get_mask(active_input_o_i)  # tot x d x obs

                        encoding_target = self.model._encoding(active_input_o_target_slice,
                                                               mask_active_target)  # total x 2*latent
                        encoding_pool_target = self.model._encoding(active_input_o_i_target,
                                                                    mask_active_pool_target)  # tot  x d x 2*latent
                        encoding_o_i = self.model._encoding(active_input_o_i, mask_active)  # total  x d x 2*latent

                        encoding_o = self.model._encoding(torch.unsqueeze(data_input[user_i,:],dim=0), torch.unsqueeze(data_input_mask[user_i,:],dim=0))  # 1 x 2*latent

                        # clear memory
                        encoding_target = encoding_target.clone().detach()
                        encoding_pool_target = encoding_pool_target.clone().detach()
                        encoding_o_i = encoding_o_i.clone().detach()
                        encoding_o = encoding_o.clone().detach()

                        encoding_target = torch.unsqueeze(encoding_target, dim=1).repeat(1,
                                                                                         encoding_pool_target.shape[1],
                                                                                         1)  # tot x d x 2*latent
                        KL_2 = self._gaussian_KL(encoding_pool_target, encoding_target)  # tot x d

                        encoding_o = torch.unsqueeze(encoding_o, dim=0).repeat(encoding_o_i.shape[0], 1, 1) # tot x 1 x latent
                        encoding_o = encoding_o.repeat(1, encoding_o_i.shape[1],
                                                                               1)  # tot x d x 2*latent
                        KL_1 = self._gaussian_KL(encoding_o_i, encoding_o)  # tot  x d

                        loss = KL_1 - KL_2
                        mean_loss = torch.mean(loss, dim=0)  # d

                        _, idx_max = torch.max(mean_loss, dim=0)

                        # original index
                        idx_selected = non_zero_idx[idx_max]  # N
                        # update data_input
                        test_input[counter_idx, idx_selected] = pool_data_tensor[counter_idx, idx_selected]
                        # update the pool by removing the selected ones
                        pool_data_tensor[counter_idx, idx_selected] = 0.
                        counter_idx += 1
                pre_idx=idx

        else:
            raise NotImplementedError

        # Selection stat (Debug Purpose )
        choice_now=get_choice(test_input)
        choice_orig=get_choice(test_input_orig)
        choice_stat=choice_now-choice_orig
        return test_input,choice_stat,pool_data_tensor

    def _gaussian_KL(self,encoding1,encoding2):
        # assert if the shape matches, normally is tot x d x 2*latent
        assert encoding1.shape==encoding2.shape, ' Inconsistent encoding shapes'
        mean1,mean2=encoding1[:,:,0:self.model.latent_dim],encoding2[:,:,0:self.model.latent_dim] # tot x d x latent
        if self.model.flag_log_q==True:
            sigma1, sigma2 = torch.sqrt(torch.exp(encoding1[:, :, self.model.latent_dim:])), torch.sqrt(torch.exp(
                encoding2[:, :, self.model.latent_dim:]))  # tot x d x latent
        else:
            sigma1,sigma2=torch.sqrt(encoding1[:,:,self.model.latent_dim:]**2),torch.sqrt(encoding2[:,:,self.model.latent_dim:]**2) # tot x d x latent
        KL=torch.log(sigma2/sigma1)+(sigma1**2+(mean1-mean2)**2)/(2*sigma2**2)-0.5 # tot x d x latent
        KL=torch.sum(KL,dim=2) # tot x d
        return KL

    def _gaussian_KL_all(self,encoding1,encoding2):
        # assert if the shape matches, normally is tot x N x d x 2*latent
        assert encoding1.shape==encoding2.shape, ' Inconsistent encoding shapes'
        mean1,mean2=encoding1[:,:,:,0:self.model.latent_dim],encoding2[:,:,:,0:self.model.latent_dim] # tot x N x d x latent
        if self.model.flag_log_q==True:
            sigma1, sigma2 = torch.sqrt(torch.exp(encoding1[:,:, :, self.model.latent_dim:])), torch.sqrt(torch.exp(
                encoding2[:,:, :, self.model.latent_dim:]))  # tot x N x d x latent
        else:
            sigma1,sigma2=torch.sqrt(encoding1[:,:,:,self.model.latent_dim:]**2),torch.sqrt(encoding2[:,:,:,self.model.latent_dim:]**2) # tot x N x d x latent
        KL=torch.log(sigma2/sigma1)+(sigma1**2+(mean1-mean2)**2)/(2*sigma2**2)-0.5 # tot x N x d x latent
        KL=torch.sum(KL,dim=3) # tot x N x d
        return KL


class base_Active_Learning_SGHMC_Decoder(base_Active_Learning_SGHMC):
    def __init__(self,*args,**kwargs):
        super(base_Active_Learning_SGHMC_Decoder, self).__init__(*args, **kwargs)
    def get_target_variable(self,observed_train,observed_train_before,target_dim,train_data=None):
        observed_train=observed_train.clone().detach()
        diff_observed=torch.abs(observed_train-observed_train_before) # N x out_dim
        sum_diff=torch.sum(diff_observed,dim=1) # N
        # Apply the target variable
        if train_data is None:
            observed_train[sum_diff>0.,target_dim]=self.train_data_tensor[sum_diff>0.,target_dim] #Assume train_data_tensor has the same arrangement as observed_train
        else:
            observed_train[sum_diff > 0., target_dim] = train_data[sum_diff > 0., target_dim]
        return observed_train.clone().detach()
    def _transform_idx(self,idx,N,obs,num_unobserved,step):
        num_observed=idx.shape[1]-num_unobserved
        counter = 0
        if num_observed==0:
            idx=[]
            return [],True
        else:
            idx=idx[0,:num_observed] #remove the unobserved
            current_size=idx.shape[0]
            if current_size<=step:
                idx_selected=idx
            else:
                idx_selected=idx[0:step]
            row = (idx_selected / obs).view(-1, 1)
            column = (idx_selected % obs).view(-1, 1)
            return torch.cat((row, column), dim=1), False
    def _transform_idx_pure(self,idx,obs):
        # similar to _transform_idx but used as different components in different functions
        row = (idx / obs).view(-1, 1)
        column = (idx % obs).view(-1, 1)
        return torch.cat((row, column), dim=1)

    def _apply_selected_idx(self,observed_train,pool_data,idx):
        idx = idx.long()
        for i in range(idx.shape[0]):
            row=idx[i,0].item()
            column=idx[i,1].item()
            #Assign to observed train
            observed_train[row,column]=pool_data[row,column]
            # Remove pool_data
            pool_data[row,column]=0.
        return observed_train,pool_data
    
    def _H_Xid_Xo(self,decode,comp_mean,sigma_out,flag_reduce_size=False,**kwargs):
        # Compute H[p(x_id|x_o)]
        if flag_reduce_size==True:
            perm1, perm2 = torch.randperm(decode.shape[0])[0:10], torch.randperm(decode.shape[1])[0:5]

            decode = reduce_size(decode, perm1, perm2)  # red_w x red_z x N x obs
            comp_mean=reduce_size(comp_mean,perm1,perm2)
        N=decode.shape[2]
        obs_dim=decode.shape[3]
        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
        decode=decode.view(-1,N,obs_dim) # (N_W x N_z) x N x obs

        comp_mean=comp_mean.view(-1,N,obs_dim) #(N_w x N_z) x N x obs

        decode_exapnd=torch.unsqueeze(decode,dim=1) #  (NwNz) x 1 x N x obs
        comp_mean_expand=torch.unsqueeze(comp_mean,dim=0).repeat(comp_mean.shape[0],1,1,1)# (NwNz) x (NwNz) x N x obs

        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            # Compute log likeihood
            log_likelihood=-0.5*math.log(2*np.pi)-0.5*math.log(sigma_out**2)-1./(2*sigma_out**2)*(decode_exapnd-comp_mean_expand)**2 # (NwNz) x (NwNz) x N x obs
        # Compute logsumexp
        marginal_log_likelihood=torch.logsumexp(log_likelihood,dim=1)-math.log(float(decode_exapnd.shape[0])) # (Nw_Nz) x N x obs
        entropy= -torch.mean(marginal_log_likelihood,dim=0) # N x obs
        # Clear memory
        entropy=torch.tensor(entropy.data)
        return entropy

    def _H_X_id_W_Xo(self,decode,comp_mean,sigma_out,flag_reduce_size=False,**kwargs): # Compute H[p(X_id|X_o,W)]
        # decode has shape nw x nz x n x obs
        # Reshape
        if flag_reduce_size==True:
            perm1, perm2 = torch.randperm(decode.shape[0])[0:10], torch.randperm(decode.shape[1])[0:5]

            decode = reduce_size(decode, perm1, perm2)  # red_w x red_z x N x obs
            comp_mean = reduce_size(comp_mean, perm1, perm2)

        N = decode.shape[2]
        obs_dim = decode.shape[3]
        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')

        decode_exapnd = torch.unsqueeze(decode, dim=2)  # N_w x N_z x 1 x N x obs
        comp_mean_expand = torch.unsqueeze(comp_mean, dim=1).repeat(1, comp_mean.shape[1], 1,
                                                                    1,1)  # N_W x N_z x N_z x N x obs
        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            # Compute log likelihood
            log_likelihood=-0.5*math.log(2*np.pi)-0.5*math.log(sigma_out**2)-1./(2*sigma_out**2)*(decode_exapnd-comp_mean_expand)**2 # N_w x N_z x N_z x N x obs

        # Compute logsumexp
        marginal_log_likelihood=torch.logsumexp(log_likelihood,dim=2)-math.log(float(decode_exapnd.shape[1])) # N_w x N_z x N x obs
        entropy=-torch.mean(marginal_log_likelihood,dim=1) # N_w x N x obs
        entropy=torch.tensor(entropy.data)
        return entropy

    def _H_X_Phi_O_id(self, decode, observed_data_row, W_sample,row_idx,target_dim=-1):
        # Remove memory
        observed_data_row=observed_data_row.clone().detach() # N x obs
        decode=decode.clone().detach() # N_w x N_z x N x obs
        decode=decode[:,:,row_idx,:] # N_w x N_z x obs
        perm1, perm2 = torch.randperm(decode.shape[0])[0:5], torch.randperm(decode.shape[1])[0:5]
        decode = reduce_size(decode, perm1, perm2,flag_one_row=True)  # red_w x red_z  x obs
        decode=decode.view(-1,decode.shape[-1]) # tot x obs
        tot=decode.shape[0]
        # Get vacance location
        idx = (torch.abs(observed_data_row[0:-1]) == 0).nonzero().view(-1)  # N_id (get rid of target variable)
        if idx.shape[0] == 0:
            return 0 * torch.ones(observed_data_row.shape)
        size_idx=idx.shape[0] # N_pool
        observed_data_row_dup=torch.unsqueeze(torch.unsqueeze(observed_data_row,dim=0),dim=0).repeat(tot,size_idx,1) # tot x N_pool x obs
        idx_dup=torch.unsqueeze(torch.unsqueeze(idx,dim=0).repeat(tot,1),dim=2) # tot x N_pool x 1
        mask_candidate=1-get_mask(observed_data_row)
        mask_candidate[-1]=0
        decode=decode*torch.unsqueeze(mask_candidate,dim=0) # tot x obs with zero's
        decode=torch.t(remove_zero_row_2D(torch.t(decode))) # tot x N_pool
        decode_dup=torch.unsqueeze(decode,dim=2) # tot x N_pool x 1
        observed_data_row_dup.scatter_(2,idx_dup,decode_dup) # tot x N_pool x obs

        # Sample X_phi
        observed_data_row_dup = observed_data_row_dup.view(-1, observed_data_row_dup.shape[2])  # (totxN_pool) x obs
        # Sample z
        mask_observed_data_exp = get_mask(observed_data_row_dup)
        z, _ = self.model.sample_latent_variable(observed_data_row_dup, mask_observed_data_exp,
                                                 size=10)  # size_z x (totxN_pool) x latent

        # Transform back
        z = z.view(z.shape[0], tot, size_idx, z.shape[2])  # size_z x tot x N_pool x latent
        z = z.clone().detach()

        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:

            decode_ = self.Infer_model.sample_X(z, W_sample)  # N_w x Nz x tot x N_pool x obs
        decode_=decode_.clone().detach()
        # Reduce sample size to save memory
        perm1, perm2 = torch.randperm(decode_.shape[0])[0:10], torch.randperm(decode_.shape[1])[0:5]

        decode_ = reduce_size(decode_, perm1, perm2)  # red_w x red_z x tot x N_pool x obs
        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
        decode_phi=decode_[:,:,:,:,target_dim] # red_w x red_z x tot x N_pool

        decode_phi=decode_phi.view(-1,decode_phi.shape[2],decode_phi.shape[3]) # (w x z) x tot x N_pool

        comp_mean=torch.unsqueeze(decode_phi,dim=0).repeat(decode_phi.shape[0],1,1,1) # (wxz) x (wxz) x tot x N_pool
        decode_phi_exp=torch.unsqueeze(decode_phi,dim=1) # (wxz) x 1 x tot x N_pool

        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            log_likelihood = -0.5 * math.log(2 * np.pi) - 0.5 * math.log(self.sigma_out ** 2) - 1. / (
                2 * self.sigma_out ** 2) * (
                                 decode_phi_exp - comp_mean) ** 2  # (wxz) x (wxz) x tot x N_pool
        marginal_log_likelihood = torch.logsumexp(log_likelihood, dim=1) - math.log(
            float(decode_phi_exp.shape[0]))  # (w x z) x tot x N_pool
        entropy = -torch.mean(marginal_log_likelihood, dim=0)  # tot x N_pool
        E_entropy_value = torch.mean(entropy,dim=0) # N_pool
        E_entropy = 0 * torch.ones(observed_data_row.shape)  # obs
        E_entropy[idx] = E_entropy_value  # obs
        return E_entropy
    
    def _H_X_Phi_W_id_O(self,decode,observed_data_row,W_sample,row_idx,target_dim=-1):
        # Remove memory
        observed_data_row = observed_data_row.clone().detach()  # N x obs
        decode = decode.clone().detach()  # N_w x N_z x N x obs
        decode = decode[:, :, row_idx, :]  # N_w x N_z x obs
        perm1, perm2 = torch.randperm(decode.shape[0])[0:5], torch.randperm(decode.shape[1])[0:5]
        decode = reduce_size(decode, perm1, perm2, flag_one_row=True)  # red_w x red_z  x obs
        decode = decode.view(-1, decode.shape[-1])  # tot x obs
        tot = decode.shape[0]
        # Get vacance location
        idx = (torch.abs(observed_data_row[0:-1]) == 0).nonzero().view(-1)  #  N_id (get rid of target variable)
        if idx.shape[0] == 0:
            # just set a large value to aviod being selected
            return 10e5 * torch.ones(observed_data_row.shape)

        size_idx = idx.shape[0]  # N_pool
        observed_data_row_dup = torch.unsqueeze(torch.unsqueeze(observed_data_row, dim=0), dim=0).repeat(tot, size_idx,
                                                                                                         1)  # tot x N_pool x obs
        idx_dup = torch.unsqueeze(torch.unsqueeze(idx, dim=0).repeat(tot,1),dim=2)  # tot x N_pool x 1
        mask_candidate = 1 - get_mask(observed_data_row)
        mask_candidate[-1] = 0
        decode = decode * torch.unsqueeze(mask_candidate, dim=0)  # tot x obs with zero's
        decode = torch.t(remove_zero_row_2D(torch.t(decode)))  # tot x N_pool
        decode_dup = torch.unsqueeze(decode, dim=2)  # tot x N_pool x 1
        observed_data_row_dup.scatter_(2, idx_dup, decode_dup)  # tot x N_pool x obs
        # Sample X_phi
        observed_data_row_dup = observed_data_row_dup.view(-1, observed_data_row_dup.shape[2])  # (totxN_pool) x obs
        # Sample z
        mask_observed_data_exp = get_mask(observed_data_row_dup)
        z, _ = self.model.sample_latent_variable(observed_data_row_dup, mask_observed_data_exp,
                                                 size=10)  # size_z x (totxN_pool) x latent

        # Transform back
        z = z.view(z.shape[0], tot, size_idx, z.shape[2])  # size_z x tot x N_pool x latent
        z = z.clone().detach() # P(z|X_id,X_O)

        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            decode_ = self.Infer_model.sample_X(z, W_sample)  # N_w x Nz x tot x N_pool x obs
        decode_ = decode_.clone().detach()
        # Reduce sample size
        perm1, perm2 = torch.randperm(decode_.shape[0])[0:20], torch.randperm(decode_.shape[1])[0:5]

        decode_ = reduce_size(decode_, perm1, perm2)  # red_w x red_z x tot x N_pool x obs

        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')

        decode_phi = decode_[:, :, :, :, target_dim]  # red_w x red_z x tot x N_pool

        comp_mean = torch.unsqueeze(decode_phi, dim=1).repeat(1,decode_phi.shape[1], 1, 1,
                                                              1)  # W x z x z x tot x N_pool
        decode_phi_exp = torch.unsqueeze(decode_phi, dim=2)  # W x z x 1 x tot x N_pool

        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            log_likelihood = -0.5 * math.log(2 * np.pi) - 0.5 * math.log(self.sigma_out ** 2) - 1. / (
                2 * self.sigma_out ** 2) * (
                                 decode_phi_exp - comp_mean) ** 2  # w x z x z x tot x N_pool

        marginal_log_likelihood = torch.logsumexp(log_likelihood, dim=2) - math.log(
            float(decode_phi_exp.shape[1]))  # w x z x tot x N_pool

        entropy = -torch.mean(marginal_log_likelihood, dim=1)  # w x tot x N_pool
        E_entropy_value = torch.mean(torch.mean(entropy, dim=0),dim=0)  # N_pool
        E_entropy = 10e5 * torch.ones(observed_data_row.shape)  # obs
        E_entropy[idx] = E_entropy_value  # obs
        return E_entropy

    def base_active_learning_decoder(self,coef=0.999,balance_prop=0.25,temp_scale=0.25,flag_initial=False,split=50,strategy='Alpha',Select_Split=True,**kwargs):
        # balance_prop is the % that we selected from the already observed users
        observed_train=kwargs['observed_train']
        pool_data=kwargs['pool_data']
        step=kwargs['step'] #This is to define how many data points to select for each active learning
        sigma_out=kwargs['sigma_out']
        W_sample=kwargs['W_sample']
        sample_z_num = 10

        selection_scheme=kwargs['selection_scheme']

        if selection_scheme=='overall':
            print('%s selection scheme with Select_Split %s'%(selection_scheme,Select_Split))
        else:
            raise NotImplementedError

        if split>1 and split<pool_data.shape[0]:
            # split is to reduce the memory consumption
            batch_size = int(math.ceil(pool_data.shape[0] / split))
            pre_idx = 0
            counter=0
            for counter_idx in range(split + 1):
                idx = min((counter_idx + 1) * batch_size, pool_data.shape[0])
                if pre_idx == idx:
                    # finished computing BALD value for all pool data
                    break

                pool_data_split=pool_data[pre_idx:idx,:]
                observed_train_split=observed_train[pre_idx:idx,:]
                observed_train_mask=get_mask(observed_train_split)

                # Sample X_id
                z, _ = self.model.sample_latent_variable(observed_train_split, observed_train_mask,
                                                         size=sample_z_num)  # size_z x N_pN x latent

                if self.Infer_model.flag_LV:
                    raise RuntimeError('flag_LV must be False')
                else:
                    decode = self.Infer_model.sample_X(z, W_sample)  # N_w x Nz x N x obs

                # Remove memory
                z=z.clone().detach()
                decode=decode.clone().detach()

                if strategy=='Opposite_Alpha':
                    if self.Infer_model.flag_LV:
                        raise RuntimeError('flag_LV must be False')

                    else:
                        H_xid_xo = self._H_Xid_Xo(decode, decode.clone().detach(), sigma_out)  # N x obs

                        ############### Sum of I ########################
                        # Compute H[p(x_id|W,x_o)]
                        H_xid_w_xo = self._H_X_id_W_Xo(decode, torch.tensor(decode.data), sigma_out,
                                                   split=None)  # N_w x N x obs
                    # Compute E[H[p(x_id|W,x_o)]]
                    E_H_xid_w_xo = torch.mean(H_xid_w_xo, dim=0)  # N x obs
                    for row_id in range(observed_train_split.shape[0]):
                        E_H_Phi_id=self._H_X_Phi_O_id(decode,observed_train_split[row_id,:],W_sample,row_id) # obs
                        E_H_Phi_W_id=self._H_X_Phi_W_id_O(decode,observed_train_split[row_id,:],W_sample,row_id) # obs
                        if row_id==0:
                            E_loss_2=torch.unsqueeze(E_H_Phi_id-E_H_Phi_W_id,dim=0)
                        else:
                            E_loss_2_comp = torch.unsqueeze(E_H_Phi_id-E_H_Phi_W_id, dim=0)
                            E_loss_2 = torch.cat((E_loss_2, E_loss_2_comp), dim=0)

                    BALD_split = coef * (H_xid_xo - E_H_xid_w_xo) + (1. - coef) * E_loss_2  # N_split x obs
                elif strategy=='MC':

                    if self.Infer_model.flag_LV:
                        raise RuntimeError('flag_LV must be False')

                    else:
                        with torch.no_grad():
                            H_xid_xo = self._H_Xid_Xo(decode, decode.clone().detach(), sigma_out, split=None,flag_reduce_size=True)  # N x obs

                            ############### Sum of I ########################
                            # Compute H[p(x_id|W,x_o)]
                            H_xid_w_xo = self._H_X_id_W_Xo(decode, torch.tensor(decode.data), sigma_out,
                                                   split=None,flag_reduce_size=True)  # N_w x N x obs
                            E_H_xid_w_xo = torch.mean(H_xid_w_xo, dim=0)  # N x obs
                    BALD_split = (H_xid_xo - E_H_xid_w_xo)
                else:
                    raise NotImplementedError('strategy must be Opposite_Alpha or MC')

                if counter==0:
                    BALD=BALD_split.clone().detach()
                else:
                    BALD=torch.cat((BALD,BALD_split),dim=0)
                counter+=1
                pre_idx=idx
        else:
            raise NotImplementedError

        # Clear memory
        BALD=BALD.clone().detach()
        pool_mask=get_mask(pool_data)
        if selection_scheme=='overall':
            if not flag_initial:
                # instead of selecting top K
                print('%s Exploration'%(self.model_name))

                # Sample according to BALD
                pool_mask_Byte = ~pool_mask.bool()

                ########## No new user exploration ########
                #BALD = assign_zero_row_2D_with_target(BALD,observed_train)  # In exploration, no new user can be selected, only select entries from the observed users
                ###########################################
                if Select_Split==True:
                    # Select_Split: Some of the features are selected from old users, others are selected from new users
                    # these numbers are controlled by balance_prop
                    BALD[pool_mask_Byte] = 0.
                    BALD[BALD < 0.] = 0.
                    step_new=int((1.-balance_prop)*step)

                    # Select from the new user
                    # Apply softmax

                    # Tempering a little
                    BALD_unobserved = assign_zero_row_2D_with_target_reverse(BALD, observed_train)
                    mean_scale=torch.mean(BALD_unobserved[BALD_unobserved>0.])
                    temp=(temp_scale+0.35)*mean_scale
                    BALD_unobserved[BALD_unobserved>0.]=torch.clamp(F.softmax(BALD_unobserved[BALD_unobserved>0.]/temp),max=1.,min=1e-10)

                    BALD_weight_unob=torch.squeeze(BALD_unobserved.view(1,-1))
                    idx_un, flag_full_un,select_num_un = BALD_Select_Explore(BALD_weight_unob, step_new)
                    if not flag_full_un:
                        idx_selected_unobserved = self._transform_idx_pure(idx_un, BALD.shape[1])
                        step_old=step-select_num_un
                    else:
                        # pool data has not remaining users, thus, all selected from the old users
                        print('BALD No more new user')
                        step_old=step-select_num_un

                    # Select from the observed
                    BALD_observed = assign_zero_row_2D_with_target(BALD, observed_train)
                    mean_scale=torch.mean(BALD_observed[BALD_observed>0.])
                    temp=temp_scale*mean_scale
                    # Apply softmax
                    BALD_observed[BALD_observed>0.]=torch.clamp(F.softmax(BALD_observed[BALD_observed>0.]/temp),max=1.,min=1e-10)

                    BALD_weight_ob=torch.squeeze(BALD_observed.view(1,-1))

                    idx_ob,flag_full_ob,select_num_ob=BALD_Select_Explore(BALD_weight_ob,step_old)

                    if not flag_full_ob:
                        # Still has remaining data in pool
                        idx_selected_observed=self._transform_idx_pure(idx_ob,BALD.shape[1])
                    else:
                        print('BALD No more old user')
                    # Concat together
                    if not flag_full_un:
                        flag_full = False
                        if not flag_full_ob:
                            idx_selected = torch.cat((idx_selected_unobserved, idx_selected_observed), dim=0)
                        else:
                            idx_selected = idx_selected_unobserved
                    elif not flag_full_ob:
                        flag_full = False
                        if not flag_full_un:
                            idx_selected = torch.cat((idx_selected_unobserved, idx_selected_observed), dim=0)
                        else:
                            idx_selected = idx_selected_observed
                    else:
                        print('No possible candidate')
                        flag_full = True
                else:
                    # select from pool, this does not distingush old/new users
                    BALD[pool_mask_Byte] = 0.
                    BALD[BALD < 0.] = 0.
                    BALD_ = BALD.clone().detach()
                    mean_scale = torch.mean(BALD_[BALD_ > 0.])
                    temp = (temp_scale + 0.) * mean_scale
                    BALD_[BALD_ > 0.] = torch.clamp(
                        F.softmax(BALD_[BALD_ > 0.] / temp), max=1.,min=1e-10)
                    BALD_weight_=torch.squeeze(BALD_.view(1,-1))

                    idx_, flag_full, select_num_ = BALD_Select_Explore(BALD_weight_, step)

                    if not flag_full:
                        idx_selected = self._transform_idx_pure(idx_, BALD.shape[1])
                    else:
                        print('BALD No more remaining pool data')

            else:
                # Select top K values initially instead of sampling
                pool_mask_Byte = ~pool_mask.bool()
                BALD[pool_mask_Byte] = 0.
                BALD[BALD < 0.] = 0.
                # pool_mask_Byte = 1 - pool_mask.byte()
                pool_mask_Byte = ~pool_mask.bool()
                BALD[pool_mask_Byte] = -(10e8)

                idx, num = BALD_Select(BALD)

                idx_selected, flag_full = self._transform_idx(idx, BALD.shape[0], BALD.shape[1], num,
                                                              step=step)  # (Nxobs) x 2

            # Apply to observed_train
            if not flag_full:
                num_selected = idx_selected.shape[0]
                observed_train,pool_data=self._apply_selected_idx(observed_train,pool_data,idx_selected)
            else:
                print('Full train data selected')
                num_selected=0
        else:
            raise NotImplementedError

        return observed_train,pool_data,flag_full,num_selected


def train_SGHMC(Infer_model,observed_train,eps=0.01,max_sample_size=20,tot_epoch=100,thinning=50,hyper_param_update=200,sample_int=1,flag_hybrid=False,flag_results=True,W_dict_init=None,flag_imputation=False,**kwargs):
    # This is the function to train the model with SGHMC methods
    if flag_hybrid:
        conditional_coef = kwargs['conditional_coef']
        target_dim = kwargs['target_dim']
    else:
        conditional_coef=1.

    Adam_encoder_PNP_SGHMC=kwargs['Adam_encoder']
    Adam_embedding_PNP_SGHMC=kwargs['Adam_embedding']
    observed_train_size = observed_train.shape[0]
    batch_size = int(observed_train_size / 2.)
    Drop_p=kwargs['Drop_p']
    list_p_z=kwargs['list_p_z']
    test_input_tensor=kwargs['test_input_tensor']
    test_target_tensor=kwargs['test_target_tensor']
    valid_data=kwargs['valid_data']
    valid_data_target=kwargs['valid_data_target']
    W_sample=collections.OrderedDict()
    sigma_out=kwargs['sigma_out']
    scale_data=kwargs['scale_data']
    update_noise=kwargs['noisy_update']
    if W_dict_init is not None:
        W_dict=W_dict_init
    else:
        W_dict=Infer_model.model.decoder._get_W_dict()

    RMSE_mat,MAE_mat,NLL_mat=np.zeros(int(tot_epoch/sample_int)),np.zeros(int(tot_epoch/sample_int)),np.zeros(int(tot_epoch/sample_int))

    if batch_size > 100:
        batch_size = 100
    train_dataset = base_UCI_Dataset(observed_train, transform=None, flag_GPU=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator('cuda'))

    iter = 1
    Acc_ELBO = 0
    sample_counter=0
    valid_NLL_list=collections.deque(maxlen=10)
    mean_Acc_valid_min=10.
    counter_valid_break=0
    counter=1
    for ep in range(tot_epoch):

        flag_update=True

        for _, data in enumerate(train_loader):
            Adam_embedding_PNP_SGHMC.zero_grad()
            Adam_encoder_PNP_SGHMC.zero_grad()
            mask = get_mask(data)
            Drop_p_var = np.minimum(np.random.rand(1), Drop_p)
            mask_drop = np.array([bernoulli.rvs(1 - Drop_p_var, size=data.shape[1])] * data.shape[0])
            mask_drop = torch.from_numpy(mask_drop).float().cuda()

            if flag_hybrid:
                # mask_drop_hybrid = torch.tensor(mask_drop.data)
                mask_drop_hybrid = mask_drop.clone().detach()
                mask_drop_hybrid[:, target_dim] = 1.  # Reserve the target dim
                # mask_drop_Xdy = torch.tensor(mask_drop.data)
                mask_drop_Xdy = mask_drop.clone().detach()
                mask_drop_Xdy[:, -1] = 0.
                mask_target_ELBO = mask * mask_drop_hybrid
                mask = mask * mask_drop
            else:
                mask =mask*mask_drop

            # Flag Optimizer (This indicate the burn-in period, if true, it is in burn-in, if false, start sampling)
            flag_optimizer = (ep < (tot_epoch/2))
            if flag_optimizer:
                scale_data=1
            else:
                scale_data=scale_data
            ep_stop=10000 # Set a large value to make sure always update encoder
            if  ep<ep_stop:
                flag_record_memory=True
                flag_encoder_update=True
            else:
                # These two flags mean stop update encoder
                flag_record_memory=False
                flag_encoder_update=False
            # Hyper_param Update
            if (ep == 0 or (ep+1) % hyper_param_update == 0) and flag_update==True:
                # Only used for initialize necessary matrix for SGHMC, so only run 1 iter.
                if flag_hybrid:
                    W_dict, r, M, V = Infer_model.sample_pos_W(observed_train, eps=eps, tot_iter=1,
                                                               thinning=200,
                                                               flag_burnin=True,
                                                               flag_reset_r=False, flag_hybrid=flag_hybrid,
                                                               target_dim=target_dim,
                                                               conditional_coef=conditional_coef,W_dict_init=W_dict_init,sigma_out=sigma_out,Drop_p=Drop_p,update_noise=update_noise)
                else:
                    W_dict, r, M, V = Infer_model.sample_pos_W(observed_train, eps=eps, tot_iter=1,
                                                               thinning=200,
                                                               flag_burnin=True,
                                                               flag_reset_r=False,W_dict_init=W_dict_init,sigma_out=sigma_out,Drop_p=Drop_p,update_noise=update_noise)
                flag_update = False
                _,_,r,_,_=Infer_model._init_mat()

            if flag_hybrid:
                ELBO,ELBO_mean = Infer_model._compute_target_ELBO(data, mask, mask_target_ELBO,observed_train_size*scale_data, W_dict, size_Z=10, sigma_out=sigma_out,
                                                 z_sigma_prior=1., W_mean_prior=0., W_sigma_prior=1,
                                                 record_memory=flag_record_memory,target_dim=target_dim,conditional_coef=conditional_coef)
            else:
                ELBO, ELBO_mean = Infer_model._compute_ELBO(data, mask, observed_train_size*scale_data, W_dict, size_Z=10, sigma_out=sigma_out,
                                                 z_sigma_prior=1., W_mean_prior=0., W_sigma_prior=1,
                                                 record_memory=flag_record_memory)

            Acc_ELBO += ELBO_mean.cpu().data.numpy()

            # Obtain grad of ELBO
            (ELBO).backward()
            G_dict = Infer_model.model.decoder._get_grad_W_dict()  # No graph attached

            # SGHMC step
            M_old=copy.deepcopy(M)
            if ep==(tot_epoch/2):
                #V,M,r,_,_=Infer_model._init_mat()
                pass

            W_dict, r,V,M = Infer_model._SGHMC_step(W_dict=W_dict, G_dict=G_dict, eps=eps,eps2=eps, M=M, V=V, r=r,flag_optimizer=flag_optimizer,counter=counter,flag_compensator=True,M_old=M_old,update_noise=update_noise)

            counter+=1
            if flag_encoder_update:
                # Update encoder
                list_grad_p_z = grap_modify_grad(list_p_z, 1, observed_train.shape[0])
                zero_grad(list_p_z)
                assign_grad(list_p_z, list_grad_p_z)
                Adam_embedding_PNP_SGHMC.step()
                Adam_encoder_PNP_SGHMC.step()

            iter+=1
        
        # Initialize W_sample
        if ep < thinning or flag_optimizer:
            W_sample['sample_1'] = copy.deepcopy(W_dict)
        # Store Samples
        if (ep+1) % thinning == 0 and not flag_optimizer:
            sample_counter += 1
            W_sample = Update_W_sample(W_dict, W_sample, sample_counter, maxsize=max_sample_size)

        if (ep+1)%sample_int==0 and flag_results==True:
            RMSE,MAE,NLL=Test_UCI_batch(Infer_model.model, test_input_tensor, test_target_tensor, sigma_out_scale=sigma_out, split=10,
                           flag_model='PNP_SGHMC', size=10, Infer_model=Infer_model, W_sample=W_sample)
            RMSE_mat[int(ep/sample_int)],MAE_mat[int(ep/sample_int)],NLL_mat[int(ep/sample_int)]=RMSE.cpu().data.numpy(),MAE.cpu().data.numpy(),NLL.cpu().data.numpy()

        # Evaluate the validation NLL
        if (ep + 1) % 10 == 0 and valid_data is not None:
            mask_valid_NLL = get_mask(valid_data)
            mean_valid_NLL, _ = Infer_model.test_log_likelihood(valid_data, valid_data_target,
                                                                W_sample, mask_valid_NLL, sigma_out=sigma_out, size=10)
            valid_NLL_list.append(-mean_valid_NLL.cpu().data.numpy())

            mean_Acc_valid_now = np.mean(valid_NLL_list)
            if (ep+1)>200:
                if mean_Acc_valid_now < mean_Acc_valid_min :
                    mean_Acc_valid_min = mean_Acc_valid_now
                    counter_valid_break = 0
                else:
                    counter_valid_break += 1
            if counter_valid_break > 15:
                RMSE, MAE, NLL = Test_UCI_batch(Infer_model.model, test_input_tensor, test_target_tensor,
                                                sigma_out_scale=sigma_out, split=10,
                                                flag_model='PNP_SGHMC', size=10, Infer_model=Infer_model,
                                                W_sample=W_sample)
                print('validation worse, break at ep:%s with test NLL:%s' % (ep + 1,NLL.cpu().data.numpy()))
                break

        if (ep+1)%100==0:
            RMSE, MAE, NLL = Test_UCI_batch(Infer_model.model, test_input_tensor, test_target_tensor,
                                            sigma_out_scale=sigma_out,  split=10,
                                            flag_model='PNP_SGHMC', size=10, Infer_model=Infer_model, W_sample=W_sample)
            # Evaluate the observed train NLL
            mask_train_NLL=get_mask(observed_train)
            if not flag_imputation:
                mean_train_NLL,_=Infer_model.test_log_likelihood(observed_train, observed_train.clone().detach(), W_sample, mask_train_NLL, sigma_out=sigma_out, size=10)
            else:
                mean_train_NLL='Not Applicable'

            # Evaluate the valid NLL
            # Evaluate the observed train NLL
            if valid_data is not None:
                raise NotImplementedError('Early stop has not been implemented')
            else:
                print('conditional_coef:%s ep:%s train_NLL:%s NLL:%s RMSE:%s'%(conditional_coef,ep+1,mean_train_NLL,NLL.cpu().data.numpy(),RMSE.cpu().data.numpy()))

    return W_sample,RMSE_mat,MAE_mat,NLL_mat

