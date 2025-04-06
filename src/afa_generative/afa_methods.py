import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm.auto import tqdm
from afa_generative.utils import generate_uniform_mask, restore_parameters, MaskLayerGrouped
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from copy import deepcopy


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


class PVAE(nn.Module):
    '''
    Partial VAE (PVAE): a variational autoencoder trained with random feature subsets.
    
    Original paper: https://arxiv.org/abs/1809.11142v4
    
    Args:
      encoder: encoder network.
      decoder: decoder network.
      mask_layer: layer to perform masking on encoder input.
      num_samples: number of latent variable samples to use during training.
      decoder_distribution: distribution for reconstruction, 'gaussian' or
        'bernoulli'.
      deterministic_kl: calculate prior/posterior KL divergence
        deterministically or stochastically.
    '''

    def __init__(self,
                 encoder,
                 decoder,
                 mask_layer,
                 num_samples=128,
                 decoder_distribution='gaussian',
                 deterministic_kl=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_layer = mask_layer
        self.num_samples = num_samples
        self.deterministic_kl = deterministic_kl
        
        assert decoder_distribution in ('gaussian', 'bernoulli')
        self.decoder_distribution = decoder_distribution
    
    def forward(self, x, mask):
        # Get latent encoding.
        x_masked = self.mask_layer(x, mask)
        latent = self.encoder(x_masked)
    
        # Sample latent variables and decode.
        dims = latent.shape[1] // 2
        mean = latent[:, :dims]
        std = torch.exp(latent[:, dims:])
        eps = torch.randn(mean.shape[0], self.num_samples, mean.shape[1], device=mean.device)
        z =  torch.unsqueeze(mean, 1) + eps * torch.unsqueeze(std, 1)
    
        # Decode and return.
        recon = self.decoder(z)
        return latent, z, recon
    
    def loss(self, x, mask):
        # Get latent representation and reconstruction.
        latent, z, recon = self.forward(x, mask)
        
        # Calculate latent KL divergence.
        latent_dims = latent.shape[1] // 2
        latent_mean = latent[:, :latent_dims]
        latent_std = torch.exp(latent[:, latent_dims:])
        if self.deterministic_kl:
            kl = torch.distributions.kl_divergence(
                Normal(latent_mean, latent_std),
                Normal(0.0, 1.0)).sum(1)
            kl = torch.unsqueeze(kl, 1)
        else:
            # Set up prior and posterior distributions.
            p_dist = Normal(0.0, 1.0)
            q_dist = Normal(latent_mean, latent_std)
            
            # Estimate KL divergence.
            log_p = p_dist.log_prob(z)
            log_q = q_dist.log_prob(z.permute(1, 0, 2)).permute(1, 0, 2)
            kl = (log_q - log_p).sum(dim=2)
        
        # Calculate output log prob.
        if self.decoder_distribution == 'gaussian':
            # TODO learned std version: unstable training
            # dims = recon.shape[2] // 2
            # mean = recon[:, :, :dims]
            # std = torch.exp(recon[:, :, dims:])
            mean = recon
            std = torch.ones_like(mean)
            dist = Normal(mean, std)
        elif self.decoder_distribution == 'bernoulli':
            p = recon.sigmoid()
            dist = Bernoulli(p)
            # x = (x > 0.5).float()  # TODO included this only for MNIST, not usually necessary
        log_prob = dist.log_prob(torch.unsqueeze(x, 1))
        if isinstance(self.mask_layer, MaskLayerGrouped):  # TODO support for groups is not elegant
            mask_multiply = mask @ self.mask_layer.group_matrix
        else:
            mask_multiply = mask
        log_prob = (log_prob * torch.unsqueeze(mask_multiply, 1)).sum(dim=2)
        
        # Calculate loss.
        return kl - log_prob
    
    def fit(self,
            train_loader,
            val_loader,
            lr,
            nepochs,
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
          factor:
          patience:
          min_lr:
          early_stopping_epochs:
          verbose:
        '''
        # Set up optimizer and lr scheduler.
        mask_layer = self.mask_layer
        device = next(self.parameters()).device
        opt = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=factor, patience=patience,
            min_lr=min_lr)
        
        # Determine mask size.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            x, _ = next(iter(val_loader))
            assert len(x.shape) == 2
            mask_size = x.shape[1]

        # For tracking best model and early stopping.
        best_encoder = None
        best_decoder = None
        num_bad_epochs = 0
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
            
        # for epoch in tqdm(range(nepochs), desc="Epoch number", ncols=80):
        for epoch in range(nepochs):
            # Switch model to training mode.
            self.train()

            for x, _ in train_loader:
                # Calculate loss.
                x = x.to(device)
                m = generate_uniform_mask(len(x), mask_size).to(device)
                loss = self.loss(x, m).mean()

                # Take gradient step.
                loss.backward()
                opt.step()
                self.zero_grad()
                
            # Calculate validation loss.
            self.eval()
            with torch.no_grad():
                # For mean loss.
                val_loss = 0
                n = 0

                for x, _ in val_loader:
                    # Calculate loss.
                    # TODO mask should be precomputed and shared across epochs
                    x = x.to(device)
                    m = generate_uniform_mask(len(x), mask_size).to(device)
                    loss = self.loss(x, m).mean()
                    
                    # Update mean.
                    val_loss = (loss * len(x) + val_loss * n) / (n + len(x))
                    n += len(x)

            # Print progress.
            if verbose:
                print(f'{"-"*8}Epoch {epoch+1}{"-"*8}')
                print(f'Val loss = {val_loss:.4f}\n')
                
            # Update scheduler.
            scheduler.step(val_loss)

            # Check if best model.
            if val_loss == scheduler.best:
                best_encoder = deepcopy(self.encoder)
                best_decoder = deepcopy(self.decoder)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
            
            print(num_bad_epochs, early_stopping_epochs)
                
            # Early stopping.
            if num_bad_epochs > early_stopping_epochs:
                if verbose:
                    print(f'Stopping early at epoch {epoch+1}')
                break

        # Copy parameters from best model.
        restore_parameters(self.encoder, best_encoder)
        restore_parameters(self.decoder, best_decoder)
    
    def impute(self, x, mask):
        '''Impute using a partial input.'''
        _, _, recon = self.forward(x, mask)
        return self.output_sample(recon)
    
    def generate(self, num_samples):
        '''Generate new samples by sampling from the latent distribution.'''
        dim = list(self.decoder.parameters())[0].shape[1]
        device = next(self.decoder.parameters()).device
        z = torch.randn(num_samples, dim, device=device)
        
        # Decode.
        recon = self.decoder(z)
        return self.output_sample(recon)
        
    def output_sample(self, params):
        '''Generate output sample given decoder parameters.'''
        if self.decoder_distribution == 'gaussian':
            # Return mean.
            mean = params
            return mean

        elif self.decoder_distribution == 'bernoulli':
            # Return probabilities.
            p = params.sigmoid()
            return p


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
                best_model = deepcopy(model)
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

