import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm.auto import tqdm
from copy import deepcopy
import os
from pathlib import Path
from afa_discriminative.utils import restore_parameters, make_onehot, get_entropy, ind_to_onehot, ConcreteSelector, MaskLayer
from afa_discriminative.models import fc_Net
from common.custom_types import AFAMethod, AFASelection, FeatureMask, Label, MaskedFeatures


class GreedyDynamicSelection(nn.Module):
    '''
    Greedy adaptive feature selection.
    
    Args:
      selector:
      predictor:
      mask_layer:
      selector_layer:
    '''

    def __init__(self, selector, predictor, mask_layer):
        super().__init__()
        
        # Set up models and mask layer.
        self.selector = selector
        self.predictor = predictor
        self.mask_layer = mask_layer
        
        # Set up selector layer.
        self.selector_layer = ConcreteSelector()

    def fit(self,
            train_loader,
            val_loader,
            lr,
            nepochs,
            max_features,
            loss_fn,
            val_loss_fn=None,
            val_loss_mode=None,
            factor=0.2,
            patience=2,
            min_lr=1e-5,
            early_stopping_epochs=None,
            start_temp=1.0,
            end_temp=0.1,
            temp_steps=5,
            argmax=False,
            verbose=True):
        '''
        Train model to perform greedy adaptive feature selection.
        
        Args:
          train_loader:
          val_loader:
          lr:
          nepochs:
          max_features:
          loss_fn:
          val_loss_fn:
          val_loss_mode:
          factor:
          patience:
          min_lr:
          early_stopping_epochs:
          start_temp:
          end_temp:
          temp_steps:
          argmax:
          verbose:
        '''
        # Verify arguments.
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = 'min'
        else:
            if val_loss_mode is None:
                raise ValueError('must specify val_loss_mode (min or max) when validation_loss_fn is specified')
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
        
        # Set up models.
        selector = self.selector
        predictor = self.predictor
        mask_layer = self.mask_layer
        selector_layer = self.selector_layer
        device = next(predictor.parameters()).device
        val_loss_fn.to(device)
        
        # Determine mask size.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            x, y = next(iter(val_loader))
            assert len(x.shape) == 2
            mask_size = x.shape[1]
        
        # For tracking best models with zero temperature.
        best_val = None
        best_zerotemp_selector = None
        best_zerotemp_predictor = None
        
        # Train separately with each temperature.
        total_epochs = 0
        for temp in np.geomspace(start_temp, end_temp, temp_steps):
            if verbose:
                print(f'Starting training with temp = {temp:.4f}\n')

            # Set up optimizer and lr scheduler.
            opt = optim.Adam(set(list(predictor.parameters()) + list(selector.parameters())), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=val_loss_mode, factor=factor, patience=patience,
                min_lr=min_lr)
            
            # For tracking best models and early stopping.
            best_selector = deepcopy(selector)
            best_predictor = deepcopy(predictor)
            num_bad_epochs = 0

            for epoch in range(nepochs):
                # Switch models to training mode.
                selector.train()
                predictor.train()

                for x, y in train_loader:
                    # Move to device.
                    x = x.to(device)
                    y = y.to(device)
                    
                    # Setup.
                    m = torch.zeros(len(x), mask_size, dtype=x.dtype, device=device)
                    selector.zero_grad()
                    predictor.zero_grad()
                    
                    for _ in range(max_features):
                        # Evaluate selector model.
                        x_masked = mask_layer(x, m)
                        logits = selector(x_masked).flatten(1)
                        
                        # Get selections.
                        soft = selector_layer(logits, temp)
                        m_soft = torch.max(m, soft)
                        
                        # Evaluate predictor model.
                        x_masked = mask_layer(x, m_soft)
                        pred = predictor(x_masked)
                        
                        # Calculate loss.
                        loss = loss_fn(pred, y)
                        (loss / max_features).backward()
                        
                        # Update mask, ensure no repeats.
                        m = torch.max(m, make_onehot(selector_layer(logits - 1e6 * m, 1e-6)))

                    # Take gradient step.
                    opt.step()
                    
                # Calculate validation loss.
                selector.eval()
                predictor.eval()
                with torch.no_grad():
                    # For mean loss.
                    pred_list = []
                    hard_pred_list = []
                    label_list = []

                    for x, y in val_loader:
                        # Move to device.
                        x = x.to(device)
                        y = y.to(device)

                        # Setup.
                        m = torch.zeros(len(x), mask_size, dtype=x.dtype, device=device)

                        for _ in range(max_features):
                            # Evaluate selector model.
                            x_masked = mask_layer(x, m)
                            logits = selector(x_masked).flatten(1)
                            
                            # Get selections, ensure no repeats.
                            logits = logits - 1e6 * m
                            if argmax:
                                soft = selector_layer(logits, temp, deterministic=True)
                            else:
                                soft = selector_layer(logits, temp)
                            m_soft = torch.max(m, soft)
                            
                            # For calculating temp = 0 loss.
                            m = torch.max(m, make_onehot(soft))
                            
                            # Evaluate predictor with soft sample.
                            x_masked = mask_layer(x, m_soft)
                            pred = predictor(x_masked)
                            
                            # Evaluate predictor with hard sample.
                            x_masked = mask_layer(x, m)
                            hard_pred = predictor(x_masked)

                            # Append predictions and labels.
                            pred_list.append(pred.cpu())
                            hard_pred_list.append(hard_pred.cpu())
                            label_list.append(y.cpu())

                    # Calculate mean loss.
                    pred = torch.cat(pred_list, 0)
                    hard_pred = torch.cat(hard_pred_list, 0)
                    y = torch.cat(label_list, 0)
                    val_loss = val_loss_fn(pred, y)
                    val_hard_loss = val_loss_fn(hard_pred, y)

                # Print progress.
                if verbose:
                    print(f'{"-"*8}Epoch {epoch+1} ({epoch + 1 + total_epochs} total){"-"*8}')
                    print(f'Val loss = {val_loss:.4f}, Zero-temp loss = {val_hard_loss:.4f}\n')

                # Update scheduler.
                scheduler.step(val_loss)

                # Check if best model.
                if val_loss == scheduler.best:
                    best_selector = deepcopy(selector)
                    best_predictor = deepcopy(predictor)
                    num_bad_epochs = 0
                else:
                    num_bad_epochs += 1

                # Check if best model with zero temperature.
                if ((best_val is None)
                    or (val_loss_mode == 'min' and val_hard_loss < best_val)
                    or (val_loss_mode == 'max' and val_hard_loss > best_val)):
                    best_val = val_hard_loss
                    best_zerotemp_selector = deepcopy(selector)
                    best_zerotemp_predictor = deepcopy(predictor)
                    
                # Early stopping.
                if num_bad_epochs > early_stopping_epochs:
                    break

            # Update total epoch count.
            if verbose:
                print(f'Stopping temp = {temp:.4f} at epoch {epoch+1}\n')
            total_epochs += (epoch + 1)

            # Copy parameters from best model.
            restore_parameters(selector, best_selector)
            restore_parameters(predictor, best_predictor)

        # Copy parameters from best model with zero temperature.
        restore_parameters(selector, best_zerotemp_selector)
        restore_parameters(predictor, best_zerotemp_predictor)

    def forward(self, x, max_features, argmax=True):
        '''
        Make predictions using selected features.

        Args:
          x:
          max_features:
          argmax:
        '''
        # Setup.
        selector = self.selector
        predictor = self.predictor
        mask_layer = self.mask_layer
        selector_layer = self.selector_layer
        device = next(predictor.parameters()).device
        
        # Determine mask size.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = self.mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            assert len(x.shape) == 2
            mask_size = x.shape[1]
        m = torch.zeros(len(x), mask_size, device=device)

        for _ in range(max_features):
            # Evaluate selector model.
            x_masked = mask_layer(x, m)
            logits = selector(x_masked).flatten(1)

            # Update selections, ensure no repeats.
            logits = logits - 1e6 * m
            if argmax:
                m = torch.max(m, make_onehot(logits))
            else:
                m = torch.max(m, make_onehot(selector_layer(logits, 1e-6)))

        # Make predictions.
        x_masked = mask_layer(x, m)
        pred = predictor(x_masked)
        return pred, x_masked, m

    def evaluate(self,
                 loader,
                 max_features,
                 metric,
                 argmax=True):
        '''
        Evaluate mean performance across a dataset.
        
        Args:
          loader:
          max_features:
          metric:
          argmax:
        '''
        # Setup.
        self.selector.eval()
        self.predictor.eval()
        device = next(self.predictor.parameters()).device

        # For calculating mean loss.
        pred_list = []
        label_list = []

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device)

                # Calculate loss.
                pred, _, _ = self.forward(x, max_features, argmax)
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
    

class Covert2023AFAMethod(AFAMethod):
    def __init__(self, selector, predictor):
        super().__init__()
        
        # Set up models and mask layer.
        self.selector = selector
        self.predictor = predictor

    def predict(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> Label:
        x_masked = torch.cat([masked_features, feature_mask], dim=1)
        predictor = self.predictor
        pred = predictor(x_masked)
        return pred
    
    def select(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> AFASelection:
        # mask_layer = self.mask_layer
        selector = self.selector
        # x_masked = mask_layer(feature, feature_mask)
        x_masked = torch.cat([masked_features, feature_mask], dim=1)
        logits = selector(x_masked).flatten(1)
        logits = logits - 1e6 * feature_mask
        # TODO Do we need +1 here?
        next_feature_idx = logits.argmax(dim=1) + 1
        return next_feature_idx
    
    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        arch = checkpoint['architecture']
        d_in = arch['d_in']
        d_out = arch['d_out']
        selector_hidden_layers = arch['selector_hidden_layers']
        predictor_hidden_layers = arch['predictor_hidden_layers']
        dropout = arch['dropout']
        predictor = fc_Net(
            input_dim=d_in * 2,
            output_dim=d_out,
            hidden_layer_num=len(predictor_hidden_layers),
            hidden_unit=predictor_hidden_layers,
            activations='ReLU',
            drop_out_rate=dropout,
            flag_drop_out=True,
            flag_only_output_layer=False
        )
        selector = fc_Net(
            input_dim=d_in * 2,
            output_dim=d_in,
            hidden_layer_num=len(selector_hidden_layers),
            hidden_unit=selector_hidden_layers,
            activations='ReLU',
            drop_out_rate=dropout,
            flag_drop_out=True,
            flag_only_output_layer=False
        )

        model = cls(selector, predictor)
        model.selector.load_state_dict(checkpoint['selector_state_dict'])
        model.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        model.selector.eval()
        model.predictor.eval()
        return model

    def save(self, path: Path):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'selector_state_dict': self.selector.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'architecture': {
                'd_in': self.selector.output_dim,
                'd_out': self.predictor.output_dim,
                'selector_hidden_layers': [128, 128],
                'predictor_hidden_layers': [128, 128],
                'dropout': 0.3,
            }}, os.path.join(path,f'model.pt'))


class CMIEstimator(nn.Module):
    '''
    Greedy CMI estimation module.
    '''

    def __init__(self,
                 value_network,
                 predictor,
                 mask_layer):
        super().__init__()

        # Save network modules.
        self.value_network = value_network
        self.predictor = predictor
        self.mask_layer = mask_layer

    def set_stopping_criterion(self, budget=None, lam=None, confidence=None):
        '''Set parameters for stopping criterion.'''
        if sum([budget is None, lam is None, confidence is None]) != 2:
            raise ValueError('Must specify exactly one of budget, lam, and confidence')
        if budget is not None:
            self.budget = budget
            self.mode = 'budget'
        elif lam is not None:
            self.lam = lam
            self.mode = 'penalized'
        elif confidence is not None:
            self.confidence = confidence
            self.mode = 'confidence'

    def fit(self,
            train_loader,
            val_loader,
            lr,
            nepochs,
            max_features,
            eps,
            loss_fn,
            val_loss_fn,
            val_loss_mode,
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            early_stopping_epochs=None,
            eps_decay=0.2,
            eps_steps=1,
            feature_costs=None,
            cmi_scaling='bounded',
            verbose=True):
        
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = 'min'
        else:
            if val_loss_mode is None:
                raise ValueError('must specify val_loss_mode (min or max) when validation_loss_fn is specified')
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
        
        value_network = self.value_network
        predictor = self.predictor
        mask_layer = self.mask_layer

        device = next(predictor.parameters()).device
        val_loss_fn = val_loss_fn.to(device)
        value_network = value_network.to(device)

        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            x, y = next(iter(val_loader))
            assert len(x.shape) == 2
            mask_size = x.shape[1]

        if feature_costs is None:
            feature_costs = torch.ones(mask_size).to(device)
        elif isinstance(feature_costs, np.ndarray):
            feature_costs = torch.tensor(feature_costs).to(device)

        opt = optim.Adam(set(list(value_network.parameters()) + list(predictor.parameters())), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=val_loss_mode, factor=factor, patience=patience,
            min_lr=min_lr)
        
        # For tracking best models and early stopping.
        best_value_network = deepcopy(value_network)
        best_predictor = deepcopy(predictor)
        num_bad_epochs = 0
        num_epsilon_steps = 0

        for epoch in range(nepochs):
            # Switch models to training mode.
            value_network.train()
            predictor.train()
            value_losses = []
            pred_losses = []

            for x, y in train_loader:
                # Move to device.
                x = x.to(device)
                y = y.to(device)
                
                # Setup.
                m = torch.zeros(len(x), mask_size, dtype=x.dtype, device=device)
                value_network.zero_grad()
                predictor.zero_grad()
                value_network_loss_total = 0
                pred_loss_total = 0

                # Predictor loss with no features.
                x_masked = self.mask_layer(x, m)
                pred_without_next_feature = predictor(x_masked)
                loss_without_next_feature = loss_fn(pred_without_next_feature, y)
                pred_loss = loss_without_next_feature.mean()
                pred_loss_total += pred_loss.detach()

                (pred_loss / (max_features + 1)).backward()
                pred_without_next_feature = pred_without_next_feature.detach()
                loss_without_next_feature = loss_without_next_feature.detach()
                
                for _ in range(max_features):
                    # Estimate CMI using value network.
                    x_masked = mask_layer(x, m)
                    if cmi_scaling == 'bounded':
                        entropy = get_entropy(pred_without_next_feature).unsqueeze(1)
                        pred_cmi = value_network(x_masked).sigmoid() * entropy
                    elif cmi_scaling == 'positive':
                        pred_cmi = torch.nn.functional.softplus(value_network(x_masked))
                    else:
                        pred_cmi = value_network(x_masked)
                
                    best = torch.argmax(pred_cmi / feature_costs, dim=1)
                    random = torch.tensor(np.random.choice(mask_size, size=len(x)), device=x.device)
                    exploit = (torch.rand(len(x), device=x.device) > eps).int()
                    actions = exploit * best + (1 - exploit) * random
                    m = torch.max(m, ind_to_onehot(actions, mask_size))

                    # Predictor loss.
                    x_masked = self.mask_layer(x, m)
                    pred_with_next_feature = predictor(x_masked)
                    loss_with_next_feature = loss_fn(pred_with_next_feature, y)

                    # Value network loss.
                    delta = loss_without_next_feature - loss_with_next_feature.detach()
                    value_network_loss = nn.functional.mse_loss(pred_cmi[torch.arange(len(x)), actions], delta)

                    # Calculate gradients.
                    total_loss = torch.mean(value_network_loss) + torch.mean(loss_with_next_feature)
                    (total_loss / (max_features + 1)).backward()

                    # Updates.
                    value_network_loss_total += torch.mean(value_network_loss)
                    pred_loss_total += torch.mean(loss_with_next_feature)
                    loss_without_next_feature = loss_with_next_feature.detach()
                    pred_without_next_feature = pred_with_next_feature.detach()

                # Take gradient step.
                opt.step()
                opt.zero_grad()

                value_losses.append(value_network_loss_total / max_features)
                pred_losses.append(pred_loss_total / (max_features + 1))
                
            # Calculate validation loss.
            value_network.eval()
            predictor.eval()
            val_preds = [[] for _ in range(max_features + 1)]
            val_targets = []

            with torch.no_grad():
                for x, y in val_loader:
                    # Move to device.
                    x = x.to(device)
                    y = y.to(device)

                    # Setup.
                    m = torch.zeros(len(x), mask_size, dtype=x.dtype, device=device)
                    x_masked = self.mask_layer(x, m)
                    pred = predictor(x_masked)
                    val_preds[0].append(pred)

                    for i in range(1, max_features + 1):
                        # Estimate CMI using value network.
                        x_masked = mask_layer(x, m)
                        if cmi_scaling == 'bounded':
                            entropy = get_entropy(pred).unsqueeze(1)
                            pred_cmi = value_network(x_masked).sigmoid() * entropy
                        elif cmi_scaling == 'positive':
                            pred_cmi = torch.nn.functional.softplus(value_network(x_masked))
                        else:
                            pred_cmi = value_network(x_masked)

                        # Select next feature, ensure no repeats.
                        pred_cmi -= 1e6 * m
                        best_feature_index = torch.argmax(pred_cmi / feature_costs, dim=1)
                        m = torch.max(m, ind_to_onehot(best_feature_index, mask_size))

                        # Make prediction.
                        x_masked = self.mask_layer(x, m)
                        pred = self.predictor(x_masked)
                        val_preds[i].append(pred)

                    val_targets.append(y)

                # Calculate mean loss.
                y_val = torch.cat(val_targets)
                preds_cat = [torch.cat(p) for p in val_preds]
                pred_losses = [loss_fn(p, y_val).mean() for p in preds_cat]
                val_scores = [val_loss_fn(p, y_val) for p in preds_cat]
                val_loss_mean = torch.stack(pred_losses).mean()
                val_perf_mean = torch.stack(val_scores).mean()
                val_loss_final = pred_losses[-1]
                val_perf_final = val_scores[-1]

            # Print progress.
            if verbose:
                print(f'{"-"*8}Epoch {epoch+1}{"-"*8}')
                print(f'Loss Val/Mean = {val_loss_mean}')
                print(f'Perf Val/Mean = {val_perf_mean}')
                print(f'Loss Val/Final = {val_loss_final}')
                print(f'Perf Val/Final = {val_perf_final}')
                print(f'Eps Value = {eps}\n')

            # Update scheduler.
            scheduler.step(val_loss_mean)

            # Check if best model.
            if val_loss_mean == scheduler.best:
                best_value_network = deepcopy(value_network)
                best_predictor = deepcopy(predictor)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                
            # Decay epsilon.
            if num_bad_epochs > early_stopping_epochs:
                eps *= eps_decay
                num_bad_epochs = 0
                num_epsilon_steps += 1
                print(f'Decaying eps to {eps:.5f}, step = {num_epsilon_steps}')

                # Early stopping.
                if num_epsilon_steps >= eps_steps:
                    break
            
                # Reset optimizer learning rate. Could fully reset optimizer and scheduler, but this is simpler.
                for g in opt.param_groups:
                    g['lr'] = lr

        # Copy parameters from best model.
        restore_parameters(value_network, best_value_network)
        restore_parameters(predictor, best_predictor)


class Gadgil2023AFAMethod(AFAMethod):
    def __init__(self,
                 value_network,
                 predictor):
        super().__init__()

        # Save network modules.
        self.value_network = value_network
        self.predictor = predictor

    def predict(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> Label:
        x_masked = torch.cat([masked_features, feature_mask], dim=1)
        predictor = self.predictor
        pred = predictor(x_masked)
        return pred
    
    def select(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> AFASelection:
        x_masked = torch.cat([masked_features, feature_mask], dim=1)
        pred = self.predict(masked_features, feature_mask)
        entropy = get_entropy(pred).unsqueeze(1)
        value_network = self.value_network
        pred_cmi = value_network(x_masked).sigmoid() * entropy
        # check_pos_pred_cmi = pred_cmi.max(dim=1).values >= 0
        pred_cmi -= 1e6 * feature_mask
        next_feature_idx = torch.argmax(pred_cmi, dim=1) + 1
        # selection = ind_to_onehot(next_feature_idx, masked_features.shape[-1])

        # if sum(check_pos_pred_cmi) == 0:
        #     return 0

        return next_feature_idx

    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        arch = checkpoint['architecture']
        d_in = arch['d_in']
        d_out = arch['d_out']
        value_network_hidden_layers = arch['value_network_hidden_layers']
        predictor_hidden_layers = arch['predictor_hidden_layers']
        dropout = arch['dropout']
        predictor = fc_Net(
            input_dim=d_in * 2,
            output_dim=d_out,
            hidden_layer_num=len(predictor_hidden_layers),
            hidden_unit=predictor_hidden_layers,
            activations='ReLU',
            drop_out_rate=dropout,
            flag_drop_out=True,
            flag_only_output_layer=False
        )
        value_network = fc_Net(
            input_dim=d_in * 2,
            output_dim=d_in,
            hidden_layer_num=len(value_network_hidden_layers),
            hidden_unit=value_network_hidden_layers,
            activations='ReLU',
            drop_out_rate=dropout,
            flag_drop_out=True,
            flag_only_output_layer=False
        )
        # Tie weights
        value_network.hidden[0] = predictor.hidden[0]
        value_network.hidden[1] = predictor.hidden[1]

        model = cls(value_network, predictor)
        model.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        model.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        model.value_network.eval()
        model.predictor.eval()
        return model
    
    def save(self, path: Path):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'value_network_state_dict': self.value_network.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'architecture': {
                'd_in': self.value_network.output_dim,
                'd_out': self.predictor.output_dim,
                'value_network_hidden_layers': [128, 128],
                'predictor_hidden_layers': [128, 128],
                'dropout': 0.3,
            }}, os.path.join(path,f'model.pt'))
