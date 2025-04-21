import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from afa_generative.utils import generate_uniform_mask, restore_parameters, MaskLayerGrouped
from copy import deepcopy
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
import collections
import copy
import math


class BaseModel(nn.Module):
    '''
    Base model, no missing features.
    
    Args:
      model:
    '''

    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def fit(self,
            train_loader,
            val_loader,
            lr,
            nepochs,
            loss_fn,
            val_loss_fn=None,
            val_loss_mode=None,
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
        device = next(model.parameters()).device
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=val_loss_mode, factor=factor, patience=patience,
            min_lr=min_lr)

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

                # Calculate loss.
                pred = model(x)
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
                    
                    # Calculate prediction.
                    pred = model(x)
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
        
    def evaluate(self, loader, metric):
        '''
        Evaluate mean performance across a dataset.
        
        Args:
          loader:
          metric:
        '''
        # Setup.
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
                pred = self.model(x)
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
    
    def forward(self, x):
        '''
        Generate model prediction.
        
        Args:
          x:
        '''
        return self.model(x)


class MaskingPretrainer(nn.Module):
    '''Pretrain model with missing features.'''

    def __init__(self, model, mask_layer):
        super().__init__()
        self.model = model
        self.mask_layer = mask_layer
        
    def fit(self,
            train_loader,
            val_loader,
            lr,
            nepochs,
            loss_fn,
            val_loss_fn=None,
            val_loss_mode=None,
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

    def evaluate(self, loader, metric):
        '''
        Evaluate mean performance across a dataset.
        
        Args:
          loader:
          metric:
        '''
        # Setup.
        self.model.eval()
        device = next(self.model.parameters()).device

        # Determine mask size.
        if hasattr(self.mask_layer, 'mask_size') and (self.mask_layer.mask_size is not None):
            mask_size = self.mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            x, y = next(iter(loader))
            assert len(x.shape) == 2
            mask_size = x.shape[1]

        # For calculating mean loss.
        pred_list = []
        label_list = []

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device)
                mask = torch.ones(len(x), mask_size, device=device)

                # Calculate loss.
                pred = self.forward(x, mask)
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
    
    def forward(self, x, mask):
        '''
        Generate model prediction.
        
        Args:
          x:
          mask:
        '''
        x_masked = self.mask_layer(x, mask)
        return self.model(x_masked)


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


class fc_Net(nn.Module):
    '''
    This class implements the base network structure for fully connected encoder/decoder/predictor.
    This also support the dynamically adding new nodes at the output layer for decoder.
    '''
    def __init__(self,input_dim,output_dim,hidden_layer_num=2,hidden_unit=[100,50],activations='ReLU',activations_output=None,flag_only_output_layer=False,drop_out_rate=0.,flag_drop_out=False,flag_LV=False,output_const=1.,add_const=0.):
        '''
        Init method
        :param input_dim: The input dimensions
        :type input_dim: int
        :param output_dim: The output dimension of the network
        :type output_dim: int
        :param hidden_layer_num: The number of hidden layers excluding the output layer
        :type hidden_layer_num: int
        :param hidden_unit: The hidden unit size
        :type hidden_unit: list
        :param activations: The activation function for hidden layers
        :type activations: string
        :param flag_only_output_layer: If we only use output layer, so one hidden layer nerual net
        :type flag_only_output_layer: bool
        :param drop_out_rate: The disable percentage of the hidden node
        :param flag_drop_out: Bool, whether to use drop out
        '''
        super(fc_Net,self).__init__()
        self.drop_out_rate=drop_out_rate
        self.flag_drop_out=flag_drop_out
        self.output_dim=output_dim
        self.input_dim=input_dim
        self.hidden_layer_num=hidden_layer_num
        self.hidden_unit=hidden_unit
        self.flag_only_output_layer=flag_only_output_layer
        self.flag_LV=flag_LV
        self.output_const=output_const
        self.add_const=add_const
        if self.flag_LV==True:
            self.LV_dim=output_dim
        self.enable_output_act=False
        self.drop_out=nn.Dropout(self.drop_out_rate)
        # activation functions
        self.activations = activations
        if activations=='ReLU':
            self.act=F.relu
        elif activations=='Sigmoid':
            self.act=F.sigmoid
        elif activations=='Tanh':
            self.act=F.tanh
        elif activations=='Elu':
            self.act=F.elu
        elif activations=='Selu':
            self.act=F.selu
        else:
            raise NotImplementedError

        if activations_output=='ReLU':
            self.enable_output_act=True
            self.act_out=F.relu
        elif activations_output=='Sigmoid':
            self.enable_output_act = True
            self.act_out=F.sigmoid
        elif activations_output=='Tanh':
            self.enable_output_act = True
            self.act_out=F.tanh
        elif activations_output=='Elu':
            self.enable_output_act = True
            self.act_out=F.elu
        elif activations_output=='Selu':
            self.enable_output_act = True
            self.act_out=F.selu
        elif activations_output=='Softplus':
            self.enable_output_act = True
            self.act_out=F.softplus

        # whether to use multi NN or single layer NN
        if self.flag_only_output_layer==False:
            assert len(self.hidden_unit)==hidden_layer_num,'Hidden layer unit length %s inconsistent with layer number %s'%(len(self.hidden_unit),self.hidden_layer_num)

            # build hidden layers
            self.hidden=nn.ModuleList()
            for layer_ind in range(self.hidden_layer_num):
                if layer_ind==0:
                    self.hidden.append(nn.Linear(self.input_dim,self.hidden_unit[layer_ind]))
                else:
                    self.hidden.append(nn.Linear(self.hidden_unit[layer_ind-1],self.hidden_unit[layer_ind]))

            # output layer
            self.out=nn.Linear(self.hidden_unit[-1],self.output_dim)
            if self.flag_LV:
                self.out_LV=nn.Linear(self.hidden_unit[-1],self.LV_dim)

        else:
            self.out=nn.Linear(self.input_dim,self.output_dim)
            if self.flag_LV:
                self.out_LV = nn.Linear(self.hidden_unit[-1], self.LV_dim)


    def _assign_weight(self,W_dict):
        if self.flag_only_output_layer==False:
            for layer_ind in range(self.hidden_layer_num):
                layer_weight=W_dict['weight_layer_%s'%(layer_ind)]
                layer_bias=W_dict['bias_layer_%s'%(layer_ind)]

                self.hidden[layer_ind].weight=torch.nn.Parameter(layer_weight.data)
                self.hidden[layer_ind].bias=torch.nn.Parameter(layer_bias.data)
            out_weight=W_dict['weight_out']
            out_bias=W_dict['bias_out']
            self.out.weight=torch.nn.Parameter(out_weight.data)
            self.out.bias=torch.nn.Parameter(out_bias.data)
            if self.flag_LV:
                out_weight_LV = W_dict['weight_out_LV']
                out_bias_LV = W_dict['bias_out_LV']
                self.out_LV.weight = torch.nn.Parameter(out_weight_LV.data)
                self.out_LV.bias = torch.nn.Parameter(out_bias_LV.data)

        else:
            out_weight = W_dict['weight_out']
            out_bias = W_dict['bias_out']
            self.out.weight = torch.nn.Parameter(out_weight.data)
            self.out.bias = torch.nn.Parameter(out_bias.data)
            if self.flag_LV:
                out_weight_LV = W_dict['weight_out_LV']
                out_bias_LV = W_dict['bias_out_LV']
                self.out_LV.weight = torch.nn.Parameter(out_weight_LV.data)
                self.out_LV.bias = torch.nn.Parameter(out_bias_LV.data)


    def _get_W_dict(self):
        W_dict=collections.OrderedDict()
        if self.flag_only_output_layer == False:
            for layer_ind in range(self.hidden_layer_num):
                W_dict['weight_layer_%s'%(layer_ind)]=self.hidden[layer_ind].weight.clone().detach()
                W_dict['bias_layer_%s'%(layer_ind)]=self.hidden[layer_ind].bias.clone().detach()
            W_dict['weight_out']=self.out.weight.clone().detach()
            W_dict['bias_out']=self.out.bias.clone().detach()
            if self.flag_LV:
                W_dict['weight_out_LV'] = self.out_LV.weight.clone().detach()
                W_dict['bias_out_LV'] = self.out_LV.bias.clone().detach()
        else:
            W_dict['weight_out'] = self.out.weight.clone().detach()
            W_dict['bias_out'] = self.out.bias.clone().detach()
            if self.flag_LV:
                W_dict['weight_out_LV'] = self.out_LV.weight.clone().detach()
                W_dict['bias_out_LV'] = self.out_LV.bias.clone().detach()
        return W_dict
    

    def _get_grad_W_dict(self):
        G_dict=collections.OrderedDict()
        if self.flag_only_output_layer == False:
            for layer_ind in range(self.hidden_layer_num):
                if self.hidden[layer_ind].weight.grad is not None:
                    G_dict['weight_layer_%s'%(layer_ind)]=-self.hidden[layer_ind].weight.grad.clone().detach()
                    G_dict['bias_layer_%s'%(layer_ind)]=-self.hidden[layer_ind].bias.grad.clone().detach()
            G_dict['weight_out']=-self.out.weight.grad.clone().detach()
            G_dict['bias_out']=-self.out.bias.grad.clone().detach()
            if self.flag_LV:
                G_dict['weight_out_LV'] = -self.out_LV.weight.grad.clone().detach()
                G_dict['bias_out_LV'] = -self.out_LV.bias.grad.clone().detach()
        else:
            G_dict['weight_out'] = -self.out.weight.grad.clone().detach()
            G_dict['bias_out'] = -self.out.bias.grad.clone().detach()
            if self.flag_LV:
                G_dict['weight_out_LV'] = -self.out_LV.weight.grad.clone().detach()
                G_dict['bias_out_LV'] = -self.out_LV.bias.grad.clone().detach()
        return G_dict
    

    def _flatten_stat(self):
        if self.flag_only_output_layer==False:
            for idx,layer in enumerate(self.hidden):
                W_weight,b_weight=layer.weight.view(1,-1),layer.bias.view(1,-1) # 1 x dim
                weight_comp=torch.cat((W_weight,b_weight),dim=1) # 1 x dim
                if idx==0:
                    weight_flat=weight_comp
                else:
                    weight_flat=torch.cat((weight_flat,weight_comp),dim=1)

            # Output layer (need to account for the mask)
            Out_weight,Out_b_weight=self.out.weight,self.out.bias # N_in x N_out or N_out

            if self.flag_LV==True:
                W_weight_LV,b_weight_LV=self.out_LV.weight,self.out_LV.bias
                Out_weight,Out_b_weight=torch.cat((Out_weight,W_weight_LV),dim=1),torch.cat((Out_b_weight,b_weight_LV),dim=0) #' This should be N_in x (2 x N_out) or (2 X N_out)'

            return weight_flat,Out_weight,Out_b_weight
        else:
            Out_weight, Out_b_weight = self.out.weight, self.out.bias  # N_in x N_out or N_out
            if self.flag_LV == True:
                W_weight_LV, b_weight_LV = self.out_LV.weight, self.out_LV.bias
                Out_weight, Out_b_weight = torch.cat((Out_weight, W_weight_LV), dim=1), torch.cat(
                    (Out_b_weight, b_weight_LV), dim=0)  # ' This should be N_in x (2 x N_out) or (2 X N_out)'

            return [],Out_weight, Out_b_weight


    def forward(self,x):
        '''
        The forward pass
        :param x: Input Tensor
        :type x: Tensor
        :return: output from the network
        :rtype: Tensor
        '''
        min_sigma=-4.6
        max_sigma=2
        if self.flag_only_output_layer==False:
            for layer in self.hidden:

                x=self.act(layer(x))
                if self.flag_drop_out:
                    x=self.drop_out(x)
            if self.enable_output_act==True:
                output=self.act_out(self.out(x))
                if self.flag_LV:
                    output_LV=self.act_out(self.out_LV(x))
                    # clamp
                    output_LV=torch.clamp(output_LV,min=min_sigma,max=max_sigma) # Corresponds to 0.1 sigma
            else:
                output=self.out(x)
                if self.flag_LV:
                    output_LV=self.out_LV(x)
                    # clamp
                    output_LV = torch.clamp(output_LV, min=min_sigma,max=max_sigma)  # Corresponds to 0.1 sigma


        else:
            if self.enable_output_act==True:
                output=self.act_out(self.out(x))
                if self.flag_LV:
                    output_LV=self.act_out(self.out_LV(x))
                    # clamp
                    output_LV = torch.clamp(output_LV, min=min_sigma,max=max_sigma)  # Corresponds to 0.1 sigma
            else:
                output=self.out(x)
                if self.flag_LV:
                    output_LV=self.out_LV(x)
                    # clamp
                    output_LV = torch.clamp(output_LV, min=min_sigma,max=max_sigma)  # Corresponds to 0.1 sigma

        output=self.add_const+self.output_const*output
        if self.flag_LV:
            output=torch.cat((output,output_LV),dim=-1)
        return output


    def _add_new_node_last_random(self,new_dimension):
        '''
        This is to add new node at the output layer (Used for streaming train only)
        :param new_dimension:
        :type new_dimension:
        :return:
        :rtype:
        '''
        if self.flag_LV:
            raise NotImplementedError
        current_out_weight=self.out.weight.data
        current_out_bias=self.out.bias.data

        # new weight matrix and bias
        w_new=0.01*torch.randn(new_dimension,self.hidden_unit[-1])
        bias_new=0.01*torch.randn(new_dimension)

        # concat to old matrix and bias
        w_new_mat=torch.cat((current_out_weight,w_new),dim=0)

        bias_new_mat=torch.cat((current_out_bias,bias_new),dim=0)

        # set current weight/bias to new weight/bias
        if self.flag_only_output_layer == False:
            self.out=nn.Linear(self.hidden_unit[-1],self.output_dim+new_dimension)
        else:
            self.out = nn.Linear(self.input_dim, self.output_dim+new_dimension)

        self.out.weight.data=torch.tensor(w_new_mat.data,requires_grad=True)
        self.out.weight.bias=torch.tensor(bias_new_mat.data,requires_grad=True)


class Point_Net_Plus_BNN_SGHMC(object):
    def __init__(self,latent_dim,obs_dim,dim_before_agg,encoder_layer_num_before_agg=1,encoder_hidden_before_agg=None,encoder_layer_num_after_agg=1,
               encoder_hidden_after_agg=[100],embedding_dim=10,decoder_layer_num=2,decoder_hidden=[50,100],pooling='Sum',output_const=1.,add_const=0.,sample_z=1,sample_W=1,W_sigma_prior=0.1, pooling_act='Sigmoid',flag_log_q=False
               ):
        # Store argument
        self.latent_dim = latent_dim
        self.encoder_layer_num_before_agg = encoder_layer_num_before_agg
        self.encoder_layer_num_after_agg = encoder_layer_num_after_agg
        self.encoder_hidden_before_agg = encoder_hidden_before_agg
        self.encoder_hidden_after_agg = encoder_hidden_after_agg
        self.embedding_dim = embedding_dim
        self.obs_dim = obs_dim
        self.dim_before_agg = dim_before_agg
        self.decoder_layer_num = decoder_layer_num
        self.decoder_hidden = decoder_hidden
        self.output_const = output_const
        self.pooling = pooling
        self.sample_z = sample_z
        self.sample_W=sample_W
        self.W_sigma_prior=W_sigma_prior
        self.pooling_act=pooling_act
        self.add_const=add_const
        self.flag_log_q=flag_log_q
        self.flag_LV=False
        # Default arguments
        self.flag_encoder_before_agg=True

        # Define Pooling function
        if pooling=='Sum':
            self.pool=torch.sum
        elif pooling=='Mean':
            self.pool=torch.mean
        elif pooling=='Max':
            pass
        else:
            raise NotImplemented
        if self.pooling_act=='ReLU':
            self.act_pool=F.relu
        elif self.pooling_act=='Sigmoid':
            self.act_pool=F.sigmoid
        elif self.pooling_act=='Tanh':
            self.act_pool=F.tanh
        elif type(self.pooling_act)==type(None):
            self.act_pool=None
        elif self.pooling_act=='Softplus':
            self.act_pool=F.softplus
        else:
            raise NotImplementedError
        self.encode_embedding,self.encode_bias=self._generate_default_embedding() # 1 x obs_dim x embed_dim and 1 x obs_dim x 1

        # BNN Decoder network
        self.decoder=fc_Net(self.latent_dim,self.obs_dim,hidden_layer_num=decoder_layer_num,hidden_unit=decoder_hidden,activations='ReLU',activations_output=None,flag_only_output_layer=False,drop_out_rate=0.,flag_drop_out=False,
                                        output_const=output_const,add_const=add_const,flag_LV=self.flag_LV)

        # Whether apply the transform before pooling function
        if type(self.encoder_layer_num_before_agg)==type(None):
            self.flag_encoder_before_agg=False
            self.dim_before_agg=self.embedding_dim
        else:
            self.encoder_before_agg = fc_Net(self.embedding_dim + 2, output_dim=self.dim_before_agg,
                                                         hidden_layer_num=self.encoder_layer_num_before_agg,
                                                         hidden_unit=self.encoder_hidden_before_agg
                                                         , activations='ReLU', flag_only_output_layer=False,
                                                         activations_output='ReLU')
        # Apply after pooling
        self.encoder_after_agg = fc_Net(self.dim_before_agg, output_dim=2 * self.latent_dim,
                                                    hidden_layer_num=self.encoder_layer_num_after_agg,
                                                    hidden_unit=self.encoder_hidden_after_agg,
                                                    activations='ReLU', flag_only_output_layer=False,
                                                    activations_output=None
                                                    )

    def _extract_state_dict(self):
        decoder_state=copy.deepcopy(self.decoder.state_dict())
        encoder_before_state=copy.deepcopy(self.encoder_before_agg.state_dict())
        encoder_after_state=copy.deepcopy(self.encoder_after_agg.state_dict())
        embedding_state,embedding_bias=self.encode_embedding.clone().detach(),self.encode_bias.clone().detach()
        return decoder_state,encoder_before_state,encoder_after_state,embedding_state,embedding_bias

    def _load_state_dict(self,decoder_state,encoder_before_state,encoder_after_state,embedding_state,embedding_bias):
        self.decoder.load_state_dict(decoder_state)
        self.encoder_before_agg.load_state_dict(encoder_before_state)
        self.encoder_after_agg.load_state_dict(encoder_after_state)
        self.encode_embedding=embedding_state.clone().detach()
        self.encode_embedding.requires_grad=True
        self.encode_bias=embedding_bias.clone().detach()
        self.encode_bias.requires_grad=True

    def _generate_default_embedding(self):
        '''
        Generate the initial embeddings and bias with shape 1 x obs_dim x embedding_dim and 1 x obs_dim x 1
        :return: embedding,bias
        :rtype: Tensor, Tensor
        '''
        embedding=torch.randn(self.obs_dim,self.embedding_dim)
        embedding=torch.unsqueeze(embedding,dim=0) # 1 x obs_dim x embed_dim
        # embedding=torch.tensor(embedding.data,requires_grad=True)
        embedding = embedding.clone().detach().requires_grad_(True)

        bias=torch.randn(self.obs_dim,1)
        bias=torch.unsqueeze(bias,dim=0) # 1 x obs_dim x 1
        # bias=torch.tensor(bias.data,requires_grad=True)
        bias = bias.clone().detach().requires_grad_(True)
        return embedding,bias
    
    def _encoding(self,X,mask):
        batch_size=X.shape[0] # N x obs_dim
        mask=torch.unsqueeze(mask,dim=len(mask.shape)) # N x obs x 1
        X_expand=torch.unsqueeze(X,dim=len(X.shape))  # N x obs x 1
        # Multiplicate embedding
        if len(self.encode_embedding.shape)!=len(X_expand.shape):
            encode_embedding=torch.unsqueeze(self.encode_embedding,dim=0)
        else:
            encode_embedding=self.encode_embedding
        X_embedding=X_expand*encode_embedding # N x obs x embed_dim

        if len(self.encode_bias.shape)!=len(X_embedding.shape):
            X_bias=self.encode_bias.expand_as(X_embedding)
            X_bias=torch.index_select(X_bias,dim=-1,index=torch.tensor([0]))
        else:
            X_bias=self.encode_bias.repeat(batch_size,1,1) # N x obs x 1

        if self.flag_encoder_before_agg==True:
            X_aug=torch.cat((X_expand,X_embedding,X_bias),dim=len(X_expand.shape)-1) # N x obs_dim x (embed+2)
            output_before_agg=1*self.encoder_before_agg.forward(X_aug) # N x obs x dim_before_agg
        else:
            output_before_agg=X_embedding+X_bias # N x obs x dim_before_agg

        mask_output_before_agg=mask*output_before_agg # N x obs x dim_before_agg

        if self.pooling != 'Max':
            agg_mask_output=self.pool(mask_output_before_agg, dim=len(mask_output_before_agg.shape)-2)
            if self.act_pool:
                agg_mask_output = self.act_pool(agg_mask_output)  # N x dim_before_agg
        elif self.pooling == 'Max':
            agg_mask_output=torch.max(mask_output_before_agg,dim=len(mask_output_before_agg.shape)-2)[0]
            if self.act_pool:
                agg_mask_output=self.act_pool(agg_mask_output)
        encoding=self.encoder_after_agg.forward(agg_mask_output)
        # Encoding -3
        encoding[:,self.latent_dim:]=encoding[:,self.latent_dim:]-0.
        return encoding

    def sample_latent_variable(self,X,mask,size=10):
        batch_size = X.shape[0]
        encoding = self._encoding(X, mask)
        mean = encoding[:, :self.latent_dim]
        if self.flag_log_q == True:
            sigma = torch.clamp(torch.sqrt(torch.exp(encoding[:, self.latent_dim:])),min=1e-5,max=300.)
        else:
            sigma = torch.clamp(torch.sqrt(torch.clamp((encoding[:, self.latent_dim:]) ** 2,min=1e-8)),min=1e-5,max=300.)

        if size == 1:
            eps = torch.randn(batch_size, self.latent_dim, requires_grad=True)
            z = mean + sigma * eps  # N x latent_dim
        elif size > 1:
            eps = torch.randn(size, batch_size, self.latent_dim, requires_grad=True)  # size x N  x latent_dim
            z = torch.unsqueeze(mean, dim=0) + torch.unsqueeze(sigma, dim=0) * eps  # size x N x latent_dim
        else:
            raise NotImplementedError
        return z, encoding

    def _KL_encoder(self,mean,sigma,z_sigma_prior=1.):
        KL_z = 0.5 * (
                    math.log(z_sigma_prior ** 2) - torch.sum(torch.log(sigma ** 2), dim=1) - sigma.shape[1] + torch.sum(
                sigma ** 2 / (z_sigma_prior ** 2), dim=1) + torch.sum(mean ** 2 / (z_sigma_prior ** 2), dim=1))  # N

        return KL_z  # N
    def test_log_likelihood(self,Infer_model,X_in,X_test,W_sample,mask,sigma_out,size=10):
        mean_pred_log_likelihood,tot_pred_ll=Infer_model.test_log_likelihood(X_in, X_test, W_sample, mask, sigma_out, size=size)
        return mean_pred_log_likelihood,tot_pred_ll
    
    def completion(self,Infer_model,X,mask,W_sample,size_Z=10,record_memory=False):
        complete=Infer_model.completion(X,mask,W_sample,size_Z=size_Z,record_memory=False)
        return complete
    
    def _KL_encoder_target_ELBO(self,q_mean_XUy, q_sigma_XUy, q_mean_Xdy, q_sigma_Xdy):
        # clamp to avoid zero
        sigma = torch.clamp(torch.abs(q_sigma_XUy), min=0.001, max=10.)
        sigma_Xdy = torch.clamp(torch.abs(q_sigma_Xdy), min=0.001, max=10.)
        KL_z = torch.log(torch.abs(sigma_Xdy) / torch.abs(sigma)) + 0.5 * (sigma ** 2) / (sigma_Xdy ** 2) + 0.5 * (
                    (q_mean_XUy - q_mean_Xdy) ** 2) / (sigma_Xdy ** 2) - 0.5 # N x latent_dim
        return torch.sum(KL_z, dim=1)

    def decoding(self,Infer_model,z,W_sample):
        X=Infer_model.sample_X(z,W_sample)
        return X

