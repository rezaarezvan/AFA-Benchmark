from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn, optim

import wandb
from afa_discriminative.utils import restore_parameters
from afa_rl.utils import mask_data


class MaskingPretrainer(nn.Module):
    """Pretrain model with missing features."""

    def __init__(self, model, mask_layer):
        super().__init__()
        self.model = model
        self.mask_layer = mask_layer

    def fit(
        self,
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
        verbose=True,
        min_mask=0.1,
        max_mask=0.9,
        tag="eval_loss",
    ):
        """Train model."""
        wandb.watch(self, log="all", log_freq=100)
        # Verify arguments.
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = "min"
        elif val_loss_mode is None:
            raise ValueError(
                "must specify val_loss_mode (min or max) when validation_loss_fn is specified"
            )

        # Set up optimizer and lr scheduler.
        model = self.model
        mask_layer = self.mask_layer
        device = next(model.parameters()).device
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=val_loss_mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

        # Determine mask size.
        if hasattr(mask_layer, "mask_size") and (
            mask_layer.mask_size is not None
        ):
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
            epoch_train_loss = 0.0
            for x, y in train_loader:
                # Move to device.
                x = x.to(device)
                y = y.to(device)

                # Generate missingness.
                p = min_mask + torch.rand(1).item() * (max_mask - min_mask)
                _, m, _ = mask_data(x, p)

                # Calculate loss.
                x_masked = mask_layer(x, m)
                pred = model(x_masked)
                loss = loss_fn(pred, y)

                # Take gradient step.
                loss.backward()
                opt.step()
                model.zero_grad()
                epoch_train_loss += loss.item()

            avg_train = epoch_train_loss / len(train_loader)

            # Calculate validation loss.
            model.eval()
            with torch.no_grad():
                # For mean loss.
                pred_list = []
                label_list = []

                for x, y in val_loader:
                    # Move to device.
                    x = x.to(device)
                    y = y.to(device)

                    # Generate missingness.
                    p = min_mask + torch.rand(1).item() * (max_mask - min_mask)

                    # Calculate prediction.
                    _, m, _ = mask_data(x, p)
                    x_masked = mask_layer(x, m)
                    pred = model(x_masked)
                    pred_list.append(pred)
                    label_list.append(y)

                # Calculate loss.
                y = torch.cat(label_list, 0)
                pred = torch.cat(pred_list, 0)
                val_loss = val_loss_fn(pred, y).item()

            if tag == "eval_loss":
                wandb.log(
                    {
                        "masked_predictor/train_loss": avg_train,
                        "masked_predictor/val_loss": val_loss,
                    },
                    step=epoch,
                )
            elif tag == "eval_accuracy":
                wandb.log(
                    {
                        "masked_predictor/train_loss": avg_train,
                        "masked_predictor/val_accuracy": val_loss,
                    },
                    step=epoch,
                )
            else:
                raise KeyError("Invalid tag")

            # Print progress.
            if verbose:
                print(f"{'-' * 8}Epoch {epoch + 1}{'-' * 8}")
                print(f"Val loss = {val_loss:.4f}\n")

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
                    print(f"Stopping early at epoch {epoch + 1}")
                break

        # Copy parameters from best model.
        restore_parameters(model, best_model)
        wandb.unwatch(self)

    def evaluate(self, loader, metric):
        """
        Evaluate mean performance across a dataset.

        Args:
          loader:
          metric:

        """
        # Setup.
        self.model.eval()
        device = next(self.model.parameters()).device

        # Determine mask size.
        if hasattr(self.mask_layer, "mask_size") and (
            self.mask_layer.mask_size is not None
        ):
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
        """
        Generate model prediction.

        Args:
          x:
          mask:

        """
        x_masked = self.mask_layer(x, mask)
        return self.model(x_masked)


class fc_Net(nn.Module):
    """This class implements the base network structure for fully connected encoder/decoder/predictor."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_num=2,
        hidden_unit=[100, 50],
        activations="ReLU",
        activations_output=None,
        flag_only_output_layer=False,
        drop_out_rate=0.0,
        flag_drop_out=False,
        flag_LV=False,
        output_const=1.0,
        add_const=0.0,
    ):
        """
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
        """
        super(fc_Net, self).__init__()
        self.drop_out_rate = drop_out_rate
        self.flag_drop_out = flag_drop_out
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_layer_num = hidden_layer_num
        self.hidden_unit = hidden_unit
        self.flag_only_output_layer = flag_only_output_layer
        self.flag_LV = flag_LV
        self.output_const = output_const
        self.add_const = add_const
        if self.flag_LV == True:
            self.LV_dim = output_dim
        self.enable_output_act = False
        self.drop_out = nn.Dropout(self.drop_out_rate)
        # activation functions
        self.activations = activations
        if activations == "ReLU":
            self.act = F.relu
        elif activations == "Sigmoid":
            self.act = F.sigmoid
        elif activations == "Tanh":
            self.act = F.tanh
        elif activations == "Elu":
            self.act = F.elu
        elif activations == "Selu":
            self.act = F.selu
        else:
            raise NotImplementedError

        if activations_output == "ReLU":
            self.enable_output_act = True
            self.act_out = F.relu
        elif activations_output == "Sigmoid":
            self.enable_output_act = True
            self.act_out = F.sigmoid
        elif activations_output == "Tanh":
            self.enable_output_act = True
            self.act_out = F.tanh
        elif activations_output == "Elu":
            self.enable_output_act = True
            self.act_out = F.elu
        elif activations_output == "Selu":
            self.enable_output_act = True
            self.act_out = F.selu
        elif activations_output == "Softplus":
            self.enable_output_act = True
            self.act_out = F.softplus

        # whether to use multi NN or single layer NN
        if self.flag_only_output_layer == False:
            assert len(self.hidden_unit) == hidden_layer_num, (
                "Hidden layer unit length %s inconsistent with layer number %s"
                % (len(self.hidden_unit), self.hidden_layer_num)
            )

            # build hidden layers
            self.hidden = nn.ModuleList()
            for layer_ind in range(self.hidden_layer_num):
                if layer_ind == 0:
                    self.hidden.append(
                        nn.Linear(self.input_dim, self.hidden_unit[layer_ind])
                    )
                else:
                    self.hidden.append(
                        nn.Linear(
                            self.hidden_unit[layer_ind - 1],
                            self.hidden_unit[layer_ind],
                        )
                    )

            # output layer
            self.out = nn.Linear(self.hidden_unit[-1], self.output_dim)
            if self.flag_LV:
                self.out_LV = nn.Linear(self.hidden_unit[-1], self.LV_dim)

        else:
            self.out = nn.Linear(self.input_dim, self.output_dim)
            if self.flag_LV:
                self.out_LV = nn.Linear(self.hidden_unit[-1], self.LV_dim)

    def forward(self, x):
        """
        The forward pass
        :param x: Input Tensor
        :type x: Tensor
        :return: output from the network
        :rtype: Tensor
        """
        min_sigma = -4.6
        max_sigma = 2
        if self.flag_only_output_layer == False:
            for layer in self.hidden:
                x = self.act(layer(x))
                if self.flag_drop_out:
                    x = self.drop_out(x)
            if self.enable_output_act == True:
                output = self.act_out(self.out(x))
                if self.flag_LV:
                    output_LV = self.act_out(self.out_LV(x))
                    # clamp
                    output_LV = torch.clamp(
                        output_LV, min=min_sigma, max=max_sigma
                    )  # Corresponds to 0.1 sigma
            else:
                output = self.out(x)
                if self.flag_LV:
                    output_LV = self.out_LV(x)
                    # clamp
                    output_LV = torch.clamp(
                        output_LV, min=min_sigma, max=max_sigma
                    )  # Corresponds to 0.1 sigma

        elif self.enable_output_act == True:
            output = self.act_out(self.out(x))
            if self.flag_LV:
                output_LV = self.act_out(self.out_LV(x))
                # clamp
                output_LV = torch.clamp(
                    output_LV, min=min_sigma, max=max_sigma
                )  # Corresponds to 0.1 sigma
        else:
            output = self.out(x)
            if self.flag_LV:
                output_LV = self.out_LV(x)
                # clamp
                output_LV = torch.clamp(
                    output_LV, min=min_sigma, max=max_sigma
                )  # Corresponds to 0.1 sigma

        output = self.add_const + self.output_const * output
        if self.flag_LV:
            output = torch.cat((output, output_LV), dim=-1)
        return output
