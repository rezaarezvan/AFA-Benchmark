from copy import deepcopy
import os
import math

import torch
import torch.nn.functional as F
from torch import nn, optim

import wandb
from afa_discriminative.utils import restore_parameters
from afa_rl.utils import mask_data


models_dir = "./models/pretrained_resnet_models"
model_name = {
    "resnet18": "resnet18-5c106cde.pth",
}

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def ResNet18Backbone(model):
    # Remove avgpool and fc
    backbone_modules = nn.ModuleList(model.children())[:-2]
    return nn.Sequential(*backbone_modules), model.expansion


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        try:
            ckpt_path = os.path.join(models_dir, model_name["resnet18"])
            state_dict = torch.load(ckpt_path, map_location="cpu")
        except FileNotFoundError:
            from torch.hub import load_state_dict_from_url

            state_dict = load_state_dict_from_url(
                model_urls["resnet18"],
                weights_only=False,
            )
        model.load_state_dict(state_dict)
    return model


def make_layer(block, in_planes, planes, num_blocks, stride, expansion):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        if stride >= 1:
            downsample = None

            if stride != 1 or in_planes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers.append(block(in_planes, planes, stride, downsample))
        else:
            if expansion == 1:
                layers.append(UpsamplingBlock(in_planes, planes))
            else:
                layers.append(UpsamplingBottleneckBlock(in_planes, planes))
        in_planes = planes * block.expansion
    return nn.Sequential(*layers)


class UpsamplingBlock(nn.Module):
    """Custom residual block for performing upsampling."""

    expansion = 1

    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_planes, planes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(
                in_planes,
                self.expansion * planes,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.expansion * planes),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class UpsamplingBottleneckBlock(nn.Module):
    """Custom residual block for performing upsampling."""

    expansion = 4

    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_planes, planes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(
                in_planes,
                planes * self.expansion,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(planes * self.expansion),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


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
                if x.dim() == 4:
                    n = self.mask_layer.mask_size
                    m = (torch.rand(x.size(0), n, device=device) < p).float()
                else:
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
                    if x.dim() == 4:
                        n = self.mask_layer.mask_size
                        m = (
                            torch.rand(x.size(0), n, device=device) < p
                        ).float()
                    else:
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.expansion = block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Predictor(nn.Module):
    def __init__(self, backbone, expansion, num_classes=10):
        super(Predictor, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv = nn.Conv2d(512, 256, kernel_size=3)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(256 * expansion, num_classes)
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(self.bn(self.conv(x)))
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, backbone, expansion=1, block_layer_stride=0.5):
        super(ConvNet, self).__init__()
        self.backbone = backbone
        if expansion == 1:
            # BasicBlock
            self.block_layer = make_layer(
                BasicBlock,
                512,
                256,
                2,
                stride=block_layer_stride,
                expansion=expansion,
            )
            self.conv = nn.Conv2d(256, 1, 1)
        else:
            # Bottleneck block
            # BasicBlock
            self.block_layer = make_layer(
                Bottleneck,
                2048,
                512,
                2,
                stride=block_layer_stride,
                expansion=expansion,
            )
            self.conv = nn.Conv2d(2048, 1, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.backbone(x)
        x = self.block_layer(x)
        # print(x.shape)
        x = self.conv(x)
        x = self.flatten(x)
        return x
