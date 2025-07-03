from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Self, final, override

from common.config_classes import Kachuee2019PQModuleConfig
from common.custom_types import (
    AFAClassifier,
    AFAPredictFn,
    FeatureMask,
    Label,
    MaskedFeatures,
)


@final
class Kachuee2019PQModule(nn.Module):
    """The architecture proposed in the paper "Opportunistic Learning: Budgeted Cost-Sensitive Learning from Data Streams", slightly simplified from the implementation found at https://github.com/mkachuee/Opportunistic/blob/master/Demo_OL_DQN.ipynb"""

    def __init__(self, n_features: int, n_classes: int, cfg: Kachuee2019PQModuleConfig):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.cfg = cfg

        # Create network
        self.layers_p = nn.ModuleList()
        self.layers_q = nn.ModuleList()

        # initialize the P-Net
        size_last = self.n_features
        for n_h in self.cfg.n_hiddens + [self.n_classes]:
            self.layers_p.append(nn.Linear(size_last, n_h))
            size_last = n_h

        # initialize the Q-Net
        size_last = self.n_features
        if self.share_pq:
            # self.n_hiddens_q = [n_h//4 for n_h in self.n_hiddens]
            self.n_hiddens_q = []
            for ind in range(len(self.cfg.n_hiddens)):
                if ind == 0:
                    self.n_hiddens_q.append(self.cfg.n_hiddens[ind])
                else:
                    n_h = (self.cfg.n_hiddens[ind - 1] * self.cfg.n_hiddens[ind]) // (
                        self.cfg.n_hiddens[ind - 1] + self.n_hiddens_q[-1]
                    )
                    self.n_hiddens_q.append(n_h)
            for n_h, n_h_q in zip(self.cfg.n_hiddens, self.n_hiddens_q):
                self.layers_q.append(nn.Linear(size_last, n_h_q))
                size_last = n_h + n_h_q
            self.layers_q.append(nn.Linear(size_last, self.n_features + 1))
        else:
            self.n_hiddens_q = self.cfg.n_hiddens
            for n_h in self.n_hiddens_q + [self.n_features + 1]:
                self.layers_q.append(nn.Linear(size_last, n_h))
                size_last = n_h

    @override
    def forward(
        self, masked_features: MaskedFeatures
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # P-Net forward path
        act_last = masked_features
        acts_p = []
        for f_layer in self.layers_p[:-1]:
            act_last = F.dropout(F.relu(f_layer(act_last)), p=self.cfg.p_dropout)
            acts_p.append(act_last)
        class_logits = self.layers_p[-1](act_last)

        # Q-Net forward path
        act_last = masked_features
        act_last = F.relu(self.layers_q[0](act_last))
        for f_layer, p_act in zip(self.layers_q[1:-1], acts_p[:-1]):
            p_act = p_act.detach()
            act_last = F.relu(f_layer(torch.cat([act_last, p_act], dim=1)))
        p_act = acts_p[-1].detach()
        qvalues = self.layers_q[-1](torch.cat([act_last, p_act], dim=1))

        return class_logits, qvalues

    def confidence(self, x: torch.Tensor, mcdrop_samples: int = 1):
        """
        calculate the confidence histogrram for each class given a sample.
        x: input sample
        mcdrop_samples: mc dropout samples to use
        """
        x_rep = x.expand(mcdrop_samples, -1)
        class_logits, _qvalues = self.forward(x_rep)
        class_probabilities = class_logits.softmax(dim=-1)
        return class_probabilities.mean(dim=0)

    def predict(self, x: torch.Tensor, mcdrop_samples: int = 1):
        """
        make class prediction for a sample.
        x: input sample
        mcdrop_samples: mc dropout samples to use
        """
        conf = self.confidence(x, mcdrop_samples)
        pred = torch.argmax(conf)
        return pred


@final
class Kachuee2019AFAPredictFn(AFAPredictFn):
    """A wrapper for Kachuee2019PQModule to make it compatible with the AFAPredictFn interface."""

    def __init__(self, pq_module: Kachuee2019PQModule):
        super().__init__()
        self.pq_module = pq_module

    @override
    def __call__(
        self, masked_features: MaskedFeatures, _feature_mask: FeatureMask
    ) -> Label:
        class_logits, _qvalues = self.pq_module.forward(masked_features)
        return class_logits.softmax(dim=-1)


@final
class Kachuee2019AFAClassifier(AFAClassifier):
    """A wrapper for Kachuee2019PQModule to make it compatible with the AFAClassifier interface."""

    def __init__(
        self,
        pq_module: Kachuee2019PQModule,
        device: torch.device,
    ):
        super().__init__()
        self._device = device
        self.pq_module = pq_module.to(self._device)

    @override
    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        class_logits, _qvalues = self.pq_module(masked_features)
        return class_logits.softmax(dim=-1).to(original_device)

    @override
    def save(self, path: Path) -> None:
        torch.save(self.pq_module.cpu(), path)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        pq_module = torch.load(path, weights_only=False, map_location=device)
        return cls(pq_module, device)

    @override
    def to(self, device: torch.device) -> Self:
        self.pq_module = self.pq_module.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device
