from pathlib import Path
import torch
from afa_generative.utils import *
from common.custom_types import (
    AFAMethod,
    AFASelection,
    FeatureMask,
    Label,
    MaskedFeatures,
)


class Ma2018AFAMethod(AFAMethod):
    def __init__(self, sampler, predictor, num_classes, device=torch.device("cpu")):
        super().__init__()
        self.sampler = sampler
        self.predictor = predictor
        self.num_classes = num_classes
        self._device: torch.device = device

    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features=None,
        label=None,
    ) -> Label:
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)
        B, F = masked_features.shape
        zeros_label = torch.zeros(B, self.num_classes, device=self._device)
        zeros_mask = torch.zeros(
            B, self.num_classes, device=self._device, dtype=feature_mask.dtype
        )
        augmented_masked_feature = torch.cat([masked_features, zeros_label], dim=-1).to(
            self._device
        )
        augmented_feature_mask = torch.cat([feature_mask, zeros_mask], dim=-1).to(
            self._device
        )

        with torch.no_grad():
            _, _, _, z, _ = self.sampler(
                augmented_masked_feature, augmented_feature_mask
            )

        return self.predictor(z).softmax(dim=-1)

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features=None,
        label=None,
    ) -> AFASelection:
        device = self._device
        B, F = masked_features.shape
        zeros_label = torch.zeros(B, self.num_classes, device=self._device)
        zeros_mask = torch.zeros(
            B, self.num_classes, device=self._device, dtype=feature_mask.dtype
        )
        augmented_masked_feature = torch.cat([masked_features, zeros_label], dim=-1).to(
            self._device
        )
        augmented_feature_mask = torch.cat([feature_mask, zeros_mask], dim=-1).to(
            self._device
        )

        with torch.no_grad():
            _, _, _, _, x_full = self.sampler.forward(
                augmented_masked_feature, augmented_feature_mask
            )
        x_full = x_full.view(B, F)
        x_full = torch.cat([x_full, zeros_label], dim=-1).to(self._device)

        feature_mask_all = augmented_feature_mask[:, :F]
        feature_indices = torch.eye(F, device=device, dtype=feature_mask_all.dtype)
        # onehot mask (BxFxF)
        mask_features_all = feature_mask_all.unsqueeze(1) | feature_indices.unsqueeze(0)
        mask_features_flat = mask_features_all.reshape(B * F, F)
        mask_label_all = zeros_mask.unsqueeze(1).expand(B, F, -1)
        mask_label_flat = mask_label_all.reshape(B * F, self.num_classes)
        # (B*F x (F+num_classes))
        mask_tests = torch.cat([mask_features_flat, mask_label_flat], dim=1)

        x_rep = x_full.unsqueeze(1).expand(B, F, F + self.num_classes)
        x_masks = x_rep.reshape(B * F, F + self.num_classes) * mask_tests

        with torch.no_grad():
            _, _, _, z_all, _ = self.sampler(x_masks, mask_tests)
            preds_all = self.predictor(z_all)

        if preds_all.ndim == 2 and preds_all.size(1) > 1:
            if not ((preds_all >= 0) & (preds_all <= 1)).all():
                preds_all = preds_all.softmax(dim=1)
        else:
            preds_all = preds_all.sigmoid().view(-1, 1)
            preds_all = torch.cat([1 - preds_all, preds_all], dim=1)

        mean_all = preds_all.mean(dim=0, keepdim=True)
        kl_all = torch.sum(
            preds_all * torch.log(preds_all / (mean_all + 1e-6) + 1e-6), dim=1
        )

        scores = kl_all.view(B, F)
        # avoid choosing the masked feature j
        scores = scores.masked_fill(feature_mask_all.bool(), float("-inf"))
        best_j = scores.argmax(dim=1)
        return best_j

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(
            str(path / "model.pt"), map_location=device, weights_only=False
        )
        sampler = checkpoint["sampler"]
        predictor = checkpoint["predictor"]
        num_classes = checkpoint["num_classes"]

        predictor = predictor.to(device)
        sampler = sampler.to(device)
        return cls(sampler, predictor, num_classes, device)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sampler": self.sampler,
                "predictor": self.predictor,
                "num_classes": self.num_classes,
            },
            str(path / "model.pt"),
        )

    def to(self, device):
        self.sampler = self.sampler.to(device)
        self.predictor = self.predictor.to(device)
        self._device = device
        return self

    @property
    def device(self) -> torch.device:
        return self._device
