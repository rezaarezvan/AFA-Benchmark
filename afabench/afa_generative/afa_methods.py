import torch

from pathlib import Path

from afabench.common.custom_types import (
    AFAMethod,
    AFASelection,
    FeatureMask,
    Label,
    MaskedFeatures,
)


class Ma2018AFAMethod(AFAMethod):
    def __init__(
        self,
        sampler,
        predictor,
        num_classes,
        device=torch.device("cpu"),
        lambda_threshold: float = -float("inf"),
        feature_costs: torch.Tensor | None = None,
        num_mc_samples: int = 128,
    ):
        super().__init__()
        self.sampler = sampler
        self.predictor = predictor
        self.num_classes = num_classes
        self._device: torch.device = device
        self.lambda_threshold = lambda_threshold
        self._feature_costs = feature_costs
        self.num_mc_samples = num_mc_samples

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 2 and logits.size(1) > 1:
            return logits.softmax(dim=1)
        else:
            probs = logits.sigmoid().view(-1, 1)
            return torch.cat([1 - probs, probs], dim=1)

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
        augmented_masked_feature = torch.cat(
            [masked_features, zeros_label], dim=-1
        ).to(self._device)
        augmented_feature_mask = torch.cat(
            [feature_mask, zeros_mask], dim=-1
        ).to(self._device)

        probs_list = []
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                _, _, _, z, _ = self.sampler(
                    augmented_masked_feature, augmented_feature_mask
                )
                logits = self.predictor(z)
                probs = logits.softmax(dim=-1)
                probs_list.append(probs)
        probs_mean = torch.stack(probs_list, dim=0).mean(dim=0)
        return probs_mean

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features=None,
        label=None,
    ) -> AFASelection:
        device = self._device
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)
        B, F = masked_features.shape
        zeros_label = torch.zeros(B, self.num_classes, device=self._device)
        zeros_mask = torch.zeros(
            B, self.num_classes, device=self._device, dtype=feature_mask.dtype
        )
        augmented_masked_feature = torch.cat(
            [masked_features, zeros_label], dim=-1
        ).to(self._device)
        augmented_feature_mask = torch.cat(
            [feature_mask, zeros_mask], dim=-1
        ).to(self._device)

        with torch.no_grad():
            _, _, _, _, x_full = self.sampler.forward(
                augmented_masked_feature, augmented_feature_mask
            )
        x_full = x_full.view(B, F)
        x_filled = masked_features.clone()
        missing = ~feature_mask.bool()
        x_filled[missing] = x_full[missing]
        x_full = torch.cat([x_filled, zeros_label], dim=-1).to(self._device)

        feature_mask_all = augmented_feature_mask[:, :F]
        feature_indices = torch.eye(
            F, device=device, dtype=feature_mask_all.dtype
        )
        # onehot mask (BxFxF)
        mask_features_all = feature_mask_all.unsqueeze(
            1
        ) | feature_indices.unsqueeze(0)
        mask_features_flat = mask_features_all.reshape(B * F, F)
        mask_label_all = zeros_mask.unsqueeze(1).expand(B, F, -1)
        mask_label_flat = mask_label_all.reshape(B * F, self.num_classes)
        # (B*F x (F+num_classes))
        mask_tests = torch.cat([mask_features_flat, mask_label_flat], dim=1)

        x_rep = x_full.unsqueeze(1).expand(B, F, F + self.num_classes)
        x_masks = x_rep.reshape(B * F, F + self.num_classes) * mask_tests

        preds_mc = []
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                _, _, _, z_all, _ = self.sampler(x_masks, mask_tests)
                logits_all = self.predictor(z_all)
                probs_all = self._logits_to_probs(logits_all)
                preds_mc.append(probs_all)
        preds_all = torch.stack(preds_mc, dim=0)
        # S: num_mc_samples
        S, BF, C = preds_all.shape
        # 1/n Σ p(y|x_s, x_i^j)
        mean_preds = preds_all.mean(dim=0)
        # KL(p_s || mean), (S, B*F)
        kl_all = (
            preds_all * (
                (preds_all + 1e-6).log() - (mean_preds.unsqueeze(0) + 1e-6).log()
            )
        ).sum(dim=-1)
        kl_mean_flat = kl_all.mean(dim=0)

        scores = kl_mean_flat.view(B, F)
        # avoid choosing the already masked features
        scores = scores.masked_fill(feature_mask_all.bool(), float("-inf"))
        if self._feature_costs is not None:
            costs = self._feature_costs.to(self._device)
            costs = torch.clamp(costs, min=1e-12)
            scores = scores / costs.unsqueeze(0)
        best_scores, best_idx = scores.max(dim=1)
        lam = self.lambda_threshold
        stop_mask = best_scores < lam
        stop_mask = stop_mask | (best_scores < -1e5)
        selections = (best_idx + 1).to(torch.long)
        selections = selections.masked_fill(stop_mask, 0)
        return selections
        # best_j = scores.argmax(dim=1)
        # return best_j

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

    @property
    def has_builtin_classifier(self) -> bool:
        return True

    @property
    def cost_param(self) -> float | None:
        return float(self.lambda_threshold)

    def set_cost_param(self, cost_param: float) -> None:
        self.lambda_threshold = cost_param
