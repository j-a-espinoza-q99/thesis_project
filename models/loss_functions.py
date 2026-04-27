"""
Multi-objective loss functions for the custom model.

Implements the loss from the Evaluation Framework:
L = L_BPR + λ_adv * L_adv + λ_div * L_div + λ_pop * L_pop + λ_aug * L_aug

Where:
- L_BPR: Bayesian Personalized Ranking (pairwise ranking)
- L_adv: Adversarial Fairness (discriminator for protected attributes)
- L_div: Diversity Regularization (Determinantal Point Process via pairwise similarity)
- L_pop: Popularity Debiasing (Inverse Propensity Scoring)
- L_aug: Augmented Description Consistency (InfoNCE contrastive)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class MultiObjectiveLoss(nn.Module):
    """
    Combined multi-objective loss for fair, diverse, and debiased recommendations.
    """

    def __init__(
            self,
            bpr_weight: float = 1.0,
            adv_weight: float = 0.1,
            div_weight: float = 0.05,
            pop_weight: float = 0.1,
            aug_weight: float = 0.2,
            aug_temperature: float = 0.07,
            div_topk: int = 10,
            adv_hidden_dim: int = 128,
            protected_attribute: Optional[str] = None,
            device: str = 'cuda',
    ):
        super().__init__()
        self.bpr_weight = bpr_weight
        self.adv_weight = adv_weight
        self.div_weight = div_weight
        self.pop_weight = pop_weight
        self.aug_weight = aug_weight
        self.aug_temperature = aug_temperature
        self.div_topk = div_topk
        self.device = device

        # Adversarial fairness discriminator
        self.discriminator = FairnessDiscriminator(
            input_dim=768,  # Will be set dynamically
            hidden_dim=adv_hidden_dim,
            num_classes=2,  # Binary protected attribute
        )

        # Track losses for logging
        self.loss_history = {
            'bpr_loss': [],
            'adv_loss': [],
            'div_loss': [],
            'pop_loss': [],
            'aug_loss': [],
            'total_loss': [],
        }

    def forward(
            self,
            user_embeddings: torch.Tensor,
            pos_item_embeddings: torch.Tensor,
            neg_item_embeddings: torch.Tensor,
            all_item_embeddings: torch.Tensor,
            orig_item_embeddings: Optional[torch.Tensor] = None,
            aug_item_embeddings: Optional[torch.Tensor] = None,
            item_popularities: Optional[torch.Tensor] = None,
            protected_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined multi-objective loss.

        Args:
            user_embeddings: (batch_size, dim) user representations
            pos_item_embeddings: (batch_size, dim) positive item representations
            neg_item_embeddings: (batch_size, num_neg, dim) negative item representations
            all_item_embeddings: (num_items, dim) all item representations for diversity
            orig_item_embeddings: original item embeddings for augmentation loss
            aug_item_embeddings: augmented item embeddings for augmentation loss
            item_popularities: popularity of each item for debiasing
            protected_labels: protected attribute labels for fairness

        Returns:
            total_loss, loss_dict
        """
        losses = {}

        # 1. Bayesian Personalized Ranking (BPR) Loss
        bpr_loss = self._bpr_loss(
            user_embeddings, pos_item_embeddings, neg_item_embeddings
        )
        losses['bpr_loss'] = bpr_loss

        # 2. Adversarial Fairness Loss
        if protected_labels is not None and self.adv_weight > 0:
            adv_loss = self._adversarial_fairness_loss(
                user_embeddings, pos_item_embeddings, protected_labels
            )
            losses['adv_loss'] = adv_loss
        else:
            losses['adv_loss'] = torch.tensor(0.0, device=self.device)

        # 3. Diversity Regularization (DPP via pairwise similarity)
        if self.div_weight > 0:
            div_loss = self._diversity_loss(all_item_embeddings)
            losses['div_loss'] = div_loss
        else:
            losses['div_loss'] = torch.tensor(0.0, device=self.device)

        # 4. Popularity Debiasing (Inverse Propensity Scoring)
        if item_popularities is not None and self.pop_weight > 0:
            pop_loss = self._popularity_debiasing_loss(
                user_embeddings, pos_item_embeddings, item_popularities
            )
            losses['pop_loss'] = pop_loss
        else:
            losses['pop_loss'] = torch.tensor(0.0, device=self.device)

        # 5. Augmented Description Consistency (InfoNCE)
        if orig_item_embeddings is not None and aug_item_embeddings is not None and self.aug_weight > 0:
            aug_loss = self._augmentation_consistency_loss(
                orig_item_embeddings, aug_item_embeddings
            )
            losses['aug_loss'] = aug_loss
        else:
            losses['aug_loss'] = torch.tensor(0.0, device=self.device)

        # Combine losses
        total_loss = (
                self.bpr_weight * bpr_loss +
                self.adv_weight * losses['adv_loss'] +
                self.div_weight * losses['div_loss'] +
                self.pop_weight * losses['pop_loss'] +
                self.aug_weight * losses['aug_loss']
        )
        losses['total_loss'] = total_loss

        # Log losses
        for key, value in losses.items():
            self.loss_history[key].append(value.item() if isinstance(value, torch.Tensor) else value)

        return total_loss, {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    def _bpr_loss(
            self,
            user_emb: torch.Tensor,
            pos_item_emb: torch.Tensor,
            neg_item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bayesian Personalized Ranking loss.

        L_BPR = -Σ log σ(x_ui - x_uj)
        where x_ui = user · positive_item, x_uj = user · negative_item
        """
        pos_scores = (user_emb * pos_item_emb).sum(dim=-1)  # (batch_size,)

        # Handle multiple negatives
        if neg_item_emb.dim() == 3:
            # (batch_size, num_negatives, dim) -> average over negatives
            neg_scores = (user_emb.unsqueeze(1) * neg_item_emb).sum(dim=-1).mean(dim=-1)
        else:
            neg_scores = (user_emb * neg_item_emb).sum(dim=-1)

        diff = pos_scores - neg_scores
        loss = -F.logsigmoid(diff).mean()

        return loss

    def _adversarial_fairness_loss(
            self,
            user_emb: torch.Tensor,
            item_emb: torch.Tensor,
            protected_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adversarial fairness loss.

        A discriminator tries to predict protected attributes from embeddings.
        The main model is trained to fool the discriminator (gradient reversal).

        L_adv = min_θ max_φ E[-ln p_φ(a | e_u, e_i)]
        """
        combined = torch.cat([user_emb, item_emb], dim=-1)  # (batch_size, 2*dim)

        # Gradient reversal: we want to maximize discriminator loss
        # (i.e., make embeddings independent of protected attributes)
        combined_reversed = GradientReversal.apply(combined, 1.0)

        pred = self.discriminator(combined_reversed)
        loss = F.cross_entropy(pred, protected_labels.long())

        return loss

    def _diversity_loss(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Diversity regularization using Determinantal Point Process (DPP).

        Penalizes high pairwise cosine similarity among top-K items.

        L_div = (1/|U|) Σ_u (1/(K(K-1))) Σ_{i≠j} sim(e_i, e_j)
        """
        # Sample a subset of items for computational efficiency
        num_items = item_embeddings.shape[0]
        k = min(self.div_topk, num_items)
        indices = torch.randperm(num_items, device=self.device)[:k]
        selected = item_embeddings[indices]  # (k, dim)

        # Normalize embeddings
        selected = F.normalize(selected, p=2, dim=-1)

        # Compute pairwise cosine similarity matrix
        sim_matrix = selected @ selected.T  # (k, k)

        # Zero out diagonal (self-similarity)
        mask = torch.eye(k, device=self.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, 0.0)

        # Average pairwise similarity
        num_pairs = k * (k - 1)
        if num_pairs > 0:
            div_loss = sim_matrix.abs().sum() / num_pairs
        else:
            div_loss = torch.tensor(0.0, device=self.device)

        return div_loss

    def _popularity_debiasing_loss(
            self,
            user_emb: torch.Tensor,
            pos_item_emb: torch.Tensor,
            item_popularities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Popularity debiasing using Inverse Propensity Scoring (IPS).

        Down-weights popular items to prevent over-recommendation.

        L_pop = -Σ (1/n_i) ln x_{u,i}
        where n_i is the interaction count (popularity) of item i.
        """
        scores = (user_emb * pos_item_emb).sum(dim=-1)  # (batch_size,)

        # Apply inverse propensity weights (1/popularity)
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        weights = 1.0 / (item_popularities.float() + eps)

        # Normalize weights
        weights = weights / weights.sum()

        loss = -(weights * F.logsigmoid(scores)).sum()

        return loss

    def _augmentation_consistency_loss(
            self,
            orig_embeddings: torch.Tensor,
            aug_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive InfoNCE loss for augmented description consistency.

        Encourages alignment between original and augmented item embeddings.

        L_aug = -Σ log (exp(sim(e_i^orig, e_i^aug)/τ) / Σ_j exp(sim(e_i^orig, e_j^aug)/τ))
        """
        batch_size = orig_embeddings.shape[0]

        # Normalize embeddings
        orig = F.normalize(orig_embeddings, p=2, dim=-1)
        aug = F.normalize(aug_embeddings, p=2, dim=-1)

        # Compute similarity matrix
        sim_matrix = orig @ aug.T / self.aug_temperature  # (batch_size, batch_size)

        # Labels: diagonal is positive pair
        labels = torch.arange(batch_size, device=self.device)

        # InfoNCE loss (symmetric)
        loss_orig_to_aug = F.cross_entropy(sim_matrix, labels)
        loss_aug_to_orig = F.cross_entropy(sim_matrix.T, labels)

        loss = (loss_orig_to_aug + loss_aug_to_orig) / 2

        return loss


class FairnessDiscriminator(nn.Module):
    """Adversarial discriminator for fairness."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None