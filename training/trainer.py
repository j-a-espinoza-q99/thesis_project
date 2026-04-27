"""
Custom model trainer with multi-objective loss, prompt-based augmentation,
and comprehensive logging.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, List, Tuple
import logging
from tqdm import tqdm
import numpy as np
import wandb

from data.dataset import SequentialRecommendationDataset
from prompts.prompt_templates import format_prompt
from evaluation.metrics import RecommendationMetrics

logger = logging.getLogger(__name__)


class CustomModelTrainer:
    """
    Trainer for the custom recommendation model with multi-objective loss.
    """

    def __init__(
            self,
            model: nn.Module,
            config: Dict,
            device: str = 'cuda',
            use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.use_wandb = use_wandb

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('custom_learning_rate', 5e-4),
            weight_decay=config.get('custom_weight_decay', 0.01),
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('custom_epochs', 20),
        )

        # Metrics
        self.metrics = RecommendationMetrics(
            topk_values=[5, 10, 50, 100]
        )

        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = config.get('early_stopping_patience', 5)

        # Initialize wandb
        if use_wandb:
            wandb.init(project="thesis_custom_model", config=config)

    def train(
            self,
            domain_data: Dict,
            domain: str,
    ) -> Dict:
        """Train the custom model on a domain."""
        logger.info(f"Training custom model on {domain}...")

        # Build datasets
        train_dataset = SequentialRecommendationDataset(
            sequences_df=domain_data['train_sequences'],
            item_metadata=domain_data['item_metadata'],
            mode='train',
        )
        val_dataset = SequentialRecommendationDataset(
            sequences_df=domain_data['val_sequences'],
            item_metadata=domain_data['item_metadata'],
            mode='eval',
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('custom_batch_size', 256),
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('custom_batch_size', 256),
            shuffle=False,
            num_workers=4,
        )

        # Build item popularity statistics
        item_popularities = self._compute_item_popularities(domain_data)

        # Training loop
        train_losses = []
        val_metrics = []

        for epoch in range(self.config.get('custom_epochs', 20)):
            # Train one epoch
            epoch_losses = self._train_epoch(
                train_loader, item_popularities, epoch
            )
            train_losses.append(np.mean(epoch_losses))

            # Validate
            val_results = self._validate(val_loader, epoch)
            val_metrics.append(val_results)

            # Learning rate scheduling
            self.scheduler.step()

            # Early stopping
            current_metric = val_results.get('NDCG@10', 0.0)
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.patience_counter = 0
                self._save_checkpoint(domain, epoch, val_results)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Log
            logger.info(
                f"Epoch {epoch}: train_loss={np.mean(epoch_losses):.4f}, "
                f"val_NDCG@10={current_metric:.4f}"
            )

            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': np.mean(epoch_losses),
                    'val_NDCG@10': current_metric,
                    **{f'val_{k}': v for k, v in val_results.items()},
                })

        return {
            'train_losses': train_losses,
            'val_metrics': val_metrics,
            'best_val_metric': self.best_val_metric,
        }

    def _train_epoch(
            self,
            train_loader: DataLoader,
            item_popularities: torch.Tensor,
            epoch: int,
    ) -> List[float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            user_ids = batch['input_indices'][:, 0].to(self.device)  # First item as user proxy
            pos_item_ids = batch['target_idx'].to(self.device)

            # Sample negative items
            batch_size = user_ids.shape[0]
            neg_item_ids = torch.randint(
                0, self.model.item_id_embedding.num_embeddings,
                (batch_size,), device=self.device
            )

            # Get item popularities for this batch
            batch_popularities = item_popularities[pos_item_ids]

            # Forward pass with loss
            total_loss, loss_dict = self.model(
                user_ids=user_ids,
                pos_item_ids=pos_item_ids,
                neg_item_ids=neg_item_ids,
                item_popularities=batch_popularities,
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_losses.append(total_loss.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'bpr': loss_dict.get('bpr_loss', 0),
                'div': loss_dict.get('div_loss', 0),
                'pop': loss_dict.get('pop_loss', 0),
            })

        return epoch_losses

    @torch.no_grad()
    def _validate(
            self,
            val_loader: DataLoader,
            epoch: int,
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        all_predictions = []
        all_ground_truth = []

        for batch in val_loader:
            user_ids = batch['input_indices'][:, 0].to(self.device)
            target_ids = batch['target_idx'].to(self.device)

            # Get user embeddings
            user_emb = self.model.get_user_embedding(user_ids)

            # Score all items
            item_emb = self.model.item_id_embedding.weight
            scores = user_emb @ item_emb.T

            # Get top-k predictions
            _, top_k = torch.topk(scores, k=100, dim=-1)

            all_predictions.extend(top_k.cpu().tolist())
            all_ground_truth.extend([[t.item()] for t in target_ids])

        # Compute metrics
        results = {}
        for k in [5, 10, 50, 100]:
            ndcg = np.mean([
                self.metrics.ndcg_at_k(p, g, k)
                for p, g in zip(all_predictions, all_ground_truth)
            ])
            recall = np.mean([
                self.metrics.recall_at_k(p, g, k)
                for p, g in zip(all_predictions, all_ground_truth)
            ])
            results[f'NDCG@{k}'] = ndcg
            results[f'Recall@{k}'] = recall

        return results

    def evaluate(
            self,
            domain_data: Dict,
            domain: str,
    ) -> Dict:
        """Final evaluation on test set."""
        test_dataset = SequentialRecommendationDataset(
            sequences_df=domain_data['test_sequences'],
            item_metadata=domain_data['item_metadata'],
            mode='eval',
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('custom_batch_size', 256),
            shuffle=False,
            num_workers=4,
        )

        results = self._validate(test_loader, epoch=-1)

        # Also compute fairness and diversity metrics
        self.model.eval()
        item_embeddings = self.model.item_id_embedding.weight.detach().cpu().numpy()

        item_popularities = self._compute_item_popularities(domain_data).cpu().numpy()

        diversity_results = self.metrics.evaluate_all(
            predictions=[[0] * 10],  # Placeholder
            ground_truth=[[0]],
            item_embeddings=item_embeddings,
            item_popularities=item_popularities,
        )

        results.update({
            'diversity': diversity_results.get('diversity', {}),
            'popularity_bias': diversity_results.get('popularity_bias', {}),
        })

        return results

    def _compute_item_popularities(self, domain_data: Dict) -> torch.Tensor:
        """Compute item popularity counts."""
        item_counts = {}
        for _, row in domain_data['train'].iterrows():
            asin = row['parent_asin']
            item_counts[asin] = item_counts.get(asin, 0) + 1

        # Build popularity tensor aligned with item embedding indices
        num_items = self.model.item_id_embedding.num_embeddings
        popularities = torch.ones(num_items, device=self.device)

        for asin, count in item_counts.items():
            # Map ASIN to index if possible
            pass  # Implementation depends on item ID mapping

        return popularities

    def _save_checkpoint(self, domain: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint_dir = f"checkpoints/custom/{domain}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = f"{checkpoint_dir}/epoch_{epoch}_ndcg_{metrics.get('NDCG@10', 0):.4f}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, checkpoint_path)

        logger.info(f"Checkpoint saved to {checkpoint_path}")