#!/usr/bin/env python
"""
BLAIR fine-tuning script for sequential recommendation.
Uses BLAIR embedding as item features, fine-tunes a lightweight UniSRec model
or directly uses BLAIR embeddings in a zero-shot manner.
"""
import os, sys, argparse, logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.blair_model import BLAIRBaseline
from data.preprocessing import AmazonReviewsPreprocessor
from data.dataset import SequentialRecommendationDataset
from evaluation.evaluate import RecommenderEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleSequentialRecommender(torch.nn.Module):
    """A simple model that uses BLAIR embeddings with an MLP to predict next item."""
    def __init__(self, embedding_dim: int, num_items: int):
        super().__init__()
        self.item_embeddings = torch.nn.Parameter(torch.randn(num_items, embedding_dim))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, input_seq_emb: torch.Tensor) -> torch.Tensor:
        # input_seq_emb: (batch, embedding_dim)
        user_emb = self.mlp(input_seq_emb)
        scores = user_emb @ self.item_embeddings.T
        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='hyp1231/blair-roberta-base')
    parser.add_argument('--task', type=str, default='sequential_recommendation')
    parser.add_argument('--domains', nargs='+', default=['All_Beauty'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='results/blair/seq_rec')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    blair = BLAIRBaseline(model_name=args.model, device=device)
    preprocessor = AmazonReviewsPreprocessor(domains=args.domains)

    for domain in args.domains:
        logger.info(f"Training BLAIR-based sequential recommender on {domain}")
        data = preprocessor.process_all_domains()
        domain_data = data[domain]

        # Prepare datasets
        train_dataset = SequentialRecommendationDataset(
            sequences_df=domain_data['train_sequences'],
            item_metadata=domain_data['item_metadata'],
            mode='train',
        )
        test_dataset = SequentialRecommendationDataset(
            sequences_df=domain_data['test_sequences'],
            item_metadata=domain_data['item_metadata'],
            mode='eval',
        )
        # Extract texts for all items and encode
        all_item_texts = list(domain_data['item_metadata'].values())
        item_embs_tensor = blair.encode_items(all_item_texts).to(device)  # (num_items, dim)
        num_items = item_embs_tensor.shape[0]

        # Initialize simple model
        model = SimpleSequentialRecommender(embedding_dim=blair.embedding_dim, num_items=num_items).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                # Convert input sequences to embeddings using BLAIR
                input_texts = batch['input_texts']  # list of list of strings
                # Average embedding of sequence items
                seq_embs = []
                for texts in input_texts:
                    emb = blair.encode(texts).mean(axis=0)
                    seq_embs.append(emb)
                seq_embs = torch.tensor(np.stack(seq_embs), device=device)
                targets = batch['target_idx'].to(device)
                scores = model(seq_embs)
                loss = criterion(scores, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # Evaluate
        model.eval()
        evaluator = RecommenderEvaluator()
        test_predictions = []
        test_truth = []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in test_loader:
                input_texts = batch['input_texts']
                seq_embs = []
                for texts in input_texts:
                    emb = blair.encode(texts).mean(axis=0)
                    seq_embs.append(emb)
                seq_embs = torch.tensor(np.stack(seq_embs), device=device)
                scores = model(seq_embs)
                _, topk = torch.topk(scores, k=100, dim=-1)
                test_predictions.extend(topk.cpu().numpy().tolist())
                test_truth.extend([[t.item()] for t in batch['target_idx']])

        results = evaluator.metrics.evaluate_all(
            predictions=test_predictions,
            ground_truth=test_truth,
            item_embeddings=item_embs_tensor.cpu().numpy(),
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"{domain}_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Domain {domain} results: {results['accuracy']}")


if __name__ == '__main__':
    main()