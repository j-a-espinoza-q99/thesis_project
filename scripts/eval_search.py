#!/usr/bin/env python
"""
Standalone evaluation for product search tasks.

Usage:
    python eval_search.py --model blair-base --dataset esci --output_dir results/esci
    python eval_search.py --model blair-base --dataset amazon_c4 --output_dir results/amazon_c4
"""
import os, sys, argparse, logging, json
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.blair_model import BLAIRBaseline
from evaluation.metrics import RecommendationMetrics
from data.esci_dataset import ESCIDataset
from data.amazon_c4_dataset import AmazonC4Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name for BLAIR')
    parser.add_argument('--dataset', type=str, choices=['esci', 'amazon_c4'], required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    blair = BLAIRBaseline(model_name=args.model, device=device)
    metrics = RecommendationMetrics(topk_values=[5, 10, 50, 100])

    if args.dataset == 'esci':
        dataset = ESCIDataset()
        # For simplicity, we provide a dummy metadata provider that loads from a file.
        # In practice, you'd load the Amazon Reviews 2023 metadata. Here we'll
        # use a placeholder because the focus is on search evaluation flow.
        def dummy_meta_provider(asins):
            # Return empty metadata; product search can be run without metadata
            return {asin: "" for asin in asins}
        queries, item_ids, item_metadata = dataset.load(item_metadata_provider=dummy_meta_provider)
    else:
        dataset = AmazonC4Dataset()
        queries, item_ids, item_metadata = dataset.load()

    # Encode all items
    all_texts = list(item_metadata.values())
    all_ids = list(item_metadata.keys())
    item_embs = blair.encode_items(all_texts, batch_size=args.batch_size)

    pred_lists = []
    true_lists = []
    for query, target in tqdm(zip(queries, item_ids), total=len(queries), desc=f"Search {args.dataset}"):
        q_emb = blair.encode_query(query)
        scores = q_emb @ item_embs.T
        topk = scores.argsort(descending=True)[0, :100].cpu().numpy()
        pred_items = [all_ids[i] for i in topk if i < len(all_ids)]
        pred_lists.append(pred_items)
        true_lists.append([target])

    results = metrics.evaluate_all(predictions=pred_lists, ground_truth=true_lists,
                                   item_embeddings=item_embs.cpu().numpy())
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {args.output_dir}/results.json")
    for k, v in results['accuracy'].items():
        logger.info(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()