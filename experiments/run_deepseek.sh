#!/usr/bin/env python
"""
DeepSeek API Evaluation Script.

Evaluates DeepSeek embeddings and chat-based ranking for:
1. Product search (ESCI)
2. Complex product search (Amazon-C4)
"""
import os
import sys
import json
import argparse
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.deepseek_model import DeepSeekRecommender
from data.preprocessing import AmazonReviewsPreprocessor
from evaluation.metrics import RecommendationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', default=['All_Beauty'])
    parser.add_argument('--output_dir', type=str, default='results/deepseek')
    parser.add_argument('--sample_size', type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize models
    deepseek = DeepSeekRecommender()
    metrics = RecommendationMetrics(topk_values=[5, 10, 50, 100])
    preprocessor = AmazonReviewsPreprocessor(domains=args.domains)

    results = {}

    for domain in args.domains:
        logger.info(f"\n{'='*50}\nEvaluating DeepSeek on {domain}\n{'='*50}")

        # Load data
        data = preprocessor.load_and_process_domain(domain)
        item_metadata = data[0]
        reviews_df = data[1]

        item_texts = list(item_metadata.values())
        item_ids = list(item_metadata.keys())

        # Sample test queries
        test_queries = reviews_df.sample(n=min(args.sample_size, len(reviews_df)))

        all_predictions = []
        all_ground_truth = []

        for _, row in tqdm(test_queries.iterrows(), total=len(test_queries)):
            query = row['cleaned_review']
            target = row['parent_asin']

            # Sample candidates
            candidate_indices = np.random.choice(
                len(item_texts), min(50, len(item_texts)), replace=False
            )
            candidates = [item_texts[i] for i in candidate_indices]

            try:
                ranked = deepseek.rank_items(query, candidates)
                pred_indices = [candidate_indices[r] for r in ranked if r < len(candidate_indices)]
            except Exception as e:
                logger.warning(f"Ranking failed: {e}")
                pred_indices = candidate_indices[:10].tolist()

            all_predictions.append(pred_indices)
            all_ground_truth.append([target])

        # Compute metrics
        results[domain] = metrics.evaluate_all(
            predictions=all_predictions,
            ground_truth=[[item_ids.index(g) if g in item_ids else 0 for g in truths]
                         for truths in all_ground_truth],
        )

        logger.info(f"DeepSeek {domain} results:")
        for k, v in results[domain].get('accuracy', {}).items():
            logger.info(f"  {k}: {v:.4f}")

    # Save results
    with open(f"{args.output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {args.output_dir}/results.json")


if __name__ == '__main__':
    main()