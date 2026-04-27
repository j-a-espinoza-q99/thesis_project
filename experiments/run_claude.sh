#!/usr/bin/env python
"""
Claude + Voyage AI Evaluation Script.

Uses Voyage AI for embeddings and Claude for chat-based ranking.
"""
import os
import sys
import json
import argparse
import logging

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.claude_voyage_model import ClaudeVoyageRecommender
from data.preprocessing import AmazonReviewsPreprocessor
from evaluation.metrics import RecommendationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', default=['All_Beauty'])
    parser.add_argument('--output_dir', type=str, default='results/claude_voyage')
    parser.add_argument('--sample_size', type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    claude = ClaudeVoyageRecommender()
    metrics = RecommendationMetrics(topk_values=[5, 10, 50, 100])
    preprocessor = AmazonReviewsPreprocessor(domains=args.domains)

    results = {}

    for domain in args.domains:
        logger.info(f"\nEvaluating Claude/Voyage on {domain}")

        data = preprocessor.load_and_process_domain(domain)
        item_metadata = data[0]
        reviews_df = data[1]

        item_texts = list(item_metadata.values())
        item_ids = list(item_metadata.keys())

        # Get Voyage embeddings for items
        logger.info("Generating Voyage embeddings for items...")
        item_embeddings = claude.encode_items_batch(item_texts)

        # Sample test queries
        test_queries = reviews_df.sample(n=min(args.sample_size, len(reviews_df)))

        all_predictions = []
        all_ground_truth = []

        for _, row in tqdm(test_queries.iterrows(), total=len(test_queries)):
            query = row['cleaned_review']
            target = row['parent_asin']

            candidate_indices = np.random.choice(
                len(item_texts), min(50, len(item_texts)), replace=False
            )
            candidates = [item_texts[i] for i in candidate_indices]

            try:
                ranked = claude.rank_items_with_claude(query, candidates)
                pred_indices = [candidate_indices[r] for r in ranked if r < len(candidate_indices)]
            except Exception as e:
                logger.warning(f"Ranking failed: {e}")
                pred_indices = candidate_indices[:10].tolist()

            all_predictions.append(pred_indices)
            all_ground_truth.append([target])

        results[domain] = metrics.evaluate_all(
            predictions=all_predictions,
            ground_truth=[[item_ids.index(g) if g in item_ids else 0 for g in truths]
                         for truths in all_ground_truth],
        )

    with open(f"{args.output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()