"""
Main evaluation script for recommender models.

Supports sequential recommendation, product search, and complex product search.
Generates prediction files and computes all metrics using the comprehensive
metrics module.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Callable

import numpy as np
import torch
from tqdm import tqdm

from .metrics import RecommendationMetrics
from .fairness_metrics import FairnessMetrics

logger = logging.getLogger(__name__)


class RecommenderEvaluator:
    """
    Unified evaluator for recommendation tasks.
    """

    def __init__(
            self,
            metrics_calculator: Optional[RecommendationMetrics] = None,
            topk_values: List[int] = None,
            include_fairness: bool = False,
    ):
        self.metrics = metrics_calculator or RecommendationMetrics(topk_values=topk_values or [5, 10, 50, 100])
        self.topk_values = topk_values or [5, 10, 50, 100]
        self.include_fairness = include_fairness
        if include_fairness:
            self.fairness_metrics = FairnessMetrics()

    def evaluate_sequential(
            self,
            model,
            test_sequences: pd.DataFrame,
            item_metadata: Dict[str, str],
            item_encoder: Callable[[List[str]], np.ndarray],
            batch_size: int = 128,
            verbose: bool = True,
    ) -> Dict:
        """
        Evaluate sequential recommendation: next-item prediction.

        Args:
            model: recommendation model (e.g., UniSRec) or None for zero-shot retrieval.
            test_sequences: DataFrame with 'item_sequence' column.
            item_metadata: dict mapping ASIN -> text.
            item_encoder: function that takes list of texts and returns embeddings.
        """
        id_to_idx = {item_id: idx for idx, item_id in enumerate(item_metadata.keys())}
        idx_to_id = {v: k for k, v in id_to_idx.items()}

        # Encode all items once
        all_texts = list(item_metadata.values())
        all_embeddings = item_encoder(all_texts)
        all_embeddings = torch.tensor(all_embeddings).cuda() if torch.cuda.is_available() else torch.tensor(all_embeddings)

        pred_lists = []
        true_items = []

        pbar = tqdm(test_sequences.iterrows(), total=len(test_sequences), desc="Eval", disable=not verbose)
        for _, row in pbar:
            seq = row['item_sequence']
            if len(seq) < 2:
                continue
            input_seq = seq[:-1]
            target = seq[-1]

            # Encode input (using last item as context, simple heuristic)
            last_item_text = item_metadata.get(input_seq[-1], "")
            input_emb = item_encoder([last_item_text])[0]
            input_emb = torch.tensor(input_emb).unsqueeze(0).cuda() if torch.cuda.is_available() else torch.tensor(input_emb).unsqueeze(0)

            scores = input_emb @ all_embeddings.T
            topk_idx = scores.argsort(descending=True)[0, :max(self.topk_values)].cpu().numpy()
            pred_items = [idx_to_id[i] for i in topk_idx if i < len(idx_to_id)]
            pred_lists.append(pred_items)
            true_items.append([target])

        results = self.metrics.evaluate_all(
            predictions=pred_lists,
            ground_truth=true_items,
            item_embeddings=all_embeddings.cpu().numpy(),
        )
        return results

    def evaluate_product_search(
            self,
            queries: List[str],
            ground_truth_items: List[str],
            item_metadata: Dict[str, str],
            query_encoder: Callable[[str], np.ndarray],
            item_encoder: Callable[[List[str]], np.ndarray],
            candidate_pool_provider: Optional[Callable[[str], List[str]]] = None,
            verbose: bool = True,
    ) -> Dict:
        """
        Evaluate product search (conventional or complex).
        If candidate_pool_provider is given, it supplies a pool for each query; otherwise,
        uses all items as candidates.
        """
        id_to_idx = {item_id: idx for idx, item_id in enumerate(item_metadata.keys())}
        idx_to_id = {v: k for k, v in id_to_idx.items()}

        all_texts = list(item_metadata.values())
        all_embeddings = item_encoder(all_texts)
        all_embeddings = torch.tensor(all_embeddings).cuda() if torch.cuda.is_available() else torch.tensor(all_embeddings)

        pred_lists = []
        true_lists = []

        for query, target in tqdm(zip(queries, ground_truth_items), total=len(queries), desc="Search", disable=not verbose):
            q_emb = query_encoder(query)
            q_emb = torch.tensor(q_emb).unsqueeze(0).cuda() if torch.cuda.is_available() else torch.tensor(q_emb).unsqueeze(0)

            if candidate_pool_provider:
                candidate_ids = candidate_pool_provider(target)
                candidate_texts = [item_metadata.get(c, "") for c in candidate_ids]
                cand_emb = item_encoder(candidate_texts)
                cand_emb = torch.tensor(cand_emb).cuda() if torch.cuda.is_available() else torch.tensor(cand_emb)
                scores = q_emb @ cand_emb.T
                _, topk_local = torch.topk(scores[0], min(len(candidate_ids), max(self.topk_values)))
                preds = [candidate_ids[i] for i in topk_local.cpu().numpy()]
            else:
                scores = q_emb @ all_embeddings.T
                topk_idx = scores.argsort(descending=True)[0, :max(self.topk_values)].cpu().numpy()
                preds = [idx_to_id[i] for i in topk_idx if i < len(idx_to_id)]

            pred_lists.append(preds)
            true_lists.append([target])

        results = self.metrics.evaluate_all(
            predictions=pred_lists,
            ground_truth=true_lists,
            item_embeddings=all_embeddings.cpu().numpy(),
        )
        return results

    def evaluate_fairness_aspects(
            self,
            predictions: List[List[str]],
            ground_truth: List[List[str]],
            user_groups: Dict[str, List[int]],
            item_popularities: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute additional fairness metrics if demographic information is available.
        """
        if not self.include_fairness:
            return {}
        return self.fairness_metrics.compute_all(
            predictions, ground_truth, user_groups, item_popularities
        )

    def save_predictions(self, predictions: List[List[str]], output_path: str):
        """Save predictions to a JSON file for later analysis."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Predictions saved to {output_path}")