"""
Comprehensive evaluation metrics for recommendation systems.

Covers:
- Accuracy: Recall@K, NDCG@K, MRR@K, Hit@K
- Fairness: Demographic Parity, Equal Opportunity, Disparate Impact
- Diversity: Intra-List Diversity, Coverage, Entropy
- Popularity Bias: Average Popularity, Popularity Lift, Gini Coefficient
"""
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from scipy import stats


class RecommendationMetrics:
    """Comprehensive evaluation metrics for recommender systems."""

    def __init__(self, topk_values: List[int] = None):
        self.topk_values = topk_values or [5, 10, 50, 100]

    # -------------------------------------------------------------------------
    # Accuracy Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def recall_at_k(predictions: List[int], ground_truth: List[int], k: int) -> float:
        """Recall@K: fraction of relevant items found in top-K."""
        if len(ground_truth) == 0:
            return 0.0
        top_k = set(predictions[:k])
        ground_truth = set(ground_truth)
        return len(top_k & ground_truth) / len(ground_truth)

    @staticmethod
    def ndcg_at_k(predictions: List[int], ground_truth: List[int], k: int) -> float:
        """NDCG@K: Normalized Discounted Cumulative Gain."""
        if len(ground_truth) == 0:
            return 0.0

        # Binary relevance: 1 if in ground truth, 0 otherwise
        ground_truth_set = set(ground_truth)

        # DCG
        dcg = 0.0
        for i, item in enumerate(predictions[:k]):
            if item in ground_truth_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

        # IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(ground_truth), k)):
            idcg += 1.0 / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def mrr_at_k(predictions: List[int], ground_truth: List[int], k: int) -> float:
        """MRR@K: Mean Reciprocal Rank."""
        ground_truth_set = set(ground_truth)
        for i, item in enumerate(predictions[:k]):
            if item in ground_truth_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def hit_at_k(predictions: List[int], ground_truth: List[int], k: int) -> float:
        """Hit@K: 1 if any ground truth item is in top-K."""
        return 1.0 if len(set(predictions[:k]) & set(ground_truth)) > 0 else 0.0

    # -------------------------------------------------------------------------
    # Fairness Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def demographic_parity(
            recommendations_per_group: Dict[str, List[List[int]]],
            item_groups: Dict[int, str],
            k: int = 10,
    ) -> float:
        """
        Demographic parity: measures disparity in exposure across groups.

        Lower standard deviation = better fairness.
        """
        group_exposures = defaultdict(float)
        total_exposure = 0.0

        for group, recs in recommendations_per_group.items():
            for rec_list in recs:
                for item in rec_list[:k]:
                    # Weight by position (higher weight for top positions)
                    position_weight = 1.0 / np.log2(rec_list[:k].index(item) + 2)
                    group_exposures[group] += position_weight
                    total_exposure += position_weight

        # Normalize exposures
        if total_exposure > 0:
            for group in group_exposures:
                group_exposures[group] /= total_exposure

        # Compute standard deviation of group exposures
        exposures = list(group_exposures.values())
        if len(exposures) < 2:
            return 0.0

        return np.std(exposures)

    @staticmethod
    def equal_opportunity(
            predictions: Dict[str, List[Tuple[List[int], List[int]]]],
            k: int = 10,
    ) -> float:
        """
        Equal opportunity: true positive rate parity across groups.

        For each group, compute TPR = (relevant items in top-K) / (total relevant items).
        Returns the standard deviation of TPR across groups.
        """
        group_tpr = {}

        for group, pred_truth_pairs in predictions.items():
            tprs = []
            for pred_list, truth_list in pred_truth_pairs:
                relevant_found = len(set(pred_list[:k]) & set(truth_list))
                tpr = relevant_found / max(len(truth_list), 1)
                tprs.append(tpr)
            group_tpr[group] = np.mean(tprs) if tprs else 0.0

        values = list(group_tpr.values())
        if len(values) < 2:
            return 0.0

        return np.std(values)

    @staticmethod
    def disparate_impact(
            recommendations_per_group: Dict[str, List[List[int]]],
            item_groups: Dict[int, str],
            k: int = 10,
    ) -> float:
        """
        Disparate impact ratio: ratio of positive outcomes between groups.

        Values closer to 1 indicate better fairness.
        Values < 0.8 or > 1.2 typically indicate potential bias.
        """
        group_positive_rates = {}

        for group, recs in recommendations_per_group.items():
            positive_count = sum(
                1 for rec in recs
                if any(item_groups.get(item, '') == 'positive' for item in rec[:k])
            )
            group_positive_rates[group] = positive_count / max(len(recs), 1)

        if len(group_positive_rates) < 2:
            return 1.0

        rates = list(group_positive_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)

        if max_rate == 0:
            return 1.0

        return min_rate / max_rate

    # -------------------------------------------------------------------------
    # Diversity Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def intra_list_diversity(
            recommendations: List[List[int]],
            item_embeddings: Optional[np.ndarray] = None,
            item_similarities: Optional[np.ndarray] = None,
            k: int = 10,
    ) -> float:
        """
        Intra-List Diversity (ILD): average pairwise dissimilarity within recommendation lists.

        ILD = (2/(K(K-1))) * Σ_{i=1}^{K} Σ_{j=i+1}^{K} (1 - sim(i, j))
        """
        if item_embeddings is None and item_similarities is None:
            return 0.0

        ilds = []
        for rec_list in recommendations:
            items = rec_list[:k]
            if len(items) < 2:
                continue

            similarities = []
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if item_embeddings is not None:
                        sim = np.dot(item_embeddings[items[i]], item_embeddings[items[j]])
                        sim = sim / (np.linalg.norm(item_embeddings[items[i]]) *
                                     np.linalg.norm(item_embeddings[items[j]]))
                    else:
                        sim = item_similarities[items[i]][items[j]]
                    similarities.append(1 - sim)  # Dissimilarity

            ilds.append(np.mean(similarities) if similarities else 0.0)

        return np.mean(ilds) if ilds else 0.0

    @staticmethod
    def coverage(recommendations: List[List[int]], num_items: int, k: int = 10) -> float:
        """Coverage: fraction of items that appear in any recommendation list."""
        recommended_items = set()
        for rec_list in recommendations:
            recommended_items.update(rec_list[:k])
        return len(recommended_items) / max(num_items, 1)

    @staticmethod
    def entropy(recommendations: List[List[int]], k: int = 10) -> float:
        """Entropy of recommendation distribution."""
        item_counts = defaultdict(int)
        total = 0

        for rec_list in recommendations:
            for item in rec_list[:k]:
                item_counts[item] += 1
                total += 1

        if total == 0:
            return 0.0

        probs = np.array(list(item_counts.values())) / total
        return -np.sum(probs * np.log2(probs + 1e-8)) / np.log2(max(len(item_counts), 2))

    # -------------------------------------------------------------------------
    # Popularity Bias Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def average_popularity(
            recommendations: List[List[int]],
            item_popularities: np.ndarray,
            k: int = 10,
    ) -> float:
        """Average popularity of recommended items."""
        popularities = []
        for rec_list in recommendations:
            for item in rec_list[:k]:
                if item < len(item_popularities):
                    popularities.append(item_popularities[item])
        return np.mean(popularities) if popularities else 0.0

    @staticmethod
    def popularity_lift(
            recommendations: List[List[int]],
            item_popularities: np.ndarray,
            k: int = 10,
    ) -> float:
        """
        Popularity lift: ratio of average recommendation popularity to overall average popularity.
        Lower values indicate less popularity bias.
        """
        avg_pop = RecommendationMetrics.average_popularity(recommendations, item_popularities, k)
        overall_avg = np.mean(item_popularities) if len(item_popularities) > 0 else 1.0

        if overall_avg == 0:
            return 0.0

        return avg_pop / overall_avg

    @staticmethod
    def gini_coefficient(recommendations: List[List[int]], num_items: int, k: int = 10) -> float:
        """
        Gini coefficient of recommendation frequency distribution.

        0 = perfect equality, 1 = perfect inequality.
        Lower values indicate less popularity bias.
        """
        item_counts = np.zeros(num_items)
        for rec_list in recommendations:
            for item in rec_list[:k]:
                if item < num_items:
                    item_counts[item] += 1

        # Sort by frequency
        sorted_counts = np.sort(item_counts)
        n = len(sorted_counts)

        if n == 0 or sorted_counts.sum() == 0:
            return 0.0

        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

        return gini

    # -------------------------------------------------------------------------
    # Comprehensive Evaluation
    # -------------------------------------------------------------------------

    def evaluate_all(
            self,
            predictions: List[List[int]],
            ground_truth: List[List[int]],
            item_embeddings: Optional[np.ndarray] = None,
            item_popularities: Optional[np.ndarray] = None,
            group_info: Optional[Dict] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run all evaluation metrics.

        Returns:
            Dict with keys: 'accuracy', 'fairness', 'diversity', 'popularity_bias'
            Each containing metric_name -> value mappings.
        """
        results = {
            'accuracy': {},
            'fairness': {},
            'diversity': {},
            'popularity_bias': {},
        }

        # Accuracy metrics
        for k in self.topk_values:
            recall = np.mean([self.recall_at_k(p, g, k) for p, g in zip(predictions, ground_truth)])
            ndcg = np.mean([self.ndcg_at_k(p, g, k) for p, g in zip(predictions, ground_truth)])
            mrr = np.mean([self.mrr_at_k(p, g, k) for p, g in zip(predictions, ground_truth)])
            hit = np.mean([self.hit_at_k(p, g, k) for p, g in zip(predictions, ground_truth)])

            results['accuracy'][f'Recall@{k}'] = recall
            results['accuracy'][f'NDCG@{k}'] = ndcg
            results['accuracy'][f'MRR@{k}'] = mrr
            results['accuracy'][f'Hit@{k}'] = hit

        # Diversity metrics
        if item_embeddings is not None:
            for k in self.topk_values:
                ild = self.intra_list_diversity(predictions, item_embeddings, k=k)
                results['diversity'][f'ILD@{k}'] = ild

        for k in self.topk_values:
            num_items = item_embeddings.shape[0] if item_embeddings is not None else 10000
            cov = self.coverage(predictions, num_items, k=k)
            ent = self.entropy(predictions, k=k)
            results['diversity'][f'Coverage@{k}'] = cov
            results['diversity'][f'Entropy@{k}'] = ent

        # Popularity bias metrics
        if item_popularities is not None:
            for k in self.topk_values:
                avg_pop = self.average_popularity(predictions, item_popularities, k=k)
                pop_lift = self.popularity_lift(predictions, item_popularities, k=k)
                gini = self.gini_coefficient(predictions, len(item_popularities), k=k)

                results['popularity_bias'][f'AvgPopularity@{k}'] = avg_pop
                results['popularity_bias'][f'PopularityLift@{k}'] = pop_lift
                results['popularity_bias'][f'GiniCoefficient@{k}'] = gini

        # Fairness metrics (if group info available)
        if group_info is not None:
            for k in self.topk_values:
                dp = self.demographic_parity(
                    group_info.get('recommendations_per_group', {}),
                    group_info.get('item_groups', {}),
                    k=k,
                )
                results['fairness'][f'DemographicParity@{k}'] = dp

        return results