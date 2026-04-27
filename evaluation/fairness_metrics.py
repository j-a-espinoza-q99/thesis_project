"""
Extended fairness metrics for evaluation.
"""
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class FairnessMetrics:
    """Suite of group fairness metrics for recommendation."""

    @staticmethod
    def demographic_parity_difference(
            predictions: List[List[str]],
            user_groups: Dict[str, List[int]],  # group_name -> list of user indices
            item_attribute: Optional[Dict[str, int]] = None,
    ) -> float:
        """
        Difference in proportion of 'advantaged' items recommended.
        Lower → fairer.
        """
        group_rates = {}
        for group, indices in user_groups.items():
            recommended_items = []
            for idx in indices:
                if idx < len(predictions):
                    recommended_items.extend(predictions[idx][:10])
            if not recommended_items:
                continue
            # Count items with a certain attribute (e.g., 'popular' if item_attribute provided)
            if item_attribute:
                advantaged = sum(1 for i in recommended_items if item_attribute.get(i, 0) == 1)
                rate = advantaged / len(recommended_items)
            else:
                # Use average popularity rank as proxy - just dummy here
                rate = 0.5
            group_rates[group] = rate

        values = list(group_rates.values())
        if len(values) < 2:
            return 0.0
        return max(values) - min(values)

    @staticmethod
    def equalized_odds_difference(
            predictions: List[List[str]],
            ground_truth: List[List[str]],
            user_groups: Dict[str, List[int]],
    ) -> float:
        """TPR difference across groups."""
        group_tpr = {}
        for group, indices in user_groups.items():
            tp = 0
            total_pos = 0
            for idx in indices:
                if idx >= len(predictions):
                    continue
                pred_set = set(predictions[idx][:10])
                true_set = set(ground_truth[idx])
                tp += len(pred_set & true_set)
                total_pos += len(true_set)
            tpr = tp / max(total_pos, 1)
            group_tpr[group] = tpr

        if len(group_tpr) < 2:
            return 0.0
        return max(group_tpr.values()) - min(group_tpr.values())

    def compute_all(self, predictions, ground_truth, user_groups, item_popularities=None) -> Dict:
        """Return a dict of fairness metrics."""
        results = {}
        results['demographic_parity_diff'] = self.demographic_parity_difference(predictions, user_groups)
        results['equalized_odds_diff'] = self.equalized_odds_difference(predictions, ground_truth, user_groups)
        return results