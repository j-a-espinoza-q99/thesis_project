"""
Amazon-C4 dataset loader: Complex Contexts Created by ChatGPT.

This dataset mirrors the BLAIR paper's semi-synthetic evaluation set for the
complex product search task. Long, narrative-style queries are paired with
ground-truth items and a multi-domain candidate pool.
"""
import os
import json
import logging
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AmazonC4Dataset:
    """
    Loader for the Amazon-C4 dataset (McAuley-Lab/Amazon-C4 on HuggingFace).
    Provides query-item pairs and candidate sampling for evaluation.
    """

    def __init__(
            self,
            dataset_name: str = "McAuley-Lab/Amazon-C4",
            split: str = "test",
            min_query_words: int = 10,
            seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.min_query_words = min_query_words
        self.seed = seed
        np.random.seed(seed)

        self.queries: List[str] = []
        self.item_ids: List[str] = []  # ground-truth ASINs
        self.item_texts: Dict[str, str] = {}  # ASIN -> metadata text
        self.domain_map: Dict[str, str] = {}  # ASIN -> category

    def load(self) -> Tuple[List[str], List[str], Dict[str, str]]:
        """Load Amazon-C4 data and return (queries, item_ids, item_metadata)."""
        logger.info(f"Loading {self.dataset_name} ({self.split})")
        try:
            dataset = load_dataset(self.dataset_name, split=self.split)
        except Exception as e:
            logger.error(f"Could not load Amazon-C4: {e}")
            raise

        # Extract queries and ground-truth items
        for example in dataset:
            query = example.get("query") or example.get("context")
            item_id = example.get("item_id") or example.get("parent_asin")
            metadata = example.get("item_metadata") or example.get("description", "")

            if query and len(query.split()) >= self.min_query_words and item_id:
                self.queries.append(query)
                self.item_ids.append(item_id)
                self.item_texts[item_id] = metadata
                if "category" in example:
                    self.domain_map[item_id] = example["category"]

        logger.info(f"Loaded {len(self.queries)} valid query-item pairs.")
        return self.queries, self.item_ids, self.item_texts

    def sample_candidate_pool(
            self,
            positive_item: str,
            pool_size: int = 50,
            in_domain_only: bool = True,
    ) -> List[str]:
        """
        Sample a candidate pool of items for a given positive item.
        Includes the positive item and (pool_size-1) negative items.
        """
        all_items = list(self.item_texts.keys())
        if in_domain_only and self.domain_map:
            domain = self.domain_map.get(positive_item, "")
            if domain:
                domain_items = [i for i, d in self.domain_map.items() if d == domain]
                candidates = [positive_item]
                if len(domain_items) > pool_size:
                    negatives = list(np.random.choice(
                        [i for i in domain_items if i != positive_item],
                        size=pool_size - 1,
                        replace=False,
                    ))
                else:
                    # Fall back to all items
                    negatives = list(np.random.choice(
                        [i for i in all_items if i != positive_item],
                        size=pool_size - 1,
                        replace=False,
                    ))
                candidates.extend(negatives)
                return candidates

        # Without domain constraint, sample from all items
        negatives = list(np.random.choice(
            [i for i in all_items if i != positive_item],
            size=pool_size - 1,
            replace=False,
        ))
        return [positive_item] + negatives

    def get_item_texts_batch(self, item_ids: List[str]) -> List[str]:
        """Return metadata texts for a list of item IDs, handling missing keys."""
        return [self.item_texts.get(item_id, "") for item_id in item_ids]

    def build_eval_set(
            self,
            candidate_pool_size: int = 50,
    ) -> List[Dict]:
        """
        Build a list of evaluation examples: each has query, positive item, and
        a candidate pool containing the positive item.
        """
        self.load()
        eval_examples = []
        for query, pos_item in zip(self.queries, self.item_ids):
            candidate_pool = self.sample_candidate_pool(pos_item, candidate_pool_size)
            eval_examples.append({
                "query": query,
                "positive_item": pos_item,
                "candidate_pool": candidate_pool,
                "candidate_texts": self.get_item_texts_batch(candidate_pool),
            })
        return eval_examples