"""
ESCI dataset loader for conventional product search evaluation.

The ESCI (Exact, Substitute, Complement, Irrelevant) dataset contains real
Amazon search queries with relevance labels. We use only 'Exact' matches as
per BLAIR's methodology.
"""
import logging
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ESCIDataset:
    """
    Loader for ESCI-S (small) dataset, linking ASINs to AMAZON REVIEWS 2023 metadata.
    Filters non-English queries and timestamps to match BLAIR's test period.
    """

    def __init__(
            self,
            dataset_name: str = "tasksource/esci",
            split: str = "test",
            label_type: str = "exact",
            seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.label_type = label_type  # 'exact' only for binary relevance
        self.seed = seed
        np.random.seed(seed)

        self.queries: List[str] = []
        self.item_ids: List[str] = []       # ground-truth ASINs
        self.item_texts: Dict[str, str] = {} # ASIN -> concatenated metadata

    def load(
            self,
            item_metadata_provider: Optional[callable] = None,
    ) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Load ESCI data and enrich with item metadata from an external source.
        If item_metadata_provider is provided, it should be a function that
        takes a list of ASINs and returns a dict of ASIN->metadata text.
        """
        logger.info(f"Loading ESCI ({self.dataset_name})")
        dataset = load_dataset(self.dataset_name, split=self.split)

        # Filter for 'Exact' labels only
        exact_pairs = []
        for example in dataset:
            if example.get("esci_label") == "Exact":
                query = example.get("query", "")
                item_id = example.get("product_id") or example.get("asin")
                if query and item_id:
                    exact_pairs.append((query, item_id))
        logger.info(f"Found {len(exact_pairs)} Exact (query, item) pairs")

        # Remove duplicates (keep first occurrence)
        seen = set()
        unique_pairs = []
        for q, i in exact_pairs:
            if (q, i) not in seen:
                seen.add((q, i))
                unique_pairs.append((q, i))
        exact_pairs = unique_pairs
        logger.info(f"After deduplication: {len(exact_pairs)} pairs")

        # Enrich with metadata
        if item_metadata_provider:
            all_asins = set(i for _, i in exact_pairs)
            metadata_map = item_metadata_provider(list(all_asins))
        else:
            metadata_map = {}

        for query, item_id in exact_pairs:
            self.queries.append(query)
            self.item_ids.append(item_id)
            self.item_texts[item_id] = metadata_map.get(item_id, "")

        return self.queries, self.item_ids, self.item_texts

    def get_item_texts_batch(self, item_ids: List[str]) -> List[str]:
        """Return metadata texts for a list of items."""
        return [self.item_texts.get(asi, "") for asi in item_ids]

    def sample_candidate_pool(
            self,
            positive_item: str,
            all_item_pool: List[str],
            pool_size: int = 50,
    ) -> List[str]:
        """Sample a candidate pool of items for product search evaluation."""
        if positive_item not in all_item_pool:
            all_item_pool = all_item_pool + [positive_item]
        negatives = list(np.random.choice(
            [i for i in all_item_pool if i != positive_item],
            size=min(pool_size - 1, len(all_item_pool) - 1),
            replace=False,
        ))
        return [positive_item] + negatives