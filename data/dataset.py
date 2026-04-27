"""
PyTorch Dataset classes for all recommendation tasks.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


class SequentialRecommendationDataset(Dataset):
    """
    Dataset for sequential recommendation.

    Each sample: (item_sequence, target_item)
    For text-based methods, items are represented by their metadata text.
    """

    def __init__(
            self,
            sequences_df: pd.DataFrame,
            item_metadata: Dict[str, str],
            max_seq_length: int = 50,
            mode: str = 'train',
    ):
        self.sequences_df = sequences_df
        self.item_metadata = item_metadata
        self.max_seq_length = max_seq_length
        self.mode = mode

        # Build item ID mapping
        self._build_item_mapping()

        # Prepare samples
        self._prepare_samples()

    def _build_item_mapping(self):
        """Build item ID to index and index to metadata mappings."""
        all_items = set()
        for seq in self.sequences_df['item_sequence']:
            all_items.update(seq)

        self.item_to_idx = {item: idx for idx, item in enumerate(sorted(all_items))}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        self.num_items = len(self.item_to_idx)

        # Item text representations
        self.item_texts = []
        for idx in range(self.num_items):
            item_id = self.idx_to_item[idx]
            text = self.item_metadata.get(item_id, "")
            self.item_texts.append(text)

    def _prepare_samples(self):
        """Prepare training/validation/test samples."""
        self.samples = []

        for _, row in self.sequences_df.iterrows():
            items = row['item_sequence']
            if len(items) < 2:
                continue

            if self.mode == 'train':
                # Use all but last item as input, last as target
                for i in range(1, len(items)):
                    input_seq = items[max(0, i - self.max_seq_length):i]
                    target = items[i]
                    self.samples.append((input_seq, target))
            else:
                # For evaluation: use all but last as input
                input_seq = items[:-1][-self.max_seq_length:]
                target = items[-1]
                self.samples.append((input_seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]

        # Convert items to indices
        input_indices = [self.item_to_idx.get(item, 0) for item in input_seq]
        target_idx = self.item_to_idx.get(target, 0)

        # Get item texts
        input_texts = [self.item_texts[i] for i in input_indices]
        target_text = self.item_texts[target_idx]

        return {
            'input_indices': torch.tensor(input_indices, dtype=torch.long),
            'target_idx': torch.tensor(target_idx, dtype=torch.long),
            'input_texts': input_texts,
            'target_text': target_text,
            'seq_length': len(input_indices),
        }


class ProductSearchDataset(Dataset):
    """
    Dataset for product search (ESCI and Amazon-C4).

    Each sample: (query, positive_item, negative_items)
    """

    def __init__(
            self,
            queries: List[str],
            positive_items: List[str],
            item_metadata: Dict[str, str],
            candidate_pool: List[str] = None,
            num_negatives: int = 50,
    ):
        self.queries = queries
        self.positive_items = positive_items
        self.item_metadata = item_metadata
        self.candidate_pool = candidate_pool or list(item_metadata.keys())
        self.num_negatives = num_negatives

        assert len(queries) == len(positive_items)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        positive_item = self.positive_items[idx]
        positive_text = self.item_metadata.get(positive_item, "")

        # Sample negative items
        negative_items = np.random.choice(
            [item for item in self.candidate_pool if item != positive_item],
            size=self.num_negatives,
            replace=False,
        )
        negative_texts = [
            self.item_metadata.get(item, "") for item in negative_items
        ]

        all_items = [positive_item] + list(negative_items)
        all_texts = [positive_text] + negative_texts

        return {
            'query': query,
            'items': all_items,
            'item_texts': all_texts,
            'label': 0,  # positive item is at index 0
        }


class ContrastivePretrainingDataset(Dataset):
    """
    Dataset for BLAIR-style contrastive pretraining.
    Pairs (context, item_metadata) for supervised contrastive learning.
    """

    def __init__(self, pairs_file: str, max_seq_length: int = 64):
        self.df = pd.read_csv(pairs_file, sep='\t')
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'context': str(row['review']),
            'metadata': str(row['meta']),
        }