"""
Data preprocessing pipeline for Amazon Reviews 2023 dataset.

Handles:
- Loading from HuggingFace datasets
- Timestamp-based train/val/test splitting
- Item metadata extraction and cleaning
- User review concatenation and filtering
"""
import os
import random
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AmazonReviewsPreprocessor:
    """
    Preprocessor for Amazon Reviews 2023 dataset.
    Mirrors BLAIR's preprocessing methodology.
    """

    def __init__(
            self,
            dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023",
            domains: List[str] = None,
            min_chars: int = 30,
            min_interactions: int = 5,
            num_workers: int = 8,
            seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.domains = domains or ["All_Beauty", "Video_Games", "Office_Products"]
        self.min_chars = min_chars
        self.min_interactions = min_interactions
        self.num_workers = num_workers
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

    def concat_item_metadata(self, item: Dict) -> Dict:
        """
        Concatenate item metadata fields: title + features + description.
        Mirrors BLAIR's 'concat_item_metadata' function.
        """
        meta_parts = []

        if item.get('title') and item['title'] is not None:
            meta_parts.append(item['title'])

        if item.get('features') and len(item.get('features', [])) > 0:
            meta_parts.append(' '.join(item['features']))

        if item.get('description') and len(item.get('description', [])) > 0:
            meta_parts.append(' '.join(item['description']))

        cleaned = ' '.join(meta_parts)
        # Remove problematic whitespace
        cleaned = cleaned.replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip()

        item['cleaned_metadata'] = cleaned
        return item

    def concat_review(self, review: Dict) -> Dict:
        """
        Concatenate review title and text.
        """
        parts = []

        if review.get('title') and review['title'] is not None:
            parts.append(review['title'])

        if review.get('text') and review['text'] is not None:
            parts.append(review['text'])

        cleaned = ' '.join(parts)
        cleaned = cleaned.replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip()

        review['cleaned_review'] = cleaned
        return review

    def filter_metadata(self, item: Dict) -> bool:
        """Filter items with insufficient metadata."""
        return len(item.get('cleaned_metadata', '')) > self.min_chars

    def filter_review(self, review: Dict) -> bool:
        """Filter reviews with insufficient content."""
        return len(review.get('cleaned_review', '')) > self.min_chars

    def load_and_process_domain(self, domain: str) -> Tuple[Dict, pd.DataFrame]:
        """
        Load and process a single domain from Amazon Reviews 2023.
        
        Returns:
            Tuple of (item_metadata_dict, reviews_dataframe)
        """
        logger.info(f"Processing domain: {domain}")

        # Load item metadata
        meta_dataset = load_dataset(
            self.dataset_name,
            f'raw_meta_{domain}',
            split='full',
            trust_remote_code=True,
        )

        # Process metadata
        meta_dataset = meta_dataset.map(self.concat_item_metadata, num_proc=self.num_workers)
        meta_dataset = meta_dataset.filter(self.filter_metadata, num_proc=self.num_workers)

        # Build item metadata dictionary: {parent_asin: cleaned_metadata}
        item_metadata = {}
        for item_id, meta in zip(meta_dataset['parent_asin'], meta_dataset['cleaned_metadata']):
            item_metadata[item_id] = meta

        # Load reviews
        review_dataset = load_dataset(
            self.dataset_name,
            f'raw_review_{domain}',
            split='full',
            trust_remote_code=True,
        )

        # Process reviews
        review_dataset = review_dataset.map(self.concat_review, num_proc=self.num_workers)
        review_dataset = review_dataset.filter(self.filter_review, num_proc=self.num_workers)

        # Convert to DataFrame
        reviews_df = pd.DataFrame({
            'user_id': review_dataset['user_id'],
            'parent_asin': review_dataset['parent_asin'],
            'timestamp': review_dataset['timestamp'],
            'rating': review_dataset['rating'],
            'cleaned_review': review_dataset['cleaned_review'],
        })

        # Filter to items with valid metadata
        reviews_df = reviews_df[reviews_df['parent_asin'].isin(item_metadata)]

        logger.info(f"  Items with metadata: {len(item_metadata)}")
        logger.info(f"  Reviews: {len(reviews_df)}")

        return item_metadata, reviews_df

    def timestamp_split(
            self,
            df: pd.DataFrame,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by timestamp (not by user's last interactions).
        This aligns with BLAIR's methodology for realistic evaluation.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        df = df.sort_values('timestamp')

        # Find timestamp boundaries
        timestamps = df['timestamp'].values
        n = len(timestamps)

        train_end_idx = int(n * train_ratio)
        val_end_idx = int(n * (train_ratio + val_ratio))

        train_ts = timestamps[train_end_idx - 1]
        val_ts = timestamps[val_end_idx - 1]

        train_df = df[df['timestamp'] <= train_ts]
        val_df = df[(df['timestamp'] > train_ts) & (df['timestamp'] <= val_ts)]
        test_df = df[df['timestamp'] > val_ts]

        logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        return train_df, val_df, test_df

    def build_sequences(
            self,
            df: pd.DataFrame,
            max_seq_length: int = 50,
    ) -> pd.DataFrame:
        """
        Build interaction sequences per user, ordered by timestamp.
        Used for sequential recommendation tasks.
        """
        df = df.sort_values(['user_id', 'timestamp'])

        sequences = []
        for user_id, group in df.groupby('user_id'):
            items = group['parent_asin'].tolist()
            timestamps = group['timestamp'].tolist()

            # Take most recent max_seq_length interactions
            if len(items) > max_seq_length:
                items = items[-max_seq_length:]
                timestamps = timestamps[-max_seq_length:]

            sequences.append({
                'user_id': user_id,
                'item_sequence': items,
                'timestamp_sequence': timestamps,
                'seq_length': len(items),
            })

        return pd.DataFrame(sequences)

    def process_all_domains(
            self,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
    ) -> Dict[str, Dict]:
        """
        Process all specified domains and return structured data.
        """
        results = {}

        for domain in self.domains:
            logger.info(f"\n{'='*60}\nProcessing {domain}\n{'='*60}")

            item_metadata, reviews_df = self.load_and_process_domain(domain)

            train_df, val_df, test_df = self.timestamp_split(
                reviews_df, train_ratio, val_ratio, test_ratio
            )

            # Build sequences
            train_seq = self.build_sequences(train_df)
            val_seq = self.build_sequences(val_df)
            test_seq = self.build_sequences(test_df)

            results[domain] = {
                'item_metadata': item_metadata,
                'train': train_df,
                'val': val_df,
                'test': test_df,
                'train_sequences': train_seq,
                'val_sequences': val_seq,
                'test_sequences': test_seq,
            }

        return results


def create_pretraining_pairs(
        reviews_df: pd.DataFrame,
        item_metadata: Dict[str, str],
        output_path: str = "data/processed/pretrain_pairs.tsv",
):
    """
    Create (review, metadata) pairs for supervised contrastive pretraining.
    This mirrors BLAIR's sample_pretraining_data.py output format.
    """
    pairs = []

    for _, row in tqdm(reviews_df.iterrows(), total=len(reviews_df), desc="Creating pairs"):
        asin = row['parent_asin']
        if asin in item_metadata:
            pairs.append({
                'review': row['cleaned_review'],
                'meta': item_metadata[asin],
            })

    df = pd.DataFrame(pairs)
    df.to_csv(output_path, sep='\t', index=False, lineterminator='\n')

    logger.info(f"Saved {len(df)} pretraining pairs to {output_path}")
    return df