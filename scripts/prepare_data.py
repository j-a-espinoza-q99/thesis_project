#!/usr/bin/env python
"""
Data preparation script: processes raw Amazon Reviews 2023 data and
creates train/val/test splits and pretraining pairs.
"""
import os, sys, argparse, logging
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.preprocessing import AmazonReviewsPreprocessor, create_pretraining_pairs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', default=['All_Beauty', 'Video_Games', 'Office_Products'])
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    preprocessor = AmazonReviewsPreprocessor(domains=args.domains, num_workers=args.num_workers)
    data = preprocessor.process_all_domains()

    for domain, domain_data in data.items():
        logger.info(f"Saving processed data for {domain}")
        # Save metadata
        with open(os.path.join(args.output_dir, f"{domain}_metadata.json"), 'w') as f:
            json.dump(domain_data['item_metadata'], f)
        # Save train/val/test sequences as CSV
        domain_data['train_sequences'].to_csv(os.path.join(args.output_dir, f"{domain}_train_seq.csv"), index=False)
        domain_data['val_sequences'].to_csv(os.path.join(args.output_dir, f"{domain}_val_seq.csv"), index=False)
        domain_data['test_sequences'].to_csv(os.path.join(args.output_dir, f"{domain}_test_seq.csv"), index=False)
        # Create pretraining pairs
        pairs_df = create_pretraining_pairs(domain_data['train'], domain_data['item_metadata'],
                                            os.path.join(args.output_dir, f"{domain}_pretrain_pairs.tsv"))
        logger.info(f"Created {len(pairs_df)} pretraining pairs for {domain}")

    logger.info("Data preparation complete.")


if __name__ == '__main__':
    main()