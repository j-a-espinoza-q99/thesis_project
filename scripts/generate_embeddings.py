#!/usr/bin/env python
"""
Precompute and cache item embeddings for a given model to speed up evaluation.
"""
import os, sys, argparse, logging
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.blair_model import BLAIRBaseline
from data.preprocessing import AmazonReviewsPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='hyp1231/blair-roberta-base')
    parser.add_argument('--domains', nargs='+', default=['All_Beauty'])
    parser.add_argument('--output_dir', type=str, default='data/cache/embeddings')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BLAIRBaseline(model_name=args.model_name, device=device)
    preprocessor = AmazonReviewsPreprocessor(domains=args.domains)

    os.makedirs(args.output_dir, exist_ok=True)
    for domain in args.domains:
        logger.info(f"Processing domain: {domain}")
        item_metadata, _ = preprocessor.load_and_process_domain(domain)
        item_texts = list(item_metadata.values())
        item_ids = list(item_metadata.keys())

        embeddings = model.encode_items(item_texts, batch_size=args.batch_size)
        save_path = os.path.join(args.output_dir, f"{domain}_{args.model_name.replace('/', '_')}.npy")
        np.save(save_path, embeddings.cpu().numpy())
        # Also save item IDs mapping
        with open(save_path.replace('.npy', '_ids.txt'), 'w') as f:
            f.write('\n'.join(item_ids))
        logger.info(f"Saved {len(item_ids)} embeddings to {save_path}")


if __name__ == '__main__':
    main()