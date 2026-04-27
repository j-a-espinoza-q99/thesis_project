#!/usr/bin/env python
"""
Custom model training with multi-objective loss.

Supports ablation studies by disabling individual loss components.
"""
import os
import sys
import argparse
import logging
import yaml

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.custom_model import CustomRecommendationModel
from models.loss_functions import MultiObjectiveLoss
from training.trainer import CustomModelTrainer
from data.preprocessing import AmazonReviewsPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--domains', nargs='+', default=['All_Beauty'])
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['no_adv', 'no_div', 'no_pop', 'no_aug', None])
    parser.add_argument('--output_dir', type=str, default='results/custom')
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply ablation
    if args.ablation:
        logger.info(f"Running ablation: {args.ablation}")
        loss_config = config['loss'].copy()
        if args.ablation == 'no_adv':
            loss_config['adv_weight'] = 0.0
        elif args.ablation == 'no_div':
            loss_config['div_weight'] = 0.0
        elif args.ablation == 'no_pop':
            loss_config['pop_weight'] = 0.0
        elif args.ablation == 'no_aug':
            loss_config['aug_weight'] = 0.0
        config['loss'] = loss_config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocessor = AmazonReviewsPreprocessor(domains=args.domains)

    all_results = {}

    for domain in args.domains:
        logger.info(f"\n{'='*50}\nTraining custom model on {domain}\n{'='*50}")

        # Load and process data
        data = preprocessor.process_all_domains()
        domain_data = data[domain]

        # Create model
        model = CustomRecommendationModel(
            blair_model_name=config['custom']['backbone'],
            embedding_dim=config['custom']['embedding_dim'],
            num_experts=config['custom']['num_experts'],
            lora_rank=config['custom']['lora_rank'],
            lora_alpha=config['custom']['lora_alpha'],
            use_pca=config['custom']['use_pca'],
            pca_dim=config['custom']['pca_dim'],
            loss_config=config['loss'],
            device=device,
        )

        # Train
        trainer = CustomModelTrainer(
            model=model,
            config=config['training'],
            device=device,
            use_wandb=args.use_wandb,
        )

        train_results = trainer.train(domain_data, domain)
        eval_results = trainer.evaluate(domain_data, domain)

        all_results[domain] = {
            'training': train_results,
            'evaluation': eval_results,
        }

        # Log final metrics
        logger.info(f"\nFinal results for {domain}:")
        for metric, value in eval_results.items():
            if isinstance(value, (int, float, np.floating)):
                logger.info(f"  {metric}: {value:.4f}")

        # Save model
        suffix = f"_ablation_{args.ablation}" if args.ablation else ""
        os.makedirs(f"{args.output_dir}{suffix}", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"{args.output_dir}{suffix}/{domain}_model.pt"
        )

    # Save results
    import json
    with open(f"{args.output_dir}{suffix}/results.json", 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj

        json.dump(convert(all_results), f, indent=2)

    logger.info(f"Results saved to {args.output_dir}{suffix}/")


if __name__ == '__main__':
    main()