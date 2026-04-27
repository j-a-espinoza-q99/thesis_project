"""
Main benchmarking script that runs all experiments and compares models.

Orchestrates:
1. BLAIR baseline evaluation (sequential recommendation + product search)
2. DeepSeek API evaluation
3. Claude/Voyage AI evaluation
4. Custom model evaluation with multi-objective loss
"""
import os
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.model_configs import (
    BLAIRConfig, DeepSeekConfig, ClaudeVoyageConfig,
    CustomModelConfig, LossConfig, PromptConfig, load_config,
)
from data.preprocessing import AmazonReviewsPreprocessor
from data.dataset import SequentialRecommendationDataset, ProductSearchDataset
from models.blair_model import BLAIRBaseline
from models.deepseek_model import DeepSeekRecommender
from models.claude_voyage_model import ClaudeVoyageRecommender
from models.custom_model import CustomRecommendationModel
from models.loss_functions import MultiObjectiveLoss
from prompts.prompt_templates import format_prompt
from evaluation.metrics import RecommendationMetrics
from training.trainer import CustomModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runs the full benchmark: BLAIR → DeepSeek → Claude/Voyage → Custom.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_calculator = RecommendationMetrics(
            topk_values=self.config['evaluation']['topk_values']
        )

        # Initialize results storage
        self.results = {
            'blair': {},
            'deepseek': {},
            'claude_voyage': {},
            'custom': {},
        }

        # Initialize preprocessor
        self.preprocessor = AmazonReviewsPreprocessor(
            domains=self.config['data']['domains'],
            min_interactions=self.config['data']['min_interactions'],
        )

    def run_all(self):
        """Run the complete benchmark pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Full Benchmark Pipeline")
        logger.info("=" * 60)

        # Step 1: Process data
        logger.info("\n[Step 1/5] Processing data...")
        data = self.preprocessor.process_all_domains()

        # Step 2: BLAIR baseline
        logger.info("\n[Step 2/5] Running BLAIR baseline...")
        self.run_blair_baseline(data)

        # Step 3: DeepSeek
        logger.info("\n[Step 3/5] Running DeepSeek evaluation...")
        self.run_deepseek(data)

        # Step 4: Claude/Voyage
        logger.info("\n[Step 4/5] Running Claude/Voyage AI evaluation...")
        self.run_claude_voyage(data)

        # Step 5: Custom model
        logger.info("\n[Step 5/5] Running custom model evaluation...")
        self.run_custom_model(data)

        # Save all results
        self.save_results()

        # Print comparison table
        self.print_comparison()

        return self.results

    def run_blair_baseline(self, data: Dict):
        """Run BLAIR baseline evaluation."""
        blair_config = BLAIRConfig(
            **self.config['blair']['base']
        )

        blair_model = BLAIRBaseline(
            model_name=blair_config.model_name,
            device=str(self.device),
        )

        results = {}

        for domain in self.config['data']['domains']:
            logger.info(f"  Evaluating BLAIR on {domain}...")
            domain_data = data[domain]

            # Sequential recommendation
            seq_results = self._evaluate_sequential_recommendation(
                blair_model, domain_data, domain
            )

            # Product search
            search_results = self._evaluate_product_search(
                blair_model, domain_data, domain
            )

            results[domain] = {
                'sequential_recommendation': seq_results,
                'product_search': search_results,
            }

        self.results['blair'] = results

    def run_deepseek(self, data: Dict):
        """Run DeepSeek API evaluation."""
        try:
            deepseek_model = DeepSeekRecommender()

            results = {}

            for domain in self.config['data']['domains']:
                logger.info(f"  Evaluating DeepSeek on {domain}...")
                domain_data = data[domain]

                # Get item representations
                item_texts = list(domain_data['item_metadata'].values())

                # Product search with DeepSeek
                search_results = self._evaluate_product_search_llm(
                    deepseek_model, domain_data, domain, model_type='deepseek'
                )

                results[domain] = {
                    'product_search': search_results,
                }

            self.results['deepseek'] = results

        except Exception as e:
            logger.error(f"DeepSeek evaluation failed: {e}")
            self.results['deepseek'] = {'error': str(e)}

    def run_claude_voyage(self, data: Dict):
        """Run Claude/Voyage AI evaluation."""
        try:
            claude_model = ClaudeVoyageRecommender()

            results = {}

            for domain in self.config['data']['domains']:
                logger.info(f"  Evaluating Claude/Voyage on {domain}...")
                domain_data = data[domain]

                # Product search with Claude/Voyage
                search_results = self._evaluate_product_search_llm(
                    claude_model, domain_data, domain, model_type='claude_voyage'
                )

                results[domain] = {
                    'product_search': search_results,
                }

            self.results['claude_voyage'] = results

        except Exception as e:
            logger.error(f"Claude/Voyage evaluation failed: {e}")
            self.results['claude_voyage'] = {'error': str(e)}

    def run_custom_model(self, data: Dict):
        """Run custom model with multi-objective loss and custom prompts."""
        custom_config = CustomModelConfig(**self.config['custom'])
        loss_config = LossConfig(**self.config['loss'])

        model = CustomRecommendationModel(
            blair_model_name=custom_config.backbone,
            embedding_dim=custom_config.embedding_dim,
            num_experts=custom_config.num_experts,
            lora_rank=custom_config.lora_rank,
            lora_alpha=custom_config.lora_alpha,
            use_pca=custom_config.use_pca,
            pca_dim=custom_config.pca_dim,
            loss_config={
                'bpr_weight': loss_config.bpr_weight,
                'adv_weight': loss_config.adv_weight,
                'div_weight': loss_config.div_weight,
                'pop_weight': loss_config.pop_weight,
                'aug_weight': loss_config.aug_weight,
                'aug_temperature': loss_config.aug_temperature,
            },
            device=str(self.device),
        )

        # Train the model
        trainer = CustomModelTrainer(
            model=model,
            config=self.config['training'],
            device=str(self.device),
        )

        results = {}

        for domain in self.config['data']['domains']:
            logger.info(f"  Training custom model on {domain}...")
            domain_data = data[domain]

            # Train
            train_results = trainer.train(domain_data, domain)

            # Evaluate
            eval_results = trainer.evaluate(domain_data, domain)

            results[domain] = {
                'training': train_results,
                'evaluation': eval_results,
            }

        self.results['custom'] = results

    def _evaluate_sequential_recommendation(
            self,
            model,
            domain_data: Dict,
            domain: str,
    ) -> Dict:
        """Evaluate sequential recommendation performance."""
        # Build dataset
        test_df = domain_data['test_sequences']
        item_metadata = domain_data['item_metadata']

        # Get item embeddings
        item_texts = list(item_metadata.values())
        item_ids = list(item_metadata.keys())

        item_embeddings = model.encode_items(item_texts)

        # Evaluate
        all_predictions = []
        all_ground_truth = []

        for _, row in tqdm(test_df.iterrows(), desc=f"Evaluating {domain}"):
            input_seq = row['item_sequence'][:-1]  # All but last
            target = row['item_sequence'][-1]

            if len(input_seq) == 0:
                continue

            # Use last item in sequence as "query"
            last_item_text = item_metadata.get(input_seq[-1], "")
            query_emb = model.encode_query(last_item_text)

            scores = query_emb @ item_embeddings.T
            top_k = scores.squeeze(0).argsort(descending=True)[:100].cpu().tolist()

            predictions = [item_ids[i] for i in top_k if i < len(item_ids)]
            all_predictions.append(predictions)
            all_ground_truth.append([target])

        # Compute metrics
        metrics_results = self.metrics_calculator.evaluate_all(
            predictions=[[self._item_to_idx(p, item_ids) for p in preds]
                         for preds in all_predictions],
            ground_truth=[[self._item_to_idx(g, item_ids) for g in truths]
                          for truths in all_ground_truth],
            item_embeddings=item_embeddings.cpu().numpy() if isinstance(item_embeddings, torch.Tensor) else None,
        )

        return metrics_results

    def _evaluate_product_search(
            self,
            model,
            domain_data: Dict,
            domain: str,
    ) -> Dict:
        """Evaluate product search performance."""
        item_metadata = domain_data['item_metadata']
        item_texts = list(item_metadata.values())
        item_ids = list(item_metadata.keys())

        # For product search, use test reviews as queries
        test_reviews = domain_data['test']['cleaned_review'].tolist()
        test_items = domain_data['test']['parent_asin'].tolist()

        item_embeddings = model.encode_items(item_texts)

        all_predictions = []
        all_ground_truth = []

        for query, target_item in tqdm(
                zip(test_reviews[:1000], test_items[:1000]),  # Sample for efficiency
                total=min(1000, len(test_reviews)),
                desc=f"Product search {domain}"
        ):
            query_emb = model.encode_query(query)
            scores = query_emb @ item_embeddings.T
            top_k = scores.squeeze(0).argsort(descending=True)[:100].cpu().tolist()

            predictions = [item_ids[i] for i in top_k if i < len(item_ids)]
            all_predictions.append(predictions)
            all_ground_truth.append([target_item])

        metrics_results = self.metrics_calculator.evaluate_all(
            predictions=[[self._item_to_idx(p, item_ids) for p in preds]
                         for preds in all_predictions],
            ground_truth=[[self._item_to_idx(g, item_ids) for g in truths]
                          for truths in all_ground_truth],
            item_embeddings=item_embeddings.cpu().numpy(),
        )

        return metrics_results

    def _evaluate_product_search_llm(
            self,
            model,
            domain_data: Dict,
            domain: str,
            model_type: str,
    ) -> Dict:
        """Evaluate LLM-based product search (DeepSeek or Claude)."""
        item_metadata = domain_data['item_metadata']
        item_texts = list(item_metadata.values())

        test_reviews = domain_data['test']['cleaned_review'].tolist()
        test_items = domain_data['test']['parent_asin'].tolist()

        all_predictions = []
        all_ground_truth = []

        # Sample for API efficiency
        sample_size = min(200, len(test_reviews))
        indices = np.random.choice(len(test_reviews), sample_size, replace=False)

        for idx in tqdm(indices, desc=f"LLM search {domain}"):
            query = test_reviews[idx]
            target = test_items[idx]

            # Sample candidates
            candidate_indices = np.random.choice(len(item_texts), 50, replace=False)
            candidates = [item_texts[i] for i in candidate_indices]

            try:
                if hasattr(model, 'rank_items'):
                    ranked = model.rank_items(query, candidates)
                else:
                    ranked = list(range(10))  # Fallback

                predictions = [candidate_indices[r] for r in ranked if r < len(candidate_indices)]
            except Exception as e:
                logger.warning(f"LLM ranking failed: {e}")
                predictions = candidate_indices[:10].tolist()

            all_predictions.append(predictions)
            all_ground_truth.append([target])

        metrics_results = self.metrics_calculator.evaluate_all(
            predictions=all_predictions,
            ground_truth=[[self._item_to_idx(g, list(item_metadata.keys()))
                           for g in truths] for truths in all_ground_truth],
        )

        return metrics_results

    @staticmethod
    def _item_to_idx(item_id: str, item_ids: List[str]) -> int:
        """Convert item ID to index."""
        try:
            return item_ids.index(item_id)
        except ValueError:
            return 0

    def save_results(self):
        """Save all benchmark results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/benchmark_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Save full results
        with open(f"{output_dir}/full_results.json", 'w') as f:
            # Convert any non-serializable objects
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {output_dir}/full_results.json")

    def _make_serializable(self, obj):
        """Convert results to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def print_comparison(self):
        """Print a comparison table of all models."""
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("=" * 80)

        for domain in self.config['data']['domains']:
            logger.info(f"\n--- {domain} ---")

            for model_name in ['blair', 'deepseek', 'claude_voyage', 'custom']:
                if model_name in self.results and domain in self.results[model_name]:
                    domain_results = self.results[model_name][domain]
                    logger.info(f"\n{model_name.upper()}:")

                    for task, metrics in domain_results.items():
                        if isinstance(metrics, dict) and 'accuracy' in metrics:
                            for metric_name, value in metrics['accuracy'].items():
                                logger.info(f"  {task}/{metric_name}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark")
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--skip-api', action='store_true', help='Skip API-based models')
    parser.add_argument('--domains', nargs='+', default=None, help='Domains to evaluate')
    args = parser.parse_args()

    runner = BenchmarkRunner(config_path=args.config)

    if args.domains:
        runner.config['data']['domains'] = args.domains

    if args.skip_api:
        logger.info("Skipping API-based models (DeepSeek, Claude/Voyage)")
        # Run only BLAIR and custom
        data = runner.preprocessor.process_all_domains()
        runner.run_blair_baseline(data)
        runner.run_custom_model(data)
        runner.save_results()
        runner.print_comparison()
    else:
        runner.run_all()


if __name__ == '__main__':
    main()