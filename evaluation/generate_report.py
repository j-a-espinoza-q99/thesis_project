#!/usr/bin/env python
"""
Generate final comparison report from benchmark results.
Creates tables, plots, and a PDF report.
"""
import os
import sys
import json
import argparse
from typing import Dict, List
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results(results_dir: str) -> Dict:
    """Load all benchmark results."""
    results = {}
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if os.path.isdir(model_path):
            json_path = os.path.join(model_path, 'results.json')
            if os.path.exists(json_path):
                with open(json_path) as f:
                    results[model_dir] = json.load(f)
    return results


def create_comparison_table(results: Dict, metric: str = 'NDCG@10') -> pd.DataFrame:
    """Create a comparison table for a specific metric."""
    models = list(results.keys())
    domains = set()

    for model_results in results.values():
        if isinstance(model_results, dict):
            domains.update(model_results.keys())

    data = {}
    for domain in sorted(domains):
        data[domain] = {}
        for model in models:
            if model in results and domain in results[model]:
                model_domain = results[model][domain]
                # Extract metric from nested structure
                value = extract_metric(model_domain, metric)
                data[domain][model] = value

    return pd.DataFrame(data)


def extract_metric(results_dict: Dict, metric: str) -> float:
    """Extract a specific metric from nested results."""
    if 'evaluation' in results_dict:
        results_dict = results_dict['evaluation']
    if 'accuracy' in results_dict:
        return results_dict['accuracy'].get(metric, 0.0)
    if 'product_search' in results_dict:
        return results_dict['product_search'].get('accuracy', {}).get(metric, 0.0)
    if 'sequential_recommendation' in results_dict:
        return results_dict['sequential_recommendation'].get('accuracy', {}).get(metric, 0.0)
    return results_dict.get(metric, 0.0)


def plot_comparison(results: Dict, output_path: str):
    """Create comparison plots."""
    metrics_to_plot = ['NDCG@10', 'NDCG@50', 'Recall@10', 'Recall@50']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        df = create_comparison_table(results, metric)

        if df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        df.T.plot(kind='bar', ax=ax)
        ax.set_title(metric)
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.legend(loc='best')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_loss_curves(results: Dict, output_path: str):
    """Plot training loss curves for custom model."""
    custom_results = results.get('custom', {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for domain, domain_results in custom_results.items():
        if 'training' in domain_results:
            train_losses = domain_results['training'].get('train_losses', [])
            axes[0].plot(train_losses, label=domain, marker='o', markersize=3)

    axes[0].set_title('Training Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot loss decomposition if available
    for domain, domain_results in custom_results.items():
        if 'evaluation' in domain_results:
            eval_results = domain_results['evaluation']
            diversity = eval_results.get('diversity', {})
            popularity = eval_results.get('popularity_bias', {})

            metrics_vals = []
            metric_names = []
            for k, v in {**diversity, **popularity}.items():
                if isinstance(v, (int, float)):
                    metrics_vals.append(v)
                    metric_names.append(k)

            if metrics_vals:
                axes[1].bar(metric_names, metrics_vals, alpha=0.7, label=domain)

    axes[1].set_title('Fairness & Diversity Metrics')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {output_path}")


def generate_html_report(
        results: Dict,
        output_path: str,
):
    """Generate an HTML report with all results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Thesis Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .best {{ font-weight: bold; color: #4CAF50; }}
        img {{ max-width: 100%; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Thesis Benchmark Report</h1>
    <p>Generated: {timestamp}</p>
    
    <h2>Model Comparison: NDCG@10</h2>
    {create_comparison_table(results, 'NDCG@10').to_html()}
    
    <h2>Model Comparison: Recall@10</h2>
    {create_comparison_table(results, 'Recall@10').to_html()}
    
    <h2>Comparison Plots</h2>
    <img src="comparison_plots.png" alt="Comparison Plots">
    
    <h2>Training Curves</h2>
    <img src="loss_curves.png" alt="Loss Curves">
    
    <h2>Raw Results</h2>
    <pre>{json.dumps(results, indent=2, default=str)}</pre>
</body>
</html>
    """

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"HTML report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/final_report')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load results
    results = load_results(args.results_dir)

    if not results:
        print("No results found!")
        return

    # Generate plots
    plot_comparison(results, os.path.join(args.output, 'comparison_plots.png'))
    plot_loss_curves(results, os.path.join(args.output, 'loss_curves.png'))

    # Generate HTML report
    generate_html_report(results, os.path.join(args.output, 'report.html'))

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for metric in ['NDCG@10', 'Recall@10', 'NDCG@50', 'Recall@50']:
        df = create_comparison_table(results, metric)
        if not df.empty:
            print(f"\n{metric}:")
            print(df.to_string())


if __name__ == '__main__':
    main()