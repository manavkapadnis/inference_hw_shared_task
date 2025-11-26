"""
Main script to run all reranking methods on InfoBench outputs.
Section 2.4: Comparing methods for reranking
# Andrew id: mkapadni
"""

import json
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from rerank_outputs_KapadnisManavNitin_mkapadni import (
    compute_model_prob,
    compute_scalar_reward,
    compute_pairwise_reward,
    mbr_bleu,
    mbr_bertscore
)


def load_data(data_path: str = "results/all_results_processed.json"):
    with open(data_path) as f:
        return json.load(f)


def save_data(data, output_path: str = "results/all_results_with_scores.json"):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


def compute_all_scores(all_results, methods_to_run=None):
    """
    Args:
        all_results: The loaded data structure
        methods_to_run: List of method names to run (or None for all)
    """
    if methods_to_run is None:
        methods_to_run = ['qwen3_4b', 'qwen3_14b', 'scalar', 'pairwise', 'bleu', 'bertscore']
    
    results = all_results['results']
    
    for i, result in enumerate(tqdm(results, desc="Processing questions")):
        question_id = result['question_id']
        prompt = result['prompt']['intermediate_prompt']
        candidates = result['candidates']
        
        # Extract outputs
        outputs = [c['generated_text'] for c in candidates]
        
        print(f"\nQuestion {question_id}:")
        
        if 'qwen3_4b' in methods_to_run:
            print("  Computing Qwen3-4B log-probs...")
            try:
                scores = compute_model_prob(outputs, prompt, model="Qwen/Qwen3-4B")
                for j, score in enumerate(scores):
                    candidates[j]['scores']['qwen3_4b'] = score
            except Exception as e:
                print(f"    Error: {e}")
        
        if 'qwen3_14b' in methods_to_run:
            print("  Computing Qwen3-14B log-probs...")
            try:
                scores = compute_model_prob(outputs, prompt, model="Qwen/Qwen3-14B")
                for j, score in enumerate(scores):
                    candidates[j]['scores']['qwen3_14b'] = score
            except Exception as e:
                print(f"    Error: {e}")
        
        if 'scalar' in methods_to_run:
            print("  Computing scalar rewards...")
            try:
                scores = compute_scalar_reward(outputs, prompt)
                for j, score in enumerate(scores):
                    candidates[j]['scores']['r_scalar'] = score
            except Exception as e:
                print(f"    Error: {e}")
        
        if 'pairwise' in methods_to_run:
            print("  Computing pairwise rewards...")
            try:
                scores = compute_pairwise_reward(outputs, prompt)
                for j, score in enumerate(scores):
                    candidates[j]['scores']['r_pairwise'] = score
            except Exception as e:
                print(f"    Error: {e}")
        
        if 'bleu' in methods_to_run:
            print("  Computing MBR-BLEU...")
            try:
                scores = mbr_bleu(outputs, prompt)
                for j, score in enumerate(scores):
                    candidates[j]['scores']['mbr_bleu'] = score
            except Exception as e:
                print(f"    Error: {e}")
        
        if 'bertscore' in methods_to_run:
            print("  Computing MBR-BERTScore...")
            try:
                scores = mbr_bertscore(outputs, prompt)
                for j, score in enumerate(scores):
                    candidates[j]['scores']['mbr_bert'] = score
            except Exception as e:
                print(f"    Error: {e}")
    
    return all_results


def evaluate_reranking(all_results):
    """
    Computes:
    1. Top-1 score (accuracy when selecting top-scoring output)
    2. Average rank of best output
    3. Spearman rank correlation with oracle scores
    """
    results = all_results['results']
    
    methods = {
        'oracle': 'infobench',
        'qwen3_4b': 'qwen3_4b',
        'qwen3_14b': 'qwen3_14b',
        'scalar': 'r_scalar',
        'pairwise': 'r_pairwise',
        'bleu': 'mbr_bleu',
        'bertscore': 'mbr_bert'
    }
    
    evaluation = {}
    
    for method_name, score_key in methods.items():
        top1_scores = []
        best_ranks = []
        correlations = []
        top1_lengths = []
        
        for result in results:
            candidates = result['candidates']
        
            oracle_scores = [c['scores']['infobench'] for c in candidates]
            best_oracle_score = max(oracle_scores)
            
            method_scores = [c['scores'][score_key] for c in candidates]
            
            if any(s is None for s in method_scores):
                continue
            
            # Top-1: score of the highest-ranked output by this method
            top_idx = np.argmax(method_scores)
            top1_score = oracle_scores[top_idx]
            top1_scores.append(top1_score)
            
            # Length of top-1 output
            top1_length = candidates[top_idx]['generation_len']
            top1_lengths.append(top1_length)
            
            # Rank of best output according to this method
            # Sort by method scores (descending) and find rank of best oracle output
            sorted_indices = np.argsort(method_scores)[::-1]
            oracle_best_idx = np.argmax(oracle_scores)
            rank = np.where(sorted_indices == oracle_best_idx)[0][0] + 1  # 1-indexed
            best_ranks.append(rank)
            
            # Spearman correlation
            if len(set(method_scores)) > 1:  # Need variation for correlation
                corr, _ = spearmanr(method_scores, oracle_scores)
                correlations.append(corr)
        
        evaluation[method_name] = {
            'top1_score': np.mean(top1_scores),
            'avg_rank_of_best': np.mean(best_ranks),
            'spearman': np.mean(correlations) if correlations else 0.0,
            'avg_top1_length': np.mean(top1_lengths) if top1_lengths else 0.0,
            'n_evaluated': len(top1_scores)
        }
    
    return evaluation


def print_results_table(evaluation):
    """Print results in table format."""
    print("\n" + "="*80)
    print("RERANKING EVALUATION RESULTS")
    print("="*80)
    print(f"\n{'Method':<20} {'Top-1 Score':<15} {'Avg. Rank':<15} {'Spearman':<15}")
    print("-"*80)
    
    method_order = ['oracle', 'qwen3_4b', 'qwen3_14b', 'scalar', 'pairwise', 'bleu', 'bertscore']
    method_names = {
        'oracle': 'Oracle',
        'qwen3_4b': 'Qwen3-4B logprobs',
        'qwen3_14b': 'Qwen3-14B logprobs',
        'scalar': 'Scalar reward',
        'pairwise': 'Pairwise reward',
        'bleu': 'MBR-BLEU',
        'bertscore': 'MBR-BERTScore'
    }
    
    for method in method_order:
        if method in evaluation:
            stats = evaluation[method]
            print(f"{method_names[method]:<20} "
                  f"{stats['top1_score']:<15.4f} "
                  f"{stats['avg_rank_of_best']:<15.2f} "
                  f"{stats['spearman']:<15.4f}")
    
    print("-"*80)
    
    # Length analysis
    print("\n" + "="*80)
    print("LENGTH ANALYSIS")
    print("="*80)
    print(f"\n{'Method':<20} {'Avg. Top-1 Length':<20}")
    print("-"*80)
    
    for method in method_order:
        if method in evaluation:
            stats = evaluation[method]
            print(f"{method_names[method]:<20} {stats['avg_top1_length']:<20.2f}")
    
    print("-"*80)


def create_scatter_plots(all_results, output_dir="plots"):
    """Create scatter plots comparing method scores to oracle scores."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = all_results['results']
    
    methods = {
        'Qwen3-4B log-probs': 'qwen3_4b',
        'Qwen3-14B log-probs': 'qwen3_14b',
        'Scalar reward': 'r_scalar',
        'Pairwise reward': 'r_pairwise',
        'MBR-BLEU': 'mbr_bleu',
        'MBR-BERTScore': 'mbr_bert'
    }
    
    for method_name, score_key in methods.items():
        oracle_scores = []
        method_scores = []
        
        for result in results:
            for candidate in result['candidates']:
                oracle_score = candidate['scores']['infobench']
                method_score = candidate['scores'][score_key]
                
                if method_score is not None:
                    oracle_scores.append(oracle_score)
                    method_scores.append(method_score)
        
        if len(oracle_scores) == 0:
            continue
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(method_scores, oracle_scores, alpha=0.5, s=10)
        plt.xlabel(f'{method_name} Score')
        plt.ylabel('Oracle (InfoBench) Score')
        plt.title(f'{method_name} vs Oracle Scores')
        plt.grid(True, alpha=0.3)
        
        # Add correlation
        corr, _ = spearmanr(method_scores, oracle_scores)
        plt.text(0.05, 0.95, f'Spearman: {corr:.3f}',
                transform=plt.gca().transAxes,
                verticalalignment='top')
        
        filename = f"{output_dir}/{score_key}_scatter.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='results/all_results_processed.json',
                       help='Path to input data')
    parser.add_argument('--output', type=str, default='results/all_results_with_scores.json',
                       help='Path to output data')
    parser.add_argument('--methods', type=str, nargs='+',
                       choices=['qwen3_4b', 'qwen3_14b', 'scalar', 'pairwise', 'bleu', 'bertscore', 'all'],
                       default=['all'],
                       help='Which methods to run')
    parser.add_argument('--skip-compute', action='store_true',
                       help='Skip computing scores (use if already computed)')
    parser.add_argument('--plots', action='store_true',
                       help='Generate scatter plots')
    
    args = parser.parse_args()

    print("Loading data...")
    all_results = load_data(args.data)
    
    if not args.skip_compute:
        methods = None if 'all' in args.methods else args.methods
        print("\nComputing scores for reranking methods...")
        all_results = compute_all_scores(all_results, methods_to_run=methods)

        save_data(all_results, args.output)
    
    print("\nEvaluating reranking methods...")
    evaluation = evaluate_reranking(all_results)
    
    print_results_table(evaluation)
    
    if args.plots:
        print("\nCreating scatter plots...")
        create_scatter_plots(all_results)
    
