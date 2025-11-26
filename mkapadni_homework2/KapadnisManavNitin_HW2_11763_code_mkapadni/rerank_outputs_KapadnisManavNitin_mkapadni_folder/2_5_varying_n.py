"""
Script for Section 2.5: Varying n
Analyzes reranking performance with different numbers of candidates (5, 10, 20, 50).
# Andrew id: mkapadni
"""

import json
import numpy as np
from scipy.stats import spearmanr
import random
from tqdm import tqdm

from rerank_outputs_KapadnisManavNitin_mkapadni import (
    mbr_bleu,
    mbr_bertscore,
    compute_pairwise_reward
)


def subsample_candidates(all_results, n, seed=42):
    """
    Args:
        all_results: Full dataset
        n: Number of candidates to keep
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    subsampled = {
        'metadata': all_results['metadata'].copy(),
        'results': []
    }
    
    subsampled['metadata']['subsampled_n'] = n
    
    for result in all_results['results']:
        candidates = result['candidates']
        selected_indices = random.sample(range(len(candidates)), n)
        selected_candidates = [candidates[i] for i in selected_indices]
        
        subsampled_result = {
            'question_id': result['question_id'],
            'prompt': result['prompt'],
            'question_data': result['question_data'],
            'candidates': selected_candidates,
            'original_indices': selected_indices 
        }
        
        subsampled['results'].append(subsampled_result)
    
    return subsampled


def recompute_mbr_scores(subsampled_data):
    """
    Args:
        subsampled_data: Data with n candidates per question
        
    Returns:
        Updated data with recomputed MBR scores
    """
    for result in tqdm(subsampled_data['results'], desc="Recomputing MBR scores"):
        candidates = result['candidates']
        outputs = [c['generated_text'] for c in candidates]
        prompt = result['prompt']['intermediate_prompt']
        
        bleu_scores = mbr_bleu(outputs, prompt)
        for i, score in enumerate(bleu_scores):
            candidates[i]['scores']['mbr_bleu'] = score
        
        bert_scores = mbr_bertscore(outputs, prompt)
        for i, score in enumerate(bert_scores):
            candidates[i]['scores']['mbr_bert'] = score
    
    return subsampled_data


def recompute_pairwise_scores(subsampled_data):
    """
    Recompute pairwise scores for the subsampled data.
    Pairwise scores depend on comparisons with all other candidates.
    """
    for result in tqdm(subsampled_data['results'], desc="Recomputing pairwise scores"):
        candidates = result['candidates']
        outputs = [c['generated_text'] for c in candidates]
        prompt = result['prompt']['intermediate_prompt']
        
        # Recompute pairwise scores
        pairwise_scores = compute_pairwise_reward(outputs, prompt)
        for i, score in enumerate(pairwise_scores):
            candidates[i]['scores']['r_pairwise'] = score
    
    return subsampled_data


def evaluate_subsampled(subsampled_data, recompute_methods=['mbr_bleu', 'mbr_bert', 'r_pairwise']):
    """
    Args:
        subsampled_data: Subsampled dataset
        recompute_methods: Which methods need recomputation
        
    Returns:
        Evaluation dictionary
    """
    results = subsampled_data['results']
    
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
        
        for result in results:
            candidates = result['candidates']
            
            oracle_scores = [c['scores']['infobench'] for c in candidates]
            best_oracle_score = max(oracle_scores)
            
            method_scores = [c['scores'][score_key] for c in candidates]
            
            # Skip if scores are not computed
            if any(s is None for s in method_scores):
                continue
            
            top_idx = np.argmax(method_scores)
            top1_score = oracle_scores[top_idx]
            top1_scores.append(top1_score)
            
            # Rank of best output
            sorted_indices = np.argsort(method_scores)[::-1]
            oracle_best_idx = np.argmax(oracle_scores)
            rank = np.where(sorted_indices == oracle_best_idx)[0][0] + 1
            best_ranks.append(rank)
            
            # Spearman correlation
            if len(set(method_scores)) > 1:
                corr, _ = spearmanr(method_scores, oracle_scores)
                correlations.append(corr)
        
        evaluation[method_name] = {
            'top1_score': np.mean(top1_scores) if top1_scores else 0.0,
            'avg_rank_of_best': np.mean(best_ranks) if best_ranks else 0.0,
            'spearman': np.mean(correlations) if correlations else 0.0,
            'n_evaluated': len(top1_scores)
        }
    
    return evaluation


def print_comparison_table(evaluations, n_values):
    print("\n" + "="*100)
    print("COMPARISON ACROSS DIFFERENT VALUES OF N")
    print("="*100)
    
    methods = ['oracle', 'qwen3_4b', 'qwen3_14b', 'scalar', 'pairwise', 'bleu', 'bertscore']
    method_names = {
        'oracle': 'Oracle',
        'qwen3_4b': 'Qwen3-4B',
        'qwen3_14b': 'Qwen3-14B',
        'scalar': 'Scalar',
        'pairwise': 'Pairwise',
        'bleu': 'MBR-BLEU',
        'bertscore': 'MBR-BERT'
    }
    
    for metric in ['top1_score', 'avg_rank_of_best', 'spearman']:
        print(f"\n{metric.replace('_', ' ').title()}:")
        print("-"*100)
        
        header = f"{'Method':<20}"
        for n in n_values:
            header += f" n={n:<10}"
        print(header)
        print("-"*100)
        
        for method in methods:
            row = f"{method_names[method]:<20}"
            for n in n_values:
                if method in evaluations[n]:
                    value = evaluations[n][method][metric]
                    if metric == 'top1_score' or metric == 'spearman':
                        row += f" {value:<18}"
                    else:
                        row += f" {value:<10.2f}"
                else:
                    row += f" {'N/A':<10}"
            print(row)
        
        print("-"*100)


def analyze_oracle_improvement(evaluations, n_values):
    print("\n" + "="*80)
    print("ORACLE SCORE IMPROVEMENT WITH n")
    print("="*80)
    
    print(f"\n{'n':<10} {'Oracle Top-1':<15} {'Improvement':<15}")
    print("-"*80)
    
    baseline = None
    for n in n_values:
        oracle_score = evaluations[n]['oracle']['top1_score']
        
        if baseline is None:
            baseline = oracle_score
            improvement = 0.0
        else:
            improvement = oracle_score - baseline
        
        print(f"{n:<10} {oracle_score:<15.4f} {improvement:+<15.4f}")
    
    print("-"*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='results/all_results_with_scores.json',
                       help='Path to data with computed scores')
    parser.add_argument('--n-values', type=int, nargs='+', default=[5, 10, 20, 50],
                       help='Values of n to test')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for subsampling')
    parser.add_argument('--skip-recompute', action='store_true',
                       help='Skip recomputing MBR and pairwise scores')
    
    args = parser.parse_args()
    
    print("Loading data...")
    with open(args.data) as f:
        all_results = json.load(f)
    
    evaluations = {}
    
    for n in args.n_values:
        print(f"\n{'='*80}")
        print(f"EVALUATING WITH n={n}")
        print(f"{'='*80}")
        
        if n == 50:
            subsampled = all_results
        else:
            print(f"Subsampling to {n} candidates per question...")
            subsampled = subsample_candidates(all_results, n, seed=args.seed)
            
            if not args.skip_recompute:
                # Recompute MBR scores
                print("Recomputing MBR and pairwise scores...")
                subsampled = recompute_mbr_scores(subsampled)
                subsampled = recompute_pairwise_scores(subsampled)
                
                # Save subsampled data
                output_path = f"results/subsampled_n{n}.json"
                with open(output_path, 'w') as f:
                    json.dump(subsampled, f, indent=2)
                print(f"Saved to {output_path}")
        
        print(f"Evaluating with n={n}...")
        evaluation = evaluate_subsampled(subsampled)
        evaluations[n] = evaluation

    print_comparison_table(evaluations, args.n_values)
    
    analyze_oracle_improvement(evaluations, args.n_values)
    
    print("\n" + "="*80)
