"""
Warm-up analysis for Section 2.1
Analyzes unique generations and InfoBench score differences.
# Andrew id: mkapadni
"""

import json
import numpy as np
from collections import Counter


def analyze_warmup(data_path: str = "results/all_results_processed.json"):
    """
    Calculates:
    1. Mean, median, and std of number of unique generations per prompt
    2. Average difference in InfoBench score between best and worst completion per prompt
    """
    
    # Load data
    with open(data_path) as f:
        all_results = json.load(f)
    
    print("="*60)
    print("WARM-UP ANALYSIS")
    print("="*60)
    print(f"\nMetadata:")
    print(f"  Model: {all_results['metadata']['model']}")
    print(f"  Sampling params: {all_results['metadata']['sampling_params']}")
    print(f"  Total questions: {all_results['metadata']['total_questions']}")
    print(f"  Total candidates: {all_results['metadata']['total_candidates']}")
    
    unique_counts = []
    score_differences = []
    diverse_examples = []
    similar_examples = []
    
    for result in all_results['results']:
        question_id = result['question_id']
        candidates = result['candidates']
        generated_texts = [c['generated_text'] for c in candidates]

        unique_texts = set(generated_texts)
        unique_count = len(unique_texts)
        unique_counts.append(unique_count)

        scores = [c['scores']['infobench'] for c in candidates]
        
        max_score = max(scores)
        min_score = min(scores)
        score_diff = max_score - min_score
        score_differences.append(score_diff)
        
        if unique_count <= 10:  # Low diversity
            similar_examples.append({
                'question_id': question_id,
                'unique_count': unique_count,
                'total_count': len(generated_texts),
                'prompt': result['prompt']['intermediate_prompt'][:100] + "...",
                'samples': list(unique_texts)[:3]
            })
        
        if unique_count >= 40:  # High diversity
            diverse_examples.append({
                'question_id': question_id,
                'unique_count': unique_count,
                'score_range': (min_score, max_score),
                'prompt': result['prompt']['intermediate_prompt'][:100] + "..."
            })
    
    mean_unique = np.mean(unique_counts)
    median_unique = np.median(unique_counts)
    std_unique = np.std(unique_counts)
    
    avg_score_diff = np.mean(score_differences)
    
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"\nUnique Generations per Prompt:")
    print(f"  Mean:   {mean_unique:.2f}")
    print(f"  Median: {median_unique:.2f}")
    print(f"  Std:    {std_unique:.2f}")
    
    print(f"\nInfoBench Score Difference (Best - Worst):")
    print(f"  Average: {avg_score_diff:.4f}")
    
    print(f"\nDistribution of Unique Counts:")
    unique_dist = Counter(unique_counts)
    for count in sorted(unique_dist.keys())[:10]:
        print(f"  {count:2d} unique: {unique_dist[count]:3d} prompts")
    
    print("\n" + "="*60)
    print("EXAMPLES FOR REFLECTION")
    print("="*60)
    
    if similar_examples:
        print(f"\nLow Diversity Examples (≤10 unique out of 50):")
        for ex in similar_examples[:2]:
            print(f"\n  Question ID: {ex['question_id']}")
            print(f"  Unique: {ex['unique_count']}/{ex['total_count']}")
            print(f"  Prompt: {ex['prompt']}")
            print(f"  Sample outputs:")
            for i, sample in enumerate(ex['samples'], 1):
                print(f"    {i}. {sample[:100]}...")
    
    if diverse_examples:
        print(f"\n\nHigh Diversity Examples (≥40 unique out of 50):")
        for ex in diverse_examples[:2]:
            print(f"\n  Question ID: {ex['question_id']}")
            print(f"  Unique: {ex['unique_count']}/50")
            print(f"  Score range: {ex['score_range']}")
            print(f"  Prompt: {ex['prompt']}")
    
    print("\n" + "="*60)
    print("LENGTH ANALYSIS")
    print("="*60)
    
    all_lengths = []
    for result in all_results['results']:
        lengths = [c['generation_len'] for c in result['candidates']]
        all_lengths.extend(lengths)
    
    print(f"\nGeneration Lengths:")
    print(f"  Mean:   {np.mean(all_lengths):.2f}")
    print(f"  Median: {np.median(all_lengths):.2f}")
    print(f"  Min:    {np.min(all_lengths)}")
    print(f"  Max:    {np.max(all_lengths)}")
    
    return {
        'mean_unique': mean_unique,
        'median_unique': median_unique,
        'std_unique': std_unique,
        'avg_score_diff': avg_score_diff
    }


def reflection_analysis(data_path: str = "results/all_results_processed.json"):
    with open(data_path) as f:
        all_results = json.load(f)
    
    print("\n" + "="*60)
    print("REFLECTION QUESTIONS")
    print("="*60)
    
    # 1. How do generations differ?
    print("\n1. How do generations tend to differ?")
    
    # Look at a few examples
    sample_result = all_results['results'][0]
    texts = [c['generated_text'] for c in sample_result['candidates']]
    
    # Check for length variation
    lengths = [len(t) for t in texts]
    print(f"   - Length variation: {np.std(lengths):.2f} (std)")
    
    # Check for semantic similarity (simple approach: check overlap)
    unique_texts = list(set(texts))
    if len(unique_texts) < len(texts):
        print(f"   - Some generations are exact duplicates")
        print(f"   - {len(texts)} total, {len(unique_texts)} unique")
    
    # 2. Look at task types
    print("\n2. Task types and diversity:")
    
    # Group by instruction type
    task_diversity = {}
    for result in all_results['results']:
        instruction = result['question_data']['instruction']
        texts = [c['generated_text'] for c in result['candidates']]
        unique_count = len(set(texts))
        
        if instruction not in task_diversity:
            task_diversity[instruction] = []
        task_diversity[instruction].append(unique_count)
    
    print("\n   Average diversity by task type:")
    for task, counts in sorted(task_diversity.items(), key=lambda x: np.mean(x[1])):
        if len(counts) >= 3:  # Only show tasks with multiple examples
            print(f"   - {task[:50]:50s}: {np.mean(counts):.1f} unique (avg)")


if __name__ == "__main__":
    import sys
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else "results/all_results_processed.json"
    
    stats = analyze_warmup(data_path)
    
    reflection_analysis(data_path)
    
    print("\n" + "="*60)
    print(f"\nMean unique generations per prompt: {stats['mean_unique']:.2f}")
    print(f"Median unique generations per prompt: {stats['median_unique']:.2f}")
    print(f"Std unique generations per prompt: {stats['std_unique']:.2f}")
    print(f"Average InfoBench score difference (best-worst): {stats['avg_score_diff']:.4f}")
    print("\n" + "="*60)