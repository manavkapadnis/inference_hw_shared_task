"""
analyze_specdec_results.py

Analyzes benchmark outputs from benchmark_dir and generates tables and plots.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    """Parse a benchmark log file to extract metrics."""
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found")
        return None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    metrics = {}
    
    for line in lines:
        line = line.strip()
        
        # Extract acceptance rate
        if 'Average Acceptance Rate' in line:
            try:
                rate_str = line.split(':')[1].strip().rstrip('%')
                metrics['acceptance_rate'] = float(rate_str) / 100
            except:
                pass
        
        # Extract speedup
        if 'Average Empirical Speedup' in line or 'Empirical Speedup' in line:
            try:
                speedup_str = line.split(':')[1].strip().rstrip('x')
                metrics['speedup'] = float(speedup_str)
            except:
                pass
    
    return metrics if metrics else None

def extract_config_from_filename(filename):
    """
    Extract target, draft, and gamma from filename.
    Expected format: {target}_{draft}_gamma{gamma}_overall.log or similar
    """
    # Remove extension
    name = filename.replace('.log', '')
    
    parts = name.split('_')
    
    # Try to extract gamma
    gamma = None
    for part in parts:
        if 'gamma' in part.lower():
            try:
                gamma = int(part.replace('gamma', '').replace('Gamma', ''))
            except:
                pass
    
    # Try to identify target and draft models
    # This is heuristic-based
    if 'Qwen' in name and 'Qwen3-8B' in name:
        target = "Qwen/Qwen3-8B"
        if 'Qwen3-1.7B' in name:
            draft = "Qwen/Qwen3-1.7B"
        elif 'Qwen3-0.6B' in name:
            draft = "Qwen/Qwen3-0.6B"
        else:
            return None
    elif 'Llama' in name or 'llama' in name:
        target = "meta-llama/Llama-3.1-8B"
        draft = "meta-llama/Llama-3.2-1B"
    else:
        return None
    
    return {"target": target, "draft": draft, "gamma": gamma}

def parse_json_results(benchmark_dir):
    """Parse JSON result files if they exist."""
    results = []
    
    for json_file in Path(benchmark_dir).glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract configuration from filename or data
            config = extract_config_from_filename(json_file.stem)
            if config and 'acceptance_rate' in data and 'speedup' in data:
                results.append({
                    **config,
                    'acceptance_rate': data['acceptance_rate'],
                    'speedup': data['speedup']
                })
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
    
    return results

def main():
    """Main analysis function."""
    print("="*80)
    print("SPECULATIVE DECODING RESULTS ANALYSIS")
    print("="*80)
    
    benchmark_dir = "benchmark_dir"
    
    if not os.path.exists(benchmark_dir):
        print(f"Error: {benchmark_dir} directory not found!")
        print("Make sure you've run the benchmark.py commands first.")
        return
    
    results = []
    
    # Parse all log files
    print("\nParsing log files...")
    log_files = list(Path(benchmark_dir).glob("*.log"))
    
    if not log_files:
        print("No log files found in benchmark_dir!")
        return
    
    for log_file in log_files:
        print(f"  Processing: {log_file.name}")
        
        # Try to extract config from filename
        config = extract_config_from_filename(log_file.name)
        
        if not config:
            # Try manual patterns based on actual benchmark.py output names
            name = log_file.stem
            
            # Pattern: {target_sanitized}_{draft_sanitized}_overall_speedup
            if '_overall_speedup' in name:
                parts = name.replace('_overall_speedup', '').split('_')
                
                # Reconstruct model names
                if 'Qwen' in name:
                    if 'Qwen3-8B' in name or 'Qwen_Qwen3-8B' in name:
                        target = "Qwen/Qwen3-8B"
                        if 'Qwen3-1.7B' in name:
                            draft = "Qwen/Qwen3-1.7B"
                        elif 'Qwen3-0.6B' in name:
                            draft = "Qwen/Qwen3-0.6B"
                        else:
                            continue
                    else:
                        continue
                elif 'Llama' in name or 'llama' in name:
                    target = "meta-llama/Llama-3.1-8B"
                    draft = "meta-llama/Llama-3.2-1B"
                else:
                    continue
                
                # Gamma should be in the directory structure or we need to track it
                # Let's check if there's a pattern
                config = {"target": target, "draft": draft, "gamma": None}
        
        # Parse metrics
        metrics = parse_log_file(log_file)
        
        if metrics:
            if config and config['gamma'] is not None:
                results.append({
                    "Target": config['target'],
                    "Draft": config['draft'],
                    "Gamma": config['gamma'],
                    "Acceptance Rate": metrics['acceptance_rate'],
                    "Speedup": metrics['speedup']
                })
                print(f"    ✓ Gamma={config['gamma']}, α={metrics['acceptance_rate']:.2%}, Speedup={metrics['speedup']:.2f}x")
            else:
                # Store without gamma for manual assignment
                print(f"    ⚠ Could not extract gamma from filename")
                print(f"      Metrics: α={metrics.get('acceptance_rate', 'N/A')}, Speedup={metrics.get('speedup', 'N/A')}")
    
    # If we couldn't parse gamma from filenames, try to infer from separate log files per gamma
    # Or prompt user to manually organize
    if not results:
        print("\n⚠ No results with gamma values found.")
        print("The log files may need manual organization.")
        print("Expected pattern: separate runs with gamma in filename or separate directories")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort for better display
    df = df.sort_values(['Target', 'Draft', 'Gamma'])
    
    # Save to CSV
    df.to_csv("specdec_results_summary.csv", index=False)
    print(f"\n✓ Results saved to specdec_results_summary.csv")
    
    # Print formatted table
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(f"{'Target':<30} {'Draft':<30} {'γ':<5} {'α':<10} {'Speedup':<10}")
    print("-"*85)
    for _, row in df.iterrows():
        target_short = row['Target'].split('/')[-1]
        draft_short = row['Draft'].split('/')[-1]
        print(f"{target_short:<30} {draft_short:<30} {row['Gamma']:<5} "
              f"{row['Acceptance Rate']:<10.2%} {row['Speedup']:<10.2f}x")
    
    # Create LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    print("\\begin{tabular}{llccc}")
    print("\\toprule")
    print("Target Model & Draft Model & $\\gamma$ & $\\alpha$ & Speedup \\\\")
    print("\\midrule")
    for _, row in df.iterrows():
        target_short = row['Target'].split('/')[-1]
        draft_short = row['Draft'].split('/')[-1]
        print(f"{target_short} & {draft_short} & {row['Gamma']} & "
              f"{row['Acceptance Rate']:.2%} & {row['Speedup']:.2f}x \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    # Plot 1: Speedup vs Gamma
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for (target, draft), group in df.groupby(['Target', 'Draft']):
        label = f"{target.split('/')[-1]} + {draft.split('/')[-1]}"
        ax.plot(group['Gamma'], group['Speedup'], marker='o', 
                linewidth=2.5, markersize=10, label=label)
        
        # Add value labels
        for _, row in group.iterrows():
            ax.text(row['Gamma'], row['Speedup'], f"{row['Speedup']:.2f}x",
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Lookahead (γ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=13, fontweight='bold')
    ax.set_title('Speculative Decoding: Speedup vs Lookahead', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks([2, 3, 5, 7])
    plt.tight_layout()
    plt.savefig('specdec_speedup_vs_gamma.png', dpi=300, bbox_inches='tight')
    print("✓ Saved specdec_speedup_vs_gamma.png")
    plt.close()
    
    # Plot 2: Acceptance Rate vs Gamma
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for (target, draft), group in df.groupby(['Target', 'Draft']):
        label = f"{target.split('/')[-1]} + {draft.split('/')[-1]}"
        ax.plot(group['Gamma'], group['Acceptance Rate'] * 100, marker='s',
                linewidth=2.5, markersize=10, label=label)
        
        # Add value labels
        for _, row in group.iterrows():
            ax.text(row['Gamma'], row['Acceptance Rate'] * 100, 
                   f"{row['Acceptance Rate']:.1%}",
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Lookahead (γ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Acceptance Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Speculative Decoding: Acceptance Rate vs Lookahead', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks([2, 3, 5, 7])
    plt.tight_layout()
    plt.savefig('specdec_acceptance_vs_gamma.png', dpi=300, bbox_inches='tight')
    print("✓ Saved specdec_acceptance_vs_gamma.png")
    plt.close()
    
    # Find best configuration
    best_idx = df['Speedup'].idxmax()
    best = df.loc[best_idx]
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"Target Model:    {best['Target']}")
    print(f"Draft Model:     {best['Draft']}")
    print(f"Lookahead (γ):   {best['Gamma']}")
    print(f"Acceptance Rate: {best['Acceptance Rate']:.2%}")
    print(f"Speedup:         {best['Speedup']:.2f}x")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - specdec_results_summary.csv")
    print("  - specdec_speedup_vs_gamma.png")
    print("  - specdec_acceptance_vs_gamma.png")

if __name__ == "__main__":
    main()
