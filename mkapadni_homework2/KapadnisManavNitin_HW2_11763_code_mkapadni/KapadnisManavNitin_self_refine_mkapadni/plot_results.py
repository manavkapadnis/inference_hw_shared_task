# plot_results.py
# Andrew id: mkapadni

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

def plot_accuracy_curves(analysis_files, output_dir):
    """Plot accuracy and best-accuracy-so-far curves - separate by dataset."""
    
    dataset_files = defaultdict(list)
    for f in analysis_files:
        filename = Path(f).stem
        if 'graphdev' in filename.lower():
            dataset_files['GraphDev'].append(f)
        elif 'mmlu' in filename.lower():
            dataset_files['MMLU_Med'].append(f)

    for dataset_name, files in dataset_files.items():
        if not files:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for analysis_file in files:
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)

            # Create shortened label
            filename = Path(analysis_file).stem.replace("analysis_", "")
            parts = filename.split('_')
            model = parts[0] if parts else "model"
            temp_info = [p for p in parts if 'temp' in p]
            temp_str = temp_info[0].replace('temp', 'T=') if temp_info else ""

            label = f"{model} {temp_str}"

            iterations = list(range(1, len(analysis['accuracy_by_iteration']) + 1))

            # Plot accuracy by iteration
            ax1.plot(iterations, analysis['accuracy_by_iteration'], marker='o', label=label, linewidth=2)

            # Plot best accuracy so far
            ax2.plot(iterations, analysis['best_accuracy_by_iteration'], marker='s', label=label, linewidth=2)

        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title(f'{dataset_name}: Iter Accuracy', fontsize=14)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)

        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Best Accuracy', fontsize=12)
        ax2.set_title(f'{dataset_name}: Best So Far', fontsize=14)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.0)

        plt.tight_layout()
        output_file = output_dir / f'accuracy_curves_{dataset_name.lower()}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def plot_conditional_probabilities(analysis_files, output_dir):
    """Plot conditional probability curves - one plot per file."""
    for analysis_file in analysis_files:
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)

        filename = Path(analysis_file).stem.replace("analysis_", "")

        # Determine dataset name
        if 'graphdev' in filename.lower():
            dataset_tag = 'GraphDev'
        elif 'mmlu' in filename.lower():
            dataset_tag = 'MMLU_Med'
        else:
            dataset_tag = 'Unknown'

        # Extract model and temp
        parts = filename.split('_')
        model = parts[0] if parts else "model"
        temp_info = [p for p in parts if 'temp' in p]
        temp_str = temp_info[0].replace('temp', 'T=') if temp_info else ""

        title = f"{dataset_tag}: {model} {temp_str}"

        if not analysis['conditional_probabilities']:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        iterations = [cp['iteration'] for cp in analysis['conditional_probabilities']]
        p_correct_given_correct = [cp['P(correct_i+1|correct_i)'] for cp in analysis['conditional_probabilities']]
        p_correct_given_incorrect = [cp['P(correct_i+1|incorrect_i)'] for cp in analysis['conditional_probabilities']]

        x = np.arange(len(iterations))
        width = 0.35

        ax.bar(x - width/2, p_correct_given_correct, width, 
               label='P(correct_i+1 | correct_i)', alpha=0.8, color='#2E7D32')
        ax.bar(x + width/2, p_correct_given_incorrect, width, 
               label='P(correct_i+1 | incorrect_i)', alpha=0.8, color='#C62828')

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'Conditional Probabilities: {title}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(iterations)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        output_file = output_dir / f'conditional_probs_{filename}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot self-refine results")
    parser.add_argument("--analysis_files", nargs='+', required=True, help="Analysis JSON files")
    parser.add_argument("--output_dir", type=str, default="./results/figures", help="Output directory for plots")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plot_accuracy_curves(args.analysis_files, output_dir)
    plot_conditional_probabilities(args.analysis_files, output_dir)
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()