# Q3 Self-Refine Implementation 

This repository contains the implementation of Self-Refine for the inference class homework.

## File Structure

```
.
├── dataset.py              # Dataset handlers for GraphDev and MMLU_Med
├── self_refine.py          # Main self-refine implementation
├── plot_results.py         # Plotting script for analysis
├── run_self_refine.sh      # Bash script to run all experiments
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── results/               # Output directory (created automatically)
    ├── results_*.json     # Raw results for each run
    ├── analysis_*.json    # Analysis metrics
    └── figures/           # Generated plots
```

## Installation

```
# Install dependencies
pip install -r requirements.txt

# Set HuggingFace cache (optional)
export TRANSFORMERS_CACHE="/path/to/your/cache"
```

## Usage

### Quick Start

Run all experiments with default configurations:

```
chmod +x run_self_refine.sh
./run_self_refine.sh
```

### Manual Execution

Run individual experiments:

```
# GraphDev with Qwen3-4B
python self_refine.py \
    --model Qwen/Qwen3-4B \
    --dataset graphdev \
    --draft_temp 0.7 \
    --critique_temp 0.7 \
    --refine_temp 0.7 \
    --output_dir ./results

# MMLU_Med with Qwen3-0.6B
python self_refine.py \
    --model Qwen/Qwen3-0.6B \
    --dataset mmlu_med \
    --draft_temp 0.7 \
    --critique_temp 0.7 \
    --refine_temp 0.7 \
    --output_dir ./results
```

### Generate Plots

After running experiments:

```
python plot_results.py \
    --analysis_files results/analysis_*.json \
    --output_dir ./results/figures
```

## Configuration

### Temperature Settings

The script supports different temperature configurations for each stage:

- **Configuration 1**: Low temperature (0.3) - More deterministic, consistent outputs
  - Draft: 0.3, Critique: 0.3, Refine: 0.3
  
- **Configuration 2**: High draft, low refinement (0.9/0.3/0.3) - Diverse initial generation, focused refinement
  - Draft: 0.9, Critique: 0.3, Refine: 0.3

### Command Line Arguments

```
--model: Model path (e.g., Qwen/Qwen3-4B)
--dataset: Dataset name (graphdev or mmlu_med)
--max_iterations: Max iterations (default: 4)
--draft_temp: Temperature for draft generation
--critique_temp: Temperature for critique
--refine_temp: Temperature for refinement
--output_dir: Output directory
--max_examples: Limit number of examples (for testing)
```

## Outputs

### JSON Files

1. **results_*.json**: Raw results with all iterations
   - Contains full responses, parsed answers, correctness for each iteration
   
2. **analysis_*.json**: Computed metrics
   - Accuracy by iteration
   - Best accuracy so far
   - Conditional probabilities P(correct_{i+1} | correct_i) and P(correct_{i+1} | incorrect_i)
   - Example cases where refinement helped/harmed

### Plots

1. **accuracy_curves.png**: 
   - Accuracy at each iteration
   - Best accuracy up to each iteration

2. **conditional_probabilities.png**:
   - P(correct_{i+1} | correct_i)
   - P(correct_{i+1} | incorrect_i)
   - For each model and dataset

