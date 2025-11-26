#!/bin/bash
#SBATCH --job-name=self_refine
#SBATCH --output=self_refine.out
#SBATCH --error=self_refine.err
#SBATCH --partition=shire-general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12  # Adjusted to match the number of workers in your Python script
#SBATCH --gres=gpu:A100_80GB:1  # Requesting one A100 80GB GPU
#SBATCH --mem=100G  # Memory allocation as per your interactive command
#SBATCH --time=72:00:00  # Adjusted to match your interactive command

# Load necessary modules or activate environments
source /home/mkapadni/.bashrc  # Ensure this points to the correct file if different
conda activate gpt_oss

# Navigate to the project directory
cd /home/mkapadni/work/inference_algo/homework2/self_refine_code_solutions

set -e  # Exit on error

# Set environment variables
export TRANSFORMERS_CACHE="/data/user_data/mkapadni/hf_cache/models"
export HF_HOME="/data/user_data/mkapadni/hf_cache"
export CUDA_VISIBLE_DEVICES=0


# Create results directory
mkdir -p results
mkdir -p results/figures

# Configuration 1: Lower temperature for all stages (more deterministic)
echo "Running Config 1: Low Temperature (0.3) for all stages"
python self_refine.py \
    --model Qwen/Qwen3-4B \
    --dataset graphdev \
    --draft_temp 0.3 \
    --critique_temp 0.3 \
    --refine_temp 0.3 \
    --output_dir ./results

python self_refine.py \
    --model Qwen/Qwen3-0.6B \
    --dataset graphdev \
    --draft_temp 0.3 \
    --critique_temp 0.3 \
    --refine_temp 0.3 \
    --output_dir ./results

python self_refine.py \
    --model Qwen/Qwen3-4B \
    --dataset mmlu_med \
    --draft_temp 0.3 \
    --critique_temp 0.3 \
    --refine_temp 0.3 \
    --output_dir ./results

python self_refine.py \
    --model Qwen/Qwen3-0.6B \
    --dataset mmlu_med \
    --draft_temp 0.3 \
    --critique_temp 0.3 \
    --refine_temp 0.3 \
    --output_dir ./results

# Configuration 2: Higher draft temp, lower critique/refine (diverse generation, focused refinement)
echo "Running Config 2: High Draft (0.9), Low Critique/Refine (0.3)"
python self_refine.py \
    --model Qwen/Qwen3-4B \
    --dataset graphdev \
    --draft_temp 0.9 \
    --critique_temp 0.3 \
    --refine_temp 0.3 \
    --output_dir ./results

python self_refine.py \
    --model Qwen/Qwen3-0.6B \
    --dataset graphdev \
    --draft_temp 0.9 \
    --critique_temp 0.3 \
    --refine_temp 0.3 \
    --output_dir ./results

python self_refine.py \
    --model Qwen/Qwen3-4B \
    --dataset mmlu_med \
    --draft_temp 0.9 \
    --critique_temp 0.3 \
    --refine_temp 0.3 \
    --output_dir ./results

python self_refine.py \
    --model Qwen/Qwen3-0.6B \
    --dataset mmlu_med \
    --draft_temp 0.9 \
    --critique_temp 0.3 \
    --refine_temp 0.3 \
    --output_dir ./results

echo "Generating plots..."
python plot_results.py \
    --analysis_files results/analysis_*.json \
    --output_dir ./results/figures

echo "All experiments completed! Check the results directory for outputs."
