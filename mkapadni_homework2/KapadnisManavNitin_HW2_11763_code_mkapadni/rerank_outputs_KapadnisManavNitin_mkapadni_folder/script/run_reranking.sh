#!/bin/bash
#SBATCH --job-name=q2_rerank
#SBATCH --output=q2_rerank.out
#SBATCH --error=q2_rerank.err
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
cd /home/mkapadni/work/inference_algo/homework2/code_solutions


# Set environment variables
export TRANSFORMERS_CACHE="/data/user_data/mkapadni/hf_cache/models"
export HF_HOME="/data/user_data/mkapadni/hf_cache"

python run_reranking.py --plots