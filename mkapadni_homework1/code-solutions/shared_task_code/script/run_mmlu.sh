#!/bin/bash
#SBATCH --job-name=mmlu
#SBATCH --output=mmlu.out
#SBATCH --error=mmlu.err
#SBATCH --partition=shire-general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12  # Adjusted to match the number of workers in your Python script
#SBATCH --gres=gpu:A100_80GB:1  # Requesting one A100 80GB GPU
#SBATCH --mem=128G  # Memory allocation as per your interactive command
#SBATCH --time=72:00:00  # Adjusted to match your interactive command

# Load necessary modules or activate environments
source /home/mkapadni/.bashrc  # Ensure this points to the correct file if different
conda activate gpu_env

# Navigate to the project directory
cd /home/mkapadni/work/inference_algo/homework1/code-solutions/shared_task_code

# Create the save directory if it doesn't exist and run the Python script

python mmlu_benchmark_hf.py