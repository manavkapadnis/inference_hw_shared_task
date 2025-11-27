#!/bin/bash
#SBATCH --job-name=system5_mmlu
#SBATCH --partition=shire-general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=100GB
#SBATCH --time=100:00:00
#SBATCH --output=/home/mkapadni/work/inference_algo/homework4/attempt_1/out_file/system5_mmlu.out
#SBATCH --error=/home/mkapadni/work/inference_algo/homework4/attempt_1/err_file/system5_mmlu.err

# Load environment
source ~/.bashrc
conda activate gpt_oss

# Set working directory
cd /home/mkapadni/work/inference_algo/homework4/attempt_1

# Set HuggingFace cache
export TRANSFORMERS_CACHE=/data/user_data/mkapadni/hf_cache
export HF_HOME=/data/user_data/mkapadni/hf_cache

# Run evaluation - System 5: Qwen3-8B + Qwen3-1.7B (8-bit) with Enhanced Inference
python evaluate_local.py \
    --task mmlu_med \
    --large_model Qwen/Qwen3-8B \
    --small_model Qwen/Qwen3-1.7B \
    --use_8bit \
    --use_enhanced \
    --batch_size 1 \
    --output results/system5_mmlu.json

echo "System 5 MMLU evaluation complete!"
