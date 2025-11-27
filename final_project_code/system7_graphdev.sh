#!/bin/bash
#SBATCH --job-name=system7_graphdev
#SBATCH --partition=shire-general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=100GB
#SBATCH --time=100:00:00
#SBATCH --output=/home/mkapadni/work/inference_algo/homework4/attempt_1/out_file/system7_graphdev_new_function.out
#SBATCH --error=/home/mkapadni/work/inference_algo/homework4/attempt_1/err_file/system7_graphdev_new_function.err

# Load environment
source ~/.bashrc
conda activate gpt_oss

# Set working directory
cd /home/mkapadni/work/inference_algo/homework4/attempt_1

# Set HuggingFace cache
export TRANSFORMERS_CACHE=/data/user_data/mkapadni/hf_cache
export HF_HOME=/data/user_data/mkapadni/hf_cache

# Run evaluation - System 6: Qwen3-1.7B + Qwen3-0.6B, both 4-bit
python evaluate_local.py \
    --task graphdev \
    --large_model Qwen/Qwen3-1.7B \
    --small_model Qwen/Qwen3-0.6B \
    --use_4bit \
    --batch_size 1 \
    --output results/system6_graphdev_smallest_model.json

echo "System 7 GraphDev evaluation complete!"
