#!/bin/bash
#SBATCH --job-name=system4_infobench
#SBATCH --partition=shire-general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=100GB
#SBATCH --time=100:00:00
#SBATCH --output=/home/mkapadni/work/inference_algo/homework4/attempt_1/out_file/system4_infobench.out
#SBATCH --error=/home/mkapadni/work/inference_algo/homework4/attempt_1/err_file/system4_infobench.err

# Load environment
source ~/.bashrc
conda activate gpt_oss

# Set working directory
cd /home/mkapadni/work/inference_algo/homework4/attempt_1

# Set HuggingFace cache
export TRANSFORMERS_CACHE=/data/user_data/mkapadni/hf_cache
export HF_HOME=/data/user_data/mkapadni/hf_cache

# Run evaluation - System 4: Qwen3-14B + Qwen3-8B, both 4-bit
python evaluate_local.py \
    --task infobench \
    --large_model Qwen/Qwen3-14B \
    --small_model Qwen/Qwen3-8B \
    --use_4bit \
    --batch_size 1 \
    --output results/system4_infobench.json

echo "System 4 InfoBench evaluation complete!"
