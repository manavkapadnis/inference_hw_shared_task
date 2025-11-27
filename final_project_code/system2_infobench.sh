#!/bin/bash
#SBATCH --job-name=sys2_infobench
#SBATCH --output=/home/mkapadni/work/inference_algo/homework4/attempt_1/out_file/sys2_infobench.out
#SBATCH --error=/home/mkapadni/work/inference_algo/homework4/attempt_1/err_file/sys2_infobench.err
#SBATCH --partition=shire-general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=100G
#SBATCH --time=100:00:00

source /home/mkapadni/.bashrc
conda activate gpt_oss

cd /home/mkapadni/work/inference_algo/homework4/attempt_1

export TRANSFORMERS_CACHE="/data/user_data/mkapadni/hf_cache/models"
export HF_HOME="/data/user_data/mkapadni/hf_cache"

# System 2: Speed-Optimized (Qwen3-4B + Qwen3-0.6B, 8-bit quantization)
python evaluate_local.py \
    --task infobench \
    --large_model Qwen/Qwen3-4B \
    --small_model Qwen/Qwen3-0.6B \
    --use_8bit \
    --batch_size 1 \
    --output results/system2_infobench.json
