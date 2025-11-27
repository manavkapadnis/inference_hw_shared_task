#!/bin/bash
#SBATCH --job-name=sys1_infobench
#SBATCH --output=/home/mkapadni/work/inference_algo/homework4/attempt_1/out_file/sys1_infobench.out
#SBATCH --error=/home/mkapadni/work/inference_algo/homework4/attempt_1/err_file/sys1_infobench.err
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

# System 1: Accuracy-Optimized (Qwen3-8B + Qwen3-4B, full precision)
python evaluate_local.py \
    --task infobench \
    --large_model Qwen/Qwen3-8B \
    --small_model Qwen/Qwen3-4B \
    --batch_size 1 \
    --output results/system1_infobench.json
