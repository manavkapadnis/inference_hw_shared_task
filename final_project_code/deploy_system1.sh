#!/bin/bash
# Deploy System 1: Accuracy-Optimized
# Qwen3-8B + Qwen3-4B, full precision

echo "Deploying System 1 (Accuracy-Optimized)..."
echo "Large: Qwen3-8B | Small: Qwen3-4B | Precision: bfloat16"

export LARGE_MODEL="Qwen/Qwen3-8B"
export SMALL_MODEL="Qwen/Qwen3-4B"
export USE_4BIT="false"
export USE_8BIT="false"

modal deploy modal_deploy_api.py

echo "System 1 deployed successfully!"
echo "URL will be displayed above (format: https://yourModalID--mkapadni-inference-system-inferenceapi-completions.modal.run)"