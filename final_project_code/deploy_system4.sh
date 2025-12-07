#!/bin/bash
# Deploy System 1: mixed-Optimized
# Qwen3-14B + Qwen3-4B, full precision

echo "Deploying System 1 (Accuracy-Optimized)..."
echo "Large: Qwen3-14B | Small: Qwen3-4B | Precision: 4 bit"

export LARGE_MODEL="Qwen/Qwen3-8B"
export SMALL_MODEL="Qwen/Qwen3-4B"
export USE_4BIT="true"
export USE_8BIT="false"

modal deploy modal_deploy_api.py

echo "System 4 deployed successfully!"
echo "URL will be displayed above (format: https://yourModalID--mkapadni-inference-system-inferenceapi-completions.modal.run)"