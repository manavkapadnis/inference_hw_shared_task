#!/bin/bash
# Deploy System 6: Balanced 4-bit
# Qwen3-8B + Qwen3-1.7B, both 4-bit

echo "Deploying System 6 (Balanced 4-bit)..."
echo "Large: Qwen3-8B | Small: Qwen3-1.7B | Quantization: 4-bit"

export LARGE_MODEL="Qwen/Qwen3-8B"
export SMALL_MODEL="Qwen/Qwen3-1.7B"
export USE_4BIT="true"
export USE_8BIT="false"

modal deploy modal_deploy_api.py

echo "System 6 deployed successfully!"
echo "URL will be displayed above (format: https://yourModalID--mkapadni-inference-system-inferenceapi-completions.modal.run)"
