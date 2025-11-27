#!/bin/bash

# Deployment script for Final Project Inference System
# This script deploys the inference system to Modal

set -e  # Exit on error

echo "========================================="
echo "Final Project Inference System Deployment"
echo "========================================="
echo ""

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "Error: Modal CLI not found. Please install with:"
    echo "  pip install modal"
    exit 1
fi

# Check if user is authenticated
if ! modal token show &> /dev/null; then
    echo "Error: Not authenticated with Modal. Please run:"
    echo "  modal token new"
    exit 1
fi

echo "Deploying inference_system.py as a Modal volume..."
modal volume put inference-system inference_system.py /root/inference_system.py || {
    echo "Creating volume and uploading..."
    modal volume create inference-system
    modal volume put inference-system inference_system.py /root/inference_system.py
}

echo ""
echo "Deploying to Modal..."
modal deploy modal_deploy.py

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "Your inference system is now live at:"
echo "https://YOUR_MODAL_USERNAME--mkapadni-system-1-model-completions.modal.run"
echo ""
echo "To test the endpoint, update hit_endpoint.py with your Modal username"
echo "and run: python hit_endpoint.py"
echo ""
echo "To view logs: modal app logs mkapadni-system-1"
echo "To stop the app: modal app stop mkapadni-system-1"
echo "========================================="
