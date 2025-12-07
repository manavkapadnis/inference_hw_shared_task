#!/bin/bash

# Deploy the optimized inference system to Modal

echo "Deploying inference system to Modal..."
modal deploy server_system_1_accuracy.py

echo "Deployment complete!"
echo "Your endpoint will be available shortly at the URL shown above."