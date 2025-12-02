# LLM Inference System - Modal API Deployment & Evaluation

## Overview
Complete inference system with GPU placement, continuous batching, and robust GraphDev parsing. Supports Systems 1, 4, and 6 with Modal deployment and automated evaluation.

## Key Features
1. **GPU Placement**: Large model on GPU:0, small model on GPU:1 (or both on GPU:0 if single GPU)
2. **Continuous Batching**: Handles variable batch sizes up to 300 concurrent requests
3. **Robust GraphDev Parsing**: Multi-level regex extraction with fallback to original parameters
4. **Variable max_tokens**: Respects max_tokens from API requests (default: 2048)
5. **Evaluation Pipeline**: Automated accuracy and throughput measurement

## Quick Start

### 1. Deploy System to Modal

```bash
# System 1 (Accuracy): Qwen3-8B + Qwen3-4B, bfloat16
./deploy_system1.sh

# System 4 (High-Capacity): Qwen3-14B + Qwen3-8B, 4-bit
./deploy_system4.sh

# System 6 (Balanced): Qwen3-8B + Qwen3-1.7B, 4-bit
./deploy_system6.sh
```

After deployment, Modal will output your API endpoint URL:
```
https://yourModalID--mkapadni-inference-system-inferenceapi-completions.modal.run
```

### 2. Evaluate Deployed System

```bash
cd evaluation

# Create .env file with OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Run evaluation with batch arrivals
python eval_api.py \
  --url "YOUR_MODAL_URL" \
  --batch_arrivals batch_arrivals.json \
  --hidden_test combined_dataset.jsonl \
  --output_dir eval_results_system1 \
  --max_concurrent 300
```

### 3. View Results

Results are saved in `eval_results_system*/`:
- `api_results.json` - Raw API responses and timing
- `eval_results.jsonl` - Per-example evaluation
- `metrics.json` - Accuracy and throughput metrics

Example metrics:
```json
{
  "total_examples": 100,
  "task_metrics": {
    "graphdev": {"accuracy": 0.9133, "count": 33},
    "mmlu_med": {"accuracy": 0.8750, "count": 33},
    "infobench": {"accuracy": 0.8235, "count": 34}
  },
  "overall_accuracy": 0.8706
}
```

## System Configurations

| System | Large Model | Small Model | Quantization | Target |
|--------|-------------|-------------|--------------|--------|
| 1 | Qwen3-8B | Qwen3-4B | bfloat16 | Accuracy (>90%) |
| 4 | Qwen3-14B | Qwen3-8B | 4-bit | High Accuracy + Efficiency |
| 6 | Qwen3-8B | Qwen3-1.7B | 4-bit | Balanced Performance |

## API Usage

### Request Format
```python
import requests

response = requests.post(
    "YOUR_MODAL_URL",
    json={
        "prompt": ["prompt1", "prompt2"],  # Single string or list
        "max_tokens": 2048,                # Optional, default: 2048
        "temperature": 0.7                 # Optional, default: 0.7
    }
)
```

### Response Format
```json
{
  "choices": [
    {
      "text": "generated text here",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "model": "inference-system",
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  },
  "metadata": {
    "generation_time": 2.5,
    "batch_size": 2
  }
}
```

## GraphDev Robust Parsing

The system now extracts graph parameters from various prompt formats:

**Supported Formats:**
- Standard: `"graph with 10 nodes"`, `"top 1 shortest path"`
- Alternative: `"nodes numbered 0 to 9"`, `"find 3 shortest"`
- Edges: `"0 -> 1, weight: 5"`, `"edge (0, 1) with weight 5"`, `"[0, 1, 5]"`

**Extraction Process:**
1. Try extracting from LLM response
2. If fails, extract from original prompt  
3. If fails, use original example parameters
4. Validate and call pathfinding algorithm

## Files Structure

```
final_project/
├── inference_system.py       # Core inference with GPU placement
├── dataset_handlers.py        # Enhanced GraphDev parsing
├── modal_deploy_api.py       # Modal deployment script
├── deploy_system1.sh         # Deploy System 1
├── deploy_system4.sh         # Deploy System 4
├── deploy_system6.sh         # Deploy System 6
├── evaluate_local.py         # Local SLURM evaluation
├── system1_*.sh              # SLURM scripts (9 total)
├── system4_*.sh              # SLURM scripts (6 total)
├── system5_*.sh              # SLURM scripts (6 total)
├── system6_*.sh              # SLURM scripts (6 total)
└── evaluation/
    ├── grader.py             # Evaluation logic (MMLU, Graph, InfoBench)
    ├── eval_api.py           # API evaluation script
    ├── requirements.txt      # Evaluation dependencies
    └── .env                  # OpenAI API key (create this)
```

## Troubleshooting

**GPU Placement Not Working:**
- Check `torch.cuda.device_count()` in logs
- Verify Modal has 2 GPUs allocated
- System automatically falls back to single GPU if needed

**Evaluation Timeouts:**
- Increase `--max_concurrent` (default: 300)
- Check Modal timeout settings (default: 600s)
- Verify API endpoint is responsive

**GraphDev Parsing Errors:**
- System automatically falls back through 3 extraction methods
- Check logs for "method" field: "regex_extraction", "original_params", or "failed"
- Most failures resolved by fallback to original parameters

## Commands Summary

```bash
# Deploy systems
./deploy_system1.sh  # System 1
./deploy_system4.sh  # System 4
./deploy_system6.sh  # System 6

# Evaluate deployed system
cd evaluation
python eval_api.py --url "YOUR_URL" \
  --batch_arrivals batch_arrivals.json \
  --hidden_test combined_dataset.jsonl \
  --output_dir results

# SLURM evaluation (if needed)
cd ..
sbatch system1_graphdev.sh
sbatch system4_graphdev.sh
# ... etc
```

## Notes
- TAs will test with batch_arrivals.json containing mixed tasks
- max_tokens can vary per request (default: 2048 if not specified)
- Continuous batching handles variable batch sizes automatically
- All systems support up to 300 concurrent requests