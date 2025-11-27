# Final Project: Optimized LLM Inference System

An optimized inference system for handling three tasks: GraphDev (shortest path finding), MMLU Medicine, and InfoBench. Built with Qwen3 models and deployed on Modal.

## System Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Request Handler                         │
│  (OpenAI-compatible API with batching support)             │
└─────────────────┬───────────────────────────────────────────┘
                  │
         ┌────────▼─────────┐
         │   Task Router    │
         │ (Identifies task │
         │  from prompt)    │
         └────────┬─────────┘
                  │
     ┌────────────┼────────────┐
     │            │            │
┌────▼────┐  ┌───▼────┐  ┌───▼─────┐
│  Graph  │  │  MMLU  │  │InfoBench│
│ Handler │  │Handler │  │ Handler │
└────┬────┘  └───┬────┘  └────┬────┘
     │            │            │
     └────────────┼────────────┘
                  │
         ┌────────▼─────────┐
         │  Model Selection │
         │  (Large/Small)   │
         └────────┬─────────┘
                  │
     ┌────────────┼────────────┐
     │                         │
┌────▼─────┐           ┌──────▼──────┐
│Qwen3-8B  │           │Qwen3-1.7B   │
│(Large)   │           │(Small)      │
└──────────┘           └─────────────┘
```

### Key Features

1. **Intelligent Task Routing**: Automatically identifies task type from prompt patterns
2. **Model Selection**: Routes to appropriate model based on task complexity
3. **Batch Processing**: Efficient batching for multiple requests
4. **Continuous Batching**: Dynamic request scheduling
5. **Optional Speculative Decoding**: 2-3x speedup for certain tasks
6. **Modal Deployment**: Scalable cloud deployment with concurrent request handling

## Hardware Requirements

- **Development**: 1x GPU with 40GB+ VRAM (e.g., A100, A6000)
- **Production (Modal)**: 2x A100 80GB GPUs (as specified)

## Installation

### 1. Setup Environment

```bash
# Create conda environment
conda create -n inference_project python=3.10
conda activate inference_project

# Install dependencies
pip install torch transformers accelerate datasets
pip install modal fastapi sentencepiece protobuf
pip install tqdm
```

### 2. Setup Modal

```bash
# Install Modal CLI
pip install modal

# Create Modal account and authenticate
modal token new
```

### 3. Clone/Copy Project Files

Place all project files in a directory:
```
final_project/
├── inference_system.py      # Main inference system
├── enhanced_inference.py    # Optional: with speculative decoding
├── dataset_handlers.py      # Task-specific handlers
├── modal_deploy.py          # Modal deployment
├── evaluate_local.py        # Local evaluation
├── deploy.sh                # Deployment script
├── hit_endpoint.py          # Endpoint testing
└── README.md                # This file
```

## Usage

### Local Testing

#### Test on a Small Dataset

```bash
# Test GraphDev task
python evaluate_local.py \
    --task graphdev \
    --limit 10 \
    --batch_size 4 \
    --output results/graphdev_test.json

# Test MMLU Medicine task
python evaluate_local.py \
    --task mmlu_med \
    --limit 10 \
    --batch_size 4 \
    --output results/mmlu_test.json

# Test InfoBench task
python evaluate_local.py \
    --task infobench \
    --limit 10 \
    --batch_size 4 \
    --output results/infobench_test.json
```

#### Test with 8-bit Quantization (for memory efficiency)

```bash
python evaluate_local.py \
    --task graphdev \
    --limit 50 \
    --use_8bit \
    --output results/graphdev_8bit.json
```

### Modal Deployment

#### 1. Update Andrew ID

Edit `modal_deploy.py` and replace `mkapadni` with your Andrew ID:

```python
app = modal.App("YOUR_ANDREW_ID-system-1")
```

#### 2. Deploy

```bash
# Make deploy script executable
chmod +x deploy.sh

# Deploy to Modal
./deploy.sh
```

#### 3. Test Deployed Endpoint

Update `hit_endpoint.py` with your Modal username, then:

```bash
python hit_endpoint.py
```

### Testing Different Configurations

#### Configuration 1: Accuracy-Focused (Default)

Uses larger models for all tasks.

```python
# In modal_deploy.py
self.inference_system = InferenceSystem(
    large_model_path="Qwen/Qwen3-8B",
    small_model_path="Qwen/Qwen3-1.7B",
    use_8bit=False
)
```

#### Configuration 2: Speed-Focused

Uses smaller models and more aggressive batching.

```python
self.inference_system = InferenceSystem(
    large_model_path="Qwen/Qwen3-4B",  # Smaller
    small_model_path="Qwen/Qwen3-0.6B",
    use_8bit=True  # Quantization for speed
)
```

#### Configuration 3: With Speculative Decoding

For maximum speed on generation-heavy tasks:

```python
from enhanced_inference import EnhancedInferenceSystem

self.inference_system = EnhancedInferenceSystem(
    large_model_path="Qwen/Qwen3-8B",
    small_model_path="Qwen/Qwen3-1.7B",
    use_speculative=True,
    draft_model_path="Qwen/Qwen3-0.6B"
)
```

## System Configurations

### System 1: Accuracy-Optimized

**Target**: Maximize task accuracy scores

**Configuration**:
- Large Model: Qwen3-8B (full precision)
- Small Model: Qwen3-4B (for simple tasks)
- Temperature: 0.3 for structured tasks, 0.7 for open-ended
- Batch Size: 4-8
- No quantization

**Expected Performance**:
- Task Accuracy: 90%+ on baseline
- Throughput: ~15-20 requests/min

### System 2: Speed-Optimized

**Target**: Maximize throughput

**Configuration**:
- Large Model: Qwen3-4B (8-bit)
- Small Model: Qwen3-0.6B (8-bit)
- Speculative Decoding: Enabled
- Batch Size: 16-32
- Aggressive batching

**Expected Performance**:
- Task Accuracy: 80%+ on baseline
- Throughput: ~50-100 requests/min

### System 3: Pareto-Optimal (Hybrid)

**Target**: Balance accuracy and throughput

**Configuration**:
- Large Model: Qwen3-8B (8-bit)
- Small Model: Qwen3-1.7B (8-bit)
- Intelligent routing based on task complexity
- Batch Size: 8-16
- Selective speculative decoding

**Expected Performance**:
- Task Accuracy: 85-90% on baseline
- Throughput: ~30-40 requests/min

# System 4 and System 5 - LLM Inference Final Project

## System 4: High-Capacity 4-bit Models
**Configuration:**
- Large Model: Qwen3-14B (4-bit quantization)
- Small Model: Qwen3-8B (4-bit quantization)
- Batch Size: 6
- Quantization: 4-bit with BitsAndBytes

**Expected Performance:**
- Accuracy: ~92-95% (larger models)
- Throughput: 25-35 req/min
- Target: Best accuracy with efficient memory usage

**SLURM Scripts:**
- `system4_graphdev.sh` - GraphDev task
- `system4_mmlu.sh` - MMLU Medicine task
- `system4_infobench.sh` - InfoBench task

## System 5: Enhanced Inference with Speculative Decoding
**Configuration:**
- Large Model: Qwen3-8B (8-bit quantization)
- Small Model: Qwen3-1.7B (8-bit quantization)
- Batch Size: 6
- Quantization: 8-bit
- **Special Feature:** Uses enhanced_inference.py with speculative decoding support

**Expected Performance:**
- Accuracy: 85-90%
- Throughput: 35-45 req/min (faster with speculative decoding)
- Target: Balanced performance with optional speed boost

**SLURM Scripts:**
- `system5_graphdev.sh` - GraphDev task
- `system5_mmlu.sh` - MMLU Medicine task
- `system5_infobench.sh` - InfoBench task

## Setup Instructions

1. **Create output directories:**
```bash
mkdir -p /home/mkapadni/work/inference_algo/homework4/attempt_1/out_file
mkdir -p /home/mkapadni/work/inference_algo/homework4/attempt_1/err_file
mkdir -p /home/mkapadni/work/inference_algo/homework4/attempt_1/results
```

2. **Create .env file with OpenAI API key:**
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

3. **Submit jobs:**
```bash
# System 4 jobs
sbatch system4_graphdev.sh
sbatch system4_mmlu.sh
sbatch system4_infobench.sh

# System 5 jobs
sbatch system5_graphdev.sh
sbatch system5_mmlu.sh
sbatch system5_infobench.sh
```

4. **Monitor jobs:**
```bash
squeue -u mkapadni
```

5. **Check results:**
```bash
ls -lh results/system4_*.json
ls -lh results/system5_*.json
```

## Key Differences

**System 4 vs System 3:**
- Uses larger models (14B vs 8B large, 8B vs 1.7B small)
- Uses 4-bit quantization instead of 8-bit
- Expected higher accuracy with similar memory footprint

**System 5 vs System 3:**
- Same model sizes and quantization
- Uses enhanced_inference.py with speculative decoding capability
- Expected similar/better throughput with potential accuracy benefits

## Files Modified/Added
- `inference_system.py` - Added 4-bit quantization support
- `evaluate_local.py` - Added --use_4bit and --use_enhanced flags
- `enhanced_inference.py` - Already supports 4-bit/8-bit (no changes needed)
- 6 new SLURM scripts for System 4 and 5

## Dependencies
All required packages are in requirements.txt. Main additions:
- `bitsandbytes` - For 4-bit quantization
- `accelerate` - For efficient model loading

## Task-Specific Optimizations

### GraphDev (Shortest Path)
- Uses larger model for accuracy
- Lower temperature (0.3) for structured output
- Max tokens: 1024
- JSON parsing with fallback mechanisms

### MMLU Medicine
- Uses larger model (medical questions are tricky)
- Standard temperature (0.7)
- Max tokens: 256 (multiple choice is short)
- Pattern-based answer extraction

### InfoBench
- Routes based on prompt complexity
- Long/complex → Large model
- Short/simple → Small model
- Max tokens: 512
- Thinking tag removal

## Routing Logic

```python
def route_to_model(prompt, task, prompt_length):
    if task == "graph":
        return "large"  # Always use large for graphs
    
    if task == "mmlu":
        return "large"  # Medical questions need accuracy
    
    if task == "infobench":
        if prompt_length > 200 or "detailed" in prompt:
            return "large"
        return "small"  # Quick responses ok for simple queries
    
    return "large"  # Default to large
```

## Evaluation

The system is evaluated on:

1. **Task Accuracy**: Average score across three tasks
   - GraphDev: Fraction of correct (path, weight) pairs
   - MMLU: Exact match accuracy
   - InfoBench: GPT-5-nano evaluation (handled by staff)

2. **Throughput**: Requests processed in 20 minutes

3. **Pareto Efficiency**: Distance to empirical Pareto frontier

## Monitoring

### View Modal Logs

```bash
modal app logs YOUR_ANDREW_ID-system-1
```

### Check System Stats

The `/completions` endpoint returns metadata including:
- `elapsed_time`: Time taken for request
- `requests`: Number of prompts processed
- Token usage statistics

## Troubleshooting

### Out of Memory

1. Enable 8-bit quantization: `use_8bit=True`
2. Reduce batch size
3. Use smaller models: Qwen3-4B, Qwen3-1.7B

### Slow Inference

1. Enable batching (already default)
2. Try speculative decoding with `EnhancedInferenceSystem`
3. Use smaller models for non-critical tasks
4. Increase Modal GPU count if available

### Poor Accuracy

1. Use larger models: Qwen3-8B
2. Disable quantization
3. Lower temperature for structured outputs
4. Check routing logic is working correctly

### Modal Deployment Issues

```bash
# Re-authenticate
modal token new

# Check deployment status
modal app list

# View recent logs
modal app logs YOUR_ANDREW_ID-system-1 --tail

# Redeploy
./deploy.sh
```

## Performance Optimization Tips

### 1. Batch Efficiently

Group similar requests together for better GPU utilization:
```python
# Good: batch similar tasks
graph_prompts = [...]
results = system.process_batch(graph_prompts)

# Less efficient: mixed tasks
mixed_prompts = [graph1, mmlu1, graph2, mmlu2]
```

### 2. Temperature Tuning

- Graph tasks: 0.1-0.3 (need precision)
- MMLU: 0.5-0.7 (structured but allow variety)
- InfoBench: 0.7-0.9 (creative responses)

### 3. Model Selection

Use the smallest model that maintains 80%+ baseline accuracy:
- Easy tasks → Qwen3-0.6B or Qwen3-1.7B
- Medium tasks → Qwen3-4B
- Hard tasks → Qwen3-8B

### 4. Quantization Trade-offs

- 8-bit: ~2x speed, minimal accuracy loss
- 4-bit: ~4x speed, noticeable accuracy loss
- Full precision: Best accuracy, slowest

## References

- [Qwen3 Models](https://huggingface.co/Qwen)
- [Modal Documentation](https://modal.com/docs)
- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
- Course Homeworks 1-3 implementations

## Contact

For issues or questions:
- Check Modal logs first
- Review this README
- Contact course staff via Piazza

---

**Note**: Remember to stop your Modal app when not in use to avoid unnecessary charges:
```bash
modal app stop YOUR_ANDREW_ID-system-1
```
