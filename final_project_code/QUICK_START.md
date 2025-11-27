# Final Project Inference System

## Setup

### 1. Create .env file

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key for InfoBench evaluation
```

### 2. Submit All Jobs

```bash
# Make scripts executable
chmod +x *.sh

# Submit all 9 evaluation jobs
sbatch system1_graphdev.sh
sbatch system1_mmlu.sh
sbatch system1_infobench.sh
sbatch system2_graphdev.sh
sbatch system2_mmlu.sh
sbatch system2_infobench.sh
sbatch system3_graphdev.sh
sbatch system3_mmlu.sh
sbatch system3_infobench.sh
```

## How It Works

### GraphDev Task
- LLM receives graph problem as text prompt
- LLM makes tool call: `find_shortest_paths(edges=[[...]], N=X, P=Y)`
- System extracts parameters using regex
- Calls pathfinding function with extracted parameters
- Compares computed paths with ground truth

### MMLU Medicine
- Standard multiple choice evaluation
- Exact match on answer (A/B/C/D)

### InfoBench  
- Uses GPT-5-nano for evaluation
- Loads API key from .env file
- Returns ratio of yes/no answers

## Results

Results saved to `results/` directory:
- `systemX_TASK.json` - Contains accuracy, timing, throughput
- Each JSON has:
  - `accuracy` - Task score
  - `total_time_seconds` - Inference time
  - `throughput_examples_per_second` - Examples/sec

## System Configurations

**System 1:** Qwen3-8B + Qwen3-4B (full precision, batch=4)
**System 2:** Qwen3-4B + Qwen3-0.6B (8-bit, batch=8)  
**System 3:** Qwen3-8B + Qwen3-1.7B (8-bit, batch=6)

## Check Status

```bash
squeue -u mkapadni
tail -f sys1_graphdev.out
```