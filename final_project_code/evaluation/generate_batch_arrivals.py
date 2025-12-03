#!/usr/bin/env python3
"""
Generate batch_arrivals.json for evaluation
"""

import json
import numpy as np
from datasets import load_dataset
from typing import List, Dict, Any


def format_mmlu_prompt(example: Dict[str, Any]) -> str:
    """Format MMLU question as prompt"""
    question = example.get("question", "")
    choices = example.get("choices", [])
    subject = example.get("subject", "medicine")
    
    prompt = (
        f"The following is a multiple choice question (with answers) about {subject}. "
        f"Output the answer in the format of \"The answer is (X)\" at the end.\n\n"
        f"Question: {question}\n Options:\n"
    )
    
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    
    prompt += "Answer:"
    return prompt


def format_infobench_prompt(example: Dict[str, Any]) -> str:
    """Format InfoBench as prompt"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    
    if input_text:
        return f"Instruction: {instruction}\nQuestion: {input_text}\nGeneration:"
    else:
        return f"Instruction: {instruction}\nGeneration:"


def load_prompts() -> List[str]:
    """Load prompts from all datasets"""
    print("Loading datasets...")
    
    graph_ds = load_dataset("vashistht/11763_datasets", "graph_dev", split="dev_test")
    infobench_ds = load_dataset("vashistht/11763_datasets", "infobench", split="dev_test")
    mmlu_ds = load_dataset("vashistht/11763_datasets", "mmlu_med", split="dev_test")
    
    prompts = []
    
    # Graph prompts
    for ex in graph_ds:
        prompts.append(ex["prompt"])
    
    # InfoBench prompts
    # for ex in infobench_ds:
    #     prompts.append(format_infobench_prompt(ex))
    
    # MMLU prompts
    for ex in mmlu_ds:
        prompts.append(format_mmlu_prompt(ex))
    
    print(f"Loaded {len(prompts)} total prompts")
    return prompts


def simulate_batch_arrivals(
    prompts: List[str],
    total_time_seconds: float = 600.0,
    mean_batch_size: float = 3.0,
    min_batch_size: int = 1,
    max_batch_size: int = 8,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """Simulate Poisson batch arrivals ensuring all prompts are used"""
    
    np.random.seed(random_seed)
    
    # Shuffle prompt indices
    remaining_idxs = list(range(len(prompts)))
    np.random.shuffle(remaining_idxs)
    
    # Estimate arrival rate
    estimated_batches = int(np.ceil(len(prompts) / mean_batch_size))
    mean_arrival_rate = estimated_batches / (total_time_seconds * 0.95)
    
    batches = []
    batch_id = 0
    current_time = 0.0
    
    while remaining_idxs:
        # Generate inter-arrival time
        if len(batches) == 0:
            inter_arrival_time = np.random.exponential(0.5 / mean_arrival_rate)
        else:
            inter_arrival_time = np.random.exponential(1.0 / mean_arrival_rate)
        
        current_time += inter_arrival_time
        
        # Determine batch size (Poisson)
        batch_size = np.random.poisson(mean_batch_size)
        batch_size = max(min_batch_size, min(batch_size, max_batch_size))
        batch_size = min(batch_size, len(remaining_idxs))
        
        if batch_size == 0:
            continue
        
        # Extract prompts for this batch
        batch_idxs = remaining_idxs[:batch_size]
        remaining_idxs = remaining_idxs[batch_size:]
        batch_prompts = [prompts[idx] for idx in batch_idxs]
        
        batches.append({
            "batch_id": batch_id,
            "arrival_time": current_time,
            "batch_size": batch_size,
            "prompt_idxs": batch_idxs,
            "prompts": batch_prompts,
            "max_length": 2048
        })
        
        batch_id += 1
    
    print(f"Generated {len(batches)} batches over {current_time:.2f}s")
    return batches


def main():
    print("Generating batch_arrivals.json...")
    
    # Load prompts
    prompts = load_prompts()
    
    # Generate batches
    batches = simulate_batch_arrivals(prompts)
    
    # Save to file
    with open("batch_arrivals.json", "w") as f:
        json.dump(batches, f, indent=2)
    
    print(f"✓ Saved batch_arrivals.json ({len(batches)} batches, {len(prompts)} prompts)")


if __name__ == "__main__":
    main()