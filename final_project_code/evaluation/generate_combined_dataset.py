#!/usr/bin/env python3
"""
Generate combined_dataset.jsonl from batch_arrivals.json
Creates ground truth dataset aligned with request indices
"""

import json
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, Any, List


def process_graph_entry(ex, idx: int) -> Dict[str, Any]:
    """Process graph example"""
    return {
        "index": idx,
        "task": "graph",
        "prompt": ex["prompt"],
        "gold_answer": ex.get("solution"),
        "meta": {
            "graph_params": ex.get("graph_params"),
            "edges": ex.get("edges"),
            "original_id": ex.get("id")
        }
    }


def process_infobench_entry(ex, idx: int) -> Dict[str, Any]:
    """Process InfoBench example"""
    instruction = ex.get("instruction", "")
    input_text = ex.get("input", "")
    
    if input_text:
        prompt = f"Instruction: {instruction}\nQuestion: {input_text}\nGeneration:"
    else:
        prompt = f"Instruction: {instruction}\nGeneration:"
    
    return {
        "index": idx,
        "task": "infobench",
        "prompt": prompt,
        "gold_answer": None,
        "meta": {
            "decomposed_questions": ex.get("decomposed_questions"),
            "category": ex.get("category"),
            "subset": ex.get("subset"),
            "input": ex.get("input"),
            "original_id": ex.get("id")
        }
    }


def process_mmlu_entry(ex, idx: int) -> Dict[str, Any]:
    """Process MMLU example"""
    question = ex.get("question", "")
    choices = ex.get("choices", [])
    subject = ex.get("subject", "medicine")
    
    prompt = (
        f"The following is a multiple choice question (with answers) about {subject}. "
        f"Output the answer in the format of \"The answer is (X)\" at the end.\n\n"
        f"Question: {question}\n Options:\n"
    )
    
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    
    prompt += "Answer:"
    
    answer_idx = ex.get("answer")
    gold_letter = chr(65 + answer_idx) if answer_idx is not None else None
    
    return {
        "index": idx,
        "task": "mmlu_med",
        "prompt": prompt,
        "gold_answer": gold_letter,
        "meta": {
            "subject": ex.get("subject"),
            "answer_index": answer_idx,
            "raw_choices": choices,
            "original_id": ex.get("id")
        }
    }


def find_match(prompt_text: str, graph_ds, infobench_ds, mmlu_ds) -> Dict[str, Any]:
    """Find matching source record for a prompt"""
    
    # Check Graph
    for ex in graph_ds:
        if ex["prompt"].strip() in prompt_text.strip():
            solution = ex.get("solution")
            if isinstance(solution, str):
                try:
                    solution = json.loads(solution)
                except:
                    pass
            
            return {
                "task": "graph",
                "gold_answer": solution,
                "meta": {
                    "graph_params": ex.get("graph_params"),
                    "edges": ex.get("edges"),
                    "original_id": ex.get("id")
                }
            }
    
    # Check MMLU
    for ex in mmlu_ds:
        if ex["question"].strip() in prompt_text:
            answer_idx = ex.get("answer")
            gold_letter = chr(65 + answer_idx) if answer_idx is not None else None
            return {
                "task": "mmlu_med",
                "gold_answer": gold_letter,
                "meta": {
                    "subject": ex.get("subject"),
                    "answer_index": answer_idx,
                    "raw_choices": ex.get("choices"),
                    "original_id": ex.get("id")
                }
            }
    
    # Check InfoBench
    for ex in infobench_ds:
        if ex["instruction"].strip() in prompt_text:
            return {
                "task": "infobench",
                "gold_answer": None,
                "meta": {
                    "decomposed_questions": ex.get("decomposed_questions"),
                    "input": ex.get("input"),
                    "category": ex.get("category"),
                    "subset": ex.get("subset"),
                    "original_id": ex.get("id")
                }
            }
    
    return None


def main():
    print("Loading batch_arrivals.json...")
    with open("batch_arrivals.json", "r") as f:
        batches = json.load(f)
    
    print("Loading source datasets...")
    graph_ds = load_dataset("vashistht/11763_datasets", "graph_dev", split="dev_test")
    infobench_ds = load_dataset("vashistht/11763_datasets", "infobench", split="dev_test")
    mmlu_ds = load_dataset("vashistht/11763_datasets", "mmlu_med", split="dev_test")
    
    print("Matching prompts to ground truth...")
    aligned_data = []
    seen_indices = set()
    
    for batch in tqdm(batches):
        prompts = batch["prompts"]
        indices = batch["prompt_idxs"]
        
        for prompt_text, idx in zip(prompts, indices):
            if idx in seen_indices:
                continue
            
            match = find_match(prompt_text, graph_ds, infobench_ds, mmlu_ds)
            
            if match:
                entry = {
                    "index": idx,
                    "task": match["task"],
                    "prompt": prompt_text,
                    "gold_answer": match["gold_answer"],
                    "meta": match["meta"]
                }
                aligned_data.append(entry)
                seen_indices.add(idx)
            else:
                print(f"WARNING: Could not find match for index {idx}")
    
    # Sort by index
    aligned_data.sort(key=lambda x: x["index"])
    
    print(f"Found matches for {len(aligned_data)} items")
    print("Saving to combined_dataset.jsonl...")
    
    with open("combined_dataset.jsonl", "w") as f:
        for item in aligned_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✓ Saved combined_dataset.jsonl ({len(aligned_data)} examples)")


if __name__ == "__main__":
    main()