"""
Local Evaluation Script
Test the inference system locally before deploying to Modal
"""

import torch
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from inference_system import InferenceSystem
from dataset_handlers import get_handler
import dataset_handlers


def load_test_data(task: str, split: str = "dev_test", limit: int = None) -> List[Dict[str, Any]]:
    """Load test data from HuggingFace"""
    print(f"Loading {task} dataset...")
    
    if task.lower() in ["graph", "graphdev"]:
        raw_dataset = load_dataset("vashistht/11763_datasets", "graph_dev")
    elif task.lower() in ["mmlu", "mmlu_med"]:
        raw_dataset = load_dataset("vashistht/11763_datasets", "mmlu_med")
    elif task.lower() == "infobench":
        raw_dataset = load_dataset("vashistht/11763_datasets", "infobench")
    else:
        raise ValueError(f"Unknown task: {task}")
    
    examples = list(raw_dataset[split])
    
    if limit:
        examples = examples[:limit]
    
    print(f"Loaded {len(examples)} examples")
    return examples


def evaluate_system(
    system: InferenceSystem,
    examples: List[Dict[str, Any]],
    handler,
    batch_size: int = 4,
    output_file: str = None
) -> Dict[str, Any]:
    """Evaluate the inference system on examples"""
    
    import time
    
    results = []
    correct_count = 0
    batch_times = []
    
    print(f"Evaluating {len(examples)} examples...")
    
    total_start = time.time()
    
    # Process in batches
    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i + batch_size]
        
        # Format prompts
        prompts = [handler.format_prompt(ex) for ex in batch]
        
        # Generate responses with timing
        batch_start = time.time()
        responses = system.process_batch(
            prompts, 
            max_tokens=512, 
            temperature=0.7
        )
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Evaluate each response
        for ex, response in zip(batch, responses):
            # Special handling for graph task
            if isinstance(handler, dataset_handlers.GraphHandler):
                # For graph task, LLM should make tool call, we extract params and compute
                parsed = handler.parse_response(response, ex)  # Pass example for fallback
                ground_truth = handler.get_ground_truth(ex)
                score = handler.evaluate(parsed, ground_truth)
            else:
                # For other tasks, parse and evaluate normally
                parsed = handler.parse_response(response)
                ground_truth = handler.get_ground_truth(ex)
                score = handler.evaluate(parsed, ground_truth)
            
            correct_count += score
            
            results.append({
                "example": ex,
                "prompt": handler.format_prompt(ex),
                "response": response,
                "parsed": parsed,
                "ground_truth": ground_truth,
                "score": score
            })
    
    total_time = time.time() - total_start
    
    # Compute metrics
    accuracy = correct_count / len(examples) if examples else 0.0
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0.0
    throughput = len(examples) / total_time if total_time > 0 else 0.0
    
    eval_results = {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(examples),
        "total_time_seconds": total_time,
        "avg_batch_time_seconds": avg_batch_time,
        "throughput_examples_per_second": throughput,
        "results": results,
        "stats": system.get_stats()
    }
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save without individual results (too large)
        summary = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(examples),
            "total_time_seconds": total_time,
            "avg_batch_time_seconds": avg_batch_time,
            "throughput_examples_per_second": throughput,
            "stats": system.get_stats()
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate inference system locally")
    parser.add_argument("--task", type=str, required=True, 
                       choices=["graphdev", "mmlu_med", "infobench"],
                       help="Task to evaluate")
    parser.add_argument("--split", type=str, default="dev_test",
                       help="Dataset split to use")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of examples (for testing)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results")
    parser.add_argument("--large_model", type=str, default="Qwen/Qwen3-8B",
                       help="Large model path")
    parser.add_argument("--small_model", type=str, default="Qwen/Qwen3-1.7B",
                       help="Small model path")
    parser.add_argument("--use_8bit", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--use_4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--use_enhanced", action="store_true",
                       help="Use enhanced inference system with speculative decoding")
    
    args = parser.parse_args()
    
    # Initialize system
    print("Initializing inference system...")
    
    if args.use_enhanced:
        from enhanced_inference import EnhancedInferenceSystem
        system = EnhancedInferenceSystem(
            large_model_path=args.large_model,
            small_model_path=args.small_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_8bit=args.use_8bit,
            use_4bit=args.use_4bit
        )
    else:
        system = InferenceSystem(
            large_model_path=args.large_model,
            small_model_path=args.small_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_8bit=args.use_8bit,
            use_4bit=args.use_4bit
        )
    
    # Load data
    examples = load_test_data(args.task, args.split, args.limit)
    
    # Get handler
    handler = get_handler(args.task)
    
    # Evaluate
    results = evaluate_system(
        system, 
        examples, 
        handler,
        batch_size=args.batch_size,
        output_file=args.output
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Task: {args.task}")
    print(f"Examples: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"\nTiming Metrics:")
    print(f"  Total Time: {results['total_time_seconds']:.2f}s")
    print(f"  Avg Batch Time: {results['avg_batch_time_seconds']:.4f}s")
    print(f"  Throughput: {results['throughput_examples_per_second']:.2f} examples/s")
    print(f"\nSystem Stats:")
    print(f"  Total Requests: {results['stats']['total_requests']}")
    print(f"  Total Tokens: {results['stats']['total_tokens']}")
    print(f"  Avg Tokens/Request: {results['stats']['avg_tokens_per_request']:.2f}")
    print("="*50)


if __name__ == "__main__":
    main()