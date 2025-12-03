#!/usr/bin/env python3
"""
Complete evaluation pipeline:
1. Load batch_arrivals.json
2. Send requests to Modal API
3. Generate simulation_summary.json
4. Transform to student_outputs.jsonl
5. Generate combined_dataset.jsonl (if needed)
6. Evaluate and produce metrics
"""

import json
import asyncio
import aiohttp
import time
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

from grader import InfoBenchEvaluator, evaluate_single
from dotenv import load_dotenv

# Load .env
load_dotenv()


# ============================================================================
# STEP 1: Send Requests to API
# ============================================================================

async def send_batch_request(
    session: aiohttp.ClientSession,
    batch: Dict[str, Any],
    url: str,
    start_time: float,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Send a single batch request at its scheduled arrival time"""
    current_elapsed = time.time() - start_time
    wait_time = max(0, batch['arrival_time'] - current_elapsed)
    
    if wait_time > 0:
        await asyncio.sleep(wait_time)
    
    async with semaphore:
        actual_send_time = time.time() - start_time
        
        print(f"[t={actual_send_time:.2f}s] Sending batch {batch['batch_id']} "
              f"(size: {batch['batch_size']})")
        
        request_start = time.time()
        try:
            async with session.post(
                url,
                json={
                    "prompt": batch['prompts'],
                    "max_tokens": batch.get('max_length', 2048),
                },
                timeout=aiohttp.ClientTimeout(total=600)
            ) as response:
                response_data = await response.json()
                request_duration = time.time() - request_start
                
                result = {
                    "batch_id": batch['batch_id'],
                    "batch_size": batch['batch_size'],
                    "scheduled_arrival_time": batch['arrival_time'],
                    "actual_send_time": actual_send_time,
                    "request_duration": request_duration,
                    "completion_time": time.time() - start_time,
                    "status_code": response.status,
                    "prompt_idxs": batch.get('prompt_idxs', []),
                    "response": response_data if response.status == 200 else None,
                    "error": None if response.status == 200 else f"HTTP {response.status}"
                }
                
                print(f"[t={result['completion_time']:.2f}s] Completed batch {batch['batch_id']} "
                      f"(duration: {request_duration:.2f}s)")
                
                return result
                
        except Exception as e:
            request_duration = time.time() - request_start
            result = {
                "batch_id": batch['batch_id'],
                "batch_size": batch['batch_size'],
                "scheduled_arrival_time": batch['arrival_time'],
                "actual_send_time": actual_send_time,
                "request_duration": request_duration,
                "completion_time": time.time() - start_time,
                "status_code": None,
                "prompt_idxs": batch.get('prompt_idxs', []),
                "response": None,
                "error": str(e)
            }
            print(f"[t={result['completion_time']:.2f}s] ERROR batch {batch['batch_id']}: {e}")
            return result


async def run_api_simulation(batches: List[Dict[str, Any]], url: str, max_concurrent: int = 300) -> List[Dict[str, Any]]:
    print(f"🚀 API simulation with {len(batches)} batches")
    print(f"Total prompts: {sum(b['batch_size'] for b in batches)}")
    print(f"Max concurrent requests: {max_concurrent}")
    
    start_time = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent * 10)
    
    # ✅ Set timeout at session level with all components
    timeout = aiohttp.ClientTimeout(
        total=600,      # Total request timeout
        connect=60,     # Connection timeout
        sock_read=600,  # Socket read timeout - THIS IS KEY!
        sock_connect=60 # Socket connect timeout
    )
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout  # Apply to all requests from this session
    ) as session:
        tasks = [send_batch_request(session, batch, url, start_time, semaphore) 
                 for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Exception in batch {i}: {result}")
            processed_results.append({
                "batch_id": batches[i]['batch_id'],
                "error": str(result)
            })
        else:
            processed_results.append(result)
    
    print(f"\n{'='*60}")
    print(f"API simulation complete!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Successful: {sum(1 for r in processed_results if r.get('status_code') == 200)}/{len(batches)}")
    print(f"{'='*60}\n")
    
    return processed_results


# ============================================================================
# STEP 2: Transform to student_outputs.jsonl
# ============================================================================

def transform_to_student_outputs(simulation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform simulation results to student outputs format"""
    print("Transforming to student_outputs.jsonl...")
    
    outputs = []
    
    for batch in simulation_results:
        if batch.get("status_code") != 200 or not batch.get("response"):
            print(f"Skipping failed batch ID: {batch.get('batch_id')}")
            continue
        
        prompt_idxs = batch.get("prompt_idxs", [])
        choices = batch.get("response", {}).get("choices", [])
        
        if len(prompt_idxs) != len(choices):
            print(f"Warning: Mismatch in batch {batch['batch_id']}")
            continue
        
        for idx, choice in zip(prompt_idxs, choices):
            outputs.append({
                "index": idx,
                "output": choice.get("text", "")
            })
    
    print(f"Generated {len(outputs)} student outputs")
    return outputs


# ============================================================================
# STEP 3: Evaluate
# ============================================================================

def calculate_metrics(results: list, student_id: str) -> dict:
    """Calculate task-wise and overall metrics"""
    task_scores = {"mmlu_med": [], "graph": [], "infobench": []}
    
    for r in results:
        task = r["task"]
        if task in task_scores:
            task_scores[task].append(r["score"])
    
    metrics = {
        "student_id": student_id,
        "total_examples": len(results),
        "task_metrics": {},
        "overall_accuracy": 0.0
    }
    
    all_scores = []
    for task, scores in task_scores.items():
        if scores:
            metrics["task_metrics"][task] = {
                "count": len(scores),
                "accuracy": sum(scores) / len(scores),
                "total_score": sum(scores)
            }
            all_scores.extend(scores)
    
    if all_scores:
        metrics["overall_accuracy"] = sum(all_scores) / len(all_scores)
    
    return metrics


def run_evaluation(
    student_outputs: Dict[int, str],
    combined_dataset: List[Dict[str, Any]],
    infobench_evaluator: InfoBenchEvaluator
) -> tuple:
    """Run evaluation on all examples"""
    print("Running evaluation...")
    
    results = []
    
    for idx, test_item in enumerate(tqdm(combined_dataset, desc="Evaluating")):
        index = test_item["index"]
        student_response = student_outputs.get(index, "")
        result = evaluate_single(idx, test_item, student_response, infobench_evaluator)
        results.append(result)
    
    metrics = calculate_metrics(results, "test_student")
    
    return metrics, results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run complete evaluation pipeline")
    parser.add_argument("--url", type=str, required=True, help="Modal API endpoint URL")
    parser.add_argument("--batch_arrivals", type=str, default="batch_arrivals.json",
                       help="Path to batch_arrivals.json")
    parser.add_argument("--combined_dataset", type=str, default="combined_dataset.jsonl",
                       help="Path to combined_dataset.jsonl")
    parser.add_argument("--max_concurrent", type=int, default=300,
                       help="Max concurrent requests")
    
    args = parser.parse_args()
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    print("="*60)
    print("EVALUATION PIPELINE")
    print("="*60)
    
    # Step 1: Load batch arrivals
    print(f"\n[1/6] Loading {args.batch_arrivals}...")
    with open(args.batch_arrivals, "r") as f:
        batches = json.load(f)
    print(f"  Loaded {len(batches)} batches")
    
    # Step 2: Run API simulation
    print(f"\n[2/6] Running API simulation...")
    simulation_results = asyncio.run(run_api_simulation(batches, args.url, args.max_concurrent))
    
    # Save simulation summary
    with open("simulation_summary.json", "w") as f:
        json.dump({"results": simulation_results}, f, indent=2)
    print(f"  ✓ Saved simulation_summary.json")
    
    # Step 3: Transform to student outputs
    print(f"\n[3/6] Transforming to student_outputs.jsonl...")
    student_outputs_list = transform_to_student_outputs(simulation_results)
    
    with open("student_outputs.jsonl", "w") as f:
        for item in student_outputs_list:
            f.write(json.dumps(item) + "\n")
    print(f"  ✓ Saved student_outputs.jsonl ({len(student_outputs_list)} outputs)")
    
    # Step 4: Load/generate combined dataset
    print(f"\n[4/6] Loading {args.combined_dataset}...")
    if not Path(args.combined_dataset).exists():
        print(f"  {args.combined_dataset} not found, generating...")
        from generate_combined_dataset import main as generate_dataset
        generate_dataset()
    
    combined_dataset = []
    with open(args.combined_dataset, "r") as f:
        for line in f:
            if line.strip():
                combined_dataset.append(json.loads(line))
    print(f"  Loaded {len(combined_dataset)} examples")
    
    # Step 5: Prepare student outputs dict
    print(f"\n[5/6] Preparing student outputs...")
    student_outputs_dict = {item["index"]: item["output"] for item in student_outputs_list}
    
    # Step 6: Evaluate
    print(f"\n[6/6] Evaluating...")
    infobench_evaluator = InfoBenchEvaluator(openai_key)
    metrics, results = run_evaluation(student_outputs_dict, combined_dataset, infobench_evaluator)
    
    def sanitize_for_json(obj):
        """Recursively replace ellipsis with None"""
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        elif obj is ...:
            return None
        return obj

# Use it like:
    with open("test_student_results.jsonl", "w") as f:
        for result in results:
            clean_result = sanitize_for_json(result)
            f.write(json.dumps(clean_result) + "\n")

    # Save results
    # with open("test_student_results.jsonl", "w") as f:
    #     for result in results:
    #         f.write(json.dumps(result) + "\n")
    print(f"  ✓ Saved test_student_results.jsonl")
    
    with open("test_student_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Saved test_student_metrics.json")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Total examples: {metrics['total_examples']}")
    for task, task_metrics in metrics["task_metrics"].items():
        print(f"{task:12s}: {task_metrics['accuracy']:.4f} ({task_metrics['count']} examples)")
    print(f"{'Overall':12s}: {metrics['overall_accuracy']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()