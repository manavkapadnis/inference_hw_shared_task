#!/usr/bin/env python3
"""
Evaluate Modal API endpoint with batch arrivals simulation
"""

import json
import asyncio
import aiohttp
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

from grader import InfoBenchEvaluator, evaluate_single


async def send_batch_request(
    session: aiohttp.ClientSession,
    batch: Dict[str, Any],
    url: str,
    start_time: float,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Send a single batch request at its scheduled arrival time."""
    current_elapsed = time.time() - start_time
    wait_time = max(0, batch['arrival_time'] - current_elapsed)
    
    if wait_time > 0:
        await asyncio.sleep(wait_time)
    
    async with semaphore:
        actual_send_time = time.time() - start_time
        
        print(f"[t={actual_send_time:.2f}s] Sending batch {batch['batch_id']} "
              f"(scheduled: {batch['arrival_time']:.2f}s, size: {batch['batch_size']})")
        
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
                      f"(duration: {request_duration:.2f}s, status: {response.status})")
                
                return result
                
        except asyncio.TimeoutError:
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
                "error": "Timeout"
            }
            print(f"[t={result['completion_time']:.2f}s] TIMEOUT batch {batch['batch_id']}")
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


async def run_api_evaluation(
    batches: List[Dict[str, Any]],
    hidden_test: List[Dict[str, Any]],
    url: str,
    max_concurrent_requests: int = 300
) -> List[Dict[str, Any]]:
    """Run batch simulation and collect API responses."""
    print(f"Starting API evaluation with {len(batches)} batches")
    print(f"Total prompts: {sum(b['batch_size'] for b in batches)}")
    print(f"Max concurrent requests: {max_concurrent_requests}\n")
    
    start_time = time.time()
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    connector = aiohttp.TCPConnector(limit=max_concurrent_requests + 10)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            send_batch_request(session, batch, url, start_time, semaphore)
            for batch in batches
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    # Process results
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
    print(f"API requests complete!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Successful: {sum(1 for r in processed_results if r.get('status_code') == 200)}/{len(batches)}")
    print(f"Failed: {sum(1 for r in processed_results if r.get('status_code') != 200)}/{len(batches)}")
    print(f"{'='*60}\n")
    
    return processed_results


def evaluate_results(
    api_results: List[Dict[str, Any]],
    hidden_test: List[Dict[str, Any]],
    infobench_evaluator: InfoBenchEvaluator
) -> Dict[str, Any]:
    """Evaluate API responses against ground truth."""
    print("Evaluating responses...")
    
    # Create index mapping
    index_to_test = {item["index"]: item for item in hidden_test}
    
    eval_results = []
    task_scores = {"mmlu_med": [], "graph": [], "infobench": []}
    
    for batch_result in tqdm(api_results, desc="Evaluating batches"):
        if batch_result.get("status_code") != 200 or not batch_result.get("response"):
            continue
            
        prompt_idxs = batch_result.get("prompt_idxs", [])
        choices = batch_result["response"].get("choices", [])
        
        for prompt_idx, choice in zip(prompt_idxs, choices):
            if prompt_idx not in index_to_test:
                continue
                
            test_item = index_to_test[prompt_idx]
            student_response = choice.get("text", "")
            
            result = evaluate_single(
                idx=prompt_idx,
                test_item=test_item,
                student_response=student_response,
                infobench_evaluator=infobench_evaluator
            )
            
            eval_results.append(result)
            task = result["task"]
            if task in task_scores:
                task_scores[task].append(result["score"])
    
    # Calculate metrics
    metrics = {
        "total_examples": len(eval_results),
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
    
    return metrics, eval_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Modal API endpoint")
    parser.add_argument("--url", type=str, required=True, help="Modal API endpoint URL")
    parser.add_argument("--batch_arrivals", type=str, default="batch_arrivals.json",
                       help="Batch arrivals JSON file")
    parser.add_argument("--hidden_test", type=str, default="combined_dataset.jsonl",
                       help="Hidden test JSONL file")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                       help="Output directory for results")
    parser.add_argument("--max_concurrent", type=int, default=300,
                       help="Max concurrent requests")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    with open(args.batch_arrivals, 'r') as f:
        batches = json.load(f)
    
    hidden_test = []
    with open(args.hidden_test, 'r') as f:
        for line in f:
            if line.strip():
                hidden_test.append(json.loads(line))
    
    print(f"Loaded {len(batches)} batches")
    print(f"Loaded {len(hidden_test)} test examples")
    
    # Initialize InfoBench evaluator
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    infobench_evaluator = InfoBenchEvaluator(openai_key)
    
    # Run API evaluation
    api_results = asyncio.run(run_api_evaluation(
        batches, hidden_test, args.url, args.max_concurrent
    ))
    
    # Evaluate results
    metrics, eval_results = evaluate_results(api_results, hidden_test, infobench_evaluator)
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "api_results.json", 'w') as f:
        json.dump(api_results, f, indent=2)
    
    with open(output_path / "eval_results.jsonl", 'w') as f:
        for result in eval_results:
            f.write(json.dumps(result) + '\n')
    
    with open(output_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total examples: {metrics['total_examples']}")
    for task, task_metrics in metrics["task_metrics"].items():
        print(f"{task:12s}: {task_metrics['accuracy']:.4f} ({task_metrics['count']} examples)")
    print(f"{'Overall':12s}: {metrics['overall_accuracy']:.4f}")
    print("="*60)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()