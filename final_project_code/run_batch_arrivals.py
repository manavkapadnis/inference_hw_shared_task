#!/usr/bin/env python3
import json
import requests
import time
import argparse
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


ENDPOINT = "https://lbaghel--mkapadni-system-1-model-completions.modal.run"


def load_batch_arrivals(file_path: str) -> List[Dict[str, Any]]:
    """Load batch arrivals from JSON file"""
    print(f"Loading batch arrivals from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        batches = data
    elif isinstance(data, dict) and 'batches' in data:
        batches = data['batches']
    else:
        raise ValueError(f"Unexpected format in {file_path}")

    # Sort by arrival_time
    batches.sort(key=lambda x: x.get('arrival_time', 0))
    print(f"✓ Loaded {len(batches)} batches")
    return batches


async def send_batch_request_async(
    session: aiohttp.ClientSession,
    batch: Dict[str, Any],
    batch_idx: int,
    endpoint: str,
    output_file: str,
    start_time: float,
) -> Dict[str, Any]:
    """Send a batch request to the API asynchronously and save results immediately"""
    if 'prompts' in batch:
        prompts = batch['prompts']
    elif 'prompt' in batch:
        prompts = [batch['prompt']]
    else:
        raise ValueError(f"No prompts found in batch {batch_idx}")

    if 'prompt_idxs' in batch:
        prompt_idxs = batch['prompt_idxs']
    elif 'indices' in batch:
        prompt_idxs = batch['indices']
    else:
        prompt_idxs = list(range(len(prompts)))

    payload = {
        "prompt": prompts,
        # pass indices so server can attach them to errors if needed
        "indices": prompt_idxs,
    }

    # Record actual send time
    actual_send_time = time.time() - start_time
    request_start = time.time()

    try:
        async with session.post(
            endpoint,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as response:
            request_duration = time.time() - request_start
            completion_time = time.time() - start_time
            
            if response.status != 200:
                text = await response.text()
                error_msg = f"HTTP {response.status}: {text[:200]}"
                
                # Save errors immediately for all prompts in this batch
                with open(output_file, 'a') as f:
                    for idx in prompt_idxs:
                        line = {
                            "index": idx,
                            "output": f"[ERROR] {error_msg}",
                        }
                        f.write(json.dumps(line) + "\n")
                
                return {
                    "batch_id": batch_idx,
                    "batch_size": len(prompts),
                    "scheduled_arrival_time": batch.get('arrival_time', 0),
                    "actual_send_time": actual_send_time,
                    "request_duration": request_duration,
                    "completion_time": completion_time,
                    "status_code": response.status,
                    "prompt_idxs": prompt_idxs,
                    "response": None,
                    "error": error_msg,
                }

            result = await response.json()

            if "choices" not in result:
                error_msg = "Invalid response format (no 'choices')"
                
                # Save errors immediately
                with open(output_file, 'a') as f:
                    for idx in prompt_idxs:
                        line = {
                            "index": idx,
                            "output": f"[ERROR] {error_msg}",
                        }
                        f.write(json.dumps(line) + "\n")
                
                return {
                    "batch_id": batch_idx,
                    "batch_size": len(prompts),
                    "scheduled_arrival_time": batch.get('arrival_time', 0),
                    "actual_send_time": actual_send_time,
                    "request_duration": request_duration,
                    "completion_time": completion_time,
                    "status_code": response.status,
                    "prompt_idxs": prompt_idxs,
                    "response": None,
                    "error": error_msg,
                }

            answers = [choice.get("text", "") for choice in result["choices"]]

            # Save successful answers immediately
            with open(output_file, 'a') as f:
                for idx, ans in zip(prompt_idxs, answers):
                    line = {"index": idx, "output": ans}
                    f.write(json.dumps(line) + "\n")

            # Handle endpoint-specific errors (if any)
            endpoint_errors = []
            for err in result.get("errors", []):
                idx = err.get("index")
                if idx is not None:
                    error_line = {
                        "index": idx,
                        "output": f"[ERROR] {err.get('error', 'unknown error from endpoint')}",
                    }
                    # Save endpoint errors immediately
                    with open(output_file, 'a') as f:
                        f.write(json.dumps(error_line) + "\n")
                    endpoint_errors.append(err)

            return {
                "batch_id": batch_idx,
                "batch_size": len(prompts),
                "scheduled_arrival_time": batch.get('arrival_time', 0),
                "actual_send_time": actual_send_time,
                "request_duration": request_duration,
                "completion_time": completion_time,
                "status_code": response.status,
                "prompt_idxs": prompt_idxs,
                "response": result,
                "error": None,
            }

    except asyncio.TimeoutError:
        request_duration = time.time() - request_start
        completion_time = time.time() - start_time
        error_msg = "Request timeout (client-side asyncio.TimeoutError)"
        
        # Save timeout errors immediately
        with open(output_file, 'a') as f:
            for idx in prompt_idxs:
                line = {
                    "index": idx,
                    "output": f"[ERROR] {error_msg}",
                }
                f.write(json.dumps(line) + "\n")
        
        return {
            "batch_id": batch_idx,
            "batch_size": len(prompts),
            "scheduled_arrival_time": batch.get('arrival_time', 0),
            "actual_send_time": actual_send_time,
            "request_duration": request_duration,
            "completion_time": completion_time,
            "status_code": None,
            "prompt_idxs": prompt_idxs,
            "response": None,
            "error": error_msg,
        }
    except Exception as e:
        request_duration = time.time() - request_start
        completion_time = time.time() - start_time
        error_msg = str(e)
        
        # Save exception errors immediately
        with open(output_file, 'a') as f:
            for idx in prompt_idxs:
                line = {
                    "index": idx,
                    "output": f"[ERROR] {error_msg}",
                }
                f.write(json.dumps(line) + "\n")
        
        return {
            "batch_id": batch_idx,
            "batch_size": len(prompts),
            "scheduled_arrival_time": batch.get('arrival_time', 0),
            "actual_send_time": actual_send_time,
            "request_duration": request_duration,
            "completion_time": completion_time,
            "status_code": None,
            "prompt_idxs": prompt_idxs,
            "response": None,
            "error": error_msg,
        }


async def process_batches_with_arrival_times(
    batches: List[Dict[str, Any]],
    output_file: str,
    endpoint: str,
):
    """Process batches respecting their arrival_time"""
    print(f"\nProcessing {len(batches)} batches with arrival time scheduling...")
    print(f"Endpoint: {endpoint}")
    print(f"Output: {output_file}\n")

    # Truncate output file at start
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")  # empty file

    successful_batches = 0
    failed_batches = 0
    total_outputs = 0
    all_batch_results = []
    tasks = []

    start_time = time.time()
    pbar = tqdm(total=len(batches), desc="Dispatching batches")

    async with aiohttp.ClientSession() as session:
        for idx, batch in enumerate(batches):
            arrival_time = batch.get("arrival_time", 0)
            elapsed = time.time() - start_time
            wait_time = arrival_time - elapsed

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Dispatch request (non-blocking)
            task = asyncio.create_task(
                send_batch_request_async(session, batch, idx, endpoint, output_file, start_time)
            )
            tasks.append(task)
            pbar.update(1)

        pbar.close()
        print("\nWaiting for all responses...")

        # Process tasks as they finish
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Receiving responses"):
            result = await coro
            all_batch_results.append(result)

            if result["status_code"] == 200:
                successful_batches += 1
                total_outputs += len(result["prompt_idxs"])
            else:
                failed_batches += 1
                total_outputs += len(result["prompt_idxs"])

    elapsed = time.time() - start_time

    # Calculate scheduled duration (last batch arrival time)
    scheduled_duration = batches[-1].get('arrival_time', 0) if batches else 0

    # Generate simulation summary
    summary = {
        "total_batches": len(batches),
        "total_prompts": sum(b.get('batch_size', len(b.get('prompts', []))) for b in batches),
        "scheduled_duration": scheduled_duration,
        "actual_duration": elapsed,
        "successful_batches": successful_batches,
        "failed_batches": failed_batches,
        "results": all_batch_results
    }

    # Save simulation summary
    summary_path = out_path.parent / "simulation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total batches: {len(batches)}")
    print(f"Successful batches: {successful_batches}")
    print(f"Failed batches: {failed_batches}")
    print(f"Total outputs (including error lines): {total_outputs}")
    print(f"Scheduled duration: {scheduled_duration:.1f}s")
    print(f"Actual duration: {elapsed:.1f}s")
    print(f"Throughput: {total_outputs/elapsed:.2f} outputs/s")
    print(f"\n✓ Results saved to: {output_file}")
    print(f"✓ Summary saved to: {summary_path}")

    if failed_batches > 0:
        failed_results = [r for r in all_batch_results if r['status_code'] != 200]
        print(f"\n⚠️ {failed_batches} batches failed (logged as [ERROR] in outputs):")
        for fail in failed_results[:5]:
            print(f"  Batch {fail['batch_id']}: {fail['error']}")

    return total_outputs, all_batch_results


def main():
    parser = argparse.ArgumentParser(
        description="Process batch_arrivals.json with arrival time scheduling"
    )
    parser.add_argument(
        "--batch-file",
        type=str,
        default="/home/mkapadni/work/inference_algo/homework4/attempt_1/evaluation/batch_arrivals.json",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/home/mkapadni/work/inference_algo/homework4/attempt_1/evaluation/student_outputs_manav.jsonl",
    )
    parser.add_argument("--endpoint", type=str, default=ENDPOINT)
    args = parser.parse_args()

    print("=" * 80)
    print("BATCH PROCESSING (Arrival Time Mode)")
    print("=" * 80)
    print(f"Endpoint: {args.endpoint}")
    print(f"Batch file: {args.batch_file}")
    print(f"Output file: {args.output_file}")
    print("=" * 80)

    if not Path(args.batch_file).exists():
        print(f"\n❌ Error: {args.batch_file} not found!")
        return 1

    try:
        batches = load_batch_arrivals(args.batch_file)
        total_outputs, batch_results = asyncio.run(
            process_batches_with_arrival_times(
                batches, args.output_file, args.endpoint
            )
        )
        return 0 if all(r['status_code'] == 200 for r in batch_results) else 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())