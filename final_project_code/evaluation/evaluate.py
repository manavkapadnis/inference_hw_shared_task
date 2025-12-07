#!/usr/bin/env python3
"""
Standalone Evaluation Script
Supports: MMLU Medicine, Graph Shortest Paths, InfoBench
Includes: Accuracy metrics + Throughput metrics
Usage: python evaluate.py --output_file student_outputs_manav_system2.jsonl --test_file combined_dataset.jsonl
"""

import json
import os
import re
import time
import argparse
import statistics
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# InfoBench system prompt
SYS_MSG = ("Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. "
           "Your selection should be based on your judgment as well as the following rules:\n\n"
           "- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. "
           "However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. "
           "As an illustration, consider a question that asks, \"Does each sentence in the generated text use a second person?\" "
           "If even one sentence does not use the second person, the answer should NOT be 'YES'. "
           "To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n"
           "- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information "
           "that could be utilized to answer the question. For instance, if the question asks, "
           "\"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence, "
           "it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.''")


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_json(data: Dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# THROUGHPUT METRICS
# =============================================================================

def calculate_throughput_metrics(simulation_file: str) -> Dict[str, Any]:
    """Calculate throughput metrics from simulation_summary.json"""
    if not Path(simulation_file).exists():
        return {"error": f"Simulation file not found: {simulation_file}"}
    
    with open(simulation_file, 'r') as f:
        data = json.load(f)
    
    results = data.get("results", data) if isinstance(data, dict) else data
    
    successful = [r for r in results if r.get("status_code") == 200]
    failed = [r for r in results if r.get("status_code") != 200]
    
    if not successful:
        return {"error": "No successful requests found", "total_batches": len(results), "failed_batches": len(failed)}
    
    # Basic counts
    total_prompts = sum(r.get("batch_size", len(r.get("prompt_idxs", []))) for r in successful)
    total_batches = len(results)
    successful_batches = len(successful)
    
    # Time metrics
    send_times = [r["actual_send_time"] for r in successful if "actual_send_time" in r]
    completion_times = [r["completion_time"] for r in successful if "completion_time" in r]
    request_durations = [r["request_duration"] for r in successful if "request_duration" in r]
    
    first_send = min(send_times) if send_times else 0
    last_completion = max(completion_times) if completion_times else 0
    total_time = last_completion - first_send if last_completion > first_send else sum(request_durations)
    
    # Latency percentiles
    latency_stats = {}
    if request_durations:
        sorted_durations = sorted(request_durations)
        n = len(sorted_durations)
        latency_stats = {
            "min_latency": min(request_durations),
            "max_latency": max(request_durations),
            "avg_latency": statistics.mean(request_durations),
            "median_latency": statistics.median(request_durations),
            "p90_latency": sorted_durations[int(n * 0.90)] if n > 0 else 0,
            "p95_latency": sorted_durations[int(n * 0.95)] if n > 0 else 0,
            "p99_latency": sorted_durations[int(n * 0.99)] if n > 0 else 0,
            "std_latency": statistics.stdev(request_durations) if len(request_durations) > 1 else 0,
        }
    
    # Throughput calculations
    throughput = total_prompts / total_time if total_time > 0 else 0
    batch_throughput = successful_batches / total_time if total_time > 0 else 0
    
    return {
        "total_batches": total_batches,
        "successful_batches": successful_batches,
        "failed_batches": len(failed),
        "success_rate": successful_batches / total_batches if total_batches > 0 else 0,
        "total_prompts": total_prompts,
        "total_time_seconds": round(total_time, 2),
        "throughput_prompts_per_sec": round(throughput, 2),
        "throughput_batches_per_sec": round(batch_throughput, 2),
        **{k: round(v, 3) for k, v in latency_stats.items()},
    }


def print_throughput_metrics(metrics: Dict[str, Any]):
    """Print throughput metrics summary"""
    print("\n" + "=" * 80)
    print("THROUGHPUT METRICS")
    print("=" * 80)
    
    if "error" in metrics:
        print(f"⚠️  {metrics['error']}")
        return
    
    print(f"{'Total Batches':25s}: {metrics['total_batches']}")
    print(f"{'Successful Batches':25s}: {metrics['successful_batches']}")
    print(f"{'Failed Batches':25s}: {metrics['failed_batches']}")
    print(f"{'Success Rate':25s}: {metrics['success_rate']:.2%}")
    print("-" * 80)
    print(f"{'Total Prompts':25s}: {metrics['total_prompts']}")
    print(f"{'Total Time':25s}: {metrics['total_time_seconds']:.2f}s")
    print(f"{'Throughput (prompts/s)':25s}: {metrics['throughput_prompts_per_sec']:.2f}")
    print(f"{'Throughput (batches/s)':25s}: {metrics['throughput_batches_per_sec']:.2f}")
    print("-" * 80)
    if "avg_latency" in metrics:
        print(f"{'Avg Latency':25s}: {metrics['avg_latency']:.3f}s")
        print(f"{'Median Latency':25s}: {metrics['median_latency']:.3f}s")
        print(f"{'P90 Latency':25s}: {metrics['p90_latency']:.3f}s")
        print(f"{'P95 Latency':25s}: {metrics['p95_latency']:.3f}s")
        print(f"{'P99 Latency':25s}: {metrics['p99_latency']:.3f}s")
        print(f"{'Min/Max Latency':25s}: {metrics['min_latency']:.3f}s / {metrics['max_latency']:.3f}s")
    print("=" * 80)


# =============================================================================
# RESPONSE PARSERS
# =============================================================================

class ResponseParser:
    @staticmethod
    def parse_mmlu(response: str) -> Optional[str]:
        if not response:
            return None
        response = response.strip()
        answer_match = re.search(r'The answer is\s*\(?([A-D])\)?', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        boxed_match = re.search(r'\\boxed\{([A-D])\}', response)
        if boxed_match:
            return boxed_match.group(1).upper()
        answer_match2 = re.search(r'Answer:\s*([A-D])', response, re.IGNORECASE)
        if answer_match2:
            return answer_match2.group(1).upper()
        letter_match = re.search(r'\b([A-D])\b', response)
        if letter_match:
            return letter_match.group(1)
        return None

    @staticmethod
    def parse_graph(response: str) -> Optional[Dict[str, Any]]:
        if not response:
            return None
        response = response.strip()
        func_patterns = [
            r'submit_paths\s*\(\s*paths\s*=\s*(\[.*?\])\s*,\s*weights\s*=\s*(\[.*?\])\s*\)',
            r'submit_paths\s*\(\s*weights\s*=\s*(\[.*?\])\s*,\s*paths\s*=\s*(\[.*?\])\s*\)',
        ]
        for i, pattern in enumerate(func_patterns):
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    if i == 0:
                        paths = eval(match.group(1))
                        weights = eval(match.group(2))
                    else:
                        weights = eval(match.group(1))
                        paths = eval(match.group(2))
                    return {"paths": paths, "weights": weights}
                except:
                    continue
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if "paths" in parsed and isinstance(parsed["paths"], list):
                    if parsed["paths"] and isinstance(parsed["paths"][0], dict) and "path" in parsed["paths"][0]:
                        paths = [p["path"] for p in parsed["paths"]]
                        weights = [p["weight"] for p in parsed["paths"]]
                        return {"paths": paths, "weights": weights}
                    elif "weights" in parsed:
                        return {"paths": parsed["paths"], "weights": parsed["weights"]}
        except:
            pass
        return None


# =============================================================================
# EVALUATORS
# =============================================================================

class MMLUEvaluator:
    @staticmethod
    def evaluate(response: str, gold_answer: str) -> Tuple[float, Dict[str, Any]]:
        parsed_answer = ResponseParser.parse_mmlu(response)
        is_correct = parsed_answer is not None and parsed_answer.upper() == gold_answer.upper()
        return (1.0 if is_correct else 0.0, {"parsed_answer": parsed_answer, "gold_answer": gold_answer, "correct": is_correct})


class GraphEvaluator:
    @staticmethod
    def evaluate(response: str, gold_answer: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        parsed = ResponseParser.parse_graph(response)
        if parsed is None:
            return (0.0, {"parsed_paths": None, "gold_paths": gold_answer, "parse_error": True, "matches": 0, "total": len(gold_answer.get("paths", []))})
        gold_paths_set = set()
        for path_info in gold_answer.get("paths", []):
            gold_paths_set.add((tuple(path_info["path"]), path_info["weight"]))
        parsed_paths_set = set()
        for i, path in enumerate(parsed.get("paths", [])):
            if i < len(parsed.get("weights", [])):
                parsed_paths_set.add((tuple(path), parsed["weights"][i]))
        matches = len(gold_paths_set.intersection(parsed_paths_set))
        total = len(gold_paths_set)
        return (matches / total if total > 0 else 0.0, {"parsed_paths": parsed, "gold_paths": gold_answer, "matches": matches, "total": total, "parse_error": False})


class InfoBenchEvaluator:
    def __init__(self, openai_api_key: str, eval_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=openai_api_key)
        self.eval_model = eval_model

    def _parse_yes_no(self, generation: str) -> Optional[bool]:
        generation = generation.strip()
        if generation.lower().startswith("yes"):
            return True
        elif generation.lower().startswith("no"):
            return False
        elif "YES" in generation and "NO" not in generation:
            return True
        elif "YES" not in generation and "NO" in generation:
            return False
        return None

    def evaluate(self, request_i: int, meta: Dict[str, Any], predicted_solution: str) -> Tuple[float, Dict[str, Any]]:
        input_task = meta.get('input', '')
        decomposed_questions = meta.get("decomposed_questions", [])
        if not decomposed_questions:
            return 0.0, {"error": "No decomposed questions found"}
        
        message = []
        bool_results = []
        for i, question in enumerate(decomposed_questions):
            if len(message) == 0:
                if input_task:
                    content = f"{SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{predicted_solution}\"\n\nQuestion:\n{question}\n"
                else:
                    content = f"{SYS_MSG}\n\nGenerated Text:\n\"{predicted_solution}\"\n\nQuestion:\n{question}\n"
            else:
                content = f"{question}\n"
            message.append({"role": "user", "content": content})
            
            result = None
            for attempt in range(3):
                try:
                    completion = self.client.chat.completions.create(model=self.eval_model, messages=message, temperature=1.0)
                    generation = completion.choices[0].message.content
                    message.append({"role": "assistant", "content": generation})
                    result = self._parse_yes_no(generation)
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep((2 ** attempt) + (0.1 * request_i))
                    else:
                        result = None
            bool_results.append(result)
            if result is None:
                bool_results.extend([None] * (len(decomposed_questions) - len(bool_results)))
                break
        
        num_yes = sum(r is True for r in bool_results)
        return num_yes / len(decomposed_questions), {"question_results": bool_results, "total_questions": len(decomposed_questions)}


# =============================================================================
# EVALUATION LOGIC
# =============================================================================

def evaluate_single(test_item: Dict, student_response: str, infobench_evaluator: Optional[InfoBenchEvaluator] = None, idx: int = 0) -> Dict:
    index = test_item["index"]
    task = test_item["task"]
    prompt = test_item["prompt"]
    gold_answer = test_item["gold_answer"]
    meta = test_item.get("meta", {})

    if not student_response:
        score, details = 0.0, {"error": "No response"}
    elif task == "mmlu_med":
        score, details = MMLUEvaluator.evaluate(student_response, gold_answer)
    elif task == "graph":
        score, details = GraphEvaluator.evaluate(student_response, gold_answer)
    elif task == "infobench":
        if infobench_evaluator is None:
            score, details = 0.0, {"error": "InfoBench evaluator not initialized"}
        else:
            score, details = infobench_evaluator.evaluate(idx, meta, student_response)
    else:
        score, details = 0.0, {"error": f"Unknown task {task}"}

    return {"index": index, "task": task, "prompt": prompt, "student_output": student_response, "gold_answer": str(gold_answer)[:200], "score": score, "eval_details": details}


def calculate_accuracy_metrics(results: List[Dict], system_id: str) -> Dict:
    task_scores = {"mmlu_med": [], "graph": [], "infobench": []}
    for r in results:
        if r["task"] in task_scores:
            task_scores[r["task"]].append(r["score"])
    metrics = {"system_id": system_id, "total_examples": len(results), "task_metrics": {}, "overall_accuracy": 0.0}
    all_scores = []
    for task, scores in task_scores.items():
        if scores:
            metrics["task_metrics"][task] = {"count": len(scores), "accuracy": sum(scores) / len(scores), "total_score": sum(scores)}
            all_scores.extend(scores)
    if all_scores:
        metrics["overall_accuracy"] = sum(all_scores) / len(all_scores)
    return metrics


def print_accuracy_metrics(metrics: Dict):
    print("\n" + "=" * 80)
    print(f"ACCURACY METRICS: {metrics['system_id']}")
    print("=" * 80)
    for task in ["mmlu_med", "graph", "infobench"]:
        if task in metrics["task_metrics"]:
            m = metrics["task_metrics"][task]
            print(f"{task:15s}: {m['accuracy']:.4f} ({m['count']} examples)")
    print("-" * 80)
    print(f"{'OVERALL':15s}: {metrics['overall_accuracy']:.4f}")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate outputs with accuracy and throughput metrics")
    parser.add_argument("--output_file", required=True, help="Path to student_outputs.jsonl")
    parser.add_argument("--test_file", required=True, help="Path to combined_dataset.jsonl")
    parser.add_argument("--simulation_file", default="simulation_summary.json", help="Path to simulation_summary.json for throughput metrics (auto-detected if not specified)")
    parser.add_argument("--output_dir", default="./eval_results", help="Output directory")
    parser.add_argument("--system_id", default="system1", help="System identifier")
    args = parser.parse_args()
    
    print("=" * 80)
    print("EVALUATION (Accuracy + Throughput)")
    print("=" * 80)

    # Auto-detect simulation_summary.json if not specified
    if args.simulation_file is None:
        output_path = Path(args.output_file)
        simulation_file = output_path.parent / "simulation_summary.json"
        if simulation_file.exists():
            args.simulation_file = str(simulation_file)
            print(f"✓ Auto-detected simulation file: {args.simulation_file}")
        else:
            # Try current directory as fallback
            args.simulation_file = "simulation_summary.json"
            if Path(args.simulation_file).exists():
                print(f"✓ Found simulation file in current directory: {args.simulation_file}")
            else:
                print(f"⚠️  No simulation_summary.json found - throughput metrics will be unavailable")
    else:
        print(f"Using specified simulation file: {args.simulation_file}")

    # Initialize InfoBench evaluator
    openai_key = os.getenv("OPENAI_API_KEY")
    infobench_evaluator = None
    if openai_key:
        infobench_evaluator = InfoBenchEvaluator(openai_key)
        print("✓ InfoBench evaluator initialized")
    else:
        print("⚠️  OPENAI_API_KEY not set - InfoBench evaluation disabled")

    # Load data
    print(f"\nLoading outputs from: {args.output_file}")
    output_data = load_jsonl(args.output_file)
    outputs_dict = {item["index"]: item.get("output", item.get("text", "")) for item in output_data if "index" in item}
    print(f"✓ Loaded {len(outputs_dict)} outputs")

    print(f"\nLoading test set from: {args.test_file}")
    test_data = load_jsonl(args.test_file)
    print(f"✓ Loaded {len(test_data)} test examples")
    test_dict = {item["index"]: item for item in test_data}

    # Evaluate accuracy
    print("\nEvaluating accuracy...")
    results = []
    for idx, (index, student_response) in enumerate(tqdm(outputs_dict.items(), desc="Evaluating")):
        test_item = test_dict.get(index)
        if test_item is None:
            print(f"⚠️  Warning: No test item found for index {index}")
            continue
        result = evaluate_single(test_item, student_response, infobench_evaluator, idx)
        results.append(result)

    # Calculate metrics
    accuracy_metrics = calculate_accuracy_metrics(results, args.system_id)
    throughput_metrics = calculate_throughput_metrics(args.simulation_file)

    # Combined metrics
    combined_metrics = {
        "system_id": args.system_id,
        "accuracy": accuracy_metrics,
        "throughput": throughput_metrics,
    }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    def sanitize(obj):
        if obj is ... or obj is Ellipsis:
            return None
        elif isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [sanitize(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    results_path = os.path.join(args.output_dir, f"{args.system_id}_results.jsonl")
    with open(results_path, 'w') as f:
        for r in results:
            f.write(json.dumps(sanitize(r)) + '\n')
    
    metrics_path = os.path.join(args.output_dir, f"{args.system_id}_metrics.json")
    save_json(combined_metrics, metrics_path)

    # Print summaries
    print_accuracy_metrics(accuracy_metrics)
    print_throughput_metrics(throughput_metrics)

    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()