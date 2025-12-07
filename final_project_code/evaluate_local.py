"""
Local Evaluation Script
Test the inference system locally before deploying to Modal
"""

import torch
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv
from collections import defaultdict
import heapq

load_dotenv()


# ============================================================================
# Graph Solver (same as in modal_deploy_api.py)
# ============================================================================

class DijkstraSolver:
    """Pure algorithmic solver using modified Dijkstra for K-shortest paths."""
    
    @staticmethod
    def dijkstra_k_shortest_paths(
        num_nodes: int,
        edges: List[List[int]],
        K: int,
        source: int = 0,
        target: int = None
    ) -> List[Tuple[List[int], int]]:
        """Find K shortest simple paths using modified Dijkstra"""
        if target is None:
            target = num_nodes - 1
        
        graph = defaultdict(list)
        for edge in edges:
            src, dst, weight = edge[0], edge[1], edge[2]
            graph[src].append((dst, weight))
        
        pq = [(0.0, tuple([source]))]
        found_paths = []
        seen_paths = set()
        
        while pq and len(found_paths) < K:
            cost, path_tuple = heapq.heappop(pq)
            path = list(path_tuple)
            
            path_signature = (path_tuple, int(cost))
            if path_signature in seen_paths:
                continue
            seen_paths.add(path_signature)
            
            current_node = path[-1]
            
            if current_node == target:
                found_paths.append((path, int(cost)))
                continue
            
            for neighbor, edge_weight in graph[current_node]:
                if neighbor not in path:
                    new_cost = cost + edge_weight
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (new_cost, tuple(new_path)))
        
        return found_paths[:K]
    
    @staticmethod
    def format_solution(paths: List[Tuple[List[int], int]]) -> str:
        """Format as submit_paths() - REQUIRED FORMAT FOR GRADER"""
        if not paths:
            return "submit_paths(paths=[], weights=[])"
        
        paths_list = [p for p, _ in paths]
        weights_list = [w for _, w in paths]
        
        return f"submit_paths(paths={paths_list}, weights={weights_list})"


def extract_graph_params_from_prompt(prompt: str) -> Dict:
    """Extract graph parameters using regex patterns."""
    try:
        nodes_match = re.search(r'graph\s+with\s+(\d+)\s+nodes?', prompt, re.IGNORECASE)
        if not nodes_match:
            return None
        num_nodes = int(nodes_match.group(1))
        
        k_match = re.search(r'top[-\s]?(\d+)|(\d+)\s+shortest', prompt, re.IGNORECASE)
        if k_match:
            K = int(k_match.group(1) or k_match.group(2))
        else:
            K = 1
        
        edges = []
        for match in re.finditer(r'(\d+)\s*-+>\s*(\d+),?\s*weight:?\s*(\d+)', prompt, re.IGNORECASE):
            src = int(match.group(1))
            dst = int(match.group(2))
            weight = int(match.group(3))
            edges.append([src, dst, weight])
        
        if not edges:
            for match in re.finditer(r'[\(\[](\d+),\s*(\d+),\s*(\d+)[\)\]]', prompt):
                src = int(match.group(1))
                dst = int(match.group(2))
                weight = int(match.group(3))
                edges.append([src, dst, weight])
        
        if not edges:
            return None
        
        return {'num_nodes': num_nodes, 'edges': edges, 'K': K}
    except:
        return None


def solve_graph_problem(prompt: str) -> str:
    """Solve graph problem directly"""
    parsed = extract_graph_params_from_prompt(prompt)
    
    if parsed is None:
        return "submit_paths(paths=[], weights=[])"
    
    try:
        paths = DijkstraSolver.dijkstra_k_shortest_paths(
            num_nodes=parsed['num_nodes'],
            edges=parsed['edges'],
            K=parsed['K']
        )
        return DijkstraSolver.format_solution(paths)
    except:
        return "submit_paths(paths=[], weights=[])"


# ============================================================================
# Dataset Handlers
# ============================================================================

class GraphHandler:
    """Handler for graph tasks"""
    
    def format_prompt(self, example: Dict) -> str:
        return example.get('prompt', '')
    
    def parse_response(self, response: str, example: Dict = None) -> Dict:
        """Parse response - returns dict with paths and weights"""
        if not response:
            return {"paths": [], "weights": []}
        
        # Parse submit_paths format
        func_match = re.search(
            r'submit_paths\s*\(\s*paths\s*=\s*(\[.*?\])\s*,\s*weights\s*=\s*(\[.*?\])\s*\)',
            response, re.DOTALL
        )
        if func_match:
            try:
                paths = eval(func_match.group(1))
                weights = eval(func_match.group(2))
                return {"paths": paths, "weights": weights}
            except:
                pass
        
        return {"paths": [], "weights": []}
    
    def get_ground_truth(self, example: Dict) -> Dict:
        """Get ground truth from example"""
        solution = example.get('solution', {})
        if isinstance(solution, str):
            try:
                solution = json.loads(solution)
            except:
                solution = {}
        
        if not isinstance(solution, dict):
            return {"paths": [], "weights": []}
        
        paths = []
        weights = []
        for path_obj in solution.get('paths', []):
            if isinstance(path_obj, dict):
                paths.append(path_obj.get('path', []))
                weights.append(path_obj.get('weight', 0))
        
        return {"paths": paths, "weights": weights}
    
    def evaluate(self, parsed: Dict, ground_truth: Dict) -> float:
        """Evaluate: |pred ∩ gold| / P"""
        pred_paths = parsed.get("paths", [])
        pred_weights = parsed.get("weights", [])
        gold_paths = ground_truth.get("paths", [])
        gold_weights = ground_truth.get("weights", [])
        
        P = len(gold_paths)
        if P == 0:
            return 0.0
        
        pred_pairs = set()
        for i in range(min(len(pred_paths), len(pred_weights))):
            pred_pairs.add((tuple(pred_paths[i]), pred_weights[i]))
        
        gold_pairs = set()
        for i in range(min(len(gold_paths), len(gold_weights))):
            gold_pairs.add((tuple(gold_paths[i]), gold_weights[i]))
        
        intersection = len(pred_pairs & gold_pairs)
        return intersection / P


class MMLUHandler:
    """Handler for MMLU tasks"""
    
    def format_prompt(self, example: Dict) -> str:
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
    
    def parse_response(self, response: str) -> str:
        if not response:
            return ""
        
        answer_match = re.search(r'The answer is\s*\(?([A-Z])\)?', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        
        letter_match = re.search(r'\b([A-D])\b', response)
        if letter_match:
            return letter_match.group(1)
        
        return ""
    
    def get_ground_truth(self, example: Dict) -> str:
        answer_idx = example.get("answer")
        if answer_idx is not None:
            return chr(65 + answer_idx)
        return ""
    
    def evaluate(self, parsed: str, ground_truth: str) -> float:
        return 1.0 if parsed.upper() == ground_truth.upper() else 0.0


class InfoBenchHandler:
    """Handler for InfoBench tasks"""
    
    def format_prompt(self, example: Dict) -> str:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        
        if input_text:
            return f"Instruction: {instruction}\nQuestion: {input_text}\nGeneration:"
        return f"Instruction: {instruction}\nGeneration:"
    
    def parse_response(self, response: str) -> str:
        return response if response else ""
    
    def get_ground_truth(self, example: Dict) -> None:
        return None
    
    def evaluate(self, parsed: str, ground_truth: None) -> float:
        # InfoBench requires GPT evaluation - return placeholder
        return 0.5 if parsed else 0.0


def get_handler(task: str):
    """Get appropriate handler for task"""
    if task.lower() in ["graph", "graphdev"]:
        return GraphHandler()
    elif task.lower() in ["mmlu", "mmlu_med"]:
        return MMLUHandler()
    elif task.lower() == "infobench":
        return InfoBenchHandler()
    else:
        raise ValueError(f"Unknown task: {task}")


# ============================================================================
# Local Testing (without full inference system)
# ============================================================================

def test_graph_solver(limit: int = 20):
    """Test graph solver locally without loading models"""
    print("Testing Graph Solver...")
    
    dataset = load_dataset("vashistht/11763_datasets", "graph_dev", split="dev_test")
    examples = list(dataset)[:limit]
    
    handler = GraphHandler()
    
    correct = 0
    total = 0
    
    for ex in tqdm(examples, desc="Graph"):
        prompt = handler.format_prompt(ex)
        response = solve_graph_problem(prompt)
        
        parsed = handler.parse_response(response, ex)
        ground_truth = handler.get_ground_truth(ex)
        score = handler.evaluate(parsed, ground_truth)
        
        if score >= 0.99:
            correct += 1
        total += 1
        
        if total <= 3:
            print(f"\nExample {total}:")
            print(f"  Response: {response[:100]}...")
            print(f"  Parsed: {parsed}")
            print(f"  Ground truth: {ground_truth}")
            print(f"  Score: {score}")
    
    print(f"\nGraph Accuracy: {correct}/{total} = {correct/total:.4f}")
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Evaluate inference system locally")
    parser.add_argument("--task", type=str, default="graphdev",
                       choices=["graphdev", "mmlu_med", "infobench"],
                       help="Task to evaluate")
    parser.add_argument("--limit", type=int, default=20,
                       help="Limit number of examples")
    parser.add_argument("--test-graph-only", action="store_true",
                       help="Only test graph solver (no model loading)")
    
    args = parser.parse_args()
    
    if args.test_graph_only or args.task == "graphdev":
        test_graph_solver(args.limit)
    else:
        print(f"Use --test-graph-only for local testing without models")


if __name__ == "__main__":
    main()