# ANDREW ID = MKAPADNI
import os
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/mkapadni/hf_cache/models"
import re
import json
from typing import Dict, Any, List
from tqdm import tqdm
import torch
from inference import (
    load_custom_dataset,          
    load_hf_model_and_tokenizer,  
    hf_generate_once,            
    HF_QWEN_MODELS,               
)
from graph_path_finder import (
    GraphPathSolution,
    PathInfo,
)

DECODING_SETTINGS = ["default", "greedy", "temp_0_25", "temp_1_5", "beam_3", "beam_25", "typical"]

def build_json_prompt_from_edges(edges: List[List[int]], N: int, P: int) -> str:
    lines = [f"You are given a directed graph with {N} nodes (numbered 0 to {N-1}) and the following weighted edges (src -> dst, weight):"]
    for (s, d, w) in edges:
        lines.append(f"{s} -> {d}, weight: {w}")
    lines.append("")
    lines.append(f"Return the top {P} shortest path(s) from node 0 to node {N-1} as strict JSON with keys 'paths' and 'weights'.")
    lines.append('Example format: {"paths": [[0, 2, 4]], "weights": [10]}')
    lines.append("Output JSON only with no extra text.")
    return "\n".join(lines)

def parse_paths_weights(text: str) -> Dict[str, Any]:
    """
    Improved parsing function to extract paths and weights from model output.
    Handles conversation format, arithmetic expressions, and various JSON structures.
    """
    
    assistant_match = re.search(r'assistant\n.*?(\{.*?\})', text, re.DOTALL)
    if assistant_match:
        json_candidate = assistant_match.group(1)
    else:
        
        json_match = re.search(r'\{[^{}]*"paths"[^{}]*"weights"[^{}]*\}', text, re.DOTALL)
        if json_match:
            json_candidate = json_match.group(0)
        else:
            # Try to find any JSON object
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_candidate = json_match.group(0)
            else:
                return {"paths": [], "weights": []}
    try:
        brace_count = 0
        end_pos = 0
        for i, char in enumerate(json_candidate):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        
        if end_pos > 0:
            json_candidate = json_candidate[:end_pos]

        data = json.loads(json_candidate)
        paths = data.get("paths", [])
        weights = data.get("weights", [])
        
        norm_paths = []
        for p in paths:
            if isinstance(p, list):
                try:
                    norm_paths.append([int(x) for x in p])
                except (ValueError, TypeError):
                    continue
        
        norm_weights = []
        for w in weights:
            try:
                if isinstance(w, str):
                    if re.match(r'^[\d\s+\-*/().]+$', w):
                        norm_weights.append(int(eval(w)))
                    else:
                        norm_weights.append(0)
                elif isinstance(w, (int, float)):
                    norm_weights.append(int(w))
                else:
                    norm_weights.append(0)
            except (ValueError, TypeError, SyntaxError, NameError):
                norm_weights.append(0)
        return {"paths": norm_paths, "weights": norm_weights}
        
    except json.JSONDecodeError:
        return extract_paths_weights_fallback(text)

def extract_paths_weights_fallback(text: str) -> Dict[str, Any]:
    """
    Fallback method to extract paths and weights using regex when JSON parsing fails.
    """
    paths = []
    weights = []

    paths_match = re.search(r'"paths"\s*:\s*\[([^\]]+(?:\],\s*\[[^\]]+)*)\]', text)
    if paths_match:
        paths_str = '[' + paths_match.group(1) + ']'
        try:
            paths_raw = json.loads(paths_str)
            for p in paths_raw:
                if isinstance(p, list):
                    try:
                        paths.append([int(x) for x in p])
                    except (ValueError, TypeError):
                        continue
        except json.JSONDecodeError:
            pass

    weights_match = re.search(r'"weights"\s*:\s*\[([^\]]+)\]', text)
    if weights_match:
        weights_str = '[' + weights_match.group(1) + ']'
        try:
            weights_raw = json.loads(weights_str)
            for w in weights_raw:
                try:
                    if isinstance(w, str) and re.match(r'^[\d\s+\-*/().]+$', w):
                        weights.append(int(eval(w)))
                    else:
                        weights.append(int(w))
                except (ValueError, TypeError, SyntaxError, NameError):
                    weights.append(0)
        except json.JSONDecodeError:
            pass
    
    return {"paths": paths, "weights": weights}

def to_solution_from_parsed(obj: Dict[str, Any]) -> GraphPathSolution:
    paths = []
    p_list = obj.get("paths", [])
    w_list = obj.get("weights", [])
    for i, path in enumerate(p_list):
        w = w_list[i] if i < len(w_list) else 0
        paths.append(PathInfo(path=path, weight=w))
    return GraphPathSolution(paths=paths)

def to_solution_from_dataset(sol_obj: Dict[str, Any]) -> GraphPathSolution:
    # Dataset solution looks like: {"paths": [{"path":[...], "weight": 123}, ...]}
    paths = []
    for ent in sol_obj.get("paths", []):
        try:
            path = [int(x) for x in ent.get("path", [])]
            weight = int(ent.get("weight", 0))
            paths.append(PathInfo(path=path, weight=weight))
        except Exception:
            continue
    return GraphPathSolution(paths=paths)

def evaluate_paths_exact(gold: GraphPathSolution, pred: GraphPathSolution, P: int) -> float:
    gold_paths = [tuple(pi.path) for pi in gold.paths[:P]]
    pred_paths = set(tuple(pi.path) for pi in pred.paths)
    hits = sum(1 for gp in gold_paths if gp in pred_paths)
    return float(hits) / float(max(1, P))

def run_graph_dev(model_name: str, out_dir: str, max_new_tokens: int = 128, limit: int = None):
    os.makedirs(out_dir, exist_ok=True)

    dataset = load_custom_dataset("graphdev") 
    if limit:
        dataset = dataset[:limit]

    model, tokenizer, device = load_hf_model_and_tokenizer(model_name)

    for setting in DECODING_SETTINGS:
        total = 0.0
        results = []
        for i, row in tqdm(enumerate(dataset, 1), total=len(dataset), desc=f"{model_name} | {setting}"):
            params = row["graph_params"] if "graph_params" in row else row.get("graph_params", {})
            edges = row["edges"] if "edges" in row else row.get("edges", [])
            sol = row["solution"] if "solution" in row else row.get("solution", {"paths": []})
            norm_edges = []
            for e in edges:
                try:
                    s, d, w = int(e[0]), int(e[1]), int(e[2])
                    norm_edges.append([s, d, w])
                except Exception:
                    continue

            N = int(params.get("N", 0))
            P = int(params.get("P", 1))
            if N <= 1:
                # skip degenerate
                results.append({
                    "id": i,
                    "params": params,
                    "edges": norm_edges,
                    "prompt": "",
                    "raw_generation": "",
                    "parsed": {"paths": [], "weights": []},
                    "score": 0.0,
                })
                continue

            prompt = build_json_prompt_from_edges(norm_edges, N, P)
            text = hf_generate_once(model, tokenizer, prompt, setting, max_new_tokens=max_new_tokens)
            parsed = parse_paths_weights(text)
            pred = to_solution_from_parsed(parsed)
            gold = to_solution_from_dataset(sol)
            score = evaluate_paths_exact(gold, pred, P)

            results.append({
                "id": i,
                "params": params,
                "edges": norm_edges,
                "prompt": prompt,
                "raw_generation": text,
                "parsed": parsed,
                "score": score,
            })
            total += score

        avg = total / len(dataset) if dataset else 0.0
        tag = f"GraphDev__{model_name.split('/')[-1]}__{setting}"
        with open(os.path.join(out_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "task": "GraphDev",
                "model": model_name,
                "setting": setting,
                "avg_score": avg,
                "n": len(dataset),
                "results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"{tag} avg_score={avg:.4f}")

if __name__ == "__main__":
    out_dir = "results"
    limit =  None
    for m in HF_QWEN_MODELS:
        run_graph_dev(m, out_dir=out_dir, max_new_tokens=1024, limit=limit)
