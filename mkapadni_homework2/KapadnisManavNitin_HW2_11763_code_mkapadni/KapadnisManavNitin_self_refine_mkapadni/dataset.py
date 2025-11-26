# dataset.py
# Andrew id: mkapadni

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import ast
import json
import re

# Dataset abstract class
class DatasetHandler(ABC):
    @abstractmethod
    def format_question(self, example: Dict[str, Any]) -> str:
        """Format the question for the model."""
        pass

    @abstractmethod
    def parse_answer(self, response: str) -> Any:
        """Parse the model response to extract the answer."""
        pass

    @abstractmethod
    def verify_answer(self, predicted: Any, ground_truth: Any) -> bool:
        """Implement answer verification logic."""
        pass

    @abstractmethod
    def get_ground_truth(self, example: Dict[str, Any]) -> Any:
        """Extract ground truth from the example."""
        pass

    @abstractmethod
    def get_feedback_prompt(self, question: str, answer: str) -> str:
        """Generate feedback prompt for the answer."""
        pass

    @abstractmethod
    def get_refine_prompt(self, question: str, answer: str, feedback: str) -> str:
        """Generate refinement prompt based on feedback."""
        pass


class GraphHandler(DatasetHandler):
    """Handler for graph pathfinding dataset (GraphDev) - Based on graph_benchmark_hf.py."""

    def _extract_graph_info(self, example: Dict[str, Any]):
        """Extract N, P, and edges from the example."""
        params = example.get("graph_params", {})
        edges = example.get("edges", [])
        N = int(params.get("N", 0))
        P = int(params.get("P", 1))

        norm_edges = []
        for e in edges:
            try:
                s, d, w = int(e[0]), int(e[1]), int(e[2])
                norm_edges.append([s, d, w])
            except Exception:
                continue

        return N, P, norm_edges

    def format_question(self, example: Dict[str, Any]) -> str:
        """Format graph question using the prompt from graph_benchmark_hf.py."""
        N, P, edges = self._extract_graph_info(example)

        lines = [f"You are given a directed graph with {N} nodes (numbered 0 to {N-1}) and the following weighted edges (src -> dst, weight):"]
        for (s, d, w) in edges:
            lines.append(f"{s} -> {d}, weight: {w}")
        lines.append("")
        lines.append(f"Return the top {P} shortest path(s) from node 0 to node {N-1} as strict JSON with keys \'paths\' and \'weights\'.")
        lines.append("Example format: {'paths': [[0, 2, 4]], 'weights': [10]}")
        lines.append("Output JSON only with no extra text.")

        return "\n".join(lines)

    def parse_answer(self, response: str) -> Any:
        """Parse graph answer from model response - Improved parsing from graph_benchmark_hf.py."""
        
        assistant_match = re.search(r"assistant\n.*?(\{.*?\})", response, re.DOTALL)
        if assistant_match:
            json_candidate = assistant_match.group(1)
        else:
            
            json_match = re.search(r'\{[^{}]*"paths"[^{}]*"weights"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_candidate = json_match.group(0)
            else:
                
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_candidate = json_match.group(0)
                else:
                    return []

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

            # Normalize paths and weights
            result = []
            for i, p in enumerate(paths):
                if isinstance(p, list):
                    try:
                        path_nodes = [int(x) for x in p]
                        weight = int(weights[i]) if i < len(weights) else 0
                        result.append({"path": path_nodes, "weight": weight})
                    except (ValueError, TypeError):
                        continue

            return result
        except json.JSONDecodeError:
            return self._fallback_parse(response)

    def _fallback_parse(self, text: str) -> List[Dict[str, Any]]:
        """Fallback parsing method."""
        paths = []
        weights = []

        # Try to extract paths
        # Try to extract paths
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

        # Try to extract weights
        weights_match = re.search(r'"weights"\s*:\s*\[([^\]]+)\]', text)
        if weights_match:
            weights_str = '[' + weights_match.group(1) + ']'
            try:
                weights_raw = json.loads(weights_str)
                for w in weights_raw:
                    try:
                        weights.append(int(w))
                    except (ValueError, TypeError):
                        weights.append(0)
            except json.JSONDecodeError:
                pass

        result = []
        for i, path in enumerate(paths):
            weight = weights[i] if i < len(weights) else 0
            result.append({"path": path, "weight": weight})

        return result

    def verify_answer(self, predicted: Any, ground_truth: Any) -> bool:
        """Verify graph answer - relaxed verification as specified."""
        if not predicted or not ground_truth:
            return False

        pred_paths = predicted if isinstance(predicted, list) else []
        gt_paths = ground_truth if isinstance(ground_truth, list) else []

        # Relaxed verification: at least one path matches (as per instructor note)
        for pred_path_info in pred_paths:
            for gt_path_info in gt_paths:
                if (pred_path_info.get("path") == gt_path_info.get("path") and 
                    pred_path_info.get("weight") == gt_path_info.get("weight")):
                    return True
        return False

    def get_ground_truth(self, example: Dict[str, Any]) -> Any:
        """Extract ground truth from graph example."""
        # Handle both formats
        if "solution" in example:
            sol = example["solution"]
            if isinstance(sol, dict) and "paths" in sol:
                return sol["paths"]

        answer = example.get('answer', example.get('paths', []))
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except:
                return []
        return answer

    def get_feedback_prompt(self, question: str, answer: str) -> str:
        """Generate feedback prompt for graph answer."""
        return f"""You are reviewing a solution to a graph shortest path problem. Provide constructive, specific feedback.

Problem:
{question}

Current Solution:
{answer}

Analyze the solution carefully:
1. Are the paths valid from source to destination?
2. Are the path weights correctly calculated?
3. Are these actually the shortest paths?
4. Is the JSON format correct?

Provide specific feedback on what needs to be corrected or improved."""

    def get_refine_prompt(self, question: str, answer: str, feedback: str) -> str:
        """Generate refinement prompt for graph answer."""
        return f"""Based on the feedback below, provide an improved solution to the graph problem.

Problem:
{question}

Previous Solution:
{answer}

Feedback:
{feedback}

Please provide a corrected solution in the exact JSON format:
{{"paths": [[node1, node2, ...]], "weights": [weight1, ...]}}

Output only the JSON with no extra text."""


class MMLUMedHandler(DatasetHandler):
    """Handler for MMLU medical dataset."""

    def __init__(self):
        self.choices = ["A", "B", "C", "D"]

    def format_subject(self, subject: str) -> str:
        """Format subject name."""
        return " ".join(subject.split("_"))

    def format_question(self, example: Dict[str, Any]) -> str:
        """Format MMLU question."""
        subject = example.get('subject', 'medicine')
        question = example.get('question', '')
        choices = example.get('choices', [])

        prompt = f"""The following is a multiple choice question about {self.format_subject(subject)}. 
Answer by selecting the correct option (A, B, C, or D).

Question: {question}

Options:
"""
        for i, choice in enumerate(choices):
            prompt += f"{self.choices[i]}. {choice}\n"

        prompt += "\nProvide your answer in the format: The answer is (X), where X is one of A, B, C, or D."
        return prompt

    def parse_answer(self, response: str) -> Any:
        """Parse MMLU answer from model response."""
        
        response = response.replace('**', '')

        
        pattern = r"answer is \(?([A-D])\)?"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        
        match = re.search(r"Answer:\s*([A-D])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        
        pattern = r"\b([A-D])\b(?!.*\b[A-D]\b)"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).upper()

        return None

    def verify_answer(self, predicted: Any, ground_truth: Any) -> bool:
        """Verify MMLU answer."""
        if predicted is None:
            return False
        return str(predicted).upper() == str(ground_truth).upper()

    def get_ground_truth(self, example: Dict[str, Any]) -> Any:
        """Extract ground truth from MMLU example."""
        answer_idx = example.get("answer", 0)
        if isinstance(answer_idx, int):
            return self.choices[answer_idx]
        return str(answer_idx).upper()

    def get_feedback_prompt(self, question: str, answer: str) -> str:
        """Generate feedback prompt for MMLU answer."""
        return f"""You are reviewing an answer to a medical multiple choice question. Provide constructive, specific feedback.

Question:
{question}

Current Answer:
{answer}

Carefully analyze:
1. Is the medical reasoning correct?
2. Is the selected option accurate based on medical knowledge?
3. Are there any logical errors or misconceptions?
4. What evidence supports or contradicts this answer?

Provide specific, actionable feedback to improve the answer."""

    def get_refine_prompt(self, question: str, answer: str, feedback: str) -> str:
        """Generate refinement prompt for MMLU answer."""
        return f"""Based on the feedback below, reconsider and provide an improved answer to the medical question.

Question:
{question}

Previous Answer:
{answer}

Feedback:
{feedback}

Provide your improved answer in the format: The answer is (X), where X is one of A, B, C, or D."""
