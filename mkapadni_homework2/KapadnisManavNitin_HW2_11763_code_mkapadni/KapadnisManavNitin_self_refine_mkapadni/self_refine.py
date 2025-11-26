# self_refine.py
# Andrew id: mkapadni

import os
import json
import time
import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import random
import numpy as np

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import the handlers
import dataset

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

@dataclass
class RefineConfig:
    """Configuration for self-refine process."""
    model_path: str
    dataset_name: str
    max_iterations: int = 4  # 1 draft + 3 refinements
    max_new_tokens: int = 512
    draft_temperature: float = 0.7
    critique_temperature: float = 0.7
    refine_temperature: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    output_dir: str = "./results"
    batch_size: int = 1  # Process one at a time for iterative refinement


class Generator:
    """LLM Engine for generation, feedback, and refinement - Updated for Qwen3 thinking models."""
    
    def __init__(self, cfg: RefineConfig):
        self.cfg = cfg
        print(f"Loading model: {cfg.model_path}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype="auto",  # Use auto for Qwen3
            device_map="auto"
        )
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        print(f"Model loaded successfully")
    
    def _format_prompt_qwen3(self, prompt: str) -> str:
        """Format prompt for Qwen3 thinking models with thinking disabled."""
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode
        )
        return text
    
    @torch.inference_mode()
    def _generate(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 512) -> str:
        """Generate text from prompt using Qwen3 models."""
        # Format prompt for Qwen3
        formatted_prompt = self._format_prompt_qwen3(prompt)
        
        # Tokenize
        model_inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.model.device)
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": 0.95 if temperature > 0 else 1.0,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        
        # Generate
        generated_ids = self.model.generate(**model_inputs, **gen_kwargs)
        
        # Extract only the new tokens (remove input)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Decode (no thinking parsing needed since enable_thinking=False)
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        return content
    
    def draft(self, questions: List[Any], handler: dataset.DatasetHandler) -> List[str]:
        """Generate initial drafts for questions."""
        drafts = []
        for q in questions:
            prompt = handler.format_question(q)
            response = self._generate(prompt, self.cfg.draft_temperature, self.cfg.max_new_tokens)
            drafts.append(response)
        return drafts
    
    def feedback(self, questions: List[Any], attempts: List[str], handler: dataset.DatasetHandler) -> List[str]:
        """Generate feedback for question-attempt pairs."""
        feedbacks = []
        for q, attempt in zip(questions, attempts):
            q_text = handler.format_question(q)
            prompt = handler.get_feedback_prompt(q_text, attempt)
            response = self._generate(prompt, self.cfg.critique_temperature, self.cfg.max_new_tokens)
            feedbacks.append(response)
        return feedbacks
    
    def refine(self, questions: List[Any], attempts: List[str], feedbacks: List[str], handler: dataset.DatasetHandler) -> List[str]:
        """Generate refinements based on feedback."""
        refinements = []
        for q, attempt, fb in zip(questions, attempts, feedbacks):
            q_text = handler.format_question(q)
            prompt = handler.get_refine_prompt(q_text, attempt, fb)
            response = self._generate(prompt, self.cfg.refine_temperature, self.cfg.max_new_tokens)
            refinements.append(response)
        return refinements
    
    def cleanup(self):
        """Clean up model to free memory."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Model cleaned up and memory freed")


def run_self_refine(
    examples: List[Dict[str, Any]],
    handler: dataset.DatasetHandler,
    generator: Generator,
    config: RefineConfig,
) -> Dict[str, Any]:
    """
    Implement the self-refinement algorithm.
    
    Returns results with iteration-by-iteration tracking for analysis.
    """
    all_results = []
    
    for ex_idx, example in enumerate(tqdm(examples, desc="Processing examples")):
        result = {
            "example_id": ex_idx,
            "example": example,
            "iterations": [],
            "ground_truth": handler.get_ground_truth(example)
        }
        
        # Iteration 1: Initial draft
        draft = generator.draft([example], handler)[0]
        parsed_draft = handler.parse_answer(draft)
        is_correct = handler.verify_answer(parsed_draft, result["ground_truth"])
        
        result["iterations"].append({
            "iteration": 1,
            "type": "draft",
            "response": draft,
            "parsed_answer": parsed_draft,
            "correct": is_correct
        })
        
        # Iterations 2-4: Feedback and refinement
        current_answer = draft
        for iter_num in range(2, config.max_iterations + 1):
            # Get feedback
            feedback = generator.feedback([example], [current_answer], handler)[0]
            
            # Refine based on feedback
            refined = generator.refine([example], [current_answer], [feedback], handler)[0]
            parsed_refined = handler.parse_answer(refined)
            is_correct = handler.verify_answer(parsed_refined, result["ground_truth"])
            
            result["iterations"].append({
                "iteration": iter_num,
                "type": "refine",
                "feedback": feedback,
                "response": refined,
                "parsed_answer": parsed_refined,
                "correct": is_correct
            })
            
            current_answer = refined
        
        all_results.append(result)
    
    return {"results": all_results, "config": vars(config)}


def analyze_results(results: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Analyze self-refine results and generate statistics."""
    all_results = results["results"]
    config = results["config"]
    max_iters = config["max_iterations"]
    
    # Compute metrics for each iteration
    accuracy_by_iter = []
    best_accuracy_by_iter = []
    
    # Conditional probabilities
    conditional_probs = []
    
    for iter_idx in range(max_iters):
        correct_count = 0
        total_count = len(all_results)
        best_so_far_count = 0
        
        # For conditional probabilities
        correct_to_correct = 0
        correct_to_incorrect = 0
        incorrect_to_correct = 0
        incorrect_to_incorrect = 0
        
        for result in all_results:
            # Current iteration correctness
            curr_correct = result["iterations"][iter_idx]["correct"]
            if curr_correct:
                correct_count += 1
            
            # Best so far (if any iteration up to current is correct)
            best_so_far = any(result["iterations"][j]["correct"] for j in range(iter_idx + 1))
            if best_so_far:
                best_so_far_count += 1
            
            # Conditional probabilities (for iter > 0)
            if iter_idx > 0:
                prev_correct = result["iterations"][iter_idx - 1]["correct"]
                
                if prev_correct and curr_correct:
                    correct_to_correct += 1
                elif prev_correct and not curr_correct:
                    correct_to_incorrect += 1
                elif not prev_correct and curr_correct:
                    incorrect_to_correct += 1
                elif not prev_correct and not curr_correct:
                    incorrect_to_incorrect += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        best_accuracy = best_so_far_count / total_count if total_count > 0 else 0
        
        accuracy_by_iter.append(accuracy)
        best_accuracy_by_iter.append(best_accuracy)
        
        if iter_idx > 0:
            # Calculate conditional probabilities
            p_correct_given_correct = correct_to_correct / (correct_to_correct + correct_to_incorrect) if (correct_to_correct + correct_to_incorrect) > 0 else 0
            p_correct_given_incorrect = incorrect_to_correct / (incorrect_to_correct + incorrect_to_incorrect) if (incorrect_to_correct + incorrect_to_incorrect) > 0 else 0
            
            conditional_probs.append({
                "iteration": iter_idx + 1,
                "P(correct_i+1|correct_i)": p_correct_given_correct,
                "P(correct_i+1|incorrect_i)": p_correct_given_incorrect,
                "correct_to_correct": correct_to_correct,
                "correct_to_incorrect": correct_to_incorrect,
                "incorrect_to_correct": incorrect_to_correct,
                "incorrect_to_incorrect": incorrect_to_incorrect
            })
    
    analysis = {
        "accuracy_by_iteration": accuracy_by_iter,
        "best_accuracy_by_iteration": best_accuracy_by_iter,
        "conditional_probabilities": conditional_probs,
        "total_examples": len(all_results),
        "max_iterations": max_iters
    }
    
    # Find examples where refinement helps/harms
    improved_examples = []
    harmed_examples = []
    
    for result in all_results:
        for iter_idx in range(1, max_iters):
            current_correct = result["iterations"][iter_idx]["correct"]
            prev_correct = result["iterations"][iter_idx - 1]["correct"]
            
            # Improved: was incorrect, now correct
            if not prev_correct and current_correct:
                improved_examples.append({
                    "example_id": result["example_id"],
                    "iteration": iter_idx + 1,
                    "example": result["example"],
                    "previous": result["iterations"][iter_idx - 1],
                    "current": result["iterations"][iter_idx]
                })
            
            # Harmed: was correct, now incorrect
            if prev_correct and not current_correct:
                harmed_examples.append({
                    "example_id": result["example_id"],
                    "iteration": iter_idx + 1,
                    "example": result["example"],
                    "previous": result["iterations"][iter_idx - 1],
                    "current": result["iterations"][iter_idx]
                })
    
    analysis["improved_examples"] = improved_examples[:5]
    analysis["harmed_examples"] = harmed_examples[:5]
    
    return analysis


def load_dataset_by_name(dataset_name: str, split: str = "dev_test"):
    """Load dataset from HuggingFace."""
    if dataset_name.lower() == "graphdev":
        raw_dataset = load_dataset("vashistht/11763_datasets", "graph_dev")
    elif dataset_name.lower() == "mmlu_med":
        raw_dataset = load_dataset("vashistht/11763_datasets", "mmlu_med")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return list(raw_dataset[split])


def main():
    parser = argparse.ArgumentParser(description="Self-Refine Implementation")
    parser.add_argument("--model", type=str, required=True, help="Model path (e.g., Qwen/Qwen3-4B)")
    parser.add_argument("--dataset", type=str, required=True, choices=["graphdev", "mmlu_med"], help="Dataset name")
    parser.add_argument("--max_iterations", type=int, default=4, help="Max refinement iterations")
    parser.add_argument("--draft_temp", type=float, default=0.7, help="Temperature for draft")
    parser.add_argument("--critique_temp", type=float, default=0.7, help="Temperature for critique")
    parser.add_argument("--refine_temp", type=float, default=0.7, help="Temperature for refine")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--max_examples", type=int, default=None, help="Max examples to process (for testing)")
    parser.add_argument("--split", type=str, default="dev_test", help="Dataset split")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set max_new_tokens based on dataset (1024 for GraphDev, 512 for MMLU)
    max_new_tokens = 1024 if args.dataset.lower() == "graphdev" else 512
    
    # Initialize config
    config = RefineConfig(
        model_path=args.model,
        dataset_name=args.dataset,
        max_iterations=args.max_iterations,
        max_new_tokens=max_new_tokens,
        draft_temperature=args.draft_temp,
        critique_temperature=args.critique_temp,
        refine_temperature=args.refine_temp,
        output_dir=args.output_dir
    )
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    examples = load_dataset_by_name(args.dataset, args.split)
    
    if args.max_examples:
        examples = examples[:args.max_examples]
    
    print(f"Loaded {len(examples)} examples")
    
    # Initialize handler
    HANDLERS = {
        "graphdev": dataset.GraphHandler,
        "mmlu_med": dataset.MMLUMedHandler,
    }
    handler = HANDLERS[args.dataset.lower()]()
    
    # Initialize generator
    generator = Generator(config)
    
    # Run self-refine
    print("Running self-refine...")
    results = run_self_refine(examples, handler, generator, config)
    
    # Clean up generator
    generator.cleanup()
    
    # Save raw results
    model_name = args.model.split("/")[-1]
    results_file = output_dir / f"results_{model_name}_{args.dataset}_temp{args.draft_temp}-{args.critique_temp}-{args.refine_temp}.json"
    
    print(f"Saving results to {results_file}")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Analyze results
    print("Analyzing results...")
    analysis = analyze_results(results, output_dir)
    
    # Save analysis
    analysis_file = output_dir / f"analysis_{model_name}_{args.dataset}_temp{args.draft_temp}-{args.critique_temp}-{args.refine_temp}.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary with 4 decimal places
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Max tokens: {max_new_tokens}")
    print(f"Temperatures: draft={args.draft_temp}, critique={args.critique_temp}, refine={args.refine_temp}")
    print(f"Total examples: {analysis['total_examples']}")
    print("\nAccuracy by iteration:")
    for i, acc in enumerate(analysis["accuracy_by_iteration"]):
        print(f"  Iteration {i+1}: {acc:.4f}")
    print("\nBest accuracy so far by iteration:")
    for i, acc in enumerate(analysis["best_accuracy_by_iteration"]):
        print(f"  Up to iteration {i+1}: {acc:.4f}")
    
    if analysis["conditional_probabilities"]:
        print("\nConditional Probabilities:")
        for cp in analysis["conditional_probabilities"]:
            print(f"  Iteration {cp['iteration']}: P(correct|correct)={cp['P(correct_i+1|correct_i)']:.4f}, P(correct|incorrect)={cp['P(correct_i+1|incorrect_i)']:.4f}")
    
    print(f"\nImproved examples: {len(analysis['improved_examples'])}")
    print(f"Harmed examples: {len(analysis['harmed_examples'])}")
    print("="*50)


if __name__ == "__main__":
    main()
