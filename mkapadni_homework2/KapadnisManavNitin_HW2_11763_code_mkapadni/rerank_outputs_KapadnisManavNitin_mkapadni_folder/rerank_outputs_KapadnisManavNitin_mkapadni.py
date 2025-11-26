"""
Reranking methods for Best-of-n and MBR decoding.
Section 2: Best-of-n and MBR

Each method returns a list of scores for each candidate output,
in the same order as the outputs were passed to the method.
# Andrew id: mkapadni
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import llm_blender
# For BLEU
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# For BERTScore
from bert_score import BERTScorer

# For scalar reward model
from transformers import AutoModelForSequenceClassification

# For pairwise reward model
import torch.nn as nn


class RerankingMethods:
    """Class to hold all reranking methods with shared model caching."""
    
    def __init__(self):
        self.models_cache = {}
        self.tokenizers_cache = {}
        
    def _get_model_and_tokenizer(self, model_name: str, device: str = 'cuda:0'):
        """Load and cache models to avoid reloading."""
        if model_name not in self.models_cache:
            print(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with appropriate dtype
            if "Qwen" in model_name:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto"
                )
            
            model.eval()
            self.models_cache[model_name] = model
            self.tokenizers_cache[model_name] = tokenizer
            
        return self.models_cache[model_name], self.tokenizers_cache[model_name]


def compute_model_prob(outputs: List[str], prompt: str, model: str = "Qwen/Qwen3-4B") -> List[float]:
    """
    Args:
        outputs: List of candidate output strings
        prompt: The input prompt
        model: Model name (default: "Qwen/Qwen3-4B")
        
    Returns:
        List of log-likelihood scores for each output
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Load model and tokenizer
    print(f"Loading model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    llm.eval()
    
    scores = []
    
    for output in tqdm(outputs, desc=f"Computing log-probs with {model}"):
        chat_form = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output}
        ]
        
        full_chat_tokens = tokenizer.apply_chat_template(
            chat_form,
            tokenize=True,
            return_tensors="pt"
        ).to(device)
        
        prompt_chat = [{"role": "user", "content": prompt}]
        prompt_tokens = tokenizer.apply_chat_template(
            prompt_chat,
            tokenize=True,
            add_generation_prompt=True
        )
        prompt_len = len(prompt_tokens)
        output_tokens = full_chat_tokens[0][prompt_len:]
        
        with torch.no_grad():
            logits = llm(full_chat_tokens).logits
            log_probs = F.log_softmax(logits[0], dim=-1)
            
            cumulative_logprob = 0.0
            for i, token_id in enumerate(output_tokens):
                # Get log prob for this token at the previous position
                if prompt_len + i < log_probs.shape[0]:
                    token_logprob = log_probs[prompt_len + i - 1, token_id].item()
                    cumulative_logprob += token_logprob
            
        scores.append(cumulative_logprob)
    
    return scores


def compute_scalar_reward(outputs: List[str], prompt: str) -> List[float]:
    """
    Args:
        outputs: List of candidate output strings
        prompt: The input prompt
        
    Returns:
        List of scalar reward scores for each output
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    print(f"Loading reward model: {model_name}")
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        num_labels=1
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    reward_model.eval()
    
    scores = []
    
    for output in tqdm(outputs, desc="Computing scalar rewards"):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output}
        ]

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(device)
        
        with torch.no_grad():
            reward = reward_model(**inputs).logits[0][0].item()
        
        scores.append(reward)
    
    return scores


def compute_pairwise_reward(outputs: List[str], prompt: str) -> List[float]:
    """
    Args:
        outputs: List of candidate output strings
        prompt: The input prompt

    Returns:
        List[float]: Pairwise scores (win counts) for each output
    """

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("Loading Pairwise Reward Model (llm-blender/PairRM)...")
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")

    n = len(outputs)
    win_counts = [0] * n

    for i in tqdm(range(n), desc="Computing pairwise comparisons"):
        for j in range(i + 1, n):
            comparison = blender.compare(
                [prompt],               
                [outputs[i]],          
                [outputs[j]]     
            )
            if comparison[0]:
                win_counts[i] += 1
            else:
                win_counts[j] += 1

    return [float(c) for c in win_counts]


def mbr_bleu(outputs: List[str], prompt: str = None) -> List[float]:
    """
    Args:
        outputs: List of candidate output strings
        prompt: The input prompt (not used for MBR-BLEU)
        
    Returns:
        List of MBR-BLEU scores for each output
    """
    n = len(outputs)
    scores = []

    tokenized_outputs = [output.split() for output in outputs]
    smoothing = SmoothingFunction()
    
    for i in tqdm(range(n), desc="Computing MBR-BLEU"):
        bleu_sum = 0.0
        for j in range(n):
            if i != j:
                bleu = sentence_bleu(
                    [tokenized_outputs[j]],
                    tokenized_outputs[i],
                    smoothing_function=smoothing.method1
                )
                bleu_sum += bleu

        avg_bleu = bleu_sum / (n - 1) if n > 1 else 0.0
        scores.append(avg_bleu)
    
    return scores


def mbr_bertscore(outputs: List[str], prompt: str = None) -> List[float]:
    """
    Args:
        outputs: List of candidate output strings
        prompt: The input prompt (not used for MBR-BERTScore)
        
    Returns:
        List of MBR-BERTScore scores for each output
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print("Initializing BERTScorer")
    scorer = BERTScorer(
        model_type="microsoft/deberta-xlarge-mnli",
        device=device,
        batch_size=8
    )
    
    n = len(outputs)
    scores = []
    
    for i in tqdm(range(n), desc="Computing MBR-BERTScore"):
        bertscore_sum = 0.0
        
        candidates = [outputs[i]] * (n - 1)
        references = [outputs[j] for j in range(n) if j != i]
        
        P, R, F1 = scorer.score(candidates, references)
        bertscore_sum = F1.mean().item()
        scores.append(bertscore_sum)
    
    return scores


# Convenience functions for direct use
def compute_qwen3_4b_logprobs(outputs: List[str], prompt: str) -> List[float]:
    """Convenience function for Qwen3-4B log-probs."""
    return compute_model_prob(outputs, prompt, model="Qwen/Qwen3-4B")


def compute_qwen3_14b_logprobs(outputs: List[str], prompt: str) -> List[float]:
    """Convenience function for Qwen3-14B log-probs."""
    return compute_model_prob(outputs, prompt, model="Qwen/Qwen3-14B")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing reranking methods...")
    
    # Sample data
    prompt = "What is the capital of France?"
    outputs = [
        "The capital of France is Paris.",
        "Paris is the capital city of France.",
        "France's capital is Paris, a beautiful city."
    ]
    
    print("\n1. Testing log-probability (Qwen3-4B)...")
    logprobs = compute_model_prob(outputs, prompt)
    print(f"Log-probs: {logprobs}")
    
    print("\n2. Testing scalar reward...")
    rewards = compute_scalar_reward(outputs, prompt)
    print(f"Scalar rewards: {rewards}")
    
    print("\n3. Testing pairwise reward...")
    pairwise = compute_pairwise_reward(outputs, prompt)
    print(f"Pairwise scores: {pairwise}")
    
    print("\n4. Testing MBR-BLEU...")
    bleu_scores = mbr_bleu(outputs, prompt)
    print(f"MBR-BLEU scores: {bleu_scores}")
    
    print("\n5. Testing MBR-BERTScore...")
    bert_scores = mbr_bertscore(outputs, prompt)
    print(f"MBR-BERTScore scores: {bert_scores}")
    
    print("\nAll tests completed!")