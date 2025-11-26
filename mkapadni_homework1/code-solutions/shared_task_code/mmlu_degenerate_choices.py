# ANDREW ID = MKAPADNI
import os
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/mkapadni/hf_cache/models"
import json
from tqdm import tqdm
from inference import (
    load_custom_dataset,
    generate_problem_prompt,
    convert_llm_response_to_solution,
    evaluate_solution,
    load_hf_model_and_tokenizer,
)
import torch

def hf_generate_degenerate(model, tokenizer, prompt, max_new_tokens=128):
    """Generate with deliberately bad hyperparameters for larger model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=3.5,      # Very high temperature for chaos
            top_p=0.05,          # Extremely restrictive top_p 
            top_k=3,             # Very low top_k
            repetition_penalty=2.5,  # High repetition penalty
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text

def hf_generate_stable(model, tokenizer, prompt, max_new_tokens=128):
    """Generate with stable, conservative settings for smaller model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,      # Low temperature for stability
            top_p=0.9,           # Conservative top_p
            top_k=50,            # Reasonable top_k
            repetition_penalty=1.1,  # Mild repetition penalty
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text

MAX_NEW_TOKENS = 128
OUT_DIR = "results_degenerate_choices"
os.makedirs(OUT_DIR, exist_ok=True)

def run_mmlu_single(model_name: str, setting: str, generate_fn, max_new_tokens: int, limit: int = None):
    dataset = load_custom_dataset("MMLU")
    if limit:
        dataset = dataset[:limit]

    model, tokenizer, device = load_hf_model_and_tokenizer(model_name)

    results = []
    correct = 0.0

    for i, example in tqdm(enumerate(dataset, 1), total=len(dataset), desc=f"{model_name} | {setting}"):
        prompt = generate_problem_prompt("MMLU", example)
        text = generate_fn(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        pred = convert_llm_response_to_solution(text, "MMLU")
        score = float(evaluate_solution(example, pred, "MMLU", model_name, api_key="") == True)
        correct += score
        results.append({
            "id": i,
            "prompt_subject": example.get("subject", ""),
            "prompt": prompt,
            "raw_generation": text,
            "pred_answer": pred,
            "gold_answer": example["choices"][example["answer"]],
            "score": score,
        })

    avg = correct / len(dataset) if dataset else 0.0
    tag = f"MMLU__{model_name.split('/')[-1]}__{setting}"
    with open(os.path.join(OUT_DIR, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "task": "MMLU",
            "model": model_name,
            "setting": setting,
            "max_new_tokens": max_new_tokens,
            "avg_score": avg,
            "n": len(dataset),
            "hyperparameters": get_hyperparameters(setting),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"{tag} avg_score={avg:.4f}")
    return avg

def get_hyperparameters(setting):
    if setting == "degenerate_chaos":
        return {
            "do_sample": True,
            "temperature": 3.5,
            "top_p": 0.05,
            "top_k": 3,
            "repetition_penalty": 2.5
        }
    elif setting == "stable_conservative":
        return {
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1
        }

def main():
    limit = 100 

    print("Running MMLU Degenerate Choices Experiment")
    print("=" * 50)
    
    # Strategy A: Degenerate decoding on larger model
    print("\nStrategy A: Qwen3-4B with chaotic sampling")
    print("Hyperparameters: temp=3.5, top_p=0.05, top_k=3, rep_penalty=2.5")
    
    score_4b = run_mmlu_single(
        model_name="Qwen/Qwen3-4B",
        setting="degenerate_chaos",
        generate_fn=hf_generate_degenerate,
        max_new_tokens=MAX_NEW_TOKENS,
        limit=limit
    )

    # Strategy B: Stable decoding on smaller model  
    print("\nStrategy B: Qwen3-1.7B with conservative sampling")
    print("Hyperparameters: temp=0.3, top_p=0.9, top_k=50, rep_penalty=1.1")
    
    score_1_7b = run_mmlu_single(
        model_name="Qwen/Qwen3-1.7B",
        setting="stable_conservative",
        generate_fn=hf_generate_stable,
        max_new_tokens=MAX_NEW_TOKENS,
        limit=limit
    )

    # Summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Qwen3-4B (degenerate):     {score_4b:.3f}")
    print(f"Qwen3-1.7B (stable):       {score_1_7b:.3f}")
    
    if score_4b < score_1_7b:
        print(f"\n✅ SUCCESS: Larger model underperformed by {score_1_7b - score_4b:.3f}")
        print("Degenerate decoding successfully inverted model ranking!")
    else:
        print(f"\n❌ Larger model still outperformed. Try more extreme settings.")

if __name__ == "__main__":
    main()