# Code to test perplexity predictions using Qwen2.5-7B model
# ANDREW ID = MKAPADNI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import math
import os
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/mkapadni/hf_cache/models"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# model_name = "Qwen/Qwen2.5-7B"
model_name = "Qwen/Qwen2.5-7B-Instruct"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")

def calculate_perplexity(model, tokenizer, prompt, max_new_tokens=64):
    """
    Calculate per-token and global perplexity for generated sequence using greedy sampling
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    prompt_length = input_ids.shape[1]
    
    # Generate with greedy sampling 
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy sampling
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    generated_ids = outputs.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_only = tokenizer.decode(generated_ids[prompt_length:], skip_special_tokens=True)
    
    # Calculate perplexity for the generated portion
    total_log_likelihood = 0
    per_token_perplexities = []
    
    # Get logits for the entire sequence
    with torch.no_grad():
        model_outputs = model(generated_ids.unsqueeze(0))
        logits = model_outputs.logits[0]  
    
    # Calculate per-token perplexity for generated tokens only
    for i in range(prompt_length, len(generated_ids)):
        if i == 0:  # Skip first token as there's no previous context
            continue
            
        # Get the logits for predicting token at position i
        token_logits = logits[i-1] 
        
        # Convert to probabilities
        probs = torch.softmax(token_logits, dim=-1)
        
        # Get probability of the actual token
        actual_token_id = generated_ids[i]
        token_prob = probs[actual_token_id].item()
        
        # Calculate per-token perplexity
        token_perplexity = 1 / token_prob if token_prob > 0 else float('inf')
        per_token_perplexities.append(token_perplexity)
        
        # Add to total log likelihood
        if token_prob > 0:
            total_log_likelihood += math.log(token_prob)
    
    # Calculate global perplexity
    num_generated_tokens = len(generated_ids) - prompt_length
    if num_generated_tokens > 0:
        avg_log_likelihood = total_log_likelihood / num_generated_tokens
        global_perplexity = math.exp(-avg_log_likelihood)
    else:
        global_perplexity = float('inf')
    
    return {
        'prompt': prompt,
        'generated_text': generated_text,
        'generated_only': generated_only,
        'num_generated_tokens': num_generated_tokens,
        'per_token_perplexities': per_token_perplexities,
        'avg_per_token_perplexity': np.mean(per_token_perplexities) if per_token_perplexities else float('inf'),
        'global_perplexity': global_perplexity,
        'generated_tokens': [tokenizer.decode([token_id]) for token_id in generated_ids[prompt_length:]]
    }


low_perplexity_prompts = [
    "The capital of France is",
    "Dear Sir or Madam, I am writing to inform you that",
    "The first law of thermodynamics states that energy"
]

high_perplexity_prompts = [
    "The purple elephant's quantum consciousness merged with the digital symphony, creating",
    "In the ancient Sumerian language, the word 'zukratum' means",
    "My grandmother's secret recipe calls for exactly 2.7 cups of"
]

print("Testing prompts...")
print("=" * 80)

results = []


print("\nLOW PERPLEXITY PROMPTS:")
print("=" * 50)
for i, prompt in enumerate(low_perplexity_prompts, 1):
    print(f"\nPrompt {i}: '{prompt}'")
    result = calculate_perplexity(model, tokenizer, prompt)
    results.append(('low', i, result))
    
    print(f"Generated: {result['generated_only']}")
    print(f"Number of generated tokens: {result['num_generated_tokens']}")
    print(f"Global perplexity: {result['global_perplexity']:.4f}")
    print(f"Average per-token perplexity: {result['avg_per_token_perplexity']:.4f}")
    print(f"Per-token perplexities: {[round(p, 4) for p in result['per_token_perplexities'][:10]]}")  # Show first 10


print("\n\nHIGH PERPLEXITY PROMPTS:")
print("=" * 50)
for i, prompt in enumerate(high_perplexity_prompts, 1):
    print(f"\nPrompt {i}: '{prompt}'")
    result = calculate_perplexity(model, tokenizer, prompt)
    results.append(('high', i, result))
    
    print(f"Generated: {result['generated_only']}")
    print(f"Number of generated tokens: {result['num_generated_tokens']}")
    print(f"Global perplexity: {result['global_perplexity']:.4f}")
    print(f"Average per-token perplexity: {result['avg_per_token_perplexity']:.4f}")
    print(f"Per-token perplexities: {[round(p, 4) for p in result['per_token_perplexities'][:10]]}")  # Show first 10

print("\n" + "=" * 80)
print("SUMMARY TABLE:")
print("=" * 80)
print(f"{'Type':<6} {'Prompt#':<8} {'Global PPL':<12} {'Avg Per-Token PPL':<18} {'# Tokens':<10}")
print("-" * 60)

for result_type, prompt_num, result in results:
    print(f"{result_type:<6} {prompt_num:<8} {result['global_perplexity']:<12.4f} {result['avg_per_token_perplexity']:<18.4f} {result['num_generated_tokens']:<10}")

# print("\nModel used: Qwen/Qwen2.5-7B")
print("\nModel used: Qwen/Qwen2.5-7B-Instruct")
print("Sampling method: Greedy (deterministic)")
print("Max new tokens: 64")