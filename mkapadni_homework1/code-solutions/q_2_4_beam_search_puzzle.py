"""
Simple Beam Search Puzzle Solver for Qwen3-1.7B
To Find cases where different beams produce identical text
"""
# ANDREW ID = MKAPADNI

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

def load_model():
    """Load Qwen3-1.7B model and tokenizer"""
    model_name = "Qwen/Qwen3-1.7B"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    print(f"Model loaded on device: {model.device}")
    return model, tokenizer

def test_beam_search(model, tokenizer, prompt, num_beams=20, max_new_tokens=50):
    print(f"\nTesting: '{prompt}' with {num_beams} beams")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    # Generate with beam search
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_outputs = outputs[:, input_length:]
    decoded_outputs = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
    
    # Check for identical outputs 
    unique_outputs = set(decoded_outputs)
    has_identical = len(decoded_outputs) > len(unique_outputs)
    
    print(f"Generated {len(decoded_outputs)} beams")
    print(f"Found {len(unique_outputs)} unique outputs")
    
    if has_identical:
        print("🎯 FOUND IDENTICAL BEAMS!")
        output_counts = {}
        for i, output in enumerate(decoded_outputs):
            if output not in output_counts:
                output_counts[output] = []
            output_counts[output].append(i)
        
        for output, beam_indices in output_counts.items():
            if len(beam_indices) > 1:
                print(f"Beams {beam_indices} produced identical text:")
                print(f"'{output}'")
        
        return True, decoded_outputs, unique_outputs
    else:
        print("❌ No identical beams found")
        return False, decoded_outputs, unique_outputs

def main():
    
    print("QWEN3-1.7B BEAM SEARCH PUZZLE SOLVER")
    print("=" * 50)
    model, tokenizer = load_model()
    
    test_cases = [
        # Repetitive inputs
        "The the the the",
        "A A A A A",
        "Hello hello hello",
        "Yes yes yes yes",
        
        # Degenerate cases
        "",
        " ",
        ".",
        "!",
        
        # Symbol-heavy
        "!@#$%^&*()",
        ".........",
        "????????",
        "||||||||",
        
        # Very long words or repetitive patterns
        "antidisestablishmentarianism",
        "supercalifragilisticexpialidocious",
        "pneumonoultramicroscopicsilicovolcanoconiosispneumonoultramicroscopicsilicovolcanoconiosis",
        
        # Numbers and sequences
        "1 1 1 1 1",
        "123 123 123",
        "0000000000",
        
        # Short prompts that might lead to deterministic continuations
        "The",
        "I",
        "It",
        "This",
    ]
    
    print(f"Testing {len(test_cases)} different inputs...")
    successful_cases = []
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}]")
        
        try:
            found_identical, outputs, unique_outputs = test_beam_search(
                model, tokenizer, prompt, 
                num_beams=30,  # High beam size 
                max_new_tokens=30
            )
            
            if found_identical:
                successful_cases.append({
                    'prompt': prompt,
                    'outputs': outputs,
                    'unique_outputs': unique_outputs
                })
                print(f"✅ SUCCESS with prompt: '{prompt}'")
                
                # Stop after finding first success for assignment
                break
                
        except Exception as e:
            print(f"❌ Error with prompt '{prompt}': {e}")
            continue
    
    # Report results
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    
    if successful_cases:
        case = successful_cases[0]
        print(f"\n🎯 SOLUTION FOUND!")
        print(f"Input: '{case['prompt']}'")
        print(f"Configuration: 30 beams, max_new_tokens=30")
        print(f"Result: {len(case['outputs'])} beams generated {len(case['unique_outputs'])} unique outputs")
        output_counts = {}
        for i, output in enumerate(case['outputs']):
            if output not in output_counts:
                output_counts[output] = []
            output_counts[output].append(i)
        
        print("\nIdentical beam pairs:")
        for output, beam_indices in output_counts.items():
            if len(beam_indices) > 1:
                print(f"Beams {beam_indices}: '{output}'")
        
    else:
        print("\n❌ No identical beams found in any test case")

if __name__ == "__main__":
    main()