import torch
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os
from typing import List, Dict, Tuple
import numpy as np

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
NUM_WARMUP = 3
NUM_ITERATIONS = 10
BATCH_SIZE = 8

def get_gpu_name():
    """Get the name of the GPU being used"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def load_model(model_name: str):
    """Load model and tokenizer"""
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def prepare_batch_inputs(tokenizer, batch_size: int, input_length: int, is_qwen3: bool = True):
    """
    Prepare random batch inputs for benchmarking.
    For Qwen3 models, use chat template with thinking disabled.
    """
    vocab_size = tokenizer.vocab_size
    
    if is_qwen3:
        # For Qwen3, use chat template with thinking disabled
        messages = [{"role": "user", "content": "x" * input_length}]
        
        # Use apply_chat_template with enable_thinking=False
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode
        )
        
        # Create batch
        texts = [text] * batch_size
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    else:
        # For non-Qwen3 models, use random tokens
        random_tokens = torch.randint(0, vocab_size, (batch_size, input_length), dtype=torch.long)
        inputs = {"input_ids": random_tokens.to(DEVICE), "attention_mask": torch.ones_like(random_tokens).to(DEVICE)}
    
    return inputs

def benchmark_generation(
    model,
    tokenizer,
    input_length: int,
    output_length: int,
    batch_size: int,
    is_qwen3: bool = True,
    num_warmup: int = NUM_WARMUP,
    num_iterations: int = NUM_ITERATIONS
) -> Tuple[float, float]:
    """
    Benchmark generation with given configuration.
    Returns: (average_time, average_throughput)
    """
    times = []
    
    # Warmup
    for _ in range(num_warmup):
        try:
            inputs = prepare_batch_inputs(tokenizer, batch_size, input_length, is_qwen3)
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=output_length,
                    min_new_tokens=output_length,  # Force exact length
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            torch.cuda.synchronize()
            clear_memory()
        except Exception as e:
            print(f"Warmup failed: {e}")
            return None, None
    
    # Actual benchmarking
    for i in range(num_iterations):
        try:
            inputs = prepare_batch_inputs(tokenizer, batch_size, input_length, is_qwen3)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=output_length,
                    min_new_tokens=output_length,  # Force exact length
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            torch.cuda.synchronize()
            end_time = time.time()
            
            times.append(end_time - start_time)
            clear_memory()
            
        except torch.cuda.OutOfMemoryError:
            print(f"OOM Error at iteration {i+1}")
            clear_memory()
            return None, None
        except Exception as e:
            print(f"Error at iteration {i+1}: {e}")
            return None, None
    
    avg_time = np.mean(times)
    total_tokens = batch_size * output_length
    avg_throughput = total_tokens / avg_time
    
    return avg_time, avg_throughput

def task1_input_sweep():
    """Task 1: Sweep over input sequence lengths"""
    print("\n" + "="*80)
    print("TASK 1: Varying Input Sequence Lengths")
    print("="*80)
    
    model_name = "Qwen/Qwen3-8B"
    output_length = 64
    
    model, tokenizer = load_model(model_name)
    
    # Input lengths: 2^n for n in [0, 15]
    input_lengths = [2**n for n in range(16)]
    
    results = []
    
    for input_len in input_lengths:
        print(f"\nTesting input_length={input_len}...")
        
        try:
            avg_time, avg_throughput = benchmark_generation(
                model, tokenizer, input_len, output_length, BATCH_SIZE, is_qwen3=True
            )
            
            if avg_time is None:
                print(f"  ❌ OOM at input_length={input_len}")
                results.append({
                    "input_length": input_len,
                    "time_s": "OOM",
                    "throughput_tokens_per_s": "OOM"
                })
            else:
                print(f"  ✓ Time: {avg_time:.4f}s, Throughput: {avg_throughput:.2f} tokens/s")
                results.append({
                    "input_length": input_len,
                    "time_s": f"{avg_time:.4f}",
                    "throughput_tokens_per_s": f"{avg_throughput:.2f}"
                })
        except Exception as e:
            print(f"  ❌ Error at input_length={input_len}: {e}")
            results.append({
                "input_length": input_len,
                "time_s": f"Error: {str(e)[:50]}",
                "throughput_tokens_per_s": "Error"
            })
        
        clear_memory()
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("task1_input_sweep_results.csv", index=False)
    print("\n✓ Results saved to task1_input_sweep_results.csv")
    
    # Create plots
    plot_results = [r for r in results if r["time_s"] != "OOM" and not r["time_s"].startswith("Error")]
    if plot_results:
        plot_df = pd.DataFrame(plot_results)
        plot_df["time_s"] = plot_df["time_s"].astype(float)
        plot_df["throughput_tokens_per_s"] = plot_df["throughput_tokens_per_s"].astype(float)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(plot_df["input_length"], plot_df["time_s"], marker='o')
        ax1.set_xlabel("Input Length (tokens)")
        ax1.set_ylabel("Time (s)")
        ax1.set_title("Generation Time vs Input Length")
        ax1.set_xscale("log", base=2)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(plot_df["input_length"], plot_df["throughput_tokens_per_s"], marker='o', color='green')
        ax2.set_xlabel("Input Length (tokens)")
        ax2.set_ylabel("Throughput (tokens/s)")
        ax2.set_title("Throughput vs Input Length")
        ax2.set_xscale("log", base=2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("task1_input_sweep_plots.png", dpi=300, bbox_inches='tight')
        print("✓ Plots saved to task1_input_sweep_plots.png")
        plt.close()
    
    # Cleanup
    del model, tokenizer
    clear_memory()
    
    return results

def task2_output_sweep():
    """Task 2: Sweep over output sequence lengths"""
    print("\n" + "="*80)
    print("TASK 2: Varying Output Sequence Lengths")
    print("="*80)
    
    model_name = "Qwen/Qwen3-8B"
    input_length = 64
    
    model, tokenizer = load_model(model_name)
    
    # Output lengths: 2^n for n in [0, 8]
    output_lengths = [2**n for n in range(9)]
    
    results = []
    
    for output_len in output_lengths:
        print(f"\nTesting output_length={output_len}...")
        
        try:
            avg_time, avg_throughput = benchmark_generation(
                model, tokenizer, input_length, output_len, BATCH_SIZE, is_qwen3=True
            )
            
            if avg_time is None:
                print(f"  ❌ OOM at output_length={output_len}")
                results.append({
                    "output_length": output_len,
                    "time_s": "OOM",
                    "throughput_tokens_per_s": "OOM"
                })
            else:
                print(f"  ✓ Time: {avg_time:.4f}s, Throughput: {avg_throughput:.2f} tokens/s")
                results.append({
                    "output_length": output_len,
                    "time_s": f"{avg_time:.4f}",
                    "throughput_tokens_per_s": f"{avg_throughput:.2f}"
                })
        except Exception as e:
            print(f"  ❌ Error at output_length={output_len}: {e}")
            results.append({
                "output_length": output_len,
                "time_s": f"Error: {str(e)[:50]}",
                "throughput_tokens_per_s": "Error"
            })
        
        clear_memory()
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("task2_output_sweep_results.csv", index=False)
    print("\n✓ Results saved to task2_output_sweep_results.csv")
    
    # Create plots
    plot_results = [r for r in results if r["time_s"] != "OOM" and not r["time_s"].startswith("Error")]
    if plot_results:
        plot_df = pd.DataFrame(plot_results)
        plot_df["time_s"] = plot_df["time_s"].astype(float)
        plot_df["throughput_tokens_per_s"] = plot_df["throughput_tokens_per_s"].astype(float)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(plot_df["output_length"], plot_df["time_s"], marker='o', color='red')
        ax1.set_xlabel("Output Length (tokens)")
        ax1.set_ylabel("Time (s)")
        ax1.set_title("Generation Time vs Output Length")
        ax1.set_xscale("log", base=2)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(plot_df["output_length"], plot_df["throughput_tokens_per_s"], marker='o', color='green')
        ax2.set_xlabel("Output Length (tokens)")
        ax2.set_ylabel("Throughput (tokens/s)")
        ax2.set_title("Throughput vs Output Length")
        ax2.set_xscale("log", base=2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("task2_output_sweep_plots.png", dpi=300, bbox_inches='tight')
        print("✓ Plots saved to task2_output_sweep_plots.png")
        plt.close()
    
    # Cleanup
    del model, tokenizer
    clear_memory()
    
    return results

def task3_model_comparison():
    """Task 3: Compare different models"""
    print("\n" + "="*80)
    print("TASK 3: Model Comparison")
    print("="*80)
    
    input_length = 64
    output_length = 64
    
    models_to_test = [
        ("Qwen/Qwen3-1.7B", True),
        ("Qwen/Qwen3-8B", True),
        ("allenai/OLMo-7B-0724-hf", False)
    ]
    
    results = []
    
    for model_name, is_qwen3 in models_to_test:
        print(f"\nTesting {model_name}...")
        
        try:
            model, tokenizer = load_model(model_name)
            
            avg_time, avg_throughput = benchmark_generation(
                model, tokenizer, input_length, output_length, BATCH_SIZE, is_qwen3=is_qwen3
            )
            
            if avg_time is None:
                print(f"  ❌ OOM for {model_name}")
                results.append({
                    "model": model_name,
                    "time_s": "OOM",
                    "throughput_tokens_per_s": "OOM"
                })
            else:
                print(f"  ✓ Time: {avg_time:.4f}s, Throughput: {avg_throughput:.2f} tokens/s")
                results.append({
                    "model": model_name,
                    "time_s": f"{avg_time:.4f}",
                    "throughput_tokens_per_s": f"{avg_throughput:.2f}"
                })
            
            # Cleanup after each model
            del model, tokenizer
            clear_memory()
            
        except Exception as e:
            print(f"  ❌ Error for {model_name}: {e}")
            results.append({
                "model": model_name,
                "time_s": f"Error: {str(e)[:50]}",
                "throughput_tokens_per_s": "Error"
            })
            clear_memory()
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("task3_model_comparison_results.csv", index=False)
    print("\n✓ Results saved to task3_model_comparison_results.csv")
    
    return results

def main():
    """Main function to run all benchmarks"""
    print("="*80)
    print("LLM INFERENCE BENCHMARKING SUITE")
    print("="*80)
    print(f"\nGPU: {get_gpu_name()}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Dtype: {DTYPE}")
    print(f"Warmup Iterations: {NUM_WARMUP}")
    print(f"Benchmark Iterations: {NUM_ITERATIONS}")
    
    # Create results directory
    os.makedirs("benchmark_results", exist_ok=True)
    os.chdir("benchmark_results")
    
    # Run all tasks
    try:
        task1_results = task1_input_sweep()
    except Exception as e:
        print(f"\n❌ Task 1 failed: {e}")
        task1_results = None
    
    try:
        task2_results = task2_output_sweep()
    except Exception as e:
        print(f"\n❌ Task 2 failed: {e}")
        task2_results = None
    
    try:
        task3_results = task3_model_comparison()
    except Exception as e:
        print(f"\n❌ Task 3 failed: {e}")
        task3_results = None
    
    # Summary
    print("\n" + "="*80)
    print("BENCHMARKING COMPLETE")
    print("="*80)
    print("\nResults saved in './benchmark_results/' directory:")
    print("  - task1_input_sweep_results.csv")
    print("  - task1_input_sweep_plots.png")
    print("  - task2_output_sweep_results.csv")
    print("  - task2_output_sweep_plots.png")
    print("  - task3_model_comparison_results.csv")
    print("\nGPU used:", get_gpu_name())

if __name__ == "__main__":
    main()
