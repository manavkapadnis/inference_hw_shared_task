import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import gc
import json

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def benchmark_forward_pass(model, batch_size, seq_length=256, num_warmup=3, num_trials=5):
    """
    Benchmark forward pass time for a given batch size and sequence length.
    Uses torch.cuda.Event for accurate GPU timing.
    
    Args:
        model: The model to benchmark
        batch_size: Batch size to test
        seq_length: Sequence length (default 256)
        num_warmup: Number of warmup iterations
        num_trials: Number of trials to average
    
    Returns:
        Average forward pass time in milliseconds
    """
    device = next(model.parameters()).device
    vocab_size = model.config.vocab_size
    
    # Create random input tokens
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    
    # Warmup
    print(f"  Warming up (batch_size={batch_size})...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    torch.cuda.synchronize()
    
    # Benchmark trials
    print(f"  Running {num_trials} trials...")
    times = []
    
    with torch.no_grad():
        for trial in range(num_trials):
            # Create CUDA events for timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Record start
            start_event.record()
            
            # Forward pass
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Record end
            end_event.record()
            
            # Wait for completion
            torch.cuda.synchronize()
            
            # Calculate elapsed time in milliseconds
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)
            
            print(f"    Trial {trial+1}/{num_trials}: {elapsed_time:.2f} ms")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
    
    return avg_time, std_time, times

def run_batch_size_sweep():
    """
    Main function to benchmark Qwen3-4B with different batch sizes.
    """
    print("="*80)
    print("SPECULATIVE DECODING: Forward Pass Benchmarking")
    print("="*80)
    
    model_name = "Qwen/Qwen3-4B"
    seq_length = 256
    batch_sizes = [1, 2, 4, 8, 16]
    
    print(f"\nModel: {model_name}")
    print(f"Sequence Length: {seq_length}")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✓ Model loaded")
    print()
    
    # Store results
    results = {
        "batch_sizes": [],
        "avg_times_ms": [],
        "std_times_ms": [],
        "all_trials": {}
    }
    
    # Benchmark each batch size
    for batch_size in batch_sizes:
        print(f"Benchmarking batch_size={batch_size}")
        print("-" * 40)
        
        try:
            avg_time, std_time, all_times = benchmark_forward_pass(
                model, batch_size, seq_length
            )
            
            results["batch_sizes"].append(batch_size)
            results["avg_times_ms"].append(avg_time)
            results["std_times_ms"].append(std_time)
            results["all_trials"][batch_size] = all_times
            
            print()
            clear_memory()
            
        except torch.cuda.OutOfMemoryError:
            print(f"  ❌ OOM at batch_size={batch_size}")
            print()
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
            break
    
    # Save results to JSON
    with open("forward_pass_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✓ Results saved to forward_pass_benchmark_results.json")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Batch Size':<15} {'Avg Time (ms)':<20} {'Std Dev (ms)':<15}")
    print("-" * 50)
    for bs, avg_t, std_t in zip(results["batch_sizes"], 
                                  results["avg_times_ms"], 
                                  results["std_times_ms"]):
        print(f"{bs:<15} {avg_t:<20.2f} {std_t:<15.2f}")
    
    # Create plot
    print("\nGenerating plot...")
    plt.figure(figsize=(10, 6))
    
    batch_sizes_arr = np.array(results["batch_sizes"])
    avg_times_arr = np.array(results["avg_times_ms"])
    std_times_arr = np.array(results["std_times_ms"])
    
    # Plot with error bars
    plt.errorbar(batch_sizes_arr, avg_times_arr, yerr=std_times_arr, 
                 marker='o', linewidth=2, markersize=8, capsize=5, capthick=2)
    
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Forward Pass Time (ms)', fontsize=12)
    plt.title(f'Qwen3-4B Forward Pass Time vs Batch Size (seq_len={seq_length})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(batch_sizes)
    
    # Add value labels on points
    for bs, avg_t in zip(batch_sizes_arr, avg_times_arr):
        plt.text(bs, avg_t, f'{avg_t:.1f}ms', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('forward_pass_batch_size_scaling.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved to forward_pass_batch_size_scaling.png")
    plt.close()
    
    # Calculate and print scaling analysis
    print("\n" + "="*80)
    print("SCALING ANALYSIS")
    print("="*80)
    
    if len(results["batch_sizes"]) >= 2:
        # Calculate time per sample
        print("\nTime per sample in batch:")
        print(f"{'Batch Size':<15} {'Time/Sample (ms)':<20} {'Efficiency':<15}")
        print("-" * 50)
        
        baseline_time = results["avg_times_ms"][0]  # batch_size=1
        
        for bs, avg_t in zip(results["batch_sizes"], results["avg_times_ms"]):
            time_per_sample = avg_t / bs
            efficiency = (baseline_time * bs) / avg_t * 100  # % of ideal scaling
            print(f"{bs:<15} {time_per_sample:<20.2f} {efficiency:<15.1f}%")
        
        # Calculate speedup
        print("\nSpeedup relative to batch_size=1:")
        for i, (bs, avg_t) in enumerate(zip(results["batch_sizes"], 
                                             results["avg_times_ms"])):
            if i == 0:
                continue
            speedup = baseline_time / avg_t
            ideal_speedup = 1.0  # Since we're comparing per-token time
            throughput_increase = bs / baseline_time * baseline_time / avg_t
            print(f"  Batch {bs}: {throughput_increase:.2f}x throughput increase")
    
    print("\n" + "="*80)
    print("BENCHMARKING COMPLETE")
    print("="*80)
    
    # Cleanup
    del model, tokenizer
    clear_memory()
    
    return results

if __name__ == "__main__":
    results = run_batch_size_sweep()
