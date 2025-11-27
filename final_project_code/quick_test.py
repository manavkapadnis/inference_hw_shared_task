"""
Quick Test Script
Fast validation of the inference system with sample inputs
"""

import torch
from inference_system import InferenceSystem


def test_system():
    """Quick test of the inference system"""
    
    print("=" * 70)
    print("QUICK SYSTEM TEST")
    print("=" * 70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Using CPU (will be slow)")
        device = "cpu"
    else:
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    
    print("\n" + "-" * 70)
    print("Initializing system...")
    print("-" * 70)
    
    # Initialize with small models for quick testing
    system = InferenceSystem(
        large_model_path="Qwen/Qwen3-1.7B",  # Using smaller for quick test
        small_model_path="Qwen/Qwen3-0.6B",
        device=device,
        use_8bit=False
    )
    
    print("✓ System initialized\n")
    
    # Test samples
    test_cases = [
        {
            "name": "Graph Task",
            "prompt": """You are given a directed graph with 4 nodes (numbered 0 to 3) and the following weighted edges (src -> dst, weight):
0 -> 1, weight: 5
0 -> 2, weight: 10
1 -> 3, weight: 3
2 -> 3, weight: 1

Return the top 1 shortest path(s) from node 0 to node 3 as strict JSON with keys 'paths' and 'weights'.
Example format: {"paths": [[0, 2, 3]], "weights": [11]}
Output JSON only with no extra text.""",
            "max_tokens": 256
        },
        {
            "name": "MMLU Task",
            "prompt": """The following is a multiple choice question about medicine. Strictly give output the answer in the format of "The answer is (X)" at the end.

Question: Which organ is primarily responsible for filtering blood?

Options:
A. Liver
B. Kidney
C. Heart
D. Lungs

Answer:""",
            "max_tokens": 128
        },
        {
            "name": "InfoBench Task",
            "prompt": "Instruction: Explain what photosynthesis is.\n\nResponse:",
            "max_tokens": 256
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print("-" * 70)
        
        try:
            # Identify task
            task = system.router.identify_task(test['prompt'])
            print(f"Identified task: {task}")
            
            # Generate
            response = system.process_request(
                test['prompt'],
                max_tokens=test['max_tokens'],
                temperature=0.7
            )
            
            print(f"Response: {response[:200]}...")
            print("✓ PASSED\n")
            
            results.append({
                "test": test['name'],
                "status": "PASSED",
                "task": task,
                "response_length": len(response)
            })
            
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            results.append({
                "test": test['name'],
                "status": "FAILED",
                "error": str(e)
            })
    
    # Test batch processing
    print("\nTest 4: Batch Processing")
    print("-" * 70)
    
    try:
        batch_prompts = [
            "What is 2+2?",
            "What is the capital of France?",
            "Explain gravity in one sentence."
        ]
        
        batch_results = system.process_batch(
            batch_prompts,
            max_tokens=100,
            temperature=0.7
        )
        
        print(f"Processed {len(batch_results)} prompts in batch")
        for i, (prompt, result) in enumerate(zip(batch_prompts, batch_results), 1):
            print(f"{i}. Q: {prompt}")
            print(f"   A: {result[:80]}...")
        
        print("✓ PASSED\n")
        results.append({
            "test": "Batch Processing",
            "status": "PASSED",
            "batch_size": len(batch_prompts)
        })
        
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        results.append({
            "test": "Batch Processing",
            "status": "FAILED",
            "error": str(e)
        })
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    total = len(results)
    
    for result in results:
        status_symbol = "✓" if result['status'] == 'PASSED' else "✗"
        print(f"{status_symbol} {result['test']}: {result['status']}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    # System stats
    stats = system.get_stats()
    print(f"\nSystem Statistics:")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Total Tokens: {stats['total_tokens']}")
    print(f"  Avg Tokens/Request: {stats['avg_tokens_per_request']:.2f}")
    
    print("=" * 70)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for deployment.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 70)


if __name__ == "__main__":
    test_system()
