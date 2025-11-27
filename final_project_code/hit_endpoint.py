"""
Test the deployed Modal endpoint
"""

import requests
import json
import time


# Update this with your Modal username after deployment
MODAL_USERNAME = "your-modal-username"
URL = f"https://{MODAL_USERNAME}--mkapadni-system-1-model-completions.modal.run"


def test_single_request():
    """Test single prompt inference"""
    print("Testing single request...")
    print("-" * 50)
    
    response = requests.post(
        URL,
        json={
            "prompt": "What is the capital of France?",
            "max_tokens": 100,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: SUCCESS")
        print(f"Response: {result['choices'][0]['text']}")
        print(f"Tokens: {result['usage']['total_tokens']}")
        print(f"Time: {result['metadata']['elapsed_time']:.2f}s")
    else:
        print(f"Status: FAILED ({response.status_code})")
        print(f"Error: {response.text}")
    
    print()


def test_batch_request():
    """Test batch inference"""
    print("Testing batch request...")
    print("-" * 50)
    
    prompts = [
        "What is 2 + 2?",
        "Explain photosynthesis in one sentence.",
        "What is the speed of light?"
    ]
    
    response = requests.post(
        URL,
        json={
            "prompt": prompts,
            "max_tokens": 100,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: SUCCESS")
        print(f"Processed {len(result['choices'])} prompts")
        print(f"Total tokens: {result['usage']['total_tokens']}")
        print(f"Time: {result['metadata']['elapsed_time']:.2f}s")
        print(f"Throughput: {len(prompts) / result['metadata']['elapsed_time']:.2f} req/s")
        
        for i, choice in enumerate(result['choices']):
            print(f"\nPrompt {i+1}: {prompts[i]}")
            print(f"Response: {choice['text'][:100]}...")
    else:
        print(f"Status: FAILED ({response.status_code})")
        print(f"Error: {response.text}")
    
    print()


def test_graph_task():
    """Test with a graph pathfinding task"""
    print("Testing graph task...")
    print("-" * 50)
    
    graph_prompt = """You are given a directed graph with 5 nodes (numbered 0 to 4) and the following weighted edges (src -> dst, weight):
0 -> 1, weight: 10
0 -> 2, weight: 5
1 -> 3, weight: 1
2 -> 1, weight: 3
2 -> 3, weight: 9
3 -> 4, weight: 2

Return the top 1 shortest path(s) from node 0 to node 4 as strict JSON with keys 'paths' and 'weights'.
Example format: {"paths": [[0, 2, 4]], "weights": [10]}
Output JSON only with no extra text."""
    
    response = requests.post(
        URL,
        json={
            "prompt": graph_prompt,
            "max_tokens": 512,
            "temperature": 0.3
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: SUCCESS")
        print(f"Response:\n{result['choices'][0]['text']}")
        print(f"Tokens: {result['usage']['total_tokens']}")
        print(f"Time: {result['metadata']['elapsed_time']:.2f}s")
    else:
        print(f"Status: FAILED ({response.status_code})")
        print(f"Error: {response.text}")
    
    print()


def test_mmlu_task():
    """Test with an MMLU medical question"""
    print("Testing MMLU task...")
    print("-" * 50)
    
    mmlu_prompt = """The following is a multiple choice question about medicine. Strictly give output the answer in the format of "The answer is (X)" at the end.

Question: Which of the following is a common symptom of dehydration?

Options:
A. Increased urination
B. Dark urine
C. Excessive sweating
D. High blood pressure

Answer:"""
    
    response = requests.post(
        URL,
        json={
            "prompt": mmlu_prompt,
            "max_tokens": 128,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: SUCCESS")
        print(f"Response: {result['choices'][0]['text']}")
        print(f"Tokens: {result['usage']['total_tokens']}")
        print(f"Time: {result['metadata']['elapsed_time']:.2f}s")
    else:
        print(f"Status: FAILED ({response.status_code})")
        print(f"Error: {response.text}")
    
    print()


def test_concurrent_requests(n_requests: int = 10):
    """Test concurrent requests"""
    import concurrent.futures
    
    print(f"Testing {n_requests} concurrent requests...")
    print("-" * 50)
    
    def send_request(i):
        start = time.time()
        response = requests.post(
            URL,
            json={
                "prompt": f"Tell me a fact about the number {i}.",
                "max_tokens": 50
            }
        )
        elapsed = time.time() - start
        return response.status_code, elapsed
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_requests) as executor:
        results = list(executor.map(send_request, range(n_requests)))
    
    total_time = time.time() - start_time
    
    success_count = sum(1 for status, _ in results if status == 200)
    avg_latency = sum(elapsed for _, elapsed in results) / len(results)
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Successful requests: {success_count}/{n_requests}")
    print(f"Throughput: {n_requests / total_time:.2f} req/s")
    print(f"Average latency: {avg_latency:.2f}s")
    print()


def main():
    print("=" * 50)
    print("Testing Modal Inference Endpoint")
    print("=" * 50)
    print(f"URL: {URL}")
    print()
    
    try:
        # Run tests
        test_single_request()
        test_batch_request()
        test_graph_task()
        test_mmlu_task()
        test_concurrent_requests(10)
        
        print("=" * 50)
        print("All tests completed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to update MODAL_USERNAME in this file!")


if __name__ == "__main__":
    main()
