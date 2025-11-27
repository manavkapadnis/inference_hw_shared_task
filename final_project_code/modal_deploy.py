"""
Modal Deployment for Final Project Inference System
Deploys the inference system as a REST API with OpenAI-compatible interface
"""

import modal

# Define app name (replace with your Andrew ID)
app = modal.App("mkapadni-system-1")

# Define the image with required dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers>=4.40.0",
    "torch>=2.0.0",
    "accelerate>=0.20.0",
    "fastapi[standard]",
    "sentencepiece",
    "protobuf",
)


@app.cls(
    image=image,
    gpu="A100-80GB:2",  # 2x A100 80GB as specified
    timeout=600,  # 10 minute timeout
    container_idle_timeout=300,  # Keep warm for 5 minutes
)
class Model:
    """
    Main model class for inference
    Supports concurrent requests up to 300
    """
    
    @modal.enter()
    def load_model(self):
        """Load models on container startup"""
        import sys
        sys.path.append("/root")
        
        from inference_system import InferenceSystem
        
        print("Initializing inference system...")
        self.inference_system = InferenceSystem(
            large_model_path="Qwen/Qwen3-8B",
            small_model_path="Qwen/Qwen3-1.7B",
            device="cuda",
            use_8bit=False  # Use full precision for accuracy
        )
        print("Inference system ready!")
    
    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 512, 
                temperature: float = 0.7) -> str:
        """Generate completion for a single prompt"""
        return self.inference_system.process_request(
            prompt, max_tokens=max_tokens, temperature=temperature
        )
    
    @modal.method()
    def generate_batch(self, prompts: list, max_tokens: int = 512,
                      temperature: float = 0.7) -> list:
        """Generate completions for a batch of prompts"""
        return self.inference_system.process_batch(
            prompts, max_tokens=max_tokens, temperature=temperature
        )
    
    @modal.web_endpoint(method="POST", docs=True)
    async def completions(self, request: dict):
        """
        OpenAI-compatible completions endpoint
        
        Accepts either:
        - Single prompt: {"prompt": "text"}
        - Batch of prompts: {"prompt": ["text1", "text2", ...]}
        
        Optional parameters:
        - max_tokens: maximum tokens to generate (default: 512)
        - temperature: sampling temperature (default: 0.7)
        """
        import time
        
        # Extract parameters
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 512)
        temperature = request.get("temperature", 0.7)
        
        start_time = time.time()
        
        # Handle both single and batch requests
        if isinstance(prompt, str):
            prompts = [prompt]
            is_batch = False
        else:
            prompts = prompt
            is_batch = True
        
        # Generate completions
        generated_texts = self.inference_system.process_batch(
            prompts, max_tokens=max_tokens, temperature=temperature
        )
        
        # Create OpenAI-style response
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for i, generated_text in enumerate(generated_texts):
            # Estimate token counts (rough approximation)
            prompt_tokens = len(prompts[i].split()) * 1.3  # ~1.3 tokens per word
            completion_tokens = len(generated_text.split()) * 1.3
            
            choices.append({
                "text": generated_text,
                "index": i,
                "finish_reason": "stop",
                "logprobs": None
            })
            
            total_prompt_tokens += int(prompt_tokens)
            total_completion_tokens += int(completion_tokens)
        
        elapsed_time = time.time() - start_time
        
        # Return OpenAI-style response
        response = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "mkapadni-system-1",
            "choices": choices,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            },
            "metadata": {
                "elapsed_time": elapsed_time,
                "requests": len(prompts)
            }
        }
        
        return response


# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """Test the deployment locally"""
    model = Model()
    
    # Test single request
    print("Testing single request...")
    result = model.generate.remote("What is 2+2?")
    print(f"Result: {result}\n")
    
    # Test batch request
    print("Testing batch request...")
    results = model.generate_batch.remote([
        "What is the capital of France?",
        "Explain photosynthesis in one sentence."
    ])
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result}")
