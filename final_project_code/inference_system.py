"""
Final Project Inference System
Optimized inference server for GraphDev, MMLU, and InfoBench tasks
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class ModelConfig:
    """Configuration for a model in the system"""
    name: str
    path: str
    max_tokens: int = 512
    quantization: Optional[str] = None  # "4bit", "8bit", or None


class TaskRouter:
    """Routes requests to appropriate models based on task complexity"""
    
    def __init__(self):
        # Patterns to identify task types
        self.graph_patterns = [
            r"directed graph",
            r"nodes.*edges",
            r"shortest path",
            r"->.*weight",
            r"node \d+ to node \d+"
        ]
        
        self.mmlu_patterns = [
            r"multiple choice",
            r"Options:\s*A\.",
            r"college_medicine",
            r"professional_medicine",
            r"The following is a.*question.*about"
        ]
        
        self.infobench_patterns = [
            r"Instruction:",
            r"Question:.*Generation:",
        ]
    
    def identify_task(self, prompt: str) -> str:
        """Identify which task type a prompt belongs to"""
        prompt_lower = prompt.lower()
        
        # Check graph patterns
        graph_matches = sum(1 for pattern in self.graph_patterns 
                          if re.search(pattern, prompt_lower, re.IGNORECASE))
        if graph_matches >= 2:
            return "graph"
        
        # Check MMLU patterns
        mmlu_matches = sum(1 for pattern in self.mmlu_patterns 
                         if re.search(pattern, prompt, re.IGNORECASE))
        if mmlu_matches >= 1:
            return "mmlu"
        
        # Check InfoBench patterns
        infobench_matches = sum(1 for pattern in self.infobench_patterns 
                               if re.search(pattern, prompt, re.IGNORECASE))
        if infobench_matches >= 1:
            return "infobench"
        
        # Default to infobench for open-ended queries
        return "infobench"
    
    def route_to_model(self, prompt: str, task: str, prompt_length: int) -> str:
        """Decide which model to use based on task and complexity"""
        # Graph tasks: always use larger model for accuracy
        if task == "graph":
            return "large"
        
        # MMLU: use large model for medical questions (they're tricky)
        if task == "mmlu":
            return "large"
        
        # InfoBench: route based on prompt length and complexity
        if task == "infobench":
            # Long prompts or complex queries -> large model
            if prompt_length > 200 or "detailed" in prompt.lower() or "comprehensive" in prompt.lower():
                return "large"
            # Short prompts -> small model for speed
            return "small"
        
        return "large"  # Default to large model


class ContinuousBatcher:
    """Implements continuous batching for efficient request processing"""
    
    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        self.pending_requests: List[Dict[str, Any]] = []
    
    def add_request(self, request_id: str, prompt: str, max_tokens: int, 
                   temperature: float, model_name: str) -> None:
        """Add a request to the batch queue"""
        self.pending_requests.append({
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model_name": model_name,
            "timestamp": time.time()
        })
    
    def get_batch(self, model_name: str) -> List[Dict[str, Any]]:
        """Get a batch of requests for a specific model"""
        # Filter requests for this model
        model_requests = [r for r in self.pending_requests 
                         if r["model_name"] == model_name]
        
        if not model_requests:
            return []
        
        # Take up to max_batch_size requests
        batch = model_requests[:self.max_batch_size]
        
        # Remove from pending
        for req in batch:
            self.pending_requests.remove(req)
        
        return batch
    
    def is_empty(self) -> bool:
        return len(self.pending_requests) == 0


class InferenceSystem:
    """Main inference system with multiple models and optimizations"""
    
    def __init__(self, 
                 large_model_path: str = "Qwen/Qwen3-8B",
                 small_model_path: str = "Qwen/Qwen3-1.7B",
                 device: str = "cuda",
                 use_8bit: bool = False,
                 use_4bit: bool = False):
        
        self.device = device
        self.router = TaskRouter()
        self.batcher = ContinuousBatcher(max_batch_size=8)
        
        print("Loading models...")
        
        # Configure quantization
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        if use_4bit:
            # 4-bit quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            load_kwargs["quantization_config"] = quantization_config
            print("Using 4-bit quantization")
        elif use_8bit:
            load_kwargs["load_in_8bit"] = True
            print("Using 8-bit quantization")
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
            print("Using full precision (bfloat16)")
        
        # Load large model
        print(f"Loading large model: {large_model_path}")
        self.large_tokenizer = AutoTokenizer.from_pretrained(large_model_path, trust_remote_code=True)
        self.large_model = AutoModelForCausalLM.from_pretrained(
            large_model_path, **load_kwargs
        )
        self.large_model.eval()
        
        # Set pad token
        if self.large_tokenizer.pad_token is None:
            self.large_tokenizer.pad_token = self.large_tokenizer.eos_token
        
        # Load small model
        print(f"Loading small model: {small_model_path}")
        self.small_tokenizer = AutoTokenizer.from_pretrained(small_model_path, trust_remote_code=True)
        self.small_model = AutoModelForCausalLM.from_pretrained(
            small_model_path, **load_kwargs
        )
        self.small_model.eval()
        
        if self.small_tokenizer.pad_token is None:
            self.small_tokenizer.pad_token = self.small_tokenizer.eos_token
        
        print("Models loaded successfully!")
        
        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0
    
    def _format_prompt_for_qwen3(self, tokenizer, prompt: str, task: str) -> str:
        """Format prompt for Qwen3 models based on task type"""
        # For Qwen3 models, use chat template with thinking disabled
        messages = [{"role": "user", "content": prompt}]
        
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            return text
        except:
            # Fallback for models without chat template
            return prompt
    
    @torch.inference_mode()
    def generate_batch(self, 
                      prompts: List[str],
                      model_size: str = "large",
                      max_new_tokens: int = 512,
                      temperature: float = 0.7,
                      task: str = "infobench") -> List[str]:
        """Generate completions for a batch of prompts"""
        
        # Select model and tokenizer
        if model_size == "large":
            model = self.large_model
            tokenizer = self.large_tokenizer
        else:
            model = self.small_model
            tokenizer = self.small_tokenizer
        
        # Format prompts
        formatted_prompts = [
            self._format_prompt_for_qwen3(tokenizer, p, task) 
            for p in prompts
        ]
        
        # Tokenize batch
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": 0.95 if temperature > 0 else 1.0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True
        }
        
        # Generate
        outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode only new tokens
        generated_texts = []
        for i, output in enumerate(outputs):
            # Get only the new tokens (excluding input)
            new_tokens = output[len(inputs.input_ids[i]):]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        # Update metrics
        self.request_count += len(prompts)
        self.total_tokens += sum(len(output) for output in outputs)
        
        return generated_texts
    
    def process_request(self, prompt: str, max_tokens: int = 512, 
                       temperature: float = 0.7) -> str:
        """Process a single request"""
        # Identify task
        task = self.router.identify_task(prompt)
        
        # Route to model
        prompt_length = len(prompt.split())
        model_size = self.router.route_to_model(prompt, task, prompt_length)
        
        # Adjust max_tokens and temperature based on task
        if task == "graph":
            max_tokens = min(max_tokens, 2048)  # Increased for graph tasks to allow full tool call
            temperature = 0.1  # Very low temperature for structured output
        elif task == "mmlu":
            max_tokens = min(max_tokens, 256)  # MMLU needs less
            temperature = 0.3  # Lower for multiple choice
        else:
            temperature = 0.7  # Default for open-ended
        
        # Generate
        results = self.generate_batch(
            [prompt],
            model_size=model_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
            task=task
        )
        
        return results[0]
    
    def process_batch(self, prompts: List[str], max_tokens: int = 512,
                     temperature: float = 0.7) -> List[str]:
        """Process a batch of requests with intelligent routing"""
        
        # Group requests by task and model
        groups = defaultdict(list)
        for idx, prompt in enumerate(prompts):
            task = self.router.identify_task(prompt)
            prompt_length = len(prompt.split())
            model_size = self.router.route_to_model(prompt, task, prompt_length)
            
            # Adjust max_tokens and temperature based on task
            if task == "graph":
                adjusted_max_tokens = min(max_tokens, 2048)  # More tokens for graph
                adjusted_temperature = 0.1  # Very low for structured output
            elif task == "mmlu":
                adjusted_max_tokens = min(max_tokens, 256)
                adjusted_temperature = 0.3
            else:
                adjusted_max_tokens = max_tokens
                adjusted_temperature = temperature
            
            groups[(model_size, task, adjusted_max_tokens, adjusted_temperature)].append((idx, prompt))
        
        # Process each group
        results = [""] * len(prompts)
        for (model_size, task, adj_max_tokens, adj_temp), group_items in groups.items():
            indices, group_prompts = zip(*group_items)
            
            group_results = self.generate_batch(
                list(group_prompts),
                model_size=model_size,
                max_new_tokens=adj_max_tokens,
                temperature=adj_temp,
                task=task
            )
            
            # Place results back in original order
            for idx, result in zip(indices, group_results):
                results[idx] = result
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_requests": self.request_count,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_request": self.total_tokens / max(1, self.request_count)
        }