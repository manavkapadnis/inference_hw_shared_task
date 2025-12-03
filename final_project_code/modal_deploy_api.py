# # Create the updated modal_deploy_api.py with graph parameter extraction
# """
# Advanced Modal Deployment with Task-based Inference System
# Integrated with inference_system.py approach for consistent behavior
# Includes regex-based graph parameter extraction to avoid LLM calls
# """

# import modal
# import os
# from typing import Dict, List, Any, Tuple, Optional
# import asyncio
# import time
# import torch
# from collections import defaultdict, deque
# from dataclasses import dataclass
# import re
# import heapq

# # Create Modal app
# app = modal.App("mkapadni-inference-system-system_4")

# # Define image with dependencies
# image = modal.Image.debian_slim(python_version="3.10").pip_install(
#     # Core ML libraries
#     "torch>=2.0.0",
#     "transformers>=4.40.0",
#     "accelerate>=0.20.0",
#     "sentencepiece",
#     "protobuf",
#     # Dataset and evaluation
#     "datasets",
#     "tqdm",
#     # Deployment
#     "fastapi[standard]",
#     # Testing / utilities
#     "requests",
#     "openai",
#     "python-dotenv",
#     "aiohttp",
#     # Optional enhancements
#     "bitsandbytes",
# )

# # Create volume for model caching
# volume = modal.Volume.from_name("model-cache", create_if_missing=True)


# def find_top_p_paths(edges: List[List[int]], N: int, P: int) -> List[Dict[str, Any]]:
#     """
#     Find the top P shortest paths from node 0 to node N-1 using optimized Yen's algorithm.
#     Adapted from dataset_handlers.py
#     """
#     # Build adjacency list
#     graph = {i: [] for i in range(N)}
#     for src, dst, weight in edges:
#         graph[src].append((dst, weight))
    
#     # Helper function: Optimized Dijkstra with edge blocking
#     def dijkstra(source, target, blocked_edges):
#         dist = [float('inf')] * N
#         dist[source] = 0
#         parent = [-1] * N
#         pq = [(0, source)]
#         visited = [False] * N
        
#         while pq:
#             d, u = heapq.heappop(pq)
#             if visited[u]:
#                 continue
#             visited[u] = True
#             if u == target:
#                 break
            
#             for v, edge_weight in graph[u]:
#                 if (u, v) in blocked_edges:
#                     continue
#                 new_dist = d + edge_weight
#                 if new_dist < dist[v]:
#                     dist[v] = new_dist
#                     parent[v] = u
#                     heapq.heappush(pq, (new_dist, v))
        
#         if dist[target] == float('inf'):
#             return None, float('inf')
        
#         # Reconstruct path
#         path = []
#         node = target
#         while node != -1:
#             path.append(node)
#             node = parent[node]
#         path.reverse()
#         return path, int(dist[target])
    
#     # Find P shortest paths using optimized Yen's algorithm
#     paths_found = []
    
#     # First shortest path
#     path, weight = dijkstra(0, N - 1, set())
#     if not path:
#         return []
#     paths_found.append({"path": path, "weight": weight})
    
#     if P == 1:
#         return paths_found
    
#     # Use a min-heap for candidates (weight, path_tuple)
#     candidates_heap = []
#     seen_paths = {tuple(path)}  # Track seen paths for O(1) duplicate detection
    
#     for k in range(1, P):
#         prev_path = paths_found[-1]["path"]
#         prev_len = len(prev_path)
        
#         # For each node in previous path (except last)
#         for i in range(prev_len - 1):
#             spur_node = prev_path[i]
#             root_path = prev_path[:i+1]
            
#             # Build blocked edges set
#             blocked = set()
#             for p_dict in paths_found:
#                 p_path = p_dict["path"]
#                 if len(p_path) > i and p_path[:i+1] == root_path and len(p_path) > i + 1:
#                     blocked.add((p_path[i], p_path[i+1]))
            
#             # Skip if no valid spur path exists (optimization)
#             if not blocked and i > 0:
#                 continue
            
#             # Find spur path from spur_node to target
#             spur_path, spur_weight = dijkstra(spur_node, N - 1, blocked)
#             if not spur_path:
#                 continue
            
#             # Combine root + spur
#             total_path = root_path[:-1] + spur_path
#             total_path_tuple = tuple(total_path)
            
#             # Skip if already seen
#             if total_path_tuple in seen_paths:
#                 continue
#             seen_paths.add(total_path_tuple)
            
#             # Calculate total weight incrementally
#             total_weight = 0
#             # Weight from start to spur_node
#             for j in range(i):
#                 u, v = total_path[j], total_path[j+1]
#                 for dst, w in graph[u]:
#                     if dst == v:
#                         total_weight += w
#                         break
#             # Add spur path weight
#             total_weight += spur_weight
            
#             # Push to heap
#             heapq.heappush(candidates_heap, (total_weight, total_path_tuple))
        
#         # Get next shortest path from heap
#         if not candidates_heap:
#             break
        
#         next_weight, next_path_tuple = heapq.heappop(candidates_heap)
#         paths_found.append({"path": list(next_path_tuple), "weight": next_weight})
    
#     return paths_found


# def extract_graph_params_from_prompt(prompt: str) -> Optional[Tuple[List[List[int]], int, int]]:
#     """
#     IMPROVED: Extract graph parameters using robust regex patterns
#     Returns (edges, N, P) if successful, None otherwise
#     """
#     try:
#         # Extract N (number of nodes)
#         N = None
#         n_patterns = [
#             r'graph\s+with\s+(\d+)\s+nodes?',
#             r'(\d+)\s+nodes?\s+\(numbered',
#             r'nodes?\s+numbered\s+0\s+to\s+(\d+)',
#         ]
#         for pattern in n_patterns:
#             match = re.search(pattern, prompt, re.IGNORECASE)
#             if match:
#                 N = int(match.group(1))
#                 if "numbered 0 to" in match.group(0).lower() or "numbered from 0 to" in match.group(0).lower():
#                     N = N + 1
#                 break
        
#         if N is None:
#             return None
        
#         # Extract P (number of paths)
#         P = None
#         p_patterns = [
#             r'top\s*(\d+)',
#             r'top[-\s](\d+)',
#             r'(\d+)\s+shortest',
#         ]
#         for pattern in p_patterns:
#             match = re.search(pattern, prompt, re.IGNORECASE)
#             if match:
#                 P = int(match.group(1))
#                 break
        
#         if P is None:
#             P = 1  # Default
        
#         # Extract edges - IMPROVED PATTERNS
#         edges = []
        
#         # Pattern 1: "X -> Y, weight: Z" or "X -> Y, weight Z"
#         pattern1 = r'(\d+)\s*-+>\s*(\d+),?\s*weight:?\s*(\d+)'
#         for match in re.finditer(pattern1, prompt, re.IGNORECASE):
#             src = int(match.group(1))
#             dst = int(match.group(2))
#             weight = int(match.group(3))
#             edges.append([src, dst, weight])
        
#         # Pattern 2: "(X, Y, Z)" or "[X, Y, Z]" - only if Pattern 1 failed
#         if not edges:
#             pattern2 = r'[\(\[](\d+),\s*(\d+),\s*(\d+)[\)\]]'
#             for match in re.finditer(pattern2, prompt):
#                 src = int(match.group(1))
#                 dst = int(match.group(2))
#                 weight = int(match.group(3))
#                 edges.append([src, dst, weight])
        
#         # Pattern 3: "edge (X, Y) with weight Z" - fallback
#         if not edges:
#             pattern3 = r'edge\s*\(?(\d+),?\s*(\d+)\)?\s*(?:with)?\s*weight:?\s*(\d+)'
#             for match in re.finditer(pattern3, prompt, re.IGNORECASE):
#                 src = int(match.group(1))
#                 dst = int(match.group(2))
#                 weight = int(match.group(3))
#                 edges.append([src, dst, weight])
        
#         if not edges:
#             return None
        
#         # Normalize edges
#         normalized_edges = []
#         for e in edges:
#             if isinstance(e, (list, tuple)) and len(e) >= 3:
#                 try:
#                     s, d, w = int(e[0]), int(e[1]), int(e[2])
#                     normalized_edges.append([s, d, w])
#                 except:
#                     continue
        
#         if not normalized_edges:
#             return None
        
#         return (normalized_edges, N, P)
        
#     except Exception as e:
#         print(f"[GRAPH] Regex extraction error: {e}")
#         return None


# class TaskRouter:
#     """Routes requests to appropriate models based on task complexity"""
    
#     def __init__(self):
#         # Patterns to identify task types (from inference_system.py)
#         self.graph_patterns = [
#             r"directed graph",
#             r"nodes.*edges",
#             r"shortest path",
#             r"->.*weight",
#             r"node \\d+ to node \\d+"
#         ]
        
#         self.mmlu_patterns = [
#             r"multiple choice",
#             r"Options:\\s*A\\.",
#             r"college_medicine",
#             r"professional_medicine",
#             r"The following is a.*question.*about"
#         ]
        
#         self.infobench_patterns = [
#             r"Instruction:",
#             r"Question:.*Generation:",
#         ]
    
#     def identify_task(self, prompt: str) -> str:
#         """Identify which task type a prompt belongs to"""
#         prompt_lower = prompt.lower()
        
#         # Check graph patterns
#         graph_matches = sum(1 for pattern in self.graph_patterns
#                           if re.search(pattern, prompt_lower, re.IGNORECASE))
#         if graph_matches >= 2:
#             return "graph"
        
#         # Check MMLU patterns
#         mmlu_matches = sum(1 for pattern in self.mmlu_patterns
#                          if re.search(pattern, prompt, re.IGNORECASE))
#         if mmlu_matches >= 1:
#             return "mmlu"
        
#         # Check InfoBench patterns
#         infobench_matches = sum(1 for pattern in self.infobench_patterns
#                               if re.search(pattern, prompt, re.IGNORECASE))
#         if infobench_matches >= 1:
#             return "infobench"
        
#         # Default to infobench for open-ended queries
#         return "infobench"
    
#     def route_to_model(self, prompt: str, task: str, prompt_length: int) -> str:
#         """Decide which model to use based on task and complexity"""
#         # Graph tasks: always use larger model for accuracy
#         if task == "graph":
#             return "large"
        
#         # MMLU: use large model for medical questions (they're tricky)
#         if task == "mmlu":
#             return "large"
        
#         # InfoBench: route based on prompt length and complexity
#         if task == "infobench":
#             # Long prompts or complex queries -> large model
#             if prompt_length > 200 or "detailed" in prompt.lower() or "comprehensive" in prompt.lower():
#                 return "large"
#             # Short prompts -> small model for speed
#             return "small"
        
#         return "large"  # Default to large model


# @dataclass
# class QueuedRequest:
#     """Represents a queued inference request"""
#     prompts: List[str]
#     max_tokens: int
#     temperature: float
#     task: str
#     arrival_time: float
#     timeout: float
#     future: asyncio.Future
#     prompt_indices: List[int]


# @app.cls(
#     image=image,
#     gpu=modal.gpu.A100(count=2, size="80GB"),
#     timeout=900,
#     container_idle_timeout=600,
#     volumes={"/cache": volume},
# )
# @modal.concurrent(max_inputs=300)
# class InferenceAPI:
#     """
#     Advanced LLM Inference API with:
#     - Task-specific routing (graph, mmlu, infobench)
#     - Intelligent model selection
#     - Regex-based graph parameter extraction (no LLM needed if successful)
#     - Timeout-aware processing
#     - Qwen3-specific prompt formatting
#     """
#     BATCH_DELAY = 0.25   # Wait window (sec) to accumulate more requests
#     MAX_BATCH_SIZE = 8   # Max requests per batch
    
#     @modal.enter()
#     def load_model(self):
#         """Load models on container startup"""
#         from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
#         # Get configuration
#         large_model = os.environ.get("LARGE_MODEL", "Qwen/Qwen3-14B")
#         small_model = os.environ.get("SMALL_MODEL", "Qwen/Qwen3-0.6B")
#         use_4bit = os.environ.get("USE_4BIT", "false").lower() == "true"
#         use_8bit = os.environ.get("USE_8BIT", "false").lower() == "true"
        
#         print(f"Loading models...")
#         print(f"Large: {large_model}, Small: {small_model}")
#         print(f"4-bit: {use_4bit}, 8-bit: {use_8bit}")
        
#         # Configure quantization
#         load_kwargs = {"trust_remote_code": True}
#         if use_4bit:
#             quantization_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_compute_dtype=torch.bfloat16,
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_quant_type="nf4"
#             )
#             load_kwargs["quantization_config"] = quantization_config
#         elif use_8bit:
#             load_kwargs["load_in_8bit"] = True
#         else:
#             load_kwargs["torch_dtype"] = torch.bfloat16
        
#         # Load both models on separate GPUs (for efficient GPU usage)
#         if torch.cuda.device_count() >= 2:
#             print("Loading models on both GPUs for efficient utilization")
#             # Large model primarily on GPU 0
#             self.large_tokenizer = AutoTokenizer.from_pretrained(
#                 large_model, trust_remote_code=True, cache_dir="/cache"
#             )
#             self.large_model = AutoModelForCausalLM.from_pretrained(
#                 large_model, cache_dir="/cache", device_map={"": 0}, **load_kwargs
#             )
#             self.large_model.eval()
            
#             # Small model primarily on GPU 1
#             self.small_tokenizer = AutoTokenizer.from_pretrained(
#                 small_model, trust_remote_code=True, cache_dir="/cache"
#             )
#             self.small_model = AutoModelForCausalLM.from_pretrained(
#                 small_model, cache_dir="/cache", device_map={"": 1}, **load_kwargs
#             )
#             self.small_model.eval()
#         else:
#             print("Single GPU mode")
#             self.large_tokenizer = AutoTokenizer.from_pretrained(
#                 large_model, trust_remote_code=True, cache_dir="/cache"
#             )
#             self.large_model = AutoModelForCausalLM.from_pretrained(
#                 large_model, cache_dir="/cache", device_map="auto", **load_kwargs
#             )
#             self.large_model.eval()
            
#             self.small_tokenizer = AutoTokenizer.from_pretrained(
#                 small_model, trust_remote_code=True, cache_dir="/cache"
#             )
#             self.small_model = AutoModelForCausalLM.from_pretrained(
#                 small_model, cache_dir="/cache", device_map="auto", **load_kwargs
#             )
#             self.small_model.eval()
        
#         # Set pad tokens
#         if self.large_tokenizer.pad_token is None:
#             self.large_tokenizer.pad_token = self.large_tokenizer.eos_token
#         if self.small_tokenizer.pad_token is None:
#             self.small_tokenizer.pad_token = self.small_tokenizer.eos_token
        
#         print("Models loaded successfully!")
        
#         # Initialize task router
#         self.router = TaskRouter()
        
#         # Initialize queues for each task
#         self.task_queues = {
#             "graph": deque(),
#             "mmlu": deque(),
#             "infobench": deque()
#         }
        
#         # Batch size config (can be tuned)
#         self.max_batch_sizes = {
#             "graph": 3,      # Complex, needs accuracy
#             "mmlu": 3,       # Medium complexity
#             "infobench": 1   # Variable length
#         }
        
#         # Timeout threshold (seconds before deadline to force flush)
#         self.timeout_threshold = 5.0
        
#         # GPU availability flags
#         self.gpu_busy = {0: False, 1: False}
        
#         # Background processing task
#         self.processing_task = None
#         self.shutdown_flag = False
        
#         # Stats tracking
#         self.stats = {
#             "graph_regex_success": 0,
#             "graph_llm_fallback": 0,
#             "total_requests": 0
#         }
    
#     def _format_prompt_for_qwen3(self, tokenizer, prompt: str, task: str) -> str:
#         """Format prompt for Qwen3 models based on task type"""
#         # For Qwen3 models, use chat template with thinking disabled
#         messages = [{"role": "user", "content": prompt}]
#         try:
#             text = tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True,
#                 enable_thinking=False
#             )
#             return text
#         except:
#             # Fallback for models without chat template
#             return prompt
    
#     def select_model_and_gpu(self, task: str, prompt: str) -> Tuple[Any, Any, int]:
#         """Select appropriate model and GPU based on task"""
#         prompt_length = len(prompt.split())
#         model_size = self.router.route_to_model(prompt, task, prompt_length)
        
#         if model_size == "large":
#             # Use large model on GPU 0
#             return self.large_model, self.large_tokenizer, 0
#         else:
#             # Use small model on GPU 1
#             return self.small_model, self.small_tokenizer, 1
    
#     def get_task_params(self, task: str, max_tokens: int, temperature: float) -> Tuple[int, float]:
#         """Get task-specific parameters (from inference_system.py logic)"""
#         if task == "graph":
#             # Increased for graph tasks to allow full tool call
#             adjusted_max_tokens = min(max_tokens, 2048)
#             adjusted_temperature = 0.1  # Very low temperature for structured output
#         elif task == "mmlu":
#             adjusted_max_tokens = min(max_tokens, 256)  # MMLU needs less
#             adjusted_temperature = 0.3  # Lower for multiple choice
#         else:
#             adjusted_max_tokens = max_tokens
#             adjusted_temperature = temperature  # Default for open-ended
        
#         return adjusted_max_tokens, adjusted_temperature
    
#     async def solve_graph_directly(self, prompt: str) -> Optional[str]:
#         """
#         Try to solve graph problem directly using regex extraction from prompt.
#         Returns formatted result string if successful, None if extraction fails.
#         """
#         try:
#             # Try to extract parameters from prompt
#             extracted = extract_graph_params_from_prompt(prompt)
            
#             if extracted is None:
#                 print("[GRAPH] Regex extraction failed, will use LLM fallback")
#                 self.stats["graph_llm_fallback"] += 1
#                 return None
            
#             edges, N, P = extracted
#             print(f"[GRAPH] ✓ Regex extraction successful: N={N}, P={P}, edges={len(edges)}")
#             self.stats["graph_regex_success"] += 1
            
#             # Compute solution directly
#             paths = find_top_p_paths(edges, N, P)
            
#             if not paths:
#                 print("[GRAPH] No valid paths found")
#                 return None
            
#             # Format result as structured output (like what LLM would generate)
#             result_lines = [
#                 f"Based on the graph with {N} nodes, I'll find the top {P} shortest path(s).",
#                 "",
#                 "Solution:"
#             ]
            
#             for i, path_info in enumerate(paths, 1):
#                 path = path_info["path"]
#                 weight = path_info["weight"]
#                 path_str = " -> ".join(map(str, path))
#                 result_lines.append(f"Path {i}: {path_str} (weight: {weight})")
            
#             result = "\\n".join(result_lines)
#             print(f"[GRAPH] ✓ Solution computed without LLM")
#             return result
            
#         except Exception as e:
#             print(f"[GRAPH] Error in direct solving: {e}, will use LLM fallback")
#             self.stats["graph_llm_fallback"] += 1
#             return None
    
#     async def process_batch(self, task: str, requests: List[QueuedRequest]):
#         """Process a batch of requests for a specific task"""
#         if not requests:
#             return
        
#         # Special handling for graph tasks: try regex extraction first
#         if task == "graph":
#             for req in requests:
#                 # Try to solve each graph prompt directly without LLM
#                 direct_results = []
#                 needs_llm = []
                
#                 for prompt_idx, prompt in enumerate(req.prompts):
#                     direct_result = await self.solve_graph_directly(prompt)
#                     if direct_result is not None:
#                         direct_results.append((prompt_idx, direct_result))
#                     else:
#                         needs_llm.append(prompt_idx)
                
#                 # If all prompts were solved directly, set result and skip LLM
#                 if not needs_llm:
#                     results = [""] * len(req.prompts)
#                     for prompt_idx, result in direct_results:
#                         results[prompt_idx] = result
#                     if not req.future.done():
#                         req.future.set_result(results)
#                     continue
                
#                 # If some need LLM, we'll process them below
#                 # For now, store direct results and continue with LLM for the rest
#                 if direct_results:
#                     # Filter prompts to only those needing LLM
#                     req.prompts = [req.prompts[i] for i in needs_llm]
#                     req._direct_results = direct_results  # Store for later
#                     req._original_indices = needs_llm
        
#         # Filter out requests that were fully solved
#         requests = [r for r in requests if not r.future.done()]
#         if not requests:
#             return
        
#         # Get first prompt to determine model selection
#         first_prompt = requests[0].prompts[0] if requests[0].prompts else ""
#         model, tokenizer, gpu_id = self.select_model_and_gpu(task, first_prompt)
        
#         # Wait for GPU to be free
#         while self.gpu_busy[gpu_id]:
#             await asyncio.sleep(0.01)
        
#         self.gpu_busy[gpu_id] = True
        
#         try:
#             # Collect all prompts
#             all_prompts = []
#             request_map = []  # Maps output index to (request_idx, prompt_idx_in_request)
            
#             for req_idx, req in enumerate(requests):
#                 for prompt_idx, prompt in enumerate(req.prompts):
#                     all_prompts.append(prompt)
#                     request_map.append((req_idx, prompt_idx))
            
#             # Get task-specific parameters from first request
#             base_max_tokens = requests[0].max_tokens
#             base_temperature = requests[0].temperature
#             max_tokens, temperature = self.get_task_params(task, base_max_tokens, base_temperature)
            
#             # Format prompts for Qwen3
#             formatted_prompts = [
#                 self._format_prompt_for_qwen3(tokenizer, p, task)
#                 for p in all_prompts
#             ]
            
#             # Batch tokenization
#             inputs = tokenizer(
#                 formatted_prompts,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=2048
#             ).to(model.device)
            
#             # Generation parameters (from inference_system.py)
#             gen_kwargs = {
#                 "max_new_tokens": max_tokens,
#                 "temperature": temperature,
#                 "do_sample": temperature > 0,
#                 "top_p": 0.95 if temperature > 0 else 1.0,
#                 "pad_token_id": tokenizer.pad_token_id,
#                 "eos_token_id": tokenizer.eos_token_id,
#                 "use_cache": True
#             }
            
#             # Generate
#             with torch.no_grad():
#                 outputs = model.generate(**inputs, **gen_kwargs)
            
#             # Decode only new tokens (like inference_system.py)
#             generated_texts = []
#             for i, output in enumerate(outputs):
#                 # Get only the new tokens (excluding input)
#                 new_tokens = output[len(inputs.input_ids[i]):]
#                 generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
#                 generated_texts.append(generated_text)
            
#             # Group results back to original requests
#             for req_idx, req in enumerate(requests):
#                 results = []
                
#                 # Check if this request had direct results
#                 if hasattr(req, '_direct_results'):
#                     # Initialize with empty strings for all original prompts
#                     original_count = len(req._direct_results) + len(req.prompts)
#                     results = [""] * original_count
                    
#                     # Fill in direct results
#                     for prompt_idx, result in req._direct_results:
#                         results[prompt_idx] = result
                    
#                     # Fill in LLM results
#                     for local_idx, original_idx in enumerate(req._original_indices):
#                         # Find this prompt's LLM result
#                         for i, (r_idx, p_idx) in enumerate(request_map):
#                             if r_idx == req_idx and p_idx == local_idx:
#                                 results[original_idx] = generated_texts[i]
#                                 break
#                 else:
#                     # No direct results, just use LLM results
#                     for prompt_idx in range(len(req.prompts)):
#                         # Find this prompt's result
#                         for i, (r_idx, p_idx) in enumerate(request_map):
#                             if r_idx == req_idx and p_idx == prompt_idx:
#                                 results.append(generated_texts[i])
#                                 break
                
#                 # Set result on future
#                 if not req.future.done():
#                     req.future.set_result(results)
        
#         except Exception as e:
#             print(f"Error processing batch for {task}: {e}")
#             for req in requests:
#                 if not req.future.done():
#                     req.future.set_exception(e)
        
#         finally:
#             self.gpu_busy[gpu_id] = False
    
#     async def queue_processor(self):
#         """Background task to process queues"""
#         while not self.shutdown_flag:
#             current_time = time.time()
            
#             # Check each task queue
#             for task, queue in self.task_queues.items():
#                 if not queue:
#                     continue
                
#                 max_batch_size = self.max_batch_sizes[task]
#                 batch_to_process = []
                
#                 # Check for urgent requests (near timeout)
#                 urgent_requests = []
#                 for req in queue:
#                     time_remaining = req.timeout - (current_time - req.arrival_time)
#                     if time_remaining <= self.timeout_threshold:
#                         urgent_requests.append(req)
                
#                 if urgent_requests:
#                     # Process urgent requests immediately
#                     batch_to_process = urgent_requests[:max_batch_size]
#                     for req in batch_to_process:
#                         queue.remove(req)
#                     print(f"[URGENT] Processing {len(batch_to_process)} {task} requests (near timeout)")
#                     await self.process_batch(task, batch_to_process)
                
#                 elif len(queue) >= max_batch_size:
#                     # Process full batch
#                     batch_to_process = [queue.popleft() for _ in range(max_batch_size)]
#                     print(f"[FULL] Processing {len(batch_to_process)} {task} requests")
#                     await self.process_batch(task, batch_to_process)
            
#             # Small sleep to avoid busy waiting
#             await asyncio.sleep(0.1)
    
#     @modal.web_endpoint(method="POST")
#     async def completions(self, request: dict):
#         """
#         Completions endpoint with task-based routing and inference.
#         For graph tasks, tries regex extraction first to avoid LLM calls.
#         Uses the same inference approach as inference_system.py for LLM fallback.
#         """
#         # Start background processor if not running
#         if self.processing_task is None or self.processing_task.done():
#             self.processing_task = asyncio.create_task(self.queue_processor())
        
#         # Extract parameters
#         prompt = request.get("prompt", "")
#         max_tokens = request.get("max_tokens", 512)
#         temperature = request.get("temperature", 0.7)
        
#         # Handle both single and batch
#         if isinstance(prompt, str):
#             prompts = [prompt]
#         else:
#             prompts = prompt
        
#         # Detect task from first prompt (using TaskRouter)
#         task = self.router.identify_task(prompts[0])
        
#         # Update stats
#         self.stats["total_requests"] += 1
        
#         # Create queued request
#         arrival_time = time.time()
#         timeout = 880.0  # 590 seconds (10 sec buffer before Modal timeout)
#         future = asyncio.Future()
        
#         queued_req = QueuedRequest(
#             prompts=prompts,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             task=task,
#             arrival_time=arrival_time,
#             timeout=timeout,
#             future=future,
#             prompt_indices=list(range(len(prompts)))
#         )
        
#         # Add to appropriate queue
#         self.task_queues[task].append(queued_req)
        
#         # Wait for result
#         try:
#             results = await asyncio.wait_for(future, timeout=timeout)
#         except asyncio.TimeoutError:
#             return {
#                 "error": "Request timeout",
#                 "choices": [],
#                 "model": "inference-system"
#             }
        
#         # Format response
#         choices = []
#         for i, text in enumerate(results):
#             choices.append({
#                 "text": text,
#                 "index": i,
#                 "finish_reason": "stop"
#             })
        
#         # Add stats for graph tasks
#         extra_info = {}
#         if task == "graph":
#             extra_info["graph_stats"] = {
#                 "regex_success_rate": f"{self.stats['graph_regex_success']}/{self.stats['graph_regex_success'] + self.stats['graph_llm_fallback']}",
#                 "total_graph_requests": self.stats['graph_regex_success'] + self.stats['graph_llm_fallback']
#             }
        
#         return {
#             "choices": choices,
#             "model": "inference-system",
#             "task": task,
#             **extra_info,
#             "usage": {
#                 "prompt_tokens": len(prompts) * 100,
#                 "completion_tokens": len(results) * max_tokens // 2,
#                 "total_tokens": len(prompts) * 100 + len(results) * max_tokens // 2
#             }
#         }


# # Local deployment for testing
# @app.local_entrypoint()
# def main():
#     """Local test endpoint"""
#     api = InferenceAPI()
    
#     # Test prompts for different tasks
#     test_prompts = [
#         # Graph task (should work with regex extraction)
#         "Find the top 1 shortest path from node 0 to node 3 in a directed graph with 4 nodes. Edges: 0 -> 1, weight: 5, 1 -> 3, weight: 2, 0 -> 2, weight: 10, 2 -> 3, weight: 1",
        
#         # MMLU task
#         "The following is a multiple choice question about medicine. Question: What is the most common cause of death? Options: A. Cancer B. Heart Disease C. Stroke D. Diabetes",
        
#         # InfoBench task
#         "Instruction: Write a short story about a cat."
#     ]
    
#     for prompt in test_prompts:
#         result = api.completions.remote({"prompt": prompt, "max_tokens": 256})
#         print(f"\\nPrompt: {prompt[:100]}...")
#         print(f"Task: {result.get('task')}")
#         if result.get('graph_stats'):
#             print(f"Graph Stats: {result['graph_stats']}")
#         print(f"Response: {result['choices'][0]['text'][:200]}...")


# # Write to file
# # with open('modal_deploy_api.py', 'w') as f:
# #     f.write(updated_code)

# # print("✓ Updated modal_deploy_api.py with graph parameter extraction")
# # print("\n" + "="*80)
# # print("KEY ADDITIONS:")
# # print("="*80)
# # print("""
# # ✅ NEW: extract_graph_params_from_prompt() function
# #    - Extracts edges, N, P directly from user prompt using regex
# #    - Supports multiple edge formats: "0 -> 1, weight: 5", "[0,1,5]", etc.
# #    - Validates all parameters before returning

# # ✅ NEW: find_top_p_paths() function
# #    - Full Yen's algorithm implementation (from dataset_handlers.py)
# #    - Computes shortest paths directly without LLM

# # ✅ NEW: solve_graph_directly() method
# #    - Step 1: Try regex extraction from prompt
# #    - Step 2: If successful, compute solution directly (NO LLM CALL)
# #    - Step 3: If fails, return None for LLM fallback

# # ✅ UPDATED: process_batch() for graph tasks
# #    - First attempts direct solving for all graph prompts
# #    - Only calls LLM for prompts where regex extraction failed
# #    - Combines direct results + LLM results seamlessly

# # ✅ NEW: Stats tracking
# #    - graph_regex_success: Count of direct solves
# #    - graph_llm_fallback: Count of LLM fallbacks
# #    - Returned in API response for monitoring

# # FLOW FOR GRAPH TASKS:
# # 1. User sends graph prompt
# # 2. extract_graph_params_from_prompt() tries regex extraction
# # 3. If successful → compute solution directly (NO LLM) ✓
# # 4. If fails → fallback to LLM with temp=0.1, max_tokens=2048
# # 5. LLM generates tool call, dataset_handlers.py parses it
# # """)

"""
Advanced Modal Deployment with Task-based Inference System

- Two-model deployment on 2× A100 80GB GPUs using Modal
- Dynamic online batching with per-task queues and short wait window
- Task-based routing (graph, MMLU, InfoBench-style)
- Direct graph solving via regex + Yen's algorithm (avoids LLM when possible)
- Warmup generation to avoid first-request latency
- All timeouts ≤ 600 seconds to satisfy ASG requirement
"""

import os
import time
import asyncio
import re
import heapq
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import modal
import torch

# Global CUDA / Torch settings for A100s
torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul on A100 [file:1]
torch.set_float32_matmul_precision("high")    # Prefer higher-performance kernels [file:1]
torch.backends.cudnn.benchmark = True        # Optimize kernels for input shapes [file:1]

# Create Modal app
app = modal.App("mkapadni-inference-system-system_4")  # [file:1]

# Define image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        # Core ML libraries
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.20.0",
        "sentencepiece",
        "protobuf",
        # Dataset and evaluation
        "datasets",
        "tqdm",
        # Deployment
        "fastapi[standard]",
        # Testing / utilities
        "requests",
        "openai",
        "python-dotenv",
        "aiohttp",
        # Optional enhancements
        "bitsandbytes",
    )
)

# Create volume for model caching
volume = modal.Volume.from_name("model-cache", create_if_missing=True)  # [file:1]


# =========================
# Graph utilities
# =========================

def find_top_p_paths(edges: List[List[int]], N: int, P: int) -> List[Dict[str, Any]]:
    """
    Find the top P shortest paths from node 0 to node N-1 using an optimized
    variant of Yen's algorithm. [file:1]
    """
    # Build adjacency list
    graph = {i: [] for i in range(N)}
    for src, dst, weight in edges:
        graph[src].append((dst, weight))

    def dijkstra(source: int, target: int, blocked_edges: set) -> Tuple[Optional[List[int]], float]:
        dist = [float("inf")] * N
        dist[source] = 0.0
        parent = [-1] * N
        pq = [(0.0, source)]
        visited = [False] * N

        while pq:
            d, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            if u == target:
                break
            for v, w in graph[u]:
                if (u, v) in blocked_edges:
                    continue
                new_d = d + w
                if new_d < dist[v]:
                    dist[v] = new_d
                    parent[v] = u
                    heapq.heappush(pq, (new_d, v))

        if dist[target] == float("inf"):
            return None, float("inf")

        # Reconstruct path
        path = []
        node = target
        while node != -1:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path, dist[target]

    # First shortest path
    first_path, first_weight = dijkstra(0, N - 1, set())
    if not first_path:
        return []

    paths_found = [{"path": first_path, "weight": int(first_weight)}]
    if P == 1:
        return paths_found

    candidates_heap: List[Tuple[float, Tuple[int, ...]]] = []
    seen_paths = {tuple(first_path)}

    for _ in range(1, P):
        prev_path = paths_found[-1]["path"]
        prev_len = len(prev_path)

        for i in range(prev_len - 1):
            spur_node = prev_path[i]
            root_path = prev_path[: i + 1]

            blocked = set()
            for p_dict in paths_found:
                p_path = p_dict["path"]
                if len(p_path) > i and p_path[: i + 1] == root_path and len(p_path) > i + 1:
                    blocked.add((p_path[i], p_path[i + 1]))

            if not blocked and i > 0:
                continue

            spur_path, spur_weight = dijkstra(spur_node, N - 1, blocked)
            if not spur_path:
                continue

            total_path = root_path[:-1] + spur_path
            total_path_tuple = tuple(total_path)
            if total_path_tuple in seen_paths:
                continue
            seen_paths.add(total_path_tuple)

            total_weight = 0
            for j in range(i):
                u, v = total_path[j], total_path[j + 1]
                for dst, w in graph[u]:
                    if dst == v:
                        total_weight += w
                        break
            total_weight += spur_weight

            heapq.heappush(candidates_heap, (total_weight, total_path_tuple))

        if not candidates_heap:
            break

        next_weight, next_path_tuple = heapq.heappop(candidates_heap)
        paths_found.append({"path": list(next_path_tuple), "weight": int(next_weight)})

    return paths_found


def extract_graph_params_from_prompt(
    prompt: str,
) -> Optional[Tuple[List[List[int]], int, int]]:
    """
    Try to extract (edges, N, P) from a natural language prompt using regex. [file:1]
    Returns None if extraction fails.
    """
    try:
        N: Optional[int] = None

        # Try to find number of nodes
        n_patterns = [
            r"graph\s+with\s+(\d+)\s+nodes?",
            r"(\d+)\s+nodes?\s+\(numbered",
            r"nodes?\s+numbered\s+0\s+to\s+(\d+)",
        ]
        for pattern in n_patterns:
            m = re.search(pattern, prompt, re.IGNORECASE)
            if m:
                N = int(m.group(1))
                if "numbered 0 to" in m.group(0).lower() or "numbered from 0 to" in m.group(0).lower():
                    N = N + 1
                break
        if N is None:
            return None

        # Try to find number of paths to compute
        P: Optional[int] = None
        p_patterns = [
            r"top\s*(\d+)",
            r"top[-\s](\d+)",
            r"(\d+)\s+shortest",
        ]
        for pattern in p_patterns:
            m = re.search(pattern, prompt, re.IGNORECASE)
            if m:
                P = int(m.group(1))
                break
        if P is None:
            P = 1

        edges: List[List[int]] = []

        # Format: "0 -> 1, weight: 5"
        pattern1 = r"(\d+)\s*-+>\s*(\d+),?\s*weight:?\s*(\d+)"
        for m in re.finditer(pattern1, prompt, re.IGNORECASE):
            edges.append([int(m.group(1)), int(m.group(2)), int(m.group(3))])

        # Fallback: "(0, 1, 5)" or "[0, 1, 5]"
        if not edges:
            pattern2 = r"[\(\[](\d+),\s*(\d+),\s*(\d+)[\)\]]"
            for m in re.finditer(pattern2, prompt):
                edges.append([int(m.group(1)), int(m.group(2)), int(m.group(3))])

        # Fallback: "edge (0, 1) with weight 5"
        if not edges:
            pattern3 = r"edge\s*\(?(\d+),?\s*(\d+)\)?\s*(?:with)?\s*weight:?\s*(\d+)"
            for m in re.finditer(pattern3, prompt, re.IGNORECASE):
                edges.append([int(m.group(1)), int(m.group(2)), int(m.group(3))])

        if not edges:
            return None

        normalized_edges: List[List[int]] = []
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) >= 3:
                try:
                    s, d, w = int(e[0]), int(e[1]), int(e[2])
                    normalized_edges.append([s, d, w])
                except Exception:
                    continue

        if not normalized_edges:
            return None

        return normalized_edges, N, P
    except Exception as e:
        print(f"[GRAPH] Regex extraction error: {e}")
        return None


# =========================
# Task routing
# =========================

class TaskRouter:
    """Routes requests to appropriate models based on task type and complexity."""  # [file:1]

    def __init__(self) -> None:
        # Simple patterns to identify tasks
        self.graph_patterns = [
            r"directed graph",
            r"nodes.*edges",
            r"shortest path",
            r"->.*weight",
            r"node \d+ to node \d+",
        ]
        self.mmlu_patterns = [
            r"multiple choice",
            r"Options:\s*A\.",
            r"college_medicine",
            r"professional_medicine",
            r"The following is a.*question.*about",
        ]
        self.infobench_patterns = [
            r"Instruction:",
            r"Question:.*Generation:",
        ]

    def identify_task(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        graph_matches = sum(
            1 for pattern in self.graph_patterns if re.search(pattern, prompt_lower, re.IGNORECASE)
        )
        if graph_matches >= 2:
            return "graph"

        mmlu_matches = sum(
            1 for pattern in self.mmlu_patterns if re.search(pattern, prompt, re.IGNORECASE)
        )
        if mmlu_matches >= 1:
            return "mmlu"

        infobench_matches = sum(
            1 for pattern in self.infobench_patterns if re.search(pattern, prompt, re.IGNORECASE)
        )
        if infobench_matches >= 1:
            return "infobench"

        return "infobench"

    def route_to_model(self, prompt: str, task: str, prompt_length: int) -> str:
        """
        Decide whether to use the large or small model. [file:1]
        """
        if task == "graph":
            return "large"
        if task == "mmlu":
            return "large"

        if task == "infobench":
            if (
                prompt_length > 200
                or "detailed" in prompt.lower()
                or "comprehensive" in prompt.lower()
            ):
                return "large"
            return "small"

        return "large"


# =========================
# Request data structure
# =========================

@dataclass
class QueuedRequest:
    prompts: List[str]
    max_tokens: int
    temperature: float
    task: str
    arrival_time: float
    timeout: float
    future: asyncio.Future
    prompt_indices: List[int]


# =========================
# Inference API
# =========================

@app.cls(
    image=image,
    gpu=modal.gpu.A100(count=2, size="80GB"),
    timeout=600,                # Hard cap ≤ 600s as requested [file:1]
    scaledown_window=600, # Also ≤ 600s [file:1]
    volumes={"/cache": volume},
)
@modal.concurrent(max_inputs=300)
class InferenceAPI:
    """
    Advanced LLM Inference API with:
    - Task-specific routing (graph, mmlu, infobench)
    - Dynamic online batching with short wait window
    - Direct graph solving when possible
    - Warmup generation on startup to avoid first-call latency
    """

    # Max time to wait to accumulate a batch before flushing (seconds)
    BATCH_DELAY = 0.05  # short window for online batching [file:1]

    @modal.enter()
    def load_model(self) -> None:
        """
        Load models on container startup and run a small warmup generation on
        both models to avoid latency spikes on the first user query. [file:1]
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        large_model_name = os.environ.get("LARGE_MODEL", "Qwen/Qwen3-14B")
        small_model_name = os.environ.get("SMALL_MODEL", "Qwen/Qwen3-0.6B")
        use_4bit = os.environ.get("USE_4BIT", "false").lower() == "true"
        use_8bit = os.environ.get("USE_8BIT", "false").lower() == "true"
        use_4bit = True
        use_8bit = False

        print("Loading models...")
        print(f"  Large: {large_model_name}")
        print(f"  Small: {small_model_name}")
        print(f"  4-bit: {use_4bit}, 8-bit: {use_8bit}")

        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if use_4bit:
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = qconfig
        elif use_8bit:
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        if torch.cuda.device_count() >= 2:
            print("Using 2× A100: large on GPU 0, small on GPU 1")
            self.large_tokenizer = AutoTokenizer.from_pretrained(
                large_model_name,
                trust_remote_code=True,
                cache_dir="/cache",
            )
            self.large_model = AutoModelForCausalLM.from_pretrained(
                large_model_name,
                cache_dir="/cache",
                device_map={"": 0},
                **load_kwargs,
            )
            self.large_model.eval()

            self.small_tokenizer = AutoTokenizer.from_pretrained(
                small_model_name,
                trust_remote_code=True,
                cache_dir="/cache",
            )
            self.small_model = AutoModelForCausalLM.from_pretrained(
                small_model_name,
                cache_dir="/cache",
                device_map={"": 1},
                **load_kwargs,
            )
            self.small_model.eval()
        else:
            print("Single-GPU mode (auto device_map)")
            self.large_tokenizer = AutoTokenizer.from_pretrained(
                large_model_name,
                trust_remote_code=True,
                cache_dir="/cache",
            )
            self.large_model = AutoModelForCausalLM.from_pretrained(
                large_model_name,
                cache_dir="/cache",
                device_map="auto",
                **load_kwargs,
            )
            self.large_model.eval()

            self.small_tokenizer = AutoTokenizer.from_pretrained(
                small_model_name,
                trust_remote_code=True,
                cache_dir="/cache",
            )
            self.small_model = AutoModelForCausalLM.from_pretrained(
                small_model_name,
                cache_dir="/cache",
                device_map="auto",
                **load_kwargs,
            )
            self.small_model.eval()

        if self.large_tokenizer.pad_token is None:
            self.large_tokenizer.pad_token = self.large_tokenizer.eos_token
        if self.small_tokenizer.pad_token is None:
            self.small_tokenizer.pad_token = self.small_tokenizer.eos_token

        # Initialize routing and queues
        self.router = TaskRouter()
        self.task_queues: Dict[str, deque[QueuedRequest]] = {
            "graph": deque(),
            "mmlu": deque(),
            "infobench": deque(),
        }
        # Per-task max batch sizes (tunable)
        self.max_batch_sizes: Dict[str, int] = {
            "graph": 2,
            "mmlu": 2,
            "infobench": 2,
        }

        self.timeout_threshold = 5.0  # flush when within 5s of per-request timeout [file:1]
        self.gpu_busy: Dict[int, bool] = {0: False, 1: False}
        self.processing_task: Optional[asyncio.Task] = None
        self.shutdown_flag: bool = False

        self.stats = {
            "graph_regex_success": 0,
            "graph_llm_fallback": 0,
            "total_requests": 0,
        }

        # -------- Warmup: dummy generation on both models ----------
        try:
            warmup_prompt = "Warmup for inference service."
            for model, tokenizer, task in [
                (self.large_model, self.large_tokenizer, "infobench"),
                (self.small_model, self.small_tokenizer, "infobench"),
            ]:
                formatted = self._format_prompt_for_qwen3(tokenizer, warmup_prompt, task)
                inputs = tokenizer(
                    [formatted],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64,
                ).to(model.device)
                with torch.inference_mode():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        temperature=0.0,
                        do_sample=False,
                        use_cache=True,
                    )
            print("Warmup generation completed for both models.")
        except Exception as e:
            print(f"Warmup generation failed (non-fatal): {e}")

    # -------- Helper methods --------

    def _format_prompt_for_qwen3(self, tokenizer, prompt: str, task: str) -> str:
        """
        Format prompt for Qwen3 chat-style models; falls back to raw text if
        chat templates are unavailable. [file:1]
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            return prompt

    def _select_model_and_gpu(
        self, task: str, prompt: str
    ) -> Tuple[Any, Any, int]:
        prompt_length = len(prompt.split())
        model_size = self.router.route_to_model(prompt, task, prompt_length)
        if model_size == "large":
            return self.large_model, self.large_tokenizer, 0
        else:
            return self.small_model, self.small_tokenizer, 1

    def _get_task_params(
        self, task: str, max_tokens: int, temperature: float
    ) -> Tuple[int, float]:
        if task == "graph":
            return min(max_tokens, 2048), 0.1
        if task == "mmlu":
            return min(max_tokens, 256), 0.3
        return max_tokens, temperature

    async def _solve_graph_directly(self, prompt: str) -> Optional[str]:
        """
        Try to solve a graph problem via regex + shortest-path algorithm
        instead of calling the LLM. [file:1]
        """
        try:
            extracted = extract_graph_params_from_prompt(prompt)
            if extracted is None:
                self.stats["graph_llm_fallback"] += 1
                return None

            edges, N, P = extracted
            print(f"[GRAPH] Regex extraction OK: N={N}, P={P}, edges={len(edges)}")
            paths = find_top_p_paths(edges, N, P)
            if not paths:
                print("[GRAPH] No valid paths found.")
                self.stats["graph_llm_fallback"] += 1
                return None

            self.stats["graph_regex_success"] += 1
            lines = [
                f"Based on the graph with {N} nodes, here are the top {P} shortest path(s):",
                "",
                "Solution:",
            ]
            for idx, info in enumerate(paths, 1):
                path_str = " -> ".join(map(str, info["path"]))
                lines.append(f"Path {idx}: {path_str} (weight: {info['weight']})")
            print("[GRAPH] Solved via direct algorithm (no LLM).")
            return "\n".join(lines)
        except Exception as e:
            print(f"[GRAPH] Direct solver error, using LLM fallback: {e}")
            self.stats["graph_llm_fallback"] += 1
            return None

    async def _process_batch(self, task: str, requests: List[QueuedRequest]) -> None:
        """
        Process a batch of requests for a given task, including dynamic online
        batching and direct-graph solving where possible. [file:1]
        """
        if not requests:
            return

        # Graph: attempt direct solution first per request
        if task == "graph":
            remaining_requests: List[QueuedRequest] = []
            for req in requests:
                direct_results: Dict[int, str] = {}
                needs_llm_indices: List[int] = []

                for idx, p in enumerate(req.prompts):
                    direct = await self._solve_graph_directly(p)
                    if direct is not None:
                        direct_results[idx] = direct
                    else:
                        needs_llm_indices.append(idx)

                if not needs_llm_indices:
                    # Entire request solved without LLM
                    ordered = [direct_results[i] for i in range(len(req.prompts))]
                    if not req.future.done():
                        req.future.set_result(ordered)
                else:
                    # Partially solved; keep the rest for LLM processing
                    new_prompts = [req.prompts[i] for i in needs_llm_indices]
                    req._direct_results = direct_results
                    req._original_indices = needs_llm_indices
                    req.prompts = new_prompts
                    remaining_requests.append(req)

            # Nothing left to send to LLM
            if not remaining_requests:
                return

            requests = remaining_requests

        # Select model based on first prompt
        first_prompt = requests[0].prompts[0] if requests[0].prompts else ""
        model, tokenizer, gpu_id = self._select_model_and_gpu(task, first_prompt)

        # Wait for target GPU to be free
        while self.gpu_busy[gpu_id]:
            await asyncio.sleep(0.005)
        self.gpu_busy[gpu_id] = True

        try:
            # Flatten prompts across all queued requests
            all_prompts: List[str] = []
            request_map: List[Tuple[int, int]] = []
            for r_idx, req in enumerate(requests):
                for p_idx, p in enumerate(req.prompts):
                    all_prompts.append(p)
                    request_map.append((r_idx, p_idx))

            base_max_tokens = requests[0].max_tokens
            base_temperature = requests[0].temperature
            max_tokens, temperature = self._get_task_params(
                task, base_max_tokens, base_temperature
            )

            # Format prompts for Qwen3
            formatted = [
                self._format_prompt_for_qwen3(tokenizer, p, task) for p in all_prompts
            ]

            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0.0,
                "top_p": 0.95 if temperature > 0.0 else 1.0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }

            with torch.inference_mode():
                outputs = model.generate(**inputs, **gen_kwargs)

            generated_texts: List[str] = []
            for i, output in enumerate(outputs):
                input_len = inputs.input_ids[i].shape[0]
                new_tokens = output[input_len:]
                gen_text = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
                generated_texts.append(gen_text)

            # Reassemble outputs per original request
            for req_idx, req in enumerate(requests):
                if hasattr(req, "_direct_results"):
                    # Graph mixed case: some prompts solved by algorithm
                    total_count = len(req._direct_results) + len(req.prompts)
                    combined = [""] * total_count

                    # Fill in algorithmic results
                    for idx, text in req._direct_results.items():
                        combined[idx] = text

                    # Fill in LLM results
                    for local_idx, original_idx in enumerate(req._original_indices):
                        # Find flattened index for this (req_idx, local_idx)
                        for flat_i, (r_i, p_i) in enumerate(request_map):
                            if r_i == req_idx and p_i == local_idx:
                                combined[original_idx] = generated_texts[flat_i]
                                break
                    if not req.future.done():
                        req.future.set_result(combined)
                else:
                    # Non-graph or fully LLM-based graph request
                    out: List[str] = []
                    for p_idx in range(len(req.prompts)):
                        for flat_i, (r_i, p_i) in enumerate(request_map):
                            if r_i == req_idx and p_i == p_idx:
                                out.append(generated_texts[flat_i])
                                break
                    if not req.future.done():
                        req.future.set_result(out)
        except Exception as e:
            print(f"Error in batch processing for task={task}: {e}")
            for req in requests:
                if not req.future.done():
                    req.future.set_exception(e)
        finally:
            self.gpu_busy[gpu_id] = False

    async def _queue_processor(self) -> None:
        """
        Background task that performs dynamic online batching:
        - Flushes "urgent" requests close to their per-request timeout
        - Batches up to per-task max size
        - If not full, flushes once the head request has waited ≥ BATCH_DELAY
        """
        while not self.shutdown_flag:
            now = time.time()
            for task, queue in self.task_queues.items():
                if not queue:
                    continue

                max_batch = self.max_batch_sizes[task]
                batch: List[QueuedRequest] = []

                # 1. Urgent requests: close to timeout
                urgent: List[QueuedRequest] = []
                for req in list(queue):
                    time_left = req.timeout - (now - req.arrival_time)
                    if time_left <= self.timeout_threshold:
                        urgent.append(req)
                if urgent:
                    for req in urgent[:max_batch]:
                        queue.remove(req)
                        batch.append(req)
                    print(f"[URGENT] Processing {len(batch)} {task} requests near timeout")
                else:
                    # 2. Full batch
                    if len(queue) >= max_batch:
                        for _ in range(max_batch):
                            batch.append(queue.popleft())
                        print(f"[FULL] Processing {len(batch)} {task} requests")
                    else:
                        # 3. Dynamic flush after short wait window
                        head = queue[0]
                        waited = now - head.arrival_time
                        if waited >= self.BATCH_DELAY:
                            flush_size = min(len(queue), max_batch)
                            for _ in range(flush_size):
                                batch.append(queue.popleft())
                            print(
                                f"[DELAY] Processing {len(batch)} {task} requests "
                                f"after waiting {waited:.3f}s"
                            )

                if batch:
                    await self._process_batch(task, batch)

            await asyncio.sleep(0.01)

    # =========================
    # Web endpoint
    # =========================

    @modal.web_endpoint(method="POST")
    async def completions(self, request: dict) -> Dict[str, Any]:
        """
        Completions endpoint with task-based routing and dynamic online batching.
        All per-request waits are capped to < 600 seconds. [file:1]
        """
        # Start background processor lazily
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._queue_processor())

        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 512)
        temperature = request.get("temperature", 0.7)

        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

        task = self.router.identify_task(prompts[0])
        self.stats["total_requests"] += 1

        arrival_time = time.time()
        # Per-request timeout: keep some headroom under 600s function timeout
        request_timeout = 590.0  # seconds [file:1]

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        queued = QueuedRequest(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            task=task,
            arrival_time=arrival_time,
            timeout=request_timeout,
            future=fut,
            prompt_indices=list(range(len(prompts))),
        )

        self.task_queues[task].append(queued)

        try:
            results = await asyncio.wait_for(fut, timeout=request_timeout)
        except asyncio.TimeoutError:
            return {
                "error": "Request timeout",
                "choices": [],
                "model": "inference-system",
            }

        choices = []
        for i, txt in enumerate(results):
            choices.append(
                {
                    "text": txt,
                    "index": i,
                    "finish_reason": "stop",
                }
            )

        extra_info: Dict[str, Any] = {}
        if task == "graph":
            total_graph = self.stats["graph_regex_success"] + self.stats["graph_llm_fallback"]
            regex_success_rate = (
                f"{self.stats['graph_regex_success']}/{total_graph}" if total_graph > 0 else "0/0"
            )
            extra_info["graph_stats"] = {
                "regex_success_rate": regex_success_rate,
                "total_graph_requests": total_graph,
            }

        return {
            "choices": choices,
            "model": "inference-system",
            "task": task,
            **extra_info,
            "usage": {
                # Rough placeholders; replace with real token accounting if needed
                "prompt_tokens": len(prompts) * 100,
                "completion_tokens": len(results) * max_tokens // 2,
                "total_tokens": len(prompts) * 100 + len(results) * max_tokens // 2,
            },
        }


# =========================
# Local entrypoint for testing
# =========================

@app.local_entrypoint()
def main() -> None:
    """
    Simple local test harness; in actual deployment you will hit the
    /completions web endpoint. [file:1]
    """
    api = InferenceAPI()

    test_prompts = [
        # Graph task (should be solved by regex + algorithm)
        "Find the top 1 shortest path from node 0 to node 3 in a directed graph with 4 nodes. "
        "Edges: 0 -> 1, weight: 5, 1 -> 3, weight: 2, 0 -> 2, weight: 10, 2 -> 3, weight: 1",
        # MMLU-like task
        "The following is a multiple choice question about medicine. "
        "Question: What is the most common cause of death? "
        "Options: A. Cancer B. Heart Disease C. Stroke D. Diabetes",
        # InfoBench-style task
        "Instruction: Write a short story about a cat.",
    ]

    for p in test_prompts:
        result = api.completions.remote({"prompt": p, "max_tokens": 256})
        print("\nPrompt:", p[:120], "...")
        print("Task:", result.get("task"))
        if result.get("graph_stats"):
            print("Graph stats:", result["graph_stats"])
        print("Response:", result["choices"][0]["text"][:200], "...")
