import json
import re
import modal
import asyncio
import time
import torch
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter

app = modal.App("mkapadni-system-1")

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen3-14B",
    "use_4bit": False,
    "use_8bit": False,
    "gpu_type": "A100-80GB:2",
    "max_concurrent": 300,
    
    # Max tokens per task
    "infobench_max_tokens": 512,
    "mmlu_max_tokens": 64,
    "graph_max_tokens": 2048,
    
    # Batch settings
    "batch_wait_time": 2,
    "max_batch_size": 32,
    "max_wait_time": 10,
}

# Image with required dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "accelerate",
    "fastapi[standard]",
    "numpy",
    "bitsandbytes",
)

@dataclass
class BatchRequest:
    """Represents a single request in the batch queue"""
    prompt: str
    max_tokens: int
    result_future: asyncio.Future
    enqueue_time: float = field(default_factory=time.time)

@dataclass
class TaskQueue:
    """Queue for a specific task type"""
    requests: List[BatchRequest] = field(default_factory=list)
    processing_gpus: set = field(default_factory=set)
    last_process_time: float = 0.0

@app.cls(
    image=image,
    gpu="A100-80GB:2",
    startup_timeout=300,
    scaledown_window=600,
    timeout=600,
)
@modal.concurrent(max_inputs=300)
class Model:
    @modal.enter()
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        model_name = DEFAULT_CONFIG["model_name"]
        use_4bit = DEFAULT_CONFIG["use_4bit"]
        use_8bit = DEFAULT_CONFIG["use_8bit"]
        
        print("=" * 80)
        print("SYSTEM INITIALIZATION - DIRECT ANSWERS")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Quantization: {'4-bit' if use_4bit else '8-bit' if use_8bit else 'FP16/BF16'}")
        print(f"Strategy: Greedy + Direct Answers (No Thinking) + MBR")
        print(f"GPUs: 2x A100-80GB")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization config
        quantization_config = None
        if use_4bit:
            print("Loading with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif use_8bit:
            print("Loading with 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Base model kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Load models on both GPUs
        print(f"[GPU 0] Loading model...")
        self.model_gpu0 = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": "cuda:0"},
            **model_kwargs
        )
        
        print(f"[GPU 1] Loading model...")
        self.model_gpu1 = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": "cuda:1"},
            **model_kwargs
        )
        
        self.models = [self.model_gpu0, self.model_gpu1]
        self.devices = [torch.device("cuda:0"), torch.device("cuda:1")]
        
        print(f"✓ Model 0 loaded on: cuda:0")
        print(f"✓ Model 1 loaded on: cuda:1")
        
        # Initialize dynamic batching queues
        self.task_queues = {
            "graph": TaskQueue(),
            "mmlu": TaskQueue(),
            "infobench": TaskQueue()
        }
        
        # Batching configuration
        self.batch_wait_time = DEFAULT_CONFIG["batch_wait_time"]
        self.max_batch_size = DEFAULT_CONFIG["max_batch_size"]
        self.max_wait_time = DEFAULT_CONFIG["max_wait_time"]
        
        # GPU assignment counter for round-robin
        self.gpu_counters = {"graph": 0, "mmlu": 0, "infobench": 0}
        
        print("\n" + "=" * 80)
        print("✓ System Ready")
        print("=" * 80 + "\n")
    
    def _route_task(self, text: str) -> str:
        """Route task type with lightweight classification"""
        import torch
        
        routing_prompt = f"""You are a task classifier. Given a user prompt, classify it into exactly ONE of these categories:

1. "graph" - Questions about graphs, shortest paths, nodes, edges
2. "mmlu" - Multiple choice questions with options A, B, C, D  
3. "infobench" - Open-ended questions requiring detailed explanations

Respond with ONLY the category name and nothing else.

User prompt: {text}

Category:"""
        
        inputs = self.tokenizer(routing_prompt, return_tensors="pt", truncation=True).to(self.devices[0])
        
        with torch.no_grad():
            outputs = self.models[0].generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
            )
        
        decoded = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        ).strip().lower()
        
        if "graph" in decoded:
            return "graph"
        elif "mmlu" in decoded:
            return "mmlu"
        else:
            return "infobench"
    
    async def _maybe_process_batch(self, task_type: str):
        """Check if we should process a batch and do so if needed"""
        queue = self.task_queues[task_type]
        
        if not queue.requests:
            return
        
        current_time = time.time()
        queue_size = len(queue.requests)
        time_since_last = current_time - queue.last_process_time
        oldest_request_age = current_time - queue.requests[0].enqueue_time if queue.requests else 0
        
        # Process conditions
        should_process = (
            queue_size >= self.max_batch_size or
            (queue_size > 0 and time_since_last >= self.batch_wait_time) or
            oldest_request_age >= self.max_wait_time
        )
        
        if not should_process:
            return
        
        # Select a GPU that's not already busy
        available_gpus = [i for i in range(len(self.models)) if i not in queue.processing_gpus]
        
        if not available_gpus:
            return
        
        gpu_idx = available_gpus[self.gpu_counters[task_type] % len(available_gpus)]
        self.gpu_counters[task_type] += 1
        
        # Extract batch
        batch_size = min(len(queue.requests), self.max_batch_size)
        batch = queue.requests[:batch_size]
        queue.requests = queue.requests[batch_size:]
        queue.last_process_time = current_time
        
        print(f"[GPU {gpu_idx}] Processing {task_type} batch: {len(batch)} requests, {len(queue.requests)} remaining")
        
        queue.processing_gpus.add(gpu_idx)
        
        try:
            prompts = [req.prompt for req in batch]
            max_tokens_list = [req.max_tokens for req in batch]
            
            if task_type == "mmlu":
                results = self._infer_mmlu_batch(prompts, max_tokens_list, gpu_idx)
            elif task_type == "graph":
                results = self._infer_graph_batch(prompts, max_tokens_list, gpu_idx)
            else:  # infobench
                results = self._infer_infobench_batch(prompts, max_tokens_list, gpu_idx)
            
            for req, result in zip(batch, results):
                if not req.result_future.done():
                    req.result_future.set_result(result)
                    
        except Exception as e:
            print(f"Error processing batch for {task_type} on GPU {gpu_idx}: {e}")
            import traceback
            traceback.print_exc()
            error_result = (f"[ERROR] {str(e)}", 0, 0)
            for req in batch:
                if not req.result_future.done():
                    req.result_future.set_result(error_result)
        finally:
            queue.processing_gpus.discard(gpu_idx)
            
            if queue.requests:
                asyncio.create_task(self._maybe_process_batch(task_type))
    
    # async def _process_batch(self, batch: List[BatchRequest], task_type: str, gpu_idx: int):
    #     """Process a batch on a specific GPU"""
    #     try:
    #         prompts = [req.prompt for req in batch]
    #         max_tokens_list = [req.max_tokens for req in batch]
            
    #         # Run inference based on task type
    #         if task_type == "graph":
    #             results = self._infer_graph_batch(prompts, max_tokens_list, gpu_idx)
    #         elif task_type == "mmlu":
    #             results = self._infer_mmlu_batch(prompts, max_tokens_list, gpu_idx)
    #         else:  # infobench
    #             results = self._infer_infobench_batch(prompts, max_tokens_list, gpu_idx)
            
    #         # Set results
    #         for req, result in zip(batch, results):
    #             req.result_future.set_result(result)
        
    #     except Exception as e:
    #         print(f"[GPU {gpu_idx}] Batch processing error: {e}")
    #         for req in batch:
    #             if not req.result_future.done():
    #                 req.result_future.set_exception(e)
        
    #     finally:
    #         # Release GPU
    #         queue = self.task_queues[task_type]
    #         queue.processing_gpus.discard(gpu_idx)
            
    #         # Try to process next batch immediately
    #         await self._maybe_process_batch(task_type)
    
    async def _add_to_queue_and_wait(self, prompt: str, max_tokens: int, task_type: str) -> Tuple[str, int, int]:
        """Add request to queue and wait for result"""
        loop = asyncio.get_event_loop()
        result_future = loop.create_future()
        
        batch_req = BatchRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            result_future=result_future
        )
        
        queue = self.task_queues[task_type]
        queue.requests.append(batch_req)
        
        # Trigger processing checks
        asyncio.create_task(self._maybe_process_batch(task_type))
        asyncio.create_task(self._maybe_process_batch(task_type))
        
        # Periodic checks
        async def periodic_check():
            for _ in range(150):
                await asyncio.sleep(0.2)
                if result_future.done():
                    break
                await self._maybe_process_batch(task_type)
        
        asyncio.create_task(periodic_check())
        
        # Wait for result with timeout
        try:
            return await asyncio.wait_for(result_future, timeout=600.0)
        except asyncio.TimeoutError:
            return (f"[ERROR] Request timeout after 600s", 0, 0)
    
    # =============================================================================
    # ANSWER EXTRACTION HELPERS
    # =============================================================================
    
    def _extract_answer_from_response(self, text: str, task_type: str) -> str:
        """
        Extract clean answer from model response using Answer: keyword
        
        Args:
            text: Raw model output
            task_type: "mmlu" or "infobench"
        
        Returns:
            Extracted answer or original text if extraction fails
        """
        # Try to extract content after "Answer:" keyword
        answer_patterns = [
            r'Answer:\s*(.+?)(?:\n|$)',  # Answer: <content> until newline or end
            r'Answer:\s*(.+)',             # Answer: <content> until end
            r'answer:\s*(.+?)(?:\n|$)',    # Lowercase variant
            r'answer:\s*(.+)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                
                # For MMLU, extract single letter choice
                if task_type == "mmlu":
                    # Look for A, B, C, D (possibly with parentheses or periods)
                    choice_match = re.search(r'[(\[]?([A-Da-d])[)\]]?', answer)
                    if choice_match:
                        return choice_match.group(1).upper()
                    # If just a letter at the start
                    if answer and answer[0].upper() in 'ABCD':
                        return answer[0].upper()
                
                return answer
        
        # Fallback: if no "Answer:" found, try to extract last meaningful content
        if task_type == "mmlu":
            # Look for last occurrence of A/B/C/D
            choices = re.findall(r'\b([A-Da-d])\b', text)
            if choices:
                return choices[-1].upper()
        
        # Return original if nothing works
        return text.strip()
    
    def _parse_mmlu_response(self, text: str) -> str:
        """
        Parse MMLU response to extract choice (A, B, C, or D)
        
        Args:
            text: Raw model output
        
        Returns:
            Single letter choice or original text
        """
        return self._extract_answer_from_response(text, "mmlu")
    
    def _parse_infobench_response(self, text: str) -> str:
        """
        Parse InfoBench response to extract detailed answer
        
        Args:
            text: Raw model output
        
        Returns:
            Extracted answer or original text
        """
        # For InfoBench with direct answer, just return the text as-is
        # since we're not asking for "Answer:" prefix anymore
        return text.strip()
    
    # =============================================================================
    # GRAPH PROCESSING - REGEX + YEN'S ALGORITHM
    # =============================================================================
    
    def submit_paths(self, edges, source, target, p):
        """Compute top-p shortest simple paths using Yen's algorithm."""
        from collections import defaultdict
        import heapq

        adj = defaultdict(list)
        nodes_set = set()
        for u, v, w in edges:
            adj[u].append((v, w))
            nodes_set.add(u); nodes_set.add(v)

        def dijkstra(src, dst, banned_nodes=set(), banned_edges=set()):
            heap = [(0, src, [src])]
            visited_cost = {}
            while heap:
                cost, node, path = heapq.heappop(heap)
                if node == dst:
                    return path, cost
                if node in visited_cost and visited_cost[node] <= cost:
                    continue
                visited_cost[node] = cost
                for (nbr, w) in adj.get(node, []):
                    if nbr in banned_nodes:
                        continue
                    if (node, nbr) in banned_edges:
                        continue
                    if nbr in path:
                        continue
                    heapq.heappush(heap, (cost + w, nbr, path + [nbr]))
            return None, None

        A = []
        first_path, first_cost = dijkstra(source, target)
        if first_path is None:
            return {"paths": [], "weights": []}
        A.append((first_path, first_cost))

        B = []
        for k in range(1, p):
            prev_path = A[-1][0]
            for i in range(len(prev_path) - 1):
                spur_node = prev_path[i]
                root_path = prev_path[:i+1]
                banned_edges = set()
                banned_nodes = set(root_path[:-1])
                for (path_k, _) in A:
                    if len(path_k) > i and path_k[:i+1] == root_path:
                        banned_edges.add((path_k[i], path_k[i+1]))
                spur_path, _ = dijkstra(spur_node, target, banned_nodes, banned_edges)
                if spur_path is not None:
                    total_path = root_path[:-1] + spur_path
                    total_cost = sum(
                        next((w for nbr, w in adj[u] if nbr == v), 10**9)
                        for u, v in zip(total_path[:-1], total_path[1:])
                    )
                    heapq.heappush(B, (total_cost, tuple(total_path)))
            if not B:
                break
            while B:
                cost, path_tuple = heapq.heappop(B)
                path = list(path_tuple)
                if not any(p == path for p, _ in A):
                    A.append((path, cost))
                    break
            else:
                break
        return {"paths": [p for p, _ in A[:p]], "weights": [int(c) for _, c in A[:p]]}
    
    def extract_tool_call(self, text):
        """Extract tool call arguments from LLM output."""
        try:
            pattern = r'<tool_call>\s*\{.*?"arguments"\s*:\s*(\{.*?\})\s*\}\s*</tool_call>'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                args_str = match.group(1)
                args = json.loads(args_str)
                return args
            return None
        except:
            return None
    
    def _parse_graph_with_regex(self, prompt: str) -> Tuple[Optional[List], Optional[int], Optional[int], Optional[int]]:
        """
        Robust regex-based graph parsing
        Returns: (edges, source, target, p) or (None, None, None, None) if parsing fails
        """
        try:
            # Flexible edge pattern matching
            edge_patterns = [
                r'(\d+)\s*->\s*(\d+)\s*,?\s*weight:?\s*(\d+)',  # "1 -> 2, weight: 5"
                r'(\d+)\s*->\s*(\d+)\s*:\s*(\d+)',              # "1 -> 2: 5"
                r'(\d+)\s*->\s*(\d+)\s+(\d+)',                  # "1 -> 2 5"
            ]
            
            edges = []
            for pattern in edge_patterns:
                edge_matches = re.findall(pattern, prompt)
                if edge_matches:
                    edges = [[int(u), int(v), int(w)] for u, v, w in edge_matches]
                    break
            
            if not edges:
                return None, None, None, None
            
            # Parse source node
            source_patterns = [
                r'from node[s]?\s+(\d+)',
                r'source[s]?\s*:?\s*(\d+)',
                r'start[ing]*\s*(?:at|from|node)?\s*(\d+)',
            ]
            source = None
            for pattern in source_patterns:
                source_match = re.search(pattern, prompt, re.IGNORECASE)
                if source_match:
                    source = int(source_match.group(1))
                    break
            
            if source is None:
                source = 0  # Default
            
            # Parse target node
            target_patterns = [
                r'to node[s]?\s+(\d+)',
                r'target[s]?\s*:?\s*(\d+)',
                r'end[ing]*\s*(?:at|node)?\s*(\d+)',
                r'destination[s]?\s*:?\s*(\d+)',
            ]
            target = None
            for pattern in target_patterns:
                target_match = re.search(pattern, prompt, re.IGNORECASE)
                if target_match:
                    target = int(target_match.group(1))
                    break
            
            if target is None:
                # Default to max node in edges
                target = max(max(u, v) for u, v, w in edges)
            
            # Parse p (number of paths)
            p_patterns = [
                r'top[-\s]?(\d+)\s+shortest',
                r'(\d+)\s+shortest\s+path',
                r'find\s+(\d+)\s+path',
                r'k\s*=\s*(\d+)',
                r'p\s*=\s*(\d+)',
            ]
            p = 5  # Default
            for pattern in p_patterns:
                p_match = re.search(pattern, prompt, re.IGNORECASE)
                if p_match:
                    p = int(p_match.group(1))
                    break
            
            return edges, source, target, p
            
        except Exception as e:
            return None, None, None, None
    
    def _infer_graph_batch(self, prompts: List[str], max_tokens_list: List[int], gpu_idx: int) -> List[Tuple[str, int, int]]:
        """Batch process graph requests (regex first, LLM fallback)"""
        import torch
        
        results = []
        llm_needed = []
        batch_start = time.time()
        print(f"[GPU {gpu_idx}] Graphdev: {len(prompts)} prompts, regex first, LLM fallback")
        # First pass: try regex for all
        for idx, prompt in enumerate(prompts):
            edges, source, target, p = self._parse_graph_with_regex(prompt)
            
            if edges is not None:
                result = self.submit_paths(edges, source, target, p)
                try:
                    paths_repr = result["paths"]
                    weights_repr = result["weights"]
                    out_text = f"submit_paths(paths={paths_repr}, weights={weights_repr})"
                    out_tokenized = self.tokenizer(out_text, return_tensors="pt", truncation=True).input_ids
                    completion_tokens = out_tokenized.shape[1]
                    prompt_tokens = len(self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids[0])
                    results.append((out_text, prompt_tokens, completion_tokens))
                except Exception as e:
                    results.append((f"[ERROR] submit_paths execution failure: {str(e)}", 0, 0))
            else:
                llm_needed.append(idx)
                results.append(None)
        
        # Second pass: batch LLM calls for failures
        if llm_needed:
            llm_prompts = [prompts[i] for i in llm_needed]
            llm_results = self._batch_graph_llm(llm_prompts, [max_tokens_list[i] for i in llm_needed], gpu_idx)
            
            for idx, result in zip(llm_needed, llm_results):
                results[idx] = result
        
        batch_time = time.time() - batch_start
        print(f"[GPU {gpu_idx}]: Graphdev COMPLETE (Regex parsing) in {batch_time:.2f}s")
        
        return results
    
    def _batch_graph_llm(self, prompts: List[str], max_tokens_list: List[int], gpu_idx: int) -> List[Tuple[str, int, int]]:
        """Batch LLM-based graph parsing (fallback)"""
        import torch
        
        model = self.models[gpu_idx]
        device = self.devices[gpu_idx]
        
        results = []
        
        for prompt in prompts:
            llm_prompt = """You will extract the graph specification from the user's prompt and return a JSON object representing a function call of the following shape:

{
  "name": "parse_edges",
  "arguments": {
     "edges": [[u,v,w], ...],
     "source": <int>,
     "target": <int>,
     "p": <int>
  }
}

Requirements:
- edges must be an array of triples [u, v, w] with integers
- source, target, p must be integers
- Output EXACTLY the JSON object above and nothing else.

""" + prompt

            tool_schema = [
                {
                    "type": "function",
                    "function": {
                        "name": "parse_edges",
                        "description": "Parse graph edges",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "edges": {"type": "array"},
                                "source": {"type": "integer"},
                                "target": {"type": "integer"},
                                "p": {"type": "integer"}
                            },
                            "required": ["edges", "source", "target", "p"]
                        }
                    }
                }
            ]

            messages = [{"role": "user", "content": llm_prompt}]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tools=tool_schema, 
                tool_choice={"type": "function", "function": {"name": "parse_edges"}},
                enable_thinking=False,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids, 
                    max_new_tokens=DEFAULT_CONFIG["graph_max_tokens"],
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            args = self.extract_tool_call(decoded)
            
            if args is None:
                results.append((f"[ERROR] Could not parse graph specification", len(input_ids[0]), 0))
                continue

            try:
                edges = [[int(e[0]), int(e[1]), int(e[2])] for e in args.get("edges", [])]
                source = int(args.get("source"))
                target = int(args.get("target"))
                p = int(args.get("p"))
                
                result = self.submit_paths(edges, source, target, p)
                paths_repr = result["paths"]
                weights_repr = result["weights"]
                out_text = f"submit_paths(paths={paths_repr}, weights={weights_repr})"
                out_tokenized = self.tokenizer(out_text, return_tensors="pt", truncation=True).input_ids
                completion_tokens = out_tokenized.shape[1]
                results.append((out_text, len(input_ids[0]), completion_tokens))
            except Exception as e:
                results.append((f"[ERROR] {str(e)}", len(input_ids[0]), 0))
        
        return results
    
    # =============================================================================
    # MMLU - CONFIDENCE-BASED SELECTION
    # =============================================================================
    
    def _infer_mmlu_batch(self, prompts: List[str], max_tokens_list: List[int], gpu_idx: int) -> List[Tuple[str, int, int]]:
        """Batched MMLU inference with semantic similarity + confidence pruning."""

        model = self.models[gpu_idx]
        device = self.devices[gpu_idx]

        batch_start = time.time()
        print(f"[GPU {gpu_idx}] MMLU: {len(prompts)} prompts, Semantic Similarity + Confidence Pruning")

        try:
            # Format prompts
            formatted_prompts = [
                "You are an expert across multiple medical academic domains."
                "Read the question + answer choices and output ONLY the single letter choice (A, B, C, or D) and nothing else.\n\n"
                f"{p}\n\nAnswer:"
                for p in prompts
            ]

            encoded = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(device)
            input_lengths = [len(ids) for ids in encoded.input_ids]

            # Generate multiple candidates
            n_candidates = 3
            max_new_tokens = 64

            with torch.no_grad():
                all_outputs = model.generate(
                    **encoded,
                    temperature=0.4,
                    do_sample=True,
                    top_p=0.9,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=n_candidates,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            results = []
            batch_size = len(prompts)

            for batch_idx in range(batch_size):
                input_len = input_lengths[batch_idx]
                start_idx = batch_idx * n_candidates
                end_idx = start_idx + n_candidates
                batch_outputs = all_outputs.sequences[start_idx:end_idx]

                candidates = []
                candidate_tokens = []
                candidate_logprobs = []

                # Extract candidates and compute confidence scores
                for seq_idx, output in enumerate(batch_outputs):
                    decoded = self.tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
                    candidates.append(decoded)
                    candidate_tokens.append(len(output) - input_len)

                    # Compute average log probability as confidence
                    if hasattr(all_outputs, 'scores') and all_outputs.scores:
                        logprobs = []
                        for score in all_outputs.scores:
                            # scores is tuple of (batch_size, vocab_size) for each generation step
                            if seq_idx < score.shape[0]:
                                logprobs.append(torch.nn.functional.log_softmax(score[seq_idx], dim=-1).max().item())
                        avg_logprob = np.mean(logprobs) if logprobs else 0.0
                    else:
                        avg_logprob = 0.0

                    candidate_logprobs.append(avg_logprob)

                # Extract letter choices
                def extract_choice(text):
                    import re
                    m = re.search(r'[Tt]he answer is \(?([A-D])\)?', text)
                    if not m:
                        m = re.search(r'\(([A-D])\)', text)
                    if not m:
                        m = re.search(r'\b([A-D])\b', text.upper())
                    if m:
                        return m.group(1).upper()
                    if text and text[0].upper() in 'ABCD':
                        return text[0].upper()
                    return None

                choices = [extract_choice(c) for c in candidates]
                valid_choices = [ch for ch in choices if ch is not None]

                # OPTION 1: Confidence-Based Selection (Faster for MMLU - single letter choice)
                if valid_choices:
                    # For MMLU, just pick the choice with highest confidence
                    best_idx = int(np.argmax(candidate_logprobs))
                    final_answer = extract_choice(candidates[best_idx])
                    if not final_answer:
                        final_answer = Counter(valid_choices).most_common(1)[0][0] if valid_choices else "A"
                else:
                    final_answer = "A"

                total_completion = sum(candidate_tokens)
                results.append((final_answer, input_len, total_completion))

            batch_time = time.time() - batch_start
            print(f"[GPU {gpu_idx}] MMLU COMPLETE (Confidence Pruning) in {batch_time:.2f}s")

            return results

        except Exception as e:
            print(f"[GPU {gpu_idx}] MMLU ERROR: {e}")
            import traceback
            traceback.print_exc()
            return [(f"[ERROR] {str(e)}", 0, 0) for _ in prompts]



    # =============================================================================
    # INFOBENCH - DIRECT ANSWER (NO THINKING)
    # =============================================================================
    
    def _infer_infobench_batch(self, prompts: List[str], max_tokens_list: List[int], gpu_idx: int) -> List[Tuple[str, int, int]]:
        """Batched InfoBench inference with semantic embedding similarity + AMBR-style pruning."""
        

        model = self.models[gpu_idx]
        device = self.devices[gpu_idx]

        batch_start = time.time()
        print(f"[GPU {gpu_idx}] InfoBench: {len(prompts)} prompts, Semantic Similarity + Adaptive Pruning")

        try:
            base_prompts = [
                "You are a knowledgeable assistant. Provide a detailed and precise answer to the instruction below. "
                "Focus on accuracy, clarity, and completeness. Address all components of the question systematically.\n\n"
                f"{p}\n\n"
                "Answer:"
                for p in prompts
            ]

            # Apply chat template if available
            messages_list = [[{"role": "user", "content": bp}] for bp in base_prompts]

            try:
                texts = [
                    self.tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for msgs in messages_list
                ]
            except:
                texts = base_prompts

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(device)

            input_lens = inputs["attention_mask"].sum(dim=1).tolist()

            # Generate candidates
            n_candidates = 3
            max_new_tokens = 512

            with torch.no_grad():
                all_outputs = model.generate(
                    **inputs,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=n_candidates,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            results = []
            batch_size = len(prompts)

            for batch_idx in range(batch_size):
                input_len = input_lens[batch_idx]
                start_idx = batch_idx * n_candidates
                end_idx = start_idx + n_candidates
                batch_outputs = all_outputs.sequences[start_idx:end_idx]

                candidates = []
                candidate_tokens = []
                candidate_embeddings = []

                # Extract candidates and compute embeddings
                for output in batch_outputs:
                    decoded = self.tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
                    candidates.append(decoded)
                    candidate_tokens.append(len(output) - input_len)

                # OPTION 2: Semantic Similarity Scoring (Better for InfoBench - longer responses)
                # Use lightweight embedding pooling instead of expensive n-gram computation
                def get_embedding(text):
                    """Get lightweight embedding by tokenizing and averaging token embeddings."""
                    if not text or len(text.split()) < 2:
                        return np.zeros(768)  # Default embedding size

                    # Tokenize
                    tokens = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)

                    # Get embeddings from model (use last hidden state)
                    with torch.no_grad():
                        outputs = model(**tokens, output_hidden_states=True)
                        # Average pooling over sequence
                        hidden = outputs.hidden_states[-1]
                        mask = tokens['attention_mask'].unsqueeze(-1).expand(hidden.size()).float()
                        sum_hidden = (hidden * mask).sum(1)
                        len_hidden = mask.sum(1)
                        embedding = sum_hidden / len_hidden
                    return embedding.cpu().numpy()[0]

                # Compute embeddings for candidates (cached)
                for cand in candidates:
                    emb = get_embedding(cand)
                    candidate_embeddings.append(emb)

                # Compute pairwise semantic similarity using cosine distance
                def cosine_similarity(v1, v2):
                    """Fast cosine similarity."""
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 == 0 or norm2 == 0:
                        return 0.0
                    return np.dot(v1, v2) / (norm1 * norm2)

                # MBR-style scoring: select candidate with highest average similarity to others
                if len(candidates) > 1:
                    mbr_scores = []
                    for i, emb_i in enumerate(candidate_embeddings):
                        score = sum(cosine_similarity(emb_i, emb_j) 
                                for j, emb_j in enumerate(candidate_embeddings) if i != j)
                        mbr_scores.append(score)
                    best_idx = int(np.argmax(mbr_scores))
                else:
                    best_idx = 0

                best_candidate = candidates[best_idx]
                total_completion = sum(candidate_tokens)
                results.append((best_candidate, input_len, total_completion))

            batch_time = time.time() - batch_start
            print(f"[GPU {gpu_idx}] InfoBench COMPLETE (Semantic Similarity) in {batch_time:.2f}s")

            return results

        except Exception as e:
            print(f"[GPU {gpu_idx}] InfoBench ERROR: {e}")
            import traceback
            traceback.print_exc()
            return [(f"[ERROR] {str(e)}", 0, 0) for _ in prompts]



    # =============================================================================
    # API ENDPOINT WITH ANSWER EXTRACTION
    # =============================================================================
    
    @modal.fastapi_endpoint(method="POST")
    async def completions(self, request: dict):
        """OpenAI-style completions endpoint with dynamic batching and answer extraction"""
        body_prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 128)
        extract_answer = request.get("extract_answer", True)
        
        if isinstance(body_prompt, str):
            prompts = [body_prompt]
        else:
            prompts = list(body_prompt)
        
        # Route all prompts
        tasks = [self._route_task(p) for p in prompts]
        
        # Add to queues and collect futures with task types
        futures_with_tasks = []
        for prompt, task_type in zip(prompts, tasks):
            future = self._add_to_queue_and_wait(prompt, max_tokens, task_type)
            futures_with_tasks.append((future, task_type))
        
        # Wait for all results
        results = await asyncio.gather(*[f for f, _ in futures_with_tasks])
        
        # Format response with optional answer extraction
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for i, ((text_out, ptoks, ctoks), (_, task_type)) in enumerate(zip(results, futures_with_tasks)):
            # Extract clean answer if requested
            if extract_answer and task_type in ["mmlu", "infobench"]:
                if task_type == "mmlu":
                    extracted = self._parse_mmlu_response(text_out)
                else:  # infobench - direct answer, no parsing needed
                    extracted = self._parse_infobench_response(text_out)
                
                choices.append({
                    "text": extracted,
                    "prompt": prompts[i],  # NEW: Add original prompt
                    "index": i,            # NEW: Index within the batch (0, 1, 2, ...)
                    "finish_reason": "stop",
                })
            else:
                choices.append({
                    "text": text_out,
                    "prompt": prompts[i],  # NEW: Add original prompt
                    "index": i,            # NEW: Index within the batch (0, 1, 2, ...)
                    "finish_reason": "stop",
                })
            
            total_prompt_tokens += int(ptoks or 0)
            total_completion_tokens += int(ctoks or 0)
        
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "manav-accuracy-system",
            "choices": choices,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            }
        }