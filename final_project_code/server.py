import json
import re
import modal
import asyncio
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter

app = modal.App("rithviks-1")

# image with required deps
image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "accelerate",
    "fastapi[standard]",
    "numpy",
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
    scaledown_window=600,
    timeout=600
)

@modal.concurrent(max_inputs=300)
class Model:
    @modal.enter()
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        model_name = "Qwen/Qwen3-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # CRITICAL: Set padding side to left for decoder-only models
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load separate models on each GPU
        self.model_gpu0 = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": "cuda:0"},
            torch_dtype=torch.bfloat16,
        )
        
        self.model_gpu1 = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": "cuda:1"},
            torch_dtype=torch.bfloat16,
        )
        
        self.models = [self.model_gpu0, self.model_gpu1]
        self.devices = [torch.device("cuda:0"), torch.device("cuda:1")]
        
        print(f"Model 0 loaded on: cuda:0")
        print(f"Model 1 loaded on: cuda:1")
        
        # Initialize dynamic batching queues
        self.task_queues = {
            "graph": TaskQueue(),
            "mmlu": TaskQueue(),
            "infobench": TaskQueue()
        }
        
        # IMPROVED batching configuration
        self.batch_wait_time = 5  # 4 second wait time
        self.max_batch_size = 16  # Smaller for faster turnaround
        self.max_wait_time = 10  # Force process after 10s regardless of batch size
        
        # GPU assignment counter for round-robin
        self.gpu_counters = {"graph": 0, "mmlu": 0, "infobench": 0}

    async def _maybe_process_batch(self, task_type: str):
        """Check if we should process a batch and do so if needed"""
        queue = self.task_queues[task_type]
        
        # Early exit if no requests
        if not queue.requests:
            return
        
        current_time = time.time()
        queue_size = len(queue.requests)
        time_since_last = current_time - queue.last_process_time
        oldest_request_age = current_time - queue.requests[0].enqueue_time if queue.requests else 0
        
        # FIXED: Process if ANY of these conditions are met:
        should_process = (
            queue_size >= self.max_batch_size or  # Batch is full
            (queue_size > 0 and time_since_last >= self.batch_wait_time) or  # Some time has passed
            oldest_request_age >= self.max_wait_time  # Request is getting old (prevents timeout)
        )
        
        if not should_process:
            return
        
        # Select a GPU that's not already busy with this task
        available_gpus = [i for i in range(len(self.models)) if i not in queue.processing_gpus]
        
        if not available_gpus:
            return
        
        # Round-robin among available GPUs for this task type
        gpu_idx = available_gpus[self.gpu_counters[task_type] % len(available_gpus)]
        self.gpu_counters[task_type] += 1
        
        # Mark this GPU as processing for this task type
        queue.processing_gpus.add(gpu_idx)
        
        # Extract batch (take up to max_batch_size)
        batch_size = min(len(queue.requests), self.max_batch_size)
        batch = queue.requests[:batch_size]
        queue.requests = queue.requests[batch_size:]
        queue.last_process_time = current_time
        
        print(f"[GPU {gpu_idx}] Processing {task_type} batch: {len(batch)} requests, {len(queue.requests)} remaining in queue")
        
        try:
            # Process the batch
            prompts = [req.prompt for req in batch]
            max_tokens_list = [req.max_tokens for req in batch]
            
            if task_type == "mmlu":
                results = self._infer_mmlu_batch(prompts, max_tokens_list, gpu_idx)
            elif task_type == "graph":
                results = self._infer_graph_batch(prompts, max_tokens_list, gpu_idx)
            else:  # infobench
                results = self._infer_infobench_batch(prompts, max_tokens_list, gpu_idx)
            
            # Set results for each request
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
            # Mark this GPU as done processing
            queue.processing_gpus.discard(gpu_idx)
            
            # If there are still requests in queue, trigger another batch immediately
            if queue.requests:
                asyncio.create_task(self._maybe_process_batch(task_type))

    async def _add_to_queue_and_wait(self, prompt: str, max_tokens: int, task_type: str) -> Tuple[str, int, int]:
        """Add request to queue and wait for result with timeout"""
        loop = asyncio.get_event_loop()
        result_future = loop.create_future()
        
        batch_req = BatchRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            result_future=result_future
        )
        
        queue = self.task_queues[task_type]
        queue.requests.append(batch_req)
        
        # Trigger processing checks immediately and at intervals
        asyncio.create_task(self._maybe_process_batch(task_type))
        asyncio.create_task(self._maybe_process_batch(task_type))  # Try both GPUs
        
        # Schedule periodic checks
        async def periodic_check():
            for _ in range(10):  # Check 10 times over 2 seconds
                await asyncio.sleep(0.2)
                if result_future.done():
                    break
                await self._maybe_process_batch(task_type)
                await self._maybe_process_batch(task_type)
        
        asyncio.create_task(periodic_check())
        
        # Wait for result with timeout
        try:
            return await asyncio.wait_for(result_future, timeout=600.0)
        except asyncio.TimeoutError:
            return (f"[ERROR] Request timeout after 600s", 0, 0)

    def _route_task(self, text: str) -> str:
        """
        Use the LLM to intelligently route tasks by identifying the task type.
        Returns: "graph", "mmlu", or "infobench"
        """
        import torch
        
        routing_prompt = f"""You are a task classifier. Given a user prompt, classify it into exactly ONE of these categories:

1. "graph" - Questions about graphs, shortest paths, nodes, edges, directed/undirected graphs, path finding
2. "mmlu" - Multiple choice questions with options labeled A, B, C, D (medical or general knowledge MCQs)
3. "infobench" - Open-ended questions requiring detailed explanations or answers

Respond with ONLY the category name (graph, mmlu, or infobench) and nothing else.

User prompt: {text}

Category:"""
        
        # Use GPU 0 for routing (lightweight task)
        inputs = self.tokenizer(routing_prompt, return_tensors="pt", truncation=True).to(self.devices[0])
        
        with torch.no_grad():
            outputs = self.models[0].generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
            )
        
        decoded = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        
        if "graph" in decoded:
            return "graph"
        elif "mmlu" in decoded:
            return "mmlu"
        else:
            return "infobench"

    # -----------------------
    # Yen's K-shortest simple paths
    # -----------------------
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
        B = []
        first_path, first_cost = dijkstra(source, target)
        if first_path is None:
            return {"paths": [], "weights": []}
        A.append((first_path, first_cost))

        for k in range(1, p):
            for i in range(len(A[-1][0]) - 1):
                root_path = A[-1][0][: i + 1]
                spur_node = root_path[-1]

                banned_edges = set()
                banned_nodes = set(root_path[:-1])

                for path_k, cost_k in A:
                    if len(path_k) > i and path_k[: i + 1] == root_path:
                        banned_edges.add((path_k[i], path_k[i + 1]))

                spur_path, spur_cost = dijkstra(spur_node, target, banned_nodes=banned_nodes, banned_edges=banned_edges)
                if spur_path is not None:
                    total_path = root_path[:-1] + spur_path
                    total_cost = 0
                    for u_idx, v_idx in zip(total_path[:-1], total_path[1:]):
                        w_found = None
                        for (a, b, wt) in edges:
                            if a == u_idx and b == v_idx:
                                w_found = wt
                                break
                        if w_found is None:
                            w_found = 10**9
                        total_cost += w_found
                    candidate = (total_path, total_cost)
                    heapq.heappush(B, (candidate[1], candidate))

            if not B:
                break
            while B:
                cost_cand, cand = heapq.heappop(B)
                if cand not in A:
                    A.append(cand)
                    break
            else:
                break

        paths = [p_[0] for p_ in A[:p]]
        weights = [p_[1] for p_ in A[:p]]
        return {"paths": paths, "weights": weights}

    # -----------------------
    # IMPROVED MMLU with Conditional Self-Consistency
    # -----------------------
    def _infer_mmlu_batch(self, prompts: List[str], max_tokens_list: List[int], gpu_idx: int) -> List[Tuple[str, int, int]]:
        """
        Batched MMLU inference with conditional self-consistency.
        Uses majority voting only when confidence is low.
        """
        import torch
        import re
        import torch.nn.functional as F
        
        model = self.models[gpu_idx]
        device = self.devices[gpu_idx]
        
        model_prompts = [
            "You are a precise medical multiple-choice assistant. "
            "Read the question + answer choices and output ONLY the single letter choice (A, B, C, or D) and nothing else.\n\n"
            f"{p}\n\nAnswer:"
            for p in prompts
        ]
        
        # Tokenize with LEFT padding
        inputs = self.tokenizer(
            model_prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=2048
        ).to(device)
        
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        
        # First pass: greedy decode with confidence scores
        with torch.no_grad():
            greedy_outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.0,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        greedy_sequences = greedy_outputs.sequences
        greedy_scores = greedy_outputs.scores
        
        # Calculate confidence for each batch item
        results = []
        low_confidence_indices = []
        
        for batch_idx in range(len(prompts)):
            input_len = input_lens[batch_idx]
            output = greedy_sequences[batch_idx]
            
            # Decode greedy result
            decoded = self.tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            
            # Extract answer
            m = re.search(r"[Tt]he answer is \(?([A-D])\)?", decoded)
            if not m:
                m = re.search(r"\(([A-D])\)", decoded)
            if not m:
                m = re.search(r"\b([A-D])\b", decoded.upper())                                
            if m:
                greedy_answer = m.group(1)
            else:
                greedy_answer = decoded[:1].upper() if decoded and decoded[0].upper() in "ABCD" else "A"
            
            # Calculate confidence from first token probability
            if greedy_scores and len(greedy_scores) > 0:
                first_token_logits = greedy_scores[0][batch_idx]
                first_token_probs = F.softmax(first_token_logits, dim=-1)
                max_prob = first_token_probs.max().item()
            else:
                max_prob = 0.0
            
            completion_tokens = len(output) - input_len
            
            # Use self-consistency if confidence is low (< 0.85 threshold)
            if max_prob < 0.85:
                low_confidence_indices.append(batch_idx)
                results.append(None)  # Placeholder for low-confidence results
            else:
                results.append((greedy_answer, input_len, completion_tokens))
        
        # Second pass: self-consistency for low-confidence items
        if low_confidence_indices:
            n_samples = 5
            low_conf_inputs = {
                'input_ids': inputs['input_ids'][low_confidence_indices],
                'attention_mask': inputs['attention_mask'][low_confidence_indices]
            }
            
            with torch.no_grad():
                sampled_outputs = model.generate(
                    **low_conf_inputs,
                    max_new_tokens=20,
                    temperature=0.4,
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=n_samples,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Process sampled results
            for i, batch_idx in enumerate(low_confidence_indices):
                input_len = input_lens[batch_idx]
                
                # Extract samples for this item
                start_idx = i * n_samples
                end_idx = start_idx + n_samples
                batch_outputs = sampled_outputs[start_idx:end_idx]
                
                answers = []
                total_completion_tokens = 0
                
                for output in batch_outputs:
                    decoded = self.tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
                    total_completion_tokens += len(output) - input_len
                    
                    # Extract letter answer
                    m = re.search(r"[Tt]he answer is \(?([A-D])\)?", decoded)
                    if not m:
                        m = re.search(r"\(([A-D])\)", decoded)
                    if not m:
                        m = re.search(r"\b([A-D])\b", decoded.upper())                    
                    if m:
                        answers.append(m.group(1))
                    if not m:
                        first_char = decoded[:1].upper()
                        if first_char in "ABCD":
                            answers.append(first_char)
                
                # Majority vote
                if answers:
                    answer_counts = Counter(answers)
                    final_answer = answer_counts.most_common(1)[0][0]
                else:
                    final_answer = "A"
                
                avg_completion_tokens = total_completion_tokens // n_samples
                results[batch_idx] = (final_answer, input_len, avg_completion_tokens)
        
        return results

    def _infer_infobench_batch(self, prompts: List[str], max_tokens_list: List[int], gpu_idx: int) -> List[Tuple[str, int, int]]:
        """Batched InfoBench inference with MBR decoding."""
        import torch
        import numpy as np
        
        model = self.models[gpu_idx]
        device = self.devices[gpu_idx]
        
        base_prompts = [
            "You are a helpful assistant. Provide a clear, detailed answer to the question below.\n\n"
            f"{p}\n\nAnswer:"
            for p in prompts
        ]
        
        messages_list = [[{"role": "user", "content": bp}] for bp in base_prompts]
        
        texts = [
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for msgs in messages_list
        ]
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(device)
        
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        
        n_candidates = 3
        max_new_tokens = min(max(max_tokens_list), 512)
        
        with torch.no_grad():
            all_outputs = model.generate(
                **inputs,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                max_new_tokens=256,
                num_return_sequences=n_candidates,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        results = []
        batch_size = len(prompts)
        
        for batch_idx in range(batch_size):
            input_len = input_lens[batch_idx]
            
            start_idx = batch_idx * n_candidates
            end_idx = start_idx + n_candidates
            batch_outputs = all_outputs[start_idx:end_idx]
            
            candidates = []
            candidate_tokens = []
            
            for output in batch_outputs:
                decoded = self.tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
                candidates.append(decoded)
                candidate_tokens.append(len(output) - input_len)
            
            # Fast MBR scoring
            def get_ngrams(text, n=2):
                words = text.lower().split()
                return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
            
            def ngram_overlap_score(cand1, cand2):
                unigrams1 = set(cand1.lower().split())
                unigrams2 = set(cand2.lower().split())
                
                if len(unigrams1) == 0 or len(unigrams2) == 0:
                    return 0.0
                
                unigram_overlap = len(unigrams1 & unigrams2) / (len(unigrams1) + len(unigrams2)) * 2
                
                bigrams1 = get_ngrams(cand1, 2)
                bigrams2 = get_ngrams(cand2, 2)
                
                if len(bigrams1) > 0 and len(bigrams2) > 0:
                    bigram_overlap = len(bigrams1 & bigrams2) / (len(bigrams1) + len(bigrams2)) * 2
                    return 0.6 * unigram_overlap + 0.4 * bigram_overlap
                return unigram_overlap
            
            if len(candidates) > 1 and all(len(c) > 0 for c in candidates):
                mbr_scores = []
                for i, cand_i in enumerate(candidates):
                    score = sum(ngram_overlap_score(cand_i, cand_j) 
                               for j, cand_j in enumerate(candidates) if i != j)
                    mbr_scores.append(score)
                best_idx = np.argmax(mbr_scores)
            else:
                best_idx = 0
            
            best_candidate = candidates[best_idx]
            total_completion = sum(candidate_tokens)
            
            results.append((best_candidate, input_len, total_completion))
        
        return results

    def extract_tool_call(self, decoded: str):
        """Extract the last JSON tool call between <tool_call>...</tool_call>."""
        matches = re.findall(r"<tool_call>(.*?)</tool_call>", decoded, re.DOTALL)
        if not matches:
            return None
        
        tool_str = matches[-1].strip()
        try:
            tool_json = json.loads(tool_str)
            if "arguments" in tool_json and isinstance(tool_json["arguments"], str):
                tool_json["arguments"] = json.loads(tool_json["arguments"])
            return tool_json['arguments']
        except json.JSONDecodeError:
            return None

    def _parse_graph_with_regex(self, prompt: str):
        """Attempt to parse graph specification using regex."""
        import re
        
        try:
            edge_pattern = r'(\d+)\s*->\s*(\d+)\s*,\s*weight:\s*(\d+)'
            edge_matches = re.findall(edge_pattern, prompt)
            
            if not edge_matches:
                return None, None, None, None
            
            edges = [[int(u), int(v), int(w)] for u, v, w in edge_matches]
            
            source_pattern = r'from node[s]?\s+(\d+)'
            source_match = re.search(source_pattern, prompt, re.IGNORECASE)
            
            if not source_match:
                return None, None, None, None
            
            source = int(source_match.group(1))
            
            target_pattern = r'to node[s]?\s+(\d+)'
            target_match = re.search(target_pattern, prompt, re.IGNORECASE)
            
            if not target_match:
                return None, None, None, None
            
            target = int(target_match.group(1))
            
            p_pattern = r'top[-\s]?(\d+)\s+shortest'
            p_match = re.search(p_pattern, prompt, re.IGNORECASE)
            
            if p_match:
                p = int(p_match.group(1))
            else:
                p = 5
            
            return edges, source, target, p
            
        except Exception as e:
            return None, None, None, None

    def _infer_graph_batch(self, prompts: List[str], max_tokens_list: List[int], gpu_idx: int) -> List[Tuple[str, int, int]]:
        """Batch process graph requests (mostly regex, LLM fallback for failures)."""
        import torch
        
        results = []
        llm_needed = []
        
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
                    results.append((f"[ERROR]  _paths execution failure: {str(e)}", 0, 0))
            else:
                llm_needed.append(idx)
                results.append(None)
        
        # Second pass: batch LLM calls for failures
        if llm_needed:
            llm_prompts = [prompts[i] for i in llm_needed]
            llm_results = self._batch_graph_llm(llm_prompts, [max_tokens_list[i] for i in llm_needed], gpu_idx)
            
            for idx, result in zip(llm_needed, llm_results):
                results[idx] = result
        
        return results

    def _batch_graph_llm(self, prompts: List[str], max_tokens_list: List[int], gpu_idx: int) -> List[Tuple[str, int, int]]:
        """Batch LLM-based graph parsing."""
        import torch
        
        model = self.models[gpu_idx]
        device = self.devices[gpu_idx]
        
        results = []
        
        for prompt in prompts:
            llm_prompt = """You will extract the graph specification from the user's prompt and return a JSON "
                "object representing a function call of the following shape:\n\n"
                {\n
                  "name": "parse_edges",\n
                  "arguments": {\n
                     "edges": [[u,v,w], ...],\n
                     "source": <int>,\n
                     "target": <int>,\n
                     "p": <int>\n
                  }\n
                }\n\n
                Requirements:\n
                - edges must be an array of triples [u, v, w] with integers\n
                - source, target, p must be integers\n
                - Output EXACTLY the JSON object above and nothing else.\n\n""" + prompt

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
                    max_new_tokens=2048,
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

    @modal.fastapi_endpoint(method="POST")
    async def completions(self, request: dict):
        """OpenAI-style completions endpoint with dynamic batching."""
        body_prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 128)

        if isinstance(body_prompt, str):
            prompts = [body_prompt]
        else:
            prompts = list(body_prompt)

        # Route all prompts
        tasks = [self._route_task(p) for p in prompts]
        
        # Add all requests to their respective queues and collect futures
        futures = []
        for prompt, task_type in zip(prompts, tasks):
            future = self._add_to_queue_and_wait(prompt, max_tokens, task_type)
            futures.append(future)
        
        # Wait for all results
        results = await asyncio.gather(*futures)
        
        # Format response
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for i, (text_out, ptoks, ctoks) in enumerate(results):
            choices.append({
                "text": text_out,
                "index": i,
                "finish_reason": "stop",
            })
            total_prompt_tokens += int(ptoks or 0)
            total_completion_tokens += int(ctoks or 0)

        return {
            "choices": choices,
            "model": "rithviks-system-1",
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            }
        }