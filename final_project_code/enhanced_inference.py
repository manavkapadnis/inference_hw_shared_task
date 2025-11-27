"""
Enhanced Inference System with Speculative Decoding
Optional optimization for speed-focused deployment
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
from inference_system import InferenceSystem


class SpeculativeDecoder:
    """
    Implements speculative decoding for faster inference
    Based on HW3 implementation
    """
    
    def __init__(self, 
                 target_model,
                 draft_model,
                 target_tokenizer,
                 draft_tokenizer,
                 lookahead: int = 3):
        self.target_model = target_model
        self.draft_model = draft_model
        self.target_tokenizer = target_tokenizer
        self.draft_tokenizer = draft_tokenizer
        self.lookahead = lookahead
    
    @torch.inference_mode()
    def generate(self, 
                input_ids: torch.Tensor,
                max_new_tokens: int = 512,
                temperature: float = 0.7) -> torch.Tensor:
        """
        Generate with speculative decoding
        Returns generated token IDs
        """
        device = self.target_model.device
        generated = input_ids.clone()
        target_len = input_ids.shape[1] + max_new_tokens
        
        acceptance_count = 0
        draft_count = 0
        
        while generated.shape[1] < target_len:
            n = generated.shape[1]
            corrected_lookahead = min(self.lookahead, target_len - n)
            
            if corrected_lookahead == 0:
                break
            
            # Draft phase: generate K tokens with draft model
            draft_outputs = self.draft_model.generate(
                generated,
                max_new_tokens=corrected_lookahead,
                min_new_tokens=corrected_lookahead,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=self.draft_tokenizer.pad_token_id,
                eos_token_id=None,
                use_cache=True
            )
            
            draft_tokens = draft_outputs[:, n:]
            
            if draft_tokens.shape[1] < corrected_lookahead:
                corrected_lookahead = draft_tokens.shape[1]
                if corrected_lookahead == 0:
                    break
            
            draft_count += corrected_lookahead
            
            # Get draft probabilities
            with torch.no_grad():
                draft_logits_output = self.draft_model(draft_outputs)
                draft_logits = draft_logits_output.logits[:, n-1:n-1+corrected_lookahead, :]
                draft_probs = F.softmax(draft_logits / temperature, dim=-1)
            
            # Verify with target model
            with torch.no_grad():
                target_outputs = self.target_model(draft_outputs)
                target_logits = target_outputs.logits[:, n-1:n-1+corrected_lookahead, :]
                target_probs = F.softmax(target_logits / temperature, dim=-1)
            
            # Accept/reject loop
            accepted_tokens = []
            for t in range(corrected_lookahead):
                draft_token = draft_tokens[:, t:t+1]
                
                if draft_token.shape[1] == 0:
                    break
                
                p_target = target_probs[:, t, draft_token[0, 0]]
                p_draft = draft_probs[:, t, draft_token[0, 0]]
                
                # Acceptance probability
                accept_prob = torch.min(
                    torch.ones_like(p_target), 
                    p_target / (p_draft + 1e-10)
                )
                
                r = torch.rand_like(accept_prob)
                
                if r < accept_prob:
                    # Accept
                    accepted_tokens.append(draft_token)
                    acceptance_count += 1
                else:
                    # Reject and resample
                    adjusted_probs = torch.clamp(
                        target_probs[:, t, :] - draft_probs[:, t, :],
                        min=0.0
                    )
                    adjusted_probs = adjusted_probs / (adjusted_probs.sum(dim=-1, keepdim=True) + 1e-10)
                    
                    new_token = torch.multinomial(adjusted_probs, num_samples=1)
                    accepted_tokens.append(new_token)
                    break
            
            # Append accepted tokens
            if accepted_tokens:
                generated = torch.cat([generated] + accepted_tokens, dim=1)
            
            # Bonus token if all accepted
            if len(accepted_tokens) == corrected_lookahead and generated.shape[1] < target_len:
                with torch.no_grad():
                    bonus_outputs = self.target_model(generated)
                    bonus_logits = bonus_outputs.logits[:, -1:, :]
                    bonus_probs = F.softmax(bonus_logits / temperature, dim=-1)
                    bonus_token = torch.multinomial(bonus_probs[:, 0, :], num_samples=1)
                
                generated = torch.cat([generated, bonus_token], dim=1)
            
            if len(accepted_tokens) == 0:
                break
        
        acceptance_rate = acceptance_count / draft_count if draft_count > 0 else 0.0
        
        return generated


class EnhancedInferenceSystem(InferenceSystem):
    """
    Enhanced inference system with speculative decoding option
    """
    
    def __init__(self, *args, use_speculative: bool = False, 
                 draft_model_path: str = "Qwen/Qwen3-0.6B", **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_speculative = use_speculative
        
        if use_speculative:
            print(f"Loading draft model for speculative decoding: {draft_model_path}")
            self.draft_tokenizer = AutoTokenizer.from_pretrained(
                draft_model_path, trust_remote_code=True
            )
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.draft_model.eval()
            
            if self.draft_tokenizer.pad_token is None:
                self.draft_tokenizer.pad_token = self.draft_tokenizer.eos_token
            
            # Create speculative decoder (uses large model as target)
            self.spec_decoder = SpeculativeDecoder(
                target_model=self.large_model,
                draft_model=self.draft_model,
                target_tokenizer=self.large_tokenizer,
                draft_tokenizer=self.draft_tokenizer,
                lookahead=3  # Optimal from HW3 analysis
            )
            
            print("Speculative decoding enabled!")
    
    @torch.inference_mode()
    def generate_batch_speculative(self,
                                  prompts: List[str],
                                  max_new_tokens: int = 512,
                                  temperature: float = 0.7,
                                  task: str = "infobench") -> List[str]:
        """
        Generate using speculative decoding (single prompt at a time for now)
        """
        results = []
        
        for prompt in prompts:
            # Format prompt
            formatted = self._format_prompt_for_qwen3(
                self.large_tokenizer, prompt, task
            )
            
            # Tokenize
            inputs = self.large_tokenizer(
                [formatted],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=2048
            ).to(self.large_model.device)
            
            # Generate with speculative decoding
            outputs = self.spec_decoder.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            # Decode only new tokens
            new_tokens = outputs[0][len(inputs.input_ids[0]):]
            text = self.large_tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(text)
        
        return results
    
    def process_request(self, prompt: str, max_tokens: int = 512,
                       temperature: float = 0.7) -> str:
        """
        Process request with optional speculative decoding
        """
        # Identify task
        task = self.router.identify_task(prompt)
        
        # For graph tasks or if speculative decoding is enabled for large model
        if self.use_speculative and task in ["graph", "infobench"]:
            results = self.generate_batch_speculative(
                [prompt],
                max_new_tokens=max_tokens,
                temperature=temperature,
                task=task
            )
            return results[0]
        else:
            # Use standard generation
            return super().process_request(prompt, max_tokens, temperature)
