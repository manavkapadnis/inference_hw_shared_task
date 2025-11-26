"""
Speculative Decoding Implementation

Reference Papers:
1. Fast Inference from Transformers via Speculative Decoding (https://arxiv.org/pdf/2211.17192)
2. Accelerating Large Language Model Decoding with Speculative Sampling (https://arxiv.org/pdf/2302.01318)

This implementation follows Algorithm 2 from Paper 2 (DeepMind).
See Theorem 1 for why the rejection sampling preserves the target distribution.
"""

import torch
import transformers
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored
from tqdm import tqdm

torch.manual_seed(42)
transformers.set_seed(42)

class SamplingConfig:
    def __init__(self,
                 max_new_tokens: int=50,
                 temperature: float=1.0,
                 lookahead_K: int=3,
                 device: str = "cuda:0",
                 debug: bool = False):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.lookahead_K = lookahead_K
        self.debug = debug
        self.dtype = torch.bfloat16

class SpecDecSamplingConfig(SamplingConfig):
    def __init__(self,
                 target_name: str,
                 draft_name: str):
        super().__init__()
        self.target_name = target_name
        self.draft_name = draft_name

class SpeculativeDecoder:
    def __init__(self, config: SpecDecSamplingConfig):
        """
        Initialize target model, draft model, and tokenizer.
        Set models to eval mode.
        """
        self.config = config
        self.device = config.device
        self.temperature = config.temperature
        
        # TODO: Load models and tokenizer
        print("Loading models...")
        # Load tokenizer (shared between target and draft)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.target_name, 
            trust_remote_code=True
        )
        
        # Fix for Llama models: set pad_token to something other than eos_token
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
            # Use unk_token if available, otherwise add a new pad token
            if self.tokenizer.unk_token:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load target model
        print(f"  Loading target model: {config.target_name}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            config.target_name,
            torch_dtype=config.dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.target_model.eval()
        
        # Resize embeddings if we added a new token
        if len(self.tokenizer) != self.target_model.config.vocab_size:
            self.target_model.resize_token_embeddings(len(self.tokenizer))
        
        # Load draft model
        print(f"  Loading draft model: {config.draft_name}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            config.draft_name,
            torch_dtype=config.dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.draft_model.eval()
        
        # Resize embeddings for draft model too
        if len(self.tokenizer) != self.draft_model.config.vocab_size:
            self.draft_model.resize_token_embeddings(len(self.tokenizer))
        
        print("✓ Models loaded\n")

    def max_fn(self, x):
        """Max function from paper 2 (f)_+"""
        return torch.clamp(x, min=0.0)

    def get_distribution(self, logits, temperature, epsilon=1e-8):
        """Get probability distribution from logits"""
        # Softmax with temperature
        if temperature <= epsilon:
            # Greedy decoding
            probs = torch.zeros_like(logits)
            probs[..., logits.argmax(dim=-1)] = 1.0
            return probs
        else:
            # temperature scaling
            logits = logits / temperature
            # normalize
            probs = F.softmax(logits, dim=-1)
            return probs

    @torch.inference_mode()
    def ar_sample(self, model, tokenized_prompt, max_new_tokens, temperature=1.0):
        """
        Standard autoregressive sampling.
        Returns generated sequence and temperature temp-normalized probs."""
        # TODO: Implement autoregressive generation
        generated = tokenized_prompt
        
        # Use model.generate for efficiency with KV caching
        outputs = model.generate(
            generated,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,  # Force exact number of tokens
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=None,  # Don't stop early
            use_cache=True
        )
        
        # Get the newly generated tokens
        new_tokens = outputs[:, generated.shape[1]:]
        
        # For speculative decoding, we also need the probabilities
        # Run forward pass to get probs
        with torch.no_grad():
            full_outputs = model(outputs)
            logits = full_outputs.logits[:, generated.shape[1]-1:-1, :]
            probs = self.get_distribution(logits, temperature)
        
        return outputs, probs

    @torch.inference_mode()
    def sd_sample(self, tokenized_prompt, max_new_tokens, lookahead, temperature):
        """
        Speculative decoding (Algorithm 2 from Paper 2).
        
        Args:
            tokenized_prompt: [batch_size, seq_len]
            max_new_tokens: Total tokens to generate
            lookahead: Number of speculative tokens (K)
            temperature: Sampling temperature
        
        Returns:
            generated_tokens: [batch_size, max_new_tokens]
            acceptance_rate: Fraction of draft tokens accepted
        """
        debug = self.config.debug
        bsz, n = tokenized_prompt.shape
        assert bsz == 1, 'Batch size should be 1'
        
        generated = tokenized_prompt.clone()
        target_len = n + max_new_tokens
        
        # Metrics
        accepted_count = 0
        draft_token_num = 0
        n_orig = n
        
        # Progress bar for token generation
        pbar = tqdm(total=max_new_tokens, desc="Generating", leave=False, disable=not debug)
        
        while generated.shape[1] < target_len:
            n = generated.shape[1]
            tokens_generated = 0
            
            # HINT: you dont want to overshoot on max_new_tokens
            corrected_lookahead = min(lookahead, target_len - n)
            
            # TODO: Generate K draft tokens
            with torch.no_grad():
                draft_outputs = self.draft_model.generate(
                    generated,
                    max_new_tokens=corrected_lookahead,
                    min_new_tokens=corrected_lookahead,  # Force exact count
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=None,  # Don't stop early on EOS
                    use_cache=True
                )
            
            draft_tokens = draft_outputs[:, n:]
            
            # Check if we got the expected number of tokens
            if draft_tokens.shape[1] < corrected_lookahead:
                # Draft model stopped early, handle gracefully
                corrected_lookahead = draft_tokens.shape[1]
                if corrected_lookahead == 0:
                    # No tokens generated, break
                    break
            
            draft_token_num += corrected_lookahead
            
            # Get draft probabilities
            with torch.no_grad():
                draft_logits_output = self.draft_model(draft_outputs)
                draft_logits = draft_logits_output.logits[:, n-1:n-1+corrected_lookahead, :]
                draft_probs = self.get_distribution(draft_logits, temperature)
            
            if debug:
                drafted_text = self.tokenizer.decode(draft_outputs[0, n:],
                                                     skip_special_tokens=False)
                print(colored(f"Possible continuations: {drafted_text}", 'blue', 'on_black'))
            
            # TODO: Run target model on draft sequence to verify
            with torch.no_grad():
                target_outputs = self.target_model(draft_outputs)
                target_logits = target_outputs.logits[:, n-1:n-1+corrected_lookahead, :]
                target_probs = self.get_distribution(target_logits, temperature)
            
            # TODO: For each draft token, compute acceptance probability and accept/reject
            accepted_tokens = []
            for t in range(corrected_lookahead):
                draft_token = draft_tokens[:, t:t+1]
                
                # Safety check
                if draft_token.shape[1] == 0:
                    break
                
                # Get probabilities for this token
                p_target = target_probs[:, t, draft_token[0, 0]]
                p_draft = draft_probs[:, t, draft_token[0, 0]]
                
                # Acceptance probability: min(1, p/q)
                accept_prob = torch.min(torch.ones_like(p_target), p_target / (p_draft + 1e-10))
                
                # Sample uniform random
                r = torch.rand_like(accept_prob)
                
                # accept loop
                if r < accept_prob:
                    accepted_tokens.append(draft_token)
                    accepted_count += 1
                    tokens_generated += 1
                    if debug:
                        accepted_token = self.tokenizer.decode(draft_token[0])
                        print(f"Accepted token: '{accepted_token}'")
                # reject loop
                else:
                    # TODO: Reject and resample from adjusted distribution
                    adjusted_probs = self.max_fn(target_probs[:, t, :] - draft_probs[:, t, :])
                    adjusted_probs = adjusted_probs / (adjusted_probs.sum(dim=-1, keepdim=True) + 1e-10)
                    
                    new_token = torch.multinomial(adjusted_probs, num_samples=1)
                    accepted_tokens.append(new_token)
                    tokens_generated += 1
                    
                    if debug:
                        rejected_token = self.tokenizer.decode(draft_token[0])
                        new_token_text = self.tokenizer.decode(new_token[0])
                        print(colored(f"Rejected: {rejected_token}", 'red', 'on_black'))
                        print(colored(f"Replaced with: {new_token_text}", 'green', 'on_black'))
                    break
            
            # Append accepted tokens
            if accepted_tokens:
                generated = torch.cat([generated] + accepted_tokens, dim=1)
            
            # TODO: Sample bonus token if all accepted
            if len(accepted_tokens) == corrected_lookahead and generated.shape[1] < target_len:
                with torch.no_grad():
                    bonus_outputs = self.target_model(generated)
                    bonus_logits = bonus_outputs.logits[:, -1:, :]
                    bonus_probs = self.get_distribution(bonus_logits, temperature)
                    bonus_token = torch.multinomial(bonus_probs[:, 0, :], num_samples=1)
                
                generated = torch.cat([generated, bonus_token], dim=1)
                tokens_generated += 1
                if debug:
                    print(colored(f"Bonus token: {self.tokenizer.decode(bonus_token[0])}", 'yellow'))
            
            pbar.update(tokens_generated)
            
            # Safety break
            if tokens_generated == 0:
                break
        
        pbar.close()
        
        # Calculate acceptance rate
        acceptance_rate = accepted_count / draft_token_num if draft_token_num > 0 else 0.0
        
        return generated, acceptance_rate

    def decode(self, 
               prompts,
               max_new_tokens,
               speculative=True,
               lookahead=None,
               temperature=1.0,
               show_progress=True):
        """
        Main decode function with progress tracking.
        """
        if lookahead is None:
            lookahead = self.config.lookahead_K
        
        all_generated = []
        all_acceptance_rates = []
        
        # Wrap prompts with tqdm
        prompt_iter = tqdm(prompts, desc="Prompts", disable=not show_progress)
        
        for prompt in prompt_iter:
            tokenized = self.tokenizer(
                prompt, 
                return_tensors='pt',
                padding=False,
                truncation=False
            ).input_ids.to(self.device)
            
            if speculative:
                generated, acceptance_rate = self.sd_sample(
                    tokenized, max_new_tokens, lookahead, temperature
                )
                all_acceptance_rates.append(acceptance_rate)
            else:
                # Autoregressive baseline
                with torch.no_grad():
                    generated = self.target_model.generate(
                        tokenized,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else 1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True
                    )
                all_acceptance_rates.append(0.0)
            
            all_generated.append(generated)
        
        avg_acceptance = sum(all_acceptance_rates) / len(all_acceptance_rates) if all_acceptance_rates else 0.0
        
        return all_generated, avg_acceptance
