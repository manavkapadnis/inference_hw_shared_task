import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import os
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/mkapadni/hf_cache/models"


def mirostat(model, tokenizer, prompt, max_length=50, device='cpu', temperature=1.0, target_ce=3.0, learning_rate=0.1):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    mu = 2 * target_ce  # Initial mu value / "maximal surprisal"


    # TODO: YOUR CODE HERE -- additional variable init
    # We will not be checking this section for correctness,
    # But you will probably eventually want to set up some 
    # extra variables here for plotting metrics.
    # Our advice is to fill out the other sections first!
    k_hist = []
    s_hist = []
    mu_hist = [mu]
    ce_hist = []


    for step in range(max_length):
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
            adjusted_logits = logits / temperature
            adjusted_probs = torch.softmax(adjusted_logits, dim=-1)
            
            sorted_logits, sorted_inds = torch.sort(adjusted_logits, descending = True)
        
        # TODO: YOUR CODE HERE -- Estimate Zipf's exponent
        # Following Basu et al, use m=100 (i.e. use only the top 100 tokens(' diffs) to estimate the exponent)
        # Refer to Equation 30 [https://arxiv.org/pdf/2007.14966#equation.C.30](https://arxiv.org/pdf/2007.14966#equation.C.30) for pointers
        
        # Use top-m tokens to estimate s via the difference-based least-squares slope:
        # Let b_i = log p_i - log p_{i+1}, t_i = log(i+1) - log(i), then s_hat = sum_i t_i*b_i / sum_i t_i^2
        vocab_size = adjusted_probs.size(-1)
        m = min(100, vocab_size)
        topm_idx = sorted_inds[:, :m]                      # [1, m]
        topm_probs = adjusted_probs.gather(-1, topm_idx)   # [1, m]
        logp = torch.log(topm_probs + 1e-12)[0]            # [m]
        i = torch.arange(1, m+1, device=logp.device, dtype=logp.dtype)
        t = torch.log(i[1:]) - torch.log(i[:-1])           # [m-1]
        b = logp[:-1] - logp[1:]                           # [m-1]
        denom = torch.sum(t * t) + 1e-12
        s_hat = torch.sum(t * b) / denom
        s_hat = torch.clamp(s_hat, min=1.001)  # ensure s_hat > 1 to avoid epsilon<=0
        s_hist.append(s_hat.item())


        # TODO: YOUR CODE HERE -- Compute k using Zipf exponent
        # From Alg. 1 / Eq. (2): with ε_hat = s_hat - 1,
        # k = ((ε_hat * 2^mu) / (1 - N^{-ε_hat}))^(1/s_hat)
        eps_hat = s_hat - 1.0
        denom_k = 1.0 - (vocab_size ** (-eps_hat))
        denom_k = torch.clamp(denom_k, min=1e-12)
        k_float = ((eps_hat * (2.0 ** mu)) / denom_k) ** (1.0 / s_hat)
        # Convert to int in valid range [1, |V|]
        k = int(max(1, min(vocab_size, int(k_float.item()))))
        k_hist.append(k)


        # top k sampling
        topk_logits = sorted_logits[:,0:k]
        topk_inds = sorted_inds[:,0:k]
        topk_probs = torch.softmax(topk_logits, dim=1)
        next_tok = topk_inds[0, torch.multinomial(topk_probs, num_samples=1)]
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break


        # TODO: YOUR CODE HERE -- Compute surprisal error and adjust mu accordingly
        # Observed surprisal S(X) = -log P(X) under the adjusted distribution;
        # error e = S(X) - tau; update mu <- mu - eta * e
        p_next = adjusted_probs[0, next_tok.item()].item()
        surprisal = -math.log(max(p_next, 1e-12))
        ce_hist.append(surprisal)
        err = surprisal - target_ce
        mu = mu - learning_rate * err
        mu_hist.append(mu)
        
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    prompt = "Once upon a time,"
    result = mirostat(model, tokenizer, prompt, max_length=256, device=device, temperature=1.0, target_ce=3.0, learning_rate=0.1)
    print(result)
