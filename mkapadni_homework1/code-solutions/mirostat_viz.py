# ANDREW ID = MKAPADNI
import os
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class MirostatResult:
    text: str
    per_token_surprisal: List[float]
    per_token_ppl: List[float]
    k_hist: List[int]
    s_hist: List[float]
    mu_hist: List[float]
    err_hist: List[float]
    logits_snapshots: Dict[int, np.ndarray] 

def mirostat(
    model,
    tokenizer,
    prompt: str,
    device: str = "cpu",
    temperature: float = 0.9,
    target_ce: float = 3.0,
    learning_rate: float = 0.1,
    max_total_seq_len: int = 128,
    record_steps_for_logits: Optional[List[int]] = None,
) -> MirostatResult:
    """
    Mirostat decoding with dynamic top-k based on Zipf exponent; collects traces and optional logits snapshots.
    """
    model.eval()
    torch.set_grad_enabled(False)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    mu = 2.0 * target_ce

    per_token_surprisal: List[float] = []
    per_token_ppl: List[float] = []
    k_hist: List[int] = []
    s_hist: List[float] = []
    mu_hist: List[float] = [mu]
    err_hist: List[float] = []
    logits_snapshots: Dict[int, np.ndarray] = {}

    step_idx = 0
    while input_ids.shape[1] < max_total_seq_len:
        step_idx += 1
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
            adjusted_logits = logits / temperature
            adjusted_probs = torch.softmax(adjusted_logits, dim=-1)
            sorted_logits, sorted_inds = torch.sort(adjusted_logits, descending=True)

        # Estimate Zipf exponent s_hat using top-m (m=100) via Eq. (30) 
        vocab_size = adjusted_probs.size(-1)
        m = min(100, vocab_size)
        topm_idx = sorted_inds[:, :m]                     # [1, m]
        topm_probs = adjusted_probs.gather(-1, topm_idx)  # [1, m]
        logp = torch.log(topm_probs + 1e-12)[0]           # [m]
        i = torch.arange(1, m + 1, device=logp.device, dtype=logp.dtype)
        t = torch.log(i[1:]) - torch.log(i[:-1])          # [m-1]
        b = logp[:-1] - logp[1:]                          # [m-1]
        denom = torch.sum(t * t) + 1e-12
        s_hat = (torch.sum(t * b) / denom).clamp(min=1.001)
        s_hist.append(float(s_hat.item()))

        # Compute k from s_hat and mu (Alg. 1 / Eq. (2)) with epsilon_hat = s_hat - 1
        eps_hat = s_hat - 1.0
        denom_k = 1.0 - (vocab_size ** (-eps_hat))
        denom_k = torch.clamp(denom_k, min=1e-12)
        k_float = ((eps_hat * (2.0 ** mu)) / denom_k) ** (1.0 / s_hat)
        k = int(max(1, min(vocab_size, int(k_float.item()))))
        k_hist.append(k)

        # Top-k sampling from adjusted distribution
        topk_logits = sorted_logits[:, :k]      # [1, k]
        topk_inds = sorted_inds[:, :k]          # [1, k]
        topk_probs = torch.softmax(topk_logits, dim=-1)  # [1, k]
        next_in_topk = torch.multinomial(topk_probs[0], num_samples=1)  # [1]
        next_tok = topk_inds[0, next_in_topk]

        # Optional logits snapshots at specific steps (store adjusted logits as numpy for plotting)
        if record_steps_for_logits and (step_idx in record_steps_for_logits):
            logits_snapshots[step_idx] = adjusted_logits[0].detach().cpu().numpy()

        # Compute surprisal under the adjusted distribution and update mu via error feedback
        next_id_int = int(next_tok.item())
        p_next = float(adjusted_probs[0, next_id_int].item())
        surprisal = -math.log(max(p_next, 1e-12))
        per_token_surprisal.append(surprisal)
        per_token_ppl.append(math.exp(surprisal))
        err = surprisal - float(target_ce)
        err_hist.append(err)
        mu = float(mu - learning_rate * err)
        mu_hist.append(mu)

        # Append token; stop at EOS or when reaching max length
        input_ids = torch.cat([input_ids, next_tok.view(1, 1)], dim=1)
        if next_id_int == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return MirostatResult(
        text=text,
        per_token_surprisal=per_token_surprisal,
        per_token_ppl=per_token_ppl,
        k_hist=k_hist,
        s_hist=s_hist,
        mu_hist=mu_hist,
        err_hist=err_hist,
        logits_snapshots=logits_snapshots,
    )

# Utils

def slug(s: str) -> str:
    s = s.strip().replace(" ", "_").replace(",", "").replace("'", "")
    s = s.replace("/", "_").replace("+", "plus").replace("=", "eq")
    return s[:60]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def summarize_perplexity(per_token_ppl: List[float], per_token_surprisal: List[float]) -> Dict[str, float]:
    arr = np.array(per_token_ppl, dtype=np.float64)
    mean_ppl = float(np.mean(arr)) if arr.size > 0 else float("nan")
    median_ppl = float(np.median(arr)) if arr.size > 0 else float("nan")
    std_ppl = float(np.std(arr)) if arr.size > 0 else float("nan")
    # Sequence-level perplexity as exp(mean surprisal)
    seq_ppl = float(math.exp(float(np.mean(per_token_surprisal)))) if len(per_token_surprisal) > 0 else float("nan")
    return {
        "mean_token_ppl": mean_ppl,
        "median_token_ppl": median_ppl,
        "std_token_ppl": std_ppl,
        "sequence_ppl": seq_ppl,
    }

def plot_traces(steps, k_hist, s_hist, mu_hist, err_hist, out_path: str, title: str):
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(steps, k_hist, marker="o", linewidth=1)
    ax1.set_title("k vs step")
    ax1.set_xlabel("step")
    ax1.set_ylabel("k")

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(steps, s_hist, marker="o", color="tab:orange", linewidth=1)
    ax2.set_title("s_hat vs step")
    ax2.set_xlabel("step")
    ax2.set_ylabel("s_hat")

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(range(len(mu_hist)), mu_hist, marker="o", color="tab:green", linewidth=1)
    ax3.set_title("mu vs step (incl. init)")
    ax3.set_xlabel("step")
    ax3.set_ylabel("mu")

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(steps, err_hist, marker="o", color="tab:red", linewidth=1)
    ax4.set_title("surprisal error vs step")
    ax4.set_xlabel("step")
    ax4.set_ylabel("error")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_logits_distributions(logits_snapshots: Dict[int, np.ndarray], out_path: str, title: str, bins: int = 50):
    steps_sorted = sorted(logits_snapshots.keys())
    
    plt.figure(figsize=(8, 12))
    
    for i, step in enumerate(steps_sorted):
        if i >= 3:  # Only plot first 3 steps
            break
            
        logits = logits_snapshots[step]
        
        # Calculate statistics
        mean_val = np.mean(logits)
        std_val = np.std(logits)
        min_val = np.min(logits)
        max_val = np.max(logits)
        
        plt.subplot(3, 1, i + 1)
        
        # Create histogram
        n, bins_edges, patches = plt.hist(logits, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add statistics text box
        stats_text = f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.title(f"Logits Distribution at Step {step}")
        plt.xlabel("Logit Values")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_one_combo(
    model_name: str,
    tokenizer,
    model,
    device: str,
    prompt: str,
    tau: float,
    viz_dir: str,
    steps_for_logits: Optional[List[int]] = None,
) -> Dict[str, object]:
    res = mirostat(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        temperature=0.9,
        target_ce=tau,
        learning_rate=0.1,
        max_total_seq_len=128,
        record_steps_for_logits=steps_for_logits,
    )

    ppl_stats = summarize_perplexity(res.per_token_ppl, res.per_token_surprisal)
    model_slug = slug(model_name.split("/")[-1])
    prompt_slug = slug(prompt if len(prompt) > 0 else "EMPTY")
    tag = f"{model_slug}__{prompt_slug}__tau{tau:.2f}"

    with open(os.path.join(viz_dir, f"{tag}__sequence.txt"), "w", encoding="utf-8") as f:
        f.write(res.text)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "tau": tau,
        "temperature": 0.9,
        "learning_rate": 0.1,
        "sequence_len": len(res.text),
        "mean_token_ppl": ppl_stats["mean_token_ppl"],
        "median_token_ppl": ppl_stats["median_token_ppl"],
        "std_token_ppl": ppl_stats["std_token_ppl"],
        "sequence_ppl": ppl_stats["sequence_ppl"],
        "num_steps": len(res.k_hist),
    }
    with open(os.path.join(viz_dir, f"{tag}__metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    steps = list(range(1, len(res.k_hist) + 1))
    plot_traces(
        steps=steps,
        k_hist=res.k_hist,
        s_hist=res.s_hist,
        mu_hist=res.mu_hist,
        err_hist=res.err_hist,
        out_path=os.path.join(viz_dir, f"{tag}__trace.png"),
        title=f"Traces: {model_name}, prompt='{prompt}', tau={tau:.2f}",
    )

    return {
        "result": res,
        "stats": ppl_stats,
        "tag": tag,
    }

if __name__ == "__main__":
    VIZ_DIR = "viz"
    ensure_dir(VIZ_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_names = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.1-8B",
    ]

    prompts = [
        "Once upon a time,",
        "The capital of France is,",
    ]

    tau_values = [2.0, 3.0, 4.0]
    logits_plot_model = "meta-llama/Llama-3.2-1B"
    logits_plot_prompt = "Once upon a time,"
    steps_for_logits = [1, 10, 100]
    model_cache: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}

    # Run the 3 x 2 x 2 = 12 combinations
    for model_name in model_names:
        if model_name not in model_cache:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to(device)
            model_cache[model_name] = (tokenizer, model)
        else:
            tokenizer, model = model_cache[model_name]

        for prompt in prompts:
            for tau in tau_values:
                # Only collect logits snapshots when generating the logit distribution figures case to save time/memory
                record_steps = steps_for_logits if (model_name == logits_plot_model and prompt == logits_plot_prompt) else None
                _ = run_one_combo(
                    model_name=model_name,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    prompt=prompt,
                    tau=tau,
                    viz_dir=VIZ_DIR,
                    steps_for_logits=record_steps,
                )

    tokenizer, model = model_cache[logits_plot_model]
    for tau in tau_values:
        res = mirostat(
            model=model,
            tokenizer=tokenizer,
            prompt=logits_plot_prompt,
            device=device,
            temperature=0.9,
            target_ce=tau,
            learning_rate=0.1,
            max_total_seq_len=128,
            record_steps_for_logits=steps_for_logits,
        )
        model_slug = slug(logits_plot_model.split("/")[-1])
        prompt_slug = slug(logits_plot_prompt)
        tag = f"{model_slug}__{prompt_slug}__tau{tau:.2f}"
        
        # Make histogram distribution plots for steps 1, 10, 100
        plot_logits_distributions(
            logits_snapshots=res.logits_snapshots,
            out_path=os.path.join(VIZ_DIR, f"{tag}__logits_distribution.png"),
            title=f"Logit Distribution: {logits_plot_model}, prompt='{logits_plot_prompt}', τ={tau:.1f}",
            bins=50,
        )

    print(f"All results and plots saved under: {os.path.abspath(VIZ_DIR)}")
