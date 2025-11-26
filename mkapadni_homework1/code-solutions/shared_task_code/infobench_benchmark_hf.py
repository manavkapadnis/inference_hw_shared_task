# ANDREW ID = MKAPADNI
import os
os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/mkapadni/hf_cache/models"
from dotenv import load_dotenv
import json
from tqdm import tqdm
import torch
from inference import (
    load_custom_dataset,
    generate_problem_prompt,
    convert_llm_response_to_solution,
    evaluate_solution,
    load_hf_model_and_tokenizer,
    hf_generate_once,
    HF_QWEN_MODELS,
)

load_dotenv() 
DECODING_SETTINGS = ["default", "greedy", "temp_0_25", "temp_1_5", "beam_3", "beam_25", "typical"]

def run_infobench(model_name: str, out_dir: str, max_new_tokens: int = 256, limit: int = None):
    os.makedirs(out_dir, exist_ok=True)
    dataset = load_custom_dataset("InfoBench")  
    if limit:
        dataset = dataset[:limit]
        
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if openai_api_key == "":
        print("Warning: OPENAI_API_KEY not set; InfoBench eval will fail.")

    model, tokenizer, device = load_hf_model_and_tokenizer(model_name)
    for setting in DECODING_SETTINGS:
        results = []
        total = 0.0
        for i, example in tqdm(enumerate(dataset, 1), total=len(dataset), desc=f"{model_name} | {setting}"):
            prompt = generate_problem_prompt("InfoBench", example)
            # print("Below is prompt")
            # print(prompt)
            # print("\n\n")
            text = hf_generate_once(model, tokenizer, prompt, setting, max_new_tokens=max_new_tokens)
            # print("Below is model output")
            # print(text)
            # print("\n\n")
            pred = convert_llm_response_to_solution(text, "InfoBench")
            # print("Below is model preds formatted")
            # print(pred)
            # print("\n\n")
            score = float(evaluate_solution(example, pred, "InfoBench", model_name, api_key="", openai_api_key=openai_api_key))
            # print("score: ", score)
            # print("\n\n")
            total += score
            results.append({
                "id": i,
                "instruction": example.get("instruction", ""),
                "input": example.get("input", ""),
                "prompt": prompt,
                "raw_generation": text,
                "pred": pred,
                "score": score,
            })
        avg = total / len(dataset) if dataset else 0.0
        tag = f"InfoBench__{model_name.split('/')[-1]}__{setting}"
        with open(os.path.join(out_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
            json.dump({"task": "InfoBench", "model": model_name, "setting": setting, "avg_score": avg, "n": len(dataset), "results": results}, f, indent=2, ensure_ascii=False)
        print(f"{tag} avg_score={avg:.4f}")

if __name__ == "__main__":
    out_dir = "results"
    limit = None
    for m in HF_QWEN_MODELS:
        run_infobench(m, out_dir=out_dir, max_new_tokens=512, limit=limit)
