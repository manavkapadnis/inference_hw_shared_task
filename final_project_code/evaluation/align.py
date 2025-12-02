import json
from datasets import load_dataset
from tqdm import tqdm

# === CONFIGURATION ===
# Path to the input file used by make_requests.py
# Verify this is the file that generated your current simulation output!
INPUT_BATCH_FILE = "/home/mkapadni/work/inference_algo/homework4/attempt_1/evaluation/batch_arrivals.json" 
OUTPUT_FILE = "/home/mkapadni/work/inference_algo/homework4/attempt_1/evaluation/combined_dataset.jsonl"

def load_source_datasets():
    print("Loading source datasets (Graph, InfoBench, MMLU)...")
    graph_ds = load_dataset('vashistht/11763_datasets', 'graph_dev', split='dev_test')
    infobench_ds = load_dataset('vashistht/11763_datasets', 'infobench', split='dev_test')
    mmlu_ds = load_dataset('vashistht/11763_datasets', 'mmlu_med', split='dev_test')
    return graph_ds, infobench_ds, mmlu_ds

def find_match(prompt_text, graph_ds, infobench_ds, mmlu_ds):
    """
    Finds the source record for a given prompt text by checking
    if the source content is a substring of the prompt.
    """
    # 1. Check Graph (Exact or high overlap match)
    for ex in graph_ds:
        # Graph prompts are usually exact matches
        if ex['prompt'].strip() in prompt_text.strip():
            # Parse solution if it's a string
            solution = ex.get('solution')
            if isinstance(solution, str):
                try:
                    solution = json.loads(solution)
                except:
                    pass
            
            return {
                "task": "graph",
                "gold_answer": solution,
                "meta": {"graph_params": ex.get('graph_params'), "original_id": ex.get('id')}
            }

    # 2. Check MMLU (Question text should be in prompt)
    for ex in mmlu_ds:
        # MMLU prompts in batch are wrapped with "Question: ... Choices: ..."
        # So we check if the raw question is inside the batch prompt
        if ex['question'].strip() in prompt_text:
            answer_idx = ex.get("answer")
            gold_letter = chr(65 + answer_idx) if answer_idx is not None else None
            return {
                "task": "mmlu_med",
                "gold_answer": gold_letter,
                "meta": {"subject": ex.get("subject"), "choices": ex.get("choices")}
            }

    # 3. Check InfoBench (Instruction should be in prompt)
    for ex in infobench_ds:
        if ex['instruction'].strip() in prompt_text:
            return {
                "task": "infobench",
                "gold_answer": None,
                "meta": {
                    "decomposed_questions": ex.get('decomposed_questions'),
                    "input": ex.get('input'),
                    "subset": ex.get('subset'),
                    "original_id": ex.get('id')
                }
            }
            
    return None

def main():
    # 1. Load the batch file to get (Index -> Prompt) mapping
    print(f"Reading batch file: {INPUT_BATCH_FILE}")
    try:
        with open(INPUT_BATCH_FILE, 'r') as f:
            batches = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_BATCH_FILE}.")
        print("Please ensure this matches the filename in make_requests.py")
        return

    # 2. Load source datasets for lookup
    graph_ds, infobench_ds, mmlu_ds = load_source_datasets()

    aligned_data = []
    seen_indices = set()
    
    print("Matching prompts to ground truth...")
    
    # Iterate through all batches and all requests
    for batch in tqdm(batches):
        prompts = batch['prompts']
        indices = batch['prompt_idxs']
        
        for prompt_text, idx in zip(prompts, indices):
            if idx in seen_indices:
                continue # Skip duplicates if any
            
            match = find_match(prompt_text, graph_ds, infobench_ds, mmlu_ds)
            
            if match:
                entry = {
                    "index": idx, # This ensures alignment with student_outputs.jsonl
                    "task": match['task'],
                    "prompt": prompt_text,
                    "gold_answer": match['gold_answer'],
                    "meta": match['meta']
                }
                aligned_data.append(entry)
                seen_indices.add(idx)
            else:
                print(f"WARNING: Could not find match for index {idx}")

    # 3. Save aligned dataset
    aligned_data.sort(key=lambda x: x['index'])
    
    print(f"\nFound matches for {len(aligned_data)} items.")
    print(f"Saving to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w') as f:
        for item in aligned_data:
            f.write(json.dumps(item) + "\n")
            
    print("Done! You can now run eval.ipynb.")

if __name__ == "__main__":
    main()