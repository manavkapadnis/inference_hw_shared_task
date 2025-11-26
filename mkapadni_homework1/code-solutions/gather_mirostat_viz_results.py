# ANDREW ID = MKAPADNI
import os
import json
import glob

def load_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def main(viz_dir="viz"):
    os.makedirs(viz_dir, exist_ok=True)  
    metrics_paths = sorted(glob.glob(os.path.join(viz_dir, "*__metrics.json")))
    combined = []

    for mpath in metrics_paths:
        with open(mpath, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        seq_path = mpath.replace("__metrics.json", "__sequence.txt")
        sequence_text = load_text(seq_path)
        rec = {
            "tag": os.path.basename(mpath).replace("__metrics.json", ""),
            "model": metrics.get("model"),
            "prompt": metrics.get("prompt"),
            "tau": metrics.get("tau"),
            "temperature": metrics.get("temperature"),
            "learning_rate": metrics.get("learning_rate"),
            "sequence_len": metrics.get("sequence_len"),
            "mean_token_ppl": metrics.get("mean_token_ppl"),
            "median_token_ppl": metrics.get("median_token_ppl"),
            "std_token_ppl": metrics.get("std_token_ppl"),
            "sequence_ppl": metrics.get("sequence_ppl"),
            "num_steps": metrics.get("num_steps"),
            "sequence": sequence_text,
        }
        combined.append(rec)

    out_path = os.path.join(viz_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(combined)} records to {out_path}")

if __name__ == "__main__":
    main()
