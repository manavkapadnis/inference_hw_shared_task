## Evaluation Folder Structure

```
evaluation/
├── README.md
├── generate_batch_arrivals.py
├── generate_combined_dataset.py
├── run_evaluation.py
├── grader.py
└── math_verify.py
```

---

### `evaluation/README.md`

```markdown
# Evaluation Pipeline

## Quick Start

1. **Generate test data:**
   ```bash
   python generate_batch_arrivals.py
   python generate_combined_dataset.py
   ```

2. **Run evaluation:**
   ```bash
   export OPENAI_API_KEY=your_key_here
   python run_evaluation.py --url YOUR_MODAL_URL
   ```

## Outputs

- `batch_arrivals.json` - Input request schedule
- `combined_dataset.jsonl` - Ground truth dataset
- `simulation_summary.json` - Raw API responses
- `student_outputs.jsonl` - Extracted model outputs
- `test_student_metrics.json` - Final accuracy scores
- `test_student_results.jsonl` - Per-example results
```

