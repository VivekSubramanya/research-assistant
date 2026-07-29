import json
import sys
sys.path.insert(0, "training")
from generate_training_data import validate_arxiv_params

arxiv = json.load(open("training/data/arxiv_params.json", encoding="utf-8"))
for i, ex in enumerate(arxiv):
    errors = validate_arxiv_params(ex)
    if errors:
        print(f"Example {i}: {errors}")
        print(f"  Input: {ex['input']}")
        print(f"  Output: {ex['output']}")
        print()
