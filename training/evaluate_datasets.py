"""Quick evaluation of generated training datasets."""

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("training/data")


def load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    arxiv = load(DATA_DIR / "arxiv_params.json")
    rag = load(DATA_DIR / "rag_responses.json")
    intent = load(DATA_DIR / "intent_classification.json")

    print("=== Dataset sizes ===")
    print(f"arxiv_params: {len(arxiv)}")
    print(f"rag_responses: {len(rag)}")
    print(f"intent_classification: {len(intent)}")

    print("\n=== Intent distribution in RAG responses ===")
    rag_intents = []
    for ex in rag:
        try:
            assistant = json.loads(ex["messages"][-1]["content"])
            rag_intents.append(assistant.get("intent", "missing"))
        except Exception:
            rag_intents.append("parse_error")
    for intent_label, count in Counter(rag_intents).most_common():
        print(f"  {intent_label}: {count}")

    print("\n=== Intent distribution in intent_classification ===")
    for intent_label, count in Counter(ex["output"] for ex in intent).most_common():
        print(f"  {intent_label}: {count}")

    print("\n=== Message types in RAG responses ===")
    msg_types = []
    for ex in rag:
        try:
            assistant = json.loads(ex["messages"][-1]["content"])
            for msg in assistant.get("messages", []):
                msg_types.append(msg.get("type", "missing"))
        except Exception:
            pass
    for msg_type, count in Counter(msg_types).most_common():
        print(f"  {msg_type}: {count}")

    print("\n=== arXiv params key usage ===")
    keys = Counter()
    id_count = 0
    submitted_date_count = 0
    for ex in arxiv:
        try:
            params = json.loads(ex["output"])
            keys.update(params.keys())
            if "id_list" in params:
                id_count += 1
            search_query = params.get("search_query", "")
            if "submittedDate" in search_query:
                submitted_date_count += 1
        except Exception:
            pass
    for key, count in keys.most_common():
        print(f"  {key}: {count}")
    print(f"  Examples with id_list: {id_count}")
    print(f"  Examples with submittedDate: {submitted_date_count}")

    print("\n=== Sample arXiv params ===")
    for ex in arxiv[:3]:
        print(f"Input: {ex['input']}")
        print(f"Output: {ex['output']}")
        print()

    print("=== Sample RAG response ===")
    sample = json.dumps(rag[0], indent=2, ensure_ascii=False)
    print(sample[:1200])
    if len(sample) > 1200:
        print("... (truncated)")

    print("\n=== Sample intent classification ===")
    for ex in intent[:5]:
        print(f"{ex['output']}: {ex['input']}")


if __name__ == "__main__":
    main()
