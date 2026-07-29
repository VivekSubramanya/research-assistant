"""Post-process arXiv params to normalize non-standard keys."""

import json
from pathlib import Path


def clean_arxiv_params(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    cleaned = []
    for item in dataset:
        try:
            params = json.loads(item["output"])
        except json.JSONDecodeError:
            cleaned.append(item)
            continue

        # Normalize 'sort' -> 'sortBy'/'sortOrder' if present.
        if "sort" in params and isinstance(params["sort"], str):
            sort_value = params.pop("sort").lower()
            if sort_value in {"relevance", "lastupdateddate", "submitteddate"}:
                params["sortBy"] = sort_value
                params["sortOrder"] = "descending"

        # Move standalone 'category' into search_query only when not using id_list.
        if "category" in params and "id_list" not in params:
            category = params.pop("category")
            if isinstance(category, list):
                category_clause = " OR ".join(f"cat:{c}" for c in category)
            else:
                category_clause = f"cat:{category}"

            search_query = params.get("search_query", "")
            if search_query:
                params["search_query"] = f"({category_clause}) AND ({search_query})"
            else:
                params["search_query"] = category_clause

        # If id_list is present, search_query must be omitted.
        if "id_list" in params and "search_query" in params:
            params.pop("search_query")

        item["output"] = json.dumps(params, ensure_ascii=False)
        cleaned.append(item)

    return cleaned


def main():
    path = Path("training/data/arxiv_params.json")
    cleaned = clean_arxiv_params(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"Cleaned {len(cleaned)} arXiv params examples")


if __name__ == "__main__":
    main()
