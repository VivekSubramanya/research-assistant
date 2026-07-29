"""Synthetic training data generator for the arXiv Research Assistant.

This script produces three LLaMA-Factory compatible datasets:
- arxiv_params.json (Alpaca format)
- rag_responses.json (ShareGPT chat format)
- intent_classification.json (Alpaca format)

It embeds the real system prompt used by llm.py and validates every generated
example against the application schema before writing to disk.
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")


def ollama_generate(prompt: str, system: str | None = None, require_json: bool = False) -> str:
    """Call a local Ollama model directly via its /api/generate endpoint."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 2048},
    }
    if system:
        payload["system"] = system
    if require_json:
        payload["format"] = "json"

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


# ---------------------------------------------------------------------------
# Domain / intent distributions
# ---------------------------------------------------------------------------

DOMAINS = [
    "AI/ML",
    "Quantitative Biology",
    "Astrophysics",
    "Mathematics",
    "Quantum Chemistry",
    "High Energy Physics",
    "Computer Vision",
    "NLP",
]
INTENTS = ["default", "comparison", "citation_request", "paper_level_query"]


# ---------------------------------------------------------------------------
# Real system prompt extracted from llm.py (must stay in sync with runtime)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """You are a research assistant restricted to the provided arXiv context only.

Rules:
- Use only facts present in Context.
- If Context is insufficient, state that clearly.
- Never invent citations.
- Return JSON only.

Frontend response JSON schema:
{
    "query": "<original user query>",
    "intent": "default|comparison|citation_request|paper_level_query",
    "messages": [
        {"type": "text", "content": "...", "title": "optional"},
        {"type": "table", "content": [{"column": "value"}]},
        {"type": "citations", "content": ["citation string"]}
    ],
    "sources": ["optional source ids"],
    "meta": {}
}

Allowed message types: text, table, graph, list, citations, code.
"""

INTENT_PROMPTS = {
    "comparison": "Provide a structured side-by-side comparison. Prefer one table message then one short text takeaway.",
    "citation_request": "Answer concisely and include a citations message with only citations found in the provided context. The citation should include the title, authors, and arXiv ID if available.",
    "paper_level_query": "Answer using chunk details from the requested paper only.",
    "default": "Answer concisely using only the provided context.",
}


def build_rag_system_prompt(intent: str = "default") -> str:
    """Return the exact system prompt the runtime will use for a given intent."""
    return (
        SYSTEM_PROMPT_BASE
        + "\n"
        + INTENT_PROMPTS.get(intent, INTENT_PROMPTS["default"])
        + "\nContext (arXiv-only):\n{context}\n\nQuery: {query}\n\nImportant: Only use facts that appear in the Context above. Do not use outside knowledge or sources. If the Context is empty or insufficient, say that the answer cannot be determined from the provided arXiv material.\n"
    )


# ---------------------------------------------------------------------------
# Schema validators
# ---------------------------------------------------------------------------

VALID_INTENTS = {"default", "comparison", "citation_request", "paper_level_query"}
VALID_MESSAGE_TYPES = {"text", "table", "graph", "list", "citations", "code"}


def validate_arxiv_params(item: dict[str, Any]) -> list[str]:
    """Validate one Alpaca example for arXiv parameter extraction."""
    errors: list[str] = []
    if not isinstance(item, dict):
        return ["item is not a dict"]

    output_str = item.get("output")
    if not isinstance(output_str, str):
        errors.append("'output' must be a string")
        return errors

    try:
        params = json.loads(output_str)
    except json.JSONDecodeError as exc:
        errors.append(f"'output' is not valid JSON: {exc}")
        return errors

    if not isinstance(params, dict):
        errors.append("parsed 'output' is not a JSON object")
        return errors

    if "id_list" in params:
        id_list = params["id_list"]
        if not isinstance(id_list, str):
            errors.append("'id_list' must be a comma-separated string")
        elif id_list.strip().lower() in {
            "",
            "optional",
            "none",
            "null",
            "comma-separated ids",
            "optional comma-separated ids",
            "optional, comma-separated ids",
        }:
            errors.append("'id_list' contains placeholder text")
        if "search_query" in params:
            errors.append("'search_query' must be omitted when 'id_list' is present")
        return errors

    if "search_query" not in params:
        errors.append("'search_query' is required unless 'id_list' is provided")

    for field in ("start", "max_results"):
        if field in params and not isinstance(params[field], int):
            errors.append(f"'{field}' must be an integer")

    if "sortBy" in params and params["sortBy"] not in {
        "relevance",
        "lastUpdatedDate",
        "submittedDate",
    }:
        errors.append("'sortBy' has invalid value")

    if "sortOrder" in params and params["sortOrder"] not in {
        "ascending",
        "descending",
    }:
        errors.append("'sortOrder' has invalid value")

    search_query = params.get("search_query", "")
    if search_query:
        if "submittedDate" in search_query and not re.search(
            r"submittedDate:\[\d{12}\+TO\+\d{12}\]", search_query
        ):
            errors.append(
                "submittedDate clause must use [YYYYMMDD0000+TO+YYYYMMDD0000] format"
            )

    return errors


def validate_rag_response(item: dict[str, Any]) -> list[str]:
    """Validate one ShareGPT example for RAG response generation."""
    errors: list[str] = []
    if not isinstance(item, dict):
        return ["item is not a dict"]

    messages = item.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        errors.append("'messages' must contain at least system/user/assistant")
        return errors

    assistant_msg = messages[-1]
    if assistant_msg.get("role") != "assistant":
        errors.append("last message must be from assistant")
        return errors

    content = assistant_msg.get("content")
    if not isinstance(content, str):
        errors.append("assistant content must be a string")
        return errors

    try:
        response = json.loads(content)
    except json.JSONDecodeError as exc:
        errors.append(f"assistant content is not valid JSON: {exc}")
        return errors

    if not isinstance(response, dict):
        errors.append("assistant content is not a JSON object")
        return errors

    intent = response.get("intent")
    if intent not in VALID_INTENTS:
        errors.append(f"invalid intent '{intent}'")

    for msg in response.get("messages", []):
        if not isinstance(msg, dict):
            errors.append("each message must be a dict")
            continue
        msg_type = msg.get("type")
        if msg_type not in VALID_MESSAGE_TYPES:
            errors.append(f"invalid message type '{msg_type}'")

        content_value = msg.get("content")
        if msg_type == "table" and not isinstance(content_value, list):
            errors.append("table content must be a list of dicts")
        if msg_type == "table" and isinstance(content_value, list):
            for row in content_value:
                if not isinstance(row, dict):
                    errors.append("each table row must be a dict")
                    break
        if msg_type == "citations" and not isinstance(content_value, list):
            errors.append("citations content must be a list")
        if msg_type == "citations" and isinstance(content_value, list):
            for citation in content_value:
                if not isinstance(citation, str):
                    errors.append("each citation must be a string")
                    break

    sources = response.get("sources", [])
    if not isinstance(sources, list):
        errors.append("'sources' must be a list")
    else:
        for source in sources:
            if not isinstance(source, str):
                errors.append("each source must be a string")
                break

    if "meta" in response and not isinstance(response["meta"], dict):
        errors.append("'meta' must be a dict")

    return errors


def validate_intent_classification(item: dict[str, Any]) -> list[str]:
    """Validate one Alpaca example for intent classification."""
    errors: list[str] = []
    if not isinstance(item, dict):
        return ["item is not a dict"]

    output = item.get("output")
    if output not in VALID_INTENTS:
        errors.append(f"invalid intent label '{output}'")

    return errors


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _load_existing_jsonl(path: Path | None) -> list[dict[str, Any]]:
    """Load previously generated examples from a JSONL file if it exists."""
    if not path or not path.exists():
        return []
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed line in {path}")
    print(f"Resumed from {path}: loaded {len(items)} existing examples")
    return items


def _append_jsonl(path: Path | None, item: dict[str, Any]) -> None:
    """Append a single example to a JSONL file for crash recovery."""
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_REFERENCE_DATE = datetime(2026, 7, 22, tzinfo=timezone.utc)


def default_date_window() -> str:
    """Return the exact submittedDate clause used for capped 12-month windows."""
    start = _REFERENCE_DATE.replace(year=_REFERENCE_DATE.year - 1)
    return (
        f"submittedDate:[{start.strftime('%Y%m%d%H%M')}+TO+"
        f"{_REFERENCE_DATE.strftime('%Y%m%d%H%M')}]"
    )


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------


def _generate_arxiv_query(domain: str, is_id_query: bool) -> str:
    """Step 1: generate a natural-language user query."""
    system_prompt = """You are generating realistic user queries for an arXiv research assistant.
Return a single JSON object with key 'query' containing a natural-language user query.

The query should sound like something a researcher would type. It may mention:
- a topic (e.g., "large language models", "black holes", "CRISPR")
- an author (e.g., "papers by Yann LeCun")
- recency (e.g., "recent", "last year", "since 2025")
- specific arXiv paper IDs if requested

Do NOT include instructions like "Domain:" or "Generate a" in the query. Just return the raw user query."""

    user_prompt = f"Domain: {domain}. "
    if is_id_query:
        user_prompt += "Generate a query asking about one or more specific arXiv papers by ID."
    else:
        user_prompt += "Generate a natural language search query for papers on a topic in this domain."

    raw_response = ollama_generate(
        prompt=f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:",
        require_json=True,
    )
    parsed = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("query generation did not return a dict")
    return parsed.get("query", "").strip()


def _generate_arxiv_params_from_query(query: str) -> dict[str, Any]:
    """Step 2: convert a user query into arXiv API parameters."""
    system_prompt = f"""You convert user queries to arXiv API parameters.
Return a single JSON object (not a string) with the API parameters.

Rules:
- If the query names specific arXiv paper ID(s):
  - Use 'id_list' as a comma-separated STRING (e.g., "2303.08774,2305.18290").
  - OMIT 'search_query' entirely.
- Otherwise:
  - Provide 'search_query' using arXiv field prefixes such as all:, ti:, abs:, au:, cat:.
  - Expand well-known acronyms with OR clauses, e.g., (all:"LLM" OR all:"Large Language Model").
  - Strip conversational filler.
  - For recency requests, embed submittedDate INSIDE 'search_query' like this:
    all:"machine learning" AND submittedDate:[YYYYMMDD0000+TO+YYYYMMDD0000]
    (use +TO+, no spaces). Default 12-month window: {default_date_window()}
- 'start' and 'max_results' must be integers.
- Do not include placeholder text in id_list.
"""

    raw_response = ollama_generate(
        prompt=f"{system_prompt}\n\nQuery: {query}\nAssistant:",
        require_json=True,
    )
    return json.loads(raw_response)


def generate_arxiv_params(count: int = 1000, output_dir: Path | None = None) -> list[dict[str, Any]]:
    jsonl_path = None
    if output_dir:
        jsonl_path = output_dir / "arxiv_params.jsonl"

    dataset = _load_existing_jsonl(jsonl_path) if jsonl_path else []
    starting_count = len(dataset)
    remaining = count - starting_count
    print(
        f"Generating {remaining} arXiv parameter extraction examples using {model_name} "
        f"(already have {starting_count})..."
    )
    if remaining <= 0:
        print(f"Skipping arXiv params: target {count} already met.")
        return dataset

    instruction = "Extract arXiv API search parameters from the user query. Output valid JSON only."

    attempts = 0
    max_attempts = remaining * 3
    while len(dataset) < count and attempts < max_attempts:
        attempts += 1
        domain = random.choice(DOMAINS)
        is_id_query = random.random() < 0.2

        try:
            query = _generate_arxiv_query(domain, is_id_query)
            if not query:
                print(f"Validation failed (attempt {attempts}): empty query generated")
                continue

            params = _generate_arxiv_params_from_query(query)

            # Post-process: move standalone submittedDate into search_query if needed.
            if "submittedDate" in params and "search_query" in params:
                date_clause = params.pop("submittedDate")
                date_clause = date_clause.strip("[]")
                if "submittedDate:" not in date_clause:
                    date_clause = f"submittedDate:[{date_clause}]"
                params["search_query"] = f"({params['search_query']}) AND {date_clause}"

            output_string = json.dumps(params, ensure_ascii=False)

            item = {
                "instruction": instruction,
                "input": query,
                "output": output_string,
            }

            errors = validate_arxiv_params(item)
            if errors:
                print(f"Validation failed (attempt {attempts}): {errors}")
                continue

            dataset.append(item)
            _append_jsonl(jsonl_path, item)
            if len(dataset) % 100 == 0:
                print(f"arXiv params: validated {len(dataset)}/{count}")
        except Exception as exc:
            print(f"Error generating sample (attempt {attempts}): {exc}")

    if len(dataset) < count:
        print(f"WARNING: only generated {len(dataset)}/{count} valid arXiv params examples")

    return dataset


def _generate_rag_context(intent: str, domain: str, is_unanswerable: bool) -> tuple[str, str] | None:
    """Step 1: generate a synthetic query + context for a RAG example."""
    system_prompt = f"""You are generating synthetic training data for an arXiv RAG assistant.
Return a single JSON object with keys 'query', 'context'.

- 'query' is a realistic user query matching the given intent.
- 'context' is a string containing 1-5 chunks. Each chunk line must start with "[Chunk N] (arXiv:YYMM.NNNNN): '...'".

If unanswerable is true, make the context completely irrelevant to the query.
Otherwise, provide context that directly supports a clear answer."""

    user_prompt = f"Domain: {domain}, Intent: {intent}, Unanswerable: {is_unanswerable}."
    raw_response = ollama_generate(
        prompt=f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:",
        require_json=True,
    )
    parsed = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("context generation did not return a dict")
    return parsed.get("query", ""), parsed.get("context", "")


def _generate_rag_assistant(query: str, context: str, intent: str) -> dict[str, Any]:
    """Step 2: generate the assistant JSON response for a given query and context."""
    system_prompt = f"""{SYSTEM_PROMPT_BASE}
{INTENT_PROMPTS[intent]}

You must return a single JSON object (not a string) containing exactly:
- "query": the original query
- "intent": one of {VALID_INTENTS}
- "messages": a list of message objects
  - default -> one or two "text" messages. Occasionally use "list" for enumerations or "code" for short pseudocode.
  - comparison -> one "table" message with content as a LIST OF DICTS, followed by one "text" takeaway
  - citation_request -> one short "text" message followed by one "citations" message with content as a LIST OF STRINGS
  - paper_level_query -> one "text" message; "sources" must include the paper arXiv ID
- "sources": list of arXiv IDs actually used from the context
- "meta": dict; use empty dict {{}} when not adding metadata

Also vary message types across examples: include some "list" and "code" messages when the content naturally fits.

INSUFFICIENT CONTEXT RULE:
If the context does not contain the answer, keep the predicted intent, set "sources" to [], set "meta" to {{}}, and return a text message saying the answer cannot be determined from the provided arXiv material. NEVER use "insufficient_context" as an intent value.
"""

    user_prompt = f"Context (arXiv-only):\n{context}\n\nQuery: {query}\n\nImportant: Only use facts that appear in the Context above."
    raw_response = ollama_generate(
        prompt=f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:",
        require_json=True,
    )
    return json.loads(raw_response)


def generate_rag_responses(count: int = 1000, output_dir: Path | None = None) -> list[dict[str, Any]]:
    jsonl_path = None
    if output_dir:
        jsonl_path = output_dir / "rag_responses.jsonl"

    dataset = _load_existing_jsonl(jsonl_path) if jsonl_path else []
    starting_count = len(dataset)
    remaining = count - starting_count
    print(
        f"Generating {remaining} RAG response examples using {model_name} "
        f"(already have {starting_count})..."
    )
    if remaining <= 0:
        print(f"Skipping RAG responses: target {count} already met.")
        return dataset

    attempts = 0
    max_attempts = remaining * 4
    intent_cycle = 0
    while len(dataset) < count and attempts < max_attempts:
        attempts += 1
        domain = random.choice(DOMAINS)
        # Cycle through intents evenly instead of pure random sampling.
        intent = INTENTS[intent_cycle % len(INTENTS)]
        intent_cycle += 1
        is_unanswerable = random.random() < 0.15

        try:
            query, context = _generate_rag_context(intent, domain, is_unanswerable)
            if not query or not context:
                print(f"Validation failed (attempt {attempts}): missing query or context")
                continue

            assistant_payload = _generate_rag_assistant(query, context, intent)

            item = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_BASE + "\n" + INTENT_PROMPTS[intent],
                    },
                    {
                        "role": "user",
                        "content": f"Context (arXiv-only):\n{context}\n\nQuery: {query}\n\nImportant: Only use facts that appear in the Context above.",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(assistant_payload, ensure_ascii=False),
                    },
                ]
            }

            errors = validate_rag_response(item)
            if errors:
                print(f"Validation failed (attempt {attempts}): {errors}")
                continue

            dataset.append(item)
            _append_jsonl(jsonl_path, item)
            if len(dataset) % 100 == 0:
                print(f"RAG responses: validated {len(dataset)}/{count}")
        except Exception as exc:
            print(f"Error generating sample (attempt {attempts}): {exc}")

    if len(dataset) < count:
        print(f"WARNING: only generated {len(dataset)}/{count} valid RAG response examples")

    return dataset


def _generate_intent_query(intent: str, domain: str) -> str:
    """Generate a user query matching a specific intent and domain."""
    system_prompt = """You are generating realistic user queries for an arXiv research assistant.
Return a single JSON object with key 'query' containing a natural-language user query.

The query should match the given intent:
- default: a general question or explanation request.
- comparison: explicitly compares two or more methods, models, or theories.
- citation_request: asks for papers, sources, references, or citations.
- paper_level_query: asks about a specific arXiv paper ID.

Do NOT include instructions like "Domain:" or "Generate a" in the query. Just return the raw user query."""

    user_prompt = f"Domain: {domain}, Intent: {intent}. Generate a realistic user query."
    raw_response = ollama_generate(
        prompt=f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:",
        require_json=True,
    )
    parsed = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("intent query generation did not return a dict")
    return parsed.get("query", "").strip()


def generate_intent_dataset(count: int = 500, output_dir: Path | None = None) -> list[dict[str, Any]]:
    jsonl_path = None
    if output_dir:
        jsonl_path = output_dir / "intent_classification.jsonl"

    dataset = _load_existing_jsonl(jsonl_path) if jsonl_path else []
    starting_count = len(dataset)
    remaining = count - starting_count
    print(
        f"Generating {remaining} intent classification examples using {model_name} "
        f"(already have {starting_count})..."
    )
    if remaining <= 0:
        print(f"Skipping intent classification: target {count} already met.")
        return dataset

    instruction = (
        "Classify the user intent into one of: default, comparison, citation_request, paper_level_query."
    )

    attempts = 0
    max_attempts = remaining * 3
    intent_cycle = 0
    while len(dataset) < count and attempts < max_attempts:
        attempts += 1
        # Cycle through intents evenly.
        intent = INTENTS[intent_cycle % len(INTENTS)]
        intent_cycle += 1
        domain = random.choice(DOMAINS)

        try:
            query = _generate_intent_query(intent, domain)
            if not query:
                print(f"Validation failed (attempt {attempts}): empty query generated")
                continue

            item = {
                "instruction": instruction,
                "input": query,
                "output": intent,
            }

            errors = validate_intent_classification(item)
            if errors:
                print(f"Validation failed (attempt {attempts}): {errors}")
                continue

            dataset.append(item)
            _append_jsonl(jsonl_path, item)
            if len(dataset) % 100 == 0:
                print(f"Intent classification: validated {len(dataset)}/{count}")
        except Exception as exc:
            print(f"Error generating sample (attempt {attempts}): {exc}")

    if len(dataset) < count:
        print(f"WARNING: only generated {len(dataset)}/{count} valid intent examples")

    return dataset


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for the arXiv Research Assistant."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory where generated datasets and dataset_info.json will be written. Default: 'data'.",
    )
    parser.add_argument(
        "--arxiv-count",
        type=int,
        default=1000,
        help="Number of arXiv parameter extraction examples to generate.",
    )
    parser.add_argument(
        "--rag-count",
        type=int,
        default=1000,
        help="Number of RAG response examples to generate.",
    )
    parser.add_argument(
        "--intent-count",
        type=int,
        default=500,
        help="Number of intent classification examples to generate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = {
        "arxiv_params": {
            "file_name": "arxiv_params.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
        "rag_responses": {
            "file_name": "rag_responses.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
        },
        "intent_classification": {
            "file_name": "intent_classification.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
    }

    dataset_info_path = output_dir / "dataset_info.json"
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Wrote {dataset_info_path}")

    arxiv_params = generate_arxiv_params(args.arxiv_count, output_dir=output_dir)
    arxiv_path = output_dir / "arxiv_params.json"
    with open(arxiv_path, "w", encoding="utf-8") as f:
        json.dump(arxiv_params, f, indent=2)
    print(f"Wrote {arxiv_path} with {len(arxiv_params)} examples")

    rag_responses = generate_rag_responses(args.rag_count, output_dir=output_dir)
    rag_path = output_dir / "rag_responses.json"
    with open(rag_path, "w", encoding="utf-8") as f:
        json.dump(rag_responses, f, indent=2)
    print(f"Wrote {rag_path} with {len(rag_responses)} examples")

    intent_classification = generate_intent_dataset(args.intent_count, output_dir=output_dir)
    intent_path = output_dir / "intent_classification.json"
    with open(intent_path, "w", encoding="utf-8") as f:
        json.dump(intent_classification, f, indent=2)
    print(f"Wrote {intent_path} with {len(intent_classification)} examples")

    return 0


if __name__ == "__main__":
    sys.exit(main())
