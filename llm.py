import ast
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
import logging
import os
import re
import subprocess

import requests

ARXIV_CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "arXiv_categories.json")
LOGGER = logging.getLogger("research_assistant")
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "qwen2.5-14b-research-assistant.Q4_K_M.gguf"
)
LLAMA_CLI_PATH = os.path.join(os.path.dirname(__file__), "llama-cpp precompiled", "llama-cli.exe")
DEFAULT_CTX_SIZE = int(os.environ.get("RESEARCH_ASSISTANT_CTX_SIZE", "8192"))
DEFAULT_LLM_TIMEOUT_SECONDS = int(os.environ.get("RESEARCH_ASSISTANT_LLM_TIMEOUT_SECONDS", "180"))
DEFAULT_JSON_TOKENS = int(os.environ.get("RESEARCH_ASSISTANT_JSON_TOKENS", "256"))
DEFAULT_ANSWER_TOKENS = int(os.environ.get("RESEARCH_ASSISTANT_ANSWER_TOKENS", "768"))

# Built-in expansions for common acronyms / synonyms used in arXiv search.
# Keys are lower-cased; values are the canonical variants to OR together.
_TERM_EXPANSIONS = {
    "rag": ["RAG", "retrieval-augmented generation", "retrieval augmented generation", "document retrieval"],
    "retrieval-augmented generation": ["RAG", "retrieval augmented generation", "document retrieval"],
    "retrieval augmented generation": ["RAG", "retrieval-augmented generation", "document retrieval"],
    "llm": ["LLM", "large language model"],
    "llms": ["LLMs", "large language models"],
    "gnn": ["GNN", "graph neural network"],
    "gnns": ["GNNs", "graph neural networks"],
    "nlp": ["NLP", "natural language processing"],
    "cv": ["CV", "computer vision"],
    "rl": ["RL", "reinforcement learning"],
    "transformer": ["transformer", "transformers", "attention mechanism"],
    "transformers": ["transformers", "attention mechanism"],
    "fine-tuning": ["fine-tuning", "fine tuning", "finetuning"],
    "fine tuning": ["fine-tuning", "fine tuning", "finetuning"],
    "qa": ["QA", "question answering"],
    "mt": ["MT", "machine translation"],
    "sr": ["SR", "symbolic regression"],
}


def _expand_terms_for_query(query_text):
    """Return a set of additional terms/phrases for known acronyms/synonyms."""
    query_lower = query_text.lower()
    extra = set()
    for term, variants in _TERM_EXPANSIONS.items():
        pattern = r"(?:^|\W)" + re.escape(term) + r"(?:$|\W)"
        if re.search(pattern, query_lower):
            extra.update(variants)
    return extra


def _build_arxiv_or_clause(terms):
    """Build an arXiv OR clause from a list of terms, e.g. all:\"a\" OR all:\"b\"."""
    cleaned = [term.strip() for term in terms if term.strip()]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return f'all:"{cleaned[0]}"'
    clauses = [f'all:"{term}"' for term in cleaned]
    return "(" + " OR ".join(clauses) + ")"


def _core_search_terms(query_text):
    """Return the cleaned core search phrase after stripping conversational fluff."""
    candidate = _strip_code_fences(str(query_text).strip())
    if not candidate:
        return ""
    candidate = _strip_leading_phrases(candidate)
    lines = [line.strip(" -\t") for line in candidate.splitlines() if line.strip()]
    if lines:
        candidate = lines[0]
    candidate = re.sub(r"^(search terms?|query|keywords?)\s*:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;!?")
    return candidate


def _unique_terms_case_insensitive(terms):
    """Return unique terms preserving order, deduplicated case-insensitively."""
    seen = set()
    unique_terms = []
    for term in terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            unique_terms.append(term)
    return unique_terms


def expand_search_query(query_text):
    """Expand query_text into an arXiv search_query using OR synonyms."""
    core = _core_search_terms(query_text)
    if not core:
        return None

    # If the core contains comparison conjunctions, split and expand each side with AND.
    conjunction_pattern = r"\b(?:and|vs|versus|or)\b"
    parts = [part.strip(" ,:;") for part in re.split(conjunction_pattern, core, flags=re.IGNORECASE) if part.strip(" ,:;")]
    if len(parts) > 1:
        clauses = []
        for part in parts:
            extra = _expand_terms_for_query(part)
            terms = _unique_terms_case_insensitive([part] + sorted(extra))
            clause = _build_arxiv_or_clause(terms)
            if clause:
                clauses.append(clause)
        if clauses:
            return "(" + " AND ".join(clauses) + ")"

    extra = _expand_terms_for_query(core)
    if not extra:
        return f'all:"{core}"'
    terms = _unique_terms_case_insensitive([core] + sorted(extra))
    return _build_arxiv_or_clause(terms)


def expand_query_for_embedding(query_text):
    """Return a natural-language query enriched with known synonyms for embedding search."""
    core = _core_search_terms(query_text)
    base = (core or query_text).strip()
    extra = _expand_terms_for_query(base)
    if not extra:
        return base
    terms = _unique_terms_case_insensitive([base] + sorted(extra))
    return " ".join(terms)


def _heuristic_search_query(text):
    """Backward-compatible wrapper around expand_search_query."""
    return expand_search_query(text)


SYSTEM_PROMPT_BASE = """
You are a research assistant restricted to the provided arXiv context only.

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

Few-shot response examples:
Example A:
Query: "Tell me about aphids"
Output:
{"query":"Tell me about aphids","intent":"default","messages":[{"type":"text","content":"The provided arXiv context does not contain enough information about aphids to answer fully."}],"sources":[],"meta":{}}

Example B:
Query: "Compare method X and method Y"
Output:
{"query":"Compare method X and method Y","intent":"comparison","messages":[{"type":"table","content":[{"aspect":"dataset","method_x":"...","method_y":"..."}]},{"type":"text","content":"Comparison is based only on the provided arXiv context."}],"sources":[],"meta":{}}

Example C:
Query: "In arxiv:2101.12345, what dataset is used?"
Output:
{"query":"In arxiv:2101.12345, what dataset is used?","intent":"paper_level_query","messages":[{"type":"text","content":"From the provided chunks for arxiv:2101.12345, the paper uses [dataset if present in context]. If the dataset is not mentioned in context, state that it cannot be determined from the provided arXiv material."}],"sources":["2101.12345"],"meta":{}}

Example D:
Query: "Give citations for transformer papers"
Output:
{"query":"Give citations for transformer papers","intent":"citation_request","messages":[{"type":"text","content":"Here are citations found in the provided arXiv context."},{"type":"citations","content":["<title> - <authors> - arXiv:<id>"]}],"sources":[],"meta":{}}
"""

INTENT_PROMPTS = {
        "comparison": "Provide a structured side-by-side comparison. Prefer one table message then one short text takeaway.",
        "citation_request": "Answer concisely and include a citations message with only citations found in the provided context. The citation should include the title, authors, and arXiv ID if available.",
        "paper_level_query": "Answer using chunk details from the requested paper only.",
        "default": "Answer concisely using only the provided context.",
}


def call_ollama_mixtral(prompt, model=None, require_json=False):
    model_path = model or DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        LOGGER.error("stage=llm_call output=%s", {"transport": "local", "status": "missing"})
        return None

    if not os.path.exists(LLAMA_CLI_PATH):
        LOGGER.error("stage=llm_call output=%s", {"transport": "local", "status": "missing_cli"})
        return None

    LOGGER.info("stage=llm_call input=%s", {"transport": "local", "require_json": require_json})
    max_tokens = DEFAULT_JSON_TOKENS if require_json else DEFAULT_ANSWER_TOKENS
    command = [
        LLAMA_CLI_PATH,
        "-m",
        model_path,
        "-p",
        str(prompt),
        "-n",
        str(max_tokens),
        "--temp",
        "0",
        "--ctx-size",
        str(DEFAULT_CTX_SIZE),
        "--log-disable",
        "--simple-io",
        "--no-display-prompt",
        "--no-warmup",
        "--no-conversation",
        "--single-turn",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=DEFAULT_LLM_TIMEOUT_SECONDS,
            cwd=os.path.dirname(__file__),
            check=False,
        )
    except subprocess.TimeoutExpired:
        LOGGER.error("stage=llm_call output=%s", {"transport": "local", "status": "timeout"})
        return None
    except OSError as exc:
        LOGGER.exception("stage=llm_call output=%s", {"transport": "local", "status": "spawn_error", "error": str(exc)})
        return None

    cleaned_output = _extract_llama_cli_response(result.stdout, prompt)
    if cleaned_output:
        LOGGER.info(
            "stage=llm_call output=%s",
            {"transport": "local", "status": "ok", "returncode": result.returncode},
        )
        return cleaned_output

    stderr_text = (result.stderr or "").strip()
    LOGGER.error(
        "stage=llm_call output=%s",
        {
            "transport": "local",
            "status": "empty",
            "returncode": result.returncode,
            "stderr": stderr_text[:500],
        },
    )
    return None


def _extract_llama_cli_response(stdout_text, prompt_text):
    text = (stdout_text or "").replace("\r\n", "\n")
    lines = text.split("\n")

    # Trim runtime footer first so we can focus on prompt/response extraction.
    end = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if line.startswith("[ Prompt:") or stripped == "Exiting...":
            end = i
            break
    lines = lines[:end]

    prompt_idx = None
    for i, line in enumerate(lines):
        if line.startswith(">"):
            prompt_idx = i
            break

    if prompt_idx is not None:
        candidate_lines = lines[prompt_idx + 1 :]
    else:
        candidate_lines = lines

    while candidate_lines and not candidate_lines[0].strip():
        candidate_lines.pop(0)
    while candidate_lines and not candidate_lines[-1].strip():
        candidate_lines.pop()

    if candidate_lines:
        # llama-cli may echo large prompts; keep only text after known prompt tails.
        boundary_markers = (
            "Output JSON:",
            "Important: Only use facts that appear in the Context above.",
        )
        for marker in boundary_markers:
            for i in range(len(candidate_lines) - 1, -1, -1):
                if candidate_lines[i].strip().startswith(marker):
                    candidate_lines = candidate_lines[i + 1 :]
                    break

    # If prompt echo remains, take the last parseable object suffix.
    parsed_suffix = _extract_parseable_tail_block(candidate_lines)
    if parsed_suffix:
        return parsed_suffix

    candidate = "\n".join(candidate_lines).strip()
    candidate = _strip_llama_cli_startup_noise(candidate)
    if candidate:
        return candidate

    # Last-resort cleanup when no prompt marker was detected.
    cleaned = _strip_llama_cli_startup_noise("\n".join(lines))
    return cleaned or None


def _extract_parseable_tail_block(lines):
    """Return the last parseable object suffix from a list of lines, if any."""
    lines = [line for line in (lines or [])]
    for start in range(len(lines) - 1, -1, -1):
        block = "\n".join(lines[start:]).strip()
        if not block:
            continue
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(block)
                if isinstance(parsed, (dict, list, str, int, float, bool)):
                    return block
            except (json.JSONDecodeError, ValueError, SyntaxError, TypeError):
                continue
    return None


def _strip_llama_cli_startup_noise(text):
    """Remove llama-cli startup banner/help text while preserving model answers."""
    lines = (text or "").replace("\r\n", "\n").split("\n")
    if not lines:
        return ""

    metadata_pattern = re.compile(r"^(build|model|ftype|modalities)\s*:", re.IGNORECASE)
    metadata_hits = sum(1 for line in lines[:80] if metadata_pattern.match(line.strip()))
    saw_loading = any(line.strip().lower().startswith("loading model") for line in lines[:20])

    # Only strip aggressively if this looks like llama-cli startup output.
    if not (saw_loading or metadata_hits >= 2):
        return "\n".join(lines).strip()

    available_idx = None
    for index, line in enumerate(lines[:120]):
        if line.strip().lower() == "available commands:":
            available_idx = index
            break

    if available_idx is not None:
        cut = available_idx + 1
        while cut < len(lines):
            stripped = lines[cut].strip().lower()
            if not stripped:
                cut += 1
                continue
            if stripped.startswith("/"):
                cut += 1
                continue
            break
        lines = lines[cut:]
    else:
        # Fallback: trim leading banner-ish lines when command help block is absent.
        def _is_startup_line(raw_line):
            stripped = raw_line.strip()
            lower = stripped.lower()
            if not stripped:
                return True
            if lower.startswith("loading model"):
                return True
            if metadata_pattern.match(stripped):
                return True
            if lower == "available commands:":
                return True
            if lower.startswith("/"):
                return True
            # ASCII/Unicode art lines tend to contain no alphanumeric content.
            if not re.search(r"[a-z0-9]", lower):
                return True
            return False

        start = 0
        while start < len(lines) and _is_startup_line(lines[start]):
            start += 1
        lines = lines[start:]

    return "\n".join(lines).strip()


def warm_start_llm():
    """Optionally warm a persistent Ollama server if available."""
    try:
        call_ollama_mixtral("Hello")
    except Exception:
        pass


def _paper_chunk_priority(chunk):
    """Score chunk informativeness for paper-level summaries."""
    text = str(chunk.get("text") or chunk.get("chunk_text") or "")
    if not text:
        return -1.0

    lower = text.lower()
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    total_chars = max(len(text), 1)
    alpha_ratio = alpha_chars / total_chars

    keyword_bonus = 0.0
    for token in ("title", "abstract", "summary", "introduction", "we propose", "this paper"):
        if token in lower:
            keyword_bonus += 0.25

    # Penalize formula-heavy snippets that are less useful for overview queries.
    symbol_chars = sum(1 for ch in text if ch in "=+-*/^<>[]{}()")
    symbol_ratio = symbol_chars / total_chars
    symbol_penalty = 0.6 if symbol_ratio > 0.08 else 0.0

    length_bonus = min(total_chars / 4000.0, 0.6)
    return alpha_ratio + keyword_bonus + length_bonus - symbol_penalty


def _tokenize_relevance_terms(text):
    tokens = re.findall(r"[a-z0-9]{3,}", str(text or "").lower())
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into", "about", "paper", "arxiv",
        "tell", "what", "which", "where", "when", "does", "have", "show", "give", "details",
        "detail", "please", "explain", "summary",
    }
    return [token for token in tokens if token not in stopwords]


def _is_detailed_paper_query(query_text):
    query_lower = str(query_text or "").lower()
    detailed_markers = (
        "in detail",
        "detailed",
        "comprehensive",
        "deep dive",
        "thorough",
        "full summary",
        "summarize the paper",
        "tell me about paper",
        "tell me about arxiv",
        "overview of",
    )
    return any(marker in query_lower for marker in detailed_markers)


def _is_precise_paper_query(query_text):
    if _is_detailed_paper_query(query_text):
        return False

    query_lower = str(query_text or "").lower()
    precise_markers = (
        "dataset",
        "method",
        "architecture",
        "equation",
        "result",
        "metric",
        "ablation",
        "baseline",
        "limitation",
        "contribution",
        "experiment",
        "table",
        "figure",
        "section",
        "author",
    )
    wh_markers = ("what ", "which ", "where ", "when ", "who ", "how ")
    has_question_form = "?" in query_lower or any(marker in query_lower for marker in wh_markers)
    return has_question_form or any(marker in query_lower for marker in precise_markers)


def _paper_chunk_relevance_score(chunk, query_text):
    base = _paper_chunk_priority(chunk)
    query_terms = set(_tokenize_relevance_terms(query_text))
    text = str(chunk.get("text") or chunk.get("chunk_text") or "")
    chunk_terms = set(_tokenize_relevance_terms(text))
    overlap = len(query_terms.intersection(chunk_terms))
    return base + (0.4 * overlap)


def _paper_context_char_budget():
    # Reserve room for instructions + schema text; keep budget conservative.
    token_budget = max(DEFAULT_CTX_SIZE - 2500, 2500)
    return max(7500, int(token_budget * 1.8))


def _looks_like_context_overflow(raw_response):
    text = str(raw_response or "").lower()
    return "exceeds the available context size" in text or "request (" in text and "context size" in text


def _requested_top_n(query_text):
    query = str(query_text or "")
    match = re.search(r"\btop\s+(\d+)\s+(?:papers?|articles?|studies?)\b", query, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _is_arxiv_fallback_chunk(chunk):
    text = str(chunk.get("text") or chunk.get("chunk_text") or "").strip()
    if not text:
        return False
    return text.startswith("Title:") and "\nSummary:" in text


def _prompt_max_chunks(intent, query_text, retrieved_chunks, options=None):
    opts = options or {}
    try:
        configured = int(opts.get("max_context_chunks", 2))
    except (TypeError, ValueError):
        configured = 2

    max_chunks = max(1, configured)
    items = list(retrieved_chunks or [])
    if not items or intent == "paper_level_query":
        return max_chunks

    requested_top_n = _requested_top_n(query_text)
    if requested_top_n:
        max_chunks = max(max_chunks, min(max(requested_top_n, 3), 10))

    # Fallback results are already pre-ranked API entries; keep all of them.
    if items and all(_is_arxiv_fallback_chunk(chunk) for chunk in items):
        max_chunks = max(max_chunks, len(items))

    return max_chunks


def _select_context_chunks(intent, chunks, max_chunks):
    items = list(chunks or [])
    if not items:
        return []

    if intent != "paper_level_query":
        return items[:max_chunks]

    target = max(max_chunks, 8)
    scored = sorted(items, key=_paper_chunk_priority, reverse=True)
    return scored[:target]


def build_prompt(intent, query_text, chunks, max_chunks=2, max_context_chars=6000):
    selected_chunks = _select_context_chunks(intent, chunks, max_chunks)

    if intent == "paper_level_query":
        is_precise = _is_precise_paper_query(query_text)
        is_detailed = _is_detailed_paper_query(query_text)
        paper_budget = _paper_context_char_budget()
        if is_precise:
            selected_chunks = sorted(
                list(chunks or []),
                key=lambda chunk: _paper_chunk_relevance_score(chunk, query_text),
                reverse=True,
            )[:8]
            max_context_chars = max(max_context_chars, min(10000, paper_budget))
        elif is_detailed:
            selected_chunks = sorted(list(chunks or []), key=_paper_chunk_priority, reverse=True)
            max_context_chars = max(max_context_chars, paper_budget)
        else:
            selected_chunks = sorted(list(chunks or []), key=_paper_chunk_priority, reverse=True)[:20]
            max_context_chars = max(max_context_chars, min(14000, paper_budget))

    context_parts = []
    total_chars = 0
    truncated = False
    paper_ids = []
    for chunk in selected_chunks:
        arxiv_id = str(chunk.get("arxiv_id") or "").strip()
        if arxiv_id and arxiv_id not in paper_ids:
            paper_ids.append(arxiv_id)

    if intent == "paper_level_query" and paper_ids:
        context_parts.append("Requested paper context IDs: " + ", ".join(paper_ids))
        total_chars += len(context_parts[-1]) + 2

    for chunk in selected_chunks:
        text = chunk.get("text") or chunk.get("chunk_text", "") or ""
        text = str(text).strip()
        if not text:
            continue
        arxiv_id = str(chunk.get("arxiv_id") or "").strip()
        if arxiv_id:
            text = f"[arXiv:{arxiv_id}]\n{text}"
        if total_chars + len(text) + 2 > max_context_chars:
            remaining = max(max_context_chars - total_chars - 80, 0)
            if remaining > 0:
                context_parts.append(text[:remaining])
            truncated = True
            break
        context_parts.append(text)
        total_chars += len(text) + 2

    context = "\n\n".join(context_parts)
    if truncated or len(context) >= max_context_chars:
        context = context[: max_context_chars - 80].rstrip() + "\n\n[Context truncated due to length]"

    intent_instruction = INTENT_PROMPTS.get(intent, INTENT_PROMPTS["default"])
    if intent == "paper_level_query":
        if _is_precise_paper_query(query_text):
            intent_instruction += " Prioritize exact details directly related to the user's requested aspect of the paper."
        elif _is_detailed_paper_query(query_text):
            intent_instruction += " Provide a detailed paper summary covering problem, method, data, experiments, and key findings from context."

    return f"""
{SYSTEM_PROMPT_BASE}

{intent_instruction}

Context (arXiv-only):
{context}

Query: {query_text}

Important: Only use facts that appear in the Context above. Do not use outside knowledge or sources. If the Context is empty or insufficient, say that the answer cannot be determined from the provided arXiv material.
"""


def _strip_code_fences(text):
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|python|plaintext)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    return text


@lru_cache(maxsize=1)
def _load_arxiv_categories():
    try:
        with open(ARXIV_CATEGORIES_PATH, "r", encoding="utf-8") as category_file:
            data = json.load(category_file)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _subcategory_ids(prefix):
    category_ids = []
    for item in _load_arxiv_categories():
        category_id = item.get("id")
        if isinstance(category_id, str) and category_id.startswith(prefix + "."):
            category_ids.append(category_id)
    return category_ids


def _qbio_example_categories():
    category_ids = _subcategory_ids("q-bio")
    if category_ids:
        return category_ids[:4]
    return ["q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.NC"]


def _utc_now():
    return datetime.now(timezone.utc)


def _format_submitted_date(days_back):
    end = _utc_now()
    start = end - timedelta(days=int(days_back))
    return f"submittedDate:[{start.strftime('%Y%m%d%H%M')}+TO+{end.strftime('%Y%m%d%H%M')}]"


def _requested_window_days(query_text):
    query_lower = query_text.lower()

    duration_match = re.search(r"(?:last|past|previous)\s+(\d+)\s+(day|week|month|year)s?", query_lower)
    if duration_match:
        amount = int(duration_match.group(1))
        unit = duration_match.group(2)
        multipliers = {
            "day": 1,
            "week": 7,
            "month": 30,
            "year": 365,
        }
        return amount * multipliers[unit]

    if re.search(r"\brecent\b|\blatest\b|\bcurrent\b|\bright now\b|\bnew developments\b|\bwhat'?s happening\b", query_lower):
        return 90

    return None


def _strip_leading_phrases(text):
    cleaned = text.strip().strip('"').strip("'")
    patterns = [
        r"^(tell me about|tell me more about)\s+",
        r"^(what is|what's|whats)\s+",
        r"^(what are)\s+",
        r"^(give me|show me|find me|find|get me|list)\s+",
        r"^(papers? (about|on|for|describing|discussing|of))\s+",
        r"^(i want to know about)\s+",
        r"^(can you explain)\s+",
        r"^(explain|describe)\s+",
        r"^(the arxiv ids of|arxiv ids of|ids of|the ids of)\s+",
        r"^(two papers describing|two different types of|different types of|different approaches to|types of|approaches to)\s+",
        r"^(some|a few|several)\s+(papers?|articles?|studies?)\s+(about|on|for|describing|discussing|of)?\s*",
        r"^(compare|comparing|comparison of|difference between|differences between)\s+",
    ]
    # Apply repeatedly because some prefixes stack (e.g. "two papers describing two different types of ...").
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            new_cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
            if new_cleaned != cleaned:
                cleaned = new_cleaned
                changed = True
    return cleaned.strip(" .,:;!?\t\n\r")


def _heuristic_search_query(text):
    candidate = _strip_code_fences(str(text).strip())
    if not candidate:
        return None

    json_field_match = re.search(r'"search_query"\s*:\s*"([^"]+)"', candidate)
    if json_field_match:
        return json_field_match.group(1)

    quoted_all_match = re.search(r'all:\"(.+?)\"', candidate)
    if quoted_all_match:
        return f'all:"{quoted_all_match.group(1).strip()}"'

    candidate = _strip_leading_phrases(candidate)
    if not candidate:
        return None

    lines = [line.strip(" -\t") for line in candidate.splitlines() if line.strip()]
    if lines:
        candidate = lines[0]

    candidate = re.sub(r"^(search terms?|query|keywords?)\s*:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;!?")
    if not candidate:
        return None

    return f'all:"{candidate}"'


def _apply_temporal_window(params, query_text):
    requested_days = _requested_window_days(query_text)
    if requested_days is None:
        return params

    days_back = min(requested_days, 365)
    date_clause = _format_submitted_date(days_back)
    search_query = params.get("search_query")

    if search_query:
        if "submittedDate:[" in search_query:
            search_query = re.sub(r"submittedDate:\[[^\]]+\]", date_clause, search_query)
        else:
            search_query = f"({search_query}) AND {date_clause}"
        params["search_query"] = search_query

    params.setdefault("sortBy", "submittedDate")
    params.setdefault("sortOrder", "descending")
    return params


def _apply_broad_category_defaults(params, query_text):
    query_lower = query_text.lower()
    search_query = params.get("search_query", "")
    if params.get("id_list") or "cat:" in search_query:
        return params

    if "biology" not in query_lower and "q-bio" not in query_lower:
        return params

    qbio_categories = _qbio_example_categories()
    category_query = " OR ".join(f"cat:{category_id}*" for category_id in qbio_categories)
    params["search_query"] = f"({category_query})"
    params.setdefault("sortBy", "submittedDate")
    params.setdefault("sortOrder", "descending")
    return _apply_temporal_window(params, query_text)


def _coerce_response_payload(parsed, query_text=None, intent="default"):
    if isinstance(parsed, dict):
        if "messages" in parsed:
            return parsed
        return {
            "query": query_text,
            "intent": intent,
            "messages": [
                {"type": "text", "content": parsed.get("content") or parsed.get("answer") or str(parsed)}
            ],
            "sources": parsed.get("sources", []),
            "meta": {},
        }

    if isinstance(parsed, list):
        return {
            "query": query_text,
            "intent": intent,
            "messages": [{"type": "text", "content": str(parsed)}],
            "sources": [],
            "meta": {},
        }

    return {
        "query": query_text,
        "intent": intent,
        "messages": [{"type": "text", "content": str(parsed)}],
        "sources": [],
        "meta": {},
    }


def _fallback_response_from_chunks(query_text, intent, retrieved_chunks):
    chunk_texts = []
    sources = []
    for chunk in retrieved_chunks or []:
        if not chunk:
            continue
        text = chunk.get("text") or chunk.get("chunk_text") or ""
        text = str(text).strip()
        if not text:
            continue
        chunk_texts.append(text)
        arxiv_id = chunk.get("arxiv_id")
        if arxiv_id and arxiv_id not in sources:
            sources.append(arxiv_id)

    if not chunk_texts:
        return {
            "query": query_text,
            "intent": intent,
            "messages": [{"type": "text", "content": "Unable to generate answer from the provided context."}],
            "sources": [],
            "meta": {},
        }

    context_preview = "\n\n".join(chunk_texts[:3])
    if len(context_preview) > 1800:
        context_preview = context_preview[:1800].rstrip() + "..."

    return {
        "query": query_text,
        "intent": intent,
        "messages": [{
            "type": "text",
            "content": "The model did not return a usable response, so I’m answering from the available context:\n\n" + context_preview,
        }],
        "sources": sources,
        "meta": {},
    }


def normalize_llm_response(raw_response, query_text=None, intent="default"):
    if isinstance(raw_response, dict):
        return _coerce_response_payload(raw_response, query_text=query_text, intent=intent)

    if isinstance(raw_response, bytes):
        raw_response = raw_response.decode("utf-8", errors="ignore")

    if not raw_response:
        return {
            "query": query_text,
            "intent": intent,
            "messages": [{"type": "text", "content": "Unable to generate answer"}],
            "sources": [],
            "meta": {},
        }

    text = str(raw_response).strip()
    if not text:
        return {
            "query": query_text,
            "intent": intent,
            "messages": [{"type": "text", "content": "Unable to generate answer"}],
            "sources": [],
            "meta": {},
        }

    text = _strip_code_fences(text)

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            return _coerce_response_payload(parsed, query_text=query_text, intent=intent)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            return _coerce_response_payload(parsed, query_text=query_text, intent=intent)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
                return _coerce_response_payload(parsed, query_text=query_text, intent=intent)
            except (ValueError, SyntaxError):
                pass

    return {
        "query": query_text,
        "intent": intent,
        "messages": [{"type": "text", "content": text}],
        "sources": [],
        "meta": {},
    }


def extract_search_terms(query_text):
    params = extract_arxiv_search_params(query_text, max_results=5)
    search_query = params.get("search_query", "")
    # Single quoted term
    match = re.match(r'^all:\"(.+)\"$', search_query)
    if match:
        return match.group(1).strip()
    # OR clause: return the first quoted term
    match = re.search(r'all:\"([^\"]+)\"', search_query)
    if match:
        return match.group(1).strip()
    return query_text


def extract_arxiv_search_params(query_text, max_results=5):
    """Build arXiv API params semantically from user query."""
    LOGGER.info("stage=extract_arxiv_params_start input=%s", {"query": query_text, "max_results": max_results})
    qbio_categories = _qbio_example_categories()
    qbio_example_query = " OR ".join(f"cat:{category_id}*" for category_id in qbio_categories)
    extract_prompt = f"""
You convert user queries to arXiv API params.
Return STRICT JSON ONLY using this schema:
{{
    "search_query": "string",
    "id_list": "optional comma-separated ids",
    "start": 0,
    "max_results": 5,
    "sortBy": "optional: relevance|lastUpdatedDate|submittedDate",
    "sortOrder": "optional: ascending|descending"
}}

arXiv field rules (from API manual usage pattern):
- search_query: use field prefixes like all:, ti:, au:, abs:, cat:, jr:, rn:.
- id_list: use this for explicit arXiv ids instead of search_query=id:.
- start: integer >= 0.
- max_results: integer > 0.
- sortBy/sortOrder are optional.
- For vague recency requests such as recent, latest, current, right now, or new developments, add a submittedDate range covering the last 3 months.
- If the user asks for a relative time period longer than 1 year, cap the submittedDate range to the last 12 months.
- For broad archive-level category requests, expand to a few representative subcategories instead of a single generic keyword.
- Expand well-known acronyms and synonyms using OR clauses when they appear in the query (e.g., RAG, LLM, GNN, fine-tuning).
- Strip conversational filler such as "give me", "find me", "two papers describing", "different types of", etc.
- Never include placeholder text like "optional, comma-separated ids" in id_list; omit the field entirely if no IDs are present.

Few-shot examples:
Input: "Tell me about aphids"
Output: {{"search_query":"all:\"aphids\"","start":0,"max_results":5}}

Input: "papers by Yann LeCun on convolution"
Output: {{"search_query":"au:\"Yann LeCun\" AND all:\"convolution\"","start":0,"max_results":5}}

Input: "recent graph neural network papers"
Output: {{"search_query":"all:\"graph neural network\"","start":0,"max_results":5,"sortBy":"submittedDate","sortOrder":"descending"}}

Input: "Tell me about recent developments in data science"
Output: {{"search_query":"all:\"data science\" AND submittedDate:[202604210000+TO+202607210000]","start":0,"max_results":5,"sortBy":"submittedDate","sortOrder":"descending"}}

Input: "Compare machine learning papers from the last 2 years"
Output: {{"search_query":"all:\"machine learning\" AND submittedDate:[202507210000+TO+202607210000]","start":0,"max_results":5,"sortBy":"submittedDate","sortOrder":"descending"}}

Input: "What's happening in biology right now"
Output: {{"search_query":"({qbio_example_query}) AND submittedDate:[202604210000+TO+202607210000]","start":0,"max_results":5,"sortBy":"submittedDate","sortOrder":"descending"}}

Input: "Give me papers about RAG"
Output: {{"search_query":"(all:\"RAG\" OR all:\"retrieval-augmented generation\" OR all:\"document retrieval\")","start":0,"max_results":5}}

Input: "Compare RAG and fine-tuning"
Output: {{"search_query":"(all:\"RAG\" OR all:\"retrieval-augmented generation\") AND (all:\"fine-tuning\" OR all:\"fine tuning\")","start":0,"max_results":5}}

Input: "What are two different approaches to retrieval-augmented generation?"
Output: {{"search_query":"(all:\"retrieval-augmented generation\" OR all:\"RAG\")","start":0,"max_results":5}}

Input: "arxiv 2301.12345"
Output: {{"id_list":"2301.12345","start":0,"max_results":5}}

Now convert:
Input: "{query_text}"
Output JSON:
"""

    raw_response = call_ollama_mixtral(extract_prompt, require_json=True)
    # If the caller requested strict JSON and the helper returned a plain
    # human-readable status string (e.g. "Local GGUF model path selected: ..."
    # or any non-JSON text), treat it as no response so we fall back to the
    # heuristic/default parsing based on the original user query.
    if isinstance(raw_response, str):
        text_try = _strip_code_fences(raw_response).strip()
        parsed_ok = False
        for parser in (json.loads, ast.literal_eval):
            try:
                parser(text_try)
                parsed_ok = True
                break
            except Exception:
                continue
        if not parsed_ok:
            raw_response = None
    default_params = {
        "search_query": f'all:"{query_text}"',
        "start": 0,
        "max_results": int(max_results),
    }
    heuristic_default_params = {
        "search_query": _heuristic_search_query(query_text) or default_params["search_query"],
        "start": 0,
        "max_results": int(max_results),
    }
    if not raw_response:
        result = _apply_broad_category_defaults(_apply_temporal_window(heuristic_default_params, query_text), query_text)
        LOGGER.info("stage=extract_arxiv_params_end output=%s", {"source": "default", "params": result})
        return result

    text = _strip_code_fences(str(raw_response).strip())
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, dict):
                search_query = str(parsed.get("search_query", default_params["search_query"]))
                start = int(parsed.get("start", 0))
                max_r = int(parsed.get("max_results", max_results))
                if start < 0:
                    start = 0
                if max_r <= 0:
                    max_r = int(max_results)
                result = {
                    "search_query": search_query,
                    "start": start,
                    "max_results": max_r,
                }
                id_list = parsed.get("id_list")
                if id_list and str(id_list).strip().lower() not in {
                    "",
                    "optional",
                    "none",
                    "null",
                    "comma-separated ids",
                    "optional comma-separated ids",
                    "optional, comma-separated ids",
                    "comma separated ids",
                }:
                    result["id_list"] = str(id_list).strip()
                sort_by = parsed.get("sortBy")
                if sort_by in {"relevance", "lastUpdatedDate", "submittedDate"}:
                    result["sortBy"] = sort_by
                sort_order = parsed.get("sortOrder")
                if sort_order in {"ascending", "descending"}:
                    result["sortOrder"] = sort_order
                result = _apply_temporal_window(result, query_text)
                result = _apply_broad_category_defaults(result, query_text)
                LOGGER.info("stage=extract_arxiv_params_end output=%s", {"source": "llm", "params": result})
                return result
        except (json.JSONDecodeError, ValueError, SyntaxError, TypeError):
            continue

    heuristic_query = _heuristic_search_query(text)
    if heuristic_query:
        result = {
            "search_query": heuristic_query,
            "start": 0,
            "max_results": int(max_results),
        }
        result = _apply_temporal_window(result, query_text)
        result = _apply_broad_category_defaults(result, query_text)
        LOGGER.info("stage=extract_arxiv_params_end output=%s", {"source": "heuristic_parse", "params": result})
        return result

    result = _apply_broad_category_defaults(_apply_temporal_window(heuristic_default_params, query_text), query_text)
    LOGGER.info("stage=extract_arxiv_params_end output=%s", {"source": "fallback_parse", "params": result})
    return result


def generate_answer(request):
    if isinstance(request, tuple):
        query_text, chunks, intent = request
        request = {
            "query": query_text,
            "intent": intent,
            "retrieved_chunks": chunks,
            "options": {"max_messages": 3},
        }

    query_text = request.get("query", "")
    intent = request.get("intent", "default")
    retrieved_chunks = request.get("retrieved_chunks", [])
    options = request.get("options") or {}
    max_chunks = _prompt_max_chunks(intent, query_text, retrieved_chunks, options)
    prompt = build_prompt(intent, query_text, retrieved_chunks, max_chunks=max_chunks)
    raw_response = call_ollama_mixtral(prompt)

    if intent == "paper_level_query" and _looks_like_context_overflow(raw_response):
        LOGGER.warning("stage=llm_retry reason=context_overflow intent=%s", intent)
        compact_prompt = build_prompt(intent, query_text, retrieved_chunks, max_context_chars=7000)
        raw_response = call_ollama_mixtral(compact_prompt)

    response = normalize_llm_response(raw_response, query_text=query_text, intent=intent)
    messages = response.get("messages") or []
    if not messages:
        return _fallback_response_from_chunks(query_text, intent, retrieved_chunks)

    first_message = messages[0] if isinstance(messages[0], dict) else {}
    content = str(first_message.get("content", "") or "").strip().lower()
    if content in {"", "unable to generate answer"} and retrieved_chunks:
        return _fallback_response_from_chunks(query_text, intent, retrieved_chunks)

    return response
