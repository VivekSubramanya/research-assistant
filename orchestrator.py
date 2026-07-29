import logging
import os
import sqlite3
import re
import time

from chunk_and_embed import process_document
from llm import generate_answer, extract_arxiv_search_params, expand_query_for_embedding
from APIs import arxiv_api
from query_processing import process_query, get_chunks_for_paper


ARXIV_ONLY_ERROR = "Unable to answer from the provided arXiv context."

LOGGER = logging.getLogger("research_assistant")
if not LOGGER.handlers:
    logging.basicConfig(
        filename="logs.txt",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def build_llm_request(query, intent, retrieved_chunks, options=None):
    normalized_chunks = []
    for chunk in retrieved_chunks or []:
        if not chunk:
            continue
        text = chunk.get("chunk_text") if "chunk_text" in chunk else chunk.get("text")
        arxiv_id = chunk.get("arxiv_id")
        normalized_chunks.append({
            "chunk_id": chunk.get("chunk_id"),
            "arxiv_id": arxiv_id,
            "text": text,
            "score": chunk.get("score"),
        })

    return {
        "query": query,
        "intent": intent,
        "retrieved_chunks": normalized_chunks,
        "options": options or {"max_messages": 3},
    }


def _coerce_arxiv_only_response(response, fallback_message=ARXIV_ONLY_ERROR):
    if isinstance(response, dict):
        if isinstance(response.get("messages"), list) and response["messages"]:
            response.setdefault("sources", [])
            response.setdefault("meta", {})
            return response
        if response.get("error"):
            response["messages"] = [{"type": "text", "content": fallback_message}]
            response.setdefault("sources", [])
            response.setdefault("meta", {})
            return response
    return {
        "query": None,
        "intent": None,
        "messages": [{"type": "text", "content": fallback_message}],
        "sources": [],
        "meta": {},
    }


def _build_arxiv_error_response(query_text, intent, message):
    return {
        "query": query_text,
        "intent": intent,
        "messages": [{"type": "text", "content": message}],
        "sources": [],
        "meta": {"degraded": True, "source": "arxiv_fallback"},
    }


def _build_arxiv_fallback_results(query_text, top_k=5):
    LOGGER.info("stage=arxiv_fallback input=%s", {"query": query_text, "top_k": top_k})
    params = extract_arxiv_search_params(query_text, max_results=top_k)
    try:
        entries = arxiv_api.query_arxiv(params)
    except Exception as exc:
        LOGGER.exception("stage=arxiv_fallback error=%s", exc)
        return {
            "results": [],
            "error_message": str(exc) or ARXIV_ONLY_ERROR,
        }

    fallback_results = []
    for entry in entries[:top_k]:
        parsed = arxiv_api.parse_entry(entry)
        title = parsed.get("title") or ""
        summary = parsed.get("summary") or ""
        arxiv_id = parsed.get("id") or ""
        if arxiv_id:
            arxiv_id = arxiv_id.split("/abs/")[-1].split("/pdf/")[-1].replace("v", "") if "/" in arxiv_id else arxiv_id
        text = f"Title: {title}\nSummary: {summary}".strip()
        if text:
            fallback_results.append({
                "chunk_text": text,
                "arxiv_id": arxiv_id,
                "chunk_id": None,
                "score": None,
            })
    LOGGER.info("stage=arxiv_fallback output=%s", {"count": len(fallback_results), "params": params})
    return {
        "results": fallback_results,
        "error_message": None,
    }


def _answer_with_arxiv_context(query_text, intent, results):
    LOGGER.info("stage=answer_start input=%s", {"query": query_text, "intent": intent, "chunk_count": len(results or [])})
    fallback_error_message = None
    if not results:
        LOGGER.info("stage=empty_retrieval action=arxiv_fallback")
        fallback = _build_arxiv_fallback_results(query_text)
        results = fallback.get("results", [])
        fallback_error_message = fallback.get("error_message")
        LOGGER.info(
            "stage=arxiv_fallback_complete output=%s",
            {"chunk_count": len(results or []), "degraded": bool(fallback_error_message)},
        )

    if not results:
        if fallback_error_message:
            response = _build_arxiv_error_response(query_text, intent, fallback_error_message)
            LOGGER.info("stage=answer_end output=%s", response)
            return response
        response = _coerce_arxiv_only_response({"messages": []})
        LOGGER.info("stage=answer_end output=%s", response)
        return response

    llm_request = build_llm_request(query_text, intent, results)
    LOGGER.info("stage=llm_request output=%s", {"query": query_text, "intent": intent, "chunk_count": len(llm_request["retrieved_chunks"])})
    if not llm_request["retrieved_chunks"]:
        response = _coerce_arxiv_only_response({"messages": []})
        LOGGER.info("stage=answer_end output=%s", response)
        return response

    response = generate_answer(llm_request)
    if not response.get("messages"):
        response = _coerce_arxiv_only_response(response)
    response.setdefault("query", query_text)
    response.setdefault("intent", intent)
    response.setdefault("sources", [])
    response.setdefault("meta", {})
    LOGGER.info("stage=answer_end output=%s", response)
    return response


# --- Hybrid Reasoning Model ---
class HybridReasoningModel:
    def __init__(self, classifier):
        self.classifier = classifier

    def predict(self, query_text):
        q = query_text.lower()

        if re.search(r"\barxiv[:\s]*\d+\.\d{4,5}(?:v\d+)?\b", q) or re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", q):
            return "paper_level_query"
        if "source" in q or "reference" in q or "cite" in q:
            return "citation_request"
        if "compare" in q or "difference" in q:
            return "comparison"

        return self.classifier.predict(query_text)


# --- Orchestrator ---
class Orchestrator:
    def __init__(self, reasoning_model):
        self.reasoning_model = reasoning_model

    def document_exists(self, arxiv_id):
        conn = sqlite3.connect("research.db")
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM documents WHERE arxiv_id=?", (arxiv_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    @staticmethod
    def _normalize_arxiv_id(arxiv_id):
        if not arxiv_id:
            return None
        normalized = str(arxiv_id).strip()
        normalized = re.sub(r"^arxiv[:\s]*", "", normalized, flags=re.IGNORECASE)
        normalized = normalized.strip(" ,.;:!?()[]{}<>")
        normalized = re.sub(r"v\d+$", "", normalized, flags=re.IGNORECASE)
        return normalized or None

    def _paper_chunk_count(self, arxiv_id):
        conn = sqlite3.connect("research.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE arxiv_id=?", (arxiv_id,))
        row = cursor.fetchone()
        conn.close()
        return int(row[0]) if row and row[0] is not None else 0

    def _paper_is_ready(self, arxiv_id):
        conn = sqlite3.connect("research.db")
        cursor = conn.cursor()
        cursor.execute("SELECT title, file_path FROM documents WHERE arxiv_id=?", (arxiv_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return False
        title, file_path = row
        if not title:
            return False
        if not file_path or not os.path.exists(file_path):
            return False
        return self._paper_chunk_count(arxiv_id) > 0

    def _upsert_document_metadata(self, arxiv_id, file_path):
        title = None
        authors = ""
        abstract = ""
        categories = ""
        published_date = ""
        try:
            entries = arxiv_api.query_arxiv({"id_list": arxiv_id, "start": 0, "max_results": 1})
            if entries:
                parsed = arxiv_api.parse_entry(entries[0])
                title = parsed.get("title")
                authors = ", ".join(parsed.get("authors") or [])
                abstract = parsed.get("summary") or ""
                categories = ", ".join(parsed.get("categories") or [])
                published = parsed.get("published")
                published_date = str(published) if published is not None else ""
        except Exception as exc:
            LOGGER.warning("stage=paper_metadata_lookup warning=%s", str(exc))

        conn = sqlite3.connect("research.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO documents (arxiv_id, title, authors, abstract, categories, published_date, file_path, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (arxiv_id, title, authors, abstract, categories, published_date, file_path, time.time()),
        )
        conn.commit()
        conn.close()

    def extract_arxiv_id(self, query_text):
        match = re.search(r"arxiv[:\s]*(\d+\.\d{4,5}(?:v\d+)?)", query_text, re.IGNORECASE)
        if match:
            return self._normalize_arxiv_id(match.group(1))
        match = re.search(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b", query_text)
        return self._normalize_arxiv_id(match.group(1)) if match else None

    def execute(self, query_text, llm_expansion=None, override_top_k=None):
        start_time = time.time()
        intent = self.reasoning_model.predict(llm_expansion or query_text)
        searchable_query = expand_query_for_embedding(query_text)
        LOGGER.info("stage=intent output=%s", {"query": query_text, "intent": intent})

        if intent == "paper_level_query":
            arxiv_id = self.extract_arxiv_id(query_text)
            LOGGER.info("stage=paper_lookup output=%s", {"arxiv_id": arxiv_id})
            if arxiv_id and not self._paper_is_ready(arxiv_id):
                LOGGER.info("stage=ingest_missing_paper input=%s", {"arxiv_id": arxiv_id})
                print(f"Document {arxiv_id} is missing or incomplete. Ingesting automatically...")
                pdf_path = f"papers/{arxiv_id}.pdf"
                try:
                    # Re-download to avoid stale/corrupt local files for this ID.
                    arxiv_api.download_arxiv_pdf(arxiv_id, pdf_path)
                except Exception as exc:
                    LOGGER.exception("stage=ingest_missing_paper error=%s", exc)
                    print(f"Failed to download PDF for {arxiv_id}: {exc}")
                    return _coerce_arxiv_only_response({
                        "messages": [{"type": "text", "content": f"Unable to download paper {arxiv_id} from arXiv."}]
                    })
                process_document(arxiv_id, pdf_path)
                self._upsert_document_metadata(arxiv_id, pdf_path)
            if arxiv_id:
                paper_top_k = override_top_k if override_top_k is not None else None
                LOGGER.info(
                    "stage=retrieval input=%s",
                    {"source": "paper_chunks", "arxiv_id": arxiv_id, "top_k": paper_top_k if paper_top_k is not None else "all"},
                )
                results = get_chunks_for_paper(arxiv_id, top_k=paper_top_k)
            else:
                LOGGER.info("stage=retrieval input=%s", {"source": "semantic_search", "top_k": override_top_k or 10})
                results = process_query(searchable_query, top_k=override_top_k or 10)
            response = _answer_with_arxiv_context(query_text, intent, results)
            LOGGER.info("stage=orchestrator_complete output=%s", {"duration_s": round(time.time() - start_time, 3)})
            return response

        if intent == "comparison":
            LOGGER.info("stage=retrieval input=%s", {"source": "semantic_search", "top_k": 12})
            results = process_query(searchable_query, top_k=12)
            response = _answer_with_arxiv_context(query_text, intent, results)
            LOGGER.info("stage=orchestrator_complete output=%s", {"duration_s": round(time.time() - start_time, 3)})
            return response

        if intent == "citation_request":
            LOGGER.info("stage=retrieval input=%s", {"source": "semantic_search", "top_k": 8})
            results = process_query(searchable_query, top_k=8)
            response = _answer_with_arxiv_context(query_text, intent, results)
            LOGGER.info("stage=orchestrator_complete output=%s", {"duration_s": round(time.time() - start_time, 3)})
            return response

        LOGGER.info("stage=retrieval input=%s", {"source": "semantic_search", "top_k": override_top_k or 10})
        results = process_query(searchable_query, top_k=override_top_k or 10)
        response = _answer_with_arxiv_context(query_text, "default", results)
        LOGGER.info("stage=orchestrator_complete output=%s", {"duration_s": round(time.time() - start_time, 3)})
        return response


def handle_query(query_text):
    class DummyClassifier:
        def predict(self, _):
            return "default"

    orchestrator = Orchestrator(HybridReasoningModel(DummyClassifier()))
    return orchestrator.execute(query_text)
