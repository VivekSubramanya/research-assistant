import importlib
import os
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import requests
import chunk_and_embed
import data_ingestion

from APIs import arxiv_api
from data_ingestion import init_db, download_pdf
from query_processing import adaptive_top_k
from llm import normalize_llm_response, extract_arxiv_search_params, expand_search_query, expand_query_for_embedding, build_prompt, generate_answer, call_ollama_mixtral, _extract_llama_cli_response
from orchestrator import build_llm_request, Orchestrator, HybridReasoningModel


class ArxivApiTests(unittest.TestCase):
    def test_parse_entry_extracts_expected_fields(self):
        entry = {
            "id": "http://arxiv.org/abs/2401.00001v1",
            "title": "Test Paper",
            "summary": "A summary",
            "authors": [{"name": "Ada"}, {"name": "Grace"}],
            "published": "2024-01-01T00:00:00Z",
            "tags": [{"term": "cs.AI"}],
            "links": [{"href": "https://example.com/paper.pdf", "title": "pdf"}],
        }

        parsed = arxiv_api.parse_entry(entry)
        self.assertEqual(parsed["title"], "Test Paper")
        self.assertEqual(parsed["authors"], ["Ada", "Grace"])
        self.assertEqual(parsed["pdf_url"], "https://example.com/paper.pdf")

    @patch("APIs.arxiv_api.time.sleep", return_value=None)
    @patch("APIs.arxiv_api.requests.get")
    def test_query_arxiv_retries_timeout_then_raises_graceful_error(self, mock_get, _mock_sleep):
        mock_get.side_effect = requests.exceptions.ReadTimeout("timed out")

        with self.assertRaises(arxiv_api.ArxivQueryError) as context:
            arxiv_api.query_arxiv({"search_query": 'all:"data sourcing"', "start": 0, "max_results": 5})

        self.assertTrue(context.exception.is_temporary)
        self.assertIn("taking too long", str(context.exception))
        self.assertEqual(mock_get.call_count, arxiv_api.ARXIV_RETRY_ATTEMPTS)


class IngestionTests(unittest.TestCase):
    def test_init_db_creates_expected_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            import importlib
            import data_ingestion
            data_ingestion.DB_PATH = db_path
            init_db()

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
            self.assertIsNotNone(cursor.fetchone())
            conn.close()

    @patch("data_ingestion.requests.get")
    def test_download_pdf_writes_file(self, mock_get):
        class Response:
            def __init__(self):
                self.content = b"pdf-bytes"
            def raise_for_status(self):
                return None

        mock_get.return_value = Response()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_pdf_dir = data_ingestion.PDF_DIR
            data_ingestion.PDF_DIR = tmpdir + os.sep
            try:
                path = download_pdf("https://example.com/paper.pdf", "1234")
                self.assertTrue(os.path.exists(path))
                self.assertTrue(path.endswith("1234.pdf"))
            finally:
                data_ingestion.PDF_DIR = old_pdf_dir


class PdfExtractionTests(unittest.TestCase):
    def test_extract_text_from_pdf_uses_pymupdf_and_closes_document(self):
        page = unittest.mock.MagicMock()
        page.get_text.return_value = [(0, 0, 0, 0, " Primary paragraph ")]
        document = unittest.mock.MagicMock()
        document.__iter__.return_value = iter([page])
        pymupdf = unittest.mock.MagicMock()
        pymupdf.open.return_value = document

        with patch.object(chunk_and_embed, "fitz", pymupdf):
            paragraphs = chunk_and_embed.extract_text_from_pdf("paper.pdf")

        self.assertEqual(paragraphs, ["Primary paragraph"])
        document.close.assert_called_once()

    def test_extract_text_from_pdf_falls_back_to_pypdf(self):
        page = unittest.mock.MagicMock()
        page.extract_text.return_value = "First line\nSecond line"
        reader = unittest.mock.MagicMock()
        reader.return_value.pages = [page]

        with patch.object(chunk_and_embed, "fitz", None), \
             patch.object(chunk_and_embed, "PdfReader", reader):
            paragraphs = chunk_and_embed.extract_text_from_pdf("paper.pdf")

        self.assertEqual(paragraphs, ["First line", "Second line"])


class RetrievalTests(unittest.TestCase):
    def test_adaptive_top_k_returns_expected_values(self):
        self.assertEqual(adaptive_top_k([[0.01, 0.02, 0.03]]), 5)
        self.assertEqual(adaptive_top_k([[0.01, 0.2, 0.4]]), 20)

    def test_import_does_not_load_faiss_index_eagerly(self):
        import query_processing

        with patch("faiss.read_index", side_effect=AssertionError("faiss should not be read at import time")):
            importlib.reload(query_processing)

        self.assertIsNone(query_processing.faiss_index)
        self.assertEqual(adaptive_top_k([[0.01, 0.5, 1.0]]), 20)


class OrchestratorContractTests(unittest.TestCase):
    def test_build_llm_request_uses_chunk_text(self):
        payload = build_llm_request("Q", "precise_qa", [{"chunk_text": "hello", "arxiv_id": "123", "chunk_id": 7}])
        self.assertEqual(payload["retrieved_chunks"][0]["text"], "hello")

    def test_execute_falls_back_to_arxiv_api_when_local_results_are_empty(self):
        class DummyClassifier:
            def predict(self, _):
                return "default"

        orchestrator = Orchestrator(HybridReasoningModel(DummyClassifier()))

        with patch("orchestrator.process_query", return_value=[]), \
             patch("orchestrator._build_arxiv_fallback_results", return_value=[{"chunk_text": "Title: Test\nSummary: summary", "arxiv_id": "1234.5678"}]) as mock_fallback, \
             patch("orchestrator.generate_answer", return_value={"messages": [{"type": "text", "content": "ok"}]}) as mock_generate:
            response = orchestrator.execute("aphids")

        self.assertEqual(mock_fallback.call_count, 1)
        self.assertEqual(mock_generate.call_count, 1)
        self.assertEqual(mock_generate.call_args[0][0]["retrieved_chunks"][0]["arxiv_id"], "1234.5678")
        self.assertEqual(response["messages"][0]["content"], "ok")

    def test_normalize_llm_response_handles_plain_text(self):
        response = normalize_llm_response("Plain answer", query_text="What?", intent="default")
        self.assertIn("messages", response)
        self.assertEqual(response["messages"][0]["content"], "Plain answer")

    def test_normalize_llm_response_parses_python_style_object_string(self):
        raw = "{'messages': [{'type': 'text', 'content': 'Parsed answer'}]}"
        response = normalize_llm_response(raw, query_text="What?", intent="default")
        self.assertEqual(response["messages"][0]["content"], "Parsed answer")

    @patch("llm.call_ollama_mixtral", return_value="data sourcing")
    def test_extract_arxiv_search_params_uses_plain_text_llm_output(self, _mock_call):
        params = extract_arxiv_search_params("Tell me about data sourcing", max_results=5)
        self.assertEqual(params["search_query"], 'all:"data sourcing"')

    @patch("llm.call_ollama_mixtral", return_value=None)
    def test_extract_arxiv_search_params_uses_heuristic_query_fallback(self, _mock_call):
        params = extract_arxiv_search_params("Tell me about data sourcing", max_results=5)
        self.assertEqual(params["search_query"], 'all:"data sourcing"')

    @patch("llm.call_ollama_mixtral", return_value=None)
    def test_generate_answer_falls_back_to_chunk_summary_when_llm_is_unavailable(self, _mock_call):
        response = generate_answer({
            "query": "Explain the paper",
            "intent": "default",
            "retrieved_chunks": [{"text": "Paper summary about symbolic regression."}],
            "options": {"max_messages": 3},
        })

        self.assertIn("messages", response)
        self.assertTrue(response["messages"])
        self.assertIn("symbolic regression", response["messages"][0]["content"].lower())

    def test_generate_answer_includes_all_fallback_papers_for_top_n_query(self):
        captured = {}

        def _fake_llm(prompt, model=None, require_json=False):
            captured["prompt"] = prompt
            return '{"query":"q","intent":"default","messages":[{"type":"text","content":"ok"}],"sources":[],"meta":{}}'

        chunks = [
            {"text": f"Title: Paper {i}\nSummary: Summary {i}", "arxiv_id": f"2401.0000{i}"}
            for i in range(1, 6)
        ]

        with patch("llm.call_ollama_mixtral", side_effect=_fake_llm):
            response = generate_answer({
                "query": "What are the top 5 papers for computer science?",
                "intent": "default",
                "retrieved_chunks": chunks,
                "options": {"max_messages": 3},
            })

        self.assertEqual(response["messages"][0]["content"], "ok")
        prompt = captured.get("prompt", "")
        for i in range(1, 6):
            self.assertIn(f"Title: Paper {i}", prompt)

    def test_execute_returns_graceful_message_when_arxiv_fallback_times_out(self):
        class DummyClassifier:
            def predict(self, _):
                return "default"

        orchestrator = Orchestrator(HybridReasoningModel(DummyClassifier()))

        with patch("orchestrator.process_query", return_value=[]), \
             patch("orchestrator.arxiv_api.query_arxiv", side_effect=arxiv_api.ArxivQueryError("arXiv is taking too long to respond right now. Please try again shortly.", is_temporary=True)):
            response = orchestrator.execute("Tell me about data sourcing")

        self.assertEqual(
            response["messages"][0]["content"],
            "arXiv is taking too long to respond right now. Please try again shortly.",
        )
        self.assertTrue(response["meta"]["degraded"])
        self.assertEqual(response["meta"]["source"], "arxiv_fallback")

    def test_hybrid_reasoning_detects_versioned_arxiv_id(self):
        class DummyClassifier:
            def predict(self, _):
                return "default"

        model = HybridReasoningModel(DummyClassifier())
        self.assertEqual(model.predict("Tell me about paper 2406.15531v2"), "paper_level_query")

    def test_extract_arxiv_id_normalizes_version_suffix(self):
        class DummyClassifier:
            def predict(self, _):
                return "default"

        orchestrator = Orchestrator(HybridReasoningModel(DummyClassifier()))
        self.assertEqual(orchestrator.extract_arxiv_id("Tell me about arXiv:2406.15531v2"), "2406.15531")

    def test_execute_reingests_when_paper_record_is_incomplete(self):
        class DummyClassifier:
            def predict(self, _):
                return "default"

        orchestrator = Orchestrator(HybridReasoningModel(DummyClassifier()))

        with patch.object(orchestrator, "_paper_is_ready", return_value=False), \
             patch("orchestrator.arxiv_api.download_arxiv_pdf", return_value="papers/2406.15531.pdf") as mock_download, \
             patch("orchestrator.process_document") as mock_process, \
             patch.object(orchestrator, "_upsert_document_metadata") as mock_upsert, \
             patch("orchestrator.get_chunks_for_paper", return_value=[{"chunk_text": "Title: X\nSummary: Y", "arxiv_id": "2406.15531"}]), \
             patch("orchestrator.generate_answer", return_value={"messages": [{"type": "text", "content": "ok"}]}) as mock_generate:
            response = orchestrator.execute("Tell me about paper 2406.15531v2")

        self.assertEqual(mock_download.call_count, 1)
        self.assertEqual(mock_process.call_count, 1)
        self.assertEqual(mock_upsert.call_count, 1)
        self.assertEqual(mock_generate.call_count, 1)
        self.assertEqual(response["messages"][0]["content"], "ok")

    def test_extract_llama_cli_response_returns_completion_only(self):
        stdout = """
Loading model...

> Reply with exactly: OK
OK

[ Prompt: 830.9 t/s | Generation: 102.4 t/s ]

Exiting...
"""

        self.assertEqual(_extract_llama_cli_response(stdout, "Reply with exactly: OK"), "OK")

    def test_extract_llama_cli_response_handles_truncated_prompt_echo(self):
        stdout = """
Loading model...

available commands:
    /exit or Ctrl+C     stop or exit

> 
You are a research assistant restricted to the provided arXiv context only.
Rules:
- Return JSON only.
Output JSON:
{"search_query":"all:\"data sourcing\"","start":0,"max_results":5}

[ Prompt: 3246.2 t/s | Generation: 62.6 t/s ]

Exiting...
"""

        self.assertEqual(
            _extract_llama_cli_response(stdout, "ignored prompt"),
            '{"search_query":"all:\"data sourcing\"","start":0,"max_results":5}',
        )

    def test_extract_llama_cli_response_handles_prompt_echo_without_boundary_marker(self):
        stdout = """
Loading model...

> 
You are a research assistant restricted to the provided arXiv context only.
Frontend response JSON schema:
{
    "query": "<original user query>",
    "messages": [
        {"type": "text", "content": "value ... (truncated)
{"query":"Tell me about data sourcing","intent":"default","messages":[{"type":"text","content":"Short answer"}],"sources":[],"meta":{}}

[ Prompt: 1200.0 t/s | Generation: 70.0 t/s ]
Exiting...
"""

        self.assertEqual(
            _extract_llama_cli_response(stdout, "ignored prompt"),
            '{"query":"Tell me about data sourcing","intent":"default","messages":[{"type":"text","content":"Short answer"}],"sources":[],"meta":{}}',
        )

    @patch("llm.os.path.exists", return_value=True)
    @patch("llm.subprocess.run")
    def test_call_ollama_mixtral_accepts_valid_output_with_nonzero_exit(self, mock_run, _mock_exists):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["llama-cli.exe"],
            returncode=1,
            stdout="""
Loading model...

> Reply with exactly: OK
OK

[ Prompt: 830.9 t/s | Generation: 102.4 t/s ]

Exiting...
""",
            stderr="",
        )

        self.assertEqual(call_ollama_mixtral("Reply with exactly: OK"), "OK")
        command = mock_run.call_args.args[0]
        self.assertIn("--single-turn", command)


class PromptFormattingTests(unittest.TestCase):
    def test_build_prompt_truncates_large_context(self):
        long_text = "A" * 15000
        prompt = build_prompt("default", "What happened?", [{"text": long_text}], max_chunks=1, max_context_chars=4000)
        self.assertLessEqual(len(prompt), 7000)
        self.assertIn("[Context truncated", prompt)

    def test_build_prompt_paper_level_includes_arxiv_labels_and_extra_chunks(self):
        chunks = [
            {"arxiv_id": "2406.15531", "text": "Title: cp3-bench"},
            {"arxiv_id": "2406.15531", "text": "Abstract: benchmark for symbolic regression."},
            {"arxiv_id": "2406.15531", "text": "Introduction: we propose a benchmark suite."},
        ]
        prompt = build_prompt("paper_level_query", "Tell me about paper 2406.15531", chunks, max_chunks=2, max_context_chars=4000)
        self.assertIn("Requested paper context IDs: 2406.15531", prompt)
        self.assertGreaterEqual(prompt.count("[arXiv:2406.15531]"), 3)

    def test_build_prompt_paper_level_precise_query_limits_to_top_eight_chunks(self):
        chunks = [{"arxiv_id": "2406.15531", "text": f"chunk {i} with dataset and method details"} for i in range(12)]
        prompt = build_prompt(
            "paper_level_query",
            "What dataset is used in paper 2406.15531?",
            chunks,
            max_chunks=2,
            max_context_chars=12000,
        )
        self.assertLessEqual(prompt.count("[arXiv:2406.15531]"), 8)

    def test_build_prompt_paper_level_detailed_query_uses_more_than_eight_chunks(self):
        chunks = [{"arxiv_id": "2406.15531", "text": f"Title and summary content block {i} for detailed overview"} for i in range(18)]
        prompt = build_prompt(
            "paper_level_query",
            "Tell me about paper 2406.15531 in detail",
            chunks,
            max_chunks=2,
            max_context_chars=25000,
        )
        self.assertGreater(prompt.count("[arXiv:2406.15531]"), 8)


class QueryExpansionTests(unittest.TestCase):
    def test_expand_search_query_expands_rag(self):
        query = "Give me papers about RAG"
        result = expand_search_query(query)
        self.assertIn('all:"RAG"', result)
        self.assertIn('all:"retrieval-augmented generation"', result)
        self.assertIn("OR", result)

    def test_expand_search_query_without_expansion(self):
        query = "data sourcing"
        result = expand_search_query(query)
        self.assertEqual(result, 'all:"data sourcing"')

    def test_expand_query_for_embedding_enriches_terms(self):
        query = "What is RAG?"
        result = expand_query_for_embedding(query)
        self.assertIn("RAG", result)
        self.assertIn("retrieval-augmented generation", result)

    def test_expand_search_query_strips_conversational_filler(self):
        query = "Give me the arXiv IDs of two papers describing two different types of retrieval-augmented generation"
        result = expand_search_query(query)
        self.assertIn('all:"RAG"', result)
        self.assertIn('all:"retrieval-augmented generation"', result)
        self.assertNotIn("Give me", result)
        self.assertNotIn("two papers", result)

    @patch("llm.call_ollama_mixtral", return_value='{"search_query": "all:\\\"data sourcing\\\"", "id_list": "optional, comma-separated ids", "start": 0, "max_results": 5}')
    def test_extract_arxiv_search_params_filters_placeholder_id_list(self, _mock_call):
        params = extract_arxiv_search_params("Tell me about data sourcing", max_results=5)
        self.assertEqual(params["search_query"], 'all:"data sourcing"')
        self.assertNotIn("id_list", params)


if __name__ == "__main__":
    unittest.main()
