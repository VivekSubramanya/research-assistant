import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from llm import normalize_llm_response
from orchestrator import build_llm_request
import query_processing


class EdgeCaseTests(unittest.TestCase):
    def test_empty_retrieval_results_produce_fallback_response(self):
        response = normalize_llm_response(None, query_text="What now?", intent="default")
        self.assertEqual(response["messages"][0]["content"], "Unable to generate answer")

    def test_build_llm_request_handles_missing_chunk_fields(self):
        payload = build_llm_request("Query", "default", [{"chunk_id": 1}])
        self.assertEqual(payload["retrieved_chunks"][0]["text"], None)

    def test_process_query_initializes_storage_before_search(self):
        order = []

        def fake_search(embedding, top_k=10):
            order.append("search")
            return []

        with patch.object(query_processing, "run_faiss_search", side_effect=fake_search) as search_mock, \
             patch.object(query_processing, "get_embed_model", return_value=type("Encoder", (), {"encode": lambda self, texts: [[0.0, 0.0, 0.0]]})()), \
             patch.object(query_processing, "init_storage_db", side_effect=lambda: order.append("init")) as init_mock:
            query_processing.process_query("hello")
            self.assertEqual(order, ["init", "search"])
            init_mock.assert_called_once()
            search_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
