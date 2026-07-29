import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import faiss
import numpy as np

import chunk_and_embed
import data_ingestion
import query_processing
from llm import normalize_llm_response
from orchestrator import build_llm_request


class ResearchAssistantIntegrationTests(unittest.TestCase):
    def test_happy_path_ingestion_to_answer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "research.db")
            pdf_dir = os.path.join(tmpdir, "pdfs")
            os.makedirs(pdf_dir, exist_ok=True)

            fake_paper = SimpleNamespace(
                title="Test Paper",
                authors=[SimpleNamespace(name="Ada")],
                summary="This paper introduces a new method.",
                categories=["cs.AI"],
                published=SimpleNamespace(date=lambda: SimpleNamespace(isoformat=lambda: "2024-01-01")),
                pdf_url="https://example.com/paper.pdf",
            )

            class FakeSearch:
                def __init__(self, id_list=None, query=None, max_results=10):
                    self.id_list = id_list or []

                def results(self):
                    yield fake_paper

            class FakeResponse:
                def __init__(self, payload):
                    self.content = payload
                    self.text = payload
                def raise_for_status(self):
                    return None

            class FakeEncoder:
                def encode(self, chunks):
                    embeddings = []
                    for chunk in chunks:
                        if "method" in chunk.lower():
                            embeddings.append([0.1, 0.2, 0.3])
                        else:
                            embeddings.append([0.4, 0.5, 0.6])
                    return embeddings

            def fake_search(query_vec, k):
                index = faiss.IndexFlatL2(3)
                index.add(np.array([[0.1, 0.2, 0.3]], dtype="float32"))
                return index.search(query_vec, k)

            class FakeLLMResponse:
                def __init__(self):
                    self.stdout = b'{"messages": [{"type": "text", "content": "The paper introduces a new method."}]}'

            # Patch module-level paths and external dependencies
            with patch.object(data_ingestion, "DB_PATH", db_path), \
                 patch.object(chunk_and_embed, "DB_PATH", db_path), \
                 patch.object(query_processing, "DB_PATH", db_path), \
                 patch.object(data_ingestion, "PDF_DIR", pdf_dir), \
                 patch.object(data_ingestion.arxiv, "Search", FakeSearch), \
                 patch.object(data_ingestion.requests, "get", return_value=FakeResponse(b"pdfbytes")), \
                 patch.object(chunk_and_embed.requests, "post", return_value=FakeResponse("<TEI><text><p>This paper introduces a new method.</p></text></TEI>")), \
                 patch.object(chunk_and_embed, "get_embed_model", return_value=FakeEncoder()), \
                 patch.object(query_processing, "get_embed_model", return_value=FakeEncoder()), \
                 patch.object(chunk_and_embed, "faiss_index", faiss.IndexFlatL2(3)), \
                 patch.object(query_processing, "faiss_index", faiss.IndexFlatL2(3)), \
                 patch.object(query_processing, "run_faiss_search", side_effect=lambda embedding, top_k=10: [{"chunk_id": 1, "chunk_text": "This paper introduces a new method.", "arxiv_id": "1234", "faiss_pos": 0}]), \
                 patch.object(chunk_and_embed, "save_faiss_index", return_value=None), \
                 patch("llm.subprocess.run", return_value=FakeLLMResponse()):
                data_ingestion.init_db()
                data_ingestion.ingest_paper("1234")
                chunk_and_embed.init_db = data_ingestion.init_db
                chunk_and_embed.process_document("1234", os.path.join(pdf_dir, "1234.pdf"))

                results = query_processing.process_query("What is the main idea?", top_k=5)
                self.assertTrue(results)

                request = build_llm_request(
                    query="What is the main idea?",
                    intent="precise_qa",
                    retrieved_chunks=results,
                )
                response = normalize_llm_response(
                    '{"messages": [{"type": "text", "content": "The paper introduces a new method."}]}',
                    query_text=request["query"],
                    intent=request["intent"],
                )
                self.assertEqual(response["messages"][0]["content"], "The paper introduces a new method.")


if __name__ == "__main__":
    unittest.main()
