import unittest

from orchestrator import build_llm_request
from llm import normalize_llm_response


class ResponseContractTests(unittest.TestCase):
    def test_build_llm_request_contains_expected_fields(self):
        request = build_llm_request(
            query="What are the main contributions?",
            intent="precise_qa",
            retrieved_chunks=[
                {
                    "chunk_id": 1,
                    "arxiv_id": "2401.12345",
                    "text": "This paper introduces a new retrieval method.",
                    "score": 0.95,
                }
            ],
        )

        self.assertEqual(request["query"], "What are the main contributions?")
        self.assertEqual(request["intent"], "precise_qa")
        self.assertEqual(request["retrieved_chunks"][0]["text"], "This paper introduces a new retrieval method.")
        self.assertEqual(request["options"]["max_messages"], 3)

    def test_normalize_llm_response_wraps_plain_text(self):
        response = normalize_llm_response("A short answer", query_text="What is this?", intent="default")

        self.assertIn("messages", response)
        self.assertEqual(response["messages"][0]["type"], "text")
        self.assertEqual(response["messages"][0]["content"], "A short answer")


if __name__ == "__main__":
    unittest.main()
