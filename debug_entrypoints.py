import os
import time
import json
import logging
import traceback

LOG_FILE = "debug_entrypoints.log"
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

from llm import expand_query_for_embedding, extract_arxiv_search_params, call_ollama_mixtral, generate_answer
from query_processing import embed_query, process_query, run_faiss_search, get_chunks_for_paper
from orchestrator import Orchestrator, HybridReasoningModel
from APIs import arxiv_api

QUERY = "Tell me about data sourcing"

def safe_dump(obj):
    try:
        return json.dumps(obj, default=str, indent=2)
    except Exception:
        try:
            return repr(obj)
        except Exception:
            return '<unserializable>'


def trace():
    logging.info('=== debug run start')
    try:
        t0 = time.time()
        logging.info('Query: %s', QUERY)

        logging.info('Stage: expand_query_for_embedding')
        searchable = expand_query_for_embedding(QUERY)
        logging.debug('expand_query_for_embedding -> %s', searchable)

        logging.info('Stage: extract_arxiv_search_params')
        params = extract_arxiv_search_params(QUERY, max_results=5)
        logging.debug('extract_arxiv_search_params -> %s', safe_dump(params))

        logging.info('Stage: arXiv API hit (preview)')
        try:
            entries = arxiv_api.query_arxiv(params)
            logging.info('arXiv entries returned: %d', len(entries))
            for i, e in enumerate(entries[:5]):
                parsed = arxiv_api.parse_entry(e)
                logging.debug('entry %d: %s', i+1, parsed.get('title'))
        except Exception as e:
            logging.exception('arXiv API call failed')

        logging.info('Stage: embed_query')
        try:
            emb = embed_query(QUERY)
            logging.info('embed_query -> length %d, norm %.4f', len(emb), float((emb**2).sum()**0.5))
        except Exception:
            logging.exception('embed_query failed')
            emb = None

        logging.info('Stage: process_query (semantic search)')
        try:
            results = process_query(QUERY, top_k=10)
            logging.info('process_query -> %d results', len(results))
            if results:
                logging.debug('first result sample: %s', results[0].get('chunk_text')[:200])
        except Exception:
            logging.exception('process_query failed')
            results = []

        logging.info('Stage: orchestrator.handle_query')
        try:
            resp = None
            # Use orchestrator entrypoint
            from orchestrator import handle_query
            resp = handle_query(QUERY)
            logging.info('handle_query returned type %s', type(resp))
            logging.debug('handle_query response: %s', safe_dump(resp))
        except Exception:
            logging.exception('handle_query failed')

        logging.info('Stage: generate_answer dry-run (if LLM call available)')
        try:
            # build a minimal llm_request to call generate_answer
            llm_request = {
                'query': QUERY,
                'intent': 'default',
                'retrieved_chunks': results,
                'options': {'max_messages': 3}
            }
            llm_resp = generate_answer(llm_request)
            logging.info('generate_answer -> messages: %s', [m.get('content')[:200] for m in llm_resp.get('messages', [])])
            logging.debug('generate_answer raw: %s', safe_dump(llm_resp))
        except Exception:
            logging.exception('generate_answer failed')

        dt = time.time() - t0
        logging.info('=== debug run complete in %.3fs', dt)
    except Exception:
        logging.exception('Unexpected error in trace')


if __name__ == '__main__':
    trace()
    print('Wrote', LOG_FILE)
