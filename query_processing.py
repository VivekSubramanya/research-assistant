import os
import sqlite3
import numpy as np
import faiss

from data_ingestion import init_db as init_storage_db

SentenceTransformer = None

try:
    import torch
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    TORCH_CUDA_AVAILABLE = False

DB_PATH = "research.db"
EMBED_MODEL = None
FAISS_PATH = "faiss_index.bin"
faiss_index = None


def load_faiss_index(path=FAISS_PATH):
    if os.path.exists(path):
        try:
            index = faiss.read_index(path)
        except Exception as exc:
            print(f"FAISS index load failed ({exc}); starting empty index")
            index = faiss.IndexFlatL2(384)
    else:
        index = faiss.IndexFlatL2(384)
    print(f"FAISS index loaded from {path}")
    return index


def reload_faiss_index():
    global faiss_index
    faiss_index = load_faiss_index()
    return faiss_index


def ensure_storage_ready():
    init_storage_db()


def get_embed_model():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        global SentenceTransformer
        if SentenceTransformer is None:
            try:
                from sentence_transformers import SentenceTransformer as LoadedSentenceTransformer
            except Exception as exc:
                raise ImportError("sentence_transformers is required for embedding") from exc
            SentenceTransformer = LoadedSentenceTransformer
        # Force CPU execution to prevent CUDA conflicts with LLM inference
        device = "cpu"
        EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return EMBED_MODEL


def warm_start_embedder():
    try:
        _ = get_embed_model()
    except Exception:
        pass

# --- Step 1: Embed query ---
def embed_query(query_text):
    model = get_embed_model()
    try:
        emb = model.encode([query_text], batch_size=1, convert_to_numpy=True, show_progress_bar=False)
    except TypeError:
        emb = model.encode([query_text])
    return np.asarray(emb[0], dtype="float32")

# --- Step 2: Adaptive top_k based on confidence spread ---
def adaptive_top_k(distances):
    spread = max(distances[0]) - min(distances[0])

    if spread < 0.1:       # tight cluster -> high confidence
        return 5
    elif spread < 0.3:     # moderate spread
        return 10
    else:                  # wide spread -> low confidence
        return 20

# --- Step 3: Run FAISS search with adaptive top_k ---
def run_faiss_search(query_embedding, top_k=10, distance_threshold=1.25):
    ensure_storage_ready()
    reload_faiss_index()
    query_vec = np.array([query_embedding]).astype("float32")
    search_top_k = max(1, min(int(top_k), 50))
    
    if hasattr(faiss_index, "nprobe") and hasattr(faiss_index, "nlist"):
        try:
            faiss_index.nprobe = min(12, faiss_index.nlist)
        except Exception:
            pass
            
    distances, indices = faiss_index.search(query_vec, search_top_k)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    results = []
    
    for i, pos in enumerate(indices[0]):
        dist = distances[0][i]
        
        # If the chunk is mathematically too distant, it's irrelevant. 
        # Skip it so we can trigger the arXiv API fallback.
        if pos == -1 or dist > distance_threshold:
            continue

        cursor.execute("SELECT chunk_id FROM faiss_map WHERE faiss_pos=?", (int(pos),))
        row = cursor.fetchone()
        if row:
            chunk_id = row[0]
            cursor.execute("SELECT chunk_text, arxiv_id FROM chunks WHERE chunk_id=?", (chunk_id,))
            chunk_row = cursor.fetchone()
            if chunk_row:
                results.append({
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_row[0],
                    "arxiv_id": chunk_row[1],
                    "faiss_pos": int(pos),
                    "score": float(dist)
                })
                
    conn.close()
    return results

def get_chunks_for_paper(arxiv_id, top_k=None):
    """Return chunks for a single paper without running FAISS search."""
    ensure_storage_ready()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if top_k is None:
        cursor.execute("SELECT chunk_id, chunk_text FROM chunks WHERE arxiv_id=? ORDER BY created_at DESC", (arxiv_id,))
    else:
        cursor.execute(
            "SELECT chunk_id, chunk_text FROM chunks WHERE arxiv_id=? ORDER BY created_at DESC LIMIT ?",
            (arxiv_id, int(top_k)),
        )
    rows = cursor.fetchall()
    conn.close()
    results = []
    for row in rows:
        results.append({"chunk_id": row[0], "chunk_text": row[1], "arxiv_id": arxiv_id, "faiss_pos": None})
    return results

# --- Orchestration ---
def process_query(query_text, top_k=10):
    ensure_storage_ready()
    query_embedding = embed_query(query_text)
    results = run_faiss_search(query_embedding, top_k=top_k)
    return results