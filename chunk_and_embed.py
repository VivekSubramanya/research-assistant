import os
import re
import sqlite3
import time
import numpy as np
from data_ingestion import init_db as init_storage_db
import faiss

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

SentenceTransformer = None

try:
    import torch
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    TORCH_CUDA_AVAILABLE = False

DB_PATH = "research.db"
EMBED_MODEL = None
dimension = 384
FAISS_PATH = "faiss_index.bin"
faiss_index = None

# --- FAISS Persistence ---

def save_faiss_index(faiss_index_obj, path=FAISS_PATH):
    faiss.write_index(faiss_index_obj, path)
    print(f"FAISS index saved to {path}")


def load_faiss_index(path=FAISS_PATH):
    if os.path.exists(path):
        try:
            index = faiss.read_index(path)
        except Exception as exc:
            print(f"FAISS index load failed ({exc}); starting empty index")
            index = faiss.IndexFlatL2(dimension)
    else:
        index = faiss.IndexFlatL2(dimension)
    print(f"FAISS index loaded from {path}")
    return index


def get_faiss_index(force_reload=False):
    global faiss_index
    if force_reload or faiss_index is None:
        faiss_index = load_faiss_index()
    return faiss_index

# --- Step 1: Extract text ---
def extract_text_from_pdf(file_path):
    """Extract PDF paragraphs with PyMuPDF, then fall back to pypdf."""
    try:
        if fitz is None:
            raise ImportError("PyMuPDF (pymupdf) is required for PDF extraction")

        document = fitz.open(file_path)
        try:
            paragraphs = []
            for page in document:
                for block in page.get_text("blocks"):
                    text = block[4].strip()
                    if text:
                        paragraphs.append(text)
        finally:
            document.close()

        if paragraphs:
            return paragraphs
    except Exception as exc:
        print(f"PyMuPDF extraction failed ({exc}); falling back to pypdf extraction.")

    # pypdf is intentionally kept in this method because it is the continuation
    # of the same extraction operation, not a separately callable strategy.
    if PdfReader is None:
        raise ImportError("pypdf is required for fallback PDF extraction")

    paragraphs = []
    for page in PdfReader(file_path).pages:
        text = page.extract_text()
        if text:
            paragraphs.extend(text.splitlines())
    return paragraphs

# --- Step 2: Semantic chunking ---
def chunk_paragraphs(paragraphs, max_tokens=1000, min_tokens=100, overlap_sentences=2):
    chunks = []
    for para in paragraphs:
        tokens = para.split()
        if len(tokens) > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para.strip())
            current = []
            for i, sent in enumerate(sentences):
                current.append(sent)
                if len(" ".join(current).split()) > max_tokens:
                    chunks.append(" ".join(current))
                    current = sentences[max(0, i-overlap_sentences):i+1]
            if current:
                chunks.append(" ".join(current))
        elif len(tokens) < min_tokens:
            if chunks:
                chunks[-1] += " " + para
            else:
                chunks.append(para)
        else:
            chunks.append(para)
    return chunks

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
    """Force model to load into memory (use at app startup)."""
    try:
        _ = get_embed_model()
    except Exception:
        pass

# --- Step 3: Embeddings ---
def embed_chunks(chunks, batch_size=64):
    model = get_embed_model()
    try:
        embeddings = model.encode(chunks, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    except TypeError:
        embeddings = model.encode(chunks)
    return np.asarray(embeddings, dtype="float32")

def store_chunks(arxiv_id, chunks, embeddings, faiss_index_obj=None, db_path=None):
    global faiss_index
    db_path = DB_PATH if db_path is None else db_path
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    index = faiss_index_obj if faiss_index_obj is not None else get_faiss_index()

    for chunk_text, embedding in zip(chunks, embeddings):
        cursor.execute("""
        INSERT INTO chunks (arxiv_id, chunk_text, chunk_embedding, section, created_at)
        VALUES (?, ?, ?, ?, ?)
        """, (arxiv_id, chunk_text, embedding.tobytes(), None, time.time()))
        chunk_id = cursor.lastrowid

        vec = np.asarray(embedding, dtype="float32")
        vec = np.ascontiguousarray(vec.reshape(1, -1))
        index.add(vec)
        cursor.execute("INSERT INTO faiss_map (faiss_pos, chunk_id) VALUES (?, ?)", (index.ntotal - 1, chunk_id))

    if faiss_index_obj is None:
        faiss_index = index

    conn.commit()
    conn.close()
    save_faiss_index(index)

# --- Orchestration ---
def _ensure_document_row(arxiv_id, file_path, db_path=None):
    db_path = DB_PATH if db_path is None else db_path
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM documents WHERE arxiv_id=?", (arxiv_id,))
    if cursor.fetchone() is None:
        cursor.execute("""
        INSERT OR REPLACE INTO documents (arxiv_id, title, authors, abstract, categories, published_date, file_path, last_accessed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (arxiv_id, None, None, None, None, None, file_path, time.time()))
        conn.commit()
    conn.close()


def process_document(arxiv_id, file_path, faiss_index_obj=None, db_path=None):
    db_path = DB_PATH if db_path is None else db_path
    init_storage_db()
    _ensure_document_row(arxiv_id, file_path, db_path=db_path)
    paragraphs = extract_text_from_pdf(file_path)
    chunks = chunk_paragraphs(paragraphs)
    embeddings = embed_chunks(chunks)
    store_chunks(arxiv_id, chunks, embeddings, faiss_index_obj=faiss_index_obj, db_path=db_path)
    print(f"Processed {arxiv_id}: {len(chunks)} chunks embedded.")