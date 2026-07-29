import sqlite3
import requests
import time
import os
from APIs import arxiv_api as arxiv

DB_PATH = "research.db"
PDF_DIR = "pdfs/"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Queries table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        query_id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_text TEXT NOT NULL,
        query_type TEXT,
        query_embedding BLOB,
        response_text TEXT,
        document_ids TEXT,
        chunk_ids TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        arxiv_id TEXT PRIMARY KEY,
        title TEXT,
        authors TEXT,
        abstract TEXT,
        categories TEXT,
        published_date TEXT,
        file_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_accessed TIMESTAMP
    )
    """)

    # Chunks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        arxiv_id TEXT NOT NULL,
        chunk_text TEXT,
        chunk_embedding BLOB,
        section TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (arxiv_id) REFERENCES documents(arxiv_id)
    )
    """)

    # FAISS mapping table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS faiss_map (
        faiss_pos INTEGER PRIMARY KEY,   -- FAISS vector position
        chunk_id INTEGER NOT NULL,       -- maps to chunks table
        FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
    )
    """)


    conn.commit()
    conn.close()

def download_pdf(pdf_url, arxiv_id):
    os.makedirs(PDF_DIR, exist_ok=True)
    file_path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)
    return file_path

def ingest_paper(arxiv_id):
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(search.results(), None)
    if paper is None:
        raise ValueError(f"No paper found for {arxiv_id}")

    title = paper.title
    if isinstance(paper.authors, list):
        author_names = [getattr(author, "name", str(author)) for author in paper.authors]
        authors = ", ".join(author_names)
    else:
        authors = str(paper.authors)
    abstract = paper.summary
    categories = ", ".join(paper.categories) if isinstance(paper.categories, list) else str(paper.categories)
    published_date = paper.published.date().isoformat() if hasattr(paper.published, "date") else str(paper.published)
    pdf_url = paper.pdf_url

    file_path = download_pdf(pdf_url, arxiv_id)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO documents (arxiv_id, title, authors, abstract, categories, published_date, file_path, last_accessed)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (arxiv_id, title, authors, abstract, categories, published_date, file_path, time.time()))
    conn.commit()
    conn.close()

    print(f"Ingested {arxiv_id}: {title}")
