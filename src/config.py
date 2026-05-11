"""
Central configuration — all settings loaded from .env here,
imported everywhere else. Never hardcode paths or model names.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DOCS_PATH = ROOT_DIR / os.getenv("DOCS_PATH", "docs/")
VECTORSTORE_PATH = ROOT_DIR / os.getenv("VECTORSTORE_PATH", "data/vectorstore")

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ── Retrieval ──────────────────────────────────────────────────────────────
RETRIEVER_K = 6  # number of chunks to retrieve per query
