"""
retriever.py — loads the ChromaDB vectorstore and returns a retriever.

The retriever is the "search" part of RAG. Given a query string, it
converts it to a vector using the same embedding model we used during
ingestion, then finds the K most similar chunks in the store.
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import VECTORSTORE_PATH, EMBEDDING_MODEL, OLLAMA_BASE_URL, RETRIEVER_K

COLLECTION_NAME = "governance_docs"


def get_retriever() -> VectorStoreRetriever:
    """
    Load the persisted vectorstore and return a retriever.

    The retriever will return RETRIEVER_K chunks per query (default 6).
    We use the same embedding model as ingestion — if they don't match,
    the similarity search produces garbage.
    """
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_PATH),
    )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )