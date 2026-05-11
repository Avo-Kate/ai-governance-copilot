"""
chain.py — the RAG chain that answers questions using the governance docs.

How it works:
  1. User asks a question
  2. Retriever finds the top K relevant chunks from ChromaDB
  3. Those chunks are injected into a prompt as "context"
  4. Ollama reads the context + question and generates a grounded answer
  5. Source metadata is returned alongside the answer so we can cite it

The LLM is instructed to only answer from the provided context — this
prevents hallucination and keeps answers grounded in the actual documents.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List

from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from src.rag.retriever import get_retriever

# ── Prompt ─────────────────────────────────────────────────────────────────
# This is the most important part of a RAG system.
# "context" gets filled with retrieved chunks. "question" is the user query.
# We explicitly tell the model to stay within the documents and cite sources.

SYSTEM_PROMPT = """You are an expert AI governance analyst with deep knowledge \
of regulatory frameworks including the EU AI Act, NIST AI RMF, and ISO 42001.

Answer the question using ONLY the context provided below. If the answer is \
not in the context, say "I couldn't find this in the provided documents" — \
do not make things up.

When possible, mention which document and section your answer comes from.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


# ── Helpers ────────────────────────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    """
    Concatenate retrieved chunks into a single context string.
    Each chunk is prefixed with its source and page so the LLM can cite them.
    """
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Chain ──────────────────────────────────────────────────────────────────

def build_chain():
    """
    Build and return the RAG chain.

    The chain is a pipeline built with LangChain's | operator:
      retriever → format_docs → prompt → LLM → string output

    RunnablePassthrough keeps the original question flowing through
    while the retriever runs in parallel on the same input.
    """
    retriever = get_retriever()
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,  # 0 = deterministic, better for factual Q&A
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(question: str) -> dict:
    """
    Ask a question and get back an answer with sources.

    Returns:
        {
            "answer": str,
            "sources": [{"source": str, "page": int}, ...]
        }
    """
    chain, retriever = build_chain()

    # Get the answer
    answer = chain.invoke(question)

    # Get the source chunks separately so we can show citations
    source_docs = retriever.invoke(question)
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
        }
        for doc in source_docs
    ]

    # Deduplicate sources
    seen = set()
    unique_sources = []
    for s in sources:
        key = (s["source"], s["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {"answer": answer, "sources": unique_sources}