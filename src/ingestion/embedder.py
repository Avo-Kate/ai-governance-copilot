import sys
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rich.console import Console

from src.config import VECTORSTORE_PATH, EMBEDDING_MODEL, OLLAMA_BASE_URL
from src.ingestion.loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents, preview_chunks

console = Console()
COLLECTION_NAME = "governance_docs"


def get_embeddings() -> OllamaEmbeddings:
    """
    Use Ollama for embeddings — same process you already have running,
    no PyTorch required in our code at all.
    """
    console.print(f"[dim]Using Ollama embedding model: {EMBEDDING_MODEL}[/dim]")
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_vectorstore(embeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_PATH),
    )


def ingest(force: bool = False) -> Chroma:
    console.rule("[bold]AI Governance Copilot — Document Ingestion[/bold]")

    if VECTORSTORE_PATH.exists() and any(VECTORSTORE_PATH.iterdir()) and not force:
        console.print("[yellow]Vectorstore already exists.[/yellow] Use --force to re-ingest.")
        return get_vectorstore(get_embeddings())

    if force and VECTORSTORE_PATH.exists():
        import shutil
        shutil.rmtree(VECTORSTORE_PATH)
        console.print("[yellow]Cleared existing vectorstore.[/yellow]")

    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold]Step 1/3 — Loading PDFs[/bold]")
    documents = load_all_pdfs()
    if not documents:
        console.print("[red]No documents found. Add PDFs to docs/ first.[/red]")
        sys.exit(1)

    console.print("\n[bold]Step 2/3 — Chunking[/bold]")
    chunks = chunk_documents(documents)
    preview_chunks(chunks, n=2)

    console.print("\n[bold]Step 3/3 — Embedding & storing[/bold]")
    console.print("[dim]First run downloads ~90MB model, then it's cached.[/dim]")

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_PATH),
    )

    count = vectorstore._collection.count()
    console.print(f"\n[bold green]✓ Done! {count} chunks stored in {VECTORSTORE_PATH}[/bold green]")
    return vectorstore


if __name__ == "__main__":
    force = "--force" in sys.argv
    ingest(force=force)