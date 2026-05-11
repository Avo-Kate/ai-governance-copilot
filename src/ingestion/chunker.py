from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

console = Console()


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,         # max characters per chunk (1000 in config)
        chunk_overlap=CHUNK_OVERLAP,   # characters shared between adjacent chunks (150)
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    console.print(f"[bold]Split {len(documents)} pages → {len(chunks)} chunks[/bold]")
    return chunks


def preview_chunks(chunks: List[Document], n: int = 3) -> None:
    console.print(f"\n[bold underline]Preview (first {n} chunks):[/bold underline]")
    for chunk in chunks[:n]:
        console.print(
            f"\n[dim]── {chunk.metadata.get('source')} "
            f"p.{chunk.metadata.get('page')} ──[/dim]"
        )
        console.print(chunk.page_content[:300] + "...")