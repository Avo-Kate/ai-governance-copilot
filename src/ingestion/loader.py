from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from rich.console import Console
from rich.progress import track

from src.config import DOCS_PATH

console = Console()


def find_pdfs(docs_path: Path = DOCS_PATH) -> List[Path]:
    pdfs = sorted(docs_path.glob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]⚠ No PDFs found in {docs_path}. See docs/README.md[/yellow]")
    return pdfs


def load_pdf(pdf_path: Path) -> List[Document]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    for i, page in enumerate(pages):
        page.metadata["source"] = pdf_path.name
        page.metadata["page"] = i + 1
        page.metadata["total_pages"] = len(pages)

    return pages


def load_all_pdfs(docs_path: Path = DOCS_PATH) -> List[Document]:
    pdfs = find_pdfs(docs_path)
    if not pdfs:
        return []

    all_documents: List[Document] = []
    for pdf_path in track(pdfs, description="Loading PDFs..."):
        docs = load_pdf(pdf_path)
        all_documents.extend(docs)
        console.print(f"  [green]✓[/green] {pdf_path.name} — {len(docs)} pages")

    console.print(f"\n[bold]Loaded {len(all_documents)} pages from {len(pdfs)} file(s)[/bold]")
    return all_documents