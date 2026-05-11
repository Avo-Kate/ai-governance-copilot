# AI Governance Copilot

A fully local AI assistant for AI governance work. Point it at real regulatory documents and it helps you classify system risk, identify compliance gaps, generate model cards, and draft assessment reports — all running on your machine, no cloud required.

**Built with:** Python · LangChain · Ollama · ChromaDB · Streamlit

**Frameworks covered:** EU AI Act · NIST AI RMF · ISO/IEC 42001

---

## What it does

| Module | What it does |
|---|---|
| **Risk Classifier** | Describes an AI system → outputs risk tier with framework citations |
| **Gap Analyzer** | Maps your system against multiple frameworks → flags what's missing |
| **Model Card Generator** | Structured inputs → formatted model card (HuggingFace standard) |
| **Report Writer** | Combines everything → drafts a formal risk assessment report |

---

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Ollama](https://ollama.com) — install, then `ollama pull llama3.2`

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-governance-copilot.git
cd ai-governance-copilot

# 2. Install dependencies
uv sync

# 3. Copy and configure environment
cp .env.example .env
# Edit .env if you want to change the model or paths

# 4. Add your documents
# Put EU AI Act, NIST RMF, ISO 42001 PDFs in the docs/ folder
# (see docs/README.md for download links)

# 5. Ingest documents into the vector store
uv run python -m src.ingestion.embedder

# 6. Launch the app
uv run streamlit run src/ui/app.py
```

---

## Project structure

```
ai-governance-copilot/
├── docs/               # Put your PDFs here (gitignored)
├── src/
│   ├── config.py       # Central config loaded from .env
│   ├── ingestion/      # PDF loading, chunking, embedding
│   ├── rag/            # Retrieval chain and Ollama integration
│   ├── modules/        # Risk classifier, gap analyzer, model card, report writer
│   └── ui/             # Streamlit app and components
├── data/
│   └── vectorstore/    # ChromaDB persisted store (gitignored)
├── tests/
├── .env.example
└── pyproject.toml
```

---

## Document sources

Free, legal downloads:

- **EU AI Act** — [EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689)
- **NIST AI RMF 1.0** — [NIST](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf)
- **ISO/IEC 42001** — Summary only (full standard is paywalled); NIST and EU AI Act alone give excellent coverage

---

## Running tests

```bash
uv run pytest
```

---

## Roadmap

- [x] Step 1 — Project setup & skeleton
- [ ] Step 2 — Document ingestion pipeline
- [ ] Step 3 — RAG core
- [ ] Step 4 — Risk classifier
- [ ] Step 5 — Gap analyzer
- [ ] Step 6 — Model card generator
- [ ] Step 7 — Report writer
- [ ] Step 8 — Streamlit UI
