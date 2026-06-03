# AI Governance Copilot

A fully local AI assistant for AI governance work. Point it at real regulatory documents and it helps you classify AI system risk, identify compliance gaps, generate model cards, and draft formal assessment reports — all running on your own machine, no internet connection required after setup.

**Built with:** Python · LangChain · Ollama · ChromaDB · Streamlit

**Frameworks covered:** EU AI Act · NIST AI RMF · ISO/IEC 42001

---

## What it does

| Module | What it does |
|---|---|
| **Risk Classifier** | Describe an AI system → get its EU AI Act risk tier (Unacceptable / High / Limited / Minimal) with citations |
| **Gap Analyzer** | Map your system against EU AI Act, NIST AI RMF, and ISO 42001 → see exactly what's missing |
| **Model Card Generator** | Fill in system metadata → get a structured model card in the HuggingFace standard format |
| **Report Writer** | Combine all of the above → draft a complete, formal AI risk assessment report |

---

## How it works (plain English)

This tool is built around a technique called **RAG (Retrieval-Augmented Generation)**. Here's what that means without the jargon:

1. **You supply the source documents.** You download the actual EU AI Act PDF, the NIST AI RMF PDF, etc., and put them in a folder. The tool reads every page.

2. **The documents get indexed.** A process called *ingestion* reads all the PDFs, breaks them into small chunks, and stores them in a local database (ChromaDB) in a way that makes them searchable by *meaning* — not just keywords. This happens once and takes a few minutes.

3. **You describe your AI system.** Through the web interface, you type a plain-English description of what your AI system does, who uses it, and what data it handles.

4. **The AI looks it up.** When you ask a question (e.g. "what risk tier is this?"), the tool finds the most relevant paragraphs from the regulatory documents and hands them to the AI model along with your question.

5. **The AI answers from the documents.** The AI (running locally via Ollama) reads those paragraphs and generates a structured answer — a risk tier, a list of gaps, a model card section — citing the exact articles and clauses it found. It is instructed not to make things up.

**Why fully local?** Governance work often involves sensitive system descriptions. Nothing you type ever leaves your machine.

---

## What you need before starting

You need four things installed on your computer. This guide walks through each one.

| Requirement | What it is | Why you need it |
|---|---|---|
| **Python 3.10+** | The programming language this project is written in | To run the code at all |
| **uv** | A Python package manager | Installs all the Python libraries this project depends on |
| **Ollama** | A program that runs AI models locally | The "brain" — generates answers from the documents |
| **Two AI models** (via Ollama) | `llama3.2` (language) + `nomic-embed-text` (search) | One answers questions; the other makes documents searchable |

---

## Step-by-step setup

> **Note on the terminal:** All commands below are run in a terminal (also called the command line, shell, or command prompt). On Mac, open **Terminal** from Applications → Utilities. On Windows, open **PowerShell**. You type a command and press Enter to run it.

---

### Step 1 — Check Python

Open your terminal and run:

```bash
python3 --version
```

You should see something like `Python 3.10.x` or higher. If you get `command not found`, download Python from [python.org](https://www.python.org/downloads/) and install it, then come back here.

---

### Step 2 — Install uv

`uv` is a fast Python package manager. Think of it like an App Store for Python libraries — it downloads and installs everything this project needs in one command, and keeps it all neatly isolated so it doesn't interfere with other projects on your machine.

**Mac / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.sh | iex"
```

After installation, close and reopen your terminal, then verify it worked:
```bash
uv --version
```

You should see a version number like `uv 0.5.x`.

---

### Step 3 — Install Ollama

Ollama is a program that runs AI language models locally on your computer. Instead of your questions going to OpenAI's servers, everything stays on your machine.

1. Go to [ollama.com](https://ollama.com) and click **Download**
2. Install it like any normal application (double-click the installer on Mac/Windows)
3. Once installed, verify it's running:

```bash
ollama --version
```

---

### Step 4 — Download the AI models

This project uses two AI models, both free:

- **`llama3.2`** — the language model that reads documents and writes answers (about 2 GB)
- **`nomic-embed-text`** — the embedding model that makes the documents searchable by meaning (about 270 MB)

Run these two commands (they may take a few minutes depending on your internet speed):

```bash
ollama pull llama3.2
```
```bash
ollama pull nomic-embed-text
```

You'll see a download progress bar. Wait for both to complete before continuing.

> **What's an embedding model?** When you search for something, you usually search by keyword. An embedding model converts text into numbers that capture *meaning*, so a search for "what are the rules for employment AI?" can find a paragraph that talks about "automated hiring decisions" even if it uses different words.

---

### Step 5 — Get the project files

If you have git installed:
```bash
git clone https://github.com/YOUR_USERNAME/ai-governance-copilot.git
cd ai-governance-copilot
```

Or download the ZIP from GitHub (green "Code" button → "Download ZIP"), unzip it, and navigate to the folder in your terminal:
```bash
cd path/to/ai-governance-copilot
```

---

### Step 6 — Install project dependencies

This installs all the Python libraries the project needs (LangChain, ChromaDB, Streamlit, etc.). Run this inside the project folder:

```bash
uv sync
```

`uv` will create a self-contained environment inside the project folder and install everything into it. It will not touch anything else on your computer. This takes about a minute on first run.

---

### Step 7 — Create your configuration file

The project uses a `.env` file to store settings (model names, file paths). A template is included. Copy it:

**Mac / Linux:**
```bash
cp .env.example .env
```

**Windows:**
```powershell
copy .env.example .env
```

You do not need to edit this file unless you want to change the default AI model or folder locations. The defaults work out of the box.

---

### Step 8 — Download the regulatory documents

The tool needs the actual regulatory PDFs to work. These are free and publicly available. Download them and place them in the `docs/` folder inside the project.

| Document | Download link | Filename to use |
|---|---|---|
| EU AI Act | [EUR-Lex (PDF)](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689) | `eu_ai_act.pdf` |
| NIST AI RMF 1.0 | [NIST (PDF)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) | `nist_ai_rmf.pdf` |

> **ISO/IEC 42001** — The full standard is paywalled. The tool works well with just the EU AI Act and NIST RMF. If you have access to the ISO document, name it `iso_42001.pdf` and place it in `docs/` too.

Your `docs/` folder should look like:
```
docs/
├── eu_ai_act.pdf
└── nist_ai_rmf.pdf
```

---

### Step 9 — Ingest the documents

This is the one-time setup step that reads your PDFs and builds the searchable database. Run:

```bash
uv run python -m src.ingestion.embedder
```

What's happening under the hood:
1. Every page of every PDF is read
2. Each page is split into small overlapping chunks (about 1,000 characters each)
3. Each chunk is converted to a vector (numbers representing its meaning) using `nomic-embed-text`
4. Everything is stored in a local database in `data/vectorstore/`

This takes **3–10 minutes** depending on your computer speed. You'll see progress messages. You only need to do this once (or again if you add new documents).

Expected output when it completes:
```
✓ eu_ai_act.pdf — 144 pages
✓ nist_ai_rmf.pdf — 26 pages
Split 170 pages → 1,842 chunks
✓ Done! 1842 chunks stored in data/vectorstore
```

---

### Step 10 — Launch the app

```bash
uv run streamlit run src/ui/app.py
```

Streamlit will start a local web server and automatically open the app in your browser at `http://localhost:8501`. You should see the AI Governance Copilot interface.

To stop the app, go back to your terminal and press `Ctrl + C`.

---

## Using the app

Once the app is open in your browser:

1. **Risk Classifier** — Type a description of your AI system (what it does, who uses it, what data it processes). Click **Classify** to get its EU AI Act risk tier with an explanation.

2. **Gap Analyzer** — Use the same description to map your system against EU AI Act, NIST AI RMF, and ISO 42001. The tool returns a table of requirements: what's met, what's missing.

3. **Model Card Generator** — Fill in a form with your system's metadata (name, type, training data, metrics, etc.). The tool generates a complete model card you can copy or download.

4. **Report Writer** — Combine all of the above into a single formal AI risk assessment report, ready to share with your team or regulators. You can provide a system description and let it run all four modules at once.

> **Tip:** The more detail you put in your system description, the better the outputs. Include: what problem it solves, what data it uses, who the users are, where it's deployed, and any known limitations.

---

## Troubleshooting

**"No PDFs found in docs/"**
→ Make sure your PDF files are inside the `docs/` folder in the project directory, and that the folder is named exactly `docs` (lowercase).

**"Connection refused" or Ollama errors**
→ Ollama must be running in the background. On Mac, look for the Ollama icon in your menu bar. On Windows, check the system tray. If it's not there, open the Ollama application again.

**Ingestion is very slow**
→ This is normal on first run. The embedding model has to process every chunk. Let it finish — subsequent runs will be instant (it detects the database already exists and skips re-ingestion).

**The app opens but answers seem wrong or generic**
→ Make sure ingestion completed successfully (check that `data/vectorstore/` exists and isn't empty). If in doubt, re-run the ingestion step with `--force` to rebuild from scratch:
```bash
uv run python -m src.ingestion.embedder --force
```

**"Module not found" errors**
→ Make sure you're running commands from inside the project folder (`cd ai-governance-copilot`) and using `uv run` before Python commands.

---

## Project structure (for the curious)

```
ai-governance-copilot/
│
├── docs/                   ← Put your PDFs here (not committed to git)
│
├── src/
│   ├── config.py           ← All settings (model names, paths) — loaded from .env
│   ├── ingestion/          ← Step 1 of the pipeline: PDF → chunks → database
│   │   ├── loader.py         Reads PDFs, attaches source/page metadata
│   │   ├── chunker.py        Splits pages into overlapping chunks
│   │   └── embedder.py       Converts chunks to vectors, stores in ChromaDB
│   ├── rag/                ← Step 2: search and answer
│   │   ├── retriever.py      Loads the database, searches by meaning
│   │   └── chain.py          Connects retriever → AI model → answer
│   └── modules/            ← The four tools
│       ├── risk_classifier.py
│       ├── gap_analyzer.py
│       ├── model_card_generator.py
│       └── report_writer.py
│
├── data/
│   └── vectorstore/        ← The searchable document database (generated, not in git)
│
├── tests/                  ← Automated tests
├── .env.example            ← Settings template — copy to .env
└── pyproject.toml          ← Project definition and dependencies
```

---

## Running tests

To verify everything is working correctly (no Ollama or PDFs needed for this):

```bash
uv run --no-active python -m pytest
```

You should see all tests passing. This is useful after making any changes to the code.

---

## Roadmap

- [x] Step 1 — Project setup & skeleton
- [x] Step 2 — Document ingestion pipeline
- [x] Step 3 — RAG core
- [x] Step 4 — Risk classifier
- [x] Step 5 — Gap analyzer
- [x] Step 6 — Model card generator
- [x] Step 7 — Report writer
- [x] Step 8 — Streamlit UI
