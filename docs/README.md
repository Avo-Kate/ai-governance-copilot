# Governance Documents

Put your PDF files in this folder. They are gitignored (too large for version control).

## Download links

| Document | URL | Filename to use |
|---|---|---|
| EU AI Act (2024) | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689 | `eu_ai_act.pdf` |
| NIST AI RMF 1.0 | https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf | `nist_ai_rmf.pdf` |
| NIST AI RMF Playbook | https://airc.nist.gov/Docs/2 | `nist_ai_rmf_playbook.pdf` |

## After downloading

Run the ingestion pipeline to embed them into the vector store:

```bash
uv run python -m src.ingestion.embedder
```

This only needs to be run once, or again when you add new documents.
