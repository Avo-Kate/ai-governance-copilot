"""
risk_classifier.py — classifies an AI system's risk tier per the EU AI Act.

How it works:
  1. Build a retrieval query from the system description + risk-tier keywords
  2. Retrieve the top-K regulatory chunks most relevant to risk classification
  3. Feed the description + context into a structured classification prompt
  4. Parse the LLM's response into a RiskClassification object

The four EU AI Act tiers:
  Unacceptable — prohibited outright (Article 5)
  High         — strict obligations apply (Annex III)
  Limited      — transparency obligations only
  Minimal      — no specific requirements
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, field_validator

from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from src.rag.chain import format_docs
from src.rag.retriever import get_retriever


# ── Output schema ──────────────────────────────────────────────────────────────

VALID_TIERS = {"Unacceptable", "High", "Limited", "Minimal"}


class RiskClassification(BaseModel):
    tier: str
    reasoning: str
    key_factors: list[str]
    sources: list[dict[str, Any]]

    @field_validator("tier")
    @classmethod
    def tier_must_be_valid(cls, v: str) -> str:
        if v not in VALID_TIERS:
            # Fuzzy-match common LLM variations before giving up
            normalised = v.strip().capitalize()
            if normalised in VALID_TIERS:
                return normalised
            raise ValueError(f"Unknown tier '{v}'. Must be one of {VALID_TIERS}.")
        return v


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a regulatory expert specialising in the EU AI Act.
Classify the AI system described below into exactly one of these four risk tiers:

  • Unacceptable — prohibited by Article 5 (e.g. social scoring, real-time \
remote biometric ID in public spaces, subliminal manipulation, exploitation \
of vulnerabilities of specific groups).
  • High — listed in Annex III (e.g. critical infrastructure, education, \
employment, essential services, law enforcement, migration, justice, \
biometric categorisation that is not prohibited).
  • Limited — transparency obligations only (e.g. chatbots that must disclose \
they are AI, deepfake generators, emotion recognition disclosed to users).
  • Minimal — no specific obligations (e.g. spam filters, AI in video games, \
recommendation engines outside high-risk contexts).

Use ONLY the regulatory context below to ground your reasoning. \
If a relevant article or annex is cited in the context, mention it.

Regulatory context:
{context}

AI system description:
{system_description}

Reply with EXACTLY this format — no extra text before or after:
TIER: <Unacceptable | High | Limited | Minimal>
REASONING: <2-4 sentences citing specific articles or annexes>
KEY_FACTORS: <factor 1> | <factor 2> | <factor 3>\
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
])


# ── Helpers ────────────────────────────────────────────────────────────────────

# Extra keywords appended to the retrieval query so we pull risk-tier chunks
# even when the system description doesn't mention them explicitly.
_RETRIEVAL_SUFFIX = (
    " EU AI Act risk classification prohibited high-risk Annex III Article 5 "
    "limited minimal transparency obligations"
)


def _parse_response(text: str) -> tuple[str, str, list[str]]:
    """Extract tier, reasoning, and key_factors from the structured LLM output."""
    tier = reasoning = key_factors_raw = ""

    for line in text.splitlines():
        if line.startswith("TIER:"):
            tier = line.removeprefix("TIER:").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.removeprefix("REASONING:").strip()
        elif line.startswith("KEY_FACTORS:"):
            key_factors_raw = line.removeprefix("KEY_FACTORS:").strip()

    key_factors = [f.strip() for f in key_factors_raw.split("|") if f.strip()]
    return tier, reasoning, key_factors


# ── Public API ─────────────────────────────────────────────────────────────────

def classify(system_description: str) -> RiskClassification:
    """
    Classify an AI system description into an EU AI Act risk tier.

    Args:
        system_description: Plain-language description of the AI system,
            its purpose, deployment context, and data it processes.

    Returns:
        RiskClassification with tier, reasoning, key_factors, and sources.
    """
    retriever = get_retriever()
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    retrieval_query = system_description + _RETRIEVAL_SUFFIX
    docs = retriever.invoke(retrieval_query)
    context = format_docs(docs)

    chain = _prompt | llm | StrOutputParser()
    raw = chain.invoke({"context": context, "system_description": system_description})

    tier, reasoning, key_factors = _parse_response(raw)

    sources = [
        {"source": d.metadata.get("source", "unknown"), "page": d.metadata.get("page", "?")}
        for d in docs
    ]
    seen: set[tuple] = set()
    unique_sources = []
    for s in sources:
        key = (s["source"], s["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return RiskClassification(
        tier=tier,
        reasoning=reasoning,
        key_factors=key_factors,
        sources=unique_sources,
    )


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    examples = [
        (
            "Biometric surveillance",
            "A system deployed by a city government that uses live CCTV feeds "
            "to identify individuals in real time across public spaces for general "
            "law-enforcement purposes.",
        ),
        (
            "CV screening tool",
            "An automated resume screening tool used by HR departments to rank "
            "job applicants and decide who proceeds to the interview stage.",
        ),
        (
            "Customer chatbot",
            "A customer-service chatbot on an e-commerce website that answers "
            "questions about orders and returns. Users are informed they are "
            "talking to an AI.",
        ),
        (
            "Spam filter",
            "An email spam filter that classifies incoming messages as spam or "
            "not-spam using a trained machine-learning model.",
        ),
    ]

    for name, description in examples:
        console.rule(f"[bold]{name}[/bold]")
        result = classify(description)

        tier_colours = {
            "Unacceptable": "red",
            "High": "orange3",
            "Limited": "yellow",
            "Minimal": "green",
        }
        colour = tier_colours.get(result.tier, "white")

        console.print(Panel(
            f"[bold {colour}]{result.tier} Risk[/bold {colour}]\n\n"
            f"{result.reasoning}\n\n"
            f"[dim]Key factors:[/dim] {', '.join(result.key_factors)}",
            title=name,
            expand=False,
        ))

        if result.sources:
            table = Table("Source", "Page", show_header=True, header_style="dim")
            for s in result.sources[:3]:
                table.add_row(str(s["source"]), str(s["page"]))
            console.print(table)

        console.print()
