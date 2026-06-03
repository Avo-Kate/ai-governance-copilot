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

import re
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
Classify the AI system into exactly one of these four risk tiers:

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

Use ONLY the regulatory context provided to ground your reasoning. \
If a relevant article or annex is cited in the context, mention it.

Reply with EXACTLY this format — no extra text before or after:
TIER: <Unacceptable | High | Limited | Minimal>
REASONING: <2-4 sentences citing specific articles or annexes>
KEY_FACTORS: <factor 1> | <factor 2> | <factor 3>\
"""

_HUMAN = """\
Regulatory context:
{context}

AI system description:
{system_description}

Classify this AI system.\
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])


# ── Helpers ────────────────────────────────────────────────────────────────────

# Extra keywords appended to the retrieval query so we pull risk-tier chunks
# even when the system description doesn't mention them explicitly.
_RETRIEVAL_SUFFIX = (
    " EU AI Act risk classification prohibited high-risk Annex III Article 5 "
    "limited minimal transparency obligations"
)


# Matches a labelled line tolerantly: optional leading bullets/quotes/markdown,
# the label, optional whitespace before the colon, and the rest as the value.
# Case-insensitive so "tier:" works as well as "TIER:".
_LABEL_RE = re.compile(
    r"^\s*[>#\-\*_]*\s*(TIER|REASONING|KEY_FACTORS)\s*:\s*(.*)$",
    re.IGNORECASE,
)


def _strip_md(s: str) -> str:
    """Drop markdown emphasis / code characters and surrounding whitespace."""
    return s.replace("**", "").replace("*", "").replace("`", "").replace("__", "").strip()


def _scan_for_tier(text: str) -> str:
    """
    Fallback when the structured TIER line is missing or unparseable:
    return the first valid tier name mentioned anywhere in the response.
    """
    lowered = text.lower()
    best_pos: int | None = None
    best_tier = ""
    for tier in VALID_TIERS:
        pos = lowered.find(tier.lower())
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_pos, best_tier = pos, tier
    return best_tier


def _parse_response(text: str) -> tuple[str, str, list[str]]:
    """Extract tier, reasoning, and key_factors from the structured LLM output.

    Tolerates real-world LLM formatting drift: markdown emphasis around labels
    or values (``**TIER:** High``), lowercase labels, leading bullets, and
    preamble lines. Falls back to scanning the whole response for a tier name
    when the TIER line itself can't be parsed.
    """
    fields = {"TIER": "", "REASONING": "", "KEY_FACTORS": ""}

    for line in text.splitlines():
        match = _LABEL_RE.match(line)
        if match:
            label = match.group(1).upper()
            fields[label] = _strip_md(match.group(2))

    tier = fields["TIER"]
    if tier not in VALID_TIERS:
        tier = _scan_for_tier(text) or tier

    reasoning = fields["REASONING"]
    key_factors = [f.strip() for f in fields["KEY_FACTORS"].split("|") if f.strip()]
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

    # Surface the model's actual output when we can't extract a tier — far more
    # useful for diagnosis than the validator's opaque "Unknown tier ''".
    if tier not in VALID_TIERS:
        snippet = raw.strip() or "(the model returned an empty response)"
        raise ValueError(
            "Could not parse a risk tier from the model's response. "
            "The model may not have followed the requested format.\n\n"
            f"--- Raw model output ---\n{snippet[:1000]}"
        )

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
