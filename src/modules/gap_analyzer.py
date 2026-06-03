"""
gap_analyzer.py — maps an AI system against multiple governance frameworks
and returns per-framework compliance gaps.

How it works:
  For each framework (EU AI Act, NIST AI RMF, ISO 42001):
    1. Build a retrieval query from the system description + framework keywords
    2. Retrieve the top-K chunks most relevant to that framework's obligations
    3. Ask the LLM: given this system and these rules, what is required and what is missing?
    4. Parse the structured response into a FrameworkGap object
  Return a GapAnalysis collecting all per-framework results.

Calling analyze() with no `frameworks` argument runs all three frameworks.
Pass a subset list (e.g. ["EU AI Act"]) to run only those.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from src.rag.chain import format_docs
from src.rag.retriever import get_retriever


# ── Framework registry ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Framework:
    name: str
    description: str   # injected into the prompt so the LLM knows which rules apply
    keywords: str      # appended to retrieval query to bias chunks toward this framework


FRAMEWORKS: dict[str, _Framework] = {
    "EU AI Act": _Framework(
        name="EU AI Act",
        description=(
            "EU AI Act (Regulation (EU) 2024/1689). Key obligations for high-risk systems: "
            "risk management system (Article 9), data governance (Article 10), technical "
            "documentation (Article 11), record-keeping and logging (Article 12), "
            "transparency and user information (Article 13), human oversight measures "
            "(Article 14), accuracy, robustness, and cybersecurity (Article 15), "
            "conformity assessment before deployment (Article 43), and registration in "
            "the EU database (Article 71)."
        ),
        keywords=(
            "EU AI Act obligations requirements high-risk conformity assessment "
            "technical documentation human oversight transparency data governance "
            "accuracy robustness Article 9 10 11 12 13 14 15"
        ),
    ),
    "NIST AI RMF": _Framework(
        name="NIST AI RMF",
        description=(
            "NIST AI Risk Management Framework 1.0. Structured around four functions: "
            "GOVERN (policies, roles, culture), MAP (context and risk identification), "
            "MEASURE (analysis and assessment), MANAGE (prioritise, respond, monitor). "
            "Targets seven trustworthy AI characteristics: valid and reliable, safe, "
            "secure and resilient, explainable and interpretable, privacy-enhanced, "
            "fair with bias managed, accountable and transparent."
        ),
        keywords=(
            "NIST AI RMF risk management govern map measure manage trustworthy "
            "valid reliable safe secure explainable privacy fair accountable "
            "bias documentation monitoring"
        ),
    ),
    "ISO 42001": _Framework(
        name="ISO 42001",
        description=(
            "ISO/IEC 42001:2023 — AI Management System (AIMS) standard. Key clauses: "
            "leadership and AI policy (clause 5), AI risk and impact assessment (clause 6), "
            "support — resources, competence, awareness, communication, documented "
            "information (clause 7), operational planning and AI system lifecycle controls "
            "(clause 8), performance evaluation — monitoring, audit, management review "
            "(clause 9), and continual improvement (clause 10)."
        ),
        keywords=(
            "ISO 42001 AI management system policy objectives risk impact assessment "
            "documented information competence awareness audit management review "
            "continual improvement lifecycle"
        ),
    ),
}

ALL_FRAMEWORK_NAMES = list(FRAMEWORKS.keys())


# ── Output schema ──────────────────────────────────────────────────────────────

class FrameworkGap(BaseModel):
    framework: str
    requirements: list[str]   # what the framework mandates for this system
    gaps: list[str]           # what the described system is missing
    met: list[str]            # requirements that appear to be addressed
    sources: list[dict[str, Any]]


class GapAnalysis(BaseModel):
    system_description: str
    framework_gaps: list[FrameworkGap]

    @property
    def frameworks_analysed(self) -> list[str]:
        return [fg.framework for fg in self.framework_gaps]

    @property
    def total_gaps(self) -> int:
        return sum(len(fg.gaps) for fg in self.framework_gaps)


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a compliance expert specialising in AI governance frameworks.
Analyse the AI system against the specific framework rules provided.

Your task:
1. Identify which requirements of this framework apply to this type of AI system.
2. Based on what is described, determine which requirements appear to be met.
3. Identify which requirements are NOT addressed and represent compliance gaps.

If the context does not contain enough information about a particular requirement, \
list it as a gap (assume the worst — the burden of proof is on the system owner).

Reply with EXACTLY this format — no extra text before or after:
REQUIREMENTS: <req 1> | <req 2> | <req 3> | ...
MET: <met req 1> | <met req 2> | ... (or NONE if nothing is addressed)
GAPS: <gap 1> | <gap 2> | <gap 3> | ... (or NONE if fully compliant)\
"""

_HUMAN = """\
Framework: {framework_name}
Framework summary: {framework_description}

Relevant regulatory context extracted from the official documents:
{context}

AI system description:
{system_description}

Analyse this AI system against the {framework_name} framework.\
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_response(text: str) -> tuple[list[str], list[str], list[str]]:
    """Extract requirements, met items, and gaps from the structured LLM output."""
    requirements: list[str] = []
    met: list[str] = []
    gaps: list[str] = []

    def _split(raw: str) -> list[str]:
        items = [i.strip() for i in raw.split("|") if i.strip()]
        return [] if items == ["NONE"] else items

    for line in text.splitlines():
        if line.startswith("REQUIREMENTS:"):
            requirements = _split(line.removeprefix("REQUIREMENTS:").strip())
        elif line.startswith("MET:"):
            met = _split(line.removeprefix("MET:").strip())
        elif line.startswith("GAPS:"):
            gaps = _split(line.removeprefix("GAPS:").strip())

    return requirements, met, gaps


def _dedup_sources(docs: list) -> list[dict[str, Any]]:
    seen: set[tuple] = set()
    result = []
    for doc in docs:
        key = (doc.metadata.get("source", "unknown"), doc.metadata.get("page", "?"))
        if key not in seen:
            seen.add(key)
            result.append({"source": key[0], "page": key[1]})
    return result


def _analyse_framework(
    system_description: str,
    framework: _Framework,
    retriever,
    llm,
) -> FrameworkGap:
    """Run gap analysis for a single framework. Returns a FrameworkGap."""
    query = system_description + " " + framework.keywords
    docs = retriever.invoke(query)
    context = format_docs(docs)

    chain = _prompt | llm | StrOutputParser()
    raw = chain.invoke({
        "framework_name": framework.name,
        "framework_description": framework.description,
        "context": context,
        "system_description": system_description,
    })

    requirements, met, gaps = _parse_response(raw)

    return FrameworkGap(
        framework=framework.name,
        requirements=requirements,
        met=met,
        gaps=gaps,
        sources=_dedup_sources(docs),
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze(
    system_description: str,
    frameworks: list[str] | None = None,
) -> GapAnalysis:
    """
    Map an AI system description against governance frameworks and return gaps.

    Args:
        system_description: Plain-language description of the AI system,
            its purpose, deployment context, and data it processes.
        frameworks: Names of frameworks to check. Defaults to all three:
            ["EU AI Act", "NIST AI RMF", "ISO 42001"].
            Pass a subset to run only those.

    Returns:
        GapAnalysis with a FrameworkGap for each requested framework.

    Raises:
        ValueError: If an unknown framework name is passed.
    """
    if frameworks is None:
        frameworks = ALL_FRAMEWORK_NAMES

    unknown = set(frameworks) - set(FRAMEWORKS)
    if unknown:
        raise ValueError(f"Unknown framework(s): {unknown}. Choose from {ALL_FRAMEWORK_NAMES}.")

    retriever = get_retriever()
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    framework_gaps = [
        _analyse_framework(system_description, FRAMEWORKS[name], retriever, llm)
        for name in frameworks
    ]

    return GapAnalysis(
        system_description=system_description,
        framework_gaps=framework_gaps,
    )


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    console = Console()

    EXAMPLE = (
        "An automated CV screening and ranking tool used by a large employer to "
        "shortlist job applicants. The tool uses a machine-learning model trained "
        "on historical hiring data. No human reviews the shortlist before candidates "
        "are rejected. The system processes names, addresses, and work history. "
        "There is no documented risk assessment, no logging of decisions, and "
        "applicants are not informed that an AI made the decision."
    )

    console.rule("[bold]Gap Analysis — CV Screening Tool[/bold]")
    console.print(f"\n[dim]{EXAMPLE}[/dim]\n")

    result = analyze(EXAMPLE)

    for fg in result.framework_gaps:
        gap_colour = "red" if fg.gaps else "green"
        gap_label = f"{len(fg.gaps)} gap(s)" if fg.gaps else "No gaps found"

        table = Table(
            title=f"{fg.framework}  [{gap_colour}]{gap_label}[/{gap_colour}]",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Status", width=6)
        table.add_column("Requirement")

        for req in fg.requirements:
            if req in fg.met:
                table.add_row("[green]✓[/green]", req)
            else:
                table.add_row("[red]✗[/red]", req)

        console.print(table)

        if fg.sources:
            src_str = ", ".join(f"{s['source']} p.{s['page']}" for s in fg.sources[:3])
            console.print(f"  [dim]Sources: {src_str}[/dim]\n")

    console.print(
        f"\n[bold]Total gaps across {len(result.framework_gaps)} framework(s): "
        f"[red]{result.total_gaps}[/red][/bold]"
    )
