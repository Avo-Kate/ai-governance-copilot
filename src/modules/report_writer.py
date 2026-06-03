"""
report_writer.py — combines risk classification and gap analysis into a
formal AI risk assessment report, with an LLM-authored executive summary
and prioritised recommended actions.

How it works:
  1. Accept a ReportInput (system description + optional pre-computed results)
  2. Run the risk classifier if no pre-computed RiskClassification is provided
  3. Run the gap analyzer if no pre-computed GapAnalysis is provided
  4. Retrieve regulatory context chunks scoped to remediation and obligations
  5. Single LLM call → executive summary + prioritised recommended actions
  6. Assemble everything into an AssessmentReport with a to_markdown() method

Passing pre-computed results avoids redundant LLM calls when the caller
has already run the classifier and analyzer (e.g. the Streamlit UI).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from src.rag.chain import format_docs
from src.rag.retriever import get_retriever
from src.modules.risk_classifier import RiskClassification, classify
from src.modules.gap_analyzer import GapAnalysis, analyze


# ── Input schema ───────────────────────────────────────────────────────────────

class ReportInput(BaseModel):
    system_description: str
    system_name: str = "AI System"
    organization: str = "Not specified"
    assessor: str = "Not specified"

    # Pre-computed results — supply these to skip redundant LLM calls.
    # If None, write() will run the classifier / analyzer automatically.
    risk_classification: RiskClassification | None = Field(default=None)
    gap_analysis: GapAnalysis | None = Field(default=None)


# ── Output schema ──────────────────────────────────────────────────────────────

class AssessmentReport(BaseModel):
    system_name: str
    organization: str
    assessor: str
    assessment_date: str
    system_description: str
    risk_classification: RiskClassification
    gap_analysis: GapAnalysis
    executive_summary: str
    recommended_actions: list[str]
    sources: list[dict[str, Any]]

    @property
    def total_gaps(self) -> int:
        return self.gap_analysis.total_gaps

    def to_markdown(self) -> str:
        """Render the full assessment report as a Markdown document."""
        rc = self.risk_classification
        ga = self.gap_analysis

        tier_emoji = {
            "Unacceptable": "🔴",
            "High": "🟠",
            "Limited": "🟡",
            "Minimal": "🟢",
        }
        tier_badge = tier_emoji.get(rc.tier, "⚪")

        lines: list[str] = []

        # ── Title block ────────────────────────────────────────────────────────
        lines += [
            f"# AI Risk Assessment Report",
            "",
            f"**System:** {self.system_name}  ",
            f"**Organization:** {self.organization}  ",
            f"**Assessor:** {self.assessor}  ",
            f"**Date:** {self.assessment_date}  ",
            "",
            "---",
            "",
        ]

        # ── Executive summary ──────────────────────────────────────────────────
        lines += [
            "## Executive Summary",
            "",
            self.executive_summary,
            "",
            "---",
            "",
        ]

        # ── System description ─────────────────────────────────────────────────
        lines += [
            "## System Description",
            "",
            self.system_description,
            "",
            "---",
            "",
        ]

        # ── Risk classification ────────────────────────────────────────────────
        lines += [
            "## Risk Classification",
            "",
            f"**Risk Tier:** {tier_badge} {rc.tier}  ",
            f"**Framework:** EU AI Act  ",
            "",
            f"**Reasoning:** {rc.reasoning}",
            "",
        ]
        if rc.key_factors:
            lines += ["**Key determining factors:**", ""]
            for factor in rc.key_factors:
                lines.append(f"- {factor}")
            lines.append("")
        lines += ["---", ""]

        # ── Gap analysis ───────────────────────────────────────────────────────
        lines += [
            "## Compliance Gap Analysis",
            "",
            f"**Frameworks assessed:** {', '.join(ga.frameworks_analysed)}  ",
            f"**Total gaps identified:** {self.total_gaps}",
            "",
        ]

        for fg in ga.framework_gaps:
            gap_count = len(fg.gaps)
            met_count = len(fg.met)
            lines += [
                f"### {fg.framework}",
                "",
                f"**Requirements identified:** {len(fg.requirements)} | "
                f"**Met:** {met_count} | "
                f"**Gaps:** {gap_count}",
                "",
            ]

            if fg.requirements:
                lines += ["| Status | Requirement |", "|---|---|"]
                for req in fg.requirements:
                    status = "✅ Met" if req in fg.met else "❌ Gap"
                    lines.append(f"| {status} | {req} |")
                lines.append("")

        lines += ["---", ""]

        # ── Recommended actions ────────────────────────────────────────────────
        lines += [
            "## Recommended Actions",
            "",
            "_Prioritised remediation steps based on identified gaps and applicable obligations:_",
            "",
        ]
        for i, action in enumerate(self.recommended_actions, 1):
            lines.append(f"{i}. {action}")
        lines += ["", "---", ""]

        # ── Sources ────────────────────────────────────────────────────────────
        lines += ["## Regulatory Sources", ""]
        if self.sources:
            for s in self.sources:
                lines.append(f"- {s['source']}, page {s['page']}")
        else:
            lines.append("_No source documents retrieved._")

        return "\n".join(lines)


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a senior AI governance consultant writing a formal risk assessment report.

Write two sections of the report. Be specific and professional. \
Reference articles, annexes, or framework functions from the context where relevant.

Reply with EXACTLY this format — no extra text before or after:
EXECUTIVE_SUMMARY: <2-3 paragraphs summarising the system, its risk tier and why, \
and the most critical compliance gaps requiring urgent attention>
RECOMMENDED_ACTIONS: <action 1> | <action 2> | <action 3> | <action 4> | <action 5>\
"""

_HUMAN = """\
System name: {system_name}
System description: {system_description}

Risk classification result:
  Tier: {risk_tier}
  Reasoning: {risk_reasoning}
  Key factors: {risk_factors}

Compliance gaps identified ({total_gaps} total across {framework_count} framework(s)):
{gaps_summary}

Regulatory context from the governance documents:
{context}

Write the executive summary and recommended actions.\
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])

_RETRIEVAL_KEYWORDS = (
    "obligations requirements remediation conformity assessment registration "
    "documentation human oversight risk management GOVERN MANAGE technical "
    "documentation EU AI Act NIST RMF ISO 42001"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_gaps_summary(ga: GapAnalysis) -> str:
    """Flatten all gaps into a compact summary string for the prompt."""
    lines = []
    for fg in ga.framework_gaps:
        if fg.gaps:
            gap_list = "; ".join(fg.gaps)
            lines.append(f"  {fg.framework}: {gap_list}")
    return "\n".join(lines) if lines else "  No gaps identified."


def _parse_response(text: str) -> tuple[str, list[str]]:
    """Extract executive summary and recommended actions from the LLM output."""
    summary = ""
    actions: list[str] = []

    current_key: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("EXECUTIVE_SUMMARY:"):
            current_key = "summary"
            current_lines = [line.removeprefix("EXECUTIVE_SUMMARY:").strip()]
        elif line.startswith("RECOMMENDED_ACTIONS:"):
            if current_key == "summary":
                summary = " ".join(current_lines).strip()
            current_key = "actions"
            current_lines = [line.removeprefix("RECOMMENDED_ACTIONS:").strip()]
        elif current_key:
            current_lines.append(line.strip())

    remainder = " ".join(current_lines).strip()
    if current_key == "summary":
        summary = remainder
    elif current_key == "actions":
        actions = [a.strip() for a in remainder.split("|") if a.strip()]

    return summary, actions


def _dedup_sources(*source_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge and deduplicate sources from multiple module results."""
    seen: set[tuple] = set()
    result = []
    for sources in source_lists:
        for s in sources:
            key = (s.get("source", "unknown"), s.get("page", "?"))
            if key not in seen:
                seen.add(key)
                result.append({"source": key[0], "page": key[1]})
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def write(input: ReportInput) -> AssessmentReport:
    """
    Generate a formal AI risk assessment report.

    Runs the risk classifier and gap analyzer if pre-computed results are not
    supplied in the input. Use pre-computed results to avoid redundant LLM calls
    when the caller has already run those modules.

    Args:
        input: ReportInput with system metadata and optional pre-computed results.

    Returns:
        AssessmentReport with all sections populated and a to_markdown() method.
    """
    rc = input.risk_classification or classify(input.system_description)
    ga = input.gap_analysis or analyze(input.system_description)

    retriever = get_retriever()
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    query = input.system_description + " " + _RETRIEVAL_KEYWORDS
    docs = retriever.invoke(query)
    context = format_docs(docs)

    chain = _prompt | llm | StrOutputParser()
    raw = chain.invoke({
        "system_name": input.system_name,
        "system_description": input.system_description,
        "risk_tier": rc.tier,
        "risk_reasoning": rc.reasoning,
        "risk_factors": ", ".join(rc.key_factors),
        "total_gaps": ga.total_gaps,
        "framework_count": len(ga.framework_gaps),
        "gaps_summary": _build_gaps_summary(ga),
        "context": context,
    })

    executive_summary, recommended_actions = _parse_response(raw)

    retriever_sources = [
        {"source": d.metadata.get("source", "unknown"), "page": d.metadata.get("page", "?")}
        for d in docs
    ]
    all_sources = _dedup_sources(rc.sources, ga.framework_gaps[0].sources if ga.framework_gaps else [], retriever_sources)

    return AssessmentReport(
        system_name=input.system_name,
        organization=input.organization,
        assessor=input.assessor,
        assessment_date=date.today().isoformat(),
        system_description=input.system_description,
        risk_classification=rc,
        gap_analysis=ga,
        executive_summary=executive_summary,
        recommended_actions=recommended_actions,
        sources=all_sources,
    )


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    console = Console()

    inp = ReportInput(
        system_name="HireScore v2",
        organization="ExampleCorp Ltd",
        assessor="AI Governance Team",
        system_description=(
            "An automated CV screening and ranking tool used by a large employer to "
            "shortlist job applicants for entry-level roles. The system uses a "
            "machine-learning model trained on five years of historical hiring data. "
            "No human reviews the shortlist before candidates are rejected. The system "
            "processes names, addresses, and work history. There is no documented risk "
            "assessment, no logging of decisions, and applicants are not informed that "
            "an AI made the decision."
        ),
    )

    console.rule("[bold]AI Risk Assessment Report — HireScore v2[/bold]")
    report = write(inp)
    md = report.to_markdown()

    console.print(Panel(
        Syntax(md, "markdown", theme="monokai", word_wrap=True),
        title="Assessment Report",
        expand=True,
    ))

    console.print(f"\n[dim]Risk tier: {report.risk_classification.tier}[/dim]")
    console.print(f"[dim]Total gaps: {report.total_gaps}[/dim]")
    console.print(f"[dim]Recommended actions: {len(report.recommended_actions)}[/dim]")
