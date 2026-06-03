"""
model_card_generator.py — generates a HuggingFace-format model card from
structured system metadata, with LLM-authored governance sections grounded
in the regulatory documents.

How it works:
  1. User fills in a ModelCardInput (factual metadata about the model)
  2. A retrieval query is built from the model type + intended use + governance keywords
  3. Retrieved chunks provide regulatory grounding for the written sections
  4. A single LLM call produces: ethical considerations, limitations expansion,
     caveats, and regulatory recommendations — all in structured format
  5. Everything is assembled into a ModelCard whose to_markdown() method
     renders a complete, standards-compliant model card document

The LLM only writes the sections that require governance expertise.
All factual sections (metrics, training data, intended use) come
directly from the user's input — no hallucination risk there.
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


# ── Input schema ───────────────────────────────────────────────────────────────

class ModelCardInput(BaseModel):
    """Structured metadata about the AI system. Fill in as much as you know."""

    # Required
    model_name: str
    model_type: str = Field(description="e.g. 'binary classifier', 'LLM', 'regression model'")
    intended_use: str = Field(description="What problem does it solve and in what context?")
    training_data_description: str = Field(description="What data was the model trained on?")

    # Organisational
    organization: str = "Not specified"
    version: str = "1.0.0"
    license_type: str = "Not specified"
    contact: str = "Not specified"

    # Use context
    primary_users: str = "Not specified"
    deployment_context: str = "Not specified"
    out_of_scope_uses: str = "Not specified"

    # Performance
    metrics: str = "Not specified"
    evaluation_data: str = "Not specified"

    # Known issues (user-supplied, LLM will expand on these)
    known_limitations: str = "Not specified"


# ── Output schema ──────────────────────────────────────────────────────────────

class ModelCard(BaseModel):
    """Complete model card — factual fields from input, written fields from LLM."""

    # Passed through from input
    input: ModelCardInput
    generated_date: str

    # LLM-generated governance sections
    ethical_considerations: str
    limitations_expanded: str
    caveats: str
    recommendations: str

    # Sources used for the LLM-generated sections
    sources: list[dict[str, Any]]

    def to_markdown(self) -> str:
        """Render the full model card as a Markdown string (HuggingFace format)."""
        i = self.input
        _na = "Not specified"

        def _section(title: str, body: str) -> str:
            return f"## {title}\n\n{body}\n"

        def _row(label: str, value: str) -> str:
            return f"| {label} | {value} |"

        details_rows = "\n".join([
            "| Field | Value |",
            "|---|---|",
            _row("Model name", i.model_name),
            _row("Version", i.version),
            _row("Model type", i.model_type),
            _row("Organization", i.organization),
            _row("License", i.license_type),
            _row("Contact", i.contact),
            _row("Generated", self.generated_date),
        ])

        limitations_body = ""
        if i.known_limitations and i.known_limitations != _na:
            limitations_body += f"**Known limitations (user-supplied):**\n{i.known_limitations}\n\n"
        limitations_body += f"**Expanded analysis:**\n{self.limitations_expanded}"

        sources_body = _na
        if self.sources:
            lines = [f"- {s['source']}, page {s['page']}" for s in self.sources[:5]]
            sources_body = "\n".join(lines)

        sections = [
            f"# Model Card: {i.model_name}\n",
            _section("Model Details", details_rows),
            _section(
                "Intended Use",
                f"**Primary intended use:** {i.intended_use}\n\n"
                f"**Primary users:** {i.primary_users}\n\n"
                f"**Deployment context:** {i.deployment_context}\n\n"
                f"**Out-of-scope uses:** {i.out_of_scope_uses}",
            ),
            _section("Training Data", i.training_data_description),
            _section(
                "Evaluation",
                f"**Metrics:** {i.metrics}\n\n"
                f"**Evaluation data:** {i.evaluation_data}",
            ),
            _section("Limitations", limitations_body),
            _section("Ethical Considerations", self.ethical_considerations),
            _section("Caveats and Recommendations", f"{self.caveats}\n\n{self.recommendations}"),
            _section(
                "Regulatory Grounding",
                f"The governance sections of this card were generated using retrieved "
                f"excerpts from EU AI Act, NIST AI RMF, and ISO/IEC 42001.\n\n{sources_body}",
            ),
        ]

        return "\n---\n\n".join(sections)


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an AI governance expert writing the regulatory sections of a model card \
for an AI system. Use ONLY the provided regulatory context to ground your writing.

Model name: {model_name}
Model type: {model_type}
Intended use: {intended_use}
Deployment context: {deployment_context}
Known limitations (user-supplied): {known_limitations}

Regulatory context from EU AI Act, NIST AI RMF, and ISO 42001:
{context}

Write four governance sections for this model card. Be specific — reference \
articles, framework functions, or clauses from the context where relevant. \
Keep each section focused and practical (3-6 sentences or bullet points).

Reply with EXACTLY this format — no extra text before or after:
ETHICAL_CONSIDERATIONS: <paragraph covering data ethics, fairness, transparency obligations, and human rights impacts relevant to this system>
LIMITATIONS_EXPANDED: <paragraph expanding on technical and deployment limitations, referencing regulatory expectations around accuracy and robustness>
CAVEATS: <paragraph on conditions under which the model should not be used or results should be treated with caution>
RECOMMENDATIONS: <paragraph of concrete steps the deploying organisation should take to meet regulatory obligations>\
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
])

_RETRIEVAL_KEYWORDS = (
    "transparency technical documentation data governance human oversight "
    "accuracy robustness ethical considerations limitations fairness bias "
    "EU AI Act NIST RMF ISO 42001 model card requirements"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_response(text: str) -> tuple[str, str, str, str]:
    """Extract the four LLM-generated sections from the structured response."""
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    keys = {"ETHICAL_CONSIDERATIONS", "LIMITATIONS_EXPANDED", "CAVEATS", "RECOMMENDATIONS"}

    for line in text.splitlines():
        matched = False
        for key in keys:
            if line.startswith(f"{key}:"):
                if current_key:
                    sections[current_key] = " ".join(current_lines).strip()
                current_key = key
                current_lines = [line.removeprefix(f"{key}:").strip()]
                matched = True
                break
        if not matched and current_key:
            current_lines.append(line.strip())

    if current_key:
        sections[current_key] = " ".join(current_lines).strip()

    return (
        sections.get("ETHICAL_CONSIDERATIONS", ""),
        sections.get("LIMITATIONS_EXPANDED", ""),
        sections.get("CAVEATS", ""),
        sections.get("RECOMMENDATIONS", ""),
    )


def _dedup_sources(docs: list) -> list[dict[str, Any]]:
    seen: set[tuple] = set()
    result = []
    for doc in docs:
        key = (doc.metadata.get("source", "unknown"), doc.metadata.get("page", "?"))
        if key not in seen:
            seen.add(key)
            result.append({"source": key[0], "page": key[1]})
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def generate(input: ModelCardInput) -> ModelCard:
    """
    Generate a model card for the described AI system.

    Args:
        input: ModelCardInput with factual metadata about the system.

    Returns:
        ModelCard with all sections populated and a to_markdown() method.
    """
    retriever = get_retriever()
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    query = (
        f"{input.model_type} {input.intended_use} {input.deployment_context} "
        + _RETRIEVAL_KEYWORDS
    )
    docs = retriever.invoke(query)
    context = format_docs(docs)

    chain = _prompt | llm | StrOutputParser()
    raw = chain.invoke({
        "model_name": input.model_name,
        "model_type": input.model_type,
        "intended_use": input.intended_use,
        "deployment_context": input.deployment_context,
        "known_limitations": input.known_limitations,
        "context": context,
    })

    ethical, limitations_expanded, caveats, recommendations = _parse_response(raw)

    return ModelCard(
        input=input,
        generated_date=date.today().isoformat(),
        ethical_considerations=ethical,
        limitations_expanded=limitations_expanded,
        caveats=caveats,
        recommendations=recommendations,
        sources=_dedup_sources(docs),
    )


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    console = Console()

    example_input = ModelCardInput(
        model_name="HireScore v2",
        model_type="Binary classifier (shortlist / reject)",
        intended_use=(
            "Automated screening of job applications for entry-level roles. "
            "Ranks applicants and produces a binary shortlist/reject decision "
            "passed to a human recruiter."
        ),
        training_data_description=(
            "Trained on 5 years of historical hiring decisions (120,000 applications) "
            "from a single large European employer. Labels derived from interviewer "
            "outcome scores. Data includes CV text, self-reported demographics "
            "(optional), and role metadata."
        ),
        organization="ExampleCorp Ltd",
        version="2.1.0",
        license_type="Proprietary",
        contact="ai-governance@examplecorp.com",
        primary_users="HR departments and recruitment teams",
        deployment_context=(
            "Deployed as an internal SaaS tool; decisions reviewed by a human "
            "recruiter before rejection letters are sent."
        ),
        out_of_scope_uses=(
            "Not for use in promotion or disciplinary decisions. "
            "Not for roles outside entry-level. "
            "Not for use without human review of rejections."
        ),
        metrics="Precision 0.81, Recall 0.74, F1 0.77 on hold-out set (2024 Q4)",
        evaluation_data="Random 20% hold-out from the training corpus, stratified by role category",
        known_limitations=(
            "Model may perpetuate historical biases present in training labels. "
            "Lower recall on applicants from non-English-speaking backgrounds. "
            "Not validated on roles outside the original employer's context."
        ),
    )

    console.rule("[bold]Model Card Generator — HireScore v2[/bold]")
    card = generate(example_input)
    markdown = card.to_markdown()

    console.print(Panel(
        Syntax(markdown, "markdown", theme="monokai", word_wrap=True),
        title="Generated Model Card",
        expand=True,
    ))

    console.print(f"\n[dim]Total length: {len(markdown)} characters[/dim]")
    console.print(f"[dim]Sources used: {len(card.sources)}[/dim]")
