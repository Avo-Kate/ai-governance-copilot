"""
components.py — shared Streamlit UI helpers.

Pure presentation logic lives here so app.py stays focused on page
flow and module orchestration. Nothing in this file calls Ollama;
the renderers take already-computed module results and lay them out.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config import VECTORSTORE_PATH
from src.modules.risk_classifier import RiskClassification
from src.modules.gap_analyzer import GapAnalysis
from src.modules.model_card_generator import ModelCard
from src.modules.report_writer import AssessmentReport


# ── Risk-tier presentation ───────────────────────────────────────────────────

# Emoji badges match report_writer.to_markdown() so the UI and the exported
# report speak the same visual language.
TIER_BADGES = {
    "Unacceptable": "🔴",
    "High": "🟠",
    "Limited": "🟡",
    "Minimal": "🟢",
}

# Streamlit status-box flavour per tier — drives the coloured callout.
TIER_STATUS = {
    "Unacceptable": "error",
    "High": "warning",
    "Limited": "info",
    "Minimal": "success",
}


def tier_label(tier: str) -> str:
    """'High' → '🟠 High Risk'."""
    return f"{TIER_BADGES.get(tier, '⚪')} {tier} Risk"


# ── Environment readiness ────────────────────────────────────────────────────

def vectorstore_ready() -> bool:
    """
    True if the ChromaDB store has been built. We check for the sqlite file
    Chroma writes on persist — an empty/absent directory means the user
    hasn't run ingestion yet.
    """
    if not VECTORSTORE_PATH.exists():
        return False
    return any(VECTORSTORE_PATH.glob("chroma.sqlite3")) or any(
        VECTORSTORE_PATH.iterdir()
    )


def require_vectorstore() -> bool:
    """
    Render a blocking warning if the vectorstore is missing.
    Returns True when the store is ready (caller may proceed), False otherwise.
    """
    if vectorstore_ready():
        return True
    st.warning(
        "**No document index found.** Before using the copilot you need to "
        "ingest the regulatory PDFs.\n\n"
        "1. Place the PDFs in the `docs/` folder\n"
        "2. Run `uv run python -m src.ingestion.embedder`\n\n"
        "Then refresh this page.",
        icon="📚",
    )
    return False


# ── Source citations ─────────────────────────────────────────────────────────

def render_sources(sources: list[dict[str, Any]], *, limit: int | None = None) -> None:
    """Render a deduplicated list of (source, page) citations."""
    if not sources:
        st.caption("No source documents retrieved.")
        return

    shown = sources if limit is None else sources[:limit]
    lines = [f"- **{s.get('source', 'unknown')}** — page {s.get('page', '?')}" for s in shown]
    st.markdown("\n".join(lines))
    if limit is not None and len(sources) > limit:
        st.caption(f"…and {len(sources) - limit} more.")


# ── Result renderers ─────────────────────────────────────────────────────────

def render_classification(rc: RiskClassification) -> None:
    """Lay out a RiskClassification: tier callout, reasoning, factors, sources."""
    status = TIER_STATUS.get(rc.tier, "info")
    getattr(st, status)(f"**Risk tier: {tier_label(rc.tier)}**")

    st.markdown("**Reasoning**")
    st.write(rc.reasoning)

    if rc.key_factors:
        st.markdown("**Key determining factors**")
        for factor in rc.key_factors:
            st.markdown(f"- {factor}")

    with st.expander("Regulatory sources"):
        render_sources(rc.sources)


def render_gap_analysis(ga: GapAnalysis) -> None:
    """Lay out a GapAnalysis: a summary metric row, then per-framework tables."""
    col1, col2 = st.columns(2)
    col1.metric("Frameworks assessed", len(ga.framework_gaps))
    col2.metric("Total gaps identified", ga.total_gaps)

    for fg in ga.framework_gaps:
        gap_count = len(fg.gaps)
        header = f"{fg.framework} — {gap_count} gap(s)" if gap_count else f"{fg.framework} — no gaps"
        with st.expander(header, expanded=bool(gap_count)):
            c1, c2, c3 = st.columns(3)
            c1.metric("Requirements", len(fg.requirements))
            c2.metric("Met", len(fg.met))
            c3.metric("Gaps", gap_count)

            if fg.requirements:
                rows = [
                    {
                        "Status": "✅ Met" if req in fg.met else "❌ Gap",
                        "Requirement": req,
                    }
                    for req in fg.requirements
                ]
                st.table(rows)
            else:
                st.caption("No specific requirements were identified for this system.")

            st.markdown("**Sources**")
            render_sources(fg.sources, limit=5)


def render_model_card(card: ModelCard) -> None:
    """Show a generated model card: rendered preview + raw-markdown download."""
    markdown = card.to_markdown()

    preview_tab, raw_tab = st.tabs(["Preview", "Markdown"])
    with preview_tab:
        st.markdown(markdown)
    with raw_tab:
        st.code(markdown, language="markdown")

    st.download_button(
        "⬇ Download model card (.md)",
        data=markdown,
        file_name=f"model_card_{_slug(card.input.model_name)}.md",
        mime="text/markdown",
    )


def render_report(report: AssessmentReport) -> None:
    """Show a generated assessment report: rendered preview + download."""
    markdown = report.to_markdown()

    preview_tab, raw_tab = st.tabs(["Preview", "Markdown"])
    with preview_tab:
        st.markdown(markdown)
    with raw_tab:
        st.code(markdown, language="markdown")

    st.download_button(
        "⬇ Download report (.md)",
        data=markdown,
        file_name=f"risk_assessment_{_slug(report.system_name)}.md",
        mime="text/markdown",
    )


# ── Internal ─────────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    """'HireScore v2' → 'hirescore_v2' for safe filenames."""
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_") or "untitled"
