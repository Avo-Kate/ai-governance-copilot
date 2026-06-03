"""
Tests for the report writer — all LLM, retriever, and upstream module
calls are mocked so the suite runs without a live Ollama instance.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.modules.gap_analyzer import FrameworkGap, GapAnalysis
from src.modules.risk_classifier import RiskClassification
from src.modules.report_writer import (
    AssessmentReport,
    ReportInput,
    _build_gaps_summary,
    _dedup_sources,
    _parse_response,
    write,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_risk_classification(**kwargs) -> RiskClassification:
    defaults = dict(
        tier="High",
        reasoning="CV screening falls under Annex III Article 4(4).",
        key_factors=["employment context", "automated rejection", "Annex III"],
        sources=[{"source": "eu_ai_act.pdf", "page": 45}],
    )
    defaults.update(kwargs)
    return RiskClassification(**defaults)


def _make_gap_analysis(**kwargs) -> GapAnalysis:
    defaults = dict(
        system_description="CV screening tool",
        framework_gaps=[
            FrameworkGap(
                framework="EU AI Act",
                requirements=["Risk management system", "Logging", "Human oversight"],
                met=["Human oversight"],
                gaps=["Risk management system", "Logging"],
                sources=[{"source": "eu_ai_act.pdf", "page": 9}],
            ),
            FrameworkGap(
                framework="NIST AI RMF",
                requirements=["GOVERN policies", "MEASURE bias"],
                met=[],
                gaps=["GOVERN policies", "MEASURE bias"],
                sources=[{"source": "nist_rmf.pdf", "page": 12}],
            ),
        ],
    )
    defaults.update(kwargs)
    return GapAnalysis(**defaults)


_FAKE_DOCS = [
    Document(page_content="Article 9: risk management system.", metadata={"source": "eu_ai_act.pdf", "page": 9}),
    Document(page_content="GOVERN: establish AI policies.", metadata={"source": "nist_rmf.pdf", "page": 5}),
]

_FAKE_LLM_RESPONSE = (
    "EXECUTIVE_SUMMARY: HireScore v2 has been assessed as High risk under the EU AI Act "
    "due to its use in employment decisions (Annex III). Critical gaps include missing risk "
    "management documentation and lack of decision logging.\n"
    "This system requires immediate remediation before continued deployment.\n"
    "RECOMMENDED_ACTIONS: Implement risk management system per Article 9 | "
    "Enable decision logging per Article 12 | "
    "Register in EU AI Act database | "
    "Conduct conformity assessment per Article 43 | "
    "Notify applicants of automated processing"
)


# ── _parse_response ────────────────────────────────────────────────────────────

def test_parse_response_extracts_summary_and_actions():
    summary, actions = _parse_response(_FAKE_LLM_RESPONSE)
    assert "High risk" in summary
    assert "Annex III" in summary
    assert len(actions) == 5
    assert "Article 9" in actions[0]


def test_parse_response_multi_line_summary():
    raw = (
        "EXECUTIVE_SUMMARY: First paragraph.\n"
        "Second paragraph continues here.\n"
        "RECOMMENDED_ACTIONS: Action A | Action B"
    )
    summary, actions = _parse_response(raw)
    assert "First paragraph." in summary
    assert "Second paragraph continues here." in summary
    assert actions == ["Action A", "Action B"]


def test_parse_response_missing_actions():
    raw = "EXECUTIVE_SUMMARY: Summary only."
    summary, actions = _parse_response(raw)
    assert summary == "Summary only."
    assert actions == []


def test_parse_response_missing_summary():
    raw = "RECOMMENDED_ACTIONS: Do this | Do that"
    summary, actions = _parse_response(raw)
    assert summary == ""
    assert actions == ["Do this", "Do that"]


def test_parse_response_empty():
    summary, actions = _parse_response("")
    assert summary == ""
    assert actions == []


# ── _build_gaps_summary ────────────────────────────────────────────────────────

def test_build_gaps_summary_formats_all_frameworks():
    ga = _make_gap_analysis()
    result = _build_gaps_summary(ga)
    assert "EU AI Act" in result
    assert "Risk management system" in result
    assert "NIST AI RMF" in result
    assert "GOVERN policies" in result


def test_build_gaps_summary_no_gaps():
    ga = GapAnalysis(
        system_description="compliant system",
        framework_gaps=[
            FrameworkGap(framework="EU AI Act", requirements=["Logging"], met=["Logging"], gaps=[], sources=[]),
        ],
    )
    result = _build_gaps_summary(ga)
    assert "No gaps identified" in result


# ── _dedup_sources ─────────────────────────────────────────────────────────────

def test_dedup_sources_merges_multiple_lists():
    a = [{"source": "eu.pdf", "page": 5}]
    b = [{"source": "nist.pdf", "page": 3}]
    c = [{"source": "eu.pdf", "page": 5}]  # duplicate of a
    result = _dedup_sources(a, b, c)
    assert len(result) == 2


def test_dedup_sources_empty_lists():
    result = _dedup_sources([], [], [])
    assert result == []


def test_dedup_sources_single_list():
    sources = [{"source": "x.pdf", "page": 1}, {"source": "x.pdf", "page": 1}]
    result = _dedup_sources(sources)
    assert len(result) == 1


# ── AssessmentReport model ─────────────────────────────────────────────────────

def _make_report(**kwargs) -> AssessmentReport:
    defaults = dict(
        system_name="HireScore v2",
        organization="ExampleCorp",
        assessor="AI Team",
        assessment_date="2026-06-03",
        system_description="A CV screening tool.",
        risk_classification=_make_risk_classification(),
        gap_analysis=_make_gap_analysis(),
        executive_summary="This system is High risk due to employment context.",
        recommended_actions=["Implement risk management", "Enable logging"],
        sources=[{"source": "eu_ai_act.pdf", "page": 9}],
    )
    defaults.update(kwargs)
    return AssessmentReport(**defaults)


def test_assessment_report_total_gaps():
    report = _make_report()
    assert report.total_gaps == 4  # 2 from EU AI Act + 2 from NIST RMF


def test_to_markdown_contains_system_name():
    md = _make_report().to_markdown()
    assert "HireScore v2" in md


def test_to_markdown_contains_all_sections():
    md = _make_report().to_markdown()
    for heading in [
        "Executive Summary",
        "System Description",
        "Risk Classification",
        "Compliance Gap Analysis",
        "Recommended Actions",
        "Regulatory Sources",
    ]:
        assert heading in md, f"Missing section: {heading}"


def test_to_markdown_risk_tier_present():
    md = _make_report().to_markdown()
    assert "High" in md
    assert "🟠" in md


def test_to_markdown_gaps_table_present():
    md = _make_report().to_markdown()
    assert "❌ Gap" in md
    assert "✅ Met" in md


def test_to_markdown_recommended_actions_numbered():
    md = _make_report().to_markdown()
    assert "1. Implement risk management" in md
    assert "2. Enable logging" in md


def test_to_markdown_sources_listed():
    md = _make_report().to_markdown()
    assert "eu_ai_act.pdf" in md


def test_to_markdown_all_frameworks_in_gap_section():
    md = _make_report().to_markdown()
    assert "EU AI Act" in md
    assert "NIST AI RMF" in md


# ── write() — integration with mocks ──────────────────────────────────────────

@patch("src.modules.report_writer.get_retriever")
@patch("src.modules.report_writer.ChatOllama")
def test_write_with_precomputed_results_skips_classifier_and_analyzer(
    mock_llm_cls, mock_get_retriever
):
    """When pre-computed results are provided, classify() and analyze() should not be called."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = _FAKE_DOCS
    mock_get_retriever.return_value = mock_retriever
    mock_llm_cls.return_value = MagicMock()

    with patch("src.modules.report_writer._prompt") as mock_prompt, \
         patch("src.modules.report_writer.classify") as mock_classify, \
         patch("src.modules.report_writer.analyze") as mock_analyze:

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _FAKE_LLM_RESPONSE
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        inp = ReportInput(
            system_name="HireScore v2",
            system_description="A CV screening tool.",
            risk_classification=_make_risk_classification(),
            gap_analysis=_make_gap_analysis(),
        )
        result = write(inp)

    mock_classify.assert_not_called()
    mock_analyze.assert_not_called()
    assert isinstance(result, AssessmentReport)


@patch("src.modules.report_writer.get_retriever")
@patch("src.modules.report_writer.ChatOllama")
def test_write_without_precomputed_calls_classifier_and_analyzer(
    mock_llm_cls, mock_get_retriever
):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = _FAKE_DOCS
    mock_get_retriever.return_value = mock_retriever
    mock_llm_cls.return_value = MagicMock()

    with patch("src.modules.report_writer._prompt") as mock_prompt, \
         patch("src.modules.report_writer.classify", return_value=_make_risk_classification()) as mock_classify, \
         patch("src.modules.report_writer.analyze", return_value=_make_gap_analysis()) as mock_analyze:

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _FAKE_LLM_RESPONSE
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        inp = ReportInput(
            system_name="HireScore v2",
            system_description="A CV screening tool.",
        )
        result = write(inp)

    mock_classify.assert_called_once_with("A CV screening tool.")
    mock_analyze.assert_called_once_with("A CV screening tool.")
    assert result.risk_classification.tier == "High"


@patch("src.modules.report_writer.get_retriever")
@patch("src.modules.report_writer.ChatOllama")
def test_write_report_contains_parsed_summary_and_actions(
    mock_llm_cls, mock_get_retriever
):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = _FAKE_DOCS
    mock_get_retriever.return_value = mock_retriever
    mock_llm_cls.return_value = MagicMock()

    with patch("src.modules.report_writer._prompt") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _FAKE_LLM_RESPONSE
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        result = write(ReportInput(
            system_description="A CV screening tool.",
            risk_classification=_make_risk_classification(),
            gap_analysis=_make_gap_analysis(),
        ))

    assert "High risk" in result.executive_summary
    assert len(result.recommended_actions) == 5
    assert "Article 9" in result.recommended_actions[0]


@patch("src.modules.report_writer.get_retriever")
@patch("src.modules.report_writer.ChatOllama")
def test_write_deduplicates_sources_across_modules(mock_llm_cls, mock_get_retriever):
    # rc, ga, and retriever all reference the same eu_ai_act.pdf page 9
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="x", metadata={"source": "eu_ai_act.pdf", "page": 9}),
    ]
    mock_get_retriever.return_value = mock_retriever
    mock_llm_cls.return_value = MagicMock()

    rc = _make_risk_classification(sources=[{"source": "eu_ai_act.pdf", "page": 9}])
    ga = _make_gap_analysis()  # ga.framework_gaps[0].sources has eu_ai_act.pdf page 9

    with patch("src.modules.report_writer._prompt") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _FAKE_LLM_RESPONSE
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        result = write(ReportInput(
            system_description="A CV screening tool.",
            risk_classification=rc,
            gap_analysis=ga,
        ))

    # eu_ai_act.pdf p.9 appears in rc, ga, and retriever — should appear once
    eu9_entries = [s for s in result.sources if s["source"] == "eu_ai_act.pdf" and s["page"] == 9]
    assert len(eu9_entries) == 1
