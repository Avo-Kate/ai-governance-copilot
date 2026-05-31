"""
Tests for the gap analyzer — all LLM and retriever calls are mocked
so the suite runs without a live Ollama instance.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.modules.gap_analyzer import (
    ALL_FRAMEWORK_NAMES,
    FrameworkGap,
    GapAnalysis,
    _analyse_framework,
    _dedup_sources,
    _parse_response,
    analyze,
    FRAMEWORKS,
)


# ── _parse_response ────────────────────────────────────────────────────────────

def test_parse_response_full():
    raw = (
        "REQUIREMENTS: Risk management system | Technical documentation | Human oversight\n"
        "MET: Technical documentation\n"
        "GAPS: Risk management system | Human oversight"
    )
    reqs, met, gaps = _parse_response(raw)
    assert reqs == ["Risk management system", "Technical documentation", "Human oversight"]
    assert met == ["Technical documentation"]
    assert gaps == ["Risk management system", "Human oversight"]


def test_parse_response_none_sentinels():
    raw = (
        "REQUIREMENTS: Transparency disclosure\n"
        "MET: Transparency disclosure\n"
        "GAPS: NONE"
    )
    reqs, met, gaps = _parse_response(raw)
    assert gaps == []
    assert met == ["Transparency disclosure"]


def test_parse_response_met_none():
    raw = (
        "REQUIREMENTS: Logging | Conformity assessment\n"
        "MET: NONE\n"
        "GAPS: Logging | Conformity assessment"
    )
    _, met, gaps = _parse_response(raw)
    assert met == []
    assert len(gaps) == 2


def test_parse_response_missing_section():
    raw = "REQUIREMENTS: Something\nGAPS: Something"
    reqs, met, gaps = _parse_response(raw)
    assert met == []  # MET section absent → empty list
    assert reqs == ["Something"]


def test_parse_response_extra_whitespace():
    raw = "REQUIREMENTS:  A | B  \nMET: A  \nGAPS:  B "
    reqs, met, gaps = _parse_response(raw)
    assert reqs == ["A", "B"]
    assert met == ["A"]
    assert gaps == ["B"]


# ── _dedup_sources ─────────────────────────────────────────────────────────────

def test_dedup_sources_removes_duplicates():
    docs = [
        Document(page_content="a", metadata={"source": "eu.pdf", "page": 5}),
        Document(page_content="b", metadata={"source": "eu.pdf", "page": 5}),
        Document(page_content="c", metadata={"source": "eu.pdf", "page": 9}),
    ]
    result = _dedup_sources(docs)
    assert len(result) == 2
    assert result[0] == {"source": "eu.pdf", "page": 5}
    assert result[1] == {"source": "eu.pdf", "page": 9}


def test_dedup_sources_preserves_order():
    docs = [
        Document(page_content="x", metadata={"source": "nist.pdf", "page": 1}),
        Document(page_content="y", metadata={"source": "eu.pdf", "page": 2}),
    ]
    result = _dedup_sources(docs)
    assert result[0]["source"] == "nist.pdf"
    assert result[1]["source"] == "eu.pdf"


# ── FrameworkGap model ─────────────────────────────────────────────────────────

def test_framework_gap_model():
    fg = FrameworkGap(
        framework="EU AI Act",
        requirements=["Risk management", "Logging"],
        met=["Logging"],
        gaps=["Risk management"],
        sources=[{"source": "eu.pdf", "page": 9}],
    )
    assert fg.framework == "EU AI Act"
    assert len(fg.gaps) == 1


# ── GapAnalysis model ──────────────────────────────────────────────────────────

def test_gap_analysis_total_gaps():
    ga = GapAnalysis(
        system_description="test system",
        framework_gaps=[
            FrameworkGap(framework="EU AI Act", requirements=[], met=[], gaps=["a", "b"], sources=[]),
            FrameworkGap(framework="NIST AI RMF", requirements=[], met=[], gaps=["c"], sources=[]),
        ],
    )
    assert ga.total_gaps == 3
    assert ga.frameworks_analysed == ["EU AI Act", "NIST AI RMF"]


def test_gap_analysis_zero_gaps():
    ga = GapAnalysis(
        system_description="compliant system",
        framework_gaps=[
            FrameworkGap(framework="EU AI Act", requirements=["Logging"], met=["Logging"], gaps=[], sources=[]),
        ],
    )
    assert ga.total_gaps == 0


# ── _analyse_framework ─────────────────────────────────────────────────────────

_FAKE_DOCS = [
    Document(page_content="Article 9: risk management system required.", metadata={"source": "eu.pdf", "page": 9}),
    Document(page_content="Article 12: automatic logging of events.", metadata={"source": "eu.pdf", "page": 12}),
]

_FAKE_LLM_RESPONSE = (
    "REQUIREMENTS: Risk management system | Logging of decisions | Human oversight\n"
    "MET: NONE\n"
    "GAPS: Risk management system | Logging of decisions | Human oversight"
)


def test_analyse_framework_returns_framework_gap():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = _FAKE_DOCS

    mock_llm = MagicMock()

    with patch("src.modules.gap_analyzer._prompt") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _FAKE_LLM_RESPONSE
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        fg = _analyse_framework(
            "A CV screening tool with no human review.",
            FRAMEWORKS["EU AI Act"],
            mock_retriever,
            mock_llm,
        )

    assert fg.framework == "EU AI Act"
    assert len(fg.gaps) == 3
    assert fg.met == []
    assert len(fg.sources) == 2  # two distinct source+page combos


def test_analyse_framework_deduplicates_sources():
    duplicate_docs = [
        Document(page_content="a", metadata={"source": "eu.pdf", "page": 9}),
        Document(page_content="b", metadata={"source": "eu.pdf", "page": 9}),
    ]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = duplicate_docs

    with patch("src.modules.gap_analyzer._prompt") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "REQUIREMENTS: X\nMET: NONE\nGAPS: X"
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        fg = _analyse_framework("some system", FRAMEWORKS["NIST AI RMF"], mock_retriever, MagicMock())

    assert len(fg.sources) == 1


# ── analyze() — public API ─────────────────────────────────────────────────────

@patch("src.modules.gap_analyzer.get_retriever")
@patch("src.modules.gap_analyzer.ChatOllama")
def test_analyze_all_frameworks(mock_llm_cls, mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = _FAKE_DOCS
    mock_get_retriever.return_value = mock_retriever
    mock_llm_cls.return_value = MagicMock()

    with patch("src.modules.gap_analyzer._prompt") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _FAKE_LLM_RESPONSE
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        result = analyze("A CV screening tool.")

    assert len(result.framework_gaps) == 3
    assert result.frameworks_analysed == ALL_FRAMEWORK_NAMES
    assert result.total_gaps == 9  # 3 gaps × 3 frameworks


@patch("src.modules.gap_analyzer.get_retriever")
@patch("src.modules.gap_analyzer.ChatOllama")
def test_analyze_single_framework(mock_llm_cls, mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = _FAKE_DOCS
    mock_get_retriever.return_value = mock_retriever
    mock_llm_cls.return_value = MagicMock()

    with patch("src.modules.gap_analyzer._prompt") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _FAKE_LLM_RESPONSE
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        result = analyze("A CV screening tool.", frameworks=["EU AI Act"])

    assert result.frameworks_analysed == ["EU AI Act"]
    assert len(result.framework_gaps) == 1


def test_analyze_unknown_framework_raises():
    with pytest.raises(ValueError, match="Unknown framework"):
        analyze("some system", frameworks=["Made Up Framework"])


@patch("src.modules.gap_analyzer.get_retriever")
@patch("src.modules.gap_analyzer.ChatOllama")
def test_analyze_preserves_system_description(mock_llm_cls, mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_get_retriever.return_value = mock_retriever
    mock_llm_cls.return_value = MagicMock()

    with patch("src.modules.gap_analyzer._prompt") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "REQUIREMENTS: X\nMET: X\nGAPS: NONE"
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        result = analyze("My specific system.", frameworks=["NIST AI RMF"])

    assert result.system_description == "My specific system."
