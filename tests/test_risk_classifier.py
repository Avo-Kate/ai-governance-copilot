"""
Tests for the risk classifier — all LLM and retriever calls are mocked
so the suite runs without a live Ollama instance.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.modules.risk_classifier import (
    RiskClassification,
    _parse_response,
    classify,
)


# ── _parse_response ────────────────────────────────────────────────────────────

def test_parse_response_all_fields():
    raw = (
        "TIER: High\n"
        "REASONING: CV screening tools fall under Annex III because they are used "
        "in employment decisions.\n"
        "KEY_FACTORS: employment context | automated decision-making | Annex III item 4"
    )
    tier, reasoning, key_factors = _parse_response(raw)
    assert tier == "High"
    assert "Annex III" in reasoning
    assert key_factors == ["employment context", "automated decision-making", "Annex III item 4"]


def test_parse_response_extra_whitespace():
    raw = "TIER:   Minimal  \nREASONING: Spam filter.\nKEY_FACTORS: low risk | no personal data"
    tier, reasoning, _ = _parse_response(raw)
    assert tier == "Minimal"
    assert reasoning == "Spam filter."


def test_parse_response_missing_fields():
    raw = "TIER: Limited"
    tier, reasoning, key_factors = _parse_response(raw)
    assert tier == "Limited"
    assert reasoning == ""
    assert key_factors == []


def test_parse_response_markdown_bold_labels():
    # llama3.2 frequently wraps the labels and/or values in markdown bold.
    raw = (
        "**TIER:** High\n"
        "**REASONING:** Employment decisions fall under Annex III.\n"
        "**KEY_FACTORS:** employment | automated decision"
    )
    tier, reasoning, key_factors = _parse_response(raw)
    assert tier == "High"
    assert "Annex III" in reasoning
    assert key_factors == ["employment", "automated decision"]


def test_parse_response_bold_value_only():
    raw = "TIER: **Unacceptable**\nREASONING: Prohibited by Article 5."
    tier, _, _ = _parse_response(raw)
    assert tier == "Unacceptable"


def test_parse_response_lowercase_labels():
    raw = "tier: Minimal\nreasoning: Spam filter.\nkey_factors: low risk"
    tier, reasoning, key_factors = _parse_response(raw)
    assert tier == "Minimal"
    assert reasoning == "Spam filter."
    assert key_factors == ["low risk"]


def test_parse_response_preamble_then_label():
    # The model adds a chatty preamble before the structured block.
    raw = (
        "Here is my classification:\n\n"
        "TIER: High\n"
        "REASONING: Used for promotion and firing decisions (Annex III)."
    )
    tier, _, _ = _parse_response(raw)
    assert tier == "High"


def test_parse_response_bulleted_labels():
    raw = "- TIER: Limited\n- REASONING: Chatbot with disclosure."
    tier, reasoning, _ = _parse_response(raw)
    assert tier == "Limited"
    assert reasoning == "Chatbot with disclosure."


def test_parse_response_fallback_scans_for_tier():
    # No structured TIER line at all — fall back to scanning the prose. This is
    # the regression guard for the empty-tier ValidationError the UI surfaced.
    raw = (
        "This system clearly falls into the High risk category because it makes "
        "employment decisions about promotion and termination."
    )
    tier, _, _ = _parse_response(raw)
    assert tier == "High"


def test_parse_response_fallback_picks_first_mentioned():
    raw = "It is not Unacceptable, but it is High risk under Annex III."
    tier, _, _ = _parse_response(raw)
    # Unacceptable appears first in the text, so the fallback returns it.
    assert tier == "Unacceptable"


def test_parse_response_truly_unparseable_returns_empty():
    # If nothing resembles a tier, we return "" and let the validator reject it
    # rather than fabricating a classification.
    raw = "I cannot determine this from the provided context."
    tier, _, _ = _parse_response(raw)
    assert tier == ""


# ── RiskClassification model ───────────────────────────────────────────────────

@pytest.mark.parametrize("tier", ["Unacceptable", "High", "Limited", "Minimal"])
def test_valid_tiers_accepted(tier):
    rc = RiskClassification(tier=tier, reasoning="test", key_factors=[], sources=[])
    assert rc.tier == tier


def test_invalid_tier_raises():
    with pytest.raises(Exception):
        RiskClassification(tier="Unknown", reasoning="x", key_factors=[], sources=[])


def test_capitalise_normalisation():
    # Some LLMs might return lowercase
    rc = RiskClassification(tier="High", reasoning="r", key_factors=[], sources=[])
    assert rc.tier == "High"


# ── classify() — integration with mocks ───────────────────────────────────────

_FAKE_DOCS = [
    Document(
        page_content="Article 5 prohibits AI systems used for real-time biometric surveillance.",
        metadata={"source": "eu_ai_act.pdf", "page": 12},
    ),
    Document(
        page_content="Annex III lists employment-related AI as high-risk.",
        metadata={"source": "eu_ai_act.pdf", "page": 45},
    ),
]

_UNACCEPTABLE_LLM_RESPONSE = (
    "TIER: Unacceptable\n"
    "REASONING: Real-time biometric surveillance in public spaces is explicitly prohibited "
    "by Article 5 of the EU AI Act.\n"
    "KEY_FACTORS: real-time biometric ID | public space | Article 5 prohibition"
)

_HIGH_LLM_RESPONSE = (
    "TIER: High\n"
    "REASONING: CV screening is an employment decision tool listed in Annex III item 4.\n"
    "KEY_FACTORS: employment | automated ranking | Annex III"
)


@patch("src.modules.risk_classifier.get_retriever")
@patch("src.modules.risk_classifier.ChatOllama")
def test_classify_unacceptable(mock_llm_cls, mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = _FAKE_DOCS
    mock_get_retriever.return_value = mock_retriever

    mock_llm = MagicMock()
    mock_llm_cls.return_value = mock_llm

    # Simulate the chain: _prompt | llm | StrOutputParser()
    # We patch at the chain level by making the final chain.invoke return the raw text.
    with patch("src.modules.risk_classifier._prompt") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _UNACCEPTABLE_LLM_RESPONSE
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        # chain = _prompt | llm | StrOutputParser() — stub the whole pipe
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        result = classify(
            "A city government system that identifies individuals in real time "
            "via CCTV for law enforcement."
        )

    assert result.tier == "Unacceptable"
    assert "Article 5" in result.reasoning
    assert len(result.key_factors) == 3
    # Sources are deduped — both docs come from same file/page combo? No, different pages.
    assert len(result.sources) == 2
    assert result.sources[0]["source"] == "eu_ai_act.pdf"


@patch("src.modules.risk_classifier.get_retriever")
@patch("src.modules.risk_classifier.ChatOllama")
def test_classify_high_risk(mock_llm_cls, mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = _FAKE_DOCS
    mock_get_retriever.return_value = mock_retriever

    mock_llm = MagicMock()
    mock_llm_cls.return_value = mock_llm

    with patch("src.modules.risk_classifier._prompt") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _HIGH_LLM_RESPONSE
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        result = classify("An automated CV screening tool used in hiring decisions.")

    assert result.tier == "High"
    assert len(result.key_factors) >= 1


@patch("src.modules.risk_classifier.get_retriever")
@patch("src.modules.risk_classifier.ChatOllama")
def test_classify_deduplicates_sources(mock_llm_cls, mock_get_retriever):
    # All docs from the same source+page — should collapse to one source entry.
    duplicate_docs = [
        Document(page_content="text", metadata={"source": "eu_ai_act.pdf", "page": 5}),
        Document(page_content="more", metadata={"source": "eu_ai_act.pdf", "page": 5}),
    ]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = duplicate_docs
    mock_get_retriever.return_value = mock_retriever

    mock_llm_cls.return_value = MagicMock()

    with patch("src.modules.risk_classifier._prompt") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = (
            "TIER: Minimal\nREASONING: Low risk.\nKEY_FACTORS: no personal data"
        )
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        result = classify("A spam filter.")

    assert len(result.sources) == 1
