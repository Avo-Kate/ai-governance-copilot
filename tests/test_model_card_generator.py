"""
Tests for the model card generator — all LLM and retriever calls are mocked
so the suite runs without a live Ollama instance.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.modules.model_card_generator import (
    ModelCard,
    ModelCardInput,
    _dedup_sources,
    _parse_response,
    generate,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _minimal_input(**kwargs) -> ModelCardInput:
    defaults = dict(
        model_name="TestModel",
        model_type="binary classifier",
        intended_use="Screen job applicants",
        training_data_description="Historical hiring data from 2019-2023",
    )
    defaults.update(kwargs)
    return ModelCardInput(**defaults)


_FAKE_LLM_RESPONSE = (
    "ETHICAL_CONSIDERATIONS: The system processes personal data and must comply "
    "with Article 10 of the EU AI Act on data governance. Fairness audits are required.\n"
    "LIMITATIONS_EXPANDED: Article 15 requires accuracy and robustness testing. "
    "The model may underperform on underrepresented groups.\n"
    "CAVEATS: Do not use without human review. Results should not be treated as final.\n"
    "RECOMMENDATIONS: Conduct a conformity assessment per Article 43. Register the "
    "system in the EU AI Act database before deployment."
)

_FAKE_DOCS = [
    Document(page_content="Article 10: data governance requirements.", metadata={"source": "eu_ai_act.pdf", "page": 10}),
    Document(page_content="Article 15: accuracy and robustness.", metadata={"source": "eu_ai_act.pdf", "page": 15}),
    Document(page_content="GOVERN function: establish AI policies.", metadata={"source": "nist_rmf.pdf", "page": 8}),
]


# ── _parse_response ────────────────────────────────────────────────────────────

def test_parse_response_all_four_sections():
    ethical, limitations, caveats, recommendations = _parse_response(_FAKE_LLM_RESPONSE)
    assert "Article 10" in ethical
    assert "Article 15" in limitations
    assert "human review" in caveats
    assert "Article 43" in recommendations


def test_parse_response_multi_line_section():
    raw = (
        "ETHICAL_CONSIDERATIONS: Line one.\n"
        "Still ethical considerations.\n"
        "LIMITATIONS_EXPANDED: Limitations here.\n"
        "CAVEATS: Be careful.\n"
        "RECOMMENDATIONS: Do this."
    )
    ethical, limitations, caveats, recommendations = _parse_response(raw)
    assert "Line one." in ethical
    assert "Still ethical considerations." in ethical
    assert limitations == "Limitations here."


def test_parse_response_missing_sections_return_empty():
    raw = "ETHICAL_CONSIDERATIONS: Only this section."
    ethical, limitations, caveats, recommendations = _parse_response(raw)
    assert ethical == "Only this section."
    assert limitations == ""
    assert caveats == ""
    assert recommendations == ""


def test_parse_response_empty_string():
    ethical, limitations, caveats, recommendations = _parse_response("")
    assert ethical == limitations == caveats == recommendations == ""


# ── _dedup_sources ─────────────────────────────────────────────────────────────

def test_dedup_sources_collapses_duplicates():
    docs = [
        Document(page_content="a", metadata={"source": "eu.pdf", "page": 5}),
        Document(page_content="b", metadata={"source": "eu.pdf", "page": 5}),
        Document(page_content="c", metadata={"source": "nist.pdf", "page": 3}),
    ]
    result = _dedup_sources(docs)
    assert len(result) == 2


def test_dedup_sources_empty():
    assert _dedup_sources([]) == []


# ── ModelCardInput ─────────────────────────────────────────────────────────────

def test_model_card_input_required_fields():
    inp = _minimal_input()
    assert inp.model_name == "TestModel"
    assert inp.organization == "Not specified"  # default


def test_model_card_input_optional_override():
    inp = _minimal_input(organization="ACME Corp", version="3.0.0")
    assert inp.organization == "ACME Corp"
    assert inp.version == "3.0.0"


def test_model_card_input_missing_required_raises():
    with pytest.raises(Exception):
        ModelCardInput(model_name="X")  # missing model_type, intended_use, training_data


# ── ModelCard.to_markdown() ────────────────────────────────────────────────────

def _make_card(**overrides) -> ModelCard:
    defaults = dict(
        input=_minimal_input(),
        generated_date="2026-06-03",
        ethical_considerations="Fairness and transparency required.",
        limitations_expanded="May underperform on edge cases.",
        caveats="Do not use without human review.",
        recommendations="Conduct conformity assessment.",
        sources=[{"source": "eu_ai_act.pdf", "page": 10}],
    )
    defaults.update(overrides)
    return ModelCard(**defaults)


def test_to_markdown_contains_model_name():
    card = _make_card()
    md = card.to_markdown()
    assert "TestModel" in md


def test_to_markdown_contains_all_sections():
    card = _make_card()
    md = card.to_markdown()
    for heading in [
        "Model Details", "Intended Use", "Training Data",
        "Evaluation", "Limitations", "Ethical Considerations",
        "Caveats and Recommendations", "Regulatory Grounding",
    ]:
        assert heading in md, f"Missing section: {heading}"


def test_to_markdown_ethical_considerations_present():
    card = _make_card(ethical_considerations="EU AI Act Article 13 applies.")
    md = card.to_markdown()
    assert "EU AI Act Article 13 applies." in md


def test_to_markdown_known_limitations_shown_when_provided():
    inp = _minimal_input(known_limitations="Biased on non-English CVs.")
    card = _make_card(input=inp)
    md = card.to_markdown()
    assert "Biased on non-English CVs." in md


def test_to_markdown_sources_listed():
    card = _make_card(sources=[
        {"source": "eu_ai_act.pdf", "page": 10},
        {"source": "nist_rmf.pdf", "page": 5},
    ])
    md = card.to_markdown()
    assert "eu_ai_act.pdf" in md
    assert "nist_rmf.pdf" in md


def test_to_markdown_no_sources_shows_not_specified():
    card = _make_card(sources=[])
    md = card.to_markdown()
    assert "Not specified" in md


def test_to_markdown_is_string():
    md = _make_card().to_markdown()
    assert isinstance(md, str)
    assert len(md) > 100


# ── generate() — integration with mocks ───────────────────────────────────────

@patch("src.modules.model_card_generator.get_retriever")
@patch("src.modules.model_card_generator.ChatOllama")
def test_generate_returns_model_card(mock_llm_cls, mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = _FAKE_DOCS
    mock_get_retriever.return_value = mock_retriever
    mock_llm_cls.return_value = MagicMock()

    with patch("src.modules.model_card_generator._prompt") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _FAKE_LLM_RESPONSE
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        result = generate(_minimal_input())

    assert isinstance(result, ModelCard)
    assert result.input.model_name == "TestModel"
    assert "Article 10" in result.ethical_considerations
    assert len(result.sources) == 3  # three distinct source+page combos


@patch("src.modules.model_card_generator.get_retriever")
@patch("src.modules.model_card_generator.ChatOllama")
def test_generate_deduplicates_sources(mock_llm_cls, mock_get_retriever):
    duplicate_docs = [
        Document(page_content="a", metadata={"source": "eu.pdf", "page": 5}),
        Document(page_content="b", metadata={"source": "eu.pdf", "page": 5}),
    ]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = duplicate_docs
    mock_get_retriever.return_value = mock_retriever
    mock_llm_cls.return_value = MagicMock()

    with patch("src.modules.model_card_generator._prompt") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _FAKE_LLM_RESPONSE
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        result = generate(_minimal_input())

    assert len(result.sources) == 1


@patch("src.modules.model_card_generator.get_retriever")
@patch("src.modules.model_card_generator.ChatOllama")
def test_generate_markdown_roundtrip(mock_llm_cls, mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = _FAKE_DOCS
    mock_get_retriever.return_value = mock_retriever
    mock_llm_cls.return_value = MagicMock()

    with patch("src.modules.model_card_generator._prompt") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = _FAKE_LLM_RESPONSE
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        result = generate(_minimal_input(
            model_name="ScreenBot",
            organization="TestOrg",
        ))

    md = result.to_markdown()
    assert "ScreenBot" in md
    assert "TestOrg" in md
    assert "Ethical Considerations" in md
