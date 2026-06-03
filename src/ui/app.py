"""
app.py — Streamlit frontend for the AI Governance Copilot.

A single-file multi-page app (sidebar navigation) wiring all four modules:
  • Risk Classifier      → src.modules.risk_classifier.classify
  • Gap Analyzer         → src.modules.gap_analyzer.analyze
  • Model Card Generator → src.modules.model_card_generator.generate
  • Report Writer        → src.modules.report_writer.write

Run with:  uv run streamlit run src/ui/app.py

Results are kept in st.session_state so they survive Streamlit's re-runs and
can be handed between pages — the Report Writer reuses an already-computed
classification and gap analysis instead of paying for the LLM calls twice.
"""

from __future__ import annotations

import streamlit as st

from src.config import OLLAMA_MODEL
from src.ui import components as ui
from src.modules.risk_classifier import classify
from src.modules.gap_analyzer import analyze, ALL_FRAMEWORK_NAMES
from src.modules.model_card_generator import ModelCardInput, generate
from src.modules.report_writer import ReportInput, write


# ── Page config & session state ──────────────────────────────────────────────

st.set_page_config(
    page_title="AI Governance Copilot",
    page_icon="🛡️",
    layout="wide",
)

# Persistent slots for cross-page reuse. Each *_desc slot records the system
# description a result was computed for, so we only offer to reuse a result
# when it actually matches what the user is now assessing.
_DEFAULTS = {
    "classification": None,
    "classification_desc": "",
    "gap_analysis": None,
    "gap_analysis_desc": "",
    "model_card": None,
    "report": None,
}
for key, default in _DEFAULTS.items():
    st.session_state.setdefault(key, default)


# ── Sidebar ──────────────────────────────────────────────────────────────────

PAGES = [
    "🏠 Home",
    "⚖️ Risk Classifier",
    "🔍 Gap Analyzer",
    "📇 Model Card Generator",
    "📄 Report Writer",
]

with st.sidebar:
    st.title("🛡️ Governance Copilot")
    st.caption("Fully local · no data leaves your machine")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")

    st.divider()
    st.caption(f"**Model:** `{OLLAMA_MODEL}`")
    if ui.vectorstore_ready():
        st.success("Document index ready", icon="✅")
    else:
        st.error("No document index", icon="⚠️")


# ── Home ─────────────────────────────────────────────────────────────────────

def page_home() -> None:
    st.title("AI Governance Copilot")
    st.markdown(
        "A local assistant for AI governance work, grounded in the **EU AI Act**, "
        "**NIST AI RMF**, and **ISO/IEC 42001**. Everything runs on-device via Ollama."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⚖️ Risk Classifier")
        st.write("Classify an AI system into its EU AI Act risk tier with citations.")
        st.subheader("🔍 Gap Analyzer")
        st.write("Map a system against all three frameworks and surface compliance gaps.")
    with c2:
        st.subheader("📇 Model Card Generator")
        st.write("Produce a HuggingFace-format model card with regulatory grounding.")
        st.subheader("📄 Report Writer")
        st.write("Combine the above into a formal AI risk assessment report.")

    st.divider()
    if not ui.vectorstore_ready():
        ui.require_vectorstore()
    else:
        st.info(
            "Pick a tool from the sidebar to get started. Tip: run the **Risk "
            "Classifier** and **Gap Analyzer** first — the **Report Writer** can "
            "then reuse those results without re-running the analysis.",
            icon="👈",
        )


# ── Risk Classifier ──────────────────────────────────────────────────────────

def page_risk_classifier() -> None:
    st.title("⚖️ Risk Classifier")
    st.write("Describe an AI system to classify it under the EU AI Act risk tiers.")

    if not ui.require_vectorstore():
        return

    with st.form("risk_form"):
        description = st.text_area(
            "System description",
            height=180,
            placeholder=(
                "Describe what the AI system does, who uses it, where it is deployed, "
                "and what data it processes…"
            ),
            value=st.session_state.classification_desc,
        )
        submitted = st.form_submit_button("Classify", type="primary")

    if submitted:
        if not description.strip():
            st.error("Please enter a system description.")
            return
        with st.spinner("Classifying against the EU AI Act…"):
            result = classify(description)
        st.session_state.classification = result
        st.session_state.classification_desc = description

    if st.session_state.classification is not None:
        st.divider()
        ui.render_classification(st.session_state.classification)


# ── Gap Analyzer ─────────────────────────────────────────────────────────────

def page_gap_analyzer() -> None:
    st.title("🔍 Gap Analyzer")
    st.write("Map an AI system against governance frameworks to find compliance gaps.")

    if not ui.require_vectorstore():
        return

    with st.form("gap_form"):
        description = st.text_area(
            "System description",
            height=180,
            placeholder="Describe the AI system and its deployment context…",
            value=st.session_state.gap_analysis_desc,
        )
        frameworks = st.multiselect(
            "Frameworks to assess",
            options=ALL_FRAMEWORK_NAMES,
            default=ALL_FRAMEWORK_NAMES,
        )
        submitted = st.form_submit_button("Analyze gaps", type="primary")

    if submitted:
        if not description.strip():
            st.error("Please enter a system description.")
            return
        if not frameworks:
            st.error("Select at least one framework.")
            return
        with st.spinner(f"Analyzing against {len(frameworks)} framework(s)…"):
            result = analyze(description, frameworks=frameworks)
        st.session_state.gap_analysis = result
        st.session_state.gap_analysis_desc = description

    if st.session_state.gap_analysis is not None:
        st.divider()
        ui.render_gap_analysis(st.session_state.gap_analysis)


# ── Model Card Generator ─────────────────────────────────────────────────────

def page_model_card() -> None:
    st.title("📇 Model Card Generator")
    st.write(
        "Fill in what you know about the model. Governance sections are written "
        "by the LLM and grounded in the regulatory documents; everything else "
        "comes straight from your input."
    )

    if not ui.require_vectorstore():
        return

    with st.form("model_card_form"):
        st.subheader("Required")
        model_name = st.text_input("Model name", placeholder="e.g. HireScore v2")
        model_type = st.text_input(
            "Model type", placeholder="e.g. binary classifier, LLM, regression model"
        )
        intended_use = st.text_area(
            "Intended use", placeholder="What problem does it solve, and in what context?"
        )
        training_data = st.text_area(
            "Training data description", placeholder="What data was the model trained on?"
        )

        with st.expander("Optional details"):
            c1, c2 = st.columns(2)
            with c1:
                organization = st.text_input("Organization", value="Not specified")
                version = st.text_input("Version", value="1.0.0")
                license_type = st.text_input("License", value="Not specified")
                contact = st.text_input("Contact", value="Not specified")
                primary_users = st.text_input("Primary users", value="Not specified")
            with c2:
                deployment_context = st.text_input("Deployment context", value="Not specified")
                out_of_scope_uses = st.text_input("Out-of-scope uses", value="Not specified")
                metrics = st.text_input("Metrics", value="Not specified")
                evaluation_data = st.text_input("Evaluation data", value="Not specified")
            known_limitations = st.text_area("Known limitations", value="Not specified")

        submitted = st.form_submit_button("Generate model card", type="primary")

    if submitted:
        missing = [
            label
            for label, val in [
                ("Model name", model_name),
                ("Model type", model_type),
                ("Intended use", intended_use),
                ("Training data description", training_data),
            ]
            if not val.strip()
        ]
        if missing:
            st.error(f"Please fill in the required field(s): {', '.join(missing)}.")
            return

        card_input = ModelCardInput(
            model_name=model_name,
            model_type=model_type,
            intended_use=intended_use,
            training_data_description=training_data,
            organization=organization,
            version=version,
            license_type=license_type,
            contact=contact,
            primary_users=primary_users,
            deployment_context=deployment_context,
            out_of_scope_uses=out_of_scope_uses,
            metrics=metrics,
            evaluation_data=evaluation_data,
            known_limitations=known_limitations,
        )
        with st.spinner("Writing governance sections from the regulatory docs…"):
            st.session_state.model_card = generate(card_input)

    if st.session_state.model_card is not None:
        st.divider()
        ui.render_model_card(st.session_state.model_card)


# ── Report Writer ────────────────────────────────────────────────────────────

def page_report_writer() -> None:
    st.title("📄 Report Writer")
    st.write(
        "Generate a formal AI risk assessment report combining the risk "
        "classification and gap analysis with an executive summary and "
        "prioritised recommended actions."
    )

    if not ui.require_vectorstore():
        return

    # Offer to reuse cached results — but only the ones computed for the *same*
    # description the user is about to assess, so we never staple a mismatched
    # classification onto a different system.
    rc = st.session_state.classification
    ga = st.session_state.gap_analysis

    with st.form("report_form"):
        c1, c2 = st.columns(2)
        with c1:
            system_name = st.text_input("System name", value="AI System")
            organization = st.text_input("Organization", value="Not specified")
        with c2:
            assessor = st.text_input("Assessor", value="Not specified")

        description = st.text_area(
            "System description",
            height=180,
            placeholder="Describe the AI system and its deployment context…",
            value=st.session_state.classification_desc or st.session_state.gap_analysis_desc,
        )

        reuse = st.checkbox(
            "Reuse existing Risk Classifier & Gap Analyzer results when they "
            "match this description (skips redundant LLM calls)",
            value=True,
        )
        submitted = st.form_submit_button("Write report", type="primary")

    if submitted:
        if not description.strip():
            st.error("Please enter a system description.")
            return

        # Only reuse a cached result if it was computed for this exact description.
        reuse_rc = rc if (reuse and st.session_state.classification_desc == description) else None
        reuse_ga = ga if (reuse and st.session_state.gap_analysis_desc == description) else None

        notices = []
        if reuse_rc:
            notices.append("risk classification")
        if reuse_ga:
            notices.append("gap analysis")
        if notices:
            st.info(f"Reusing cached {', '.join(notices)}.", icon="♻️")

        report_input = ReportInput(
            system_description=description,
            system_name=system_name,
            organization=organization,
            assessor=assessor,
            risk_classification=reuse_rc,
            gap_analysis=reuse_ga,
        )
        with st.spinner("Drafting the assessment report…"):
            report = write(report_input)

        # Cache the freshly computed sub-results too, so other pages benefit.
        st.session_state.report = report
        st.session_state.classification = report.risk_classification
        st.session_state.classification_desc = description
        st.session_state.gap_analysis = report.gap_analysis
        st.session_state.gap_analysis_desc = description

    if st.session_state.report is not None:
        st.divider()
        ui.render_report(st.session_state.report)


# ── Router ───────────────────────────────────────────────────────────────────

ROUTES = {
    "🏠 Home": page_home,
    "⚖️ Risk Classifier": page_risk_classifier,
    "🔍 Gap Analyzer": page_gap_analyzer,
    "📇 Model Card Generator": page_model_card,
    "📄 Report Writer": page_report_writer,
}

ROUTES[page]()
