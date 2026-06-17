from pathlib import Path
import sys
import json

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.monitor_agent import InvoiceMonitorAgent
from agents.rag_agents.indexing_agent import IndexingAgent
from agents.rag_agents.retrieval_agent import RetrievalAgent
from agents.rag_agents.generation_agent import GenerationAgent
from agents.rag_agents.reflection_agent import ReflectionAgent

INCOMING_DIR = PROJECT_ROOT / "data" / "incoming"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FEEDBACK_DIR = PROJECT_ROOT / "outputs" / "feedback"

INCOMING_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="AI Invoice Auditor",
    layout="wide"
)

st.title("AI Invoice Auditor")
st.write("Agentic AI-powered multilingual invoice validation system")

tab_upload, tab_scan, tab_process, tab_reports, tab_feedback, tab_rag = st.tabs(
    [
        "Upload Invoice",
        "Scan Incoming",
        "Run Workflow",
        "Reports",
        "Human Feedback",
        "RAG Q&A"
    ]
)

with tab_upload:
    st.header("Upload Invoice or Metadata")

    uploaded_file = st.file_uploader(
        "Upload invoice or .meta.json",
        type=["pdf", "docx", "png", "json"]
    )

    if uploaded_file is not None:
        save_path = INCOMING_DIR / uploaded_file.name

        with open(save_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        st.success(f"Uploaded to: {save_path}")

with tab_scan:
    st.header("Scan Incoming Folder")

    if st.button("Scan Files"):
        monitor = InvoiceMonitorAgent(
            incoming_dir=str(INCOMING_DIR),
            rules_path=str(PROJECT_ROOT / "configs" / "rules.yaml")
        )

        detected_files = monitor.scan_existing_files_for_ui()

        if not detected_files:
            st.warning("No invoice files found.")
        else:
            st.success(f"Detected {len(detected_files)} invoice file(s).")
            st.json(detected_files)

with tab_process:
    st.header("Run Full Workflow")

    st.write(
        "Flow: Monitor → Extractor → Translation → Validation → ERP Check → Reporting → RAG Indexing"
    )

    if st.button("Run Workflow"):
        monitor = InvoiceMonitorAgent(
            incoming_dir=str(INCOMING_DIR),
            rules_path=str(PROJECT_ROOT / "configs" / "rules.yaml")
        )

        results = monitor.run_once()

        for result in results:
            st.subheader("Workflow Result")
            st.json(result)

            if result.get("report_json_path"):
                st.success(f"JSON report: {result.get('report_json_path')}")

            if result.get("report_html_path"):
                st.success(f"HTML report: {result.get('report_html_path')}")

with tab_reports:
    st.header("Validation Reports")

    report_files = list(REPORT_DIR.glob("*_report.json"))

    if not report_files:
        st.info("No reports generated yet.")
    else:
        selected_report = st.selectbox(
            "Select report",
            report_files,
            format_func=lambda path: path.name
        )

        with open(selected_report, "r", encoding="utf-8") as file:
            report = json.load(file)

        st.json(report)

        st.subheader("Recommendation")
        st.write(report.get("recommendation"))

        st.subheader("Missing Fields")
        st.write(report.get("missing_fields"))

        st.subheader("Discrepancies")
        st.write(report.get("discrepancies"))

with tab_feedback:
    st.header("Human-in-the-Loop Feedback")

    invoice_no = st.text_input("Invoice No")
    correction_notes = st.text_area("Correction Notes")

    if st.button("Save Feedback"):
        if not invoice_no:
            st.warning("Please enter Invoice No.")
        else:
            feedback = {
                "invoice_no": invoice_no,
                "correction_notes": correction_notes,
                "status": "saved"
            }

            feedback_path = FEEDBACK_DIR / f"{invoice_no}_feedback.json"

            with open(feedback_path, "w", encoding="utf-8") as file:
                json.dump(feedback, file, indent=2)

            st.success(f"Feedback saved: {feedback_path}")

with tab_rag:
    st.header("RAG-based Invoice Q&A")

    if st.button("Index Reports"):
        indexer = IndexingAgent(
            reports_dir=str(REPORT_DIR),
            index_dir=str(PROJECT_ROOT / "outputs" / "faiss_index")
        )

        result = indexer.index_reports()
        st.json(result)

    question = st.text_input("Ask a question about invoices")

    if st.button("Ask") and question:
        retriever = RetrievalAgent(
            index_dir=str(PROJECT_ROOT / "outputs" / "faiss_index")
        )
        generator = GenerationAgent()
        reflector = ReflectionAgent()

        docs = retriever.retrieve(question)

        if not docs:
            st.warning("No RAG index found. Please index reports first.")
        else:
            generation = generator.answer(question, docs)
            evaluation = reflector.evaluate(
                question,
                generation["answer"],
                generation["context"]
            )

            st.subheader("Answer")
            st.write(generation["answer"])

            st.subheader("RAG Triad Evaluation")
            st.json(evaluation)

            st.subheader("Retrieved Context")
            for doc in docs:
                st.text(doc.page_content)
