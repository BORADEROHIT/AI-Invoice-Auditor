import json
from pathlib import Path
from datetime import datetime

from litellm import completion

class ReportingAgent:
    """
    Reporting Agent
    Persona: Compliance Reporter
    Uses LLM: Yes

    Flow:
        reporting_agent.py -> rag_agents/
    """

    def __init__(
        self,
        rules_path: str = "configs/rules.yaml",
        persona_path: str = "configs/persona_invoice_agent.yaml"
    ):
        self.rules_path = Path(rules_path)
        self.persona_path = Path(persona_path)
        self.reporting_config = self._load_reporting_config()
        self.model = self._load_model_from_persona()

        self.output_dir = Path(
            self.reporting_config.get("output_dir", "./outputs/reports")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_reporting_config(self) -> dict:
        config = {
            "include_translation_confidence": True,
            "include_discrepancy_summary": True,
            "report_format": "HTML",
            "output_dir": "./outputs/reports"
        }

        if not self.rules_path.exists():
            return config

        inside_reporting = False

        with open(self.rules_path, "r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()

                if not stripped or stripped.startswith("#"):
                    continue

                if stripped == "reporting:":
                    inside_reporting = True
                    continue

                if inside_reporting and not line.startswith(" "):
                    break

                if inside_reporting and ":" in stripped:
                    key, value = stripped.split(":", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")

                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False

                    config[key] = value

        return config

    def _load_model_from_persona(self) -> str:
        default_model = "bedrock/amazon.nova-lite-v1:0"

        if not self.persona_path.exists():
            return default_model

        inside_agent = False

        with open(self.persona_path, "r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()

                if stripped == "reporting_agent:":
                    inside_agent = True
                    continue

                if inside_agent and stripped.startswith("model:"):
                    _, value = stripped.split(":", 1)
                    return value.strip().strip('"').strip("'")

                if inside_agent and not line.startswith("  "):
                    break

        return default_model

    def generate_report(self, state: dict) -> dict:
        invoice_data = state.get("invoice_data", {})

        invoice_no = invoice_data.get("invoice_no") 
        
        if not invoice_no:
            file_name = Path(state.get("file_name", "unknown_invoice")).stem
            invoice_no = f"unknown_invoice_{file_name}"

        missing_fields = state.get("missing_fields", [])
        all_discrepancies = state.get("all_discrepancies", [])

        recommendation = self._decide_recommendation(
            missing_fields,
            all_discrepancies,
            state.get("translation_confidence", 0)
        )

        summary = self._generate_llm_summary(state, recommendation)

        report = {
            "invoice_no": invoice_no,
            "invoice_date": invoice_data.get("invoice_date"),
            "vendor_id": invoice_data.get("vendor_id"),
            "vendor_name": invoice_data.get("vendor_name"),
            "po_number": invoice_data.get("po_number"),
            "currency": invoice_data.get("currency"),
            "total_amount": invoice_data.get("total_amount"),
            "email_metadata": state.get("email_metadata", {}),
            "source_language": state.get("source_language"),
            "translation_confidence": state.get("translation_confidence"),
            "missing_fields": missing_fields,
            "discrepancies": all_discrepancies,
            "recommendation": recommendation,
            "summary": summary,
            "audit_trail": state.get("audit_trail", []),
            "generated_at": datetime.utcnow().isoformat()
        }

        json_path = self.output_dir / f"{invoice_no}_report.json"
        html_path = self.output_dir / f"{invoice_no}_report.html"

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        with open(html_path, "w", encoding="utf-8") as file:
            file.write(self._build_html(report))

        state["report"] = report
        state["report_json_path"] = str(json_path)
        state["report_html_path"] = str(html_path)
        state["status"] = "reported"

        try:
            from agents.rag_agents.indexing_agent import IndexingAgent

            indexer = IndexingAgent(
                reports_dir=str(self.output_dir),
                index_dir="./outputs/faiss_index"
            )
            state["rag_indexing_result"] = indexer.index_reports()
        except Exception as error:
            state["rag_indexing_result"] = {
                "status": "indexing_failed",
                "error": str(error)
            }

        return state

    def _decide_recommendation(
        self,
        missing_fields: list,
        discrepancies: list,
        translation_confidence: float
    ) -> str:
        if any(item.get("severity") == "high" for item in discrepancies):
            return "Reject"

        if missing_fields:
            return "Manual Review"

        if discrepancies:
            return "Manual Review"

        if translation_confidence and translation_confidence < 0.95:
            return "Manual Review"

        return "Approve"

    def _generate_llm_summary(self, state: dict, recommendation: str) -> str:
        prompt = f"""
    You are the Reporting Agent for AI Invoice Auditor.

    Create a concise audit summary.

    Invoice data:
    {json.dumps(state.get("invoice_data", {}), indent=2)}

    Missing fields:
    {json.dumps(state.get("missing_fields", []), indent=2)}

    Discrepancies:
    {json.dumps(state.get("all_discrepancies", []), indent=2)}

    Recommendation:
    {recommendation}
    """

        try:
            response = completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a compliance invoice reporting expert."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response["choices"][0]["message"]["content"]

        except Exception:
            return f"Invoice processed with recommendation: {recommendation}."

    def _build_html(self, report: dict) -> str:
        return f"""
    AI Invoice Auditor Report
    Invoice: {report.get("invoice_no")}
    <p><strong>Vendor ID:</strong> {report.get("vendor_id")}</p>
    <p><strong>Vendor Name:</strong> {report.get("vendor_name")}</p>
    <p><strong>PO Number:</strong> {report.get("po_number")}</p>
    <p><strong>Currency:</strong> {report.get("currency")}</p>
    <p><strong>Total Amount:</strong> {report.get("total_amount")}</p>
    <p><strong>Translation Confidence:</strong> {report.get("translation_confidence")}</p>
    <p><strong>Recommendation:</strong> {report.get("recommendation")}</p>

    <h3>Email Metadata</h3>
    <pre>{json.dumps(report.get("email_metadata"), indent=2)}</pre>

    <h3>Summary</h3>
    <pre>{report.get("summary")}</pre>

    <h3>Missing Fields</h3>
    <pre>{json.dumps(report.get("missing_fields"), indent=2)}</pre>

    <h3>Discrepancies</h3>
    <pre>{json.dumps(report.get("discrepancies"), indent=2)}</pre>

    <h3>Audit Trail</h3>
    <pre>{json.dumps(report.get("audit_trail"), indent=2)}</pre>
    """

