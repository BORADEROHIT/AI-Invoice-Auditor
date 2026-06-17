import json
from datetime import datetime
from pathlib import Path

from litellm import completion

class TranslationAgent:
    """
    Translation Agent
    Persona: Multilingual Translator
    Uses LLM: Yes

    Flow:
        translation_agent.py -> validation_agent.py
    """

    def __init__(
        self,
        persona_path: str = "configs/persona_invoice_agent.yaml"
    ):
        self.persona_path = Path(persona_path)
        self.model = self._load_model_from_persona()

    def _load_model_from_persona(self) -> str:
        default_model = "bedrock/cohere.command-r-plus-v1:0"

        if not self.persona_path.exists():
            return default_model

        inside_translation_agent = False

        with open(self.persona_path, "r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()

                if not stripped or stripped.startswith("#"):
                    continue

                if stripped == "translation_agent:":
                    inside_translation_agent = True
                    continue

                if inside_translation_agent and stripped.startswith("model:"):
                    _, value = stripped.split(":", 1)
                    return value.strip().strip('"').strip("'")

                if inside_translation_agent and not line.startswith("  "):
                    break

        return default_model

    def translate(self, extraction_state: dict) -> dict:
        raw_text = extraction_state.get("raw_text", "")

        if not raw_text:
            extraction_state["translation_status"] = "failed"
            extraction_state["translation_error"] = "No raw text available for translation."
            return self._send_to_validation(extraction_state)

        prompt = self._build_prompt(raw_text, extraction_state.get("email_metadata", {}))

        try:
            response = completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a multilingual financial invoice translation expert."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            llm_output = response["choices"][0]["message"]["content"]
            parsed_output = self._parse_json(llm_output)

            extraction_state["source_language"] = parsed_output.get(
                "source_language",
                extraction_state.get("email_metadata", {}).get("language", "unknown")
            )
            extraction_state["translated_text"] = parsed_output.get("translated_text", raw_text)
            extraction_state["translation_confidence"] = parsed_output.get("translation_confidence", 0.80)
            extraction_state["translation_status"] = "success"

            extraction_state.setdefault("audit_trail", []).append(
                {
                    "agent": "Translation Agent",
                    "persona": "Multilingual Translator",
                    "model": self.model,
                    "status": "success",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Invoice text translated into English."
                }
            )

        except Exception as error:
            extraction_state["source_language"] = extraction_state.get("email_metadata", {}).get("language", "unknown")
            extraction_state["translated_text"] = raw_text
            extraction_state["translation_confidence"] = 0.50
            extraction_state["translation_status"] = "fallback"
            extraction_state["translation_error"] = str(error)

            extraction_state.setdefault("audit_trail", []).append(
                {
                    "agent": "Translation Agent",
                    "persona": "Multilingual Translator",
                    "model": self.model,
                    "status": "fallback",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Translation failed. Raw text passed forward as fallback.",
                    "error": str(error)
                }
            )

        return self._send_to_validation(extraction_state)

    def _send_to_validation(self, extraction_state: dict) -> dict:
        from agents.validation_agent import ValidationAgent

        validation_agent = ValidationAgent()
        return validation_agent.validate(extraction_state)

    def _build_prompt(self, raw_text: str, email_metadata: dict) -> str:
        return f"""
    You are the Translation Agent for AI Invoice Auditor.

    Email metadata:
    {json.dumps(email_metadata, indent=2)}

    Tasks:

    Detect source language.
    Translate non-English invoice content into English.
    Preserve invoice_no, invoice_date, vendor_id, po_number, item_code, currency, qty, unit_price, total, and total_amount.
    Return translation confidence between 0 and 1.
    Return ONLY valid JSON:

    {{
    "source_language": "detected language",
    "translated_text": "English translated invoice text",
    "translation_confidence": 0.95
    }}

    Invoice text:
    {raw_text}
    """

    def _parse_json(self, llm_output: str) -> dict:
        cleaned = llm_output.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "source_language": "unknown",
                "translated_text": llm_output,
                "translation_confidence": 0.80
            }
