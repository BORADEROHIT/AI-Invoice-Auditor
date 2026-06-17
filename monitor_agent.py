import json
import time
from pathlib import Path
from datetime import datetime

from agents.extractor_agent import ExtractorAgent


"""
    Invoice Monitor Agent
    Persona: System Watchdog
    Uses LLM: No
    Flow:
        monitor_agent.py -> extractor_agent.py
"""

class InvoiceMonitorAgent:
    
    

    def __init__(
        self,
        incoming_dir: str = "data/incoming",
        rules_path: str = "configs/rules.yaml"
    ):
        self.incoming_dir = Path(incoming_dir)
        self.rules_path = Path(rules_path)
        self.supported_extensions = [".pdf", ".docx", ".png"]

        self.incoming_dir.mkdir(parents=True, exist_ok=True)

        self.logging_config = self._load_logging_config()
        self.extractor_agent = ExtractorAgent()

    def _load_logging_config(self) -> dict:
        default_config = {
            "enable_audit_log": True,
            "log_file": "./logs/invoice_auditor.log",
            "log_level": "INFO"
        }

        if not self.rules_path.exists():
            return default_config

        config = default_config.copy()
        inside_logging = False

        with open(self.rules_path, "r", encoding="utf-8") as file:
            for line in file:
                clean_line = line.strip()

                if not clean_line or clean_line.startswith("#"):
                    continue

                if clean_line == "logging:":
                    inside_logging = True
                    continue

                if inside_logging and not line.startswith(" "):
                    break

                if inside_logging and ":" in clean_line:
                    key, value = clean_line.split(":", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")

                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False

                    config[key] = value

        return config

    def _write_log(self, message: str):
        if not self.logging_config.get("enable_audit_log", True):
            return

        log_file = Path(
            self.logging_config.get(
                "log_file",
                "./logs/invoice_auditor.log"
            )
        )

        log_file.parent.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().isoformat()
        log_level = self.logging_config.get("log_level", "INFO")

        with open(log_file, "a", encoding="utf-8") as file:
            file.write(
                f"{timestamp} | {log_level} | "
                f"InvoiceMonitorAgent | {message}\n"
            )

    def _get_metadata_path(self, invoice_path: Path) -> Path:
        return invoice_path.with_suffix(".meta.json")

    def _load_email_metadata(self, invoice_path: Path) -> dict:
        metadata_path = self._get_metadata_path(invoice_path)

        if not metadata_path.exists():
            return {}

        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as error:
            self._write_log(
                f"Failed to read metadata file {metadata_path}: {str(error)}"
            )
            return {}

    def scan_incoming_folder(self) -> list:
        detected_files = []

        for file_path in self.incoming_dir.iterdir():
            if not file_path.is_file():
                continue

            if file_path.name.endswith(".meta.json"):
                continue

            if file_path.suffix.lower() in self.supported_extensions:
                detected_files.append(str(file_path))
                self._write_log(f"Detected invoice file: {file_path}")
            else:
                self._write_log(f"Unsupported file ignored: {file_path}")

        return detected_files

    def scan_existing_files_for_ui(self) -> list:
        detected_files = []

        for file_path in self.incoming_dir.iterdir():
            if not file_path.is_file():
                continue

            if file_path.name.endswith(".meta.json"):
                continue

            if file_path.suffix.lower() in self.supported_extensions:
                metadata = self._load_email_metadata(file_path)

                detected_files.append(
                    {
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "file_type": file_path.suffix.lower(),
                        "metadata_available": bool(metadata),
                        "metadata": metadata,
                        "status": "detected"
                    }
                )

        return detected_files

    def trigger_extraction(self, file_path: str) -> dict:
        invoice_path = Path(file_path)
        metadata = self._load_email_metadata(invoice_path)

        self._write_log(f"Triggering Extractor Agent for file: {file_path}")

        try:
            result = self.extractor_agent.extract(
                file_path=file_path,
                email_metadata=metadata
            )

            self._write_log(
                f"Extractor Agent completed for file: {file_path} "
                f"with status: {result.get('status')}"
            )

            return result

        except Exception as error:
            self._write_log(
                f"Extractor Agent failed for file: {file_path}. "
                f"Error: {str(error)}"
            )

            return {
                "status": "extraction_failed",
                "file_path": file_path,
                "email_metadata": metadata,
                "error": str(error)
            }

    def run_once(self) -> list:
        detected_files = self.scan_incoming_folder()
        processed_files = set()
   

        if not detected_files:
            return [
                {
                    "status": "no_files_found",
                    "message": "No invoice files found in data/incoming."
                }
            ]

        results = []

        for file_path in detected_files:
                if file_path not in processed_files:
                    self.trigger_extraction(file_path)
                    processed_files.add(file_path)

        return results

    def poll_forever(self, interval_seconds: int = 5):
        processed_files = set()

        self._write_log("Continuous polling started.")

        while True:
            detected_files = self.scan_incoming_folder()

            for file_path in detected_files:
                if file_path not in processed_files:
                    self.trigger_extraction(file_path)
                    processed_files.add(file_path)

            time.sleep(interval_seconds)

    def run(self) -> list:
        return self.run_once()

if __name__ == "main":
    agent = InvoiceMonitorAgent()
    output = agent.run_once()

    for item in output:
        print(item)

