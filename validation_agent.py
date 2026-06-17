import json
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen
from pydantic import BaseModel,Field
from urllib.error import HTTPError, URLError


from agents.reporting_agent import ReportingAgent

class InvoiceLineItem(BaseModel):
    item_code: str | None = None
    description: str | None = None
    qty: float | None = None
    unit_price: float | None = None
    total: float | None = None

class InvoiceData(BaseModel):
    invoice_no: str | None = None
    invoice_date: str | None = None
    vendor_id: str | None = None
    vendor_name: str | None = None
    po_number: str | None = None
    currency: str | None = None
    total_amount: float | None = None
    line_items: list[InvoiceLineItem] = Field(default_factory=list)

class ValidationAgent:
    """
    Invoice Data Validation Agent + Business Validation Agent

    Personas:
        Data Auditor
        ERP Auditor

    Flow:
        validation_agent.py -> mock_erp/app.py
        validation_agent.py -> reporting_agent.py
    """

    def __init__(
        self,
        rules_path: str = "configs/rules.yaml",
        erp_base_url: str = "http://localhost:8501"

    
    ):
        self.rules_path = Path(rules_path)
        self.erp_base_url = erp_base_url.rstrip("/")
        self.rules = self._load_rules()
        self.reporting_agent = ReportingAgent()

    def _load_rules(self) -> dict:
        rules = {
            "required_fields": {
                "header": [],
                "line_item": []
            },
            "data_types": {},
            "tolerances": {
                "price_difference_percent": 5,
                "quantity_difference_percent": 0,
                "tax_difference_percent": 2
            },
            "accepted_currencies": [],
            "currency_symbol_map": {},
            "validation_policies": {
                "missing_field_action": "flag",
                "total_mismatch_action": "manual_review",
                "invalid_currency_action": "reject",
                "auto_approve_confidence_threshold": 0.95
            }
        }

        if not self.rules_path.exists():
            return rules

        current_section = None
        current_subsection = None

        with open(self.rules_path, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.rstrip("\n")
                stripped = line.strip()

                if not stripped or stripped.startswith("#"):
                    continue

                # Top-level section
                if not line.startswith(" ") and stripped.endswith(":"):
                    section_name = stripped.replace(":", "").strip()

                    if section_name in rules:
                        current_section = section_name
                        current_subsection = None
                    else:
                        current_section = None
                        current_subsection = None

                    continue

                # required_fields:
                #   header:
                #   line_item:
                if current_section == "required_fields":
                    if stripped.endswith(":"):
                        current_subsection = stripped.replace(":", "").strip()

                        if current_subsection not in rules["required_fields"]:
                            rules["required_fields"][current_subsection] = []

                        continue

                    if stripped.startswith("-") and current_subsection:
                        value = stripped.replace("-", "", 1).strip()
                        rules["required_fields"][current_subsection].append(value)

                elif current_section == "accepted_currencies":
                    if stripped.startswith("-"):
                        value = stripped.replace("-", "", 1).strip()
                        rules["accepted_currencies"].append(value)

                elif current_section in [
                    "data_types",
                    "tolerances",
                    "currency_symbol_map",
                    "validation_policies"
                ]:
                    if ":" in stripped:
                        key, value = stripped.split(":", 1)
                        key = key.strip().strip('"').strip("'")
                        value = value.split("#")[0].strip().strip('"').strip("'")

                        if value.lower() == "true":
                            value = True
                        elif value.lower() == "false":
                            value = False
                        else:
                            try:
                                if "." in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except ValueError:
                                pass

                        rules[current_section][key] = value

        rules.setdefault("required_fields", {})
        rules["required_fields"].setdefault("header", [])
        rules["required_fields"].setdefault("line_item", [])

        return rules

    def validate(self, state: dict) -> dict:
        invoice_data = state.get("invoice_data", {})

        try:
            InvoiceData(**invoice_data)
            state["pydantic_validation_status"] = "success"
        except Exception as error:
            state["pydantic_validation_status"] = "failed"
            state["pydantic_error"] = str(error)

        missing_fields = self._check_required_fields(invoice_data)
        data_discrepancies = self._validate_invoice_totals(invoice_data)
        currency_discrepancies = self._validate_currency(invoice_data)
        business_discrepancies = self._business_validate_against_erp(invoice_data)

        all_discrepancies = []
        all_discrepancies.extend(data_discrepancies)
        all_discrepancies.extend(currency_discrepancies)
        all_discrepancies.extend(business_discrepancies)

        state["missing_fields"] = missing_fields
        state["validation_discrepancies"] = data_discrepancies
        state["business_discrepancies"] = business_discrepancies
        state["all_discrepancies"] = all_discrepancies
        state["validation_status"] = "validated"

        state.setdefault("audit_trail", []).append(
            {
                "agent": "Validation Agent",
                "persona": "Data Auditor + ERP Auditor",
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Invoice validation and ERP validation completed."
            }
        )

        return self.reporting_agent.generate_report(state)

    def _check_required_fields(self, invoice_data: dict) -> list:
        missing = []

        required_fields = self.rules.get("required_fields", {})
        required_header = required_fields.get("header", [])
        required_line_item = required_fields.get("line_item", [])

        for field in required_header:
            if invoice_data.get(field) in [None, ""]:
                missing.append(field)

        line_items = invoice_data.get("line_items", [])

        if not line_items:
            missing.append("line_items")
            return missing

        for index, item in enumerate(line_items):
            for field in required_line_item:
                if item.get(field) in [None, ""]:
                    missing.append(f"line_items[{index}].{field}")

        return missing

    def _validate_invoice_totals(self, invoice_data: dict) -> list:
        """
        Validate line item totals and invoice total_amount.

        Uses:
            qty * unit_price = total
            sum(line item totals) = total_amount
        """

        discrepancies = []
        line_items = invoice_data.get("line_items", [])
        calculated_total = 0.0

        for index, item in enumerate(line_items):
            qty = item.get("qty")
            unit_price = item.get("unit_price")
            total = item.get("total")

            if qty is not None and unit_price is not None:
                expected_total = round(float(qty) * float(unit_price), 2)
                calculated_total += expected_total

                if total is not None and round(float(total), 2) != expected_total:
                    discrepancies.append(
                        {
                            "field": f"line_items[{index}].total",
                            "issue": "line_total_mismatch",
                            "expected": expected_total,
                            "actual": total,
                            "severity": "medium",
                            "message": "Line item total does not match qty * unit_price."
                        }
                    )

        total_amount = invoice_data.get("total_amount")

        if total_amount is not None and line_items:
            if round(float(total_amount), 2) != round(calculated_total, 2):
                discrepancies.append(
                    {
                        "field": "total_amount",
                        "issue": "invoice_total_mismatch",
                        "expected": round(calculated_total, 2),
                        "actual": total_amount,
                        "severity": "medium",
                        "message": "Invoice total amount does not match calculated line item total."
                    }
                )

        return discrepancies

    def _validate_currency(self, invoice_data: dict) -> list:
        currency = invoice_data.get("currency")
        accepted = self.rules.get("accepted_currencies", [])

        if currency and currency not in accepted:
            return [
                {
                    "field": "currency",
                    "issue": "invalid_currency",
                    "expected": accepted,
                    "actual": currency,
                    "severity": "high",
                    "message": "Invoice currency is not in accepted currency list."
                }
            ]

        return []

    def _get_json_from_erp(self, endpoint: str):
        try:
            with urlopen(f"{self.erp_base_url}{endpoint}", timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            return None

    def _business_validate_against_erp(self, invoice_data: dict) -> list:
        discrepancies = []

        po_number = invoice_data.get("po_number")
        vendor_id = invoice_data.get("vendor_id")
        invoice_currency = invoice_data.get("currency")

        if not po_number:
            return [
                {
                    "field": "po_number",
                    "issue": "missing_po_number",
                    "severity": "high",
                    "message": "PO number is missing from invoice."
                }
            ]

        po = self._get_json_from_erp(f"/po/{po_number}")

        if not po:
            return [
                {
                    "field": "po_number",
                    "issue": "po_not_found",
                    "actual": po_number,
                    "severity": "high",
                    "message": "PO number not found in mock ERP."
                }
            ]

        vendor = self._get_json_from_erp(f"/vendor/{vendor_id}") if vendor_id else None

        if vendor_id and not vendor:
            discrepancies.append(
                {
                    "field": "vendor_id",
                    "issue": "vendor_not_found",
                    "actual": vendor_id,
                    "severity": "high",
                    "message": "Vendor ID not found in mock ERP."
                }
            )

        if vendor_id and po.get("vendor_id") != vendor_id:
            discrepancies.append(
                {
                    "field": "vendor_id",
                    "issue": "vendor_po_mismatch",
                    "expected": po.get("vendor_id"),
                    "actual": vendor_id,
                    "severity": "high",
                    "message": "Invoice vendor does not match ERP PO vendor."
                }
            )

        erp_items = {
            item.get("item_code"): item
            for item in po.get("line_items", [])
        }

        tolerance = float(
            self.rules.get("tolerances", {}).get("price_difference_percent", 5)
        )

        for item in invoice_data.get("line_items", []):
            item_code = item.get("item_code")
            erp_item = erp_items.get(item_code)

            if not erp_item:
                discrepancies.append(
                    {
                        "field": "item_code",
                        "issue": "item_not_found_in_po",
                        "actual": item_code,
                        "severity": "medium",
                        "message": "Invoice item code not found in ERP PO."
                    }
                )
                continue

            erp_currency = erp_item.get("currency")

            if invoice_currency and erp_currency and invoice_currency != erp_currency:
                discrepancies.append(
                    {
                        "field": "currency",
                        "issue": "currency_mismatch",
                        "expected": erp_currency,
                        "actual": invoice_currency,
                        "severity": "high",
                        "message": "Invoice currency does not match ERP line item currency."
                    }
                )

            invoice_qty = item.get("qty")
            erp_qty = erp_item.get("qty")

            if invoice_qty != erp_qty:
                discrepancies.append(
                    {
                        "field": "qty",
                        "issue": "quantity_mismatch",
                        "expected": erp_qty,
                        "actual": invoice_qty,
                        "severity": "medium",
                        "message": "Invoice quantity does not match ERP quantity."
                    }
                )

            invoice_price = item.get("unit_price")
            erp_price = erp_item.get("unit_price")

            if invoice_price is not None and erp_price:
                diff_percent = (
                    abs(float(invoice_price) - float(erp_price))
                    / float(erp_price)
                    * 100
                )

                if diff_percent > tolerance:
                    discrepancies.append(
                        {
                            "field": "unit_price",
                            "issue": "price_difference_exceeds_tolerance",
                            "expected": erp_price,
                            "actual": invoice_price,
                            "difference_percent": round(diff_percent, 2),
                            "tolerance_percent": tolerance,
                            "severity": "medium",
                            "message": "Unit price difference exceeds configured tolerance."
                        }
                    )

        return discrepancies