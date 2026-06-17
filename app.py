import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Mock ERP API")

DATA_DIR = Path(file).parent / "data"

def load_json(file_name: str):
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"ERP data file not found: {file_name}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
    
@app.get("/health")
def health():
    return {
    "status": "ok",
    "service": "mock_erp"
    }

@app.get("/vendor/{vendor_id}")
def get_vendor(vendor_id: str):
    vendors = load_json("vendors.json")

    for vendor in vendors:
        if vendor.get("vendor_id") == vendor_id:
            return vendor

    raise HTTPException(status_code=404, detail="Vendor not found")

@app.get("/po/{po_number}")
def get_po(po_number: str):
    po_records = load_json("po_records.json")

    for po in po_records:
        if po.get("po_number") == po_number:
            return po

    raise HTTPException(status_code=404, detail="PO not found")


@app.get("/sku/{item_code}")
def get_sku(item_code: str):
    sku_records = load_json("sku_master.json")

    for sku in sku_records:
        if sku.get("item_code") == item_code:
            return sku

    raise HTTPException(status_code=404, detail="SKU not found")
