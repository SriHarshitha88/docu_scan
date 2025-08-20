# DocuScan – Agentic Document Extraction

Intelligent, agentic extraction of structured data (key-value JSON) from PDFs and images with automatic routing, OCR, schema-aware validation, and trustworthy confidence scoring. Includes a modern Streamlit UI for uploads, visualization, and JSON export.

## Highlights

- Auto-detect document type (invoice, bill/medical bill, prescription, receipt, contract, financial, legal)
- OCR for scans with table handling and totals analysis (OpenAI + fallbacks)
- Agentic extraction with strict JSON output and self-consistency aggregation
- Validation rules: regex/date/amount checks and cross-field `totals_match`
- Confidence scoring per field and overall, shown in the UI
- JSON copy/download, per-field confidence bars, and summary QA report

## Confidence Formulas

- Field confidence: c = (llm^0.4) × (source^0.3) × (consensus^0.2) × (regex^0.1)
  - `llm`: LLM confidence (0..1)
  - `source`: evidence from OCR/position (0..1)
  - `consensus`: multi-sample agreement (self-consistency)
  - `regex`: 0.9 if field format validated, else 0.5

- Overall confidence: overall = 0.6 × avg(field_conf) + 0.25 × classification_conf + 0.15 × validation_pass_rate

## Sample Output

```json
{
  "doc_type": "invoice|medical_bill|prescription",
  "fields": [
    {
      "name": "PatientName",
      "value": "Priya Sharma",
      "confidence": 0.91,
      "source": {"page": 1, "bbox": [x1, y1, x2, y2]}
    }
  ],
  "overall_confidence": 0.88,
  "qa": {
    "passed_rules": ["totals_match"],
    "failed_rules": [],
    "notes": "2 low-confidence fields"
  }
}
```

## Setup

1) Clone and enter the repo
```bash
git clone <your-repo-url>
cd docu_scan
```

2) Python environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

3) Configure API key
- Preferred: create `.env` in the project root (not committed):
```
OPENAI_API_KEY=your_key_here
OPENAI_VISION_MODEL=gpt-4o
OPENAI_TEXT_MODEL=gpt-4o-mini
OCR_MAX_TOKENS=2000
LOG_LEVEL=INFO
UPLOAD_MAX_SIZE_MB=10
```

4) Run the app
```bash
streamlit run ui/app.py
```

