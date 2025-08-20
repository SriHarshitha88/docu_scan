from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import re
from datetime import datetime

from logger import app_logger


@dataclass
class ValidationResult:
    passed_rules: List[str] = field(default_factory=list)
    failed_rules: List[str] = field(default_factory=list)
    field_validity: Dict[str, bool] = field(default_factory=dict)  # normalized field name -> valid?
    pass_rate: float = 1.0
    notes: Optional[str] = None


DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",  # ISO
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
]

AMOUNT_PATTERN = r"^-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$|^-?\d+(?:\.\d{2})?$"


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()


def _is_date(value: str) -> bool:
    s = str(value).strip()
    for p in DATE_PATTERNS:
        if re.search(p, s):
            return True
    # ISO parse try
    try:
        datetime.fromisoformat(s)
        return True
    except Exception:
        return False


def _is_amount(value: str) -> bool:
    s = str(value).strip()
    return re.match(AMOUNT_PATTERN, s) is not None


def validate_extraction(extraction, ocr=None) -> ValidationResult:
    try:
        passed: List[str] = []
        failed: List[str] = []
        field_validity: Dict[str, bool] = {}

        # Field-level checks based on simple heuristics
        for f in extraction.fields:
            key = _norm(f.name)
            val_str = str(f.value)
            valid = True

            if any(k in key for k in ["date", "dob", "due"]):
                valid = _is_date(val_str)
            elif any(k in key for k in ["amount", "total", "subtotal", "tax", "balance"]):
                valid = _is_amount(val_str)
            elif "invoice number" in key or "account number" in key:
                valid = len(val_str.strip()) >= 3

            field_validity[key] = valid

        # Cross-field: totals math check
        def to_num(s: str) -> Optional[float]:
            try:
                return float(str(s).replace(",", ""))
            except Exception:
                return None

        totals = { _norm(f.name): to_num(f.value) for f in extraction.fields }
        subtotal = totals.get("subtotal")
        tax = totals.get("tax amount") or totals.get("tax")
        total = totals.get("total amount") or totals.get("amount due") or totals.get("total")

        if subtotal is not None and tax is not None and total is not None:
            if abs((subtotal + tax) - total) < 0.01:
                passed.append("totals_match")
            else:
                failed.append("totals_match")

        # Compute pass rate: count of valid fields / total fields
        if extraction.fields:
            pass_rate = sum(1 for v in field_validity.values() if v) / max(1, len(field_validity))
        else:
            pass_rate = 0.0

        notes = None
        low_conf = [f for f in extraction.fields if (f.confidence or 0.0) < 0.5]
        if low_conf:
            notes = f"{len(low_conf)} low-confidence fields"

        return ValidationResult(
            passed_rules=passed,
            failed_rules=failed,
            field_validity=field_validity,
            pass_rate=pass_rate,
            notes=notes,
        )

    except Exception as e:
        app_logger.warning(f"Validation failed: {e}")
        return ValidationResult(passed_rules=[], failed_rules=["validation_error"], field_validity={}, pass_rate=0.0, notes=str(e))


