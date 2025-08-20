from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json
import re

import openai
from pydantic import BaseModel, Field, validator

from config import settings
from logger import app_logger
from document_classifier import DocumentType, ClassificationResult
from ocr_service import OCRResult, ExtractedText, BoundingBox
from validation import ValidationResult, validate_extraction
from scoring import compute_field_confidence, compute_overall_confidence


class SourceBBox(BaseModel):
    page: int = 1
    bbox: List[float] = Field(default_factory=list, description="[x1, y1, x2, y2]")


class ExtractedField(BaseModel):
    name: str
    value: Any
    confidence: float = 0.0
    source: Optional[SourceBBox] = None

    @validator("confidence")
    def _clip_conf(cls, v: float) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return 0.0


class QAReport(BaseModel):
    passed_rules: List[str] = Field(default_factory=list)
    failed_rules: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class ExtractionOutput(BaseModel):
    doc_type: str
    fields: List[ExtractedField]
    overall_confidence: float
    qa: QAReport


@dataclass
class AgentConfig:
    num_samples: int = 1
    temperature: float = 0.1
    timeout_s: int = 60


DOC_TYPE_MAP = {
    DocumentType.INVOICE: "invoice",
    DocumentType.BILL: "medical_bill",  # may be adjusted by medical context
    DocumentType.MEDICAL: "medical_bill",
    DocumentType.RECEIPT: "receipt",
    DocumentType.CONTRACT: "contract",
    DocumentType.FINANCIAL: "financial",
    DocumentType.LEGAL: "legal",
}


def _normalize_field_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()


class ExtractionAgent:
    """
    Orchestrates structured extraction with OpenAI and post-processing.
    """

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key is required for extraction.")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = settings.openai_text_model

    def _build_prompt(self, text: str, user_fields: Optional[List[str]], doc_type_hint: str) -> str:
        schema_hint = {
            "doc_type": "invoice|medical_bill|prescription|receipt|contract|financial|legal",
            "fields": [
                {
                    "name": "Field name (e.g., Invoice Number)",
                    "value": "Field value as string or number",
                    "confidence": 0.0,
                    "source": {"page": 1, "bbox": [0, 0, 0, 0]}
                }
            ],
            "overall_confidence": 0.0,
            "qa": {
                "passed_rules": ["totals_match"],
                "failed_rules": [],
                "notes": ""
            }
        }

        requested = ", ".join(user_fields) if user_fields else "auto-detect key fields"
        return (
            "You are a meticulous information extraction agent.\n"
            f"Document type hint: {doc_type_hint}.\n"
            "Extract the requested fields while also including other obvious key fields.\n"
            "Return STRICT JSON matching this schema (no markdown, no explanations):\n"
            f"{json.dumps(schema_hint)}\n\n"
            "Guidelines:\n"
            "- Provide confidence per field in [0,1] reflecting how certain you are.\n"
            "- If a field is missing, omit it (do not include nulls).\n"
            "- Use simple strings for dates in ISO format YYYY-MM-DD when possible.\n"
            "- Currency values should be numeric without currency symbol.\n"
            f"Requested fields: {requested}.\n\n"
            "Document Content:\n" + text[:120000]
        )

    def _call_llm(self, prompt: str, temperature: float) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except Exception:
            # Attempt to extract JSON substring
            match = re.search(r"\{[\s\S]*\}", content)
            return json.loads(match.group(0)) if match else {}

    def _aggregate_samples(self, samples: List[ExtractionOutput]) -> ExtractionOutput:
        if not samples:
            return ExtractionOutput(doc_type="unknown", fields=[], overall_confidence=0.0, qa=QAReport())

        # Majority vote for doc_type
        type_counts: Dict[str, int] = {}
        for s in samples:
            type_counts[s.doc_type] = type_counts.get(s.doc_type, 0) + 1
        doc_type = max(type_counts, key=type_counts.get)

        # Aggregate fields by normalized name
        buckets: Dict[str, List[ExtractedField]] = {}
        for s in samples:
            for f in s.fields:
                key = _normalize_field_name(f.name)
                buckets.setdefault(key, []).append(f)

        merged_fields: List[ExtractedField] = []
        for key, group in buckets.items():
            # Pick most common value; average confidence
            value_counts: Dict[str, int] = {}
            for f in group:
                v = str(f.value).strip()
                value_counts[v] = value_counts.get(v, 0) + 1
            best_value = max(value_counts, key=value_counts.get)
            avg_conf = sum(f.confidence for f in group) / max(1, len(group))
            merged_fields.append(ExtractedField(name=group[0].name, value=best_value, confidence=avg_conf, source=group[0].source))

        # Simple QA merge
        passed_rules = sorted({r for s in samples for r in s.qa.passed_rules})
        failed_rules = sorted({r for s in samples for r in s.qa.failed_rules})

        overall_conf = sum(s.overall_confidence for s in samples) / len(samples)

        return ExtractionOutput(
            doc_type=doc_type,
            fields=merged_fields,
            overall_confidence=overall_conf,
            qa=QAReport(passed_rules=passed_rules, failed_rules=failed_rules, notes=None)
        )

    def _map_doc_type(self, classification: ClassificationResult, ocr: Optional[OCRResult]) -> str:
        # Start with classifier mapping
        base = DOC_TYPE_MAP.get(classification.document_type, "unknown")
        if base == "medical_bill" and ocr:
            text_lower = ocr.raw_text.lower()
            if "prescription" in text_lower or "rx" in text_lower:
                return "prescription"
        return base

    def extract(self,
                text: str,
                classification: ClassificationResult,
                ocr: Optional[OCRResult] = None,
                user_fields: Optional[List[str]] = None,
                config: Optional[AgentConfig] = None) -> ExtractionOutput:
        cfg = config or AgentConfig()
        doc_type_hint = self._map_doc_type(classification, ocr)
        prompt = self._build_prompt(text=text, user_fields=user_fields, doc_type_hint=doc_type_hint)

        samples: List[ExtractionOutput] = []
        for i in range(max(1, cfg.num_samples)):
            try:
                raw = self._call_llm(prompt, temperature=cfg.temperature if cfg.num_samples > 1 else 0.1)
                model_out = ExtractionOutput(
                    doc_type=raw.get("doc_type", doc_type_hint),
                    fields=[ExtractedField(**f) for f in raw.get("fields", [])],
                    overall_confidence=float(raw.get("overall_confidence", 0.0)),
                    qa=QAReport(**raw.get("qa", {})) if raw.get("qa") else QAReport()
                )
                samples.append(model_out)
            except Exception as e:
                app_logger.warning(f"Extraction sample {i+1} failed: {e}")

        # Aggregate if multiple
        aggregated = self._aggregate_samples(samples)

        # Validation pass
        validation: ValidationResult = validate_extraction(aggregated, ocr)

        # Confidence scoring per field (recompute, override LLM confidence using formula)
        enriched_fields: List[ExtractedField] = []
        for f in aggregated.fields:
            source_conf = 0.5
            source_bbox: Optional[SourceBBox] = None
            if ocr:
                # Heuristic: find a structured element containing the value
                bbox = _find_bbox_for_value(ocr, str(f.value))
                if bbox:
                    source_conf = 0.9
                    source_bbox = SourceBBox(page=1, bbox=[bbox.x, bbox.y, bbox.x2, bbox.y2])
            field_conf = compute_field_confidence(
                llm_confidence=f.confidence or 0.7,
                source_confidence=source_conf,
                regex_valid=validation.field_validity.get(_normalize_field_name(f.name), True),
                consensus_strength=1.0 if len(samples) > 1 else 0.7
            )
            enriched_fields.append(ExtractedField(name=f.name, value=f.value, confidence=field_conf, source=source_bbox))

        overall_conf = compute_overall_confidence(
            field_confidences=[f.confidence for f in enriched_fields],
            classification_confidence=classification.confidence,
            validation_pass_rate=validation.pass_rate
        )

        return ExtractionOutput(
            doc_type=aggregated.doc_type,
            fields=enriched_fields,
            overall_confidence=overall_conf,
            qa=QAReport(
                passed_rules=validation.passed_rules,
                failed_rules=validation.failed_rules,
                notes=validation.notes,
            ),
        )


def _find_bbox_for_value(ocr: OCRResult, value: str) -> Optional[BoundingBox]:
    try:
        target = value.strip().lower()
        best: Optional[ExtractedText] = None
        for t in ocr.structured_text:
            if target and target in t.text.lower():
                best = t
                break
        return best.bbox if best else None
    except Exception:
        return None


def create_extraction_agent() -> ExtractionAgent:
    return ExtractionAgent()


