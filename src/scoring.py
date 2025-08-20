"""
Confidence scoring utilities.

We compute per-field confidence from multiple signals and aggregate overall.
"""

from __future__ import annotations

from typing import List


def compute_field_confidence(
    llm_confidence: float,
    source_confidence: float,
    regex_valid: bool,
    consensus_strength: float,
) -> float:
    """
    Compute field confidence using a weighted geometric mean style blend.

    Formula (bounded 0..1):
      c = (llm^0.4) * (source^0.3) * (consensus^0.2) * (regex^0.1)
    where regex = 0.9 if valid else 0.5.
    """
    import math

    llm = max(0.0, min(1.0, float(llm_confidence)))
    src = max(0.0, min(1.0, float(source_confidence)))
    cons = max(0.0, min(1.0, float(consensus_strength)))
    rx = 0.9 if regex_valid else 0.5

    conf = (llm ** 0.4) * (src ** 0.3) * (cons ** 0.2) * (rx ** 0.1)
    return max(0.0, min(1.0, conf))


def compute_overall_confidence(
    field_confidences: List[float],
    classification_confidence: float,
    validation_pass_rate: float,
) -> float:
    """
    Overall confidence combines average field confidence, classification, and validation.

    Overall formula:
      avg_fields = average(field_confidences)
      overall = 0.6 * avg_fields + 0.25 * classification_confidence + 0.15 * validation_pass_rate
    """
    if field_confidences:
        avg_fields = sum(field_confidences) / len(field_confidences)
    else:
        avg_fields = 0.0

    overall = 0.6 * avg_fields + 0.25 * max(0.0, min(1.0, classification_confidence)) + 0.15 * max(0.0, min(1.0, validation_pass_rate))
    return max(0.0, min(1.0, overall))


