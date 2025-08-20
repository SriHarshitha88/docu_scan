"""
Evaluation utilities to compare extracted JSON with ground truth.
Computes field-level precision/recall/F1 and overall accuracy, plus notes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import math


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def evaluate_extraction(pred: Dict, truth: Dict, field_key: str = "name", value_key: str = "value") -> Dict:
    pred_fields = { _norm(f.get(field_key)): str(f.get(value_key, "")) for f in pred.get("fields", []) }
    true_fields = { _norm(f.get(field_key)): str(f.get(value_key, "")) for f in truth.get("fields", []) }

    keys = set(pred_fields.keys()) | set(true_fields.keys())
    tp = 0
    fp = 0
    fn = 0
    per_field = {}

    for k in keys:
        pv = pred_fields.get(k)
        tv = true_fields.get(k)
        if pv is None and tv is not None:
            fn += 1
            per_field[k] = {"match": False, "reason": "missing"}
        elif pv is not None and tv is None:
            fp += 1
            per_field[k] = {"match": False, "reason": "spurious"}
        else:
            # both present
            if _norm(pv) == _norm(tv):
                tp += 1
                per_field[k] = {"match": True}
            else:
                fp += 1
                fn += 1
                per_field[k] = {"match": False, "reason": "mismatch", "pred": pv, "truth": tv}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_field": per_field,
    }


