"""Confidence gating (DR-04).

Dual threshold:
  score >= soft_warn   -> "found"           (answer normally)
  hard <= score < soft -> "low_confidence"  (answer, but flag verify-with-source)
  score <  hard_reject -> "not_found"       (never call the LLM; return NOT-FOUND)

The score is the reranker relevance of the top retrieved chunk. Defaults are
sensible starting points; calibrate them empirically with
eval/calibrate_threshold.py once real documents and queries exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings

FOUND = "found"
LOW_CONFIDENCE = "low_confidence"
NOT_FOUND = "not_found"


@dataclass
class GateDecision:
    band: str
    score: float
    should_generate: bool


def evaluate_confidence(top_score: float) -> GateDecision:
    s = get_settings()
    if top_score < s.CONFIDENCE_HARD_REJECT:
        return GateDecision(NOT_FOUND, top_score, should_generate=False)
    if top_score < s.CONFIDENCE_SOFT_WARN:
        return GateDecision(LOW_CONFIDENCE, top_score, should_generate=True)
    return GateDecision(FOUND, top_score, should_generate=True)
