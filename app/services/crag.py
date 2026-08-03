"""Corrective RAG (CRAG): a retrieval-evaluation step between retrieval and generation.

Pipeline addition: retrieve -> [evaluate_retrieval] -> route -> generate.

  CORRECT   -> use retrieved chunks as-is.
  INCORRECT -> discard retrieved chunks entirely (no chunks survive to
               generation, so the existing NOT_FOUND path in generation.py
               takes over — nothing outside the retrieved documents is used).
  AMBIGUOUS -> keep only the chunks the evaluator judged relevant. If none
               are judged relevant, this also collapses to NOT_FOUND.

This app answers only from the official documents (see MASTER_SYSTEM_PROMPT /
NOT_FOUND_RESPONSE in app.prompts) — there is deliberately no web search or any
other out-of-corpus fallback here.

The evaluator calls the LLM directly (via the Groq-compatible OpenAI client)
rather than going through app.services.llm.LLM, since that interface is
generation-oriented and doesn't have a structured-JSON method. Any evaluator
failure (bad JSON, network error) falls back to AMBIGUOUS with all chunks
marked relevant — the safest default, since it never silently discards
context it couldn't grade.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings
from app.prompts import CRAG_EVALUATOR_SYSTEM_PROMPT, CRAG_EVALUATOR_USER_TEMPLATE

logger = logging.getLogger(__name__)

CRAGBand = Literal["CORRECT", "INCORRECT", "AMBIGUOUS"]


class CRAGEvaluation(BaseModel):
    band: CRAGBand
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: str = ""
    relevant_chunk_ids: list[str] = Field(default_factory=list)


def _client():
    from openai import AsyncOpenAI

    s = get_settings()
    return AsyncOpenAI(api_key=s.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


def _format_passages(chunks: list[dict[str, Any]]) -> str:
    lines = []
    for c in chunks:
        text = (c.get("text") or "")[:500]
        lines.append(f"[chunk_id={c['chunk_id']}]\n{text}")
    return "\n\n".join(lines)


def _extract_json(raw: str) -> dict[str, Any]:
    """Strip markdown fences if the model added them, then parse JSON."""
    cleaned = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    return json.loads(cleaned)


def _fallback_evaluation(chunks: list[dict[str, Any]], reason: str) -> CRAGEvaluation:
    logger.warning("CRAG evaluator falling back to AMBIGUOUS: %s", reason)
    return CRAGEvaluation(
        band="AMBIGUOUS",
        confidence=0.0,
        reasoning=f"evaluator_fallback: {reason}",
        relevant_chunk_ids=[c["chunk_id"] for c in chunks],
    )


async def evaluate_retrieval(query: str, chunks: list[dict[str, Any]]) -> CRAGEvaluation:
    """Grade retrieved chunks against the query. Empty input is always INCORRECT."""
    if not chunks:
        return CRAGEvaluation(band="INCORRECT", confidence=1.0, reasoning="no chunks retrieved")

    s = get_settings()
    try:
        resp = await _client().chat.completions.create(
            model=s.CRAG_EVAL_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": CRAG_EVALUATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": CRAG_EVALUATOR_USER_TEMPLATE.format(
                        query=query, passages=_format_passages(chunks)
                    ),
                },
            ],
        )
        raw = resp.choices[0].message.content or ""
        data = _extract_json(raw)
        return CRAGEvaluation.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        return _fallback_evaluation(chunks, f"unparseable evaluator output ({e})")
    except Exception as e:
        logger.error("CRAG evaluator call failed: %s", e, exc_info=True)
        return _fallback_evaluation(chunks, f"evaluator call error ({e})")


def _filter_relevant(
    chunks: list[dict[str, Any]], relevant_ids: list[str]
) -> list[dict[str, Any]]:
    keep = set(relevant_ids)
    return [c for c in chunks if c["chunk_id"] in keep]


async def apply_crag_eval(
    query: str, retrieved_chunks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    """Orchestrator: evaluate retrieval quality and route.

    Returns (final_context_chunks, band) — band is the CRAGEvaluation band
    ("CORRECT" | "INCORRECT" | "AMBIGUOUS"), useful for logging/response
    metadata the same way confidence_band already is. final_context_chunks is
    ready to feed straight into the existing answer generator; an empty list
    (INCORRECT, or AMBIGUOUS with nothing judged relevant) should be treated
    exactly like the existing "no chunks retrieved" NOT_FOUND path.
    """
    evaluation = await evaluate_retrieval(query, retrieved_chunks)

    if evaluation.band == "CORRECT":
        return retrieved_chunks, evaluation.band

    if evaluation.band == "INCORRECT":
        return [], evaluation.band

    # AMBIGUOUS: keep only the chunks judged relevant. No backfill — if
    # nothing survives, this collapses to NOT_FOUND same as INCORRECT.
    kept = _filter_relevant(retrieved_chunks, evaluation.relevant_chunk_ids)
    return kept, evaluation.band
