"""Corrective RAG (CRAG): a retrieval-evaluation step between retrieval and generation.

Pipeline addition: retrieve -> [evaluate_retrieval] -> route -> generate.

  CORRECT   -> keep only the chunks judged relevant (falls back to the full
               set if the evaluator returned CORRECT but no ids, which would
               otherwise discard good context on a formatting slip).
  INCORRECT -> rewrite the query and re-retrieve once against the SAME corpus,
               then re-evaluate. If that retry is still unusable, no chunks
               survive to generation and the existing NOT_FOUND path in
               generation.py takes over.
  AMBIGUOUS -> keep only the chunks judged relevant. If none survive the
               filter, that's functionally the same failure as INCORRECT, so
               it gets the same rewrite-and-retry treatment.

This app answers only from the official documents (see MASTER_SYSTEM_PROMPT /
NOT_FOUND_RESPONSE in app.prompts) — there is deliberately no web search or any
other out-of-corpus fallback here. The corrective action is entirely internal:
rewrite the query, search the same vector store again.

The retry is capped at CRAG_MAX_RETRIES (default 1) so a persistently bad
query can't loop forever — after the cap is hit, an unusable result always
resolves to NOT_FOUND rather than retrying indefinitely.

The evaluator and retry-rewriter call the LLM directly (via the Groq-compatible
OpenAI client) rather than going through app.services.llm.LLM, since that
interface is generation-oriented and doesn't have a structured-JSON method.
Any evaluator failure (bad JSON, network error) falls back to AMBIGUOUS with
all chunks marked relevant — the safest default, since it never silently
discards context it couldn't grade.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings
from app.prompts import (
    CRAG_EVALUATOR_SYSTEM_PROMPT,
    CRAG_EVALUATOR_USER_TEMPLATE,
    CRAG_REQUERY_PROMPT,
)
from app.services.retrieval import RetrievedChunk, get_retrieval_engine

logger = logging.getLogger(__name__)

CRAGBand = Literal["CORRECT", "INCORRECT", "AMBIGUOUS"]


class CRAGEvaluation(BaseModel):
    band: CRAGBand
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: str = ""
    relevant_chunk_ids: list[str] = Field(default_factory=list)


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import AsyncOpenAI

        s = get_settings()
        _CLIENT = AsyncOpenAI(api_key=s.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    return _CLIENT


def chunk_to_dict(c: RetrievedChunk) -> dict[str, Any]:
    return dataclasses.asdict(c)


def dict_to_chunk(d: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=d["chunk_id"],
        text=d["text"],
        metadata=d["metadata"],
        score=d.get("score", 0.0),
        cosine_similarity=d.get("cosine_similarity", 0.0),
    )


def _format_passages(chunks: list[dict[str, Any]]) -> str:
    lines = []
    for c in chunks:
        text = (c.get("text") or "")[:500]
        lines.append(f"[chunk_id={c['chunk_id']}]\n{text}")
    return "\n\n".join(lines)


def _extract_json(raw: str) -> dict[str, Any]:
    """Parse JSON out of an LLM response that may have markdown fences or a
    prose preamble around the actual object."""
    cleaned = raw.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise json.JSONDecodeError("no JSON object found in evaluator output", cleaned, 0)


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


async def rewrite_query_for_retry(query: str) -> list[str]:
    """Generate alternative search phrasings for a corrective re-retrieval.

    Deliberately a different prompt (and non-zero temperature) from the
    upfront query-expansion rewrite in app.services.llm — retrying with the
    same deterministic rewrite on the same input would likely reproduce the
    same search terms that already failed. Falls back to the original query
    on any failure.
    """
    s = get_settings()
    try:
        resp = await _client().chat.completions.create(
            model=s.CRAG_EVAL_MODEL,
            temperature=0.5,
            messages=[{"role": "user", "content": CRAG_REQUERY_PROMPT.format(query=query)}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        cleaned = []
        for line in raw.split("\n"):
            l = re.sub(r"^[\s\-\*\d\.\:\)\(]+", "", line).strip().strip("\"'")
            if l:
                cleaned.append(l)
        return cleaned[:3] or [query]
    except Exception as e:
        logger.warning(
            "CRAG retry rewrite failed, reusing original query: %s", e, exc_info=True
        )
        return [query]


def _filter_relevant(
    chunks: list[dict[str, Any]], relevant_ids: list[str]
) -> list[dict[str, Any]]:
    keep = set(relevant_ids)
    return [c for c in chunks if c["chunk_id"] in keep]


async def _retry_retrieval(query: str) -> list[dict[str, Any]]:
    search_terms = await rewrite_query_for_retry(query)
    logger.info("CRAG retry: rewrote query=%r -> search_terms=%r", query, search_terms)
    engine = get_retrieval_engine()
    new_chunks = await engine.retrieve(search_terms, original_query=query)
    return [chunk_to_dict(c) for c in new_chunks]


async def _resolve(
    query: str,
    chunks: list[dict[str, Any]],
    evaluation: CRAGEvaluation,
    retries_left: int,
) -> tuple[list[dict[str, Any]], str]:
    if evaluation.band == "CORRECT":
        kept = _filter_relevant(chunks, evaluation.relevant_chunk_ids) or chunks
        return kept, evaluation.band

    if evaluation.band == "AMBIGUOUS":
        kept = _filter_relevant(chunks, evaluation.relevant_chunk_ids)
        if kept:
            return kept, evaluation.band
        # Nothing survived the filter — treat like INCORRECT below.

    if retries_left <= 0:
        logger.info(
            "CRAG: band=%s and out of retries for query=%r -- returning no chunks",
            evaluation.band, query,
        )
        return [], evaluation.band

    new_chunks = await _retry_retrieval(query)
    if not new_chunks:
        return [], "INCORRECT"

    re_evaluation = await evaluate_retrieval(query, new_chunks)
    return await _resolve(query, new_chunks, re_evaluation, retries_left - 1)


async def apply_crag_eval(
    query: str, retrieved_chunks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    """Orchestrator: evaluate retrieval quality, rewrite-and-retry on failure,
    and return the final context.

    Returns (final_context_chunks, band) — band is the CRAGEvaluation band of
    whichever attempt produced the final result ("CORRECT" | "INCORRECT" |
    "AMBIGUOUS"), useful for logging/response metadata the same way
    confidence_band already is. final_context_chunks is ready to feed straight
    into the existing answer generator; an empty list means every attempt
    (including retries) failed and should be treated exactly like the existing
    "no chunks retrieved" NOT_FOUND path.
    """
    s = get_settings()
    evaluation = await evaluate_retrieval(query, retrieved_chunks)
    return await _resolve(query, retrieved_chunks, evaluation, s.CRAG_MAX_RETRIES)
