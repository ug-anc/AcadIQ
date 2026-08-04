"""LLM-based reflection (self-critique/self-correction): a check-and-revise
step between generation and returning the response to the caller.

Pipeline addition: generate -> [reflect_on_response] -> route -> return.

  PASS            -> use the generated answer as-is.
  NEEDS_REVISION  -> rewrite once via refine_response, then re-check. If it
                     still doesn't pass after REFLECTION_MAX_RETRIES rewrites,
                     fall back to a safe "could not verify" response rather
                     than return an ungrounded or incomplete answer.

The evaluator calls the LLM directly (via the Groq-compatible OpenAI client),
mirroring app.services.crag's pattern, since this is a structured-JSON grading
call rather than a generation call. If the evaluator call itself fails (bad
JSON, network error), that is treated as PASS on the current answer — the
answer already went through the pipeline's confidence gate and citation
enforcement, so an unreadable *evaluator* response shouldn't trigger a
rewrite based on a critique we don't actually have.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings
from app.prompts import (
    CRITIQUE_PROMPT,
    CRITIQUE_USER_TEMPLATE,
    REFINER_PROMPT,
    REFLECTION_FALLBACK_RESPONSE,
)

logger = logging.getLogger(__name__)

ReflectionVerdict = Literal["PASS", "NEEDS_REVISION"]


class ReflectionResult(BaseModel):
    is_grounded: bool
    answers_query: bool
    critique_notes: str = ""
    verdict: ReflectionVerdict = Field(default="NEEDS_REVISION")


def _client():
    from openai import AsyncOpenAI

    s = get_settings()
    return AsyncOpenAI(api_key=s.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


def _extract_json(raw: str) -> dict:
    """Strip markdown fences if the model added them, then parse JSON."""
    cleaned = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    return json.loads(cleaned)


def _format_context(retrieved_context: list[str]) -> str:
    return "\n\n---\n\n".join(retrieved_context) if retrieved_context else "(no context retrieved)"


def _pass_fallback(reason: str) -> ReflectionResult:
    logger.warning("Reflection evaluator falling back to PASS: %s", reason)
    return ReflectionResult(
        is_grounded=True,
        answers_query=True,
        critique_notes=f"reflection_fallback: {reason}",
        verdict="PASS",
    )


async def reflect_on_response(
    query: str, retrieved_context: list[str], generated_answer: str
) -> ReflectionResult:
    """Grade a generated answer for groundedness, answerability, and safety."""
    s = get_settings()
    try:
        resp = await _client().chat.completions.create(
            model=s.REFLECTION_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": CRITIQUE_PROMPT},
                {
                    "role": "user",
                    "content": CRITIQUE_USER_TEMPLATE.format(
                        query=query,
                        context=_format_context(retrieved_context),
                        answer=generated_answer,
                    ),
                },
            ],
        )
        raw = resp.choices[0].message.content or ""
        data = _extract_json(raw)
        return ReflectionResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        return _pass_fallback(f"unparseable evaluator output ({e})")
    except Exception as e:
        logger.error("Reflection evaluator call failed: %s", e, exc_info=True)
        return _pass_fallback(f"evaluator call error ({e})")


async def refine_response(
    query: str, retrieved_context: list[str], failed_answer: str, critique: str
) -> str:
    """Rewrite failed_answer to address critique, grounded only in retrieved_context."""
    s = get_settings()
    try:
        resp = await _client().chat.completions.create(
            model=s.REFLECTION_MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": REFINER_PROMPT.format(
                        query=query,
                        context=_format_context(retrieved_context),
                        failed_answer=failed_answer,
                        critique=critique,
                    ),
                },
            ],
        )
        refined = (resp.choices[0].message.content or "").strip()
        return refined or failed_answer
    except Exception as e:
        logger.error("Refiner call failed, keeping prior answer: %s", e, exc_info=True)
        return failed_answer


async def generate_with_reflection(
    query: str,
    context: list[str],
    initial_answer: str,
    max_reflections: int = 1,
) -> str:
    """Critique-and-revise loop, bounded to at most max_reflections rewrites.

    Note: unlike the spec's bare (query, context, max_reflections) signature,
    this takes initial_answer explicitly — reflection runs on the answer the
    existing generation pipeline already produced (with its citation-aware
    master prompt), rather than re-generating from scratch here.

    Terminates in at most max_reflections + 1 reflect calls and at most
    max_reflections refine calls: reflect -> (refine -> reflect)*max_reflections.
    Falls back to REFLECTION_FALLBACK_RESPONSE if it still doesn't pass.
    """
    answer = initial_answer
    for attempt in range(max_reflections + 1):
        result = await reflect_on_response(query, context, answer)
        if result.verdict == "PASS":
            return answer
        if attempt >= max_reflections:
            logger.warning(
                "Reflection did not reach PASS after %d retries for query '%s': %s",
                max_reflections, query[:80], result.critique_notes,
            )
            return REFLECTION_FALLBACK_RESPONSE
        answer = await refine_response(query, context, answer, result.critique_notes)
    return REFLECTION_FALLBACK_RESPONSE  # unreachable, satisfies type checkers
