"""LLM-based reflection (self-critique / self-correction).

Pipeline addition: generate -> [reflect_on_response] -> (accept | refine) ->
final answer. Runs *after* the existing generation step and *before* the
response is returned — insert it right before you build QueryResponse.

  PASS           -> the generated answer is returned unchanged.
  NEEDS_REVISION -> refine_response() rewrites the answer using the critique,
                     then it is reflected on again. Capped at max_reflections
                     rewrite attempts (default 1) so a persistently bad answer
                     can't loop forever; if it's still failing when the cap is
                     hit, REFLECTION_FALLBACK_RESPONSE is returned instead of
                     ever shipping an unverified answer.

reflect_on_response() calls the LLM directly (via the Groq-compatible OpenAI
client) rather than through app.services.llm.LLM, since that interface is
generation-oriented and doesn't have a structured-JSON method. Any critique
failure (bad JSON, network error) fails closed to NEEDS_REVISION rather than
PASS — an ungraded answer must never be assumed safe, matching this app's
zero-hallucination contract.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.prompts import (
    CRITIQUE_SYSTEM_PROMPT,
    CRITIQUE_USER_TEMPLATE,
    REFINER_SYSTEM_PROMPT,
    REFINER_USER_TEMPLATE,
    REFLECTION_FALLBACK_RESPONSE,
)

logger = logging.getLogger(__name__)

ReflectionVerdict = Literal["PASS", "NEEDS_REVISION"]


class ReflectionResult(BaseModel):
    is_grounded: bool
    answers_query: bool
    critique_notes: str = ""
    verdict: ReflectionVerdict


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import AsyncOpenAI

        s = get_settings()
        _CLIENT = AsyncOpenAI(api_key=s.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    return _CLIENT


def _format_context(retrieved_context: list[str]) -> str:
    if not retrieved_context:
        return "(no context retrieved)"
    return "\n\n---\n\n".join(retrieved_context)


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

    raise json.JSONDecodeError("no JSON object found in critique output", cleaned, 0)


def _fallback_reflection(reason: str) -> ReflectionResult:
    """Fail closed: an ungraded answer is never assumed to PASS."""
    logger.warning("Reflection critique falling back to NEEDS_REVISION: %s", reason)
    return ReflectionResult(
        is_grounded=False,
        answers_query=False,
        critique_notes=f"reflection_fallback: {reason}",
        verdict="NEEDS_REVISION",
    )


async def reflect_on_response(
    query: str, retrieved_context: list[str], generated_answer: str
) -> ReflectionResult:
    """Critique a generated answer for groundedness, answerability, and
    safety/policy adherence."""
    s = get_settings()
    try:
        resp = await _client().chat.completions.create(
            model=s.REFLECTION_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
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
        return _fallback_reflection(f"unparseable critique output ({e})")
    except Exception as e:
        logger.error("Reflection critique call failed: %s", e, exc_info=True)
        return _fallback_reflection(f"critique call error ({e})")


async def refine_response(
    query: str, retrieved_context: list[str], failed_answer: str, critique: str
) -> str:
    """Rewrite failed_answer using the critique. Falls back to the original
    answer on failure — the caller's next reflect_on_response call will
    re-grade it and the retry cap still applies."""
    s = get_settings()
    try:
        resp = await _client().chat.completions.create(
            model=s.REFLECTION_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": REFINER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": REFINER_USER_TEMPLATE.format(
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
        logger.error("Reflection refine call failed: %s", e, exc_info=True)
        return failed_answer


async def generate_with_reflection(
    query: str,
    context: list[str],
    generated_answer: str,
    max_reflections: int = 1,
) -> str:
    """Reflect on `generated_answer` and refine it up to `max_reflections`
    times. Insert this right after your existing generation call and before
    building QueryResponse — pass in the answer your generator already
    produced, and use the returned string in its place.

    Runs at most max_reflections refine attempts: 1 initial critique, then up
    to max_reflections (critique -> refine) cycles. If the answer still
    fails after the cap, returns REFLECTION_FALLBACK_RESPONSE rather than an
    unverified answer.
    """
    answer = generated_answer
    for attempt in range(max_reflections + 1):
        result = await reflect_on_response(query, context, answer)
        if result.verdict == "PASS":
            return answer

        if attempt >= max_reflections:
            logger.warning(
                "Reflection exhausted %d retries, still NEEDS_REVISION: %s",
                max_reflections, result.critique_notes,
            )
            return REFLECTION_FALLBACK_RESPONSE

        logger.info(
            "Reflection verdict=NEEDS_REVISION (attempt %d/%d): %s",
            attempt + 1, max_reflections, result.critique_notes,
        )
        answer = await refine_response(query, context, answer, result.critique_notes)

    return REFLECTION_FALLBACK_RESPONSE  # unreachable; defensive
