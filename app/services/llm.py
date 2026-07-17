"""LLM providers for generation, query rewriting, and multi-turn contextualization.

OpenAI provider follows the PRD. The local provider is a faithful *demo*: it does
not invent anything — it answers extractively from the single most relevant chunk
and attaches a correctly formatted citation, or returns NOT-FOUND. This keeps the
zero-key demo honest about the zero-hallucination contract.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.prompts import (
    CONTEXTUALIZER_SYSTEM_PROMPT,
    CONTEXTUALIZER_USER_TEMPLATE,
    INJECTION_RESPONSE,
    MASTER_SYSTEM_PROMPT,
    NOT_FOUND_RESPONSE,
    QUERY_REWRITE_PROMPT,
    QUERY_TRANSLATION_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    document_name: str
    section_number: str
    page_number: int


def format_context(chunks: list[Chunk]) -> str:
    formatted = ""
    for i, chunk in enumerate(chunks, 1):
        # Access attributes directly, not via .metadata
        header = f"[{i}] SOURCE: {chunk.document_name}, Section {chunk.section_number}, Page {chunk.page_number}"
        formatted += f"{header}\n{chunk.text}\n\n"
    return formatted


class LLM(ABC):
    @abstractmethod
    async def generate(self, system_prompt: str, user_query: str) -> str:
        ...

    @abstractmethod
    async def rewrite_query_to_search_terms(self, query: str) -> list[str]:
        ...

    @abstractmethod
    async def contextualize_query(
        self, raw_query: str, history: list[dict]
    ) -> str:
        """Rewrite *raw_query* into a standalone question using conversation
        history.  If history is empty, return raw_query unchanged."""
        ...


class LocalExtractiveLLM(LLM):
    """Demo generator. Echoes the grounding chunk verbatim with a citation.

    `generate` receives the fully rendered master prompt; we recover the context
    block and the top chunk's citation header from it, then format a structured,
    fully-cited answer. No free-form text is produced, so nothing can be
    hallucinated.
    """

    async def generate(self, system_prompt: str, user_query: str) -> str:
        # for debugging
        print("DEBUG: LOCAL EXTRACTIVE LLM IS RUNNING!")
        context = _extract_context_section(system_prompt)
        if not context.strip():
            return NOT_FOUND_RESPONSE
        first_block = context.split("\n\n---\n\n")[0].strip()
        lines = first_block.split("\n", 1)
        citation = lines[0].strip() if lines else ""
        body = lines[1].strip() if len(lines) > 1 else ""
        # if not citation.startswith("[Source:") or not body:
        #     return NOT_FOUND_RESPONSE
        return body if body else NOT_FOUND_RESPONSE
        snippet = body[:600].rstrip()
        sources = _collect_sources(context)
        return (
            "DIRECT ANSWER:\n"
            f"{snippet} {citation}\n\n"
            "DETAILS:\n"
            f"The above is quoted directly from the official document. {citation}\n\n"
            "SOURCES REFERENCED:\n" + "\n".join(f"- {s}" for s in sources)
        )

    async def rewrite_query(self, query: str) -> str:
        return query  # local mode skips rewriting

    async def rewrite_query_to_search_terms(self, query: str) -> list[str]:
        return [query]

    async def contextualize_query(
        self, raw_query: str, history: list[dict]
    ) -> str:
        """Demo mode: return raw_query unchanged.

        This is by-design — the local extractive LLM has no generative
        capability for query rewriting.  Multi-turn contextualization is
        a no-op in demo mode; RULE H2 (fresh retrieval) still runs because
        the returned query always goes through the retrieval pipeline.
        """
        logger.info(
            "contextualize_query: demo mode — returning raw_query unchanged "
            "(session_id contextualization is a no-op for LocalExtractiveLLM)"
        )
        return raw_query


class OpenAILLM(LLM):
    def __init__(self, settings: Settings):
        from openai import AsyncOpenAI

        # Point the client to Groq's URL
        self._client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        # Choose a fast, free model supported by Groq
        self._gen_model = settings.LLM_GEN_MODEL
        self._rewrite_model = settings.LLM_REWRITE_MODEL
        self._contextualizer_model = settings.SESSION_CONTEXTUALIZER_MODEL


    async def generate(self, system_prompt: str, user_query: str) -> str:
        logger.info("Calling LLM generate with model: %s", self._gen_model)
        resp = await self._client.chat.completions.create(
            model=self._gen_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )
        answer = resp.choices[0].message.content or NOT_FOUND_RESPONSE
        logger.info("LLM generate call successful. Response length: %d characters", len(answer))
        return answer

    async def rewrite_query(self, query: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._rewrite_model,
            temperature=0.0,
            messages=[
                {"role": "user", "content": QUERY_REWRITE_PROMPT.format(query=query)}
            ],
        )
        return (resp.choices[0].message.content or query).strip()

    async def rewrite_query_to_search_terms(self, query: str) -> list[str]:
        import re
        try:
            resp = await self._client.chat.completions.create(
                model=self._rewrite_model,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": QUERY_TRANSLATION_PROMPT.format(query=query)}
                ],
            )
            raw = (resp.choices[0].message.content or query).strip()
            lines = [line.strip() for line in raw.split("\n") if line.strip()]
            cleaned = []
            for line in lines:
                l = re.sub(r"^[\s\-\*\d\.\:\)\(]+", "", line).strip()
                l = l.strip('"\'')
                if l:
                    cleaned.append(l)
            if not cleaned:
                return [query]
            return cleaned[:3]
        except Exception as e:
            logger.error(f"Failed to rewrite query to search terms: {e}", exc_info=True)
            return [query]

    async def contextualize_query(
        self, raw_query: str, history: list[dict]
    ) -> str:
        """Use the contextualizer LLM to rewrite a follow-up into a standalone
        query.  Implements Fix #2: any exception/timeout falls back to
        raw_query so the turn is never blocked.  RULE H2 guarantees fresh
        retrieval regardless of contextualizer outcome.
        """
        if not history:
            return raw_query

        # Build the history block for the prompt
        history_lines: list[str] = []
        for turn in history:
            role_label = "Student" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{role_label}: {turn['content']}")
        history_text = "\n".join(history_lines)

        user_msg = CONTEXTUALIZER_USER_TEMPLATE.format(
            history=history_text, raw_query=raw_query
        )

        try:
            resp = await self._client.chat.completions.create(
                model=self._contextualizer_model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": CONTEXTUALIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            standalone = (resp.choices[0].message.content or raw_query).strip()
            if not standalone:
                standalone = raw_query
            logger.info(
                "Contextualizer: '%s' → '%s'", raw_query[:80], standalone[:80]
            )
            return standalone
        except Exception:
            # Fix #2: Never block the turn — fall back to raw_query.
            # RULE H2 still runs because the caller always does fresh retrieval.
            logger.warning(
                "Contextualizer call failed — falling back to raw_query: '%s'",
                raw_query[:80],
                exc_info=True,
            )
            return raw_query


def _extract_context_section(rendered_prompt: str) -> str:
    marker = "RETRIEVED MANUAL CHUNKS (FACTUAL DOCUMENTATION - TRUTH SOURCE):"
    if marker not in rendered_prompt:
        legacy_marker = "CONTEXT — RETRIEVED DOCUMENT CHUNKS:"
        if legacy_marker not in rendered_prompt:
            return ""
        marker = legacy_marker
    after = rendered_prompt.split(marker, 1)[1]
    return after.split("\n---\nSTUDENT QUESTION:", 1)[0].strip("-\n ")


def _collect_sources(context: str) -> list[str]:
    sources: list[str] = []
    for block in context.split("\n\n---\n\n"):
        header = block.strip().split("\n", 1)[0].strip()
        if header.startswith("[Source:") and header not in sources:
            sources.append(header.strip("[]"))
    return sources


def format_history(history: list[dict]) -> str:
    if not history:
        return "No prior conversation history."
    formatted = []
    for turn in history:
        role = "Student" if turn["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {turn['content']}")
    return "\n".join(formatted)


def render_prompt(
    college_name: str,
    chunks: list[Chunk],
    user_query: str,
    history: list[dict] | None = None,
) -> str:
    history_str = format_history(history or [])
    return MASTER_SYSTEM_PROMPT.format(
        COLLEGE_NAME=college_name,
        CONVERSATION_HISTORY=history_str,
        RETRIEVED_CHUNKS=format_context(chunks),
        USER_QUERY=user_query,
    )


_LLM: LLM | None = None


def get_llm() -> LLM:
    global _LLM
    if _LLM is not None:
        return _LLM
    settings = get_settings()
    if settings.LLM_PROVIDER == "openai":
        _LLM = OpenAILLM(settings)
    else:
        _LLM = LocalExtractiveLLM()
    return _LLM


__all__ = [
    "Chunk",
    "LLM",
    "get_llm",
    "render_prompt",
    "format_context",
    "INJECTION_RESPONSE",
    "NOT_FOUND_RESPONSE",
]
