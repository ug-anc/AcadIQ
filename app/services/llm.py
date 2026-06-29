"""LLM providers for generation and query rewriting.

OpenAI provider follows the PRD. The local provider is a faithful *demo*: it does
not invent anything — it answers extractively from the single most relevant chunk
and attaches a correctly formatted citation, or returns NOT-FOUND. This keeps the
zero-key demo honest about the zero-hallucination contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.prompts import (
    INJECTION_RESPONSE,
    MASTER_SYSTEM_PROMPT,
    NOT_FOUND_RESPONSE,
    QUERY_REWRITE_PROMPT,
)


@dataclass
class Chunk:
    text: str
    document_name: str
    section_number: str
    page_number: int


# def format_context(chunks: list[Chunk]) -> str:
#     blocks = []
#     for c in chunks:
#         header = (
#             f"[Source: {c.document_name}, Section {c.section_number}, "
#             f"Page {c.page_number}]"
#         )
#         blocks.append(f"{header}\n{c.text}")
#     return "\n\n---\n\n".join(blocks)

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
    async def rewrite_query(self, query: str) -> str:
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

    # def __init__(self, settings: Settings):
    #     from openai import AsyncOpenAI

    #     self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    #     self._gen_model = settings.OPENAI_GENERATION_MODEL
    #     self._rewrite_model = settings.OPENAI_QUERY_REWRITE_MODEL

    async def generate(self, system_prompt: str, user_query: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._gen_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )
        return resp.choices[0].message.content or NOT_FOUND_RESPONSE

    async def rewrite_query(self, query: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._rewrite_model,
            temperature=0.0,
            messages=[
                {"role": "user", "content": QUERY_REWRITE_PROMPT.format(query=query)}
            ],
        )
        return (resp.choices[0].message.content or query).strip()


def _extract_context_section(rendered_prompt: str) -> str:
    marker = "CONTEXT — RETRIEVED DOCUMENT CHUNKS:"
    if marker not in rendered_prompt:
        return ""
    after = rendered_prompt.split(marker, 1)[1]
    return after.split("\n---\nSTUDENT QUESTION:", 1)[0].strip("-\n ")


def _collect_sources(context: str) -> list[str]:
    sources: list[str] = []
    for block in context.split("\n\n---\n\n"):
        header = block.strip().split("\n", 1)[0].strip()
        if header.startswith("[Source:") and header not in sources:
            sources.append(header.strip("[]"))
    return sources


def render_prompt(college_name: str, chunks: list[Chunk], user_query: str) -> str:
    return MASTER_SYSTEM_PROMPT.format(
        COLLEGE_NAME=college_name,
        RETRIEVED_CHUNKS=format_context(chunks),
        USER_QUERY=user_query,
    )


_LLM: LLM | None = None


def get_llm() -> LLM:
    global _LLM
    if _LLM is not None:
        return _LLM
    settings = get_settings()
    # _LLM = OpenAILLM(settings) if settings.LLM_PROVIDER == "openai" else LocalExtractiveLLM()
    print("DEBUG: FORCING OPENAILLM (Groq) PROVIDER")
    _LLM = OpenAILLM(settings)

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
