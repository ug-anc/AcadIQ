"""Section-anchored chunking (DR-03).

Each numbered section becomes one atomic chunk so a rule's condition and
consequence never split across chunks — the dominant source of policy
hallucination. A heading is detected by a numbered-section pattern (4, 4.3,
4.3.2 ...) or by a line that is meaningfully larger/bold than body text. Sections
over MAX_SECTION_TOKENS fall back to a recursive character split, preserving the
section path on every piece. Tables are emitted as their own `table` chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config import get_settings
from app.ingestion.extractor import ExtractedDoc, Line

# "4", "4.3", "4.3.2", optionally followed by a title on the same line.
_SECTION_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){0,4})\.?\s+(?P<title>\S.*)$")
_CHAPTER_RE = re.compile(r"^(chapter|section|part)\s+\d+", re.IGNORECASE)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)  # ~4 chars/token heuristic


def _recursive_split(text: str, max_tokens: int) -> list[str]:
    separators = ["\n\n", "\n", ". ", " "]
    if _estimate_tokens(text) <= max_tokens:
        return [text]
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks: list[str] = []
            current = ""
            for part in parts:
                candidate = (current + sep + part) if current else part
                if _estimate_tokens(candidate) > max_tokens and current:
                    chunks.append(current.strip())
                    current = part
                else:
                    current = candidate
            if current.strip():
                chunks.append(current.strip())
            # recurse on any piece still too big
            out: list[str] = []
            for c in chunks:
                out.extend(_recursive_split(c, max_tokens) if _estimate_tokens(c) > max_tokens else [c])
            return out
    return [text[: max_tokens * 4]]  # hard cut as last resort


def _is_heading(line: Line, body_size: float) -> tuple[bool, str, str]:
    m = _SECTION_RE.match(line.text)
    if m and len(line.text) < 120:
        return True, m.group("num"), m.group("title").strip()
    if _CHAPTER_RE.match(line.text) and len(line.text) < 80:
        return True, "", line.text.strip()
    if line.bold and line.font_size >= body_size + 1.5 and len(line.text) < 80:
        return True, "", line.text.strip()
    return False, "", ""


@dataclass
class _Section:
    number: str
    title: str
    path: list[str]
    page_number: int
    lines: list[str] = field(default_factory=list)


def chunk_document(doc: ExtractedDoc) -> list[Chunk]:
    s = get_settings()
    body = doc.body_font_size
    now = datetime.now(timezone.utc).isoformat()

    sections: list[_Section] = []
    current = _Section(number="0", title="Preamble", path=["Preamble"], page_number=1)
    path_stack: list[tuple[str, str]] = []  # (number, title)

    for line in doc.lines:
        is_head, num, title = _is_heading(line, body)
        if is_head:
            if current.lines:
                sections.append(current)
            # maintain a hierarchical path from numbered depth
            if num:
                depth = num.count(".")
                path_stack = path_stack[:depth]
                path_stack.append((num, title))
            else:
                path_stack = [("", title)]
            path = [f"{n} {t}".strip() for n, t in path_stack]
            current = _Section(
                number=num or title[:24],
                title=title,
                path=path,
                page_number=line.page_number,
            )
        else:
            current.lines.append(line.text)
    if current.lines:
        sections.append(current)

    chunks: list[Chunk] = []
    base = doc.source_document.replace(" ", "_")

    for sec in sections:
        text = (f"{sec.number} {sec.title}\n".strip() + "\n" + "\n".join(sec.lines)).strip()
        if not text:
            continue
        section_path = " > ".join(sec.path) if sec.path else sec.title
        pieces = (
            [text]
            if _estimate_tokens(text) <= s.MAX_SECTION_TOKENS
            else _recursive_split(text, s.MAX_SECTION_TOKENS)
        )
        for i, piece in enumerate(pieces):
            suffix = f"#{i}" if len(pieces) > 1 else ""
            import hashlib
            content_hash = hashlib.md5(piece.encode()).hexdigest()[:8]
            cid = f"{base}::{sec.page_number}::{sec.number}::{i}::{content_hash}"
            # cid = f"{base}::{sec.number}{suffix}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=piece,
                    metadata={
                        "source_document": doc.source_document,
                        "section_path": section_path,
                        "page_number": int(sec.page_number),
                        "chunk_type": "rule" if _looks_like_rule(piece) else "prose",
                        "section_number": sec.number,
                        "extraction_date": now,
                        "chunk_id": cid,
                    },
                )
            )

    # Tables become their own chunks.
    for t_index, table in enumerate(doc.tables):
        content_hash = hashlib.md5(table.text.encode()).hexdigest()[:8]
        cid = f"{base}::table_{table.page_number}_{t_index}::{content_hash}"
        # cid = f"{base}::table_{table.page_number}_{t_index}"
        chunks.append(
            Chunk(
                chunk_id=cid,
                text=table.text,
                metadata={
                    "source_document": doc.source_document,
                    "section_path": f"Table (page {table.page_number})",
                    "page_number": int(table.page_number),
                    "chunk_type": "table",
                    "section_number": f"T{table.page_number}.{t_index}",
                    "extraction_date": now,
                    "chunk_id": cid,
                },
            )
        )

    return chunks


def _looks_like_rule(text: str) -> bool:
    keywords = ("must", "shall", "minimum", "maximum", "eligible", "required", "not permitted")
    low = text.lower()
    return any(k in low for k in keywords)
