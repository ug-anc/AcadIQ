"""Run a few sample queries through the full pipeline (no server needed).

Usage:  python -m app.scripts.demo_query
"""

import asyncio

from app.services.generation import answer_query

QUERIES = [
    "What is the minimum attendance requirement?",
    "max credits?",  # short query -> exercises BM25 + rewrite path
    "branch change CGPA?",
    "What is the fee structure for hostel rooms?",  # adversarial: not in docs
    "What will the CGPA cutoff be next year?",  # adversarial: speculative
]


async def _run() -> None:
    for q in QUERIES:
        resp = await answer_query(q, session_id="demo")
        print("\n" + "=" * 72)
        print(f"Q: {q}")
        print(f"   band={resp.confidence_band} score={resp.confidence_score} "
              f"found={resp.is_found}")
        print("-" * 72)
        print(resp.answer)
        if resp.citations:
            print("   citations:",
                  ", ".join(f"{c.document_name} §{c.section} p{c.page_number}"
                            for c in resp.citations))


if __name__ == "__main__":
    asyncio.run(_run())
