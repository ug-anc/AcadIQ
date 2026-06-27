"""Adversarial suite (PRD section 08.2).

Every adversarial category must resolve to a safe non-answer: either NOT-FOUND
from the confidence gate / citation contract, or an injection rejection. A
hallucinated answer to any of these is a hard system defect.

Run as part of pytest, or standalone:  python -m tests.adversarial_suite
"""

import asyncio
import json

import pytest

from app.models.schemas import looks_like_injection
from app.services.generation import answer_query

with open("eval/golden_set.json") as _f:
    _ADVERSARIAL = json.load(_f)["adversarial"]


async def _is_safe(item: dict) -> bool:
    if item["category"] == "prompt_injection":
        # The input guard rejects these before generation.
        return looks_like_injection(item["query"])
    resp = await answer_query(item["query"])
    return resp.is_found is False


@pytest.mark.asyncio
@pytest.mark.parametrize("item", _ADVERSARIAL, ids=[i["category"] for i in _ADVERSARIAL])
async def test_adversarial_query_is_safe(item):
    assert await _is_safe(item), f"adversarial leak on: {item['query']}"


def _main() -> None:
    async def run():
        passed = 0
        for item in _ADVERSARIAL:
            ok = await _is_safe(item)
            print(f"  [{'PASS' if ok else 'FAIL'}] {item['category']}: {item['query']}")
            passed += ok
        print(f"\n{passed}/{len(_ADVERSARIAL)} adversarial queries handled safely.")

    asyncio.run(run())


if __name__ == "__main__":
    _main()
