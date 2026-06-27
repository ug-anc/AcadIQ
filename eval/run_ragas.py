"""RAGAS evaluation harness (PRD section 08).

Builds the dataset by running answerable golden-set queries through the live
pipeline, then scores faithfulness / answer relevancy / context precision /
recall and asserts the PRD phase gates. Requires OPENAI_API_KEY (RAGAS uses an
LLM judge) and `pip install ragas datasets`.

Usage:  python -m eval.run_ragas
"""

import asyncio
import json

from app.services.generation import answer_query
from app.services.retrieval import get_retrieval_engine

GOLDEN = "eval/golden_set.json"


async def _build_rows() -> list[dict]:
    with open(GOLDEN) as f:
        golden = json.load(f)
    engine = get_retrieval_engine()
    rows: list[dict] = []
    for item in golden["answerable"]:
        q = item["query"]
        contexts = [c.text for c in await engine.retrieve(q)]
        resp = await answer_query(q)
        rows.append(
            {
                "question": q,
                "answer": resp.answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            }
        )
    return rows


def main() -> None:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    rows = asyncio.run(_build_rows())
    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    print(result)

    assert result["faithfulness"] >= 0.95, "GATE FAIL: faithfulness < 0.95"
    assert result["context_precision"] >= 0.90, "GATE FAIL: context_precision < 0.90"
    print("All RAGAS phase gates passed.")


if __name__ == "__main__":
    main()
