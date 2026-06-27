"""DR-04 confidence-threshold calibration.

Records the top reranker relevance score for answerable vs. adversarial queries,
then sweeps candidate thresholds and reports the F1-maximizing operating point.
Use the printed hard/soft values to set CONFIDENCE_HARD_REJECT and
CONFIDENCE_SOFT_WARN in .env.

Usage:  python -m eval.calibrate_threshold
"""

import asyncio
import json

from app.services.retrieval import get_retrieval_engine

GOLDEN = "eval/golden_set.json"


async def _top_scores() -> tuple[list[float], list[float]]:
    with open(GOLDEN) as f:
        golden = json.load(f)
    engine = get_retrieval_engine()

    answerable: list[float] = []
    for item in golden["answerable"]:
        chunks = await engine.retrieve(item["query"])
        answerable.append(chunks[0].score if chunks else 0.0)

    unanswerable: list[float] = []
    for item in golden["adversarial"]:
        if item["category"] == "prompt_injection":
            continue  # handled by the injection guard, not the gate
        chunks = await engine.retrieve(item["query"])
        unanswerable.append(chunks[0].score if chunks else 0.0)

    return answerable, unanswerable


def _sweep(pos: list[float], neg: list[float]) -> None:
    print(f"answerable top-scores:   {[round(x, 3) for x in sorted(pos)]}")
    print(f"unanswerable top-scores: {[round(x, 3) for x in sorted(neg)]}")
    print("\nthreshold |  precision  recall    F1")
    best = (0.0, -1.0)
    t = 0.10
    while t <= 0.90 + 1e-9:
        tp = sum(1 for s in pos if s >= t)
        fp = sum(1 for s in neg if s >= t)
        fn = sum(1 for s in pos if s < t)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"   {t:0.2f}   |   {precision:0.2f}      {recall:0.2f}    {f1:0.2f}")
        if f1 > best[1]:
            best = (t, f1)
        t += 0.05

    soft = round(best[0], 2)
    hard = round(max(0.0, soft - 0.2), 2)
    print(f"\nF1-optimal soft-warn threshold: {soft}")
    print(f"Suggested hard-reject threshold: {hard}")
    print("Set CONFIDENCE_SOFT_WARN and CONFIDENCE_HARD_REJECT accordingly.")


def main() -> None:
    pos, neg = asyncio.run(_top_scores())
    _sweep(pos, neg)


if __name__ == "__main__":
    main()
