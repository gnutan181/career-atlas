"""Run the 120-case retrieval release gate against configured services.

Usage: uv run python scripts/run_retrieval_benchmark.py
Exits non-zero when Recall@10 < .80 or nDCG@10 < .72.
"""
from __future__ import annotations

import asyncio

from app.evaluation.retrieval_benchmark import evaluate, load_cases
from app.gap_analysis.hybrid_retrieval import hybrid_retrieve


async def main() -> None:
    rankings = []
    for case in load_cases():
        results = await hybrid_retrieve(list(case.known_skills), case.target_role_title)
        rankings.append((case, [row["skill_name"] for row in results]))
    result = evaluate(rankings)
    print(
        f"cases={result.cases} Recall@10={result.recall_at_10:.3f} "
        f"nDCG@10={result.ndcg_at_10:.3f} passed={result.passes}"
    )
    if not result.passes:
        raise SystemExit("Retrieval release gate failed")


if __name__ == "__main__":
    asyncio.run(main())
