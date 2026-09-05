"""Deterministic 120-case benchmark for the curated role taxonomy."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

RECALL_AT_10_GATE = 0.80
NDCG_AT_10_GATE = 0.72
CASES_PER_ROLE = 15
_TAXONOMY_PATH = Path(__file__).parents[1] / "gap_analysis" / "taxonomy.json"
_ROLE_TITLES = {
    "ml_engineer": "Machine Learning Engineer",
    "data_scientist": "Data Scientist",
    "data_analyst": "Data Analyst",
    "swe_backend": "Backend Engineer",
    "devops_engineer": "DevOps Engineer",
    "frontend_engineer": "Frontend Engineer",
    "fullstack_engineer": "Full-Stack Engineer",
    "product_manager": "Associate Product Manager",
}


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    target_role_title: str
    known_skills: tuple[str, ...]
    relevant_skills: frozenset[str]


@dataclass(frozen=True)
class BenchmarkResult:
    cases: int
    recall_at_10: float
    ndcg_at_10: float

    @property
    def passes(self) -> bool:
        return self.recall_at_10 >= RECALL_AT_10_GATE and self.ndcg_at_10 >= NDCG_AT_10_GATE


def load_cases() -> list[RetrievalCase]:
    """Create 15 fixed profile variants for each of eight curated roles.

    Each case holds out one relevant taxonomy skill and varies the known-skill
    context.  The taxonomy is repository-owned and reviewed, making these 120
    fixtures reproducible without a production database snapshot.
    """
    taxonomy = json.loads(_TAXONOMY_PATH.read_text(encoding="utf-8"))
    by_role: dict[str, list[str]] = {}
    for row in taxonomy:
        role = row["role"]
        if role in _ROLE_TITLES:
            by_role.setdefault(role, []).append(row["skill_name"])

    cases: list[RetrievalCase] = []
    for role, title in _ROLE_TITLES.items():
        skills = by_role.get(role, [])
        if not skills:
            raise ValueError(f"Curated benchmark role {role!r} has no taxonomy rows")
        for index in range(CASES_PER_ROLE):
            expected = skills[index % len(skills)]
            # Context mimics a partly complete profile while keeping the target
            # skill absent, so it remains a valid gap-retrieval expectation.
            known = tuple(skill for offset, skill in enumerate(skills) if skill != expected and offset % 3 == index % 3)[:4]
            cases.append(RetrievalCase(
                id=f"{role}-{index + 1:02d}",
                target_role_title=title,
                known_skills=known,
                relevant_skills=frozenset({expected}),
            ))
    if len(cases) != 120:
        raise AssertionError(f"Expected 120 benchmark cases, found {len(cases)}")
    return cases


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int = 10) -> float:
    if not relevant:
        return 1.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 1.0


def evaluate(rankings: Iterable[tuple[RetrievalCase, Sequence[str]]]) -> BenchmarkResult:
    rows = list(rankings)
    if len(rows) != 120:
        raise ValueError(f"Benchmark must receive all 120 cases, received {len(rows)}")
    return BenchmarkResult(
        cases=len(rows),
        recall_at_10=sum(recall_at_k(ranking, case.relevant_skills) for case, ranking in rows) / len(rows),
        ndcg_at_10=sum(ndcg_at_k(ranking, case.relevant_skills) for case, ranking in rows) / len(rows),
    )
