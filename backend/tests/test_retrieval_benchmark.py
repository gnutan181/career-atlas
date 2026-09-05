from app.evaluation.retrieval_benchmark import (
    NDCG_AT_10_GATE,
    RECALL_AT_10_GATE,
    evaluate,
    load_cases,
)


def test_curated_benchmark_contains_120_cases():
    cases = load_cases()
    assert len(cases) == 120
    assert len({case.id for case in cases}) == 120
    assert all(case.relevant_skills.isdisjoint(case.known_skills) for case in cases)


def test_perfect_rankings_pass_release_gates():
    cases = load_cases()
    result = evaluate([(case, list(case.relevant_skills)) for case in cases])
    assert result.recall_at_10 == 1.0
    assert result.ndcg_at_10 == 1.0
    assert result.recall_at_10 >= RECALL_AT_10_GATE
    assert result.ndcg_at_10 >= NDCG_AT_10_GATE
