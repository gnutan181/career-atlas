from app.deep_researcher import agent
from app.deep_researcher.schemas import GapIn, Milestone, Pathway


def test_node_structure_retries_schema_validation_failure(monkeypatch):
    pathway = Pathway(
        target_role="Backend Engineer",
        rationale="Build foundations before advanced topics.",
        milestones=[
            Milestone(
                phase="Foundations",
                skill="Python",
                estimated_weeks=2,
                objective="Learn Python fundamentals.",
                resources=[],
                checklist=["Read", "Practice", "Review"],
                mini_project="Build a command-line tool.",
            )
        ],
    )
    temperatures = []

    class _Chain:
        def invoke(self, _input):
            if len(temperatures) == 1:
                raise RuntimeError("json_validate_failed")
            return pathway

    def fake_build(_prompt, _schema, *, temperature, strict_json_schema):
        assert strict_json_schema is True
        temperatures.append(temperature)
        return _Chain()

    monkeypatch.setattr(agent, "build_groq_structured_chain", fake_build)

    result = agent.node_structure(
        {
            "target_role": "Backend Engineer",
            "gaps": [GapIn(skill="Python", category="language", relevance=10, difficulty="easy")],
        }
    )

    assert result["pathway"] == pathway
    assert temperatures == [0.2, 0.0]


def test_recover_failed_pathway_fills_omitted_required_fields():
    class _SchemaFailure(Exception):
        body = {
            "error": {
                "failed_generation": """
                {
                  "target_role": "Backend Engineer",
                  "milestones": [{
                    "phase": "Foundations",
                    "skill": "Python",
                    "estimated_weeks": 2,
                    "objective": "Learn Python.",
                    "resources": [{
                      "title": "Python docs", "kind": "doc", "provider": "Python",
                      "url": "https://docs.python.org", "why": "Official reference."
                    }]
                  }]
                }
                """
            }
        }

    result = agent._recover_failed_pathway(
        _SchemaFailure("json_validate_failed"),
        {
            "target_role": "Backend Engineer",
            "gaps": [
                GapIn(skill="Python", category="language", relevance=10, difficulty="easy"),
                GapIn(skill="Testing", category="concept", relevance=8, difficulty="medium"),
            ],
        },
    )

    assert result is not None
    assert result.rationale
    assert [m.skill for m in result.milestones] == ["Python", "Testing"]
    assert result.milestones[0].checklist
    assert result.milestones[0].mini_project
    assert result.milestones[0].resources[0].url == "https://docs.python.org"
