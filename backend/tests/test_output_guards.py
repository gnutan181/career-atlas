"""Guardrails AI output validation.  LLM calls are mocked; validators run locally."""
import asyncio
import json

import pytest

from app.deep_researcher import agent
from app.deep_researcher.schemas import GapIn, Milestone, Pathway
from app.gap_analysis import service as gap_service
from app.gap_analysis.schemas import GapAnalysisResponse, GapSchema
from app.resume_extraction import service as resume_service
from app.security import output_guards as og


@pytest.fixture(autouse=True)
def guards_on(monkeypatch):
    monkeypatch.setattr(og.settings, "output_guards_enabled", True)
    monkeypatch.setattr(og.settings, "toxicity_guard_enabled", False)
    monkeypatch.setattr(og.settings, "prompt_shields_enabled", False)


# ── unsupported "verified" skill claims ─────────────────────────────────────

def test_unsupported_verified_skill_is_flagged():
    text = "Your verified Kubernetes experience makes this role a natural next step."
    assert og.find_unsupported_verified_claims(text, ["Python"], ["Kubernetes", "Python"]) == ["kubernetes"]


def test_supported_and_negated_claims_pass():
    vocab = ["Kubernetes", "Python"]
    assert og.find_unsupported_verified_claims("Your Python skills are verified by GitHub.", ["Python"], vocab) == []
    assert og.find_unsupported_verified_claims("Kubernetes is not yet verified.", ["Python"], vocab) == []
    assert og.find_unsupported_verified_claims("Learn Kubernetes next.", ["Python"], vocab) == []


def test_validate_texts_rejects_unsupported_verified_claim():
    with pytest.raises(og.OutputGuardError) as err:
        og.validate_texts(["You have proven Docker expertise."], ["Python"], ["Docker"], "gap_analysis_output")
    assert "Docker" not in str(err.value)  # errors are content-free


def test_validate_texts_rejects_profanity_and_passes_clean_text():
    with pytest.raises(og.OutputGuardError):
        og.validate_texts(["This roadmap is for idiots, you fucking loser."], [], [], "roadmap_output")
    og.validate_texts(["Learn Docker fundamentals, then deploy a FastAPI service."], [], ["Docker"], "roadmap_output")


def test_guards_disabled_is_a_no_op(monkeypatch):
    monkeypatch.setattr(og.settings, "output_guards_enabled", False)
    og.validate_texts(["you fucking loser"], [], [], "roadmap_output")


def test_guardrails_hub_telemetry_is_disabled():
    from guardrails.utils.hub_telemetry_utils import HubTelemetry

    og.validate_texts(["Learn Docker."], [], [], "roadmap_output")
    assert HubTelemetry()._enabled is False


# ── re-ask then safe error ──────────────────────────────────────────────────

def test_with_reask_retries_once_then_raises():
    calls = []

    def generate(attempt):
        calls.append(attempt)
        return "bad"

    def validate(_result):
        raise og.OutputGuardError("check")

    with pytest.raises(og.OutputGuardError):
        og.with_reask(generate, validate, "check")
    assert calls == [0, 1]


def test_with_reask_returns_second_answer_when_valid():
    answers = iter(["bad", "good"])

    def validate(result):
        if result == "bad":
            raise og.OutputGuardError("check")

    assert og.with_reask(lambda _a: next(answers), validate, "check") == "good"


# ── roadmap (deep researcher structurer) ────────────────────────────────────

def _pathway(objective: str) -> Pathway:
    return Pathway(
        target_role="Backend Engineer",
        rationale="Foundations first.",
        milestones=[
            Milestone(
                phase="Foundations", skill="Kubernetes", estimated_weeks=2, objective=objective,
                resources=[], checklist=["Install minikube", "Deploy a pod", "Expose a service"],
                mini_project="Deploy a FastAPI app on Kubernetes.",
            )
        ],
    )


def _structure_with(monkeypatch, pathways):
    queue = iter(pathways)

    class _Chain:
        def invoke(self, _input):
            return next(queue)

    monkeypatch.setattr(agent, "build_groq_structured_chain", lambda *_a, **_k: _Chain())
    return agent.node_structure({
        "target_role": "Backend Engineer",
        "gaps": [GapIn(skill="Kubernetes", category="tool", relevance=80, difficulty="Medium")],
        "evidence_skills": ["Python", "FastAPI"],
    })


def test_toxic_roadmap_is_rejected_after_one_reask(monkeypatch):
    toxic = _pathway("Stop being a lazy idiot and learn this shit.")
    with pytest.raises(og.OutputGuardError):
        _structure_with(monkeypatch, [toxic, toxic])


def test_roadmap_reask_recovers_with_clean_output(monkeypatch):
    bad = _pathway("Build on your verified Kubernetes expertise.")  # not in evidence
    good = _pathway("Deploy and operate services on Kubernetes.")
    assert _structure_with(monkeypatch, [bad, good])["pathway"] == good


def test_deep_research_router_returns_safe_error(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.deep_researcher import router as dr_router
    from app.dependencies.auth import require_user_id

    app = FastAPI()
    app.include_router(dr_router.router)
    app.dependency_overrides[require_user_id] = lambda: "user-1"
    ctx = dr_router.ResearchContext("r", "Backend Engineer", "backend", "resume", [], [], ["Python"])
    monkeypatch.setattr(dr_router, "_prepare_research_context", lambda *_a: ctx)

    class _Agent:
        def invoke(self, *_a, **_k):
            raise og.OutputGuardError("roadmap_output")

    monkeypatch.setattr(dr_router, "deep_researcher_agent", _Agent())
    response = TestClient(app).post(dr_router.router.prefix + "/", json={"target_role_id": "r"})
    assert response.status_code == 502
    assert response.json()["detail"] == og.SAFE_ERROR_MESSAGE


# ── gap analysis ────────────────────────────────────────────────────────────

def _gap_response(why: str) -> GapAnalysisResponse:
    return GapAnalysisResponse(
        gaps=[GapSchema(skill="Kubernetes", category="tool", relevance=80, difficulty="Medium",
                        prerequisites=["Docker"], why=why)],
        justifications={"Kubernetes": "Needed to run services in production."},
    )


def _run_gap(monkeypatch, responses):
    queue = iter(responses)
    calls = []

    async def fake_ainvoke(**_kwargs):
        calls.append(1)
        return next(queue)

    async def fake_taxonomy(_role):
        return "curated"

    async def fake_retrieve(**_kwargs):
        return [{"skill_name": "Kubernetes", "relevance_score": 0.3}]

    monkeypatch.setattr(gap_service, "ainvoke_gemini", fake_ainvoke)
    monkeypatch.setattr(gap_service, "ensure_role_taxonomy", fake_taxonomy)
    monkeypatch.setattr(gap_service, "hybrid_retrieve", fake_retrieve)
    result = asyncio.run(gap_service.generate_gaps_for_user(["Python", "Docker"], "Backend Engineer"))
    return result, calls


def test_gap_analysis_rejects_unsupported_verified_claim_then_safe_error(monkeypatch):
    bad = _gap_response("Your verified Kubernetes skills already cover this.")
    with pytest.raises(og.OutputGuardError):
        _run_gap(monkeypatch, [bad, bad])


def test_gap_analysis_reask_returns_clean_result(monkeypatch):
    bad = _gap_response("Your verified Kubernetes skills already cover this.")
    good = _gap_response("Production backends run on Kubernetes.")
    (gaps, _explain), calls = _run_gap(monkeypatch, [bad, good])
    assert len(calls) == 2 and gaps[0].why == good.gaps[0].why


# ── profile extraction: only the candidate's own contact PII ───────────────

def _llm_returning(monkeypatch, outputs):
    queue = iter(outputs)
    prompts = []

    def fake(prompt):
        prompts.append(prompt)
        return json.dumps(next(queue))

    monkeypatch.setattr(resume_service, "llm_generate_resume_json", fake)
    return prompts


def _profile(tokens, bullet):
    return {
        "full_name": tokens["name"],
        "contact": {"email": tokens["email"], "phone_raw": tokens["phone"], "location": "Bengaluru"},
        "skills": ["Python"],
        "experience": [{"company": "Infosys", "title": "Engineer", "description_bullets": [bullet]}],
    }


RESUME = (
    "# Priya Sharma\npriya.sharma@example.com | +91 98450 12345\n"
    "Infosys - Engineer. Built Python services. Reference: Rahul Verma\n"
)


def _tokens(monkeypatch):
    from app.security import pii

    monkeypatch.setattr(pii.settings, "pii_engine", "presidio")
    _masked, replacements = resume_service.redact_pii(RESUME)
    by_value = {v: t for t, v in replacements.items()}
    return {
        "name": by_value.get("Priya Sharma", "Priya Sharma"),
        "email": by_value["priya.sharma@example.com"],
        "phone": by_value["+91 98450 12345"],
        "other": by_value.get("Rahul Verma"),
    }


def test_profile_restores_only_candidate_contact_and_redacts_third_parties(monkeypatch):
    tokens = _tokens(monkeypatch)
    bullet = f"Built Python services with {tokens['other'] or 'a colleague'}."
    _llm_returning(monkeypatch, [_profile(tokens, bullet)])
    profile = resume_service.extract_structured_resume_data(RESUME)
    assert profile.full_name == "Priya Sharma"
    assert profile.contact.email == "priya.sharma@example.com"
    assert profile.contact.phone_raw == "+91 98450 12345"
    assert "Rahul Verma" not in profile.model_dump_json()


def test_profile_with_extra_pii_is_reasked_then_safe_error(monkeypatch):
    tokens = _tokens(monkeypatch)
    leaky = _profile(tokens, "Escalations: ops-team@example.com, Aadhaar 2341 2341 2346")
    prompts = _llm_returning(monkeypatch, [leaky, leaky])
    with pytest.raises(og.OutputGuardError):
        resume_service.extract_structured_resume_data(RESUME)
    assert len(prompts) == 2
    assert "outside the candidate's own full_name and contact fields" in prompts[1]
    assert "2341 2341 2346" not in prompts[1].split("Previous JSON:")[0]


def test_profile_reask_recovers(monkeypatch):
    tokens = _tokens(monkeypatch)
    leaky = _profile(tokens, "Escalations: ops-team@example.com")
    clean = _profile(tokens, "Built Python services.")
    _llm_returning(monkeypatch, [leaky, clean])
    profile = resume_service.extract_structured_resume_data(RESUME)
    assert profile.experience[0].description_bullets == ["Built Python services."]
