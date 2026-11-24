"""LLM output validation with Guardrails AI.

* Profile extraction: the existing ``ResumeExtraction`` Pydantic schema
  (``Guard.for_pydantic``) plus ``OnlyCandidatePII`` — outside the candidate's
  own contact fields, the profile may not contain any other email, phone,
  street address or Indian ID.
* Gap analysis / roadmap text: ``ProfanityFree`` (Hub, local model), optional
  ``ToxicLanguage`` (Hub, local BERT; only when installed and
  ``TOXICITY_GUARD_ENABLED``), and ``EvidenceBackedVerifiedSkills`` which
  flags "verified/proven/demonstrated <skill>" claims for skills that the
  resume or confirmed GitHub evidence does not contain.

Policy everywhere: validate → on failure re-ask the LLM once → validate →
otherwise raise :class:`OutputGuardError` (callers turn it into a safe error).
Error messages and logs never contain the validated content.

Guardrails Hub telemetry is switched off for every Guard built here: the
library otherwise enables it by default (``RC.enable_metrics`` defaults to
True and ``~/.guardrailsrc`` may set it) and exports spans to a remote
collector.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any, Awaitable, Callable, Iterable, TypeVar

from app.config import settings
from app.security.guard_log import log_guard, timed

logger = logging.getLogger(__name__)

T = TypeVar("T")

SAFE_ERROR_MESSAGE = "The AI response did not pass our safety checks. Please try again."

# Entities that may never appear in a profile outside the candidate's own contact block.
_PROFILE_PII_ENTITIES = [
    "EMAIL_ADDRESS", "PHONE_NUMBER", "STREET_ADDRESS",
    "IN_PAN", "IN_AADHAAR", "IN_PASSPORT", "IN_VOTER",
]
_EMAIL_FALLBACK = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

_CLAIM_VERB = re.compile(
    r"\b(verified|proven|demonstrated|confirmed|validated|certified|evidenced)\b", re.IGNORECASE
)
_NEGATION = re.compile(r"\b(not|no|never|unverified|without|lacks?|missing|yet to|isn't|aren't|hasn't|haven't)\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;\n])\s+")


class OutputGuardError(RuntimeError):
    """An LLM output failed validation (after the allowed re-ask)."""

    def __init__(self, check: str, reason: str = "failed"):
        super().__init__(f"{check}: {reason}")
        self.check = check
        self.reason = reason


# ── Guardrails plumbing ─────────────────────────────────────────────────────

def _quiet(guard):
    """Disable Hub telemetry on a Guard (and globally for this process)."""
    from guardrails.settings import settings as gr_settings

    guard.configure(allow_metrics_collection=False)
    try:
        gr_settings.rc.enable_metrics = False
    except Exception:  # pragma: no cover - defensive, attribute layout may change
        pass
    return guard


@lru_cache(maxsize=1)
def _validators():
    """Register custom validators once (Guardrails keeps a global registry)."""
    from guardrails.validator_base import FailResult, PassResult, Validator, register_validator

    @register_validator(name="careeratlas/only_candidate_pii", data_type="string")
    class OnlyCandidatePII(Validator):
        """Fails when text holds PII that is not the candidate's own contact data.

        metadata: ``allowed_values`` (candidate name/email/phone) and
        ``forbidden_values`` (originals of masked non-candidate tokens)."""

        def _validate(self, value: Any, metadata: dict) -> Any:
            text = str(value or "")
            allowed = [a for a in metadata.get("allowed_values", []) if a]
            allowed_digits = {re.sub(r"\D", "", a) for a in allowed if re.sub(r"\D", "", a)}
            allowed_lower = {a.lower() for a in allowed}
            lowered = text.lower()
            for forbidden in metadata.get("forbidden_values", []):
                if forbidden and forbidden.lower() in lowered:
                    return FailResult(error_message="masked third-party value present")
            try:
                from app.security.pii import detect_pii

                spans = [(s.entity_type, text[s.start:s.end]) for s in detect_pii(text, entities=_PROFILE_PII_ENTITIES)]
            except Exception as exc:
                logger.warning("PII output check: Presidio unavailable (%s); email regex only", type(exc).__name__)
                spans = [("EMAIL_ADDRESS", m.group(0)) for m in _EMAIL_FALLBACK.finditer(text)]
            for entity, found in spans:
                digits = re.sub(r"\D", "", found)
                if found.lower() in allowed_lower:
                    continue
                if entity == "PHONE_NUMBER" and digits and any(
                    digits.endswith(d[-10:]) or d.endswith(digits[-10:]) for d in allowed_digits
                ):
                    continue
                return FailResult(error_message=f"unexpected {entity}")
            return PassResult()

    @register_validator(name="careeratlas/evidence_backed_verified_skills", data_type="string")
    class EvidenceBackedVerifiedSkills(Validator):
        """Fails when a sentence calls a skill verified/proven/demonstrated
        but that skill is not in ``metadata['evidence_skills']``.

        ``metadata['skill_vocabulary']`` lists the skill names to look for
        (gap skills, evidence skills, common technologies)."""

        def _validate(self, value: Any, metadata: dict) -> Any:
            unsupported = find_unsupported_verified_claims(
                str(value or ""),
                metadata.get("evidence_skills", []),
                metadata.get("skill_vocabulary", []),
            )
            if unsupported:
                return FailResult(error_message=f"{len(unsupported)} unsupported verified-skill claim(s)")
            return PassResult()

    return OnlyCandidatePII, EvidenceBackedVerifiedSkills


def _norm_skill(skill: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", (skill or "").lower()).strip()


def find_unsupported_verified_claims(
    text: str, evidence_skills: Iterable[str], skill_vocabulary: Iterable[str]
) -> list[str]:
    """Skill names claimed as verified in ``text`` without supporting evidence."""
    evidence = {_norm_skill(s) for s in evidence_skills if s}
    vocab = sorted({_norm_skill(s) for s in skill_vocabulary if s} | evidence, key=len, reverse=True)
    vocab = [v for v in vocab if v]
    unsupported: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text or ""):
        if not _CLAIM_VERB.search(sentence) or _NEGATION.search(sentence):
            continue
        norm_sentence = f" {_norm_skill(sentence)} "
        for skill in vocab:
            if f" {skill} " in norm_sentence:
                if skill not in evidence and skill not in unsupported:
                    unsupported.append(skill)
                norm_sentence = norm_sentence.replace(f" {skill} ", " ")
    return unsupported


@lru_cache(maxsize=1)
def _profile_schema_guard():
    from guardrails import Guard

    from app.resume_extraction.schemas import ResumeExtraction

    return _quiet(Guard.for_pydantic(ResumeExtraction))


@lru_cache(maxsize=1)
def _profile_pii_guard():
    from guardrails import Guard, OnFailAction

    only_pii, _ = _validators()
    return _quiet(Guard().use(only_pii(on_fail=OnFailAction.EXCEPTION)))


@lru_cache(maxsize=1)
def _text_guard():
    from guardrails import Guard, OnFailAction
    from guardrails_ai.profanity_free import ProfanityFree

    _, verified = _validators()
    validators = [ProfanityFree(on_fail=OnFailAction.EXCEPTION), verified(on_fail=OnFailAction.EXCEPTION)]
    if settings.toxicity_guard_enabled:
        try:
            from guardrails_ai.toxic_language import ToxicLanguage  # optional heavy dependency

            validators.append(ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail=OnFailAction.EXCEPTION))
        except ImportError:
            logger.warning("TOXICITY_GUARD_ENABLED but guardrails-ai-toxic-language is not installed")
    return _quiet(Guard().use(*validators))


def _run(guard, text: str, check: str, metadata: dict | None = None) -> None:
    """Run a Guard; translate any failure into a content-free OutputGuardError."""
    with timed() as elapsed:
        try:
            outcome = guard.validate(text, metadata=metadata or {})
            passed = bool(getattr(outcome, "validation_passed", False))
        except Exception as exc:  # guardrails raises ValidationError on EXCEPTION
            log_guard(check, "block", elapsed(), reason=type(exc).__name__[:64])
            raise OutputGuardError(check, "validation_failed") from None
        if not passed:
            log_guard(check, "block", elapsed(), reason="validation_not_passed")
            raise OutputGuardError(check, "validation_not_passed")
        log_guard(check, "pass", elapsed())


# ── Public checks ───────────────────────────────────────────────────────────

def validate_profile(profile: dict, allowed_values: list[str], forbidden_values: list[str]) -> None:
    """Schema + PII check for an extracted (restored) resume profile."""
    if not settings.output_guards_enabled:
        return
    _run(_profile_schema_guard(), json.dumps(profile, ensure_ascii=False), "profile_schema")
    # The contact block and full_name legitimately hold the candidate's details;
    # everything else is checked for any PII at all.
    rest = {k: v for k, v in profile.items() if k not in {"contact", "full_name"}}
    _run(
        _profile_pii_guard(),
        json.dumps(rest, ensure_ascii=False),
        "profile_pii",
        {"allowed_values": allowed_values, "forbidden_values": forbidden_values},
    )


def validate_texts(texts: Iterable[str], evidence_skills: Iterable[str],
                   skill_vocabulary: Iterable[str], check: str) -> None:
    """Profanity / toxicity / unsupported-verified-claim checks on free text."""
    if not settings.output_guards_enabled:
        return
    joined = "\n".join(t.strip() for t in texts if t and t.strip())
    if not joined:
        return
    _run(
        _text_guard(),
        joined,
        check,
        {"evidence_skills": list(evidence_skills), "skill_vocabulary": list(skill_vocabulary)},
    )


def with_reask(generate: Callable[[int], T], validate: Callable[[T], None], check: str) -> T:
    """generate(attempt) → validate; one re-ask on OutputGuardError; then a safe error."""
    for attempt in range(2):
        result = generate(attempt)
        try:
            validate(result)
            return result
        except OutputGuardError:
            if attempt == 0:
                log_guard(check, "reask", 0.0)
                continue
            raise
    raise OutputGuardError(check, "exhausted")  # pragma: no cover


async def awith_reask(generate: Callable[[int], Awaitable[T]], validate: Callable[[T], None], check: str) -> T:
    """Async twin of :func:`with_reask`."""
    for attempt in range(2):
        result = await generate(attempt)
        try:
            validate(result)
            return result
        except OutputGuardError:
            if attempt == 0:
                log_guard(check, "reask", 0.0)
                continue
            raise
    raise OutputGuardError(check, "exhausted")  # pragma: no cover
