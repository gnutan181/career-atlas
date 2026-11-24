"""Azure AI Content Safety — Prompt Shields (document attack detection).

API (verified 2026-11 against learn.microsoft.com):
  POST {endpoint}/contentsafety/text:shieldPrompt?api-version=2024-09-01
  header  Ocp-Apim-Subscription-Key
  body    {"userPrompt": str, "documents": [str, ...]}
  result  {"userPromptAnalysis": {"attackDetected": bool},
           "documentsAnalysis": [{"attackDetected": bool}, ...]}
  limits  userPrompt <= 10K chars; <= 5 documents totalling <= 10K chars.

Untrusted text is split into overlapping chunks of at most 2,000 characters and
packed five per request, so every request stays inside the documented limits.
Each named item (a resume, one GitHub file) gets its own verdict.

Callers decide policy: resumes fail closed, GitHub files are skipped.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Iterable

import httpx

from app.config import settings
from app.security.guard_log import log_guard, timed

API_VERSION = "2024-09-01"
MAX_DOC_CHARS = 2000
MAX_DOCS_PER_REQUEST = 5
CHUNK_OVERLAP = 200
_RETRY_STATUS = {429, 500, 502, 503, 504}
# Our own (trusted) task statement.  The API requires a userPrompt; the
# untrusted content goes only in `documents`.
_USER_PROMPT = "Extract factual career information from the attached documents."


class PromptShieldError(RuntimeError):
    """Prompt Shields could not produce a verdict (timeout, HTTP error, config)."""


class ResumeInjectionDetected(ValueError):
    """The resume contains text that Prompt Shields classifies as an attack."""


@dataclass(frozen=True)
class ShieldVerdict:
    attack_detected: bool
    checked: bool = True
    error: str | None = None


def is_enabled() -> bool:
    return bool(settings.prompt_shields_enabled)


def chunk_text(text: str, size: int = MAX_DOC_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split into chunks <= size chars, preferring paragraph/line boundaries.
    Overlap keeps an instruction that straddles a boundary visible in one chunk."""
    text = text or ""
    if len(text) <= size:
        return [text] if text.strip() else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:
                end = cut
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c.strip()]


def _pack(items: dict[str, str]) -> list[list[tuple[str, str]]]:
    """[(item_key, chunk), ...] batches of at most 5 docs / 10K chars."""
    pairs = [(key, chunk) for key, text in items.items() for chunk in chunk_text(text)]
    return [pairs[i:i + MAX_DOCS_PER_REQUEST] for i in range(0, len(pairs), MAX_DOCS_PER_REQUEST)]


def _request_parts() -> tuple[str, dict[str, str]]:
    endpoint = settings.azure_content_safety_endpoint.rstrip("/")
    key = settings.azure_content_safety_key
    if not endpoint or not key:
        raise PromptShieldError("not_configured")
    url = f"{endpoint}/contentsafety/text:shieldPrompt?api-version={API_VERSION}"
    return url, {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"}


def _parse(response: httpx.Response, expected: int) -> list[bool]:
    data = response.json()
    analyses = data.get("documentsAnalysis")
    if not isinstance(analyses, list) or len(analyses) != expected:
        raise PromptShieldError("malformed_response")
    return [bool(a.get("attackDetected")) for a in analyses]


def _retry_after(response: httpx.Response | None) -> float:
    if response is None:
        return 1.0
    try:
        return min(max(float(response.headers.get("Retry-After", "1")), 0.0), 5.0)
    except ValueError:
        return 1.0


def _merge(items: dict[str, str], batches: list[list[tuple[str, str]]],
           outcomes: list[list[bool] | str]) -> dict[str, ShieldVerdict]:
    attacked: dict[str, bool] = {key: False for key in items}
    errors: dict[str, str] = {}
    for batch, outcome in zip(batches, outcomes):
        for idx, (key, _) in enumerate(batch):
            if isinstance(outcome, str):
                errors.setdefault(key, outcome)
            elif outcome[idx]:
                attacked[key] = True
    verdicts: dict[str, ShieldVerdict] = {}
    for key in items:
        if attacked[key]:
            verdicts[key] = ShieldVerdict(attack_detected=True)  # an attack outranks a partial error
        elif key in errors:
            verdicts[key] = ShieldVerdict(attack_detected=False, checked=False, error=errors[key])
        else:
            verdicts[key] = ShieldVerdict(attack_detected=False)
    return verdicts


def _log(source: str, verdicts: dict[str, ShieldVerdict], latency_ms: float, requests: int) -> None:
    blocked = sum(v.attack_detected for v in verdicts.values())
    errored = sum(1 for v in verdicts.values() if v.error)
    outcome = "block" if blocked else "error" if errored else "pass"
    log_guard("prompt_shields", outcome, latency_ms, source=source, items=len(verdicts),
              blocked=blocked, errored=errored, requests=requests)


def shield_documents(items: dict[str, str], source: str) -> dict[str, ShieldVerdict]:
    """Synchronous check of several named texts.  Never raises for HTTP errors;
    an item that could not be checked gets ``checked=False`` + ``error``."""
    if not is_enabled():
        return {key: ShieldVerdict(attack_detected=False, checked=False, error="disabled") for key in items}
    with timed() as elapsed:
        batches = _pack(items)
        try:
            url, headers = _request_parts()
        except PromptShieldError as exc:
            verdicts = {key: ShieldVerdict(False, False, str(exc)) for key in items}
            _log(source, verdicts, elapsed(), 0)
            return verdicts
        outcomes: list[list[bool] | str] = []
        with httpx.Client(timeout=settings.prompt_shields_timeout_seconds) as client:
            for batch in batches:
                outcomes.append(_send_sync(client, url, headers, [c for _, c in batch]))
        verdicts = _merge(items, batches, outcomes)
        _log(source, verdicts, elapsed(), len(batches))
        return verdicts


def _send_sync(client: httpx.Client, url: str, headers: dict[str, str], docs: list[str]) -> list[bool] | str:
    payload = {"userPrompt": _USER_PROMPT, "documents": docs}
    for attempt in range(2):  # one retry on transient errors
        response = None
        try:
            response = client.post(url, headers=headers, json=payload)
            if response.status_code in _RETRY_STATUS and attempt == 0:
                time.sleep(_retry_after(response))
                continue
            if response.status_code != 200:
                return f"http_{response.status_code}"
            return _parse(response, len(docs))
        except httpx.TimeoutException:
            if attempt == 0:
                continue
            return "timeout"
        except (httpx.HTTPError, ValueError, PromptShieldError) as exc:
            if attempt == 0 and isinstance(exc, httpx.TransportError):
                time.sleep(_retry_after(response))
                continue
            return type(exc).__name__[:48]
    return "retry_exhausted"


async def ashield_documents(items: dict[str, str], source: str) -> dict[str, ShieldVerdict]:
    """Async twin of :func:`shield_documents`."""
    if not is_enabled():
        return {key: ShieldVerdict(attack_detected=False, checked=False, error="disabled") for key in items}
    with timed() as elapsed:
        batches = _pack(items)
        try:
            url, headers = _request_parts()
        except PromptShieldError as exc:
            verdicts = {key: ShieldVerdict(False, False, str(exc)) for key in items}
            _log(source, verdicts, elapsed(), 0)
            return verdicts
        outcomes: list[list[bool] | str] = []
        async with httpx.AsyncClient(timeout=settings.prompt_shields_timeout_seconds) as client:
            for batch in batches:
                outcomes.append(await _send_async(client, url, headers, [c for _, c in batch]))
        verdicts = _merge(items, batches, outcomes)
        _log(source, verdicts, elapsed(), len(batches))
        return verdicts


async def _send_async(client: httpx.AsyncClient, url: str, headers: dict[str, str], docs: list[str]) -> list[bool] | str:
    payload = {"userPrompt": _USER_PROMPT, "documents": docs}
    for attempt in range(2):
        response = None
        try:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code in _RETRY_STATUS and attempt == 0:
                await asyncio.sleep(_retry_after(response))
                continue
            if response.status_code != 200:
                return f"http_{response.status_code}"
            return _parse(response, len(docs))
        except httpx.TimeoutException:
            if attempt == 0:
                continue
            return "timeout"
        except (httpx.HTTPError, ValueError, PromptShieldError) as exc:
            if attempt == 0 and isinstance(exc, httpx.TransportError):
                await asyncio.sleep(_retry_after(response))
                continue
            return type(exc).__name__[:48]
    return "retry_exhausted"


def check_resume(masked_text: str) -> None:
    """Resume policy: attack → ResumeInjectionDetected; no verdict → PromptShieldError
    (fail closed).  No-op when Prompt Shields is disabled."""
    if not is_enabled():
        return
    verdict = shield_documents({"resume": masked_text}, "resume")["resume"]
    if verdict.attack_detected:
        raise ResumeInjectionDetected("resume_prompt_injection")
    if not verdict.checked:
        raise PromptShieldError(verdict.error or "unchecked")


def iter_flagged(verdicts: dict[str, ShieldVerdict]) -> Iterable[tuple[str, str]]:
    """(key, reason) for every item that must be skipped (GitHub policy)."""
    for key, verdict in verdicts.items():
        if verdict.attack_detected:
            yield key, "prompt_injection_detected"
        elif not verdict.checked and verdict.error != "disabled":
            yield key, "safety_check_unavailable"
