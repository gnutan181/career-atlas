"""Content-free structured logging for guardrail checks.

Every guardrail emits exactly one line per check: which check ran, its outcome
and its latency.  Callers may add small scalar fields (counts, entity types,
status codes) but never text, PII values, keys or tokens.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger("app.guardrails")

_ALLOWED_TYPES = (int, float, bool, type(None))


def log_guard(check: str, outcome: str, latency_ms: float, **fields: Any) -> None:
    """Log one guardrail result.  Non-scalar / string extras are dropped unless
    they are short identifiers explicitly passed through ``fields``' ``reason``
    or ``source`` keys (fixed vocabulary chosen by our code, never content)."""
    record: dict[str, Any] = {
        "check": check,
        "outcome": outcome,
        "latency_ms": round(latency_ms, 1),
    }
    for key, value in fields.items():
        if isinstance(value, _ALLOWED_TYPES):
            record[key] = value
        elif key in {"reason", "source", "provider", "engine"} and isinstance(value, str) and len(value) <= 64:
            record[key] = value
        elif key == "entity_counts" and isinstance(value, dict):
            record[key] = {str(k)[:32]: int(v) for k, v in value.items()}
    level = logging.INFO if outcome == "pass" else logging.WARNING
    logger.log(level, "guardrail %s", json.dumps(record, sort_keys=True))


@contextmanager
def timed() -> Iterator[callable]:
    """``with timed() as elapsed: ...; elapsed()`` → milliseconds so far."""
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000.0
