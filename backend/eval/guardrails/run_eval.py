"""Guardrails evaluation on the synthetic set in eval/guardrails/data.

Usage (from backend/):
    python eval/guardrails/run_eval.py                 # offline: PII, output guards, latency
    python eval/guardrails/run_eval.py --extraction    # + live LLM extraction before/after Presidio
    python eval/guardrails/run_eval.py --extraction --limit 10

Prompt Shields metrics are measured only when PROMPT_SHIELDS_ENABLED=true and
AZURE_CONTENT_SAFETY_ENDPOINT / AZURE_CONTENT_SAFETY_KEY are set; otherwise they
are reported as "not measured".  Nothing here is estimated: every number comes
from running the code on the files in data/.

Results are printed and written to eval/guardrails/results/latest.json.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.security import pii  # noqa: E402

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
TOKEN = re.compile(r"\[\[CA_([A-Z]+)_\d+\]\]")
KIND_TO_TYPE = {v: k for k, v in pii.TOKEN_KIND.items()}


def original_regex_redact(text: str) -> tuple[str, dict[str, str]]:
    """The pre-change ``redact_pii`` (email + phone regex), verbatim, as the baseline."""
    email = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.+-])")
    phone = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{5,}\d)(?!\w)")
    replacements: dict[str, str] = {}
    counters = {"EMAIL": 0, "PHONE": 0}

    def replace(kind, match):
        counters[kind] += 1
        token = f"[[CA_{kind}_{counters[kind]}]]"
        replacements[token] = match.group(0)
        return token

    redacted = email.sub(lambda m: replace("EMAIL", m), text)
    redacted = phone.sub(lambda m: replace("PHONE", m), redacted)
    return redacted, replacements


ENGINES = {
    "baseline_regex": original_regex_redact,                       # what production ran before this change
    "regex_fallback": lambda t: (setattr(settings, "pii_engine", "regex"), pii.mask_pii(t))[1],
    "presidio": lambda t: (setattr(settings, "pii_engine", "presidio"), pii.mask_pii(t))[1],
}


# ── helpers ─────────────────────────────────────────────────────────────────

def masked_spans(original: str, masked: str, replacements: dict[str, str]) -> list[tuple[int, int, str]]:
    """Recover (start, end, entity_type) in the original text from the masked text."""
    spans, o_pos, m_pos = [], 0, 0
    for match in TOKEN.finditer(masked):
        literal = masked[m_pos:match.start()]
        assert original.startswith(literal, o_pos), "alignment failed"
        o_pos += len(literal)
        value = replacements[match.group(0)]
        assert original.startswith(value, o_pos), "alignment failed"
        spans.append((o_pos, o_pos + len(value), KIND_TO_TYPE.get(match.group(1), match.group(1))))
        o_pos += len(value)
        m_pos = match.end()
    assert original[o_pos:] == masked[m_pos:], "alignment failed"
    return spans


def occurrences(text: str, value: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(re.escape(value), text)]


def covered(start: int, end: int, spans) -> bool:
    chars = set(range(start, end))
    for s, e, _ in spans:
        chars -= set(range(s, e))
    return not chars


def touched(start: int, end: int, spans) -> bool:
    return any(s < end and start < e for s, e, _ in spans)


def pct(num: float, den: float) -> float | None:
    return round(100.0 * num / den, 1) if den else None


def latency(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean_ms": round(statistics.fmean(values), 1),
        "p50_ms": round(ordered[len(ordered) // 2], 1),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1),
    }


# ── 1. PII masking ──────────────────────────────────────────────────────────

def eval_pii(resumes: list[dict]) -> dict:
    out: dict = {}
    # Warm the Presidio engine (model load is a one-off cold-start cost).
    t0 = time.perf_counter()
    pii._analyzer()
    out["presidio_cold_start_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    for engine, masker in ENGINES.items():
        stats = defaultdict(lambda: {"labelled": 0, "masked": 0, "pred": 0, "pred_correct": 0})
        keep_total = keep_kept = 0
        pred_any = pred_any_correct = 0
        times: list[float] = []
        for doc in resumes:
            text = doc["text"]
            start = time.perf_counter()
            masked, replacements = masker(text)
            times.append((time.perf_counter() - start) * 1000)
            spans = masked_spans(text, masked, replacements)
            labelled = [(s, e, item["type"]) for item in doc["pii"] for s, e in occurrences(text, item["value"])]
            for s, e, etype in labelled:
                stats[etype]["labelled"] += 1
                stats[etype]["masked"] += covered(s, e, spans)
            for s, e, ptype in spans:
                stats[ptype]["pred"] += 1
                stats[ptype]["pred_correct"] += any(ls < e and s < le and lt == ptype for ls, le, lt in labelled)
                pred_any += 1
                pred_any_correct += any(ls < e and s < le for ls, le, _ in labelled)
            for item in doc["keep"]:
                for s, e in occurrences(text, item["value"]):
                    keep_total += 1
                    keep_kept += not touched(s, e, spans)
        per_type = {
            etype: {
                "labelled": v["labelled"],
                "recall_pct": pct(v["masked"], v["labelled"]),
                "predicted": v["pred"],
                "precision_pct": pct(v["pred_correct"], v["pred"]),
            }
            for etype, v in sorted(stats.items())
        }
        total_lab = sum(v["labelled"] for v in stats.values())
        total_masked = sum(v["masked"] for v in stats.values())
        out[engine] = {
            "per_entity": per_type,
            "overall_recall_pct": pct(total_masked, total_lab),
            "overall_precision_pct": pct(pred_any_correct, pred_any),
            "facts_kept_pct": pct(keep_kept, keep_total),
            "facts_checked": keep_total,
            "latency": latency(times),
        }
    settings.pii_engine = "presidio"
    out["added_latency_mean_ms"] = round(
        out["presidio"]["latency"]["mean_ms"] - out["baseline_regex"]["latency"]["mean_ms"], 1
    )
    return out


# ── 2. Prompt Shields ───────────────────────────────────────────────────────

def eval_injection(docs: list[dict]) -> dict:
    from app.security import prompt_shields as ps

    if not (settings.prompt_shields_enabled and settings.azure_content_safety_endpoint and settings.azure_content_safety_key):
        return {"status": "not measured",
                "reason": "PROMPT_SHIELDS_ENABLED / AZURE_CONTENT_SAFETY_ENDPOINT / AZURE_CONTENT_SAFETY_KEY not set",
                "documents": {"attack": sum(d["label"] == "attack" for d in docs),
                              "clean": sum(d["label"] == "clean" for d in docs)}}
    tp = fp = attacks = cleans = errors = 0
    times: list[float] = []
    for doc in docs:
        text = pii.mask_pii(doc["text"])[0] if doc["kind"] == "resume" else doc["text"]
        start = time.perf_counter()
        verdict = ps.shield_documents({"doc": text}, "eval")["doc"]
        times.append((time.perf_counter() - start) * 1000)
        if not verdict.checked:
            errors += 1
            continue
        if doc["label"] == "attack":
            attacks += 1
            tp += verdict.attack_detected
        else:
            cleans += 1
            fp += verdict.attack_detected
    return {"status": "measured", "detection_rate_pct": pct(tp, attacks), "false_block_rate_pct": pct(fp, cleans),
            "attacks": attacks, "clean": cleans, "errors": errors, "latency": latency(times)}


# ── 3. Output guards (offline) ─────────────────────────────────────────────

def eval_output_guards(resumes: list[dict]) -> dict:
    from app.security import output_guards as og

    settings.output_guards_enabled = True
    profile_times, text_times = [], []
    profile_false_blocks = leak_caught = 0
    for doc in resumes:
        exp = doc["expected"]
        profile = {
            "full_name": exp["full_name"],
            "contact": {"email": exp["email"], "phone_raw": exp["phone_raw"]},
            "summary": f"{exp['titles'][0]} experienced in {', '.join(exp['skills'][:3])}.",
            "skills": exp["skills"],
            "experience": [{"company": c, "title": t, "description_bullets": ["Built data services."]}
                           for c, t in zip(exp["companies"], exp["titles"])],
            "education": [{"institution": i} for i in exp["institutions"]],
            "projects": [], "certifications": [], "programming_languages": [], "spoken_languages": [], "keywords": [],
        }
        allowed = [exp["full_name"], exp["email"], exp["phone_raw"]]
        start = time.perf_counter()
        try:
            og.validate_profile(profile, allowed, [])
        except og.OutputGuardError:
            profile_false_blocks += 1
        profile_times.append((time.perf_counter() - start) * 1000)
        # Same profile with a third-party email + Aadhaar leaked into a bullet.
        leaky = json.loads(json.dumps(profile))
        leaky["experience"][0]["description_bullets"].append("Escalate to ops-desk@example.org, Aadhaar 2341 2341 2346.")
        try:
            og.validate_profile(leaky, allowed, [])
        except og.OutputGuardError:
            leak_caught += 1

        texts = [f"{s} is required for {exp['titles'][0]} roles." for s in exp["skills"][:4]]
        start = time.perf_counter()
        try:
            og.validate_texts(texts, exp["skills"], exp["skills"], "eval_text")
        except og.OutputGuardError:
            pass
        text_times.append((time.perf_counter() - start) * 1000)

    n = len(resumes)
    return {
        "profile_guard_false_block_pct_on_clean_profiles": pct(profile_false_blocks, n),
        "profile_guard_leak_detection_pct": pct(leak_caught, n),
        "profile_guard_latency": latency(profile_times),
        "text_guard_latency": latency(text_times),
    }


# ── 4. Extraction quality before vs after Presidio (live LLM) ──────────────

def _norm(value) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", str(value or "").lower()).strip()


def _recall(expected: list[str], got: list[str]) -> float:
    got_n = [_norm(g) for g in got]
    hits = sum(any(_norm(e) == g or _norm(e) in g for g in got_n) for e in expected)
    return hits / len(expected) if expected else 1.0


def score_extraction(profile: dict, exp: dict) -> dict:
    contact = profile.get("contact") or {}
    digits = lambda s: re.sub(r"\D", "", s or "")[-10:]  # noqa: E731
    return {
        "full_name": float(_norm(profile.get("full_name")) == _norm(exp["full_name"])),
        "email": float((contact.get("email") or "").lower() == exp["email"].lower()),
        "phone": float(bool(digits(contact.get("phone_raw"))) and digits(contact.get("phone_raw")) == digits(exp["phone_raw"])),
        "companies": _recall(exp["companies"], [x.get("company") for x in profile.get("experience", [])]),
        "titles": _recall(exp["titles"], [x.get("title") for x in profile.get("experience", [])]),
        "institutions": _recall(exp["institutions"], [x.get("institution") for x in profile.get("education", [])]),
        "skills": _recall(exp["skills"], profile.get("skills", []) + profile.get("programming_languages", [])),
    }


def eval_extraction(resumes: list[dict]) -> dict:
    from app.resume_extraction import service
    from app.resume_extraction.service import extract_structured_resume_data

    settings.prompt_shields_enabled = False
    settings.output_guards_enabled = True
    out = {}
    current_redact = service.redact_pii
    for engine in ("baseline_regex", "presidio"):
        service.redact_pii = original_regex_redact if engine == "baseline_regex" else current_redact
        settings.pii_engine = "presidio"
        scores, failures, times = defaultdict(list), [], []
        for doc in resumes:
            start = time.perf_counter()
            try:
                profile = extract_structured_resume_data(doc["text"]).model_dump(mode="json")
            except Exception as exc:  # count, never hide
                failures.append({"id": doc["id"], "error": type(exc).__name__})
                continue
            finally:
                times.append((time.perf_counter() - start) * 1000)
            for field, value in score_extraction(profile, doc["expected"]).items():
                scores[field].append(value)
        out[engine] = {
            "resumes": len(resumes),
            "failures": failures,
            "field_accuracy_pct": {f: round(100 * statistics.fmean(v), 1) for f, v in scores.items()},
            "mean_field_accuracy_pct": round(100 * statistics.fmean([x for v in scores.values() for x in v]), 1) if scores else None,
            "latency": latency(times),
        }
    service.redact_pii = current_redact
    return out


# ── report ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction", action="store_true", help="run live LLM extraction (uses your LLM key)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    resumes = json.loads((DATA / "pii_resumes.json").read_text(encoding="utf-8"))[: args.limit]
    holdout = json.loads((DATA / "pii_resumes_holdout.json").read_text(encoding="utf-8"))
    docs = json.loads((DATA / "injection_docs.json").read_text(encoding="utf-8"))

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {"pii_resumes": len(resumes), "pii_resumes_holdout": len(holdout), "injection_docs": len(docs)},
        "config": {"pii_entities": settings.pii_entity_list, "pii_score_threshold": settings.pii_score_threshold,
                   "spacy_model": settings.presidio_spacy_model},
        "pii_masking": eval_pii(resumes),
        # Recognizers were tuned after a first run on pii_resumes.json; the
        # held-out set (different seed) was generated afterwards.
        "pii_masking_holdout": eval_pii(holdout),
        "prompt_shields": eval_injection(docs),
        "output_guards": eval_output_guards(resumes),
        "extraction": eval_extraction(resumes) if args.extraction else {"status": "not run (pass --extraction)"},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
