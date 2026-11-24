"""PII detection and masking for untrusted resume text (Microsoft Presidio).

Contract (kept identical to the original regex ``redact_pii``):
``mask_pii(text) -> (masked_text, {token: original_value})`` where tokens look
like ``[[CA_EMAIL_1]]``.  ``restore_redacted_values`` in ``ingestion.py``
reverses the mapping after the LLM output has been validated.

Design notes
------------
* Only *direct identifiers* are masked (configurable via ``PII_ENTITIES``).
  LOCATION / ORGANIZATION / DATE_TIME are never requested, because the
  extractor needs cities, companies, universities and dates.
* PERSON spans produced by spaCy NER are pruned when they look like an
  organisation, institution or a technology name — the common false positives
  on resumes ("Django", "Infosys Technologies", "Anna University").
* Street addresses and a stricter PAN rule are custom recognizers; Presidio's
  built-in Aadhaar recognizer already validates the Verhoeff checksum.
* After Presidio, the original email / phone regexes run over the remaining
  text as a safety net, so this layer is never weaker than the one it replaced.
* Any Presidio failure falls back to the regex masking.  Unmasked text is never
  returned.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from app.config import settings
from app.security.guard_log import log_guard, timed

logger = logging.getLogger(__name__)
# Presidio logs one warning per skipped non-English recognizer and per unmapped
# spaCy label on every load/call; that is noise, not a signal.
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

# ── Regex layer (original implementation, now the fallback/safety net) ──────

_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.+-])")
# Deliberately conservative: at least seven digits avoids replacing dates and
# version numbers while covering common international resume phone formats.
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{5,}\d)(?!\w)")
_TOKEN = re.compile(r"\[\[CA_[A-Z]+_\d+\]\]")

TOKEN_KIND = {
    "PERSON": "NAME",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "STREET_ADDRESS": "ADDRESS",
    "IN_PAN": "PAN",
    "IN_AADHAAR": "AADHAAR",
    "IN_PASSPORT": "PASSPORT",
    "IN_VOTER": "VOTER",
}

# Words that make a "PERSON" span almost certainly an organisation/institution.
_ORG_HINTS = re.compile(
    r"\b(universit\w*|institut\w*|college|school|academy|vidyalaya|technolog\w*|labs?|"
    r"inc|ltd|llc|llp|pvt|private|limited|corp\w*|solutions|systems|services|software|"
    r"consult\w*|bank|group|foundation|analytics|networks|global|digital|infotech|"
    r"iit|nit|iiit|bits|ai|ml|cloud|data)\b",
    re.IGNORECASE,
)

# Technology names spaCy sometimes tags as PERSON.  Exact (case-insensitive)
# matches are never masked.
_TECH_ALLOW = {
    t.lower()
    for t in (
        "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "Ruby", "Scala", "Kotlin", "Swift",
        "Dart", "Julia", "Perl", "Haskell", "Elixir", "Django", "Flask", "FastAPI", "React", "Angular",
        "Vue", "Svelte", "Next.js", "Node.js", "Express", "Spring", "Spring Boot", "Laravel", "Rails",
        "Pandas", "NumPy", "SciPy", "Keras", "PyTorch", "TensorFlow", "XGBoost", "LightGBM", "OpenCV",
        "Hadoop", "Spark", "Kafka", "Airflow", "Tableau", "Power BI", "Excel", "Jira", "Confluence",
        "Figma", "Docker", "Kubernetes", "Jenkins", "Terraform", "Ansible", "Git", "GitHub", "GitLab",
        "Linux", "Bash", "AWS", "GCP", "Azure", "Heroku", "Vercel", "Netlify", "Firebase", "Supabase",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Cassandra", "Elasticsearch", "Snowflake",
        "Databricks", "LangChain", "LangGraph", "Hugging Face", "Streamlit", "Gradio", "Selenium",
        "Postman", "Matplotlib", "Seaborn", "Plotly", "Scikit-learn", "NLTK", "spaCy", "Pinecone",
        "Groq", "Gemini", "OpenAI", "Claude", "Llama", "Mistral", "Ollama", "Celery", "RabbitMQ",
        "GraphQL", "Redux", "Tailwind", "Bootstrap", "Sass", "Webpack", "Vite", "Jest", "Pytest",
        "Cypress", "Playwright", "Unity", "Unreal", "Blender", "MATLAB", "Simulink", "AutoCAD",
        "SolidWorks", "Salesforce", "SAP", "Oracle", "Hindi", "English", "Marathi", "Tamil",
        "Telugu", "Kannada", "Bengali", "Gujarati", "Punjabi", "Malayalam", "Urdu",
    )
}


# Public lower-cased list (also the default skill vocabulary for output guards).
TECH_TERMS: list[str] = sorted(_TECH_ALLOW)


@dataclass(frozen=True)
class PiiSpan:
    entity_type: str
    start: int
    end: int
    score: float


_YEAR_RANGE = re.compile(r"^(?:19|20)\d{2}\D+(?:19|20)\d{2}$")


def _plausible_phone(value: str) -> bool:
    """Keep the regex from eating date ranges ("2016 - 2020") or short numbers."""
    digits = re.sub(r"\D", "", value)
    return 9 <= len(digits) <= 15 and not _YEAR_RANGE.match(value.strip())


def regex_mask(text: str) -> tuple[str, dict[str, str]]:
    """Original email + phone masking.  Used as fallback and as safety net."""
    replacements: dict[str, str] = {}
    return _regex_mask_into(text, replacements, {}), replacements


def _regex_mask_into(text: str, replacements: dict[str, str], counters: dict[str, int]) -> str:
    def make_replacer(kind: str):
        def replace(match: re.Match[str]) -> str:
            value = match.group(0)
            if _TOKEN.fullmatch(value):
                return value
            if kind == "PHONE" and not _plausible_phone(value):
                return value
            return _token_for(kind, value, replacements, counters)
        return replace

    out = _EMAIL.sub(make_replacer("EMAIL"), text)
    # Never let the phone regex eat digits inside an existing token.
    parts = _TOKEN.split(out)
    tokens = _TOKEN.findall(out)
    phone = make_replacer("PHONE")
    rebuilt = []
    for idx, part in enumerate(parts):
        rebuilt.append(_PHONE.sub(phone, part))
        if idx < len(tokens):
            rebuilt.append(tokens[idx])
    return "".join(rebuilt)


def _token_for(kind: str, value: str, replacements: dict[str, str], counters: dict[str, int]) -> str:
    for token, original in replacements.items():
        if original == value and token.startswith(f"[[CA_{kind}_"):
            return token
    counters[kind] = counters.get(kind, 0) + 1
    token = f"[[CA_{kind}_{counters[kind]}]]"
    replacements[token] = value
    return token


# ── Presidio layer ───────────────────────────────────────────────────────────

def _build_custom_recognizers() -> list:
    from presidio_analyzer import Pattern, PatternRecognizer

    street_suffix = (
        r"(?:road|rd|street|st|lane|ln|avenue|ave|marg|nagar|colony|layout|sector|block|"
        r"cross|main|apartments?|apts?|society|enclave|vihar|boulevard|blvd|drive|"
        r"chowk|gali|residency|heights)"
    )
    address = PatternRecognizer(
        supported_entity="STREET_ADDRESS",
        name="CA street address",
        patterns=[
            # "Flat 4B, 12 MG Road", "221B Baker Street", "No. 7, 3rd Cross".
            # The house number must start a line / field (after "|", ",", ":")
            # or follow flat/house/no./plot/door/#, so prose such as
            # "across 3 microservices in phase 2" is not an address.
            Pattern(
                "street line",
                r"(?:(?<=^)|(?<=^[ \t#*>-]{1,6})|(?<=[,:|•·][ \t]{0,3})"
                r"|\b(?:flat|house|h\.?\s?no\.?|plot|door|no\.?|#)[ \t]{0,3})"
                r"\d{1,5}[A-Za-z]?(?:[/-]\d{1,4}[A-Za-z]?)?"
                r",?[ \t]+(?:[A-Za-z0-9.'-]+,?[ \t]+){0,4}?" + street_suffix
                + r"\b\.?(?:[ \t]+" + street_suffix + r"\b\.?)?",
                0.6,
            ),
            # Indian PIN code after a place name / comma: "Bengaluru 560001", "Pune - 411 001"
            Pattern("in pin after place", r"(?<=[A-Za-z][,\s\-]{1,3})[1-9]\d{2}\s?\d{3}(?!\d)", 0.55),
            Pattern("in pin labelled", r"\b(?:pin(?:\s?code)?|postal\s?code|zip)\s*[:\-]?\s*[1-9]\d{2}\s?\d{3}(?!\d)", 0.7),
        ],
        context=["address", "residence", "pin", "pincode"],
    )

    class StrictPanRecognizer(PatternRecognizer):
        """PAN: AAAAA9999A where the 4th letter is a valid holder type."""

        def validate_result(self, pattern_text: str):  # noqa: D401
            return bool(re.fullmatch(r"[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z]", pattern_text.upper()))

    pan = StrictPanRecognizer(
        supported_entity="IN_PAN",
        name="CA strict PAN",
        patterns=[Pattern("pan", r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b", 0.4)],
        context=["pan", "permanent account number"],
    )

    in_mobile = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        name="CA india mobile",
        patterns=[Pattern("in mobile", r"(?<![\w+])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)", 0.6)],
        context=["phone", "mobile", "contact", "tel"],
    )
    return [address, pan, in_mobile]


@lru_cache(maxsize=1)
def _analyzer():
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": settings.presidio_spacy_model}],
        }
    )
    registry = RecognizerRegistry(supported_languages=["en"])
    registry.load_predefined_recognizers(languages=["en"])
    # India recognizers ship with Presidio but are not in the default English
    # set (verified against presidio-analyzer 2.2.366), so register them here.
    from presidio_analyzer.predefined_recognizers import (
        InAadhaarRecognizer,
        InPassportRecognizer,
        InVoterRecognizer,
    )
    for recognizer in (InAadhaarRecognizer(), InPassportRecognizer(), InVoterRecognizer()):
        registry.add_recognizer(recognizer)
    for recognizer in _build_custom_recognizers():
        registry.add_recognizer(recognizer)
    return AnalyzerEngine(registry=registry, nlp_engine=provider.create_engine(), supported_languages=["en"])


def _looks_like_non_person(value: str) -> bool:
    stripped = value.strip(" .,:;-")
    if stripped.lower() in _TECH_ALLOW:
        return True
    if _ORG_HINTS.search(stripped):
        return True
    # Single all-caps token ("IIT", "TCS") or anything with digits is not a name.
    if re.search(r"\d", stripped) or (stripped.isupper() and " " not in stripped):
        return True
    return False


_NAME_WORD = r"[A-Z][a-zA-Z'\-]+\.?|[A-Z]\."
_HEADER_NAME = re.compile(rf"^[#*_\s>]*(?:name\s*[:\-]\s*)?((?:{_NAME_WORD})(?:\s+(?:{_NAME_WORD})){{1,3}})[*_\s]*$", re.IGNORECASE)
_LABELLED_NAME = re.compile(rf"\b(?:full\s+)?name\s*[:\-]\s*((?:{_NAME_WORD})(?:\s+(?:{_NAME_WORD})){{1,3}})")


_NOT_NAME_WORDS = {
    "resume", "curriculum", "vitae", "cv", "profile", "summary", "objective", "contact", "experience",
    "education", "skills", "projects", "engineer", "developer", "scientist", "analyst", "manager",
    "intern", "designer", "architect", "consultant", "lead", "senior", "junior", "software", "data",
    "full", "stack", "frontend", "backend", "student", "fresher", "professional", "personal", "details",
}


def _header_name_spans(text: str) -> list[PiiSpan]:
    """Resume names usually sit alone on the first line (often a markdown
    heading), where spaCy NER is weakest.  Catch that line and "Name: …"."""
    spans: list[PiiSpan] = []
    offset = 0
    seen_lines = 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            seen_lines += 1
            match = _HEADER_NAME.match(line.rstrip("\r\n"))
            if match:
                candidate = match.group(1)
                words = candidate.split()
                if (
                    all(w[0].isupper() for w in words)
                    and not any(w.lower().strip(".") in _NOT_NAME_WORDS for w in words)
                    and not _looks_like_non_person(candidate)
                ):
                    spans.append(PiiSpan("PERSON", offset + match.start(1), offset + match.end(1), 0.85))
                    break
            if seen_lines >= 3:
                break
        offset += len(line)
    for match in _LABELLED_NAME.finditer(text):
        words = match.group(1).split()
        if not any(w.lower().strip(".") in _NOT_NAME_WORDS for w in words) and not _looks_like_non_person(match.group(1)):
            spans.append(PiiSpan("PERSON", match.start(1), match.end(1), 0.85))
    return spans


def _resolve_overlaps(spans: Iterable[PiiSpan]) -> list[PiiSpan]:
    chosen: list[PiiSpan] = []
    for span in sorted(spans, key=lambda s: (-s.score, -(s.end - s.start), s.start)):
        if all(span.end <= c.start or span.start >= c.end for c in chosen):
            chosen.append(span)
    return sorted(chosen, key=lambda s: s.start)


def detect_pii(text: str, entities: list[str] | None = None, threshold: float | None = None) -> list[PiiSpan]:
    """Return non-overlapping PII spans (Presidio).  Raises on engine failure."""
    entities = entities or settings.pii_entity_list
    threshold = settings.pii_score_threshold if threshold is None else threshold
    results = _analyzer().analyze(
        text=text,
        language="en",
        entities=entities,
        score_threshold=0.0,
        allow_list=sorted(_TECH_ALLOW),
        allow_list_match="exact",
    )
    spans: list[PiiSpan] = []
    for r in results:
        value = text[r.start:r.end]
        if _TOKEN.search(value):
            continue
        keep = r.score >= threshold
        # Presidio's PhoneRecognizer validates with libphonenumber but scores 0.4;
        # a libphonenumber-valid match is strong enough on its own.
        if not keep and r.entity_type == "PHONE_NUMBER":
            explanation = getattr(r, "recognition_metadata", {}) or {}
            keep = explanation.get("recognizer_name") == "PhoneRecognizer"
        if not keep:
            continue
        if r.entity_type == "PERSON":
            if _looks_like_non_person(value):
                continue
            # Trim trailing newline + following line that NER sometimes swallows.
            first_line = value.split("\n", 1)[0].rstrip()
            if not first_line:
                continue
            spans.append(PiiSpan(r.entity_type, r.start, r.start + len(first_line), r.score))
            continue
        spans.append(PiiSpan(r.entity_type, r.start, r.end, r.score))
    if "PERSON" in entities:
        spans.extend(_header_name_spans(text))
    spans = _resolve_overlaps(spans)
    # spaCy often tags one-word company names ("Freshworks") as PERSON.  Keep a
    # single-word PERSON span only when it is part of a multi-word name found
    # elsewhere in the same text.
    multi_word_names = [text[s.start:s.end] for s in spans
                        if s.entity_type == "PERSON" and len(text[s.start:s.end].split()) > 1]
    name_words = {w.lower() for name in multi_word_names for w in name.split()}
    spans = [
        s for s in spans
        if s.entity_type != "PERSON"
        or len(text[s.start:s.end].split()) > 1
        or text[s.start:s.end].lower() in name_words
    ]
    # A name found once (e.g. in the header) is masked everywhere it recurs,
    # including spots where NER missed it.
    names = {text[s.start:s.end] for s in spans if s.entity_type == "PERSON"}
    extra: list[PiiSpan] = []
    for name in names:
        for match in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text):
            extra.append(PiiSpan("PERSON", match.start(), match.end(), 0.8))
    return _resolve_overlaps(spans + extra)


def _presidio_mask(text: str) -> tuple[str, dict[str, str], dict[str, int]]:
    spans = detect_pii(text)
    replacements: dict[str, str] = {}
    counters: dict[str, int] = {}
    entity_counts: dict[str, int] = {}
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        kind = TOKEN_KIND.get(span.entity_type, re.sub(r"[^A-Z]", "", span.entity_type) or "PII")
        pieces.append(text[cursor:span.start])
        pieces.append(_token_for(kind, text[span.start:span.end], replacements, counters))
        cursor = span.end
        entity_counts[span.entity_type] = entity_counts.get(span.entity_type, 0) + 1
    pieces.append(text[cursor:])
    masked = "".join(pieces)
    # Safety net: original regexes over whatever Presidio left behind.
    before = len(replacements)
    masked = _regex_mask_into(masked, replacements, counters)
    if len(replacements) > before:
        entity_counts["REGEX_SAFETY_NET"] = len(replacements) - before
    return masked, replacements, entity_counts


def mask_pii(text: str) -> tuple[str, dict[str, str]]:
    """Mask direct identifiers.  Never returns unmasked text."""
    with timed() as elapsed:
        if settings.pii_engine.lower() != "presidio":
            masked, replacements = regex_mask(text)
            log_guard("pii_mask", "pass", elapsed(), engine="regex", masked=len(replacements))
            return masked, replacements
        try:
            masked, replacements, counts = _presidio_mask(text)
        except Exception as exc:  # engine/model missing, spaCy error, ...
            logger.warning("Presidio PII masking failed (%s); using regex fallback", type(exc).__name__)
            masked, replacements = regex_mask(text)
            log_guard("pii_mask", "fallback", elapsed(), engine="regex", reason=type(exc).__name__[:64],
                      masked=len(replacements))
            return masked, replacements
        log_guard("pii_mask", "pass", elapsed(), engine="presidio", masked=len(replacements), entity_counts=counts)
        return masked, replacements
