"""Presidio PII masking (app.security.pii) — runs locally, no network."""
import importlib.util
import logging

import pytest

from app.security import pii
from app.security.ingestion import redact_pii, restore_redacted_values

requires_spacy_model = pytest.mark.skipif(
    importlib.util.find_spec(pii.settings.presidio_spacy_model) is None,
    reason="spaCy model for Presidio is not installed",
)

# Synthetic data only.  Aadhaar 2341 2341 2346 passes the Verhoeff checksum;
# PAN ABCPS1234K has a valid holder-type letter (P).
RESUME = """# Priya Sharma
Software Engineer | priya.sharma@example.com | +91 98450 12345 | Flat 4B, 12 MG Road, Bengaluru 560001
PAN: ABCPS1234K  Aadhaar: 2341 2341 2346  Alt phone: +1 415 555 0132

## Experience
Infosys Technologies, Bengaluru - Software Engineer, Jan 2021 - Present
Built Django and FastAPI services with PostgreSQL, Docker and Kubernetes.
Mentored by Rahul Verma on Kafka streaming.

## Education
Anna University, Chennai - B.Tech Computer Science, 2016 - 2020
"""

PII_VALUES = [
    "Priya Sharma", "priya.sharma@example.com", "+91 98450 12345", "12 MG Road", "560001",
    "ABCPS1234K", "2341 2341 2346", "+1 415 555 0132", "Rahul Verma",
]
KEPT_VALUES = [
    "Infosys Technologies", "Anna University", "Bengaluru", "Chennai", "Software Engineer",
    "Django", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "Kafka", "Jan 2021", "2016 - 2020",
    "B.Tech Computer Science",
]


@requires_spacy_model
def test_presidio_masks_all_direct_identifiers_and_keeps_extraction_facts():
    masked, replacements = redact_pii(RESUME)
    for value in PII_VALUES:
        assert value not in masked, f"PII leaked: {value!r}"
    for value in KEPT_VALUES:
        assert value in masked, f"needed fact was masked: {value!r}"
    kinds = {token.split("_")[1] for token in replacements}
    assert {"NAME", "EMAIL", "PHONE", "ADDRESS", "PAN", "AADHAAR"} <= kinds


@requires_spacy_model
def test_mask_then_restore_returns_original_text():
    masked, replacements = redact_pii(RESUME)
    assert restore_redacted_values(masked, replacements) == RESUME
    nested = restore_redacted_values({"a": [masked]}, replacements)
    assert nested == {"a": [RESUME]}


@requires_spacy_model
def test_repeated_name_maps_to_one_token():
    masked, replacements = redact_pii("# Priya Sharma\nSummary: Priya Sharma builds APIs.")
    name_tokens = [t for t, v in replacements.items() if v == "Priya Sharma"]
    assert len(name_tokens) == 1
    assert masked.count(name_tokens[0]) == 2


@requires_spacy_model
def test_aadhaar_with_bad_checksum_is_not_labelled_aadhaar():
    spans = pii.detect_pii("Aadhaar 2341 2341 2345")
    assert all(s.entity_type != "IN_AADHAAR" for s in spans)


@requires_spacy_model
def test_tech_and_org_names_are_not_masked_as_person():
    text = "Skills: Django, Flask, Pandas, Tableau. Worked at Tata Consultancy Services and IIT Madras."
    masked, replacements = redact_pii(text)
    assert masked == text
    assert replacements == {}


@requires_spacy_model
def test_entity_list_is_configurable(monkeypatch):
    monkeypatch.setattr(pii.settings, "pii_entities", "EMAIL_ADDRESS")
    masked, _ = redact_pii("# Priya Sharma\npriya.sharma@example.com")
    assert "Priya Sharma" in masked
    assert "priya.sharma@example.com" not in masked


def test_presidio_failure_falls_back_to_regex(monkeypatch, caplog):
    def boom(*_a, **_k):
        raise RuntimeError("spaCy model missing")

    monkeypatch.setattr(pii, "detect_pii", boom)
    monkeypatch.setattr(pii.settings, "pii_engine", "presidio")
    with caplog.at_level(logging.WARNING):
        masked, replacements = redact_pii("Mail jane@example.com or call +91 98765 43210")
    assert "jane@example.com" not in masked and "98765" not in masked
    assert set(replacements.values()) == {"jane@example.com", "+91 98765 43210"}
    assert "regex fallback" in caplog.text
    # The structured guard log records the fallback without content.
    assert '"outcome": "fallback"' in caplog.text
    assert "jane@example.com" not in caplog.text


def test_regex_engine_setting(monkeypatch):
    monkeypatch.setattr(pii.settings, "pii_engine", "regex")
    masked, replacements = redact_pii("jane@example.com")
    assert masked == "[[CA_EMAIL_1]]"
    assert replacements == {"[[CA_EMAIL_1]]": "jane@example.com"}
