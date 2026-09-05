import asyncio

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from app.security.ingestion import (
    MAX_RESUME_BYTES,
    read_validated_pdf,
    redact_pii,
    restore_redacted_values,
)


def _upload(data: bytes, content_type: str = "application/pdf") -> UploadFile:
    from io import BytesIO
    return UploadFile(BytesIO(data), filename="resume.pdf", headers=Headers({"content-type": content_type}))


def test_redact_and_restore_pii_in_nested_output():
    redacted, replacements = redact_pii("Contact me at jane@example.com or +91 98765 43210")
    assert "jane@example.com" not in redacted
    assert "98765" not in redacted
    restored = restore_redacted_values({"contact": {"email": redacted}}, replacements)
    assert restored["contact"]["email"] == "Contact me at jane@example.com or +91 98765 43210"


def test_pdf_validation_rejects_non_pdf_bytes():
    with pytest.raises(HTTPException, match="not a valid PDF"):
        asyncio.run(read_validated_pdf(_upload(b"not a pdf")))


def test_pdf_validation_rejects_oversized_upload():
    with pytest.raises(HTTPException) as error:
        asyncio.run(read_validated_pdf(_upload(b"%PDF-" + b"0" * MAX_RESUME_BYTES)))
    assert error.value.status_code == 413
