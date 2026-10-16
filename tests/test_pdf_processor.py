"""Tests for safe PDF validation and text extraction."""

import pymupdf
import pytest

from app import pdf_processor
from app.pdf_processor import PDFValidationError, extract_text_from_pdf


def test_extracts_text_from_valid_pdf(make_pdf):
    data = make_pdf(["Project Phoenix checklist", "Budget is 5000 credits"])
    text, pages = extract_text_from_pdf(data)
    assert "Project Phoenix" in text
    assert "5000 credits" in text
    assert pages == 1


def test_empty_upload_is_rejected():
    with pytest.raises(PDFValidationError):
        extract_text_from_pdf(b"")


def test_missing_pdf_header_is_rejected():
    with pytest.raises(PDFValidationError):
        extract_text_from_pdf(b"this is definitely not a pdf")


def test_pdf_without_text_is_rejected(make_pdf):
    # A blank page has no extractable text (e.g. a scanned image).
    with pytest.raises(PDFValidationError):
        extract_text_from_pdf(make_pdf([]))


def test_too_many_pages_is_rejected(monkeypatch):
    monkeypatch.setattr(pdf_processor, "MAX_PAGES", 1)
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    data = doc.tobytes()

    with pytest.raises(PDFValidationError):
        extract_text_from_pdf(data)


def test_oversized_file_is_rejected(monkeypatch, make_pdf):
    monkeypatch.setattr(pdf_processor, "MAX_UPLOAD_MB", 0.0001)
    data = make_pdf(["some content"])
    with pytest.raises(PDFValidationError):
        extract_text_from_pdf(data)
