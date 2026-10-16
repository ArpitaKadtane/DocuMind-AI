"""Safe PDF validation and text extraction.

This module is the only place that touches uploaded file bytes. It refuses
anything that is not a well-formed, reasonably sized PDF so that a malicious
or broken upload cannot crash or exhaust the server.
"""

import pymupdf  # PyMuPDF

from app.config import settings

MAX_UPLOAD_MB = settings.max_upload_mb
MAX_PAGES = settings.max_pdf_pages
MAX_TOTAL_CHARS = settings.max_pdf_chars
PDF_MAGIC = b"%PDF-"


class PDFValidationError(Exception):
    """Raised when an upload fails a validity or safety check."""


def _max_bytes() -> int:
    return int(MAX_UPLOAD_MB * 1024 * 1024)


def validate_pdf_bytes(data: bytes) -> None:
    if not data:
        raise PDFValidationError("The uploaded file is empty.")
    if len(data) > _max_bytes():
        raise PDFValidationError(
            f"File is larger than the {MAX_UPLOAD_MB:g} MB limit."
        )
    if PDF_MAGIC not in data[:1024]:
        raise PDFValidationError("File is not a valid PDF (missing %PDF header).")


def extract_pages_from_pdf(data: bytes) -> tuple[list[tuple[int, str]], int]:
    """Return ``([(page_number, text), ...], page_count)`` for a PDF.

    Page numbers are the original one-based PDF page numbers, so a chunk can be
    traced back to the exact page it came from. Raises PDFValidationError with a
    safe, user-facing message on any problem.
    """
    validate_pdf_bytes(data)

    # Silences MuPDF's own stderr chatter so corrupted files cannot spam logs.
    pymupdf.TOOLS.mupdf_display_errors(False)

    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            page_count = doc.page_count
            if page_count == 0:
                raise PDFValidationError("PDF contains no pages.")
            if page_count > MAX_PAGES:
                raise PDFValidationError(
                    f"PDF has {page_count} pages; the limit is {MAX_PAGES}."
                )

            pages: list[tuple[int, str]] = []
            total = 0
            for page_number, page in enumerate(doc, start=1):
                page_text = page.get_text("text")
                total += len(page_text)
                if total > MAX_TOTAL_CHARS:
                    raise PDFValidationError("PDF text exceeds the allowed limit.")
                pages.append((page_number, page_text))
    except PDFValidationError:
        raise
    except Exception as exc:
        raise PDFValidationError(
            "PDF could not be read; it may be corrupted or malicious."
        ) from exc

    if not any(page_text.strip() for _, page_text in pages):
        raise PDFValidationError(
            "No text could be extracted. Scanned/image-only PDFs are not supported."
        )
    return pages, page_count


def extract_text_from_pdf(data: bytes) -> tuple[str, int]:
    """Return ``(text, page_count)`` for a PDF, with all pages joined.

    Page breaks are represented by a blank line. Use ``extract_pages_from_pdf``
    when you need per-page text (for example, to cite page numbers).
    """
    pages, page_count = extract_pages_from_pdf(data)
    text = "\n\n".join(page_text for _, page_text in pages).strip()
    return text, page_count
