"""Whole-document summarization.

Unlike question answering, a summary should cover the *whole* document rather
than only the chunks most similar to a query. We therefore read the document's
chunks directly (instead of the vector index). When a document is too large to
fit the model's context window, chunks are sampled evenly across it so the
beginning, middle, and end are all represented -- naive truncation would only
ever see the start.

The result is grounded in those chunks (never invented) and, like Q&A, carries
the one-based page each excerpt came from so the summary can be traced back.

Flow:
  document chunks -> (maybe even-sample) -> labelled contexts -> LLM ->
  parse {summary, key points}
"""

import re

from app.config import settings
from app.llm import generate_summary


def _label(page) -> str:
    """Render a page number as a human label for the model."""
    return f"page {page}" if page is not None else "page unavailable"


def select_chunks(
    chunks: list[str], max_chars: int, max_chunks: int
) -> tuple[list[int], bool]:
    """Choose which chunk indices to summarize.

    Returns ``(indices, truncated)``. If every chunk fits within ``max_chars``
    and ``max_chunks``, all are used and ``truncated`` is False. Otherwise up to
    ``max_chunks`` chunks are picked evenly across the document (including the
    first and last) and ``truncated`` is True. Indices are always in ascending
    document order. The caller derives ``max_chunks`` from ``max_chars`` so the
    selected chunks respect the character budget.
    """
    n = len(chunks)
    if n == 0:
        return [], False

    total_chars = sum(len(chunk) for chunk in chunks)
    if total_chars <= max_chars and n <= max_chunks:
        return list(range(n)), False

    count = min(max_chunks, n)
    if count <= 1:
        return [0], True

    # Evenly spaced indices from the first to the last chunk.
    indices = sorted(
        {round(i * (n - 1) / (count - 1)) for i in range(count)}
    )
    return indices, True


def _is_bullet(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*\u2022]|\d+[.)])\s+\S", line))


def _clean_bullet(line: str) -> str:
    return re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", line).strip()


def parse_summary(raw: str) -> tuple[str, list[str]]:
    """Split the model's reply into ``(summary, key_points)``.

    The prompt asks for a ``SUMMARY:`` line followed by a ``KEY POINTS:`` list.
    Real models drift, so this is forgiving: if the marker is missing the whole
    reply becomes the summary, and a reply that is nothing but bullets is
    treated as key points. Nothing is ever discarded entirely.
    """
    text = (raw or "").strip()
    if not text:
        return "", []

    match = re.search(r"key\s*points?\s*:?", text, flags=re.IGNORECASE)
    if match:
        before, after = text[: match.start()], text[match.end() :]
    else:
        before, after = text, ""

    summary = re.sub(
        r"^\s*summary\s*:?\s*", "", before, flags=re.IGNORECASE
    ).strip()
    key_points = [_clean_bullet(line) for line in after.splitlines()]
    key_points = [point for point in key_points if point]

    if not match:
        lines = [line for line in text.splitlines() if line.strip()]
        if lines and all(_is_bullet(line) for line in lines):
            return "", [_clean_bullet(line) for line in lines]

    if not summary and not key_points:
        return text, []
    return summary, key_points


def summarize_document(
    chunks: list[str], pages: list, filename: str | None = None
) -> dict:
    """Summarize a document from its chunks and return a structured result.

    Returns ``{"summary", "key_points", "found", "truncated", "sources"}``.
    ``found`` is False (and nothing is sent to the LLM) when there is no text
    to summarize. ``sources`` reuses the citation shape of Q&A: each selected
    chunk's ``filename``, ``page`` (or None), and ``text``.
    """
    if not chunks:
        return {
            "summary": "",
            "key_points": [],
            "found": False,
            "truncated": False,
            "sources": [],
        }

    n = len(chunks)
    avg_len = max(1, sum(len(chunk) for chunk in chunks) // n)
    max_chunks = max(1, settings.summary_max_chars // avg_len)
    indices, truncated = select_chunks(
        chunks, settings.summary_max_chars, max_chunks
    )

    sources = [
        {
            "filename": filename,
            "page": pages[index] if index < len(pages) else None,
            "text": chunks[index],
        }
        for index in indices
    ]
    contexts = [f"[{_label(source['page'])}] {source['text']}" for source in sources]

    raw = generate_summary(contexts)
    summary, key_points = parse_summary(raw)

    return {
        "summary": summary,
        "key_points": key_points,
        "found": True,
        "truncated": truncated,
        "sources": sources,
    }
