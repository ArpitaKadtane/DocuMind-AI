"""Split extracted document text into retrieval-sized chunks.

Why chunks? Embeddings (the numbers vector search compares) work best on
short, focused passages. One vector for a whole PDF averages everything
together and retrieves poorly. Each chunk becomes one searchable unit.

Chunks are sentence-aware and overlap slightly so an answer that straddles a
boundary is still retrievable from both neighbouring chunks.
"""

import re

from app.config import settings

CHUNK_SIZE_CHARS = settings.chunk_size_chars
CHUNK_OVERLAP_CHARS = settings.chunk_overlap_chars

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Return overlapping, sentence-aware chunks of ``text``.

    Raises ValueError on bad configuration (overlap >= chunk_size).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        # A single sentence longer than chunk_size gets hard-split.
        while len(sentence) > chunk_size:
            if current:
                chunks.append(" ".join(current).strip())
                current, current_len = [], 0
            chunks.append(sentence[:chunk_size].strip())
            sentence = sentence[chunk_size:]

        candidate_len = current_len + len(sentence) + (1 if current else 0)
        if current and candidate_len > chunk_size:
            chunks.append(" ".join(current).strip())
            # Carry the tail of the finished chunk forward as overlap.
            current, current_len = [], 0
            for prev in reversed(chunks[-1].split(". ")):
                if current_len + len(prev) <= overlap:
                    current.insert(0, prev)
                    current_len += len(prev) + 2
                else:
                    break
        current.append(sentence)
        current_len += len(sentence) + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[dict]:
    """Chunk each page separately and remember each chunk's source page.

    Chunking page by page guarantees every chunk maps to exactly one page, so a
    citation is never ambiguous. ``pages`` is a list of ``(page_number, text)``
    with one-based page numbers. Returns a list of ``{"text": str, "page": int}``.
    """
    records: list[dict] = []
    for page_number, page_text in pages:
        for chunk in chunk_text(page_text, chunk_size=chunk_size, overlap=overlap):
            records.append({"text": chunk, "page": page_number})
    return records
