"""Tests for the sentence-aware chunker."""

import pytest

from app.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ") == []


def test_chunks_respect_max_size():
    text = ("Project Phoenix launch checklist. " * 40).strip()
    chunks = chunk_text(text, chunk_size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_chunks_overlap():
    text = ("Sentence number one here. " * 20).strip()
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) >= 2
    # Some tail text of chunk 0 should reappear at the start of chunk 1.
    assert chunks[0].split()[-1] in chunks[1]


def test_long_sentence_is_hard_split():
    chunks = chunk_text("x" * 1000, chunk_size=300, overlap=60)
    assert all(len(c) <= 300 for c in chunks)
    assert sum(len(c) for c in chunks) >= 1000


@pytest.mark.parametrize("size,overlap", [(100, 100), (100, -1), (0, 0)])
def test_invalid_configuration_raises(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a. b. c.", chunk_size=size, overlap=overlap)
