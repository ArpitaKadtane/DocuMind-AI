"""Tests for the FAISS-backed vector store."""

import numpy as np
import pytest

from app.vector_store import VectorStore


def _unit(*values):
    vec = np.array([values], dtype="float32")
    return vec / np.linalg.norm(vec)


def test_search_returns_most_similar_chunk():
    store = VectorStore()
    chunks = ["about cats", "about budgets", "about coffee"]
    embeddings = np.eye(3, dtype="float32")
    store.add(chunks, embeddings)

    results = store.search(_unit(0.0, 1.0, 0.0)[0], top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "about budgets"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_top_k_is_capped_at_number_of_chunks():
    store = VectorStore()
    store.add(["a", "b"], np.eye(2, dtype="float32"))
    results = store.search(_unit(1.0, 0.0)[0], top_k=10)
    assert len(results) == 2


def test_search_on_empty_store_returns_nothing():
    store = VectorStore()
    assert store.size == 0
    assert store.search(_unit(1.0, 0.0)[0], top_k=3) == []


def test_mismatched_counts_raise():
    store = VectorStore()
    with pytest.raises(ValueError):
        store.add(["only one chunk"], np.eye(3, dtype="float32"))
