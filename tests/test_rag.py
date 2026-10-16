"""Tests for the RAG decision logic (score gating and not-found handling)."""

import pytest

from app import rag
from app.llm import NOT_FOUND_MESSAGE


class StubStore:
    def __init__(self, results=None, size=1):
        self._results = results or []
        self._size = size

    @property
    def size(self):
        return self._size

    def search(self, query_vector, top_k):
        return self._results


def test_empty_store_returns_not_found_without_calling_llm(monkeypatch):
    called = False

    def boom(*args, **kwargs):
        nonlocal called
        called = True
        return "should not run"

    monkeypatch.setattr(rag, "generate_answer", boom)
    result = rag.answer_question("anything?", StubStore(size=0))

    assert result["found"] is False
    assert result["answer"] == NOT_FOUND_MESSAGE
    assert called is False


def test_low_score_is_gated_without_calling_llm(monkeypatch):
    called = False

    def boom(*args, **kwargs):
        nonlocal called
        called = True
        return "should not run"

    monkeypatch.setattr(rag, "generate_answer", boom)
    store = StubStore(results=[{"text": "irrelevant", "score": 0.05}])
    result = rag.answer_question("something unrelated", store)

    assert result["found"] is False
    assert called is False


def test_high_score_calls_llm_and_returns_answer(monkeypatch):
    captured = {}

    def fake_answer(question, contexts):
        captured["question"] = question
        captured["contexts"] = contexts
        return "The budget is 5000 credits."

    monkeypatch.setattr(rag, "generate_answer", fake_answer)
    store = StubStore(results=[{"text": "The budget is 5000 credits.", "score": 0.8}])
    result = rag.answer_question("What is the budget?", store)

    assert result["found"] is True
    assert result["answer"] == "The budget is 5000 credits."
    assert captured["contexts"] == ["The budget is 5000 credits."]


def test_llm_saying_not_found_marks_found_false(monkeypatch):
    monkeypatch.setattr(rag, "generate_answer", lambda q, c: NOT_FOUND_MESSAGE)
    store = StubStore(results=[{"text": "some text", "score": 0.9}])
    result = rag.answer_question("unanswerable", store)

    assert result["found"] is False
    assert result["answer"] == NOT_FOUND_MESSAGE
