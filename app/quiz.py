"""AI-generated multiple-choice quizzes, grounded in a document.

A quiz is requested in two steps so the answer key never reaches the browser
before the user has answered (see ``app/main.py``):

  1. ``create_quiz`` builds questions from the document's own chunks. Like
     summaries, large documents are sampled evenly across their chunks so the
     whole document is covered within the model's context window.
  2. The generated questions, including the correct option and explanation,
     are held **server-side**. Only the question text and options are returned;
     grading happens later, on submit.

Model output is untrusted: it is parsed as JSON, then every question and option
is validated. Malformed or duplicate questions are dropped rather than
repaired, and the requested count and option limits are enforced. Page numbers
are accepted only when they refer to a page that actually appeared in the
sampled context; otherwise the page becomes ``None`` ("unavailable"). No fact,
option, or page number is ever invented.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.llm import generate_quiz
from app.summary import select_chunks


class QuizGenerationError(Exception):
    """Raised when the model's quiz output cannot be turned into questions."""


def _label(page) -> str:
    """Render a page number as a human label for the model."""
    return f"page {page}" if page is not None else "page unavailable"


def build_contexts(
    chunks: list[str], pages: list, max_chars: int | None = None
) -> tuple[list[str], set[int], bool]:
    """Build labelled contexts from a document's chunks.

    Returns ``(contexts, source_pages, truncated)``. ``source_pages`` is the set
    of real page numbers that actually appear in the selected context -- the
    only pages a generated question may legitimately cite. ``truncated`` is True
    when the document was too large to send whole and was sampled evenly.
    """
    if not chunks:
        return [], set(), False

    budget = settings.quiz_max_chars if max_chars is None else max_chars
    n = len(chunks)
    avg_len = max(1, sum(len(chunk) for chunk in chunks) // n)
    max_chunks = max(1, budget // avg_len)
    indices, truncated = select_chunks(chunks, budget, max_chunks)

    contexts: list[str] = []
    source_pages: set[int] = set()
    for index in indices:
        page = pages[index] if index < len(pages) else None
        if page is not None:
            source_pages.add(page)
        contexts.append(f"[{_label(page)}] {chunks[index]}")
    return contexts, source_pages, truncated


def _extract_json(raw: str) -> object:
    """Pull a JSON value out of the model's reply, tolerating code fences."""
    text = (raw or "").strip()
    if not text:
        raise QuizGenerationError("The model returned an empty quiz.")
    text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    # The reply may be a JSON object or a bare array; the earliest opening
    # bracket of either kind marks the start, the latest closing bracket the end.
    starts = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
    ends = [pos for pos in (text.rfind("}"), text.rfind("]")) if pos != -1]
    if not starts or not ends:
        raise QuizGenerationError("The model did not return valid JSON.")
    start, end = min(starts), max(ends)
    if end <= start:
        raise QuizGenerationError("The model did not return valid JSON.")
    try:
        return json.loads(text[start : end + 1])
    except (ValueError, TypeError) as exc:
        raise QuizGenerationError("The model did not return valid JSON.") from exc


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _clean_question(item: object, num_options: int, valid_pages: set[int]) -> dict | None:
    """Validate one raw question, or return None if it is unusable."""
    if not isinstance(item, dict):
        return None

    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        return None

    options = item.get("options")
    if not isinstance(options, list):
        return None
    options = [opt.strip() for opt in options if isinstance(opt, str) and opt.strip()]
    if len(options) < 2:
        return None
    if len(options) > num_options:
        options = options[:num_options]

    answer_index = item.get("answer_index")
    if isinstance(answer_index, bool) or not isinstance(answer_index, int):
        return None
    if not 0 <= answer_index < len(options):
        return None

    explanation = item.get("explanation")
    explanation = explanation.strip() if isinstance(explanation, str) else ""

    page = item.get("page")
    if isinstance(page, bool) or not isinstance(page, int) or page not in valid_pages:
        page = None

    return {
        "question": question.strip(),
        "options": options,
        "answer_index": answer_index,
        "explanation": explanation,
        "page": page,
    }


def parse_quiz(
    raw: str, count: int, num_options: int, valid_pages: set[int]
) -> list[dict]:
    """Parse and validate the model's quiz, returning at most ``count`` items."""
    data = _extract_json(raw)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("questions")
    else:
        items = None
    if not isinstance(items, list):
        raise QuizGenerationError("The model's quiz was missing its questions.")

    questions: list[dict] = []
    seen: set[str] = set()
    for item in items:
        question = _clean_question(item, num_options, valid_pages)
        if question is None:
            continue
        key = _normalize(question["question"])
        if key in seen:
            continue
        seen.add(key)
        questions.append(question)
        if len(questions) >= count:
            break

    if not questions:
        raise QuizGenerationError("The model did not produce any usable questions.")
    return questions


def create_quiz(
    chunks: list[str], pages: list, count: int, num_options: int
) -> dict:
    """Generate and validate a quiz for a document.

    Returns ``{"questions": [...], "truncated": bool}``. Each question carries
    its ``answer_index`` and ``explanation`` for server-side grading; callers
    must not expose those before submission.
    """
    contexts, source_pages, truncated = build_contexts(chunks, pages)
    if not contexts:
        raise QuizGenerationError("This document has no text to build a quiz from.")

    raw = generate_quiz(contexts, count, num_options)
    questions = parse_quiz(raw, count, num_options, source_pages)
    return {"questions": questions, "truncated": truncated}


class QuizStore:
    """A small in-memory store for generated quizzes.

    Quizzes are ephemeral: they are discarded once they expire, and the oldest
    are evicted when the store grows past ``max_stored`` so memory stays bounded.
    Nothing is persisted, so quizzes do not survive a restart and are not shared
    between worker processes.
    """

    def __init__(self, *, ttl_minutes: int, max_stored: int) -> None:
        self._ttl = timedelta(minutes=max(1, ttl_minutes))
        self._max = max(1, max_stored)
        self._items: dict[str, dict] = {}

    def _prune(self, now: datetime) -> None:
        expired = [key for key, quiz in self._items.items() if quiz["expires_at"] <= now]
        for key in expired:
            self._items.pop(key, None)
        if len(self._items) > self._max:
            ordered = sorted(self._items.items(), key=lambda kv: kv[1]["created_at"])
            for key, _ in ordered[: len(self._items) - self._max]:
                self._items.pop(key, None)

    def add(
        self,
        *,
        quiz_id: str,
        document_id: str,
        owner_id: str | None,
        filename: str,
        questions: list[dict],
        now: datetime | None = None,
    ) -> dict:
        now = now or datetime.now(timezone.utc)
        self._prune(now)
        quiz = {
            "quiz_id": quiz_id,
            "document_id": document_id,
            "owner_id": owner_id,
            "filename": filename,
            "created_at": now,
            "expires_at": now + self._ttl,
            "questions": questions,
        }
        self._items[quiz_id] = quiz
        self._prune(now)
        return quiz

    def get(self, quiz_id: str, now: datetime | None = None) -> dict | None:
        self._prune(now or datetime.now(timezone.utc))
        return self._items.get(quiz_id)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
