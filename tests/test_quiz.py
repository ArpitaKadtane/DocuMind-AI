"""Tests for AI-generated quizzes (M5).

Coverage: context selection and grounding, JSON parsing/validation, page
validation, the in-memory quiz store (expiration + capacity), the two HTTP
endpoints (generation without leaking the answer key, and server-side grading),
ownership, limits, and LLM/error handling. M1-M4 behaviour is guarded by the
rest of the suite.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from app import main
from app.llm import LLMError
from app.quiz import (
    QuizGenerationError,
    QuizStore,
    build_contexts,
    create_quiz,
    parse_quiz,
)
from app.vector_store import VectorStore


def _upload(client, data, filename="doc.pdf"):
    return client.post(
        "/documents", files={"file": (filename, data, "application/pdf")}
    )


def _quiz_json(missing_page=False, page=1, options=4, count=2):
    """A valid raw quiz payload with answer_index = i % options."""
    questions = []
    for i in range(count):
        item = {
            "question": f"What is fact number {i}?",
            "options": [f"option {j}" for j in range(options)],
            "answer_index": i % options,
            "explanation": f"Because fact {i}.",
        }
        if not missing_page:
            item["page"] = page
        questions.append(item)
    return json.dumps({"questions": questions})


# --- build_contexts --------------------------------------------------------


def test_build_contexts_labels_pages():
    contexts, pages, truncated = build_contexts(
        ["The budget is 5000 credits", "Coffee machine broken"], [1, 2], max_chars=1000
    )
    assert contexts == [
        "[page 1] The budget is 5000 credits",
        "[page 2] Coffee machine broken",
    ]
    assert pages == {1, 2}
    assert truncated is False


def test_build_contexts_marks_unavailable_page():
    contexts, pages, _ = build_contexts(["legacy text"], [None], max_chars=1000)
    assert contexts == ["[page unavailable] legacy text"]
    assert pages == set()


def test_build_contexts_samples_large_document():
    chunks = [f"chunk {i} " + "x" * 100 for i in range(20)]
    pages = list(range(1, 21))
    contexts, page_set, truncated = build_contexts(chunks, pages, max_chars=700)

    assert truncated is True
    assert len(contexts) < len(chunks)
    assert page_set.issubset(set(pages))
    assert page_set == {int(c.split("[page ")[1].split("]")[0]) for c in contexts}


# --- parse_quiz ------------------------------------------------------------


def test_parse_quiz_valid():
    result = parse_quiz(_quiz_json(count=2), count=5, num_options=4, valid_pages={1})
    assert len(result) == 2
    assert result[0]["answer_index"] == 0
    assert result[0]["page"] == 1
    assert result[0]["explanation"] == "Because fact 0."


def test_parse_quiz_strips_code_fences():
    raw = "```json\n" + _quiz_json(count=1) + "\n```"
    result = parse_quiz(raw, count=5, num_options=4, valid_pages={1})
    assert len(result) == 1


def test_parse_quiz_accepts_bare_list():
    raw = json.dumps(
        [{"question": "Q?", "options": ["a", "b"], "answer_index": 1, "page": 1}]
    )
    result = parse_quiz(raw, count=5, num_options=4, valid_pages={1})
    assert result[0]["options"] == ["a", "b"]
    assert result[0]["answer_index"] == 1


def test_parse_quiz_malformed_json_raises():
    with pytest.raises(QuizGenerationError):
        parse_quiz("{ this is not json", count=5, num_options=4, valid_pages={1})


def test_parse_quiz_missing_questions_raises():
    with pytest.raises(QuizGenerationError):
        parse_quiz('{"foo": 1}', count=5, num_options=4, valid_pages={1})


def test_parse_quiz_empty_raises():
    with pytest.raises(QuizGenerationError):
        parse_quiz("", count=5, num_options=4, valid_pages={1})


def test_parse_quiz_drops_invalid_items():
    raw = json.dumps(
        {
            "questions": [
                {"question": "Good?", "options": ["a", "b"], "answer_index": 0, "page": 1},
                {"question": "", "options": ["a", "b"], "answer_index": 0},
                {"question": "Bad index?", "options": ["a", "b"], "answer_index": 9},
                {"question": "One option?", "options": ["only"], "answer_index": 0},
                {"question": "Bad options?", "options": "nope", "answer_index": 0},
                "not even a dict",
            ]
        }
    )
    result = parse_quiz(raw, count=5, num_options=4, valid_pages={1})
    assert [q["question"] for q in result] == ["Good?"]


def test_parse_quiz_all_invalid_raises():
    raw = json.dumps(
        {"questions": [{"question": "x", "options": ["a"], "answer_index": 0}]}
    )
    with pytest.raises(QuizGenerationError):
        parse_quiz(raw, count=5, num_options=4, valid_pages={1})


def test_parse_quiz_dedupes_questions():
    item = {"question": "Same?", "options": ["a", "b"], "answer_index": 0, "page": 1}
    raw = json.dumps({"questions": [item, dict(item), dict(item)]})
    result = parse_quiz(raw, count=5, num_options=4, valid_pages={1})
    assert len(result) == 1


def test_parse_quiz_truncates_to_requested_count():
    result = parse_quiz(_quiz_json(count=6), count=2, num_options=4, valid_pages={1})
    assert len(result) == 2


def test_parse_quiz_caps_option_count():
    raw = json.dumps(
        {
            "questions": [
                {
                    "question": "Q?",
                    "options": ["a", "b", "c", "d", "e"],
                    "answer_index": 0,
                    "page": 1,
                }
            ]
        }
    )
    result = parse_quiz(raw, count=5, num_options=3, valid_pages={1})
    assert result[0]["options"] == ["a", "b", "c"]


def test_parse_quiz_rejects_bool_answer_index():
    raw = json.dumps(
        {"questions": [{"question": "Q?", "options": ["a", "b"], "answer_index": True, "page": 1}]}
    )
    with pytest.raises(QuizGenerationError):
        parse_quiz(raw, count=5, num_options=4, valid_pages={1})


def test_parse_quiz_nulls_page_not_in_context():
    raw = json.dumps(
        {"questions": [{"question": "Q?", "options": ["a", "b"], "answer_index": 0, "page": 9}]}
    )
    result = parse_quiz(raw, count=5, num_options=4, valid_pages={1, 2})
    assert result[0]["page"] is None


def test_parse_quiz_keeps_valid_page():
    raw = json.dumps(
        {"questions": [{"question": "Q?", "options": ["a", "b"], "answer_index": 0, "page": 2}]}
    )
    result = parse_quiz(raw, count=5, num_options=4, valid_pages={1, 2})
    assert result[0]["page"] == 2


# --- create_quiz -----------------------------------------------------------


def test_create_quiz_orchestration(monkeypatch):
    monkeypatch.setattr(
        "app.quiz.generate_quiz", lambda contexts, count, num_options: _quiz_json(count=2)
    )
    result = create_quiz(["alpha text", "beta text"], [1, 2], count=5, num_options=4)
    assert len(result["questions"]) == 2
    assert result["truncated"] is False


def test_create_quiz_nulls_pages_outside_context(monkeypatch):
    raw = json.dumps(
        {"questions": [{"question": "Q?", "options": ["a", "b"], "answer_index": 0, "page": 2}]}
    )
    monkeypatch.setattr("app.quiz.generate_quiz", lambda c, n, o: raw)
    # The sampled context only contains page 1, so a claimed page 2 is invalid.
    result = create_quiz(["only page one text"], [1], count=5, num_options=4)
    assert result["questions"][0]["page"] is None


def test_create_quiz_empty_document_raises():
    with pytest.raises(QuizGenerationError):
        create_quiz([], [], count=5, num_options=4)


# --- QuizStore (expiration + capacity) -------------------------------------


def test_quiz_store_get_missing_returns_none():
    store = QuizStore(ttl_minutes=10, max_stored=5)
    assert store.get("nope") is None


def test_quiz_store_expires_old_quizzes():
    store = QuizStore(ttl_minutes=10, max_stored=5)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.add(
        quiz_id="q1", document_id="d", owner_id="u", filename="f.pdf",
        questions=[], now=t0,
    )
    assert store.get("q1", now=t0) is not None
    assert store.get("q1", now=t0 + timedelta(minutes=11)) is None
    assert len(store) == 0


def test_quiz_store_evicts_oldest_beyond_capacity():
    store = QuizStore(ttl_minutes=60, max_stored=2)
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i, qid in enumerate(["a", "b", "c"]):
        store.add(
            quiz_id=qid, document_id="d", owner_id="u", filename="f",
            questions=[], now=t + timedelta(seconds=i),
        )
    assert len(store) == 2
    assert store.get("a", now=t + timedelta(seconds=2)) is None
    assert store.get("c", now=t + timedelta(seconds=2)) is not None


# --- HTTP: generate --------------------------------------------------------


def test_generate_quiz_success_does_not_leak_answers(auth_client, make_pdf, monkeypatch):
    document_id = _upload(auth_client, make_pdf(["The budget is 5000 credits"])).json()[
        "document_id"
    ]
    monkeypatch.setattr(
        "app.quiz.generate_quiz", lambda c, n, o: _quiz_json(count=3)
    )

    response = auth_client.post(f"/documents/{document_id}/quiz", json={"count": 3})
    assert response.status_code == 200
    body = response.json()

    assert body["quiz_id"]
    assert body["document_id"] == document_id
    assert body["count"] == 3
    assert len(body["questions"]) == 3
    for question in body["questions"]:
        assert set(question.keys()) == {"index", "question", "options", "page"}

    serialized = json.dumps(body)
    assert "answer_index" not in serialized
    assert "correct_index" not in serialized
    assert "explanation" not in serialized


def test_generate_quiz_defaults(auth_client, make_pdf, monkeypatch):
    document_id = _upload(auth_client, make_pdf(["hello world"])).json()["document_id"]
    captured = {}

    def fake(contexts, count, num_options):
        captured["count"] = count
        captured["num_options"] = num_options
        return _quiz_json(count=5)

    monkeypatch.setattr("app.quiz.generate_quiz", fake)
    response = auth_client.post(f"/documents/{document_id}/quiz", json={})
    assert response.status_code == 200
    assert captured == {"count": 5, "num_options": 4}


def test_generate_quiz_rejects_invalid_count(auth_client, make_pdf):
    document_id = _upload(auth_client, make_pdf(["hello"])).json()["document_id"]
    assert auth_client.post(f"/documents/{document_id}/quiz", json={"count": 0}).status_code == 422
    assert auth_client.post(f"/documents/{document_id}/quiz", json={"count": 999}).status_code == 422


def test_generate_quiz_rejects_invalid_option_count(auth_client, make_pdf):
    document_id = _upload(auth_client, make_pdf(["hello"])).json()["document_id"]
    assert auth_client.post(f"/documents/{document_id}/quiz", json={"num_options": 1}).status_code == 422
    assert auth_client.post(f"/documents/{document_id}/quiz", json={"num_options": 99}).status_code == 422


def test_generate_quiz_requires_authentication(client, auth_client, make_pdf):
    document_id = _upload(auth_client, make_pdf(["hello"])).json()["document_id"]
    assert client.post(f"/documents/{document_id}/quiz", json={}).status_code == 401


def test_generate_quiz_unknown_document_404(auth_client):
    assert auth_client.post("/documents/does-not-exist/quiz", json={}).status_code == 404


def test_generate_quiz_foreign_document_404(make_auth_client, make_pdf):
    alice = make_auth_client("alice-quiz@example.com")
    bob = make_auth_client("bob-quiz@example.com")
    document_id = _upload(alice, make_pdf(["secret budget"])).json()["document_id"]
    assert bob.post(f"/documents/{document_id}/quiz", json={}).status_code == 404


def test_generate_quiz_llm_error_is_502(auth_client, make_pdf, monkeypatch):
    document_id = _upload(auth_client, make_pdf(["hello world"])).json()["document_id"]

    def boom(contexts, count, num_options):
        raise LLMError("LLM unreachable")

    monkeypatch.setattr("app.quiz.generate_quiz", boom)
    response = auth_client.post(f"/documents/{document_id}/quiz", json={})
    assert response.status_code == 502
    assert "LLM unreachable" in response.json()["detail"]


def test_generate_quiz_malformed_output_is_502(auth_client, make_pdf, monkeypatch):
    document_id = _upload(auth_client, make_pdf(["hello world"])).json()["document_id"]
    monkeypatch.setattr("app.quiz.generate_quiz", lambda c, n, o: "{ not valid json")
    response = auth_client.post(f"/documents/{document_id}/quiz", json={})
    assert response.status_code == 502


def test_generate_quiz_empty_document_is_400(auth_client):
    user_id = auth_client.get("/auth/me").json()["user_id"]
    main.DOCUMENTS["empty-doc"] = {
        "filename": "empty.pdf",
        "owner_id": user_id,
        "pages": 0,
        "characters": 0,
        "chunks": [],
        "chunk_pages": [],
        "store": VectorStore(),
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    response = auth_client.post("/documents/empty-doc/quiz", json={})
    assert response.status_code == 400


# --- HTTP: submit ----------------------------------------------------------


def _generate(auth_client, make_pdf, monkeypatch, count=3):
    document_id = _upload(auth_client, make_pdf(["The budget is 5000 credits"])).json()[
        "document_id"
    ]
    monkeypatch.setattr(
        "app.quiz.generate_quiz", lambda c, n, o: _quiz_json(count=count)
    )
    quiz_id = auth_client.post(
        f"/documents/{document_id}/quiz", json={"count": count}
    ).json()["quiz_id"]
    return document_id, quiz_id


def test_submit_quiz_grades_answers(auth_client, make_pdf, monkeypatch):
    document_id, quiz_id = _generate(auth_client, make_pdf, monkeypatch, count=3)
    # correct indices are [0, 1, 2]; answer the first two correctly, third wrong.
    response = auth_client.post(
        f"/documents/{document_id}/quiz/{quiz_id}/submit",
        json={"answers": [0, 1, 0]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["score"] == {"correct": 2, "total": 3, "percent": 67}
    assert body["results"][2]["correct"] is False
    assert body["results"][2]["correct_index"] == 2
    assert body["results"][1]["correct"] is True
    assert body["results"][0]["explanation"]


def test_submit_quiz_unanswered_counted_wrong(auth_client, make_pdf, monkeypatch):
    document_id, quiz_id = _generate(auth_client, make_pdf, monkeypatch, count=2)
    response = auth_client.post(
        f"/documents/{document_id}/quiz/{quiz_id}/submit",
        json={"answers": [None, None]},
    )
    assert response.status_code == 200
    assert response.json()["score"]["correct"] == 0


def test_submit_quiz_wrong_answer_length_is_422(auth_client, make_pdf, monkeypatch):
    document_id, quiz_id = _generate(auth_client, make_pdf, monkeypatch, count=3)
    response = auth_client.post(
        f"/documents/{document_id}/quiz/{quiz_id}/submit", json={"answers": [0]}
    )
    assert response.status_code == 422


def test_submit_quiz_out_of_range_answer_is_422(auth_client, make_pdf, monkeypatch):
    document_id, quiz_id = _generate(auth_client, make_pdf, monkeypatch, count=2)
    response = auth_client.post(
        f"/documents/{document_id}/quiz/{quiz_id}/submit",
        json={"answers": [99, 0]},
    )
    assert response.status_code == 422


def test_submit_unknown_quiz_is_404(auth_client, make_pdf):
    document_id = _upload(auth_client, make_pdf(["hello"])).json()["document_id"]
    response = auth_client.post(
        f"/documents/{document_id}/quiz/nope/submit", json={"answers": []}
    )
    assert response.status_code == 404


def test_submit_quiz_document_mismatch_is_404(auth_client, make_pdf, monkeypatch):
    other = _upload(auth_client, make_pdf(["another document"])).json()["document_id"]
    document_id, quiz_id = _generate(auth_client, make_pdf, monkeypatch, count=1)
    # Same owner, but the quiz belongs to a different document.
    response = auth_client.post(
        f"/documents/{other}/quiz/{quiz_id}/submit", json={"answers": [0]}
    )
    assert response.status_code == 404


def test_submit_foreign_quiz_is_404(make_auth_client, make_pdf, monkeypatch):
    alice = make_auth_client("alice-sub@example.com")
    bob = make_auth_client("bob-sub@example.com")
    document_id, quiz_id = _generate(alice, make_pdf, monkeypatch, count=1)
    response = bob.post(
        f"/documents/{document_id}/quiz/{quiz_id}/submit", json={"answers": [0]}
    )
    assert response.status_code == 404


def test_submit_requires_authentication(client, auth_client, make_pdf, monkeypatch):
    document_id, quiz_id = _generate(auth_client, make_pdf, monkeypatch, count=1)
    response = client.post(
        f"/documents/{document_id}/quiz/{quiz_id}/submit", json={"answers": [0]}
    )
    assert response.status_code == 401
