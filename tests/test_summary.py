"""Tests for whole-document summarization (M4).

The embedding model and LLM are replaced with fakes elsewhere; here we focus on
chunk selection (large-PDF handling), output parsing, the orchestration in
``app.summary``, and the HTTP endpoint (auth, ownership, errors).
"""

from app import summary as summary_mod
from app.llm import LLMError


def _upload(client, data: bytes, filename="doc.pdf", content_type="application/pdf"):
    return client.post("/documents", files={"file": (filename, data, content_type)})


# --- parse_summary ---------------------------------------------------------


def test_parse_summary_extracts_summary_and_points():
    raw = (
        "SUMMARY: The report covers the budget and timeline.\n"
        "KEY POINTS:\n"
        "- The budget is 5000 credits\n"
        "- Launch is in Q3"
    )
    summary, points = summary_mod.parse_summary(raw)
    assert summary == "The report covers the budget and timeline."
    assert points == ["The budget is 5000 credits", "Launch is in Q3"]


def test_parse_summary_handles_numbered_points():
    raw = "SUMMARY: Overview.\nKEY POINTS:\n1. first\n2. second"
    summary, points = summary_mod.parse_summary(raw)
    assert summary == "Overview."
    assert points == ["first", "second"]


def test_parse_summary_missing_marker_falls_back_to_whole_text():
    summary, points = summary_mod.parse_summary("Just a plain summary.")
    assert summary == "Just a plain summary."
    assert points == []


def test_parse_summary_bullets_only():
    summary, points = summary_mod.parse_summary("- alpha\n- beta")
    assert summary == ""
    assert points == ["alpha", "beta"]


def test_parse_summary_empty():
    assert summary_mod.parse_summary("") == ("", [])
    assert summary_mod.parse_summary("   ") == ("", [])


def test_parse_summary_is_case_insensitive():
    raw = "Summary: Top line\nKey Points:\n- one"
    summary, points = summary_mod.parse_summary(raw)
    assert summary == "Top line"
    assert points == ["one"]


# --- select_chunks ---------------------------------------------------------


def test_select_chunks_uses_all_when_small():
    indices, truncated = summary_mod.select_chunks(
        ["a", "b", "c"], max_chars=100, max_chunks=10
    )
    assert indices == [0, 1, 2]
    assert truncated is False


def test_select_chunks_samples_evenly_when_large():
    chunks = ["x" * 100] * 10
    indices, truncated = summary_mod.select_chunks(
        chunks, max_chars=250, max_chunks=2
    )
    assert truncated is True
    assert len(indices) == 2
    assert indices[0] == 0
    assert indices[-1] == 9
    assert indices == sorted(indices)


def test_select_chunks_empty():
    assert summary_mod.select_chunks([], max_chars=100, max_chunks=5) == ([], False)


# --- summarize_document ----------------------------------------------------


def test_summarize_document_grounds_and_labels(monkeypatch):
    captured = {}

    def fake_generate(contexts):
        captured["contexts"] = contexts
        return "SUMMARY: All good.\nKEY POINTS:\n- point one"

    monkeypatch.setattr(summary_mod, "generate_summary", fake_generate)
    result = summary_mod.summarize_document(
        ["Budget is 5000 credits", "Coffee machine broken"],
        [1, 2],
        filename="doc.pdf",
    )

    assert result["found"] is True
    assert result["summary"] == "All good."
    assert result["key_points"] == ["point one"]
    assert result["truncated"] is False
    assert captured["contexts"] == [
        "[page 1] Budget is 5000 credits",
        "[page 2] Coffee machine broken",
    ]
    assert result["sources"][0] == {
        "filename": "doc.pdf",
        "page": 1,
        "text": "Budget is 5000 credits",
    }


def test_summarize_document_reports_missing_page(monkeypatch):
    monkeypatch.setattr(summary_mod, "generate_summary", lambda contexts: "SUMMARY: ok")
    result = summary_mod.summarize_document(["legacy text"], [None], filename="old.pdf")
    assert result["sources"][0]["page"] is None


def test_summarize_document_empty_chunks_skips_llm(monkeypatch):
    called = False

    def fake_generate(contexts):
        nonlocal called
        called = True
        return "x"

    monkeypatch.setattr(summary_mod, "generate_summary", fake_generate)
    result = summary_mod.summarize_document([], [], filename="doc.pdf")

    assert called is False
    assert result == {
        "summary": "",
        "key_points": [],
        "found": False,
        "truncated": False,
        "sources": [],
    }


def test_summarize_document_truncates_large_document(monkeypatch):
    class _Settings:
        summary_max_chars = 700
        summary_max_points = 5

    monkeypatch.setattr(summary_mod, "settings", _Settings)
    monkeypatch.setattr(summary_mod, "generate_summary", lambda contexts: "SUMMARY: ok")

    chunks = [f"chunk number {i} " + "x" * 100 for i in range(20)]
    pages = list(range(1, 21))
    result = summary_mod.summarize_document(chunks, pages, filename="big.pdf")

    assert result["truncated"] is True
    # Sampling respects the char budget and spans the whole document.
    assert sum(len(s["text"]) for s in result["sources"]) <= 700
    assert result["sources"][0]["page"] == 1
    assert result["sources"][-1]["page"] == 20
    selected_pages = [s["page"] for s in result["sources"]]
    assert selected_pages == sorted(selected_pages)


# --- HTTP endpoint ---------------------------------------------------------


def test_summary_endpoint_success(auth_client, make_pdf, monkeypatch):
    document_id = _upload(
        auth_client, make_pdf(["The budget is 5000 credits"])
    ).json()["document_id"]

    monkeypatch.setattr(
        "app.main.summarize_document",
        lambda chunks, pages, filename=None: {
            "summary": "Short summary.",
            "key_points": ["Budget is 5000 credits"],
            "found": True,
            "truncated": False,
            "sources": [{"filename": filename, "page": 1, "text": chunks[0]}],
        },
    )

    response = auth_client.post(f"/documents/{document_id}/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == document_id
    assert body["filename"] == "doc.pdf"
    assert body["summary"] == "Short summary."
    assert body["key_points"] == ["Budget is 5000 credits"]
    assert body["found"] is True
    assert body["truncated"] is False
    assert body["sources"][0]["page"] == 1


def test_summary_endpoint_uses_document_chunks(auth_client, make_pdf, monkeypatch):
    document_id = _upload(
        auth_client, make_pdf(["The budget is 5000 credits"])
    ).json()["document_id"]

    captured = {}

    def fake_generate(contexts):
        captured["contexts"] = contexts
        return "SUMMARY: grounded.\nKEY POINTS:\n- Budget is 5000 credits"

    # Run the real summarize_document but stub only the LLM call.
    monkeypatch.setattr("app.summary.generate_summary", fake_generate)

    response = auth_client.post(f"/documents/{document_id}/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "grounded."
    assert body["key_points"] == ["Budget is 5000 credits"]
    assert captured["contexts"]
    assert "[page 1]" in captured["contexts"][0]


def test_summary_unknown_document_is_404(auth_client):
    response = auth_client.post("/documents/does-not-exist/summary")
    assert response.status_code == 404


def test_summary_requires_authentication(client, auth_client, make_pdf):
    document_id = _upload(auth_client, make_pdf(["hello"])).json()["document_id"]
    response = client.post(f"/documents/{document_id}/summary")
    assert response.status_code == 401


def test_summary_cross_user_is_404(make_auth_client, make_pdf):
    owner = make_auth_client("owner@example.com")
    other = make_auth_client("other@example.com")
    document_id = _upload(owner, make_pdf(["secret budget"])).json()["document_id"]
    response = other.post(f"/documents/{document_id}/summary")
    assert response.status_code == 404


def test_summary_returns_502_on_llm_error(auth_client, make_pdf, monkeypatch):
    document_id = _upload(auth_client, make_pdf(["hello world"])).json()["document_id"]

    def boom(chunks, pages, filename=None):
        raise LLMError("LLM unreachable")

    monkeypatch.setattr("app.main.summarize_document", boom)
    response = auth_client.post(f"/documents/{document_id}/summary")
    assert response.status_code == 502
    assert "LLM unreachable" in response.json()["detail"]
