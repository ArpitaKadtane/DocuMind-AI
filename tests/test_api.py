"""End-to-end tests of the HTTP API using FastAPI's TestClient."""

from app import main
from app.llm import LLMError


def _upload(client, data: bytes, filename="doc.pdf", content_type="application/pdf"):
    return client.post(
        "/documents", files={"file": (filename, data, content_type)}
    )


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_serves_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "GenAI Document Assistant" in response.text


def test_upload_valid_pdf(client, make_pdf):
    response = _upload(client, make_pdf(["Budget is 5000 credits"]))
    assert response.status_code == 200
    body = response.json()
    assert body["pages"] == 1
    assert body["chunks"] >= 1
    assert "document_id" in body


def test_upload_rejects_wrong_content_type(client, make_pdf):
    response = _upload(
        client, make_pdf(["hello"]), filename="notes.txt", content_type="text/plain"
    )
    assert response.status_code == 415


def test_upload_rejects_fake_pdf(client):
    response = _upload(client, b"this is not really a pdf")
    assert response.status_code == 400


def test_document_metadata_and_missing(client, make_pdf):
    document_id = _upload(client, make_pdf(["hello world"])).json()["document_id"]

    found = client.get(f"/documents/{document_id}")
    assert found.status_code == 200
    assert found.json()["document_id"] == document_id

    missing = client.get("/documents/does-not-exist")
    assert missing.status_code == 404


def test_search_returns_matches(client, make_pdf):
    document_id = _upload(
        client, make_pdf(["The budget is 5000 credits", "Coffee machine is broken"])
    ).json()["document_id"]

    response = client.post(f"/documents/{document_id}/search?query=budget")
    assert response.status_code == 200
    assert len(response.json()["matches"]) >= 1


def test_ask_success(client, make_pdf, monkeypatch):
    document_id = _upload(client, make_pdf(["The budget is 5000 credits"])).json()[
        "document_id"
    ]
    monkeypatch.setattr(
        main,
        "answer_question",
        lambda q, store: {"answer": "5000 credits", "found": True, "sources": ["ctx"]},
    )
    response = client.post(
        f"/documents/{document_id}/ask", json={"question": "What is the budget?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "5000 credits"
    assert body["found"] is True


def test_ask_returns_502_on_llm_error(client, make_pdf, monkeypatch):
    document_id = _upload(client, make_pdf(["The budget is 5000 credits"])).json()[
        "document_id"
    ]

    def boom(q, store):
        raise LLMError("LLM unreachable")

    monkeypatch.setattr(main, "answer_question", boom)
    response = client.post(
        f"/documents/{document_id}/ask", json={"question": "What is the budget?"}
    )
    assert response.status_code == 502
    assert "LLM unreachable" in response.json()["detail"]


def test_ask_unknown_document_is_404(client):
    response = client.post("/documents/nope/ask", json={"question": "hello"})
    assert response.status_code == 404


def test_ask_rejects_empty_question(client, make_pdf):
    document_id = _upload(client, make_pdf(["hello world"])).json()["document_id"]
    response = client.post(
        f"/documents/{document_id}/ask", json={"question": ""}
    )
    assert response.status_code == 422


def test_oversized_declared_body_is_rejected(client):
    # A lying Content-Length header must be caught early by the middleware.
    response = client.post(
        "/documents/abc/ask",
        json={"question": "hi"},
        headers={"content-length": "999999999"},
    )
    assert response.status_code == 413
