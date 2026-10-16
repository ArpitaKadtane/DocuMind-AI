"""Ownership enforcement: users can only reach their own documents.

Missing ids and other users' ids must be indistinguishable (identical 404),
and documents uploaded before auth existed (owner_id NULL) stay inaccessible.
"""

from app import main
from app.vector_store import VectorStore


def _upload(client, make_pdf, text="The budget is 5000 credits"):
    return client.post(
        "/documents", files={"file": ("doc.pdf", make_pdf([text]), "application/pdf")}
    )


def _inject_legacy(document_id="legacy-doc"):
    main.DOCUMENTS[document_id] = {
        "filename": "old.pdf",
        "owner_id": None,
        "pages": 1,
        "characters": 5,
        "chunks": [],
        "store": VectorStore(),
        "created_at": "2026-01-01T00:00:00+00:00",
    }


def test_upload_requires_authentication(client, make_pdf):
    response = client.post(
        "/documents", files={"file": ("doc.pdf", make_pdf(["hi"]), "application/pdf")}
    )
    assert response.status_code == 401


def test_list_requires_authentication(client):
    assert client.get("/documents").status_code == 401


def test_owner_can_access_own_document(auth_client, make_pdf, monkeypatch):
    document_id = _upload(auth_client, make_pdf).json()["document_id"]

    assert auth_client.get(f"/documents/{document_id}").status_code == 200
    assert (
        auth_client.post(f"/documents/{document_id}/search?query=budget").status_code
        == 200
    )

    monkeypatch.setattr(
        main,
        "answer_question",
        lambda q, s: {"answer": "5000 credits", "found": True, "sources": ["ctx"]},
    )
    assert (
        auth_client.post(
            f"/documents/{document_id}/ask", json={"question": "budget?"}
        ).status_code
        == 200
    )


def test_other_user_cannot_read_metadata(make_auth_client, make_pdf):
    alice = make_auth_client("alice@example.com")
    bob = make_auth_client("bob@example.com")
    document_id = _upload(alice, make_pdf).json()["document_id"]

    assert bob.get(f"/documents/{document_id}").status_code == 404


def test_other_user_cannot_search(make_auth_client, make_pdf):
    alice = make_auth_client("alice@example.com")
    bob = make_auth_client("bob@example.com")
    document_id = _upload(alice, make_pdf).json()["document_id"]

    assert (
        bob.post(f"/documents/{document_id}/search?query=budget").status_code == 404
    )


def test_other_user_cannot_ask(make_auth_client, make_pdf):
    alice = make_auth_client("alice@example.com")
    bob = make_auth_client("bob@example.com")
    document_id = _upload(alice, make_pdf).json()["document_id"]

    response = bob.post(
        f"/documents/{document_id}/ask", json={"question": "What is the budget?"}
    )
    assert response.status_code == 404


def test_unknown_and_foreign_ids_are_indistinguishable(make_auth_client, make_pdf):
    alice = make_auth_client("alice@example.com")
    bob = make_auth_client("bob@example.com")
    document_id = _upload(alice, make_pdf).json()["document_id"]

    foreign = bob.get(f"/documents/{document_id}")
    unknown = bob.get("/documents/does-not-exist")

    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json() == unknown.json()


def test_list_returns_only_own_documents(make_auth_client, make_pdf):
    alice = make_auth_client("alice@example.com")
    bob = make_auth_client("bob@example.com")
    alice_id = _upload(alice, make_pdf, "Alice document").json()["document_id"]
    bob_id = _upload(bob, make_pdf, "Bob document").json()["document_id"]

    alice_docs = alice.get("/documents").json()["documents"]
    assert [doc["document_id"] for doc in alice_docs] == [alice_id]

    # Bob's document must not leak into Alice's listing.
    assert all(doc["document_id"] != bob_id for doc in alice_docs)


def test_legacy_unowned_document_is_inaccessible(auth_client, make_pdf):
    _inject_legacy("legacy-doc")

    assert auth_client.get("/documents/legacy-doc").status_code == 404
    assert (
        auth_client.post("/documents/legacy-doc/search?query=anything").status_code
        == 404
    )
    assert (
        auth_client.post(
            "/documents/legacy-doc/ask", json={"question": "anything?"}
        ).status_code
        == 404
    )
    assert all(
        doc["document_id"] != "legacy-doc"
        for doc in auth_client.get("/documents").json()["documents"]
    )


def test_claim_legacy_documents_assigns_owner(make_auth_client):
    client = make_auth_client("owner@example.com")
    store = main.get_document_store()
    store.save_document(
        document_id="old",
        filename="old.pdf",
        stored_bytes=b"%PDF-1.4",
        pages=1,
        characters=5,
        chunks=[],
        embeddings=None,
        embedding_model="m",
        created_at="2026-01-01T00:00:00+00:00",
    )
    assert store.get_document("old")["owner_id"] is None

    owner_id = client.get("/auth/me").json()["user_id"]
    assert store.claim_legacy_documents(owner_id) == 1
    assert store.get_document("old")["owner_id"] == owner_id
