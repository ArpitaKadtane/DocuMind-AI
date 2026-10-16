"""Tests for SQLite-backed persistence and restart restoration."""

import sqlite3

import numpy as np
import pytest

from app import main
from app.storage import DocumentStore


def _save(store, document_id="d1", chunks=("alpha", "beta"), dim=3, **overrides):
    chunk_list = list(chunks)
    embeddings = (
        np.eye(len(chunk_list), dim, dtype="float32") if chunk_list else None
    )
    payload = dict(
        document_id=document_id,
        filename="notes.pdf",
        stored_bytes=b"%PDF-1.4 test",
        pages=2,
        characters=15,
        chunks=chunk_list,
        embeddings=embeddings,
        embedding_model="test-model",
        created_at="2026-01-01T00:00:00+00:00",
    )
    payload.update(overrides)
    store.save_document(**payload)


def test_document_store_round_trip(tmp_path):
    store = DocumentStore(tmp_path / "data")
    _save(store, chunks=("alpha", "beta", "gamma"))

    docs = store.iter_documents()
    assert len(docs) == 1
    meta = docs[0]
    assert meta["document_id"] == "d1"
    assert meta["filename"] == "notes.pdf"
    assert meta["pages"] == 2
    assert meta["characters"] == 15
    assert meta["chunk_count"] == 3
    assert meta["embedding_dim"] == 3
    assert store.count() == 1

    chunks, embeddings = store.load_chunks("d1")
    assert chunks == ["alpha", "beta", "gamma"]
    np.testing.assert_allclose(embeddings, np.eye(3, dtype="float32"))
    assert store.stored_file_path("d1").read_bytes() == b"%PDF-1.4 test"


def test_save_document_is_idempotent(tmp_path):
    store = DocumentStore(tmp_path / "data")
    _save(store, chunks=("a", "b", "c"))
    _save(store, chunks=("only",))

    assert store.count() == 1
    chunks, embeddings = store.load_chunks("d1")
    assert chunks == ["only"]
    assert embeddings.shape == (1, 3)


def test_document_without_chunks_round_trips(tmp_path):
    store = DocumentStore(tmp_path / "data")
    _save(store, chunks=(), dim=0)

    chunks, embeddings = store.load_chunks("d1")
    assert chunks == []
    assert embeddings.shape[0] == 0


def test_stored_filename_is_generated_not_user_supplied(tmp_path):
    store = DocumentStore(tmp_path / "data")
    _save(store, filename="../../etc/passwd")

    path = store.stored_file_path("d1")
    assert path.name == "d1.pdf"
    assert path.parent == store.files_dir
    assert path.exists()


def test_documents_restored_and_searchable_after_restart(
    auth_client, make_pdf, monkeypatch
):
    # The autouse "isolated_storage" fixture already points the app at a
    # throwaway directory, so users, sessions, and documents all live there.
    main.DOCUMENTS.clear()

    pdf = make_pdf(["The budget is 5000 credits"])
    upload = auth_client.post(
        "/documents", files={"file": ("doc.pdf", pdf, "application/pdf")}
    )
    assert upload.status_code == 200
    document_id = upload.json()["document_id"]

    # Simulate a server restart: drop the in-memory cache and reload from disk.
    main.DOCUMENTS.clear()
    restored = main.load_persisted_documents()
    assert restored == 1
    assert document_id in main.DOCUMENTS
    # Ownership survives the restart, so the uploader can still reach it.
    assert main.DOCUMENTS[document_id]["owner_id"] == main.get_document_store().get_document(
        document_id
    )["owner_id"]

    # Metadata survives the restart.
    meta = auth_client.get(f"/documents/{document_id}")
    assert meta.status_code == 200
    assert meta.json()["chunks"] >= 1
    assert meta.json()["filename"] == "doc.pdf"

    # Search works against the restored index.
    search = auth_client.post(f"/documents/{document_id}/search?query=budget")
    assert search.status_code == 200
    assert len(search.json()["matches"]) >= 1

    # Ask works against the restored document (LLM replaced with a stub).
    monkeypatch.setattr(
        main,
        "answer_question",
        lambda q, s, filename=None: {
            "answer": "5000 credits",
            "found": True,
            "sources": ["ctx"],
        },
    )
    ask = auth_client.post(
        f"/documents/{document_id}/ask", json={"question": "What is the budget?"}
    )
    assert ask.status_code == 200
    assert ask.json()["found"] is True


def test_owner_id_column_is_added_to_legacy_database(tmp_path):
    """A database created before auth existed must be migrated in place."""
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    db_path = legacy_dir / "documents.sqlite3"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE documents (document_id TEXT PRIMARY KEY, "
            "filename TEXT NOT NULL, stored_filename TEXT NOT NULL, "
            "size_bytes INTEGER NOT NULL, pages INTEGER, characters INTEGER "
            "NOT NULL, chunk_count INTEGER NOT NULL, embedding_model TEXT, "
            "embedding_dim INTEGER, created_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO documents VALUES "
            "('old','a.pdf','old.pdf',3,1,3,1,'m',3,'2026-01-01T00:00:00+00:00')"
        )

    store = DocumentStore(legacy_dir)

    # The migration adds the column; the pre-existing row is left unowned.
    document = store.get_document("old")
    assert document is not None
    assert document["owner_id"] is None
    assert store.iter_documents(owner_id="nobody") == []


def test_missing_stored_file_still_restores_search(tmp_path, caplog):
    store = DocumentStore(tmp_path / "data")
    _save(store, document_id="ghost", chunks=("hello",))

    store.stored_file_path("ghost").unlink()
    main.DOCUMENTS.clear()

    with caplog.at_level("WARNING"):
        restored = main.load_persisted_documents(store=store)

    assert restored == 1
    assert "ghost" in main.DOCUMENTS
    assert any("Stored file missing" in record.message for record in caplog.records)


def test_corrupt_document_is_skipped_not_fatal(tmp_path, caplog):
    store = DocumentStore(tmp_path / "data")
    _save(store, document_id="good", chunks=("ok",))

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "INSERT INTO documents (document_id, filename, stored_filename, "
            "size_bytes, pages, characters, chunk_count, embedding_model, "
            "embedding_dim, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("bad", "b.pdf", "bad.pdf", 4, 1, 1, 1, "m", 4, "2026-01-01T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO chunks (document_id, ordinal, text, embedding) "
            "VALUES (?,?,?,?)",
            ("bad", 0, "x", b"\x01\x02\x03"),
        )

    main.DOCUMENTS.clear()
    with caplog.at_level("WARNING"):
        restored = main.load_persisted_documents(store=store)

    assert restored == 1
    assert "good" in main.DOCUMENTS
    assert "bad" not in main.DOCUMENTS
    assert any("Skipping unreadable document" in record.message for record in caplog.records)
