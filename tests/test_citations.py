"""Tests for page-aware PDF citations.

These cover the M3 behaviour end to end: per-page extraction, chunks that
remember their source page, page metadata flowing through the vector store and
SQLite, filename+page in search results, page citations from the RAG pipeline,
and safe "page unavailable" handling for legacy data.
"""

import sqlite3

import numpy as np

from app import main, rag
from app.chunking import chunk_pages
from app.embeddings import embed_texts
from app.pdf_processor import extract_pages_from_pdf
from app.storage import DocumentStore
from app.vector_store import VectorStore


def _upload(client, data, filename="report.pdf"):
    return client.post(
        "/documents", files={"file": (filename, data, "application/pdf")}
    )


def _two_page(report_lines, other_lines):
    return [report_lines, other_lines]


# --- extraction and chunking ----------------------------------------------


def test_extract_pages_returns_one_based_page_numbers(make_multipage_pdf):
    data = make_multipage_pdf([["Page one alpha"], ["Page two beta"]])
    pages, page_count = extract_pages_from_pdf(data)

    assert page_count == 2
    assert [number for number, _ in pages] == [1, 2]
    assert "alpha" in pages[0][1]
    assert "beta" in pages[1][1]


def test_chunk_pages_attaches_source_page(make_multipage_pdf):
    data = make_multipage_pdf([["alpha alpha alpha"], ["beta beta beta"]])
    pages, _ = extract_pages_from_pdf(data)
    records = chunk_pages(pages)

    assert records
    assert records[0]["page"] == 1
    assert "alpha" in records[0]["text"]
    assert records[-1]["page"] == 2
    assert "beta" in records[-1]["text"]


# --- vector store ----------------------------------------------------------


def test_vector_store_search_returns_page():
    store = VectorStore()
    chunks = ["alpha", "beta"]
    store.add(chunks, embed_texts(chunks), pages=[1, 2])

    results = store.search(embed_texts(["alpha"])[0], top_k=1)
    assert results[0]["page"] == 1


def test_vector_store_defaults_page_to_none():
    store = VectorStore()
    store.add(["alpha"], embed_texts(["alpha"]))

    results = store.search(embed_texts(["alpha"])[0], top_k=1)
    assert results[0]["page"] is None


# --- RAG citations ---------------------------------------------------------


def test_answer_cites_pages_from_multipage_document(make_multipage_pdf, monkeypatch):
    data = make_multipage_pdf(
        _two_page(["The budget is 5000 credits"], ["Coffee machine is broken"])
    )
    pages, _ = extract_pages_from_pdf(data)
    records = chunk_pages(pages)
    store = VectorStore()
    store.add(
        [record["text"] for record in records],
        embed_texts([record["text"] for record in records]),
        pages=[record["page"] for record in records],
    )

    captured = {}

    def fake_answer(question, contexts):
        captured["contexts"] = contexts
        return "The budget is 5000 credits (page 1)."

    monkeypatch.setattr(rag, "generate_answer", fake_answer)
    result = rag.answer_question("What is the budget?", store, filename="report.pdf")

    assert result["found"] is True
    assert result["sources"][0]["filename"] == "report.pdf"
    assert result["sources"][0]["page"] == 1
    assert "The budget is 5000 credits" in result["sources"][0]["text"]
    assert any(context.startswith("[page 1]") for context in captured["contexts"])


# --- HTTP API --------------------------------------------------------------


def test_search_results_include_filename_and_page(auth_client, make_multipage_pdf):
    data = make_multipage_pdf(
        _two_page(["The budget is 5000 credits"], ["Coffee machine is broken"])
    )
    document_id = _upload(auth_client, data).json()["document_id"]

    response = auth_client.post(f"/documents/{document_id}/search?query=budget")

    assert response.status_code == 200
    matches = response.json()["matches"]
    assert matches
    # Every hit carries the source filename.
    assert all(match["filename"] == "report.pdf" for match in matches)

    # Assert page attribution by content, not ranking: the fake embedding model
    # is hash-seeded, so either page can rank first. Each sentence must still be
    # tagged with the page it actually came from.
    budget_hits = [match for match in matches if "budget" in match["text"]]
    assert budget_hits
    assert budget_hits[0]["page"] == 1

    coffee_hits = [match for match in matches if "Coffee" in match["text"]]
    assert coffee_hits
    assert coffee_hits[0]["page"] == 2


# --- persistence -----------------------------------------------------------


def test_page_metadata_survives_restart(auth_client, make_multipage_pdf):
    data = make_multipage_pdf(
        _two_page(["The budget is 5000 credits"], ["Coffee machine is broken"])
    )
    document_id = _upload(auth_client, data).json()["document_id"]

    # Simulate a restart: drop the in-memory cache and reload from storage.
    main.DOCUMENTS.clear()
    main.load_persisted_documents(store=main.get_document_store())

    response = auth_client.post(f"/documents/{document_id}/search?query=budget")
    assert response.status_code == 200
    matches = response.json()["matches"]
    assert matches

    # The restored page metadata must still map the budget sentence to page 1,
    # regardless of where the hash-seeded fake embedding ranks it.
    budget_hits = [match for match in matches if "budget" in match["text"]]
    assert budget_hits
    assert budget_hits[0]["filename"] == "report.pdf"
    assert budget_hits[0]["page"] == 1


def test_legacy_chunk_without_page_is_unavailable(tmp_path):
    store = DocumentStore(tmp_path / "data")
    store.save_document(
        document_id="legacy",
        filename="old.pdf",
        stored_bytes=b"%PDF-1.4",
        pages=1,
        characters=5,
        chunks=["old text"],
        embeddings=np.ones((1, 3), dtype="float32"),
        embedding_model="m",
        created_at="2026-01-01T00:00:00+00:00",
        owner_id="u1",
    )

    chunks, _embeddings, pages = store.load_chunks_with_pages("legacy")
    assert chunks == ["old text"]
    assert pages == [None]


def test_legacy_database_gains_page_column(tmp_path):
    """A chunks table created before page tracking must be migrated in place."""
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    db_path = legacy_dir / "documents.sqlite3"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE chunks (document_id TEXT NOT NULL, ordinal INTEGER NOT NULL, "
            "text TEXT NOT NULL, embedding BLOB NOT NULL, "
            "PRIMARY KEY (document_id, ordinal))"
        )
        conn.execute(
            "INSERT INTO chunks VALUES ('d', 0, 'old', ?)",
            (np.zeros(3, dtype="float32").tobytes(),),
        )

    store = DocumentStore(legacy_dir)
    chunks, _embeddings, pages = store.load_chunks_with_pages("d")
    assert chunks == ["old"]
    assert pages == [None]


# --- ownership unchanged ---------------------------------------------------


def test_other_user_cannot_search_cited_document(
    auth_client, make_auth_client, make_multipage_pdf
):
    data = make_multipage_pdf([["The budget is 5000 credits"]])
    document_id = _upload(auth_client, data).json()["document_id"]
    bob = make_auth_client("bob-cite@example.com")

    assert bob.post(f"/documents/{document_id}/search?query=budget").status_code == 404
    assert (
        bob.post(
            f"/documents/{document_id}/ask", json={"question": "budget?"}
        ).status_code
        == 404
    )
