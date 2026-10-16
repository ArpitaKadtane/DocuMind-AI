"""Shared pytest fixtures.

Tests must run fast and offline, so we replace the real embedding model and
the real LLM with fakes. This keeps the suite deterministic and avoids
downloading models or spending API credits.
"""

import re

import numpy as np
import pytest

from app import embeddings, main
from app.storage import DocumentStore


class FakeEmbeddingModel:
    """Deterministic bag-of-words "embedding" for offline tests.

    Words are hashed into a fixed number of dimensions. Texts that share words
    end up with a higher cosine similarity, which is enough to exercise the
    retrieval plumbing without loading a real model.
    """

    dim = 64

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype="float32")
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[hash(word) % self.dim] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.asarray(vectors, dtype="float32")


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """Replace the real sentence-transformers model everywhere it is used."""
    monkeypatch.setattr(embeddings, "get_model", lambda: FakeEmbeddingModel())


@pytest.fixture(autouse=True)
def clean_documents():
    """Ensure the in-memory document store is empty before each test."""
    main.DOCUMENTS.clear()
    yield
    main.DOCUMENTS.clear()


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Point persistence at a throwaway directory so tests never touch real data."""
    monkeypatch.setattr(main, "DOCUMENT_STORE", DocumentStore(tmp_path / "data"))
    yield


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    return TestClient(main.app)


@pytest.fixture
def make_auth_client():
    """Factory returning a TestClient already authenticated as a new user.

    Each call registers and logs in a distinct email and sets the bearer token
    on the client, so tests can act as several independent users.
    """
    from fastapi.testclient import TestClient

    def _make(email: str, password: str = "password123", register: bool = True):
        test_client = TestClient(main.app)
        if register:
            created = test_client.post(
                "/auth/register", json={"email": email, "password": password}
            )
            assert created.status_code == 201, created.text
        logged_in = test_client.post(
            "/auth/login", json={"email": email, "password": password}
        )
        assert logged_in.status_code == 200, logged_in.text
        token = logged_in.json()["access_token"]
        test_client.headers.update({"Authorization": f"Bearer {token}"})
        return test_client

    return _make


@pytest.fixture
def auth_client(make_auth_client):
    """A single authenticated client for tests that do not need a second user."""
    return make_auth_client("tester@example.com")


@pytest.fixture
def make_pdf():
    """Return a helper that builds an in-memory PDF from a list of lines."""
    import pymupdf

    def _make(lines: list[str]) -> bytes:
        doc = pymupdf.open()
        page = doc.new_page()
        for i, line in enumerate(lines):
            page.insert_text((72, 72 + i * 20), line)
        return doc.tobytes()

    return _make


@pytest.fixture
def make_multipage_pdf():
    """Return a helper that builds a PDF from a list of pages (each a line list)."""
    import pymupdf

    def _make(pages: list[list[str]]) -> bytes:
        doc = pymupdf.open()
        for lines in pages:
            page = doc.new_page()
            for i, line in enumerate(lines):
                page.insert_text((72, 72 + i * 20), line)
        return doc.tobytes()

    return _make
