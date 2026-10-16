"""Persistent storage for uploaded documents, chunks, and embeddings.

The RAG pipeline keeps everything in memory, so documents normally vanish when
the server restarts. This module adds a small SQLite-backed store (Python's
built-in ``sqlite3`` -- no new dependency) plus an on-disk folder for the
original PDFs, so a restart can rebuild the exact same search index.

Design notes:
 * Chunk embeddings are stored as raw ``float32`` bytes. Because the FAISS index
   is a flat inner-product index, it can be rebuilt losslessly from those
   vectors -- no model download or re-embedding is needed on startup.
 * Stored filenames are derived from the server-generated ``document_id`` (a hex
   UUID), never from the user-supplied filename, so a malicious name cannot
   escape the data directory.
 * Filesystem paths are never returned through the API; documents are addressed
   only by ``document_id``.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

logger = logging.getLogger("document_assistant.storage")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id     TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    pages           INTEGER,
    characters      INTEGER NOT NULL,
    chunk_count     INTEGER NOT NULL,
    embedding_model TEXT,
    embedding_dim   INTEGER,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    document_id TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    PRIMARY KEY (document_id, ordinal),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
        ON DELETE CASCADE
);
"""


class DocumentStore:
    """SQLite + filesystem persistence for uploaded documents."""

    def __init__(self, data_dir: str | os.PathLike[str]) -> None:
        self.data_dir = Path(data_dir)
        self.files_dir = self.data_dir / "files"
        self.db_path = self.data_dir / "documents.sqlite3"
        self._ensure_dirs()
        self._init_schema()

    def _ensure_dirs(self) -> None:
        self.files_dir.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def stored_file_path(self, document_id: str) -> Path:
        """Return the on-disk path for a document's stored PDF bytes."""
        return self.files_dir / f"{document_id}.pdf"

    def save_document(
        self,
        *,
        document_id: str,
        filename: str,
        stored_bytes: bytes,
        pages: int | None,
        characters: int,
        chunks: list[str],
        embeddings: np.ndarray | None,
        embedding_model: str,
        created_at: str,
    ) -> None:
        """Persist a document's file, metadata, chunks, and embeddings."""
        target = self.stored_file_path(document_id)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(stored_bytes)
        os.replace(tmp, target)

        dim = (
            int(embeddings.shape[1])
            if embeddings is not None and embeddings.size
            else None
        )

        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                    (document_id, filename, stored_filename, size_bytes, pages,
                     characters, chunk_count, embedding_model, embedding_dim,
                     created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    filename,
                    target.name,
                    len(stored_bytes),
                    pages,
                    characters,
                    len(chunks),
                    embedding_model,
                    dim,
                    created_at,
                ),
            )
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            if chunks and embeddings is not None:
                rows = [
                    (
                        document_id,
                        ordinal,
                        text,
                        np.asarray(vector, dtype="float32").tobytes(),
                    )
                    for ordinal, (text, vector) in enumerate(zip(chunks, embeddings))
                ]
                conn.executemany(
                    "INSERT INTO chunks (document_id, ordinal, text, embedding) "
                    "VALUES (?, ?, ?, ?)",
                    rows,
                )

    def iter_documents(self) -> list[dict]:
        """Return metadata for every stored document, oldest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, document_id: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return dict(row) if row else None

    def load_chunks(self, document_id: str) -> tuple[list[str], np.ndarray]:
        """Return ``(chunks, embeddings)`` for a document, in stored order."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT text, embedding FROM chunks "
                "WHERE document_id = ? ORDER BY ordinal",
                (document_id,),
            ).fetchall()

        chunks = [row["text"] for row in rows]
        if rows:
            vectors = [np.frombuffer(row["embedding"], dtype="float32") for row in rows]
            embeddings = np.vstack(vectors).astype("float32")
        else:
            embeddings = np.zeros((0, 0), dtype="float32")
        return chunks, embeddings

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
