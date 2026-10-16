"""Persistent storage for documents, users, and sessions.

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
 * Every document records an ``owner_id``. Documents uploaded before accounts
   existed have a NULL owner and are deliberately left inaccessible until an
   operator explicitly claims them (see ``claim_legacy_documents``).
 * Sessions store only a SHA-256 hash of the bearer token, never the token
   itself, so a database leak does not hand over live sessions.
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
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS documents (
    document_id     TEXT PRIMARY KEY,
    owner_id        TEXT,
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
    page        INTEGER,
    embedding   BLOB NOT NULL,
    PRIMARY KEY (document_id, ordinal),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
        ON DELETE CASCADE
);
"""


class DocumentStore:
    """SQLite + filesystem persistence for uploaded documents and users."""

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
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Bring older databases up to date without destroying any data."""
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
        if "owner_id" not in columns:
            # SQLite has no "ADD COLUMN IF NOT EXISTS"; the guard above makes
            # this safe to run on every startup. Existing rows get a NULL owner.
            conn.execute("ALTER TABLE documents ADD COLUMN owner_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id)"
        )

        # Page tracking was added later. Older databases have chunks without a
        # page column; add it so page numbers can be stored from now on. Legacy
        # rows keep a NULL page ("page unavailable") -- nothing is deleted.
        chunk_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(chunks)")
        }
        if "page" not in chunk_columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN page INTEGER")

    # --- documents -------------------------------------------------------

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
        owner_id: str | None = None,
        chunk_pages: list[int | None] | None = None,
    ) -> None:
        """Persist a document's file, metadata, chunks, and embeddings.

        ``chunk_pages`` is an optional list parallel to ``chunks`` holding each
        chunk's one-based source page (or ``None`` when unknown).
        """
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
                    (document_id, owner_id, filename, stored_filename, size_bytes,
                     pages, characters, chunk_count, embedding_model, embedding_dim,
                     created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    owner_id,
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
                if chunk_pages is None:
                    chunk_pages = [None] * len(chunks)
                elif len(chunk_pages) != len(chunks):
                    raise ValueError(
                        "Number of pages does not match number of chunks."
                    )
                rows = [
                    (
                        document_id,
                        ordinal,
                        text,
                        page,
                        np.asarray(vector, dtype="float32").tobytes(),
                    )
                    for ordinal, (text, vector, page) in enumerate(
                        zip(chunks, embeddings, chunk_pages)
                    )
                ]
                conn.executemany(
                    "INSERT INTO chunks (document_id, ordinal, text, page, embedding) "
                    "VALUES (?, ?, ?, ?, ?)",
                    rows,
                )

    def iter_documents(self, owner_id: str | None = None) -> list[dict]:
        """Return metadata for stored documents, oldest first.

        With ``owner_id`` given, only that owner's documents are returned.
        Without it, every document is returned (including unowned legacy rows).
        """
        sql = "SELECT * FROM documents"
        params: tuple = ()
        if owner_id is not None:
            sql += " WHERE owner_id = ?"
            params = (owner_id,)
        sql += " ORDER BY created_at"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, document_id: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return dict(row) if row else None

    def claim_legacy_documents(self, user_id: str) -> int:
        """Assign every unowned document to ``user_id``. Returns rows changed.

        This is an explicit operator action -- the application never calls it
        automatically, so legacy documents are never silently handed to a user.
        """
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE documents SET owner_id = ? WHERE owner_id IS NULL",
                (user_id,),
            )
        return int(cursor.rowcount)

    def load_chunks_with_pages(
        self, document_id: str
    ) -> tuple[list[str], np.ndarray, list[int | None]]:
        """Return ``(chunks, embeddings, pages)`` for a document, in stored order.

        ``pages`` runs parallel to ``chunks``; a ``None`` entry means the page
        is unknown (for example a document stored before page tracking existed).
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT text, page, embedding FROM chunks "
                "WHERE document_id = ? ORDER BY ordinal",
                (document_id,),
            ).fetchall()

        chunks = [row["text"] for row in rows]
        pages = [row["page"] for row in rows]
        if rows:
            vectors = [np.frombuffer(row["embedding"], dtype="float32") for row in rows]
            embeddings = np.vstack(vectors).astype("float32")
        else:
            embeddings = np.zeros((0, 0), dtype="float32")
        return chunks, embeddings, pages

    def load_chunks(self, document_id: str) -> tuple[list[str], np.ndarray]:
        """Return ``(chunks, embeddings)`` for a document, in stored order."""
        chunks, embeddings, _ = self.load_chunks_with_pages(document_id)
        return chunks, embeddings

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    # --- users -----------------------------------------------------------

    def create_user(
        self, *, user_id: str, email: str, password_hash: str, created_at: str
    ) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO users (user_id, email, password_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, email, password_hash, created_at),
            )

    def get_user_by_email(self, email: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def count_users(self) -> int:
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    # --- sessions --------------------------------------------------------

    def create_session(
        self, *, token_hash: str, user_id: str, created_at: str, expires_at: str
    ) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token_hash, user_id, created_at, expires_at),
            )

    def get_session(self, token_hash: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE token_hash = ?", (token_hash,)
            ).fetchone()
        return dict(row) if row else None

    def delete_session(self, token_hash: str) -> bool:
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE token_hash = ?", (token_hash,)
            )
        return cursor.rowcount > 0
