"""A thin wrapper around a FAISS index for similarity search.

FAISS ("Facebook AI Similarity Search") stores vectors and quickly finds the
ones closest to a query vector. Because our embeddings are normalised, using
the inner-product index (IndexFlatIP) gives cosine similarity scores in
roughly the range -1..1 (higher = more similar).
"""

import faiss
import numpy as np


class VectorStore:
    """Holds an index plus the chunk texts their vectors were built from."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._chunks: list[str] = []

    def add(self, chunks: list[str], embeddings: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings does not match number of chunks.")

        dim = embeddings.shape[1]
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[dict]:
        """Return the top-k most similar chunks with their scores."""
        if self._index is None or self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"text": self._chunks[idx], "score": float(score)})
        return results

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index is not None else 0
