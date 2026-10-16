"""Text embeddings using a small local Sentence-Transformers model.

An embedding is a fixed-length list of numbers representing the *meaning* of a
piece of text. Texts that mean similar things end up close together in this
numeric space, which is what makes semantic search possible.

We use "all-MiniLM-L6-v2": ~384 numbers per text, small and CPU-friendly.
Embeddings are L2-normalised so that a dot product equals cosine similarity.
"""

import functools

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

MODEL_NAME = settings.embedding_model


@functools.lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load and cache the embedding model (first call downloads it once)."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings into a float32 matrix (n_texts, dim)."""
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    model = get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")
