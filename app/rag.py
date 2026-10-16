"""The RAG pipeline: retrieve relevant chunks, then generate a grounded answer.

Flow:
  question -> embed -> vector search -> (optional) gate on score -> LLM answer

The score gate is the key anti-hallucination guard: if nothing in the document
is similar enough to the question, we return "not found" without calling the
LLM at all.
"""

from app.config import settings
from app.embeddings import embed_texts
from app.llm import NOT_FOUND_MESSAGE, generate_answer
from app.vector_store import VectorStore


def answer_question(question: str, store: VectorStore) -> dict:
    if store.size == 0:
        return {"answer": NOT_FOUND_MESSAGE, "found": False, "sources": []}

    query_vector = embed_texts([question])[0]
    matches = store.search(query_vector, top_k=settings.top_k)

    if not matches or matches[0]["score"] < settings.min_score:
        return {"answer": NOT_FOUND_MESSAGE, "found": False, "sources": []}

    contexts = [m["text"] for m in matches]
    answer = generate_answer(question, contexts)

    found = answer.strip() != NOT_FOUND_MESSAGE
    return {"answer": answer, "found": found, "sources": contexts}
