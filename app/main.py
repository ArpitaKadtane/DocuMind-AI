import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.chunking import chunk_text
from app.config import settings
from app.embeddings import embed_texts
from app.llm import LLMError
from app.pdf_processor import (
    MAX_UPLOAD_MB,
    PDFValidationError,
    extract_text_from_pdf,
    _max_bytes,
)
from app.rag import answer_question
from app.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Keep third-party libraries quiet: their INFO lines are noisy and would
# drown out our own metadata-only logs.
for _noisy in (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "filelock",
    "urllib3",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

app = FastAPI(title="GenAI Document Assistant")

logger = logging.getLogger("document_assistant")

ALLOWED_CONTENT_TYPES = {"application/pdf"}
MAX_UPLOAD_BYTES = _max_bytes()

STATIC_DIR = Path(__file__).parent / "static"

# In-memory store: document_id -> metadata, chunks and vector index.
# Deliberately kept in memory only and never written to disk or logged.
DOCUMENTS: dict[str, dict] = {}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Reject requests that declare a body larger than our limit.

    This is a cheap early guard based on the Content-Length header. The real
    size is still checked while reading the file, so a lying header cannot
    bypass the limit.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES + 1024:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body too large (max {MAX_UPLOAD_MB:g} MB)."},
                )
        except ValueError:
            return JSONResponse(
                status_code=400, content={"detail": "Invalid Content-Length header."}
            )
    return await call_next(request)


@app.get("/")
def home():
    """Serve the simple web frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    """Accept a PDF, validate it, extract text, and register it for Q&A."""
    content_length = file.size
    if content_length is not None and content_length > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_MB:g} MB limit.",
        )

    if (file.content_type or "") not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are accepted (Content-Type must be application/pdf).",
        )

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_MB:g} MB limit.",
        )

    try:
        text, page_count = extract_text_from_pdf(data)
    except PDFValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document_id = uuid.uuid4().hex
    filename = os.path.basename(file.filename or "upload.pdf")
    chunks = chunk_text(text)

    store = VectorStore()
    if chunks:
        store.add(chunks, embed_texts(chunks))

    DOCUMENTS[document_id] = {
        "filename": filename,
        "pages": page_count,
        "text": text,
        "chunks": chunks,
        "store": store,
    }

    # Log metadata only -- never the extracted text.
    logger.info(
        "Stored document id=%s filename=%r pages=%d chars=%d chunks=%d",
        document_id,
        filename,
        page_count,
        len(text),
        len(chunks),
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": page_count,
        "characters": len(text),
        "chunks": len(chunks),
    }


@app.post("/documents/{document_id}/search")
def search_document(document_id: str, query: str, top_k: int | None = None):
    """Debug/retrieval endpoint: return the most relevant chunks for a query."""
    doc = DOCUMENTS.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    if top_k is None:
        top_k = settings.top_k
    top_k = max(1, min(top_k, 10))
    query_vector = embed_texts([query])[0]
    matches = doc["store"].search(query_vector, top_k=top_k)
    return {"document_id": document_id, "query": query, "matches": matches}


@app.post("/documents/{document_id}/ask")
def ask_document(document_id: str, payload: AskRequest):
    """Answer a question about an uploaded document using RAG."""
    doc = DOCUMENTS.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = answer_question(question, doc["store"])
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Log only metadata -- never the question, answer, or document text.
    logger.info(
        "asked document=%s found=%s", document_id, result["found"]
    )
    return {
        "document_id": document_id,
        "question": question,
        **result,
    }


@app.get("/documents/{document_id}")
def get_document_meta(document_id: str):
    doc = DOCUMENTS.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "document_id": document_id,
        "filename": doc["filename"],
        "pages": doc["pages"],
        "characters": len(doc["text"]),
        "chunks": len(doc["chunks"]),
    }
