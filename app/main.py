import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.auth import get_current_user
from app.auth import router as auth_router
from app.chunking import chunk_pages
from app.config import settings
from app.embeddings import embed_texts
from app.llm import LLMError
from app.pdf_processor import (
    MAX_UPLOAD_MB,
    PDFValidationError,
    extract_pages_from_pdf,
    _max_bytes,
)
from app.rag import answer_question
from app.storage import DocumentStore
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

logger = logging.getLogger("document_assistant")

ALLOWED_CONTENT_TYPES = {"application/pdf"}
MAX_UPLOAD_BYTES = _max_bytes()

STATIC_DIR = Path(__file__).parent / "static"

# In-memory cache of loaded documents: document_id -> metadata, chunks, index.
# This mirrors what is persisted on disk and is rebuilt on every startup.
DOCUMENTS: dict[str, dict] = {}

# SQLite + file persistence. Initialised lazily so tests can substitute a
# temporary directory, and used to restore documents after a restart.
DOCUMENT_STORE: DocumentStore | None = None


def get_document_store() -> DocumentStore:
    """Return the process-wide document store, creating it on first use."""
    global DOCUMENT_STORE
    if DOCUMENT_STORE is None:
        DOCUMENT_STORE = DocumentStore(settings.data_dir)
    return DOCUMENT_STORE


def load_persisted_documents(store: DocumentStore | None = None) -> int:
    """Rebuild the in-memory document cache from persistent storage.

    Returns the number of documents restored. A single missing or corrupt
    document is logged and skipped rather than crashing the whole application.
    """
    store = store or get_document_store()
    try:
        metadata = store.iter_documents()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Could not read persisted documents: %s", exc)
        return 0

    restored = 0
    for meta in metadata:
        document_id = meta["document_id"]
        try:
            if not store.stored_file_path(document_id).exists():
                logger.warning(
                    "Stored file missing for document id=%s; restoring search "
                    "data from the database only",
                    document_id,
                )
            chunks, embeddings, chunk_pages = store.load_chunks_with_pages(document_id)
            vector_store = VectorStore()
            if chunks:
                vector_store.add(chunks, embeddings, pages=chunk_pages)
            DOCUMENTS[document_id] = {
                "filename": meta["filename"],
                "owner_id": meta.get("owner_id"),
                "pages": meta["pages"],
                "characters": meta["characters"],
                "chunks": chunks,
                "chunk_pages": chunk_pages,
                "store": vector_store,
                "created_at": meta["created_at"],
            }
            restored += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Skipping unreadable document id=%s: %s", document_id, exc)
    return restored


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Restore persisted documents when the server starts."""
    restored = load_persisted_documents()
    logger.info("Restored %d persisted document(s) from storage.", restored)
    yield


app = FastAPI(title="GenAI Document Assistant", lifespan=lifespan)
app.include_router(auth_router)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


def get_owned_document(
    document_id: str,
    user: dict = Depends(get_current_user),
) -> dict:
    """Return a document only if it belongs to the authenticated user.

    A document that does not exist and one owned by someone else raise the same
    404 with the same message, so a caller cannot tell whether an id exists.
    Unowned legacy documents (owner_id is None) are inaccessible this way.
    """
    doc = DOCUMENTS.get(document_id)
    if doc is None or doc.get("owner_id") != user["user_id"]:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


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


@app.get("/documents")
def list_documents(user: dict = Depends(get_current_user)):
    """List metadata for the current user's documents (never file paths/text)."""
    owned = [
        {
            "document_id": document_id,
            "filename": doc["filename"],
            "pages": doc["pages"],
            "characters": doc["characters"],
            "chunks": len(doc["chunks"]),
            "created_at": doc["created_at"],
        }
        for document_id, doc in DOCUMENTS.items()
        if doc.get("owner_id") == user["user_id"]
    ]
    owned.sort(key=lambda item: item["created_at"])
    return {"documents": owned}


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
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
        pages, page_count = extract_pages_from_pdf(data)
    except PDFValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document_id = uuid.uuid4().hex
    filename = os.path.basename(file.filename or "upload.pdf")
    text = "\n\n".join(page_text for _, page_text in pages).strip()
    chunk_records = chunk_pages(pages)
    chunks = [record["text"] for record in chunk_records]
    chunk_page_numbers = [record["page"] for record in chunk_records]
    embeddings = embed_texts(chunks) if chunks else None
    created_at = datetime.now(timezone.utc).isoformat()

    vector_store = VectorStore()
    if chunks and embeddings is not None:
        vector_store.add(chunks, embeddings, pages=chunk_page_numbers)

    DOCUMENTS[document_id] = {
        "filename": filename,
        "owner_id": user["user_id"],
        "pages": page_count,
        "characters": len(text),
        "chunks": chunks,
        "chunk_pages": chunk_page_numbers,
        "store": vector_store,
        "created_at": created_at,
    }

    # Persist to disk so the document survives a restart. A storage failure is
    # logged but must not fail the request: the document still works in memory.
    try:
        get_document_store().save_document(
            document_id=document_id,
            filename=filename,
            stored_bytes=data,
            pages=page_count,
            characters=len(text),
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=settings.embedding_model,
            created_at=created_at,
            owner_id=user["user_id"],
            chunk_pages=chunk_page_numbers,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Could not persist document id=%s: %s", document_id, exc)

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
def search_document(
    document_id: str,
    query: str,
    top_k: int | None = None,
    doc: dict = Depends(get_owned_document),
):
    """Debug/retrieval endpoint: return the most relevant chunks for a query."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    if top_k is None:
        top_k = settings.top_k
    top_k = max(1, min(top_k, 10))
    query_vector = embed_texts([query])[0]
    matches = doc["store"].search(query_vector, top_k=top_k)
    # Enrich each hit with the source filename and page so callers can cite it.
    sources = [
        {
            "filename": doc["filename"],
            "page": match.get("page"),
            "text": match["text"],
            "score": match["score"],
        }
        for match in matches
    ]
    return {"document_id": document_id, "query": query, "matches": sources}


@app.post("/documents/{document_id}/ask")
def ask_document(
    document_id: str,
    payload: AskRequest,
    doc: dict = Depends(get_owned_document),
):
    """Answer a question about an uploaded document using RAG."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = answer_question(question, doc["store"], filename=doc["filename"])
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
def get_document_meta(document_id: str, doc: dict = Depends(get_owned_document)):
    return {
        "document_id": document_id,
        "filename": doc["filename"],
        "pages": doc["pages"],
        "characters": doc["characters"],
        "chunks": len(doc["chunks"]),
    }
