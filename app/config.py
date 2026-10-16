"""Central configuration loaded from environment variables / .env file.

Keeping every tunable in one place makes the project easy to configure and
keeps secrets (like API keys) out of the source code. Copy .env.example to
.env and fill in your values.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root (the folder that contains the "app" package). Runtime data such
# as uploaded PDFs and the SQLite database lives here by default.
BASE_DIR = Path(__file__).resolve().parent.parent


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # LLM (Groq, OpenAI-compatible API)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_base_url: str = os.getenv(
        "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
    )
    request_timeout: float = _get_float("REQUEST_TIMEOUT", 30.0)

    # Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "multi-qa-MiniLM-L6-cos-v1")

    # Chunking
    chunk_size_chars: int = _get_int("CHUNK_SIZE_CHARS", 800)
    chunk_overlap_chars: int = _get_int("CHUNK_OVERLAP_CHARS", 150)

    # Upload / PDF limits
    max_upload_mb: float = _get_float("MAX_UPLOAD_MB", 10.0)
    max_pdf_pages: int = _get_int("MAX_PDF_PAGES", 100)
    max_pdf_chars: int = _get_int("MAX_PDF_CHARS", 2_000_000)

    # Retrieval
    top_k: int = _get_int("TOP_K", 4)
    min_score: float = _get_float("MIN_SCORE", 0.15)

    # Persistence
    data_dir: str = os.getenv("DATA_DIR", str(BASE_DIR / "data"))


settings = Settings()
