"""User registration, login, logout, and bearer-token authentication.

Standard library only: passwords are hashed with scrypt and sessions are opaque
random tokens stored (hashed) in SQLite, so nothing sensitive is kept in
plaintext and there is nothing new to install.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.config import settings
from app.security import generate_token, hash_password, hash_token, verify_password

logger = logging.getLogger("document_assistant.auth")

router = APIRouter(prefix="/auth", tags=["auth"])

# Deliberately simple: full RFC 5322 validation needs the optional
# "email-validator" package, which we are avoiding. This catches obvious typos
# and keeps dependencies at zero.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# auto_error=False so a missing header becomes a 401 (with WWW-Authenticate),
# not FastAPI's default 403 -- and so callers never learn why they were denied.
_bearer = HTTPBearer(auto_error=False)


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=256)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=256)


class UserOut(BaseModel):
    user_id: str
    email: str
    created_at: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: str
    user: UserOut


def _store():
    """Return the process-wide document store.

    Imported lazily to avoid a circular import with ``app.main`` and to pick up
    the store that tests replace through monkeypatching.
    """
    from app import main

    return main.get_document_store()


def _unauthorized(detail: str = "Not authenticated.") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_user_out(user: dict) -> UserOut:
    return UserOut(
        user_id=user["user_id"], email=user["email"], created_at=user["created_at"]
    )


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=UserOut)
def register(payload: RegisterRequest) -> UserOut:
    email = payload.email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")

    store = _store()
    if store.get_user_by_email(email) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        )

    user_id = uuid.uuid4().hex
    created_at = _utcnow().isoformat()
    store.create_user(
        user_id=user_id,
        email=email,
        password_hash=hash_password(payload.password),
        created_at=created_at,
    )
    logger.info("Registered user id=%s", user_id)
    return UserOut(user_id=user_id, email=email, created_at=created_at)


@router.post("/login", response_model=TokenOut)
def login(payload: LoginRequest) -> TokenOut:
    email = payload.email.strip().lower()
    store = _store()
    user = store.get_user_by_email(email)

    # Same generic message for "no such user" and "wrong password" so callers
    # cannot enumerate which emails are registered.
    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise _unauthorized("Incorrect email or password.")

    token = generate_token()
    created = _utcnow()
    expires = created + timedelta(hours=settings.session_ttl_hours)
    store.create_session(
        token_hash=hash_token(token),
        user_id=user["user_id"],
        created_at=created.isoformat(),
        expires_at=expires.isoformat(),
    )
    logger.info("Issued session for user id=%s", user["user_id"])
    return TokenOut(access_token=token, expires_at=expires.isoformat(), user=_as_user_out(user))


def get_current_session(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """Resolve the bearer token to ``{"token_hash": ..., "user": ...}``.

    Missing, malformed, unknown, or expired tokens all raise the same 401.
    """
    if credentials is None or not credentials.credentials:
        raise _unauthorized()

    token_hash = hash_token(credentials.credentials)
    store = _store()
    session = store.get_session(token_hash)
    if session is None:
        raise _unauthorized("Invalid or expired token.")

    try:
        expires_at = datetime.fromisoformat(session["expires_at"])
    except ValueError:
        expires_at = None
    if expires_at is None or expires_at <= _utcnow():
        store.delete_session(token_hash)
        raise _unauthorized("Invalid or expired token.")

    user = store.get_user_by_id(session["user_id"])
    if user is None:
        store.delete_session(token_hash)
        raise _unauthorized("Invalid or expired token.")

    return {"token_hash": token_hash, "user": user}


def get_current_user(session: dict = Depends(get_current_session)) -> dict:
    """Dependency that yields the authenticated user row."""
    return session["user"]


@router.post("/logout")
def logout(session: dict = Depends(get_current_session)) -> dict:
    _store().delete_session(session["token_hash"])
    return {"status": "ok"}


@router.get("/me", response_model=UserOut)
def me(user: dict = Depends(get_current_user)) -> UserOut:
    return _as_user_out(user)
