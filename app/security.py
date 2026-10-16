"""Password hashing and bearer-token helpers.

Standard library only, so nothing new has to be installed and the code keeps
working on new Python versions without waiting for a third-party wheel.

 * Passwords are hashed with scrypt (memory-hard) and a fresh random salt per
   user. The stored string is ``scrypt$n$r$p$salt_b64$hash_b64``; verification
   re-derives the key and compares it in constant time.
 * Tokens are opaque, high-entropy random strings. Only a SHA-256 hash of a
   token is ever persisted, so a database leak cannot be replayed as a session.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

# scrypt work factors. ~16 MiB of memory per hash -- comfortably strong for a
# login endpoint while staying fast enough for interactive use.
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_DKLEN = 32
_SALT_BYTES = 16
_ALGORITHM = "scrypt"

_TOKEN_BYTES = 32


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def hash_password(password: str) -> str:
    """Return a salted scrypt hash for ``password``."""
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_DKLEN,
    )
    return "$".join(
        [_ALGORITHM, str(_SCRYPT_N), str(_SCRYPT_R), str(_SCRYPT_P), _b64(salt), _b64(derived)]
    )


def verify_password(password: str, encoded: str) -> bool:
    """Return True if ``password`` matches the stored hash. Never raises."""
    try:
        algorithm, n, r, p, salt_b64, hash_b64 = encoded.split("$")
        if algorithm != _ALGORITHM:
            return False
        salt = _unb64(salt_b64)
        expected = _unb64(hash_b64)
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(derived, expected)


def generate_token() -> str:
    """Return a new opaque bearer token (URL-safe, ~256 bits of entropy)."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def hash_token(token: str) -> str:
    """Return the hex SHA-256 of ``token`` -- the only form we persist."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
