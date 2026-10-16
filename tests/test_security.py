"""Unit tests for the standard-library password/token helpers."""

from app.security import generate_token, hash_password, hash_token, verify_password


def test_password_round_trip():
    encoded = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", encoded) is True


def test_wrong_password_is_rejected():
    encoded = hash_password("correct horse battery staple")
    assert verify_password("Correct horse battery staple", encoded) is False


def test_hashes_are_salted_so_identical_passwords_differ():
    first = hash_password("same-password")
    second = hash_password("same-password")
    assert first != second
    assert verify_password("same-password", first) is True
    assert verify_password("same-password", second) is True


def test_malformed_hash_returns_false_not_error():
    assert verify_password("whatever", "not-a-valid-hash") is False
    assert verify_password("whatever", "bcrypt$1$2$3$4$5") is False


def test_tokens_are_unique_and_high_entropy():
    tokens = {generate_token() for _ in range(25)}
    assert len(tokens) == 25
    assert all(len(token) >= 32 for token in tokens)


def test_hash_token_is_deterministic_and_not_the_token():
    token = generate_token()
    digest = hash_token(token)
    assert digest == hash_token(token)
    assert digest != token
    assert len(digest) == 64  # sha256 hex
