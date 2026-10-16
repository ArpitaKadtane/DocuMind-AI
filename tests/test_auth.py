"""Tests for registration, login, logout, and token validation."""

from app import main


def _register(client, email="alice@example.com", password="password123"):
    return client.post("/auth/register", json={"email": email, "password": password})


def _login(client, email="alice@example.com", password="password123"):
    return client.post("/auth/login", json={"email": email, "password": password})


def test_register_creates_user(client):
    response = _register(client)
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "alice@example.com"
    assert "user_id" in body
    assert "password" not in body
    assert "password_hash" not in body


def test_register_duplicate_email_conflicts(client):
    assert _register(client).status_code == 201
    assert _register(client).status_code == 409


def test_register_rejects_invalid_email(client):
    response = _register(client, email="not-an-email")
    assert response.status_code == 422


def test_register_rejects_short_password(client):
    response = _register(client, password="short")
    assert response.status_code == 422


def test_password_is_hashed_not_stored_plaintext(client):
    _register(client)
    user = main.get_document_store().get_user_by_email("alice@example.com")
    assert user["password_hash"] != "password123"
    assert user["password_hash"].startswith("scrypt$")


def test_login_returns_bearer_token(client):
    _register(client)
    response = _login(client)
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["user"]["email"] == "alice@example.com"


def test_login_email_is_case_insensitive(client):
    _register(client, email="Alice@Example.com")
    assert _login(client, email="alice@example.com").status_code == 200


def test_login_wrong_password_is_401(client):
    _register(client)
    assert _login(client, password="wrong-password").status_code == 401


def test_login_unknown_email_is_401(client):
    assert _login(client, email="nobody@example.com").status_code == 401


def test_me_requires_authentication(client):
    assert client.get("/auth/me").status_code == 401


def test_me_returns_current_user(auth_client):
    response = auth_client.get("/auth/me")
    assert response.status_code == 200
    assert response.json()["email"] == "tester@example.com"


def test_invalid_token_is_rejected(client):
    response = client.get("/auth/me", headers={"Authorization": "Bearer nope"})
    assert response.status_code == 401


def test_malformed_authorization_header_is_rejected(client):
    response = client.get("/auth/me", headers={"Authorization": "Token abc"})
    assert response.status_code == 401


def test_logout_revokes_the_token(make_auth_client):
    client = make_auth_client("carol@example.com")
    assert client.post("/auth/logout").status_code == 200
    # The same token must no longer work.
    assert client.get("/auth/me").status_code == 401
