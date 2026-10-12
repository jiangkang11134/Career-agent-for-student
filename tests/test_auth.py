from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from smart_agent.auth import AuthError, AuthService


def make_service() -> AuthService:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    return AuthService(engine, session_days=1)


def test_register_login_session_and_revoke() -> None:
    service = make_service()

    user, token = service.register("MaryAccount1", "password123", "Mary")

    assert user.account == "maryaccount1"
    assert user.email == "maryaccount1"
    assert user.tenant_id == user.user_id
    assert service.user_from_token(token) == user

    logged_in, login_token = service.authenticate("maryaccount1", "password123")
    assert logged_in == user
    assert service.user_from_token(login_token) == user

    service.revoke(login_token)
    assert service.user_from_token(login_token) is None


def test_duplicate_account_and_wrong_password_fail() -> None:
    service = make_service()
    service.register("maryaccount1", "password123", "Mary")

    with pytest.raises(AuthError):
        service.register("MARYACCOUNT1", "password123", "Other")

    with pytest.raises(AuthError):
        service.authenticate("maryaccount1", "wrong-password")


def test_rejects_invalid_account_and_short_password() -> None:
    service = make_service()

    with pytest.raises(AuthError):
        service.register("short", "password123", "")

    with pytest.raises(AuthError):
        service.register("maryaccount1", "short", "")


def test_career_api_requires_auth_and_uses_tenant_root(tmp_path, monkeypatch) -> None:
    testclient_module = pytest.importorskip("fastapi.testclient")
    from smart_agent.web import app as app_module

    service = make_service()
    monkeypatch.setattr(app_module, "_AUTH_SERVICE", service)
    monkeypatch.setattr(app_module, "ROOT", tmp_path)

    client = testclient_module.TestClient(app_module.app)
    assert client.get("/api/career/facts").status_code == 401

    registered = client.post(
        "/api/auth/register",
        json={"account": "tenantuser01", "password": "password123", "display_name": "Tenant"},
    )
    assert registered.status_code == 200
    payload = registered.json()
    token = payload["token"]
    tenant_id = payload["user"]["tenant_id"]

    response = client.get("/api/career/facts", headers={"Authorization": "Bearer " + token})
    assert response.status_code == 200
    assert response.json()["facts"] == []
    assert (tmp_path / ".runtime" / "career" / "tenants" / tenant_id).is_dir()
