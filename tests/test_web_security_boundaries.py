from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from smart_agent.auth import AuthService


def test_legacy_file_apis_require_auth(tmp_path, monkeypatch):
    pytest = __import__("pytest")
    testclient_module = pytest.importorskip("fastapi.testclient")
    from smart_agent.web import app as app_module

    service = AuthService(
        create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )
    )
    monkeypatch.setattr(app_module, "_AUTH_SERVICE", service)
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    client = testclient_module.TestClient(app_module.app)

    for path in ("/api/files", "/api/file?path=a.txt", "/api/report"):
        assert client.get(path).status_code == 401
    assert client.post("/api/save", json={"path": "a.txt", "content": "x"}).status_code == 401


def test_project_file_api_uses_registered_project_and_optimistic_lock(tmp_path, monkeypatch):
    pytest = __import__("pytest")
    testclient_module = pytest.importorskip("fastapi.testclient")
    from smart_agent.web import app as app_module

    service = AuthService(
        create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )
    )
    monkeypatch.setattr(app_module, "_AUTH_SERVICE", service)
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    client = testclient_module.TestClient(app_module.app)
    registration = client.post(
        "/api/auth/register",
        json={"account": "boundary01", "password": "password123"},
    ).json()
    token = registration["token"]
    tenant_id = registration["user"]["tenant_id"]
    project = tmp_path / ".runtime" / "career" / "tenants" / tenant_id / "uploaded-project"
    project.mkdir(parents=True)
    (project / "main.py").write_text("print('ok')\n", encoding="utf-8")
    headers = {"Authorization": "Bearer " + token}
    registered = client.post(
        "/api/career/project-library/register",
        headers=headers,
        json={"project_id": "demo", "source_path": str(project.relative_to(tmp_path))},
    )
    assert registered.status_code == 200

    listed = client.get("/api/files", params={"project_id": "demo"}, headers=headers)
    assert listed.status_code == 200
    assert any(item["name"] == "main.py" for item in listed.json()["items"])

    current = client.get("/api/file", params={"project_id": "demo", "path": "main.py"}, headers=headers).json()
    saved = client.post(
        "/api/save",
        headers=headers,
        json={
            "project_id": "demo",
            "path": "main.py",
            "content": "print('changed')\n",
            "expected_sha256": "0" * 64,
        },
    )
    assert saved.status_code == 409
    assert current["content"] == "print('ok')\n"
