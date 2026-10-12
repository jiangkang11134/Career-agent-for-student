from concurrent.futures import ThreadPoolExecutor

import pytest

from smart_agent.career.workspace_context import ContextConflict, WorkspaceContextStore


def test_atomic_compare_and_swap_replay_and_retry(tmp_path):
    store = WorkspaceContextStore(tmp_path, "alice")
    state = store.create()
    identity = state["session_id"]

    def update(index):
        try:
            return store.commit(identity, 1, "selected", {"stage": "resume"}, str(index))
        except ContextConflict:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(update, range(4)))
    assert sum(value is not None for value in results) == 1
    assert store.replay(identity) == store.get(identity)
    winning = str(next(i for i, value in enumerate(results) if value))
    assert store.commit(identity, 1, "selected", {"stage": "resume"}, winning)["context_revision"] == 2
    with pytest.raises(ContextConflict):
        store.commit(identity, 2, "selected", {"stage": "coach"}, winning)
    with pytest.raises(KeyError):
        WorkspaceContextStore(tmp_path, "bob").get(identity)
    assert WorkspaceContextStore(tmp_path, "alice").get(identity) == store.get(identity)


@pytest.fixture
def web(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    from smart_agent.auth import AuthService
    from smart_agent.web import app as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "_AUTH_SERVICE",
        AuthService(
            create_engine(
                "sqlite+pysqlite:///:memory:",
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        ),
    )
    client = TestClient(module.app)
    user = client.post("/api/auth/register", json={"account": "context001", "password": "password123"}).json()
    assert "token" in user, user
    client.headers["Authorization"] = "Bearer " + user["token"]
    return (
        client,
        module,
        tmp_path / ".runtime" / "career" / "tenants" / user["user"]["tenant_id"],
        user["user"]["user_id"],
    )


def test_context_api_conflict_and_ownership(web):
    client, _, _, _ = web
    state = client.post("/api/career/contexts", json={}).json()
    path = "/api/career/contexts/" + state["session_id"]
    assert (
        client.patch(path, json={"expected_revision": 1, "request_id": "one", "patch": {"stage": "coach"}}).status_code
        == 200
    )
    assert (
        client.patch(path, json={"expected_revision": 1, "request_id": "two", "patch": {"stage": "resume"}}).status_code
        == 409
    )
    assert client.get(path + "/replay").json() == client.get(path).json()
    other = client.post("/api/auth/register", json={"account": "context002", "password": "password123"}).json()
    client.headers["Authorization"] = "Bearer " + other["token"]
    assert client.get(path).status_code == 404


def test_imported_resume_can_start_runtime_without_generated_resume(web, monkeypatch):
    from smart_agent.career import (
        CareerStore,
        InterviewKnowledgeChunk,
        InterviewKnowledgeCorpus,
        InterviewKnowledgeStore,
        InterviewRuntime,
        InterviewSessionStore,
        ResumeDocumentStatus,
    )
    from smart_agent.career.interview_inputs import InterviewInputStore

    client, module, root, owner = web
    result = client.post(
        "/api/career/contexts/interview-inputs",
        json={
            "jd_text": "Python backend API engineer",
            "resume_text": "技能\nPython API SQL Redis",
            "role_title": "Engineer",
        },
    )
    assert result.status_code == 200, result.text
    bundle = result.json()
    path = "/api/career/contexts/" + bundle["session_id"]
    confirmed = client.post(path + "/confirm", json={"expected_revision": 2, "request_id": "confirm"})
    assert confirmed.status_code == 200, confirmed.text
    base = CareerStore(root)
    store = InterviewInputStore(base, owner)
    prefix = "input:" + bundle["session_id"] + ":"
    assert store.get(prefix + "resume").status is ResumeDocumentStatus.USER_PROVIDED
    knowledge = InterviewKnowledgeStore(root / "interview-knowledge")
    knowledge.put(
        InterviewKnowledgeCorpus(
            corpus_id="api",
            name="API",
            domain="backend",
            chunks=(
                InterviewKnowledgeChunk(
                    chunk_id="api-1",
                    corpus_id="api",
                    domain="backend",
                    topic="Python API",
                    text="Use indexes and caching for API latency.",
                    source_id="notes",
                ),
            ),
        )
    )
    runtime = InterviewRuntime(
        career_store=store, knowledge_store=knowledge, session_store=InterviewSessionStore(root / "interviews")
    )
    interview = runtime.create_session(jd_id=prefix + "jd", resume_document_id=prefix + "resume", corpus_ids=("api",))
    assert interview.resume_run_id is None
    assert interview.interview_points
    assert runtime.next_question(interview.interview_id).text
    from smart_agent.career import ResumeDocument

    assert base.list_latest(ResumeDocument) == ()
    workspace = client.post("/api/career/contexts", json={}).json()
    workspace_path = "/api/career/contexts/" + workspace["session_id"]
    selected = client.patch(
        workspace_path,
        json={"expected_revision": 1, "request_id": "select", "patch": {"input_bundle_id": bundle["session_id"]}},
    )
    assert selected.status_code == 200, selected.text
    from smart_agent.career import InterviewKnowledgeLibrary
    from smart_agent.tools.career import _validate_create_interview_session

    monkeypatch.setattr(InterviewKnowledgeLibrary, "metadata_summary", lambda *args, **kwargs: {"corpus_ids": ["api"]})

    def execute(name, payload, **kwargs):
        assert name == "create_interview_session"
        assert payload["jd_id"] == prefix + "jd"
        return runtime.create_session(**_validate_create_interview_session(payload)).to_dict()

    monkeypatch.setattr(module, "_execute_career_tool", execute)
    monkeypatch.setattr(
        module,
        "_execute_next_interview_question_with_trace",
        lambda identity, **kwargs: runtime.next_question(identity).to_dict(),
    )
    started = client.post(
        "/api/career/interviews/start",
        json={
            "workspace_session_id": workspace["session_id"],
            "context_revision": 2,
            "jd_id": "wrong-client-jd",
            "resume_run_id": "wrong-client-run",
        },
    )
    assert started.status_code == 200, started.text
    current = client.get(workspace_path).json()
    assert current["active_interview_id"] == started.json()["session"]["interview_id"]
    assert current["context_revision"] == 3
    assert (
        client.post(
            "/api/career/interviews/start",
            json={
                "workspace_session_id": workspace["session_id"],
                "context_revision": 2,
            },
        ).status_code
        == 409
    )


def test_extract_and_invalid_context_inputs(web):
    client, _, _, _ = web
    result = client.post("/api/career/contexts/extract", files={"file": ("resume.txt", b"Python SQL API")})
    assert result.status_code == 200, result.text
    assert "Python" in result.json()["text"]
    assert client.post("/api/career/contexts/extract", files={"file": ("bad.exe", b"x")}).status_code == 400
    assert (
        client.post(
            "/api/career/contexts/interview-inputs", json={"jd_text": "x", "resume_text": "", "role_title": "test"}
        ).status_code
        == 400
    )
    state = client.post("/api/career/contexts", json={}).json()
    assert (
        client.patch(
            "/api/career/contexts/" + state["session_id"],
            json={
                "patch": {"coach_messages": []},
                "expected_revision": 1,
                "request_id": "x",
            },
        ).status_code
        == 400
    )


def test_coach_turn_persists_with_fallback(web, monkeypatch):
    client, module, _, _ = web
    monkeypatch.setattr(module, "_interview_coach_collect_sources", lambda *args: ({}, []))
    monkeypatch.setattr(module, "_interview_coach_answer_with_model", lambda *args: ("answer", [], ["offline"], []))
    state = client.post("/api/career/contexts", json={}).json()
    result = client.post(
        "/api/career/interview-coach/ask",
        json={"session_id": state["session_id"], "context_revision": 1, "question": "What is SQL?"},
    )
    assert result.status_code == 200, result.text
    restored = client.get("/api/career/contexts/" + state["session_id"]).json()
    assert [m["role"] for m in restored["coach_messages"]] == ["user", "assistant"]
    assert restored["context_revision"] == 2
    assert (
        client.post(
            "/api/career/interview-coach/ask",
            json={"session_id": state["session_id"], "context_revision": 1, "question": "Again?"},
        ).status_code
        == 409
    )
