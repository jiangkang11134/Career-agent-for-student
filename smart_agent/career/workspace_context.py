"""Transactional session projections backed by an append-only event stream."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class ContextConflict(ValueError):
    pass


class WorkspaceContextStore:
    def __init__(self, root: Path, owner: str):
        self.path = Path(root) / "workspace-context.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.owner = owner
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    owner TEXT NOT NULL, id TEXT NOT NULL, revision INTEGER NOT NULL,
                    snapshot TEXT NOT NULL, PRIMARY KEY(owner, id));
                CREATE TABLE IF NOT EXISTS context_events (
                    owner TEXT NOT NULL, session_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    request_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
                    created_at REAL NOT NULL, PRIMARY KEY(owner, session_id, revision),
                    UNIQUE(owner, session_id, request_id));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, feature="career"):
        if feature not in {"career", "coach", "interviewer"}:
            raise ValueError("Unsupported feature")
        identity = "workspace-" + uuid.uuid4().hex
        state = dict(
            session_id=identity,
            feature=feature,
            active_run_id="",
            active_interview_id="",
            input_bundle_id="",
            stage="library",
            status="active",
            context_revision=1,
            created_at=time.time(),
        )
        with self.connect() as db:
            db.execute("INSERT INTO sessions VALUES (?, ?, 1, ?)", (self.owner, identity, json.dumps(state)))
            db.execute(
                "INSERT INTO context_events VALUES (?, ?, 1, ?, ?, ?, ?)",
                (self.owner, identity, identity, "session_created", json.dumps(state), time.time()),
            )
        return state

    def get(self, identity):
        with self.connect() as db:
            row = db.execute("SELECT snapshot FROM sessions WHERE owner=? AND id=?", (self.owner, identity)).fetchone()
        if row is None:
            raise KeyError("Workspace session not found")
        return json.loads(row[0])

    def commit(self, identity, expected_revision, kind, patch, request_id):
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("A request_id of at most 128 characters is required")
        encoded = json.dumps(patch, ensure_ascii=False, sort_keys=True)
        if len(encoded.encode()) > 512_000:
            raise ValueError("Context event exceeds 512 KB")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT revision, snapshot FROM sessions WHERE owner=? AND id=?", (self.owner, identity)
            ).fetchone()
            if row is None:
                raise KeyError("Workspace session not found")
            previous = db.execute(
                "SELECT kind, payload FROM context_events WHERE owner=? AND session_id=? AND request_id=?",
                (self.owner, identity, request_id),
            ).fetchone()
            if previous:
                if previous[0] != kind or previous[1] != encoded:
                    raise ContextConflict("request_id already used for another operation")
                return json.loads(row[1])
            if row[0] != expected_revision:
                raise ContextConflict("Workspace context changed; reload before retrying")
            state = json.loads(row[1])
            state.update(patch)
            state["context_revision"] = row[0] + 1
            state["updated_at"] = time.time()
            db.execute(
                "INSERT INTO context_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.owner, identity, state["context_revision"], request_id, kind, encoded, state["updated_at"]),
            )
            db.execute(
                "UPDATE sessions SET revision=?, snapshot=? WHERE owner=? AND id=?",
                (state["context_revision"], json.dumps(state), self.owner, identity),
            )
        return state

    def events(self, identity, after=0, limit=100):
        self.get(identity)
        with self.connect() as db:
            rows = db.execute(
                "SELECT revision, kind, payload, created_at FROM context_events WHERE owner=? AND session_id=? AND revision>? ORDER BY revision LIMIT ?",
                (self.owner, identity, after, min(100, max(1, limit))),
            ).fetchall()
        return [dict(revision=r[0], event_type=r[1], payload=json.loads(r[2]), created_at=r[3]) for r in rows]

    def replay(self, identity):
        self.get(identity)
        state = {}
        cursor = 0
        while batch := self.events(identity, cursor):
            for event in batch:
                state.update(event["payload"])
                state["context_revision"] = event["revision"]
                if event["revision"] > 1:
                    state["updated_at"] = event["created_at"]
                cursor = event["revision"]
        return state
