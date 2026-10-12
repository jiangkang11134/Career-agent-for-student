"""Durable workspace context and append-only context events."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WorkspaceSession:
    session_id: str
    tenant_id: str
    feature: str = "career"
    active_run_id: str = ""
    active_interview_id: str = ""
    input_bundle_id: str = ""
    revision: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "messages": list(self.messages)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkspaceSession":
        return cls(
            session_id=str(value.get("session_id") or ""),
            tenant_id=str(value.get("tenant_id") or ""),
            feature=str(value.get("feature") or "career"),
            active_run_id=str(value.get("active_run_id") or ""),
            active_interview_id=str(value.get("active_interview_id") or ""),
            input_bundle_id=str(value.get("input_bundle_id") or ""),
            revision=int(value.get("revision") or 0),
            messages=[dict(item) for item in value.get("messages", []) if isinstance(item, dict)],
            updated_at=float(value.get("updated_at") or time.time()),
        )


@dataclass(frozen=True)
class ContextEvent:
    event_id: str
    session_id: str
    tenant_id: str
    revision: int
    event_type: str
    payload: dict[str, Any]
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class WorkspaceSessionStore:
    def __init__(self, root: str | os.PathLike[str], tenant_id: str) -> None:
        self.root = Path(root) / "workspace-sessions" / tenant_id
        self.root.mkdir(parents=True, exist_ok=True)

    def get_or_create(self, session_id: str = "", *, feature: str = "career") -> WorkspaceSession:
        selected = str(session_id or "").strip() or f"ws-{uuid.uuid4().hex[:20]}"
        path = self._path(selected)
        if path.exists():
            return WorkspaceSession.from_dict(json.loads(path.read_text(encoding="utf-8")))
        session = WorkspaceSession(session_id=selected, tenant_id=self.root.name, feature=feature)
        self.save(session, event_type="session_created")
        return session

    def save(self, session: WorkspaceSession, *, event_type: str, payload: dict[str, Any] | None = None) -> WorkspaceSession:
        session.revision += 1
        session.updated_at = time.time()
        path = self._path(session.session_id)
        self._atomic_write(path, session.to_dict())
        event = ContextEvent(str(uuid.uuid4()), session.session_id, session.tenant_id, session.revision, event_type, payload or {}, session.updated_at)
        self._atomic_write(self._event_path(session.session_id, session.revision), event.to_dict())
        return session

    def _path(self, session_id: str) -> Path:
        if not session_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in session_id):
            raise ValueError("invalid workspace session id")
        return self.root / f"{session_id}.json"

    def _event_path(self, session_id: str, revision: int) -> Path:
        return self.root / f"{session_id}.event-{revision:08d}.json"

    @staticmethod
    def _atomic_write(path: Path, value: dict[str, Any]) -> None:
        fd, name = tempfile.mkstemp(prefix=".workspace-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
                json.dump(value, output, ensure_ascii=False, indent=2)
                output.flush(); os.fsync(output.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name): os.unlink(name)
