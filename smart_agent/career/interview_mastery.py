"""Persistent Career-owned knowledge mastery updated from completed interviews."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_agent.career.interview import InterviewSessionState, KnowledgeBeliefState

MASTERY_SCHEMA_VERSION = "1.1"
_SUPPORTED_MASTERY_SCHEMA_VERSIONS = frozenset({"1.0", MASTERY_SCHEMA_VERSION})


def _required_text(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


@dataclass(frozen=True, slots=True)
class KnowledgeMasteryProfile:
    tenant_id: str
    belief_state: KnowledgeBeliefState = field(default_factory=KnowledgeBeliefState)
    applied_interview_ids: tuple[str, ...] = ()
    revision: int = 1
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _required_text(self.tenant_id, "tenant_id"))
        interview_ids = tuple(
            dict.fromkeys(_required_text(item, "applied_interview_id") for item in self.applied_interview_ids)
        )
        object.__setattr__(self, "applied_interview_ids", interview_ids)
        if self.revision < 1:
            raise ValueError("revision must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MASTERY_SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "belief_state": self.belief_state.to_dict(),
            "applied_interview_ids": list(self.applied_interview_ids),
            "revision": self.revision,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KnowledgeMasteryProfile":
        if value.get("schema_version", "1.0") not in _SUPPORTED_MASTERY_SCHEMA_VERSIONS:
            raise ValueError("Unsupported knowledge mastery schema version")
        belief = dict(value.get("belief_state") or {})
        return cls(
            tenant_id=str(value["tenant_id"]),
            belief_state=KnowledgeBeliefState.from_dict(belief),
            applied_interview_ids=tuple(value.get("applied_interview_ids", ())),
            revision=int(value.get("revision", 1)),
            updated_at=float(value.get("updated_at", time.time())),
        )


class KnowledgeMasteryStore:
    """Idempotently merge completed interview observations into Career state."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def get(self, tenant_id: str) -> KnowledgeMasteryProfile:
        tenant_id = _required_text(tenant_id, "tenant_id")
        with self._lock:
            path = self._path(tenant_id)
            if not path.is_file():
                return KnowledgeMasteryProfile(tenant_id)
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Knowledge mastery profile must contain an object")
            profile = KnowledgeMasteryProfile.from_dict(value)
            if profile.tenant_id != tenant_id:
                raise ValueError("Knowledge mastery profile tenant mismatch")
            return profile

    def apply_interview(self, state: InterviewSessionState) -> KnowledgeMasteryProfile:
        with self._lock:
            current = self.get(state.tenant_id)
            if state.interview_id in current.applied_interview_ids:
                return current
            belief = current.belief_state.with_document_priors(
                state.belief_state.document_prior_count_by_corpus
            )
            for turn in state.turns:
                belief, _committed = belief.apply_updates(turn.knowledge_confidence_updates)
            updated = KnowledgeMasteryProfile(
                tenant_id=current.tenant_id,
                belief_state=belief,
                applied_interview_ids=(*current.applied_interview_ids, state.interview_id),
                revision=current.revision + 1,
                updated_at=time.time(),
            )
            self._write(self._path(state.tenant_id), updated.to_dict())
            return updated

    def _path(self, tenant_id: str) -> Path:
        identity = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:24]
        return self.root / f"mastery-{identity}.json"

    @staticmethod
    def _write(path: Path, value: dict[str, Any]) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=".mastery.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
