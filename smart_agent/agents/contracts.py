"""Versioned contracts exchanged between parent and child agents."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from smart_agent.domains.runtime import require_explicit_domain

AGENT_CONTRACT_VERSION = "1.0"
_FINDING_SEVERITIES = frozenset({"critical", "major", "minor", "info"})
_RUN_STATUSES = frozenset({"completed", "failed", "timed_out"})


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _require_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    return text


def _json_payload(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object.")
    try:
        serialized = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable.") from exc
    return json.loads(serialized)


def _artifact_hash(artifact: "AgentArtifact") -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "artifact_type": artifact.artifact_type,
                "domain": artifact.domain,
                "session_id": artifact.session_id,
                "created_by": artifact.created_by,
                "content": artifact.content,
                "source_artifact_ids": artifact.source_artifact_ids,
                "revision": artifact.revision,
                "schema_version": artifact.schema_version,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AgentArtifact:
    """A versioned, traceable unit of data passed across an agent boundary."""

    artifact_id: str
    artifact_type: str
    domain: str
    session_id: str
    created_by: str
    content: dict[str, Any]
    source_artifact_ids: tuple[str, ...] = ()
    content_hash: str = ""
    revision: int = 1
    schema_version: str = AGENT_CONTRACT_VERSION
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _require_text(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "artifact_type", _require_text(self.artifact_type, "artifact_type"))
        object.__setattr__(self, "domain", require_explicit_domain(self.domain).value)
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(self, "created_by", _require_text(self.created_by, "created_by"))
        object.__setattr__(self, "content", _json_payload(self.content, "content"))
        object.__setattr__(
            self,
            "source_artifact_ids",
            tuple(_require_text(item, "source_artifact_id") for item in self.source_artifact_ids),
        )
        expected_hash = _artifact_hash(self)
        if self.content_hash and self.content_hash != expected_hash:
            raise ValueError("AgentArtifact content_hash does not match its content.")
        object.__setattr__(self, "content_hash", expected_hash)
        if self.revision < 1:
            raise ValueError("revision must be at least 1.")
        if self.schema_version != AGENT_CONTRACT_VERSION:
            raise ValueError(f"Unsupported AgentArtifact schema version: {self.schema_version}")

    def verify_integrity(self) -> None:
        if self.content_hash != _artifact_hash(self):
            raise ValueError("AgentArtifact content_hash does not match its content.")

    @classmethod
    def create(
        cls,
        artifact_type: str,
        *,
        domain: str,
        session_id: str,
        created_by: str,
        content: dict[str, Any],
        source_artifact_ids: tuple[str, ...] | list[str] = (),
        revision: int = 1,
    ) -> "AgentArtifact":
        return cls(
            artifact_id=_new_id("artifact"),
            artifact_type=artifact_type,
            domain=domain,
            session_id=session_id,
            created_by=created_by,
            content=content,
            source_artifact_ids=tuple(source_artifact_ids),
            revision=revision,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentArtifact":
        return cls(
            artifact_id=value["artifact_id"],
            artifact_type=value["artifact_type"],
            domain=value["domain"],
            session_id=value["session_id"],
            created_by=value["created_by"],
            content=value.get("content", {}),
            source_artifact_ids=tuple(value.get("source_artifact_ids", ())),
            content_hash=value.get("content_hash", ""),
            revision=int(value.get("revision", 1)),
            schema_version=value.get("schema_version", AGENT_CONTRACT_VERSION),
            created_at=float(value.get("created_at", time.time())),
        )

    def to_dict(self) -> dict[str, Any]:
        self.verify_integrity()
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "domain": self.domain,
            "session_id": self.session_id,
            "created_by": self.created_by,
            "content": _json_payload(self.content, "content"),
            "source_artifact_ids": list(self.source_artifact_ids),
            "content_hash": self.content_hash,
            "revision": self.revision,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class AgentTask:
    """A bounded child-agent assignment declared by the parent agent."""

    task_id: str
    task_type: str
    domain: str
    description: str
    parent_session_id: str
    child_session_id: str
    assigned_agent: str
    input_artifact_ids: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    timeout_seconds: float
    max_steps: int
    revision: int = 1
    schema_version: str = AGENT_CONTRACT_VERSION
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _require_text(self.task_id, "task_id"))
        object.__setattr__(self, "task_type", _require_text(self.task_type, "task_type"))
        object.__setattr__(self, "domain", require_explicit_domain(self.domain).value)
        object.__setattr__(self, "description", _require_text(self.description, "description"))
        object.__setattr__(
            self,
            "parent_session_id",
            _require_text(self.parent_session_id, "parent_session_id"),
        )
        object.__setattr__(
            self,
            "child_session_id",
            _require_text(self.child_session_id, "child_session_id"),
        )
        object.__setattr__(self, "assigned_agent", _require_text(self.assigned_agent, "assigned_agent"))
        object.__setattr__(
            self,
            "input_artifact_ids",
            tuple(_require_text(item, "input_artifact_id") for item in self.input_artifact_ids),
        )
        object.__setattr__(
            self,
            "allowed_tools",
            tuple(dict.fromkeys(_require_text(item, "allowed_tool") for item in self.allowed_tools)),
        )
        if not self.input_artifact_ids:
            raise ValueError("A child-agent task requires at least one input artifact.")
        if len(self.input_artifact_ids) != len(set(self.input_artifact_ids)):
            raise ValueError("input_artifact_ids cannot contain duplicates.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.revision < 1:
            raise ValueError("revision must be at least 1.")
        if self.schema_version != AGENT_CONTRACT_VERSION:
            raise ValueError(f"Unsupported AgentTask schema version: {self.schema_version}")

    @classmethod
    def create(
        cls,
        task_type: str,
        *,
        domain: str,
        description: str,
        parent_session_id: str,
        assigned_agent: str,
        input_artifact_ids: tuple[str, ...] | list[str],
        allowed_tools: tuple[str, ...] | list[str] | set[str],
        timeout_seconds: float,
        max_steps: int,
    ) -> "AgentTask":
        return cls(
            task_id=_new_id("task"),
            task_type=task_type,
            domain=domain,
            description=description,
            parent_session_id=parent_session_id,
            child_session_id=_new_id("session"),
            assigned_agent=assigned_agent,
            input_artifact_ids=tuple(input_artifact_ids),
            allowed_tools=tuple(sorted(allowed_tools)),
            timeout_seconds=timeout_seconds,
            max_steps=max_steps,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentTask":
        return cls(
            task_id=value["task_id"],
            task_type=value["task_type"],
            domain=value["domain"],
            description=value["description"],
            parent_session_id=value["parent_session_id"],
            child_session_id=value["child_session_id"],
            assigned_agent=value["assigned_agent"],
            input_artifact_ids=tuple(value.get("input_artifact_ids", ())),
            allowed_tools=tuple(value.get("allowed_tools", ())),
            timeout_seconds=float(value["timeout_seconds"]),
            max_steps=int(value["max_steps"]),
            revision=int(value.get("revision", 1)),
            schema_version=value.get("schema_version", AGENT_CONTRACT_VERSION),
            created_at=float(value.get("created_at", time.time())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "domain": self.domain,
            "description": self.description,
            "parent_session_id": self.parent_session_id,
            "child_session_id": self.child_session_id,
            "assigned_agent": self.assigned_agent,
            "input_artifact_ids": list(self.input_artifact_ids),
            "allowed_tools": list(self.allowed_tools),
            "timeout_seconds": self.timeout_seconds,
            "max_steps": self.max_steps,
            "revision": self.revision,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class AgentFinding:
    """A structured issue produced by a child agent."""

    finding_id: str
    severity: str
    title: str
    message: str
    created_by: str
    session_id: str
    source_artifact_ids: tuple[str, ...]
    file_path: str = ""
    line: int = 0
    rule_id: str = ""
    confidence: float = 1.0
    revision: int = 1
    schema_version: str = AGENT_CONTRACT_VERSION
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "finding_id", _require_text(self.finding_id, "finding_id"))
        severity = str(self.severity or "").strip().lower()
        if severity not in _FINDING_SEVERITIES:
            raise ValueError(f"Unsupported finding severity: {self.severity}")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "title", _require_text(self.title, "title"))
        object.__setattr__(self, "message", _require_text(self.message, "message"))
        object.__setattr__(self, "created_by", _require_text(self.created_by, "created_by"))
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(
            self,
            "source_artifact_ids",
            tuple(_require_text(item, "source_artifact_id") for item in self.source_artifact_ids),
        )
        if not self.source_artifact_ids:
            raise ValueError("A finding must cite at least one source artifact.")
        if self.line < 0:
            raise ValueError("line cannot be negative.")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1.")
        if self.revision < 1:
            raise ValueError("revision must be at least 1.")
        if self.schema_version != AGENT_CONTRACT_VERSION:
            raise ValueError(f"Unsupported AgentFinding schema version: {self.schema_version}")

    @classmethod
    def create(
        cls,
        *,
        severity: str,
        title: str,
        message: str,
        created_by: str,
        session_id: str,
        source_artifact_ids: tuple[str, ...] | list[str],
        file_path: str = "",
        line: int = 0,
        rule_id: str = "",
        confidence: float = 1.0,
    ) -> "AgentFinding":
        return cls(
            finding_id=_new_id("finding"),
            severity=severity,
            title=title,
            message=message,
            created_by=created_by,
            session_id=session_id,
            source_artifact_ids=tuple(source_artifact_ids),
            file_path=str(file_path or ""),
            line=int(line or 0),
            rule_id=str(rule_id or ""),
            confidence=float(confidence),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentFinding":
        return cls(
            finding_id=value["finding_id"],
            severity=value["severity"],
            title=value["title"],
            message=value["message"],
            created_by=value["created_by"],
            session_id=value["session_id"],
            source_artifact_ids=tuple(value.get("source_artifact_ids", ())),
            file_path=value.get("file_path", ""),
            line=int(value.get("line", 0)),
            rule_id=value.get("rule_id", ""),
            confidence=float(value.get("confidence", 1.0)),
            revision=int(value.get("revision", 1)),
            schema_version=value.get("schema_version", AGENT_CONTRACT_VERSION),
            created_at=float(value.get("created_at", time.time())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "created_by": self.created_by,
            "session_id": self.session_id,
            "source_artifact_ids": list(self.source_artifact_ids),
            "file_path": self.file_path,
            "line": self.line,
            "rule_id": self.rule_id,
            "confidence": self.confidence,
            "revision": self.revision,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """The only value a child-agent runner returns to its parent."""

    task: AgentTask
    status: str
    output_artifact: AgentArtifact | None = None
    findings: tuple[AgentFinding, ...] = ()
    elapsed_seconds: float = 0.0
    tool_call_count: int = 0
    token_count: int = 0
    token_count_source: str = ""
    tool_trace: tuple[dict[str, Any], ...] = ()
    error: str = ""

    def __post_init__(self) -> None:
        if self.status not in _RUN_STATUSES:
            raise ValueError(f"Unsupported child-agent status: {self.status}")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds cannot be negative.")
        if self.tool_call_count < 0 or self.token_count < 0:
            raise ValueError("Agent run counters cannot be negative.")
        if any(not isinstance(item, dict) for item in self.tool_trace):
            raise ValueError("Agent run tool_trace must contain objects.")
        if self.output_artifact is not None and self.output_artifact.session_id != self.task.child_session_id:
            raise ValueError("Output artifact must belong to the child session.")
        for finding in self.findings:
            if finding.session_id != self.task.child_session_id:
                raise ValueError("Findings must belong to the child session.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "status": self.status,
            "output_artifact": self.output_artifact.to_dict() if self.output_artifact else None,
            "findings": [finding.to_dict() for finding in self.findings],
            "elapsed_seconds": self.elapsed_seconds,
            "tool_call_count": self.tool_call_count,
            "token_count": self.token_count,
            "token_count_source": self.token_count_source,
            "tool_trace": [dict(item) for item in self.tool_trace],
            "error": self.error,
        }
