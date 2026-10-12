"""Append-only session events and the model-visible Surface projection.

The runtime still exposes a normal ``list[dict]`` message view for existing
adapters.  This module makes the ownership explicit: events are immutable
facts, while ``SurfaceProjection`` is the replaceable view sent to a model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

EVENT_SCHEMA_VERSION = "1.0"


class SurfaceOp(str, Enum):
    """Operations that can change the model-visible Surface."""

    APPEND = "message_appended"
    REPLACE = "surface_replaced"


@dataclass(frozen=True, slots=True)
class WorkingTurn:
    """An open Turn derived from immutable session events."""

    session_id: str
    turn_id: str
    goal: str
    started_sequence: int
    source_event_seqs: tuple[int, ...]
    messages: tuple[dict[str, Any], ...]
    status: str = "open"


@dataclass(frozen=True, slots=True)
class HistoryTurn:
    """A closed Turn derived from immutable session events."""

    session_id: str
    turn_id: str
    goal: str
    started_sequence: int
    ended_sequence: int
    source_event_seqs: tuple[int, ...]
    messages: tuple[dict[str, Any], ...]
    status: str
    summary: str = ""


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """One immutable event in a session's append-only log."""

    session_id: str
    sequence: int
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = ""
    turn_id: str | None = None
    source: str = "runtime"
    created_at: float = field(default_factory=time.time)
    revision: int = 1
    content_hash: str = ""
    schema_version: str = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        session_id = str(self.session_id or "").strip()
        event_type = str(self.event_type or "").strip()
        if not session_id or not event_type:
            raise ValueError("SessionEvent requires session_id and event_type")
        if self.sequence < 1 or self.revision < 1:
            raise ValueError("SessionEvent sequence and revision must be positive")
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError("Unsupported SessionEvent schema version")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "event_id", str(self.event_id or uuid.uuid4().hex))
        object.__setattr__(self, "turn_id", str(self.turn_id or "").strip() or None)
        object.__setattr__(self, "source", str(self.source or "runtime").strip() or "runtime")
        payload = copy.deepcopy(dict(self.payload or {}))
        object.__setattr__(self, "payload", payload)
        expected = _stable_hash(
            {
                "session_id": session_id,
                "sequence": self.sequence,
                "event_type": event_type,
                "payload": payload,
                "turn_id": self.turn_id,
                "source": self.source,
                "created_at": self.created_at,
                "revision": self.revision,
                "schema_version": self.schema_version,
            }
        )
        if self.content_hash and self.content_hash != expected:
            raise ValueError("SessionEvent content_hash does not match event content")
        object.__setattr__(self, "content_hash", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "payload": copy.deepcopy(self.payload),
            "turn_id": self.turn_id,
            "source": self.source,
            "created_at": self.created_at,
            "revision": self.revision,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SessionEvent":
        return cls(
            schema_version=str(value.get("schema_version") or EVENT_SCHEMA_VERSION),
            event_id=str(value.get("event_id") or ""),
            session_id=str(value["session_id"]),
            sequence=int(value["sequence"]),
            event_type=str(value["event_type"]),
            payload=dict(value.get("payload") or {}),
            turn_id=value.get("turn_id"),
            source=str(value.get("source") or "runtime"),
            created_at=float(value.get("created_at", time.time())),
            revision=int(value.get("revision", 1)),
            content_hash=str(value.get("content_hash") or ""),
        )


@dataclass
class SurfaceProjection:
    """The current model-visible message surface derived from events."""

    session_id: str
    revision: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)

    def apply(self, event: SessionEvent) -> None:
        if event.session_id != self.session_id:
            raise ValueError("Cannot apply an event from another session")
        if event.event_type == SurfaceOp.APPEND.value:
            message = event.payload.get("message")
            if isinstance(message, dict):
                self.messages.append(copy.deepcopy(message))
            self.revision = max(self.revision, event.revision)
        elif event.event_type == SurfaceOp.REPLACE.value:
            expected = int(event.payload.get("previous_revision", self.revision))
            if expected != self.revision:
                raise ValueError("Surface replacement revision conflict during replay")
            messages = event.payload.get("messages")
            if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
                raise ValueError("Surface replacement requires a message list")
            self.messages = copy.deepcopy(messages)
            self.revision = event.revision

    def snapshot(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.messages)


class SurfaceRevisionConflict(RuntimeError):
    """Raised when a replacement was based on a stale Surface revision."""


@dataclass
class SessionEventLog:
    """Append-only event collection with deterministic Surface replay."""

    session_id: str
    events: list[SessionEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.session_id = str(self.session_id or "").strip()
        if not self.session_id:
            raise ValueError("SessionEventLog requires session_id")
        self.events = list(self.events)
        self.validate()

    @property
    def next_sequence(self) -> int:
        return len(self.events) + 1

    @property
    def surface_revision(self) -> int:
        projection = self.replay()
        return projection.revision

    def validate(self) -> None:
        for expected, event in enumerate(self.events, start=1):
            if event.session_id != self.session_id or event.sequence != expected:
                raise ValueError("Session event log sequence or session ID is invalid")

    def append(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
        source: str = "runtime",
        revision: int | None = None,
    ) -> SessionEvent:
        event = SessionEvent(
            session_id=self.session_id,
            sequence=self.next_sequence,
            event_type=event_type,
            payload=payload or {},
            turn_id=turn_id,
            source=source,
            revision=revision if revision is not None else max(1, self.surface_revision),
        )
        self.events.append(event)
        return event

    def append_messages(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        turn_id: str | None = None,
        source: str = "runtime",
    ) -> tuple[SessionEvent, ...]:
        projection = self.replay()
        created: list[SessionEvent] = []
        revision = projection.revision
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("Session messages must be dictionaries")
            revision += 1
            created.append(
                self.append(
                    SurfaceOp.APPEND.value,
                    {"message": copy.deepcopy(message)},
                    turn_id=turn_id,
                    source=source,
                    revision=revision,
                )
            )
        return tuple(created)

    def replace_surface(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        expected_revision: int | None = None,
        turn_id: str | None = None,
        source: str = "compaction",
        reason: str = "surface replacement",
        metadata: dict[str, Any] | None = None,
    ) -> SessionEvent:
        projection = self.replay()
        expected = projection.revision if expected_revision is None else int(expected_revision)
        if expected != projection.revision:
            raise SurfaceRevisionConflict(
                f"Surface revision changed: expected {expected}, current {projection.revision}"
            )
        replacement = [copy.deepcopy(item) for item in messages]
        if not all(isinstance(item, dict) for item in replacement):
            raise TypeError("Surface messages must be dictionaries")
        payload = {
                "previous_revision": expected,
                "messages": replacement,
                "reason": str(reason or "surface replacement"),
                "source_message_count": len(projection.messages),
                "replacement_message_count": len(replacement),
        }
        if metadata:
            payload["metadata"] = copy.deepcopy(metadata)
        return self.append(
            SurfaceOp.REPLACE.value,
            payload,
            turn_id=turn_id,
            source=source,
            revision=expected + 1,
        )

    def derive_turns(self) -> tuple[tuple[WorkingTurn, ...], tuple[HistoryTurn, ...]]:
        """Derive explicit open and closed Turns without mutating the log."""
        builders: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        history: list[HistoryTurn] = []
        for event in self.events:
            turn_id = event.turn_id
            if not turn_id:
                continue
            if event.event_type == "turn_started":
                if turn_id in builders:
                    raise ValueError(f"Duplicate TurnStarted event for {turn_id}")
                builders[turn_id] = {
                    "goal": str(event.payload.get("goal") or ""),
                    "started_sequence": event.sequence,
                    "source_event_seqs": [event.sequence],
                    "messages": [],
                }
                order.append(turn_id)
                continue
            builder = builders.get(turn_id)
            if builder is None:
                continue
            builder["source_event_seqs"].append(event.sequence)
            if event.event_type == SurfaceOp.APPEND.value:
                message = event.payload.get("message")
                if isinstance(message, dict):
                    builder["messages"].append(copy.deepcopy(message))
            elif event.event_type == "turn_ended":
                history.append(
                    HistoryTurn(
                        session_id=self.session_id,
                        turn_id=turn_id,
                        goal=builder["goal"],
                        started_sequence=builder["started_sequence"],
                        ended_sequence=event.sequence,
                        source_event_seqs=tuple(builder["source_event_seqs"]),
                        messages=tuple(builder["messages"]),
                        status=str(event.payload.get("status") or "completed"),
                        summary=str(event.payload.get("summary") or ""),
                    )
                )
                del builders[turn_id]

        working = tuple(
            WorkingTurn(
                session_id=self.session_id,
                turn_id=turn_id,
                goal=builders[turn_id]["goal"],
                started_sequence=builders[turn_id]["started_sequence"],
                source_event_seqs=tuple(builders[turn_id]["source_event_seqs"]),
                messages=tuple(builders[turn_id]["messages"]),
            )
            for turn_id in order
            if turn_id in builders
        )
        return working, tuple(history)

    def replay(self, initial_messages: Iterable[dict[str, Any]] = ()) -> SurfaceProjection:
        projection = SurfaceProjection(self.session_id)
        if not self.events:
            for message in initial_messages:
                projection.messages.append(copy.deepcopy(message))
            return projection
        for event in self.events:
            projection.apply(event)
        return projection

    def to_dict(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]

    @classmethod
    def from_dict(cls, session_id: str, values: Iterable[dict[str, Any]]) -> "SessionEventLog":
        return cls(session_id, [SessionEvent.from_dict(item) for item in values if isinstance(item, dict)])
