"""Append-only user fact ledger with a derived current view.

The ledger is intentionally separate from conversational text and career
evidence.  A correction never edits an old assertion; it appends a transition
event that supersedes or retracts it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


def _configured_backend() -> str:
    selected = str(os.environ.get("SMART_AGENT_FACT_LEDGER_BACKEND") or "").strip().lower()
    if selected in {"file", "mysql"}:
        return selected
    return "mysql" if str(os.environ.get("SMART_AGENT_DATABASE_URL") or "").strip() else "file"


_MEMORY_OPS_RE = re.compile(r"<memory_ops>\s*(\[.*?])\s*</memory_ops>", re.IGNORECASE | re.DOTALL)


class FactStatus(str, Enum):
    PROPOSED = "proposed"
    PENDING_CONFIRMATION = "pending_confirmation"
    ASSERTED = "asserted"
    CONFIRMED = "confirmed"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    CONFLICTED = "conflicted"
    TOMBSTONED = "tombstoned"


@dataclass(frozen=True, slots=True)
class FactEvent:
    event_id: str
    fact_id: str
    session_id: str
    tenant_id: str
    content: str
    status: FactStatus
    source_turn_id: str
    supersedes_fact_id: str = ""
    reason: str = ""
    created_at: float = 0.0
    fact_key: str = ""
    confidence: float = 1.0
    actor_type: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        confidence = float(self.confidence)
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("fact confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "fact_id": self.fact_id,
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "content": self.content,
            "status": self.status.value,
            "source_turn_id": self.source_turn_id,
            "supersedes_fact_id": self.supersedes_fact_id,
            "reason": self.reason,
            "created_at": self.created_at,
            "fact_key": self.fact_key,
            "confidence": self.confidence,
            "actor_type": self.actor_type,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FactEvent":
        return cls(
            event_id=str(value["event_id"]),
            fact_id=str(value["fact_id"]),
            session_id=str(value.get("session_id") or ""),
            tenant_id=str(value.get("tenant_id") or ""),
            content=str(value.get("content") or ""),
            status=FactStatus(value.get("status") or FactStatus.ASSERTED),
            source_turn_id=str(value.get("source_turn_id") or ""),
            supersedes_fact_id=str(value.get("supersedes_fact_id") or ""),
            reason=str(value.get("reason") or ""),
            created_at=float(value.get("created_at") or 0.0),
            fact_key=str(value.get("fact_key") or ""),
            confidence=float(value.get("confidence", 1.0)),
            actor_type=str(value.get("actor_type") or "user"),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class MemoryOperation:
    """Validated memory proposal emitted by an LLM, never a direct write."""

    operation: str
    content: str = ""
    fact_id: str = ""
    fact_key: str = ""
    confidence: float = 0.0
    requires_confirmation: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation = str(self.operation or "").strip().lower()
        if operation not in {"assert", "update", "confirm", "retract", "delete"}:
            raise ValueError("unsupported memory operation")
        object.__setattr__(self, "operation", operation)
        confidence = float(self.confidence)
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("memory operation confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryOperation":
        if not isinstance(value, dict):
            raise ValueError("memory operation must be an object")
        return cls(
            operation=str(value.get("operation") or ""),
            content=str(value.get("content") or "").strip(),
            fact_id=str(value.get("fact_id") or "").strip(),
            fact_key=str(value.get("fact_key") or "").strip(),
            confidence=float(value.get("confidence", 0.0)),
            requires_confirmation=bool(value.get("requires_confirmation", True)),
            metadata=dict(value.get("metadata") or {}),
        )


def parse_memory_operations(text: str) -> tuple[MemoryOperation, ...]:
    """Parse the optional machine-readable memory proposal block from LLM output."""
    match = _MEMORY_OPS_RE.search(str(text or ""))
    if match is None:
        return ()
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid memory_ops JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("memory_ops must be an array")
    return tuple(MemoryOperation.from_dict(item) for item in payload)


def strip_memory_operations(text: str) -> str:
    """Remove the internal memory proposal block before displaying model output."""
    return _MEMORY_OPS_RE.sub("", str(text or "")).strip()


class FactLedger:
    """Durable, tenant-scoped append-only fact events."""

    _locks: dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()
    _db_engines: dict[str, Any] = {}
    _db_schema_ready: set[str] = set()
    _db_tables: dict[str, Any] = {}
    _db_migrated_files: set[str] = set()

    def __init__(self, root: str | Path, *, tenant_id: str = "", backend: str | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.backend = str(backend or _configured_backend()).strip().lower()
        if self.backend not in {"file", "mysql"}:
            raise ValueError("fact ledger backend must be file or mysql")
        if self.backend == "file":
            self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "facts.json"
        self.tenant_id = str(tenant_id or "").strip()
        self._engine: Any | None = None
        self._redis: Any | None = None
        self._redis_disabled = False
        key = str(self.path).casefold()
        with self._locks_guard:
            self._lock = self._locks.setdefault(key, threading.RLock())

    def events(self) -> tuple[FactEvent, ...]:
        with self._lock:
            if self.backend == "mysql":
                return self._db_events()
            if not self.path.exists():
                return ()
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                raw_events = payload.get("events", [])
                if not isinstance(raw_events, list):
                    raise ValueError("events must be a list")
                return tuple(FactEvent.from_dict(item) for item in raw_events)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Cannot load fact ledger: {exc}") from exc

    def append_assertion(
        self,
        content: str,
        *,
        session_id: str,
        source_turn_id: str = "",
        status: FactStatus = FactStatus.ASSERTED,
        supersedes_fact_id: str = "",
        reason: str = "",
        fact_key: str = "",
        confidence: float = 1.0,
        actor_type: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> FactEvent:
        text = str(content or "").strip()
        if not text:
            raise ValueError("fact content is required")
        session = str(session_id or "").strip()
        if not session:
            raise ValueError("session_id is required")
        now = time.time()
        seed = f"{self.tenant_id}|{session}|{source_turn_id}|{now}|{uuid.uuid4().hex}|{text}"
        event = FactEvent(
            event_id=f"fact-event-{uuid.uuid4().hex[:16]}",
            fact_id=f"fact-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:20]}",
            session_id=session,
            tenant_id=self.tenant_id,
            content=text,
            status=FactStatus(status),
            source_turn_id=str(source_turn_id or ""),
            supersedes_fact_id=str(supersedes_fact_id or ""),
            reason=str(reason or ""),
            created_at=now,
            fact_key=str(fact_key or "").strip(),
            confidence=confidence,
            actor_type=str(actor_type or "user").strip() or "user",
            metadata=dict(metadata or {}),
        )
        with self._lock:
            current = list(self.events())
            self._write([*current, event])
        return event

    def transition(
        self,
        fact_id: str,
        status: FactStatus,
        *,
        session_id: str,
        source_turn_id: str = "",
        replacement: str = "",
        reason: str = "",
        actor_type: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> FactEvent:
        target = str(fact_id or "").strip()
        if not target:
            raise ValueError("fact_id is required")
        current = next((item for item in reversed(self.events()) if item.fact_id == target), None)
        if current is None:
            raise KeyError(f"Unknown fact: {target}")
        next_status = FactStatus.ASSERTED if replacement else FactStatus(status)
        allowed = {
            FactStatus.PROPOSED: {FactStatus.PENDING_CONFIRMATION, FactStatus.ASSERTED, FactStatus.RETRACTED, FactStatus.TOMBSTONED},
            FactStatus.PENDING_CONFIRMATION: {FactStatus.CONFIRMED, FactStatus.RETRACTED, FactStatus.CONFLICTED, FactStatus.TOMBSTONED},
            FactStatus.ASSERTED: {FactStatus.CONFIRMED, FactStatus.SUPERSEDED, FactStatus.RETRACTED, FactStatus.CONFLICTED, FactStatus.TOMBSTONED},
            FactStatus.CONFIRMED: {FactStatus.SUPERSEDED, FactStatus.RETRACTED, FactStatus.CONFLICTED, FactStatus.TOMBSTONED},
            FactStatus.CONFLICTED: {FactStatus.CONFIRMED, FactStatus.SUPERSEDED, FactStatus.RETRACTED, FactStatus.TOMBSTONED},
        }
        if not replacement and next_status not in allowed.get(current.status, set()):
            raise ValueError(f"Invalid fact transition: {current.status.value} -> {next_status.value}")
        return self.append_assertion(
            replacement or current.content,
            session_id=session_id,
            source_turn_id=source_turn_id,
            # A replacement is a new active assertion; a pure correction is
            # a terminal retraction event.
            status=next_status,
            supersedes_fact_id=target,
            reason=reason,
            fact_key=current.fact_key,
            confidence=current.confidence,
            actor_type=actor_type,
            metadata=metadata or current.metadata,
        )

    def apply_memory_operation(
        self,
        operation: MemoryOperation | dict[str, Any],
        *,
        session_id: str,
        source_turn_id: str = "",
        actor_type: str = "llm",
    ) -> FactEvent:
        """Apply one LLM proposal through the guarded fact state machine."""
        proposal = operation if isinstance(operation, MemoryOperation) else MemoryOperation.from_dict(operation)
        active = list(self.active(session_id=session_id))
        target = None
        if proposal.fact_id:
            target = next((item for item in active if item.fact_id == proposal.fact_id), None)
        elif proposal.fact_key:
            target = next((item for item in reversed(active) if item.fact_key == proposal.fact_key), None)
        if target is None and proposal.operation in {"confirm", "retract", "delete"}:
            history = [item for item in self.events() if item.session_id == session_id]
            if proposal.fact_id:
                target = next((item for item in reversed(history) if item.fact_id == proposal.fact_id), None)
            elif proposal.fact_key:
                target = next((item for item in reversed(history) if item.fact_key == proposal.fact_key), None)
        if proposal.operation == "assert":
            if not proposal.content:
                raise ValueError("assert operation requires content")
            return self.append_assertion(
                proposal.content,
                session_id=session_id,
                source_turn_id=source_turn_id,
                status=(
                    FactStatus.ASSERTED
                    if not proposal.requires_confirmation
                    else FactStatus.PENDING_CONFIRMATION
                ),
                fact_key=proposal.fact_key,
                confidence=proposal.confidence,
                actor_type=actor_type,
                metadata=proposal.metadata,
                reason=(
                    "LLM proposal; durable fact"
                    if not proposal.requires_confirmation
                    else "LLM proposal; user confirmation required"
                ),
            )
        if target is None:
            raise KeyError("memory operation target was not found in the active fact view")
        if proposal.operation == "confirm":
            if actor_type != "user":
                raise PermissionError("only the user can confirm a fact")
            return self.transition(
                target.fact_id,
                FactStatus.CONFIRMED,
                session_id=session_id,
                source_turn_id=source_turn_id,
                reason="user confirmation",
                actor_type="user",
            )
        if proposal.operation == "update":
            if not proposal.content:
                raise ValueError("update operation requires content")
            if actor_type != "user":
                return self.append_assertion(
                    proposal.content,
                    session_id=session_id,
                    source_turn_id=source_turn_id,
                    status=(
                        FactStatus.ASSERTED
                        if not proposal.requires_confirmation
                        else FactStatus.PENDING_CONFIRMATION
                    ),
                    supersedes_fact_id=target.fact_id,
                    fact_key=proposal.fact_key or target.fact_key,
                    confidence=proposal.confidence,
                    actor_type=actor_type,
                    metadata=proposal.metadata,
                    reason=(
                        "LLM proposal update; durable fact"
                        if not proposal.requires_confirmation
                        else "LLM proposal update; user confirmation required"
                    ),
                )
            return self.transition(
                target.fact_id,
                FactStatus.SUPERSEDED,
                session_id=session_id,
                source_turn_id=source_turn_id,
                replacement=proposal.content,
                reason="LLM proposal update",
                actor_type=actor_type,
                metadata=proposal.metadata,
            )
        if actor_type != "user":
            raise PermissionError("only the user can retract or delete a fact")
        return self.transition(
            target.fact_id,
            FactStatus.TOMBSTONED if proposal.operation == "delete" else FactStatus.RETRACTED,
            session_id=session_id,
            source_turn_id=source_turn_id,
            reason="user requested deletion" if proposal.operation == "delete" else "user retraction",
            actor_type="user",
        )

    def active(self, *, session_id: str | None = None) -> tuple[FactEvent, ...]:
        selected_session = str(session_id or "").strip()
        if self.backend == "mysql":
            cached = self._cache_get(selected_session)
            if cached is not None:
                return cached
        events = [
            event for event in self.events()
            if not selected_session or event.session_id == selected_session
        ]
        state: dict[str, FactEvent] = {}
        for event in events:
            if event.supersedes_fact_id:
                state.pop(event.supersedes_fact_id, None)
            if event.status in {FactStatus.ASSERTED, FactStatus.CONFIRMED, FactStatus.CONFLICTED}:
                state[event.fact_id] = event
        result = tuple(state.values())
        if self.backend == "mysql":
            self._cache_set(selected_session, result)
        return result

    def render_context(self, *, session_id: str, limit: int = 32) -> str:
        # Facts are tenant-level long-term memory, not conversation-local
        # context. A new browser window/session must receive the same active
        # user facts.
        active = self.active(session_id="")
        if not active:
            return ""
        lines = ["## User Fact Ledger (authoritative current view)"]
        lines.append("Use these facts as the latest user-provided state. Superseded or retracted facts are not active.")
        for event in active[-max(1, int(limit)):]:
            marker = "[CONFLICT] " if event.status is FactStatus.CONFLICTED else ""
            lines.append(f"- {marker}{event.content} (fact_id={event.fact_id})")
        return "\n".join(lines)

    def record_user_message(
        self,
        text: str,
        *,
        session_id: str,
        source_turn_id: str = "",
    ) -> tuple[FactEvent, ...]:
        """Capture explicit facts and deterministic corrections from a turn.

        This intentionally has a narrow trigger surface.  Ambiguous prose is
        left in the normal conversation instead of silently becoming memory.
        """
        content = str(text or "").strip()
        if not content:
            return ()
        # Corrections may arrive in a different conversation from the
        # original assertion, so resolve them against the tenant-wide ledger.
        active = list(self.active(session_id=""))
        correction = self._correction_target(content, active)
        created: list[FactEvent] = []
        if correction is not None:
            target, replacement, status, reason = correction
            created.append(
                self.transition(
                    target.fact_id,
                    status,
                    session_id=session_id,
                    source_turn_id=source_turn_id,
                    replacement=replacement,
                    reason=reason,
                )
            )
            if not replacement:
                return tuple(created)
        assertion = self._extract_assertion(content)
        if assertion:
            created.append(
                self.append_assertion(
                    assertion,
                    session_id=session_id,
                    source_turn_id=source_turn_id,
                    reason="explicit user statement",
                )
            )
        return tuple(created)

    def record_memory_request(
        self,
        text: str,
        *,
        session_id: str,
        source_turn_id: str = "",
    ) -> tuple[FactEvent, ...]:
        """Persist an explicit memory request without involving an agent loop."""
        if not self.is_memory_request(text):
            raise ValueError("message is not an explicit memory request")
        events = self.record_user_message(
            text,
            session_id=session_id,
            source_turn_id=source_turn_id,
        )
        if not events:
            raise ValueError("未识别到要记住的具体内容")
        return events

    @staticmethod
    def memory_response(events: tuple[FactEvent, ...] | list[FactEvent]) -> str:
        """Render a concise acknowledgement for the memory fast path."""
        if not events:
            raise ValueError("memory events are required")
        latest = events[-1]
        if latest.status is FactStatus.RETRACTED:
            return f"已撤回：{latest.content}"
        if latest.supersedes_fact_id:
            return f"已更新记忆：{latest.content}"
        return f"已记住：{latest.content}"

    @staticmethod
    def is_memory_request(text: str) -> bool:
        """Return true for an explicit memory assertion/correction only."""
        content = str(text or "").strip()
        if not content:
            return False
        # Common UTF-8 Chinese profile forms (kept separate from legacy
        # compatibility patterns that may contain mojibake).
        if re.match(r"^(?:我叫|我的名字是|我的名字叫|名字是|名字叫)\s*[^，。,。！？!?]{1,80}\s*[。！？!?]?$", content):
            return True
        if re.match(r"^(?:请|帮我|请帮我)?\s*(?:记住|牢记|记下来|保存到记忆|remember)", content, re.IGNORECASE):
            return True
        # Direct profile statements are implicit memory requests even when
        # the user does not say “remember”.
        if re.match(
            r"^(?:我的|我目前的|我现在的)[^。！？!?]{1,160}$",
            content,
        ) and re.search(r"(?:是|为|叫|在|有|来自|就读于|毕业于)", content):
            return not bool(re.search(r"(?i)(?:写入|修改文件|运行测试|扫描项目|代码|代码库|仓库|部署|调试|重构)", content))
        if re.search(r"(?:撤回|作废|刚才.*(?:错|不对)|更正|实际是|应该是)", content):
            return not bool(re.search(r"(?i)(?:写入|修改文件|运行测试|扫描项目|代码|代码库|仓库|部署)", content))
        return False

    @staticmethod
    def _extract_assertion(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        direct_name = re.match(
            r"^(?:我叫|我的名字是|我的名字叫|名字是|名字叫)\s*([^，。,。！？!?]{1,80})\s*[。！？!?]?$",
            normalized,
        )
        if direct_name:
            return f"我的名字是{direct_name.group(1).strip()}"
        if re.match(r"^(?:我的|我目前的|我现在的)[^。！？!?]{1,160}$", normalized) and re.search(
            r"(?:是|为|叫|在|有|来自|就读于|毕业于)", normalized
        ):
            return normalized
        patterns = (
            r"^(?:记住|请记住|牢记|我的事实是|事实是)[:：]?\s*(.+)$",
            r"^(?:我(?:的|目前的|现在的)|项目(?:的)?)([^。！？!?]{2,120})$",
            r"^(?:必须|不要|不能|我想要|我需要|以后都)([^。！？!?]{2,160})$",
        )
        for pattern in patterns:
            match = re.match(pattern, normalized, re.IGNORECASE)
            if match:
                value = match.group(1).strip(" ：:，,。")
                return value if len(value) >= 2 else ""
        return ""

    @staticmethod
    def _correction_target(
        text: str,
        active: list[FactEvent],
    ) -> tuple[FactEvent, str, FactStatus, str] | None:
        if not active:
            return None
        lowered = text.casefold()
        retract_markers = ("撤回", "作废", "不算", "是错的", "错了", "不对", "不正确")
        if not any(marker in lowered for marker in retract_markers):
            return None
        target = active[-1]
        for candidate in active:
            words = [word for word in re.split(r"\W+", candidate.content.casefold()) if len(word) >= 2]
            if words and sum(word in lowered for word in words) >= max(1, len(words) // 2):
                target = candidate
                break
        replacement = ""
        replace_match = re.search(r"(?:应该是|改成|更正为|实际是)[:：]?\s*(.+)$", text)
        if replace_match:
            replacement = replace_match.group(1).strip(" ：:，,。")
        status = FactStatus.SUPERSEDED if replacement else FactStatus.RETRACTED
        return target, replacement, status, "user correction"

    def _write(self, events: list[FactEvent]) -> None:
        if self.backend == "mysql":
            self._db_write(events)
            return
        payload = {
            "schema_version": "1.0",
            "tenant_id": self.tenant_id,
            "events": [event.to_dict() for event in events],
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.root)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    def _db_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from sqlalchemy import create_engine

            from smart_agent.auth import default_database_url

            url = default_database_url()
            with self._locks_guard:
                self._engine = self._db_engines.get(url)
                if self._engine is None:
                    self._engine = create_engine(url, pool_pre_ping=True, future=True)
                    self._db_engines[url] = self._engine
                schema_ready = url in self._db_schema_ready
            if not schema_ready:
                self._ensure_db_schema(url)
                with self._locks_guard:
                    self._db_schema_ready.add(url)
            else:
                self._fact_events_table = self._db_tables[url]
                self._migrate_legacy_once(url)
            return self._engine
        except Exception as exc:
            raise RuntimeError(f"fact ledger database unavailable: {exc}") from exc

    def _ensure_db_schema(self, url: str) -> None:
        from sqlalchemy import Column, Float, Index, MetaData, String, Table, Text

        metadata = MetaData()
        table = Table(
            "fact_events",
            metadata,
            Column("event_id", String(64), primary_key=True),
            Column("fact_id", String(64), nullable=False, unique=True),
            Column("session_id", String(128), nullable=False),
            Column("tenant_id", String(128), nullable=False),
            Column("content", Text, nullable=False),
            Column("status", String(24), nullable=False),
            Column("source_turn_id", String(128), nullable=False),
            Column("supersedes_fact_id", String(64), nullable=False),
            Column("reason", String(255), nullable=False),
            Column("created_at", Float, nullable=False),
            Column("fact_key", String(128), nullable=False),
            Column("confidence", Float, nullable=False),
            Column("actor_type", String(32), nullable=False),
            Column("metadata_json", Text, nullable=False),
            Index("ix_fact_events_tenant_session_created", "tenant_id", "session_id", "created_at"),
        )
        metadata.create_all(self._engine)
        self._fact_events_table = table
        self._db_tables[url] = table
        self._migrate_legacy_once(url)

    def _migrate_legacy_once(self, url: str) -> None:
        key = f"{url}|{self.tenant_id}|{self.path}"
        with self._locks_guard:
            if key in self._db_migrated_files:
                return
        self._import_legacy_file(self._fact_events_table)
        with self._locks_guard:
            self._db_migrated_files.add(key)

    def _import_legacy_file(self, table: Any) -> None:
        """Import matching file events once when switching a tenant to MySQL."""
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            raw_events = payload.get("events", [])
            legacy_events = [
                FactEvent.from_dict(item)
                for item in raw_events
                if isinstance(item, dict) and str(item.get("tenant_id") or "") == self.tenant_id
            ]
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot migrate fact ledger: {exc}") from exc
        if not legacy_events:
            return
        from sqlalchemy import select

        with self._engine.begin() as conn:
            existing = {
                str(value)
                for value in conn.execute(
                    select(table.c.event_id).where(table.c.tenant_id == self.tenant_id)
                ).scalars()
            }
            for event in legacy_events:
                if event.event_id in existing:
                    continue
                conn.execute(
                    table.insert().values(
                        event_id=event.event_id,
                        fact_id=event.fact_id,
                        session_id=event.session_id,
                        tenant_id=event.tenant_id,
                        content=event.content,
                        status=event.status.value,
                        source_turn_id=event.source_turn_id,
                        supersedes_fact_id=event.supersedes_fact_id,
                        reason=event.reason,
                        created_at=event.created_at,
                        fact_key=event.fact_key,
                        confidence=event.confidence,
                        actor_type=event.actor_type,
                        metadata_json=json.dumps(event.metadata, ensure_ascii=False, sort_keys=True),
                    )
                )

    def _db_events(self) -> tuple[FactEvent, ...]:
        engine = self._db_engine()
        table = self._fact_events_table
        from sqlalchemy import select

        with engine.begin() as conn:
            rows = conn.execute(
                select(table)
                .where(table.c.tenant_id == self.tenant_id)
                .order_by(table.c.created_at, table.c.event_id)
            ).mappings().all()
        return tuple(
            FactEvent(
                event_id=str(row["event_id"]),
                fact_id=str(row["fact_id"]),
                session_id=str(row["session_id"]),
                tenant_id=str(row["tenant_id"]),
                content=str(row["content"]),
                status=FactStatus(str(row["status"])),
                source_turn_id=str(row["source_turn_id"]),
                supersedes_fact_id=str(row["supersedes_fact_id"]),
                reason=str(row["reason"]),
                created_at=float(row["created_at"]),
                fact_key=str(row["fact_key"] or ""),
                confidence=float(row["confidence"]),
                actor_type=str(row["actor_type"] or "user"),
                metadata=json.loads(str(row["metadata_json"] or "{}")),
            )
            for row in rows
        )

    def _db_write(self, events: list[FactEvent]) -> None:
        engine = self._db_engine()
        table = self._fact_events_table
        from sqlalchemy import select

        with engine.begin() as conn:
            existing = {
                str(value)
                for value in conn.execute(
                    select(table.c.event_id).where(table.c.tenant_id == self.tenant_id)
                ).scalars()
            }
            for event in events:
                if event.event_id in existing:
                    continue
                conn.execute(
                    table.insert().values(
                        event_id=event.event_id,
                        fact_id=event.fact_id,
                        session_id=event.session_id,
                        tenant_id=event.tenant_id,
                        content=event.content,
                        status=event.status.value,
                        source_turn_id=event.source_turn_id,
                        supersedes_fact_id=event.supersedes_fact_id,
                        reason=event.reason,
                        created_at=event.created_at,
                        fact_key=event.fact_key,
                        confidence=event.confidence,
                        actor_type=event.actor_type,
                        metadata_json=json.dumps(event.metadata, ensure_ascii=False, sort_keys=True),
                    )
                )
        for session_id in {event.session_id for event in events}:
            self._cache_delete(session_id)
        self._cache_delete("")

    def _redis_client(self):
        if self._redis_disabled:
            return None
        if self._redis is not None:
            return self._redis
        try:
            from redis import Redis

            url = str(os.environ.get("SMART_AGENT_REDIS_URL") or "").strip()
            if not url:
                self._redis_disabled = True
                return None
            client = Redis.from_url(url, decode_responses=True, socket_connect_timeout=0.5, socket_timeout=0.5)
            client.ping()
            self._redis = client
        except Exception:
            self._redis_disabled = True
        return self._redis

    def _cache_key(self, session_id: str) -> str:
        scope = f"{self.tenant_id}:{session_id}"
        digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()
        return f"smart-agent:fact-ledger:{digest}"

    def _cache_get(self, session_id: str) -> tuple[FactEvent, ...] | None:
        client = self._redis_client()
        if client is None:
            return None
        try:
            raw = client.get(self._cache_key(session_id))
            if not raw:
                return None
            values = json.loads(raw)
            return tuple(FactEvent.from_dict(item) for item in values if isinstance(item, dict))
        except Exception:
            return None

    def _cache_set(self, session_id: str, events: tuple[FactEvent, ...]) -> None:
        client = self._redis_client()
        if client is None:
            return
        try:
            client.setex(
                self._cache_key(session_id),
                3600,
                json.dumps([event.to_dict() for event in events], ensure_ascii=False),
            )
        except Exception:
            return

    def _cache_delete(self, session_id: str) -> None:
        client = self._redis_client()
        if client is None:
            return
        try:
            client.delete(self._cache_key(session_id))
        except Exception:
            return
