"""In-process event hub for the Agent operations console.

The hub deliberately stores redacted summaries instead of prompts or source
material.  It is process-local: the admin UI can reconnect and replay the
recent buffer, while a later persistent event store can be introduced without
changing the event shape.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

_SECRET_RE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{12,}|(?:api[_-]?key|authorization|token|password|secret)\s*[:=]\s*)[^\s,;]+"
)
_WINDOWS_PATH_RE = re.compile(r"(?i)\b[A-Z]:[\\/][^\s,;\"']+")
_UNIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9])/(?:[^\s,;\"']+/){1,}[^\s,;\"']*")
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "prompt",
    "system_prompt",
    "messages",
    "content",
    "text",
    "body",
    "jd",
    "resume",
    "material",
    "source_text",
}
_PATH_KEYS = {
    "path",
    "source_path",
    "cwd",
    "workspace",
    "working_directory",
}


def _safe_path(value: Any) -> str:
    """Keep only the final path component in operational event payloads."""
    text = _SECRET_RE.sub("[redacted]", str(value or "").strip())
    if not text:
        return ""
    return re.split(r"[\\/]", text)[-1][:240]


def summarize(value: Any, *, limit: int = 600, key: str = "") -> str:
    """Return a short, secret-safe representation for operational display."""
    if key.casefold() in _SENSITIVE_KEYS and not key.casefold().endswith("_summary"):
        return "[redacted]"
    if value is None:
        return ""
    if isinstance(value, str):
        text = _SECRET_RE.sub("[redacted]", value).strip()
        text = _WINDOWS_PATH_RE.sub("<local-path>", text)
        text = _UNIX_PATH_RE.sub("<local-path>", text)
    elif isinstance(value, (dict, list, tuple)):
        if isinstance(value, dict):
            safe = {
                str(k): summarize(v, limit=220, key=str(k))
                for k, v in value.items()
                if str(k).casefold() not in _SENSITIVE_KEYS
                or str(k).casefold().endswith("_summary")
            }
        else:
            safe = [summarize(item, limit=180) for item in value[:20]]
        text = json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
    else:
        text = str(value)
    if len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


@dataclass(frozen=True, slots=True)
class AgentEvent:
    sequence: int
    event_id: str
    timestamp: float
    event_type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            **self.payload,
        }


class AgentEventHub:
    """Thread-safe recent-event buffer with blocking subscriptions."""

    def __init__(self, *, max_events: int = 2_000) -> None:
        self.max_events = max(10, int(max_events))
        self._events: deque[AgentEvent] = deque(maxlen=self.max_events)
        self._subscribers: dict[str, queue.Queue[AgentEvent]] = {}
        self._lock = threading.RLock()
        self._sequence = 0

    def publish(self, event_type: str, **fields: Any) -> dict[str, Any]:
        """Publish one event and return its serialized, redacted payload."""
        now = time.time()
        with self._lock:
            self._sequence += 1
            event = AgentEvent(
                sequence=self._sequence,
                event_id=f"agent-event-{self._sequence}-{uuid.uuid4().hex[:10]}",
                timestamp=now,
                event_type=str(event_type or "agent_progress"),
                payload=self._normalize(fields),
            )
            self._events.append(event)
            subscribers = tuple(self._subscribers.values())
        for subscriber in subscribers:
            # A slow browser must not block an Agent. Drop the oldest queued
            # event for that subscriber before enqueueing the latest one.
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    pass
                try:
                    subscriber.put_nowait(event)
                except queue.Full:
                    pass
        return event.to_dict()

    @staticmethod
    def _normalize(fields: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in fields.items():
            name = str(key)
            if name.casefold() in _PATH_KEYS:
                payload[name] = _safe_path(value)
                continue
            if name.casefold() in _SENSITIVE_KEYS and not name.casefold().endswith("_summary"):
                payload[name] = "[redacted]"
            elif name.endswith("_summary") or name in {"message", "error", "result"}:
                payload[name] = summarize(value, key=name)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                payload[name] = summarize(value, key=name)
            else:
                payload[name] = summarize(value, key=name)
        return payload

    def snapshot(self, *, limit: int = 200, after: int = 0) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit or 200), self.max_events))
        with self._lock:
            events = [event for event in self._events if event.sequence > int(after or 0)]
        return [event.to_dict() for event in events[-bounded:]]

    def subscribe(self, *, max_queue: int = 256) -> tuple[str, queue.Queue[AgentEvent]]:
        token = uuid.uuid4().hex
        with self._lock:
            self._subscribers[token] = queue.Queue(maxsize=max(10, int(max_queue)))
            return token, self._subscribers[token]

    def subscribe_after(self, after: int = 0, *, max_queue: int = 256) -> tuple[str, queue.Queue[AgentEvent], list[dict[str, Any]]]:
        """Create a subscription and atomically return its replay prefix."""
        token, subscriber = self.subscribe(max_queue=max_queue)
        return token, subscriber, self.events_for(after=after, limit=self.max_events)

    def unsubscribe(self, token: str) -> None:
        with self._lock:
            self._subscribers.pop(str(token), None)

    def events_for(self, *, after: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        return self.snapshot(limit=limit, after=after)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._sequence = 0
            for subscriber in self._subscribers.values():
                while True:
                    try:
                        subscriber.get_nowait()
                    except queue.Empty:
                        break


agent_event_hub = AgentEventHub()


__all__ = ["AgentEvent", "AgentEventHub", "agent_event_hub", "summarize"]
