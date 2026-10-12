"""Small compatibility supervisor for the agent-loop control signals.

The supervisor intentionally aggregates observations only.  It does not own
tool execution or model routing, which keeps the control plane optional and
safe to disable in minimal deployments.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

SUPERVISOR_STATE_PATH = Path(
    os.environ.get("SMART_AGENT_SUPERVISOR_STATE", str(Path.cwd() / ".runtime" / "cybernetic_supervisor.json"))
)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ControlSnapshot:
    name: str
    health: float
    risk: float
    action: str = "observe"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ControlSnapshot name is required")
        object.__setattr__(self, "health", max(0.0, min(1.0, float(self.health))))
        object.__setattr__(self, "risk", max(0.0, min(1.0, float(self.risk))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "health": self.health,
            "risk": self.risk,
            "action": self.action,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ControlSnapshot":
        return cls(
            name=str(value.get("name") or "unknown"),
            health=float(value.get("health", 0.0)),
            risk=float(value.get("risk", 0.0)),
            action=str(value.get("action") or "observe"),
            metadata=dict(value.get("metadata") or {}),
            created_at=float(value.get("created_at", time.time())),
        )


@dataclass(frozen=True, slots=True)
class SupervisorReport:
    overall_health: float
    risk_level: RiskLevel
    recommended_actions: tuple[str, ...] = ()
    snapshots: tuple[ControlSnapshot, ...] = ()
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_health": self.overall_health,
            "risk_level": self.risk_level.value,
            "recommended_actions": list(self.recommended_actions),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SupervisorReport":
        return cls(
            overall_health=float(value.get("overall_health", 1.0)),
            risk_level=RiskLevel(value.get("risk_level", RiskLevel.LOW.value)),
            recommended_actions=tuple(str(item) for item in value.get("recommended_actions", ())),
            snapshots=tuple(ControlSnapshot.from_dict(item) for item in value.get("snapshots", ())),
            created_at=float(value.get("created_at", time.time())),
        )


class CyberneticSupervisor:
    """Deterministically aggregate control snapshots into a report."""

    def snapshot_from_context(self, context: dict[str, Any]) -> ControlSnapshot:
        sensor = dict(context.get("sensor") or {})
        predictor = dict(context.get("predictor") or {})
        usage = max(0.0, min(1.0, float(sensor.get("current_usage", 0.0))))
        urgency = max(0.0, min(1.0, float(predictor.get("urgency", 0.0))))
        risk = max(usage, urgency)
        action = "compact" if usage >= 0.85 else "observe"
        return ControlSnapshot("context", 1.0 - risk, risk, action)

    def report(self, snapshots: list[ControlSnapshot] | tuple[ControlSnapshot, ...]) -> SupervisorReport:
        values = tuple(snapshots)
        if not values:
            return SupervisorReport(1.0, RiskLevel.LOW, (), ())
        health = sum(item.health for item in values) / len(values)
        risk = max(item.risk for item in values)
        level = (
            RiskLevel.CRITICAL if risk >= 0.9 else
            RiskLevel.HIGH if risk >= 0.7 else
            RiskLevel.MEDIUM if risk >= 0.4 else
            RiskLevel.LOW
        )
        actions = tuple(dict.fromkeys(item.action for item in values if item.action and item.action != "observe"))
        return SupervisorReport(round(health, 6), level, actions, values)


def save_supervisor_report(report: SupervisorReport, path: str | Path | None = None) -> Path:
    target = Path(path) if path is not None else SUPERVISOR_STATE_PATH
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def load_supervisor_report(path: str | Path | None = None) -> SupervisorReport | None:
    target = Path(path) if path is not None else SUPERVISOR_STATE_PATH
    try:
        return SupervisorReport.from_dict(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
