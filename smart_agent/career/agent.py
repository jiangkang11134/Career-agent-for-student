"""Single-agent Career turn orchestration.

The generic agent loop remains responsible for model/tool execution. This
module owns the Career-specific boundary around one turn: workspace preflight,
career-only runtime configuration, user confirmation detection, and a
structured result artifact.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smart_agent.agent_loop import run_agent_turn
from smart_agent.agents.contracts import AgentArtifact
from smart_agent.context_manager import token_count_with_estimation
from smart_agent.domains import get_domain_module
from smart_agent.fact_ledger import FactLedger, parse_memory_operations, strip_memory_operations
from smart_agent.tooling import ToolContext, ToolRegistry
from smart_agent.types import ModelAdapter, RuntimeEvent

_CAREER_STATUSES = frozenset({"completed", "awaiting_user", "blocked", "failed"})
CAREER_AGENT_RUN_SCHEMA_VERSION = "1.0"


def _text(value: Any, field_name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field_name} is required")
    return result


@dataclass(frozen=True, slots=True)
class CareerAgentRun:
    run_id: str
    session_id: str
    goal: str
    status: str
    elapsed_ms: float
    tool_call_count: int
    token_count: int
    token_count_source: str
    workspace_before_stage: str
    workspace_after_stage: str
    confirmation_target_count: int
    tool_trace: tuple[dict[str, Any], ...] = ()
    error: str = ""
    created_at: float = field(default_factory=time.time)
    schema_version: str = CAREER_AGENT_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in _CAREER_STATUSES:
            raise ValueError(f"Unsupported Career Agent status: {self.status}")
        if self.elapsed_ms < 0 or self.tool_call_count < 0 or self.token_count < 0:
            raise ValueError("Career Agent run metrics cannot be negative")
        if any(not isinstance(item, dict) for item in self.tool_trace):
            raise ValueError("Career Agent tool_trace must contain objects")
        if self.schema_version != CAREER_AGENT_RUN_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Career Agent run schema: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CareerAgentRun":
        return cls(**value)


class CareerAgentRunStore:
    _ROOT_LOCKS: dict[str, threading.RLock] = {}
    _ROOT_LOCKS_GUARD = threading.Lock()

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self._ROOT_LOCKS_GUARD:
            self._lock = self._ROOT_LOCKS.setdefault(str(self.root).casefold(), threading.RLock())

    def put(self, run: CareerAgentRun) -> CareerAgentRun:
        path = self.root / f"{run.run_id}.json"
        with self._lock:
            if path.exists():
                raise ValueError(f"Career Agent run already exists: {run.run_id}")
            descriptor, temporary = tempfile.mkstemp(prefix="career-run-", suffix=".tmp", dir=self.root)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(run.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                try:
                    Path(temporary).unlink()
                except OSError:
                    pass
        return run

    def list(self) -> tuple[CareerAgentRun, ...]:
        with self._lock:
            runs = tuple(
                CareerAgentRun.from_dict(json.loads(path.read_text(encoding="utf-8")))
                for path in self.root.glob("*.json")
            )
        return tuple(sorted(runs, key=lambda item: (item.created_at, item.run_id)))

    def metrics(self) -> dict[str, Any]:
        runs = self.list()
        return {
            "run_count": len(runs),
            "failed_count": sum(item.status == "failed" for item in runs),
            "blocked_count": sum(item.status == "blocked" for item in runs),
            "awaiting_user_count": sum(item.status == "awaiting_user" for item in runs),
            "tool_call_count": sum(item.tool_call_count for item in runs),
            "token_count": sum(item.token_count for item in runs),
            "average_elapsed_ms": (
                round(sum(item.elapsed_ms for item in runs) / len(runs), 3) if runs else 0.0
            ),
        }


@dataclass(frozen=True, slots=True)
class CareerAgentRequest:
    """Input for one bounded Career Agent turn."""

    message: str
    goal: str = "resume"
    session_id: str = ""
    history: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "message", _text(self.message, "message"))
        goal = _text(self.goal, "goal").lower()
        if goal not in {"resume", "interview", "application"}:
            raise ValueError("goal must be resume, interview, or application")
        object.__setattr__(self, "goal", goal)
        session_id = str(self.session_id or "").strip() or f"career-{uuid.uuid4().hex[:12]}"
        object.__setattr__(self, "session_id", session_id)
        normalized_history: list[dict[str, Any]] = []
        for item in self.history:
            if not isinstance(item, dict) or not item.get("role"):
                raise ValueError("history must contain message objects with role")
            normalized_history.append(dict(item))
        object.__setattr__(self, "history", tuple(normalized_history))


@dataclass(frozen=True, slots=True)
class CareerAgentResult:
    """Stable result returned by one Career Agent turn."""

    status: str
    request: CareerAgentRequest
    messages: tuple[dict[str, Any], ...]
    workspace_before: dict[str, Any]
    workspace_state: dict[str, Any]
    memory_layers: dict[str, Any]
    tool_calls: tuple[dict[str, Any], ...]
    awaiting_user: bool
    confirmation_targets: tuple[str, ...]
    stop_reason: str = ""
    output_artifact: AgentArtifact | None = field(default=None)
    error: str = ""

    def __post_init__(self) -> None:
        if self.status not in _CAREER_STATUSES:
            raise ValueError(f"Unsupported Career Agent status: {self.status}")

    @property
    def assistant_text(self) -> str:
        return "\n".join(
            str(item.get("content") or "")
            for item in self.messages
            if item.get("role") == "assistant" and item.get("content")
        ).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "request": {
                "message": self.request.message,
                "goal": self.request.goal,
                "session_id": self.request.session_id,
            },
            "assistant_text": self.assistant_text,
            "workspace_before": self.workspace_before,
            "workspace_state": self.workspace_state,
            "memory_layers": self.memory_layers,
            "tool_calls": list(self.tool_calls),
            "awaiting_user": self.awaiting_user,
            "confirmation_targets": list(self.confirmation_targets),
            "stop_reason": self.stop_reason,
            "error": self.error,
            "output_artifact": self.output_artifact.to_dict() if self.output_artifact else None,
        }


class CareerAgent:
    """Run one Career turn through the existing model/tool kernel."""

    def __init__(
        self,
        *,
        model: ModelAdapter,
        cwd: str,
        tools: ToolRegistry,
        runtime: dict[str, Any] | None = None,
        permissions: Any | None = None,
        session: Any | None = None,
        run_store: CareerAgentRunStore | None = None,
        max_steps: int = 12,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.model = model
        self.cwd = str(cwd)
        self.tools = tools
        self.runtime = dict(runtime or {})
        self.permissions = permissions
        self.session = session
        self.run_store = run_store
        self.max_steps = max_steps

    def run(self, request: CareerAgentRequest) -> CareerAgentResult:
        """Preflight the Career workspace, run one turn, and return an artifact."""
        started = time.monotonic()
        try:
            from smart_agent.agent_events import agent_event_hub
            event_hub = agent_event_hub
        except Exception:  # monitoring is optional
            event_hub = None

        def emit_event(event_type: str, **payload: Any) -> None:
            if event_hub is None:
                return
            try:
                event_hub.publish(event_type, **payload)
            except Exception:
                return

        runtime = {
            **self.runtime,
            "activeDomain": "career",
            "supervisorMode": "off",
            "careerSingleAgent": True,
            "careerGoal": request.goal,
            "sessionId": request.session_id,
            "agentType": "career",
            "agentName": "Career Agent",
        }
        def event_sink(event: dict[str, Any]) -> None:
            event_type = str(event.get("event_type") or "agent_progress")
            fields = {
                "agent_type": "career",
                "agent_name": "Career Agent",
                "session_id": request.session_id,
                "parent_session_id": str(runtime.get("parentSessionId") or ""),
                "child_session_id": str(runtime.get("childSessionId") or ""),
                **{key: value for key, value in event.items() if key != "event_type"},
            }
            emit_event(event_type, **fields)

        runtime["_event_sink"] = event_sink
        emit_event(
            "agent_started",
            agent_type="career",
            agent_name="Career Agent",
            session_id=request.session_id,
        )
        context = ToolContext(
            cwd=self.cwd,
            permissions=self.permissions,
            session=self.session,
            _runtime=runtime,
        )
        try:
            workspace_before = self._inspect_workspace(request, context)
            runtime["careerWorkspaceState"] = workspace_before
            system_prompt = self._system_prompt(request, workspace_before)
            fact_ledger = FactLedger(
                Path(str(runtime.get("careerStorePath") or self.cwd)) / "fact-ledger",
                tenant_id=str(runtime.get("careerTenantId") or "legacy-career"),
            )
            fact_context = fact_ledger.render_context(session_id=request.session_id)
            if fact_context:
                system_prompt = f"{system_prompt}\n\n{fact_context}"
            messages = [*request.history, {"role": "user", "content": request.message}]
            events: list[RuntimeEvent] = []
            result_messages = run_agent_turn(
                model=self.model,
                tools=self.tools,
                messages=messages,
                cwd=self.cwd,
                permissions=self.permissions,
                session=self.session,
                runtime=runtime,
                max_steps=self.max_steps,
                system_prompt=system_prompt,
                on_runtime_event=events.append,
            )
            self._apply_model_memory_ops(
                request=request,
                runtime=runtime,
                messages=result_messages,
            )
            workspace_state = self._inspect_workspace(request, context)
            tool_calls = tuple(
                {
                    "tool": str(item.get("toolName") or ""),
                    "input": dict(item.get("input") or {}),
                }
                for item in result_messages
                if item.get("role") == "assistant_tool_call"
            )
            awaiting_user = any(item["tool"] == "ask_user" for item in tool_calls)
            confirmation_targets = self._confirmation_targets(workspace_state)
            stop_reason = next(
                (event.stop_reason for event in reversed(events) if event.stop_reason),
                "",
            )
            status = "awaiting_user" if awaiting_user or confirmation_targets else "completed"
            if stop_reason in {"blocked", "max_steps"} and not awaiting_user:
                status = "blocked"
            artifact = self._artifact(
                request=request,
                status=status,
                workspace_before=workspace_before,
                workspace_state=workspace_state,
                memory_layers=self._memory_layers(request, workspace_state),
                tool_calls=tool_calls,
                awaiting_user=awaiting_user,
                confirmation_targets=confirmation_targets,
                stop_reason=stop_reason,
                assistant_text=self._assistant_text(result_messages),
                elapsed_seconds=time.monotonic() - started,
            )
            token_usage = token_count_with_estimation(result_messages)
            self._persist_run(
                artifact=artifact,
                request=request,
                status=status,
                elapsed_ms=(time.monotonic() - started) * 1000,
                tool_call_count=len(tool_calls),
                token_count=int(token_usage["total_tokens"]),
                token_count_source=str(token_usage["source"]),
                workspace_before=workspace_before,
                workspace_state=workspace_state,
                confirmation_target_count=len(confirmation_targets),
                tool_trace=tool_calls,
            )
            emit_event(
                "agent_finished",
                runtime=runtime,
                agent_type="career",
                agent_name="Career Agent",
                session_id=request.session_id,
                status=status,
                output_summary=self._assistant_text(result_messages)[:500],
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
            return CareerAgentResult(
                status=status,
                request=request,
                messages=tuple(result_messages),
                workspace_before=workspace_before,
                workspace_state=workspace_state,
                memory_layers=self._memory_layers(request, workspace_state),
                tool_calls=tool_calls,
                awaiting_user=awaiting_user,
                confirmation_targets=confirmation_targets,
                stop_reason=stop_reason,
                output_artifact=artifact,
            )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            artifact = self._artifact(
                request=request,
                status="failed",
                workspace_before={},
                workspace_state={},
                memory_layers={},
                tool_calls=(),
                awaiting_user=False,
                confirmation_targets=(),
                stop_reason="failed",
                assistant_text="",
                elapsed_seconds=time.monotonic() - started,
                error=error,
            )
            self._persist_run(
                artifact=artifact,
                request=request,
                status="failed",
                elapsed_ms=(time.monotonic() - started) * 1000,
                tool_call_count=0,
                token_count=0,
                token_count_source="unavailable",
                workspace_before={},
                workspace_state={},
                confirmation_target_count=0,
                tool_trace=(),
                error=error,
            )
            emit_event(
                "agent_failed",
                runtime=runtime,
                agent_type="career",
                agent_name="Career Agent",
                session_id=request.session_id,
                status="failed",
                output_summary=error[:500],
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
            return CareerAgentResult(
                status="failed",
                request=request,
                messages=(),
                workspace_before={},
                workspace_state={},
                memory_layers={},
                tool_calls=(),
                awaiting_user=False,
                confirmation_targets=(),
                stop_reason="failed",
                output_artifact=artifact,
                error=error,
            )

    def _inspect_workspace(self, request: CareerAgentRequest, context: ToolContext) -> dict[str, Any]:
        result = self.tools.execute("inspect_career_workspace", {"goal": request.goal}, context)
        if not result.ok:
            raise RuntimeError(result.output)
        try:
            payload = json.loads(result.output)
        except json.JSONDecodeError as exc:
            raise RuntimeError("inspect_career_workspace returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("inspect_career_workspace must return an object")
        return payload

    def _apply_model_memory_ops(
        self,
        *,
        request: CareerAgentRequest,
        runtime: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> None:
        """Commit only validated LLM proposals; never allow direct model writes."""
        ledger = None
        turn_id = str(runtime.get("turnId") or f"turn-{uuid.uuid4().hex[:12]}")
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content") or "")
            operations = parse_memory_operations(content)
            if operations:
                if ledger is None:
                    ledger = FactLedger(
                        Path(str(runtime.get("careerStorePath") or self.cwd)) / "fact-ledger",
                        tenant_id=str(runtime.get("careerTenantId") or "legacy-career"),
                    )
                for operation in operations:
                    ledger.apply_memory_operation(
                        operation,
                        session_id=request.session_id,
                        source_turn_id=turn_id,
                        actor_type="llm",
                    )
                message["content"] = strip_memory_operations(content)

    def _persist_run(
        self,
        *,
        artifact: AgentArtifact,
        request: CareerAgentRequest,
        status: str,
        elapsed_ms: float,
        tool_call_count: int,
        token_count: int,
        token_count_source: str,
        workspace_before: dict[str, Any],
        workspace_state: dict[str, Any],
        confirmation_target_count: int,
        tool_trace: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
        error: str = "",
    ) -> None:
        if self.run_store is None:
            return
        self.run_store.put(
            CareerAgentRun(
                run_id=artifact.artifact_id,
                session_id=request.session_id,
                goal=request.goal,
                status=status,
                elapsed_ms=max(0.0, float(elapsed_ms)),
                tool_call_count=tool_call_count,
                token_count=token_count,
                token_count_source=token_count_source,
                workspace_before_stage=str(workspace_before.get("recommended_stage") or ""),
                workspace_after_stage=str(workspace_state.get("recommended_stage") or ""),
                confirmation_target_count=confirmation_target_count,
                tool_trace=tuple(dict(item) for item in tool_trace if isinstance(item, dict)),
                error=error,
            )
        )

    @staticmethod
    def _system_prompt(request: CareerAgentRequest, workspace_state: dict[str, Any]) -> str:
        module = get_domain_module("career")
        return (
            f"{module.system_prompt}\n"
            "You are running one bounded single-agent Career turn. First use the supplied workspace state to "
            "choose the smallest valid action. Use Career tools for materials and evidence; do not write a fact "
            "as confirmed without explicit user confirmation. If confirmation or missing personal information is "
            "required, call ask_user and stop; confirmation and rejection are handled by the explicit user-decision "
            "API, not by this model turn. This stage is single-agent: do not delegate to child agents. Finish with "
            "a concise explanation of what changed, what is supported, and the next action. Every explicit, first-person, "
            "stable fact in the user message (including name, education/major, occupation, interests, and preferences) "
            "MUST be represented by a machine-readable <memory_ops>[...]</memory_ops> block, even without a 'remember' request. "
            "Each item must contain operation (assert/update), content, fact_key, confidence, and requires_confirmation. "
            "Use requires_confirmation=false for stable, explicit, high-confidence facts; use true for uncertain, sensitive, "
            "or consequential claims. "
            "The runtime validates and stores these proposals; never claim a fact is confirmed from the model output.\n"
            f"Goal: {request.goal}\n"
            "Preflight workspace state (authoritative for this turn):\n"
            + json.dumps(workspace_state, ensure_ascii=False, indent=2)
        )

    @staticmethod
    def _memory_layers(request: CareerAgentRequest, workspace_state: dict[str, Any]) -> dict[str, Any]:
        index = workspace_state.get("index") if isinstance(workspace_state.get("index"), dict) else {}
        return {
            "turn": {
                "session_id": request.session_id,
                "history_message_count": len(request.history),
            },
            "workspace": {
                "recommended_stage": workspace_state.get("recommended_stage", ""),
                "material_catalog_version": index.get("material_catalog_version", ""),
            },
            "personal_material": {
                "usable_evidence_count": int(index.get("usable_evidence_count", 0) or 0),
                "unverified_fact_count": int(index.get("unverified_fact_count", 0) or 0),
            },
            "project": {
                "source_count": int(index.get("project_source_count", 0) or 0),
                "needs_analysis": list(index.get("project_ids_needing_analysis", ()) or ()),
                "needs_confirmation": list(index.get("project_descriptions_needing_confirmation", ()) or ()),
            },
        }

    @staticmethod
    def _assistant_text(messages: list[dict[str, Any]]) -> str:
        return "\n".join(
            str(item.get("content") or "")
            for item in messages
            if item.get("role") == "assistant" and item.get("content")
        ).strip()

    @staticmethod
    def _confirmation_targets(state: dict[str, Any]) -> tuple[str, ...]:
        index = state.get("index") if isinstance(state.get("index"), dict) else {}
        targets = [
            *(str(item) for item in index.get("unhandled_candidate_fact_ids", ()) if str(item).strip()),
            *(str(item) for item in index.get("project_descriptions_needing_confirmation", ()) if str(item).strip()),
        ]
        return tuple(dict.fromkeys(targets))

    @staticmethod
    def _artifact(
        *,
        request: CareerAgentRequest,
        status: str,
        workspace_before: dict[str, Any],
        workspace_state: dict[str, Any],
        memory_layers: dict[str, Any],
        tool_calls: tuple[dict[str, Any], ...],
        awaiting_user: bool,
        confirmation_targets: tuple[str, ...],
        stop_reason: str,
        assistant_text: str,
        elapsed_seconds: float,
        error: str = "",
    ) -> AgentArtifact:
        return AgentArtifact.create(
            "career_turn_result",
            domain="career",
            session_id=request.session_id,
            created_by="career_agent",
            content={
                "request_id": request.session_id,
                "goal": request.goal,
                "status": status,
                "assistant_text": assistant_text,
                "workspace_before": workspace_before,
                "workspace_state": workspace_state,
                "memory_layers": memory_layers,
                "tool_calls": list(tool_calls),
                "awaiting_user": awaiting_user,
                "confirmation_targets": list(confirmation_targets),
                "stop_reason": stop_reason,
                "elapsed_seconds": round(elapsed_seconds, 6),
                "error": error,
            },
        )


__all__ = [
    "CAREER_AGENT_RUN_SCHEMA_VERSION",
    "CareerAgent",
    "CareerAgentRequest",
    "CareerAgentResult",
    "CareerAgentRun",
    "CareerAgentRunStore",
]
