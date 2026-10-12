"""Isolated child-agent execution with artifact-only context transfer."""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from smart_agent.agent_loop import run_agent_turn
from smart_agent.agents.contracts import AgentArtifact, AgentRunResult, AgentTask
from smart_agent.context_manager import ContextManager, token_count_with_estimation
from smart_agent.domains.runtime import DomainSession
from smart_agent.permissions import PermissionManager
from smart_agent.session import SessionData
from smart_agent.tooling import ToolContext, ToolRegistry

_COMPLETION_TOOL_RESULT_LIMIT = 4000
_COMPLETION_RETRY_LIMIT = 0


def _bounded_completion_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the final tool-call pass within a predictable context budget."""

    bounded: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        role = str(item.get("role") or "")
        if role == "tool_result":
            content = str(item.get("content") or "")
            if len(content) > _COMPLETION_TOOL_RESULT_LIMIT:
                half = _COMPLETION_TOOL_RESULT_LIMIT // 2
                item["content"] = (
                    content[:half]
                    + "\n...[completion context truncated; use cited paths already read]...\n"
                    + content[-half:]
                )
        elif role in {"assistant_progress", "assistant"}:
            content = str(item.get("content") or "")
            if len(content) > 2000:
                item["content"] = content[:2000] + "\n...[completion context truncated]..."
        bounded.append(item)
    return bounded


def _successful_tool_calls(
    messages: list[dict[str, Any]],
    tool_name: str,
) -> list[dict[str, Any]]:
    """Return successful calls to one completion tool from a model turn."""

    successful_call_ids = {
        str(message.get("toolUseId") or "")
        for message in messages
        if message.get("role") == "tool_result" and not message.get("isError", False)
    }
    return [
        message
        for message in messages
        if message.get("role") == "assistant_tool_call"
        and message.get("toolName") == tool_name
        and str(message.get("toolUseId") or "") in successful_call_ids
    ]


def _model_transport_failure(messages: list[dict[str, Any]]) -> str:
    """Return a provider/network failure surfaced as an assistant message."""

    markers = (
        "model api error",
        "model api timeout",
        "missing choices",
        "provider availability failure",
        "openai-compatible request failed",
        "read operation timed out",
        "network error",
        "connection failed or dropped",
        "connection refused",
        "connection reset",
        "connection reset",
        "connection refused",
        "temporarily unavailable",
        "upstream service unavailable",
    )
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if content and any(marker in content.casefold() for marker in markers):
            return content[:1000]
    return ""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract one JSON object from a plain-text compatibility response."""

    candidate = str(text or "").strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


@dataclass(frozen=True, slots=True)
class SubagentSpec:
    """Static policy for one child-agent role."""

    agent_type: str
    display_name: str
    domain: str
    system_prompt: str
    allowed_tools: frozenset[str]
    max_steps: int
    timeout_seconds: float
    read_only: bool = True
    completion_tool_name: str = ""
    completion_reserve_steps: int = 0
    # Keep an individual provider outage from consuming the whole child-task
    # deadline.  None preserves the process-wide adapter defaults.
    model_timeout_seconds: float | None = None
    model_max_retries: int | None = None


def _default_model_factory(runtime: dict[str, Any], tools: ToolRegistry):
    from smart_agent.model_registry import create_model_adapter

    return create_model_adapter(
        model=str(runtime.get("model") or ""),
        tools=tools,
        runtime=runtime,
    )


class SubagentRunner:
    """Run a child agent without copying or mutating the parent conversation."""

    def __init__(
        self,
        *,
        cwd: str,
        runtime: dict[str, Any],
        tools: ToolRegistry,
        permissions: Any | None = None,
        model_factory: Callable[[dict[str, Any], ToolRegistry], Any] = _default_model_factory,
        run_turn: Callable[..., list[dict[str, Any]]] = run_agent_turn,
    ) -> None:
        self.cwd = cwd
        self.runtime = dict(runtime)
        self.tools = tools
        self.permissions = permissions
        self.model_factory = model_factory
        self.run_turn = run_turn

    def _validate_boundary(
        self,
        task: AgentTask,
        artifacts: list[AgentArtifact],
        spec: SubagentSpec,
    ) -> None:
        if task.assigned_agent != spec.agent_type or task.task_type != spec.agent_type:
            raise ValueError("Task type does not match the selected child-agent policy.")
        if task.domain != spec.domain:
            raise ValueError("Task domain does not match the selected child-agent policy.")
        if frozenset(task.allowed_tools) != spec.allowed_tools:
            raise ValueError("Task tool allowlist does not match the child-agent policy.")
        if task.max_steps != spec.max_steps:
            raise ValueError("Task max_steps does not match the child-agent policy.")
        if task.timeout_seconds != spec.timeout_seconds:
            raise ValueError("Task timeout does not match the child-agent policy.")
        artifact_ids = [artifact.artifact_id for artifact in artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("Duplicate input artifacts are not allowed.")
        if set(artifact_ids) != set(task.input_artifact_ids):
            raise ValueError("Child agent received artifacts not declared by AgentTask.")
        if any(artifact.domain != task.domain for artifact in artifacts):
            raise ValueError("Cross-domain artifacts cannot enter a child-agent session.")
        for artifact in artifacts:
            artifact.verify_integrity()

    def _filtered_tools(self, spec: SubagentSpec) -> ToolRegistry:
        selected = [tool for tool in self.tools.list() if tool.name in spec.allowed_tools]
        selected_names = {tool.name for tool in selected}
        missing = spec.allowed_tools - selected_names
        if missing:
            raise ValueError(
                f"Child agent '{spec.agent_type}' is missing allowed tools: "
                + ", ".join(sorted(missing))
            )
        if spec.read_only:
            writable = [tool.name for tool in selected if not tool.is_read_only]
            if writable:
                raise ValueError(
                    f"Read-only child agent cannot receive writable tools: {', '.join(sorted(writable))}"
                )
        return ToolRegistry(selected)

    @staticmethod
    def _tool_trace(messages: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        """Build a safe, path-only trace for both success and failure results."""

        successful_call_ids = {
            str(message.get("toolUseId") or "")
            for message in messages
            if message.get("role") == "tool_result" and not message.get("isError", False)
        }
        return tuple(
            {
                "tool_name": str(message.get("toolName") or "").strip(),
                "ok": str(message.get("toolUseId") or "") in successful_call_ids,
                **(
                    {"path": str(message.get("input", {}).get("path") or "")}
                    if str(message.get("toolName") or "").strip()
                    in {"read_file", "read_project_document"}
                    and isinstance(message.get("input"), dict)
                    and str(message.get("input", {}).get("path") or "").strip()
                    else {}
                ),
            }
            for message in messages
            if message.get("role") == "assistant_tool_call"
            and str(message.get("toolName") or "").strip()
            and isinstance(message.get("input"), dict)
        )

    @staticmethod
    def _messages(
        task: AgentTask,
        artifacts: list[AgentArtifact],
        spec: SubagentSpec,
    ) -> list[dict[str, Any]]:
        payload = {
            "task": task.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in artifacts],
        }
        return [
            {
                "role": "system",
                "content": (
                    spec.system_prompt
                    + "\n\nYou are running in an isolated child session. "
                    "Use only the supplied artifacts and allowed tools. "
                    "Do not ask the user, modify parent state, or claim access to parent chat history."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ]

    def run(
        self,
        task: AgentTask,
        artifacts: list[AgentArtifact],
        spec: SubagentSpec,
    ) -> AgentRunResult:
        event_sink = self.runtime.get("_event_sink")

        def emit(event_type: str, **payload: Any) -> None:
            if not callable(event_sink):
                return
            try:
                event_sink({
                    "event_type": event_type,
                    "agent_type": spec.agent_type,
                    "agent_name": spec.display_name,
                    "task_id": task.task_id,
                    "parent_session_id": task.parent_session_id,
                    "child_session_id": task.child_session_id,
                    **payload,
                })
            except Exception:
                return

        self._validate_boundary(task, artifacts, spec)
        tools = self._filtered_tools(spec)
        now = time.time()
        child_session = SessionData(
            session_id=task.child_session_id,
            created_at=now,
            updated_at=now,
            workspace=self.cwd,
            domain_session=DomainSession.create(
                task.domain,
                session_id=task.child_session_id,
                created_at=now,
            ),
        )

        child_runtime = {
            **self.runtime,
            "activeDomain": task.domain,
            "domainStage": "default",
            "runtimeProfile": "harness",
            "isSubagent": True,
            "subagentType": spec.agent_type,
            "parentSessionId": task.parent_session_id,
            "childSessionId": task.child_session_id,
            "agentName": spec.display_name,
        }
        emit("agent_started", status="running")
        if spec.model_timeout_seconds is not None:
            child_runtime["modelTimeoutSeconds"] = spec.model_timeout_seconds
        if spec.model_max_retries is not None:
            child_runtime["modelMaxRetries"] = spec.model_max_retries
        setup_started = time.monotonic()
        try:
            model = self.model_factory(child_runtime, tools)
            child_permissions = (
                PermissionManager(self.cwd, prompt=None)
                if spec.read_only
                else PermissionManager(
                    self.cwd,
                    prompt=getattr(self.permissions, "prompt", None),
                )
            )
        except Exception as exc:  # noqa: BLE001
            emit("agent_failed", status="failed", output_summary=f"{type(exc).__name__}: {exc}")
            return AgentRunResult(
                task=task,
                status="failed",
                elapsed_seconds=time.monotonic() - setup_started,
                error=f"{type(exc).__name__}: {exc}",
            )
        messages = self._messages(task, artifacts, spec)
        # Every child gets an isolated short-term context.  It tracks the
        # complete tool transcript and enables the shared compaction policy
        # without inheriting the parent conversation.
        child_context = ContextManager(
            model=str(child_runtime.get("model") or self.runtime.get("model") or "default"),
            messages=list(messages),
        )
        result_queue: queue.Queue[tuple[str, Any, list[dict[str, Any]]]] = queue.Queue(maxsize=1)

        def _invoke() -> None:
            result_messages = list(messages)
            try:
                exploration_steps = spec.max_steps
                if spec.completion_tool_name:
                    exploration_steps -= spec.completion_reserve_steps
                result_messages = self.run_turn(
                    model=model,
                    # The final structured-completion schema is deliberately
                    # absent while exploring. This keeps the prompt small and
                    # prevents relays from selecting the completion tool early.
                    tools=(
                        ToolRegistry(
                            [
                                tool
                                for tool in tools.list()
                                if tool.name != spec.completion_tool_name
                            ]
                        )
                        if spec.completion_tool_name
                        else tools
                    ),
                    messages=messages,
                    cwd=self.cwd,
                    permissions=child_permissions,
                    session=child_session,
                    context_manager=child_context,
                    max_steps=exploration_steps,
                    runtime=child_runtime,
                    enable_work_chain=False,
                )
                transport_failure = _model_transport_failure(result_messages)
                if transport_failure:
                    raise RuntimeError(transport_failure)
                completion_calls = _successful_tool_calls(result_messages, spec.completion_tool_name)
                if spec.completion_tool_name and not completion_calls:
                    completion_tool = tools.find(spec.completion_tool_name)
                    if completion_tool is None:
                        raise RuntimeError(f"{spec.display_name} completion tool is unavailable.")
                    completion_tools = ToolRegistry([completion_tool])
                    completion_runtime = {
                        **child_runtime,
                        "runtimeProfile": "harness",
                        # The completion pass has exactly one valid action.
                        # Adapters translate this into their native tool-choice
                        # parameter so a model cannot end with prose instead.
                        "forceToolName": spec.completion_tool_name,
                    }
                    completion_messages = _bounded_completion_messages(result_messages)
                    completion_error = "the previous completion pass returned no successful tool call"
                    for attempt in range(_COMPLETION_RETRY_LIMIT + 1):
                        if completion_messages and completion_messages[-1].get("role") == "assistant":
                            completion_messages.pop()
                        if attempt == 0:
                            instruction = (
                                "The exploration phase is over. Based only on evidence already gathered, call "
                                f"{spec.completion_tool_name} now. No prose response is valid. "
                                "Do not claim unsupported facts. If this endpoint cannot emit a tool call, return only "
                                "the JSON object that would be passed to the tool."
                            )
                        else:
                            instruction = (
                                f"Completion attempt {attempt} failed: {completion_error}. "
                                f"Call {spec.completion_tool_name} exactly once now. No prose is valid. "
                                "If tool calls are unavailable, return only the complete JSON object matching the tool schema."
                            )
                        completion_messages.append({"role": "user", "content": instruction})
                        completion_model = self.model_factory(completion_runtime, completion_tools)
                        candidate_messages = self.run_turn(
                            model=completion_model,
                            tools=completion_tools,
                            messages=completion_messages,
                            cwd=self.cwd,
                            permissions=child_permissions,
                            session=child_session,
                            context_manager=child_context,
                            max_steps=spec.completion_reserve_steps,
                            runtime=completion_runtime,
                            enable_work_chain=False,
                        )
                        transport_failure = _model_transport_failure(candidate_messages)
                        if transport_failure:
                            # A provider failure cannot be repaired by another
                            # completion prompt. Surface it once so the Career
                            # layer can apply its explicit offline fallback.
                            raise RuntimeError(transport_failure)
                        completion_calls = _successful_tool_calls(candidate_messages, spec.completion_tool_name)
                        result_messages = candidate_messages
                        if completion_calls:
                            break
                        completion_error = (
                            next(
                                (
                                    str(message.get("content") or "").strip()
                                    for message in reversed(candidate_messages)
                                    if message.get("role") == "tool_result"
                                    and message.get("isError", False)
                                    and str(message.get("content") or "").strip()
                                ),
                                next(
                                    (
                                        str(message.get("content") or "").strip()
                                        for message in reversed(candidate_messages)
                                        if message.get("role") == "assistant"
                                        and str(message.get("content") or "").strip()
                                    ),
                                    completion_error,
                                ),
                            )
                        )[:500]
                        completion_messages = _bounded_completion_messages(candidate_messages)
                    else:
                        # Some OpenAI-compatible relays silently strip tool calls.
                        # If the model still returned a complete JSON object, execute
                        # the same validated tool locally rather than discarding the analysis.
                        plain_payload = next(
                            (
                                _extract_json_object(str(message.get("content") or ""))
                                for message in reversed(result_messages)
                                if message.get("role") == "assistant"
                                and _extract_json_object(str(message.get("content") or "")) is not None
                            ),
                            None,
                        )
                        if plain_payload is not None:
                            compat_id = f"compat-{uuid.uuid4().hex}"
                            compat_context = ToolContext(
                                cwd=self.cwd,
                                permissions=child_permissions,
                                session=child_session,
                                _runtime=completion_runtime,
                            )
                            compat_result = completion_tools.execute(
                                spec.completion_tool_name,
                                plain_payload,
                                compat_context,
                            )
                            if compat_result.ok:
                                result_messages = [
                                    *result_messages,
                                    {
                                        "role": "assistant_tool_call",
                                        "toolUseId": compat_id,
                                        "toolName": spec.completion_tool_name,
                                        "input": plain_payload,
                                        "compatibilityFallback": True,
                                    },
                                    {
                                        "role": "tool_result",
                                        "toolUseId": compat_id,
                                        "toolName": spec.completion_tool_name,
                                        "content": compat_result.output,
                                        "isError": False,
                                    },
                                    {
                                        "role": "assistant",
                                        "content": "Project analysis submitted.",
                                    },
                                ]
                                completion_calls = _successful_tool_calls(
                                    result_messages, spec.completion_tool_name
                                )
                        if not completion_calls:
                            raise RuntimeError(
                                f"{spec.display_name} completion failed after "
                                f"{_COMPLETION_RETRY_LIMIT + 1} attempts: {completion_error}"
                            )
                result_queue.put(("ok", result_messages, result_messages))
            except BaseException as exc:  # noqa: BLE001
                # Preserve the exploration transcript so callers can diagnose
                # a failed completion or provider outage without re-reading
                # the project and without exposing raw tool output.
                result_queue.put(("error", exc, result_messages))

        started = time.monotonic()
        worker = threading.Thread(
            target=_invoke,
            name=f"subagent-{task.task_id}",
            daemon=True,
        )
        worker.start()
        worker.join(spec.timeout_seconds)
        elapsed = time.monotonic() - started

        if worker.is_alive():
            emit("agent_timed_out", status="timed_out", output_summary=f"{spec.display_name} timed out after {spec.timeout_seconds:g}s.", elapsed_ms=elapsed * 1000)
            return AgentRunResult(
                task=task,
                status="timed_out",
                elapsed_seconds=elapsed,
                error=f"{spec.display_name} timed out after {spec.timeout_seconds:g}s.",
            )

        try:
            state, value, partial_messages = result_queue.get_nowait()
        except queue.Empty:
            emit("agent_failed", status="failed", output_summary=f"{spec.display_name} ended without returning a result.", elapsed_ms=elapsed * 1000)
            return AgentRunResult(
                task=task,
                status="failed",
                elapsed_seconds=elapsed,
                error=f"{spec.display_name} ended without returning a result.",
            )
        if state == "error":
            emit("agent_failed", status="failed", output_summary=f"{type(value).__name__}: {value}", elapsed_ms=elapsed * 1000)
            token_usage = token_count_with_estimation(partial_messages)
            return AgentRunResult(
                task=task,
                status="failed",
                elapsed_seconds=elapsed,
                tool_call_count=sum(
                    message.get("role") == "assistant_tool_call" for message in partial_messages
                ),
                token_count=int(token_usage["total_tokens"]),
                token_count_source=str(token_usage["source"]),
                tool_trace=self._tool_trace(partial_messages),
                error=f"{type(value).__name__}: {value}",
            )

        result_messages = value
        child_session.messages = list(result_messages)
        token_usage = token_count_with_estimation(result_messages)
        tool_call_count = sum(
            message.get("role") == "assistant_tool_call" for message in result_messages
        )
        final_text = next(
            (
                str(message.get("content") or "").strip()
                for message in reversed(result_messages)
                if message.get("role") == "assistant" and str(message.get("content") or "").strip()
            ),
            "",
        )
        if not final_text:
            emit("agent_failed", status="failed", output_summary=f"{spec.display_name} returned no final assistant message.", elapsed_ms=elapsed * 1000)
            return AgentRunResult(
                task=task,
                status="failed",
                elapsed_seconds=elapsed,
                tool_call_count=tool_call_count,
                token_count=int(token_usage["total_tokens"]),
                token_count_source=str(token_usage["source"]),
                error=f"{spec.display_name} returned no final assistant message.",
            )

        successful_call_ids = {
            str(message.get("toolUseId") or "")
            for message in result_messages
            if message.get("role") == "tool_result" and not message.get("isError", False)
        }
        tool_trace = [
            {
                "tool_name": str(message.get("toolName") or "").strip(),
                "input": dict(message.get("input") or {}),
                "ok": str(message.get("toolUseId") or "") in successful_call_ids,
            }
            for message in result_messages
            if message.get("role") == "assistant_tool_call"
            and str(message.get("toolName") or "").strip()
            and isinstance(message.get("input"), dict)
        ]
        safe_trace = tuple(
            {
                "tool_name": item["tool_name"],
                "ok": bool(item["ok"]),
                **(
                    {"path": str(item["input"].get("path") or "")}
                    if item["tool_name"] in {"read_file", "read_project_document"}
                    and str(item["input"].get("path") or "").strip()
                    else {}
                ),
            }
            for item in tool_trace
        )

        output = AgentArtifact.create(
            f"{spec.agent_type}_result",
            domain=task.domain,
            session_id=task.child_session_id,
            created_by=spec.agent_type,
            content={"text": final_text, "tool_trace": tool_trace},
            source_artifact_ids=task.input_artifact_ids,
        )
        emit("agent_finished", status="completed", output_summary=final_text[:500], elapsed_ms=elapsed * 1000, tool_call_count=tool_call_count, token_count=int(token_usage["total_tokens"]))
        return AgentRunResult(
            task=task,
            status="completed",
            output_artifact=output,
            elapsed_seconds=elapsed,
            tool_call_count=tool_call_count,
            token_count=int(token_usage["total_tokens"]),
            token_count_source=str(token_usage["source"]),
            tool_trace=safe_trace,
        )
