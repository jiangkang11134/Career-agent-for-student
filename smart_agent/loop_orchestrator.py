"""Agent 循环编排。

``run_agent_turn`` uses one ReAct loop: the model chooses an answer or tool
call, tool observations are appended, and the model chooses again.  Setup,
context safety, metrics, and persistence are orthogonal lifecycle concerns;
they do not gate the model with explore/execute/verify phases.

辅助函数：
  _upsert_stable_task_state_message — 消息列表中替换"稳定任务状态"
  _is_empty_assistant_response — 判断模型是否返回空内容
  _extract_task_description — 从消息中提取用户原始请求
  _build_work_chain_task — 构建 TaskObject
  _format_diagnostics — 格式化诊断信息
  _is_recoverable_thinking_stop — 判断思考中断是否可恢复
  _should_treat_assistant_as_progress — 判断模型输出是进度还是最终答案
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import time
import uuid
from collections.abc import Callable
from typing import Any

from smart_agent.agent_intelligence import ErrorClassifier, NudgeGenerator, ToolScheduler

# ── Agent 智能辅助 ──
from smart_agent.agent_metrics import AgentMetricsCollector
from smart_agent.circuit_breaker import CompactionCircuitBreaker

# ── 上下文管理 ──
from smart_agent.context_manager import ContextManager, estimate_message_tokens
from smart_agent.cybernetic_orchestrator import CyberneticOrchestrator
from smart_agent.decision_audit import DecisionOutcome, get_auditor
from smart_agent.hooks import HookEvent, fire_hook_sync

# ── 任务系统 ──
from smart_agent.intent_parser import parse_intent
from smart_agent.logging_config import get_logger

# ── 子模块（拆分目标） ──
from smart_agent.model_caller import (
    _infer_active_model_id,
    _is_at_blocking_limit,
    _looks_like_provider_availability_error,
    _model_next,
    _summarize_model_api_failure,
)
from smart_agent.model_errors import ContextWindowExceeded
from smart_agent.permissions import PermissionManager
from smart_agent.runtime_profiles import resolve_code_run_policy
from smart_agent.state import AppState, Store
from smart_agent.task_graph import TaskGraph
from smart_agent.task_graph import TaskState as GraphTaskState
from smart_agent.task_object import TaskObject, TaskState, build_task
from smart_agent.tool_executor import (
    _execute_single_tool,
    _register_tool_capabilities,
    get_active_review_hooks,
    init_review_hooks,
)
from smart_agent.tool_result_budget import ToolResultBudgetManager
from smart_agent.tooling import ToolContext, ToolRegistry, ToolResult

# ── 回合状态机 ──
from smart_agent.turn_kernel import (
    TurnLifecycleState,
    TurnRecurrentState,
    TurnTaskContract,
    TurnVerificationState,
    _decide_react_assistant_turn,
    build_turn_coda_summary,
    finalize_work_chain_task,
)
from smart_agent.types import (
    AgentStep,
    ChatMessage,
    ModelAdapter,
    RuntimeEvent,
)
from smart_agent.verification_guard import VerificationLedger
from smart_agent.working_memory import begin_turn, end_turn

logger = get_logger("loop_orchestrator")

_STRONG_VERIFY_TOOLS = frozenset({
    "diff_viewer", "code_review", "test_runner", "sandbox_test", "run_command", "git",
})

_MUTATING_ACTIONS = frozenset({
    "create", "update", "delete", "move", "rename", "merge", "split",
})
_EXECUTION_ACTIONS = frozenset({"execute"})
_MUTATION_TEXT_MARKERS = (
    "\u521b\u5efa", "\u5b9e\u73b0", "\u4fee\u6539", "\u7f16\u8f91", "\u4fee\u590d",
    "\u8c03\u8bd5", "\u91cd\u6784", "\u4f18\u5316", "\u5220\u9664", "\u91cd\u547d\u540d", "\u79fb\u52a8",
    "write", "create", "implement", "add", "generate", "modify", "update",
    "change", "edit", "fix", "refactor", "optimize", "delete", "remove",
    "rename", "move", "commit", "鍒涘缓", "瀹炵幇", "淇敼", "缂栬緫",
    "淇", "璋冭瘯", "閲嶆瀯", "浼樺寲", "鍒犻櫎", "閲嶅懡鍚?", "绉诲姩",
)
_EXECUTION_TEXT_MARKERS = (
    "\u8fd0\u884c", "\u6d4b\u8bd5", "\u6267\u884c",
    "run", "execute", "test", "杩愯", "娴嬭瘯", "鎵ц",
)


def _infer_task_contract(
    messages: list[dict[str, Any]],
    runtime: dict[str, Any],
    active_domain: str,
) -> TurnTaskContract:
    """Infer obligations conservatively; side effects can escalate the mode.

    Ambiguous text starts in a permissive ReAct mode.  The contract becomes
    strict when intent metadata clearly requests a mutation or when a
    side-effecting tool is actually used.
    """
    if active_domain != "code":
        return TurnTaskContract(mode="conversation", source="non_code_domain")

    preflight = runtime.get("preflight")
    if isinstance(preflight, dict):
        intent = str(preflight.get("intent") or "").strip().lower()
        if intent in {"general", "memory"}:
            return TurnTaskContract(mode="conversation", source="preflight")
        if intent == "project_info":
            return TurnTaskContract(
                mode="read_only", source="preflight", evidence_required=True
            )

    user_text = ""
    for message in reversed(messages):
        if message.get("role") == "user" and str(message.get("content") or "").strip():
            user_text = str(message.get("content") or "").strip()
            break
    parsed = parse_intent(user_text) if user_text else None
    action_type = str(getattr(parsed, "action_type", "") or "").lower()
    if hasattr(getattr(parsed, "action_type", None), "value"):
        action_type = str(parsed.action_type.value).lower()
    if action_type in _MUTATING_ACTIONS:
        return TurnTaskContract(
            mode="mutation", source="intent", requires_execute=True,
            requires_verify=True, evidence_required=True,
        )
    if action_type in _EXECUTION_ACTIONS:
        return TurnTaskContract(
            mode="execution", source="intent", requires_execute=True,
            requires_verify=True, evidence_required=True,
        )
    normalized_text = user_text.lower()
    if any(marker in normalized_text for marker in _EXECUTION_TEXT_MARKERS):
        return TurnTaskContract(
            mode="execution", source="text", requires_execute=True,
            requires_verify=True, evidence_required=True,
        )
    if any(marker in normalized_text for marker in _MUTATION_TEXT_MARKERS):
        return TurnTaskContract(
            mode="mutation", source="text", requires_execute=True,
            requires_verify=True, evidence_required=True,
        )
    if action_type in {"read", "analyze", "compare"}:
        return TurnTaskContract(mode="read_only", source="intent")

    # Unknown requests must be answerable.  A write tool, if selected by the
    # model, will escalate this contract through TurnRecurrentState.
    return TurnTaskContract(mode="conversation", source="ambiguous")


def _contract_tool_violation(*, tool: Any, contract: TurnTaskContract) -> str:
    """Compatibility hook retained for callers; ReAct imposes no workflow gate."""
    return ""


def _tool_action_kinds(
    call: dict[str, Any],
    tool: Any,
    contract: TurnTaskContract,
) -> tuple[bool, bool]:
    """Return ``(is_execution, is_verification)`` for one successful call."""
    name = str(call.get("toolName") or "").strip().lower()
    if tool is None or name == "complete_task":
        return False, False
    verification_only = {"diff_viewer", "code_review", "test_runner", "sandbox_test"}
    if name in verification_only:
        # A requested test run is both the execution and its own verification.
        return contract.mode == "execution", True
    if bool(getattr(tool, "is_read_only", False)):
        return False, bool(contract.mode == "read_only" and contract.evidence_required)
    if name == "git":
        action = str((call.get("input") or {}).get("action") or "").strip().lower()
        if action in {"status", "diff", "log", "review"}:
            return False, True
        return True, False
    if name == "run_command":
        command = str((call.get("input") or {}).get("command") or "").lower()
        verifies = any(
            marker in command
            for marker in ("pytest", "unittest", "test", "check", "lint", "typecheck")
        )
        return True, verifies
    return not bool(getattr(tool, "is_read_only", False)), False


def _tool_signature(call: dict[str, Any]) -> str:
    return json.dumps(
        {"tool": call.get("toolName", ""), "input": call.get("input", {})},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _context_summary_provider(
    model: ModelAdapter,
    store: Store[AppState] | None,
) -> Callable[[str, list[dict[str, Any]]], Any]:
    """Build the structured LLM summarizer used only by L3 and L4."""

    def summarize(mode: str, payload: list[dict[str, Any]]) -> Any:
        if mode == "l3_lock":
            instruction = (
                "Summarize the supplied terminal user turns as one immutable history lock. "
                "For every turn include User, Tool Call, Tool Result, Assistant, and explicitly "
                "protected content. Do not add unresolved-task lists or progress summaries."
            )
        else:
            instruction = (
                "Return only a JSON array with exactly the same number, order, roles, and tool "
                "call identifiers as the supplied current-turn messages. Summarize content and "
                "tool inputs, but do not add, remove, reorder, or change message roles."
            )
        request = [
            {
                "role": "system",
                "content": (
                    "You are a context compression model. Preserve established facts and explicit "
                    "requirements. Never invent facts. " + instruction
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            },
        ]
        step = _model_next(
            model,
            request,
            on_stream_chunk=None,
            on_thinking_chunk=None,
            store=store,
        )
        if step.type != "assistant" or not isinstance(step.content, str):
            raise ValueError("context summary model did not return assistant text")
        content = step.content.strip()
        if mode == "l3_lock":
            return content
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1]).strip()
            if content.startswith("json"):
                content = content[4:].lstrip()
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            raise ValueError("current-turn summary must be a JSON array")
        return parsed

    return summarize

# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════

NUDGE_CONTINUE = (
    "Continue immediately from your <progress> update with concrete tool calls, "
    "code changes, or an explicit <final> answer only if the task is complete. "
    "Prefer taking the next concrete action over explaining what you plan to do."
)

NUDGE_AFTER_TOOL_RESULT = (
    "You have received tool results. Review them briefly, then take the next "
    "concrete action: call another tool, edit code, or give an explicit <final> "
    "answer only if the task is truly complete. Do not restate what you just saw."
)

NUDGE_AFTER_EMPTY_RESPONSE = (
    "Your last response was empty. This often happens after tool errors or when "
    "the model is uncertain. Pick the most likely next action and try it — you can "
    "adjust based on results. Call a tool, edit code, or give <final> if done."
)

NUDGE_AFTER_EMPTY_NO_TOOLS = (
    "Your last response was empty but you have not used any tools yet. Start by "
    "inspecting the relevant files (read_file, grep_files, list_files) to understand "
    "the codebase before making changes."
)

RESUME_AFTER_PAUSE = (
    "Resume from the previous pause. Continue with the next concrete tool call, "
    "code change, or <final> answer."
)

RESUME_AFTER_MAX_TOKENS = (
    "Your previous response was cut short by the token limit. Resume immediately "
    "with the next concrete action — pick up where you left off."
)

STABLE_TASK_STATE_MARKER = "[Stable task state]"


# ══════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════

def _upsert_stable_task_state_message(
    messages: list[ChatMessage],
    stable_text: str,
) -> list[ChatMessage]:
    """替换"稳定任务状态"消息，避免旧状态无限堆积浪费 token。

    【为什么需要】每步迭代后任务状态会变化，如果每次都 append 新状态，
    旧状态会无限堆积。这个函数用"替换"代替"追加"，同一时间只有一条最新的状态消息。

    实现：删除所有以 [Stable task state] 开头的 system 消息，然后把新的追加进去。

    参数:
        messages: 当前对话消息列表
        stable_text: 当前稳定的任务状态文本

    返回:
        更新后的消息列表（只有一条稳定状态消息）
    """
    filtered = [
        m for m in messages
        if not (m.get("role") == "system" and str(m.get("content", "")).startswith(STABLE_TASK_STATE_MARKER))
    ]
    filtered.append({"role": "system", "content": f"{STABLE_TASK_STATE_MARKER}\n{stable_text}"})
    return filtered


def _is_empty_assistant_response(content: str) -> bool:
    """判断模型返回的内容是否为空（纯空白字符也算空）。

    【为什么需要】模型返回空响应通常意味着卡住了或不确定。
    后面 decide_assistant_turn 会根据这个判断是重试还是给 nudge 提示。

    参数:
        content: 模型返回的内容字符串

    返回:
        True=内容是空的
    """
    return len(content.strip()) == 0


def _extract_task_description(messages: list[ChatMessage]) -> str:
    """从消息列表中找到用户原始的请求。

    【为什么需要】消息列表里混着系统提示、nudge 提示语（"Continue from your progress..."）、
    历史对话等各种内容。原始任务描述只有一条（用户第一次提出的请求），后面的都是辅助指令。
    这个函数把真正的那条任务描述捞出来用于构建 TaskObject。

    实现：遍历消息，找第一条 role=user 且不以 "Continue" 或 "Your last" 开头的消息。

    参数:
        messages: 对话消息列表

    返回:
        任务描述文本（前 500 字符），没找到则返回 "Unknown task"
    """
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            content = str(msg["content"])
            if not content.startswith("Continue") and not content.startswith("Your last"):
                return content[:500]
    return "Unknown task"


def _build_work_chain_task(messages: list[ChatMessage]) -> tuple[TaskObject | None, dict]:
    """构建工作链任务对象。

    【为什么需要】agent 不能只在"字符串层面"理解任务。把用户的自然语言输入转成
    结构化的 TaskObject（含意图类型、动作类型、复杂度等），后续路由、进度跟踪、
    记忆反馈才有结构化数据可用。

    流程：_extract_task_description() → parse_intent() → build_task()

    参数:
        messages: 对话消息列表

    返回:
        (TaskObject, metadata) 元组，如果无法识别任务则返回 (None, {})
    """
    raw_input = _extract_task_description(messages)
    if raw_input == "Unknown task":
        return None, {}
    intent = parse_intent(raw_input)
    task = build_task(intent, raw_input)
    metadata = {
        "intent_type": intent.intent_type.value,
        "action_type": intent.action_type.value,
        "confidence": intent.confidence,
        "entities": intent.entities,
        "complexity": intent.complexity_hint,
    }
    return task, metadata


def _format_diagnostics(stop_reason, block_types, ignored_block_types) -> str:
    """格式化诊断信息字符串。

    【为什么需要】decide_assistant_turn 需要把模型的停止原因和阻塞信息拼到 nudge 提示里。
    这个函数统一格式，生成类似 "stop_reason=max_tokens; blocks=thinking" 的文本。

    参数:
        stop_reason: 停止原因（如 done, max_tokens, pause_turn）
        block_types: 阻塞类型列表（如 thinking, tool_use）
        ignored_block_types: 被忽略的阻塞类型列表

    返回:
        格式化的诊断字符串，无内容则返回空字符串
    """
    parts: list[str] = []
    if stop_reason:
        parts.append(f"stop_reason={stop_reason}")
    if block_types:
        parts.append(f"blocks={','.join(block_types)}")
    if ignored_block_types:
        parts.append(f"ignored={','.join(ignored_block_types)}")
    return f" Diagnostics: {'; '.join(parts)}." if parts else ""


def _is_recoverable_thinking_stop(*, is_empty, stop_reason, ignored_block_types) -> bool:
    """判断空响应是否是"思考被截断"（可恢复）还是"模型不会答"（需换策略）。

    【为什么需要】模型有时因为思考过程被截断或用户打断了而返回空内容。
    这不同于"模型不知道该怎么回答"的空响应。前者只需要告诉模型"接着想"，
    后者需要换策略。区分两者避免在可恢复的情况下浪费一次重试。

    参数:
        is_empty: 响应是否为空
        stop_reason: 停止原因
        ignored_block_types: 被忽略的阻塞类型列表

    返回:
        True=是可恢复的思考中断，可以重试
    """
    if not is_empty:
        return False
    if stop_reason not in {"pause_turn", "max_tokens"}:
        return False
    return "thinking" in (ignored_block_types or [])


def _should_treat_assistant_as_progress(*, kind, content, saw_tool_result) -> bool:
    """判断模型输出是进度消息还是最终答案。

    【为什么需要】模型有时会先输出一段"我正在分析..."的中间思考再回答问题。
    如果不区分进度消息和最终答案，agent 可能把中间思考误认为最终结论而过早停止工具调用。

    参数:
        kind: 响应类型（progress/final/None）
        content: 响应内容（未使用，保留为接口一致）
        saw_tool_result: 是否已看到工具执行结果

    返回:
        True=应视为进度消息，False=应视为最终答案
    """
    if kind == "progress":
        return True
    if kind == "final":
        return False
    if not saw_tool_result:
        return False
    return False


def _model_for_final_synthesis(model: ModelAdapter) -> ModelAdapter:
    """Return a shallow model clone with tool calls disabled for final recovery."""
    try:
        synthesis_model = copy.copy(model)
        if hasattr(synthesis_model, "tools"):
            synthesis_model.tools = ToolRegistry([])
        runtime_config = getattr(synthesis_model, "runtime", None)
        if isinstance(runtime_config, dict):
            synthesis_model.runtime = {**runtime_config, "disableTools": True}
        return synthesis_model
    except Exception:
        return model


def _synthesize_after_tool_budget(
    model: ModelAdapter,
    messages: list[ChatMessage],
    *,
    store: Store[AppState] | None,
) -> AgentStep | None:
    """Ask for a text-only handoff after tool budget exhaustion."""
    synthesis_messages = list(messages)
    synthesis_messages.append({
        "role": "user",
        "content": (
            "Tool-call budget reached. Stop using tools and provide the best final answer now. "
            "Summarize only what the tool results establish, state any remaining uncertainty, "
            "and do not describe this internal budget instruction."
        ),
    })
    try:
        step = _model_next(
            _model_for_final_synthesis(model),
            synthesis_messages,
            on_stream_chunk=None,
            on_thinking_chunk=None,
            store=store,
        )
    except Exception:  # noqa: BLE001 - budget recovery must never crash the turn
        logger.debug("Text-only synthesis after tool budget failed", exc_info=True)
        return None
    if step.type != "assistant" or not str(step.content or "").strip():
        return None
    return step


# ══════════════════════════════════════════════════════════════
# 核心
# ══════════════════════════════════════════════════════════════

def run_agent_turn(
    *,
    model: ModelAdapter,
    tools: ToolRegistry,
    messages: list[ChatMessage],
    cwd: str,
    permissions: PermissionManager | None = None,
    session: Any | None = None,
    store: Store[AppState] | None = None,
    max_steps: int = 50,
    on_tool_start: Callable[[str, dict], None] | None = None,
    on_tool_result: Callable[[str, str, bool], None] | None = None,
    on_assistant_message: Callable[[str], None] | None = None,
    on_progress_message: Callable[[str], None] | None = None,
    on_runtime_event: Callable[[RuntimeEvent], None] | None = None,
    on_assistant_stream_chunk: Callable[[str], None] | None = None,
    on_thinking_chunk: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    context_manager: ContextManager | None = None,
    runtime: dict | None = None,
    metrics_collector: AgentMetricsCollector | None = None,
    system_prompt: str = "",
    project_context: str = "",
    enable_work_chain: bool = True,
) -> list[ChatMessage]:
    """运行单轮 agent 交互循环。

    每次迭代都由模型在当前证据上选择下一步：直接回答、报告进度，或调用一个/多个工具。
    工具结果追加回同一消息面，再交给模型继续决策。上下文压缩、预算、指标和持久化
    只负责运行安全，不会强制任何工作流阶段。

    异常处理：ConnectionError/TimeoutError/其他→错误消息返回，不崩溃。
    """
    # ════════════════════════════════════════════════════════════
    # Prepare the conversation surface and runtime resources.
    # ════════════════════════════════════════════════════════════

    current_messages = list(messages)
    # ``system_prompt`` and ``project_context`` are request-scoped instructions,
    # so they must be present on the model surface, not only in token accounting.
    # Merge them into an existing system message when a caller already supplied
    # one; this keeps a single stable system prefix across compaction cycles.
    request_context = "\n\n".join(
        str(part).strip()
        for part in (system_prompt, project_context)
        if str(part or "").strip()
    )
    if request_context:
        system_index = next(
            (index for index, message in enumerate(current_messages)
             if message.get("role") == "system"),
            None,
        )
        if system_index is None:
            current_messages.insert(
                0,
                {
                    "role": "system",
                    "content": request_context,
                    "_request_context_content": request_context,
                },
            )
        else:
            system_message = current_messages[system_index]
            existing = str(system_message.get("content") or "").strip()
            previous_context = str(system_message.get("_request_context_content") or "").strip()
            if previous_context and previous_context in existing:
                existing = existing.replace(previous_context, "", 1).strip()
            current_messages[system_index] = {
                **system_message,
                "content": "\n\n".join(part for part in (existing, request_context) if part),
                "_request_context_content": request_context,
            }
    runtime = runtime or {}
    # Optional operational observer used by the web admin. It is deliberately
    # best-effort so monitoring cannot change the agent's behavior.
    external_event_sink = runtime.get("_event_sink")
    supplied_event_sink = callable(external_event_sink)
    if not callable(external_event_sink):
        try:
            from smart_agent.agent_events import agent_event_hub

            def external_event_sink(event: dict[str, Any]) -> None:
                event_type = str(event.get("event_type") or "agent_progress")
                fields = {
                    "agent_type": str(runtime.get("subagentType") or runtime.get("activeDomain") or "agent"),
                    "agent_name": str(runtime.get("agentName") or runtime.get("subagentType") or runtime.get("activeDomain") or "Agent"),
                    "session_id": str(runtime.get("sessionId") or runtime.get("session_id") or ""),
                    "parent_session_id": str(runtime.get("parentSessionId") or ""),
                    "child_session_id": str(runtime.get("childSessionId") or ""),
                }
                fields.update({key: value for key, value in event.items() if key != "event_type"})
                agent_event_hub.publish(event_type, **fields)
        except Exception:
            external_event_sink = None

    def _safe_event_summary(value: Any, limit: int = 500) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
        except Exception:
            text = str(value)
        return text[:limit]

    def emit_external(event_type: str, **payload: Any) -> None:
        if not callable(external_event_sink):
            return
        try:
            external_event_sink({"event_type": event_type, **payload})
        except Exception:  # pragma: no cover - observers are non-critical
            logger.debug("External agent event sink failed", exc_info=True)

    def observed_tool_start(tool_name: str, tool_input: dict) -> None:
        emit_external(
            "tool_started",
            phase="tool",
            tool_name=str(tool_name),
            status="running",
            input_summary=_safe_event_summary(tool_input),
        )
        if on_tool_start:
            on_tool_start(tool_name, tool_input)

    def observed_tool_result(tool_name: str, output: str, is_error: bool) -> None:
        emit_external(
            "tool_finished",
            phase="tool",
            tool_name=str(tool_name),
            status="failed" if is_error else "completed",
            output_summary=_safe_event_summary(output),
        )
        if on_tool_result:
            on_tool_result(tool_name, output, is_error)

    if not supplied_event_sink:
        emit_external(
            "agent_started",
            agent_type=str(runtime.get("subagentType") or runtime.get("activeDomain") or "agent"),
            agent_name=str(runtime.get("agentName") or runtime.get("subagentType") or runtime.get("activeDomain") or "Agent"),
            session_id=str(runtime.get("sessionId") or runtime.get("session_id") or ""),
            parent_session_id=str(runtime.get("parentSessionId") or ""),
            child_session_id=str(runtime.get("childSessionId") or ""),
            status="running",
        )

    verification_ledger = VerificationLedger()
    runtime["_verificationLedger"] = verification_ledger

    # The optional Root Supervisor admits only non-simple work in auto mode.
    # Child-agent turns always stay on the existing Domain loop.
    if not runtime.get("isSubagent"):
        from smart_agent.agents.orchestrator import decide_supervision, run_orchestrated_turn

        supervisor_goal = str(runtime.get("originalUserInput") or "").strip()
        if not supervisor_goal:
            supervisor_goal = next(
                (
                    str(message.get("content") or "").strip()
                    for message in reversed(current_messages)
                    if message.get("role") == "user" and str(message.get("content") or "").strip()
                ),
                "Continue the selected Domain task.",
            )
        supervisor_mode = runtime.get("supervisorMode", runtime.get("orchestratorMode", "off"))
        supervisor_decision = decide_supervision(supervisor_goal, supervisor_mode)
        runtime["supervisorAdmission"] = supervisor_decision.to_dict()
        if supervisor_decision.should_supervise:
            return run_orchestrated_turn(
                model=model,
                tools=tools,
                messages=current_messages,
                cwd=cwd,
                permissions=permissions,
                session=session,
                runtime=runtime,
                on_progress_message=on_progress_message,
            )

    # 确定模型名
    configured_runtime_model = (
        str(runtime.get("configuredModel", "")).strip()
        or str(runtime.get("model", "")).strip()
        or str(getattr(model, "model_id", "") or "").strip()
    )
    if configured_runtime_model:
        runtime.setdefault("configuredModel", configured_runtime_model)

    # Code Agent uses file review hooks. Career Agent uses its dedicated
    # evidence review tool only after resume generation.
    active_domain = str(runtime.get("activeDomain", "code")).strip().lower()
    if active_domain == "code" and not runtime.get("isSubagent"):
        init_review_hooks(cwd, tools, ToolContext(cwd=cwd, permissions=permissions, session=session, _runtime=runtime))

    # profile 和状态跟踪器
    run_policy = resolve_code_run_policy(runtime, fallback_max_steps=max_steps)
    configured_model_name = str(
        runtime.get("configuredModel")
        or runtime.get("model")
        or getattr(model, "model_id", "")
        or ""
    ).strip().lower()
    task_contract = _infer_task_contract(current_messages, runtime, active_domain)
    runtime["_taskContract"] = task_contract
    runtime["taskContract"] = task_contract.summary()
    turn_state = TurnRecurrentState(
        max_steps=run_policy.max_steps,
        profile_name=run_policy.name,
        task_contract=task_contract,
        widen_after_step=run_policy.widen_after_step,
        empty_response_retry_limit=run_policy.empty_response_retry_limit,
        recoverable_thinking_retry_limit=run_policy.recoverable_thinking_retry_limit,
        max_tool_calls=run_policy.max_tool_calls,
        verification_state=TurnVerificationState(
            strict=run_policy.strict_step_verification,
            requires_explicit_final=(
                task_contract.requires_verify and task_contract.verification_required
            ),
        ),
    )
    max_steps = run_policy.max_steps

    # 事件通知
    def emit_runtime_event(
        *, category, message, emit_progress=True,
        stop_reason="", widening_reason="", evidence_summary="",
    ) -> None:
        # Guard messages are control traffic for the model.  Keep them in the
        # runtime callback for diagnostics, but never publish them to the
        # external event bus or user-facing progress stream.
        is_internal_guard = category == "guard"
        if is_internal_guard:
            emit_progress = False
        event = RuntimeEvent(
            category=category, message=message,
            step=turn_state.step or None,
            profile=run_policy.name,
            phase="react",
            verification_focus="normal",
            stop_reason=stop_reason, widening_reason=widening_reason,
            evidence_summary=evidence_summary,
        )
        if on_runtime_event and not is_internal_guard:
            on_runtime_event(event)
        if not is_internal_guard:
            emit_external(
                "agent_progress",
                phase=event.phase or event.category,
                step=event.step,
                status="running",
                message=event.message[:500],
                output_summary=event.evidence_summary[:500],
            )
        if category == "stop":
            emit_external(
                "agent_failed" if event.stop_reason in {"blocked", "failed", "error"} else "agent_finished",
                phase="agent",
                status="failed" if event.stop_reason in {"blocked", "failed", "error"} else "completed",
                output_summary=event.message[:500],
                stop_reason=event.stop_reason,
            )
        if emit_progress and on_progress_message:
            on_progress_message(message)

    def stop_if_cancel_requested() -> list[ChatMessage] | None:
        if should_cancel is None or not should_cancel():
            return None
        fallback = "Agent paused by user."
        turn_state.set_stop_reason("await_user")
        emit_runtime_event(
            category="stop",
            message=fallback,
            emit_progress=False,
            stop_reason="await_user",
            evidence_summary=turn_state.latest_tool_result_summary,
        )
        if on_assistant_message:
            on_assistant_message(fallback)
        current_messages.append({"role": "assistant", "content": fallback})
        return current_messages

    def with_provider_usage(message: dict[str, Any], step_result: AgentStep) -> dict[str, Any]:
        diagnostics = step_result.diagnostics
        if diagnostics is None:
            return message
        input_tokens = max(0, int(getattr(diagnostics, "inputTokens", 0) or 0))
        output_tokens = max(0, int(getattr(diagnostics, "outputTokens", 0) or 0))
        if input_tokens or output_tokens:
            message["providerUsage"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        if step_result.providerState:
            message_id = str(
                message.get("toolUseId")
                or message.get("message_id")
                or f"message-{uuid.uuid4().hex[:12]}"
            )
            message["message_id"] = message_id
            message["_provider_state"] = {
                **copy.deepcopy(step_result.providerState),
                "session_id": runtime_session_id,
                "turn_id": runtime_turn_id,
                "message_id": message_id,
            }
        return message

    tool_scheduler = ToolScheduler(metrics_collector=metrics_collector)
    tool_result_budget = ToolResultBudgetManager(workspace=cwd)
    # A malformed path, unavailable symbol index, or other persistent tool
    # failure must not keep the model exploring indefinitely. Individual
    # failures remain recoverable, but a turn is stopped after a small number
    # of failed executions and reports the evidence collected so far.
    max_tool_errors = max(1, int(runtime.get("maxToolErrors", 4) or 4))
    max_reflections = max(0, int(runtime.get("maxReflections", 2) or 2))
    reflection_count = 0
    lifecycle = TurnLifecycleState(auditor=get_auditor() if enable_work_chain else None)
    cybernetic = CyberneticOrchestrator(runtime=runtime) if enable_work_chain else None
    if cybernetic is not None:
        cybernetic.wire_memory(context_manager=context_manager, runtime=runtime)
        cybernetic.wire_healing(runtime=runtime)
        cybernetic.inject_memories(messages=current_messages)

    # 构建任务（非控制论，是任务跟踪用的）
    if enable_work_chain:
        lifecycle.task, lifecycle.task_metadata = _build_work_chain_task(current_messages)
        if lifecycle.task:
            lifecycle.task_graph = TaskGraph(name=f"turn-{lifecycle.task.id}")
            graph_task = lifecycle.task_graph.add_task(
                name=lifecycle.task.title or lifecycle.task.id,
                description=lifecycle.task.goal or lifecycle.task.description,
            )
            lifecycle.task_graph_id = graph_task.id
            slot = lifecycle.task_graph.assign_slot(graph_task.id, slot_name="turn")
            lifecycle.task_slot_key = f"{slot.slot_name}:{slot.task_id}"
            lifecycle.task_graph.start_task(lifecycle.task_slot_key)
        _register_tool_capabilities(tools)

    # 上下文预检 + 压缩
    compaction_breaker = CompactionCircuitBreaker()

    runtime_session_id = str(
        getattr(session, "session_id", "")
        or runtime.get("sessionId")
        or runtime.get("session_id")
        or f"ephemeral-{uuid.uuid4().hex[:12]}"
    )
    runtime_turn_id = str(runtime.get("turnId") or runtime.get("turn_id") or f"turn-{uuid.uuid4().hex[:12]}")
    turn_goal = _extract_task_description(current_messages)
    working_turn_state, working_turn_token = begin_turn(
        runtime_session_id,
        runtime_turn_id,
        goal="" if turn_goal == "Unknown task" else turn_goal,
    )
    runtime["_workingTurnState"] = working_turn_state

    begin_session_turn = getattr(session, "begin_turn", None) if session is not None else None
    try:
        if context_manager:
            context_manager.active_turn_id = runtime_turn_id
            context_manager.configure_request(
                tool_definitions=[
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
                    for tool in tools.list()
                ],
                # Request context is merged into the system message above and
                # must not be counted a second time as an out-of-band prefix.
                dynamic_context=[],
            )
        if callable(begin_session_turn):
            begin_session_turn(runtime_turn_id, goal=working_turn_state.goal)
    except Exception:
        end_turn(working_turn_token)
        runtime.pop("_workingTurnState", None)
        if context_manager:
            context_manager.active_turn_id = None
        raise

    # Persist the unmodified incoming Surface before any compaction can replace
    # the model view.  Legacy SessionData implementations remain supported.
    sync_surface = getattr(session, "sync_surface", None) if session is not None else None

    def _sync_session_surface(reason: str) -> None:
        if callable(sync_surface):
            sync_surface(
                current_messages,
                turn_id=runtime_turn_id,
                source="agent_loop",
                reason=reason,
            )

    try:
        if callable(sync_surface):
            _sync_session_surface("turn input")

        if context_manager:
            if context_manager.summary_provider is None:
                context_manager.summary_provider = _context_summary_provider(model, store)
            context_manager.messages = current_messages
            stats = context_manager.get_stats()
            logger.info("Context: %d tokens (%.0f%%), %d messages",
                        stats.total_tokens, stats.usage_percentage, stats.messages_count)

            # 滚动窗口压缩
            if context_manager.should_auto_compact() and compaction_breaker.is_allowed():
                try:
                    logger.warning("Context near limit, auto-compacting...")
                    current_messages = getattr(context_manager, 'compact_messages', lambda: current_messages)()
                    compaction_breaker.record_success()
                except Exception as exc:
                    compaction_breaker.record_failure()
                    logger.warning("Auto-compact failed: %s", exc)
    except BaseException:
        working_snapshot = end_turn(working_turn_token)
        runtime.pop("_workingTurnState", None)
        if context_manager:
            context_manager.active_turn_id = None
        end_session_turn = getattr(session, "end_turn", None) if session is not None else None
        if callable(end_session_turn):
            try:
                end_session_turn(
                    runtime_turn_id,
                    status="failed",
                    summary=working_snapshot.get("error_state", "turn setup failed"),
                )
            except Exception:
                logger.exception("Failed to persist TurnEnded after setup failure")
        raise

    # ════════════════════════════════════════════════════════════
    # ReAct loop: model decision -> optional tool observations -> next decision.
    # ════════════════════════════════════════════════════════════

    seen_tool_signatures: set[str] = set()
    try:
        while turn_state.has_remaining_steps():
            cancelled_messages = stop_if_cancel_requested()
            if cancelled_messages is not None:
                return cancelled_messages
            if (
                turn_state.max_tool_calls is not None
                and turn_state.tool_observation_count >= turn_state.max_tool_calls
            ):
                synthesized = _synthesize_after_tool_budget(
                    model,
                    current_messages,
                    store=store,
                ) if turn_state.saw_tool_result else None
                if synthesized is not None:
                    summary = str(synthesized.content).strip()
                    stop_reason = "done" if turn_state.task_contract.can_finalize else "blocked"
                    turn_state.set_stop_reason(stop_reason)
                    emit_runtime_event(
                        category="stop",
                        message=summary,
                        emit_progress=False,
                        stop_reason=stop_reason,
                        evidence_summary=turn_state.verification_state.evidence_summary
                        or turn_state.latest_tool_result_summary,
                    )
                    if on_assistant_message:
                        on_assistant_message(summary)
                    current_messages.append({"role": "assistant", "content": summary})
                    return current_messages

                fallback = (
                    "工具检查已达到本轮上限，尚未生成完整总结。"
                    "最近一次工具结果："
                    f" {turn_state.latest_tool_result_summary or '无可用结果'}"
                )
                turn_state.set_stop_reason("blocked")
                emit_runtime_event(category="stop", message=fallback, emit_progress=False, stop_reason="blocked")
                if on_assistant_message:
                    on_assistant_message(fallback)
                current_messages.append({"role": "assistant", "content": fallback})
                return current_messages
            step = turn_state.begin_step()
            if cybernetic is not None:
                cybernetic.step_start(step=step, messages=current_messages, runtime=runtime)

            # Keep the prompt within the available context window.
            # 激进步策略要求压缩
            if context_manager and context_manager.should_auto_compact() and compaction_breaker.is_allowed():
                try:
                    _sync_session_surface("before aggressive compaction")
                    current_messages = getattr(context_manager, 'compact_messages', lambda: current_messages)()
                    compaction_breaker.record_success()
                except Exception as exc:
                    compaction_breaker.record_failure()
                    logger.warning("Aggressive compaction failed: %s", exc)

            # 更新稳定任务状态
            # Hook
            fire_hook_sync(HookEvent.AGENT_START, step=step, cwd=cwd)

            # Ask the model for the next ReAct decision.
            next_step: AgentStep
            try:
                # Layer 0: 预判式上下文守卫 — 先压缩再放弃
                if context_manager:
                    cm_stats = context_manager.get_stats()
                    if _is_at_blocking_limit(cm_stats.total_tokens, context_manager.context_window):
                        logger.warning("Context near limit (%d/%d), trying compact...",
                                       cm_stats.total_tokens, context_manager.context_window)
                        try:
                            _sync_session_surface("before blocking-limit compaction")
                            current_messages = context_manager.compact_messages()
                        except Exception as exc:
                            logger.warning("Proactive compact failed: %s", exc)
                        cm_stats = context_manager.get_stats()
                        if _is_at_blocking_limit(cm_stats.total_tokens, context_manager.context_window):
                            blocking_msg = (
                                f"Context near limit ({cm_stats.total_tokens} / {context_manager.context_window} tokens). "
                                "Use /compact manually, or reduce task scope."
                            )
                            logger.warning("Preemptive guard still blocking: %s", blocking_msg)
                            emit_runtime_event(category="stop", message=blocking_msg, stop_reason="blocked")
                            if on_assistant_message:
                                on_assistant_message(blocking_msg)
                            current_messages.append({"role": "assistant", "content": blocking_msg})
                            return current_messages

                emit_external("model_started", step=step, phase="model", status="running")
                next_step = _model_next(
                    model, current_messages,
                    on_stream_chunk=on_assistant_stream_chunk,
                    on_thinking_chunk=on_thinking_chunk,
                    store=store,
                )
                diagnostics = getattr(next_step, "diagnostics", None)
                emit_external(
                    "model_finished",
                    step=step,
                    phase="model",
                    status="completed",
                    output_summary=str(getattr(next_step, "content", "") or "")[:500],
                    input_tokens=int(getattr(diagnostics, "inputTokens", 0) or 0) if diagnostics else 0,
                    output_tokens=int(getattr(diagnostics, "outputTokens", 0) or 0) if diagnostics else 0,
                )
                cancelled_messages = stop_if_cancel_requested()
                if cancelled_messages is not None:
                    return cancelled_messages
            except KeyboardInterrupt:
                raise
            except ConnectionError as error:
                emit_external("model_finished", step=step, phase="model", status="failed", output_summary=str(error)[:500])
                fallback = f"Network error (connection failed or dropped): {error}"
                logger.error("Model API connection error: %s", error)
                turn_state.set_stop_reason("blocked")
                emit_runtime_event(category="stop", message=fallback, emit_progress=False, stop_reason="blocked")
                if on_assistant_message:
                    on_assistant_message(fallback)
                current_messages.append({"role": "assistant", "content": fallback})
                return current_messages
            except TimeoutError as error:
                emit_external("model_finished", step=step, phase="model", status="failed", output_summary=str(error)[:500])
                fallback = f"Model API timeout: {error}"
                logger.error("Model API timeout: %s", error)
                turn_state.set_stop_reason("blocked")
                emit_runtime_event(category="stop", message=fallback, emit_progress=False, stop_reason="blocked")
                if on_assistant_message:
                    on_assistant_message(fallback)
                current_messages.append({"role": "assistant", "content": fallback})
                return current_messages
            except ContextWindowExceeded as error:
                emit_external("model_finished", step=step, phase="model", status="failed", output_summary=str(error)[:500])
                logger.warning("Provider rejected Surface r%d: %s", context_manager.surface_revision if context_manager else 0, error)
                if context_manager:
                    generation_before = context_manager.generation
                    _sync_session_surface("before structured overflow recovery")
                    try:
                        recovered = context_manager.compact_messages(force=True)
                    except Exception as compact_error:
                        logger.warning("Structured overflow recovery failed: %s", compact_error)
                    else:
                        if context_manager.generation > generation_before:
                            current_messages = recovered
                            emit_runtime_event(
                                category="recovery",
                                message="Context overflow recovered by a committed Surface replacement.",
                            )
                            continue
                fallback = f"Context window exceeded and no smaller Surface could be committed: {error}"
                turn_state.set_stop_reason("blocked")
                emit_runtime_event(category="stop", message=fallback, emit_progress=False, stop_reason="blocked")
                if on_assistant_message:
                    on_assistant_message(fallback)
                current_messages.append({"role": "assistant", "content": fallback})
                return current_messages
            except Exception as error:
                emit_external(
                    "model_finished",
                    step=step,
                    phase="model",
                    status="failed",
                    output_summary=str(error)[:500],
                )
                error_type = type(error).__name__
                active_model_id = _infer_active_model_id(model, runtime, error)
                logger.error("Model API error (%s): %s", error_type, error)

                # Provider outages get one bounded fallback pass. The switcher
                # records failed models, so a later retry cannot bounce back.
                fallback_errors: list[str] = []
                if active_model_id and _looks_like_provider_availability_error(str(error)):
                    try:
                        from smart_agent.model_switcher import ModelSwitcher

                        runtime.setdefault("model", active_model_id)
                        switcher = runtime.get("_modelSwitcher")
                        if not isinstance(switcher, ModelSwitcher):
                            switcher = ModelSwitcher(active_model_id, runtime, tools)
                            runtime["_modelSwitcher"] = switcher
                        switcher.sync_current_model(active_model_id, adapter=model)
                        switcher.record_runtime_failure(active_model_id)
                        switched = switcher.switch_to_fallback(reason="provider_outage")
                        if switched.success and switched.adapter is not None:
                            model = switched.adapter
                            runtime["model"] = switched.new_model
                            logger.warning("Provider outage fallback selected model %s", switched.new_model)
                            emit_runtime_event(
                                category="recovery",
                                message=f"Provider outage recovered by switching to {switched.new_model}.",
                            )
                            continue
                        fallback_errors.extend(switched.errors)
                    except Exception as switch_error:  # noqa: BLE001 - preserve original API failure
                        fallback_errors.append(f"fallback switch error: {switch_error}")

                fallback = _summarize_model_api_failure(
                    error_type=error_type,
                    error=error,
                    active_model_id=active_model_id,
                    fallback_errors=fallback_errors,
                    runtime=runtime,
                )

                if on_assistant_message:
                    on_assistant_message(fallback)
                turn_state.set_stop_reason("blocked")
                emit_runtime_event(category="stop", message=fallback, emit_progress=False, stop_reason="blocked")
                current_messages.append({"role": "assistant", "content": fallback})
                return current_messages

            # A text-only assistant response is terminal unless marked progress.
            if next_step.type == "assistant":
                is_empty = _is_empty_assistant_response(next_step.content)
                diagnostics = next_step.diagnostics
                assistant_decision = _decide_react_assistant_turn(
                    turn_state=turn_state,
                    step_content=next_step.content,
                    step_kind=getattr(next_step, "kind", None),
                    stop_reason=diagnostics.stopReason if diagnostics else None,
                    ignored_block_types=diagnostics.ignoredBlockTypes if diagnostics else None,
                    is_empty=is_empty,
                    nudge_continue=NUDGE_CONTINUE,
                    nudge_after_tool_result=NUDGE_AFTER_TOOL_RESULT,
                    nudge_after_empty_no_tools=NUDGE_AFTER_EMPTY_NO_TOOLS,
                    resume_after_pause=RESUME_AFTER_PAUSE,
                    resume_after_max_tokens=RESUME_AFTER_MAX_TOKENS,
                )

                if assistant_decision.kind == "progress":
                    if assistant_decision.assistant_content:
                        turn_state.set_progress_summary(assistant_decision.assistant_content)
                        if assistant_decision.runtime_event_category is not None:
                            emit_runtime_event(
                                category=assistant_decision.runtime_event_category,
                                message=assistant_decision.assistant_content,
                                evidence_summary=(
                                    turn_state.verification_state.evidence_summary
                                    or turn_state.latest_tool_result_summary
                                ),
                            )
                        elif on_progress_message:
                            on_progress_message(assistant_decision.assistant_content)
                        current_messages.append(with_provider_usage(
                            {"role": "assistant_progress", "content": assistant_decision.assistant_content},
                            next_step,
                        ))
                    if assistant_decision.user_content:
                        current_messages.append({"role": "user", "content": assistant_decision.user_content})
                    continue

                if assistant_decision.kind == "retry":
                    if assistant_decision.user_content:
                        current_messages.append({"role": "user", "content": assistant_decision.user_content})
                    continue

                if assistant_decision.kind == "fallback":
                    if assistant_decision.stop_reason:
                        turn_state.set_stop_reason(assistant_decision.stop_reason)
                        emit_runtime_event(
                            category="stop",
                            message=assistant_decision.assistant_content or "Turn stopped without a final answer.",
                            emit_progress=False, stop_reason=assistant_decision.stop_reason,
                            evidence_summary=turn_state.verification_state.evidence_summary or turn_state.latest_tool_result_summary,
                        )
                    if assistant_decision.assistant_content and on_assistant_message:
                        on_assistant_message(assistant_decision.assistant_content)
                    if assistant_decision.assistant_content:
                        current_messages.append(with_provider_usage(
                            {"role": "assistant", "content": assistant_decision.assistant_content},
                            next_step,
                        ))
                    return current_messages

                turn_state.set_stop_reason("done")
                if assistant_decision.assistant_content and on_assistant_message:
                    on_assistant_message(assistant_decision.assistant_content)
                if assistant_decision.assistant_content:
                    current_messages.append(with_provider_usage(
                        {"role": "assistant", "content": assistant_decision.assistant_content},
                        next_step,
                    ))
                return current_messages

            # Execute requested tools and feed their observations back to the model.
            if next_step.content:
                role = "assistant_progress" if next_step.contentKind == "progress" else "assistant"
                if role == "assistant_progress":
                    turn_state.set_progress_summary(next_step.content)
                    if on_progress_message:
                        on_progress_message(next_step.content)
                    current_messages.append(with_provider_usage(
                        {"role": role, "content": next_step.content},
                        next_step,
                    ))
                    current_messages.append({"role": "user", "content": NUDGE_CONTINUE})
                else:
                    turn_state.set_progress_summary(next_step.content)
                    if on_assistant_message:
                        on_assistant_message(next_step.content)
                    current_messages.append(with_provider_usage(
                        {"role": role, "content": next_step.content},
                        next_step,
                    ))

            if not next_step.calls and next_step.content and next_step.contentKind != "progress":
                turn_state.set_stop_reason("done")
                emit_runtime_event(category="stop", message=next_step.content, emit_progress=False, stop_reason="done")
                return current_messages

            # 执行工具
            calls = next_step.calls
            _results: list[tuple[dict, ToolResult]] = []

            if len(calls) <= 1:
                # 单工具，串行
                for call in calls:
                    signature = _tool_signature(call)
                    guard_error = (
                        "Loop guard: identical tool call already executed in this Turn; "
                        "use new evidence or change the approach."
                        if signature in seen_tool_signatures
                        else ""
                    )
                    seen_tool_signatures.add(signature)
                    verification_ledger.record_call(call["id"], call["toolName"], call.get("input", {}))
                    if metrics_collector:
                        metrics_collector.start_tool(call["toolName"])
                    result = (
                        ToolResult(ok=False, output=guard_error)
                        if guard_error
                        else _execute_single_tool(
                            call, tools, cwd, permissions, session, runtime, store, step,
                            observed_tool_start, observed_tool_result, tool_scheduler,
                        )
                    )
                    if metrics_collector:
                        metrics_collector.end_tool(success=result.ok, error=result.output if not result.ok else "")
                    _results.append((call, result))
            else:
                # 多工具：调度器分类
                concurrent_calls, serial_calls = tool_scheduler.schedule_calls(calls, tools)
                _results.clear()

                # Run independent read-only calls concurrently.
                if concurrent_calls:
                    max_workers = tool_scheduler.get_recommended_max_workers(
                        concurrent_calls,
                        error_rate=turn_state.tool_error_count / max(step, 1),
                        avg_latency=step * 2.0,
                        recent_failures=turn_state.tool_error_count,
                    )
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mc-tool") as pool:
                        future_to_call = {}
                        batch_signatures: set[str] = set()
                        for call in concurrent_calls:
                            verification_ledger.record_call(
                                call["id"], call["toolName"], call.get("input", {})
                            )
                            signature = _tool_signature(call)
                            guard_error = (
                                "Loop guard: identical tool call already executed in this Turn; "
                                "use new evidence or change the approach."
                                if signature in seen_tool_signatures or signature in batch_signatures
                                else ""
                            )
                            seen_tool_signatures.add(signature)
                            batch_signatures.add(signature)
                            if guard_error:
                                _results.append((call, ToolResult(ok=False, output=guard_error)))
                                continue
                            future_to_call[pool.submit(
                                _execute_single_tool,
                                call,
                                tools,
                                cwd,
                                permissions,
                                session,
                                runtime,
                                None,
                                step,
                                observed_tool_start,
                                observed_tool_result,
                                tool_scheduler,
                            )] = call
                        for future in concurrent.futures.as_completed(future_to_call):
                            call = future_to_call[future]
                            try:
                                result = future.result()
                            except Exception as exc:
                                result = ToolResult(ok=False, output=f"Concurrent execution error: {exc}")
                            _results.append((call, result))

                # 冲突检测：并行批次中两个以上工具都失败 → 记录冲突对
                failed_in_parallel = [call["toolName"] for call, r in _results if not r.ok]
                if len(failed_in_parallel) >= 2:
                    for i, t1 in enumerate(failed_in_parallel):
                        for t2 in failed_in_parallel[i + 1:]:
                            tool_scheduler.record_conflict(t1, t2)

                # Run side-effecting calls serially to preserve ordering.
                if serial_calls:
                    for call in serial_calls:
                        verification_ledger.record_call(call["id"], call["toolName"], call.get("input", {}))
                        if metrics_collector:
                            metrics_collector.start_tool(call["toolName"])
                        signature = _tool_signature(call)
                        guard_error = (
                            "Loop guard: identical tool call already executed in this Turn; "
                            "use new evidence or change the approach."
                            if signature in seen_tool_signatures
                            else ""
                        )
                        seen_tool_signatures.add(signature)
                        result = (
                            ToolResult(ok=False, output=guard_error)
                            if guard_error
                            else _execute_single_tool(
                                call, tools, cwd, permissions, session, runtime, store, step,
                                observed_tool_start, observed_tool_result, tool_scheduler,
                            )
                        )
                        if metrics_collector:
                            metrics_collector.end_tool(success=result.ok, error=result.output if not result.ok else "")
                        _results.append((call, result))
                        if result.awaitUser:
                            break

            # 处理所有工具结果
            call_order = {call["id"]: idx for idx, call in enumerate(calls)}
            _results.sort(key=lambda pair: call_order.get(pair[0]["id"], 999))

            for call, result in _results:
                # Hook
                fire_hook_sync(HookEvent.POST_TOOL_USE, tool_name=call["toolName"], tool_output=result.output,
                               is_error=not result.ok, step=step)

                tool_summary = f"{call['toolName']}: {result.output[:200]}"
                verification_ledger.record_result(
                    call["id"], call["toolName"], result.ok, result.output
                )
                tool_def = tools.find(call["toolName"])
                if result.ok:
                    is_execution, is_verification = _tool_action_kinds(
                        call, tool_def, turn_state.task_contract
                    )
                    if is_execution:
                        turn_state.record_execution_action(call["toolName"])
                        # A write/command creates new evidence; allow the model
                        # to re-read the same path after the state changed.
                        seen_tool_signatures.clear()
                    if is_verification:
                        turn_state.record_verification_action(call["toolName"], ok=True)
                else:
                    is_execution, is_verification = _tool_action_kinds(
                        call, tool_def, turn_state.task_contract
                    )
                    if is_execution:
                        turn_state.task_contract.require_execution()
                    if is_verification:
                        turn_state.record_verification_action(call["toolName"], ok=False)
                turn_state.record_tool_result(result.ok, summary=tool_summary)
                working_turn_state.tool_result_refs.append(call["id"])
                working_turn_state.observations.append(tool_summary)
                turn_state.set_progress_summary(f"processed tool result from {call['toolName']}")

                # 错误处理 + nudge
                if not result.ok:
                    classified = ErrorClassifier.classify(result.output, tool_name=call["toolName"])
                    nudge = NudgeGenerator.generate(classified, retry_count=turn_state.tool_error_count)
                    result_output = result.output + "\n\n[System note: " + nudge + "]"
                else:
                    result_output = result.output

                # Persist oversized results before their first model exposure.
                # Previously exposed messages remain append-only for prompt-cache stability.
                result_output, persisted_path, _saved_chars = tool_result_budget.project_result(
                    result_output,
                    tool_name=call["toolName"],
                    result_key=call["id"],
                )

                # （ReadDedup 去重已在精简版中省略，不影响核心流程）

                tool_call_message = {
                    "role": "assistant_tool_call", "toolUseId": call["id"],
                    "toolName": call["toolName"], "input": call["input"],
                }
                if not next_step.content and call is calls[0]:
                    tool_call_message = with_provider_usage(tool_call_message, next_step)
                current_messages.append(tool_call_message)
                tool_result_message = {
                    "role": "tool_result", "toolUseId": call["id"],
                    "toolName": call["toolName"], "content": result_output, "isError": not result.ok,
                }
                if persisted_path is not None:
                    tool_result_message["_persisted_path"] = persisted_path
                current_messages.append(tool_result_message)

                # 开启结构化结束时，只有声明 concludes_turn 的工具可以关闭 Turn。
                tool_def = tools.find(call["toolName"])
                completion_request = runtime.get("_completionRequest")
                if result.ok and tool_def and getattr(tool_def, "concludes_turn", False) and completion_request:
                    status = str(completion_request.get("status") or "blocked").strip().lower()
                    stop_reason = {
                        "completed": "done",
                        "blocked": "blocked",
                        "paused": "await_user",
                    }.get(status, "blocked")
                    summary = str(completion_request.get("summary") or result.output).strip()
                    turn_state.set_stop_reason(stop_reason)
                    emit_runtime_event(
                        category="stop",
                        message=summary,
                        emit_progress=False,
                        stop_reason=stop_reason,
                        evidence_summary=verification_ledger.summary(),
                    )
                    if on_assistant_message:
                        on_assistant_message(summary)
                    current_messages.append({"role": "assistant", "content": summary})
                    runtime.pop("_completionRequest", None)
                    return current_messages

                if result.awaitUser:
                    turn_state.set_stop_reason("await_user")
                    emit_runtime_event(
                        category="stop", message=result_output,
                        emit_progress=False, stop_reason="await_user",
                        evidence_summary=turn_state.latest_tool_result_summary,
                    )
                    if on_assistant_message:
                        on_assistant_message(result_output)
                    current_messages.append({"role": "assistant", "content": result_output})
                    return current_messages

            # Conditional ReAct reflection: failures get one bounded self-check
            # before the model chooses its next action. Reflection is advisory;
            # it cannot execute tools or bypass the task/verification contract.
            failed_results = [call["toolName"] for call, result in _results if not result.ok]
            if failed_results and reflection_count < max_reflections:
                reflection_count += 1
                names = ", ".join(failed_results[:4])
                current_messages.append({
                    "role": "user",
                    "content": (
                        f"[Reflection {reflection_count}/{max_reflections}] 工具 {names} 执行失败。"
                        "请重新检查目标、失败原因和已有证据，判断是否需要改变工具/参数或直接说明阻塞；"
                        "不要重复相同调用。仅在有明确下一步时继续。"
                    ),
                })
                emit_runtime_event(
                    category="reflection",
                    message=f"已触发第 {reflection_count} 次失败后自检。",
                    emit_progress=False,
                )

            if turn_state.tool_error_count >= max_tool_errors:
                failure_summary = (
                    f"工具失败达到上限（{max_tool_errors} 次），已停止继续尝试。"
                    f" 最近结果：{turn_state.latest_tool_result_summary or '无可用结果'}"
                )
                turn_state.set_stop_reason("tool_errors")
                emit_runtime_event(
                    category="stop",
                    message=failure_summary,
                    emit_progress=False,
                    stop_reason="failed",
                    evidence_summary=verification_ledger.summary(),
                )
                if on_assistant_message:
                    on_assistant_message(failure_summary)
                current_messages.append({"role": "assistant", "content": failure_summary})
                return current_messages

            continue

        # while 正常退出（步数用尽）
        fallback = "Reached the maximum tool step limit for this turn."
        turn_state.set_stop_reason("max_steps")
        emit_runtime_event(category="stop", message=fallback, emit_progress=False, stop_reason="max_steps",
                           evidence_summary=turn_state.verification_state.evidence_summary or turn_state.latest_tool_result_summary)
        if on_assistant_message:
            on_assistant_message(fallback)
        current_messages.append({"role": "assistant", "content": fallback})
        return current_messages

    finally:
        try:
            if cybernetic is not None:
                cybernetic.step_end(step=turn_state.step, runtime=runtime)
                cybernetic.reflect_on_task(
                    task=lifecycle.task,
                    stop_reason=turn_state.stop_reason,
                    messages=current_messages,
                )
            review_hooks = (
                get_active_review_hooks()
                if active_domain == "code" and not runtime.get("isSubagent")
                else None
            )
            if review_hooks:
                review_hooks.on_turn_end(current_messages)

            # Compact once at turn end, then persist the final conversation surface.
            if context_manager:
                _sync_session_surface("before coda compaction")
                context_manager.messages = current_messages
                try:
                    coda_messages = context_manager.compact_messages()
                    if coda_messages is not current_messages:
                        current_messages[:] = coda_messages
                except Exception as exc:
                    logger.warning("Turn-end compaction failed: %s", exc)

            fire_hook_sync(HookEvent.AGENT_STOP, step=turn_state.step, tool_errors=turn_state.tool_error_count)

            if metrics_collector and metrics_collector._current_turn is not None:
                total_tokens = sum(estimate_message_tokens(m) for m in current_messages) if context_manager else 0
                metrics_collector.end_turn(total_tokens=total_tokens)

            context_usage = context_manager.get_stats().usage_percentage / 100.0 if context_manager else 0.0
            turn_summary = build_turn_coda_summary(turn_state=turn_state, context_usage=context_usage)

            if enable_work_chain and lifecycle.task:
                finalize_work_chain_task(
                    task=lifecycle.task, auditor=lifecycle.auditor,
                    coda_summary=turn_summary, success_outcome=DecisionOutcome.SUCCESS,
                    failure_outcome=DecisionOutcome.FAILURE,
                )
                if lifecycle.task_graph and lifecycle.task_slot_key:
                    try:
                        if turn_summary.task_state is TaskState.COMPLETED:
                            lifecycle.task_graph.complete_task(lifecycle.task_slot_key, result=lifecycle.task.result_summary)
                        elif turn_summary.task_state is TaskState.PAUSED:
                            slot = lifecycle.task_graph.slots.get(lifecycle.task_slot_key)
                            if slot is not None:
                                slot.state = GraphTaskState.QUEUED
                                slot.result = lifecycle.task.result_summary
                                lifecycle.task_graph.updated_at = time.time()
                        else:
                            lifecycle.task_graph.fail_task(lifecycle.task_slot_key, lifecycle.task.result_summary)
                    except Exception:
                        logger.debug("TaskGraph finalization skipped", exc_info=True)

            _sync_session_surface("turn completed")
        finally:
            working_snapshot = end_turn(working_turn_token)
            runtime.pop("_workingTurnState", None)
            runtime.pop("_verificationLedger", None)
            if context_manager:
                context_manager.active_turn_id = None
            end_session_turn = getattr(session, "end_turn", None) if session is not None else None
            if callable(end_session_turn):
                end_session_turn(
                    runtime_turn_id,
                    status=turn_state.stop_reason or working_snapshot.get("status", "completed"),
                    summary=turn_state.progress_state.get("summary", ""),
                )
