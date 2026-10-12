"""Smart Harness 的结构化任务结束工具。"""

from __future__ import annotations

import json

from smart_agent.tooling import ToolDefinition, ToolResult


def _validate(input_data: dict) -> dict:
    status = str(input_data.get("status") or "").strip().lower()
    if status not in {"completed", "blocked", "paused"}:
        raise ValueError("status must be completed, blocked, or paused")
    summary = str(input_data.get("summary") or "").strip()
    if not summary:
        raise ValueError("summary is required")
    if "evidence_refs" in input_data:
        raise ValueError(
            "complete_task uses tool_evidence_refs; evidence_refs is reserved for Career records"
        )
    refs = input_data.get("tool_evidence_refs", [])
    if not isinstance(refs, list):
        raise ValueError("tool_evidence_refs must be a list")
    normalized_refs = []
    seen_refs = set()
    for item in refs:
        if not isinstance(item, dict):
            raise ValueError(
                "tool_evidence_refs items must be objects with kind=tool_call/id or kind=tool/name"
            )
        kind = str(item.get("kind") or "").strip().lower()
        if kind == "tool_call":
            value = str(item.get("id") or "").strip()
            normalized = {"kind": kind, "id": value}
        elif kind == "tool":
            value = str(item.get("name") or "").strip()
            normalized = {"kind": kind, "name": value}
        else:
            raise ValueError("tool_evidence_refs.kind must be tool_call or tool")
        if not value:
            raise ValueError(
                "tool_evidence_refs requires a non-empty id for tool_call or name for tool"
            )
        identity = (kind, value)
        if identity in seen_refs:
            raise ValueError("tool_evidence_refs must not contain duplicates")
        seen_refs.add(identity)
        normalized_refs.append(normalized)
    return {
        "status": status,
        "summary": summary,
        "tool_evidence_refs": normalized_refs,
    }


def _run(input_data: dict, context) -> ToolResult:
    runtime = context._runtime if context is not None else None
    ledger = runtime.get("_verificationLedger") if isinstance(runtime, dict) else None
    if input_data["status"] == "completed" and ledger is not None:
        # 当前 complete_task 调用本身尚未提交结果，因此 ledger 会暂时有一个
        # pending 项。这里只检查此前事件，避免把当前调用误判成悬空调用。
        if ledger.invalid_reasons:
            return ToolResult(ok=False, output=f"Cannot complete task: {ledger.invalid_reasons[-1]}")
        business_pending = {
            call_id: item for call_id, item in ledger.pending.items()
            if item[0] != "complete_task"
        }
        if business_pending:
            return ToolResult(
                ok=False,
                output="Cannot complete task: unresolved tool calls: " + ", ".join(sorted(business_pending)),
            )
        business_failures = [
            item for item in ledger.unresolved_failed_results
            if item.tool_name != "complete_task"
        ]
        if business_failures:
            failed = ", ".join(item.tool_name for item in business_failures[:3])
            return ToolResult(ok=False, output=f"Cannot complete task: tool execution failed: {failed}")
        available_refs = {
            ref
            for item in ledger.successful_results
            for ref in (item.call_id, item.tool_name)
        }
        requested_refs = {
            ref["id"] if ref["kind"] == "tool_call" else ref["name"]
            for ref in input_data["tool_evidence_refs"]
        }
        if available_refs and not requested_refs:
            return ToolResult(ok=False, output="Cannot complete task: tool_evidence_refs are required")
        unknown_refs = requested_refs - available_refs
        if unknown_refs:
            return ToolResult(
                ok=False,
                output="Cannot complete task: unknown tool_evidence_refs: " + ", ".join(sorted(unknown_refs)),
            )
    request = dict(input_data)
    if isinstance(runtime, dict):
        runtime["_completionRequest"] = request
    return ToolResult(
        ok=True,
        output="Task completion recorded: " + json.dumps(request, ensure_ascii=False, sort_keys=True),
    )


complete_task_tool = ToolDefinition(
    name="complete_task",
    description=(
        "Close the current task with a structured status. Use completed only after all required "
        "tool results are successful. For completed tasks, cite tool_evidence_refs using "
        "{kind: tool_call, id: ...} or {kind: tool, name: ...}; use blocked or paused when "
        "the task cannot be completed."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed", "blocked", "paused"]},
            "summary": {"type": "string"},
            "tool_evidence_refs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["tool_call", "tool"]},
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                    },
                    "required": ["kind"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["status", "summary"],
    },
    validator=_validate,
    run=_run,
    concludes_turn=True,
)
