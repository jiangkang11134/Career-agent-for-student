"""Smart Harness 的结构化 Turn 验证账本。

账本只记录可验证事实：工具调用、对应结果、失败状态和顺序。
模型文本不能修改账本，也不能用关键词伪造验证证据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolEvidence:
    call_id: str
    tool_name: str
    ok: bool
    output: str


@dataclass(slots=True)
class VerificationLedger:
    """一个 Turn 内的工具事件账本。

    调用必须先登记，结果必须按已登记的 call id 提交；重复、悬空或
    未完成的调用都会使收尾验证失败。
    """

    events: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    results: list[ToolEvidence] = field(default_factory=list)
    invalid_reasons: list[str] = field(default_factory=list)

    def record_call(self, call_id: str, tool_name: str, tool_input: dict[str, Any]) -> None:
        call_id = str(call_id or "").strip()
        tool_name = str(tool_name or "").strip()
        if not call_id:
            self.invalid_reasons.append("tool call has no id")
            return
        if call_id in self.pending or any(item.call_id == call_id for item in self.results):
            self.invalid_reasons.append(f"duplicate tool call id: {call_id}")
            return
        self.pending[call_id] = (tool_name, dict(tool_input or {}))
        self.events.append({
            "type": "tool_call",
            "call_id": call_id,
            "tool_name": tool_name,
            "input": dict(tool_input or {}),
        })

    def record_result(self, call_id: str, tool_name: str, ok: bool, output: str) -> None:
        call_id = str(call_id or "").strip()
        tool_name = str(tool_name or "").strip()
        pending = self.pending.pop(call_id, None)
        if pending is None:
            self.invalid_reasons.append(f"tool result without matching call: {call_id}")
        elif pending[0] != tool_name:
            self.invalid_reasons.append(
                f"tool result name mismatch for {call_id}: {pending[0]} != {tool_name}"
            )
        evidence = ToolEvidence(
            call_id=call_id,
            tool_name=tool_name,
            ok=bool(ok),
            output=str(output or ""),
        )
        self.results.append(evidence)
        self.events.append({
            "type": "tool_result",
            "call_id": call_id,
            "tool_name": tool_name,
            "ok": evidence.ok,
            "output_chars": len(evidence.output),
        })

    @property
    def successful_results(self) -> tuple[ToolEvidence, ...]:
        return tuple(item for item in self.results if item.ok)

    @property
    def failed_results(self) -> tuple[ToolEvidence, ...]:
        return tuple(item for item in self.results if not item.ok)

    @property
    def unresolved_failed_results(self) -> tuple[ToolEvidence, ...]:
        """Failures whose tool has not produced a later successful result.

        ReAct deliberately permits recovery: a failed command or validation
        attempt should not poison the whole Turn after a later successful retry.
        """
        latest_by_tool: dict[str, ToolEvidence] = {}
        for item in self.results:
            latest_by_tool[item.tool_name] = item
        return tuple(item for item in latest_by_tool.values() if not item.ok)

    def check_final(self, *, require_success: bool = True) -> tuple[bool, str]:
        """返回是否允许进入最终回答，不参考模型自述。"""
        if self.invalid_reasons:
            return False, self.invalid_reasons[-1]
        if self.pending:
            return False, f"unresolved tool calls: {', '.join(sorted(self.pending))}"
        if not self.results:
            return False, "no tool result has been recorded"
        if require_success and self.unresolved_failed_results:
            failed = ", ".join(item.tool_name for item in self.unresolved_failed_results[:3])
            return False, f"tool execution failed: {failed}"
        return True, "tool calls and results are complete"

    def cites_evidence(self, content: str) -> bool:
        """要求最终文本引用本次账本中的工具或结果事实。"""
        normalized = " ".join(str(content or "").lower().split())
        if not normalized:
            return False
        for item in self.successful_results:
            if item.tool_name.lower() in normalized or item.call_id.lower() in normalized:
                return True
            words = {
                word.strip(".,:;()[]{}'\"`")
                for word in item.output.lower().split()
                if len(word.strip(".,:;()[]{}'\"`")) >= 5
            }
            if any(word in normalized for word in list(words)[:12]):
                return True
        return False

    def summary(self) -> str:
        return (
            f"calls={len(self.events) // 2 if self.events else 0}, "
            f"results={len(self.results)}, "
            f"success={len(self.successful_results)}, "
            f"failed={len(self.unresolved_failed_results)}, "
            f"pending={len(self.pending)}"
        )
