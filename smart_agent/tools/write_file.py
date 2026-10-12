"""写入文件工具。

提供文件写入功能，将内容写入工作区中的指定文件。写入操作会经过文件审阅流程。
"""
from __future__ import annotations

from smart_agent.file_review import apply_reviewed_file_change
from smart_agent.tooling import ToolDefinition
from smart_agent.workspace import resolve_tool_path


def _validate(input_data: dict) -> dict:
    path = input_data.get("path")
    content = input_data.get("content")
    if not isinstance(path, str) or not path:
        raise ValueError("path is required")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    return {"path": path, "content": content, "group_id": input_data.get("group_id", ""), "expected_hash": str(input_data.get("expected_hash") or "").strip()}


def _run(input_data: dict, context):
    target = resolve_tool_path(context, input_data["path"], "write")
    return apply_reviewed_file_change(
        context, input_data["path"], target, input_data["content"],
        group_id=input_data.get("group_id", ""),
        expected_hash=input_data.get("expected_hash", ""),
    )


write_file_tool = ToolDefinition(
    name="write_file",
    description="Write a UTF-8 text file relative to the workspace root.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Content to write"},
            "group_id": {"type": "string", "description": "Optional group ID for atomic rewind with related files"},
            "expected_hash": {"type": "string", "description": "Optional SHA-256 from read_file; reject stale writes"},
        },
        "required": ["path", "content"],
    },
    validator=_validate,
    run=_run,
    needs_review=True,
)  #
