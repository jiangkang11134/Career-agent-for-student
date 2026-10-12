"""Read-only extraction for non-code project documents."""

from __future__ import annotations

from smart_agent.career.project_library import extract_project_document_text
from smart_agent.tooling import ToolCapability, ToolDefinition, ToolMetadata, ToolResult
from smart_agent.workspace import resolve_tool_path

DEFAULT_LINE_LIMIT = 240
MAX_LINE_LIMIT = 800


def _validate(input_data: dict) -> dict:
    path = str(input_data.get("path") or "").strip()
    if not path:
        raise ValueError("path is required")
    start_line = int(input_data.get("start_line", 1))
    line_limit = int(input_data.get("line_limit", DEFAULT_LINE_LIMIT))
    if start_line < 1:
        raise ValueError("start_line must be at least 1")
    if not 1 <= line_limit <= MAX_LINE_LIMIT:
        raise ValueError(f"line_limit must be between 1 and {MAX_LINE_LIMIT}")
    return {"path": path, "start_line": start_line, "line_limit": line_limit}


def _run(input_data: dict, context) -> ToolResult:
    target = resolve_tool_path(context, input_data["path"], "read")
    text, parser = extract_project_document_text(target)
    lines = text.splitlines()
    start = input_data["start_line"] - 1
    end = min(len(lines), start + input_data["line_limit"])
    selected = "\n".join(f"{index + 1}: {lines[index]}" for index in range(start, end))
    return ToolResult(
        ok=True,
        output=(
            f"PROJECT_DOCUMENT: {input_data['path']}\n"
            f"PARSER: {parser}\n"
            f"LINES: {start + 1}-{end} / {len(lines)}\n"
            f"TRUNCATED: {'yes' if end < len(lines) else 'no'}\n\n"
            f"{selected}"
        ),
    )


read_project_document_tool = ToolDefinition(
    name="read_project_document",
    description=(
        "Read text from a project PDF, DOCX, PPTX, XLSX, image, or plain document without executing project code. "
        "Output includes stable line numbers for source citations."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "line_limit": {"type": "integer", "minimum": 1, "maximum": MAX_LINE_LIMIT},
        },
        "required": ["path"],
    },
    validator=_validate,
    run=_run,
    metadata=ToolMetadata(
        name="read_project_document",
        description="Read a non-code project document as line-numbered text.",
        capabilities={ToolCapability.READ_ONLY, ToolCapability.CONCURRENCY_SAFE},
        max_result_size_chars=40_000,
        tags=["career", "project", "read-only"],
    ),
)
