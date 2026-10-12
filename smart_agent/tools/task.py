"""Task tool for bounded, artifact-driven Code-domain child agents."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from smart_agent.agents import (
    CODE_REVIEW_SPEC,
    AgentArtifact,
    AgentTask,
    CodeAgent,
    CodeReviewAgent,
    SubagentRunner,
    SubagentSpec,
)
from smart_agent.tooling import ToolDefinition, ToolResult
from smart_agent.agent_collection import build_default_registry

_READ_ONLY_CODE_TOOLS = frozenset({
    "read_file",
    "list_files",
    "grep_files",
    "file_tree",
    "find_symbols",
    "find_references",
    "get_ast_info",
    "diff_viewer",
    "code_review",
})

AGENT_SPECS: dict[str, SubagentSpec] = {
    "explore": SubagentSpec(
        agent_type="explore",
        display_name="Explore Agent",
        domain="code",
        system_prompt=(
            "Explore the repository using read-only tools. Locate relevant files and symbols, "
            "then return a concise evidence-based summary."
        ),
        allowed_tools=frozenset({
            "read_file",
            "list_files",
            "grep_files",
            "file_tree",
            "find_symbols",
            "find_references",
            "get_ast_info",
        }),
        max_steps=5,
        timeout_seconds=45,
    ),
    "plan": SubagentSpec(
        agent_type="plan",
        display_name="Plan Agent",
        domain="code",
        system_prompt=(
            "Analyze the supplied task and repository evidence with read-only tools. "
            "Return an actionable implementation plan; do not edit files."
        ),
        allowed_tools=_READ_ONLY_CODE_TOOLS,
        max_steps=8,
        timeout_seconds=60,
    ),
    "general": SubagentSpec(
        agent_type="general",
        display_name="Analysis Agent",
        domain="code",
        system_prompt=(
            "Perform the bounded analysis described by the supplied task artifacts. "
            "Return evidence and recommendations for the parent agent to adopt. Do not edit files."
        ),
        allowed_tools=_READ_ONLY_CODE_TOOLS,
        max_steps=10,
        timeout_seconds=60,
    ),
    "code_review": CODE_REVIEW_SPEC,
    "test": SubagentSpec(
        agent_type="test",
        display_name="Test Agent",
        domain="code",
        system_prompt="Run the requested verification in the sandbox and return its result.",
        allowed_tools=frozenset({"sandbox_test", "read_file"}),
        max_steps=3,
        timeout_seconds=90,
        read_only=False,
    ),
}


def _legacy_agent_type_view(spec: SubagentSpec) -> dict[str, Any]:
    return {
        "name": spec.display_name,
        "description": spec.system_prompt,
        "system_prompt": spec.system_prompt,
        "allowed_tools": set(spec.allowed_tools),
        "max_turns": spec.max_steps,
        "timeout_seconds": spec.timeout_seconds,
        "read_only": spec.read_only,
    }


# Compatibility for callers that inspect AGENT_TYPES. New tasks use code_review.
AGENT_TYPES = {name: _legacy_agent_type_view(spec) for name, spec in AGENT_SPECS.items()}
AGENT_TYPES["review"] = AGENT_TYPES["code_review"]


def _validate_artifacts(value: Any, *, agent_type: str) -> list[dict[str, Any]]:
    if value is None:
        if agent_type == "code_review":
            raise ValueError("code_review requires structured input artifacts.")
        return []
    if not isinstance(value, list):
        raise ValueError("artifacts must be an array.")
    artifacts: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"artifacts[{index}] must be an object.")
        artifact_type = str(item.get("artifact_type") or "").strip()
        content = item.get("content")
        if not artifact_type:
            raise ValueError(f"artifacts[{index}].artifact_type is required.")
        if not isinstance(content, dict):
            raise ValueError(f"artifacts[{index}].content must be an object.")
        artifacts.append({"artifact_type": artifact_type, "content": dict(content)})
    if agent_type == "code_review" and not artifacts:
        raise ValueError("code_review requires structured input artifacts.")
    return artifacts


def _validate(input_data: dict) -> dict:
    description = input_data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description is required.")

    requested_type = str(input_data.get("agent_type") or "general").strip().lower()
    agent_type = "code_review" if requested_type == "review" else requested_type
    if agent_type not in AGENT_SPECS:
        valid = ", ".join(AGENT_SPECS)
        raise ValueError(f"agent_type must be one of: {valid}. Got: {requested_type}")
    project_id = str(input_data.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("project_id is required; Code Agent can only access a registered uploaded project.")

    prompt = input_data.get("prompt", description)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    artifacts = _validate_artifacts(input_data.get("artifacts"), agent_type=agent_type)
    return {
        "description": description.strip(),
        "agent_type": agent_type,
        "prompt": prompt.strip(),
        "artifacts": artifacts,
        "project_id": project_id,
    }


def _parent_session_id(context: Any) -> str:
    session_id = str(getattr(getattr(context, "session", None), "session_id", "") or "").strip()
    return session_id or f"ephemeral-{uuid.uuid4().hex[:12]}"


def _record_parent_result(context: Any, result) -> None:
    session = getattr(context, "session", None)
    if session is None or not hasattr(session, "delegated_tasks"):
        return
    session.delegated_tasks.append({
        "task_id": result.task.task_id,
        "label": result.task.assigned_agent,
        "status": result.status,
        "child_session_id": result.task.child_session_id,
        "input_artifact_ids": list(result.task.input_artifact_ids),
        "output_artifact_id": (
            result.output_artifact.artifact_id if result.output_artifact else ""
        ),
        "finding_count": len(result.findings),
    })


def _run(input_data: dict, context) -> ToolResult:
    from smart_agent.config import load_runtime_config
    from smart_agent.domains import require_explicit_domain
    from smart_agent.tools import create_default_tool_registry

    runtime = dict(getattr(context, "_runtime", None) or {})
    if not runtime:
        try:
            runtime = dict(load_runtime_config(context.cwd))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, output=f"Cannot load sub-agent runtime: {exc}")
    active_domain = require_explicit_domain(runtime.get("activeDomain") or "code")
    if active_domain.value != "code":
        return ToolResult(ok=False, output="The task tool is available only in the Code domain.")
    runtime["activeDomain"] = "code"
    runtime.setdefault("domainStage", "default")

    from smart_agent.career import ProjectLibraryStore, ProjectSourceStatus

    workspace = Path(context.cwd).resolve()
    configured = runtime.get("careerStorePath")
    career_root = Path(configured).expanduser() if configured else workspace / ".runtime" / "career"
    if not career_root.is_absolute():
        career_root = workspace / career_root
    career_root = career_root.resolve()
    if not career_root.is_relative_to(workspace):
        return ToolResult(ok=False, output="careerStorePath must stay inside the active workspace.")
    try:
        source = ProjectLibraryStore(career_root / "project-library", workspace=workspace).get(
            input_data["project_id"], refresh=True
        )
    except (KeyError, FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        return ToolResult(ok=False, output=f"Registered project is unavailable: {exc}")
    if source.status in {ProjectSourceStatus.UNAVAILABLE, ProjectSourceStatus.INVALIDATED}:
        return ToolResult(ok=False, output=f"Registered project is unavailable: {source.status.value}")
    project_root = Path(source.source_location).resolve()
    if not project_root.is_dir():
        project_root = project_root.parent
    runtime["projectId"] = source.project_id
    runtime["projectRoot"] = str(project_root)
    runtime["optimisticLockRequired"] = True

    parent_session_id = _parent_session_id(context)
    created_by = "code_agent"
    artifacts = [
        AgentArtifact.create(
            item["artifact_type"],
            domain="code",
            session_id=parent_session_id,
            created_by=created_by,
            content=item["content"],
        )
        for item in input_data["artifacts"]
    ]
    if not artifacts:
        artifacts = [
            AgentArtifact.create(
                "task_context",
                domain="code",
                session_id=parent_session_id,
                created_by=created_by,
                content={"prompt": input_data["prompt"]},
            )
        ]

    spec = AGENT_SPECS[input_data["agent_type"]]
    task = AgentTask.create(
        spec.agent_type,
        domain="code",
        description=input_data["description"],
        parent_session_id=parent_session_id,
        assigned_agent=spec.agent_type,
        input_artifact_ids=[artifact.artifact_id for artifact in artifacts],
        allowed_tools=spec.allowed_tools,
        timeout_seconds=spec.timeout_seconds,
        max_steps=spec.max_steps,
    )

    base_tools = create_default_tool_registry(str(project_root), runtime=runtime)
    try:
        runner = SubagentRunner(
            cwd=str(project_root),
            runtime=runtime,
            tools=base_tools,
            permissions=getattr(context, "permissions", None),
        )
        result = (
            build_default_registry().create("code", runner).run(task, artifacts, mode="code_review")
            if spec.agent_type == "code_review"
            else runner.run(task, artifacts, spec)
        )
    finally:
        base_tools.dispose()

    _record_parent_result(context, result)
    return ToolResult(
        ok=result.status == "completed",
        output=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
    )


task_tool = ToolDefinition(
    name="task",
    description=(
        "Launch an isolated Code-domain child agent using versioned artifacts. "
        "Use code_review for read-only structured review findings, explore for repository discovery, "
        "plan for implementation analysis, general for bounded read-only analysis, or test for sandbox verification. "
        "Child agents never receive the parent chat history and cannot recursively launch tasks."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short task description.",
            },
            "prompt": {
                "type": "string",
                "description": "Task details for non-review agents.",
            },
            "agent_type": {
                "type": "string",
                "enum": ["explore", "plan", "general", "code_review", "test"],
            },
            "project_id": {"type": "string", "description": "Registered uploaded project ID."},
            "artifacts": {
                "type": "array",
                "description": "Minimal structured inputs. Required for code_review.",
                "items": {
                    "type": "object",
                    "properties": {
                        "artifact_type": {"type": "string"},
                        "content": {"type": "object"},
                    },
                    "required": ["artifact_type", "content"],
                },
            },
        },
        "required": ["description", "project_id"],
    },
    validator=_validate,
    run=_run,
)
