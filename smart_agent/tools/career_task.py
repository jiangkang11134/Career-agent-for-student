"""Career-only child-agent delegation through minimal artifacts."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from smart_agent.agents import (
    APPLICATION_SPEC,
    INTERVIEWER_SPEC,
    PROJECT_ANALYST_SPEC,
    AgentArtifact,
    AgentTask,
    ApplicationAgent,
    CodeAgent,
    InterviewerAgent,
    SubagentRunner,
    SubagentSpec,
)
from smart_agent.agent_collection import build_default_registry
from smart_agent.career import InterviewMode
from smart_agent.tooling import ToolDefinition, ToolResult

_SPECS = {
    "interviewer": INTERVIEWER_SPEC,
    "application": APPLICATION_SPEC,
    "project_analyst": PROJECT_ANALYST_SPEC,
}
_SIMULATION_ARTIFACTS = frozenset({"jd", "verified_resume", "interview_knowledge"})
_GAP_ARTIFACTS = frozenset({"interview_knowledge"})
_APPLICATION_ARTIFACTS = frozenset({"application_url", "verified_resume_pdf"})
_PROJECT_ANALYST_ARTIFACTS = frozenset({"project_analysis_request"})
_PROJECT_MODIFY_ARTIFACTS = frozenset({"project_modification_request"})
PROJECT_MODIFY_SPEC = SubagentSpec(
    agent_type="project_modify",
    display_name="Project Modification Code Agent",
    domain="code",
    system_prompt=(
        "You are an authorized project modification Code Agent. Modify only the supplied project workspace. "
        "First inspect relevant files, then make the smallest safe change requested. Run available tests after edits. "
        "Every edit to an existing file must include the SHA-256 returned by read_file as expected_hash; if it changes, stop and report a conflict. "
        "Do not read Career personal materials, other projects, secrets, or runtime data. Return a concise JSON summary "
        "of changed files, tests, and remaining risks."
    ),
    allowed_tools=frozenset({"read_file", "list_files", "grep_files", "file_tree", "find_symbols", "find_references", "edit_file", "patch_file", "write_file", "sandbox_test", "code_review"}),
    max_steps=12,
    timeout_seconds=300,
    read_only=False,
)


def _apply_project_analyst_model(runtime: dict[str, Any]) -> None:
    """Use the dedicated Project Analyst connection without changing the old loop."""

    from smart_agent.career import CareerModelConfig

    config = CareerModelConfig.from_runtime(runtime, "project_analyst", required=False)
    if config is None:
        return
    runtime.update({"model": config.model, "providerOverride": config.provider})
    if config.provider == "anthropic":
        runtime.update({"baseUrl": config.base_url, "apiKey": config.api_key})
    elif config.provider == "openrouter":
        runtime.update({"openrouterBaseUrl": config.base_url, "openrouterApiKey": config.api_key})
    elif config.provider == "openai":
        runtime.update({"openaiBaseUrl": config.base_url, "openaiApiKey": config.api_key})
    else:
        runtime.update({"customBaseUrl": config.base_url, "customApiKey": config.api_key})


def _primary_project_readme(source_path: Path) -> str:
    """Return the shallowest README path relative to the registered source."""

    from smart_agent.career import scan_project_artifacts

    root = source_path if source_path.is_dir() else source_path.parent
    candidates = [
        Path(item.path)
        for item in scan_project_artifacts(root)
        if Path(item.path).name.casefold() == "readme.md"
    ]
    if not candidates:
        raise ValueError("Project Analyst requires a readable project README.md")
    selected = min(candidates, key=lambda path: (len(path.parts), path.as_posix().casefold()))
    return selected.as_posix()


def _validate(input_data: dict) -> dict:
    agent_type = str(input_data.get("agent_type") or "").strip().lower()
    if agent_type == "project_modify":
        if input_data.get("approved") is not True:
            raise ValueError("project_modify requires explicit approved=true")
    elif agent_type not in _SPECS:
        raise ValueError("agent_type must be interviewer, application, project_analyst, or project_modify")
    mode = (
        InterviewMode(input_data.get("mode", InterviewMode.REALISTIC_SIMULATION.value))
        if agent_type == "interviewer"
        else None
    )
    description = str(input_data.get("description") or "").strip()
    if not description:
        raise ValueError("description is required")
    raw_artifacts = input_data.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("artifacts must be a non-empty array")
    artifacts: list[dict[str, Any]] = []
    for index, item in enumerate(raw_artifacts):
        if not isinstance(item, dict):
            raise ValueError(f"artifacts[{index}] must be an object")
        artifact_type = str(item.get("artifact_type") or "").strip()
        content = item.get("content")
        if not artifact_type or not isinstance(content, dict):
            raise ValueError(f"artifacts[{index}] requires artifact_type and object content")
        artifacts.append({"artifact_type": artifact_type, "content": dict(content)})
    actual_types = {item["artifact_type"] for item in artifacts}
    required_types = (
        _PROJECT_MODIFY_ARTIFACTS
        if agent_type == "project_modify"
        else
        _APPLICATION_ARTIFACTS
        if agent_type == "application"
        else _PROJECT_ANALYST_ARTIFACTS
        if agent_type == "project_analyst"
        else _GAP_ARTIFACTS
        if mode is InterviewMode.GAP_ANALYSIS
        else _SIMULATION_ARTIFACTS
    )
    if actual_types != required_types or len(artifacts) != len(required_types):
        raise ValueError(
            f"{agent_type} requires exactly these artifact types: {', '.join(sorted(required_types))}"
        )
    return {
        "agent_type": agent_type,
        "mode": mode.value if mode is not None else "",
        "description": description,
        "artifacts": artifacts,
        "approved": input_data.get("approved") is True,
    }


def _parent_session_id(context: Any) -> str:
    value = str(getattr(getattr(context, "session", None), "session_id", "") or "").strip()
    return value or f"ephemeral-{uuid.uuid4().hex[:12]}"


def _record_parent_result(context: Any, result) -> None:
    session = getattr(context, "session", None)
    if session is None or not hasattr(session, "delegated_tasks"):
        return
    session.delegated_tasks.append(
        {
            "task_id": result.task.task_id,
            "label": result.task.assigned_agent,
            "status": result.status,
            "child_session_id": result.task.child_session_id,
            "input_artifact_ids": list(result.task.input_artifact_ids),
            "output_artifact_id": result.output_artifact.artifact_id if result.output_artifact else "",
            "finding_count": len(result.findings),
        }
    )


def _run(input_data: dict, context) -> ToolResult:
    from smart_agent.config import load_runtime_config
    from smart_agent.domains import require_explicit_domain
    from smart_agent.tools import create_default_tool_registry

    runtime = dict(getattr(context, "_runtime", None) or {})
    if not runtime:
        runtime = dict(load_runtime_config(context.cwd))
    if require_explicit_domain(runtime.get("activeDomain") or "career").value != "career":
        return ToolResult(ok=False, output="career_task is available only in the Career domain.")
    runtime["activeDomain"] = "career"
    runtime.setdefault("domainStage", "default")
    if input_data["agent_type"] in {"project_analyst", "project_modify"}:
        _apply_project_analyst_model(runtime)
    parent_session_id = _parent_session_id(context)
    runner_cwd = str(context.cwd)
    normalized_artifacts = list(input_data["artifacts"])
    if input_data["agent_type"] in {"project_analyst", "project_modify"}:
        from smart_agent.career import ProjectLibraryStore, ProjectSourceStatus

        requested = normalized_artifacts[0]["content"]
        project_id = str(requested.get("project_id") or "").strip()
        if not project_id:
            raise ValueError("project_analysis_request.project_id is required")
        workspace = Path(context.cwd).resolve()
        configured = runtime.get("careerStorePath")
        career_root = Path(configured).expanduser() if configured else workspace / ".runtime" / "career"
        if not career_root.is_absolute():
            career_root = workspace / career_root
        career_root = career_root.resolve()
        if not career_root.is_relative_to(workspace):
            raise ValueError("careerStorePath must stay inside the active workspace")
        source = ProjectLibraryStore(
            career_root / "project-library",
            workspace=workspace,
        ).get(project_id, refresh=True)
        if source.status in {ProjectSourceStatus.UNAVAILABLE, ProjectSourceStatus.INVALIDATED}:
            raise ValueError(f"Project source is not readable: {source.status.value}")
        source_path = Path(source.source_location)
        if input_data["agent_type"] == "project_modify":
            runtime["optimisticLockRequired"] = True
            requested_goal = str(requested.get("change_goal") or "").strip()
            if not requested_goal:
                raise ValueError("project_modification_request.change_goal is required")
            runner_cwd = str(source_path if source_path.is_dir() else source_path.parent)
            normalized_artifacts = [{
                "artifact_type": "project_modification_request",
                "content": {
                    "project_id": source.project_id,
                    "project_version": source.project_version,
                    "change_goal": requested_goal,
                    "approved": True,
                },
            }]
        else:
            runner_cwd = str(source_path if source_path.is_dir() else source_path.parent)
        if input_data["agent_type"] == "project_analyst":
            from smart_agent.career import scan_project_artifacts
            project_artifacts = scan_project_artifacts(source_path)
            normalized_artifacts = [
            {
                "artifact_type": "project_analysis_request",
                "content": {
                    "project_id": source.project_id,
                    "project_version": source.project_version,
                    "display_name": source.display_name,
                    "intent": str(requested.get("intent") or "overview").strip(),
                    "artifact_count": source.artifact_count,
                    "required_readme_path": _primary_project_readme(source_path),
                    "artifact_paths": [item.path for item in project_artifacts],
                },
            }
            ]
    selected_spec = PROJECT_MODIFY_SPEC if input_data["agent_type"] == "project_modify" else _SPECS[input_data["agent_type"]]
    artifact_domain = "code" if input_data["agent_type"] == "project_modify" else "career"
    artifacts = [
        AgentArtifact.create(
            item["artifact_type"],
            domain=artifact_domain,
            session_id=parent_session_id,
            created_by="career_agent",
            content=item["content"],
        )
        for item in normalized_artifacts
    ]
    spec = selected_spec
    task = AgentTask.create(
        spec.agent_type,
        domain=artifact_domain,
        description=input_data["description"],
        parent_session_id=parent_session_id,
        assigned_agent=spec.agent_type,
        input_artifact_ids=[artifact.artifact_id for artifact in artifacts],
        allowed_tools=spec.allowed_tools,
        timeout_seconds=spec.timeout_seconds,
        max_steps=spec.max_steps,
    )
    if input_data["agent_type"] == "project_analyst":
        from smart_agent.tooling import ToolRegistry
        from smart_agent.tools.code_nav import find_references_tool, find_symbols_tool, get_ast_info_tool
        from smart_agent.tools.file_tree import file_tree_tool
        from smart_agent.tools.grep_files import grep_files_tool
        from smart_agent.tools.list_files import list_files_tool
        from smart_agent.tools.project_documents import read_project_document_tool
        from smart_agent.tools.read_file import read_file_tool

        tools = ToolRegistry(
            [
                list_files_tool,
                grep_files_tool,
                read_file_tool,
                read_project_document_tool,
                file_tree_tool,
                find_symbols_tool,
                find_references_tool,
                get_ast_info_tool,
            ]
        )
    else:
        runtime["activeDomain"] = "code" if input_data["agent_type"] == "project_modify" else "career"
        tools = create_default_tool_registry(runner_cwd if input_data["agent_type"] == "project_modify" else context.cwd, runtime=runtime)
    try:
        runner = SubagentRunner(
            cwd=runner_cwd,
            runtime=runtime,
            tools=tools,
            permissions=getattr(context, "permissions", None),
        )
        if input_data["agent_type"] == "project_modify":
            result = CodeAgent(runner).run(task, artifacts, mode="project_modify", spec=spec)
        elif spec.agent_type == "interviewer":
            agent = build_default_registry().create("interviewer", runner, mode=input_data["mode"])
            result = agent.run(task, artifacts)
        elif spec.agent_type == "application":
            agent = build_default_registry().create("application", runner)
            result = agent.run(task, artifacts)
        else:
            result = CodeAgent(runner).run(task, artifacts, mode="project_analysis")
    finally:
        tools.dispose()
    _record_parent_result(context, result)
    return ToolResult(
        ok=result.status == "completed",
        output=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
    )


career_task_tool = ToolDefinition(
    name="career_task",
    description=(
        "Launch an isolated Career child agent. A realistic interviewer requires JD, verified resume, and interview "
        "knowledge; a gap-analysis interviewer receives only an interview-knowledge scope and relies on its isolated "
        "runtime for live read-only RAG. Application requires URL and verified resume PDF. Project Analyst accepts "
        "one registered project ID and reads that project with read-only repository tools. project_modify requires "
        "explicit approved=true and changes only the selected registered project."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "agent_type": {
                "type": "string",
                "enum": ["interviewer", "application", "project_analyst", "project_modify"],
            },
            "mode": {
                "type": "string",
                "enum": [item.value for item in InterviewMode],
            },
            "description": {"type": "string"},
            "approved": {"type": "boolean"},
            "artifacts": {
                "type": "array",
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
        "required": ["agent_type", "description", "artifacts"],
    },
    validator=_validate,
    run=_run,
)
