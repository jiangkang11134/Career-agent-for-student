"""Read-only Project Analyst child agent and its structured output contract."""

from __future__ import annotations

import json

from smart_agent.agents.contracts import AgentArtifact, AgentRunResult, AgentTask
from smart_agent.agents.runner import SubagentRunner, SubagentSpec

PROJECT_ANALYST_INPUT_TYPES = frozenset({"project_analysis_request"})

# The archived Project Analyst prompt permits qualitative confidence labels as
# well as numeric scores. Keep the normalized representation numeric because
# the career models and persistence layer consume values in the [0, 1] range.
_QUALITATIVE_CONFIDENCE = {
    "very_high": 0.95,
    "high": 0.85,
    "medium": 0.65,
    "moderate": 0.65,
    "low": 0.35,
    "very_low": 0.15,
    "unknown": 0.0,
    "none": 0.0,
    "很高": 0.95,
    "高": 0.85,
    "中": 0.65,
    "中等": 0.65,
    "低": 0.35,
    "很低": 0.15,
    "未知": 0.0,
    "无": 0.0,
}

PROJECT_ANALYST_SPEC = SubagentSpec(
    agent_type="project_analyst",
    display_name="Project Analyst Agent",
    domain="career",
    system_prompt=(
        "You are a read-only Project Analyst. Inspect the supplied project with repository navigation tools. "
        "The final human-readable fields must be written in Simplified Chinese; keep technology names, class names, "
        "identifiers, and project-relative source paths in their original spelling. Return real JSON arrays, never "
        "Python-list strings. The first tool call MUST read the request's required_readme_path. "
        "Build a career-oriented understanding of the project without reading runtime assets, model weights, "
        "datasets, dependency directories, caches, secrets, or parent chat. Do not infer that repository code was "
        "written by the user. Return exactly one JSON object with these keys: title, goal, core_flow, core_modules, "
        "architecture, technology_choices, technologies, implemented_features, candidate_outcomes, "
        "candidate_responsibilities, technical_tags, unknowns, source_refs, confidence_by_field, analysis_summary. "
        "Every confidence_by_field value must be a JSON number from 0 to 1. "
        "Array fields must be arrays of concise strings. Each source_refs item must contain field, path, line, and "
        "excerpt. Cite actual project-relative paths. core_modules must contain exact readable source paths, and "
        "every declared core module must be read before the final response. Before returning JSON, cross-check your "
        "successful file reads: core_modules must be a subset of those successfully read paths. Never declare a "
        "module discovered only through an import, README, file tree, search result, or another source file; read it "
        "first or omit it from core_modules. source_refs must include the required README and at least one separate "
        "citation for every core_modules path; do not list a core module unless you can provide its field, exact "
        "path, positive line number, and non-empty excerpt. candidate_responsibilities and historical outcomes must stay "
        "empty or be marked unknown unless project material explicitly proves them. Explain how technology choices "
        "serve the project goal. A complete result must include a non-empty goal, core_flow, and source_refs."
    ),
    allowed_tools=frozenset(
        {
            "list_files",
            "grep_files",
            "read_file",
            "read_project_document",
            "file_tree",
            "find_symbols",
            "find_references",
            "get_ast_info",
        }
    ),
    max_steps=18,
    timeout_seconds=1200,
    read_only=True,
    model_timeout_seconds=180.0,
    model_max_retries=0,
)


def _decode_json_object(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise ValueError("Project Analyst did not return a JSON object")
    value, _end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("Project Analyst output must be a JSON object")
    return value


def _normalize_path(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").lstrip("./").casefold()


def _resolve_project_path(value: object, artifact_paths: set[str]) -> str | None:
    """Resolve a model path against the registered manifest, including ZIP roots."""

    normalized = _normalize_path(value)
    if not normalized:
        return None
    exact = next((path for path in artifact_paths if _normalize_path(path) == normalized), None)
    if exact is not None:
        return exact
    matches = [
        path
        for path in artifact_paths
        if _normalize_path(path).endswith("/" + normalized)
    ]
    return matches[0] if len(matches) == 1 else None


def _module_candidate(value: object) -> str:
    """Extract a path when a model appends a prose description after it."""

    text = str(value or "").strip().strip(chr(96))
    for separator in ("：", ":"):
        if separator in text:
            prefix = text.split(separator, 1)[0].strip().strip(chr(96))
            if "/" in prefix or "\\" in prefix or prefix.casefold().endswith(".py"):
                return prefix
    return text


def _validate_and_canonicalize_coverage(
    understanding: dict,
    request: dict,
    tool_trace: object,
) -> None:
    """Require evidence coverage while accepting unique ZIP-root aliases."""

    if not isinstance(tool_trace, list) or any(not isinstance(item, dict) for item in tool_trace):
        raise ValueError("Project Analyst tool trace is invalid")
    artifact_paths = {
        str(item).strip().replace("\\", "/")
        for item in request.get("artifact_paths", [])
        if str(item).strip()
    }
    if not artifact_paths:
        raise ValueError("Project Analyst request is missing the project artifact manifest")
    required_readme = _resolve_project_path(request.get("required_readme_path"), artifact_paths)
    if required_readme is None:
        raise ValueError("Project Analyst request has no readable required README.md")

    read_calls = [
        item
        for item in tool_trace
        if str(item.get("tool_name") or "").strip() in {"read_file", "read_project_document"}
        and isinstance(item.get("input"), dict)
    ]
    if not read_calls:
        raise ValueError("Project Analyst did not read any project files")
    first_path = _resolve_project_path(read_calls[0]["input"].get("path"), artifact_paths)
    if read_calls[0].get("tool_name") != "read_file" or first_path != required_readme:
        raise ValueError("Project Analyst must read the required README.md before other files")
    if read_calls[0].get("ok") is False:
        raise ValueError("Project Analyst could not read the required README.md")

    successful_reads = {
        resolved
        for item in read_calls
        if item.get("ok") is not False
        for resolved in (_resolve_project_path(item["input"].get("path"), artifact_paths),)
        if resolved is not None
    }
    canonical_modules: list[str] = []
    for value in understanding.get("core_modules") or []:
        candidate = _resolve_project_path(_module_candidate(value), artifact_paths)
        if candidate is None:
            raise ValueError(f"Project Analyst core module is not in the project manifest: {value}")
        canonical_modules.append(candidate)
    canonical_modules = list(dict.fromkeys(canonical_modules))
    if not canonical_modules:
        raise ValueError("core_modules must contain the project's major source modules")
    unread = [path for path in canonical_modules if path not in successful_reads]
    if unread:
        raise ValueError(f"Project Analyst did not read declared core modules: {unread}")
    understanding["core_modules"] = canonical_modules

    canonical_refs: list[dict] = []
    for item in understanding.get("source_refs") or []:
        ref = dict(item)
        original_path = ref.get("path")
        canonical = _resolve_project_path(original_path, artifact_paths)
        if canonical is None:
            raise ValueError(f"Project Analyst source reference is not in the project manifest: {original_path}")
        ref["path"] = canonical
        canonical_refs.append(ref)
    if not canonical_refs:
        raise ValueError("source_refs must contain at least one project citation")
    cited = {_normalize_path(item["path"]) for item in canonical_refs}
    missing_citations = [path for path in canonical_modules if _normalize_path(path) not in cited]
    if missing_citations:
        raise ValueError(f"Project Analyst core modules lack source citations: {missing_citations}")
    if _normalize_path(required_readme) not in cited:
        raise ValueError("Project Analyst result does not cite the required README.md")
    understanding["source_refs"] = canonical_refs


def _normalize_confidence_score(score: object, field_name: str) -> float:
    """Convert a model confidence label/number to a validated [0, 1] score."""

    if isinstance(score, str):
        normalized = score.strip().casefold().replace("-", "_").replace(" ", "_")
        if normalized in _QUALITATIVE_CONFIDENCE:
            return _QUALITATIVE_CONFIDENCE[normalized]
        candidate: object = normalized
    else:
        candidate = score

    try:
        value = float(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"confidence_by_field[{field_name!r}] must be a number in [0, 1] or a known quality level; "
            f"received {score!r}"
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence_by_field[{field_name!r}] must be between 0 and 1")
    return value


def _scalar_text(value: object, field_name: str) -> str:
    if isinstance(value, (list, dict, tuple, set)):
        raise ValueError(f"{field_name} must be a string")
    return str(value or "").strip()


def _architecture_text(value: object) -> str:
    if not isinstance(value, list):
        return _scalar_text(value, "architecture")
    if not value or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("architecture must be a string or an array of non-empty strings")
    return "\n".join(item.strip() for item in value)


def _contains_chinese(value: object) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in str(value or ""))


def _require_chinese(value: object, field_name: str) -> None:
    if not _contains_chinese(value):
        raise ValueError(f"{field_name} must be written in Simplified Chinese")


def _normalize_understanding(value: dict) -> dict:
    array_fields = (
        "core_flow",
        "core_modules",
        "technology_choices",
        "technologies",
        "implemented_features",
        "candidate_outcomes",
        "candidate_responsibilities",
        "technical_tags",
        "unknowns",
    )
    normalized = {
        "title": _scalar_text(value.get("title"), "title"),
        "goal": _scalar_text(value.get("goal"), "goal"),
        "architecture": _architecture_text(value.get("architecture")),
        "analysis_summary": _scalar_text(value.get("analysis_summary"), "analysis_summary"),
    }
    for field_name in array_fields:
        raw = value.get(field_name, [])
        if not isinstance(raw, list):
            raise ValueError(f"{field_name} must be an array")
        normalized[field_name] = list(
            dict.fromkeys(str(item or "").strip() for item in raw if str(item or "").strip())
        )
    refs = value.get("source_refs", [])
    if not isinstance(refs, list) or any(not isinstance(item, dict) for item in refs):
        raise ValueError("source_refs must be an array of objects")
    normalized["source_refs"] = [dict(item) for item in refs]
    confidence = value.get("confidence_by_field", {})
    if not isinstance(confidence, dict):
        raise ValueError("confidence_by_field must be an object")
    normalized["confidence_by_field"] = {
        str(key): _normalize_confidence_score(score, str(key)) for key, score in confidence.items()
    }
    if not normalized["title"] or not normalized["analysis_summary"]:
        raise ValueError("title and analysis_summary are required")
    if not normalized["goal"]:
        raise ValueError("goal is required")
    if not normalized["core_flow"]:
        raise ValueError("core_flow must contain at least one step")
    if not normalized["core_modules"]:
        raise ValueError("core_modules must contain the project's major source modules")
    if not normalized["architecture"]:
        raise ValueError("architecture is required")
    if not normalized["implemented_features"]:
        raise ValueError("implemented_features must contain at least one verified feature")
    if not normalized["source_refs"]:
        raise ValueError("source_refs must contain at least one project citation")
    for field_name in ("goal", "architecture", "analysis_summary"):
        _require_chinese(normalized[field_name], field_name)
    for field_name in ("core_flow", "technology_choices", "implemented_features"):
        for index, item in enumerate(normalized[field_name]):
            _require_chinese(item, f"{field_name}[{index}]")
    return normalized


class ProjectAnalystAgent:
    def __init__(self, runner: SubagentRunner) -> None:
        self.runner = runner

    def analyze(self, task: AgentTask, artifacts: list[AgentArtifact]) -> AgentRunResult:
        if task.task_type != PROJECT_ANALYST_SPEC.agent_type:
            raise ValueError("ProjectAnalystAgent only accepts project_analyst tasks")
        actual_types = {artifact.artifact_type for artifact in artifacts}
        if actual_types != PROJECT_ANALYST_INPUT_TYPES or len(artifacts) != 1:
            raise ValueError("Project Analyst requires exactly one project_analysis_request artifact")

        raw_result = self.runner.run(task, artifacts, PROJECT_ANALYST_SPEC)
        if raw_result.status != "completed" or raw_result.output_artifact is None:
            return raw_result
        try:
            payload = _decode_json_object(str(raw_result.output_artifact.content.get("text") or ""))
            understanding = _normalize_understanding(payload)
            request = artifacts[0].content
            _validate_and_canonicalize_coverage(
                understanding,
                request,
                raw_result.output_artifact.content.get("tool_trace"),
            )
            content = {
                "project_id": str(request.get("project_id") or "").strip(),
                "project_version": str(request.get("project_version") or "").strip(),
                "intent": str(request.get("intent") or "overview").strip(),
                "understanding": understanding,
            }
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return AgentRunResult(
                task=task,
                status="failed",
                elapsed_seconds=raw_result.elapsed_seconds,
                tool_call_count=raw_result.tool_call_count,
                token_count=raw_result.token_count,
                token_count_source=raw_result.token_count_source,
                tool_trace=raw_result.tool_trace,
                error=f"Invalid Project Analyst output: {exc}",
            )

        report = AgentArtifact.create(
            "project_understanding",
            domain="career",
            session_id=task.child_session_id,
            created_by=PROJECT_ANALYST_SPEC.agent_type,
            content=content,
            source_artifact_ids=task.input_artifact_ids,
        )
        return AgentRunResult(
            task=task,
            status="completed",
            output_artifact=report,
            elapsed_seconds=raw_result.elapsed_seconds,
            tool_call_count=raw_result.tool_call_count,
            token_count=raw_result.token_count,
            token_count_source=raw_result.token_count_source,
            tool_trace=raw_result.tool_trace,
        )

    def run(self, task: AgentTask, artifacts: list[AgentArtifact]) -> AgentRunResult:
        return self.analyze(task, artifacts)
