"""Code Review child agent with a structured finding contract."""

from __future__ import annotations

import json
from typing import Any

from smart_agent.agents.contracts import AgentArtifact, AgentFinding, AgentRunResult, AgentTask
from smart_agent.agents.runner import SubagentRunner, SubagentSpec

CODE_REVIEW_INPUT_TYPES = frozenset({
    "task_goal",
    "code_diff",
    "related_file",
    "verification_result",
})

CODE_REVIEW_SPEC = SubagentSpec(
    agent_type="code_review",
    display_name="Code Review Agent",
    domain="code",
    system_prompt=(
        "You are the Code Review Agent. Review correctness, security, behavioral regressions, "
        "boundary conditions, and missing tests. You are read-only. Return exactly one JSON object: "
        '{"summary":"...","findings":[{"severity":"critical|major|minor|info",'
        '"title":"...","message":"...","file_path":"...","line":0,'
        '"rule_id":"...","confidence":0.0}]}. '
        "Use an empty findings array when no issue requires action."
    ),
    allowed_tools=frozenset({
        "read_file",
        "grep_files",
        "file_tree",
        "find_symbols",
        "find_references",
        "diff_viewer",
        "code_review",
    }),
    max_steps=5,
    timeout_seconds=30,
    read_only=True,
)


def _decode_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("Code Review Agent did not return a JSON object.")
    value, _end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("Code Review Agent output must be a JSON object.")
    return value


def _required_output_text(item: dict[str, Any], field_name: str) -> str:
    value = item.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Code Review Agent finding.{field_name} is required.")
    return value.strip()


class CodeReviewAgent:
    """Validate review inputs, run an isolated reviewer, and normalize findings."""

    def __init__(self, runner: SubagentRunner) -> None:
        self.runner = runner

    def review(
        self,
        task: AgentTask,
        artifacts: list[AgentArtifact],
    ) -> AgentRunResult:
        if task.task_type != CODE_REVIEW_SPEC.agent_type:
            raise ValueError("CodeReviewAgent only accepts code_review tasks.")
        unsupported = sorted({artifact.artifact_type for artifact in artifacts} - CODE_REVIEW_INPUT_TYPES)
        if unsupported:
            raise ValueError(
                "Code Review Agent received unauthorized artifact types: "
                + ", ".join(unsupported)
            )

        raw_result = self.runner.run(task, artifacts, CODE_REVIEW_SPEC)
        if raw_result.status != "completed" or raw_result.output_artifact is None:
            return raw_result

        try:
            payload = _decode_json_object(str(raw_result.output_artifact.content.get("text") or ""))
            summary = payload.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("Code Review Agent summary is required.")
            raw_findings = payload.get("findings", [])
            if not isinstance(raw_findings, list):
                raise ValueError("Code Review Agent findings must be an array.")
            if any(not isinstance(item, dict) for item in raw_findings):
                raise ValueError("Each Code Review Agent finding must be an object.")
            findings = tuple(
                AgentFinding.create(
                    severity=_required_output_text(item, "severity"),
                    title=_required_output_text(item, "title"),
                    message=_required_output_text(item, "message"),
                    created_by=CODE_REVIEW_SPEC.agent_type,
                    session_id=task.child_session_id,
                    source_artifact_ids=task.input_artifact_ids,
                    file_path=str(item.get("file_path") or ""),
                    line=int(item.get("line") or 0),
                    rule_id=str(item.get("rule_id") or ""),
                    confidence=float(item.get("confidence", 1.0)),
                )
                for item in raw_findings
                if isinstance(item, dict)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return AgentRunResult(
                task=task,
                status="failed",
                elapsed_seconds=raw_result.elapsed_seconds,
                error=f"Invalid Code Review Agent output: {exc}",
            )

        report = AgentArtifact.create(
            "code_review_report",
            domain="code",
            session_id=task.child_session_id,
            created_by=CODE_REVIEW_SPEC.agent_type,
            content={
                "summary": summary.strip(),
                "findings": [finding.to_dict() for finding in findings],
            },
            source_artifact_ids=task.input_artifact_ids,
        )
        return AgentRunResult(
            task=task,
            status="completed",
            output_artifact=report,
            findings=findings,
            elapsed_seconds=raw_result.elapsed_seconds,
        )
