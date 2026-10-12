"""Unified entry point for bounded code-understanding modes."""

from __future__ import annotations

from typing import Literal

from smart_agent.agents.contracts import AgentArtifact, AgentRunResult, AgentTask
from smart_agent.agents.runner import SubagentRunner

CodeAgentMode = Literal["code_review", "project_analysis", "project_modify"]


class CodeAgent:
    """Run one explicit code-understanding mode in an isolated runner.

    Mode-specific output contracts remain in their own modules. This class is
    the single execution boundary used by callers that need code analysis.
    """

    def __init__(self, runner: SubagentRunner) -> None:
        self.runner = runner

    def run(
        self,
        task: AgentTask,
        artifacts: list[AgentArtifact],
        *,
        mode: CodeAgentMode | str | None = None,
        spec=None,
    ) -> AgentRunResult:
        selected = str(mode or task.task_type).strip().casefold()
        if selected == "code_review":
            from smart_agent.agents.code_review import CodeReviewAgent

            return CodeReviewAgent(self.runner).review(task, artifacts)
        if selected in {"project_analysis", "project_analyst"}:
            from smart_agent.agents.project_analyst import ProjectAnalystAgent

            return ProjectAnalystAgent(self.runner).analyze(task, artifacts)
        if selected == "project_modify":
            if spec is None:
                raise ValueError("project_modify requires an explicit writable policy")
            return self.runner.run(task, artifacts, spec)
        raise ValueError(f"Unsupported CodeAgent mode: {selected or '<empty>'}")

    def review(self, task: AgentTask, artifacts: list[AgentArtifact]) -> AgentRunResult:
        return self.run(task, artifacts, mode="code_review")

    def analyze(self, task: AgentTask, artifacts: list[AgentArtifact]) -> AgentRunResult:
        return self.run(task, artifacts, mode="project_analysis")
