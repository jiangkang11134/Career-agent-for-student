"""Career child-agent policies with explicit artifact boundaries."""

from __future__ import annotations

from smart_agent.agents.contracts import AgentArtifact, AgentRunResult, AgentTask
from smart_agent.agents.runner import SubagentRunner, SubagentSpec

REALISTIC_SIMULATION_MODE = "realistic_simulation"
GAP_ANALYSIS_MODE = "gap_analysis"
INTERVIEW_MODES = frozenset({REALISTIC_SIMULATION_MODE, GAP_ANALYSIS_MODE})
SIMULATION_INTERVIEWER_INPUT_TYPES = frozenset({"jd", "verified_resume", "interview_knowledge"})
GAP_INTERVIEWER_INPUT_TYPES = frozenset({"interview_knowledge"})
INTERVIEWER_INPUT_TYPES = SIMULATION_INTERVIEWER_INPUT_TYPES
APPLICATION_INPUT_TYPES = frozenset({"application_url", "verified_resume_pdf"})


INTERVIEWER_SPEC = SubagentSpec(
    agent_type="interviewer",
    display_name="Interviewer Agent",
    domain="career",
    system_prompt=(
        "You are an isolated, long-lived Interviewer Agent with one immutable session mode. In realistic_simulation "
        "you may see only the supplied JD, verified resume, and frozen interview knowledge. In gap_analysis you may "
        "omit JD/resume; InterviewRuntime performs tenant-scoped read-only local RAG retrieval before each question. "
        "Never inspect the parent Career session, raw personal materials, project repositories, or resume generation. "
        "Keep knowledge-source citations in every question and evaluation. "
        "Create or resume only an isolated InterviewSession through the dedicated interview tools. Treat the user's "
        "project ownership as assumed; evaluate only knowledge answers, and never accept user-supplied scores as "
        "ground truth. You may write only the isolated interview session store. Return structured JSON."
    ),
    allowed_tools=frozenset({
        "create_interview_session",
        "get_interview_session",
        "next_interview_question",
        "process_interview_answer",
        "generate_interview_depth_question",
        "generate_interview_breadth_question",
        "submit_interview_answer",
        "finish_interview_session",
    }),
    max_steps=8,
    timeout_seconds=60,
    read_only=False,
)


APPLICATION_SPEC = SubagentSpec(
    agent_type="application",
    display_name="Application Agent",
    domain="career",
    system_prompt=(
        "You are an isolated semi-automated Application Agent. You may see only an application URL and a verified "
        "resume PDF artifact. Fill only fields directly supported by the resume. Leave unsupported, sensitive, or "
        "ambiguous fields blank and request user confirmation. Never bypass CAPTCHA, submit an application, or alter "
        "the parent Career session without an explicit approval boundary. Return a structured field plan."
    ),
    allowed_tools=frozenset({"web_fetch"}),
    max_steps=8,
    timeout_seconds=60,
    read_only=False,
)


class _CareerChildAgent:
    spec: SubagentSpec
    allowed_artifact_types: frozenset[str]

    def __init__(self, runner: SubagentRunner) -> None:
        self.runner = runner

    def run(self, task: AgentTask, artifacts: list[AgentArtifact]) -> AgentRunResult:
        if task.task_type != self.spec.agent_type:
            raise ValueError(f"{type(self).__name__} only accepts {self.spec.agent_type} tasks.")
        actual_types = {artifact.artifact_type for artifact in artifacts}
        if actual_types != self.allowed_artifact_types or len(artifacts) != len(self.allowed_artifact_types):
            raise ValueError(
                f"{self.spec.display_name} requires exactly these artifact types: "
                + ", ".join(sorted(self.allowed_artifact_types))
            )
        return self.runner.run(task, artifacts, self.spec)


class InterviewerAgent(_CareerChildAgent):
    spec = INTERVIEWER_SPEC

    def __init__(
        self,
        runner: SubagentRunner,
        *,
        mode: str = REALISTIC_SIMULATION_MODE,
    ) -> None:
        super().__init__(runner)
        self.mode = str(getattr(mode, "value", mode) or "").strip()
        if self.mode not in INTERVIEW_MODES:
            raise ValueError(f"Unsupported interview mode: {self.mode or '<empty>'}")
        self.allowed_artifact_types = (
            GAP_INTERVIEWER_INPUT_TYPES
            if self.mode == GAP_ANALYSIS_MODE
            else SIMULATION_INTERVIEWER_INPUT_TYPES
        )


class ApplicationAgent(_CareerChildAgent):
    spec = APPLICATION_SPEC
    allowed_artifact_types = APPLICATION_INPUT_TYPES
