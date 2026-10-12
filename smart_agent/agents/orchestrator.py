"""Goal-driven Root Supervisor over the existing Domain Agent runtime.

The supervisor owns admission, planning, delegation, structured verification,
and bounded re-planning. Code and Career agents keep their existing
think/execute/verify loop, permissions, skills, and tools.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Protocol

from smart_agent.agents.contracts import AgentArtifact, AgentRunResult, AgentTask
from smart_agent.agents.runner import SubagentRunner, SubagentSpec
from smart_agent.domains import AgentDomain, assemble_capabilities, require_explicit_domain
from smart_agent.intent_parser import parse_intent
from smart_agent.types import ModelAdapter

ROOT_CONTRACT_VERSION = "2.0"
_STEP_STATUSES = frozenset({"pending", "running", "completed", "failed"})
_VERIFICATION_OUTCOMES = frozenset({"passed", "retry", "blocked"})
_RUN_STATUSES = frozenset({"completed", "failed", "timed_out"})
_SUPERVISOR_PHASES = frozenset({"goal", "plan", "dispatch", "execute", "verify", "replan", "finish"})
_AUTO_MODES = frozenset({"auto", "supervisor"})
_FORCE_MODES = frozenset({"force", "supervisor-force", "p-e-rp", "perp"})
_OFF_MODES = frozenset({"", "off", "single", "direct"})


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _text(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required.")
    return normalized


def _domain(value: AgentDomain | str) -> str:
    return require_explicit_domain(value).value


@dataclass(frozen=True, slots=True)
class RootPlanStep:
    """One bounded Domain Agent assignment in a root plan."""

    step_id: str
    domain: str
    objective: str
    input_artifact_ids: tuple[str, ...]
    status: str = "pending"
    attempt: int = 1
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _text(self.step_id, "step_id"))
        object.__setattr__(self, "domain", _domain(self.domain))
        object.__setattr__(self, "objective", _text(self.objective, "objective"))
        ids = tuple(_text(item, "input_artifact_id") for item in self.input_artifact_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("input_artifact_ids cannot contain duplicates.")
        object.__setattr__(self, "input_artifact_ids", ids)
        object.__setattr__(self, "depends_on", tuple(_text(item, "dependency") for item in self.depends_on))
        status = str(self.status or "").strip().lower()
        if status not in _STEP_STATUSES:
            raise ValueError(f"Unsupported root plan step status: {self.status}")
        object.__setattr__(self, "status", status)
        if self.attempt < 1:
            raise ValueError("attempt must be at least 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "domain": self.domain,
            "objective": self.objective,
            "input_artifact_ids": list(self.input_artifact_ids),
            "status": self.status,
            "attempt": self.attempt,
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True, slots=True)
class RootPlan:
    """Versioned root plan. It contains intent, not raw conversation history."""

    plan_id: str
    goal: str
    domain: str
    steps: tuple[RootPlanStep, ...]
    revision: int = 1
    max_replans: int = 1
    created_at: float = 0.0
    schema_version: str = ROOT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        object.__setattr__(self, "goal", _text(self.goal, "goal"))
        object.__setattr__(self, "domain", _domain(self.domain))
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("A root plan requires at least one step.")
        if any(step.domain != self.domain for step in steps):
            raise ValueError("A manually selected Domain plan cannot mix Code and Career steps.")
        step_ids = [step.step_id for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Root plan step IDs must be unique.")
        object.__setattr__(self, "steps", steps)
        if self.revision < 1:
            raise ValueError("revision must be at least 1.")
        if self.max_replans < 0:
            raise ValueError("max_replans cannot be negative.")
        if self.schema_version != ROOT_CONTRACT_VERSION:
            raise ValueError(f"Unsupported root contract version: {self.schema_version}")

    @classmethod
    def create(
        cls,
        goal: str,
        *,
        domain: AgentDomain | str,
        input_artifact_ids: tuple[str, ...] | list[str],
        max_replans: int = 1,
    ) -> "RootPlan":
        selected = _domain(domain)
        return cls(
            plan_id=_new_id("plan"),
            goal=goal,
            domain=selected,
            steps=(
                RootPlanStep(
                    step_id=_new_id("step"),
                    domain=selected,
                    objective=goal,
                    input_artifact_ids=tuple(input_artifact_ids),
                ),
            ),
            max_replans=max_replans,
            created_at=time.time(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "domain": self.domain,
            "steps": [step.to_dict() for step in self.steps],
            "revision": self.revision,
            "max_replans": self.max_replans,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class SupervisorDecision:
    """Deterministic admission decision for the optional Root Supervisor."""

    mode: str
    reason: str
    complexity: str

    def __post_init__(self) -> None:
        mode = str(self.mode or "").strip().lower()
        if mode not in {"direct", "supervised"}:
            raise ValueError(f"Unsupported supervisor decision: {self.mode}")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        complexity = str(self.complexity or "unknown").strip().lower()
        if complexity not in {"simple", "moderate", "complex", "unknown"}:
            raise ValueError(f"Unsupported task complexity: {self.complexity}")
        object.__setattr__(self, "complexity", complexity)

    @property
    def should_supervise(self) -> bool:
        return self.mode == "supervised"

    def to_dict(self) -> dict[str, str]:
        return {"mode": self.mode, "reason": self.reason, "complexity": self.complexity}


def decide_supervision(
    goal: str,
    configured_mode: str | None,
    *,
    force: bool = False,
) -> SupervisorDecision:
    """Choose the direct path for simple work without calling a root model."""
    normalized_goal = _text(goal, "goal")
    mode = str(configured_mode or "").strip().lower()
    complexity = parse_intent(normalized_goal).complexity_hint
    if force or mode in _FORCE_MODES:
        return SupervisorDecision(
            mode="supervised",
            reason="Supervisor execution was explicitly requested.",
            complexity=complexity,
        )
    if mode in _OFF_MODES:
        return SupervisorDecision(
            mode="direct",
            reason="Supervisor execution is disabled.",
            complexity=complexity,
        )
    if mode not in _AUTO_MODES:
        raise ValueError(
            "Supervisor mode must be off, auto, force, supervisor, supervisor-force, or legacy p-e-rp."
        )
    if complexity == "simple":
        return SupervisorDecision(
            mode="direct",
            reason="Simple task bypasses root planning and runs in the selected Domain Agent.",
            complexity=complexity,
        )
    return SupervisorDecision(
        mode="supervised",
        reason="Non-simple task uses structured planning and verification.",
        complexity=complexity,
    )


@dataclass(frozen=True, slots=True)
class RootVerification:
    """Structured verification of one Domain Agent execution."""

    outcome: str
    reason: str
    next_action: str = ""
    evidence: tuple[str, ...] = ()
    retryable: bool = False
    state_changed: bool = False
    plan_revision: int = 1

    def __post_init__(self) -> None:
        outcome = str(self.outcome or "").strip().lower()
        if outcome not in _VERIFICATION_OUTCOMES:
            raise ValueError(f"Unsupported root verification outcome: {self.outcome}")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(self, "next_action", str(self.next_action or "").strip())
        object.__setattr__(self, "evidence", tuple(_text(item, "evidence") for item in self.evidence))
        if outcome == "passed" and self.retryable:
            raise ValueError("A passed verification cannot be retryable.")
        if outcome == "blocked" and self.retryable:
            raise ValueError("A blocked verification cannot be retryable.")
        if self.plan_revision < 1:
            raise ValueError("plan_revision must be at least 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "next_action": self.next_action,
            "evidence": list(self.evidence),
            "retryable": self.retryable,
            "state_changed": self.state_changed,
            "plan_revision": self.plan_revision,
        }


@dataclass(frozen=True, slots=True)
class RootRunResult:
    """Structured result returned by the root controller to its caller."""

    status: str
    plan: RootPlan
    verifications: tuple[RootVerification, ...]
    domain_runs: tuple[AgentRunResult, ...]
    output_artifact: AgentArtifact | None
    replan_count: int
    elapsed_seconds: float
    phase_trace: tuple[str, ...]
    error: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"completed", "blocked", "failed"}:
            raise ValueError(f"Unsupported root run status: {self.status}")
        if self.replan_count < 0:
            raise ValueError("replan_count cannot be negative.")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds cannot be negative.")
        if not self.phase_trace or self.phase_trace[0] != "goal" or self.phase_trace[-1] != "finish":
            raise ValueError("Root phase_trace must run from goal to finish.")
        if any(phase not in _SUPERVISOR_PHASES for phase in self.phase_trace):
            raise ValueError("Root phase_trace contains an unsupported phase.")
        if self.output_artifact is not None and self.output_artifact.domain != self.plan.domain:
            raise ValueError("Root output artifact must belong to the selected Domain.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "plan": self.plan.to_dict(),
            "verifications": [item.to_dict() for item in self.verifications],
            "domain_runs": [item.to_dict() for item in self.domain_runs],
            "output_artifact": self.output_artifact.to_dict() if self.output_artifact else None,
            "replan_count": self.replan_count,
            "elapsed_seconds": self.elapsed_seconds,
            "phase_trace": list(self.phase_trace),
            "error": self.error,
        }


class RootPlanner(Protocol):
    def create_plan(
        self,
        goal: str,
        *,
        domain: AgentDomain,
        input_artifact_ids: tuple[str, ...],
        max_replans: int,
    ) -> RootPlan: ...

    def replan(self, plan: RootPlan, verification: RootVerification) -> RootPlan: ...


class RootVerifier(Protocol):
    def verify(self, plan: RootPlan, run: AgentRunResult) -> RootVerification: ...


class DeterministicRootPlanner:
    """Offline planner used as a safe baseline and test double."""

    def create_plan(
        self,
        goal: str,
        *,
        domain: AgentDomain,
        input_artifact_ids: tuple[str, ...],
        max_replans: int,
    ) -> RootPlan:
        return RootPlan.create(
            goal,
            domain=domain,
            input_artifact_ids=input_artifact_ids,
            max_replans=max_replans,
        )

    def replan(self, plan: RootPlan, verification: RootVerification) -> RootPlan:
        revised_steps = tuple(
            replace(
                step,
                objective=(
                    f"{step.objective}\n\n"
                    f"Replan instruction: {verification.next_action or verification.reason}"
                ),
                status="pending",
                attempt=step.attempt + 1,
            )
            for step in plan.steps
        )
        return replace(plan, steps=revised_steps, revision=plan.revision + 1)


class DeterministicRootVerifier:
    """Verify bounded execution facts without an additional model call."""

    def verify(self, plan: RootPlan, run: AgentRunResult) -> RootVerification:
        if run.status not in _RUN_STATUSES:
            raise ValueError(f"Unsupported Domain Agent status: {run.status}")
        if run.status != "completed":
            return RootVerification(
                outcome="retry",
                reason=run.error or f"Domain Agent ended with status={run.status}.",
                next_action="Retry with the execution failure and a narrower next action.",
                evidence=(f"runner_status={run.status}",),
                retryable=True,
                plan_revision=plan.revision,
            )
        if run.output_artifact is None:
            return RootVerification(
                outcome="retry",
                reason="Domain Agent completed without an output artifact.",
                next_action="Retry and require a non-empty verified output artifact.",
                evidence=("output_artifact=missing",),
                retryable=True,
                plan_revision=plan.revision,
            )
        try:
            run.output_artifact.verify_integrity()
        except ValueError as exc:
            return RootVerification(
                outcome="blocked",
                reason=f"Domain output artifact failed integrity verification: {exc}",
                evidence=("artifact_integrity=failed",),
                plan_revision=plan.revision,
            )
        content = run.output_artifact.content
        text = str(content.get("text") or "").strip()
        if not text:
            return RootVerification(
                outcome="retry",
                reason="Domain output artifact contains no final response text.",
                next_action="Retry and return a non-empty final response after Domain verification.",
                evidence=("output_text=empty",),
                retryable=True,
                plan_revision=plan.revision,
            )
        declared = str(content.get("verification") or "").strip().lower()
        if declared in {"failed", "blocked", "pending", "incomplete"}:
            return RootVerification(
                outcome="retry",
                reason=f"Domain Agent declared verification={declared}.",
                next_action="Resolve the reported verification gap before completing the goal.",
                evidence=("runner_status=completed", f"domain_verification={declared}"),
                retryable=True,
                state_changed=bool(content.get("state_changed", False)),
                plan_revision=plan.revision,
            )
        return RootVerification(
            outcome="passed",
            reason="Domain execution completed with an intact, non-empty output artifact.",
            evidence=(
                "runner_status=completed",
                "artifact_integrity=passed",
                "output_text=present",
            ),
            state_changed=bool(content.get("state_changed", False)),
            plan_revision=plan.revision,
        )


def _model_json(model: ModelAdapter, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call a root model without tools and parse one JSON object."""
    step = model.next(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]
    )
    if getattr(step, "type", "") != "assistant":
        raise ValueError("Root model must return an assistant JSON response, not a tool call.")
    raw = str(getattr(step, "content", "") or "").strip()
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if match:
        raw = match.group(1)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Root model must return one valid JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Root model JSON response must be an object.")
    return parsed


class ModelRootPlanner:
    """LLM-backed root planner with strict JSON and manual-Domain guards."""

    def __init__(self, model: ModelAdapter) -> None:
        self.model = model

    def create_plan(
        self,
        goal: str,
        *,
        domain: AgentDomain,
        input_artifact_ids: tuple[str, ...],
        max_replans: int,
    ) -> RootPlan:
        selected_domain = require_explicit_domain(domain)
        payload = _model_json(
            self.model,
            "You are the planning phase of a Smart Agent Root Supervisor. Return JSON only. "
            "The user has already selected the Domain. Produce exactly one bounded delegation step; "
            "do not execute tools, inspect private data, or switch Domains. Schema: "
            '{"objective":"short executable objective","domain":"selected domain"}.',
            {
                "goal": goal,
                "selected_domain": selected_domain.value,
                "input_artifact_ids": list(input_artifact_ids),
            },
        )
        requested_domain = require_explicit_domain(payload.get("domain") or selected_domain.value)
        if requested_domain is not selected_domain:
            raise ValueError("Root planner cannot change the manually selected Domain.")
        objective = _text(payload.get("objective"), "objective")
        plan = RootPlan.create(
            goal,
            domain=selected_domain,
            input_artifact_ids=input_artifact_ids,
            max_replans=max_replans,
        )
        return replace(plan, steps=(replace(plan.steps[0], objective=objective),))

    def replan(self, plan: RootPlan, verification: RootVerification) -> RootPlan:
        payload = _model_json(
            self.model,
            "You are the conditional re-planning phase of a Smart Agent Root Supervisor. Return JSON only. "
            "Keep the selected Domain unchanged and produce one narrower executable objective. "
            'Schema: {"objective":"revised objective"}.',
            {"plan": plan.to_dict(), "verification": verification.to_dict()},
        )
        objective = _text(payload.get("objective"), "objective")
        return replace(
            plan,
            revision=plan.revision + 1,
            steps=tuple(
                replace(step, objective=objective, status="pending", attempt=step.attempt + 1)
                for step in plan.steps
            ),
        )


def _domain_spec(domain: AgentDomain) -> SubagentSpec:
    capabilities = assemble_capabilities(domain, "default")
    display_name = "Code Agent" if domain is AgentDomain.CODE else "Career Agent"
    prompt = (
        f"You are the inner {display_name} delegated by a goal-driven Root Supervisor. "
        "Use the supplied goal and artifacts only. Follow the Smart Agent think-execute-verify loop: "
        "inspect the task, choose the smallest valid tool action, execute it, and verify the result before "
        "finishing. Return a concise structured result with status, summary, verification, blockers, and next_actions. "
        "Do not plan work for another Domain or access cross-domain data."
    )
    return SubagentSpec(
        agent_type=f"{domain.value}_agent",
        display_name=display_name,
        domain=domain.value,
        system_prompt=prompt,
        allowed_tools=capabilities.tool_names,
        max_steps=24 if domain is AgentDomain.CODE else 20,
        timeout_seconds=300.0,
        read_only=False,
    )


DOMAIN_AGENT_SPECS = {
    AgentDomain.CODE: _domain_spec(AgentDomain.CODE),
    AgentDomain.CAREER: _domain_spec(AgentDomain.CAREER),
}


class RootOrchestrator:
    """Run Goal -> Plan -> Dispatch -> Execute -> Verify -> conditional Re-plan.

    The supervisor is deliberately model-agnostic. The planner may use a model,
    while verification is deterministic by default and never adds a success-path
    model call.
    """

    def __init__(
        self,
        *,
        runner: SubagentRunner,
        planner: RootPlanner | None = None,
        verifier: RootVerifier | None = None,
        max_replans: int = 1,
    ) -> None:
        if max_replans < 0:
            raise ValueError("max_replans cannot be negative.")
        self.runner = runner
        self.planner = planner or DeterministicRootPlanner()
        self.verifier = verifier or DeterministicRootVerifier()
        self.max_replans = max_replans

    def run(
        self,
        goal: str,
        *,
        domain: AgentDomain | str,
        parent_session_id: str,
        input_artifacts: tuple[AgentArtifact, ...] | list[AgentArtifact] = (),
        max_replans: int | None = None,
    ) -> RootRunResult:
        started = time.monotonic()
        selected_domain = require_explicit_domain(domain)
        normalized_goal = _text(goal, "goal")
        parent_session_id = _text(parent_session_id, "parent_session_id")
        extras = list(input_artifacts)
        if any(artifact.domain != selected_domain.value for artifact in extras):
            raise ValueError("RootOrchestrator cannot pass cross-domain artifacts to a Domain Agent.")
        request = AgentArtifact.create(
            "root_request",
            domain=selected_domain.value,
            session_id=parent_session_id,
            created_by="root_agent",
            content={"goal": normalized_goal, "selected_domain": selected_domain.value},
            source_artifact_ids=[artifact.artifact_id for artifact in extras],
        )
        artifacts_by_id = {artifact.artifact_id: artifact for artifact in [request, *extras]}
        replans_allowed = self.max_replans if max_replans is None else int(max_replans)
        if replans_allowed < 0:
            raise ValueError("max_replans cannot be negative.")
        plan = self.planner.create_plan(
            normalized_goal,
            domain=selected_domain,
            input_artifact_ids=(request.artifact_id, *[artifact.artifact_id for artifact in extras]),
            max_replans=replans_allowed,
        )
        runs: list[AgentRunResult] = []
        verifications: list[RootVerification] = []
        phase_trace = ["goal", "plan"]
        replan_count = 0

        while True:
            step = plan.steps[0]
            plan = replace(
                plan,
                steps=(replace(step, status="running"), *plan.steps[1:]),
            )
            step = plan.steps[0]
            step_inputs = [artifacts_by_id[item] for item in step.input_artifact_ids]
            spec = DOMAIN_AGENT_SPECS[selected_domain]
            phase_trace.append("dispatch")
            task = AgentTask.create(
                spec.agent_type,
                domain=selected_domain.value,
                description=step.objective,
                parent_session_id=parent_session_id,
                assigned_agent=spec.agent_type,
                input_artifact_ids=step.input_artifact_ids,
                allowed_tools=spec.allowed_tools,
                timeout_seconds=spec.timeout_seconds,
                max_steps=spec.max_steps,
            )
            phase_trace.append("execute")
            run = self.runner.run(task, step_inputs, spec)
            runs.append(run)
            phase_trace.append("verify")
            verification = self.verifier.verify(plan, run)
            if verification.outcome == "passed" and (
                run.status != "completed" or run.output_artifact is None
            ):
                verification = RootVerification(
                    outcome="retry",
                    reason=(
                        f"Execution facts override the completion claim: status={run.status}, "
                        f"output_artifact={'present' if run.output_artifact else 'missing'}."
                    ),
                    next_action="Retry and require a verified output artifact.",
                    evidence=(
                        f"runner_status={run.status}",
                        f"output_artifact={'present' if run.output_artifact else 'missing'}",
                    ),
                    retryable=True,
                    plan_revision=plan.revision,
                )
            verifications.append(verification)
            final_step_status = "completed" if verification.outcome == "passed" else "failed"
            plan = replace(
                plan,
                steps=(replace(plan.steps[0], status=final_step_status), *plan.steps[1:]),
            )
            if verification.outcome == "passed":
                output = self._parent_result_artifact(
                    selected_domain,
                    parent_session_id,
                    run,
                    source_artifact_ids=(request.artifact_id,),
                )
                phase_trace.append("finish")
                return RootRunResult(
                    status="completed",
                    plan=plan,
                    verifications=tuple(verifications),
                    domain_runs=tuple(runs),
                    output_artifact=output,
                    replan_count=replan_count,
                    elapsed_seconds=time.monotonic() - started,
                    phase_trace=tuple(phase_trace),
                )
            if (
                verification.outcome == "blocked"
                or not verification.retryable
                or replan_count >= replans_allowed
            ):
                phase_trace.append("finish")
                return RootRunResult(
                    status="blocked",
                    plan=plan,
                    verifications=tuple(verifications),
                    domain_runs=tuple(runs),
                    output_artifact=None,
                    replan_count=replan_count,
                    elapsed_seconds=time.monotonic() - started,
                    phase_trace=tuple(phase_trace),
                    error=verification.reason,
                )
            prior = self._parent_result_artifact(
                selected_domain,
                parent_session_id,
                run,
                source_artifact_ids=(request.artifact_id,),
            )
            artifacts_by_id[prior.artifact_id] = prior
            phase_trace.append("replan")
            plan = self.planner.replan(plan, verification)
            plan = replace(
                plan,
                steps=tuple(
                    replace(step, input_artifact_ids=(*step.input_artifact_ids, prior.artifact_id))
                    for step in plan.steps
                ),
            )
            replan_count += 1

    @staticmethod
    def _parent_result_artifact(
        domain: AgentDomain,
        parent_session_id: str,
        run: AgentRunResult,
        *,
        source_artifact_ids: tuple[str, ...],
    ) -> AgentArtifact:
        return AgentArtifact.create(
            "domain_agent_result",
            domain=domain.value,
            session_id=parent_session_id,
            created_by="root_agent",
            content={
                "task_id": run.task.task_id,
                "assigned_agent": run.task.assigned_agent,
                "status": run.status,
                "child_session_id": run.task.child_session_id,
                "output": run.output_artifact.content if run.output_artifact else {},
                "findings": [finding.to_dict() for finding in run.findings],
                "elapsed_seconds": run.elapsed_seconds,
                "error": run.error,
            },
            source_artifact_ids=source_artifact_ids
            + ((run.output_artifact.artifact_id,) if run.output_artifact else ()),
        )


def run_orchestrated_turn(
    *,
    model: ModelAdapter,
    tools: Any,
    messages: list[dict[str, Any]],
    cwd: str,
    permissions: Any | None = None,
    session: Any | None = None,
    runtime: dict[str, Any] | None = None,
    on_progress_message: Any | None = None,
) -> list[dict[str, Any]]:
    """Adapt a normal Agent turn to the optional goal-driven Root Supervisor.

    The adapter keeps the existing public ``run_agent_turn`` return shape so
    TUI and headless callers can opt in through runtime configuration without
    learning a second message protocol. The child Domain Agent still uses the
    same model adapter and ToolRegistry through ``SubagentRunner``.
    """
    runtime = dict(runtime or {})
    active_domain = require_explicit_domain(runtime.get("activeDomain") or "code")
    goal = next(
        (
            str(message.get("content") or "").strip()
            for message in reversed(messages)
            if message.get("role") == "user" and str(message.get("content") or "").strip()
        ),
        "Continue the selected Domain task.",
    )
    parent_session_id = str(
        getattr(session, "session_id", "")
        or runtime.get("sessionId")
        or f"root-{uuid.uuid4().hex[:12]}"
    ).strip()
    max_replans = runtime.get("supervisorMaxReplans", runtime.get("rootMaxReplans", 1))
    try:
        max_replans = max(0, int(max_replans))
    except (TypeError, ValueError):
        max_replans = 1
    if on_progress_message:
        on_progress_message("Supervisor: planning the selected Domain task.")
    from smart_agent.model_registry import create_model_adapter
    from smart_agent.tooling import ToolRegistry

    root_model_name = str(
        runtime.get("supervisorModel")
        or runtime.get("rootModel")
        or runtime.get("model")
        or ""
    ).strip()
    root_runtime = {
        **runtime,
        "model": root_model_name,
        "configuredModel": root_model_name,
        "orchestratorMode": "off",
        "isRootAgent": True,
    }
    root_model = create_model_adapter(
        model=root_model_name,
        tools=ToolRegistry([]),
        runtime=root_runtime,
    )
    result = RootOrchestrator(
        runner=SubagentRunner(
            cwd=cwd,
            runtime={**runtime, "orchestratorMode": "off"},
            tools=tools,
            permissions=permissions,
        ),
        planner=ModelRootPlanner(root_model),
        max_replans=max_replans,
    ).run(
        goal,
        domain=active_domain,
        parent_session_id=parent_session_id,
    )
    output = result.output_artifact.content.get("output", {}) if result.output_artifact else {}
    response = str(output.get("text") or "").strip() if isinstance(output, dict) else ""
    if not response:
        response = result.error or f"Root Agent ended with status={result.status}."
    if on_progress_message:
        on_progress_message(
            f"Supervisor: verification finished with status={result.status}, replans={result.replan_count}."
        )
    return [*messages, {"role": "assistant", "content": response}]


__all__ = [
    "DOMAIN_AGENT_SPECS",
    "DeterministicRootPlanner",
    "DeterministicRootVerifier",
    "ModelRootPlanner",
    "RootOrchestrator",
    "RootPlan",
    "RootPlanStep",
    "RootVerification",
    "RootRunResult",
    "SupervisorDecision",
    "decide_supervision",
    "run_orchestrated_turn",
]
