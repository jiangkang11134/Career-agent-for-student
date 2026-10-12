"""Unified ReAct runtime policy for Smart Agent turns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class CodeRunPolicy:
    name: str = "harness"
    max_steps: int | None = 24
    # Deprecated compatibility knobs; phase control no longer uses them.
    explore_step_budget: int = 3
    execute_step_budget: int = 10
    verify_step_budget: int = 4
    max_tool_calls: int | None = 40
    empty_response_retry_limit: int = 2
    recoverable_thinking_retry_limit: int = 3
    working_memory_ttl_seconds: float | None = 1800
    working_memory_importance: float = 1.0
    strict_step_verification: bool = True
    widen_after_step: int | None = None
    widening_step_bonus: int = 0


DEFAULT_CODE_RUN_POLICY = CodeRunPolicy()


def _positive_int(value: object, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _bounded_int(value: object, default: int, *, minimum: int = 1, maximum: int = 200) -> int:
    parsed = _positive_int(value, default)
    if parsed is None:
        parsed = default
    return min(max(parsed, minimum), maximum)


def resolve_code_run_policy(
    runtime: Mapping[str, Any] | None,
    *,
    fallback_max_steps: int | None = None,
) -> CodeRunPolicy:
    """Resolve one unified harness policy.

    Legacy runtimeProfile values are ignored on purpose.  Explicit numeric
    runtime knobs still work so tests and subagents can narrow their budget.
    """
    runtime = runtime or {}
    max_steps = _positive_int(runtime.get("maxSteps"), DEFAULT_CODE_RUN_POLICY.max_steps)
    if fallback_max_steps is not None:
        max_steps = _positive_int(fallback_max_steps, max_steps)
    return replace(
        DEFAULT_CODE_RUN_POLICY,
        max_steps=max_steps,
        explore_step_budget=_bounded_int(runtime.get("exploreStepBudget"), DEFAULT_CODE_RUN_POLICY.explore_step_budget, maximum=20),
        execute_step_budget=_bounded_int(runtime.get("executeStepBudget"), DEFAULT_CODE_RUN_POLICY.execute_step_budget, maximum=100),
        verify_step_budget=_bounded_int(runtime.get("verifyStepBudget"), DEFAULT_CODE_RUN_POLICY.verify_step_budget, maximum=50),
        max_tool_calls=_positive_int(runtime.get("maxToolCalls"), DEFAULT_CODE_RUN_POLICY.max_tool_calls),
    )
