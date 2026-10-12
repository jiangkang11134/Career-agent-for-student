from smart_agent.runtime_profiles import resolve_code_run_policy


def test_legacy_profiles_resolve_to_one_harness_policy():
    default = resolve_code_run_policy({"runtimeProfile": "single"})
    legacy_deep = resolve_code_run_policy({"runtimeProfile": "single-deep"})
    assert default == legacy_deep
    assert default.name == "harness"
    assert default.strict_step_verification is True


def test_harness_policy_accepts_bounded_budget_overrides():
    policy = resolve_code_run_policy(
        {
            "maxSteps": 12,
            "exploreStepBudget": 2,
            "executeStepBudget": 5,
            "verifyStepBudget": 2,
            "maxToolCalls": 18,
        }
    )
    assert (policy.max_steps, policy.explore_step_budget) == (12, 2)
    assert (policy.execute_step_budget, policy.verify_step_budget) == (5, 2)
    assert policy.max_tool_calls == 18
