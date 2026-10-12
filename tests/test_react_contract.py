from types import SimpleNamespace

from smart_agent.context_manager import ContextManager
from smart_agent.loop_orchestrator import (
    _contract_tool_violation,
    _infer_task_contract,
    run_agent_turn,
)
from smart_agent.tooling import ToolContext, ToolDefinition, ToolRegistry, ToolResult
from smart_agent.tools.run_command import _validate as validate_run_command
from smart_agent.tools.run_command import run_command_tool
from smart_agent.turn_kernel import TurnRecurrentState, TurnTaskContract
from smart_agent.types import AgentStep
from smart_agent.verification_guard import VerificationLedger
from smart_agent.workspace import resolve_tool_path


def _tool(name: str, *, read_only: bool):
    return SimpleNamespace(name=name, is_read_only=read_only)


def test_ordinary_code_question_is_answerable_without_execution_contract():
    contract = _infer_task_contract(
        [{"role": "user", "content": "你知道我叫什么吗？"}],
        {},
        "code",
    )
    assert contract.mode == "conversation"
    assert contract.can_finalize


def test_explicit_mutation_requires_execute_and_verify():
    contract = _infer_task_contract(
        [{"role": "user", "content": "implement a function"}],
        {},
        "code",
    )
    assert contract.requires_execute
    assert not contract.can_finalize
    state = TurnRecurrentState(max_steps=5, task_contract=contract)
    state.record_execution_action("edit_file")
    assert state.task_contract.verification_required
    state.record_verification_action("test_runner", ok=True)
    assert state.task_contract.can_finalize


def test_new_mutation_invalidates_previous_verification():
    contract = TurnTaskContract(
        mode="mutation", requires_execute=True, requires_verify=True
    )
    state = TurnRecurrentState(max_steps=5, task_contract=contract)
    state.record_execution_action("edit_file")
    state.record_verification_action("test_runner", ok=True)
    assert state.task_contract.can_finalize
    state.record_execution_action("edit_file")
    assert state.task_contract.verification_required


def test_complete_tool_is_allowed_as_a_model_selected_react_action():
    contract = TurnTaskContract()
    message = _contract_tool_violation(
        tool=_tool("complete_task", read_only=False),
        contract=contract,
    )
    assert message == ""


def test_recovered_tool_failure_does_not_block_final_ledger():
    ledger = VerificationLedger()
    ledger.record_call("failed", "test_runner", {})
    ledger.record_result("failed", "test_runner", False, "one test failed")
    ledger.record_call("passed", "test_runner", {})
    ledger.record_result("passed", "test_runner", True, "all tests passed")
    assert ledger.unresolved_failed_results == ()
    assert ledger.check_final(require_success=True)[0]


def test_guard_runtime_events_are_not_published_to_external_sink(tmp_path):
    class ScriptedModel:
        model_id = "scripted"

        def next(self, messages, **kwargs):
            return AgentStep(type="assistant", content="draft")

    events = []
    runtime_events = []
    run_agent_turn(
        model=ScriptedModel(),
        tools=ToolRegistry([]),
        messages=[{"role": "user", "content": "implement a function"}],
        cwd=str(tmp_path),
        max_steps=2,
        runtime={
            "activeDomain": "code",
            "isSubagent": True,
            "supervisorMode": "off",
            "requireStructuredCompletion": False,
            "_event_sink": events.append,
        },
        on_runtime_event=runtime_events.append,
    )

    messages = [str(event.get("message") or "") for event in events]
    assert not any("Execution guard:" in message for message in messages)
    assert not any("Execution guard:" in event.message for event in runtime_events)


def test_react_allows_plain_answer_for_mutation_wording(tmp_path):
    class ScriptedModel:
        model_id = "scripted"

        def next(self, messages, **kwargs):
            return AgentStep(type="assistant", content="I do not need a tool for this answer.")

    result = run_agent_turn(
        model=ScriptedModel(),
        tools=ToolRegistry([]),
        messages=[{"role": "user", "content": "implement a function"}],
        cwd=str(tmp_path),
        max_steps=3,
        runtime={
            "activeDomain": "code",
            "isSubagent": True,
            "supervisorMode": "off",
            "_event_sink": lambda event: None,
        },
    )

    assert result[-1]["role"] == "assistant"
    assert result[-1]["content"] == "I do not need a tool for this answer."


def test_request_context_is_visible_to_model(tmp_path):
    seen = []
    context = ContextManager(model="scripted")

    class ScriptedModel:
        model_id = "scripted"

        def next(self, messages, **kwargs):
            seen.append(list(messages))
            return AgentStep(type="assistant", content="了解")

    run_agent_turn(
        model=ScriptedModel(),
        tools=ToolRegistry([]),
        messages=[{"role": "user", "content": "请说明当前目录"}],
        cwd=str(tmp_path),
        max_steps=2,
        runtime={"activeDomain": "code", "isSubagent": True, "_event_sink": lambda event: None},
        context_manager=context,
        system_prompt="当前工作目录已经是项目根目录。",
        project_context="project-root=/workspace/demo",
    )

    assert seen
    assert seen[0][0]["role"] == "system"
    assert "当前工作目录已经是项目根目录。" in seen[0][0]["content"]
    assert "project-root=/workspace/demo" in seen[0][0]["content"]
    assert context.get_stats().dynamic_tokens == 0


def test_request_context_replaces_previous_injected_context(tmp_path):
    seen = []

    class ScriptedModel:
        model_id = "scripted"

        def next(self, messages, **kwargs):
            seen.append(messages[0]["content"])
            return AgentStep(type="assistant", content="了解")

    runtime = {"activeDomain": "code", "isSubagent": True, "_event_sink": lambda event: None}
    run_agent_turn(
        model=ScriptedModel(), tools=ToolRegistry([]),
        messages=[{"role": "user", "content": "一次"}], cwd=str(tmp_path),
        max_steps=1, runtime=runtime, system_prompt="根提示", project_context="path=one",
    )
    run_agent_turn(
        model=ScriptedModel(), tools=ToolRegistry([]),
        messages=[
            {"role": "system", "content": "根提示\n\npath=one", "_request_context_content": "根提示\n\npath=one"},
            {"role": "user", "content": "二次"},
        ], cwd=str(tmp_path), max_steps=1, runtime=runtime,
        system_prompt="根提示", project_context="path=two",
    )

    assert seen[0] == "根提示\n\npath=one"
    assert seen[1] == "根提示\n\npath=two"


def test_react_terminal_answer_can_recover_from_prior_tool_error():
    contract = TurnTaskContract(mode="mutation", requires_execute=True, requires_verify=True)
    state = TurnRecurrentState(max_steps=3, task_contract=contract)
    state.record_tool_result(False, summary="test_runner: transient failure")
    state.set_stop_reason("done")

    assert state.final_task_state().value == "completed"


def test_run_command_invalid_working_directory_is_a_tool_failure(tmp_path):
    request = validate_run_command(
        {"command": "python", "args": ["-c", "print(1)"], "cwd": "missing-directory"}
    )
    result = run_command_tool.run(request, ToolContext(cwd=str(tmp_path)))

    assert not result.ok
    assert "Unable to start command" in result.output


def test_workspace_name_prefix_is_normalized_for_relative_tool_paths(tmp_path):
    workspace = tmp_path / "Smart Agent"
    workspace.mkdir()
    context = ToolContext(cwd=str(workspace))

    resolved = resolve_tool_path(context, "smart agent/.git", "read")

    assert resolved == (workspace / ".git").resolve()


def test_tool_budget_attempts_text_only_final_synthesis(tmp_path):
    calls = []

    class ScriptedModel:
        model_id = "scripted"

        def next(self, messages, **kwargs):
            calls.append(messages[-1].get("content", ""))
            if len(calls) == 1:
                return AgentStep(
                    type="tool_calls",
                    calls=[{"id": "list-1", "toolName": "list_files", "input": {"path": "."}}],
                )
            return AgentStep(type="assistant", content="已读取项目结构，目录内容如下。")

    result = run_agent_turn(
        model=ScriptedModel(),
        tools=ToolRegistry([ToolDefinition(
            name="list_files",
            description="list",
            input_schema={"type": "object"},
            validator=lambda value: value,
            run=lambda value, context: ToolResult(ok=True, output="file README.md"),
        )]),
        messages=[{"role": "user", "content": "读取项目结构"}],
        cwd=str(tmp_path),
        max_steps=3,
        runtime={"activeDomain": "code", "isSubagent": True, "maxToolCalls": 1, "_event_sink": lambda event: None},
    )

    assert result[-1]["content"] == "已读取项目结构，目录内容如下。"
    assert "budget" not in result[-1]["content"].lower()
