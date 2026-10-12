from smart_agent.tooling import ToolContext
from smart_agent.tools.complete_task import complete_task_tool
from smart_agent.turn_kernel import TurnTaskContract
from smart_agent.verification_guard import VerificationLedger


def _call(payload, ledger, tmp_path):
    runtime = {"_verificationLedger": ledger}
    return complete_task_tool.run(
        complete_task_tool.validator(payload),
        ToolContext(cwd=str(tmp_path), _runtime=runtime),
    )


def _call_with_contract(payload, ledger, contract, tmp_path):
    runtime = {"_verificationLedger": ledger, "_taskContract": contract}
    return complete_task_tool.run(
        complete_task_tool.validator(payload),
        ToolContext(cwd=str(tmp_path), _runtime=runtime),
    )


def test_complete_task_ignores_previous_complete_task_failure(tmp_path):
    ledger = VerificationLedger()
    ledger.record_call("read-1", "read_file", {"path": "README.md"})
    ledger.record_result("read-1", "read_file", True, "summary content")
    ledger.record_call("finish-1", "complete_task", {"status": "completed"})
    ledger.record_result("finish-1", "complete_task", False, "missing tool_evidence_refs")
    ledger.record_call("finish-2", "complete_task", {"status": "completed"})

    result = _call(
        {
            "status": "completed",
            "summary": "done",
            "tool_evidence_refs": [{"kind": "tool", "name": "read_file"}],
        },
        ledger,
        tmp_path,
    )

    assert result.ok
    assert "Task completion recorded" in result.output


def test_complete_task_still_blocks_business_tool_failure(tmp_path):
    ledger = VerificationLedger()
    ledger.record_call("read-1", "read_file", {"path": "missing.md"})
    ledger.record_result("read-1", "read_file", False, "file missing")
    ledger.record_call("finish-1", "complete_task", {"status": "completed"})

    result = _call(
        {
            "status": "completed",
            "summary": "done",
            "tool_evidence_refs": [{"kind": "tool", "name": "read_file"}],
        },
        ledger,
        tmp_path,
    )

    assert not result.ok
    assert "tool execution failed: read_file" in result.output


def test_complete_task_rejects_career_evidence_refs_field(tmp_path):
    ledger = VerificationLedger()
    payload = {"status": "completed", "summary": "done", "evidence_refs": ["README.md"]}

    try:
        complete_task_tool.validator(payload)
    except ValueError as exc:
        assert "tool_evidence_refs" in str(exc)
    else:
        raise AssertionError("Career evidence_refs must not be accepted by Runtime complete_task")


def test_complete_task_accepts_typed_tool_call_reference(tmp_path):
    ledger = VerificationLedger()
    ledger.record_call("read-1", "read_file", {"path": "README.md"})
    ledger.record_result("read-1", "read_file", True, "summary content")

    result = _call(
        {
            "status": "completed",
            "summary": "done",
            "tool_evidence_refs": [{"kind": "tool_call", "id": "read-1"}],
        },
        ledger,
        tmp_path,
    )

    assert result.ok


def test_complete_task_allows_model_selected_completion_before_execution(tmp_path):
    ledger = VerificationLedger()
    contract = TurnTaskContract(
        mode="mutation", requires_execute=True, requires_verify=True
    )

    result = _call_with_contract(
        {"status": "completed", "summary": "done"},
        ledger,
        contract,
        tmp_path,
    )

    assert result.ok


def test_complete_task_does_not_apply_phase_gate_after_execution(tmp_path):
    ledger = VerificationLedger()
    contract = TurnTaskContract(
        mode="mutation", requires_execute=True, requires_verify=True,
        execution_count=1, mutation_epoch=1,
    )

    result = _call_with_contract(
        {"status": "completed", "summary": "done"},
        ledger,
        contract,
        tmp_path,
    )

    assert result.ok
