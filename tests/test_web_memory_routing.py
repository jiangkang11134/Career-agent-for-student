from smart_agent.fact_ledger import FactLedger
from smart_agent.preflight import PreflightIntent, classify_input
from smart_agent.web.app import (
    _is_read_only_workspace_request,
    _original_user_message,
    _apply_memory_operations_to_messages,
    _should_use_agent_loop,
)


def test_wrapped_code_prompt_keeps_raw_memory_intent():
    payload = {
        "message": "当前项目：demo\n历史对话：...\n本轮请求：我的名字是姜康",
        "original_message": "我的名字是姜康",
        "domain": "code",
    }

    raw = _original_user_message(payload)

    assert raw == "我的名字是姜康"
    assert FactLedger.is_memory_request(raw)
    assert not FactLedger.is_memory_request(payload["message"])


def test_message_fallback_is_backward_compatible():
    payload = {"message": "我的名字是姜康"}
    assert _original_user_message(payload) == payload["message"]


def test_legacy_wrapped_message_is_unwrapped_server_side():
    payload = {"message": "当前项目：demo\n本轮请求：我的名字是姜康"}
    assert _original_user_message(payload) == "我的名字是姜康"


def test_preflight_routes_simple_inputs_without_tools():
    assert classify_input("1+1等于几").intent is PreflightIntent.GENERAL
    assert classify_input("1+1等于几").needs_tools is False
    # Memory is decided by the model's structured protocol, not a dedicated
    # preflight branch. The exact task/general route is incidental here.
    assert classify_input("我的研究方向是物理学").intent in {
        PreflightIntent.GENERAL, PreflightIntent.TASK,
    }
    assert classify_input("修改登录接口并运行测试").needs_tools is True
    intro = classify_input("介绍一下 Smart Agent")
    assert intro.intent is PreflightIntent.PROJECT_INFO
    assert intro.needs_tools is True


def test_workspace_inspection_is_read_only_without_project_id():
    assert _is_read_only_workspace_request("我先读取项目结构、仓库规范和入口代码") is True
    assert _is_read_only_workspace_request("请看看当前 smart agent 中 Code Agent 的实现") is True
    assert _is_read_only_workspace_request("修改 Code Agent 并运行测试") is False


def test_task_preflight_cannot_fall_through_to_toolless_direct_answer():
    decision = classify_input("读取项目结构和入口代码")
    assert decision.needs_tools is True
    assert _should_use_agent_loop(decision, "读取项目结构和入口代码") is True


def test_simple_response_memory_ops_are_persisted_and_hidden(tmp_path):
    ledger = FactLedger(tmp_path, tenant_id="tenant-a")
    messages = [{
        "role": "assistant",
        "content": (
            "记住了。<memory_ops>"
            '[{"operation":"assert","content":"我的姓名是姜康",'
            '"fact_key":"name","confidence":0.99,"requires_confirmation":false}]'
            "</memory_ops>"
        ),
    }]

    _apply_memory_operations_to_messages(
        messages, fact_ledger=ledger, session_id="session-1", source_turn_id="turn-1"
    )

    assert messages[0]["content"] == "记住了。"
    assert "我的姓名是姜康" in ledger.render_context(session_id="new-session")
