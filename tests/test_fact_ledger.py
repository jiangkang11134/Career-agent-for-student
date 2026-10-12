from smart_agent.fact_ledger import (
    FactLedger,
    FactStatus,
    MemoryOperation,
    parse_memory_operations,
    strip_memory_operations,
)


def test_explicit_memory_request_is_detected_and_acknowledged(tmp_path):
    ledger = FactLedger(tmp_path, tenant_id="tenant-a")

    assert FactLedger.is_memory_request("记住我的工资是5000") is True
    events = ledger.record_memory_request(
        "记住我的工资是5000",
        session_id="session-1",
        source_turn_id="turn-1",
    )

    assert events[0].content == "我的工资是5000"
    assert FactLedger.memory_response(events) == "已记住：我的工资是5000"


def test_direct_profile_statement_is_an_implicit_memory_request(tmp_path):
    ledger = FactLedger(tmp_path, tenant_id="tenant-a")

    assert FactLedger.is_memory_request("我的工资是5000") is True
    assert FactLedger.is_memory_request("我的专业是物理学") is True
    assert FactLedger.is_memory_request("我的方向是自然语言处理") is True
    assert not FactLedger.is_memory_request("请修改我的代码并运行测试")

    events = ledger.record_memory_request(
        "我的专业是物理学",
        session_id="session-1",
        source_turn_id="turn-1",
    )
    assert events[0].content == "我的专业是物理学"


def test_utf8_name_statement_persists_across_sessions(tmp_path):
    ledger = FactLedger(tmp_path, tenant_id="tenant-a")

    assert FactLedger.is_memory_request("我的名字叫姜康") is True
    events = ledger.record_memory_request(
        "我的名字叫姜康",
        session_id="session-1",
        source_turn_id="turn-1",
    )

    assert events[0].content == "我的名字是姜康"
    assert "我的名字是姜康" in ledger.render_context(session_id="session-2")


def test_memory_request_correction_uses_fast_ledger_path(tmp_path):
    ledger = FactLedger(tmp_path, tenant_id="tenant-a")
    ledger.record_memory_request(
        "记住我的工资是5000",
        session_id="session-1",
        source_turn_id="turn-1",
    )

    events = ledger.record_memory_request(
        "刚才说的工资错了，实际是8000",
        session_id="session-1",
        source_turn_id="turn-2",
    )

    assert FactLedger.memory_response(events) == "已更新记忆：8000"
    assert [item.content for item in ledger.active(session_id="session-1")] == ["8000"]


def test_mysql_backend_keeps_tenant_scoped_history_with_sqlite_driver(tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_AGENT_DATABASE_URL", f"sqlite:///{tmp_path / 'facts.db'}")
    ledger = FactLedger(tmp_path / "unused", tenant_id="tenant-a", backend="mysql")
    ledger.record_memory_request(
        "记住我的工资是5000",
        session_id="session-1",
        source_turn_id="turn-1",
    )
    other_tenant = FactLedger(tmp_path / "unused", tenant_id="tenant-b", backend="mysql")

    assert [item.content for item in ledger.active(session_id="session-1")] == ["我的工资是5000"]
    assert other_tenant.events() == ()


def test_mysql_backend_imports_matching_legacy_file_events(tmp_path, monkeypatch):
    root = tmp_path / "fact-ledger"
    FactLedger(root, tenant_id="tenant-a", backend="file").record_memory_request(
        "记住我的工资是5000",
        session_id="session-1",
        source_turn_id="turn-1",
    )
    monkeypatch.setenv("SMART_AGENT_DATABASE_URL", f"sqlite:///{tmp_path / 'facts.db'}")

    ledger = FactLedger(root, tenant_id="tenant-a", backend="mysql")

    assert [item.content for item in ledger.active(session_id="session-1")] == ["我的工资是5000"]


def test_llm_memory_proposal_requires_confirmation_before_becoming_active(tmp_path):
    ledger = FactLedger(tmp_path, tenant_id="tenant-a")
    proposed = ledger.apply_memory_operation(
        MemoryOperation(
            operation="assert",
            fact_key="salary",
            content="工资是5000",
            confidence=0.99,
        ),
        session_id="session-1",
        source_turn_id="turn-1",
    )

    assert proposed.status is FactStatus.PENDING_CONFIRMATION
    assert ledger.active(session_id="session-1") == ()

    confirmed = ledger.apply_memory_operation(
        {"operation": "confirm", "fact_id": proposed.fact_id},
        session_id="session-1",
        source_turn_id="turn-2",
        actor_type="user",
    )
    assert confirmed.status is FactStatus.CONFIRMED
    assert [item.content for item in ledger.active(session_id="session-1")] == ["工资是5000"]


def test_llm_can_persist_explicit_stable_fact_without_confirmation(tmp_path):
    ledger = FactLedger(tmp_path, tenant_id="tenant-a")
    event = ledger.apply_memory_operation(
        {
            "operation": "assert",
            "fact_key": "name",
            "content": "我的姓名是姜康",
            "confidence": 0.99,
            "requires_confirmation": False,
        },
        session_id="session-1",
        source_turn_id="turn-1",
    )

    assert event.status is FactStatus.ASSERTED
    assert "我的姓名是姜康" in ledger.render_context(session_id="new-session")


def test_memory_ops_protocol_is_parsed_and_hidden_from_user_output():
    text = '好的。<memory_ops>[{"operation":"assert","fact_key":"salary","content":"工资是5000","confidence":0.9,"requires_confirmation":true}]</memory_ops>'

    operations = parse_memory_operations(text)

    assert operations[0].fact_key == "salary"
    assert strip_memory_operations(text) == "好的。"


def test_fact_correction_keeps_history_and_updates_active_view(tmp_path):
    ledger = FactLedger(tmp_path, tenant_id="tenant-a")

    first = ledger.record_user_message(
        "记住：我在 A 公司实习",
        session_id="session-1",
        source_turn_id="turn-1",
    )
    assert len(first) == 1
    original = first[0]
    assert original.status is FactStatus.ASSERTED

    correction = ledger.record_user_message(
        "刚才说的实习公司是错的，实际是 B 公司",
        session_id="session-1",
        source_turn_id="turn-2",
    )
    assert len(correction) == 1
    assert correction[0].supersedes_fact_id == original.fact_id
    assert [item.content for item in ledger.active(session_id="session-1")] == ["B 公司"]

    events = ledger.events()
    assert len(events) == 2
    assert events[0].fact_id == original.fact_id
    assert events[1].status is FactStatus.ASSERTED

    restored = FactLedger(tmp_path, tenant_id="tenant-a")
    assert "B 公司" in restored.render_context(session_id="session-1")
    assert "A 公司" not in restored.render_context(session_id="session-1")


def test_fact_retraction_leaves_no_active_fact(tmp_path):
    ledger = FactLedger(tmp_path, tenant_id="tenant-a")
    created = ledger.append_assertion(
        "项目必须使用 FastAPI",
        session_id="session-1",
        source_turn_id="turn-1",
    )

    event = ledger.transition(
        created.fact_id,
        FactStatus.RETRACTED,
        session_id="session-1",
        source_turn_id="turn-2",
        reason="user correction",
    )
    assert event.status is FactStatus.RETRACTED
    assert ledger.active(session_id="session-1") == ()
    assert len(ledger.events()) == 2


def test_fact_views_are_session_scoped(tmp_path):
    ledger = FactLedger(tmp_path, tenant_id="tenant-a")
    ledger.append_assertion("用户 A 的事实", session_id="session-a")
    ledger.append_assertion("用户 B 的事实", session_id="session-b")

    assert [item.content for item in ledger.active(session_id="session-a")] == ["用户 A 的事实"]
    assert [item.content for item in ledger.active(session_id="session-b")] == ["用户 B 的事实"]
