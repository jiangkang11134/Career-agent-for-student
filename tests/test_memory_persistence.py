from smart_agent.memory import MemoryManager, MemoryPaths, MemoryScope


def test_global_memory_uses_workspace_fallback_when_configured_path_is_not_directory(tmp_path, monkeypatch):
    blocked = tmp_path / "legacy-memory"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("SMART_AGENT_MEMORY_DIR", str(blocked))

    paths = MemoryPaths.for_workspace(str(tmp_path / "workspace"))

    assert paths.user_memory == (tmp_path / "workspace" / ".runtime" / "user-memory").resolve()


def test_memory_write_persists_to_fallback_directory(tmp_path, monkeypatch):
    blocked = tmp_path / "legacy-memory"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("SMART_AGENT_MEMORY_DIR", str(blocked))
    manager = MemoryManager(workspace=tmp_path / "workspace")

    entry = manager.add_entry(MemoryScope.USER, category="preference", content="Use concise answers")

    assert entry.content == "Use concise answers"
    target = tmp_path / "workspace" / ".runtime" / "user-memory" / "memory.json"
    assert target.exists()
