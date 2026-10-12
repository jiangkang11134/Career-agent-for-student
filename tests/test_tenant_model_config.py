from sqlalchemy import create_engine

from smart_agent.tenant_model_config import TenantModelConfigStore


def test_model_config_isolated_by_tenant(tmp_path):
    store = TenantModelConfigStore(create_engine(f"sqlite:///{tmp_path / 'tenant-config.db'}", future=True))
    store.upsert(
        "tenant-a",
        "main_agent",
        provider="custom",
        model="model-a",
        base_url="https://a.example/v1",
        api_key="key-a",
    )
    store.upsert(
        "tenant-b",
        "main_agent",
        provider="custom",
        model="model-b",
        base_url="https://b.example/v1",
        api_key="key-b",
    )

    assert store.get_all("tenant-a")["main_agent"]["model"] == "model-a"
    assert store.get_all("tenant-a")["main_agent"]["api_key"] == "key-a"
    assert store.get_all("tenant-b")["main_agent"]["model"] == "model-b"
    assert store.get_all("tenant-c") == {}


def test_upsert_without_key_preserves_existing_key(tmp_path):
    store = TenantModelConfigStore(create_engine(f"sqlite:///{tmp_path / 'tenant-config.db'}", future=True))
    store.upsert(
        "tenant-a",
        "main_agent",
        provider="custom",
        model="model-a",
        base_url="https://a.example/v1",
        api_key="key-a",
    )
    store.upsert(
        "tenant-a",
        "main_agent",
        provider="custom",
        model="model-b",
        base_url="https://b.example/v1",
    )
    assert store.get_all("tenant-a")["main_agent"]["api_key"] == "key-a"
    assert store.get_all("tenant-a")["main_agent"]["model"] == "model-b"
