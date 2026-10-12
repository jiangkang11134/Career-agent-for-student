"""Tenant-scoped model connection persistence.

MySQL is the source of truth. Redis is a short-lived read-through cache and
is never required for correctness. Credentials are never returned by the
read API; this module keeps them server-side for model calls.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from sqlalchemy import Column, DateTime, MetaData, String, Table, create_engine, select, update
from sqlalchemy.engine import Engine

from smart_agent.auth import default_database_url

metadata = MetaData()
tenant_model_configs = Table(
    "tenant_model_configs",
    metadata,
    Column("tenant_id", String(64), primary_key=True),
    Column("agent_role", String(64), primary_key=True),
    Column("provider", String(32), nullable=False),
    Column("model", String(255), nullable=False),
    Column("base_url", String(1024), nullable=False),
    Column("api_key", String(1024), nullable=False),
    Column("updated_at", DateTime, nullable=False),
)


class TenantModelConfigStore:
    """CRUD with MySQL persistence and best-effort Redis caching."""

    def __init__(self, engine: Engine | None = None, *, redis_url: str = "") -> None:
        self.engine = engine or create_engine(default_database_url(), pool_pre_ping=True, future=True)
        self.redis_url = str(redis_url or "").strip()
        self._schema_ready = False
        self._schema_lock = Lock()
        self._redis: Any = None
        self._redis_lock = Lock()

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if not self._schema_ready:
                metadata.create_all(self.engine)
                self._schema_ready = True

    def _redis_client(self):
        if self._redis is False:
            return None
        if self._redis is not None:
            return self._redis
        with self._redis_lock:
            if self._redis is not None:
                return None if self._redis is False else self._redis
            url = str(self.redis_url or os.environ.get("SMART_AGENT_REDIS_URL") or "").strip()
            if not url:
                self._redis = False
                return None
            try:
                from redis import Redis

                client = Redis.from_url(
                    url,
                    decode_responses=True,
                    socket_connect_timeout=0.5,
                    socket_timeout=0.5,
                )
                client.ping()
                self._redis = client
            except Exception:
                self._redis = False
        return self._redis if self._redis is not False else None

    @staticmethod
    def _cache_key(tenant_id: str) -> str:
        return f"smart-agent:tenant-model-config:{tenant_id}"

    def _cache_get(self, tenant_id: str) -> dict[str, dict[str, str]] | None:
        client = self._redis_client()
        if client is None:
            return None
        try:
            raw = client.get(self._cache_key(tenant_id))
            if not raw:
                return None
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def _cache_set(self, tenant_id: str, value: dict[str, dict[str, str]]) -> None:
        client = self._redis_client()
        if client is None:
            return
        try:
            client.setex(self._cache_key(tenant_id), 900, json.dumps(value, ensure_ascii=False))
        except Exception:
            return

    def _cache_delete(self, tenant_id: str) -> None:
        client = self._redis_client()
        if client is None:
            return
        try:
            client.delete(self._cache_key(tenant_id))
        except Exception:
            return

    def get_all(self, tenant_id: str) -> dict[str, dict[str, str]]:
        tenant = str(tenant_id or "").strip()
        if not tenant:
            return {}
        cached = self._cache_get(tenant)
        if cached is not None:
            return cached
        self._ensure_schema()
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(tenant_model_configs).where(tenant_model_configs.c.tenant_id == tenant)
            ).mappings().all()
        result = {
            str(row["agent_role"]): {
                "provider": str(row["provider"]),
                "model": str(row["model"]),
                "base_url": str(row["base_url"]),
                "api_key": str(row["api_key"]),
            }
            for row in rows
        }
        self._cache_set(tenant, result)
        return result

    def upsert(
        self,
        tenant_id: str,
        agent_role: str,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None = None,
    ) -> dict[str, str]:
        tenant = str(tenant_id or "").strip()
        role = str(agent_role or "").strip()
        if not tenant or not role:
            raise ValueError("tenant_id and agent_role are required")
        values = {
            "provider": str(provider or "custom").strip().lower(),
            "model": str(model or "").strip(),
            "base_url": str(base_url or "").strip().rstrip("/"),
        }
        if not values["model"] or not values["base_url"]:
            raise ValueError("model and base_url are required")
        existing = self.get_all(tenant).get(role, {})
        values["api_key"] = str(api_key if api_key is not None and str(api_key).strip() else existing.get("api_key") or "").strip()
        if not values["api_key"]:
            raise ValueError("api_key is required for a new tenant model config")
        self._ensure_schema()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self.engine.begin() as conn:
            current = conn.execute(
                select(tenant_model_configs.c.tenant_id).where(
                    (tenant_model_configs.c.tenant_id == tenant)
                    & (tenant_model_configs.c.agent_role == role)
                )
            ).first()
            if current is None:
                conn.execute(tenant_model_configs.insert().values(tenant_id=tenant, agent_role=role, updated_at=now, **values))
            else:
                conn.execute(
                    update(tenant_model_configs)
                    .where(
                        (tenant_model_configs.c.tenant_id == tenant)
                        & (tenant_model_configs.c.agent_role == role)
                    )
                    .values(updated_at=now, **values)
                )
        self._cache_delete(tenant)
        result = self.get_all(tenant).get(role, {})
        return result


__all__ = ["TenantModelConfigStore", "metadata", "tenant_model_configs"]
