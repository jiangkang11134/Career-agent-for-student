"""Redis-backed request idempotency with an explicit in-memory fallback."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy import Column, DateTime, Integer, LargeBinary, MetaData, String, Table, Text, create_engine, delete, insert, select, update
from sqlalchemy.exc import IntegrityError

from smart_agent.auth import default_database_url


_idempotency_metadata = MetaData()
_idempotency_records = Table(
    "idempotency_records",
    _idempotency_metadata,
    Column("record_key", String(64), primary_key=True),
    Column("fingerprint", String(512), nullable=False),
    Column("status_code", Integer, nullable=False),
    Column("headers_json", Text, nullable=False),
    Column("response_body", LargeBinary, nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("expires_at", DateTime, nullable=False, index=True),
)


@dataclass
class _Entry:
    fingerprint: str
    status_code: int
    headers: dict[str, str]
    body: bytes
    expires_at: float


class IdempotencyStore:
    def __init__(self, ttl_seconds: int = 24 * 60 * 60) -> None:
        self.ttl_seconds = max(60, int(os.environ.get("SMART_AGENT_IDEMPOTENCY_TTL_SECONDS", ttl_seconds)))
        self._entries: dict[str, _Entry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()
        self._redis = None
        self._redis_init_lock = asyncio.Lock()
        self._mysql = None
        self._mysql_init_lock = asyncio.Lock()

    async def _lock_for(self, key: str) -> asyncio.Lock:
        async with self._guard:
            return self._locks.setdefault(key, asyncio.Lock())

    async def _redis_client(self):
        if self._redis is False:
            return None
        if self._redis is not None:
            return self._redis
        async with self._redis_init_lock:
            if self._redis is not None:
                return self._redis
            url = str(os.environ.get("SMART_AGENT_REDIS_URL") or "redis://127.0.0.1:6379/0").strip()
            try:
                from redis.asyncio import Redis

                client = Redis.from_url(url, decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
                await client.ping()
                self._redis = client
            except Exception:
                self._redis = False
        return self._redis if self._redis is not False else None

    async def _mysql_engine(self):
        if self._mysql is False:
            return None
        if self._mysql is not None:
            return self._mysql
        async with self._mysql_init_lock:
            if self._mysql is not None:
                return None if self._mysql is False else self._mysql
            try:
                engine = await asyncio.to_thread(create_engine, default_database_url(), pool_pre_ping=True, future=True)
                await asyncio.to_thread(_idempotency_metadata.create_all, engine)
                self._mysql = engine
            except Exception:
                self._mysql = False
        return self._mysql if self._mysql is not False else None

    async def _mysql_get(self, record_key: str, fingerprint: str):
        record_key = hashlib.sha256(record_key.encode()).hexdigest()
        engine = await self._mysql_engine()
        if engine is None:
            return None
        def read():
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            with engine.begin() as conn:
                row = conn.execute(select(_idempotency_records).where(_idempotency_records.c.record_key == record_key)).mappings().first()
                if row is None:
                    return None
                if row["expires_at"] <= now:
                    conn.execute(delete(_idempotency_records).where(_idempotency_records.c.record_key == record_key))
                    return None
                return row
        row = await asyncio.to_thread(read)
        if row is None:
            return None
        if row["fingerprint"] != fingerprint:
            return JSONResponse(status_code=409, content={"detail": "Idempotency-Key was reused with different request data"})
        remaining = max(1.0, (row["expires_at"] - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds())
        return _Entry(row["fingerprint"], int(row["status_code"]), json.loads(row["headers_json"]), bytes(row["response_body"]), time.monotonic() + remaining)

    async def _mysql_save(self, record_key: str, entry: _Entry) -> None:
        record_key = hashlib.sha256(record_key.encode()).hexdigest()
        engine = await self._mysql_engine()
        if engine is None:
            return
        def write():
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            values = {"record_key": record_key, "fingerprint": entry.fingerprint, "status_code": entry.status_code, "headers_json": json.dumps(entry.headers, ensure_ascii=False), "response_body": entry.body, "created_at": now, "expires_at": now + timedelta(seconds=self.ttl_seconds)}
            with engine.begin() as conn:
                try:
                    conn.execute(insert(_idempotency_records).values(**values))
                except IntegrityError:
                    conn.execute(update(_idempotency_records).where(_idempotency_records.c.record_key == record_key).values(**values))
        await asyncio.to_thread(write)

    @staticmethod
    def _fingerprint(request: Request, body: bytes) -> str:
        return f"{request.method.upper()}:{request.url.path}:{hashlib.sha256(body).hexdigest()}"

    @staticmethod
    def _replay(entry: _Entry) -> Response:
        return Response(content=entry.body, status_code=entry.status_code, headers=entry.headers)

    @staticmethod
    def _redis_key(scope: str, key: str) -> str:
        digest = hashlib.sha256(f"{scope}:{key}".encode()).hexdigest()
        return f"smart_agent:idempotency:v2:{digest}"

    async def execute(self, request: Request, call_next: Callable[[Request], Awaitable[Response]], scope: str) -> Response:
        key = str(request.headers.get("idempotency-key") or "").strip()
        if not key:
            return await call_next(request)
        if len(key) > 200:
            return JSONResponse(status_code=400, content={"detail": "Idempotency-Key too long"})
        body = await request.body()
        fingerprint = self._fingerprint(request, body)
        cache_key = self._redis_key(scope, key)
        lock = await self._lock_for(cache_key)
        async with lock:
            redis = await self._redis_client()
            if redis is not None:
                return await self._execute_redis(redis, cache_key, fingerprint, request, call_next)
            durable = await self._mysql_get(cache_key, fingerprint)
            if isinstance(durable, Response):
                return durable
            if durable is not None:
                return self._replay(durable)
            return await self._execute_memory(cache_key, fingerprint, request, call_next)

    async def _execute_memory(self, cache_key: str, fingerprint: str, request: Request, call_next) -> Response:
        now = time.monotonic()
        entry = self._entries.get(cache_key)
        if entry and entry.expires_at > now:
            if entry.fingerprint != fingerprint:
                return JSONResponse(status_code=409, content={"detail": "Idempotency-Key was reused with different request data"})
            return self._replay(entry)
        response = await call_next(request)
        body = b"".join([chunk async for chunk in response.body_iterator])
        if response.status_code < 500:
            headers = {k: v for k, v in response.headers.items() if k.lower() not in {"content-length", "transfer-encoding"}}
            entry = _Entry(fingerprint, response.status_code, headers, body, now + self.ttl_seconds)
            self._entries[cache_key] = entry
            await self._mysql_save(cache_key, entry)
        return Response(content=body, status_code=response.status_code, headers=dict(response.headers))

    async def _execute_redis(self, redis, cache_key: str, fingerprint: str, request: Request, call_next) -> Response:
        result_key = cache_key + ":result"
        lock_key = cache_key + ":lock"
        raw = await redis.get(result_key)
        if raw:
            return self._redis_replay(raw, fingerprint)
        durable = await self._mysql_get(cache_key, fingerprint)
        if isinstance(durable, Response):
            return durable
        if durable is not None:
            await redis.set(result_key, json.dumps({"fingerprint": durable.fingerprint, "status_code": durable.status_code, "headers": durable.headers, "body": base64.b64encode(durable.body).decode("ascii")}, ensure_ascii=False), ex=self.ttl_seconds)
            return self._replay(durable)
        owner = hashlib.sha256(f"{time.time_ns()}".encode()).hexdigest()
        acquired = await redis.set(lock_key, owner, ex=300, nx=True)
        if not acquired:
            for _ in range(120):
                await asyncio.sleep(0.25)
                raw = await redis.get(result_key)
                if raw:
                    return self._redis_replay(raw, fingerprint)
            return JSONResponse(status_code=409, content={"detail": "同一幂等请求正在处理中，请稍后重试"})
        try:
            response = await call_next(request)
            body = b"".join([chunk async for chunk in response.body_iterator])
            if response.status_code < 500:
                headers = {k: v for k, v in response.headers.items() if k.lower() not in {"content-length", "transfer-encoding"}}
                entry = _Entry(fingerprint, response.status_code, headers, body, time.monotonic() + self.ttl_seconds)
                payload = {"fingerprint": fingerprint, "status_code": response.status_code, "headers": headers, "body": base64.b64encode(body).decode("ascii")}
                await redis.set(result_key, json.dumps(payload, ensure_ascii=False), ex=self.ttl_seconds)
                await self._mysql_save(cache_key, entry)
            return Response(content=body, status_code=response.status_code, headers=dict(response.headers))
        finally:
            await redis.delete(lock_key)

    @staticmethod
    def _redis_replay(raw: str, fingerprint: str) -> Response:
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return JSONResponse(status_code=409, content={"detail": "幂等记录格式无效，请使用新的 Idempotency-Key"})
        if not isinstance(payload, dict):
            return JSONResponse(status_code=409, content={"detail": "幂等记录格式无效，请使用新的 Idempotency-Key"})
        if payload.get("fingerprint") != fingerprint:
            return JSONResponse(status_code=409, content={"detail": "Idempotency-Key was reused with different request data"})
        return Response(content=base64.b64decode(payload.get("body", "")), status_code=int(payload.get("status_code", 200)), headers=dict(payload.get("headers") or {}))


idempotency_store = IdempotencyStore()
