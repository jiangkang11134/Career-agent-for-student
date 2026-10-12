"""Durable Redis list queue with a bounded local fallback."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path


class PersistentTaskQueue:
    def __init__(self, *, name: str, root: str | Path) -> None:
        self.name = name
        self.processing_name = f"{name}:processing"
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._redis = None
        self._redis_lock = asyncio.Lock()
        self._local: asyncio.Queue[str] = asyncio.Queue()
        self._local_file = self.root / "queue.jsonl"

    async def _client(self):
        if self._redis is False:
            return None
        if self._redis is not None:
            return self._redis
        async with self._redis_lock:
            if self._redis is not None:
                return None if self._redis is False else self._redis
            try:
                from redis.asyncio import Redis

                client = Redis.from_url(
                    str(os.environ.get("SMART_AGENT_REDIS_URL") or "redis://127.0.0.1:6379/0"),
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=5,
                )
                await client.ping()
                self._redis = client
            except Exception:
                self._redis = False
        return self._redis if self._redis is not False else None

    async def enqueue(self, payload: dict) -> str:
        item = {"task_id": uuid.uuid4().hex, "attempt": 0, **dict(payload)}
        raw = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        client = await self._client()
        if client is not None:
            await client.rpush(self.name, raw)
        else:
            with self._local_file.open("a", encoding="utf-8") as handle:
                handle.write(raw + "\n")
            await self._local.put(raw)
        return str(item["task_id"])

    async def _recover_processing(self, client) -> None:
        if client is None:
            return
        stale = await client.lrange(self.processing_name, 0, -1)
        if stale:
            await client.rpush(self.name, *stale)
            await client.delete(self.processing_name)

    async def worker(
        self,
        handler: Callable[[dict], Awaitable[None]],
        stop_event: asyncio.Event,
    ) -> None:
        client = await self._client()
        await self._recover_processing(client)
        while not stop_event.is_set():
            raw = None
            if client is not None:
                result = await client.brpoplpush(self.name, self.processing_name, timeout=1)
                raw = result
            else:
                try:
                    raw = await asyncio.wait_for(self._local.get(), timeout=1)
                except asyncio.TimeoutError:
                    continue
            if not raw:
                continue
            try:
                payload = json.loads(raw)
                await handler(payload)
            except Exception:
                # Leave Redis processing items for recovery; local fallback is
                # best-effort and requeues a bounded number of attempts.
                if client is not None:
                    try:
                        payload = json.loads(raw)
                        payload["attempt"] = int(payload.get("attempt", 0)) + 1
                        await client.lrem(self.processing_name, 1, raw)
                        if payload["attempt"] <= 3:
                            await client.rpush(
                                self.name,
                                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                            )
                    except Exception:
                        pass
                    continue
                try:
                    payload = json.loads(raw)
                    payload["attempt"] = int(payload.get("attempt", 0)) + 1
                    if payload["attempt"] <= 3:
                        await self._local.put(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                except Exception:
                    pass
            else:
                if client is not None:
                    await client.lrem(self.processing_name, 1, raw)


__all__ = ["PersistentTaskQueue"]
