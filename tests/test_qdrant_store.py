from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from smart_agent.career.materials import HashingEmbedding, ProjectChunk, QdrantProjectKnowledgeStore


@dataclass
class _Point:
    id: str
    vector: list[float]
    payload: dict[str, object]
    score: float = 0.0


class _Models:
    class Distance:
        COSINE = "Cosine"

    class VectorParams:
        def __init__(self, *, size: int, distance: str) -> None:
            self.size, self.distance = size, distance

    class PointStruct:
        def __init__(self, *, id: str, vector: list[float], payload: dict[str, object]) -> None:
            self.id, self.vector, self.payload = id, vector, payload

    class PointIdsList:
        def __init__(self, *, points: list[str]) -> None:
            self.points = points

    class MatchValue:
        def __init__(self, *, value: str) -> None:
            self.value = value

    class FieldCondition:
        def __init__(self, *, key: str, match: object) -> None:
            self.key, self.match = key, match

    class Filter:
        def __init__(self, *, must: list[object]) -> None:
            self.must = must


class _Client:
    def __init__(self, **_: object) -> None:
        self.collections: dict[str, dict[str, _Point]] = {}

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, *, collection_name: str, vectors_config: object) -> None:
        self.collections[collection_name] = {}

    def count(self, *, collection_name: str, exact: bool) -> object:
        return types.SimpleNamespace(count=len(self.collections[collection_name]))

    def upsert(self, *, collection_name: str, points: list[object], wait: bool) -> None:
        self.collections[collection_name].update(
            {point.id: _Point(point.id, point.vector, point.payload) for point in points}
        )

    def delete(self, *, collection_name: str, points_selector: object, wait: bool) -> None:
        for point_id in points_selector.points:
            self.collections[collection_name].pop(point_id, None)

    def scroll(self, *, collection_name: str, limit: int, offset: object, with_payload: object, with_vectors: bool) -> tuple[list[_Point], None]:
        return list(self.collections[collection_name].values())[:limit], None

    def query_points(self, *, collection_name: str, query: list[float], query_filter: object, limit: int, with_payload: bool) -> object:
        def allowed(point: _Point) -> bool:
            if query_filter is None:
                return True
            condition = query_filter.must[0]
            return point.payload.get(condition.key) == condition.match.value

        def cosine(point: _Point) -> float:
            numerator = sum(left * right for left, right in zip(query, point.vector))
            left_norm = math.sqrt(sum(item * item for item in query)) or 1.0
            right_norm = math.sqrt(sum(item * item for item in point.vector)) or 1.0
            return numerator / (left_norm * right_norm)

        points = [point for point in self.collections[collection_name].values() if allowed(point)]
        points.sort(key=cosine, reverse=True)
        return types.SimpleNamespace(
            points=[_Point(point.id, point.vector, point.payload, cosine(point)) for point in points[:limit]]
        )

    def retrieve(self, *, collection_name: str, ids: list[str], with_payload: bool, with_vectors: bool) -> list[_Point]:
        rows = self.collections[collection_name]
        return [rows[item] for item in ids if item in rows]


def test_qdrant_store_upsert_filter_and_delete(tmp_path: Path, monkeypatch) -> None:
    fake_module = types.ModuleType("qdrant_client")
    fake_module.QdrantClient = _Client
    fake_module.models = _Models
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_module)

    store = QdrantProjectKnowledgeStore(tmp_path, embedding_backend=HashingEmbedding(32))
    chunks = [
        ProjectChunk("c1", "p1", "m1", "s1", "Python API service", {"line": 1}, "a" * 64),
        ProjectChunk("c2", "p2", "m2", "s2", "React interface", {"line": 2}, "b" * 64),
    ]
    assert store.upsert(chunks) == 2
    assert [hit.chunk_id for hit in store.search("Python", project_id="p1")] == ["c1"]
    ranked = store.search("interface", top_k=2)
    assert ranked[0].chunk_id == "c2"
    store._sparse_db.execute("DELETE FROM project_sparse")
    store._sparse_db.commit()
    assert store.delete(material_ids={"m1"}) == 1
    assert store.search("Python", project_id="p1") == ()

    with pytest.raises(ValueError, match="query is required"):
        store.search(" ")


def test_qdrant_store_real_local_client(tmp_path: Path) -> None:
    pytest.importorskip("qdrant_client")
    store = QdrantProjectKnowledgeStore(tmp_path, embedding_backend=HashingEmbedding(32))
    try:
        chunks = [
            ProjectChunk("real-1", "p1", "m1", "s1", "Python API service", {"line": 1}, "a" * 64),
            ProjectChunk("real-2", "p1", "m2", "s2", "React interface", {"line": 2}, "b" * 64),
        ]
        assert store.upsert(chunks) == 2
        assert store.search("Python", project_id="p1")[0].chunk_id == "real-1"
        assert store.delete(material_ids={"m1"}) == 1
    finally:
        store.close()
