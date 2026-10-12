"""Bridge the Career interview knowledge library to the standalone Graph RAG package."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_agent.career.interview import InterviewKnowledgeChunk, InterviewKnowledgeCorpus


def _graph_rag_package():
    root = Path(__file__).resolve().parents[2] / "Graph Rag"
    if not root.is_dir():
        raise RuntimeError(f"Graph RAG package is missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import graph_rag

    return graph_rag


class _CorpusStoreRouter:
    """Compatibility facade over one SQLite graph store per corpus."""

    def __init__(self, owner: "CareerGraphRAGIndex") -> None:
        self.owner = owner

    def _stores(self, corpus_ids: tuple[str, ...] = ()):
        ids = tuple(dict.fromkeys(str(item) for item in corpus_ids if str(item)))
        if ids:
            return tuple((cid, self.owner._store_for(cid)) for cid in ids)
        return tuple(self.owner._known_stores().items())

    def corpus_version(self, corpus_id: str) -> str:
        return self.owner._store_for(corpus_id).corpus_version(corpus_id)

    def graph_health(self, corpus_id: str = ""):
        if corpus_id:
            return self.owner._store_for(corpus_id).graph_health(corpus_id)
        stores = self._stores()
        values = [store.graph_health() for _cid, store in stores]
        nodes = sum(int(item["node_count"]) for item in values)
        edges = sum(int(item["edge_count"]) for item in values)
        isolated = sum(int(item["isolated_node_count"]) for item in values)
        return {
            "node_count": nodes,
            "edge_count": edges,
            "isolated_node_count": isolated,
            "isolated_node_rate": isolated / nodes if nodes else 0.0,
            "average_degree": 2 * edges / nodes if nodes else 0.0,
        }

    def ready_corpus_versions(self, corpus_ids: tuple[str, ...] = ()):
        rows = []
        for cid, store in self._stores(corpus_ids):
            rows.extend(store.ready_corpus_versions((cid,)))
        return tuple(sorted(rows))

    def corpus_hashes(self, corpus_ids: tuple[str, ...] = ()):
        rows = []
        for cid, store in self._stores(corpus_ids):
            rows.extend(store.corpus_hashes((cid,)))
        return tuple(sorted(rows))

    def list_nodes(self, *, corpus_ids: tuple[str, ...] = (), node_ids: tuple[str, ...] = ()):
        wanted = set(node_ids)
        rows = []
        for _cid, store in self._stores(corpus_ids):
            rows.extend(store.list_nodes(corpus_ids=corpus_ids, node_ids=node_ids))
        if wanted:
            rows = [row for row in rows if row.node_id in wanted]
        return tuple(sorted({row.node_id: row for row in rows}.values(), key=lambda row: row.node_id))

    def node_corpora(self, node_ids: tuple[str, ...]):
        return {node.node_id: node.corpus_id for node in self.list_nodes(node_ids=node_ids)}

    def get_node(self, node_id: str):
        rows = self.list_nodes(node_ids=(node_id,))
        if not rows:
            raise KeyError(f"unknown knowledge node: {node_id}")
        return rows[0]

    def vectors(self, *, corpus_ids: tuple[str, ...] = (), node_ids: tuple[str, ...] = ()):
        rows = []
        for _cid, store in self._stores(corpus_ids):
            rows.extend(store.vectors(corpus_ids=corpus_ids, node_ids=node_ids))
        return tuple(dict(rows).items())

    def keyword_search(self, query: str, *, limit: int = 30, corpus_ids: tuple[str, ...] = ()):
        rows = []
        for _cid, store in self._stores(corpus_ids):
            rows.extend(store.keyword_search(query, limit=limit, corpus_ids=corpus_ids))
        best = {}
        for node_id, score in rows:
            best[node_id] = max(float(score), float(best.get(node_id, 0.0)))
        return sorted(best.items(), key=lambda item: (-item[1], item[0]))[:limit]

    def sparse_search(self, sparse_vector: dict[str, float], *, top_k: int, corpus_ids: tuple[str, ...] = ()):
        rows = []
        for cid, store in self._stores(corpus_ids):
            rows.extend(store.sparse_search(sparse_vector, top_k=top_k, corpus_ids=(cid,)))
        best = {}
        for node_id, score in rows:
            best[node_id] = max(float(score), float(best.get(node_id, 0.0)))
        return sorted(best.items(), key=lambda item: (-item[1], item[0]))[:top_k]

    def shortest_paths(self, seed_ids, *, corpus_id: str = "", max_hops: int, minimum_edge_confidence: float = 0.7):
        if corpus_id:
            return self.owner._store_for(corpus_id).shortest_paths(
                seed_ids, corpus_id=corpus_id, max_hops=max_hops, minimum_edge_confidence=minimum_edge_confidence
            )
        paths = {}
        for cid, store in self._stores():
            paths.update(store.shortest_paths(seed_ids, corpus_id=cid, max_hops=max_hops, minimum_edge_confidence=minimum_edge_confidence))
        return paths

    def count_nodes(self, *, corpus_ids: tuple[str, ...] = ()):
        rows = {}
        for cid, store in self._stores(corpus_ids):
            rows.update(store.count_nodes(corpus_ids=(cid,)))
        if not corpus_ids:
            return {"": sum(rows.values())}
        return {cid: int(rows.get(cid, 0)) for cid in corpus_ids}

    def replace_corpus(self, **kwargs):
        return self.owner._store_for(str(kwargs["corpus_id"]), create=True).replace_corpus(**kwargs)

    def delete_corpus(self, corpus_id: str):
        return self.owner._delete_corpus_files(corpus_id)


class _RoutedDepthPrefetcher:
    def __init__(self, owner: "CareerGraphRAGIndex", *, max_workers: int, max_pending: int) -> None:
        self.owner = owner
        self.max_workers = max_workers
        self.max_pending = max_pending
        self._prefetchers: dict[str, Any] = {}
        self._question_corpora: dict[str, str] = {}

    def start(self, question_id: str, *, active_depth_chain: tuple[str, ...], **kwargs) -> None:
        if not active_depth_chain:
            raise ValueError("active_depth_chain is required")
        corpus_id = self.owner.store.node_corpora(active_depth_chain[-1:]).get(active_depth_chain[-1])
        if not corpus_id:
            raise ValueError("active depth chain seed is not indexed")
        prefetcher = self._prefetchers.get(corpus_id)
        if prefetcher is None:
            prefetcher = self.owner._graph_rag.DepthPrefetcher(
                self.owner._retriever_for(corpus_id), max_workers=self.max_workers, max_pending=self.max_pending
            )
            self._prefetchers[corpus_id] = prefetcher
        self._question_corpora[str(question_id)] = corpus_id
        prefetcher.start(question_id, active_depth_chain=active_depth_chain, **kwargs)

    def finish(self, question_id: str, **kwargs):
        corpus_id = self._question_corpora.pop(str(question_id), "")
        prefetcher = self._prefetchers.get(corpus_id)
        if prefetcher is None:
            return self.owner._graph_rag.PrefetchOutcome(str(question_id), None, stale=True, error="prefetch_not_found")
        return prefetcher.finish(question_id, **kwargs)

    def cancel_all(self) -> None:
        for prefetcher in self._prefetchers.values():
            prefetcher.cancel_all()
        self._question_corpora.clear()

    def close(self) -> None:
        for prefetcher in self._prefetchers.values():
            prefetcher.close()
        self._prefetchers.clear()
        self._question_corpora.clear()


@dataclass(frozen=True, slots=True)
class GraphRAGSyncResult:
    corpus_id: str
    content_hash: str
    node_count: int
    edge_count: int
    index_version: str
    deduplicated: bool
    embedding_backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "content_hash": self.content_hash,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "index_version": self.index_version,
            "index_status": "ready",
            "deduplicated": self.deduplicated,
            "embedding_backend": self.embedding_backend,
        }


class CareerGraphRAGIndex:
    """Persist and query Career knowledge with one Graph RAG file per corpus."""

    def __init__(self, root: str | Path, *, embedding_provider=None) -> None:
        graph_rag = _graph_rag_package()
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._graph_rag = graph_rag
        self._store_cache: dict[str, Any] = {}
        self._retriever_cache: dict[str, Any] = {}
        self._legacy_store = (
            graph_rag.SQLiteGraphStore(self.root / "knowledge.sqlite3")
            if (self.root / "knowledge.sqlite3").is_file()
            else None
        )
        self.store = _CorpusStoreRouter(self)
        self.embedding_provider = embedding_provider or graph_rag.HashingEmbedding(384)
        self._KnowledgeNode = graph_rag.KnowledgeNode
        self._KnowledgeEdge = graph_rag.KnowledgeEdge

    def _corpus_path(self, corpus_id: str) -> Path:
        identity = str(corpus_id or "").strip()
        if not identity:
            raise ValueError("corpus_id is required")
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        return self.root / "corpora" / f"{digest}.sqlite3"

    def graph_path(self, corpus_id: str) -> Path:
        return self._corpus_path(corpus_id)

    def _store_for(self, corpus_id: str, *, create: bool = False):
        identity = str(corpus_id or "").strip()
        cached = self._store_cache.get(identity)
        if cached is not None:
            path = getattr(cached, "path", None)
            if path is not None and Path(path).is_file():
                return cached
            self._store_cache.pop(identity, None)
        path = self._corpus_path(identity)
        if not create and not path.is_file() and self._legacy_store is not None:
            if self._legacy_store.corpus_version(identity):
                return self._legacy_store
        store = self._graph_rag.SQLiteGraphStore(path)
        self._store_cache[identity] = store
        return store

    def _known_stores(self) -> dict[str, Any]:
        stores: dict[str, Any] = {}
        if self._legacy_store is not None:
            for corpus_id, _version in self._legacy_store.corpus_hashes():
                stores[corpus_id] = self._legacy_store
        corpus_dir = self.root / "corpora"
        if corpus_dir.is_dir():
            for path in corpus_dir.glob("*.sqlite3"):
                store = self._graph_rag.SQLiteGraphStore(path)
                for corpus_id, _version in store.corpus_hashes():
                    stores[corpus_id] = store
                    self._store_cache[corpus_id] = store
        return stores

    def _retriever_for(self, corpus_id: str):
        identity = str(corpus_id or "").strip()
        if identity not in self._retriever_cache:
            self._retriever_cache[identity] = self._graph_rag.GraphRAGRetriever(
                self._store_for(identity), self.embedding_provider
            )
        return self._retriever_cache[identity]

    def _delete_corpus_files(self, corpus_id: str) -> dict[str, int]:
        identity = str(corpus_id or "").strip()
        totals = {"corpus_count": 0, "node_count": 0, "edge_count": 0, "breadth_cache_count": 0}
        paths = [self._corpus_path(identity)]
        if self._legacy_store is not None:
            paths.append(self.root / "knowledge.sqlite3")
        for path in paths:
            if not path.is_file():
                continue
            store = self._graph_rag.SQLiteGraphStore(path)
            result = store.delete_corpus(identity)
            for key in totals:
                totals[key] += int(result.get(key, 0))
            if path != self.root / "knowledge.sqlite3" and not store.corpus_hashes():
                for suffix in ("", "-wal", "-shm"):
                    candidate = Path(f"{path}{suffix}")
                    if candidate.exists():
                        candidate.unlink()
        self._store_cache.pop(identity, None)
        self._retriever_cache.pop(identity, None)
        return totals

    def sync_corpus(
        self,
        corpus: InterviewKnowledgeCorpus,
        *,
        content_hash: str,
        source_path: str = "",
    ) -> GraphRAGSyncResult:
        content_hash = str(content_hash or "").strip().casefold()
        if len(content_hash) != 64:
            raise ValueError("Graph RAG sync requires a SHA-256 content hash")
        backend = str(getattr(self.embedding_provider, "name", "") or "")
        graph_store = self._store_for(corpus.corpus_id, create=True)
        if graph_store.corpus_version(corpus.corpus_id) == content_hash and graph_store.corpus_embedding_backend(corpus.corpus_id) == backend:
            health = self.store.graph_health(corpus.corpus_id)
            return GraphRAGSyncResult(
                corpus.corpus_id,
                content_hash,
                int(health["node_count"]),
                int(health["edge_count"]),
                content_hash,
                True,
                getattr(self.embedding_provider, "name", ""),
            )

        nodes = [self._node(chunk) for chunk in corpus.chunks]
        documents = [f"{node.title}\n{node.text}" for node in nodes]
        vectors: list[list[float]] = []
        sparse_vectors: dict[str, dict[str, float]] = {}
        for start in range(0, len(documents), 64):
            batch = documents[start : start + 64]
            if callable(getattr(self.embedding_provider, "embed_with_sparse", None)):
                embedded, sparse = self.embedding_provider.embed_with_sparse(batch)
            else:
                embedded, sparse = self.embedding_provider.embed(batch), [{} for _ in batch]
            if len(embedded) != len(batch):
                raise ValueError("Graph RAG embedding provider returned an unexpected vector count")
            vectors.extend(embedded)
            sparse_vectors.update({node.node_id: dict(value) for node, value in zip(nodes[start:start + 64], sparse, strict=False)})
        edges = self._sequence_edges(corpus)
        graph_store.replace_corpus(
            corpus_id=corpus.corpus_id,
            name=corpus.name,
            content_hash=content_hash,
            source_path=source_path,
            rows=list(zip(nodes, vectors, strict=True)),
            edges=edges,
            sparse_vectors=sparse_vectors,
            embedding_backend=backend,
        )
        health = self.store.graph_health(corpus.corpus_id)
        return GraphRAGSyncResult(
            corpus.corpus_id,
            content_hash,
            int(health["node_count"]),
            int(health["edge_count"]),
            content_hash,
            False,
            getattr(self.embedding_provider, "name", ""),
        )

    def sync_library(self, library, *, tenant_id: str | None = None) -> tuple[GraphRAGSyncResult, ...]:
        results = []
        for source in library.list_sources(tenant_id=tenant_id):
            corpus = library.corpus_store.get(source.corpus_id)
            results.append(
                self.sync_corpus(
                    corpus,
                    content_hash=source.content_hash,
                    source_path=source.source_path,
                )
            )
            library.update_source_index_metadata(source.source_id, results[-1])
        return tuple(results)

    def ensure_corpora(
        self,
        library,
        corpus_ids: tuple[str, ...],
        *,
        tenant_id: str | None = None,
    ) -> tuple[GraphRAGSyncResult, ...]:
        sources = {
            source.corpus_id: source
            for source in library.list_sources(tenant_id=tenant_id)
        }
        results = []
        for corpus_id in corpus_ids:
            source = sources.get(corpus_id)
            if source is None:
                raise ValueError(f"Graph RAG source metadata is missing for corpus: {corpus_id}")
            current_store = self._store_for(corpus_id)
            current_version = current_store.corpus_version(corpus_id)
            backend = str(getattr(self.embedding_provider, "name", "") or "")
            if current_version == source.content_hash and current_store.corpus_embedding_backend(corpus_id) == backend:
                health = current_store.graph_health(corpus_id)
                expected_count = source.chunk_count or library.corpus_store.get_metadata(corpus_id).chunk_count
                if int(health["node_count"]) == expected_count:
                    result = GraphRAGSyncResult(
                        corpus_id,
                        source.content_hash,
                        int(health["node_count"]),
                        int(health["edge_count"]),
                        current_version,
                        True,
                        getattr(self.embedding_provider, "name", ""),
                    )
                    if source.index_status != "ready" or source.chunk_count != expected_count:
                        library.update_source_index_metadata(source.source_id, result)
                    results.append(result)
                    continue
            corpus = library.corpus_store.get(corpus_id)
            result = self.sync_corpus(
                corpus,
                content_hash=source.content_hash,
                source_path=source.source_path,
            )
            library.update_source_index_metadata(source.source_id, result)
            results.append(result)
        return tuple(results)

    def corpus_versions(self, corpus_ids: tuple[str, ...] = ()) -> dict[str, str]:
        return dict(self.store.ready_corpus_versions(corpus_ids))

    def corpus_chunk_counts(self, corpus_ids: tuple[str, ...] = ()) -> dict[str, int]:
        return self.store.count_nodes(corpus_ids=corpus_ids)

    def delete_corpus(self, corpus_id: str) -> dict[str, int]:
        return self.store.delete_corpus(corpus_id)

    def get_chunks(
        self,
        chunk_ids: tuple[str, ...],
        *,
        corpus_ids: tuple[str, ...] = (),
    ) -> tuple[InterviewKnowledgeChunk, ...]:
        if not chunk_ids:
            return ()
        nodes = {
            node.node_id: node
            for node in self.store.list_nodes(corpus_ids=corpus_ids, node_ids=chunk_ids)
        }
        missing = set(chunk_ids) - set(nodes)
        if missing:
            raise ValueError(f"Graph RAG index is missing knowledge chunks: {sorted(missing)}")
        return tuple(self._chunk(nodes[chunk_id]) for chunk_id in chunk_ids)

    def precompute_breadth(
        self,
        *,
        point_id: str,
        point_text: str,
        jd_text: str,
        corpus_ids: tuple[str, ...],
        context_version: str,
        top_k: int = 10,
    ):
        if not corpus_ids:
            raise ValueError("corpus_ids is required for file-scoped Graph RAG")
        results = [self._retriever_for(cid).precompute_breadth(
            point_id, point_text, jd_text=jd_text, corpus_ids=(cid,),
            context_version=context_version, top_k=top_k,
        ) for cid in corpus_ids]
        return max(results, key=lambda result: result.hits[0].score if result.hits else 0.0)

    def retrieve_depth(
        self,
        *,
        active_depth_chain: tuple[str, ...],
        current_question: str,
        answer_hint: str = "",
        used_node_ids: set[str] | None = None,
        corpus_ids: tuple[str, ...] = (),
        top_k: int = 3,
    ):
        if not active_depth_chain:
            raise ValueError("active_depth_chain is required")
        corpus_id = self.store.node_corpora(active_depth_chain[-1:]).get(active_depth_chain[-1])
        if not corpus_id:
            raise ValueError("active depth chain seed is not indexed")
        return self._retriever_for(corpus_id).retrieve_depth(
            active_depth_chain, current_question=current_question, answer_hint=answer_hint,
            used_node_ids=used_node_ids, corpus_ids=(corpus_id,), top_k=top_k,
        )

    def create_depth_prefetcher(self, *, max_workers: int = 2, max_pending: int = 4):
        if not self._known_stores():
            raise ValueError("no file-scoped graph is indexed")
        return _RoutedDepthPrefetcher(self, max_workers=max_workers, max_pending=max_pending)

    def result_chunks(self, result) -> tuple[InterviewKnowledgeChunk, ...]:
        return tuple(self._chunk(hit.node, hit=hit, trace=result.trace) for hit in result.hits)

    @staticmethod
    def trace_dict(result) -> dict[str, Any]:
        trace = result.trace
        return {
            "mode": trace.mode,
            "query": trace.query,
            "seed_node_ids": list(trace.seed_node_ids),
            "excluded_node_ids": list(trace.excluded_node_ids),
            "normal_hop_limit": trace.normal_hop_limit,
            "effective_hop_limit": trace.effective_hop_limit,
            "fallback_used": trace.fallback_used,
            "candidate_count": trace.candidate_count,
            "returned_node_ids": list(trace.returned_node_ids),
            "timings_ms": dict(trace.timings_ms),
            "hits": [
                {
                    "node_id": hit.node.node_id,
                    "score": hit.score,
                    "keyword_score": hit.keyword_score,
                    "vector_score": hit.vector_score,
                    "graph_score": hit.graph_score,
                    "hops": hit.hops,
                    "origin": hit.origin,
                    "path": list(hit.path.node_ids) if hit.path else [],
                    "relations": list(hit.path.relation_types) if hit.path else [],
                }
                for hit in result.hits
            ],
        }

    def _node(self, chunk: InterviewKnowledgeChunk):
        locator = {
            **chunk.locator,
            "domain": chunk.domain,
            "subdomain": chunk.subdomain,
            "knowledge_type": chunk.knowledge_type.value,
            "content_hash": chunk.content_hash,
        }
        return self._KnowledgeNode(
            node_id=chunk.chunk_id,
            corpus_id=chunk.corpus_id,
            title=chunk.topic,
            text=chunk.text,
            source_id=chunk.source_id,
            knowledge_key=hashlib.sha256(
                f"{chunk.source_id}\0{chunk.topic}\0{chunk.locator}".encode("utf-8")
            ).hexdigest(),
            locator=locator,
            difficulty=chunk.difficulty,
        )

    def _chunk(self, node, *, hit=None, trace=None) -> InterviewKnowledgeChunk:
        locator = dict(node.locator)
        domain = str(locator.pop("domain", "") or "general")
        subdomain = str(locator.pop("subdomain", "") or "")
        knowledge_type = str(locator.pop("knowledge_type", "") or "domain")
        locator.pop("content_hash", None)
        if hit is not None and trace is not None:
            locator["graph_rag"] = {
                "mode": trace.mode,
                "origin": hit.origin,
                "score": hit.score,
                "graph_score": hit.graph_score,
                "hops": hit.hops,
                "path": list(hit.path.node_ids) if hit.path else [],
                "relations": list(hit.path.relation_types) if hit.path else [],
                "fallback_used": trace.fallback_used,
                "timings_ms": dict(trace.timings_ms),
            }
        return InterviewKnowledgeChunk(
            chunk_id=node.node_id,
            corpus_id=node.corpus_id,
            domain=domain,
            subdomain=subdomain,
            topic=node.title,
            text=node.text,
            source_id=node.source_id,
            locator=locator,
            difficulty=node.difficulty,
            knowledge_type=knowledge_type,
        )

    def _sequence_edges(self, corpus: InterviewKnowledgeCorpus) -> list:
        by_id = {chunk.chunk_id: chunk for chunk in corpus.chunks}
        unique: dict[tuple[str, str], Any] = {}
        for index, chunk in enumerate(corpus.chunks):
            neighbor_ids = {
                str(chunk.locator.get("prev_chunk_id") or ""),
                str(chunk.locator.get("next_chunk_id") or ""),
            }
            if index:
                neighbor_ids.add(corpus.chunks[index - 1].chunk_id)
            if index + 1 < len(corpus.chunks):
                neighbor_ids.add(corpus.chunks[index + 1].chunk_id)
            for target_id in neighbor_ids:
                if not target_id or target_id == chunk.chunk_id or target_id not in by_id:
                    continue
                key = tuple(sorted((chunk.chunk_id, target_id)))
                unique[key] = self._KnowledgeEdge(
                    key[0],
                    key[1],
                    "related",
                    0.85,
                    True,
                )
        return list(unique.values())
