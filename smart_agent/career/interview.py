"""Framework contracts for traceable, adaptive mock interviews."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar, Protocol

INTERVIEW_SCHEMA_VERSION = "1.5"
_SUPPORTED_INTERVIEW_SCHEMA_VERSIONS = frozenset({"1.0", "1.1", "1.2", "1.3", "1.4", INTERVIEW_SCHEMA_VERSION})


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    return text


def _clone(value: Any, field_name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable.") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


class InterviewKnowledgeType(str, Enum):
    FUNDAMENTAL = "fundamental"
    DOMAIN = "domain"
    COMPANY = "company"
    USER_NOTE = "user_note"


class InterviewMode(str, Enum):
    REALISTIC_SIMULATION = "realistic_simulation"
    GAP_ANALYSIS = "gap_analysis"


class InterviewPath(str, Enum):
    BREADTH = "breadth"
    DEPTH = "depth"


class InterviewKnowledgeBlockRole(str, Enum):
    A = "A"
    B = "B"
    D = "D"


class InterviewQuestionType(str, Enum):
    KNOWLEDGE = "knowledge"
    PROJECT_SCENARIO = "project_scenario"
    CASE = "case"
    BEHAVIORAL = "behavioral"
    OPEN = "open"


class InterviewLine(str, Enum):
    INTERNSHIP = "internship"
    PROJECT = "project"
    SKILL = "skill"


class InterviewStage(str, Enum):
    INTRO = "intro"
    INTERNSHIP_SUBJECT = "internship_subject"
    PROJECT_SUBJECT = "project_subject"
    PURE_FOUNDATION = "pure_foundation"
    CANDIDATE_QUESTION_WAIT = "candidate_question_wait"
    COMPLETED = "completed"


class InterviewStopReason(str, Enum):
    USER_DOES_NOT_KNOW = "user_does_not_know"
    NO_NEW_INFORMATION = "no_new_information"
    FOLLOWUP_LIMIT = "followup_limit"
    MASTERED = "mastered"
    LOW_MASTERY = "low_mastery"
    RELEVANCE_DECLINED = "relevance_declined"
    TURN_LIMIT = "turn_limit"
    USER_FINISHED = "user_finished"
    CANCELLED = "cancelled"
    EVALUATION_FAILED = "evaluation_failed"
    KNOWLEDGE_EXHAUSTED = "knowledge_exhausted"


class InterviewAction(str, Enum):
    ASK = "ask"
    FOLLOW_UP = "follow_up"
    SWITCH_TOPIC = "switch_topic"
    END = "end"


class InterviewStatus(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class InterviewKnowledgeChunk:
    chunk_id: str
    corpus_id: str
    domain: str
    topic: str
    text: str
    source_id: str
    locator: dict[str, Any] = field(default_factory=dict)
    subdomain: str = ""
    difficulty: int = 3
    knowledge_type: InterviewKnowledgeType = InterviewKnowledgeType.FUNDAMENTAL
    content_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("chunk_id", "corpus_id", "domain", "topic", "text", "source_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "subdomain", str(self.subdomain or "").strip())
        object.__setattr__(self, "knowledge_type", InterviewKnowledgeType(self.knowledge_type))
        object.__setattr__(self, "locator", _clone(self.locator, "locator"))
        if not 1 <= self.difficulty <= 5:
            raise ValueError("difficulty must be between 1 and 5.")
        expected = _digest(
            {
                "chunk_id": self.chunk_id,
                "corpus_id": self.corpus_id,
                "domain": self.domain,
                "topic": self.topic,
                "text": self.text,
                "source_id": self.source_id,
                "locator": self.locator,
                "subdomain": self.subdomain,
                "difficulty": self.difficulty,
                "knowledge_type": self.knowledge_type.value,
            }
        )
        if self.content_hash and self.content_hash != expected:
            raise ValueError("InterviewKnowledgeChunk content_hash does not match its content.")
        object.__setattr__(self, "content_hash", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "corpus_id": self.corpus_id,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "topic": self.topic,
            "difficulty": self.difficulty,
            "knowledge_type": self.knowledge_type.value,
            "text": self.text,
            "source_id": self.source_id,
            "locator": self.locator,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class InterviewKnowledgeCorpus:
    corpus_id: str
    name: str
    chunks: tuple[InterviewKnowledgeChunk, ...]
    domain: str = "general"
    version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "corpus_id", _required_text(self.corpus_id, "corpus_id"))
        object.__setattr__(self, "name", _required_text(self.name, "name"))
        object.__setattr__(self, "domain", _required_text(self.domain, "domain"))
        object.__setattr__(self, "version", _required_text(self.version, "version"))
        chunks = tuple(self.chunks)
        if not chunks:
            raise ValueError("InterviewKnowledgeCorpus requires at least one chunk.")
        if any(chunk.corpus_id != self.corpus_id for chunk in chunks):
            raise ValueError("All knowledge chunks must belong to the corpus.")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Knowledge chunk IDs must be unique.")
        object.__setattr__(self, "chunks", chunks)

    def search(self, query: str, *, top_k: int = 5, topic: str | None = None) -> tuple[InterviewKnowledgeChunk, ...]:
        query_tokens = set(str(query or "").casefold().split())
        if not query_tokens:
            raise ValueError("query is required")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        candidates = [chunk for chunk in self.chunks if topic is None or chunk.topic == topic]
        ranked = sorted(
            candidates,
            key=lambda chunk: (
                -len(query_tokens.intersection(set(chunk.text.casefold().split()))),
                chunk.difficulty,
                chunk.chunk_id,
            ),
        )
        return tuple(ranked[:top_k])

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "name": self.name,
            "domain": self.domain,
            "version": self.version,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InterviewKnowledgeCorpus":
        corpus_id = str(value["corpus_id"])
        chunks = tuple(
            InterviewKnowledgeChunk(
                chunk_id=str(item["chunk_id"]),
                corpus_id=str(item.get("corpus_id") or corpus_id),
                domain=str(item.get("domain") or value.get("domain") or "general"),
                subdomain=str(item.get("subdomain") or ""),
                topic=str(item["topic"]),
                difficulty=int(item.get("difficulty", 3)),
                knowledge_type=InterviewKnowledgeType(item.get("knowledge_type", InterviewKnowledgeType.FUNDAMENTAL)),
                text=str(item["text"]),
                source_id=str(item["source_id"]),
                locator=dict(item.get("locator") or {}),
                content_hash=str(item.get("content_hash") or ""),
            )
            for item in value.get("chunks", ())
        )
        return cls(
            corpus_id=corpus_id,
            name=str(value["name"]),
            chunks=chunks,
            domain=str(value.get("domain") or "general"),
            version=str(value.get("version") or "1"),
        )


@dataclass(frozen=True, slots=True)
class InterviewKnowledgeCorpusMetadata:
    corpus_id: str
    name: str
    domain: str
    version: str
    chunk_count: int

    def __post_init__(self) -> None:
        for name in ("corpus_id", "name", "domain", "version"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if int(self.chunk_count) < 0:
            raise ValueError("chunk_count cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "name": self.name,
            "domain": self.domain,
            "version": self.version,
            "chunk_count": self.chunk_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InterviewKnowledgeCorpusMetadata":
        return cls(
            corpus_id=str(value["corpus_id"]),
            name=str(value.get("name") or value["corpus_id"]),
            domain=str(value.get("domain") or "general"),
            version=str(value.get("version") or "1"),
            chunk_count=int(value.get("chunk_count") or 0),
        )


class InterviewKnowledgeRetriever(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        topic: str | None = None,
    ) -> tuple[InterviewKnowledgeChunk, ...]: ...


class LexicalInterviewRetriever:
    """Deterministic baseline retriever; semantic/Reranker implementations can replace it."""

    name = "lexical-interview-baseline"

    def __init__(self, corpora: tuple[InterviewKnowledgeCorpus, ...] = ()) -> None:
        self.corpora = corpora

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        topic: str | None = None,
    ) -> tuple[InterviewKnowledgeChunk, ...]:
        hits: list[InterviewKnowledgeChunk] = []
        for corpus in self.corpora:
            hits.extend(corpus.search(query, top_k=top_k, topic=topic))
        return tuple(hits[:top_k])


class InterviewKnowledgeStore:
    """Persistent corpus store kept outside personal evidence and project knowledge."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = os.fspath(root)
        self.corpora_dir = os.path.abspath(os.path.join(self.root, "corpora"))
        self.metadata_dir = os.path.abspath(os.path.join(self.root, "corpus-metadata"))
        os.makedirs(self.corpora_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def put(self, corpus: InterviewKnowledgeCorpus) -> InterviewKnowledgeCorpus:
        path = self._path(corpus.corpus_id)
        if os.path.exists(path):
            current = InterviewKnowledgeCorpus.from_dict(self._read(path))
            if current != corpus:
                raise ValueError(f"Interview corpus {corpus.corpus_id} already exists with different content")
            self._write_metadata(current)
            return current
        self._write(path, corpus.to_dict())
        self._write_metadata(corpus)
        return corpus

    def append(self, chunk: InterviewKnowledgeChunk, *, name: str = "Interview knowledge", domain: str = "general") -> InterviewKnowledgeCorpus:
        try:
            corpus = self.get(chunk.corpus_id)
        except KeyError:
            corpus = InterviewKnowledgeCorpus(
                corpus_id=chunk.corpus_id,
                name=name,
                domain=domain,
                chunks=(chunk,),
            )
        else:
            existing = next((item for item in corpus.chunks if item.chunk_id == chunk.chunk_id), None)
            if existing is not None:
                if existing != chunk:
                    raise ValueError(f"Knowledge chunk {chunk.chunk_id} already exists with different content")
                return corpus
            corpus = InterviewKnowledgeCorpus(
                corpus_id=corpus.corpus_id,
                name=corpus.name,
                domain=corpus.domain,
                version=str(int(corpus.version) + 1) if corpus.version.isdigit() else f"{corpus.version}.1",
                chunks=(*corpus.chunks, chunk),
            )
            self._write(self._path(corpus.corpus_id), corpus.to_dict())
            self._write_metadata(corpus)
            return corpus
        return self.put(corpus)

    def get(self, corpus_id: str) -> InterviewKnowledgeCorpus:
        path = self._path(corpus_id)
        if not os.path.exists(path):
            raise KeyError(f"Unknown interview corpus: {corpus_id}")
        return InterviewKnowledgeCorpus.from_dict(self._read(path))

    def delete(self, corpus_id: str) -> int:
        """Delete one persisted corpus and its metadata, idempotently."""
        path = self._path(corpus_id)
        metadata_path = self._metadata_path(corpus_id)
        removed = 0
        for target in (path, metadata_path):
            try:
                os.unlink(target)
            except FileNotFoundError:
                continue
            removed += 1
        return removed

    def list(self) -> tuple[InterviewKnowledgeCorpus, ...]:
        return tuple(
            InterviewKnowledgeCorpus.from_dict(self._read(os.path.join(self.corpora_dir, name)))
            for name in sorted(os.listdir(self.corpora_dir))
            if name.endswith(".json")
        )

    def get_metadata(self, corpus_id: str) -> InterviewKnowledgeCorpusMetadata:
        path = self._metadata_path(corpus_id)
        if os.path.exists(path):
            return InterviewKnowledgeCorpusMetadata.from_dict(self._read(path))
        corpus_path = self._path(corpus_id)
        if not os.path.exists(corpus_path):
            raise KeyError(f"Unknown interview corpus: {corpus_id}")
        metadata = self._scan_legacy_metadata(corpus_path, corpus_id)
        self._write(path, metadata.to_dict())
        return metadata

    def list_metadata(self) -> tuple[InterviewKnowledgeCorpusMetadata, ...]:
        return tuple(
            self.get_metadata(os.path.splitext(name)[0])
            for name in sorted(os.listdir(self.corpora_dir))
            if name.endswith(".json")
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        topic: str | None = None,
        corpus_id: str | None = None,
    ) -> tuple[InterviewKnowledgeChunk, ...]:
        corpora = (self.get(corpus_id),) if corpus_id else self.list()
        return LexicalInterviewRetriever(corpora).search(query, top_k=top_k, topic=topic)

    def _path(self, corpus_id: str) -> str:
        normalized = "".join(char for char in str(corpus_id).strip() if char.isalnum() or char in "-_ .").strip()
        if not normalized:
            raise ValueError("corpus_id is required")
        return os.path.join(self.corpora_dir, f"{normalized}.json")

    def _metadata_path(self, corpus_id: str) -> str:
        normalized = "".join(char for char in str(corpus_id).strip() if char.isalnum() or char in "-_ .").strip()
        if not normalized:
            raise ValueError("corpus_id is required")
        return os.path.join(self.metadata_dir, f"{normalized}.json")

    def _write_metadata(self, corpus: InterviewKnowledgeCorpus) -> None:
        metadata = InterviewKnowledgeCorpusMetadata(
            corpus_id=corpus.corpus_id,
            name=corpus.name,
            domain=corpus.domain,
            version=corpus.version,
            chunk_count=len(corpus.chunks),
        )
        self._write(self._metadata_path(corpus.corpus_id), metadata.to_dict())

    @staticmethod
    def _scan_legacy_metadata(path: str, corpus_id: str) -> InterviewKnowledgeCorpusMetadata:
        with open(path, "rb") as handle:
            raw = handle.read()
        chunk_count = raw.count(b'"chunk_id"')
        head = raw[: min(len(raw), 8192)].decode("utf-8", errors="replace")

        def field(name: str, default: str) -> str:
            match = re.search(rf'"{name}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', head)
            if match is None:
                return default
            try:
                return str(json.loads(f'"{match.group(1)}"'))
            except json.JSONDecodeError:
                return default

        return InterviewKnowledgeCorpusMetadata(
            corpus_id=field("corpus_id", corpus_id),
            name=field("name", corpus_id),
            domain=field("domain", "general"),
            version=field("version", "1"),
            chunk_count=chunk_count,
        )

    @staticmethod
    def _read(path: str) -> dict[str, Any]:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("Interview corpus file must contain an object")
        return value

    @staticmethod
    def _write(path: str, value: dict[str, Any]) -> None:
        directory = os.path.dirname(path)
        descriptor, temporary = tempfile.mkstemp(prefix=".interview.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise


@dataclass(frozen=True, slots=True)
class InterviewerProfile:
    profile_id: str
    name: str
    temperament: str = "balanced"
    seniority: str = "senior"
    pressure_level: float = 0.5
    breadth_bias: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _required_text(self.profile_id, "profile_id"))
        object.__setattr__(self, "name", _required_text(self.name, "name"))
        object.__setattr__(self, "temperament", _required_text(self.temperament, "temperament"))
        object.__setattr__(self, "seniority", _required_text(self.seniority, "seniority"))
        for field_name in ("pressure_level", "breadth_bias"):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class InterviewerPolicy:
    """Runtime policy selected by the system, not a user-controlled chat toggle."""

    policy_id: str
    max_followups_per_thread: int = 3
    depth_threshold: float = 0.62
    breadth_threshold: float = 0.78
    project_fact_policy: str = "assume_user_control"
    knowledge_update_policy: str = "incremental-average-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _required_text(self.policy_id, "policy_id"))
        if self.max_followups_per_thread < 1:
            raise ValueError("max_followups_per_thread must be at least 1")
        for field_name in ("depth_threshold", "breadth_threshold"):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)
        if self.depth_threshold > self.breadth_threshold:
            raise ValueError("depth_threshold cannot exceed breadth_threshold")
        if self.project_fact_policy != "assume_user_control":
            raise ValueError("Project fact evaluation is intentionally disabled in this policy.")


@dataclass(frozen=True, slots=True)
class KnowledgeConfidenceUpdate:
    """One runtime-validated knowledge-point observation and its committed result."""

    knowledge_point_id: str
    corpus_id: str
    observed_confidence: float
    reason: str = ""
    confidence_before: float | None = None
    confidence_after: float | None = None
    observation_count_before: int = 0
    document_confidence_before: float | None = None
    document_confidence_after: float | None = None
    document_observation_count_before: int = 0
    document_prior_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "knowledge_point_id", _required_text(self.knowledge_point_id, "knowledge_point_id"))
        object.__setattr__(self, "corpus_id", _required_text(self.corpus_id, "corpus_id"))
        object.__setattr__(self, "reason", str(self.reason or "").strip())
        for field_name in ("observed_confidence", "confidence_before", "confidence_after", "document_confidence_before", "document_confidence_after"):
            value = getattr(self, field_name)
            if value is None:
                continue
            value = float(value)
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)
        for field_name in ("observation_count_before", "document_observation_count_before"):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative")
            object.__setattr__(self, field_name, value)
        if self.document_prior_count is not None:
            prior_count = int(self.document_prior_count)
            if prior_count < 1:
                raise ValueError("document_prior_count must be positive")
            object.__setattr__(self, "document_prior_count", prior_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_point_id": self.knowledge_point_id,
            "corpus_id": self.corpus_id,
            "observed_confidence": self.observed_confidence,
            "reason": self.reason,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
            "observation_count_before": self.observation_count_before,
            "document_confidence_before": self.document_confidence_before,
            "document_confidence_after": self.document_confidence_after,
            "document_observation_count_before": self.document_observation_count_before,
            "document_prior_count": self.document_prior_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KnowledgeConfidenceUpdate":
        return cls(
            knowledge_point_id=str(value.get("knowledge_point_id") or value.get("chunk_id") or ""),
            corpus_id=str(value.get("corpus_id") or ""),
            observed_confidence=float(value.get("observed_confidence", value.get("nc", value.get("confidence", 0.5)))),
            reason=str(value.get("reason") or ""),
            confidence_before=(float(value["confidence_before"]) if value.get("confidence_before") is not None else None),
            confidence_after=(float(value["confidence_after"]) if value.get("confidence_after") is not None else None),
            observation_count_before=int(value.get("observation_count_before", 0)),
            document_confidence_before=(
                float(value["document_confidence_before"])
                if value.get("document_confidence_before") is not None
                else None
            ),
            document_confidence_after=(
                float(value["document_confidence_after"])
                if value.get("document_confidence_after") is not None
                else None
            ),
            document_observation_count_before=int(value.get("document_observation_count_before", 0)),
            document_prior_count=(int(value["document_prior_count"]) if value.get("document_prior_count") is not None else None),
        )


@dataclass(frozen=True, slots=True)
class KnowledgeBeliefState:
    """Two-layer mastery state: knowledge blocks inside frozen interview corpora."""

    point_confidence_by_id: dict[str, float] = field(default_factory=dict)
    point_prior_count_by_id: dict[str, int] = field(default_factory=dict)
    point_observation_count_by_id: dict[str, int] = field(default_factory=dict)
    document_confidence_by_corpus: dict[str, float] = field(default_factory=dict)
    document_prior_count_by_corpus: dict[str, int] = field(default_factory=dict)
    document_observation_count_by_corpus: dict[str, int] = field(default_factory=dict)
    document_covered_point_ids_by_corpus: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Read-only compatibility snapshots for pre-1.4 callers. They are never
    # consumed by the recommendation formula.
    legacy_confidence_by_topic: dict[str, float] = field(default_factory=dict, repr=False)
    legacy_observations_by_topic: dict[str, int] = field(default_factory=dict, repr=False)
    legacy_alpha_by_topic: dict[str, float] = field(default_factory=dict, repr=False)
    legacy_beta_by_topic: dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for field_name in ("point_confidence_by_id", "document_confidence_by_corpus"):
            values = {str(key): float(value) for key, value in dict(getattr(self, field_name)).items()}
            if any(not 0 <= value <= 1 or not math.isfinite(value) for value in values.values()):
                raise ValueError(f"{field_name} values must be between 0 and 1")
            object.__setattr__(self, field_name, values)
        for field_name in (
            "point_prior_count_by_id",
            "point_observation_count_by_id",
            "document_prior_count_by_corpus",
            "document_observation_count_by_corpus",
        ):
            values = {str(key): int(value) for key, value in dict(getattr(self, field_name)).items()}
            if any(value < 0 for value in values.values()):
                raise ValueError(f"{field_name} values cannot be negative")
            object.__setattr__(self, field_name, values)
        covered = {
            str(corpus_id): tuple(dict.fromkeys(_required_text(point_id, "knowledge_point_id") for point_id in point_ids))
            for corpus_id, point_ids in dict(self.document_covered_point_ids_by_corpus).items()
        }
        object.__setattr__(self, "document_covered_point_ids_by_corpus", covered)
        for field_name in ("legacy_confidence_by_topic", "legacy_alpha_by_topic", "legacy_beta_by_topic"):
            values = {str(key): float(value) for key, value in dict(getattr(self, field_name)).items()}
            object.__setattr__(self, field_name, values)
        legacy_observations = {str(key): int(value) for key, value in dict(self.legacy_observations_by_topic).items()}
        object.__setattr__(self, "legacy_observations_by_topic", legacy_observations)

    @property
    def confidence_by_topic(self) -> dict[str, float]:
        return dict(self.legacy_confidence_by_topic)

    @property
    def observations_by_topic(self) -> dict[str, int]:
        return dict(self.legacy_observations_by_topic)

    @property
    def alpha_by_topic(self) -> dict[str, float]:
        return dict(self.legacy_alpha_by_topic)

    @property
    def beta_by_topic(self) -> dict[str, float]:
        return dict(self.legacy_beta_by_topic)

    def point_confidence(self, knowledge_point_id: str) -> float:
        return self.point_confidence_by_id.get(_required_text(knowledge_point_id, "knowledge_point_id"), 0.5)

    def point_observation_count(self, knowledge_point_id: str) -> int:
        return self.point_observation_count_by_id.get(_required_text(knowledge_point_id, "knowledge_point_id"), 0)

    def document_confidence(self, corpus_id: str) -> float:
        return self.document_confidence_by_corpus.get(_required_text(corpus_id, "corpus_id"), 0.5)

    def document_coverage(self, corpus_id: str) -> float:
        corpus_id = _required_text(corpus_id, "corpus_id")
        prior_count = self.document_prior_count_by_corpus.get(corpus_id, 0)
        if prior_count <= 0:
            return 0.0
        return min(1.0, len(self.document_covered_point_ids_by_corpus.get(corpus_id, ())) / prior_count)

    def with_document_priors(self, prior_counts: dict[str, int]) -> "KnowledgeBeliefState":
        normalized = {str(corpus_id): max(1, int(count)) for corpus_id, count in prior_counts.items()}
        if not normalized:
            return self
        return replace(
            self,
            document_prior_count_by_corpus={
                **normalized,
                **self.document_prior_count_by_corpus,
            },
        )

    def apply_updates(
        self,
        updates: tuple[KnowledgeConfidenceUpdate, ...],
    ) -> tuple["KnowledgeBeliefState", tuple[KnowledgeConfidenceUpdate, ...]]:
        """Apply the documented incremental-average update at both state layers."""
        point_confidence = dict(self.point_confidence_by_id)
        point_priors = dict(self.point_prior_count_by_id)
        point_observations = dict(self.point_observation_count_by_id)
        document_confidence = dict(self.document_confidence_by_corpus)
        document_priors = dict(self.document_prior_count_by_corpus)
        document_observations = dict(self.document_observation_count_by_corpus)
        covered = {key: tuple(value) for key, value in self.document_covered_point_ids_by_corpus.items()}
        committed: list[KnowledgeConfidenceUpdate] = []
        for update in updates:
            point_id = update.knowledge_point_id
            corpus_id = update.corpus_id
            point_before = point_confidence.get(point_id, 0.5)
            point_prior = max(1, point_priors.get(point_id, 1))
            point_observation_count = point_observations.get(point_id, 0)
            point_after = (
                point_before * (point_prior + point_observation_count) + update.observed_confidence
            ) / (point_prior + point_observation_count + 1)
            document_before = document_confidence.get(corpus_id, 0.5)
            document_prior = max(1, document_priors.get(corpus_id, update.document_prior_count or 1))
            document_observation_count = document_observations.get(corpus_id, 0)
            document_after = (
                document_before * (document_prior + document_observation_count) + update.observed_confidence
            ) / (document_prior + document_observation_count + 1)
            point_confidence[point_id] = point_after
            point_priors[point_id] = point_prior
            point_observations[point_id] = point_observation_count + 1
            document_confidence[corpus_id] = document_after
            document_priors[corpus_id] = document_prior
            document_observations[corpus_id] = document_observation_count + 1
            covered[corpus_id] = tuple(dict.fromkeys((*covered.get(corpus_id, ()), point_id)))
            committed.append(
                replace(
                    update,
                    confidence_before=point_before,
                    confidence_after=point_after,
                    observation_count_before=point_observation_count,
                    document_confidence_before=document_before,
                    document_confidence_after=document_after,
                    document_observation_count_before=document_observation_count,
                    document_prior_count=document_prior,
                )
            )
        return (
            KnowledgeBeliefState(
                point_confidence_by_id=point_confidence,
                point_prior_count_by_id=point_priors,
                point_observation_count_by_id=point_observations,
                document_confidence_by_corpus=document_confidence,
                document_prior_count_by_corpus=document_priors,
                document_observation_count_by_corpus=document_observations,
                document_covered_point_ids_by_corpus=covered,
                legacy_confidence_by_topic=self.legacy_confidence_by_topic,
                legacy_observations_by_topic=self.legacy_observations_by_topic,
                legacy_alpha_by_topic=self.legacy_alpha_by_topic,
                legacy_beta_by_topic=self.legacy_beta_by_topic,
            ),
            tuple(committed),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_confidence_by_id": self.point_confidence_by_id,
            "point_prior_count_by_id": self.point_prior_count_by_id,
            "point_observation_count_by_id": self.point_observation_count_by_id,
            "document_confidence_by_corpus": self.document_confidence_by_corpus,
            "document_prior_count_by_corpus": self.document_prior_count_by_corpus,
            "document_observation_count_by_corpus": self.document_observation_count_by_corpus,
            "document_covered_point_ids_by_corpus": {
                corpus_id: list(point_ids)
                for corpus_id, point_ids in self.document_covered_point_ids_by_corpus.items()
            },
            "legacy_confidence_by_topic": self.legacy_confidence_by_topic,
            "legacy_observations_by_topic": self.legacy_observations_by_topic,
            "legacy_alpha_by_topic": self.legacy_alpha_by_topic,
            "legacy_beta_by_topic": self.legacy_beta_by_topic,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "KnowledgeBeliefState":
        value = dict(value or {})
        if "point_confidence_by_id" not in value:
            # Legacy snapshots held topic-level Bayesian values. They cannot be
            # safely mapped to knowledge-block IDs, so preserve readability and
            # start the new block-level model from its documented neutral prior.
            return cls(
                legacy_confidence_by_topic=dict(
                    value.get("legacy_confidence_by_topic", value.get("confidence_by_topic", {})) or {}
                ),
                legacy_observations_by_topic=dict(
                    value.get("legacy_observations_by_topic", value.get("observations_by_topic", {})) or {}
                ),
                legacy_alpha_by_topic=dict(
                    value.get("legacy_alpha_by_topic", value.get("alpha_by_topic", {})) or {}
                ),
                legacy_beta_by_topic=dict(
                    value.get("legacy_beta_by_topic", value.get("beta_by_topic", {})) or {}
                ),
            )
        return cls(
            point_confidence_by_id=dict(value.get("point_confidence_by_id") or {}),
            point_prior_count_by_id=dict(value.get("point_prior_count_by_id") or {}),
            point_observation_count_by_id=dict(value.get("point_observation_count_by_id") or {}),
            document_confidence_by_corpus=dict(value.get("document_confidence_by_corpus") or {}),
            document_prior_count_by_corpus=dict(value.get("document_prior_count_by_corpus") or {}),
            document_observation_count_by_corpus=dict(value.get("document_observation_count_by_corpus") or {}),
            document_covered_point_ids_by_corpus={
                str(corpus_id): tuple(point_ids)
                for corpus_id, point_ids in dict(value.get("document_covered_point_ids_by_corpus") or {}).items()
            },
            legacy_confidence_by_topic=dict(value.get("legacy_confidence_by_topic") or {}),
            legacy_observations_by_topic=dict(value.get("legacy_observations_by_topic") or {}),
            legacy_alpha_by_topic=dict(value.get("legacy_alpha_by_topic") or {}),
            legacy_beta_by_topic=dict(value.get("legacy_beta_by_topic") or {}),
        )


@dataclass(frozen=True, slots=True)
class QuestionThread:
    thread_id: str
    root_topic: str
    path: InterviewPath
    followup_count: int = 0
    max_followups: int = 3
    status: str = "active"
    stop_reason: InterviewStopReason | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "thread_id", _required_text(self.thread_id, "thread_id"))
        object.__setattr__(self, "root_topic", _required_text(self.root_topic, "root_topic"))
        object.__setattr__(self, "path", InterviewPath(self.path))
        if self.followup_count < 0 or self.max_followups < 1 or self.followup_count > self.max_followups:
            raise ValueError("Invalid question thread follow-up count")
        if self.stop_reason is not None:
            object.__setattr__(self, "stop_reason", InterviewStopReason(self.stop_reason))


@dataclass(frozen=True, slots=True)
class QuestionUtility:
    jd_relevance: float
    resume_relevance: float
    knowledge_uncertainty: float
    weakness: float
    profile_preference: float
    repetition: float
    followup_cost: float
    score: float
    interest_factor: float = 0.5
    confidence: float = 0.5
    information_gain: float = 0.7
    verification_need: float = 0.5
    consecutive_depth: int = 0
    breadth_weight: float = 0.0
    depth_weight: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "jd_relevance",
            "resume_relevance",
            "knowledge_uncertainty",
            "weakness",
            "profile_preference",
            "repetition",
            "followup_cost",
            "score",
        ):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)
        if not 0 <= float(self.interest_factor) <= 1:
            raise ValueError("interest_factor must be between 0 and 1")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        for field_name in ("information_gain", "verification_need"):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be between 0 and 1")
        for field_name in ("breadth_weight", "depth_weight"):
            value = float(getattr(self, field_name))
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be non-negative and finite")
        if int(self.consecutive_depth) < 0:
            raise ValueError("consecutive_depth cannot be negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "jd_relevance": self.jd_relevance,
            "resume_relevance": self.resume_relevance,
            "knowledge_uncertainty": self.knowledge_uncertainty,
            "weakness": self.weakness,
            "profile_preference": self.profile_preference,
            "repetition": self.repetition,
            "followup_cost": self.followup_cost,
            "score": self.score,
            "interest_factor": self.interest_factor,
            "confidence": self.confidence,
            "information_gain": self.information_gain,
            "verification_need": self.verification_need,
            "consecutive_depth": self.consecutive_depth,
            "breadth_weight": self.breadth_weight,
            "depth_weight": self.depth_weight,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "QuestionUtility | None":
        if not value:
            return None
        payload = {key: float(value[key]) for key in (
            "jd_relevance",
            "resume_relevance",
            "knowledge_uncertainty",
            "weakness",
            "profile_preference",
            "repetition",
            "followup_cost",
            "score",
        )}
        for key in ("interest_factor", "confidence", "information_gain", "verification_need", "breadth_weight", "depth_weight"):
            if key in value:
                payload[key] = float(value[key])
        if "consecutive_depth" in value:
            payload["consecutive_depth"] = int(value["consecutive_depth"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class InterviewKnowledgeBlockRef:
    """One authorized knowledge block with its interview-routing role."""

    role: InterviewKnowledgeBlockRole
    label: str
    chunk_id: str
    point_id: str
    sequence: int = 0
    related_to_label: str | None = None
    related_to_chunk_id: str | None = None
    retrieval_trace: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", InterviewKnowledgeBlockRole(self.role))
        for field_name in ("label", "chunk_id", "point_id"):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        if int(self.sequence) < 0:
            raise ValueError("knowledge block sequence cannot be negative")
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "related_to_label", str(self.related_to_label or "").strip() or None)
        object.__setattr__(self, "related_to_chunk_id", str(self.related_to_chunk_id or "").strip() or None)
        object.__setattr__(self, "retrieval_trace", _clone(self.retrieval_trace, "retrieval_trace"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "label": self.label,
            "chunk_id": self.chunk_id,
            "point_id": self.point_id,
            "sequence": self.sequence,
            "related_to_label": self.related_to_label,
            "related_to_chunk_id": self.related_to_chunk_id,
            "retrieval_trace": self.retrieval_trace,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InterviewKnowledgeBlockRef":
        return cls(
            role=InterviewKnowledgeBlockRole(value.get("role", InterviewKnowledgeBlockRole.A.value)),
            label=str(value.get("label") or value.get("block_label") or "A1"),
            chunk_id=str(value.get("chunk_id") or value.get("knowledge_point_id") or ""),
            point_id=str(value.get("point_id") or ""),
            sequence=int(value.get("sequence", 0)),
            related_to_label=str(value.get("related_to_label") or "") or None,
            related_to_chunk_id=str(value.get("related_to_chunk_id") or "") or None,
            retrieval_trace=dict(value.get("retrieval_trace") or {}),
        )


@dataclass(frozen=True, slots=True)
class InterviewPoint:
    """A resume-anchored capability point selected before knowledge chunks."""

    point_id: str
    point: str
    category: InterviewLine
    interest_score: float
    reason: str
    jd_refs: tuple[str, ...] = ()
    resume_refs: tuple[str, ...] = ()
    question_directions: tuple[str, ...] = ()
    confidence: float = 0.5
    subject: str = ""
    static_interest_score: float | None = None
    intro_adjustment_score: float = 0.0
    cached_knowledge_refs: tuple[str, ...] = ()
    knowledge_block_refs: tuple[InterviewKnowledgeBlockRef, ...] = ()
    asked_count: int = 0
    point_status: str = "unasked"

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", _required_text(self.point_id, "point_id"))
        object.__setattr__(self, "point", _required_text(self.point, "point"))
        object.__setattr__(self, "category", InterviewLine(self.category))
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        for field_name in ("interest_score", "confidence"):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "jd_refs", tuple(_required_text(item, "jd_ref") for item in self.jd_refs))
        object.__setattr__(self, "resume_refs", tuple(_required_text(item, "resume_ref") for item in self.resume_refs))
        object.__setattr__(
            self,
            "question_directions",
            tuple(_required_text(item, "question_direction") for item in self.question_directions),
        )
        subject = str(self.subject or "").strip() or self.point
        object.__setattr__(self, "subject", subject)
        static_interest = self.interest_score if self.static_interest_score is None else float(self.static_interest_score)
        if not 0 <= static_interest <= 1 or not math.isfinite(static_interest):
            raise ValueError("static_interest_score must be between 0 and 1")
        object.__setattr__(self, "static_interest_score", static_interest)
        intro_adjustment = float(self.intro_adjustment_score)
        if not -1 <= intro_adjustment <= 1 or not math.isfinite(intro_adjustment):
            raise ValueError("intro_adjustment_score must be between -1 and 1")
        object.__setattr__(self, "intro_adjustment_score", intro_adjustment)
        object.__setattr__(
            self,
            "cached_knowledge_refs",
            tuple(_required_text(item, "cached_knowledge_ref") for item in self.cached_knowledge_refs),
        )
        object.__setattr__(
            self,
            "knowledge_block_refs",
            tuple(
                item if isinstance(item, InterviewKnowledgeBlockRef) else InterviewKnowledgeBlockRef.from_dict(dict(item))
                for item in self.knowledge_block_refs
            ),
        )
        if int(self.asked_count) < 0:
            raise ValueError("asked_count cannot be negative")
        object.__setattr__(self, "asked_count", int(self.asked_count))
        object.__setattr__(self, "point_status", str(self.point_status or "unasked").strip() or "unasked")

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "point": self.point,
            "category": self.category.value,
            "interest_score": self.interest_score,
            "reason": self.reason,
            "jd_refs": list(self.jd_refs),
            "resume_refs": list(self.resume_refs),
            "question_directions": list(self.question_directions),
            "confidence": self.confidence,
            "subject": self.subject,
            "static_interest_score": self.static_interest_score,
            "intro_adjustment_score": self.intro_adjustment_score,
            "cached_knowledge_refs": list(self.cached_knowledge_refs),
            "knowledge_block_refs": [item.to_dict() for item in self.knowledge_block_refs],
            "asked_count": self.asked_count,
            "point_status": self.point_status,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InterviewPoint":
        return cls(
            point_id=str(value["point_id"]),
            point=str(value["point"]),
            category=InterviewLine(value["category"]),
            interest_score=float(value.get("interest_score", 0.5)),
            reason=str(value.get("reason") or "Imported interview point"),
            jd_refs=tuple(value.get("jd_refs", ())),
            resume_refs=tuple(value.get("resume_refs", ())),
            question_directions=tuple(value.get("question_directions", ())),
            confidence=float(value.get("confidence", 0.5)),
            subject=str(value.get("subject") or ""),
            static_interest_score=(
                float(value["static_interest_score"])
                if value.get("static_interest_score") is not None
                else None
            ),
            intro_adjustment_score=float(value.get("intro_adjustment_score", 0.0)),
            cached_knowledge_refs=tuple(value.get("cached_knowledge_refs", ())),
            knowledge_block_refs=tuple(
                InterviewKnowledgeBlockRef.from_dict(dict(item))
                for item in value.get("knowledge_block_refs", ())
                if isinstance(item, dict)
            ),
            asked_count=int(value.get("asked_count", 0)),
            point_status=str(value.get("point_status") or "unasked"),
        )


@dataclass(frozen=True, slots=True)
class RecommendationTrace:
    """Auditable record for one recommendation decision."""

    turn: int
    thread_id: str
    point_id: str
    previous_question_id: str | None
    information_gain: str
    verification_need: str
    interest_current: float
    confidence_current_before: float
    confidence_observation: float
    confidence_current_after: float
    continuous_depth_count: int
    coefficients: dict[str, float]
    weights: dict[str, float]
    allowed_actions: tuple[str, ...]
    candidate_knowledge_point_ids: tuple[str, ...]
    selected_action: str
    selected_knowledge_point_ids: tuple[str, ...]
    selection_reason: str
    question_id: str
    policy_versions: dict[str, str]
    candidate_knowledge_blocks: tuple[InterviewKnowledgeBlockRef, ...] = ()
    selected_knowledge_blocks: tuple[InterviewKnowledgeBlockRef, ...] = ()
    retrieval_trace: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.turn < 1:
            raise ValueError("RecommendationTrace turn must be positive")
        for field_name in ("thread_id", "point_id", "information_gain", "verification_need", "selected_action", "selection_reason", "question_id"):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        object.__setattr__(self, "previous_question_id", str(self.previous_question_id or "").strip() or None)
        for field_name in ("interest_current", "confidence_current_before", "confidence_observation", "confidence_current_after"):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)
        if self.continuous_depth_count < 0:
            raise ValueError("continuous_depth_count cannot be negative")
        for field_name in ("allowed_actions", "candidate_knowledge_point_ids", "selected_knowledge_point_ids"):
            object.__setattr__(
                self,
                field_name,
                tuple(_required_text(item, field_name) for item in getattr(self, field_name)),
            )
        for field_name in ("candidate_knowledge_blocks", "selected_knowledge_blocks"):
            object.__setattr__(
                self,
                field_name,
                tuple(
                    item if isinstance(item, InterviewKnowledgeBlockRef) else InterviewKnowledgeBlockRef.from_dict(dict(item))
                    for item in getattr(self, field_name)
                ),
            )
        object.__setattr__(self, "coefficients", {str(key): float(value) for key, value in self.coefficients.items()})
        object.__setattr__(self, "weights", {str(key): float(value) for key, value in self.weights.items()})
        object.__setattr__(self, "policy_versions", {str(key): str(value) for key, value in self.policy_versions.items()})
        object.__setattr__(self, "retrieval_trace", _clone(self.retrieval_trace, "retrieval_trace"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "thread_id": self.thread_id,
            "point_id": self.point_id,
            "previous_question_id": self.previous_question_id,
            "information_gain": self.information_gain,
            "verification_need": self.verification_need,
            "interest_current": self.interest_current,
            "confidence_current_before": self.confidence_current_before,
            "confidence_observation": self.confidence_observation,
            "confidence_current_after": self.confidence_current_after,
            "continuous_depth_count": self.continuous_depth_count,
            "coefficients": self.coefficients,
            "weights": self.weights,
            "allowed_actions": list(self.allowed_actions),
            "candidate_knowledge_point_ids": list(self.candidate_knowledge_point_ids),
            "selected_action": self.selected_action,
            "selected_knowledge_point_ids": list(self.selected_knowledge_point_ids),
            "selection_reason": self.selection_reason,
            "question_id": self.question_id,
            "policy_versions": self.policy_versions,
            "candidate_knowledge_blocks": [item.to_dict() for item in self.candidate_knowledge_blocks],
            "selected_knowledge_blocks": [item.to_dict() for item in self.selected_knowledge_blocks],
            "retrieval_trace": self.retrieval_trace,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecommendationTrace":
        return cls(
            turn=int(value["turn"]),
            thread_id=str(value["thread_id"]),
            point_id=str(value["point_id"]),
            previous_question_id=str(value.get("previous_question_id") or "") or None,
            information_gain=str(value.get("information_gain") or "中"),
            verification_need=str(value.get("verification_need") or "中"),
            interest_current=float(value.get("interest_current", 0.5)),
            confidence_current_before=float(value.get("confidence_current_before", 0.5)),
            confidence_observation=float(value.get("confidence_observation", 0.5)),
            confidence_current_after=float(value.get("confidence_current_after", 0.5)),
            continuous_depth_count=int(value.get("continuous_depth_count", 0)),
            coefficients=dict(value.get("coefficients") or {}),
            weights=dict(value.get("weights") or {}),
            allowed_actions=tuple(value.get("allowed_actions", ())),
            candidate_knowledge_point_ids=tuple(value.get("candidate_knowledge_point_ids", ())),
            selected_action=str(value.get("selected_action") or "breadth"),
            selected_knowledge_point_ids=tuple(value.get("selected_knowledge_point_ids", ())),
            selection_reason=str(value.get("selection_reason") or "Restored recommendation trace"),
            question_id=str(value["question_id"]),
            policy_versions=dict(value.get("policy_versions") or {}),
            candidate_knowledge_blocks=tuple(
                InterviewKnowledgeBlockRef.from_dict(dict(item))
                for item in value.get("candidate_knowledge_blocks", ())
                if isinstance(item, dict)
            ),
            selected_knowledge_blocks=tuple(
                InterviewKnowledgeBlockRef.from_dict(dict(item))
                for item in value.get("selected_knowledge_blocks", ())
                if isinstance(item, dict)
            ),
            retrieval_trace=dict(value.get("retrieval_trace") or {}),
        )


@dataclass(frozen=True, slots=True)
class AnswerObservation:
    explicit_unknown: bool
    new_information_ratio: float
    relevance: float
    answer_length: int
    stop_reason: InterviewStopReason | None = None

    def __post_init__(self) -> None:
        for field_name in ("new_information_ratio", "relevance"):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)
        if self.answer_length < 1:
            raise ValueError("answer_length must be positive")
        if self.stop_reason is not None:
            object.__setattr__(self, "stop_reason", InterviewStopReason(self.stop_reason))

    def to_dict(self) -> dict[str, Any]:
        return {
            "explicit_unknown": self.explicit_unknown,
            "new_information_ratio": self.new_information_ratio,
            "relevance": self.relevance,
            "answer_length": self.answer_length,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "AnswerObservation | None":
        if not value:
            return None
        return cls(
            explicit_unknown=bool(value.get("explicit_unknown", False)),
            new_information_ratio=float(value.get("new_information_ratio", 1.0)),
            relevance=float(value.get("relevance", 1.0)),
            answer_length=int(value.get("answer_length", 1)),
            stop_reason=InterviewStopReason(value["stop_reason"]) if value.get("stop_reason") else None,
        )


@dataclass(frozen=True, slots=True)
class InterviewQuestion:
    question_id: str
    text: str
    topic: str
    path: InterviewPath
    thread_id: str
    knowledge_refs: tuple[str, ...] = ()
    knowledge_block_refs: tuple[InterviewKnowledgeBlockRef, ...] = ()
    jd_requirement_refs: tuple[str, ...] = ()
    resume_claim_refs: tuple[str, ...] = ()
    jd_ref: tuple[str, int] | None = None
    verified_resume_ref: tuple[str, int] | None = None
    parent_question_id: str | None = None
    followup_depth: int = 0
    question_type: InterviewQuestionType = InterviewQuestionType.KNOWLEDGE
    utility: QuestionUtility | None = None

    def __post_init__(self) -> None:
        for name in ("question_id", "text", "topic", "thread_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "path", InterviewPath(self.path))
        object.__setattr__(self, "question_type", InterviewQuestionType(self.question_type))
        object.__setattr__(self, "parent_question_id", str(self.parent_question_id or "").strip() or None)
        if self.followup_depth < 0:
            raise ValueError("followup_depth cannot be negative")
        object.__setattr__(self, "knowledge_refs", tuple(_required_text(item, "knowledge_ref") for item in self.knowledge_refs))
        object.__setattr__(
            self,
            "knowledge_block_refs",
            tuple(
                item if isinstance(item, InterviewKnowledgeBlockRef) else InterviewKnowledgeBlockRef.from_dict(dict(item))
                for item in self.knowledge_block_refs
            ),
        )
        object.__setattr__(
            self,
            "jd_requirement_refs",
            tuple(_required_text(item, "jd_requirement_ref") for item in self.jd_requirement_refs),
        )
        object.__setattr__(
            self,
            "resume_claim_refs",
            tuple(_required_text(item, "resume_claim_ref") for item in self.resume_claim_refs),
        )
        for field_name in ("jd_ref", "verified_resume_ref"):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = tuple(value)
            if len(normalized) != 2 or int(normalized[1]) < 1:
                raise ValueError(f"{field_name} must contain an ID and positive revision")
            object.__setattr__(self, field_name, (_required_text(normalized[0], field_name), int(normalized[1])))

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "topic": self.topic,
            "path": self.path.value,
            "thread_id": self.thread_id,
            "knowledge_refs": list(self.knowledge_refs),
            "knowledge_block_refs": [item.to_dict() for item in self.knowledge_block_refs],
            "jd_requirement_refs": list(self.jd_requirement_refs),
            "resume_claim_refs": list(self.resume_claim_refs),
            "jd_ref": list(self.jd_ref) if self.jd_ref else None,
            "verified_resume_ref": list(self.verified_resume_ref) if self.verified_resume_ref else None,
            "parent_question_id": self.parent_question_id,
            "followup_depth": self.followup_depth,
            "question_type": self.question_type.value,
            "utility": self.utility.to_dict() if self.utility else None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InterviewQuestion":
        return cls(
            question_id=str(value["question_id"]),
            text=str(value["text"]),
            topic=str(value["topic"]),
            path=InterviewPath(value["path"]),
            thread_id=str(value["thread_id"]),
            knowledge_refs=tuple(value.get("knowledge_refs", ())),
            knowledge_block_refs=tuple(
                InterviewKnowledgeBlockRef.from_dict(dict(item))
                for item in value.get("knowledge_block_refs", ())
                if isinstance(item, dict)
            ),
            jd_requirement_refs=tuple(value.get("jd_requirement_refs", ())),
            resume_claim_refs=tuple(value.get("resume_claim_refs", ())),
            jd_ref=tuple(value["jd_ref"]) if value.get("jd_ref") else None,
            verified_resume_ref=(
                tuple(value["verified_resume_ref"])
                if value.get("verified_resume_ref")
                else None
            ),
            parent_question_id=str(value.get("parent_question_id") or "") or None,
            followup_depth=int(value.get("followup_depth", 0)),
            question_type=InterviewQuestionType(value.get("question_type", InterviewQuestionType.KNOWLEDGE.value)),
            utility=QuestionUtility.from_dict(value.get("utility")),
        )


@dataclass(frozen=True, slots=True)
class InterviewTurn:
    question: InterviewQuestion
    answer: str
    correctness: float
    completeness: float
    confidence: float
    evaluation_reason: str = ""
    evaluation_refs: tuple[str, ...] = ()
    knowledge_confidence_updates: tuple[KnowledgeConfidenceUpdate, ...] = ()
    observation: AnswerObservation | None = None
    orchestration_audit: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer", _required_text(self.answer, "answer"))
        for field_name in ("correctness", "completeness", "confidence"):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "evaluation_reason", str(self.evaluation_reason or "").strip())
        object.__setattr__(
            self,
            "evaluation_refs",
            tuple(_required_text(item, "evaluation_ref") for item in self.evaluation_refs),
        )
        updates = tuple(
            item if isinstance(item, KnowledgeConfidenceUpdate) else KnowledgeConfidenceUpdate.from_dict(dict(item))
            for item in self.knowledge_confidence_updates
        )
        if len({item.knowledge_point_id for item in updates}) != len(updates):
            raise ValueError("Each knowledge point can be updated only once per interview turn")
        object.__setattr__(self, "knowledge_confidence_updates", updates)
        if self.orchestration_audit is not None:
            object.__setattr__(self, "orchestration_audit", _clone(self.orchestration_audit, "orchestration_audit"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question.to_dict(),
            "answer": self.answer,
            "correctness": self.correctness,
            "completeness": self.completeness,
            "confidence": self.confidence,
            "evaluation_reason": self.evaluation_reason,
            "evaluation_refs": list(self.evaluation_refs),
            "knowledge_confidence_updates": [item.to_dict() for item in self.knowledge_confidence_updates],
            "created_at": self.created_at,
            "observation": self.observation.to_dict() if self.observation else None,
            "orchestration_audit": self.orchestration_audit,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InterviewTurn":
        return cls(
            question=InterviewQuestion.from_dict(dict(value["question"])),
            answer=str(value["answer"]),
            correctness=float(value["correctness"]),
            completeness=float(value["completeness"]),
            confidence=float(value["confidence"]),
            evaluation_reason=str(value.get("evaluation_reason") or ""),
            evaluation_refs=tuple(value.get("evaluation_refs", ())),
            knowledge_confidence_updates=tuple(
                KnowledgeConfidenceUpdate.from_dict(dict(item))
                for item in value.get("knowledge_confidence_updates", ())
                if isinstance(item, dict)
            ),
            observation=AnswerObservation.from_dict(value.get("observation")),
            orchestration_audit=dict(value["orchestration_audit"]) if isinstance(value.get("orchestration_audit"), dict) else None,
            created_at=float(value.get("created_at", time.time())),
        )


@dataclass(frozen=True, slots=True)
class InterviewDecision:
    action: InterviewAction
    path: InterviewPath
    topic: str
    reason: str
    thread_id: str | None = None
    stop_reason: InterviewStopReason | None = None
    question_type: InterviewQuestionType = InterviewQuestionType.KNOWLEDGE
    utility: QuestionUtility | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", InterviewAction(self.action))
        object.__setattr__(self, "path", InterviewPath(self.path))
        object.__setattr__(self, "question_type", InterviewQuestionType(self.question_type))
        if self.stop_reason is not None:
            object.__setattr__(self, "stop_reason", InterviewStopReason(self.stop_reason))
        object.__setattr__(self, "topic", _required_text(self.topic, "topic"))
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        if self.thread_id is not None:
            object.__setattr__(self, "thread_id", _required_text(self.thread_id, "thread_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "path": self.path.value,
            "topic": self.topic,
            "reason": self.reason,
            "thread_id": self.thread_id,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "question_type": self.question_type.value,
            "utility": self.utility.to_dict() if self.utility else None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InterviewDecision":
        return cls(
            action=InterviewAction(value["action"]),
            path=InterviewPath(value["path"]),
            topic=str(value["topic"]),
            reason=str(value["reason"]),
            thread_id=str(value["thread_id"]) if value.get("thread_id") else None,
            stop_reason=InterviewStopReason(value["stop_reason"]) if value.get("stop_reason") else None,
            question_type=InterviewQuestionType(value.get("question_type", InterviewQuestionType.KNOWLEDGE.value)),
            utility=QuestionUtility.from_dict(value.get("utility")),
        )


@dataclass(frozen=True, slots=True)
class InterviewSessionState:
    interview_id: str
    jd_id: str | None
    resume_document_id: str | None
    profile: InterviewerProfile
    policy: InterviewerPolicy
    resume_run_id: str | None = None
    mode: InterviewMode = InterviewMode.REALISTIC_SIMULATION
    tenant_id: str = "default"
    jd_revision: int | None = 1
    resume_document_revision: int | None = 1
    knowledge_corpus_ids: tuple[str, ...] = ()
    knowledge_corpus_versions: dict[str, str] = field(default_factory=dict)
    random_seed: int = 0
    max_turns: int = 8
    interview_stage: InterviewStage = InterviewStage.INTRO
    duration_minutes: int = 30
    max_questions_per_subject: int = 6
    max_questions_per_point: int = 3
    max_pure_foundation_questions: int = 2
    candidate_question_wait_seconds: int = 5
    active_subject: str | None = None
    subject_question_counts: dict[str, int] = field(default_factory=dict)
    pure_foundation_question_count: int = 0
    intro_interest_updated: bool = False
    candidate_question_started_at: float | None = None
    status: InterviewStatus = InterviewStatus.CREATED
    belief_state: KnowledgeBeliefState = field(default_factory=KnowledgeBeliefState)
    interview_points: tuple[InterviewPoint, ...] = ()
    active_line: InterviewLine | None = None
    current_point_id: str | None = None
    active_depth_chain_blocks: tuple[str, ...] = ()
    used_knowledge_point_ids: tuple[str, ...] = ()
    recommendation_traces: tuple[RecommendationTrace, ...] = ()
    retry_events: tuple[dict[str, Any], ...] = ()
    threads: tuple[QuestionThread, ...] = ()
    turns: tuple[InterviewTurn, ...] = ()
    pending_question: InterviewQuestion | None = None
    last_decision: InterviewDecision | None = None
    final_stop_reason: InterviewStopReason | None = None
    revision: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "interview_id", _required_text(self.interview_id, "interview_id"))
        object.__setattr__(self, "mode", InterviewMode(self.mode))
        object.__setattr__(self, "tenant_id", _required_text(self.tenant_id, "tenant_id"))
        for name in ("jd_id", "resume_document_id", "resume_run_id"):
            value = str(getattr(self, name) or "").strip() or None
            object.__setattr__(self, name, value)
        if self.mode is InterviewMode.REALISTIC_SIMULATION and self.jd_id is None:
            raise ValueError("Realistic interview mode requires a JD")
        object.__setattr__(self, "status", InterviewStatus(self.status))
        if self.final_stop_reason is not None:
            object.__setattr__(self, "final_stop_reason", InterviewStopReason(self.final_stop_reason))
        object.__setattr__(
            self,
            "knowledge_corpus_ids",
            tuple(_required_text(item, "knowledge_corpus_id") for item in self.knowledge_corpus_ids),
        )
        versions = {
            _required_text(key, "knowledge_corpus_version_id"): _required_text(value, "knowledge_corpus_version")
            for key, value in dict(self.knowledge_corpus_versions).items()
        }
        if set(versions) - set(self.knowledge_corpus_ids):
            raise ValueError("Knowledge corpus versions must reference authorized corpus IDs")
        object.__setattr__(self, "knowledge_corpus_versions", versions)
        object.__setattr__(self, "threads", tuple(self.threads))
        object.__setattr__(self, "turns", tuple(self.turns))
        object.__setattr__(self, "interview_points", tuple(self.interview_points))
        point_ids = {point.point_id for point in self.interview_points}
        object.__setattr__(self, "active_line", InterviewLine(self.active_line) if self.active_line else None)
        object.__setattr__(self, "current_point_id", str(self.current_point_id or "").strip() or None)
        if self.current_point_id is not None and point_ids and self.current_point_id not in point_ids:
            raise ValueError("current_point_id must reference a declared interview point")
        object.__setattr__(
            self,
            "active_depth_chain_blocks",
            tuple(_required_text(item, "active_depth_chain_block") for item in self.active_depth_chain_blocks),
        )
        object.__setattr__(
            self,
            "used_knowledge_point_ids",
            tuple(_required_text(item, "used_knowledge_point_id") for item in self.used_knowledge_point_ids),
        )
        object.__setattr__(self, "recommendation_traces", tuple(self.recommendation_traces))
        object.__setattr__(
            self,
            "retry_events",
            tuple(_clone(item, "retry_event") for item in self.retry_events if isinstance(item, dict)),
        )
        for name in ("jd_revision", "resume_document_revision"):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError("JD and resume revisions must be at least 1")
            object.__setattr__(self, name, int(value) if value is not None else None)
        if (self.jd_id is None) != (self.jd_revision is None):
            raise ValueError("JD ID and revision must either both be present or both be absent")
        if (self.resume_document_id is None) != (self.resume_document_revision is None):
            raise ValueError("Resume ID and revision must either both be present or both be absent")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        object.__setattr__(self, "interview_stage", InterviewStage(self.interview_stage))
        duration = int(self.duration_minutes)
        if duration not in {30, 60}:
            duration = 60 if duration > 30 else 30
        object.__setattr__(self, "duration_minutes", duration)
        for field_name in (
            "max_questions_per_subject",
            "max_questions_per_point",
            "max_pure_foundation_questions",
            "candidate_question_wait_seconds",
        ):
            value = int(getattr(self, field_name))
            if value < 1:
                raise ValueError(f"{field_name} must be at least 1")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "active_subject", str(self.active_subject or "").strip() or None)
        object.__setattr__(
            self,
            "subject_question_counts",
            {str(key): int(value) for key, value in dict(self.subject_question_counts).items()},
        )
        if any(value < 0 for value in self.subject_question_counts.values()):
            raise ValueError("subject_question_counts cannot contain negative values")
        if int(self.pure_foundation_question_count) < 0:
            raise ValueError("pure_foundation_question_count cannot be negative")
        object.__setattr__(self, "pure_foundation_question_count", int(self.pure_foundation_question_count))
        object.__setattr__(self, "intro_interest_updated", bool(self.intro_interest_updated))
        started_at = self.candidate_question_started_at
        object.__setattr__(self, "candidate_question_started_at", float(started_at) if started_at is not None else None)
        if self.revision < 1 or self.updated_at < self.created_at:
            raise ValueError("Invalid interview session revision metadata")
        thread_ids = {thread.thread_id for thread in self.threads}
        if self.pending_question is not None and self.pending_question.thread_id not in thread_ids:
            raise ValueError("Pending question references an unknown question thread")
        if self.status in {InterviewStatus.COMPLETED, InterviewStatus.CANCELLED} and self.pending_question is not None:
            raise ValueError("A finished interview cannot retain a pending question")

    @classmethod
    def create(
        cls,
        *,
        jd_id: str | None,
        resume_document_id: str | None,
        profile: InterviewerProfile,
        policy: InterviewerPolicy,
        resume_run_id: str | None = None,
        mode: InterviewMode | str = InterviewMode.REALISTIC_SIMULATION,
        tenant_id: str = "default",
        jd_revision: int | None = 1,
        resume_document_revision: int | None = 1,
        knowledge_corpus_ids: tuple[str, ...] = (),
        knowledge_corpus_versions: dict[str, str] | None = None,
        belief_state: KnowledgeBeliefState | None = None,
        random_seed: int = 0,
        max_turns: int = 8,
        interview_points: tuple[InterviewPoint, ...] = (),
        active_line: InterviewLine | str | None = None,
        interview_stage: InterviewStage | str = InterviewStage.INTRO,
        duration_minutes: int = 30,
        max_questions_per_subject: int | None = None,
        max_questions_per_point: int = 3,
        max_pure_foundation_questions: int | None = None,
    ) -> "InterviewSessionState":
        duration = 60 if int(duration_minutes) >= 60 else 30
        return cls(
            _new_id("interview"),
            jd_id,
            resume_document_id,
            profile,
            policy,
            resume_run_id=resume_run_id,
            mode=mode,
            tenant_id=tenant_id,
            jd_revision=jd_revision,
            resume_document_revision=resume_document_revision,
            knowledge_corpus_ids=knowledge_corpus_ids,
            knowledge_corpus_versions=knowledge_corpus_versions or {},
            belief_state=belief_state or KnowledgeBeliefState(),
            random_seed=int(random_seed),
            max_turns=max_turns,
            interview_stage=interview_stage,
            duration_minutes=duration,
            max_questions_per_subject=max_questions_per_subject or (12 if duration == 60 else 6),
            max_questions_per_point=max_questions_per_point,
            max_pure_foundation_questions=max_pure_foundation_questions or (3 if duration == 60 else 2),
            candidate_question_wait_seconds=5,
            interview_points=interview_points,
            active_line=InterviewLine(active_line) if active_line else None,
        )

    def _next(self, **changes: Any) -> "InterviewSessionState":
        return replace(
            self,
            **changes,
            revision=self.revision + 1,
            updated_at=time.time(),
        )

    def activate(self) -> "InterviewSessionState":
        if self.status is not InterviewStatus.CREATED:
            return self
        return self._next(status=InterviewStatus.ACTIVE)

    def set_pending_question(self, question: InterviewQuestion) -> "InterviewSessionState":
        if self.status in {InterviewStatus.COMPLETED, InterviewStatus.CANCELLED}:
            raise ValueError("Cannot ask a question after the interview has ended")
        if self.pending_question is not None:
            if self.pending_question == question:
                return self
            raise ValueError("The current interview question must be answered first")
        if question.thread_id not in {thread.thread_id for thread in self.threads}:
            raise ValueError("Interview question references an unknown question thread")
        return self._next(status=InterviewStatus.ACTIVE, pending_question=question)

    def record_turn(self, turn: InterviewTurn) -> "InterviewSessionState":
        if self.status not in {InterviewStatus.CREATED, InterviewStatus.ACTIVE}:
            raise ValueError("Cannot record a turn after the interview has ended")
        thread = next((item for item in self.threads if item.thread_id == turn.question.thread_id), None)
        if thread is None:
            raise ValueError("Interview turn references an unknown question thread")
        if self.pending_question is not None and turn.question.question_id != self.pending_question.question_id:
            raise ValueError("Interview turn does not answer the pending question")
        is_followup = turn.question.path is InterviewPath.DEPTH
        if is_followup and thread.followup_count >= thread.max_followups:
            raise ValueError("Question thread has reached its follow-up limit")
        next_thread = replace(
            thread,
            followup_count=thread.followup_count + (1 if is_followup else 0),
        )
        next_threads = tuple(next_thread if item.thread_id == thread.thread_id else item for item in self.threads)
        belief_state, committed_updates = self.belief_state.apply_updates(turn.knowledge_confidence_updates)
        if committed_updates != turn.knowledge_confidence_updates:
            turn = replace(turn, knowledge_confidence_updates=committed_updates)
        if not turn.knowledge_confidence_updates:
            # Compatibility only for callers that construct InterviewTurn
            # directly. Production evaluators always provide block updates.
            topic = turn.question.topic
            prior_mean = self.belief_state.legacy_confidence_by_topic.get(topic, 0.5)
            prior_alpha = self.belief_state.legacy_alpha_by_topic.get(topic, max(1e-6, prior_mean * 2))
            prior_beta = self.belief_state.legacy_beta_by_topic.get(topic, max(1e-6, (1 - prior_mean) * 2))
            weighted = float(turn.confidence)
            correctness = (turn.correctness + turn.completeness) / 2
            belief_state = replace(
                belief_state,
                legacy_confidence_by_topic={
                    **belief_state.legacy_confidence_by_topic,
                    topic: (prior_alpha + weighted * correctness) / (prior_alpha + prior_beta + weighted),
                },
                legacy_observations_by_topic={
                    **belief_state.legacy_observations_by_topic,
                    topic: belief_state.legacy_observations_by_topic.get(topic, 0) + 1,
                },
                legacy_alpha_by_topic={**belief_state.legacy_alpha_by_topic, topic: prior_alpha + weighted * correctness},
                legacy_beta_by_topic={**belief_state.legacy_beta_by_topic, topic: prior_beta + weighted * (1 - correctness)},
            )
        return self._next(
            status=InterviewStatus.ACTIVE,
            belief_state=belief_state,
            threads=next_threads,
            turns=(*self.turns, turn),
            pending_question=None,
        )

    def add_thread(self, topic: str, *, path: InterviewPath = InterviewPath.BREADTH) -> "InterviewSessionState":
        if self.status in {InterviewStatus.COMPLETED, InterviewStatus.CANCELLED}:
            raise ValueError("Cannot add a question thread after the interview has ended")
        thread = QuestionThread(_new_id("thread"), topic, path, max_followups=self.policy.max_followups_per_thread)
        return self._next(threads=(*self.threads, thread), status=InterviewStatus.ACTIVE)

    def record_retry_event(self, event: dict[str, Any]) -> "InterviewSessionState":
        return self._next(retry_events=(*self.retry_events, _clone(event, "retry_event")))

    def apply_decision(self, decision: InterviewDecision) -> "InterviewSessionState":
        if self.pending_question is not None:
            raise ValueError("Cannot apply a decision before the pending question is answered")
        threads = self.threads
        if decision.action in {InterviewAction.SWITCH_TOPIC, InterviewAction.END} and decision.thread_id:
            threads = tuple(
                replace(thread, status="closed", stop_reason=decision.stop_reason)
                if thread.thread_id == decision.thread_id
                else thread
                for thread in threads
            )
        status = InterviewStatus.COMPLETED if decision.action is InterviewAction.END else InterviewStatus.ACTIVE
        return self._next(
            last_decision=decision,
            threads=threads,
            status=status,
            final_stop_reason=decision.stop_reason if decision.action is InterviewAction.END else None,
        )

    def finish(self, *, cancelled: bool = False) -> "InterviewSessionState":
        if self.pending_question is not None and not cancelled:
            raise ValueError("Answer or discard the pending question before finishing the interview")
        if self.status in {InterviewStatus.COMPLETED, InterviewStatus.CANCELLED}:
            return self
        status = InterviewStatus.CANCELLED if cancelled else InterviewStatus.COMPLETED
        reason = InterviewStopReason.CANCELLED if cancelled else InterviewStopReason.USER_FINISHED
        return self._next(status=status, pending_question=None, final_stop_reason=reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INTERVIEW_SCHEMA_VERSION,
            "interview_id": self.interview_id,
            "mode": self.mode.value,
            "tenant_id": self.tenant_id,
            "resume_run_id": self.resume_run_id,
            "jd_id": self.jd_id,
            "jd_revision": self.jd_revision,
            "resume_document_id": self.resume_document_id,
            "resume_document_revision": self.resume_document_revision,
            "knowledge_corpus_ids": list(self.knowledge_corpus_ids),
            "knowledge_corpus_versions": self.knowledge_corpus_versions,
            "random_seed": self.random_seed,
            "max_turns": self.max_turns,
            "interview_stage": self.interview_stage.value,
            "duration_minutes": self.duration_minutes,
            "max_questions_per_subject": self.max_questions_per_subject,
            "max_questions_per_point": self.max_questions_per_point,
            "max_pure_foundation_questions": self.max_pure_foundation_questions,
            "candidate_question_wait_seconds": self.candidate_question_wait_seconds,
            "active_subject": self.active_subject,
            "subject_question_counts": self.subject_question_counts,
            "pure_foundation_question_count": self.pure_foundation_question_count,
            "intro_interest_updated": self.intro_interest_updated,
            "candidate_question_started_at": self.candidate_question_started_at,
            "profile": {
                "profile_id": self.profile.profile_id,
                "name": self.profile.name,
                "temperament": self.profile.temperament,
                "seniority": self.profile.seniority,
                "pressure_level": self.profile.pressure_level,
                "breadth_bias": self.profile.breadth_bias,
            },
            "policy": {
                "policy_id": self.policy.policy_id,
                "max_followups_per_thread": self.policy.max_followups_per_thread,
                "depth_threshold": self.policy.depth_threshold,
                "breadth_threshold": self.policy.breadth_threshold,
                "project_fact_policy": self.policy.project_fact_policy,
                "knowledge_update_policy": self.policy.knowledge_update_policy,
            },
            "status": self.status.value,
            "belief_state": self.belief_state.to_dict(),
            "interview_points": [point.to_dict() for point in self.interview_points],
            "active_line": self.active_line.value if self.active_line else None,
            "current_point_id": self.current_point_id,
            "active_depth_chain_blocks": list(self.active_depth_chain_blocks),
            "used_knowledge_point_ids": list(self.used_knowledge_point_ids),
            "recommendation_traces": [trace.to_dict() for trace in self.recommendation_traces],
            "retry_events": list(self.retry_events),
            "threads": [
                {
                    "thread_id": thread.thread_id,
                    "root_topic": thread.root_topic,
                    "path": thread.path.value,
                    "followup_count": thread.followup_count,
                    "max_followups": thread.max_followups,
                    "status": thread.status,
                    "stop_reason": thread.stop_reason.value if thread.stop_reason else None,
                }
                for thread in self.threads
            ],
            "turns": [turn.to_dict() for turn in self.turns],
            "pending_question": self.pending_question.to_dict() if self.pending_question else None,
            "last_decision": self.last_decision.to_dict() if self.last_decision else None,
            "final_stop_reason": self.final_stop_reason.value if self.final_stop_reason else None,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InterviewSessionState":
        schema_version = str(value.get("schema_version") or "1.0")
        if schema_version not in _SUPPORTED_INTERVIEW_SCHEMA_VERSIONS:
            raise ValueError("Unsupported interview session schema version")
        profile = dict(value["profile"])
        policy = dict(value["policy"])
        belief = dict(value.get("belief_state") or {})
        pending = value.get("pending_question")
        decision = value.get("last_decision")
        return cls(
            interview_id=str(value["interview_id"]),
            jd_id=str(value.get("jd_id") or "") or None,
            resume_document_id=str(value.get("resume_document_id") or "") or None,
            profile=InterviewerProfile(**profile),
            policy=InterviewerPolicy(**policy),
            resume_run_id=str(value.get("resume_run_id") or "") or None,
            mode=InterviewMode(value.get("mode", InterviewMode.REALISTIC_SIMULATION.value)),
            tenant_id=str(value.get("tenant_id") or "default"),
            jd_revision=int(value["jd_revision"]) if value.get("jd_revision") is not None else None,
            resume_document_revision=(
                int(value["resume_document_revision"])
                if value.get("resume_document_revision") is not None
                else None
            ),
            knowledge_corpus_ids=tuple(value.get("knowledge_corpus_ids", ())),
            knowledge_corpus_versions=dict(value.get("knowledge_corpus_versions") or {}),
            random_seed=int(value.get("random_seed", 0)),
            max_turns=int(value.get("max_turns", 8)),
            interview_stage=InterviewStage(value.get("interview_stage", InterviewStage.INTRO.value)),
            duration_minutes=int(value.get("duration_minutes", 30)),
            max_questions_per_subject=int(value.get("max_questions_per_subject", 6)),
            max_questions_per_point=int(value.get("max_questions_per_point", 3)),
            max_pure_foundation_questions=int(value.get("max_pure_foundation_questions", 2)),
            candidate_question_wait_seconds=int(value.get("candidate_question_wait_seconds", 5)),
            active_subject=str(value.get("active_subject") or "") or None,
            subject_question_counts=dict(value.get("subject_question_counts") or {}),
            pure_foundation_question_count=int(value.get("pure_foundation_question_count", 0)),
            intro_interest_updated=bool(value.get("intro_interest_updated", False)),
            candidate_question_started_at=(
                float(value["candidate_question_started_at"])
                if value.get("candidate_question_started_at") is not None
                else None
            ),
            status=InterviewStatus(value.get("status", InterviewStatus.CREATED.value)),
            belief_state=KnowledgeBeliefState.from_dict(belief),
            interview_points=tuple(InterviewPoint.from_dict(dict(item)) for item in value.get("interview_points", ())),
            active_line=InterviewLine(value["active_line"]) if value.get("active_line") else None,
            current_point_id=str(value.get("current_point_id") or "") or None,
            active_depth_chain_blocks=tuple(value.get("active_depth_chain_blocks", ())),
            used_knowledge_point_ids=tuple(value.get("used_knowledge_point_ids", ())),
            recommendation_traces=tuple(
                RecommendationTrace.from_dict(dict(item))
                for item in value.get("recommendation_traces", ())
            ),
            retry_events=tuple(dict(item) for item in value.get("retry_events", ()) if isinstance(item, dict)),
            threads=tuple(
                QuestionThread(
                    thread_id=str(item["thread_id"]),
                    root_topic=str(item["root_topic"]),
                    path=InterviewPath(item["path"]),
                    followup_count=int(item.get("followup_count", 0)),
                    max_followups=int(item.get("max_followups", 3)),
                    status=str(item.get("status") or "active"),
                    stop_reason=(
                        InterviewStopReason(item["stop_reason"])
                        if item.get("stop_reason")
                        else None
                    ),
                )
                for item in value.get("threads", ())
            ),
            turns=tuple(InterviewTurn.from_dict(dict(item)) for item in value.get("turns", ())),
            pending_question=InterviewQuestion.from_dict(dict(pending)) if pending else None,
            last_decision=InterviewDecision.from_dict(dict(decision)) if decision else None,
            final_stop_reason=(
                InterviewStopReason(value["final_stop_reason"])
                if value.get("final_stop_reason")
                else None
            ),
            revision=int(value.get("revision", 1)),
            created_at=float(value.get("created_at", time.time())),
            updated_at=float(value.get("updated_at", time.time())),
        )


class InterviewPolicyEngine(Protocol):
    name: str

    def decide(
        self,
        state: InterviewSessionState,
        *,
        topic: str,
        correctness: float,
        completeness: float,
        observation: AnswerObservation | None = None,
        utility: QuestionUtility | None = None,
    ) -> InterviewDecision: ...


class DeterministicInterviewPolicy:
    """Replaceable baseline policy; it evaluates knowledge, never project ownership."""

    name = "threshold-adaptive-v1"
    low_mastery_threshold = 0.35
    boundary_probe_threshold = 0.78
    minimum_relevance = 0.12
    minimum_information_ratio = 0.15

    def decide(
        self,
        state: InterviewSessionState,
        *,
        topic: str,
        correctness: float,
        completeness: float,
        observation: AnswerObservation | None = None,
        utility: QuestionUtility | None = None,
    ) -> InterviewDecision:
        topic = _required_text(topic, "topic")
        score = 0.55 * float(correctness) + 0.45 * float(completeness)
        if not 0 <= score <= 1:
            raise ValueError("correctness and completeness must be between 0 and 1")
        if observation is None:
            observation = AnswerObservation(
                explicit_unknown=False,
                new_information_ratio=1.0,
                relevance=max(0.0, min(1.0, float(completeness))),
                answer_length=20,
            )
        thread = next((item for item in reversed(state.threads) if item.root_topic == topic and item.status == "active"), None)
        if thread is None:
            return InterviewDecision(InterviewAction.ASK, InterviewPath.BREADTH, topic, "Start a new topic")
        question_type = state.turns[-1].question.question_type if state.turns else InterviewQuestionType.KNOWLEDGE
        if observation.explicit_unknown:
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC,
                InterviewPath.BREADTH,
                topic,
                "Candidate explicitly indicated that they do not know",
                thread.thread_id,
                InterviewStopReason.USER_DOES_NOT_KNOW,
                question_type,
            )
        if thread.followup_count >= thread.max_followups:
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC,
                InterviewPath.BREADTH,
                topic,
                "Configured follow-up depth limit reached",
                thread.thread_id,
                InterviewStopReason.FOLLOWUP_LIMIT,
                question_type,
            )
        if observation.relevance < self.minimum_relevance:
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC,
                InterviewPath.BREADTH,
                topic,
                "Answer is unrelated to the current question; skip this topic",
                thread.thread_id,
                InterviewStopReason.RELEVANCE_DECLINED,
                question_type,
            )
        if observation.answer_length <= 2 or observation.new_information_ratio < self.minimum_information_ratio:
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC,
                InterviewPath.BREADTH,
                topic,
                "Answer contains too little useful information to justify a follow-up",
                thread.thread_id,
                InterviewStopReason.NO_NEW_INFORMATION,
                question_type,
            )
        if score >= max(state.policy.breadth_threshold, self.boundary_probe_threshold):
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC,
                InterviewPath.BREADTH,
                topic,
                "Topic is sufficiently mastered",
                thread.thread_id,
                InterviewStopReason.MASTERED,
                question_type,
            )
        if score <= self.low_mastery_threshold:
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC,
                InterviewPath.BREADTH,
                topic,
                "Topic boundary is clear: candidate lacks the current foundation",
                thread.thread_id,
                InterviewStopReason.LOW_MASTERY,
                question_type,
            )
        if thread.followup_count == 0 and score < max(state.policy.breadth_threshold, self.boundary_probe_threshold):
            return InterviewDecision(
                InterviewAction.FOLLOW_UP,
                InterviewPath.DEPTH,
                topic,
                "Answer is partially correct; ask one boundary-probing follow-up",
                thread.thread_id,
                question_type=question_type,
            )
        return InterviewDecision(
            InterviewAction.SWITCH_TOPIC,
            InterviewPath.BREADTH,
            topic,
            "Move to a new topic after sufficient coverage",
            thread.thread_id,
            question_type=question_type,
        )


class RecommendationInterviewPolicy(DeterministicInterviewPolicy):
    """Formula-based depth/breadth selector with explicit hard constraints."""

    name = "recommendation-depth-breadth-v1"

    def __init__(self, *, depth_k1: float = 0.65, breadth_k3: float = 0.55, max_questions_per_thread: int = 6) -> None:
        if depth_k1 <= 0 or breadth_k3 <= 0 or max_questions_per_thread < 1:
            raise ValueError("recommendation constants are invalid")
        self.depth_k1 = float(depth_k1)
        self.breadth_k3 = float(breadth_k3)
        self.max_questions_per_thread = int(max_questions_per_thread)

    def decide(
        self,
        state: InterviewSessionState,
        *,
        topic: str,
        correctness: float,
        completeness: float,
        observation: AnswerObservation | None = None,
        utility: QuestionUtility | None = None,
    ) -> InterviewDecision:
        topic = _required_text(topic, "topic")
        if observation is None:
            observation = AnswerObservation(False, 1.0, max(0.0, min(1.0, completeness)), 20)
        thread = next((item for item in reversed(state.threads) if item.root_topic == topic and item.status == "active"), None)
        question_type = state.turns[-1].question.question_type if state.turns else InterviewQuestionType.KNOWLEDGE
        if thread is None:
            return InterviewDecision(InterviewAction.ASK, InterviewPath.BREADTH, topic, "Start a new topic")
        if utility is not None:
            information_gain = self._level_label(utility.information_gain)
            verification_need = self._level_label(utility.verification_need)
            breadth_coefficient, depth_coefficient = self._grid_coefficients(
                information_gain,
                verification_need,
            )
            confidence = max(0.25, utility.confidence)
            depth_weight = (1.0 / confidence) * depth_coefficient * self.depth_k1
            breadth_weight = (
                breadth_coefficient
                * (1 + utility.interest_factor + 0.1 * utility.consecutive_depth)
                * self.breadth_k3
            )
            utility = replace(utility, depth_weight=depth_weight, breadth_weight=breadth_weight)
        common = {
            "topic": topic,
            "thread_id": thread.thread_id,
            "question_type": question_type,
            "utility": utility,
        }
        if observation.explicit_unknown:
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC, InterviewPath.BREADTH,
                reason="Candidate explicitly indicated that they do not know",
                stop_reason=InterviewStopReason.USER_DOES_NOT_KNOW, **common,
            )
        if observation.relevance < self.minimum_relevance:
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC, InterviewPath.BREADTH,
                reason="Answer is unrelated to the current question; skip this topic",
                stop_reason=InterviewStopReason.RELEVANCE_DECLINED, **common,
            )
        if observation.answer_length <= 2 or observation.new_information_ratio < self.minimum_information_ratio:
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC, InterviewPath.BREADTH,
                reason="Answer contains too little useful information to justify a follow-up",
                stop_reason=InterviewStopReason.NO_NEW_INFORMATION, **common,
            )
        score = 0.55 * float(correctness) + 0.45 * float(completeness)
        if score >= max(state.policy.breadth_threshold, self.boundary_probe_threshold):
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC, InterviewPath.BREADTH,
                reason="Topic is sufficiently mastered",
                stop_reason=InterviewStopReason.MASTERED, **common,
            )
        if score <= self.low_mastery_threshold:
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC, InterviewPath.BREADTH,
                reason="Topic boundary is clear: candidate lacks the current foundation",
                stop_reason=InterviewStopReason.LOW_MASTERY, **common,
            )
        if len(state.turns) >= state.max_turns or thread.followup_count >= min(thread.max_followups, self.max_questions_per_thread - 1):
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC, InterviewPath.BREADTH,
                reason="Thread hard limit reached", stop_reason=InterviewStopReason.FOLLOWUP_LIMIT, **common,
            )
        if thread.followup_count > 0:
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC, InterviewPath.BREADTH,
                reason="Boundary-probing follow-up already used for this topic",
                stop_reason=InterviewStopReason.FOLLOWUP_LIMIT, **common,
            )
        if utility is None:
            return InterviewDecision(InterviewAction.SWITCH_TOPIC, InterviewPath.BREADTH, reason="No recommendation signals", **common)
        depth_weight = utility.depth_weight
        breadth_weight = utility.breadth_weight
        if depth_weight > breadth_weight and thread.followup_count < thread.max_followups:
            return InterviewDecision(
                InterviewAction.FOLLOW_UP, InterviewPath.DEPTH,
                reason=f"W_depth={depth_weight:.3f} > W_breadth={breadth_weight:.3f}; continue depth",
                **common,
            )
        return InterviewDecision(
            InterviewAction.SWITCH_TOPIC, InterviewPath.BREADTH,
            reason=f"W_breadth={breadth_weight:.3f} >= W_depth={depth_weight:.3f}; switch breadth",
            **common,
        )

    @staticmethod
    def _level_label(value: float) -> str:
        value = float(value)
        if value >= 0.67:
            return "高"
        if value >= 0.34:
            return "中"
        return "低"

    @staticmethod
    def _grid_coefficients(information_gain: str, verification_need: str) -> tuple[float, float]:
        table = {
            ("高", "高"): (0.20, 1.00),
            ("高", "中"): (0.45, 0.80),
            ("高", "低"): (0.80, 0.45),
            ("中", "高"): (0.30, 0.95),
            ("中", "中"): (0.60, 0.60),
            ("中", "低"): (0.90, 0.30),
            ("低", "高"): (0.35, 0.85),
            ("低", "中"): (0.85, 0.30),
            ("低", "低"): (1.00, 0.10),
        }
        breadth, depth = table.get((information_gain, verification_need), (0.60, 0.60))
        return breadth, depth


class InterviewSessionStore:
    """Atomic, integrity-checked storage isolated from the parent Career session."""

    _locks_guard: ClassVar[threading.Lock] = threading.Lock()
    _root_locks: ClassVar[dict[str, threading.RLock]] = {}

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = os.path.abspath(os.fspath(root))
        self.sessions_dir = os.path.join(self.root, "sessions")
        with self._locks_guard:
            self._lock = self._root_locks.setdefault(self.root, threading.RLock())
        os.makedirs(self.sessions_dir, exist_ok=True)

    def create(self, state: InterviewSessionState) -> InterviewSessionState:
        if state.revision != 1:
            raise ValueError("A new interview session must start at revision 1")
        with self._lock:
            path = self._path(state.interview_id)
            if os.path.exists(path):
                raise ValueError(f"Interview session already exists: {state.interview_id}")
            self._write_state(path, state)
        return state

    def put(self, state: InterviewSessionState) -> InterviewSessionState:
        with self._lock:
            path = self._path(state.interview_id)
            if not os.path.exists(path):
                return self.create(state)
            current = self._read_state(path)
            if state.revision == current.revision and state == current:
                return current
            if state.revision <= current.revision:
                raise ValueError(
                    f"Stale interview session revision: {state.interview_id}@{state.revision}; "
                    f"current is {current.revision}"
                )
            self._write_state(path, state)
        return state

    def get(self, interview_id: str) -> InterviewSessionState:
        path = self._path(interview_id)
        with self._lock:
            if not os.path.exists(path):
                raise KeyError(f"Unknown interview session: {interview_id}")
            return self._read_state(path)

    def list(self) -> tuple[InterviewSessionState, ...]:
        with self._lock:
            return tuple(
                self._read_state(os.path.join(self.sessions_dir, name))
                for name in sorted(os.listdir(self.sessions_dir))
                if name.endswith(".json")
            )

    def delete(self, interview_id: str) -> None:
        path = self._path(interview_id)
        with self._lock:
            if not os.path.exists(path):
                raise KeyError(f"Unknown interview session: {interview_id}")
            os.unlink(path)

    def _path(self, interview_id: str) -> str:
        normalized = _required_text(interview_id, "interview_id")
        if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in normalized):
            raise ValueError("interview_id contains unsupported characters")
        return os.path.join(self.sessions_dir, f"{normalized}.json")

    @staticmethod
    def _read_state(path: str) -> InterviewSessionState:
        with open(path, encoding="utf-8") as handle:
            envelope = json.load(handle)
        if not isinstance(envelope, dict) or not isinstance(envelope.get("state"), dict):
            raise ValueError("Interview session file must contain an object state")
        state_payload = envelope["state"]
        if envelope.get("content_hash") != _digest(state_payload):
            raise ValueError("Interview session content hash does not match persisted state")
        return InterviewSessionState.from_dict(state_payload)

    @staticmethod
    def _write_state(path: str, state: InterviewSessionState) -> None:
        state_payload = state.to_dict()
        envelope = {
            "schema_version": INTERVIEW_SCHEMA_VERSION,
            "content_hash": _digest(state_payload),
            "state": state_payload,
        }
        directory = os.path.dirname(path)
        descriptor, temporary = tempfile.mkstemp(prefix=".session.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(envelope, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
