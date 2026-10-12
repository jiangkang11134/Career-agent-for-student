"""Requirement-level hybrid retrieval over verified Career evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

from smart_agent.career.models import (
    EvidenceFact,
    EvidenceMatch,
    JDRequirement,
    PersonalEvidence,
    SourceReference,
    SupportLevel,
    record_to_dict,
    version_ref,
)
from smart_agent.career.store import CareerStore, RecordNotFoundError


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class OpenAICompatibleEmbeddingProvider:
    """Minimal embeddings client compatible with Doubao Ark and similar APIs."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 60.0
    name: str = "openai_compatible"

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.api_key.strip() or not self.model.strip():
            raise ValueError("Embedding base_url, api_key, and model are required.")
        if self.timeout_seconds <= 0:
            raise ValueError("Embedding timeout_seconds must be greater than zero.")
        if self.name == "openai_compatible":
            object.__setattr__(self, "name", f"openai_compatible:{self.model}")

    @classmethod
    def from_runtime(
        cls,
        runtime: dict[str, Any] | None,
    ) -> "OpenAICompatibleEmbeddingProvider | None":
        runtime = runtime or {}
        base_url = str(
            runtime.get("careerEmbeddingBaseUrl")
            or os.environ.get("CAREER_EMBEDDING_BASE_URL", "")
        ).strip()
        api_key = str(
            runtime.get("careerEmbeddingApiKey")
            or os.environ.get("CAREER_EMBEDDING_API_KEY", "")
            or os.environ.get("ARK_API_KEY", "")
        ).strip()
        model = str(
            runtime.get("careerEmbeddingModel")
            or os.environ.get("CAREER_EMBEDDING_MODEL", "")
        ).strip()
        if not (base_url and api_key and model):
            return None
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=float(runtime.get("careerEmbeddingTimeout", 60.0)),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/embeddings"):
            endpoint += "/embeddings"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ValueError("Embedding API returned an unexpected number of vectors.")
        ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
        vectors = [[float(value) for value in row.get("embedding", [])] for row in ordered]
        if not vectors or any(not vector for vector in vectors):
            raise ValueError("Embedding API returned an empty vector.")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("Embedding API returned inconsistent vector dimensions.")
        return vectors


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    keyword_weight: float = 0.6
    semantic_weight: float = 0.4
    minimum_score: float = 0.18
    indirect_threshold: float = 0.25
    partial_threshold: float = 0.48
    direct_threshold: float = 0.72
    top_k_per_requirement: int = 3

    def __post_init__(self) -> None:
        values = (
            self.keyword_weight,
            self.semantic_weight,
            self.minimum_score,
            self.indirect_threshold,
            self.partial_threshold,
            self.direct_threshold,
        )
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("Retrieval weights and thresholds must be between 0 and 1.")
        if self.keyword_weight + self.semantic_weight <= 0:
            raise ValueError("At least one retrieval channel must have positive weight.")
        if not self.minimum_score <= self.indirect_threshold <= self.partial_threshold <= self.direct_threshold:
            raise ValueError("Retrieval thresholds must be monotonic.")
        if self.top_k_per_requirement < 1:
            raise ValueError("top_k_per_requirement must be at least 1.")


@dataclass(frozen=True, slots=True)
class RetrievalBatch:
    matches: tuple[EvidenceMatch, ...]
    retrieval_mode: str
    embedding_provider: str = ""

    @property
    def gaps(self) -> tuple[EvidenceMatch, ...]:
        return tuple(match for match in self.matches if match.support_level is SupportLevel.UNSUPPORTED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_mode": self.retrieval_mode,
            "embedding_provider": self.embedding_provider,
            "matches": [record_to_dict(match) for match in self.matches],
            "gap_requirement_ids": [match.requirement_ref.record_id for match in self.gaps],
        }

    def persist(self, store: CareerStore) -> "RetrievalBatch":
        persisted: list[EvidenceMatch] = []
        for match in self.matches:
            try:
                existing = store.get(match.match_id, expected_type=EvidenceMatch)
            except RecordNotFoundError:
                store.put(match)
                persisted.append(match)
            else:
                persisted.append(existing)
        return RetrievalBatch(tuple(persisted), self.retrieval_mode, self.embedding_provider)


@dataclass(frozen=True, slots=True)
class _FactDocument:
    evidence: PersonalEvidence
    fact: EvidenceFact
    text: str


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize English terms plus Chinese unigrams/bigrams without external dependencies."""

    lowered = text.casefold()
    pieces = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]*|\d+(?:\.\d+)?", lowered)
    for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
        pieces.extend(run)
        pieces.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tuple(pieces)


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must use one dimension.")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def _bm25(query: tuple[str, ...], documents: list[tuple[str, ...]]) -> list[float]:
    if not documents:
        return []
    average_length = sum(len(document) for document in documents) / len(documents) or 1.0
    frequencies = [Counter(document) for document in documents]
    document_frequency = {
        term: sum(1 for frequency in frequencies if term in frequency)
        for term in set(query)
    }
    scores: list[float] = []
    for document, frequency in zip(documents, frequencies, strict=True):
        score = 0.0
        for term in set(query):
            count = frequency.get(term, 0)
            if not count:
                continue
            appearances = document_frequency[term]
            inverse_frequency = math.log(1 + (len(documents) - appearances + 0.5) / (appearances + 0.5))
            denominator = count + 1.5 * (1 - 0.75 + 0.75 * len(document) / average_length)
            score += inverse_frequency * count * 2.5 / denominator
        query_terms = {term for term in query if len(term) > 1} or set(query)
        coverage = len(query_terms.intersection(document)) / len(query_terms) if query_terms else 0.0
        scores.append(score * math.sqrt(coverage))
    maximum = max(scores, default=0.0)
    return [score / maximum if maximum else 0.0 for score in scores]


class HybridEvidenceRetriever:
    """Retrieve evidence independently for each JD requirement and emit explicit gaps."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        requirements: tuple[JDRequirement, ...] | list[JDRequirement],
        evidence_records: tuple[PersonalEvidence, ...] | list[PersonalEvidence],
    ) -> RetrievalBatch:
        active_requirements = tuple(requirement for requirement in requirements if requirement.active)
        documents = self._documents(evidence_records)
        mode = "hybrid" if self.embedding_provider is not None else "bm25"
        provider_name = self.embedding_provider.name if self.embedding_provider is not None else ""
        if not active_requirements:
            return RetrievalBatch((), mode, provider_name)

        semantic_scores: list[list[float]] | None = None
        if self.embedding_provider is not None and documents:
            texts = [requirement.text for requirement in active_requirements] + [item.text for item in documents]
            vectors = self.embedding_provider.embed(texts)
            if len(vectors) != len(texts):
                raise ValueError("Embedding provider returned an unexpected number of vectors.")
            query_vectors = vectors[: len(active_requirements)]
            document_vectors = vectors[len(active_requirements) :]
            semantic_scores = [
                [_cosine(query, document) for document in document_vectors]
                for query in query_vectors
            ]

        matches: list[EvidenceMatch] = []
        document_tokens = [tokenize(document.text) for document in documents]
        for requirement_index, requirement in enumerate(active_requirements):
            keyword_scores = _bm25(tokenize(requirement.text), document_tokens)
            semantic = semantic_scores[requirement_index] if semantic_scores is not None else None
            combined = self._combine(keyword_scores, semantic)
            matches.extend(self._matches_for_requirement(requirement, documents, keyword_scores, semantic, combined))
        return RetrievalBatch(tuple(matches), mode, provider_name)

    @staticmethod
    def _documents(
        evidence_records: tuple[PersonalEvidence, ...] | list[PersonalEvidence],
    ) -> list[_FactDocument]:
        documents: list[_FactDocument] = []
        for evidence in evidence_records:
            for fact in evidence.facts:
                if not fact.is_usable:
                    continue
                value = fact.value if isinstance(fact.value, str) else json.dumps(fact.value, ensure_ascii=False)
                excerpts = " ".join(source.excerpt for source in fact.source_refs if source.excerpt)
                documents.append(
                    _FactDocument(
                        evidence=evidence,
                        fact=fact,
                        text=f"{evidence.label} {fact.field_name} {value} {excerpts}".strip(),
                    )
                )
        return documents

    def _combine(
        self,
        keyword_scores: list[float],
        semantic_scores: list[float] | None,
    ) -> list[float]:
        if semantic_scores is None:
            return keyword_scores
        total_weight = self.config.keyword_weight + self.config.semantic_weight
        return [
            (self.config.keyword_weight * keyword + self.config.semantic_weight * semantic) / total_weight
            for keyword, semantic in zip(keyword_scores, semantic_scores, strict=True)
        ]

    def _matches_for_requirement(
        self,
        requirement: JDRequirement,
        documents: list[_FactDocument],
        keyword_scores: list[float],
        semantic_scores: list[float] | None,
        combined_scores: list[float],
    ) -> list[EvidenceMatch]:
        if not documents:
            return [self._gap(requirement)]
        matches: list[EvidenceMatch] = []
        query_terms = {term for term in tokenize(requirement.text) if len(term) > 1}
        for index, document in enumerate(documents):
            evidence_terms = {term for term in tokenize(document.text) if len(term) > 1}
            coverage = len(query_terms.intersection(evidence_terms)) / len(query_terms) if query_terms else 0.0
            semantic_value = semantic_scores[index] if semantic_scores is not None else None
            combined_value = max(0.0, min(1.0, float(combined_scores[index])))
            score = round(combined_value * coverage, 6)
            reason = (
                f"BM25={keyword_scores[index]:.3f}; "
                + (f"semantic={semantic_value:.3f}; " if semantic_value is not None else "semantic=disabled; ")
                + f"combined={combined_value:.3f}; coverage={coverage:.3f}; matched fact="
                + document.fact.field_name
            )
            support = self._support_level(score)
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "requirement": [requirement.requirement_id, requirement.revision],
                        "evidence": [document.evidence.evidence_id, document.evidence.revision],
                        "facts": [document.fact.fact_id],
                        "score": score,
                        "mode": "hybrid" if semantic_scores is not None else "bm25",
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            matches.append(
                EvidenceMatch(
                    match_id=f"match-{digest[:16]}",
                    requirement_ref=version_ref(requirement),
                    evidence_ref=version_ref(document.evidence),
                    evidence_fact_ids=(document.fact.fact_id,),
                    support_level=support,
                    reason=reason,
                    score=score,
                    source_refs=self._source_refs(requirement, (document.fact,)),
                )
            )
        if max(combined_scores, default=0.0) < self.config.minimum_score:
            matches.append(self._gap(requirement))
        return matches

    def _support_level(self, score: float) -> SupportLevel:
        if score >= self.config.direct_threshold:
            return SupportLevel.DIRECT
        if score >= self.config.partial_threshold:
            return SupportLevel.PARTIAL
        return SupportLevel.INDIRECT

    @staticmethod
    def _source_refs(
        requirement: JDRequirement,
        facts: tuple[EvidenceFact, ...],
    ) -> tuple[SourceReference, ...]:
        unique: dict[tuple[str, str, str], SourceReference] = {}
        for source in (
            *requirement.source_refs,
            *(source for fact in facts for source in fact.source_refs),
        ):
            key = (source.source_id, source.content_hash, json.dumps(source.locator, sort_keys=True))
            unique[key] = source
        return tuple(unique.values())

    @staticmethod
    def _gap(requirement: JDRequirement) -> EvidenceMatch:
        digest = hashlib.sha256(
            f"gap\0{requirement.requirement_id}\0{requirement.revision}".encode("utf-8")
        ).hexdigest()
        return EvidenceMatch(
            match_id=f"match-{digest[:16]}",
            requirement_ref=version_ref(requirement),
            evidence_ref=None,
            evidence_fact_ids=(),
            support_level=SupportLevel.UNSUPPORTED,
            reason="No verified or user-confirmed personal evidence met the retrieval threshold.",
            score=0.0,
            source_refs=requirement.source_refs,
        )
