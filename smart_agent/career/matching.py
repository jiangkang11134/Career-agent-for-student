"""Requirement-level matching cache with dependency-aware invalidation."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from smart_agent.career.materials import MaterialCategory, MaterialManifest
from smart_agent.career.models import (
    EvidenceMatch,
    EvidenceType,
    JDRequirement,
    PersonalEvidence,
    RequirementCategory,
    RequirementMatchCache,
    RequirementMatchCacheStatus,
    SourceReference,
    VersionRef,
    version_ref,
)
from smart_agent.career.store import CareerStore, RecordNotFoundError


def normalize_requirement_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


def requirement_signature(requirement: JDRequirement) -> str:
    payload = {
        "text": normalize_requirement_text(requirement.text),
        "category": requirement.category.value,
        "required": requirement.required,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _requirement_tokens(text: str) -> set[str]:
    lowered = normalize_requirement_text(text)
    tokens = set(re.findall(r"[a-z0-9][a-z0-9+#._-]*|[\u4e00-\u9fff]", lowered))
    for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return {token for token in tokens if len(token) > 1 or token.isascii()}


def _similarity(left: str, right: str) -> float:
    left_tokens = _requirement_tokens(left)
    right_tokens = _requirement_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens.intersection(right_tokens)
    coverage = len(overlap) / len(right_tokens)
    jaccard = len(overlap) / len(left_tokens.union(right_tokens))
    return 0.6 * coverage + 0.4 * jaccard


def _relevant_categories(category: RequirementCategory) -> set[MaterialCategory]:
    if category is RequirementCategory.EDUCATION:
        return {MaterialCategory.EDUCATION}
    if category is RequirementCategory.SOFT_SKILL:
        return {MaterialCategory.INTERNSHIP, MaterialCategory.RESEARCH}
    if category is RequirementCategory.PREFERENCE:
        return set(MaterialCategory)
    return {
        MaterialCategory.PROJECT,
        MaterialCategory.INTERNSHIP,
        MaterialCategory.RESEARCH,
        MaterialCategory.SKILLS,
    }


def clone_match_for_requirement(match: EvidenceMatch, requirement: JDRequirement) -> EvidenceMatch:
    digest = hashlib.sha256(
        json.dumps(
            {
                "requirement": [requirement.requirement_id, requirement.revision],
                "evidence": [match.evidence_ref.record_id, match.evidence_ref.revision]
                if match.evidence_ref is not None
                else None,
                "facts": match.evidence_fact_ids,
                "score": match.score,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    unique: dict[tuple[str, str, str], SourceReference] = {}
    for source in (*requirement.source_refs, *match.source_refs):
        key = (source.source_id, source.content_hash, json.dumps(source.locator, sort_keys=True))
        unique[key] = source
    return EvidenceMatch(
        match_id=f"match-{digest[:16]}",
        requirement_ref=version_ref(requirement),
        evidence_ref=match.evidence_ref,
        evidence_fact_ids=match.evidence_fact_ids,
        support_level=match.support_level,
        reason=f"Reused requirement-level cache: {match.reason}",
        score=match.score,
        source_refs=tuple(unique.values()),
    )


class RequirementMatchCacheService:
    def __init__(self, store: CareerStore) -> None:
        self.store = store

    def get_reusable(
        self,
        requirement: JDRequirement,
        *,
        manifests: tuple[MaterialManifest, ...] | list[MaterialManifest],
        material_catalog_version: str,
        evidence_records: tuple[PersonalEvidence, ...] | list[PersonalEvidence] = (),
        retrieval_backend: str = "",
        embedding_model: str = "",
        config_version: str = "default",
    ) -> RequirementMatchCache | None:
        cache_id = self.cache_id(requirement)
        try:
            cache = self.store.get(cache_id, expected_type=RequirementMatchCache)
        except RecordNotFoundError:
            cache = None
            candidates = tuple(
                item
                for item in self.store.list_latest(RequirementMatchCache)
                if item.category is requirement.category
                and item.required is requirement.required
                and _similarity(item.requirement_text, requirement.text) >= 0.78
            )
            cache = max(candidates, key=lambda item: _similarity(item.requirement_text, requirement.text), default=None)
        if cache is None:
            return None
        if cache.status is not RequirementMatchCacheStatus.FRESH:
            return None
        requested_backend = str(retrieval_backend or "").strip()
        requested_embedding = str(embedding_model or "").strip()
        requested_config = str(config_version or "").strip()
        if (
            (requested_backend and cache.retrieval_backend != requested_backend)
            or (requested_embedding and cache.embedding_model != requested_embedding)
            or (requested_config and cache.config_version != requested_config)
        ):
            self._mark_stale(cache, "retrieval configuration changed")
            return None
        current_by_id = {item.material_id: item for item in manifests}
        for material_id, content_hash in cache.material_dependency_hashes.items():
            current = current_by_id.get(material_id)
            if current is None or current.content_hash != content_hash:
                self._mark_stale(cache, "dependent material changed or was removed")
                return None
        relevant_categories = _relevant_categories(requirement.category)
        cached_scope_ids = set(cache.material_dependency_hashes)
        if any(
            item.category in relevant_categories and item.material_id not in cached_scope_ids
            for item in manifests
        ):
            self._mark_stale(cache, "new relevant material requires incremental retrieval")
            return None
        current_evidence = {item.evidence_id: item for item in evidence_records}
        for evidence_id, content_hash in cache.evidence_scope_hashes.items():
            current = current_evidence.get(evidence_id)
            if current is None or current.content_hash != content_hash:
                self._mark_stale(cache, "relevant evidence changed or was removed")
                return None
        cached_evidence_ids = set(cache.evidence_scope_hashes)
        if any(
            item.evidence_type.value in {category.value for category in relevant_categories}
            and item.evidence_id not in cached_evidence_ids
            for item in evidence_records
        ):
            self._mark_stale(cache, "new relevant evidence requires incremental retrieval")
            return None
        for ref in cache.evidence_dependency_refs:
            try:
                self.store.get(ref.record_id, ref.revision, expected_type=PersonalEvidence)
            except (RecordNotFoundError, ValueError):
                self._mark_stale(cache, "dependent evidence revision is unavailable")
                return None
        return cache

    def reuse(
        self,
        cache: RequirementMatchCache,
        requirement: JDRequirement,
    ) -> tuple[EvidenceMatch, ...]:
        matches = tuple(
            self.store.get(ref.record_id, ref.revision, expected_type=EvidenceMatch)
            for ref in cache.match_refs
        )
        cloned: list[EvidenceMatch] = []
        for match in matches:
            next_match = clone_match_for_requirement(match, requirement)
            try:
                self.store.get(next_match.match_id, expected_type=EvidenceMatch)
            except RecordNotFoundError:
                self.store.put(next_match)
            cloned.append(next_match)
        return tuple(cloned)

    def persist(
        self,
        requirement: JDRequirement,
        matches: tuple[EvidenceMatch, ...] | list[EvidenceMatch],
        *,
        manifests: tuple[MaterialManifest, ...] | list[MaterialManifest],
        material_catalog_version: str,
        evidence_records: tuple[PersonalEvidence, ...] | list[PersonalEvidence] = (),
        retrieval_backend: str = "",
        embedding_model: str = "",
        config_version: str = "default",
        reused_from: str = "",
    ) -> RequirementMatchCache:
        matches = tuple(matches)
        for match in matches:
            try:
                self.store.get(match.match_id, expected_type=EvidenceMatch)
            except RecordNotFoundError:
                self.store.put(match)
        evidence_refs = tuple(
            dict.fromkeys(
                match.evidence_ref
                for match in matches
                if match.evidence_ref is not None
            )
        )
        relevant_categories = _relevant_categories(requirement.category)
        dependency_ids = tuple(
            dict.fromkeys(
                item.material_id
                for item in manifests
                if item.category in relevant_categories
            )
        )
        manifest_by_id = {item.material_id: item for item in manifests}
        dependency_hashes = {
            material_id: manifest_by_id[material_id].content_hash
            for material_id in dependency_ids
        }
        evidence_scope_hashes = {
            item.evidence_id: item.content_hash
            for item in evidence_records
            if item.evidence_type.value in {category.value for category in relevant_categories}
        }
        cache = RequirementMatchCache(
            cache_id=self.cache_id(requirement),
            requirement_signature=requirement_signature(requirement),
            requirement_text=requirement.text,
            category=requirement.category,
            required=requirement.required,
            match_refs=tuple(version_ref(match) for match in matches),
            evidence_dependency_refs=tuple(evidence_refs),
            material_dependency_ids=dependency_ids,
            material_dependency_hashes=dependency_hashes,
            material_catalog_version=material_catalog_version,
            retrieval_mode="reused" if reused_from else "retrieved",
            retrieval_backend=retrieval_backend,
            embedding_model=embedding_model,
            config_version=config_version,
            status=RequirementMatchCacheStatus.FRESH,
            reused_from=reused_from,
            evidence_scope_hashes=evidence_scope_hashes,
            source_refs=requirement.source_refs,
        )
        try:
            current = self.store.get(cache.cache_id, expected_type=RequirementMatchCache)
        except RecordNotFoundError:
            self.store.put(cache)
            return cache
        changes = {
            field_name: getattr(cache, field_name)
            for field_name in (
                "requirement_signature",
                "requirement_text",
                "category",
                "required",
                "match_refs",
                "evidence_dependency_refs",
                "material_dependency_ids",
                "material_dependency_hashes",
                "material_catalog_version",
                "retrieval_mode",
                "retrieval_backend",
                "embedding_model",
                "config_version",
                "status",
                "reused_from",
                "evidence_scope_hashes",
                "source_refs",
            )
            if getattr(current, field_name) != getattr(cache, field_name)
        }
        return self.store.revise_record(
            cache.cache_id,
            actor="career_requirement_cache",
            reason="Refresh requirement-level evidence cache",
            changes=changes,
        ) if changes else current

    def _mark_stale(self, cache: RequirementMatchCache, reason: str) -> None:
        if cache.status is RequirementMatchCacheStatus.STALE:
            return
        self.store.revise_record(
            cache.cache_id,
            actor="career_requirement_cache",
            reason=reason,
            changes={"status": RequirementMatchCacheStatus.STALE},
        )

    @staticmethod
    def cache_id(requirement: JDRequirement) -> str:
        return f"requirement-cache-{requirement_signature(requirement)[:20]}"
