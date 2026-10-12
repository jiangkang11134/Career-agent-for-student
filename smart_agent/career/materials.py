"""Classified Career material ingestion and project-specific ChromaDB indexing.

Raw documents remain immutable files with a manifest. Only user-confirmed
facts enter ``CareerStore``; project chunks are indexed separately for
retrieval and are never treated as verified resume claims by themselves.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from smart_agent.career.models import (
    EvidenceFact,
    EvidenceType,
    FactStatus,
    PersonalEvidence,
    ProjectDescriptionDocument,
    ProjectDescriptionStatus,
    ResumeSectionBlock,
    ResumeSource,
    SourceReference,
    SourceType,
    create_personal_evidence,
    split_resume_source_sections,
)
from smart_agent.career.store import CareerStore, RecordNotFoundError, RevisionConflictError


class MaterialCategory(str, Enum):
    RESUME = "resume"
    EDUCATION = "education"
    AWARD = "award"
    INTERNSHIP = "internship"
    PROJECT = "project"
    RESEARCH = "research"
    SKILLS = "skills"


_RESUME_SECTION_CATEGORIES = {
    "internships": MaterialCategory.INTERNSHIP,
    "projects": MaterialCategory.PROJECT,
    "education": MaterialCategory.EDUCATION,
    "skills": MaterialCategory.SKILLS,
    "research": MaterialCategory.RESEARCH,
    "awards": MaterialCategory.AWARD,
}


def split_resume_sections(text: str) -> dict[MaterialCategory, str]:
    """Split an imported resume by explicit section headings.

    This is deliberately deterministic: the original resume is an import
    source only, and no model receives the unsplit document afterward.
    Unknown lines are kept in the skills/other section instead of being
    silently discarded.
    """
    blocks = split_resume_source_sections(
        text,
        resume_source_id="resume-source-preview",
        source_revision=1,
    )
    return {
        _RESUME_SECTION_CATEGORIES[block.section_id]: block.text
        for block in blocks
        if block.text
    }


def material_catalog_version(
    manifests: tuple[MaterialManifest, ...] | list[MaterialManifest],
    evidence_records: tuple[PersonalEvidence, ...] | list[PersonalEvidence],
    candidate_facts: tuple[CandidateFact, ...] | list[CandidateFact],
    project_descriptions: tuple[ProjectDescriptionDocument, ...] | list[ProjectDescriptionDocument] = (),
) -> str:
    """Return a stable version for every input that can change usable Career evidence."""

    payload = {
        "materials": sorted(
            (item.material_id, item.revision, item.content_hash, item.fact_status)
            for item in manifests
        ),
        "evidence": sorted(
            (item.evidence_id, item.revision, item.content_hash)
            for item in evidence_records
        ),
        "candidate_facts": sorted(
            (
                item.candidate_fact_id,
                item.status.value,
                item.content_hash,
                item.evidence_id,
                item.fact_id,
            )
            for item in candidate_facts
        ),
        "project_descriptions": sorted(
            (
                item.project_description_id,
                item.revision,
                item.content_hash,
                item.status.value,
            )
            for item in project_descriptions
        ),
    }
    return _digest(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


_EVIDENCE_TYPES = {
    MaterialCategory.EDUCATION: EvidenceType.EDUCATION,
    MaterialCategory.AWARD: EvidenceType.AWARD,
    MaterialCategory.INTERNSHIP: EvidenceType.INTERNSHIP,
    MaterialCategory.PROJECT: EvidenceType.PROJECT,
    MaterialCategory.RESEARCH: EvidenceType.RESEARCH,
    MaterialCategory.SKILLS: EvidenceType.SKILLS,
}
_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".sql",
    ".html",
    ".css",
}
_SKIP_DIRS = {".git", ".runtime", "node_modules", "__pycache__", ".venv", "venv"}


def _digest(value: bytes | str) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "-", value.strip()).strip("-._")
    return normalized[:80] or "material"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
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
class MaterialManifest:
    material_id: str
    category: MaterialCategory
    label: str
    original_name: str
    media_type: str
    content_hash: str
    source_path: str
    raw_path: str
    extracted_text_path: str
    parser: str
    parser_confidence: float
    project_id: str = ""
    revision: int = 1
    fact_status: str = "awaiting_fact_confirmation"
    chunker: str = ""
    index_backend: str = ""
    embedding_backend: str = ""
    index_variant: str = ""
    active: bool = True
    deleted_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", MaterialCategory(self.category))
        object.__setattr__(self, "label", str(self.label or "").strip())
        object.__setattr__(self, "original_name", str(self.original_name or "").strip())
        object.__setattr__(self, "source_path", str(self.source_path or "").strip())
        object.__setattr__(self, "raw_path", str(self.raw_path or "").strip())
        object.__setattr__(self, "extracted_text_path", str(self.extracted_text_path or "").strip())
        if not self.material_id or not self.label or not self.original_name:
            raise ValueError("MaterialManifest material_id, label, and original_name are required.")
        if len(self.content_hash) != 64 or any(char not in "0123456789abcdef" for char in self.content_hash.lower()):
            raise ValueError("MaterialManifest content_hash must be a SHA-256 hex digest.")
        if not 0 <= self.parser_confidence <= 1:
            raise ValueError("MaterialManifest parser_confidence must be between 0 and 1.")
        if self.revision < 1:
            raise ValueError("MaterialManifest revision must be at least 1.")
        if self.deleted_at < 0:
            raise ValueError("MaterialManifest deleted_at cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "material_id": self.material_id,
            "category": self.category.value,
            "label": self.label,
            "original_name": self.original_name,
            "media_type": self.media_type,
            "content_hash": self.content_hash,
            "source_path": self.source_path,
            "raw_path": self.raw_path,
            "extracted_text_path": self.extracted_text_path,
            "parser": self.parser,
            "parser_confidence": self.parser_confidence,
            "project_id": self.project_id,
            "revision": self.revision,
            "fact_status": self.fact_status,
            "chunker": self.chunker,
            "index_backend": self.index_backend,
            "embedding_backend": self.embedding_backend,
            "index_variant": self.index_variant,
            "active": self.active,
            "deleted_at": self.deleted_at,
        }
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MaterialManifest":
        return cls(
            material_id=str(value["material_id"]),
            category=MaterialCategory(value["category"]),
            label=str(value["label"]),
            original_name=str(value["original_name"]),
            media_type=str(value.get("media_type") or "text/plain"),
            content_hash=str(value["content_hash"]),
            source_path=str(value.get("source_path") or ""),
            raw_path=str(value["raw_path"]),
            extracted_text_path=str(value["extracted_text_path"]),
            parser=str(value.get("parser") or "plain_text"),
            parser_confidence=float(value.get("parser_confidence", 1.0)),
            project_id=str(value.get("project_id") or ""),
            revision=int(value.get("revision", 1)),
            fact_status=str(value.get("fact_status") or "awaiting_fact_confirmation"),
            chunker=str(value.get("chunker") or ""),
            index_backend=str(value.get("index_backend") or ""),
            embedding_backend=str(value.get("embedding_backend") or ""),
            index_variant=str(value.get("index_variant") or ""),
            active=bool(value.get("active", True)),
            deleted_at=float(value.get("deleted_at", 0.0) or 0.0),
        )


@dataclass(frozen=True, slots=True)
class ProjectChunk:
    chunk_id: str
    project_id: str
    material_id: str
    source_id: str
    content: str
    locator: dict[str, Any]
    content_hash: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "material_id": self.material_id,
            "source_id": self.source_id,
            "locator": json.dumps(self.locator, ensure_ascii=False, sort_keys=True),
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class ProjectSearchHit:
    chunk_id: str
    project_id: str
    material_id: str
    source_id: str
    content: str
    locator: dict[str, Any]
    distance: float | None
    lexical_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "project_id": self.project_id,
            "material_id": self.material_id,
            "source_id": self.source_id,
            "content": self.content,
            "locator": self.locator,
            "distance": self.distance,
            "lexical_score": round(self.lexical_score, 6),
        }


class CandidateFactStatus(str, Enum):
    CANDIDATE = "candidate"
    IMPORTED = "imported"
    REJECTED = "rejected"


def _source_ref_to_dict(ref: SourceReference) -> dict[str, Any]:
    return {
        "source_id": ref.source_id,
        "source_type": ref.source_type.value,
        "locator": ref.locator,
        "content_hash": ref.content_hash,
        "excerpt": ref.excerpt,
        "captured_at": ref.captured_at,
    }


def _source_ref_from_dict(value: dict[str, Any]) -> SourceReference:
    return SourceReference(
        source_id=str(value["source_id"]),
        source_type=SourceType(value["source_type"]),
        locator=dict(value.get("locator") or {}),
        content_hash=str(value["content_hash"]),
        excerpt=str(value.get("excerpt") or ""),
        captured_at=float(value.get("captured_at") or time.time()),
    )


@dataclass(frozen=True, slots=True)
class CandidateFact:
    """A proposed resume fact extracted from raw material but not yet usable."""

    candidate_fact_id: str
    material_id: str
    category: MaterialCategory
    label: str
    field_name: str
    value: Any
    source_refs: tuple[SourceReference, ...]
    confidence: float
    extractor: str
    status: CandidateFactStatus = CandidateFactStatus.CANDIDATE
    evidence_id: str = ""
    fact_id: str = ""
    rejection_reason: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    content_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_fact_id", str(self.candidate_fact_id or "").strip())
        object.__setattr__(self, "material_id", str(self.material_id or "").strip())
        object.__setattr__(self, "category", MaterialCategory(self.category))
        object.__setattr__(self, "label", str(self.label or "").strip())
        object.__setattr__(self, "field_name", str(self.field_name or "").strip())
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        object.__setattr__(self, "extractor", str(self.extractor or "").strip())
        object.__setattr__(self, "status", CandidateFactStatus(self.status))
        if not self.candidate_fact_id or not self.material_id or not self.label or not self.field_name:
            raise ValueError("CandidateFact requires candidate_fact_id, material_id, label, and field_name.")
        if self.category is MaterialCategory.RESUME:
            raise ValueError("Resume sources cannot produce candidate personal facts.")
        if not self.source_refs:
            raise ValueError("CandidateFact requires at least one source reference.")
        if not 0 <= self.confidence <= 1:
            raise ValueError("CandidateFact confidence must be between 0 and 1.")
        if self.updated_at < self.created_at:
            raise ValueError("CandidateFact updated_at cannot precede created_at.")
        expected = self._stable_hash()
        if self.content_hash and self.content_hash != expected:
            raise ValueError("CandidateFact content_hash does not match its stable content.")
        object.__setattr__(self, "content_hash", expected)

    @classmethod
    def create(
        cls,
        *,
        manifest: MaterialManifest,
        field_name: str,
        value: Any,
        source_refs: tuple[SourceReference, ...] | list[SourceReference],
        confidence: float,
        extractor: str,
    ) -> "CandidateFact":
        seed = json.dumps(
            {
                "material_id": manifest.material_id,
                "category": manifest.category.value,
                "field_name": field_name,
                "value": value,
                "source_refs": [_source_ref_to_dict(ref) for ref in source_refs],
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return cls(
            candidate_fact_id=f"candidate-fact-{_digest(seed)[:20]}",
            material_id=manifest.material_id,
            category=manifest.category,
            label=manifest.label,
            field_name=field_name,
            value=value,
            source_refs=tuple(source_refs),
            confidence=confidence,
            extractor=extractor,
        )

    def _stable_hash(self) -> str:
        payload = {
            "candidate_fact_id": self.candidate_fact_id,
            "material_id": self.material_id,
            "category": self.category.value,
            "label": self.label,
            "field_name": self.field_name,
            "value": self.value,
            "source_refs": [_source_ref_to_dict(ref) for ref in self.source_refs],
            "confidence": self.confidence,
            "extractor": self.extractor,
        }
        return _digest(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))

    def mark_imported(self, *, evidence_id: str, fact_id: str) -> "CandidateFact":
        return replace(
            self,
            status=CandidateFactStatus.IMPORTED,
            evidence_id=str(evidence_id).strip(),
            fact_id=str(fact_id).strip(),
            updated_at=time.time(),
            content_hash="",
        )

    def reject(self, *, reason: str) -> "CandidateFact":
        reason = str(reason or "").strip()
        if not reason:
            raise ValueError("rejection reason is required.")
        return replace(
            self,
            status=CandidateFactStatus.REJECTED,
            rejection_reason=reason,
            updated_at=time.time(),
            content_hash="",
        )

    def to_evidence_fact(self) -> EvidenceFact:
        return EvidenceFact.create(
            self.field_name,
            self.value,
            source_refs=self.source_refs,
            confidence=self.confidence,
            status=FactStatus.UNVERIFIED,
            actor="candidate_fact_importer",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_fact_id": self.candidate_fact_id,
            "material_id": self.material_id,
            "category": self.category.value,
            "label": self.label,
            "field_name": self.field_name,
            "value": self.value,
            "source_refs": [_source_ref_to_dict(ref) for ref in self.source_refs],
            "confidence": self.confidence,
            "extractor": self.extractor,
            "status": self.status.value,
            "evidence_id": self.evidence_id,
            "fact_id": self.fact_id,
            "rejection_reason": self.rejection_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateFact":
        return cls(
            candidate_fact_id=str(value["candidate_fact_id"]),
            material_id=str(value["material_id"]),
            category=MaterialCategory(value["category"]),
            label=str(value["label"]),
            field_name=str(value["field_name"]),
            value=value.get("value"),
            source_refs=tuple(_source_ref_from_dict(ref) for ref in value.get("source_refs", [])),
            confidence=float(value.get("confidence", 0.0)),
            extractor=str(value.get("extractor") or "unknown"),
            status=CandidateFactStatus(value.get("status") or CandidateFactStatus.CANDIDATE),
            evidence_id=str(value.get("evidence_id") or ""),
            fact_id=str(value.get("fact_id") or ""),
            rejection_reason=str(value.get("rejection_reason") or ""),
            created_at=float(value.get("created_at") or time.time()),
            updated_at=float(value.get("updated_at") or time.time()),
            content_hash=str(value.get("content_hash") or ""),
        )


class CareerFactExtractor(Protocol):
    """Pluggable raw-material-to-candidate-fact extractor."""

    name: str

    def extract(
        self,
        manifest: MaterialManifest,
        text: str,
        *,
        max_facts: int = 12,
    ) -> tuple[CandidateFact, ...]: ...


class HeuristicCareerFactExtractor:
    """Deterministic baseline extractor; LLM extraction can replace this adapter."""

    name = "heuristic-line-fact-extractor-v1"

    def extract(
        self,
        manifest: MaterialManifest,
        text: str,
        *,
        max_facts: int = 12,
    ) -> tuple[CandidateFact, ...]:
        if manifest.category is MaterialCategory.RESUME:
            return ()
        if manifest.category is MaterialCategory.SKILLS:
            block = str(text or "").replace("\x00", "").strip()
            if not block:
                return ()
            source = SourceReference.create(
                source_id=manifest.material_id,
                source_type=SourceType.FILE,
                locator={
                    "material_id": manifest.material_id,
                    "path": manifest.extracted_text_path,
                },
                raw_content=block,
                excerpt=block[:500],
            )
            return (
                CandidateFact.create(
                    manifest=manifest,
                    field_name="content",
                    value=block,
                    source_refs=(source,),
                    confidence=0.9,
                    extractor=self.name,
                ),
            )
        candidates: list[CandidateFact] = []
        seen: set[str] = set()
        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = _normalize_candidate_fact_line(raw_line)
            if not line:
                continue
            normalized = re.sub(r"\s+", " ", line.casefold())
            if normalized in seen:
                continue
            seen.add(normalized)
            source = SourceReference.create(
                source_id=manifest.material_id,
                source_type=(
                    SourceType.REPOSITORY
                    if manifest.category is MaterialCategory.PROJECT and manifest.source_path
                    else SourceType.FILE
                ),
                locator={
                    "material_id": manifest.material_id,
                    "path": manifest.extracted_text_path,
                    "line_start": line_no,
                    "line_end": line_no,
                },
                raw_content=line,
                excerpt=line[:240],
            )
            candidates.append(
                CandidateFact.create(
                    manifest=manifest,
                    field_name=_infer_candidate_field_name(manifest.category, line),
                    value=line,
                    source_refs=(source,),
                    confidence=0.45,
                    extractor=self.name,
                )
            )
            if len(candidates) >= max_facts:
                break
        return tuple(candidates)


def _normalize_candidate_fact_line(line: str) -> str:
    text = html.unescape(str(line or "")).strip()
    if re.match(r"^#{1,6}\s+", text):
        return ""
    text = re.sub(r"^[-*+•]\s+", "", text)
    text = re.sub(r"^\d+[.)、]\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 6 or len(text) > 500:
        return ""
    if text.endswith((":", "：")) and len(text) < 40:
        return ""
    if not re.search(r"[A-Za-z0-9一-鿿]", text):
        return ""
    return text


def _infer_candidate_field_name(category: MaterialCategory, line: str) -> str:
    lower = line.casefold()
    if category is MaterialCategory.PROJECT:
        if any(token in lower for token in ("技术", "栈", "stack", "python", "java", "react", "redis", "mysql")):
            return "technology"
        if any(token in lower for token in ("指标", "提升", "降低", "%", "倍", "ms", "qps", "accuracy")):
            return "impact"
        return "project_summary"
    if category is MaterialCategory.INTERNSHIP:
        prefix = re.split(r"[:：]", line, maxsplit=1)[0].strip()
        structured_labels = {
            "实习公司": "company",
            "实习公司名称": "company",
            "公司": "company",
            "实习岗位": "role",
            "岗位": "role",
            "职位": "role",
            "项目部门": "department",
            "部门": "department",
            "实习简述": "summary",
            "实习成果": "summary",
            "简述": "summary",
            "实习内容": "content",
            "内容": "content",
            "职责": "content",
            "实习日期": "date",
            "日期": "date",
            "时间": "date",
        }
        if prefix in structured_labels:
            return structured_labels[prefix]
        if any(token in lower for token in ("title", "岗位", "职位", "算法", "后端", "工程师")):
            return "title"
        if any(token in lower for token in ("负责", "implemented", "built", "优化", "developed")):
            return "responsibility"
        return "internship_summary"
    if category is MaterialCategory.EDUCATION:
        return "education"
    if category is MaterialCategory.AWARD:
        return "award"
    if category is MaterialCategory.RESEARCH:
        return "research"
    return "summary"


class CandidateFactStore:
    """Persistent staging area between imported material and confirmed evidence."""

    name = "json-candidate-facts-v1"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "candidate_facts.json"

    def list(
        self,
        *,
        material_id: str | None = None,
        category: MaterialCategory | str | None = None,
        status: CandidateFactStatus | str | None = None,
    ) -> tuple[CandidateFact, ...]:
        facts = self._load()
        if material_id:
            selected = str(material_id).strip()
            facts = [fact for fact in facts if fact.material_id == selected]
        if category is not None:
            selected_category = MaterialCategory(category)
            facts = [fact for fact in facts if fact.category is selected_category]
        if status is not None:
            selected_status = CandidateFactStatus(status)
            facts = [fact for fact in facts if fact.status is selected_status]
        return tuple(sorted(facts, key=lambda fact: (fact.created_at, fact.candidate_fact_id)))

    def upsert(self, candidates: tuple[CandidateFact, ...] | list[CandidateFact]) -> tuple[CandidateFact, ...]:
        entries = {fact.candidate_fact_id: fact for fact in self._load()}
        for candidate in candidates:
            existing = entries.get(candidate.candidate_fact_id)
            if existing is None or existing.status is CandidateFactStatus.CANDIDATE:
                entries[candidate.candidate_fact_id] = candidate
        self._write(entries.values())
        return tuple(entries[candidate.candidate_fact_id] for candidate in candidates)

    def import_to_evidence(
        self,
        candidate_fact_ids: tuple[str, ...] | list[str],
        *,
        career_store: CareerStore,
    ) -> tuple[PersonalEvidence, tuple[CandidateFact, ...]]:
        selected = self._selected(candidate_fact_ids)
        if any(fact.status is not CandidateFactStatus.CANDIDATE for fact in selected):
            raise ValueError("Only candidate facts can be imported into PersonalEvidence.")
        groups: dict[tuple[str, MaterialCategory, str], list[CandidateFact]] = {}
        for fact in selected:
            if fact.category not in _EVIDENCE_TYPES:
                raise ValueError(f"{fact.category.value} cannot become PersonalEvidence.")
            groups.setdefault((fact.material_id, fact.category, fact.label), []).append(fact)
        records: list[PersonalEvidence] = []
        transitions: dict[str, CandidateFact] = {}
        for (_material_id, category, label), facts in groups.items():
            evidence_facts = tuple(fact.to_evidence_fact() for fact in facts)
            source_refs = tuple({ref.content_hash: ref for fact in facts for ref in fact.source_refs}.values())
            record = create_personal_evidence(
                _EVIDENCE_TYPES[category],
                label=label,
                facts=evidence_facts,
                source_refs=source_refs,
                actor="candidate_fact_importer",
            )
            career_store.put(record)
            records.append(record)
            for candidate, evidence_fact in zip(facts, evidence_facts, strict=True):
                transitions[candidate.candidate_fact_id] = candidate.mark_imported(
                    evidence_id=record.evidence_id,
                    fact_id=evidence_fact.fact_id,
                )
        entries = {fact.candidate_fact_id: fact for fact in self._load()}
        entries.update(transitions)
        self._write(entries.values())
        return tuple(records), tuple(transitions[fact.candidate_fact_id] for fact in selected)

    def reject(
        self,
        candidate_fact_ids: tuple[str, ...] | list[str],
        *,
        reason: str,
    ) -> tuple[CandidateFact, ...]:
        selected = self._selected(candidate_fact_ids)
        if any(fact.status is CandidateFactStatus.IMPORTED for fact in selected):
            raise ValueError("Imported candidate facts must be invalidated through PersonalEvidence.")
        updates = {fact.candidate_fact_id: fact.reject(reason=reason) for fact in selected}
        entries = {fact.candidate_fact_id: fact for fact in self._load()}
        entries.update(updates)
        self._write(entries.values())
        return tuple(updates[fact.candidate_fact_id] for fact in selected)

    def retire_material(self, material_id: str, *, reason: str = "Material superseded") -> tuple[CandidateFact, ...]:
        """Reject pending facts when their source material is replaced or deleted."""
        selected = self.list(material_id=material_id, status=CandidateFactStatus.CANDIDATE)
        if not selected:
            return ()
        return self.reject(
            [fact.candidate_fact_id for fact in selected],
            reason=reason,
        )

    def purge_material(self, material_id: str) -> int:
        """Remove every staged fact for a deleted material.

        This is intentionally separate from ``retire_material``: normal
        replacement keeps rejected facts for audit, while an explicit user
        deletion removes the staging records so they cannot reappear in the
        candidate-fact UI or be selected by a later workflow.
        """
        identity = str(material_id or "").strip()
        if not identity:
            raise ValueError("material_id is required")
        current = self._load()
        remaining = [fact for fact in current if fact.material_id != identity]
        removed = len(current) - len(remaining)
        if removed:
            self._write(remaining)
        return removed

    def _selected(self, candidate_fact_ids: tuple[str, ...] | list[str]) -> tuple[CandidateFact, ...]:
        ids = tuple(dict.fromkeys(str(item).strip() for item in candidate_fact_ids if str(item).strip()))
        if not ids:
            raise ValueError("candidate_fact_ids must contain at least one ID.")
        entries = {fact.candidate_fact_id: fact for fact in self._load()}
        missing = [identity for identity in ids if identity not in entries]
        if missing:
            raise KeyError(f"Unknown candidate facts: {missing}")
        return tuple(entries[identity] for identity in ids)

    def _load(self) -> list[CandidateFact]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot load candidate facts: {exc}") from exc
        return [CandidateFact.from_dict(item) for item in payload.get("candidate_facts", [])]

    def _write(self, facts: Any) -> None:
        ordered = sorted(tuple(facts), key=lambda fact: (fact.created_at, fact.candidate_fact_id))
        _atomic_write(
            self.path,
            json.dumps(
                {"schema_version": "1.0", "candidate_facts": [fact.to_dict() for fact in ordered]},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )


class EmbeddingBackend(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    # BGE-M3 providers expose this optional extension.  Keeping it optional
    # preserves the offline hashing backend and existing third-party adapters.
    def embed_with_sparse(self, texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]: ...


class MaterialTextExtractor(Protocol):
    """Pluggable document-to-text parser used by ``MaterialStore``."""

    name: str

    def extract(self, path: Path) -> tuple[str, str, str, float]: ...


class DefaultMaterialTextExtractor:
    """Built-in parser adapter; OCR can be added without changing MaterialStore."""

    name = "builtin-material-parser-v1"

    def extract(self, path: Path) -> tuple[str, str, str, float]:
        return extract_material_text(path)


class ProjectChunker(Protocol):
    """Pluggable project text segmentation strategy."""

    name: str

    def chunk(
        self,
        text: str,
        *,
        project_id: str,
        material_id: str,
        source_id: str,
    ) -> tuple[ProjectChunk, ...]: ...


class ProjectIndexBackend(Protocol):
    """Pluggable project chunk index; ChromaDB and Qdrant are supported."""

    name: str

    def upsert(self, chunks: list[ProjectChunk] | tuple[ProjectChunk, ...]) -> int: ...

    def delete(
        self,
        *,
        project_ids: set[str] | None = None,
        material_ids: set[str] | None = None,
    ) -> int: ...

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        project_id: str | None = None,
    ) -> tuple[ProjectSearchHit, ...]: ...


class LineProjectChunker:
    """Default line-window chunker with explicit, reproducible parameters."""

    def __init__(self, *, max_chars: int = 1400, overlap_chars: int = 160) -> None:
        if max_chars < 300 or overlap_chars < 0 or overlap_chars >= max_chars:
            raise ValueError("Invalid project chunk limits.")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.name = f"line-window-v1:{max_chars}:{overlap_chars}"

    def chunk(
        self,
        text: str,
        *,
        project_id: str,
        material_id: str,
        source_id: str,
    ) -> tuple[ProjectChunk, ...]:
        return chunk_project_text(
            text,
            project_id=project_id,
            material_id=material_id,
            source_id=source_id,
            max_chars=self.max_chars,
            overlap_chars=self.overlap_chars,
        )


class HashingEmbedding:
    """Offline deterministic baseline used when no embedding API is configured."""

    name = "hashing-384"

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.name = f"hashing-{dimension}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.casefold())
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                vector[index] += 1.0 if digest[4] & 1 else -1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class ProjectKnowledgeStore:
    """Persistent ChromaDB project chunk store with an offline-safe baseline."""

    name = "chromadb"

    def __init__(
        self,
        root: str | Path,
        *,
        embedding_backend: EmbeddingBackend | None = None,
        collection_variant: str = "default",
        embedding_batch_size: int = 32,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("ChromaDB is required for project knowledge indexing; install the career extra.") from exc
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedding_backend = embedding_backend or HashingEmbedding()
        self.collection_variant = str(collection_variant or "default").strip() or "default"
        try:
            self.embedding_batch_size = int(embedding_batch_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding_batch_size must be an integer") from exc
        if not 1 <= self.embedding_batch_size <= 512:
            raise ValueError("embedding_batch_size must be between 1 and 512")
        collection_key = f"{self.embedding_backend.name}:{self.collection_variant}"
        collection_suffix = hashlib.sha256(collection_key.encode("utf-8")).hexdigest()[:12]
        self.collection_name = f"project_chunks_{collection_suffix}"
        self.client = chromadb.PersistentClient(path=str(self.root))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_backend": self.embedding_backend.name,
                "collection_variant": self.collection_variant,
            },
        )
        # Chroma stores dense vectors only.  Keep sparse lexical weights in a
        # tiny sidecar SQLite index so BGE-M3 hybrid retrieval can be enabled
        # without changing the existing collection or production data layout.
        self.sparse_path = self.root / "sparse.sqlite3"
        self._sparse_db = sqlite3.connect(str(self.sparse_path), check_same_thread=False)
        self._sparse_db.execute(
            "CREATE TABLE IF NOT EXISTS project_sparse ("
            "chunk_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, material_id TEXT NOT NULL DEFAULT '', "
            "sparse_json TEXT NOT NULL DEFAULT '{}')"
        )
        columns = {str(row[1]) for row in self._sparse_db.execute("PRAGMA table_info(project_sparse)")}
        if "material_id" not in columns:
            self._sparse_db.execute("ALTER TABLE project_sparse ADD COLUMN material_id TEXT NOT NULL DEFAULT ''")
        self._sparse_db.commit()

    @staticmethod
    def _sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values())) or 1.0
        right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values())) or 1.0
        dot = sum(float(value) * float(right.get(token, 0.0)) for token, value in left.items())
        return dot / (left_norm * right_norm)

    def _embed(self, texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
        method = getattr(self.embedding_backend, "embed_with_sparse", None)
        dense_rows: list[list[float]] = []
        sparse_rows: list[dict[str, float]] = []
        for start in range(0, len(texts), self.embedding_batch_size):
            batch = texts[start : start + self.embedding_batch_size]
            if callable(method):
                dense, sparse = method(batch)
                if not isinstance(dense, list) or len(dense) != len(batch):
                    raise ValueError("Embedding backend returned an unexpected vector count")
                rows = list(sparse) if isinstance(sparse, list) else []
                if len(rows) != len(batch):
                    rows = [{} for _ in batch]
                dense_rows.extend(dense)
                sparse_rows.extend(dict(row or {}) if isinstance(row, dict) else {} for row in rows)
            else:
                dense = self.embedding_backend.embed(batch)
                if not isinstance(dense, list) or len(dense) != len(batch):
                    raise ValueError("Embedding backend returned an unexpected vector count")
                dense_rows.extend(dense)
                sparse_rows.extend({} for _ in batch)
        return dense_rows, sparse_rows

    def upsert(self, chunks: list[ProjectChunk] | tuple[ProjectChunk, ...]) -> int:
        if not chunks:
            return 0
        material_ids = {chunk.material_id for chunk in chunks}
        # Complete embedding before replacing the old rows. A worker failure must
        # leave the last usable project index intact for interview retrieval.
        embeddings, sparse_rows = self._embed([chunk.content for chunk in chunks])
        # Re-indexing one source must not delete sibling sources in the same project.
        self.delete(material_ids=material_ids)
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[chunk.to_metadata() for chunk in chunks],
            embeddings=embeddings,
        )
        self._sparse_db.executemany(
            "INSERT OR REPLACE INTO project_sparse(chunk_id, project_id, material_id, sparse_json) VALUES (?, ?, ?, ?)",
            [
                (chunk.chunk_id, chunk.project_id, chunk.material_id, json.dumps(sparse, ensure_ascii=False, sort_keys=True))
                for chunk, sparse in zip(chunks, sparse_rows, strict=True)
            ],
        )
        self._sparse_db.commit()
        return len(chunks)

    def delete(
        self,
        *,
        project_ids: set[str] | None = None,
        material_ids: set[str] | None = None,
    ) -> int:
        where: dict[str, Any] | None = None
        if project_ids and material_ids:
            expressions = [{"project_id": value} for value in project_ids] + [
                {"material_id": value} for value in material_ids
            ]
            where = expressions[0] if len(expressions) == 1 else {"$or": expressions}
        elif project_ids:
            expressions = [{"project_id": value} for value in project_ids]
            where = expressions[0] if len(expressions) == 1 else {"$or": expressions}
        elif material_ids:
            expressions = [{"material_id": value} for value in material_ids]
            where = expressions[0] if len(expressions) == 1 else {"$or": expressions}
        if where is None:
            return 0
        existing = self.collection.get(where=where, include=[])
        ids = list(existing.get("ids") or [])
        if ids:
            self.collection.delete(ids=ids)
            self._sparse_db.executemany("DELETE FROM project_sparse WHERE chunk_id = ?", [(item,) for item in ids])
            self._sparse_db.commit()
        return len(ids)

    def search(self, query: str, *, top_k: int = 5, project_id: str | None = None) -> tuple[ProjectSearchHit, ...]:
        query = " ".join(str(query or "").split())
        if not query:
            raise ValueError("Project knowledge query is required.")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50.")
        where = {"project_id": project_id} if project_id else None
        count = self.collection.count()
        if not count:
            return ()
        query_embeddings, query_sparse_rows = self._embed([query])
        query_embedding = query_embeddings[0]
        query_sparse = query_sparse_rows[0] if query_sparse_rows else {}
        candidate_limit = min(max(top_k * 8, 50), count)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        query_tokens = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", query.casefold()))
        dense_by_id = {
            str(chunk_id): (
                str(documents[index] or "") if index < len(documents) else "",
                dict(metadatas[index] or {}) if index < len(metadatas) else {},
                float(distances[index]) if index < len(distances) and distances[index] is not None else None,
            )
            for index, chunk_id in enumerate(ids)
        }
        sparse_scores: dict[str, float] = {}
        if query_sparse:
            sparse_query = "SELECT chunk_id, sparse_json FROM project_sparse"
            sparse_params: tuple[Any, ...] = ()
            if project_id:
                sparse_query += " WHERE project_id = ?"
                sparse_params = (project_id,)
            for chunk_id, sparse_json in self._sparse_db.execute(sparse_query, sparse_params):
                try:
                    row = json.loads(str(sparse_json or "{}"))
                except json.JSONDecodeError:
                    row = {}
                if isinstance(row, dict):
                    sparse_scores[str(chunk_id)] = self._sparse_cosine(query_sparse, row)
            sparse_scores = dict(sorted(sparse_scores.items(), key=lambda item: -item[1])[:candidate_limit])
        candidates = set(dense_by_id) | set(sparse_scores)
        hits: list[ProjectSearchHit] = []
        for chunk_id in candidates:
            document, metadata, distance = dense_by_id.get(chunk_id, ("", {}, None))
            if not document:
                row = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
                document = str((row.get("documents") or [""])[0] or "")
                metadata = dict((row.get("metadatas") or [{}])[0] or {})
                if not document:
                    continue
            content_tokens = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", document.casefold()))
            lexical = len(query_tokens.intersection(content_tokens)) / max(1, len(query_tokens))
            try:
                locator = json.loads(str(metadata.get("locator") or "{}"))
            except json.JSONDecodeError:
                locator = {}
            hits.append(
                ProjectSearchHit(
                    chunk_id=str(chunk_id),
                    project_id=str(metadata.get("project_id") or ""),
                    material_id=str(metadata.get("material_id") or ""),
                    source_id=str(metadata.get("source_id") or ""),
                    content=document,
                    locator=locator,
                    distance=distance,
                    lexical_score=lexical,
                )
            )
        def score(hit: ProjectSearchHit) -> float:
            dense_score = max(0.0, 1.0 - float(hit.distance or 1.0)) if hit.distance is not None else 0.0
            sparse_score = sparse_scores.get(hit.chunk_id, 0.0)
            # Keep lexical overlap as a fallback signal for hashing/legacy rows;
            # when sparse weights exist it contributes a smaller stabilizer.
            if query_sparse:
                return 0.50 * dense_score + 0.30 * sparse_score + 0.20 * hit.lexical_score
            return 0.70 * dense_score + 0.30 * hit.lexical_score
        return tuple(sorted(hits, key=lambda hit: (-score(hit), hit.chunk_id))[:top_k])


class QdrantProjectKnowledgeStore:
    """Qdrant-backed implementation of the project chunk index.

    The public contract intentionally matches ``ProjectKnowledgeStore`` so the
    material workflow and retrieval callers do not depend on a vector vendor.
    Sparse BGE-M3 weights remain in a small SQLite sidecar until Qdrant sparse
    vectors are enabled in a later migration.
    """

    name = "qdrant"

    def __init__(
        self,
        root: str | Path,
        *,
        embedding_backend: EmbeddingBackend | None = None,
        collection_variant: str = "default",
        embedding_batch_size: int = 32,
        url: str | None = None,
        api_key: str | None = None,
        collection_prefix: str = "smart_agent_project",
        timeout_seconds: float = 10.0,
    ) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is required for Qdrant project indexing; install the career extra."
            ) from exc
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedding_backend = embedding_backend or HashingEmbedding()
        self.collection_variant = str(collection_variant or "default").strip() or "default"
        self.embedding_batch_size = int(embedding_batch_size)
        if not 1 <= self.embedding_batch_size <= 512:
            raise ValueError("embedding_batch_size must be between 1 and 512")
        safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(collection_prefix or "smart_agent_project")).strip("_-")
        safe_prefix = (safe_prefix or "smart_agent_project")[:48]
        suffix = hashlib.sha256(
            f"{self.embedding_backend.name}:{self.collection_variant}".encode("utf-8")
        ).hexdigest()[:12]
        self.collection_name = f"{safe_prefix}_{suffix}"
        self._models = models
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_seconds must be a positive number") from exc
        if timeout <= 0:
            raise ValueError("timeout_seconds must be a positive number")
        if url:
            self.client = QdrantClient(
                url=str(url).strip(),
                api_key=str(api_key or "").strip() or None,
                timeout=timeout,
            )
        else:
            self.client = QdrantClient(path=str(self.root / "qdrant-local"))
        self.sparse_path = self.root / f"sparse-qdrant-{suffix}.sqlite3"
        self._lock = threading.RLock()
        self._sparse_db = sqlite3.connect(str(self.sparse_path), check_same_thread=False)
        self._sparse_db.execute(
            "CREATE TABLE IF NOT EXISTS project_sparse ("
            "chunk_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, material_id TEXT NOT NULL DEFAULT '', "
            "sparse_json TEXT NOT NULL DEFAULT '{}')"
        )
        columns = {str(row[1]) for row in self._sparse_db.execute("PRAGMA table_info(project_sparse)")}
        if "material_id" not in columns:
            self._sparse_db.execute("ALTER TABLE project_sparse ADD COLUMN material_id TEXT NOT NULL DEFAULT ''")
        self._sparse_db.commit()
        try:
            dimension = int(getattr(self.embedding_backend, "dimension", 0) or 0)
            if dimension <= 0:
                dimension = len(self._embed(["dimension probe"])[0][0])
        except Exception:
            self.close()
            raise
        try:
            exists = self.client.collection_exists(self.collection_name)
        except AttributeError:
            try:
                self.client.get_collection(self.collection_name)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status == 404 or "not found" in str(exc).casefold():
                    exists = False
                else:
                    self.close()
                    raise RuntimeError(
                        f"Qdrant service is unavailable at {url or 'local storage'}: {exc}"
                    ) from exc
            else:
                exists = True
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Qdrant service is unavailable at {url or 'local storage'}: {exc}"
            ) from exc
        if not exists:
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                )
            except Exception as exc:
                self.close()
                raise RuntimeError(
                    f"Qdrant collection initialization failed at {url or 'local storage'}: {exc}"
                ) from exc
        else:
            try:
                collection = self.client.get_collection(self.collection_name)
            except Exception as exc:
                self.close()
                raise RuntimeError(
                    f"Qdrant collection inspection failed at {url or 'local storage'}: {exc}"
                ) from exc
            configured = getattr(getattr(getattr(collection, "config", None), "params", None), "vectors", None)
            configured_size = getattr(configured, "size", None)
            if configured_size is None and isinstance(configured, dict):
                configured_size = configured.get("size")
            if configured_size is not None and int(configured_size) != dimension:
                self.close()
                raise RuntimeError(
                    f"Qdrant collection dimension mismatch: existing={configured_size}, expected={dimension}"
                )

    def close(self) -> None:
        """Release local SQLite/Qdrant resources, including Windows lock files."""
        database = getattr(self, "_sparse_db", None)
        if database is not None:
            try:
                database.close()
            except sqlite3.Error:
                pass
            self._sparse_db = None
        client = getattr(self, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"smart-agent-project:{chunk_id}"))

    @staticmethod
    def _sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values())) or 1.0
        right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values())) or 1.0
        dot = sum(float(value) * float(right.get(token, 0.0)) for token, value in left.items())
        return dot / (left_norm * right_norm)

    def _embed(self, texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
        method = getattr(self.embedding_backend, "embed_with_sparse", None)
        dense_rows: list[list[float]] = []
        sparse_rows: list[dict[str, float]] = []
        for start in range(0, len(texts), self.embedding_batch_size):
            batch = texts[start : start + self.embedding_batch_size]
            if callable(method):
                dense, sparse = method(batch)
                rows = list(sparse) if isinstance(sparse, list) else []
                if not isinstance(dense, list) or len(dense) != len(batch):
                    raise ValueError("Embedding backend returned an unexpected vector count")
                if len(rows) != len(batch):
                    rows = [{} for _ in batch]
                dense_rows.extend(dense)
                sparse_rows.extend(dict(row or {}) if isinstance(row, dict) else {} for row in rows)
            else:
                dense = self.embedding_backend.embed(batch)
                if not isinstance(dense, list) or len(dense) != len(batch):
                    raise ValueError("Embedding backend returned an unexpected vector count")
                dense_rows.extend(dense)
                sparse_rows.extend({} for _ in batch)
        return dense_rows, sparse_rows

    def upsert(self, chunks: list[ProjectChunk] | tuple[ProjectChunk, ...]) -> int:
        if not chunks:
            return 0
        with self._lock:
            embeddings, sparse_rows = self._embed([chunk.content for chunk in chunks])
            material_ids = {chunk.material_id for chunk in chunks}
            old_ids = self._matching_chunk_ids(material_ids=material_ids)
            points = [
                self._models.PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=embedding,
                    payload={**chunk.to_metadata(), "chunk_id": chunk.chunk_id, "content": chunk.content},
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]
            # Write new vectors first. If the remote write fails, the old index
            # remains available for retrieval.
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            new_ids = {chunk.chunk_id for chunk in chunks}
            self._delete_chunk_ids(old_ids - new_ids)
            self._sparse_db.executemany(
                "INSERT OR REPLACE INTO project_sparse(chunk_id, project_id, material_id, sparse_json) VALUES (?, ?, ?, ?)",
                [
                    (chunk.chunk_id, chunk.project_id, chunk.material_id, json.dumps(sparse, ensure_ascii=False, sort_keys=True))
                    for chunk, sparse in zip(chunks, sparse_rows, strict=True)
                ],
            )
            self._sparse_db.commit()
            return len(chunks)

    def _matching_chunk_ids(
        self,
        *,
        project_ids: set[str] | None = None,
        material_ids: set[str] | None = None,
    ) -> set[str]:
        clauses: list[str] = []
        params: list[str] = []
        if project_ids:
            clauses.append("project_id IN (" + ",".join("?" for _ in project_ids) + ")")
            params.extend(sorted(project_ids))
        if material_ids:
            clauses.append("material_id IN (" + ",".join("?" for _ in material_ids) + ")")
            params.extend(sorted(material_ids))
        if not clauses:
            return set()
        sql = "SELECT chunk_id FROM project_sparse WHERE " + " OR ".join(f"({clause})" for clause in clauses)
        ids = {str(row[0]) for row in self._sparse_db.execute(sql, params).fetchall()}
        scroll = getattr(self.client, "scroll", None)
        if callable(scroll):
            offset = None
            while True:
                page, offset = scroll(
                    collection_name=self.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=["chunk_id", "project_id", "material_id"],
                    with_vectors=False,
                )
                for point in page or ():
                    payload = dict(getattr(point, "payload", None) or {})
                    if (
                        (project_ids and str(payload.get("project_id") or "") in project_ids)
                        or (material_ids and str(payload.get("material_id") or "") in material_ids)
                    ):
                        chunk_id = str(payload.get("chunk_id") or "")
                        if chunk_id:
                            ids.add(chunk_id)
                if offset is None:
                    break
        return ids

    def _delete_chunk_ids(self, ids: set[str]) -> int:
        if not ids:
            return 0
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self._models.PointIdsList(points=[self._point_id(item) for item in sorted(ids)]),
            wait=True,
        )
        self._sparse_db.executemany("DELETE FROM project_sparse WHERE chunk_id = ?", [(item,) for item in ids])
        self._sparse_db.commit()
        return len(ids)

    def delete(
        self,
        *,
        project_ids: set[str] | None = None,
        material_ids: set[str] | None = None,
    ) -> int:
        with self._lock:
            return self._delete_chunk_ids(
                self._matching_chunk_ids(project_ids=project_ids, material_ids=material_ids)
            )

    def search(self, query: str, *, top_k: int = 5, project_id: str | None = None) -> tuple[ProjectSearchHit, ...]:
        query = " ".join(str(query or "").split())
        if not query:
            raise ValueError("Project knowledge query is required.")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50.")
        count = int(self.client.count(collection_name=self.collection_name, exact=True).count)
        if not count:
            return ()
        embeddings, sparse_rows = self._embed([query])
        query_sparse = sparse_rows[0] if sparse_rows else {}
        query_filter = None
        if project_id:
            query_filter = self._models.Filter(
                must=[self._models.FieldCondition(key="project_id", match=self._models.MatchValue(value=project_id))]
            )
        limit = min(max(top_k * 8, 50), count)
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=embeddings[0],
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            points = list(getattr(result, "points", result) or [])
        else:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=embeddings[0],
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        query_tokens = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", query.casefold()))
        sparse_scores: dict[str, float] = {}
        if query_sparse:
            sql = "SELECT chunk_id, sparse_json FROM project_sparse"
            params: tuple[Any, ...] = ()
            if project_id:
                sql += " WHERE project_id = ?"
                params = (project_id,)
            for chunk_id, raw in self._sparse_db.execute(sql, params):
                try:
                    row = json.loads(str(raw or "{}"))
                    if isinstance(row, dict):
                        sparse_scores[str(chunk_id)] = self._sparse_cosine(query_sparse, row)
                except (TypeError, json.JSONDecodeError):
                    continue
            sparse_scores = dict(sorted(sparse_scores.items(), key=lambda item: -item[1])[:limit])
            dense_chunk_ids = {
                str((point.payload or {}).get("chunk_id") or "")
                for point in points
                if getattr(point, "payload", None)
            }
            missing_ids = [chunk_id for chunk_id in sparse_scores if chunk_id not in dense_chunk_ids]
            retrieve = getattr(self.client, "retrieve", None)
            if missing_ids and callable(retrieve):
                records = retrieve(
                    collection_name=self.collection_name,
                    ids=[self._point_id(chunk_id) for chunk_id in missing_ids],
                    with_payload=True,
                    with_vectors=True,
                )
                query_vector = embeddings[0]
                query_norm = math.sqrt(sum(value * value for value in query_vector)) or 1.0
                for record in records or ():
                    vector = getattr(record, "vector", None)
                    if not isinstance(vector, list) or len(vector) != len(query_vector):
                        continue
                    vector_norm = math.sqrt(sum(value * value for value in vector)) or 1.0
                    similarity = sum(left * right for left, right in zip(query_vector, vector)) / (query_norm * vector_norm)
                    points.append(
                        SimpleNamespace(
                            payload=getattr(record, "payload", None) or {},
                            score=similarity,
                        )
                    )
        hits: list[ProjectSearchHit] = []
        for point in points:
            payload = dict(point.payload or {})
            content = str(payload.get("content") or "")
            chunk_id = str(payload.get("chunk_id") or "")
            tokens = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", content.casefold()))
            try:
                locator = json.loads(str(payload.get("locator") or "{}"))
            except json.JSONDecodeError:
                locator = {}
            hits.append(
                ProjectSearchHit(
                    chunk_id=chunk_id,
                    project_id=str(payload.get("project_id") or ""),
                    material_id=str(payload.get("material_id") or ""),
                    source_id=str(payload.get("source_id") or ""),
                    content=content,
                    locator=locator,
                    # Chroma exposes cosine distance while Qdrant returns cosine
                    # similarity; normalize to the shared ProjectSearchHit contract.
                    distance=(1.0 - float(point.score)) if point.score is not None else None,
                    lexical_score=len(query_tokens & tokens) / max(1, len(query_tokens)),
                )
            )
        def score(hit: ProjectSearchHit) -> float:
            # ``ProjectSearchHit.distance`` follows Chroma's contract where
            # smaller values are better. Qdrant's cosine result is converted
            # to that distance above, so restore similarity before blending
            # it with lexical/sparse scores.
            dense = max(0.0, 1.0 - float(hit.distance)) if hit.distance is not None else 0.0
            sparse = sparse_scores.get(hit.chunk_id, 0.0)
            if query_sparse:
                return 0.50 * dense + 0.30 * sparse + 0.20 * hit.lexical_score
            return 0.70 * dense + 0.30 * hit.lexical_score
        return tuple(sorted(hits, key=lambda hit: (-score(hit), hit.chunk_id))[:top_k])


@dataclass(frozen=True, slots=True)
class MaterialImportResult:
    manifest: MaterialManifest
    source_reference: SourceReference
    record: PersonalEvidence | ResumeSource | None
    indexed_chunks: int = 0
    extracted_text: str = ""

    def to_dict(self, *, include_extracted_text: bool = True) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "source_reference": {
                "source_id": self.source_reference.source_id,
                "source_type": self.source_reference.source_type.value,
                "locator": self.source_reference.locator,
                "content_hash": self.source_reference.content_hash,
                "excerpt": self.source_reference.excerpt,
            },
            "record": self.record.to_dict() if self.record is not None else None,
            "indexed_chunks": self.indexed_chunks,
            "extracted_text": self.extracted_text if include_extracted_text else None,
        }


@dataclass(frozen=True, slots=True)
class PersonalMaterialWorkflowResult:
    import_result: MaterialImportResult
    candidate_facts: tuple[CandidateFact, ...]
    structured_path: str
    status: str

    @property
    def manifest(self) -> MaterialManifest:
        return self.import_result.manifest

    def to_dict(self, *, include_extracted_text: bool = True) -> dict[str, Any]:
        return {
            "material_id": self.manifest.material_id,
            "category": self.manifest.category.value,
            "label": self.manifest.label,
            "status": self.status,
            "structured_path": self.structured_path,
            "import_result": self.import_result.to_dict(include_extracted_text=include_extracted_text),
            "candidate_facts": [fact.to_dict() for fact in self.candidate_facts],
        }


class PersonalMaterialWorkflow:
    """Deterministic material intake; semantic extraction is a replaceable capability."""

    def __init__(
        self,
        material_store: MaterialStore,
        candidate_fact_store: CandidateFactStore,
        *,
        career_store: CareerStore | None = None,
        project_store: ProjectIndexBackend | None = None,
        fact_extractor: CareerFactExtractor | None = None,
        structured_dir: str | Path | None = None,
    ) -> None:
        self.material_store = material_store
        self.candidate_fact_store = candidate_fact_store
        self.career_store = career_store
        self.project_store = project_store
        self.fact_extractor = fact_extractor or HeuristicCareerFactExtractor()
        self.structured_dir = Path(structured_dir).expanduser().resolve() if structured_dir is not None else (
            self.material_store.root / "structured"
        )
        self.structured_dir.mkdir(parents=True, exist_ok=True)

    def organize(
        self,
        *,
        category: MaterialCategory | str,
        label: str,
        text: str | None = None,
        source_path: str | Path | None = None,
        project_id: str = "",
        max_candidate_facts: int = 12,
        resume_section: ResumeSectionBlock | None = None,
    ) -> PersonalMaterialWorkflowResult:
        if max_candidate_facts < 0 or max_candidate_facts > 100:
            raise ValueError("max_candidate_facts must be between 0 and 100.")
        material_category = MaterialCategory(category)
        imported = self.material_store.ingest(
            category=material_category,
            label=label,
            text=text,
            source_path=source_path,
            project_id=project_id,
            facts=(),
            career_store=self.career_store,
            project_store=self.project_store,
            resume_section=resume_section,
        )
        candidate_facts: tuple[CandidateFact, ...] = ()
        if material_category is not MaterialCategory.RESUME and max_candidate_facts:
            candidate_facts = self.fact_extractor.extract(
                imported.manifest,
                imported.extracted_text,
                max_facts=max_candidate_facts,
            )
            candidate_facts = self.candidate_fact_store.upsert(candidate_facts)
        status = "source_only" if material_category is MaterialCategory.RESUME else "awaiting_user_confirmation"
        manifest = imported.manifest
        if manifest.fact_status != status:
            manifest = replace(manifest, fact_status=status)
            self.material_store._upsert_manifest(manifest)
            imported = replace(imported, manifest=manifest)
        structured_path = self.structured_dir / f"{manifest.material_id}.json"
        payload = self._structured_payload(
            imported=imported,
            candidate_facts=candidate_facts,
            status=status,
            structured_path=structured_path,
        )
        _atomic_write(
            structured_path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        return PersonalMaterialWorkflowResult(
            import_result=imported,
            candidate_facts=candidate_facts,
            structured_path=str(structured_path),
            status=status,
        )

    def _structured_payload(
        self,
        *,
        imported: MaterialImportResult,
        candidate_facts: tuple[CandidateFact, ...],
        status: str,
        structured_path: Path,
    ) -> dict[str, Any]:
        manifest = imported.manifest
        return {
            "schema_version": "1.0",
            "material_id": manifest.material_id,
            "category": manifest.category.value,
            "label": manifest.label,
            "status": status,
            "original_file": {
                "name": manifest.original_name,
                "sha256": manifest.content_hash,
                "media_type": manifest.media_type,
                "raw_path": manifest.raw_path,
                "source_path": manifest.source_path,
            },
            "extraction": {
                "parser": manifest.parser,
                "parser_confidence": manifest.parser_confidence,
                "text_path": manifest.extracted_text_path,
                "json_path": str(structured_path),
            },
            "source_reference": _source_ref_to_dict(imported.source_reference),
            "candidate_facts": [fact.to_dict() for fact in candidate_facts],
            "indexed_chunks": imported.indexed_chunks,
        }


class MaterialStore:
    """Raw material files, manifests, extraction cache, and Career evidence bridge."""

    @staticmethod
    def _resume_source_matches(current: ResumeSource, candidate: ResumeSource) -> bool:
        """Compare stable resume data while ignoring import-time timestamps."""

        def stable_refs(record: ResumeSource) -> tuple[tuple[Any, ...], ...]:
            return tuple(
                (
                    ref.source_id,
                    ref.source_type,
                    ref.locator,
                    ref.content_hash,
                    ref.excerpt,
                )
                for ref in record.source_refs
            )

        return (
            current.media_type == candidate.media_type
            and current.normalized_text == candidate.normalized_text
            and current.sections == candidate.sections
            and current.page_count == candidate.page_count
            and current.parser_confidence == candidate.parser_confidence
            and current.read_only == candidate.read_only
            and stable_refs(current) == stable_refs(candidate)
        )

    def __init__(
        self,
        root: str | Path,
        *,
        workspace: str | Path | None = None,
        text_extractor: MaterialTextExtractor | None = None,
        project_chunker: ProjectChunker | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.workspace = Path(workspace or Path.cwd()).expanduser().resolve()
        self.raw_dir = self.root / "raw"
        self.text_dir = self.root / "text"
        self.manifest_path = self.root / "manifest.json"
        self.text_extractor = text_extractor or DefaultMaterialTextExtractor()
        self.project_chunker = project_chunker or LineProjectChunker()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)

    def list_manifests(
        self,
        category: MaterialCategory | str | None = None,
        *,
        include_deleted: bool = False,
    ) -> tuple[MaterialManifest, ...]:
        values = self._load_manifests()
        if not include_deleted:
            values = [value for value in values if value.active]
        if category is not None:
            selected = MaterialCategory(category)
            values = [value for value in values if value.category is selected]
        return tuple(values)

    def read_text(self, material_id: str, *, max_chars: int = 50_000) -> str:
        manifest = next(
            (item for item in self._load_manifests() if item.material_id == str(material_id).strip() and item.active),
            None,
        )
        if manifest is None:
            raise KeyError(f"Unknown material_id: {material_id}")
        if max_chars < 1 or max_chars > 200_000:
            raise ValueError("max_chars must be between 1 and 200000.")
        return Path(manifest.extracted_text_path).read_text(encoding="utf-8")[:max_chars]

    def ingest(
        self,
        *,
        category: MaterialCategory | str,
        label: str,
        text: str | None = None,
        source_path: str | Path | None = None,
        project_id: str = "",
        facts: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        career_store: CareerStore | None = None,
        project_store: ProjectIndexBackend | None = None,
        resume_section: ResumeSectionBlock | None = None,
    ) -> MaterialImportResult:
        material_category = MaterialCategory(category)
        if (text is None) == (source_path is None):
            raise ValueError("Provide exactly one of text or source_path.")
        if not str(label or "").strip():
            raise ValueError("label is required.")
        source_path_value = ""
        original_name = f"{_slug(label)}.txt"
        media_type = "text/plain"
        parser = "plain_text"
        parser_confidence = 1.0
        if source_path is not None:
            path = Path(source_path).expanduser().resolve()
            if not path.is_relative_to(self.workspace):
                raise ValueError("source_path must stay inside the active workspace.")
            source_path_value = str(path)
            original_name = path.name
            text, media_type, parser, parser_confidence = self.text_extractor.extract(path)
        normalized_text = str(text or "").replace("\x00", "").strip()
        if not normalized_text:
            raise ValueError("Material text is empty after extraction.")
        content_hash = _digest(normalized_text)
        if resume_section is not None:
            expected_category = _RESUME_SECTION_CATEGORIES[resume_section.section_id]
            if material_category is not expected_category:
                raise ValueError("Derived resume material category does not match its source section.")
            if normalized_text != resume_section.text:
                raise ValueError("Derived resume material text does not match its source section.")
            identity_payload = (
                f"{resume_section.source_resume_id}@{resume_section.source_revision}:"
                f"{resume_section.section_id}:{content_hash}"
            )
            material_identity_hash = _digest(identity_payload)
        else:
            material_identity_hash = content_hash
        material_id = f"material-{material_identity_hash[:20]}"
        raw_path = self.raw_dir / f"{material_id}{Path(original_name).suffix.lower() or '.txt'}"
        extracted_path = self.text_dir / f"{material_id}.txt"
        if source_path is not None and Path(source_path).is_file():
            shutil.copyfile(Path(source_path), raw_path)
        else:
            _atomic_write(raw_path, normalized_text)
        _atomic_write(extracted_path, normalized_text)
        manifest = MaterialManifest(
            material_id=material_id,
            category=material_category,
            label=str(label).strip(),
            original_name=original_name,
            media_type=media_type,
            content_hash=content_hash,
            source_path=source_path_value,
            raw_path=str(raw_path),
            extracted_text_path=str(extracted_path),
            parser=parser,
            parser_confidence=parser_confidence,
            project_id=str(project_id or _slug(label) if material_category is MaterialCategory.PROJECT else "").strip(),
            fact_status="awaiting_fact_confirmation" if facts else "source_only",
            chunker=(getattr(self.project_chunker, "name", "custom") if material_category is MaterialCategory.PROJECT else ""),
            index_backend=(
                str(getattr(project_store, "name", "custom"))
                if material_category is MaterialCategory.PROJECT and project_store is not None
                else ""
            ),
            embedding_backend=(
                str(getattr(getattr(project_store, "embedding_backend", None), "name", ""))
                if material_category is MaterialCategory.PROJECT and project_store is not None
                else ""
            ),
            index_variant=(
                str(getattr(project_store, "collection_variant", ""))
                if material_category is MaterialCategory.PROJECT and project_store is not None
                else ""
            ),
        )
        existing_manifest = next(
            (item for item in self._load_manifests() if item.material_id == manifest.material_id),
            None,
        )
        if existing_manifest is not None:
            manifest = replace(
                manifest,
                revision=existing_manifest.revision + 1,
                active=True,
                deleted_at=0.0,
            )
        self._upsert_manifest(manifest)
        source_type = (
            SourceType.REPOSITORY
            if material_category is MaterialCategory.PROJECT and source_path is not None and Path(source_path).is_dir()
            else SourceType.RESUME
            if material_category is MaterialCategory.RESUME
            else SourceType.FILE
        )
        locator: dict[str, Any] = {"material_id": material_id, "path": str(extracted_path)}
        if resume_section is not None:
            locator["resume_source"] = {
                "record_id": resume_section.source_resume_id,
                "revision": resume_section.source_revision,
                "section_id": resume_section.section_id,
                "section_content_hash": resume_section.content_hash,
                "locator": resume_section.locator,
            }
        source = SourceReference.create(
            source_id=material_id,
            source_type=source_type,
            locator=locator,
            raw_content=normalized_text,
            excerpt=normalized_text[:500],
        )
        record: PersonalEvidence | ResumeSource | None = None
        if material_category is MaterialCategory.RESUME:
            record = ResumeSource(
                resume_source_id=f"resume-source-{content_hash[:20]}",
                original_name=original_name,
                media_type=media_type,
                normalized_text=normalized_text,
                page_count=_page_count(source_path),
                parser_confidence=parser_confidence,
                source_refs=(source,),
            )
            if career_store is not None:
                try:
                    current = career_store.get(record.resume_source_id, expected_type=ResumeSource)
                except RecordNotFoundError:
                    career_store.put(record)
                else:
                    if not self._resume_source_matches(current, record):
                        raise RevisionConflictError("A resume source with this content identity already exists with different data.")
                    record = current
        elif facts:
            evidence_facts = tuple(
                EvidenceFact.create(
                    str(fact.get("field_name") or "summary"),
                    fact.get("value"),
                    source_refs=(source,),
                    confidence=float(fact.get("confidence", 0.0)),
                    status=FactStatus.UNVERIFIED,
                    actor="career_material_importer",
                )
                for fact in facts
            )
            if not evidence_facts:
                raise ValueError("facts must contain at least one fact object when supplied.")
            existing = (
                next(
                    (
                        item
                        for item in career_store.list_latest(PersonalEvidence)
                        if item.evidence_type is _EVIDENCE_TYPES[material_category]
                        and any(ref.source_id == material_id for ref in item.source_refs)
                    ),
                    None,
                )
                if career_store is not None
                else None
            )
            record = existing or create_personal_evidence(
                _EVIDENCE_TYPES[material_category],
                label=str(label).strip(),
                facts=evidence_facts,
                source_refs=(source,),
                actor="career_material_importer",
            )
            if career_store is not None and existing is None:
                career_store.put(record)
        indexed_chunks = 0
        if material_category is MaterialCategory.PROJECT:
            if project_store is None:
                raise ValueError("Project imports require a ProjectKnowledgeStore.")
            chunks = self.project_chunker.chunk(
                normalized_text,
                project_id=manifest.project_id,
                material_id=material_id,
                source_id=material_id,
            )
            indexed_chunks = project_store.upsert(chunks)
        return MaterialImportResult(manifest, source, record, indexed_chunks, normalized_text)

    def delete(
        self,
        material_id: str,
        *,
        career_store: CareerStore | None = None,
        project_store: ProjectIndexBackend | None = None,
        reason: str = "Material deleted by user",
    ) -> MaterialManifest:
        identity = str(material_id or "").strip()
        if not identity:
            raise ValueError("material_id is required")
        current = next((item for item in self._load_manifests() if item.material_id == identity), None)
        if current is None:
            raise KeyError(f"Unknown material_id: {identity}")
        if not current.active:
            return current
        deleted = replace(current, revision=current.revision + 1, active=False, deleted_at=time.time())
        self._upsert_manifest(deleted)
        if project_store is not None and current.category is MaterialCategory.PROJECT:
            project_store.delete(material_ids={identity})
        if career_store is not None:
            for evidence in career_store.list_latest(PersonalEvidence):
                if not any(ref.source_id == identity for ref in evidence.source_refs):
                    continue
                for fact in evidence.facts:
                    if fact.status is not FactStatus.INVALIDATED:
                        career_store.invalidate_fact(
                            evidence.evidence_id,
                            fact.fact_id,
                            actor="material_deleter",
                            reason=reason,
                        )
            for description in career_store.list_latest(ProjectDescriptionDocument):
                if identity not in description.material_ids or description.status is ProjectDescriptionStatus.INVALIDATED:
                    continue
                career_store.revise_record(
                    description.project_description_id,
                    actor="material_deleter",
                    reason=reason,
                    changes={"status": ProjectDescriptionStatus.INVALIDATED},
                )
        # Only files copied into this store are removed.  ``source_path`` is
        # an audit pointer and may refer to a user's original file elsewhere.
        for persisted_path in (current.raw_path, current.extracted_text_path):
            candidate = Path(persisted_path).expanduser().resolve() if persisted_path else None
            if candidate is None or not candidate.is_relative_to(self.root) or candidate == self.root:
                continue
            try:
                if candidate.is_file():
                    candidate.unlink()
            except FileNotFoundError:
                pass
        return deleted

    def _load_manifests(self) -> list[MaterialManifest]:
        if not self.manifest_path.exists():
            return []
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot load material manifest: {exc}") from exc
        return [MaterialManifest.from_dict(item) for item in payload.get("materials", [])]

    def _upsert_manifest(self, manifest: MaterialManifest) -> None:
        entries = {item.material_id: item for item in self._load_manifests()}
        entries[manifest.material_id] = manifest
        _atomic_write(
            self.manifest_path,
            json.dumps(
                {"schema_version": "1.0", "materials": [item.to_dict() for item in entries.values()]},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )


def extract_material_text(path: Path) -> tuple[str, str, str, float]:
    """Extract text from common career files without making OCR part of core import."""

    if path.is_dir():
        blocks: list[str] = []
        for child in sorted(path.rglob("*")):
            if not child.is_file() or any(part in _SKIP_DIRS for part in child.parts):
                continue
            if child.suffix.casefold() not in _TEXT_SUFFIXES:
                continue
            try:
                content = child.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            blocks.append(f"# FILE: {child.relative_to(path)}\n{content}")
        if not blocks:
            raise ValueError("Project directory contains no supported text or code files.")
        return "\n\n".join(blocks), "inode/directory", "repository_text_walk", 0.98
    suffix = path.suffix.casefold()
    if suffix in _TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace"), "text/plain", "plain_text", 1.0
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("pypdf is required to import PDF materials.") from exc
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages), "application/pdf", "pypdf", 0.95
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("python-docx is required to import DOCX materials.") from exc
        document = Document(str(path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return text, "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "python-docx", 0.95
    if suffix in {".html", ".htm"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw))), "text/html", "html_text", 0.9
    raise ValueError(f"Unsupported material type: {suffix or '<none>'}. OCR/image import is not in this core path.")


def _page_count(source_path: str | Path | None) -> int:
    if source_path is None or Path(source_path).suffix.casefold() != ".pdf":
        return 0
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(source_path)).pages)
    except Exception:  # Parser confidence already records that this is best effort.
        return 0


def chunk_project_text(
    text: str,
    *,
    project_id: str,
    material_id: str,
    source_id: str,
    max_chars: int = 1400,
    overlap_chars: int = 160,
) -> tuple[ProjectChunk, ...]:
    if max_chars < 300 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("Invalid project chunk limits.")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    chunks: list[ProjectChunk] = []
    current: list[str] = []
    current_start = 1
    current_length = 0
    for line_number, line in enumerate(lines, start=1):
        line_length = len(line) + 1
        if current and current_length + line_length > max_chars:
            content = "\n".join(current).strip()
            if content:
                index = len(chunks)
                chunks.append(
                    ProjectChunk(
                        chunk_id=f"chunk-{_digest(f'{source_id}:{index}:{content}')[:24]}",
                        project_id=project_id,
                        material_id=material_id,
                        source_id=source_id,
                        content=content,
                        locator={"line_start": current_start, "line_end": line_number - 1},
                        content_hash=_digest(content),
                    )
                )
            overlap: list[str] = []
            overlap_length = 0
            for previous in reversed(current):
                if overlap_length + len(previous) + 1 > overlap_chars:
                    break
                overlap.insert(0, previous)
                overlap_length += len(previous) + 1
            current = overlap
            current_start = max(1, line_number - len(overlap))
            current_length = sum(len(item) + 1 for item in current)
        current.append(line)
        current_length += line_length
    if current:
        content = "\n".join(current).strip()
        if content:
            index = len(chunks)
            chunks.append(
                ProjectChunk(
                    chunk_id=f"chunk-{_digest(f'{source_id}:{index}:{content}')[:24]}",
                    project_id=project_id,
                    material_id=material_id,
                    source_id=source_id,
                    content=content,
                    locator={"line_start": current_start, "line_end": len(lines)},
                    content_hash=_digest(content),
                )
            )
    return tuple(chunks)
