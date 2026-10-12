"""Versioned Career-domain data contracts with field-level provenance."""

from __future__ import annotations

import hashlib
import json
import re
import time
import types
import uuid
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from enum import Enum
from typing import Any, ClassVar, Union, get_args, get_origin, get_type_hints

CAREER_SCHEMA_VERSION = "1.0"
_UNSET = object()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    return text


def _clone_json(value: Any, field_name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable.") from exc


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _serialize(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, tuple | list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(
        _serialize(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataclass_hash(value: Any) -> str:
    payload = {
        item.name: _serialize(getattr(value, item.name))
        for item in fields(value)
        if item.name != "content_hash"
    }
    return _sha256_payload(payload)


def _unique_texts(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    normalized = tuple(_required_text(value, field_name) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} cannot contain duplicates.")
    return normalized


def _unique_refs(values: tuple[VersionRef, ...], field_name: str) -> tuple[VersionRef, ...]:
    normalized = tuple(values)
    keys = [(ref.record_id, ref.revision) for ref in normalized]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} cannot contain duplicates.")
    return normalized


class SourceType(str, Enum):
    FILE = "file"
    URL = "url"
    USER_INPUT = "user_input"
    RESUME = "resume"
    PORTFOLIO = "portfolio"
    REPOSITORY = "repository"
    GENERATED = "generated"


class EvidenceType(str, Enum):
    EDUCATION = "education"
    AWARD = "award"
    INTERNSHIP = "internship"
    PROJECT = "project"
    RESEARCH = "research"
    SKILLS = "skills"


RESUME_SECTION_ORDER = (
    "education",
    "internships",
    "projects",
    "research",
    "awards",
    "skills",
)
RESUME_SECTION_LABELS = {
    "internships": "\u5b9e\u4e60\u7ecf\u5386",
    "projects": "\u9879\u76ee\u7ecf\u5386",
    "education": "\u6559\u80b2\u80cc\u666f",
    "skills": "\u4e2a\u4eba\u6280\u80fd/\u5176\u4ed6",
    "research": "\u79d1\u7814\u6210\u679c",
    "awards": "\u4e2a\u4eba\u5956\u9879",
}
_RESUME_SECTION_ALIASES = {
    "internship": "internships",
    "internships": "internships",
    "experience": "internships",
    "experiences": "internships",
    "work": "internships",
    "employment": "internships",
    "\u5b9e\u4e60": "internships",
    "\u5b9e\u4e60\u7ecf\u5386": "internships",
    "\u5de5\u4f5c\u7ecf\u5386": "internships",
    "project": "projects",
    "projects": "projects",
    "\u9879\u76ee": "projects",
    "\u9879\u76ee\u7ecf\u5386": "projects",
    "\u9879\u76ee\u7ecf\u9a8c": "projects",
    "education": "education",
    "educations": "education",
    "\u6559\u80b2": "education",
    "\u6559\u80b2\u80cc\u666f": "education",
    "\u6559\u80b2\u7ecf\u5386": "education",
    "\u6559\u80b2\u80cc\u666f\u7ecf\u5386": "education",
    "skill": "skills",
    "skills": "skills",
    "other": "skills",
    "others": "skills",
    "skills/other": "skills",
    "skills_other": "skills",
    "\u6280\u80fd": "skills",
    "\u6280\u80fd\u8bc1\u4e66": "skills",
    "\u4e13\u4e1a\u6280\u80fd": "skills",
    "\u5176\u4ed6": "skills",
    "\u6280\u80fd/\u5176\u4ed6": "skills",
    "\u4e2a\u4eba\u6280\u80fd/\u5176\u4ed6": "skills",
    "research": "research",
    "researches": "research",
    "publication": "research",
    "publications": "research",
    "\u79d1\u7814": "research",
    "\u79d1\u7814\u6210\u679c": "research",
    "\u79d1\u7814\u7ecf\u5386": "research",
    "\u8bba\u6587": "research",
    "\u7814\u7a76\u7ecf\u5386": "research",
    "award": "awards",
    "awards": "awards",
    "honor": "awards",
    "honors": "awards",
    "honour": "awards",
    "honours": "awards",
    "\u83b7\u5956": "awards",
    "\u83b7\u5956\u7ecf\u5386": "awards",
    "\u5956\u9879": "awards",
    "\u4e2a\u4eba\u5956\u9879": "awards",
}


def normalize_resume_section(value: Any) -> str:
    """Map all resume content into one of the six stable section IDs."""
    normalized = str(value or "").strip().casefold().replace(" ", "_").replace("-", "_")
    return _RESUME_SECTION_ALIASES.get(normalized, "skills")


def normalize_resume_sections(value: Any) -> dict[str, tuple[str, ...]]:
    """Return all six section slots and route legacy or unknown sections to skills."""
    sections: dict[str, list[str]] = {section: [] for section in RESUME_SECTION_ORDER}
    if isinstance(value, dict):
        for name, claim_ids in value.items():
            target = normalize_resume_section(name)
            sections[target].extend(str(claim_id) for claim_id in (claim_ids or ()))
    return {section: tuple(sections[section]) for section in RESUME_SECTION_ORDER}


@dataclass(frozen=True, slots=True, kw_only=True)
class ResumeSectionBlock:
    """One deterministic resume section pinned to an exact source revision."""

    section_id: str
    text: str
    source_resume_id: str
    source_revision: int
    locator: dict[str, Any]
    content_hash: str = ""

    def __post_init__(self) -> None:
        section_id = str(self.section_id or "").strip()
        if section_id not in RESUME_SECTION_ORDER:
            raise ValueError("ResumeSectionBlock section_id must be one of the six fixed sections.")
        object.__setattr__(self, "section_id", section_id)
        object.__setattr__(self, "text", str(self.text or "").strip())
        object.__setattr__(self, "source_resume_id", _required_text(self.source_resume_id, "source_resume_id"))
        if self.source_revision < 1:
            raise ValueError("ResumeSectionBlock source_revision must be at least 1.")
        locator = _clone_json(self.locator, "locator")
        ranges = locator.get("line_ranges")
        if not isinstance(ranges, list):
            raise ValueError("ResumeSectionBlock locator requires line_ranges.")
        previous_end = 0
        for line_range in ranges:
            if (
                not isinstance(line_range, list)
                or len(line_range) != 2
                or not all(isinstance(item, int) for item in line_range)
                or line_range[0] < 1
                or line_range[1] < line_range[0]
                or line_range[0] <= previous_end
            ):
                raise ValueError("ResumeSectionBlock line_ranges must be ordered positive inclusive ranges.")
            previous_end = line_range[1]
        object.__setattr__(self, "locator", locator)
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        supplied = str(self.content_hash or "").strip().lower()
        if supplied and supplied != digest:
            raise ValueError("ResumeSectionBlock content_hash does not match its text.")
        object.__setattr__(self, "content_hash", digest)


def _resume_heading(line: str) -> tuple[str, str] | None:
    cleaned = re.sub(r"^[#*+\-\u2022\u00b7\s]+", "", line).strip()
    if not cleaned:
        return None
    parts = re.split(r"[\uff1a:\u4e28|]", cleaned, maxsplit=1)
    heading = re.sub(r"\s+", "", parts[0]).casefold()
    section_id = _RESUME_SECTION_ALIASES.get(heading)
    if section_id is None:
        return None
    suffix = parts[1].strip() if len(parts) == 2 else ""
    return section_id, suffix


def _looks_like_unknown_resume_heading(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return False
    if stripped.startswith("#") or stripped.endswith(("：", ":")):
        return True
    cleaned = re.sub(r"^[*+\-\u2022\u00b7\s]+", "", stripped).strip()
    heading_endings = ("经历", "背景", "成果", "奖项", "技能", "证书", "简介", "总结", "评价", "荣誉", "爱好")
    return len(cleaned) <= 20 and cleaned.endswith(heading_endings)


def _compress_line_ranges(line_numbers: list[int]) -> list[list[int]]:
    if not line_numbers:
        return []
    ranges: list[list[int]] = []
    start = end = line_numbers[0]
    for line_number in line_numbers[1:]:
        if line_number == end + 1:
            end = line_number
            continue
        ranges.append([start, end])
        start = end = line_number
    ranges.append([start, end])
    return ranges


def split_resume_source_sections(
    text: str,
    *,
    resume_source_id: str,
    source_revision: int,
) -> tuple[ResumeSectionBlock, ...]:
    """Create all six stable slots while retaining source line locations."""

    source_id = _required_text(resume_source_id, "resume_source_id")
    if source_revision < 1:
        raise ValueError("source_revision must be at least 1.")
    section_lines: dict[str, list[str]] = {section: [] for section in RESUME_SECTION_ORDER}
    section_line_numbers: dict[str, list[int]] = {section: [] for section in RESUME_SECTION_ORDER}
    current_section: str | None = None
    for line_number, raw_line in enumerate(str(text or "").splitlines(), start=1):
        heading = _resume_heading(raw_line)
        if heading is not None:
            current_section, suffix = heading
            if suffix:
                section_lines[current_section].append(suffix)
                section_line_numbers[current_section].append(line_number)
            continue
        cleaned = raw_line.strip()
        if not cleaned:
            continue
        if _looks_like_unknown_resume_heading(raw_line):
            current_section = "skills"
        target = current_section or "skills"
        section_lines[target].append(cleaned)
        section_line_numbers[target].append(line_number)
    return tuple(
        ResumeSectionBlock(
            section_id=section_id,
            text="\n".join(section_lines[section_id]),
            source_resume_id=source_id,
            source_revision=source_revision,
            locator={"line_ranges": _compress_line_ranges(section_line_numbers[section_id])},
        )
        for section_id in RESUME_SECTION_ORDER
    )


class FactStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    USER_CONFIRMED = "user_confirmed"
    NEEDS_REVERIFICATION = "needs_reverification"
    INVALIDATED = "invalidated"


class ChangeAction(str, Enum):
    IMPORTED = "imported"
    MODIFIED = "modified"
    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"
    REVERIFIED = "reverified"


class RequirementCategory(str, Enum):
    RESPONSIBILITY = "responsibility"
    HARD_SKILL = "hard_skill"
    SOFT_SKILL = "soft_skill"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    PREFERENCE = "preference"


class SupportLevel(str, Enum):
    DIRECT = "direct"
    PARTIAL = "partial"
    INDIRECT = "indirect"
    UNSUPPORTED = "unsupported"


class ClaimStatus(str, Enum):
    DRAFT = "draft"
    SUPPORTED = "supported"
    NEEDS_REVERIFICATION = "needs_reverification"
    INVALIDATED = "invalidated"


class ResumeDocumentStatus(str, Enum):
    USER_PROVIDED = "user_provided"
    DRAFT = "draft"
    NEEDS_REVERIFICATION = "needs_reverification"
    VERIFIED = "verified"
    INVALIDATED = "invalidated"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    ISSUES_FOUND = "issues_found"
    NEEDS_USER_CONFIRMATION = "needs_user_confirmation"


class ProjectDescriptionStatus(str, Enum):
    DRAFT = "draft"
    NEEDS_USER_CONFIRMATION = "needs_user_confirmation"
    USER_CONFIRMED = "user_confirmed"
    INVALIDATED = "invalidated"


class ResumeRunStatus(str, Enum):
    CREATED = "created"
    MATCHED = "matched"
    PLANNED = "planned"
    GENERATED = "generated"
    VERIFIED = "verified"
    RENDERED = "rendered"
    STALE = "stale"
    DELETED = "deleted"


class RequirementMatchCacheStatus(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ResumeHeader:
    """User-supplied identity and contact fields rendered above resume claims."""

    full_name: str
    headline: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    links: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "full_name", _required_text(self.full_name, "full_name"))
        for field_name in ("headline", "email", "phone", "location"):
            value = str(getattr(self, field_name) or "").strip()
            if "\n" in value or "\r" in value:
                raise ValueError(f"ResumeHeader {field_name} must be a single line.")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "links", _unique_texts(self.links, "links"))


@dataclass(frozen=True, slots=True)
class VersionRef:
    """Pin a relationship to an exact immutable record revision."""

    record_id: str
    revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _required_text(self.record_id, "record_id"))
        if self.revision < 1:
            raise ValueError("VersionRef revision must be at least 1.")


@dataclass(frozen=True, slots=True)
class SourceReference:
    """W3C-PROV-style pointer to the original material and exact location."""

    source_id: str
    source_type: SourceType
    locator: dict[str, Any]
    content_hash: str
    excerpt: str = ""
    captured_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _required_text(self.source_id, "source_id"))
        object.__setattr__(self, "source_type", SourceType(self.source_type))
        object.__setattr__(self, "locator", _clone_json(self.locator, "locator"))
        digest = str(self.content_hash or "").strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("SourceReference content_hash must be a SHA-256 hex digest.")
        object.__setattr__(self, "content_hash", digest)

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        source_type: SourceType | str,
        locator: dict[str, Any],
        raw_content: str | bytes,
        excerpt: str = "",
    ) -> "SourceReference":
        raw = raw_content.encode("utf-8") if isinstance(raw_content, str) else raw_content
        return cls(
            source_id=source_id,
            source_type=SourceType(source_type),
            locator=locator,
            content_hash=hashlib.sha256(raw).hexdigest(),
            excerpt=excerpt,
        )


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    """An immutable transition in a fact or record history."""

    event_id: str
    record_id: str
    action: ChangeAction
    from_revision: int
    to_revision: int
    actor: str
    reason: str
    changed_fields: tuple[str, ...]
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _required_text(self.event_id, "event_id"))
        object.__setattr__(self, "record_id", _required_text(self.record_id, "record_id"))
        object.__setattr__(self, "action", ChangeAction(self.action))
        object.__setattr__(self, "actor", _required_text(self.actor, "actor"))
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        object.__setattr__(self, "changed_fields", tuple(self.changed_fields))
        if self.from_revision < 0 or self.to_revision != self.from_revision + 1:
            raise ValueError("ChangeEvent revisions must describe one sequential transition.")

    @classmethod
    def create(
        cls,
        *,
        record_id: str,
        action: ChangeAction | str,
        from_revision: int,
        actor: str,
        reason: str,
        changed_fields: tuple[str, ...] | list[str],
    ) -> "ChangeEvent":
        return cls(
            event_id=_new_id("event"),
            record_id=record_id,
            action=ChangeAction(action),
            from_revision=from_revision,
            to_revision=from_revision + 1,
            actor=actor,
            reason=reason,
            changed_fields=tuple(changed_fields),
        )


@dataclass(frozen=True, slots=True)
class EvidenceFact:
    """One independently sourced and confirmable fact inside an experience."""

    fact_id: str
    field_name: str
    value: Any
    source_refs: tuple[SourceReference, ...]
    status: FactStatus = FactStatus.UNVERIFIED
    confidence: float = 0.0
    revision: int = 1
    schema_version: str = CAREER_SCHEMA_VERSION
    history: tuple[ChangeEvent, ...] = ()
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    content_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _required_text(self.fact_id, "fact_id"))
        object.__setattr__(self, "field_name", _required_text(self.field_name, "field_name"))
        object.__setattr__(self, "value", _clone_json(self.value, "value"))
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        object.__setattr__(self, "status", FactStatus(self.status))
        object.__setattr__(self, "history", tuple(self.history))
        if not self.source_refs:
            raise ValueError("EvidenceFact requires at least one source reference.")
        if not 0 <= self.confidence <= 1:
            raise ValueError("EvidenceFact confidence must be between 0 and 1.")
        if self.revision < 1 or self.schema_version != CAREER_SCHEMA_VERSION:
            raise ValueError("Unsupported EvidenceFact version.")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at.")
        expected = _dataclass_hash(self)
        if self.content_hash and self.content_hash != expected:
            raise ValueError("EvidenceFact content_hash does not match its content.")
        object.__setattr__(self, "content_hash", expected)

    @classmethod
    def create(
        cls,
        field_name: str,
        value: Any,
        *,
        source_refs: tuple[SourceReference, ...] | list[SourceReference],
        confidence: float = 0.0,
        status: FactStatus | str = FactStatus.UNVERIFIED,
        actor: str = "career_importer",
    ) -> "EvidenceFact":
        fact_id = _new_id("fact")
        event = ChangeEvent.create(
            record_id=fact_id,
            action=ChangeAction.IMPORTED,
            from_revision=0,
            actor=actor,
            reason="Initial fact import",
            changed_fields=("value", "source_refs", "status"),
        )
        return cls(
            fact_id=fact_id,
            field_name=field_name,
            value=value,
            source_refs=tuple(source_refs),
            confidence=confidence,
            status=FactStatus(status),
            history=(event,),
        )

    @property
    def is_usable(self) -> bool:
        return self.status in {FactStatus.VERIFIED, FactStatus.USER_CONFIRMED}

    def verify_integrity(self) -> None:
        if self.content_hash != _dataclass_hash(self):
            raise ValueError("EvidenceFact content_hash does not match its content.")

    def _transition(
        self,
        *,
        action: ChangeAction,
        status: FactStatus,
        actor: str,
        reason: str,
        changed_fields: tuple[str, ...],
        value: Any = _UNSET,
        source_refs: tuple[SourceReference, ...] | None = None,
        confidence: float | None = None,
    ) -> "EvidenceFact":
        event = ChangeEvent.create(
            record_id=self.fact_id,
            action=action,
            from_revision=self.revision,
            actor=actor,
            reason=reason,
            changed_fields=changed_fields,
        )
        return replace(
            self,
            value=self.value if value is _UNSET else value,
            source_refs=self.source_refs if source_refs is None else source_refs,
            confidence=self.confidence if confidence is None else confidence,
            status=status,
            revision=self.revision + 1,
            history=(*self.history, event),
            updated_at=time.time(),
            content_hash="",
        )

    def revise(
        self,
        value: Any,
        *,
        actor: str,
        reason: str,
        source_refs: tuple[SourceReference, ...] | None = None,
        confidence: float | None = None,
    ) -> "EvidenceFact":
        return self._transition(
            action=ChangeAction.MODIFIED,
            status=FactStatus.NEEDS_REVERIFICATION,
            actor=actor,
            reason=reason,
            changed_fields=("value", "source_refs", "confidence", "status"),
            value=value,
            source_refs=source_refs,
            confidence=confidence,
        )

    def confirm(self, *, actor: str, reason: str = "User confirmed fact") -> "EvidenceFact":
        return self._transition(
            action=ChangeAction.CONFIRMED,
            status=FactStatus.USER_CONFIRMED,
            actor=actor,
            reason=reason,
            changed_fields=("status",),
        )

    def invalidate(self, *, actor: str, reason: str) -> "EvidenceFact":
        return self._transition(
            action=ChangeAction.INVALIDATED,
            status=FactStatus.INVALIDATED,
            actor=actor,
            reason=reason,
            changed_fields=("status",),
        )

    def reverify(
        self,
        *,
        supported: bool,
        actor: str,
        reason: str,
        source_refs: tuple[SourceReference, ...] | None = None,
        confidence: float | None = None,
    ) -> "EvidenceFact":
        return self._transition(
            action=ChangeAction.REVERIFIED,
            status=FactStatus.VERIFIED if supported else FactStatus.UNVERIFIED,
            actor=actor,
            reason=reason,
            changed_fields=("status", "source_refs", "confidence"),
            source_refs=source_refs,
            confidence=confidence,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class VersionedCareerRecord:
    """Common immutable revision and provenance fields for Career records."""

    source_refs: tuple[SourceReference, ...]
    revision: int = 1
    schema_version: str = CAREER_SCHEMA_VERSION
    history: tuple[ChangeEvent, ...] = ()
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    content_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        object.__setattr__(self, "history", tuple(self.history))
        if not self.source_refs:
            raise ValueError(f"{type(self).__name__} requires at least one source reference.")
        if self.revision < 1 or self.schema_version != CAREER_SCHEMA_VERSION:
            raise ValueError(f"Unsupported {type(self).__name__} version.")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at.")
        expected = _dataclass_hash(self)
        if self.content_hash and self.content_hash != expected:
            raise ValueError(f"{type(self).__name__} content_hash does not match its content.")
        object.__setattr__(self, "content_hash", expected)

    def to_dict(self) -> dict[str, Any]:
        return record_to_dict(self)

    def verify_integrity(self) -> None:
        if self.content_hash != _dataclass_hash(self):
            raise ValueError(f"{type(self).__name__} content_hash does not match its content.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ResumeSource(VersionedCareerRecord):
    resume_source_id: str
    original_name: str
    media_type: str
    normalized_text: str
    sections: tuple[ResumeSectionBlock, ...] = ()
    page_count: int = 0
    parser_confidence: float = 1.0
    read_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "resume_source_id", _required_text(self.resume_source_id, "resume_source_id"))
        object.__setattr__(self, "original_name", _required_text(self.original_name, "original_name"))
        object.__setattr__(self, "media_type", _required_text(self.media_type, "media_type"))
        object.__setattr__(self, "normalized_text", _required_text(self.normalized_text, "normalized_text"))
        expected_sections = split_resume_source_sections(
            self.normalized_text,
            resume_source_id=self.resume_source_id,
            source_revision=self.revision,
        )
        supplied_sections = tuple(self.sections)
        if supplied_sections and supplied_sections != expected_sections and self.content_hash:
            raise ValueError("ResumeSource sections do not match normalized_text and source revision.")
        object.__setattr__(self, "sections", expected_sections)
        if not self.read_only:
            raise ValueError("ResumeSource is immutable and must remain read_only.")
        if self.page_count < 0 or not 0 <= self.parser_confidence <= 1:
            raise ValueError("Invalid ResumeSource parser metadata.")
        VersionedCareerRecord.__post_init__(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResumeTemplate(VersionedCareerRecord):
    """A format-only resume template; source body text is never stored here."""

    template_id: str
    name: str
    source_kind: str
    original_name: str
    layout_spec: dict[str, Any]
    content_access: str = "format_only"
    resume_source_ref: VersionRef | None = None

    def __post_init__(self) -> None:
        from smart_agent.career.resume_templates import sanitize_layout_spec

        object.__setattr__(self, "template_id", _required_text(self.template_id, "template_id"))
        object.__setattr__(self, "name", _required_text(self.name, "name"))
        object.__setattr__(self, "original_name", _required_text(self.original_name, "original_name"))
        source_kind = str(self.source_kind or "").strip()
        if source_kind not in {"template_upload", "resume_source"}:
            raise ValueError("ResumeTemplate source_kind must be template_upload or resume_source.")
        object.__setattr__(self, "source_kind", source_kind)
        if self.content_access != "format_only":
            raise ValueError("ResumeTemplate content_access must remain format_only.")
        object.__setattr__(self, "layout_spec", sanitize_layout_spec(self.layout_spec))
        if (source_kind == "resume_source") != (self.resume_source_ref is not None):
            raise ValueError("A resume_source template requires exactly one ResumeSource reference.")
        VersionedCareerRecord.__post_init__(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PersonalEvidence(VersionedCareerRecord):
    evidence_id: str
    evidence_type: EvidenceType
    label: str
    facts: tuple[EvidenceFact, ...]

    expected_type: ClassVar[EvidenceType | None] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _required_text(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "evidence_type", EvidenceType(self.evidence_type))
        object.__setattr__(self, "label", _required_text(self.label, "label"))
        object.__setattr__(self, "facts", tuple(self.facts))
        if self.expected_type is not None and self.evidence_type is not self.expected_type:
            raise ValueError(f"{type(self).__name__} must use evidence_type={self.expected_type.value}.")
        if not self.facts:
            raise ValueError("PersonalEvidence requires at least one fact.")
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("PersonalEvidence fact IDs must be unique.")
        VersionedCareerRecord.__post_init__(self)

    def fact(self, fact_id: str) -> EvidenceFact:
        return next((fact for fact in self.facts if fact.fact_id == fact_id), None) or _raise_missing_fact(fact_id)

    def with_fact(self, next_fact: EvidenceFact, *, actor: str, reason: str) -> "PersonalEvidence":
        current = self.fact(next_fact.fact_id)
        if next_fact.revision != current.revision + 1:
            raise ValueError("EvidenceFact revisions must be sequential.")
        event = ChangeEvent.create(
            record_id=self.evidence_id,
            action=next_fact.history[-1].action,
            from_revision=self.revision,
            actor=actor,
            reason=reason,
            changed_fields=(f"facts.{next_fact.fact_id}",),
        )
        return replace(
            self,
            facts=tuple(next_fact if fact.fact_id == next_fact.fact_id else fact for fact in self.facts),
            revision=self.revision + 1,
            history=(*self.history, event),
            updated_at=time.time(),
            content_hash="",
        )


def _raise_missing_fact(fact_id: str):
    raise KeyError(f"Unknown evidence fact: {fact_id}")


@dataclass(frozen=True, slots=True, kw_only=True)
class EducationEvidence(PersonalEvidence):
    expected_type: ClassVar[EvidenceType] = EvidenceType.EDUCATION


@dataclass(frozen=True, slots=True, kw_only=True)
class AwardEvidence(PersonalEvidence):
    expected_type: ClassVar[EvidenceType] = EvidenceType.AWARD


@dataclass(frozen=True, slots=True, kw_only=True)
class InternshipEvidence(PersonalEvidence):
    expected_type: ClassVar[EvidenceType] = EvidenceType.INTERNSHIP


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectEvidence(PersonalEvidence):
    expected_type: ClassVar[EvidenceType] = EvidenceType.PROJECT


@dataclass(frozen=True, slots=True, kw_only=True)
class ResearchEvidence(PersonalEvidence):
    expected_type: ClassVar[EvidenceType] = EvidenceType.RESEARCH


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillsEvidence(PersonalEvidence):
    expected_type: ClassVar[EvidenceType] = EvidenceType.SKILLS


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectDescriptionDocument(VersionedCareerRecord):
    """Traceable, user-confirmable description of one imported project."""

    project_description_id: str
    project_id: str
    title: str
    project_type: str = ""
    background: str = ""
    goals: tuple[str, ...] = ()
    problem_statement: str = ""
    role: str = ""
    responsibilities: tuple[str, ...] = ()
    core_flow: tuple[str, ...] = ()
    architecture: str = ""
    technology_choices: tuple[str, ...] = ()
    core_modules: tuple[str, ...] = ()
    technologies: tuple[str, ...] = ()
    technical_tags: tuple[str, ...] = ()
    implemented_features: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    confidence_by_field: dict[str, float] = field(default_factory=dict)
    challenges: tuple[str, ...] = ()
    timeline: str = ""
    team_size: int = 0
    collaboration: str = ""
    links: tuple[str, ...] = ()
    material_ids: tuple[str, ...] = ()
    field_evidence: dict[str, tuple[SourceReference, ...]] = field(default_factory=dict)
    status: ProjectDescriptionStatus = ProjectDescriptionStatus.NEEDS_USER_CONFIRMATION
    generator: str = ""
    model: str = ""
    project_version: str = ""
    evidence_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "project_description_id",
            _required_text(self.project_description_id, "project_description_id"),
        )
        object.__setattr__(self, "project_id", _required_text(self.project_id, "project_id"))
        object.__setattr__(self, "title", _required_text(self.title, "title"))
        for field_name in (
            "project_type",
            "background",
            "problem_statement",
            "role",
            "architecture",
            "timeline",
            "collaboration",
            "generator",
            "model",
            "project_version",
            "evidence_id",
        ):
            object.__setattr__(self, field_name, str(getattr(self, field_name) or "").strip())
        for field_name in (
            "goals",
            "responsibilities",
            "core_flow",
            "technology_choices",
            "core_modules",
            "technologies",
            "technical_tags",
            "implemented_features",
            "outcomes",
            "unknowns",
            "challenges",
            "links",
            "material_ids",
        ):
            object.__setattr__(self, field_name, _unique_texts(tuple(getattr(self, field_name)), field_name))
        if self.team_size < 0:
            raise ValueError("ProjectDescriptionDocument team_size cannot be negative.")
        object.__setattr__(self, "status", ProjectDescriptionStatus(self.status))
        confidence = {str(name): float(score) for name, score in dict(self.confidence_by_field).items()}
        if any(not 0 <= score <= 1 for score in confidence.values()):
            raise ValueError("ProjectDescriptionDocument confidence scores must be between 0 and 1.")
        object.__setattr__(self, "confidence_by_field", confidence)
        normalized_evidence = {
            _required_text(name, "field_evidence key"): tuple(refs)
            for name, refs in dict(self.field_evidence).items()
        }
        if any(not refs for refs in normalized_evidence.values()):
            raise ValueError("ProjectDescriptionDocument field_evidence entries cannot be empty.")
        source_keys = {
            (ref.source_id, ref.content_hash, json.dumps(ref.locator, sort_keys=True))
            for ref in self.source_refs
        }
        for refs in normalized_evidence.values():
            for ref in refs:
                key = (ref.source_id, ref.content_hash, json.dumps(ref.locator, sort_keys=True))
                if key not in source_keys:
                    raise ValueError("ProjectDescriptionDocument field_evidence must be included in source_refs.")
        object.__setattr__(self, "field_evidence", normalized_evidence)
        VersionedCareerRecord.__post_init__(self)

    @property
    def is_usable(self) -> bool:
        return self.status is ProjectDescriptionStatus.USER_CONFIRMED


_EVIDENCE_CLASSES = {
    EvidenceType.EDUCATION: EducationEvidence,
    EvidenceType.AWARD: AwardEvidence,
    EvidenceType.INTERNSHIP: InternshipEvidence,
    EvidenceType.PROJECT: ProjectEvidence,
    EvidenceType.RESEARCH: ResearchEvidence,
    EvidenceType.SKILLS: SkillsEvidence,
}


def create_personal_evidence(
    evidence_type: EvidenceType | str,
    *,
    label: str,
    facts: tuple[EvidenceFact, ...] | list[EvidenceFact],
    source_refs: tuple[SourceReference, ...] | list[SourceReference],
    actor: str = "career_importer",
) -> PersonalEvidence:
    category = EvidenceType(evidence_type)
    evidence_id = _new_id("evidence")
    event = ChangeEvent.create(
        record_id=evidence_id,
        action=ChangeAction.IMPORTED,
        from_revision=0,
        actor=actor,
        reason="Initial personal evidence import",
        changed_fields=("facts", "source_refs"),
    )
    return _EVIDENCE_CLASSES[category](
        evidence_id=evidence_id,
        evidence_type=category,
        label=label,
        facts=tuple(facts),
        source_refs=tuple(source_refs),
        history=(event,),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class JDDocument(VersionedCareerRecord):
    jd_id: str
    raw_text: str
    role_title: str = ""
    company: str = ""
    requirement_refs: tuple[VersionRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "jd_id", _required_text(self.jd_id, "jd_id"))
        object.__setattr__(self, "raw_text", _required_text(self.raw_text, "raw_text"))
        object.__setattr__(self, "requirement_refs", _unique_refs(self.requirement_refs, "requirement_refs"))
        VersionedCareerRecord.__post_init__(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class JDRequirement(VersionedCareerRecord):
    requirement_id: str
    jd_ref: VersionRef
    category: RequirementCategory
    text: str
    original_text: str
    required: bool = True
    confidence: float = 1.0
    active: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirement_id", _required_text(self.requirement_id, "requirement_id"))
        object.__setattr__(self, "category", RequirementCategory(self.category))
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        object.__setattr__(self, "original_text", _required_text(self.original_text, "original_text"))
        if not 0 <= self.confidence <= 1:
            raise ValueError("JDRequirement confidence must be between 0 and 1.")
        VersionedCareerRecord.__post_init__(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceMatch(VersionedCareerRecord):
    match_id: str
    requirement_ref: VersionRef
    evidence_ref: VersionRef | None
    evidence_fact_ids: tuple[str, ...]
    support_level: SupportLevel
    reason: str
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "match_id", _required_text(self.match_id, "match_id"))
        object.__setattr__(
            self,
            "evidence_fact_ids",
            _unique_texts(self.evidence_fact_ids, "evidence_fact_ids"),
        )
        object.__setattr__(self, "support_level", SupportLevel(self.support_level))
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        if not 0 <= self.score <= 1:
            raise ValueError("EvidenceMatch score must be between 0 and 1.")
        if self.support_level is SupportLevel.UNSUPPORTED:
            if self.evidence_ref is not None or self.evidence_fact_ids:
                raise ValueError("Unsupported EvidenceMatch cannot cite evidence.")
        elif self.evidence_ref is None or not self.evidence_fact_ids:
            raise ValueError("Supported EvidenceMatch requires evidence and fact references.")
        VersionedCareerRecord.__post_init__(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResumeItem(VersionedCareerRecord):
    item_id: str
    section: str
    proposed_text: str
    evidence_refs: tuple[VersionRef, ...]
    evidence_fact_ids: tuple[str, ...]
    requirement_refs: tuple[VersionRef, ...]
    redundancy_group: str = ""
    features: dict[str, float] = field(default_factory=dict)
    length_cost: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _required_text(self.item_id, "item_id"))
        object.__setattr__(self, "section", normalize_resume_section(_required_text(self.section, "section")))
        object.__setattr__(self, "proposed_text", _required_text(self.proposed_text, "proposed_text"))
        object.__setattr__(self, "evidence_refs", _unique_refs(self.evidence_refs, "evidence_refs"))
        object.__setattr__(
            self,
            "evidence_fact_ids",
            _unique_texts(self.evidence_fact_ids, "evidence_fact_ids"),
        )
        object.__setattr__(self, "requirement_refs", _unique_refs(self.requirement_refs, "requirement_refs"))
        object.__setattr__(self, "redundancy_group", str(self.redundancy_group or "").strip())
        object.__setattr__(self, "features", {str(key): float(value) for key, value in self.features.items()})
        if not self.evidence_refs or not self.evidence_fact_ids:
            raise ValueError("ResumeItem requires evidence facts.")
        if self.length_cost < 0:
            raise ValueError("ResumeItem length_cost cannot be negative.")
        VersionedCareerRecord.__post_init__(self)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(ref.record_id for ref in self.evidence_refs)

    @property
    def requirement_ids(self) -> tuple[str, ...]:
        return tuple(ref.record_id for ref in self.requirement_refs)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResumePlan(VersionedCareerRecord):
    plan_id: str
    jd_ref: VersionRef
    item_refs: tuple[VersionRef, ...]
    selected_item_ids: tuple[str, ...]
    excluded_reasons: dict[str, str]
    section_order: tuple[str, ...]
    constraints: dict[str, Any]
    resume_source_ref: VersionRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _required_text(self.plan_id, "plan_id"))
        object.__setattr__(self, "item_refs", _unique_refs(self.item_refs, "item_refs"))
        object.__setattr__(
            self,
            "selected_item_ids",
            _unique_texts(self.selected_item_ids, "selected_item_ids"),
        )
        object.__setattr__(self, "section_order", RESUME_SECTION_ORDER)
        object.__setattr__(self, "excluded_reasons", _clone_json(self.excluded_reasons, "excluded_reasons"))
        object.__setattr__(self, "constraints", _clone_json(self.constraints, "constraints"))
        known_items = {ref.record_id for ref in self.item_refs}
        if not set(self.selected_item_ids).issubset(known_items):
            raise ValueError("ResumePlan selected items must be declared in item_refs.")
        VersionedCareerRecord.__post_init__(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResumeClaim(VersionedCareerRecord):
    claim_id: str
    text: str
    evidence_refs: tuple[VersionRef, ...]
    evidence_fact_ids: tuple[str, ...]
    requirement_refs: tuple[VersionRef, ...]
    status: ClaimStatus = ClaimStatus.DRAFT

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _required_text(self.claim_id, "claim_id"))
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        object.__setattr__(self, "evidence_refs", _unique_refs(self.evidence_refs, "evidence_refs"))
        object.__setattr__(
            self,
            "evidence_fact_ids",
            _unique_texts(self.evidence_fact_ids, "evidence_fact_ids"),
        )
        object.__setattr__(self, "requirement_refs", _unique_refs(self.requirement_refs, "requirement_refs"))
        object.__setattr__(self, "status", ClaimStatus(self.status))
        if not self.evidence_refs or not self.evidence_fact_ids:
            raise ValueError("ResumeClaim requires evidence_ids and evidence_fact_ids.")
        VersionedCareerRecord.__post_init__(self)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(ref.record_id for ref in self.evidence_refs)

    @property
    def requirement_ids(self) -> tuple[str, ...]:
        return tuple(ref.record_id for ref in self.requirement_refs)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResumeDocument(VersionedCareerRecord):
    resume_document_id: str
    jd_ref: VersionRef
    plan_ref: VersionRef
    claim_refs: tuple[VersionRef, ...]
    sections: dict[str, tuple[str, ...]]
    status: ResumeDocumentStatus = ResumeDocumentStatus.DRAFT
    header: ResumeHeader | None = None
    generator_provider: str = ""
    generator_model: str = ""
    generator_api_id: str = ""
    revision_round: int = 0
    verification_report_ref: VersionRef | None = None
    template_ref: VersionRef | None = None
    layout_spec: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "resume_document_id", _required_text(self.resume_document_id, "resume_document_id"))
        object.__setattr__(self, "claim_refs", _unique_refs(self.claim_refs, "claim_refs"))
        object.__setattr__(self, "layout_spec", _clone_json(self.layout_spec or {}, "layout_spec"))
        normalized_sections = normalize_resume_sections(self.sections)
        object.__setattr__(self, "sections", normalized_sections)
        object.__setattr__(self, "status", ResumeDocumentStatus(self.status))
        object.__setattr__(self, "generator_provider", str(self.generator_provider or "").strip())
        object.__setattr__(self, "generator_model", str(self.generator_model or "").strip())
        object.__setattr__(self, "generator_api_id", str(self.generator_api_id or "").strip())
        if self.revision_round < 0 or self.revision_round > 2:
            raise ValueError("ResumeDocument revision_round must be between 0 and 2.")
        generation_fields = (
            self.generator_provider,
            self.generator_model,
            self.generator_api_id,
        )
        if any(generation_fields) and not all(generation_fields):
            raise ValueError("ResumeDocument generator identity must be complete when supplied.")
        declared = {ref.record_id for ref in self.claim_refs}
        rendered_list = [claim_id for claim_ids in normalized_sections.values() for claim_id in claim_ids]
        if len(rendered_list) != len(set(rendered_list)) or set(rendered_list) != declared:
            raise ValueError("ResumeDocument sections must contain every declared claim exactly by ID.")
        VersionedCareerRecord.__post_init__(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class VerificationReport(VersionedCareerRecord):
    verification_report_id: str
    resume_ref: VersionRef
    verifier_agent: str
    provider: str
    model: str
    status: VerificationStatus
    issues: tuple[dict[str, Any], ...]
    checked_evidence_refs: tuple[VersionRef, ...]
    verifier_api_id: str = ""
    child_session_id: str = ""
    input_artifact_id: str = ""
    input_artifact_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "verification_report_id", _required_text(self.verification_report_id, "verification_report_id"))
        object.__setattr__(self, "verifier_agent", _required_text(self.verifier_agent, "verifier_agent"))
        object.__setattr__(self, "provider", _required_text(self.provider, "provider"))
        object.__setattr__(self, "model", _required_text(self.model, "model"))
        object.__setattr__(self, "status", VerificationStatus(self.status))
        object.__setattr__(self, "issues", tuple(_clone_json(item, "verification issue") for item in self.issues))
        object.__setattr__(self, "checked_evidence_refs", tuple(self.checked_evidence_refs))
        object.__setattr__(self, "verifier_api_id", str(self.verifier_api_id or "").strip())
        object.__setattr__(self, "child_session_id", str(self.child_session_id or "").strip())
        object.__setattr__(self, "input_artifact_id", str(self.input_artifact_id or "").strip())
        object.__setattr__(self, "input_artifact_hash", str(self.input_artifact_hash or "").strip())
        if self.input_artifact_hash and (
            len(self.input_artifact_hash) != 64
            or any(char not in "0123456789abcdef" for char in self.input_artifact_hash.lower())
        ):
            raise ValueError("VerificationReport input_artifact_hash must be a SHA-256 hex digest.")
        if self.status is VerificationStatus.PASSED and self.issues:
            raise ValueError("A passed VerificationReport cannot contain issues.")
        VersionedCareerRecord.__post_init__(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResumeRun(VersionedCareerRecord):
    """One isolated JD-to-resume task pinned to an immutable material snapshot."""

    resume_run_id: str
    jd_ref: VersionRef
    requirement_refs: tuple[VersionRef, ...]
    evidence_snapshot_refs: tuple[VersionRef, ...]
    material_catalog_version: str
    status: ResumeRunStatus = ResumeRunStatus.CREATED
    match_refs: tuple[VersionRef, ...] = ()
    plan_ref: VersionRef | None = None
    resume_document_ref: VersionRef | None = None
    verification_report_ref: VersionRef | None = None
    rendered_artifacts: tuple[str, ...] = ()
    customization_messages: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "resume_run_id", _required_text(self.resume_run_id, "resume_run_id"))
        for field_name in ("requirement_refs", "evidence_snapshot_refs", "match_refs"):
            object.__setattr__(self, field_name, _unique_refs(tuple(getattr(self, field_name)), field_name))
        object.__setattr__(self, "status", ResumeRunStatus(self.status))
        object.__setattr__(
            self,
            "rendered_artifacts",
            _unique_texts(tuple(self.rendered_artifacts), "rendered_artifacts"),
        )
        messages: list[dict[str, Any]] = []
        for item in tuple(self.customization_messages):
            if not isinstance(item, dict):
                raise ValueError("ResumeRun customization_messages must contain objects.")
            role = str(item.get("role") or "").strip().casefold()
            content = str(item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                raise ValueError("ResumeRun customization messages require a role and content.")
            if len(content) > 2000:
                raise ValueError("ResumeRun customization message is too long.")
            message = {
                "role": role,
                "content": content,
                "created_at": float(item.get("created_at") or 0),
            }
            document_id = str(item.get("resume_document_id") or "").strip()
            if document_id:
                message["resume_document_id"] = document_id
            messages.append(message)
        object.__setattr__(self, "customization_messages", tuple(messages[-40:]))
        digest = str(self.material_catalog_version or "").strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("ResumeRun material_catalog_version must be a SHA-256 hex digest.")
        object.__setattr__(self, "material_catalog_version", digest)
        if not self.requirement_refs:
            raise ValueError("ResumeRun requires at least one JD requirement.")
        VersionedCareerRecord.__post_init__(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RequirementMatchCache(VersionedCareerRecord):
    """Requirement-level evidence cache; never stores a complete resume."""

    cache_id: str
    requirement_signature: str
    requirement_text: str
    category: RequirementCategory
    required: bool
    match_refs: tuple[VersionRef, ...]
    evidence_dependency_refs: tuple[VersionRef, ...]
    material_dependency_ids: tuple[str, ...]
    material_dependency_hashes: dict[str, str]
    material_catalog_version: str
    retrieval_mode: str
    retrieval_backend: str = ""
    embedding_model: str = ""
    config_version: str = ""
    status: RequirementMatchCacheStatus = RequirementMatchCacheStatus.FRESH
    reused_from: str = ""
    evidence_scope_hashes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_id", _required_text(self.cache_id, "cache_id"))
        object.__setattr__(self, "requirement_signature", _required_text(self.requirement_signature, "requirement_signature"))
        object.__setattr__(self, "requirement_text", _required_text(self.requirement_text, "requirement_text"))
        object.__setattr__(self, "category", RequirementCategory(self.category))
        object.__setattr__(self, "match_refs", _unique_refs(tuple(self.match_refs), "match_refs"))
        object.__setattr__(self, "evidence_dependency_refs", _unique_refs(tuple(self.evidence_dependency_refs), "evidence_dependency_refs"))
        object.__setattr__(self, "material_dependency_ids", _unique_texts(tuple(self.material_dependency_ids), "material_dependency_ids"))
        object.__setattr__(self, "material_dependency_hashes", _clone_json(self.material_dependency_hashes, "material_dependency_hashes"))
        object.__setattr__(self, "evidence_scope_hashes", _clone_json(self.evidence_scope_hashes, "evidence_scope_hashes"))
        for field_name in ("material_catalog_version", "retrieval_mode", "retrieval_backend", "embedding_model", "config_version", "reused_from"):
            object.__setattr__(self, field_name, str(getattr(self, field_name) or "").strip())
        if len(self.material_catalog_version) != 64 or any(char not in "0123456789abcdef" for char in self.material_catalog_version.lower()):
            raise ValueError("RequirementMatchCache material_catalog_version must be a SHA-256 hex digest.")
        object.__setattr__(self, "status", RequirementMatchCacheStatus(self.status))
        VersionedCareerRecord.__post_init__(self)


CareerRecord = (
    ResumeSource
    | ResumeTemplate
    | PersonalEvidence
    | ProjectDescriptionDocument
    | JDDocument
    | JDRequirement
    | EvidenceMatch
    | ResumeItem
    | ResumePlan
    | ResumeClaim
    | ResumeDocument
    | VerificationReport
    | ResumeRun
    | RequirementMatchCache
)

_RECORD_CLASSES = {
    cls.__name__: cls
    for cls in (
        ResumeSource,
        ResumeTemplate,
        EducationEvidence,
        AwardEvidence,
        InternshipEvidence,
        ProjectEvidence,
        ResearchEvidence,
        SkillsEvidence,
        ProjectDescriptionDocument,
        JDDocument,
        JDRequirement,
        EvidenceMatch,
        ResumeItem,
        ResumePlan,
        ResumeClaim,
        ResumeDocument,
        VerificationReport,
        ResumeRun,
        RequirementMatchCache,
    )
}


def record_id(record: CareerRecord) -> str:
    for field_name in (
        "resume_source_id",
        "template_id",
        "project_description_id",
        "evidence_id",
        "jd_id",
        "requirement_id",
        "match_id",
        "item_id",
        "plan_id",
        "claim_id",
        "resume_document_id",
        "verification_report_id",
        "resume_run_id",
        "cache_id",
    ):
        value = getattr(record, field_name, None)
        if value:
            return str(value)
    raise TypeError(f"Unsupported Career record: {type(record).__name__}")


def version_ref(record: CareerRecord) -> VersionRef:
    return VersionRef(record_id(record), record.revision)


def record_to_dict(record: VersionedCareerRecord) -> dict[str, Any]:
    record.verify_integrity()
    payload = _serialize(record)
    payload["record_type"] = type(record).__name__
    if isinstance(record, (ResumeItem, ResumeClaim)):
        payload["evidence_ids"] = list(record.evidence_ids)
        payload["requirement_ids"] = list(record.requirement_ids)
    return payload


def _coerce_value(annotation: Any, value: Any) -> Any:
    if annotation is Any:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        for option in args:
            if option is type(None):
                continue
            try:
                return _coerce_value(option, value)
            except (TypeError, ValueError, KeyError):
                continue
        raise ValueError(f"Cannot coerce value for {annotation!r}")
    if origin is tuple:
        item_type = args[0] if args else Any
        return tuple(_coerce_value(item_type, item) for item in value)
    if origin is list:
        item_type = args[0] if args else Any
        return [_coerce_value(item_type, item) for item in value]
    if origin is dict:
        key_type, value_type = args or (Any, Any)
        return {
            _coerce_value(key_type, key): _coerce_value(value_type, item)
            for key, item in value.items()
        }
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _coerce_dataclass(annotation, value)
    return value


def _coerce_dataclass(cls, value: dict[str, Any]):
    hints = get_type_hints(cls)
    kwargs = {
        item.name: _coerce_value(hints.get(item.name, Any), value[item.name])
        for item in fields(cls)
        if item.name in value
    }
    return cls(**kwargs)


def record_from_dict(value: dict[str, Any]) -> CareerRecord:
    payload = dict(value)
    record_type = payload.pop("record_type", "")
    serialized_evidence_ids = payload.pop("evidence_ids", None)
    serialized_requirement_ids = payload.pop("requirement_ids", None)
    cls = _RECORD_CLASSES.get(record_type)
    if cls is None:
        raise ValueError(f"Unknown Career record_type: {record_type}")
    requires_migration = False
    if record_type == "ResumeSource" and "sections" not in payload:
        requires_migration = True
    if record_type == "ResumeItem":
        normalized_section = normalize_resume_section(payload.get("section"))
        if payload.get("section") != normalized_section:
            payload["section"] = normalized_section
            requires_migration = True
    if record_type == "ResumePlan" and tuple(payload.get("section_order") or ()) != RESUME_SECTION_ORDER:
        payload["section_order"] = list(RESUME_SECTION_ORDER)
        requires_migration = True
    if record_type == "ResumeTemplate":
        layout_spec = payload.get("layout_spec")
        if (
            isinstance(layout_spec, dict)
            and "section_order" in layout_spec
            and tuple(layout_spec.get("section_order") or ()) != RESUME_SECTION_ORDER
        ):
            payload["layout_spec"] = {
                **layout_spec,
                "section_order": list(RESUME_SECTION_ORDER),
            }
            requires_migration = True
    if record_type == "ResumeDocument":
        normalized_sections = {
            section: list(claim_ids)
            for section, claim_ids in normalize_resume_sections(payload.get("sections")).items()
        }
        if payload.get("sections") != normalized_sections:
            payload["sections"] = normalized_sections
            requires_migration = True
    if record_type == "ResumeDocument" and "template_ref" not in payload:
        payload["template_ref"] = None
        requires_migration = True
    if record_type == "ResumeDocument" and "layout_spec" not in payload:
        payload["layout_spec"] = {}
        requires_migration = True
    if record_type == "ResumeRun" and "customization_messages" not in payload:
        payload["customization_messages"] = []
        requires_migration = True
    if requires_migration:
        legacy_hash = str(payload.get("content_hash") or "")
        original_payload = {
            key: item
            for key, item in value.items()
            if key not in {"record_type", "content_hash", "evidence_ids", "requirement_ids"}
        }
        if legacy_hash != _sha256_payload(original_payload):
            raise ValueError(f"{record_type} content_hash does not match its legacy content.")
        payload["content_hash"] = ""
    record = _coerce_dataclass(cls, payload)
    if isinstance(record, (ResumeItem, ResumeClaim)):
        if serialized_evidence_ids != list(record.evidence_ids):
            raise ValueError("Serialized evidence_ids do not match evidence_refs.")
        if serialized_requirement_ids != list(record.requirement_ids):
            raise ValueError("Serialized requirement_ids do not match requirement_refs.")
    return record


def create_record_id(record_type: str) -> str:
    prefixes = {
        "ResumeSource": "resume-source",
        "ResumeTemplate": "resume-template",
        "ProjectDescriptionDocument": "project-description",
        "JDDocument": "jd",
        "JDRequirement": "requirement",
        "EvidenceMatch": "match",
        "ResumeItem": "item",
        "ResumePlan": "plan",
        "ResumeClaim": "claim",
        "ResumeDocument": "resume",
        "VerificationReport": "verification",
        "ResumeRun": "resume-run",
        "RequirementMatchCache": "requirement-cache",
    }
    try:
        return _new_id(prefixes[record_type])
    except KeyError as exc:
        raise ValueError(f"Unsupported record type: {record_type}") from exc
