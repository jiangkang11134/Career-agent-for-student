"""Traceable text-JD normalization for the Career domain."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from smart_agent.career.models import (
    JDDocument,
    JDRequirement,
    RequirementCategory,
    SourceReference,
    SourceType,
    VersionRef,
    record_to_dict,
    version_ref,
)
from smart_agent.career.store import CareerStore, RecordNotFoundError

_USER_REQUEST_SECTION = re.compile(
    r"^\s*#{1,6}\s*(?:用户需求|用户要求|user\s+requests?)\s*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class RequirementDraft:
    category: RequirementCategory
    text: str
    original_text: str
    required: bool = True
    confidence: float = 1.0
    start: int | None = None
    end: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RequirementDraft":
        if not isinstance(value, dict):
            raise ValueError("Each JD requirement must be an object.")
        text = str(value.get("text") or "").strip()
        original_text = str(value.get("original_text") or text).strip()
        if not text or not original_text:
            raise ValueError("JD requirement text and original_text are required.")
        confidence = float(value.get("confidence", 1.0))
        if not 0 <= confidence <= 1:
            raise ValueError("JD requirement confidence must be between 0 and 1.")
        start = value.get("start")
        end = value.get("end")
        if (start is None) != (end is None):
            raise ValueError("JD requirement start and end must be supplied together.")
        required = value.get("required", True)
        if not isinstance(required, bool):
            raise ValueError("JD requirement required must be a boolean.")
        return cls(
            category=RequirementCategory(value.get("category")),
            text=text,
            original_text=original_text,
            required=required,
            confidence=confidence,
            start=int(start) if start is not None else None,
            end=int(end) if end is not None else None,
        )


@dataclass(frozen=True, slots=True)
class JDAnalysis:
    document: JDDocument
    requirements: tuple[JDRequirement, ...]
    analysis_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_mode": self.analysis_mode,
            "document": record_to_dict(self.document),
            "requirements": [record_to_dict(item) for item in self.requirements],
        }

    def persist(self, store: CareerStore) -> "JDAnalysis":
        try:
            existing = store.get(self.document.jd_id, expected_type=JDDocument)
        except RecordNotFoundError:
            existing = None
        if existing is None:
            store.put(self.document)
            existing = self.document
        elif existing.raw_text != self.document.raw_text:
            raise ValueError("Stable JD ID collision with different source text.")

        persisted_requirements: list[JDRequirement] = []
        for requirement in self.requirements:
            try:
                current_requirement = store.get(
                    requirement.requirement_id,
                    expected_type=JDRequirement,
                )
            except RecordNotFoundError:
                store.put(requirement)
                current_requirement = requirement
            else:
                changes = self._requirement_changes(current_requirement, requirement)
                if changes:
                    current_requirement = store.revise_record(
                        requirement.requirement_id,
                        actor="jd_analyzer",
                        reason="Update normalized JD requirement",
                        changes=changes,
                    )
            persisted_requirements.append(current_requirement)

        desired_refs = tuple(version_ref(item) for item in persisted_requirements)
        current_document = store.get(self.document.jd_id, expected_type=JDDocument)
        if current_document.requirement_refs != desired_refs:
            current_document = store.revise_record(
                self.document.jd_id,
                actor="jd_analyzer",
                reason="Attach normalized JD requirements",
                changes={"requirement_refs": desired_refs},
            )
        return JDAnalysis(current_document, tuple(persisted_requirements), self.analysis_mode)

    @staticmethod
    def _requirement_changes(
        current: JDRequirement,
        proposed: JDRequirement,
    ) -> dict[str, Any]:
        comparable_fields = (
            "category",
            "text",
            "original_text",
            "required",
            "confidence",
            "active",
        )
        return {
            field_name: getattr(proposed, field_name)
            for field_name in comparable_fields
            if getattr(current, field_name) != getattr(proposed, field_name)
        }


class JDAnalyzer:
    """Validate model-produced JD fields against the source text and normalize them."""

    def analyze(
        self,
        text: str,
        *,
        requirements: list[dict[str, Any]] | tuple[RequirementDraft, ...] | None = None,
        role_title: str = "",
        company: str = "",
        source_id: str = "",
        source_type: SourceType | str = SourceType.USER_INPUT,
        locator: dict[str, Any] | None = None,
        validate_metadata_presence: bool = True,
    ) -> JDAnalysis:
        raw_text = str(text or "").strip()
        if not raw_text:
            raise ValueError("JD text is required.")
        digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        jd_id = f"jd-{digest[:16]}"
        resolved_source_id = str(source_id or f"source-{jd_id}").strip()
        base_locator = dict(locator or {"channel": "user_input"})
        document_source = SourceReference.create(
            source_id=resolved_source_id,
            source_type=source_type,
            locator=base_locator,
            raw_content=raw_text,
            excerpt=raw_text,
        )
        if requirements:
            drafts = tuple(
                item if isinstance(item, RequirementDraft) else RequirementDraft.from_dict(item)
                for item in requirements
            )
            mode = "model_structured"
        else:
            drafts = self._heuristic_drafts(raw_text)
            mode = "heuristic_baseline"
        drafts = tuple(
            draft
            for draft in drafts
            if not self._is_user_request_draft(draft, raw_text)
        )
        if not drafts:
            raise ValueError("No JD requirements could be extracted.")

        resolved_title = str(role_title or self._field(raw_text, ("职位", "岗位", "position", "role"))).strip()
        resolved_company = str(company or self._field(raw_text, ("公司", "company"))).strip()
        if validate_metadata_presence:
            for field_name, value in (("role_title", resolved_title), ("company", resolved_company)):
                if value and value.casefold() not in raw_text.casefold():
                    raise ValueError(f"JD {field_name} is not present in the source text.")
        document = JDDocument(
            jd_id=jd_id,
            raw_text=raw_text,
            role_title=resolved_title,
            company=resolved_company,
            source_refs=(document_source,),
        )
        normalized: list[JDRequirement] = []
        cursor = 0
        seen: set[tuple[str, int, int]] = set()
        for draft in drafts:
            start, end = self._locate(raw_text, draft, cursor)
            cursor = end
            key = (draft.text.casefold(), start, end)
            if key in seen:
                continue
            seen.add(key)
            original = raw_text[start:end]
            requirement_digest = hashlib.sha256(
                f"{jd_id}\0{start}\0{end}\0{draft.text.casefold()}".encode("utf-8")
            ).hexdigest()
            requirement_source = SourceReference.create(
                source_id=resolved_source_id,
                source_type=source_type,
                locator={
                    **base_locator,
                    "start": start,
                    "end": end,
                    "line": raw_text.count("\n", 0, start) + 1,
                },
                raw_content=raw_text,
                excerpt=original,
            )
            normalized.append(
                JDRequirement(
                    requirement_id=f"requirement-{requirement_digest[:16]}",
                    jd_ref=VersionRef(jd_id, 1),
                    category=draft.category,
                    text=draft.text,
                    original_text=original,
                    required=draft.required,
                    confidence=draft.confidence,
                    source_refs=(requirement_source,),
                )
            )
        return JDAnalysis(document, tuple(normalized), mode)

    @staticmethod
    def _field(text: str, labels: tuple[str, ...]) -> str:
        for label in labels:
            match = re.search(rf"(?im)^\s*{re.escape(label)}\s*[:：]\s*([^\n]+)", text)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _locate(text: str, draft: RequirementDraft, cursor: int) -> tuple[int, int]:
        if draft.start is not None and draft.end is not None:
            if draft.start < 0 or draft.end <= draft.start or draft.end > len(text):
                raise ValueError("JD requirement source span is outside the source text.")
            if text[draft.start:draft.end] != draft.original_text:
                raise ValueError("JD requirement source span does not match original_text.")
            return draft.start, draft.end
        start = text.find(draft.original_text, cursor)
        if start < 0:
            start = text.find(draft.original_text)
        if start < 0:
            raise ValueError(
                f"JD requirement original_text is not present in the source JD: {draft.original_text}"
            )
        return start, start + len(draft.original_text)

    @classmethod
    def _heuristic_drafts(cls, text: str) -> tuple[RequirementDraft, ...]:
        # A JD may be stored together with a separate numbered customization
        # section.  Those requests are user instructions, not employer
        # requirements, so stop heuristic extraction at the section boundary
        # while preserving offsets into the original source text.
        source_text = re.split(
            r"^\s*#{1,6}\s*(?:用户需求|用户要求|user\s+requests?)\s*:?[ \t]*$",
            text,
            maxsplit=1,
            flags=re.IGNORECASE | re.MULTILINE,
        )[0]
        drafts: list[RequirementDraft] = []
        for match in re.finditer(r"[^\n；;。]+[；;。]?", source_text):
            raw = match.group(0)
            leading = re.match(r"\s*(?:(?:[-*•·])|(?:\d+[.)、]))?\s*", raw)
            offset = leading.end() if leading else 0
            original = raw[offset:].strip()
            if not original or re.match(
                r"(?i)^(?:公司(?:名字|名称)?|company|职位|岗位|position|role)\s*[:：]",
                original,
            ):
                continue
            if original.endswith((":", "：")) and len(original) <= 24:
                continue
            start = match.start() + raw.find(original)
            category = cls._classify(original)
            preferred = bool(re.search(r"优先|加分|preferred|nice\s+to\s+have", original, re.IGNORECASE))
            drafts.append(
                RequirementDraft(
                    category=category,
                    text=original.rstrip("。；; "),
                    original_text=original,
                    required=not preferred,
                    confidence=0.55,
                    start=start,
                    end=start + len(original),
                )
            )
        return tuple(drafts)

    @staticmethod
    def _is_user_request_draft(draft: RequirementDraft, raw_text: str) -> bool:
        """Return true when a structured requirement points into user requests."""

        section = _USER_REQUEST_SECTION.search(raw_text)
        if section is None:
            return False
        section_start = section.start()
        if draft.start is not None and draft.start >= section_start:
            return True
        for value in (draft.original_text, draft.text):
            phrase = str(value or "").strip()
            if not phrase:
                continue
            cursor = 0
            found_before = False
            found_after = False
            while True:
                position = raw_text.find(phrase, cursor)
                if position < 0:
                    break
                if position >= section_start:
                    found_after = True
                else:
                    found_before = True
                cursor = position + max(1, len(phrase))
            if found_after and not found_before:
                return True
        return False

    @staticmethod
    def _classify(text: str) -> RequirementCategory:
        lowered = text.casefold()
        if re.search(r"优先|加分|preferred|nice\s+to\s+have", lowered):
            return RequirementCategory.PREFERENCE
        if re.search(r"学历|本科|硕士|博士|degree|bachelor|master|phd", lowered):
            return RequirementCategory.EDUCATION
        if re.search(r"\d+\s*\+?\s*(?:年|years?)|工作经验|experience", lowered):
            return RequirementCategory.EXPERIENCE
        if re.search(r"沟通|协作|团队|表达|communication|collaboration|leadership", lowered):
            return RequirementCategory.SOFT_SKILL
        if re.search(r"负责|职责|参与|设计|推动|responsib|deliver|build|develop", lowered):
            return RequirementCategory.RESPONSIBILITY
        return RequirementCategory.HARD_SKILL
