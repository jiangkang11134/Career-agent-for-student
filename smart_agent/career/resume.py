"""Evidence-constrained construction of structured resume documents."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from smart_agent.career.models import (
    RESUME_SECTION_ORDER,
    ClaimStatus,
    PersonalEvidence,
    ResumeClaim,
    ResumeDocument,
    ResumeDocumentStatus,
    ResumeHeader,
    ResumeItem,
    ResumePlan,
    ResumeTemplate,
    SourceReference,
    SourceType,
    VerificationReport,
    VerificationStatus,
    normalize_resume_section,
    record_id,
    record_to_dict,
    version_ref,
)
from smart_agent.career.resume_templates import DEFAULT_LAYOUT_SPEC, sanitize_layout_spec
from smart_agent.career.store import CareerStore, RecordNotFoundError, RevisionConflictError
from smart_agent.model_registry import build_provider_config

_NUMBER_PATTERN = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?")
_STRUCTURAL_LIST_MARKER = re.compile(
    r"(^|[\s;；:：])"
    r"(?:(?:[（(]\s*[1-9]\d?\s*[)）])|(?:[1-9]\d?\s*(?:[、)）]|[.:](?=\s|$)|：)))\s*"
)
_PROJECT_DESIGN_TEXT = re.compile(r"项目设计\s*[:：]\s*(.*)$")
_PROJECT_DESIGN_ITEM = re.compile(r"(?<!\d)(\d{1,2})\s*、\s*")
PROJECT_CLAIM_UNLIMITED = 0
SKILLS_CLAIM_MAX_LENGTH = 1600


def project_claim_max_length(project_count: int) -> int:
    """Return a compatibility limit large enough to represent unlimited projects."""
    return PROJECT_CLAIM_UNLIMITED


def _stable_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{prefix}-{digest[:16]}"


def _single_line(value: Any, field_name: str, *, max_length: int = 600) -> str:
    text = " ".join(str(value or "").split())
    text = text.lstrip("-*• ").strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    if max_length > 0 and len(text) > max_length:
        raise ValueError(f"{field_name} exceeds {max_length} characters.")
    return text


def _preserve_block_text(value: Any, field_name: str, *, max_length: int = 1600) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    if max_length > 0 and len(text) > max_length:
        raise ValueError(f"{field_name} exceeds {max_length} characters.")
    return text


def _without_structural_list_markers(value: str) -> str:
    """Remove list numbering while retaining factual numbers for evidence checks."""
    text = value
    while True:
        normalized = _STRUCTURAL_LIST_MARKER.sub(r"\1", text)
        if normalized == text:
            return normalized
        text = normalized


def _validate_project_design_structure(text: str) -> None:
    """Reject project claims whose rendered design list exceeds five points."""
    match = _PROJECT_DESIGN_TEXT.search(str(text or ""))
    if not match:
        return
    items = list(_PROJECT_DESIGN_ITEM.finditer(match.group(1)))
    numbers = [int(item.group(1)) for item in items]
    if not 1 <= len(items) <= 5 or numbers != list(range(1, len(items) + 1)):
        raise ValueError(
            "Project design must contain sequential points 1、 through 5、 (at most five)."
        )


@dataclass(frozen=True, slots=True)
class ModelAPIIdentity:
    """Non-secret model and API-channel identity used to prevent self-review."""

    provider: str
    model: str
    api_id: str

    @classmethod
    def from_runtime(cls, runtime: dict[str, Any]) -> "ModelAPIIdentity":
        model = str(runtime.get("model") or "").strip()
        if not model:
            raise ValueError("The active generation model is not configured.")
        config = build_provider_config(model, runtime)
        endpoint = config.base_url.rstrip("/").lower()
        if not endpoint:
            raise ValueError("The active generation API endpoint is not configured.")
        channel_hash = hashlib.sha256(endpoint.encode()).hexdigest()[:16]
        return cls(config.provider.value, config.model, f"api:{channel_hash}")


@dataclass(frozen=True, slots=True)
class ResumeBuildResult:
    document: ResumeDocument
    claims: tuple[ResumeClaim, ...]

    def persist(self, store: CareerStore) -> "ResumeBuildResult":
        persisted_claims = tuple(_put_idempotent(store, claim) for claim in self.claims)
        persisted_document = _put_idempotent(store, self.document)
        return ResumeBuildResult(persisted_document, persisted_claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "claims": [claim.to_dict() for claim in self.claims],
        }


def _put_idempotent(store: CareerStore, record):
    try:
        current = store.get(record_id(record))
    except RecordNotFoundError:
        return store.put(record)
    if _idempotent_payload(current) == _idempotent_payload(record):
        return current
    raise RevisionConflictError(
        f"A different {type(record).__name__} already exists with the deterministic identity."
    )


def _idempotent_payload(record) -> dict[str, Any]:
    payload = record_to_dict(record)
    for field_name in (
        "record_type",
        "revision",
        "history",
        "created_at",
        "updated_at",
        "content_hash",
    ):
        payload.pop(field_name, None)
    # Deterministic resume identities describe generated content. Verification
    # changes Claim/Document lifecycle fields and claim-reference revisions;
    # those transitions must not turn an exact replay into a conflict.
    if isinstance(record, ResumeClaim):
        payload.pop("status", None)
    elif isinstance(record, ResumeDocument):
        payload.pop("status", None)
        payload.pop("verification_report_ref", None)
        payload.pop("claim_refs", None)
    for source in payload.get("source_refs", []):
        source.pop("captured_at", None)
    return payload


class ResumeDocumentBuilder:
    """Turn selected ResumeItems into claims without accepting caller-supplied references."""

    def __init__(self, store: CareerStore, *, enforce_project_evidence: bool = True) -> None:
        self.store = store
        self.enforce_project_evidence = bool(enforce_project_evidence)

    def build(
        self,
        *,
        plan: ResumePlan,
        header: ResumeHeader,
        claim_drafts: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        generator: ModelAPIIdentity,
        template: ResumeTemplate | None = None,
    ) -> ResumeBuildResult:
        if not generator.provider.strip() or not generator.model.strip() or not generator.api_id.strip():
            raise ValueError("A complete generator model/API identity is required.")
        persisted_plan = self.store.get(plan.plan_id, plan.revision, expected_type=ResumePlan)
        if persisted_plan != plan:
            raise ValueError("ResumePlan does not match its immutable stored revision.")

        items = {
            ref.record_id: self.store.get(ref.record_id, ref.revision, expected_type=ResumeItem)
            for ref in plan.item_refs
        }
        selected_ids = tuple(plan.selected_item_ids)
        drafts = self._drafts_by_item(claim_drafts, selected_ids)
        project_count = sum(
            1
            for item_id in selected_ids
            if normalize_resume_section(items[item_id].section) == "projects"
        )
        project_max_length = project_claim_max_length(project_count)
        normalized_drafts = {}
        for item_id in selected_ids:
            item = items[item_id]
            max_length = (
                project_max_length
                if normalize_resume_section(item.section) == "projects"
                else SKILLS_CLAIM_MAX_LENGTH
                if normalize_resume_section(item.section) == "skills"
                else 600
            )
            if normalize_resume_section(item.section) == "projects" and not self.enforce_project_evidence:
                # Benchmark B2 keeps project formatting checks but bypasses the project evidence validator entirely.
                normalized_drafts[item_id] = _single_line(
                    drafts[item_id]["text"],
                    f"claim text for {item.item_id}",
                    max_length=max_length,
                )
            else:
                normalized_drafts[item_id] = self.validate_claim_text(
                    drafts[item_id]["text"],
                    item,
                    max_length=max_length,
                )
        layout_spec = sanitize_layout_spec(template.layout_spec if template else DEFAULT_LAYOUT_SPEC)
        resume_id = _stable_id(
            "resume",
            {
                "plan": [plan.plan_id, plan.revision],
                "header": {
                    "full_name": header.full_name,
                    "headline": header.headline,
                    "email": header.email,
                    "phone": header.phone,
                    "location": header.location,
                    "links": header.links,
                },
                "template": [template.template_id, template.revision] if template else "default",
                "layout_spec": layout_spec,
                "generator": [generator.provider, generator.model, generator.api_id],
                "claims": [[item_id, normalized_drafts[item_id]] for item_id in selected_ids],
            },
        )

        claims: list[ResumeClaim] = []
        sections: dict[str, list[str]] = {section: [] for section in RESUME_SECTION_ORDER}
        for item_id in selected_ids:
            item = items[item_id]
            text = normalized_drafts[item_id]
            claim = ResumeClaim(
                claim_id=_stable_id("claim", {"resume": resume_id, "item": item_id}),
                text=text,
                evidence_refs=item.evidence_refs,
                evidence_fact_ids=item.evidence_fact_ids,
                requirement_refs=item.requirement_refs,
                status=ClaimStatus.DRAFT,
                source_refs=item.source_refs,
            )
            claims.append(claim)
            sections[normalize_resume_section(item.section)].append(claim.claim_id)

        normalized_sections = {
            section: tuple(claim_ids)
            for section, claim_ids in sections.items()
        }
        if tuple(normalized_sections) != RESUME_SECTION_ORDER:
            raise ValueError("Resume sections do not follow the fixed six-section order.")

        header_payload = json.dumps(
            {
                "full_name": header.full_name,
                "headline": header.headline,
                "email": header.email,
                "phone": header.phone,
                "location": header.location,
                "links": header.links,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        header_source = SourceReference.create(
            source_id=f"resume-header:{resume_id}",
            source_type=SourceType.USER_INPUT,
            locator={"record": resume_id, "field": "header"},
            raw_content=header_payload,
            excerpt=header.full_name,
        )
        document = ResumeDocument(
            resume_document_id=resume_id,
            jd_ref=plan.jd_ref,
            plan_ref=version_ref(plan),
            claim_refs=tuple(version_ref(claim) for claim in claims),
            sections=normalized_sections,
            status=ResumeDocumentStatus.DRAFT,
            header=header,
            generator_provider=generator.provider,
            generator_model=generator.model,
            generator_api_id=generator.api_id,
            revision_round=0,
            source_refs=(*plan.source_refs, *(template.source_refs if template else ()), header_source),
            template_ref=None,
            layout_spec=layout_spec,
        )
        return ResumeBuildResult(document, tuple(claims))

    @staticmethod
    def _drafts_by_item(
        claim_drafts: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        selected_item_ids: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(claim_drafts, (list, tuple)) or not claim_drafts:
            raise ValueError("claim_drafts must contain one generated claim per selected ResumeItem.")
        normalized: dict[str, dict[str, Any]] = {}
        for draft in claim_drafts:
            if not isinstance(draft, dict):
                raise ValueError("Each claim draft must be an object.")
            item_id = str(draft.get("item_id") or "").strip()
            if not item_id or item_id in normalized:
                raise ValueError("Each claim draft requires a unique item_id.")
            normalized[item_id] = draft
        if set(normalized) != set(selected_item_ids):
            missing = sorted(set(selected_item_ids) - set(normalized))
            extra = sorted(set(normalized) - set(selected_item_ids))
            raise ValueError(f"Claim drafts must exactly match selected items; missing={missing}, extra={extra}.")
        return normalized

    def validate_claim_text(
        self,
        text: str,
        item: ResumeItem,
        *,
        max_length: int = 600,
    ) -> str:
        """Validate one draft before persistence so an LLM can repair it."""
        field_name = f"claim text for {item.item_id}"
        normalized = (
            _preserve_block_text(text, field_name)
            if normalize_resume_section(item.section) == "skills"
            else _single_line(text, field_name, max_length=max_length)
        )
        if normalize_resume_section(item.section) == "projects":
            _validate_project_design_structure(normalized)
        self._reject_unsupported_numbers(normalized, item)
        return normalized

    def _reject_unsupported_numbers(self, text: str, item: ResumeItem) -> None:
        # Ordered-list markers are presentation structure, not factual numbers.
        factual_text = _without_structural_list_markers(text)
        supported = _supported_numbers(
            self.store,
            item.evidence_refs,
            item.evidence_fact_ids,
        )
        unsupported = sorted(set(_NUMBER_PATTERN.findall(factual_text)) - supported)
        if unsupported:
            raise ValueError(
                f"Claim for {item.item_id} introduces numeric facts absent from cited evidence: {unsupported}"
            )


def _supported_numbers(store: CareerStore, evidence_refs, evidence_fact_ids) -> set[str]:
    evidence_text: list[str] = []
    for ref in evidence_refs:
        evidence = store.get(ref.record_id, ref.revision, expected_type=PersonalEvidence)
        evidence_text.extend(
            json.dumps(fact.value, ensure_ascii=False, sort_keys=True)
            for fact in evidence.facts
            if fact.fact_id in evidence_fact_ids
        )
    return set(_NUMBER_PATTERN.findall(" ".join(evidence_text)))


@dataclass(frozen=True, slots=True)
class ResumeRevisionResult:
    document: ResumeDocument
    revised_claims: tuple[ResumeClaim, ...]
    revision_round: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "revised_claims": [claim.to_dict() for claim in self.revised_claims],
            "revision_round": self.revision_round,
            "remaining_revision_rounds": 2 - self.revision_round,
        }


class ResumeRevisionManager:
    """Apply evidence-constrained text corrections after an independent report."""

    def __init__(self, store: CareerStore) -> None:
        self.store = store

    def revise(
        self,
        resume_document_id: str,
        *,
        verification_report_id: str,
        claim_revisions: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> ResumeRevisionResult:
        document = self.store.get(resume_document_id, expected_type=ResumeDocument)
        if document.status is not ResumeDocumentStatus.NEEDS_REVERIFICATION:
            raise ValueError("Only a resume with verification issues can be revised.")
        if document.revision_round >= 2:
            raise ValueError("The two automatic resume revision rounds are exhausted; user confirmation is required.")
        if document.verification_report_ref is None:
            raise ValueError("ResumeDocument does not reference the report that requested revision.")
        report = self.store.get(
            document.verification_report_ref.record_id,
            document.verification_report_ref.revision,
            expected_type=VerificationReport,
        )
        if report.verification_report_id != verification_report_id:
            raise ValueError("verification_report_id is not the current report for this resume.")
        if report.status is not VerificationStatus.ISSUES_FOUND:
            raise ValueError("This verification report cannot enter automatic revision.")

        affected_claims = {
            str(claim_id)
            for issue in report.issues
            for claim_id in issue.get("claim_ids", [])
            if str(claim_id).strip()
        }
        if not affected_claims:
            raise ValueError("The report has no claim-scoped correction; user confirmation is required.")
        revisions = self._normalize_revisions(claim_revisions)
        if not set(revisions).issubset(affected_claims):
            raise ValueError("Automatic revision may change only claims named by the verification report.")

        current_claim_ids = {ref.record_id for ref in document.claim_refs}
        if not set(revisions).issubset(current_claim_ids):
            raise ValueError("claim_revisions cites a claim outside the current resume.")
        revised: list[ResumeClaim] = []
        project_count = len(document.sections.get("projects", ()))
        project_max_length = project_claim_max_length(project_count)
        for claim_id, revision in revisions.items():
            claim = self.store.get(claim_id, expected_type=ResumeClaim)
            text = revision["text"]
            claim_section = normalize_resume_section(
                next(
                    (
                        section
                        for section, claim_ids in document.sections.items()
                        if claim_id in claim_ids
                    ),
                    "projects",
                )
            )
            normalized_text = _single_line(
                text,
                f"revised text for {claim_id}",
                max_length=(project_max_length if claim_section == "projects" else 600),
            )
            remove_requirement_ids = set(revision["remove_requirement_ids"])
            current_requirement_ids = {ref.record_id for ref in claim.requirement_refs}
            if not remove_requirement_ids.issubset(current_requirement_ids):
                raise ValueError(f"Revision for {claim_id} removes a requirement outside the current claim.")
            if normalized_text == claim.text and not remove_requirement_ids:
                raise ValueError(f"Revision for {claim_id} does not change the claim text.")
            unsupported = sorted(
                set(_NUMBER_PATTERN.findall(_without_structural_list_markers(normalized_text)))
                - _supported_numbers(self.store, claim.evidence_refs, claim.evidence_fact_ids)
            )
            if unsupported:
                raise ValueError(
                    f"Revision for {claim_id} introduces numeric facts absent from cited evidence: {unsupported}"
                )
            next_claim = self.store.revise_record(
                claim_id,
                actor="career_agent",
                reason=f"Correct claim using verification report {report.verification_report_id}",
                changes={
                    "text": normalized_text,
                    "status": ClaimStatus.DRAFT,
                    "requirement_refs": tuple(
                        ref for ref in claim.requirement_refs if ref.record_id not in remove_requirement_ids
                    ),
                },
            )
            revised.append(next_claim)

        current = self.store.get(resume_document_id, expected_type=ResumeDocument)
        latest_refs = tuple(
            version_ref(self.store.get(ref.record_id, expected_type=ResumeClaim))
            for ref in current.claim_refs
        )
        next_round = document.revision_round + 1
        next_document = self.store.revise_record(
            resume_document_id,
            actor="career_agent",
            reason=f"Applied evidence-constrained revision round {next_round}",
            changes={
                "claim_refs": latest_refs,
                "status": ResumeDocumentStatus.DRAFT,
                "revision_round": next_round,
                "verification_report_ref": None,
            },
        )
        return ResumeRevisionResult(next_document, tuple(revised), next_round)

    @staticmethod
    def _normalize_revisions(
        values: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("claim_revisions must be a non-empty array.")
        revisions: dict[str, dict[str, Any]] = {}
        for value in values:
            if not isinstance(value, dict):
                raise ValueError("Every claim revision must be an object.")
            claim_id = str(value.get("claim_id") or "").strip()
            if not claim_id or claim_id in revisions:
                raise ValueError("Every claim revision requires a unique claim_id.")
            remove_ids = value.get("remove_requirement_ids", [])
            if not isinstance(remove_ids, list):
                raise ValueError("remove_requirement_ids must be an array.")
            revisions[claim_id] = {
                "text": str(value.get("text") or ""),
                "remove_requirement_ids": tuple(str(item).strip() for item in remove_ids if str(item).strip()),
            }
        return revisions
