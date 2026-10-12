"""Independent, artifact-scoped verification for generated resumes."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from smart_agent.agents import AgentArtifact, AgentTask, SubagentRunner, SubagentSpec
from smart_agent.career.models import (
    ClaimStatus,
    JDRequirement,
    ResumeClaim,
    ResumeDocument,
    ResumeDocumentStatus,
    ResumeItem,
    ResumePlan,
    VerificationReport,
    VerificationStatus,
    VersionRef,
    version_ref,
)
from smart_agent.career.resume import ModelAPIIdentity
from smart_agent.career.store import CareerStore
from smart_agent.tooling import ToolRegistry

_JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_SEVERITIES = frozenset({"critical", "major", "minor", "info"})
_CATEGORIES = frozenset(
    {
        "unsupported_claim",
        "factual_mismatch",
        "identity",
        "chronology",
        "jd_alignment",
        "missing_evidence",
        "format",
        "other",
    }
)


def _decode_verifier_json(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    match = _JSON_FENCE.match(raw)
    if match:
        raw = match.group(1).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        value = None
        for offset, character in enumerate(raw):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(raw[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                value = candidate
                break
        if value is None:
            raise ValueError("Resume Verification Agent must return one valid JSON object.") from exc
    if not isinstance(value, dict):
        raise ValueError("Resume Verification Agent must return one valid JSON object.")
    return value

RESUME_VERIFICATION_SPEC = SubagentSpec(
    agent_type="resume_verification",
    display_name="Resume Verification Agent",
    domain="career",
    system_prompt=(
        "You are an independent resume verification agent. Check every resume claim against only the supplied "
        "JD, ResumePlan, selected items, and cited evidence facts. Check company names, official titles, dates, "
        "skills, metrics, project facts, chronology, and JD alignment only for requirements explicitly linked to a "
        "resume claim. JD requirements are optional filters: do not report missing_evidence or jd_alignment merely "
        "because the resume does not cover a JD requirement. A claim with no requirement_ids, such as an internship "
        "summary, must be checked for factual support but must not be rejected for lacking JD coverage. Never infer "
        "a fact absent from the artifact and never rewrite the resume. Return JSON only with one key named issues. "
        "Each issue must contain severity "
        "(critical, major, minor, or info), category (unsupported_claim, factual_mismatch, identity, chronology, "
        "jd_alignment, missing_evidence, format, or other), message, claim_ids, evidence_fact_ids, requirement_ids, "
        "and suggested_action. Return an empty issues array only when every claim is supported."
    ),
    allowed_tools=frozenset(),
    max_steps=1,
    timeout_seconds=120.0,
    read_only=True,
)


@dataclass(frozen=True, slots=True)
class VerifierModelConfig:
    provider: str
    model: str
    base_url: str
    api_key: str

    @classmethod
    def from_runtime(cls, runtime: dict[str, Any]) -> "VerifierModelConfig":
        provider = str(
            runtime.get("careerVerifierProvider")
            or os.environ.get("CAREER_VERIFIER_PROVIDER", "")
        ).strip().lower()
        model = str(
            runtime.get("careerVerifierModel")
            or os.environ.get("CAREER_VERIFIER_MODEL", "")
        ).strip()
        base_url = str(
            runtime.get("careerVerifierBaseUrl")
            or os.environ.get("CAREER_VERIFIER_BASE_URL", "")
        ).strip()
        api_key = str(
            runtime.get("careerVerifierApiKey")
            or os.environ.get("CAREER_VERIFIER_API_KEY", "")
        ).strip()
        # The web baseline can run with a configured DeepSeek generation
        # channel but no separate verifier settings. Use a different model
        # and the provider's OpenAI-compatible endpoint in that case.
        if not provider:
            generation_model = str(runtime.get("model") or "").strip().casefold()
            generation_url = str(runtime.get("baseUrl") or "").strip().lower()
            generation_key = str(runtime.get("authToken") or runtime.get("apiKey") or "").strip()
            if (
                generation_model == "deepseek-chat"
                and "api.deepseek.com" in generation_url
                and generation_key
            ):
                provider = "custom"
                model = "deepseek-reasoner"
                base_url = "https://api.deepseek.com/v1"
                api_key = generation_key
        if provider not in {"anthropic", "openai", "openrouter", "custom"}:
            raise ValueError(
                "CAREER_VERIFIER_PROVIDER must be anthropic, openai, openrouter, or custom."
            )
        if not model or not base_url or not api_key:
            raise ValueError(
                "Independent resume verification requires CAREER_VERIFIER_MODEL, "
                "CAREER_VERIFIER_BASE_URL, and CAREER_VERIFIER_API_KEY."
            )
        return cls(provider, model, base_url.rstrip("/"), api_key)

    def runtime(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": self.model,
            "providerOverride": self.provider,
            "runtimeProfile": "harness",
        }
        if self.provider == "anthropic":
            result.update({"baseUrl": self.base_url, "apiKey": self.api_key})
        elif self.provider == "openai":
            result.update({"openaiBaseUrl": self.base_url, "openaiApiKey": self.api_key})
        elif self.provider == "openrouter":
            result.update({"openrouterBaseUrl": self.base_url, "openrouterApiKey": self.api_key})
        else:
            result.update({"customBaseUrl": self.base_url, "customApiKey": self.api_key})
        return result

    @property
    def identity(self) -> ModelAPIIdentity:
        return ModelAPIIdentity.from_runtime(self.runtime())


@dataclass(frozen=True, slots=True)
class ResumeVerificationResult:
    report: VerificationReport
    document: ResumeDocument
    input_artifact: AgentArtifact
    child_session_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.report.to_dict(),
            "document": self.document.to_dict(),
            "input_artifact": self.input_artifact.to_dict(),
            "child_session_id": self.child_session_id,
        }


def _apply_verification_report(
    store: CareerStore,
    resume_document_id: str,
    report: VerificationReport,
    *,
    actor: str,
) -> ResumeDocument:
    if report.status is VerificationStatus.PASSED:
        current = store.get(resume_document_id, expected_type=ResumeDocument)
        for ref in current.claim_refs:
            claim = store.get(ref.record_id, expected_type=ResumeClaim)
            if claim.status is not ClaimStatus.SUPPORTED:
                store.revise_record(
                    claim.claim_id,
                    actor=actor,
                    reason=f"Passed verification {report.verification_report_id}",
                    changes={"status": ClaimStatus.SUPPORTED},
                )
        current = store.get(resume_document_id, expected_type=ResumeDocument)
        latest_refs = tuple(
            version_ref(store.get(ref.record_id, expected_type=ResumeClaim))
            for ref in current.claim_refs
        )
        return store.revise_record(
            resume_document_id,
            actor=actor,
            reason=f"Passed verification {report.verification_report_id}",
            changes={
                "claim_refs": latest_refs,
                "status": ResumeDocumentStatus.VERIFIED,
                "verification_report_ref": version_ref(report),
            },
        )
    current = store.get(resume_document_id, expected_type=ResumeDocument)
    return store.revise_record(
        resume_document_id,
        actor=actor,
        reason=f"Verification found issues in {report.verification_report_id}",
        changes={
            "status": ResumeDocumentStatus.NEEDS_REVERIFICATION,
            "verification_report_ref": version_ref(report),
        },
    )


class DeterministicResumeEvidenceVerifier:
    """Local fallback that verifies citation integrity, not semantic JD fit."""

    provider = "local"
    model = "deterministic-evidence-integrity-v1"
    api_id = "api:local-deterministic-evidence"

    def __init__(self, store: CareerStore) -> None:
        self.store = store

    def verify(self, resume_document_id: str, *, parent_session_id: str = "") -> ResumeVerificationResult:
        document = self.store.get(resume_document_id, expected_type=ResumeDocument)
        replay = self.store.replay_resume(document.resume_document_id, document.revision)
        issues: list[dict[str, Any]] = []
        checked_refs: dict[tuple[str, int], VersionRef] = {}
        for claim in replay.claims:
            available_facts = {}
            for evidence_ref in claim.evidence_refs:
                checked_refs[(evidence_ref.record_id, evidence_ref.revision)] = evidence_ref
                evidence = self.store.get(evidence_ref.record_id, evidence_ref.revision)
                for fact in getattr(evidence, "facts", ()):
                    available_facts[fact.fact_id] = fact
            if not str(claim.text or "").strip():
                issues.append(self._issue("critical", "unsupported_claim", "Resume claim is empty.", claim))
            for fact_id in claim.evidence_fact_ids:
                fact = available_facts.get(fact_id)
                if fact is None:
                    issues.append(
                        self._issue(
                            "critical",
                            "missing_evidence",
                            f"Claim cites a fact outside its evidence records: {fact_id}.",
                            claim,
                            evidence_fact_ids=[fact_id],
                        )
                    )
                elif not fact.is_usable:
                    issues.append(
                        self._issue(
                            "critical",
                            "missing_evidence",
                            f"Claim cites an unverified or invalidated fact: {fact_id}.",
                            claim,
                            evidence_fact_ids=[fact_id],
                        )
                    )
        normalized_issues = tuple({**issue, "issue_id": f"issue-{index + 1}"} for index, issue in enumerate(issues))
        status = VerificationStatus.PASSED if not normalized_issues else VerificationStatus.ISSUES_FOUND
        artifact = AgentArtifact.create(
            "resume_deterministic_verification_input",
            domain="career",
            session_id=parent_session_id or "career-session",
            created_by="career_agent",
            content={
                "resume_document_id": document.resume_document_id,
                "revision": document.revision,
                "claim_ids": [claim.claim_id for claim in replay.claims],
            },
        )
        report = VerificationReport(
            verification_report_id=self._report_id(document, normalized_issues),
            resume_ref=version_ref(document),
            verifier_agent="deterministic_resume_evidence_verifier",
            provider=self.provider,
            model=self.model,
            status=status,
            issues=normalized_issues,
            checked_evidence_refs=tuple(checked_refs.values()),
            verifier_api_id=self.api_id,
            child_session_id="",
            input_artifact_id=artifact.artifact_id,
            input_artifact_hash=artifact.content_hash,
            source_refs=document.source_refs,
        )
        self.store.put(report)
        transitioned = _apply_verification_report(
            self.store,
            document.resume_document_id,
            report,
            actor="deterministic_resume_evidence_verifier",
        )
        return ResumeVerificationResult(report, transitioned, artifact, "")

    @staticmethod
    def _issue(
        severity: str,
        category: str,
        message: str,
        claim: ResumeClaim,
        *,
        evidence_fact_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "severity": severity,
            "category": category,
            "message": message,
            "claim_ids": [claim.claim_id],
            "evidence_fact_ids": list(evidence_fact_ids or claim.evidence_fact_ids),
            "requirement_ids": list(claim.requirement_ids),
            "suggested_action": "Regenerate or reconfirm the cited evidence before rendering.",
        }

    def _report_id(self, document: ResumeDocument, issues: tuple[dict[str, Any], ...]) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "resume": [document.resume_document_id, document.revision, document.content_hash],
                    "verifier": [self.provider, self.model, self.api_id],
                    "issues": issues,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return f"verification-{digest[:16]}"


class ResumeVerificationAgent:
    """Invoke a different API in a clean child session and persist its verdict."""

    def __init__(
        self,
        *,
        store: CareerStore,
        cwd: str,
        verifier_config: VerifierModelConfig,
        runner: SubagentRunner | None = None,
    ) -> None:
        self.store = store
        self.cwd = cwd
        self.verifier_config = verifier_config
        self.verifier_identity = verifier_config.identity
        self.runner = runner or SubagentRunner(
            cwd=cwd,
            runtime=verifier_config.runtime(),
            tools=ToolRegistry([]),
        )

    def verify(
        self,
        resume_document_id: str,
        *,
        parent_session_id: str,
    ) -> ResumeVerificationResult:
        document = self.store.get(resume_document_id, expected_type=ResumeDocument)
        self._assert_independent(document)
        artifact = self._artifact(document, parent_session_id)
        task = AgentTask.create(
            "resume_verification",
            domain="career",
            description="Verify generated resume claims against immutable JD and personal evidence.",
            parent_session_id=parent_session_id,
            assigned_agent="resume_verification",
            input_artifact_ids=[artifact.artifact_id],
            allowed_tools=RESUME_VERIFICATION_SPEC.allowed_tools,
            timeout_seconds=RESUME_VERIFICATION_SPEC.timeout_seconds,
            max_steps=RESUME_VERIFICATION_SPEC.max_steps,
        )
        run = self.runner.run(task, [artifact], RESUME_VERIFICATION_SPEC)
        if run.status != "completed" or run.output_artifact is None:
            raise RuntimeError(run.error or f"Resume Verification Agent ended with status={run.status}.")
        issues = self._parse_issues(run.output_artifact.content.get("text"), document)
        status = (
            VerificationStatus.PASSED
            if not issues
            else (
                VerificationStatus.NEEDS_USER_CONFIRMATION
                if document.revision_round >= 2
                else VerificationStatus.ISSUES_FOUND
            )
        )
        report = VerificationReport(
            verification_report_id=self._report_id(document, artifact, issues),
            resume_ref=version_ref(document),
            verifier_agent="resume_verification",
            provider=self.verifier_identity.provider,
            model=self.verifier_identity.model,
            status=status,
            issues=issues,
            checked_evidence_refs=self._checked_evidence_refs(document),
            verifier_api_id=self.verifier_identity.api_id,
            child_session_id=task.child_session_id,
            input_artifact_id=artifact.artifact_id,
            input_artifact_hash=artifact.content_hash,
            source_refs=document.source_refs,
        )
        self.store.put(report)
        transitioned = self._apply_report(document.resume_document_id, report)
        return ResumeVerificationResult(report, transitioned, artifact, task.child_session_id)

    def _assert_independent(self, document: ResumeDocument) -> None:
        if not document.generator_model or not document.generator_api_id:
            raise ValueError("ResumeDocument does not record a generator model/API identity.")
        if document.generator_model.casefold() == self.verifier_identity.model.casefold():
            raise ValueError("Resume generator and verifier must use different models.")
        if document.generator_api_id == self.verifier_identity.api_id:
            raise ValueError("Resume generator and verifier must use different API channels.")

    def _artifact(self, document: ResumeDocument, parent_session_id: str) -> AgentArtifact:
        replay = self.store.replay_resume(document.resume_document_id, document.revision)
        requirements = tuple(
            self.store.get(ref.record_id, ref.revision, expected_type=JDRequirement)
            for ref in replay.jd.requirement_refs
        )
        linked_requirement_ids = {
            requirement_ref.record_id
            for claim in replay.claims
            for requirement_ref in claim.requirement_refs
        }
        cited_fact_ids = {fact_id for claim in replay.claims for fact_id in claim.evidence_fact_ids}
        evidence_payload = [
            {
                "evidence_id": evidence.evidence_id,
                "evidence_type": evidence.evidence_type.value,
                "label": evidence.label,
                "facts": [
                    {
                        "fact_id": fact.fact_id,
                        "field_name": fact.field_name,
                        "value": fact.value,
                        "status": fact.status.value,
                        "source_refs": [
                            {
                                "source_id": source.source_id,
                                "locator": source.locator,
                                "excerpt": source.excerpt,
                            }
                            for source in fact.source_refs
                        ],
                    }
                    for fact in evidence.facts
                    if fact.fact_id in cited_fact_ids
                ],
            }
            for evidence in replay.evidence
        ]
        selected = set(replay.plan.selected_item_ids)
        content = {
            "jd": {
                "jd_id": replay.jd.jd_id,
                "company": replay.jd.company,
                "role_title": replay.jd.role_title,
                "raw_text": replay.jd.raw_text,
                "requirements": [
                    {
                        "requirement_id": requirement.requirement_id,
                        "category": requirement.category.value,
                        "text": requirement.text,
                        "original_text": requirement.original_text,
                        "required": requirement.required,
                    }
                    for requirement in requirements
                    if requirement.active and requirement.requirement_id in linked_requirement_ids
                ],
                "coverage_policy": {
                    "unlinked_jd_requirements": "ignore",
                    "only_check_linked_requirements": True,
                },
            },
            "plan": {
                "plan_id": replay.plan.plan_id,
                "selected_item_ids": list(replay.plan.selected_item_ids),
                "section_order": list(replay.plan.section_order),
                "constraints": replay.plan.constraints,
                "selected_items": [
                    {
                        "item_id": item.item_id,
                        "section": item.section,
                        "proposed_text": item.proposed_text,
                        "evidence_fact_ids": list(item.evidence_fact_ids),
                        "requirement_ids": list(item.requirement_ids),
                    }
                    for item in replay.items
                    if item.item_id in selected
                ],
            },
            "resume": {
                "resume_document_id": document.resume_document_id,
                "revision": document.revision,
                "revision_round": document.revision_round,
                "sections": document.sections,
                "claims": [
                    {
                        "claim_id": claim.claim_id,
                        "text": claim.text,
                        "evidence_ids": list(claim.evidence_ids),
                        "evidence_fact_ids": list(claim.evidence_fact_ids),
                        "requirement_ids": list(claim.requirement_ids),
                    }
                    for claim in replay.claims
                ],
            },
            "evidence": evidence_payload,
        }
        return AgentArtifact.create(
            "resume_verification_input",
            domain="career",
            session_id=parent_session_id,
            created_by="career_agent",
            content=content,
        )

    def _parse_issues(self, text: Any, document: ResumeDocument) -> tuple[dict[str, Any], ...]:
        payload = _decode_verifier_json(text)
        if not isinstance(payload, dict) or not isinstance(payload.get("issues"), list):
            raise ValueError("Resume Verification Agent output must contain an issues array.")
        known_claims = {ref.record_id for ref in document.claim_refs}
        claim_requirements = {
            claim.claim_id: {ref.record_id for ref in claim.requirement_refs}
            for ref in document.claim_refs
            for claim in (self.store.get(ref.record_id, ref.revision, expected_type=ResumeClaim),)
        }
        normalized: list[dict[str, Any]] = []
        for index, value in enumerate(payload["issues"]):
            if not isinstance(value, dict):
                raise ValueError("Every verification issue must be an object.")
            severity = str(value.get("severity") or "").strip().lower()
            category = str(value.get("category") or "other").strip().lower()
            message = " ".join(str(value.get("message") or "").split())
            if severity not in _SEVERITIES or category not in _CATEGORIES or not message:
                raise ValueError("Verification issues require valid severity, category, and message.")
            claim_ids = _string_list(value.get("claim_ids", []))
            if not set(claim_ids).issubset(known_claims):
                raise ValueError("Verification issue cites an unknown resume claim.")
            if category in {"missing_evidence", "jd_alignment"}:
                linked_claims = set().union(*(claim_requirements.get(claim_id, set()) for claim_id in claim_ids))
                if not claim_ids or not linked_claims:
                    continue
            normalized.append(
                {
                    "issue_id": f"issue-{index + 1}",
                    "severity": severity,
                    "category": category,
                    "message": message,
                    "claim_ids": list(claim_ids),
                    "evidence_fact_ids": list(_string_list(value.get("evidence_fact_ids", []))),
                    "requirement_ids": list(_string_list(value.get("requirement_ids", []))),
                    "suggested_action": " ".join(str(value.get("suggested_action") or "").split()),
                }
            )
        return tuple(normalized)

    def _checked_evidence_refs(self, document: ResumeDocument) -> tuple[VersionRef, ...]:
        refs: dict[tuple[str, int], VersionRef] = {}
        for claim_ref in document.claim_refs:
            claim = self.store.get(claim_ref.record_id, claim_ref.revision, expected_type=ResumeClaim)
            for ref in claim.evidence_refs:
                refs[(ref.record_id, ref.revision)] = ref
        return tuple(refs.values())

    def _report_id(
        self,
        document: ResumeDocument,
        artifact: AgentArtifact,
        issues: tuple[dict[str, Any], ...],
    ) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "resume": [document.resume_document_id, document.revision],
                    "verifier": [self.verifier_identity.model, self.verifier_identity.api_id],
                    "artifact": artifact.content_hash,
                    "issues": issues,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return f"verification-{digest[:16]}"

    def _apply_report(self, resume_document_id: str, report: VerificationReport) -> ResumeDocument:
        return _apply_verification_report(
            self.store,
            resume_document_id,
            report,
            actor="resume_verification",
        )


def _string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("Verification issue reference fields must be arrays.")
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
