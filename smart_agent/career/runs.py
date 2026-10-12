"""Lifecycle service for isolated resume-generation tasks."""

from __future__ import annotations

import time

from smart_agent.career.models import (
    EvidenceMatch,
    JDDocument,
    PersonalEvidence,
    ResumeDocument,
    ResumeDocumentStatus,
    ResumePlan,
    ResumeRun,
    ResumeRunStatus,
    VerificationReport,
    VerificationStatus,
    create_record_id,
    version_ref,
)
from smart_agent.career.store import CareerStore


class ResumeRunService:
    def __init__(self, store: CareerStore) -> None:
        self.store = store

    def create(
        self,
        *,
        jd: JDDocument,
        evidence: tuple[PersonalEvidence, ...],
        material_catalog_version: str,
    ) -> ResumeRun:
        run = ResumeRun(
            resume_run_id=create_record_id("ResumeRun"),
            jd_ref=version_ref(jd),
            requirement_refs=jd.requirement_refs,
            evidence_snapshot_refs=tuple(
                version_ref(record)
                for record in evidence
                if any(fact.is_usable for fact in record.facts)
            ),
            material_catalog_version=material_catalog_version,
            source_refs=jd.source_refs,
        )
        self.store.put(run)
        return run

    def attach_matches(self, run_id: str, matches: tuple[EvidenceMatch, ...]) -> ResumeRun:
        self._require_active(run_id)
        return self.store.revise_record(
            run_id,
            actor="career_requirement_matcher",
            reason="Attach requirement-level evidence matches",
            changes={
                "match_refs": tuple(version_ref(match) for match in matches),
                "status": ResumeRunStatus.MATCHED,
            },
        )

    def refresh_project_evidence_snapshot(
        self,
        run_id: str,
        *,
        evidence: tuple[PersonalEvidence, ...],
        material_catalog_version: str,
    ) -> ResumeRun:
        """Refresh project-backed evidence before rebuilding a resume.

        Project-library changes invalidate requirement matches and downstream
        plans, while unrelated personal evidence remains pinned to the run.
        """
        current = self._require_active(run_id)
        refs = tuple(version_ref(record) for record in evidence)
        catalog = str(material_catalog_version or "").strip().lower()
        if current.evidence_snapshot_refs == refs and current.material_catalog_version == catalog:
            return current
        return self.store.revise_record(
            run_id,
            actor="career_project_snapshot_sync",
            reason="Refresh project evidence before resume generation",
            changes={
                "evidence_snapshot_refs": refs,
                "material_catalog_version": catalog,
                "match_refs": (),
                "plan_ref": None,
                "resume_document_ref": None,
                "verification_report_ref": None,
                "rendered_artifacts": (),
                "status": ResumeRunStatus.CREATED,
            },
        )

    def attach_plan(self, run_id: str, plan: ResumePlan) -> ResumeRun:
        self._require_active(run_id)
        return self.store.revise_record(
            run_id,
            actor="career_resume_planner",
            reason="Attach resume plan",
            changes={"plan_ref": version_ref(plan), "status": ResumeRunStatus.PLANNED},
        )

    def attach_document(self, run_id: str, document: ResumeDocument) -> ResumeRun:
        current = self._require_active(run_id)
        document_ref = version_ref(document)
        # A version switch may select an already verified document. Preserve
        # that document's verification contract instead of downgrading the
        # whole run to GENERATED and forcing a second verification pass.
        verified = document.status is ResumeDocumentStatus.VERIFIED
        document_report_ref = document.verification_report_ref if verified else None
        if current.resume_document_ref == document_ref:
            if not verified or (
                current.status in {ResumeRunStatus.VERIFIED, ResumeRunStatus.RENDERED}
                and current.verification_report_ref == document_report_ref
            ):
                return current
        return self.store.revise_record(
            run_id,
            actor="career_resume_generator",
            reason="Attach generated resume",
            changes={
                "resume_document_ref": document_ref,
                "verification_report_ref": (
                    version_ref(
                        self.store.get(
                            document_report_ref.record_id,
                            document_report_ref.revision,
                            expected_type=VerificationReport,
                        )
                    )
                    if document_report_ref is not None
                    else None
                ),
                "rendered_artifacts": (),
                "status": ResumeRunStatus.VERIFIED if verified else ResumeRunStatus.GENERATED,
            },
        )

    def attach_verification(self, run_id: str, report: VerificationReport) -> ResumeRun:
        current = self._require_active(run_id)
        if current.resume_document_ref is not None and (
            current.resume_document_ref.record_id != report.resume_ref.record_id
        ):
            raise ValueError("Verification report belongs to a different resume document.")
        # Verification transitions the document itself to a new immutable
        # revision. Keep the run pinned to that transitioned revision so the
        # selected/current resume is also the one used by interviews.
        document = self.store.get(report.resume_ref.record_id, expected_type=ResumeDocument)
        return self.store.revise_record(
            run_id,
            actor="career_resume_verifier",
            reason="Attach independent verification report",
            changes={
                "resume_document_ref": version_ref(document),
                "verification_report_ref": version_ref(report),
                "status": (
                    ResumeRunStatus.VERIFIED
                    if report.status is VerificationStatus.PASSED
                    else ResumeRunStatus.GENERATED
                ),
            },
        )

    def attach_rendered_artifact(self, run_id: str, path: str) -> ResumeRun:
        current = self._require_active(run_id)
        if current.status not in {ResumeRunStatus.VERIFIED, ResumeRunStatus.RENDERED}:
            raise ValueError("ResumeRun must pass verification before PDF rendering.")
        artifact = str(path or "").strip()
        if not artifact:
            raise ValueError("Rendered artifact path is required.")
        if artifact in current.rendered_artifacts:
            return current
        return self.store.revise_record(
            run_id,
            actor="career_resume_renderer",
            reason="Attach rendered resume artifact",
            changes={
                "rendered_artifacts": (*current.rendered_artifacts, artifact),
                "status": ResumeRunStatus.RENDERED,
            },
        )

    def append_customization_turn(
        self,
        run_id: str,
        *,
        user_message: str,
        assistant_message: str,
        resume_document_id: str | None = None,
    ) -> ResumeRun:
        """Persist one resume-scoped user/agent exchange without changing facts."""
        current = self._require_active(run_id)
        user_text = str(user_message or "").strip()
        assistant_text = str(assistant_message or "").strip()
        if not user_text or not assistant_text:
            raise ValueError("Resume customization turns require user and assistant messages.")
        document_id = str(resume_document_id or "").strip()
        now = time.time()
        metadata = {"resume_document_id": document_id} if document_id else {}
        messages = (
            *current.customization_messages,
            {"role": "user", "content": user_text, "created_at": now, **metadata},
            {"role": "assistant", "content": assistant_text, "created_at": now, **metadata},
        )[-40:]
        return self.store.revise_record(
            run_id,
            actor="career_resume_customization_agent",
            reason="Apply user resume customization request",
            changes={"customization_messages": messages},
        )

    def delete(self, run_id: str) -> ResumeRun:
        """Hide one task while preserving its JD requirements and match caches."""
        current = self.store.get(run_id, expected_type=ResumeRun)
        if current.status is ResumeRunStatus.DELETED:
            return current
        return self.store.revise_record(
            run_id,
            actor="career_session_manager",
            reason="User deleted resume session; reusable requirement matches are retained",
            changes={"status": ResumeRunStatus.DELETED},
        )

    def _require_active(self, run_id: str) -> ResumeRun:
        current = self.store.get(run_id, expected_type=ResumeRun)
        if current.status is ResumeRunStatus.DELETED:
            raise ValueError("ResumeRun has been deleted.")
        return current
