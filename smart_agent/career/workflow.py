"""Career runtime state index and artifact-gated workflow guidance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from smart_agent.career.interview import InterviewKnowledgeStore
from smart_agent.career.materials import (
    CandidateFactStatus,
    CandidateFactStore,
    MaterialCategory,
    MaterialStore,
    material_catalog_version,
)
from smart_agent.career.models import (
    EvidenceMatch,
    FactStatus,
    JDDocument,
    PersonalEvidence,
    ProjectDescriptionDocument,
    ProjectDescriptionStatus,
    RequirementMatchCache,
    RequirementMatchCacheStatus,
    ResumeDocument,
    ResumeDocumentStatus,
    ResumePlan,
    ResumeRun,
    ResumeRunStatus,
    ResumeSource,
    SupportLevel,
    VerificationReport,
    VerificationStatus,
    record_id,
)
from smart_agent.career.project_library import ProjectLibraryStore, ProjectSourceStatus
from smart_agent.career.store import CareerStore


class CareerGoal(str, Enum):
    RESUME = "resume"
    INTERVIEW = "interview"
    APPLICATION = "application"


class CareerWorkflowStage(str, Enum):
    MATERIALS = "materials"
    FACT_REVIEW = "fact_review"
    PROJECT_ANALYSIS = "project_analysis"
    PROJECT_CONFIRMATION = "project_confirmation"
    JD_ANALYSIS = "jd_analysis"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    RESUME_PLANNING = "resume_planning"
    RESUME_GENERATION = "resume_generation"
    VERIFICATION = "verification"
    PDF_RENDERING = "pdf_rendering"
    INTERVIEW = "interview"
    APPLICATION = "application"


@dataclass(frozen=True, slots=True)
class CareerArtifactIndex:
    material_count: int = 0
    material_counts_by_category: dict[str, int] = field(default_factory=dict)
    material_ids_without_candidates: tuple[str, ...] = ()
    resume_source_ids: tuple[str, ...] = ()
    candidate_fact_counts_by_status: dict[str, int] = field(default_factory=dict)
    unhandled_candidate_fact_ids: tuple[str, ...] = ()
    evidence_count: int = 0
    usable_evidence_count: int = 0
    unverified_evidence_count: int = 0
    usable_fact_count: int = 0
    unverified_fact_count: int = 0
    invalidated_fact_count: int = 0
    jd_ids: tuple[str, ...] = ()
    latest_jd_id: str = ""
    requirement_count: int = 0
    evidence_match_count: int = 0
    unsupported_requirement_count: int = 0
    resume_plan_ids: tuple[str, ...] = ()
    latest_resume_plan_id: str = ""
    resume_document_ids: tuple[str, ...] = ()
    latest_resume_document_id: str = ""
    verified_resume_document_ids: tuple[str, ...] = ()
    verification_report_ids: tuple[str, ...] = ()
    passed_verification_report_ids: tuple[str, ...] = ()
    issue_verification_report_ids: tuple[str, ...] = ()
    interview_corpus_count: int = 0
    interview_knowledge_chunk_count: int = 0
    rendered_pdf_count: int = 0
    latest_pdf_path: str = ""
    material_catalog_version: str = ""
    project_description_ids: tuple[str, ...] = ()
    project_descriptions_needing_confirmation: tuple[str, ...] = ()
    resume_run_ids: tuple[str, ...] = ()
    latest_resume_run_id: str = ""
    requirement_cache_count: int = 0
    fresh_requirement_cache_count: int = 0
    project_source_count: int = 0
    project_source_counts_by_status: dict[str, int] = field(default_factory=dict)
    project_ids_needing_analysis: tuple[str, ...] = ()
    unavailable_project_ids: tuple[str, ...] = ()

    @property
    def has_materials(self) -> bool:
        return self.material_count > 0

    @property
    def has_unhandled_candidate_facts(self) -> bool:
        return bool(self.unhandled_candidate_fact_ids)

    @property
    def has_usable_evidence(self) -> bool:
        return self.usable_evidence_count > 0 and self.usable_fact_count > 0

    @property
    def has_jd(self) -> bool:
        return bool(self.jd_ids) and self.requirement_count > 0

    @property
    def has_evidence_retrieval(self) -> bool:
        return self.evidence_match_count > 0

    @property
    def has_resume_plan(self) -> bool:
        return bool(self.resume_plan_ids)

    @property
    def has_resume_document(self) -> bool:
        return bool(self.resume_document_ids)

    @property
    def has_verified_resume(self) -> bool:
        return bool(self.verified_resume_document_ids)

    @property
    def has_rendered_pdf(self) -> bool:
        return self.rendered_pdf_count > 0

    @property
    def has_interview_knowledge(self) -> bool:
        return self.interview_knowledge_chunk_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_count": self.material_count,
            "material_counts_by_category": dict(self.material_counts_by_category),
            "material_ids_without_candidates": list(self.material_ids_without_candidates),
            "resume_source_ids": list(self.resume_source_ids),
            "candidate_fact_counts_by_status": dict(self.candidate_fact_counts_by_status),
            "unhandled_candidate_fact_ids": list(self.unhandled_candidate_fact_ids),
            "evidence_count": self.evidence_count,
            "usable_evidence_count": self.usable_evidence_count,
            "unverified_evidence_count": self.unverified_evidence_count,
            "usable_fact_count": self.usable_fact_count,
            "unverified_fact_count": self.unverified_fact_count,
            "invalidated_fact_count": self.invalidated_fact_count,
            "jd_ids": list(self.jd_ids),
            "latest_jd_id": self.latest_jd_id,
            "requirement_count": self.requirement_count,
            "evidence_match_count": self.evidence_match_count,
            "unsupported_requirement_count": self.unsupported_requirement_count,
            "resume_plan_ids": list(self.resume_plan_ids),
            "latest_resume_plan_id": self.latest_resume_plan_id,
            "resume_document_ids": list(self.resume_document_ids),
            "latest_resume_document_id": self.latest_resume_document_id,
            "verified_resume_document_ids": list(self.verified_resume_document_ids),
            "verification_report_ids": list(self.verification_report_ids),
            "passed_verification_report_ids": list(self.passed_verification_report_ids),
            "issue_verification_report_ids": list(self.issue_verification_report_ids),
            "interview_corpus_count": self.interview_corpus_count,
            "interview_knowledge_chunk_count": self.interview_knowledge_chunk_count,
            "rendered_pdf_count": self.rendered_pdf_count,
            "latest_pdf_path": self.latest_pdf_path,
            "material_catalog_version": self.material_catalog_version,
            "project_description_ids": list(self.project_description_ids),
            "project_descriptions_needing_confirmation": list(self.project_descriptions_needing_confirmation),
            "resume_run_ids": list(self.resume_run_ids),
            "latest_resume_run_id": self.latest_resume_run_id,
            "requirement_cache_count": self.requirement_cache_count,
            "fresh_requirement_cache_count": self.fresh_requirement_cache_count,
            "project_source_count": self.project_source_count,
            "project_source_counts_by_status": dict(self.project_source_counts_by_status),
            "project_ids_needing_analysis": list(self.project_ids_needing_analysis),
            "unavailable_project_ids": list(self.unavailable_project_ids),
        }


@dataclass(frozen=True, slots=True)
class CareerStageGate:
    stage: CareerWorkflowStage
    ready: bool
    missing: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    next_tool_names: tuple[str, ...] = ()
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "ready": self.ready,
            "missing": list(self.missing),
            "blocked_by": list(self.blocked_by),
            "next_tool_names": list(self.next_tool_names),
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class CareerWorkspaceState:
    goal: CareerGoal
    goal_ready: bool
    recommended_stage: CareerWorkflowStage
    recommended_domain_stage: str
    index: CareerArtifactIndex
    stage_gates: tuple[CareerStageGate, ...]
    next_actions: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal.value,
            "goal_ready": self.goal_ready,
            "recommended_stage": self.recommended_stage.value,
            "recommended_domain_stage": self.recommended_domain_stage,
            "index": self.index.to_dict(),
            "stage_gates": [gate.to_dict() for gate in self.stage_gates],
            "next_actions": list(self.next_actions),
            "warnings": list(self.warnings),
        }


def _ids(records: Iterable[Any]) -> tuple[str, ...]:
    return tuple(record_id(record) for record in records)


def _latest_id(records: Iterable[Any]) -> str:
    items = tuple(records)
    if not items:
        return ""
    latest = max(items, key=lambda record: (record.updated_at, record.revision, record_id(record)))
    return record_id(latest)


def _discover_pdfs(root: Path, *, resume_document_id: str = "") -> tuple[Path, ...]:
    if not root.exists():
        return ()
    matches: list[Path] = []
    for path in root.rglob("*.pdf"):
        trace_path = path.with_suffix(".trace.json")
        if not path.is_file() or not trace_path.is_file():
            continue
        if resume_document_id:
            try:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            resume_ref = trace.get("resume") if isinstance(trace, dict) else None
            if not isinstance(resume_ref, list) or not resume_ref or resume_ref[0] != resume_document_id:
                continue
        matches.append(path)
    return tuple(sorted(matches))


class CareerWorkspaceIndexer:
    """Build a compact, deterministic index over Career artifacts."""

    def __init__(
        self,
        *,
        career_store: CareerStore,
        material_store: MaterialStore,
        candidate_fact_store: CandidateFactStore,
        interview_knowledge_store: InterviewKnowledgeStore,
        interview_knowledge_library=None,
        project_library_store: ProjectLibraryStore | None = None,
        include_match_ledger: bool = True,
    ) -> None:
        self.career_store = career_store
        self.material_store = material_store
        self.candidate_fact_store = candidate_fact_store
        self.interview_knowledge_store = interview_knowledge_store
        self.interview_knowledge_library = interview_knowledge_library
        self.project_library_store = project_library_store
        self.include_match_ledger = bool(include_match_ledger)

    def build(self) -> CareerArtifactIndex:
        manifests = self.material_store.list_manifests()
        material_counts = {category.value: 0 for category in MaterialCategory}
        for manifest in manifests:
            material_counts[manifest.category.value] += 1

        candidates = self.candidate_fact_store.list()
        candidate_counts = {status.value: 0 for status in CandidateFactStatus}
        for fact in candidates:
            candidate_counts[fact.status.value] += 1
        candidate_material_ids = {fact.material_id for fact in candidates}

        active_material_ids = {manifest.material_id for manifest in manifests}
        resume_sources = tuple(
            source
            for source in self.career_store.list_latest(ResumeSource)
            if any(ref.source_id in active_material_ids for ref in source.source_refs)
        )
        evidence_records = self.career_store.list_latest(PersonalEvidence)
        jd_records = self.career_store.list_latest(JDDocument)
        all_matches = self.career_store.list_latest(EvidenceMatch) if self.include_match_ledger else ()
        all_plans = self.career_store.list_latest(ResumePlan)
        all_resume_documents = self.career_store.list_latest(ResumeDocument)
        all_reports = self.career_store.list_latest(VerificationReport)
        project_descriptions = self.career_store.list_latest(ProjectDescriptionDocument)
        resume_runs = tuple(
            record
            for record in self.career_store.list_latest(ResumeRun)
            if record.status is not ResumeRunStatus.DELETED
        )
        requirement_caches = self.career_store.list_latest(RequirementMatchCache)
        latest_resume_run_id = _latest_id(resume_runs)
        latest_resume_run = next(
            (record for record in resume_runs if record.resume_run_id == latest_resume_run_id),
            None,
        )
        latest_jd_id = latest_resume_run.jd_ref.record_id if latest_resume_run is not None else ""
        latest_jd = next((record for record in jd_records if record.jd_id == latest_jd_id), None)
        active_requirement_ids = {
            ref.record_id for ref in latest_jd.requirement_refs
        } if latest_jd is not None else set()
        matches = tuple(
            match
            for match in all_matches
            if match.requirement_ref.record_id in active_requirement_ids
        )
        plans = tuple(
            plan for plan in all_plans if plan.jd_ref.record_id == latest_jd_id
        )
        latest_plan_id = _latest_id(plans)
        resume_documents = tuple(
            document
            for document in all_resume_documents
            if document.jd_ref.record_id == latest_jd_id
            and document.plan_ref.record_id == latest_plan_id
        )
        latest_resume_document_id = _latest_id(resume_documents)
        latest_resume_document = next(
            (
                document
                for document in resume_documents
                if document.resume_document_id == latest_resume_document_id
            ),
            None,
        )
        reports = tuple(
            report
            for report in all_reports
            if report.resume_ref.record_id == latest_resume_document_id
        )

        usable_evidence_count = 0
        unverified_evidence_count = 0
        usable_fact_count = 0
        unverified_fact_count = 0
        invalidated_fact_count = 0
        for evidence in evidence_records:
            if any(fact.is_usable for fact in evidence.facts):
                usable_evidence_count += 1
            if any(fact.status is FactStatus.UNVERIFIED for fact in evidence.facts):
                unverified_evidence_count += 1
            usable_fact_count += sum(1 for fact in evidence.facts if fact.is_usable)
            unverified_fact_count += sum(1 for fact in evidence.facts if fact.status is FactStatus.UNVERIFIED)
            invalidated_fact_count += sum(1 for fact in evidence.facts if fact.status is FactStatus.INVALIDATED)

        corpus_metadata = self.interview_knowledge_store.list_metadata()
        project_sources = (
            tuple(
                # Workspace/artifact reads must stay cheap. Source manifests
                # are refreshed by the project-library endpoint and explicit
                # import/analysis flows; rescanning every project here makes
                # the main UI request block on large repositories.
                self.project_library_store.get(source.project_id, refresh=False)
                for source in self.project_library_store.list()
            )
            if self.project_library_store
            else ()
        )
        project_status_counts = {status.value: 0 for status in ProjectSourceStatus}
        for source in project_sources:
            project_status_counts[source.status.value] += 1
        pdfs = _discover_pdfs(
            self.career_store.root,
            resume_document_id=latest_resume_document_id,
        )
        catalog_version = material_catalog_version(manifests, evidence_records, candidates, project_descriptions)
        return CareerArtifactIndex(
            material_count=len(manifests),
            material_counts_by_category={key: value for key, value in material_counts.items() if value},
            material_ids_without_candidates=tuple(
                manifest.material_id
                for manifest in manifests
                if manifest.category is not MaterialCategory.RESUME
                and manifest.material_id not in candidate_material_ids
            ),
            resume_source_ids=_ids(resume_sources),
            candidate_fact_counts_by_status={key: value for key, value in candidate_counts.items() if value},
            unhandled_candidate_fact_ids=tuple(
                fact.candidate_fact_id
                for fact in candidates
                if fact.status is CandidateFactStatus.CANDIDATE
            ),
            evidence_count=len(evidence_records),
            usable_evidence_count=usable_evidence_count,
            unverified_evidence_count=unverified_evidence_count,
            usable_fact_count=usable_fact_count,
            unverified_fact_count=unverified_fact_count,
            invalidated_fact_count=invalidated_fact_count,
            jd_ids=_ids(jd_records),
            latest_jd_id=latest_jd_id,
            requirement_count=len(active_requirement_ids),
            evidence_match_count=len(matches),
            unsupported_requirement_count=sum(
                1 for match in matches if match.support_level is SupportLevel.UNSUPPORTED
            ),
            resume_plan_ids=_ids(plans),
            latest_resume_plan_id=latest_plan_id,
            resume_document_ids=_ids(resume_documents),
            latest_resume_document_id=latest_resume_document_id,
            verified_resume_document_ids=(
                (latest_resume_document_id,)
                if latest_resume_document is not None
                and latest_resume_document.status is ResumeDocumentStatus.VERIFIED
                else ()
            ),
            verification_report_ids=_ids(reports),
            passed_verification_report_ids=tuple(
                report.verification_report_id
                for report in reports
                if report.status is VerificationStatus.PASSED
            ),
            issue_verification_report_ids=tuple(
                report.verification_report_id
                for report in reports
                if report.status is not VerificationStatus.PASSED
            ),
            interview_corpus_count=len(corpus_metadata),
            interview_knowledge_chunk_count=sum(
                corpus.chunk_count for corpus in corpus_metadata
            ),
            rendered_pdf_count=len(pdfs),
            latest_pdf_path=str(pdfs[-1]) if pdfs else "",
            material_catalog_version=catalog_version,
            project_description_ids=_ids(project_descriptions),
            project_descriptions_needing_confirmation=tuple(
                item.project_description_id
                for item in project_descriptions
                if item.status is ProjectDescriptionStatus.NEEDS_USER_CONFIRMATION
            ),
            resume_run_ids=_ids(resume_runs),
            latest_resume_run_id=latest_resume_run_id,
            requirement_cache_count=len(requirement_caches),
            fresh_requirement_cache_count=sum(
                item.status is RequirementMatchCacheStatus.FRESH for item in requirement_caches
            ),
            project_source_count=len(project_sources),
            project_source_counts_by_status={
                key: value for key, value in project_status_counts.items() if value
            },
            project_ids_needing_analysis=tuple(
                source.project_id
                for source in project_sources
                if source.status in {
                    ProjectSourceStatus.REGISTERED,
                    ProjectSourceStatus.NEEDS_REANALYSIS,
                }
            ),
            unavailable_project_ids=tuple(
                source.project_id
                for source in project_sources
                if source.status is ProjectSourceStatus.UNAVAILABLE
            ),
        )


_GOAL_STAGES = {
    CareerGoal.RESUME: (
        CareerWorkflowStage.MATERIALS,
        CareerWorkflowStage.FACT_REVIEW,
        CareerWorkflowStage.PROJECT_ANALYSIS,
        CareerWorkflowStage.PROJECT_CONFIRMATION,
        CareerWorkflowStage.JD_ANALYSIS,
        CareerWorkflowStage.EVIDENCE_RETRIEVAL,
        CareerWorkflowStage.RESUME_PLANNING,
        CareerWorkflowStage.RESUME_GENERATION,
        CareerWorkflowStage.VERIFICATION,
        CareerWorkflowStage.PDF_RENDERING,
    ),
    CareerGoal.INTERVIEW: (
        CareerWorkflowStage.MATERIALS,
        CareerWorkflowStage.FACT_REVIEW,
        CareerWorkflowStage.PROJECT_ANALYSIS,
        CareerWorkflowStage.PROJECT_CONFIRMATION,
        CareerWorkflowStage.JD_ANALYSIS,
        CareerWorkflowStage.EVIDENCE_RETRIEVAL,
        CareerWorkflowStage.RESUME_PLANNING,
        CareerWorkflowStage.RESUME_GENERATION,
        CareerWorkflowStage.VERIFICATION,
        CareerWorkflowStage.INTERVIEW,
    ),
    CareerGoal.APPLICATION: (
        CareerWorkflowStage.MATERIALS,
        CareerWorkflowStage.FACT_REVIEW,
        CareerWorkflowStage.PROJECT_ANALYSIS,
        CareerWorkflowStage.PROJECT_CONFIRMATION,
        CareerWorkflowStage.JD_ANALYSIS,
        CareerWorkflowStage.EVIDENCE_RETRIEVAL,
        CareerWorkflowStage.RESUME_PLANNING,
        CareerWorkflowStage.RESUME_GENERATION,
        CareerWorkflowStage.VERIFICATION,
        CareerWorkflowStage.PDF_RENDERING,
        CareerWorkflowStage.APPLICATION,
    ),
}


class CareerWorkflowEngine:
    """Recommend the next action while preserving hard evidence boundaries."""

    def inspect(
        self,
        index: CareerArtifactIndex,
        *,
        goal: CareerGoal | str = CareerGoal.RESUME,
    ) -> CareerWorkspaceState:
        normalized_goal = CareerGoal(goal)
        gates = tuple(_build_gate(stage, index) for stage in _GOAL_STAGES[normalized_goal])
        goal_ready = all(gate.ready for gate in gates)
        recommended = next((gate for gate in gates if not gate.ready), gates[-1])
        return CareerWorkspaceState(
            goal=normalized_goal,
            goal_ready=goal_ready,
            recommended_stage=recommended.stage,
            recommended_domain_stage=_domain_stage_for(recommended.stage),
            index=index,
            stage_gates=gates,
            next_actions=_next_actions(recommended, index, normalized_goal),
            warnings=_warnings(index, normalized_goal),
        )


def _build_gate(stage: CareerWorkflowStage, index: CareerArtifactIndex) -> CareerStageGate:
    if stage is CareerWorkflowStage.MATERIALS:
        ready = index.has_materials
        return _gate(
            stage,
            ready,
            missing=("career_material",),
            tools=("import_career_material",),
            rationale="At least one personal material is required before evidence-based work can begin.",
        )
    if stage is CareerWorkflowStage.FACT_REVIEW:
        if index.has_unhandled_candidate_facts:
            return CareerStageGate(
                stage=stage,
                ready=False,
                missing=("candidate_fact_decision",),
                blocked_by=("unhandled_candidate_facts",),
                next_tool_names=("list_candidate_facts", "import_candidate_facts", "reject_candidate_facts"),
                rationale="Candidate facts must be reviewed before they can become PersonalEvidence.",
            )
        return _gate(
            stage,
            index.has_usable_evidence,
            missing=("user_confirmed_personal_evidence",),
            blocked=("no_confirmed_evidence",),
            tools=("extract_career_facts", "confirm_career_facts"),
            rationale="Resume claims can only cite verified or user-confirmed facts.",
        )
    if stage is CareerWorkflowStage.PROJECT_ANALYSIS:
        return _gate(
            stage,
            not index.project_ids_needing_analysis,
            missing=("project_analysis",),
            blocked=("project_source_needs_analysis",),
            tools=("career_task", "save_project_understanding"),
            rationale="Registered project sources must be analyzed at their current version before use.",
        )
    if stage is CareerWorkflowStage.PROJECT_CONFIRMATION:
        return _gate(
            stage,
            not index.project_descriptions_needing_confirmation,
            missing=("user_confirmed_project_experience",),
            blocked=("project_experience_needs_confirmation",),
            tools=(),
            rationale="Project responsibilities and outcomes require explicit user confirmation.",
        )
    if stage is CareerWorkflowStage.JD_ANALYSIS:
        return _gate(
            stage,
            index.has_jd,
            missing=("jd_requirements",),
            tools=("analyze_jd",),
            rationale="The target JD must be normalized before evidence matching.",
        )
    if stage is CareerWorkflowStage.EVIDENCE_RETRIEVAL:
        return _gate(
            stage,
            index.has_evidence_retrieval,
            missing=("evidence_matches",),
            blocked=_prerequisite_gaps(index, evidence=True, jd=True),
            tools=("retrieve_career_evidence",),
            rationale="Each JD requirement needs evidence support or an explicit gap.",
        )
    if stage is CareerWorkflowStage.RESUME_PLANNING:
        return _gate(
            stage,
            index.has_resume_plan,
            missing=("resume_plan",),
            blocked=_prerequisite_gaps(index, evidence=True, jd=True, matches=True),
            tools=("plan_resume_content",),
            rationale="ResumePlan selects evidence-backed content under explicit constraints.",
        )
    if stage is CareerWorkflowStage.RESUME_GENERATION:
        return _gate(
            stage,
            index.has_resume_document,
            missing=("resume_document",),
            blocked=() if index.has_resume_plan else ("resume_plan",),
            tools=("generate_resume_document",),
            rationale="Structured resume generation requires a persisted ResumePlan.",
        )
    if stage is CareerWorkflowStage.VERIFICATION:
        return _gate(
            stage,
            index.has_verified_resume,
            missing=("verified_resume_document",),
            blocked=() if index.has_resume_document else ("resume_document",),
            tools=("verify_resume_document", "revise_resume_document"),
            rationale="Independent verification is mandatory before PDF, interview, or application.",
        )
    if stage is CareerWorkflowStage.PDF_RENDERING:
        return _gate(
            stage,
            index.has_rendered_pdf,
            missing=("verified_resume_pdf",),
            blocked=() if index.has_verified_resume else ("verified_resume_document",),
            tools=("render_resume_pdf",),
            rationale="PDF rendering requires a verified ResumeDocument.",
        )
    if stage is CareerWorkflowStage.INTERVIEW:
        missing = tuple(
            name
            for name, available in (
                ("verified_resume_document", index.has_verified_resume),
                ("jd_document", index.has_jd),
                ("interview_knowledge", index.has_interview_knowledge),
            )
            if not available
        )
        return _gate(
            stage,
            not missing,
            missing=missing,
            blocked=missing,
            tools=("import_interview_knowledge", "career_task"),
            rationale="InterviewerAgent only receives JD, verified resume, and interview knowledge.",
        )
    missing = tuple(
        name
        for name, available in (
            ("verified_resume_document", index.has_verified_resume),
            ("verified_resume_pdf", index.has_rendered_pdf),
        )
        if not available
    )
    return _gate(
        CareerWorkflowStage.APPLICATION,
        not missing,
        missing=missing,
        blocked=missing,
        tools=("render_resume_pdf", "career_task"),
        rationale="ApplicationAgent only receives URL and verified resume PDF.",
    )


def _gate(
    stage: CareerWorkflowStage,
    ready: bool,
    *,
    missing: tuple[str, ...],
    tools: tuple[str, ...],
    rationale: str,
    blocked: tuple[str, ...] = (),
) -> CareerStageGate:
    return CareerStageGate(
        stage=stage,
        ready=ready,
        missing=() if ready else missing,
        blocked_by=() if ready else blocked,
        next_tool_names=() if ready else tools,
        rationale=rationale,
    )


def _prerequisite_gaps(
    index: CareerArtifactIndex,
    *,
    evidence: bool = False,
    jd: bool = False,
    matches: bool = False,
) -> tuple[str, ...]:
    gaps: list[str] = []
    if evidence and not index.has_usable_evidence:
        gaps.append("user_confirmed_personal_evidence")
    if jd and not index.has_jd:
        gaps.append("jd_document")
    if matches and not index.has_evidence_retrieval:
        gaps.append("evidence_matches")
    return tuple(gaps)


def _domain_stage_for(stage: CareerWorkflowStage) -> str:
    if stage in {
        CareerWorkflowStage.MATERIALS,
        CareerWorkflowStage.FACT_REVIEW,
        CareerWorkflowStage.PROJECT_ANALYSIS,
        CareerWorkflowStage.PROJECT_CONFIRMATION,
    }:
        return "materials"
    if stage in {
        CareerWorkflowStage.JD_ANALYSIS,
        CareerWorkflowStage.EVIDENCE_RETRIEVAL,
        CareerWorkflowStage.RESUME_PLANNING,
        CareerWorkflowStage.RESUME_GENERATION,
        CareerWorkflowStage.VERIFICATION,
        CareerWorkflowStage.PDF_RENDERING,
    }:
        return "resume"
    return "interview" if stage is CareerWorkflowStage.INTERVIEW else "application"


def _next_actions(
    gate: CareerStageGate,
    index: CareerArtifactIndex,
    goal: CareerGoal,
) -> tuple[str, ...]:
    if gate.ready:
        if goal is CareerGoal.RESUME:
            return ("Resume goal is ready for delivery or user review.",)
        if goal is CareerGoal.INTERVIEW:
            return ("Invoke InterviewerAgent with JD, verified resume, and interview knowledge.",)
        return ("Invoke ApplicationAgent after the user provides an application URL.",)
    if gate.stage is CareerWorkflowStage.MATERIALS:
        return ("Import classified personal materials.",)
    if gate.stage is CareerWorkflowStage.FACT_REVIEW:
        if index.has_unhandled_candidate_facts:
            return ("Ask the user which candidate facts to import or reject.",)
        if index.material_ids_without_candidates:
            return ("Extract candidate facts from unprocessed materials.",)
        return ("Ask the user to confirm imported unverified facts.",)
    if gate.stage is CareerWorkflowStage.PROJECT_ANALYSIS:
        return ("Analyze registered projects at their current source version.",)
    if gate.stage is CareerWorkflowStage.PROJECT_CONFIRMATION:
        return ("Ask the user to confirm project responsibilities and outcomes.",)
    messages = {
        CareerWorkflowStage.JD_ANALYSIS: "Analyze the target JD into traceable requirements.",
        CareerWorkflowStage.EVIDENCE_RETRIEVAL: "Retrieve confirmed evidence for each JD requirement.",
        CareerWorkflowStage.RESUME_PLANNING: "Plan evidence-backed resume content under constraints.",
        CareerWorkflowStage.RESUME_GENERATION: "Generate a structured ResumeDocument.",
        CareerWorkflowStage.VERIFICATION: "Run independent verification and scoped correction.",
        CareerWorkflowStage.PDF_RENDERING: "Render the verified resume to ATS-readable PDF.",
        CareerWorkflowStage.INTERVIEW: "Import interview knowledge, then invoke InterviewerAgent.",
        CareerWorkflowStage.APPLICATION: "Render a verified PDF, then invoke ApplicationAgent.",
    }
    return (messages[gate.stage],)


def _warnings(index: CareerArtifactIndex, goal: CareerGoal) -> tuple[str, ...]:
    warnings: list[str] = []
    if index.unverified_fact_count:
        warnings.append("Unverified PersonalEvidence cannot support ResumeClaim until confirmed.")
    if index.unsupported_requirement_count:
        warnings.append("Unsupported JD requirements must remain explicit gaps.")
    if index.unavailable_project_ids:
        warnings.append("Unavailable project sources cannot be analyzed or used for new claims.")
    if goal is CareerGoal.INTERVIEW and not index.has_interview_knowledge:
        warnings.append("Interview knowledge is missing; do not claim source-grounded professional questions.")
    return tuple(warnings)
