"""Run the deterministic offline Career demonstration from an empty workspace."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_agent.agents import AgentArtifact, AgentRunResult
from smart_agent.career import (
    CareerStore,
    EvidenceType,
    InterviewKnowledgeSourceType,
    ModelAPIIdentity,
    PersonalEvidence,
    ProjectAnalysisRecord,
    ProjectAnalysisStore,
    ProjectDescriptionService,
    ProjectLibraryStore,
    ProjectSourceType,
    ProjectUnderstanding,
    ResumeVerificationAgent,
    VerifierModelConfig,
)
from smart_agent.tooling import ToolContext
from smart_agent.tools import create_default_tool_registry

FIXTURE_ROOT = ROOT / "fixtures" / "career_demo"


class _OfflineVerificationRunner:
    """A deterministic pass runner; it never calls a network API."""

    def run(self, task, _artifacts, _spec):
        artifact = AgentArtifact.create(
            "resume_verification_result",
            domain="career",
            session_id=task.child_session_id,
            created_by="offline-verifier",
            content={"text": json.dumps({"issues": []})},
            source_artifact_ids=task.input_artifact_ids,
        )
        return AgentRunResult(task=task, status="completed", output_artifact=artifact)


def _prepare_workspace(path: Path, *, force: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        marker = path / ".career-demo-marker"
        if not force or not marker.is_file():
            raise RuntimeError(
                f"Workspace is not empty: {path}. Use an empty path or --force on a prior demo workspace."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / ".career-demo-marker").write_text("offline career demo\n", encoding="utf-8")
    data_root = path / "demo-data"
    shutil.copytree(FIXTURE_ROOT, data_root)
    return path


def _execute(
    workspace: Path,
    runtime: dict[str, Any],
    stage: str,
    tool_name: str,
    payload: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    parse_output: bool = True,
) -> dict[str, Any]:
    runtime["domainStage"] = stage
    registry = create_default_tool_registry(str(workspace), runtime=runtime)
    context = ToolContext(cwd=str(workspace), _runtime=runtime)
    result = registry.execute(tool_name, payload, context)
    if not result.ok:
        events.append({"stage": stage, "tool": tool_name, "status": "failed", "error": result.output[:500]})
        raise RuntimeError(f"{tool_name} failed: {result.output}")
    events.append({"stage": stage, "tool": tool_name, "status": "completed"})
    return json.loads(result.output) if parse_output else {}


def run_demo(workspace: Path, *, force: bool = False) -> dict[str, Any]:
    workspace = _prepare_workspace(workspace, force=force)
    events: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {
        "activeDomain": "career",
        "careerTenantId": "demo-candidate",
        "_careerGenerationIdentity": ModelAPIIdentity(
            "offline",
            "demo-generator-v1",
            "api:offline-generation",
        ),
    }
    career_store = CareerStore(workspace / ".runtime" / "career")
    # Share one store instance across the demo runtime and tools so in-memory
    # indexes observe every artifact written during the run.
    runtime["_careerStore"] = career_store
    runtime["_resumeVerificationAgent"] = ResumeVerificationAgent(
        store=career_store,
        cwd=str(workspace),
        verifier_config=VerifierModelConfig(
            "openai",
            "demo-verifier-v1",
            "https://offline-verifier.invalid/v1",
            "offline-only",
        ),
        runner=_OfflineVerificationRunner(),
    )

    materials = (
        ("resume", "General resume reference", "resume.txt"),
        ("education", "Education", "education.md"),
        ("award", "Awards", "awards.md"),
        ("internship", "Backend internship", "internship.md"),
        ("research", "Research", "research.md"),
    )
    manifests: dict[str, str] = {}
    for category, label, filename in materials:
        material_payload = {
            "category": category,
            "label": label,
            "project_id": "smart-agent",
        }
        if category == "resume":
            material_payload["source_path"] = f"demo-data/{filename}"
        else:
            material_payload["text"] = (workspace / "demo-data" / filename).read_text(encoding="utf-8")
        imported = _execute(
            workspace,
            runtime,
            "materials",
            "import_career_material",
            material_payload,
            events,
        )
        manifests[category] = imported["manifest"]["material_id"]

    project_library = ProjectLibraryStore(
        workspace / ".runtime" / "career" / "project-library",
        workspace=workspace,
    )
    project_source = project_library.register(
        project_id="smart-agent",
        display_name="Smart Agent project",
        source_type=ProjectSourceType.DOCUMENT_BUNDLE,
        source_location=workspace / "demo-data" / "project.md",
    )
    understanding = ProjectUnderstanding(
        title="Smart Agent Career module",
        goal="Build an evidence-backed career assistant for resumes and adaptive interviews.",
        core_flow=("Analyze JD", "Match confirmed evidence", "Generate and verify resume", "Run adaptive interview"),
        core_modules=("project.md",),
        architecture="Career workflows share a traceable personal evidence store.",
        technology_choices=("Python and FastAPI expose deterministic career workflows.",),
        technologies=("Python", "FastAPI", "BM25", "CP-SAT"),
        implemented_features=("JD analysis", "Evidence-backed resume", "Adaptive mock interview"),
        candidate_outcomes=(),
        candidate_responsibilities=(),
        technical_tags=("Career Agent", "evidence traceability"),
        unknowns=("Personal responsibility and project outcomes require user confirmation.",),
        source_refs=(
            {
                "field": "goal",
                "path": "project.md",
                "line": 2,
                "excerpt": "Smart Agent Career module.",
            },
        ),
        confidence_by_field={"goal": 0.9, "technologies": 0.9},
        analysis_summary="The fixture describes a traceable Career workflow.",
    )
    analysis = ProjectAnalysisRecord(
        analysis_run_id="demo-project-analysis-v1",
        project_id=project_source.project_id,
        project_version=project_source.project_version,
        intent="overview",
        generator="offline-demo",
        model="offline",
        understanding=understanding,
    )
    ProjectAnalysisStore(
        workspace / ".runtime" / "career" / "project-library" / "analyses"
    ).put(analysis)
    draft = ProjectDescriptionService(career_store).generate_from_understanding(
        source=project_source,
        understanding=understanding,
        analysis_run_id=analysis.analysis_run_id,
        generator=analysis.generator,
        model=analysis.model,
    )
    ProjectDescriptionService(career_store).confirm(
        draft.project_description_id,
        changes={
            "responsibilities": ["Implemented the evidence-backed Career workflow"],
            "outcomes": ["Delivered an end-to-end offline demonstration"],
        },
    )
    project_library.mark_analyzed(
        project_source.project_id,
        analysis.analysis_run_id,
        expected_project_version=project_source.project_version,
    )
    events.append({"stage": "materials", "tool": "project_library", "status": "completed"})

    candidate_groups: list[list[str]] = []
    for category, material_id in manifests.items():
        if category == "resume":
            continue
        extracted = _execute(
            workspace,
            runtime,
            "materials",
            "extract_career_facts",
            {"material_id": material_id, "max_facts": 20},
            events,
        )
        candidate_groups.append([item["candidate_fact_id"] for item in extracted["candidate_facts"]])
    for candidate_ids in candidate_groups:
        imported_facts = _execute(
            workspace,
            runtime,
            "materials",
            "import_candidate_facts",
            {"candidate_fact_ids": candidate_ids},
            events,
        )
        for record in imported_facts["records"]:
            fact_ids = [
                item["fact_id"]
                for item in imported_facts["candidate_facts"]
                if item["evidence_id"] == record["evidence_id"] and item["fact_id"]
            ]
            _execute(
                workspace,
                runtime,
                "materials",
                "confirm_career_facts",
                {
                    "evidence_id": record["evidence_id"],
                    "fact_ids": fact_ids,
                    "reason": "Confirmed by the offline demo fixture owner",
                },
                events,
                parse_output=False,
            )

    jd_text = (workspace / "demo-data" / "jd.txt").read_text(encoding="utf-8")
    jd_lines = [line.strip() for line in jd_text.splitlines() if line.strip()]
    requirements = [
        {
            "category": "hard_skill",
            "text": line,
            "original_text": line,
            "required": True,
            "confidence": 1.0,
        }
        for line in jd_lines[2:]
    ]
    jd = _execute(
        workspace,
        runtime,
        "resume",
        "analyze_jd",
        {
            "text": jd_text,
            "role_title": "Backend Engineer",
            "company": "Demo Cloud",
            "requirements": requirements,
            "source": {"source_id": "demo-jd", "source_type": "user_input", "locator": {"fixture": True}},
        },
        events,
    )
    jd_id = jd["document"]["jd_id"]
    requirement_ids = [item["requirement_id"] for item in jd["requirements"]]
    evidence = career_store.list_latest(PersonalEvidence)
    evidence_ids = [record.evidence_id for record in evidence]
    retrieved = _execute(
        workspace,
        runtime,
        "resume",
        "retrieve_career_evidence",
        {"requirement_ids": requirement_ids, "evidence_ids": evidence_ids},
        events,
    )
    requirements_by_fact: dict[tuple[str, str], list[str]] = {}
    for match in retrieved["matches"]:
        evidence_ref = match.get("evidence_ref")
        if not evidence_ref or match["support_level"] == "unsupported":
            continue
        evidence_id = evidence_ref["record_id"]
        requirement_id = match["requirement_ref"]["record_id"]
        for fact_id in match["evidence_fact_ids"]:
            requirements_by_fact.setdefault((evidence_id, fact_id), []).append(requirement_id)
    candidates: list[dict[str, Any]] = []
    for record in evidence:
        section = {
            EvidenceType.EDUCATION: "education",
            EvidenceType.AWARD: "awards",
            EvidenceType.INTERNSHIP: "internships",
            EvidenceType.PROJECT: "projects",
            EvidenceType.RESEARCH: "research",
        }[record.evidence_type]
        for fact in record.facts:
            matched_requirement_ids = requirements_by_fact.get((record.evidence_id, fact.fact_id), [])
            if not fact.is_usable or not matched_requirement_ids:
                continue
            candidates.append(
                {
                    "section": section,
                    "proposed_text": f"{record.label}: {fact.value}",
                    "evidence_ids": [record.evidence_id],
                    "evidence_fact_ids": [fact.fact_id],
                    "requirement_ids": matched_requirement_ids,
                    "length_cost": max(10, len(str(fact.value))),
                    "features": {
                        "jd_relevance": 0.85,
                        "evidence_strength": 1.0,
                        "title_and_brand_value": 0.8,
                        "result_strength": 0.7,
                        "narrative_value": 0.8,
                    },
                    "selected_by_llm": True,
                }
            )
    planned = _execute(
        workspace,
        runtime,
        "resume",
        "plan_resume_content",
        {
            "jd_id": jd_id,
            "content_budget": 1200,
            "section_order": ["education", "internships", "projects", "research", "awards"],
            "candidate_items": candidates,
        },
        events,
    )
    claim_drafts = [
        {"item_id": item["item_id"], "text": item["proposed_text"]}
        for item in planned["selected_items"]
    ]
    generated = _execute(
        workspace,
        runtime,
        "resume",
        "generate_resume_document",
        {
            "plan_id": planned["plan"]["plan_id"],
            "header": {
                "full_name": "Demo Candidate",
                "headline": "Backend Engineer",
                "email": "demo@example.com",
                "location": "Remote",
                "links": ["https://example.com/demo"],
            },
            "claim_drafts": claim_drafts,
        },
        events,
    )
    verified = _execute(
        workspace,
        runtime,
        "resume",
        "verify_resume_document",
        {"resume_document_id": generated["document"]["resume_document_id"]},
        events,
    )
    rendered = _execute(
        workspace,
        runtime,
        "resume",
        "render_resume_pdf",
        {
            "resume_document_id": verified["document"]["resume_document_id"],
            "output_dir": "exports",
            "max_pages": 2,
        },
        events,
    )

    knowledge = _execute(
        workspace,
        runtime,
        "interview",
        "import_interview_knowledge_source",
        {
            "tenant_id": "demo-candidate",
            "name": "Backend fundamentals",
            "source_type": InterviewKnowledgeSourceType.COURSE_NOTE.value,
            "authority_level": 5,
            "domains": ["computer_science"],
            "subdomains": ["backend", "distributed_systems"],
            "topics": ["API latency", "cache", "distributed systems"],
            "source_path": "demo-data/interview_notes.md",
        },
        events,
    )
    session = _execute(
        workspace,
        runtime,
        "interview",
        "create_interview_session",
        {
            "jd_id": jd_id,
            "resume_document_id": verified["document"]["resume_document_id"],
            "corpus_ids": [knowledge["corpus"]["corpus_id"]],
            "random_seed": 17,
            "max_turns": 3,
        },
        events,
    )
    interview_id = session["interview_id"]
    for _turn in range(3):
        current = _execute(workspace, runtime, "interview", "next_interview_question", {"interview_id": interview_id}, events)
        _execute(
            workspace,
            runtime,
            "interview",
            "submit_interview_answer",
            {
                "interview_id": interview_id,
                "answer": "I would measure first, then use caching and profiling while considering invalidation and trade-offs.",
            },
            events,
        )
        if current.get("question_id") is None:
            break
    # A max-turn answer records the terminal decision but leaves the session
    # active until the caller explicitly commits the final report.
    session_state = _execute(
        workspace,
        runtime,
        "interview",
        "get_interview_session",
        {"interview_id": interview_id},
        events,
    )
    if session_state.get("status") == "active":
        _execute(
            workspace,
            runtime,
            "interview",
            "finish_interview_session",
            {"interview_id": interview_id, "cancelled": False},
            events,
        )
    report = _execute(
        workspace,
        runtime,
        "interview",
        "get_interview_session",
        {"interview_id": interview_id},
        events,
    )
    summary = {
        "mode": "offline_mock",
        "workspace": str(workspace),
        "artifacts": {
            "resume_document_id": verified["document"]["resume_document_id"],
            "verification_report_id": verified["report"]["verification_report_id"],
            "pdf_path": rendered["pdf_path"],
            "interview_id": interview_id,
            "interview_status": report["status"],
        },
        "checks": {
            "resume_verified": verified["document"]["status"] == "verified",
            "pdf_exists": Path(rendered["pdf_path"]).is_file(),
            "interview_report_present": bool(report.get("report")),
            "source_trace_present": bool(report.get("report", {}).get("trace")),
        },
        "events": events,
    }
    (workspace / "run-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT / "artifacts" / "career-e2e",
        help="empty workspace to populate (default: artifacts/career-e2e)",
    )
    parser.add_argument("--force", action="store_true", help="replace a prior workspace marked by this demo")
    args = parser.parse_args()
    summary = run_demo(args.workspace, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(summary["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
