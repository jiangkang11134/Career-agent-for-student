"""Unified project-analysis and project-experience backend service."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from smart_agent.career.models import ProjectDescriptionDocument, ProjectDescriptionStatus
from smart_agent.career.project_library import (
    ProjectAnalysisRecord,
    ProjectAnalysisStore,
    ProjectLibraryStore,
    ProjectSource,
    ProjectSourceStatus,
    ProjectUnderstanding,
    project_version,
    scan_project_artifacts,
)
from smart_agent.career.projects import ProjectDescriptionService
from smart_agent.career.store import CareerStore, RecordNotFoundError, RevisionConflictError

PROJECT_RUN_SCHEMA_VERSION = "1.0"
_TEXT_SUFFIXES = frozenset(
    {
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go", ".rs",
        ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".php", ".rb", ".swift", ".scala",
        ".sql", ".sh", ".ps1", ".bat", ".html", ".css", ".scss", ".vue", ".svelte",
        ".md", ".markdown", ".txt", ".rst", ".csv", ".toml", ".yaml", ".yml", ".json",
        ".ini", ".cfg",
    }
)


def _normalize_artifact_path(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").lstrip("./").casefold()


def _resolve_cited_artifact_path(reference: object, artifact_paths: set[str]) -> str | None:
    """Resolve a model citation against the registered project manifest.

    ZIP uploads commonly contain one top-level project directory while the
    model naturally cites paths relative to that directory (for example,
    ``README.md``). Only a unique suffix match is accepted; ambiguous or
    missing paths remain invalid so citation checks stay strict.
    """

    normalized = _normalize_artifact_path(reference)
    if not normalized:
        return None
    exact = next((path for path in artifact_paths if _normalize_artifact_path(path) == normalized), None)
    if exact is not None:
        return exact
    matches = [
        path
        for path in artifact_paths
        if _normalize_artifact_path(path).endswith("/" + normalized)
    ]
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True, slots=True)
class ProjectAnalysisValidation:
    complete: bool
    missing_fields: tuple[str, ...] = ()
    invalid_source_refs: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    artifact_count: int = 0
    cited_source_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProjectAnalysisRun:
    run_id: str
    project_id: str
    project_version: str
    intent: str
    status: str
    trigger: str
    generator: str
    model: str = ""
    elapsed_ms: float = 0.0
    cache_hit: bool = False
    artifact_count: int = 0
    cited_source_count: int = 0
    candidate_fact_count: int = 0
    tool_call_count: int = 0
    token_count: int = 0
    token_count_source: str = ""
    agent_trace: tuple[dict[str, Any], ...] = ()
    validation: ProjectAnalysisValidation = field(
        default_factory=lambda: ProjectAnalysisValidation(complete=False)
    )
    error: str = ""
    created_at: float = field(default_factory=time.time)
    schema_version: str = PROJECT_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"completed", "incomplete", "failed", "cached"}:
            raise ValueError(f"Unsupported project analysis status: {self.status}")
        if (
            self.elapsed_ms < 0
            or self.artifact_count < 0
            or self.cited_source_count < 0
            or self.tool_call_count < 0
            or self.token_count < 0
        ):
            raise ValueError("Project analysis metrics cannot be negative")
        if self.schema_version != PROJECT_RUN_SCHEMA_VERSION:
            raise ValueError(f"Unsupported project run schema: {self.schema_version}")
        if any(not isinstance(item, dict) for item in self.agent_trace):
            raise ValueError("Project analysis agent_trace must contain objects")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["validation"] = self.validation.to_dict()
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectAnalysisRun":
        payload = dict(value)
        payload["validation"] = ProjectAnalysisValidation(**dict(payload.get("validation") or {}))
        payload["agent_trace"] = tuple(
            dict(item) for item in payload.get("agent_trace", ()) if isinstance(item, dict)
        )
        return cls(**payload)


class ProjectAnalysisRunStore:
    """Append-only run telemetry without project source text."""

    _ROOT_LOCKS: dict[str, threading.RLock] = {}
    _ROOT_LOCKS_GUARD = threading.Lock()

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self._ROOT_LOCKS_GUARD:
            self._lock = self._ROOT_LOCKS.setdefault(str(self.root).casefold(), threading.RLock())

    def put(self, run: ProjectAnalysisRun) -> ProjectAnalysisRun:
        path = self.root / f"{run.run_id}.json"
        with self._lock:
            if path.exists():
                existing = ProjectAnalysisRun.from_dict(json.loads(path.read_text(encoding="utf-8")))
                if existing != run:
                    raise ValueError(f"Project analysis run already exists: {run.run_id}")
                return existing
            fd, temporary = tempfile.mkstemp(prefix="project-run-", suffix=".tmp", dir=self.root)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(run.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                try:
                    Path(temporary).unlink()
                except OSError:
                    pass
        return run

    def list(self, *, project_id: str = "") -> tuple[ProjectAnalysisRun, ...]:
        selected = str(project_id or "").strip()
        records: list[ProjectAnalysisRun] = []
        with self._lock:
            for path in self.root.glob("*.json"):
                try:
                    record = ProjectAnalysisRun.from_dict(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if not selected or record.project_id == selected:
                    records.append(record)
        return tuple(sorted(records, key=lambda item: (item.created_at, item.run_id)))

    def purge(self, project_id: str) -> int:
        selected = str(project_id or "").strip()
        if not selected:
            raise ValueError("project_id is required")
        removed = 0
        with self._lock:
            for record in self.list(project_id=selected):
                path = self.root / f"{record.run_id}.json"
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                removed += 1
        return removed

    def metrics(self) -> dict[str, Any]:
        runs = self.list()
        completed = tuple(item for item in runs if item.status in {"completed", "cached"})
        scans = tuple(item for item in runs if not item.cache_hit)
        return {
            "run_count": len(runs),
            "completed_count": len(completed),
            "failed_count": sum(item.status == "failed" for item in runs),
            "incomplete_count": sum(item.status == "incomplete" for item in runs),
            "cache_hit_count": sum(item.cache_hit for item in runs),
            "cache_hit_rate": round(sum(item.cache_hit for item in runs) / len(runs), 6) if runs else 0.0,
            "average_elapsed_ms": (
                round(sum(item.elapsed_ms for item in scans) / len(scans), 3) if scans else 0.0
            ),
            "candidate_fact_count": sum(item.candidate_fact_count for item in runs),
            "cited_source_count": sum(item.cited_source_count for item in runs),
            "tool_call_count": sum(item.tool_call_count for item in runs),
            "token_count": sum(item.token_count for item in runs),
        }


class ProjectUnderstandingValidator:
    """Verify completion and source citations against the registered project."""

    def validate(
        self,
        source: ProjectSource,
        understanding: ProjectUnderstanding,
    ) -> ProjectAnalysisValidation:
        root = Path(source.source_location).resolve()
        artifacts = scan_project_artifacts(root)
        if project_version(artifacts) != source.project_version:
            return ProjectAnalysisValidation(
                complete=False,
                invalid_source_refs=("project_version_changed",),
                warnings=("项目内容已变化，需要刷新项目版本后重新分析。",),
                artifact_count=len(artifacts),
            )
        artifact_paths = {item.path for item in artifacts}
        invalid: list[str] = []
        cited: set[str] = set()
        for ref in understanding.source_refs:
            original_relative = str(ref.get("path") or "").strip().replace("\\", "/")
            relative = _resolve_cited_artifact_path(original_relative, artifact_paths)
            if relative is None:
                invalid.append(f"missing:{original_relative or '<empty>'}")
                continue
            target = root if root.is_file() else root / relative
            line = int(ref.get("line") or 0)
            if target.suffix.casefold() in _TEXT_SUFFIXES or target.name.casefold().startswith("readme"):
                try:
                    line_count = max(1, len(target.read_text(encoding="utf-8", errors="replace").splitlines()))
                except OSError:
                    invalid.append(f"unreadable:{relative}")
                    continue
                if line < 1 or line > line_count:
                    invalid.append(f"line_out_of_range:{relative}:{line}")
                    continue
            cited.add(relative)

        missing: list[str] = []
        if not understanding.goal:
            missing.append("goal")
        if not understanding.core_flow:
            missing.append("core_flow")
        if not understanding.core_modules:
            missing.append("core_modules")
        if understanding.technologies and not understanding.technology_choices:
            missing.append("technology_choices")
        if not understanding.implemented_features and not understanding.candidate_outcomes:
            missing.append("observable_deliverables")
        if not understanding.source_refs:
            missing.append("source_refs")
        complete = not missing and not invalid
        return ProjectAnalysisValidation(
            complete=complete,
            missing_fields=tuple(missing),
            invalid_source_refs=tuple(invalid),
            warnings=(() if complete else ("分析结果不完整，不能生成可确认项目档案。",)),
            artifact_count=len(artifacts),
            cited_source_count=len(cited),
        )


@dataclass(frozen=True, slots=True)
class ProjectExperienceResult:
    analysis: ProjectAnalysisRecord
    run: ProjectAnalysisRun
    description: ProjectDescriptionDocument | None

    def to_dict(self) -> dict[str, Any]:
        from smart_agent.career.models import record_to_dict

        return {
            "analysis": self.analysis.to_dict(),
            "analysis_run": self.run.to_dict(),
            "project_description": record_to_dict(self.description) if self.description else None,
        }


class ProjectExperienceService:
    """One service boundary for project analysis, drafts, and confirmation."""

    _TUPLE_FIELDS = frozenset(
        {
            "goals", "responsibilities", "core_flow", "technology_choices", "core_modules",
            "technologies", "technical_tags", "implemented_features", "outcomes", "unknowns",
            "challenges", "links",
        }
    )

    def __init__(
        self,
        *,
        career_store: CareerStore,
        project_store: ProjectLibraryStore,
        analysis_store: ProjectAnalysisStore,
        run_store: ProjectAnalysisRunStore,
        validator: ProjectUnderstandingValidator | None = None,
    ) -> None:
        self.career_store = career_store
        self.project_store = project_store
        self.analysis_store = analysis_store
        self.run_store = run_store
        self.validator = validator or ProjectUnderstandingValidator()
        self.description_service = ProjectDescriptionService(career_store)

    def list_experiences(self) -> tuple[ProjectDescriptionDocument, ...]:
        active_sources = {
            source.project_id.casefold(): source
            for source in self.project_store.list()
        }
        latest_by_project: dict[str, ProjectDescriptionDocument] = {}
        for item in self.career_store.list_latest(ProjectDescriptionDocument):
            source = active_sources.get(item.project_id.casefold())
            if source is None:
                continue
            normalized = (
                item if item.project_id == source.project_id
                else replace(item, project_id=source.project_id, content_hash="")
            )
            key = source.project_id.casefold()
            previous = latest_by_project.get(key)
            if previous is None or (normalized.updated_at, normalized.revision) > (
                previous.updated_at,
                previous.revision,
            ):
                latest_by_project[key] = normalized
        return tuple(
            sorted(
                latest_by_project.values(),
                key=lambda item: (item.updated_at, item.project_id),
                reverse=True,
            )
        )

    def get_experience(self, project_id: str) -> ProjectDescriptionDocument:
        selected = str(project_id or "").strip()
        result = next((item for item in self.list_experiences() if item.project_id == selected), None)
        if result is None:
            raise RecordNotFoundError(f"No project experience exists for project: {selected}")
        return result

    def persist_understanding(
        self,
        *,
        project_id: str,
        understanding: ProjectUnderstanding,
        analysis_run_id: str = "",
        intent: str = "overview",
        generator: str = "project-analyst-agent-v1",
        model: str = "",
        trigger: str = "manual",
        elapsed_ms: float = 0.0,
        cache_hit: bool = False,
        tool_call_count: int = 0,
        token_count: int = 0,
        token_count_source: str = "",
        agent_trace: tuple[dict[str, Any]] | list[dict[str, Any]] = (),
    ) -> ProjectExperienceResult:
        source = self.project_store.get(project_id, refresh=True)
        if source.status in {ProjectSourceStatus.UNAVAILABLE, ProjectSourceStatus.INVALIDATED}:
            raise ValueError(f"Project source is not readable: {source.status.value}")
        run_id = str(analysis_run_id or "").strip() or f"analysis-{uuid.uuid4().hex[:16]}"
        normalized_intent = str(intent or "overview").strip()
        validation = self.validator.validate(source, understanding)
        analysis = self.analysis_store.put(
            ProjectAnalysisRecord(
                analysis_run_id=run_id,
                project_id=source.project_id,
                project_version=source.project_version,
                intent=normalized_intent,
                generator=generator,
                model=model,
                understanding=understanding,
            )
        )
        description = None
        status = "cached" if cache_hit else "completed"
        if not validation.complete:
            status = "incomplete"
        elif normalized_intent == "overview":
            description = self.description_service.generate_from_understanding(
                source=source,
                understanding=understanding,
                analysis_run_id=run_id,
                generator=generator,
                model=model,
            )
            self.project_store.mark_analyzed(
                source.project_id,
                run_id,
                expected_project_version=source.project_version,
            )
        candidate_fact_count = sum(
            len(value)
            for value in (
                understanding.candidate_responsibilities,
                understanding.candidate_outcomes,
                understanding.implemented_features,
            )
        )
        run = self.run_store.put(
            ProjectAnalysisRun(
                run_id=run_id,
                project_id=source.project_id,
                project_version=source.project_version,
                intent=normalized_intent,
                status=status,
                trigger=trigger,
                generator=generator,
                model=model,
                elapsed_ms=max(0.0, float(elapsed_ms)),
                cache_hit=cache_hit,
                artifact_count=validation.artifact_count,
                cited_source_count=validation.cited_source_count,
                candidate_fact_count=candidate_fact_count,
                tool_call_count=max(0, int(tool_call_count)),
                token_count=max(0, int(token_count)),
                token_count_source=str(token_count_source or "").strip(),
                agent_trace=tuple(dict(item) for item in agent_trace if isinstance(item, dict)),
                validation=validation,
            )
        )
        return ProjectExperienceResult(analysis=analysis, run=run, description=description)

    def record_cache_hit(
        self,
        *,
        source: ProjectSource,
        analysis: ProjectAnalysisRecord,
        trigger: str,
    ) -> ProjectAnalysisRun:
        validation = self.validator.validate(source, analysis.understanding)
        return self.run_store.put(
            ProjectAnalysisRun(
                run_id=f"cache-{uuid.uuid4().hex[:16]}",
                project_id=source.project_id,
                project_version=source.project_version,
                intent=analysis.intent,
                status="cached",
                trigger=trigger,
                generator=analysis.generator,
                model=analysis.model,
                cache_hit=True,
                artifact_count=validation.artifact_count,
                cited_source_count=validation.cited_source_count,
                candidate_fact_count=sum(
                    len(value)
                    for value in (
                        analysis.understanding.candidate_responsibilities,
                        analysis.understanding.candidate_outcomes,
                        analysis.understanding.implemented_features,
                    )
                ),
                validation=validation,
            )
        )

    def record_failure(
        self,
        *,
        source: ProjectSource,
        intent: str,
        trigger: str,
        generator: str,
        model: str,
        elapsed_ms: float,
        error: Exception | str,
        tool_call_count: int = 0,
        token_count: int = 0,
        token_count_source: str = "",
    ) -> ProjectAnalysisRun:
        return self.run_store.put(
            ProjectAnalysisRun(
                run_id=f"failed-{uuid.uuid4().hex[:16]}",
                project_id=source.project_id,
                project_version=source.project_version,
                intent=str(intent or "overview").strip(),
                status="failed",
                trigger=trigger,
                generator=generator,
                model=model,
                elapsed_ms=max(0.0, float(elapsed_ms)),
                tool_call_count=max(0, int(tool_call_count)),
                token_count=max(0, int(token_count)),
                token_count_source=str(token_count_source or "").strip(),
                error=str(error),
            )
        )

    def update_draft(
        self,
        project_id: str,
        *,
        changes: dict[str, Any],
        expected_revision: int,
    ) -> ProjectDescriptionDocument:
        current = self.get_experience(project_id)
        if current.status is ProjectDescriptionStatus.USER_CONFIRMED:
            raise ValueError("Confirmed project experience must be reopened by re-analysis before editing")
        unsupported = set(changes) - ProjectDescriptionService.EDITABLE_FIELDS
        if unsupported:
            raise ValueError(f"Unsupported project description fields: {sorted(unsupported)}")
        normalized = {
            key: tuple(value) if key in self._TUPLE_FIELDS else value
            for key, value in changes.items()
        }
        return self.career_store.revise_record(
            current.project_description_id,
            actor="user",
            reason="User edited project experience draft",
            changes=normalized,
            expected_revision=expected_revision,
        )

    def confirm(
        self,
        project_id: str,
        *,
        changes: dict[str, Any] | None = None,
        expected_revision: int,
    ) -> ProjectDescriptionDocument:
        current = self.get_experience(project_id)
        source = self.project_store.get(project_id, refresh=True)
        if source.project_version != current.project_version:
            self.description_service.mark_source_changed(
                project_id,
                current_project_version=source.project_version,
            )
            raise RevisionConflictError("Project source changed; re-analysis is required before confirmation")
        edits = dict(changes or {})
        try:
            return self.description_service.confirm(
                current.project_description_id,
                changes=edits,
                actor="user",
                expected_revision=expected_revision,
            )
        except RevisionConflictError:
            # A background analysis can publish a newer draft after the UI
            # loaded its revision but before the user clicks Confirm.  A
            # confirmation without edits is safe to replay against that
            # latest draft; user edits must remain optimistic-concurrency
            # protected and therefore keep raising the conflict.
            if edits:
                raise
            latest = self.career_store.get(
                current.project_description_id,
                expected_type=ProjectDescriptionDocument,
            )
            latest_source = self.project_store.get(project_id, refresh=True)
            if latest_source.project_version != latest.project_version:
                self.description_service.mark_source_changed(
                    project_id,
                    current_project_version=latest_source.project_version,
                )
                raise RevisionConflictError(
                    "Project source changed; re-analysis is required before confirmation"
                )
            if latest.status is ProjectDescriptionStatus.USER_CONFIRMED:
                return latest
            if latest.status is not ProjectDescriptionStatus.NEEDS_USER_CONFIRMATION:
                raise
            return self.description_service.confirm(
                latest.project_description_id,
                changes={},
                actor="user",
                expected_revision=latest.revision,
            )


__all__ = [
    "PROJECT_RUN_SCHEMA_VERSION",
    "ProjectAnalysisRun",
    "ProjectAnalysisRunStore",
    "ProjectAnalysisValidation",
    "ProjectExperienceResult",
    "ProjectExperienceService",
    "ProjectUnderstandingValidator",
]
