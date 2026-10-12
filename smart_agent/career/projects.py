"""Traceable project-description generation and user confirmation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from smart_agent.career.materials import CandidateFact, MaterialManifest
from smart_agent.career.models import (
    EvidenceFact,
    EvidenceType,
    FactStatus,
    ProjectDescriptionDocument,
    ProjectDescriptionStatus,
    SourceReference,
    SourceType,
    create_personal_evidence,
)
from smart_agent.career.project_library import ProjectSource, ProjectUnderstanding
from smart_agent.career.store import CareerStore, RecordNotFoundError


class ProjectDescriptionGenerator(Protocol):
    name: str
    model: str

    def generate(
        self,
        *,
        project_id: str,
        manifests: tuple[MaterialManifest, ...],
        candidate_facts: tuple[CandidateFact, ...],
    ) -> ProjectDescriptionDocument: ...


def _source_key(ref: SourceReference) -> tuple[str, str, str]:
    return ref.source_id, ref.content_hash, json.dumps(ref.locator, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True, slots=True)
class HeuristicProjectDescriptionGenerator:
    """Offline baseline that reorganizes extracted project facts without inference."""

    name: str = "heuristic-project-description-v1"
    model: str = "offline"

    def generate(
        self,
        *,
        project_id: str,
        manifests: tuple[MaterialManifest, ...],
        candidate_facts: tuple[CandidateFact, ...],
    ) -> ProjectDescriptionDocument:
        selected_project_id = str(project_id or "").strip()
        if not selected_project_id:
            raise ValueError("project_id is required.")
        if not manifests:
            raise ValueError(f"No imported materials exist for project: {selected_project_id}")
        facts = tuple(
            fact
            for fact in candidate_facts
            if fact.material_id in {item.material_id for item in manifests}
        )
        if not facts:
            raise ValueError("Project description generation requires extracted source facts.")

        source_by_key = {
            _source_key(ref): ref
            for fact in facts
            for ref in fact.source_refs
        }
        sources = tuple(source_by_key.values())
        technologies = tuple(
            dict.fromkeys(str(fact.value).strip() for fact in facts if fact.field_name == "technology")
        )
        outcomes = tuple(
            dict.fromkeys(str(fact.value).strip() for fact in facts if fact.field_name == "impact")
        )
        features = tuple(
            dict.fromkeys(
                str(fact.value).strip()
                for fact in facts
                if fact.field_name not in {"technology", "impact"}
            )
        )
        field_evidence: dict[str, tuple[SourceReference, ...]] = {"title": (sources[0],)}
        for target, field_name in (
            ("technologies", "technology"),
            ("outcomes", "impact"),
        ):
            refs = tuple(
                {
                    _source_key(ref): ref
                    for fact in facts
                    if fact.field_name == field_name
                    for ref in fact.source_refs
                }.values()
            )
            if refs:
                field_evidence[target] = refs
        feature_refs = tuple(
            {
                _source_key(ref): ref
                for fact in facts
                if fact.field_name not in {"technology", "impact"}
                for ref in fact.source_refs
            }.values()
        )
        if feature_refs:
            field_evidence["implemented_features"] = feature_refs

        project_version = hashlib.sha256(
            json.dumps(
                sorted((item.material_id, item.revision, item.content_hash) for item in manifests),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        identity = hashlib.sha256(selected_project_id.encode("utf-8")).hexdigest()[:20]
        return ProjectDescriptionDocument(
            project_description_id=f"project-description-{identity}",
            project_id=selected_project_id,
            title=manifests[0].label,
            technologies=technologies,
            implemented_features=features,
            outcomes=outcomes,
            material_ids=tuple(item.material_id for item in manifests),
            field_evidence=field_evidence,
            status=ProjectDescriptionStatus.NEEDS_USER_CONFIRMATION,
            generator=self.name,
            model=self.model,
            project_version=project_version,
            source_refs=sources,
        )


class ProjectDescriptionService:
    EDITABLE_FIELDS = frozenset(
        {
            "title",
            "project_type",
            "background",
            "goals",
            "problem_statement",
            "role",
            "responsibilities",
            "core_flow",
            "architecture",
            "technology_choices",
            "core_modules",
            "technologies",
            "technical_tags",
            "implemented_features",
            "outcomes",
            "unknowns",
            "challenges",
            "timeline",
            "team_size",
            "collaboration",
            "links",
        }
    )

    def __init__(
        self,
        store: CareerStore,
        *,
        generator: ProjectDescriptionGenerator | None = None,
    ) -> None:
        self.store = store
        self.generator = generator or HeuristicProjectDescriptionGenerator()

    def generate(
        self,
        *,
        project_id: str,
        manifests: tuple[MaterialManifest, ...],
        candidate_facts: tuple[CandidateFact, ...],
    ) -> ProjectDescriptionDocument:
        proposed = self.generator.generate(
            project_id=project_id,
            manifests=manifests,
            candidate_facts=candidate_facts,
        )
        return self._persist_proposed(proposed)

    def generate_from_understanding(
        self,
        *,
        source: ProjectSource,
        understanding: ProjectUnderstanding,
        analysis_run_id: str,
        generator: str = "project-analyst-agent-v1",
        model: str = "",
    ) -> ProjectDescriptionDocument:
        """Convert an observed project analysis into one confirmable experience unit."""

        source_refs, refs_by_analysis_field = self._analysis_source_refs(source, understanding)
        field_evidence: dict[str, tuple[SourceReference, ...]] = {}
        mapping = {
            "title": "title",
            "goal": "goals",
            "core_flow": "core_flow",
            "architecture": "architecture",
            "technology_choices": "technology_choices",
            "core_modules": "core_modules",
            "technologies": "technologies",
            "technical_tags": "technical_tags",
            "implemented_features": "implemented_features",
            "candidate_outcomes": "outcomes",
            "candidate_responsibilities": "responsibilities",
        }
        for analysis_field, target_field in mapping.items():
            refs = refs_by_analysis_field.get(analysis_field, ())
            if refs:
                field_evidence[target_field] = tuple(
                    {
                        _source_key(ref): ref
                        for ref in (*field_evidence.get(target_field, ()), *refs)
                    }.values()
                )
        for field_name in (
            "title",
            "goals",
            "core_flow",
            "architecture",
            "technology_choices",
            "core_modules",
            "technologies",
            "technical_tags",
            "implemented_features",
            "outcomes",
            "responsibilities",
        ):
            if field_name not in field_evidence:
                field_evidence[field_name] = source_refs

        architecture_parts = [understanding.architecture]
        if understanding.core_flow:
            architecture_parts.append("核心流程：\n" + "\n".join(f"- {item}" for item in understanding.core_flow))
        if understanding.technology_choices:
            architecture_parts.append(
                "技术选型：\n" + "\n".join(f"- {item}" for item in understanding.technology_choices)
            )
        architecture = "\n\n".join(part for part in architecture_parts if part).strip()
        identity = hashlib.sha256(source.project_id.encode("utf-8")).hexdigest()[:20]
        proposed = ProjectDescriptionDocument(
            project_description_id=f"project-description-{identity}",
            project_id=source.project_id,
            title=understanding.title,
            background=understanding.analysis_summary,
            goals=(understanding.goal,) if understanding.goal else (),
            responsibilities=understanding.candidate_responsibilities,
            core_flow=understanding.core_flow,
            architecture=architecture,
            technology_choices=understanding.technology_choices,
            core_modules=understanding.core_modules,
            technologies=understanding.technologies,
            technical_tags=understanding.technical_tags,
            implemented_features=understanding.implemented_features,
            outcomes=understanding.candidate_outcomes,
            unknowns=understanding.unknowns,
            confidence_by_field=understanding.confidence_by_field,
            material_ids=(),
            field_evidence=field_evidence,
            status=ProjectDescriptionStatus.NEEDS_USER_CONFIRMATION,
            generator=f"{generator}:{analysis_run_id}",
            model=model,
            project_version=source.project_version,
            source_refs=source_refs,
        )
        return self._persist_proposed(proposed)

    @staticmethod
    def _analysis_source_refs(
        source: ProjectSource,
        understanding: ProjectUnderstanding,
    ) -> tuple[tuple[SourceReference, ...], dict[str, tuple[SourceReference, ...]]]:
        root = Path(source.source_location).resolve()
        refs: list[SourceReference] = []
        by_field: dict[str, list[SourceReference]] = {}
        for item in understanding.source_refs:
            relative_text = str(item.get("path") or "").strip().replace("\\", "/")
            if not relative_text:
                continue
            target = (root / relative_text).resolve() if root.is_dir() else root
            try:
                target.relative_to(root if root.is_dir() else root.parent)
            except ValueError:
                continue
            if not target.is_file():
                continue
            try:
                raw = target.read_bytes()
            except OSError:
                continue
            line = max(1, int(item.get("line") or 1))
            if target.suffix.casefold() in {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".webp"}:
                excerpt = str(item.get("excerpt") or "").strip()
            else:
                lines = raw.decode("utf-8", errors="ignore").splitlines()
                excerpt = (
                    "\n".join(lines[max(0, line - 2) : min(len(lines), line + 2)]).strip()
                    if lines
                    else str(item.get("excerpt") or "").strip()
                )
            ref = SourceReference.create(
                source_id=f"project:{source.project_id}:{relative_text}",
                source_type=SourceType.REPOSITORY,
                locator={
                    "project_id": source.project_id,
                    "project_version": source.project_version,
                    "path": relative_text,
                    "line": line,
                },
                raw_content=raw,
                excerpt=excerpt[:1000],
            )
            refs.append(ref)
            field_name = str(item.get("field") or "").strip()
            if field_name:
                by_field.setdefault(field_name, []).append(ref)
        if not refs:
            payload = json.dumps(source.to_dict(), ensure_ascii=False, sort_keys=True)
            refs.append(
                SourceReference.create(
                    source_id=f"project-source:{source.project_id}",
                    source_type=SourceType.REPOSITORY,
                    locator={
                        "project_id": source.project_id,
                        "project_version": source.project_version,
                        "path": ".",
                    },
                    raw_content=payload,
                    excerpt=source.display_name,
                )
            )
        unique = tuple({_source_key(ref): ref for ref in refs}.values())
        normalized_by_field = {
            name: tuple({_source_key(ref): ref for ref in field_refs}.values())
            for name, field_refs in by_field.items()
        }
        return unique, normalized_by_field

    def _persist_proposed(self, proposed: ProjectDescriptionDocument) -> ProjectDescriptionDocument:
        try:
            current = self.store.get(
                proposed.project_description_id,
                expected_type=ProjectDescriptionDocument,
            )
        except RecordNotFoundError:
            self.store.put(proposed)
            return proposed
        persisted_fields = (
            "title",
            "project_type",
            "background",
            "goals",
            "problem_statement",
            "role",
            "responsibilities",
            "core_flow",
            "architecture",
            "technology_choices",
            "core_modules",
            "technologies",
            "technical_tags",
            "implemented_features",
            "outcomes",
            "unknowns",
            "confidence_by_field",
            "challenges",
            "timeline",
            "team_size",
            "collaboration",
            "links",
            "material_ids",
            "field_evidence",
            "generator",
            "model",
            "project_version",
            "evidence_id",
            "source_refs",
            "status",
        )
        if current.project_version == proposed.project_version:
            if current.status is ProjectDescriptionStatus.USER_CONFIRMED:
                return current
            if all(getattr(current, field_name) == getattr(proposed, field_name) for field_name in persisted_fields):
                return current
        changes = {
            field_name: getattr(proposed, field_name)
            for field_name in persisted_fields
        }
        if current.evidence_id:
            try:
                previous_evidence = self.store.get(current.evidence_id)
            except RecordNotFoundError:
                previous_evidence = None
            if previous_evidence is not None and hasattr(previous_evidence, "facts"):
                for fact in previous_evidence.facts:
                    if fact.status is not FactStatus.INVALIDATED:
                        self.store.invalidate_fact(
                            previous_evidence.evidence_id,
                            fact.fact_id,
                            actor="project_description_generator",
                            reason="Project source material changed; previous description requires confirmation",
                        )
            changes["evidence_id"] = ""
        return self.store.revise_record(
            current.project_description_id,
            actor=proposed.generator or self.generator.name,
            reason="Regenerate project description after source material changed",
            changes=changes,
        )

    def confirm(
        self,
        project_description_id: str,
        *,
        changes: dict[str, Any] | None = None,
        actor: str = "user",
        expected_revision: int | None = None,
    ) -> ProjectDescriptionDocument:
        current = self.store.get(
            str(project_description_id).strip(),
            expected_type=ProjectDescriptionDocument,
        )
        edits = dict(changes or {})
        if expected_revision is not None and current.revision != int(expected_revision):
            from smart_agent.career.store import RevisionConflictError

            raise RevisionConflictError(
                f"Expected {current.project_description_id}@{expected_revision}, found revision {current.revision}."
            )
        unsupported = set(edits) - self.EDITABLE_FIELDS
        if unsupported:
            raise ValueError(f"Unsupported project description fields: {sorted(unsupported)}")
        if current.status is ProjectDescriptionStatus.USER_CONFIRMED and not edits:
            return current
        revision_changes: dict[str, Any] = {
            name: tuple(value) if name in {
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
            } else value
            for name, value in edits.items()
        }
        if edits:
            raw_edits = json.dumps(edits, ensure_ascii=False, sort_keys=True, default=str)
            user_source = SourceReference.create(
                source_id=f"user-{current.project_description_id}",
                source_type=SourceType.USER_INPUT,
                locator={"project_description_id": current.project_description_id, "action": "confirm"},
                raw_content=raw_edits,
                excerpt=raw_edits[:500],
            )
            revision_changes["source_refs"] = (*current.source_refs, user_source)
            revision_changes["field_evidence"] = {
                **current.field_evidence,
                **{name: (user_source,) for name in edits},
            }
            if current.evidence_id:
                try:
                    previous_evidence = self.store.get(current.evidence_id)
                except RecordNotFoundError:
                    previous_evidence = None
                if previous_evidence is not None and hasattr(previous_evidence, "facts"):
                    for fact in previous_evidence.facts:
                        if fact.status is not FactStatus.INVALIDATED:
                            self.store.invalidate_fact(
                                previous_evidence.evidence_id,
                                fact.fact_id,
                                actor="project_description_editor",
                                reason="Project description edited; previous evidence revision is superseded",
                            )
                revision_changes["evidence_id"] = ""
        if not revision_changes.get("evidence_id", current.evidence_id):
            facts: list[EvidenceFact] = []
            source_by_field = dict(revision_changes.get("field_evidence", current.field_evidence))
            for field_name in (
                "background",
                "goals",
                "problem_statement",
                "role",
                "responsibilities",
                "core_flow",
                "architecture",
                "technology_choices",
                "core_modules",
                "technologies",
                "implemented_features",
                "outcomes",
                "challenges",
                "timeline",
                "collaboration",
            ):
                value = revision_changes.get(field_name, getattr(current, field_name, ""))
                if isinstance(value, (tuple, list)):
                    value = list(value)
                if value in ("", (), [], None):
                    continue
                facts.append(
                    EvidenceFact.create(
                        field_name,
                        value,
                        source_refs=source_by_field.get(field_name, current.source_refs),
                        status=FactStatus.USER_CONFIRMED,
                        confidence=1.0,
                        actor="project_description_confirmer",
                    )
                )
            if not facts:
                raise ValueError("Project description confirmation requires at least one non-empty fact.")
            evidence = create_personal_evidence(
                EvidenceType.PROJECT,
                label=str(revision_changes.get("title", current.title)),
                facts=facts,
                source_refs=current.source_refs,
                actor="project_description_confirmer",
            )
            self.store.put(evidence)
            revision_changes["evidence_id"] = evidence.evidence_id
        revision_changes["status"] = ProjectDescriptionStatus.USER_CONFIRMED
        return self.store.revise_record(
            current.project_description_id,
            actor=actor,
            reason="User confirmed project description",
            changes=revision_changes,
            expected_revision=current.revision,
        )

    def mark_source_changed(
        self,
        project_id: str,
        *,
        current_project_version: str,
        actor: str = "project_library",
    ) -> ProjectDescriptionDocument | None:
        """Make prior project facts unusable as soon as the registered source version changes."""

        requested_key = str(project_id or "").strip().casefold()
        current = next(
            (
                item
                for item in self.store.list_latest(ProjectDescriptionDocument)
                if item.project_id.casefold() == requested_key
            ),
            None,
        )
        if current is None or current.project_version == current_project_version:
            return current
        warning = "项目来源版本已变化，需要重新分析并确认个人职责和成果"
        if (
            current.status is ProjectDescriptionStatus.NEEDS_USER_CONFIRMATION
            and not current.evidence_id
            and warning in current.unknowns
        ):
            return current
        if current.evidence_id:
            try:
                evidence = self.store.get(current.evidence_id)
            except RecordNotFoundError:
                evidence = None
            if evidence is not None and hasattr(evidence, "facts"):
                for fact in evidence.facts:
                    if fact.status is not FactStatus.INVALIDATED:
                        self.store.invalidate_fact(
                            evidence.evidence_id,
                            fact.fact_id,
                            actor=actor,
                            reason="Project source version changed; re-analysis and user confirmation are required",
                        )
        return self.store.revise_record(
            current.project_description_id,
            actor=actor,
            reason="Project source version changed",
            changes={
                "status": ProjectDescriptionStatus.NEEDS_USER_CONFIRMATION,
                "evidence_id": "",
                "unknowns": tuple(dict.fromkeys((*current.unknowns, warning))),
            },
        )
