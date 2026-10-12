"""Append-only storage and dependency tracking for Career records."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from smart_agent.career.models import (
    CAREER_SCHEMA_VERSION,
    CareerRecord,
    ChangeAction,
    ChangeEvent,
    ClaimStatus,
    EvidenceFact,
    EvidenceMatch,
    FactStatus,
    JDDocument,
    JDRequirement,
    PersonalEvidence,
    ResumeClaim,
    ResumeDocument,
    ResumeDocumentStatus,
    ResumeItem,
    ResumePlan,
    ResumeRun,
    ResumeSource,
    ResumeTemplate,
    SourceReference,
    VerificationReport,
    VerificationStatus,
    VersionedCareerRecord,
    VersionRef,
    record_from_dict,
    record_id,
    record_to_dict,
    version_ref,
)


class CareerStoreError(RuntimeError):
    """Base error for Career storage failures."""


class RecordNotFoundError(CareerStoreError):
    """Raised when an exact record revision does not exist."""


class RevisionConflictError(CareerStoreError):
    """Raised when an immutable revision would be overwritten or skipped."""


class ReferenceIntegrityError(CareerStoreError):
    """Raised when a record cites missing or incompatible records."""


class CareerStoreCorruptionError(CareerStoreError):
    """Raised when persisted content fails parsing or hash validation."""


@dataclass(frozen=True, slots=True)
class ClaimTrace:
    claim: ResumeClaim
    evidence: tuple[PersonalEvidence, ...]
    facts: tuple[EvidenceFact, ...]
    requirements: tuple[JDRequirement, ...]
    source_refs: tuple[SourceReference, ...]


@dataclass(frozen=True, slots=True)
class ResumeReplay:
    """All exact revisions needed to reproduce a stored resume document."""

    document: ResumeDocument
    plan: ResumePlan
    jd: JDDocument
    claims: tuple[ResumeClaim, ...]
    evidence: tuple[PersonalEvidence, ...]
    requirements: tuple[JDRequirement, ...]
    items: tuple[ResumeItem, ...]
    resume_source: ResumeSource | None


class CareerStore:
    """Small append-only JSON store whose relationships pin exact revisions."""

    _INDEX_FILE = "index.json"
    _ROOT_LOCKS: dict[str, threading.RLock] = {}
    _ROOT_LOCKS_GUARD = threading.Lock()

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.records_dir = self.root / "records"
        self.index_path = self.root / self._INDEX_FILE
        root_key = str(self.root).casefold()
        with self._ROOT_LOCKS_GUARD:
            self._lock = self._ROOT_LOCKS.setdefault(root_key, threading.RLock())
        self.records_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._index = self._load_or_rebuild_index()

    def _refresh_index(self) -> None:
        """Reload the atomic index without rescanning every immutable record."""
        try:
            stored = json.loads(self.index_path.read_text(encoding="utf-8"))
            if stored.get("schema_version") != CAREER_SCHEMA_VERSION:
                raise ValueError("unsupported index schema")
            if not isinstance(stored.get("records"), dict):
                raise ValueError("invalid records index")
            self._index = stored
        except FileNotFoundError:
            self._index = self._load_or_rebuild_index()
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise CareerStoreCorruptionError(f"Cannot refresh Career index: {exc}") from exc

    @staticmethod
    def _empty_index() -> dict[str, Any]:
        return {"schema_version": CAREER_SCHEMA_VERSION, "records": {}}

    def _load_or_rebuild_index(self) -> dict[str, Any]:
        stored: dict[str, Any] | None = None
        if self.index_path.exists():
            try:
                stored = json.loads(self.index_path.read_text(encoding="utf-8"))
                if stored.get("schema_version") != CAREER_SCHEMA_VERSION:
                    raise ValueError("unsupported index schema")
                if not isinstance(stored.get("records"), dict):
                    raise ValueError("invalid records index")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                stored = None
        if stored is not None and self._index_matches_snapshot_files(stored):
            return stored
        rebuilt = self._rebuild_index()
        if stored != rebuilt:
            self._atomic_write_json(self.index_path, rebuilt)
        return rebuilt

    def _index_matches_snapshot_files(self, index: dict[str, Any]) -> bool:
        indexed_paths = {
            str(version.get("path") or "")
            for entry in index["records"].values()
            for version in dict(entry.get("versions") or {}).values()
        }
        snapshot_paths = {
            path.relative_to(self.root).as_posix()
            for path in self.records_dir.glob("*/*.json")
        }
        return indexed_paths == snapshot_paths

    def _rebuild_index(self) -> dict[str, Any]:
        index = self._empty_index()
        for path in sorted(self.records_dir.glob("*/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                record = record_from_dict(payload)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise CareerStoreCorruptionError(f"Cannot rebuild index from {path}: {exc}") from exc
            self._register(index, record, path)
        for identity, entry in index["records"].items():
            revisions = sorted(int(value) for value in entry["versions"])
            if revisions != list(range(1, int(entry["latest_revision"]) + 1)):
                raise CareerStoreCorruptionError(f"Non-contiguous revision history for {identity}.")
        return index

    def _record_path(self, identity: str, revision: int) -> Path:
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.records_dir / digest / f"{revision}.json"

    def _register(
        self,
        index: dict[str, Any],
        record: CareerRecord,
        path: Path,
    ) -> None:
        identity = record_id(record)
        records = index["records"]
        entry = records.setdefault(
            identity,
            {
                "record_type": type(record).__name__,
                "latest_revision": 0,
                "versions": {},
            },
        )
        if entry["record_type"] != type(record).__name__:
            raise CareerStoreCorruptionError(f"Record type changed for {identity}.")
        revision_key = str(record.revision)
        if revision_key in entry["versions"]:
            raise CareerStoreCorruptionError(f"Duplicate immutable revision {identity}@{record.revision}.")
        relative_path = path.relative_to(self.root).as_posix()
        entry["versions"][revision_key] = {
            "path": relative_path,
            "content_hash": record.content_hash,
        }
        entry["latest_revision"] = max(int(entry["latest_revision"]), record.revision)

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def latest_revision(self, identity: str) -> int:
        with self._lock:
            self._refresh_index()
            try:
                return int(self._index["records"][identity]["latest_revision"])
            except KeyError as exc:
                raise RecordNotFoundError(f"Unknown Career record: {identity}") from exc

    def get(
        self,
        identity: str,
        revision: int | None = None,
        *,
        expected_type: type | tuple[type, ...] | None = None,
    ) -> CareerRecord:
        with self._lock:
            self._refresh_index()
            return self._get_locked(identity, revision, expected_type=expected_type)

    def _get_locked(
        self,
        identity: str,
        revision: int | None = None,
        *,
        expected_type: type | tuple[type, ...] | None = None,
    ) -> CareerRecord:
        try:
            entry = self._index["records"][identity]
            resolved_revision = int(revision or entry["latest_revision"])
            version = entry["versions"][str(resolved_revision)]
        except (KeyError, TypeError, ValueError) as exc:
            suffix = f"@{revision}" if revision is not None else ""
            raise RecordNotFoundError(f"Unknown Career record: {identity}{suffix}") from exc
        path = (self.root / version["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise CareerStoreCorruptionError(f"Indexed path escapes the Career store: {version['path']}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = record_from_dict(payload)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise CareerStoreCorruptionError(f"Cannot load {identity}@{resolved_revision}: {exc}") from exc
        serialized_hash = str(payload.get("content_hash") or "")
        if record.content_hash != version["content_hash"] and serialized_hash != version["content_hash"]:
            # Schema migrations can produce a new canonical record hash without
            # touching immutable JSON snapshots. Repair just this index entry
            # after record_from_dict has already verified that the payload is a
            # valid legacy or current record; do not scan the whole store on a
            # hot read path.
            next_index = json.loads(json.dumps(self._index))
            next_index["records"][identity]["versions"][str(resolved_revision)]["content_hash"] = record.content_hash
            self._atomic_write_json(self.index_path, next_index)
            self._index = next_index
        if record_id(record) != identity or record.revision != resolved_revision:
            raise CareerStoreCorruptionError(f"Identity mismatch for {identity}@{resolved_revision}.")
        if expected_type is not None and not isinstance(record, expected_type):
            raise ReferenceIntegrityError(
                f"Expected {identity}@{resolved_revision} to be {expected_type}, got {type(record).__name__}."
            )
        return record

    def list_latest(self, record_type: type | str | None = None) -> tuple[CareerRecord, ...]:
        with self._lock:
            self._refresh_index()
            if record_type is None:
                identities = sorted(self._index["records"])
            else:
                if isinstance(record_type, str):
                    record_type_names = {record_type}
                else:
                    record_type_names = {record_type.__name__}
                    pending = list(record_type.__subclasses__())
                    while pending:
                        subclass = pending.pop()
                        record_type_names.add(subclass.__name__)
                        pending.extend(subclass.__subclasses__())
                identities = sorted(
                    identity
                    for identity, entry in self._index["records"].items()
                    if entry.get("record_type") in record_type_names
                )
            records = tuple(self._get_locked(identity) for identity in identities)
        if record_type is None:
            return records
        if isinstance(record_type, str):
            return tuple(record for record in records if type(record).__name__ == record_type)
        return tuple(record for record in records if isinstance(record, record_type))

    def list_latest_payloads(self, record_type: str | None = None) -> tuple[dict[str, Any], ...]:
        """Read latest snapshots without dataclass validation for lightweight API reads."""
        with self._lock:
            self._refresh_index()
            names = {str(record_type)} if record_type else None
            payloads: list[dict[str, Any]] = []
            for identity, entry in sorted(self._index["records"].items()):
                if names is not None and str(entry.get("record_type")) not in names:
                    continue
                try:
                    version = entry["versions"][str(entry["latest_revision"])]
                    path = (self.root / version["path"]).resolve()
                    if not path.is_relative_to(self.root):
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        payloads.append(payload)
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue
            return tuple(payloads)

    def put(self, record: CareerRecord) -> CareerRecord:
        with self._lock:
            self._refresh_index()
            previous = self._current_or_none(record_id(record))
            self._put(record)
            if isinstance(record, PersonalEvidence) and isinstance(previous, PersonalEvidence):
                changed_fact_ids = self._changed_fact_ids(previous, record)
                if changed_fact_ids:
                    self._propagate_evidence_change(record, changed_fact_ids)
            elif isinstance(record, ResumeClaim) and isinstance(previous, ResumeClaim):
                self._propagate_claim_changes((record,))
            return record

    def revise_record(
        self,
        identity: str,
        *,
        actor: str,
        reason: str,
        changes: dict[str, Any],
        expected_revision: int | None = None,
    ) -> CareerRecord:
        """Append a normal revision while keeping identity and audit fields managed."""

        protected = {
            "revision",
            "schema_version",
            "history",
            "created_at",
            "updated_at",
            "content_hash",
        }
        if protected.intersection(changes):
            raise ValueError("Revision metadata is managed by CareerStore.")
        with self._lock:
            self._refresh_index()
            current = self._get_locked(identity)
            if expected_revision is not None and current.revision != int(expected_revision):
                raise RevisionConflictError(
                    f"Expected {identity}@{expected_revision}, found revision {current.revision}."
                )
            next_record = self._next_revision(
                current,
                identity=identity,
                actor=actor,
                reason=reason,
                changed_fields=tuple(sorted(changes)),
                **changes,
            )
            self.put(next_record)
            return next_record

    def _put(self, record: CareerRecord) -> None:
        record.verify_integrity()
        identity = record_id(record)
        previous = self._current_or_none(identity)
        self._validate_revision(record, previous)
        self._validate_references(record)
        path = self._record_path(identity, record.revision)
        if path.exists():
            raise RevisionConflictError(f"Immutable revision already exists: {identity}@{record.revision}")
        self._atomic_write_json(path, record_to_dict(record))
        next_index = json.loads(json.dumps(self._index))
        self._register(next_index, record, path)
        self._atomic_write_json(self.index_path, next_index)
        self._index = next_index

    def _current_or_none(self, identity: str) -> CareerRecord | None:
        if identity not in self._index["records"]:
            return None
        return self._get_locked(identity)

    def _validate_revision(self, record: CareerRecord, previous: CareerRecord | None) -> None:
        identity = record_id(record)
        if previous is None:
            if record.revision != 1:
                raise RevisionConflictError(f"First revision for {identity} must be 1.")
            return
        if type(previous) is not type(record):
            raise RevisionConflictError(f"Record type cannot change for {identity}.")
        if isinstance(record, ResumeSource):
            raise RevisionConflictError("ResumeSource is read-only and cannot receive another revision.")
        if record.revision != previous.revision + 1:
            raise RevisionConflictError(
                f"Expected {identity}@{previous.revision + 1}, got revision {record.revision}."
            )
        if record.created_at != previous.created_at:
            raise RevisionConflictError("created_at must remain stable across revisions.")
        if len(record.history) != len(previous.history) + 1 or record.history[:-1] != previous.history:
            raise RevisionConflictError("A revision must append exactly one history event.")
        event = record.history[-1]
        if (
            event.record_id != identity
            or event.from_revision != previous.revision
            or event.to_revision != record.revision
        ):
            raise RevisionConflictError("The latest history event does not describe this revision.")
        if isinstance(record, PersonalEvidence) and isinstance(previous, PersonalEvidence):
            if {fact.fact_id for fact in record.facts} != {fact.fact_id for fact in previous.facts}:
                raise RevisionConflictError("Fact IDs inside existing PersonalEvidence are immutable.")

    def _expect(self, ref: VersionRef, expected_type: type | tuple[type, ...], label: str) -> CareerRecord:
        try:
            # Reference validation runs inside the store lock after the index
            # has been refreshed. Re-reading the full index for every edge
            # makes large ResumeRun revisions grow linearly with ref count.
            return self._get_locked(ref.record_id, ref.revision, expected_type=expected_type)
        except (RecordNotFoundError, ReferenceIntegrityError) as exc:
            raise ReferenceIntegrityError(f"Invalid {label}: {ref.record_id}@{ref.revision}") from exc

    def _resolve_evidence_facts(
        self,
        evidence_refs: tuple[VersionRef, ...],
        fact_ids: tuple[str, ...],
        *,
        require_usable: bool,
    ) -> tuple[EvidenceFact, ...]:
        facts: dict[str, EvidenceFact] = {}
        for ref in evidence_refs:
            evidence = self._expect(ref, PersonalEvidence, "evidence reference")
            for fact in evidence.facts:
                if fact.fact_id in fact_ids:
                    facts[fact.fact_id] = fact
        missing = set(fact_ids) - set(facts)
        if missing:
            raise ReferenceIntegrityError(f"Unknown evidence facts: {sorted(missing)}")
        selected = tuple(facts[fact_id] for fact_id in fact_ids)
        if require_usable and any(not fact.is_usable for fact in selected):
            raise ReferenceIntegrityError("Resume content cannot cite unverified or invalidated facts.")
        return selected

    def _validate_references(self, record: CareerRecord) -> None:
        if isinstance(record, JDDocument):
            for ref in record.requirement_refs:
                requirement = self._expect(ref, JDRequirement, "JD requirement reference")
                if requirement.jd_ref.record_id != record.jd_id:
                    raise ReferenceIntegrityError("JDDocument cites a requirement from another JD.")
        elif isinstance(record, JDRequirement):
            self._expect(record.jd_ref, JDDocument, "JD document reference")
        elif isinstance(record, EvidenceMatch):
            self._expect(record.requirement_ref, JDRequirement, "JD requirement reference")
            if record.evidence_ref is not None:
                self._resolve_evidence_facts(
                    (record.evidence_ref,),
                    record.evidence_fact_ids,
                    require_usable=False,
                )
        elif isinstance(record, ResumeItem):
            self._resolve_evidence_facts(
                record.evidence_refs,
                record.evidence_fact_ids,
                require_usable=True,
            )
            for ref in record.requirement_refs:
                self._expect(ref, JDRequirement, "JD requirement reference")
        elif isinstance(record, ResumePlan):
            jd = self._expect(record.jd_ref, JDDocument, "JD document reference")
            for ref in record.item_refs:
                item = self._expect(ref, ResumeItem, "resume item reference")
                for requirement_ref in item.requirement_refs:
                    requirement = self._expect(
                        requirement_ref,
                        JDRequirement,
                        "resume item requirement reference",
                    )
                    if requirement.jd_ref.record_id != jd.jd_id:
                        raise ReferenceIntegrityError("ResumeItem cites a requirement from another JD.")
            if record.resume_source_ref is not None:
                self._expect(record.resume_source_ref, ResumeSource, "resume source reference")
        elif isinstance(record, ResumeTemplate):
            if record.resume_source_ref is not None:
                self._expect(record.resume_source_ref, ResumeSource, "resume template source reference")
        elif isinstance(record, ResumeClaim):
            self._resolve_evidence_facts(
                record.evidence_refs,
                record.evidence_fact_ids,
                require_usable=record.status in {ClaimStatus.DRAFT, ClaimStatus.SUPPORTED},
            )
            for ref in record.requirement_refs:
                self._expect(ref, JDRequirement, "JD requirement reference")
        elif isinstance(record, ResumeDocument):
            jd = self._expect(record.jd_ref, JDDocument, "JD document reference")
            plan = self._expect(record.plan_ref, ResumePlan, "resume plan reference")
            if plan.jd_ref.record_id != jd.jd_id:
                raise ReferenceIntegrityError("ResumeDocument plan belongs to another JD.")
            claims: list[ResumeClaim] = []
            for ref in record.claim_refs:
                claim = self._expect(ref, ResumeClaim, "resume claim reference")
                claims.append(claim)
                for requirement_ref in claim.requirement_refs:
                    requirement = self._expect(
                        requirement_ref,
                        JDRequirement,
                        "claim requirement reference",
                    )
                    if requirement.jd_ref.record_id != jd.jd_id:
                        raise ReferenceIntegrityError("ResumeClaim cites a requirement from another JD.")
            if (
                record.status is ResumeDocumentStatus.VERIFIED
                and any(claim.status is not ClaimStatus.SUPPORTED for claim in claims)
            ):
                raise ReferenceIntegrityError(
                    "A verified ResumeDocument can contain only supported claims."
                )
            if record.verification_report_ref is not None:
                report = self._expect(
                    record.verification_report_ref,
                    VerificationReport,
                    "verification report reference",
                )
                if report.resume_ref.record_id != record.resume_document_id:
                    raise ReferenceIntegrityError("ResumeDocument cites a report for another resume.")
                if (
                    record.status is ResumeDocumentStatus.VERIFIED
                    and report.status is not VerificationStatus.PASSED
                ):
                    raise ReferenceIntegrityError("A verified ResumeDocument requires a passed report.")
        elif isinstance(record, VerificationReport):
            self._expect(record.resume_ref, ResumeDocument, "resume document reference")
            for ref in record.checked_evidence_refs:
                self._expect(ref, PersonalEvidence, "checked evidence reference")
        elif isinstance(record, ResumeRun):
            jd = self._expect(record.jd_ref, JDDocument, "resume run JD reference")
            for ref in record.requirement_refs:
                requirement = self._expect(ref, JDRequirement, "resume run requirement reference")
                if requirement.jd_ref.record_id != jd.jd_id:
                    raise ReferenceIntegrityError("ResumeRun cites a requirement from another JD.")
            for ref in record.evidence_snapshot_refs:
                self._expect(ref, PersonalEvidence, "resume run evidence snapshot reference")
            for ref in record.match_refs:
                match = self._expect(ref, EvidenceMatch, "resume run match reference")
                if match.requirement_ref not in record.requirement_refs:
                    raise ReferenceIntegrityError("ResumeRun cites a match from another JD requirement set.")
            if record.plan_ref is not None:
                plan = self._expect(record.plan_ref, ResumePlan, "resume run plan reference")
                if plan.jd_ref.record_id != jd.jd_id:
                    raise ReferenceIntegrityError("ResumeRun plan belongs to another JD.")
            if record.resume_document_ref is not None:
                document = self._expect(
                    record.resume_document_ref,
                    ResumeDocument,
                    "resume run document reference",
                )
                if document.jd_ref.record_id != jd.jd_id:
                    raise ReferenceIntegrityError("ResumeRun document belongs to another JD.")
            if record.verification_report_ref is not None:
                report = self._expect(
                    record.verification_report_ref,
                    VerificationReport,
                    "resume run verification reference",
                )
                if (
                    record.resume_document_ref is None
                    or report.resume_ref.record_id != record.resume_document_ref.record_id
                ):
                    raise ReferenceIntegrityError("ResumeRun verification report belongs to another resume.")

    @staticmethod
    def _changed_fact_ids(previous: PersonalEvidence, current: PersonalEvidence) -> tuple[str, ...]:
        old = {fact.fact_id: fact.content_hash for fact in previous.facts}
        return tuple(fact.fact_id for fact in current.facts if old.get(fact.fact_id) != fact.content_hash)

    def affected_claims(
        self,
        evidence_id: str,
        fact_id: str | None = None,
    ) -> tuple[ResumeClaim, ...]:
        claims = self.list_latest(ResumeClaim)
        return tuple(
            claim
            for claim in claims
            if evidence_id in claim.evidence_ids
            and (fact_id is None or fact_id in claim.evidence_fact_ids)
        )

    def affected_resume_documents(self, claim_ids: tuple[str, ...] | list[str]) -> tuple[ResumeDocument, ...]:
        affected = set(claim_ids)
        return tuple(
            document
            for document in self.list_latest(ResumeDocument)
            if affected.intersection(ref.record_id for ref in document.claim_refs)
        )

    @staticmethod
    def _next_revision(
        record: VersionedCareerRecord,
        *,
        identity: str,
        actor: str,
        reason: str,
        changed_fields: tuple[str, ...],
        **changes: Any,
    ) -> VersionedCareerRecord:
        event = ChangeEvent.create(
            record_id=identity,
            action=ChangeAction.MODIFIED,
            from_revision=record.revision,
            actor=actor,
            reason=reason,
            changed_fields=changed_fields,
        )
        return replace(
            record,
            **changes,
            revision=record.revision + 1,
            history=(*record.history, event),
            updated_at=time.time(),
            content_hash="",
        )

    def _propagate_evidence_change(
        self,
        evidence: PersonalEvidence,
        changed_fact_ids: tuple[str, ...],
    ) -> None:
        affected: list[ResumeClaim] = []
        changed = set(changed_fact_ids)
        changed_facts = {fact.fact_id: fact for fact in evidence.facts if fact.fact_id in changed}
        for claim in self.affected_claims(evidence.evidence_id):
            if not changed.intersection(claim.evidence_fact_ids):
                continue
            next_refs = tuple(
                version_ref(evidence) if ref.record_id == evidence.evidence_id else ref
                for ref in claim.evidence_refs
            )
            invalidated = any(
                changed_facts[fact_id].status is FactStatus.INVALIDATED
                for fact_id in changed.intersection(claim.evidence_fact_ids)
            )
            next_claim = self._next_revision(
                claim,
                identity=claim.claim_id,
                actor="career_store",
                reason=f"Referenced evidence changed: {evidence.evidence_id}",
                changed_fields=("evidence_refs", "status"),
                evidence_refs=next_refs,
                status=ClaimStatus.INVALIDATED if invalidated else ClaimStatus.NEEDS_REVERIFICATION,
            )
            self._put(next_claim)
            affected.append(next_claim)
        self._propagate_claim_changes(tuple(affected))

    def _propagate_claim_changes(self, claims: tuple[ResumeClaim, ...]) -> None:
        if not claims:
            return
        by_id = {claim.claim_id: claim for claim in claims}
        for document in self.affected_resume_documents(tuple(by_id)):
            next_refs = tuple(
                version_ref(by_id[ref.record_id]) if ref.record_id in by_id else ref
                for ref in document.claim_refs
            )
            next_document = self._next_revision(
                document,
                identity=document.resume_document_id,
                actor="career_store",
                reason="A referenced resume claim changed",
                changed_fields=("claim_refs", "status"),
                claim_refs=next_refs,
                status=ResumeDocumentStatus.NEEDS_REVERIFICATION,
            )
            self._put(next_document)

    def revise_fact(
        self,
        evidence_id: str,
        fact_id: str,
        value: Any,
        *,
        actor: str,
        reason: str,
        source_refs: tuple[SourceReference, ...] | None = None,
        confidence: float | None = None,
    ) -> PersonalEvidence:
        with self._lock:
            evidence = self.get(evidence_id, expected_type=PersonalEvidence)
            current = evidence.fact(fact_id)
            next_fact = current.revise(
                value,
                actor=actor,
                reason=reason,
                source_refs=source_refs,
                confidence=confidence,
            )
            next_evidence = evidence.with_fact(next_fact, actor=actor, reason=reason)
            self.put(next_evidence)
            return next_evidence

    def confirm_fact(
        self,
        evidence_id: str,
        fact_id: str,
        *,
        actor: str,
        reason: str = "User confirmed fact",
    ) -> PersonalEvidence:
        return self._transition_fact(evidence_id, fact_id, "confirm", actor=actor, reason=reason)

    def invalidate_fact(
        self,
        evidence_id: str,
        fact_id: str,
        *,
        actor: str,
        reason: str,
    ) -> PersonalEvidence:
        return self._transition_fact(evidence_id, fact_id, "invalidate", actor=actor, reason=reason)

    def reverify_fact(
        self,
        evidence_id: str,
        fact_id: str,
        *,
        supported: bool,
        actor: str,
        reason: str,
        source_refs: tuple[SourceReference, ...] | None = None,
        confidence: float | None = None,
    ) -> PersonalEvidence:
        with self._lock:
            evidence = self.get(evidence_id, expected_type=PersonalEvidence)
            next_fact = evidence.fact(fact_id).reverify(
                supported=supported,
                actor=actor,
                reason=reason,
                source_refs=source_refs,
                confidence=confidence,
            )
            next_evidence = evidence.with_fact(next_fact, actor=actor, reason=reason)
            self.put(next_evidence)
            return next_evidence

    def _transition_fact(
        self,
        evidence_id: str,
        fact_id: str,
        operation: str,
        *,
        actor: str,
        reason: str,
    ) -> PersonalEvidence:
        with self._lock:
            evidence = self.get(evidence_id, expected_type=PersonalEvidence)
            fact = evidence.fact(fact_id)
            next_fact = (
                fact.confirm(actor=actor, reason=reason)
                if operation == "confirm"
                else fact.invalidate(actor=actor, reason=reason)
            )
            next_evidence = evidence.with_fact(next_fact, actor=actor, reason=reason)
            self.put(next_evidence)
            return next_evidence

    def reverify_claim(
        self,
        claim_id: str,
        *,
        supported: bool,
        actor: str,
        reason: str,
    ) -> ResumeClaim:
        with self._lock:
            claim = self.get(claim_id, expected_type=ResumeClaim)
            next_claim = self._next_revision(
                claim,
                identity=claim.claim_id,
                actor=actor,
                reason=reason,
                changed_fields=("status",),
                status=ClaimStatus.SUPPORTED if supported else ClaimStatus.INVALIDATED,
            )
            self.put(next_claim)
            return next_claim

    def mark_resume_verified(
        self,
        resume_document_id: str,
        *,
        actor: str,
        reason: str,
    ) -> ResumeDocument:
        with self._lock:
            document = self.get(resume_document_id, expected_type=ResumeDocument)
            claims = tuple(
                self.get(ref.record_id, ref.revision, expected_type=ResumeClaim)
                for ref in document.claim_refs
            )
            if any(claim.status is not ClaimStatus.SUPPORTED for claim in claims):
                raise ReferenceIntegrityError("Every resume claim must be supported before verification.")
            next_document = self._next_revision(
                document,
                identity=document.resume_document_id,
                actor=actor,
                reason=reason,
                changed_fields=("status",),
                status=ResumeDocumentStatus.VERIFIED,
            )
            self.put(next_document)
            return next_document

    def trace_claim(self, claim_id: str, revision: int | None = None) -> ClaimTrace:
        claim = self.get(claim_id, revision, expected_type=ResumeClaim)
        evidence = tuple(
            self.get(ref.record_id, ref.revision, expected_type=PersonalEvidence)
            for ref in claim.evidence_refs
        )
        fact_by_id = {
            fact.fact_id: fact
            for record in evidence
            for fact in record.facts
            if fact.fact_id in claim.evidence_fact_ids
        }
        facts = tuple(fact_by_id[fact_id] for fact_id in claim.evidence_fact_ids)
        requirements = tuple(
            self.get(ref.record_id, ref.revision, expected_type=JDRequirement)
            for ref in claim.requirement_refs
        )
        source_by_key: dict[tuple[str, str, str], SourceReference] = {}
        for source in (
            *(source for fact in facts for source in fact.source_refs),
            *(source for requirement in requirements for source in requirement.source_refs),
        ):
            key = (source.source_id, source.content_hash, json.dumps(source.locator, sort_keys=True))
            source_by_key[key] = source
        return ClaimTrace(
            claim=claim,
            evidence=evidence,
            facts=facts,
            requirements=requirements,
            source_refs=tuple(source_by_key.values()),
        )

    def replay_resume(self, resume_document_id: str, revision: int | None = None) -> ResumeReplay:
        document = self.get(resume_document_id, revision, expected_type=ResumeDocument)
        plan = self.get(document.plan_ref.record_id, document.plan_ref.revision, expected_type=ResumePlan)
        jd = self.get(document.jd_ref.record_id, document.jd_ref.revision, expected_type=JDDocument)
        claims = tuple(
            self.get(ref.record_id, ref.revision, expected_type=ResumeClaim)
            for ref in document.claim_refs
        )
        evidence_refs = {
            (ref.record_id, ref.revision): ref
            for claim in claims
            for ref in claim.evidence_refs
        }
        requirement_refs = {
            (ref.record_id, ref.revision): ref
            for claim in claims
            for ref in claim.requirement_refs
        }
        evidence = tuple(
            self.get(ref.record_id, ref.revision, expected_type=PersonalEvidence)
            for ref in evidence_refs.values()
        )
        requirements = tuple(
            self.get(ref.record_id, ref.revision, expected_type=JDRequirement)
            for ref in requirement_refs.values()
        )
        items = tuple(
            self.get(ref.record_id, ref.revision, expected_type=ResumeItem)
            for ref in plan.item_refs
        )
        resume_source = (
            self.get(
                plan.resume_source_ref.record_id,
                plan.resume_source_ref.revision,
                expected_type=ResumeSource,
            )
            if plan.resume_source_ref is not None
            else None
        )
        return ResumeReplay(
            document=document,
            plan=plan,
            jd=jd,
            claims=claims,
            evidence=evidence,
            requirements=requirements,
            items=items,
            resume_source=resume_source,
        )
