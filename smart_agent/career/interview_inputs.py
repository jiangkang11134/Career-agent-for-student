"""User-confirmed interview documents, isolated from the Career material catalog."""

from .models import (
    AwardEvidence,
    EducationEvidence,
    EvidenceFact,
    EvidenceType,
    FactStatus,
    InternshipEvidence,
    JDDocument,
    ProjectEvidence,
    ResearchEvidence,
    ResumeClaim,
    ResumeDocument,
    ResumeDocumentStatus,
    ResumePlan,
    SkillsEvidence,
    SourceReference,
    record_from_dict,
    record_id,
    record_to_dict,
    split_resume_source_sections,
    version_ref,
)
from .workspace_context import WorkspaceContextStore


def build_records(bundle_id, jd_text, resume_text, role_title):
    prefix = "input:" + bundle_id + ":"
    source = SourceReference.create(
        source_id=bundle_id, source_type="user_input", locator={"bundle_id": bundle_id}, raw_content=resume_text
    )
    jd_source = SourceReference.create(
        source_id=bundle_id + "-jd", source_type="user_input", locator={"bundle_id": bundle_id}, raw_content=jd_text
    )
    jd = JDDocument(jd_id=prefix + "jd", raw_text=jd_text, role_title=role_title, source_refs=(jd_source,))
    plan = ResumePlan(
        plan_id=prefix + "plan",
        jd_ref=version_ref(jd),
        item_refs=(),
        selected_item_ids=(),
        excluded_reasons={},
        section_order=(),
        constraints={"source": "user_provided"},
        source_refs=(source,),
    )
    records = [jd, plan]
    claims = []
    sections = {}
    for block in split_resume_source_sections(resume_text, resume_source_id=bundle_id, source_revision=1):
        if not block.text.strip():
            continue
        name = block.section_id
        fact = EvidenceFact(
            fact_id=prefix + name + "-fact",
            field_name="content",
            value=block.text,
            source_refs=(source,),
            status=FactStatus.USER_CONFIRMED,
        )
        kind = {"internships": "internship", "projects": "project", "awards": "award"}.get(name, name)
        evidence_class = {
            "education": EducationEvidence,
            "awards": AwardEvidence,
            "internships": InternshipEvidence,
            "projects": ProjectEvidence,
            "research": ResearchEvidence,
            "skills": SkillsEvidence,
        }[name]
        evidence = evidence_class(
            evidence_id=prefix + name + "-evidence",
            evidence_type=EvidenceType(kind),
            label=block.text.splitlines()[0][:100],
            facts=(fact,),
            source_refs=(source,),
        )
        claim = ResumeClaim(
            claim_id=prefix + name + "-claim",
            text=block.text,
            evidence_refs=(version_ref(evidence),),
            evidence_fact_ids=(fact.fact_id,),
            requirement_refs=(),
            source_refs=(source,),
        )
        records.extend((evidence, claim))
        claims.append(claim)
        sections[name] = (claim.claim_id,)
    document = ResumeDocument(
        resume_document_id=prefix + "resume",
        jd_ref=version_ref(jd),
        plan_ref=version_ref(plan),
        claim_refs=tuple(version_ref(c) for c in claims),
        sections=sections,
        source_refs=(source,),
        status=ResumeDocumentStatus.USER_PROVIDED,
    )
    records.append(document)
    return {record_id(item): record_to_dict(item) for item in records}


class InterviewInputStore:
    """Read-only overlay; regular business records keep their existing store."""

    def __init__(self, base, owner):
        self.base = base
        self.root = base.root
        self.contexts = WorkspaceContextStore(self.root, owner)
        self.cache = {}

    def get(self, identity, revision=None, *, expected_type=None):
        if not identity.startswith("input:"):
            return self.base.get(identity, revision, expected_type=expected_type)
        parts = identity.split(":")
        if len(parts) != 3 or revision not in (None, 1):
            raise ValueError("Invalid interview input reference")
        bundle_id = parts[1]
        if bundle_id not in self.cache:
            bundle = self.contexts.get(bundle_id)
            if not bundle.get("confirmed"):
                raise ValueError("Interview inputs are not confirmed")
            self.cache[bundle_id] = {key: record_from_dict(value) for key, value in bundle["records"].items()}
        value = self.cache[bundle_id][identity]
        if expected_type and not isinstance(value, expected_type):
            raise ValueError("Interview input type mismatch")
        return value

    def list_latest(self, record_type):
        imported = tuple(
            item for records in self.cache.values() for item in records.values() if isinstance(item, record_type)
        )
        return tuple(self.base.list_latest(record_type)) + imported
