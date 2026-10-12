"""Definitions for dynamically loaded Smart Agent domains."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentDomain(str, Enum):
    AUTO = "auto"
    CODE = "code"
    CAREER = "career"


@dataclass(frozen=True)
class DomainModule:
    name: AgentDomain
    display_name: str
    description: str
    system_prompt: str
    memory_namespace: str
    skill_agent_type: str
    tool_names: frozenset[str]
    stage_tool_names: dict[str, frozenset[str]]


_CAREER_MATERIAL_TOOLS = frozenset({
    "inspect_career_workspace",
    "import_career_material",
    "extract_career_facts",
    "list_candidate_facts",
    "import_candidate_facts",
    "reject_candidate_facts",
    "list_career_materials",
    "read_career_material",
    "confirm_career_facts",
    "invalidate_career_facts",
    "register_project_source",
    "list_project_sources",
    "save_project_understanding",
    "career_task",
})
_CAREER_RESUME_TOOLS = _CAREER_MATERIAL_TOOLS | frozenset({
    "analyze_jd",
    "retrieve_career_evidence",
    "plan_resume_content",
    "generate_resume_document",
    "verify_resume_document",
    "revise_resume_document",
    "render_resume_pdf",
    "review_resume_evidence",
})
_CAREER_INTERVIEW_TOOLS = frozenset({
    "inspect_career_workspace",
    "import_interview_knowledge",
    "import_interview_knowledge_source",
    "list_interview_knowledge",
    "search_interview_knowledge",
    "build_jd_capability_tree",
    "build_interview_plan",
    "create_interview_session",
    "get_interview_session",
    "next_interview_question",
    "process_interview_answer",
    "generate_interview_depth_question",
    "generate_interview_breadth_question",
    "submit_interview_answer",
    "finish_interview_session",
    "career_task",
})
_CAREER_APPLICATION_TOOLS = frozenset({
    "inspect_career_workspace",
    "career_task",
    "render_resume_pdf",
})
_CAREER_ALL_TOOLS = (
    _CAREER_RESUME_TOOLS
    | _CAREER_INTERVIEW_TOOLS
    | _CAREER_APPLICATION_TOOLS
)


_MODULES = {
    AgentDomain.CODE: DomainModule(
        name=AgentDomain.CODE,
        display_name="Code Agent",
        description="Repository analysis, implementation, review, testing, and debugging.",
        memory_namespace="code",
        skill_agent_type="code",
        tool_names=frozenset({
            "list_files", "grep_files", "read_file", "write_file", "edit_file",
            "patch_file", "batch_copy", "batch_move", "batch_delete", "run_command",
            "task", "git", "find_symbols", "find_references", "get_ast_info",
            "code_review", "file_tree", "diff_viewer", "sandbox_test", "test_runner",
        }),
        stage_tool_names={"default": frozenset()},
        system_prompt=(
            "Active domain: Code Agent. Work as a repository-aware software engineering agent. "
            "Inspect relevant code before editing, keep changes scoped, and verify the result. "
            "Use only Code-domain capabilities and never access Career personal materials."
        ),
    ),
    AgentDomain.CAREER: DomainModule(
        name=AgentDomain.CAREER,
        display_name="Career Agent",
        description="JD analysis, evidence-based resumes, and adaptive interview practice.",
        memory_namespace="career",
        skill_agent_type="career",
        tool_names=frozenset(),
        stage_tool_names={
            "default": _CAREER_ALL_TOOLS,
            "materials": _CAREER_MATERIAL_TOOLS,
            "resume": _CAREER_RESUME_TOOLS,
            "interview": _CAREER_INTERVIEW_TOOLS,
            "application": _CAREER_APPLICATION_TOOLS,
        },
        system_prompt=(
            "Active domain: Career Agent. Help with job descriptions, evidence-based resumes, "
            "project experience, and adaptive mock interviews. Never invent experience, metrics, "
            "education, or project facts. For multi-step goals, inspect Career workspace state, decide the next "
            "action yourself, and respect every reported evidence gate; do not force the user through a fixed UI "
            "sequence. Generate resumes through ResumePlan and generate_resume_document. "
            "Always call verify_resume_document before rendering a PDF. Correct only report-scoped claims with "
            "revise_resume_document; after two failed correction rounds, require user confirmation. "
            "Use career_task to invoke the isolated Project Analyst, Interviewer, or Application child agent with "
            "only its declared artifacts. Do not inspect project repositories in the parent Career context, never "
            "modify them, and do not invoke Code Review capabilities. Register a project source first, delegate one "
            "project_analysis_request, then persist its project_understanding with save_project_understanding. "
            "Project analysis is never user-confirmed evidence by itself."
        ),
    ),
}


def get_domain_module(domain: AgentDomain | str) -> DomainModule:
    value = domain if isinstance(domain, AgentDomain) else AgentDomain(str(domain).lower())
    if value is AgentDomain.AUTO:
        value = AgentDomain.CODE
    return _MODULES[value]


def list_domain_modules() -> tuple[DomainModule, ...]:
    """Return the manually selectable domains in stable display order."""
    return (_MODULES[AgentDomain.CODE], _MODULES[AgentDomain.CAREER])
