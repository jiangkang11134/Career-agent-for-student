"""Authenticated workspace context HTTP boundary."""

import asyncio
import hashlib
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from smart_agent.career.workspace_context import ContextConflict, WorkspaceContextStore

router = APIRouter(prefix="/api/career/contexts")


@router.post("/extract")
async def extract(file: UploadFile):
    from smart_agent.career.project_library import extract_project_document_text
    from smart_agent.web import app as web

    store()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".markdown", ".pdf", ".docx"}:
        raise HTTPException(400, "Supported formats: TXT, Markdown, PDF, DOCX")
    content = await web._read_limited_upload(file, limit=10 * 1024 * 1024)
    if not content:
        raise HTTPException(400, "Empty file")
    with tempfile.TemporaryDirectory(prefix="interview-input-") as directory:
        target = Path(directory) / ("source" + suffix)
        target.write_bytes(content)
        try:
            text, parser = await asyncio.to_thread(extract_project_document_text, target)
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
    if not text.strip() or len(text) > 20000:
        raise HTTPException(422, "Document must contain 1-20000 readable characters")
    return {
        "text": text,
        "parser": parser,
        "filename": file.filename,
        "content_hash": hashlib.sha256(content).hexdigest(),
    }


@router.post("/interview-inputs")
def import_inputs(data: dict):
    jd = data.get("jd_text")
    resume = data.get("resume_text")
    role = data.get("role_title", "")
    if any(
        not isinstance(v, str) or not v.strip() or len(v) > limit
        for v, limit in ((jd, 20000), (resume, 20000), (role, 120))
    ):
        raise HTTPException(400, "JD, resume and role title are required and must respect size limits")
    storage = store()
    state = storage.create("interviewer")
    return storage.commit(
        state["session_id"],
        1,
        "input_bundle_imported",
        {
            "jd_text": jd.strip(),
            "resume_text": resume.strip(),
            "role_title": role.strip(),
            "confirmed": False,
            "source": "user_provided",
            "content_hash": hashlib.sha256((jd + "\0" + resume).encode()).hexdigest(),
        },
        uuid.uuid4().hex,
    )


@router.post("/{identity}/confirm")
def confirm(identity: str, data: dict):
    from smart_agent.career.interview_inputs import build_records

    state = read(identity)
    if state.get("feature") != "interviewer" or not state.get("resume_text"):
        raise HTTPException(400, "Not an interview input bundle")
    if state.get("confirmed"):
        return state
    records = build_records(identity, state["jd_text"], state["resume_text"], state["role_title"])
    try:
        return store().commit(
            identity,
            data.get("expected_revision"),
            "input_bundle_confirmed",
            {"confirmed": True, "records": records},
            data.get("request_id"),
        )
    except ContextConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def store():
    from smart_agent.web import app as web

    user = web._CURRENT_AUTH_USER.get()
    if user is None:
        raise HTTPException(401, "Authentication required")
    return WorkspaceContextStore(web._career_root(), user.user_id)


def read(identity):
    try:
        return store().get(identity)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("")
def create(data: dict):
    try:
        return store().create(data.get("feature", "career"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{identity}")
def get(identity: str):
    return read(identity)


@router.get("/{identity}/events")
def events(identity: str, after: int = 0):
    read(identity)
    return {"events": store().events(identity, max(0, after))}


@router.get("/{identity}/replay")
def replay(identity: str):
    read(identity)
    return store().replay(identity)


@router.patch("/{identity}")
def select(identity: str, data: dict):
    from smart_agent.career import CareerStore, InterviewSessionStore
    from smart_agent.web import app as web

    read(identity)
    patch = data.get("patch")
    if not isinstance(patch, dict) or set(patch) - {"active_run_id", "active_interview_id", "stage", "input_bundle_id"}:
        raise HTTPException(400, "Unsupported context fields")
    patch = dict(patch)
    if any(not isinstance(value, str) or len(value) > 128 for value in patch.values()):
        raise HTTPException(400, "Invalid context field")
    if "stage" in patch and patch["stage"] not in {
        "library",
        "evidence",
        "resume",
        "coach",
        "interview",
        "history",
        "admin",
        "application",
    }:
        raise HTTPException(400, "Unsupported stage")
    if "active_run_id" in patch:
        if patch["active_run_id"]:
            web._resolve_resume_run(CareerStore(web._career_root()), patch["active_run_id"])
        patch.update(active_interview_id="", input_bundle_id="", coach_messages=[])
    if data["patch"].get("active_interview_id"):
        try:
            interview = InterviewSessionStore(web._career_root() / "interviews").get(
                data["patch"]["active_interview_id"]
            )
        except KeyError as exc:
            raise HTTPException(404, "Interview not found") from exc
        bundle_id = interview.jd_id.split(":")[1] if (interview.jd_id or "").startswith("input:") else ""
        if bundle_id:
            read(bundle_id)
        patch.update(
            active_interview_id=interview.interview_id,
            active_run_id=interview.resume_run_id or "",
            input_bundle_id=bundle_id,
        )
    if data["patch"].get("input_bundle_id"):
        bundle = read(data["patch"]["input_bundle_id"])
        if bundle.get("feature") != "interviewer" or not bundle.get("confirmed"):
            raise HTTPException(409, "Confirm interview inputs first")
        patch.update(input_bundle_id=bundle["session_id"], active_run_id="", active_interview_id="", coach_messages=[])
    try:
        return store().commit(
            identity, data.get("expected_revision"), "context_selected", patch, data.get("request_id")
        )
    except ContextConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
