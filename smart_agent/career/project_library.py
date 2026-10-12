"""Read-only project sources and structured project-understanding results."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from xml.etree import ElementTree

PROJECT_LIBRARY_SCHEMA_VERSION = "1.0"

_SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".runtime",
        ".codex",
        ".agents",
        ".tmp",
        ".idea",
        ".vscode",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "site-packages",
        "vendor",
        "dist",
        "build",
        "target",
        "coverage",
        "htmlcov",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "cache",
        "caches",
        "logs",
        "checkpoints",
        "weights",
        "models",
        "datasets",
        "data",
    }
)

_SECRET_NAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        "credentials.json",
        "secrets.json",
        "id_rsa",
        "id_ed25519",
    }
)

_SOURCE_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".kt",
        ".kts",
        ".go",
        ".rs",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".cs",
        ".php",
        ".rb",
        ".swift",
        ".scala",
        ".sql",
        ".sh",
        ".ps1",
        ".bat",
        ".html",
        ".css",
        ".scss",
        ".vue",
        ".svelte",
        ".ipynb",
    }
)

_DOCUMENT_SUFFIXES = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".csv",
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }
)
_CONFIG_SUFFIXES = frozenset({".toml", ".yaml", ".yml", ".json", ".ini", ".cfg"})
_CONFIG_NAMES = frozenset(
    {
        "dockerfile",
        "makefile",
        "requirements.txt",
        "pyproject.toml",
        "package.json",
        "cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "settings.gradle",
    }
)
_GENERATED_NAMES = frozenset(
    {
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "poetry.lock",
        "uv.lock",
        "cargo.lock",
    }
)
_MAX_TEXT_FILE_BYTES = 1_500_000
_MAX_DOCUMENT_BYTES = 20_000_000
_MAX_ANALYSIS_FILES = 5000


class ProjectSourceType(str, Enum):
    LOCAL_PATH = "local_path"
    ZIP = "zip"
    REPOSITORY = "repository"
    DOCUMENT_BUNDLE = "document_bundle"


class ProjectSourceStatus(str, Enum):
    REGISTERED = "registered"
    ANALYZED = "analyzed"
    NEEDS_REANALYSIS = "needs_reanalysis"
    UNAVAILABLE = "unavailable"
    INVALIDATED = "invalidated"


class ProjectArtifactKind(str, Enum):
    SOURCE = "source"
    DOCUMENT = "document"
    CONFIG = "config"


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _normalize_project_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", _required_text(value, "project_id")).strip("-.")
    if not normalized:
        raise ValueError("project_id must contain a letter or number")
    return normalized[:100]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def classify_project_artifact(path: Path, *, size: int | None = None) -> ProjectArtifactKind | None:
    """Return a readable artifact kind; runtime assets are deliberately absent."""

    name = path.name.casefold()
    suffix = path.suffix.casefold()
    if name in _SECRET_NAMES or name in _GENERATED_NAMES:
        return None
    if any(part.casefold() in _SKIP_DIRECTORIES for part in path.parts[:-1]):
        return None
    try:
        file_size = path.stat().st_size if size is None else int(size)
    except OSError:
        return None
    if suffix in _SOURCE_SUFFIXES and file_size <= _MAX_TEXT_FILE_BYTES:
        return ProjectArtifactKind.SOURCE
    if (suffix in _DOCUMENT_SUFFIXES or name.startswith("readme")) and file_size <= _MAX_DOCUMENT_BYTES:
        return ProjectArtifactKind.DOCUMENT
    if (suffix in _CONFIG_SUFFIXES or name in _CONFIG_NAMES) and file_size <= _MAX_TEXT_FILE_BYTES:
        return ProjectArtifactKind.CONFIG
    return None


def is_ignored_project_path(path: str | Path) -> bool:
    """Return whether a relative project path belongs to runtime or dependency data."""

    return any(part.casefold() in _SKIP_DIRECTORIES for part in Path(path).parts)


@dataclass(frozen=True, slots=True)
class ProjectArtifact:
    path: str
    kind: ProjectArtifactKind
    size: int
    content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _required_text(self.path, "path").replace("\\", "/"))
        object.__setattr__(self, "kind", ProjectArtifactKind(self.kind))
        if self.size < 0:
            raise ValueError("ProjectArtifact size cannot be negative")
        if len(self.content_hash) != 64:
            raise ValueError("ProjectArtifact content_hash must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind.value,
            "size": self.size,
            "content_hash": self.content_hash,
        }


def scan_project_artifacts(source_root: str | Path) -> tuple[ProjectArtifact, ...]:
    root = Path(source_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project source does not exist: {root}")
    artifacts: list[ProjectArtifact] = []
    if root.is_file():
        paths = iter(((root, Path(root.name)),))
    else:
        def iter_paths():
            for current, directories, filenames in os.walk(root):
                current_path = Path(current)
                relative_dir = current_path.relative_to(root)
                directories[:] = [
                    name for name in directories
                    if not is_ignored_project_path(relative_dir / name)
                ]
                for name in filenames:
                    yield current_path / name, relative_dir / name

        paths = iter_paths()
    for path, relative in paths:
        if is_ignored_project_path(relative):
            continue
        try:
            size = path.stat().st_size
            kind = classify_project_artifact(relative, size=size)
            if kind is None:
                continue
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        artifacts.append(ProjectArtifact(path=relative.as_posix(), kind=kind, size=size, content_hash=content_hash))
        if len(artifacts) >= _MAX_ANALYSIS_FILES:
            raise ValueError(f"Project has more than {_MAX_ANALYSIS_FILES} readable artifacts")
    return tuple(sorted(artifacts, key=lambda item: item.path.casefold()))


def _office_xml_text(path: Path, *, prefix: str, label: str) -> str:
    """Extract ordered visible text from PPTX/XLSX without loading executable content."""

    with zipfile.ZipFile(path) as archive:
        names = sorted(
            (
                name
                for name in archive.namelist()
                if name.startswith(prefix) and name.endswith(".xml")
            ),
            key=lambda value: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)],
        )
        blocks: list[str] = []
        for index, name in enumerate(names[:500], start=1):
            root = ElementTree.fromstring(archive.read(name))
            values = [str(node.text or "").strip() for node in root.iter() if node.tag.endswith("}t")]
            text = "\n".join(value for value in values if value)
            if text:
                blocks.append(f"# {label} {index}\n{text}")
        return "\n\n".join(blocks)


def _xlsx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root:
                shared.append("".join(str(node.text or "") for node in item.iter() if node.tag.endswith("}t")))
        sheet_names = sorted(
            (name for name in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)),
            key=lambda value: int(re.search(r"(\d+)", Path(value).stem).group(1)),
        )
        blocks: list[str] = []
        for index, name in enumerate(sheet_names[:200], start=1):
            root = ElementTree.fromstring(archive.read(name))
            rows: list[str] = []
            for row in (node for node in root.iter() if node.tag.endswith("}row")):
                values: list[str] = []
                for cell in (node for node in row if node.tag.endswith("}c")):
                    raw = next((str(node.text or "") for node in cell.iter() if node.tag.endswith("}v")), "")
                    inline = "".join(str(node.text or "") for node in cell.iter() if node.tag.endswith("}t"))
                    value = inline or raw
                    if cell.attrib.get("t") == "s" and raw.isdigit() and int(raw) < len(shared):
                        value = shared[int(raw)]
                    values.append(value)
                if any(values):
                    rows.append("\t".join(values))
            if rows:
                blocks.append(f"# Sheet {index}\n" + "\n".join(rows))
        return "\n\n".join(blocks)


def extract_project_document_text(path: str | Path) -> tuple[str, str]:
    """Read a non-code project artifact as text for the read-only Project Analyst."""

    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise ValueError(f"Project document does not exist: {target}")
    if classify_project_artifact(Path(target.name), size=target.stat().st_size) is not ProjectArtifactKind.DOCUMENT:
        raise ValueError(f"Unsupported project document type: {target.suffix or '<none>'}")
    suffix = target.suffix.casefold()
    if suffix in {".md", ".markdown", ".txt", ".rst", ".csv"} or target.name.casefold().startswith("readme"):
        return target.read_text(encoding="utf-8", errors="replace"), "plain_text"
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("pypdf is required to read PDF project documents") from exc
        reader = PdfReader(str(target))
        return "\n\n".join(
            f"# Page {index}\n{page.extract_text() or ''}"
            for index, page in enumerate(reader.pages, start=1)
        ), "pypdf"
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("python-docx is required to read DOCX project documents") from exc
        document = Document(str(target))
        return "\n".join(paragraph.text for paragraph in document.paragraphs), "python-docx"
    if suffix == ".pptx":
        return _office_xml_text(target, prefix="ppt/slides/slide", label="Slide"), "office-open-xml"
    if suffix == ".xlsx":
        return _xlsx_text(target), "office-open-xml"
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        try:
            from smart_agent.career.ocr import OCRService, PytesseractOCRAdapter

            document = OCRService(PytesseractOCRAdapter()).extract(str(target))
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("Pillow and pytesseract are required to read image project documents") from exc
        return "\n".join(region.text for region in document.regions if region.text.strip()), "pytesseract"
    raise ValueError(f"Unsupported project document type: {suffix or '<none>'}")


def project_version(artifacts: tuple[ProjectArtifact, ...] | list[ProjectArtifact]) -> str:
    payload = [(item.path, item.kind.value, item.size, item.content_hash) for item in artifacts]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ProjectSource:
    project_id: str
    display_name: str
    source_type: ProjectSourceType
    source_location: str
    project_version: str
    artifact_count: int
    status: ProjectSourceStatus = ProjectSourceStatus.REGISTERED
    managed_copy: bool = False
    last_analysis_run_id: str = ""
    revision: int = 1
    schema_version: str = PROJECT_LIBRARY_SCHEMA_VERSION
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", _normalize_project_id(self.project_id))
        object.__setattr__(self, "display_name", _required_text(self.display_name, "display_name"))
        object.__setattr__(self, "source_type", ProjectSourceType(self.source_type))
        object.__setattr__(self, "source_location", _required_text(self.source_location, "source_location"))
        object.__setattr__(self, "status", ProjectSourceStatus(self.status))
        object.__setattr__(self, "last_analysis_run_id", str(self.last_analysis_run_id or "").strip())
        if len(self.project_version) != 64:
            raise ValueError("project_version must be SHA-256")
        if self.artifact_count < 0 or self.revision < 1:
            raise ValueError("Invalid ProjectSource counters")
        if self.schema_version != PROJECT_LIBRARY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported project-library schema: {self.schema_version}")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_type"] = self.source_type.value
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectSource":
        return cls(**value)


class ProjectLibraryStore:
    """Atomic registry of project locations; project bytes remain outside this store."""

    _INDEX_FILE = "sources.json"
    _ROOT_LOCKS: dict[str, threading.RLock] = {}
    _ROOT_LOCKS_GUARD = threading.Lock()

    def __init__(self, root: str | Path, *, workspace: str | Path, allow_external: bool = False) -> None:
        self.root = Path(root).expanduser().resolve()
        self.workspace = Path(workspace).expanduser().resolve()
        self.allow_external = bool(allow_external)
        self.index_path = self.root / self._INDEX_FILE
        root_key = str(self.root).casefold()
        with self._ROOT_LOCKS_GUARD:
            self._lock = self._ROOT_LOCKS.setdefault(root_key, threading.RLock())
        self.root.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, ProjectSource]:
        if not self.index_path.exists():
            return {}
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != PROJECT_LIBRARY_SCHEMA_VERSION:
                raise ValueError("Unsupported project-library index version")
            return {
                item["project_id"]: ProjectSource.from_dict(item)
                for item in payload.get("sources", [])
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Project library is corrupted: {exc}") from exc

    def _write(self, records: dict[str, ProjectSource]) -> None:
        payload = {
            "schema_version": PROJECT_LIBRARY_SCHEMA_VERSION,
            "sources": [records[key].to_dict() for key in sorted(records)],
        }
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="sources-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.index_path)
        finally:
            try:
                Path(temporary).unlink()
            except OSError:
                pass

    @staticmethod
    def _record_key(records: dict[str, ProjectSource], project_id: str) -> str:
        """Resolve project IDs case-insensitively for legacy Career records."""
        selected = _normalize_project_id(project_id)
        if selected in records:
            return selected
        folded = selected.casefold()
        return next(
            (key for key in records if str(key).casefold() == folded),
            selected,
        )

    def _resolve_source(self, source_location: str | Path) -> Path:
        path = Path(source_location).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Project source does not exist: {path}")
        if not self.allow_external and not (
            _is_relative_to(path, self.workspace) or _is_relative_to(path, self.root)
        ):
            raise ValueError("Project source must be inside the workspace or managed project library")
        return path

    def register(
        self,
        *,
        project_id: str,
        display_name: str,
        source_type: ProjectSourceType | str,
        source_location: str | Path,
        managed_copy: bool = False,
    ) -> ProjectSource:
        selected_id = _normalize_project_id(project_id)
        path = self._resolve_source(source_location)
        artifacts = scan_project_artifacts(path)
        if not artifacts:
            raise ValueError("Project contains no readable source, document, or configuration artifacts")
        version = project_version(artifacts)
        now = time.time()
        with self._lock:
            records = self._load()
            current = records.get(selected_id)
            if current is not None and current.project_version == version and current.source_location == str(path):
                if current.status not in {ProjectSourceStatus.INVALIDATED, ProjectSourceStatus.UNAVAILABLE}:
                    return current
                record = replace(
                    current,
                    display_name=display_name,
                    source_type=ProjectSourceType(source_type),
                    status=ProjectSourceStatus.NEEDS_REANALYSIS,
                    managed_copy=managed_copy,
                    revision=current.revision + 1,
                    updated_at=now,
                )
                records[selected_id] = record
                self._write(records)
                return record
            record = ProjectSource(
                project_id=selected_id,
                display_name=display_name,
                source_type=source_type,
                source_location=str(path),
                project_version=version,
                artifact_count=len(artifacts),
                status=(ProjectSourceStatus.NEEDS_REANALYSIS if current else ProjectSourceStatus.REGISTERED),
                managed_copy=managed_copy,
                revision=current.revision + 1 if current else 1,
                created_at=current.created_at if current else now,
                updated_at=now,
            )
            records[selected_id] = record
            self._write(records)
            return record

    def get(self, project_id: str, *, refresh: bool = False) -> ProjectSource:
        with self._lock:
            records = self._load()
            selected_id = self._record_key(records, project_id)
            try:
                current = records[selected_id]
            except KeyError as exc:
                raise KeyError(f"Unknown project: {selected_id}") from exc
            if not refresh or current.status is ProjectSourceStatus.INVALIDATED:
                return current
            path = Path(current.source_location)
            if not path.exists():
                updated = replace(
                    current,
                    status=ProjectSourceStatus.UNAVAILABLE,
                    revision=current.revision + 1,
                    updated_at=time.time(),
                )
            else:
                artifacts = scan_project_artifacts(path)
                version = project_version(artifacts)
                if version == current.project_version and current.status is not ProjectSourceStatus.UNAVAILABLE:
                    return current
                updated = replace(
                    current,
                    project_version=version,
                    artifact_count=len(artifacts),
                    status=ProjectSourceStatus.NEEDS_REANALYSIS,
                    revision=current.revision + 1,
                    updated_at=time.time(),
                )
            records[selected_id] = updated
            self._write(records)
            return updated

    def list(self, *, include_invalidated: bool = False) -> tuple[ProjectSource, ...]:
        with self._lock:
            records = self._load().values()
            return tuple(
                sorted(
                    (
                        item
                        for item in records
                        if include_invalidated or item.status is not ProjectSourceStatus.INVALIDATED
                    ),
                    key=lambda item: item.display_name.casefold(),
                )
            )

    def mark_analyzed(
        self,
        project_id: str,
        analysis_run_id: str,
        *,
        expected_project_version: str,
    ) -> ProjectSource:
        selected_id = _normalize_project_id(project_id)
        with self._lock:
            records = self._load()
            current = records.get(selected_id)
            if current is None:
                raise KeyError(f"Unknown project: {selected_id}")
            if current.status in {ProjectSourceStatus.INVALIDATED, ProjectSourceStatus.UNAVAILABLE}:
                raise ValueError(f"Cannot mark project analyzed from state: {current.status.value}")
            if current.project_version != str(expected_project_version or "").strip():
                raise ValueError("Project version changed before analysis completion")
            updated = replace(
                current,
                status=ProjectSourceStatus.ANALYZED,
                last_analysis_run_id=_required_text(analysis_run_id, "analysis_run_id"),
                revision=current.revision + 1,
                updated_at=time.time(),
            )
            records[selected_id] = updated
            self._write(records)
            return updated

    def invalidate(self, project_id: str) -> ProjectSource:
        with self._lock:
            records = self._load()
            selected_id = self._record_key(records, project_id)
            current = records.get(selected_id)
            if current is None:
                raise KeyError(f"Unknown project: {selected_id}")
            if current.status is ProjectSourceStatus.INVALIDATED:
                return current
            updated = replace(
                current,
                status=ProjectSourceStatus.INVALIDATED,
                revision=current.revision + 1,
                updated_at=time.time(),
            )
            records[selected_id] = updated
            self._write(records)
            return updated

    def update(
        self,
        project_id: str,
        *,
        display_name: str | None = None,
        source_location: str | Path | None = None,
        expected_revision: int | None = None,
    ) -> ProjectSource:
        """Update project metadata or replace a local source path."""
        with self._lock:
            records = self._load()
            selected_id = self._record_key(records, project_id)
            current = records.get(selected_id)
            if current is None:
                raise KeyError(f"Unknown project: {selected_id}")
            if expected_revision is not None and current.revision != int(expected_revision):
                raise ValueError(
                    f"Expected {selected_id}@{expected_revision}, found revision {current.revision}."
                )
            next_name = str(display_name if display_name is not None else current.display_name).strip()
            if not next_name:
                raise ValueError("display_name is required")
            next_path = (
                self._resolve_source(source_location)
                if source_location is not None
                else Path(current.source_location)
            )
            if source_location is not None and current.managed_copy:
                raise ValueError("Managed ZIP projects cannot replace their source path; import a new archive")
            artifacts = scan_project_artifacts(next_path)
            if not artifacts:
                raise ValueError("Project contains no readable source, document, or configuration artifacts")
            next_version = project_version(artifacts)
            changed = next_version != current.project_version or str(next_path) != current.source_location
            updated = replace(
                current,
                display_name=next_name,
                source_location=str(next_path),
                project_version=next_version,
                artifact_count=len(artifacts),
                status=ProjectSourceStatus.NEEDS_REANALYSIS if changed else current.status,
                revision=current.revision + 1,
                updated_at=time.time(),
            )
            records[selected_id] = updated
            self._write(records)
            return updated

    def delete(self, project_id: str, *, expected_revision: int | None = None) -> ProjectSource:
        """Remove a project from the library index and delete only managed ZIP bytes."""
        with self._lock:
            records = self._load()
            selected_id = self._record_key(records, project_id)
            current = records.get(selected_id)
            if current is None:
                raise KeyError(f"Unknown project: {selected_id}")
            if expected_revision is not None and current.revision != int(expected_revision):
                raise ValueError(
                    f"Expected {selected_id}@{expected_revision}, found revision {current.revision}."
                )
            updated = replace(
                current,
                status=ProjectSourceStatus.INVALIDATED,
                revision=current.revision + 1,
                updated_at=time.time(),
            )
            # Deletion is user-facing removal, not an inactive library entry.
            # Related Career evidence is invalidated by the API layer before it
            # can be used again, while the project itself disappears from every
            # project-library listing and lookup.
            del records[selected_id]
            self._write(records)
            source_path = Path(current.source_location).resolve()
            managed_root = self.root.resolve()
            if current.managed_copy and source_path.is_relative_to(managed_root) and source_path != managed_root:
                if source_path.is_dir():
                    shutil.rmtree(source_path)
                elif source_path.exists():
                    source_path.unlink()
            return updated


def _unique_strings(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be an array")
    normalized = tuple(str(item or "").strip() for item in value if str(item or "").strip())
    return tuple(dict.fromkeys(normalized))


@dataclass(frozen=True, slots=True)
class ProjectUnderstanding:
    title: str
    goal: str
    core_flow: tuple[str, ...]
    core_modules: tuple[str, ...]
    architecture: str
    technology_choices: tuple[str, ...]
    technologies: tuple[str, ...]
    implemented_features: tuple[str, ...]
    candidate_outcomes: tuple[str, ...]
    candidate_responsibilities: tuple[str, ...]
    technical_tags: tuple[str, ...]
    unknowns: tuple[str, ...]
    source_refs: tuple[dict[str, Any], ...]
    confidence_by_field: dict[str, float]
    analysis_summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _required_text(self.title, "title"))
        object.__setattr__(self, "goal", str(self.goal or "").strip())
        object.__setattr__(self, "architecture", str(self.architecture or "").strip())
        object.__setattr__(self, "analysis_summary", _required_text(self.analysis_summary, "analysis_summary"))
        for name in (
            "core_flow",
            "core_modules",
            "technology_choices",
            "technologies",
            "implemented_features",
            "candidate_outcomes",
            "candidate_responsibilities",
            "technical_tags",
            "unknowns",
        ):
            object.__setattr__(self, name, _unique_strings(getattr(self, name), name))
        refs = tuple(dict(item) for item in self.source_refs if isinstance(item, dict))
        for item in refs:
            if not str(item.get("field") or "").strip():
                raise ValueError("source_refs.field is required")
            if not str(item.get("path") or "").strip():
                raise ValueError("source_refs.path is required")
            try:
                line = int(item.get("line") or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError("source_refs.line must be a positive integer") from exc
            if line < 1:
                raise ValueError("source_refs.line must be a positive integer")
            if not str(item.get("excerpt") or "").strip():
                raise ValueError("source_refs.excerpt is required")
        object.__setattr__(self, "source_refs", refs)
        confidence = {str(key): float(value) for key, value in dict(self.confidence_by_field).items()}
        if any(not 0 <= value <= 1 for value in confidence.values()):
            raise ValueError("confidence_by_field values must be between 0 and 1")
        object.__setattr__(self, "confidence_by_field", confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "goal": self.goal,
            "core_flow": list(self.core_flow),
            "core_modules": list(self.core_modules),
            "architecture": self.architecture,
            "technology_choices": list(self.technology_choices),
            "technologies": list(self.technologies),
            "implemented_features": list(self.implemented_features),
            "candidate_outcomes": list(self.candidate_outcomes),
            "candidate_responsibilities": list(self.candidate_responsibilities),
            "technical_tags": list(self.technical_tags),
            "unknowns": list(self.unknowns),
            "source_refs": [dict(item) for item in self.source_refs],
            "confidence_by_field": dict(self.confidence_by_field),
            "analysis_summary": self.analysis_summary,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectUnderstanding":
        if not isinstance(value, dict):
            raise ValueError("Project understanding must be an object")
        return cls(
            title=value.get("title", ""),
            goal=value.get("goal", ""),
            core_flow=value.get("core_flow", ()),
            core_modules=value.get("core_modules", ()),
            architecture=value.get("architecture", ""),
            technology_choices=value.get("technology_choices", ()),
            technologies=value.get("technologies", ()),
            implemented_features=value.get("implemented_features", ()),
            candidate_outcomes=value.get("candidate_outcomes", ()),
            candidate_responsibilities=value.get("candidate_responsibilities", ()),
            technical_tags=value.get("technical_tags", ()),
            unknowns=value.get("unknowns", ()),
            source_refs=tuple(value.get("source_refs", ())),
            confidence_by_field=dict(value.get("confidence_by_field", {})),
            analysis_summary=value.get("analysis_summary", ""),
        )


class ProjectAnalyzer(Protocol):
    name: str

    def analyze(self, source: ProjectSource) -> ProjectUnderstanding: ...


@dataclass(frozen=True, slots=True)
class ProjectAnalysisRecord:
    analysis_run_id: str
    project_id: str
    project_version: str
    intent: str
    generator: str
    model: str
    understanding: ProjectUnderstanding
    created_at: float = field(default_factory=time.time)
    schema_version: str = PROJECT_LIBRARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_run_id", _required_text(self.analysis_run_id, "analysis_run_id"))
        object.__setattr__(self, "project_id", _normalize_project_id(self.project_id))
        object.__setattr__(self, "intent", _required_text(self.intent, "intent"))
        object.__setattr__(self, "generator", _required_text(self.generator, "generator"))
        object.__setattr__(self, "model", str(self.model or "").strip())
        if len(self.project_version) != 64:
            raise ValueError("project_version must be SHA-256")
        if self.schema_version != PROJECT_LIBRARY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported project-analysis schema: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_run_id": self.analysis_run_id,
            "project_id": self.project_id,
            "project_version": self.project_version,
            "intent": self.intent,
            "generator": self.generator,
            "model": self.model,
            "understanding": self.understanding.to_dict(),
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectAnalysisRecord":
        return cls(
            analysis_run_id=value["analysis_run_id"],
            project_id=value["project_id"],
            project_version=value["project_version"],
            intent=value["intent"],
            generator=value["generator"],
            model=value.get("model", ""),
            understanding=ProjectUnderstanding.from_dict(value["understanding"]),
            created_at=float(value.get("created_at", time.time())),
            schema_version=value.get("schema_version", PROJECT_LIBRARY_SCHEMA_VERSION),
        )


class ProjectAnalysisStore:
    """Versioned structured analysis cache; it never stores raw project chunks."""

    _ROOT_LOCKS: dict[str, threading.RLock] = {}
    _ROOT_LOCKS_GUARD = threading.Lock()

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self._ROOT_LOCKS_GUARD:
            self._lock = self._ROOT_LOCKS.setdefault(str(self.root).casefold(), threading.RLock())

    @staticmethod
    def cache_key(project_version: str, intent: str) -> str:
        return hashlib.sha256(f"{project_version}:{str(intent).strip().casefold()}".encode("utf-8")).hexdigest()

    def _path(self, record: ProjectAnalysisRecord) -> Path:
        key = self.cache_key(record.project_version, record.intent)
        run_key = re.sub(r"[^a-zA-Z0-9._-]+", "-", record.analysis_run_id).strip("-.")
        return self.root / record.project_id / f"{key}-{run_key}.json"

    def put(self, record: ProjectAnalysisRecord) -> ProjectAnalysisRecord:
        path = self._path(record)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                existing = ProjectAnalysisRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
                if existing.to_dict() != record.to_dict():
                    raise ValueError("analysis_run_id already contains a different result")
                return existing
            fd, temporary = tempfile.mkstemp(prefix="analysis-", suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(record.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                try:
                    Path(temporary).unlink()
                except OSError:
                    pass
        return record

    def get(self, project_id: str, project_version: str, intent: str = "overview") -> ProjectAnalysisRecord:
        selected_id = _normalize_project_id(project_id)
        key = self.cache_key(project_version, intent)
        with self._lock:
            paths = tuple((self.root / selected_id).glob(f"{key}-*.json"))
            records = [
                ProjectAnalysisRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
                for path in paths
            ]
        if not paths:
            raise KeyError(f"No cached project analysis for {selected_id}: {intent}")
        return max(records, key=lambda item: item.created_at)

    def latest(self, project_id: str, *, intent: str = "overview") -> ProjectAnalysisRecord | None:
        selected_id = _normalize_project_id(project_id)
        directory = self.root / selected_id
        with self._lock:
            if not directory.exists():
                return None
            records: list[ProjectAnalysisRecord] = []
            for path in directory.glob("*.json"):
                try:
                    record = ProjectAnalysisRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if record.intent == intent:
                    records.append(record)
        return max(records, key=lambda item: item.created_at, default=None)

    def purge(self, project_id: str) -> int:
        """Delete cached analyses for a project after the project is removed."""
        selected_id = _normalize_project_id(project_id)
        directory = self.root / selected_id
        if not directory.exists():
            return 0
        removed = 0
        for path in directory.glob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        try:
            directory.rmdir()
        except OSError:
            pass
        return removed


class HeuristicProjectAnalyzer:
    """Offline smoke-test baseline; production analysis is delegated to Project Analyst."""

    name = "heuristic-project-analyzer-v1"

    def analyze(self, source: ProjectSource) -> ProjectUnderstanding:
        root = Path(source.source_location)
        artifacts = scan_project_artifacts(root)
        readable = [item for item in artifacts if item.kind in {ProjectArtifactKind.SOURCE, ProjectArtifactKind.CONFIG}]
        documents = [item for item in artifacts if item.kind is ProjectArtifactKind.DOCUMENT]
        title = source.display_name
        goal = ""
        refs: list[dict[str, Any]] = []
        readme = next((item for item in documents if Path(item.path).name.casefold().startswith("readme")), None)
        if readme is not None:
            readme_path = root / readme.path if root.is_dir() else root
            text = readme_path.read_text(encoding="utf-8", errors="ignore")
            heading = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")), "")
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
            if heading:
                title = heading
            goal = next((part for part in paragraphs if not part.startswith("#")), "")[:800]
            excerpt = goal[:300] or heading or source.display_name
            refs.append({"field": "goal", "path": readme.path, "line": 1, "excerpt": excerpt})
        modules = tuple(
            item.path
            for item in readable
            if len(Path(item.path).parts) <= 3 and item.kind is ProjectArtifactKind.SOURCE
        )[:20]
        technologies = self._technologies(root, readable)
        technology_choices = tuple(
            f"{technology} is declared by the project; its exact design purpose still requires Project Analyst review"
            for technology in technologies[:8]
        )
        deliverables = (
            (f"Project documentation describes the delivered objective: {goal[:300]}",)
            if goal
            else tuple(f"Readable project module: {item}" for item in modules[:5])
        )
        unknowns = (
            "个人负责部分无法仅根据项目文件确认",
            "历史成果、奖项和指标需要用户确认",
        )
        return ProjectUnderstanding(
            title=title,
            goal=goal,
            core_flow=("阅读项目入口、核心模块与配置，形成待 Project Analyst 深挖的项目地图",) if modules else (),
            core_modules=modules,
            architecture="",
            technology_choices=technology_choices,
            technologies=technologies,
            implemented_features=deliverables,
            candidate_outcomes=(),
            candidate_responsibilities=(),
            technical_tags=technologies,
            unknowns=unknowns,
            source_refs=tuple(refs),
            confidence_by_field={"title": 0.8, "goal": 0.55 if goal else 0.0, "technologies": 0.65},
            analysis_summary="离线基线只完成项目地图和技术栈提取，需要 Project Analyst 深入阅读核心流程。",
        )

    @staticmethod
    def _technologies(root: Path, artifacts: list[ProjectArtifact]) -> tuple[str, ...]:
        values: list[str] = []
        manifests = {
            "requirements.txt",
            "pyproject.toml",
            "package.json",
            "cargo.toml",
            "go.mod",
            "pom.xml",
        }
        for artifact in artifacts:
            if Path(artifact.path).name.casefold() not in manifests:
                continue
            text = (root / artifact.path).read_text(encoding="utf-8", errors="ignore")[:100_000]
            if Path(artifact.path).name.casefold() == "requirements.txt":
                values.extend(
                    re.split(r"[<>=!~\[]", line.strip(), maxsplit=1)[0]
                    for line in text.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                )
            else:
                values.extend(re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}", text))
        ignored = {"dependencies", "dependency", "version", "python", "project", "name", "true", "false"}
        return tuple(dict.fromkeys(item for item in values if item.casefold() not in ignored))[:30]
