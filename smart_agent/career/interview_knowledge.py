"""Traceable interview knowledge ingestion and pluggable hybrid retrieval."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import subprocess
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from smart_agent.career.interview import (
    InterviewKnowledgeChunk,
    InterviewKnowledgeCorpus,
    InterviewKnowledgeStore,
    InterviewKnowledgeType,
)
from smart_agent.career.models import JDDocument, JDRequirement

KNOWLEDGE_INDEX_SCHEMA_VERSION = "1.1"
KNOWLEDGE_CHUNKER_VERSION = "4"
MIN_KNOWLEDGE_CHUNK_CHARS = 20
_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
_DEFAULT_OCR_MAX_PIXELS = 100_000_000
_INJECTION_PATTERNS = (
    re.compile(r"(?i)ignore\s+(all\s+)?(?:previous|prior)?\s*(?:system\s+)?instructions?"),
    re.compile(r"(?i)(system\s+prompt|developer\s+message|tool\s+permission)"),
    re.compile(r"\u5ffd\u7565.{0,8}(\u4e4b\u524d|\u4ee5\u4e0a|\u7cfb\u7edf).{0,8}(\u6307\u4ee4|\u63d0\u793a\u8bcd)"),
    re.compile(r"(\u8c03\u7528|\u4f7f\u7528).{0,8}(\u5de5\u5177|shell|\u7ec8\u7aef).{0,8}(\u5220\u9664|\u6cc4\u9732|\u8bfb\u53d6)"),
)
_QA_QUESTION_PATTERN = re.compile(
    r"(?i)(?P<label>\u95ee\u9898|\u9898\u76ee|Question|Q)\s*"
    r"(?P<number>\d{1,4}|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e]+)\s*[:\uff1a\u3001.]"
)
_QA_ANSWER_PATTERN = re.compile(
    r"(?i)(?:\u56de\u7b54|\u7b54\u6848|\u89e3\u7b54|Answer|A)\s*[:\uff1a]"
)


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _stable_values(values: tuple[str, ...] | list[str] | None, *, default: str = "") -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(item).strip() for item in (values or ()) if str(item).strip()))
    if normalized:
        return normalized
    return (default,) if default else ()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "-", value.strip()).strip("-._")
    return normalized[:80] or "knowledge"


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
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


def tokenize_knowledge(text: str) -> tuple[str, ...]:
    lowered = str(text or "").casefold()
    pieces = re.findall(r"[a-z][a-z0-9+#.\-]*|\d+(?:\.\d+)?", lowered)
    for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
        pieces.extend(run)
        pieces.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tuple(pieces)


def detect_prompt_injection(text: str) -> tuple[str, ...]:
    return tuple(f"prompt_injection_pattern_{index + 1}" for index, pattern in enumerate(_INJECTION_PATTERNS) if pattern.search(text))


class InterviewKnowledgeSourceType(str, Enum):
    OFFICIAL_DOCUMENTATION = "official_documentation"
    TEXTBOOK = "textbook"
    COURSE_NOTE = "course_note"
    INTERVIEW_NOTE = "interview_note"
    USER_NOTE = "user_note"


@dataclass(frozen=True, slots=True)
class InterviewKnowledgeSource:
    source_id: str
    corpus_id: str
    tenant_id: str
    name: str
    source_type: InterviewKnowledgeSourceType
    authority_level: int
    domains: tuple[str, ...]
    subdomains: tuple[str, ...]
    topics: tuple[str, ...]
    difficulty: int
    content_hash: str
    media_type: str
    parser: str
    parser_version: str
    pdf_mode: str = "normal"
    chunker_version: str = "1"
    source_path: str = ""
    chapter: str = ""
    page_start: int | None = None
    page_end: int | None = None
    security_flags: tuple[str, ...] = ()
    imported_at: float = field(default_factory=time.time)
    chunk_count: int = 0
    index_status: str = "pending"
    index_version: str = ""
    indexed_at: float | None = None

    def __post_init__(self) -> None:
        for name in ("source_id", "corpus_id", "tenant_id", "name", "content_hash", "media_type", "parser"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "source_type", InterviewKnowledgeSourceType(self.source_type))
        normalized_pdf_mode = str(self.pdf_mode or "normal").strip().casefold()
        if normalized_pdf_mode not in {"normal", "qa"}:
            raise ValueError("pdf_mode must be normal or qa")
        object.__setattr__(self, "pdf_mode", normalized_pdf_mode)
        object.__setattr__(self, "chunker_version", _required_text(self.chunker_version, "chunker_version"))
        object.__setattr__(self, "domains", _stable_values(self.domains, default="general"))
        object.__setattr__(self, "subdomains", _stable_values(self.subdomains))
        object.__setattr__(self, "topics", _stable_values(self.topics, default="general"))
        object.__setattr__(self, "security_flags", _stable_values(self.security_flags))
        object.__setattr__(self, "source_path", str(self.source_path or "").strip())
        object.__setattr__(self, "chapter", str(self.chapter or "").strip())
        if not 1 <= int(self.authority_level) <= 5:
            raise ValueError("authority_level must be between 1 and 5")
        if not 1 <= int(self.difficulty) <= 5:
            raise ValueError("difficulty must be between 1 and 5")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_hash.casefold()):
            raise ValueError("content_hash must be a SHA-256 digest")
        if self.page_start is not None and self.page_start < 1:
            raise ValueError("page_start must be positive")
        if self.page_end is not None and (self.page_start is None or self.page_end < self.page_start):
            raise ValueError("page_end cannot precede page_start")
        if int(self.chunk_count) < 0:
            raise ValueError("chunk_count cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "corpus_id": self.corpus_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "source_type": self.source_type.value,
            "authority_level": self.authority_level,
            "domains": list(self.domains),
            "subdomains": list(self.subdomains),
            "topics": list(self.topics),
            "difficulty": self.difficulty,
            "content_hash": self.content_hash,
            "media_type": self.media_type,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "pdf_mode": self.pdf_mode,
            "chunker_version": self.chunker_version,
            "source_path": self.source_path,
            "chapter": self.chapter,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "security_flags": list(self.security_flags),
            "imported_at": self.imported_at,
            "chunk_count": self.chunk_count,
            "index_status": self.index_status,
            "index_version": self.index_version,
            "indexed_at": self.indexed_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InterviewKnowledgeSource":
        return cls(
            source_id=str(value["source_id"]),
            corpus_id=str(value["corpus_id"]),
            tenant_id=str(value["tenant_id"]),
            name=str(value["name"]),
            source_type=InterviewKnowledgeSourceType(value["source_type"]),
            authority_level=int(value["authority_level"]),
            domains=tuple(value.get("domains") or ()),
            subdomains=tuple(value.get("subdomains") or ()),
            topics=tuple(value.get("topics") or ()),
            difficulty=int(value["difficulty"]),
            content_hash=str(value["content_hash"]),
            media_type=str(value["media_type"]),
            parser=str(value["parser"]),
            parser_version=str(value.get("parser_version") or "1"),
            pdf_mode=str(value.get("pdf_mode") or "normal"),
            chunker_version=str(value.get("chunker_version") or "1"),
            source_path=str(value.get("source_path") or ""),
            chapter=str(value.get("chapter") or ""),
            page_start=value.get("page_start"),
            page_end=value.get("page_end"),
            security_flags=tuple(value.get("security_flags") or ()),
            imported_at=float(value.get("imported_at") or time.time()),
            chunk_count=int(value.get("chunk_count") or 0),
            index_status=str(value.get("index_status") or "pending"),
            index_version=str(value.get("index_version") or ""),
            indexed_at=float(value["indexed_at"]) if value.get("indexed_at") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class KnowledgeImportResult:
    source: InterviewKnowledgeSource
    corpus: InterviewKnowledgeCorpus
    deduplicated: bool
    parsed_blocks: int


class InterviewKnowledgeSourceChangedError(ValueError):
    pass


class InterviewKnowledgeLibrary:
    """Own source metadata and the existing isolated corpus store."""

    def __init__(self, root: str | Path, *, max_chunk_chars: int = 1200) -> None:
        self.root = Path(root).resolve()
        self.sources_dir = self.root / "sources"
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_store = InterviewKnowledgeStore(self.root)
        self.max_chunk_chars = int(max_chunk_chars)
        if self.max_chunk_chars < 200:
            raise ValueError("max_chunk_chars must be at least 200")

    def import_source(
        self,
        *,
        tenant_id: str,
        name: str,
        source_type: InterviewKnowledgeSourceType | str,
        authority_level: int,
        domains: tuple[str, ...] | list[str],
        topics: tuple[str, ...] | list[str],
        subdomains: tuple[str, ...] | list[str] = (),
        difficulty: int = 3,
        text: str | None = None,
        source_path: str | Path | None = None,
        source_name: str = "",
        corpus_id: str = "",
        pdf_mode: str = "normal",
    ) -> KnowledgeImportResult:
        tenant_id = _required_text(tenant_id, "tenant_id")
        name = _required_text(name, "name")
        if (text is None) == (source_path is None):
            raise ValueError("Provide exactly one of text or source_path")
        path = Path(source_path).expanduser().resolve() if source_path is not None else None
        if text is not None:
            raw = _required_text(text, "text").encode("utf-8")
        else:
            assert path is not None
            if not path.is_file():
                raise ValueError(f"Interview knowledge source is not a file: {path}")
            raw = path.read_bytes()
        digest = _hash_bytes(raw)
        normalized_pdf_mode = str(pdf_mode or "normal").strip().casefold()
        if normalized_pdf_mode not in {"normal", "qa"}:
            raise ValueError("pdf_mode must be normal or qa")
        if path is None or path.suffix.casefold() != ".pdf":
            normalized_pdf_mode = "normal"
        normalized_path = str(path) if path is not None else ""
        for existing in self.list_sources(tenant_id=tenant_id):
            if normalized_path and existing.source_path.casefold() == normalized_path.casefold() and existing.content_hash != digest:
                raise InterviewKnowledgeSourceChangedError(
                    "Interview knowledge source file changed; import it as an explicit new revision"
                )
            if (
                existing.content_hash == digest
                and existing.chunker_version == KNOWLEDGE_CHUNKER_VERSION
                and existing.pdf_mode == normalized_pdf_mode
            ):
                return KnowledgeImportResult(existing, self.corpus_store.get(existing.corpus_id), True, 0)
        raw, blocks, media_type, parser, parser_version = self._parse(
            text=text,
            path=path,
            raw=raw,
            source_name=source_name,
            pdf_mode=normalized_pdf_mode,
        )
        source_id = "knowledge-source-" + hashlib.sha256(
            f"{tenant_id}\0{digest}\0{KNOWLEDGE_CHUNKER_VERSION}\0{normalized_pdf_mode}".encode()
        ).hexdigest()[:20]
        resolved_corpus_id = corpus_id.strip() or (
            f"knowledge-{_slug(tenant_id)}-{digest[:12]}-c{KNOWLEDGE_CHUNKER_VERSION}-{normalized_pdf_mode}"
        )
        all_text = "\n".join(block[0] for block in blocks)
        flags = detect_prompt_injection(all_text)
        domains_tuple = _stable_values(domains, default="general")
        subdomains_tuple = _stable_values(subdomains)
        topics_tuple = _stable_values(topics, default="general")
        chunks = self._chunks(
            source_id=source_id,
            corpus_id=resolved_corpus_id,
            tenant_id=tenant_id,
            blocks=blocks,
            authority_level=authority_level,
            domains=domains_tuple,
            subdomains=subdomains_tuple,
            topics=topics_tuple,
            difficulty=difficulty,
            source_type=InterviewKnowledgeSourceType(source_type),
            security_flags=flags,
        )
        corpus = InterviewKnowledgeCorpus(
            corpus_id=resolved_corpus_id,
            name=name,
            domain=domains_tuple[0],
            version=digest[:16],
            chunks=chunks,
        )
        self.corpus_store.put(corpus)
        pages = [
            int(locator[key])
            for _, locator in blocks
            for key in ("page", "page_start", "page_end")
            if locator.get(key)
        ]
        chapters = [str(locator["chapter"]) for _, locator in blocks if locator.get("chapter")]
        source = InterviewKnowledgeSource(
            source_id=source_id,
            corpus_id=resolved_corpus_id,
            tenant_id=tenant_id,
            name=name,
            source_type=source_type,
            authority_level=authority_level,
            domains=domains_tuple,
            subdomains=subdomains_tuple,
            topics=topics_tuple,
            difficulty=difficulty,
            content_hash=digest,
            media_type=media_type,
            parser=parser,
            parser_version=parser_version,
            pdf_mode=normalized_pdf_mode,
            chunker_version=KNOWLEDGE_CHUNKER_VERSION,
            source_path=normalized_path,
            chapter=chapters[0] if len(set(chapters)) == 1 and chapters else "",
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            security_flags=flags,
            chunk_count=len(chunks),
        )
        _atomic_json(self.sources_dir / f"{source.source_id}.json", source.to_dict())
        return KnowledgeImportResult(source, corpus, False, len(blocks))

    def update_source_index_metadata(self, source_id: str, sync_result) -> InterviewKnowledgeSource:
        source = self.get_source(source_id)
        updated = replace(
            source,
            chunk_count=int(sync_result.node_count),
            index_status="ready",
            index_version=str(sync_result.index_version),
            indexed_at=time.time(),
        )
        _atomic_json(self.sources_dir / f"{updated.source_id}.json", updated.to_dict())
        return updated

    def metadata_summary(self, *, tenant_id: str | None = None) -> dict[str, Any]:
        sources = self.list_sources(tenant_id=tenant_id)
        chunk_counts = {
            source.corpus_id: (
                source.chunk_count
                or self.corpus_store.get_metadata(source.corpus_id).chunk_count
            )
            for source in sources
        }
        return {
            "source_count": len(sources),
            "chunk_count": sum(chunk_counts.values()),
            "ready_source_count": sum(source.index_status == "ready" for source in sources),
            "pending_source_count": sum(source.index_status != "ready" for source in sources),
            "corpus_ids": tuple(dict.fromkeys(source.corpus_id for source in sources)),
            "versions": {
                source.corpus_id: source.index_version or source.content_hash
                for source in sources
            },
        }

    def get_source(self, source_id: str) -> InterviewKnowledgeSource:
        path = self.sources_dir / f"{_slug(source_id)}.json"
        if not path.is_file():
            raise KeyError(f"Unknown interview knowledge source: {source_id}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Interview knowledge source metadata must be an object")
        return InterviewKnowledgeSource.from_dict(value)

    def list_sources(self, *, tenant_id: str | None = None) -> tuple[InterviewKnowledgeSource, ...]:
        sources = tuple(
            InterviewKnowledgeSource.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.sources_dir.glob("*.json"))
        )
        return tuple(source for source in sources if tenant_id is None or source.tenant_id == tenant_id)

    def delete_source(self, source_id: str, *, tenant_id: str | None = None) -> InterviewKnowledgeSource:
        """Delete source metadata and its corpus; never delete external originals."""
        source = self.get_source(source_id)
        if tenant_id is not None and source.tenant_id != tenant_id:
            raise KeyError(f"Unknown interview knowledge source: {source_id}")
        metadata_path = self.sources_dir / f"{_slug(source.source_id)}.json"
        metadata_path.unlink(missing_ok=True)
        # A corpus is content-addressed and normally belongs to one source,
        # but keep shared corpora intact if legacy data linked more than one.
        shared = any(
            item.source_id != source.source_id and item.corpus_id == source.corpus_id
            for item in self.list_sources()
        )
        if not shared:
            self.corpus_store.delete(source.corpus_id)
        return source

    def verify_source_integrity(self, source_id: str) -> bool:
        source = self.get_source(source_id)
        if not source.source_path:
            return True
        path = Path(source.source_path)
        return path.is_file() and _hash_bytes(path.read_bytes()) == source.content_hash

    def source_chunks(self, source: InterviewKnowledgeSource) -> tuple[InterviewKnowledgeChunk, ...]:
        corpus = self.corpus_store.get(source.corpus_id)
        return tuple(chunk for chunk in corpus.chunks if chunk.source_id == source.source_id)

    def _parse(
        self,
        *,
        text: str | None,
        path: Path | None,
        raw: bytes,
        source_name: str = "",
        pdf_mode: str = "normal",
    ) -> tuple[bytes, list[tuple[str, dict[str, Any]]], str, str, str]:
        if text is not None:
            normalized = _required_text(text, "text")
            return raw, self._text_blocks(normalized), "text/plain", "plain_text", "1"
        assert path is not None
        suffix = path.suffix.casefold()
        if suffix in {".txt", ".md", ".markdown"}:
            try:
                value = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                value = raw.decode("gb18030")
            parser = "markdown" if suffix in {".md", ".markdown"} else "plain_text"
            blocks = self._markdown_blocks(value) if parser == "markdown" else self._text_blocks(value)
            return raw, blocks, "text/markdown" if parser == "markdown" else "text/plain", parser, "2" if parser == "markdown" else "1"
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("pypdf is required to import PDF interview knowledge") from exc
            reader = PdfReader(path)
            pypdf_pages: list[tuple[int, str]] = []
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text(extraction_mode="layout") or ""
                except (TypeError, ValueError):
                    page_text = page.extract_text() or ""
                pypdf_pages.append((page_number, page_text))
            blocks: list[tuple[str, dict[str, Any]]] = []
            parser = "pypdf"
            try:
                blocks = (
                    self._pdf_layout_blocks(path, pdf_mode=pdf_mode)
                    if pdf_mode == "qa"
                    else self._pdf_layout_blocks(path)
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                blocks = []
            if blocks:
                parser = "pymupdf-layout"
            else:
                blocks = self._pdf_text_blocks(pypdf_pages, pdf_mode=pdf_mode)
            if not blocks:
                blocks = self._ocr_pdf_blocks(path)
                parser = "pymupdf+ocr"
            if pdf_mode == "qa":
                parser = f"{parser}+qa"
            return raw, blocks, "application/pdf", parser, "7"
        if suffix == ".docx":
            try:
                from docx import Document
            except ImportError as exc:
                raise RuntimeError("python-docx is required to import DOCX interview knowledge") from exc
            document = Document(path)
            blocks: list[tuple[str, dict[str, Any]]] = []
            chapter = ""
            for paragraph_number, paragraph in enumerate(document.paragraphs, start=1):
                content = paragraph.text.strip()
                if not content:
                    continue
                style = str(getattr(paragraph.style, "name", "") or "")
                is_heading = style.casefold().startswith("heading")
                if is_heading:
                    chapter = content
                blocks.append(
                    (
                        content,
                        {
                            "paragraph": paragraph_number,
                            "chapter": chapter,
                            "block_type": "title" if is_heading else self._block_type(content),
                        },
                    )
                )
            if not blocks:
                raise ValueError("DOCX interview knowledge contains no text")
            return raw, blocks, "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "python-docx", "1"
        if suffix == ".ipynb":
            blocks = self._notebook_blocks(raw)
            return raw, blocks, "application/x-ipynb+json", "jupyter-notebook", "1"
        if suffix in _IMAGE_MEDIA_TYPES:
            image_name = str(source_name or path.name).strip() or path.name
            self._validate_ocr_image(raw)
            content, ocr_engine, confidence = self._ocr_image(raw, image_name=image_name)
            if not content.strip():
                raise ValueError("Image interview knowledge contains no extractable text after OCR")
            return (
                raw,
                [
                    (
                        content.strip(),
                        {
                            "block_type": "ocr",
                            "image_name": image_name,
                            "ocr_engine": ocr_engine,
                            "ocr_confidence": confidence,
                        },
                    )
                ],
                _IMAGE_MEDIA_TYPES[suffix],
                "image+ocr",
                "1",
            )
        raise ValueError(
            "Interview knowledge supports TXT, Markdown, PDF, DOCX, IPYNB, PNG, JPG/JPEG, BMP, WEBP, and TIFF files"
        )

    @staticmethod
    def _validate_ocr_image(raw: bytes) -> None:
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as exc:
            raise RuntimeError("Pillow is required to import image interview knowledge") from exc
        try:
            max_pixels = int(os.environ.get("SMART_AGENT_OCR_MAX_PIXELS", str(_DEFAULT_OCR_MAX_PIXELS)) or 0)
        except ValueError as exc:
            raise RuntimeError("SMART_AGENT_OCR_MAX_PIXELS must be a positive integer") from exc
        if max_pixels <= 0:
            raise RuntimeError("SMART_AGENT_OCR_MAX_PIXELS must be a positive integer")
        try:
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError("Image interview knowledge has invalid dimensions")
                if width * height > max_pixels:
                    raise ValueError(
                        f"Image interview knowledge exceeds the {max_pixels}-pixel OCR limit"
                    )
                image.verify()
        except ValueError:
            raise
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise ValueError("Image interview knowledge is invalid or corrupted") from exc

    @staticmethod
    def _ocr_engine() -> tuple[str, Any]:
        try:
            import pytesseract

            return "pytesseract", pytesseract
        except ImportError:
            try:
                from paddleocr import PaddleOCR
            except ImportError:
                external_python = Path(
                    os.environ.get(
                        "SMART_AGENT_PADDLEOCR_PYTHON",
                        r"D:\RAG\environments\paddle_env\Scripts\python.exe",
                    )
                )
                if not external_python.is_file():
                    raise RuntimeError(
                        "Image OCR requires pytesseract+tesseract, PaddleOCR, or SMART_AGENT_PADDLEOCR_PYTHON"
                    )
                return "paddleocr-subprocess", str(external_python)
            return (
                "paddleocr",
                PaddleOCR(use_angle_cls=True, lang=os.environ.get("SMART_AGENT_PADDLEOCR_LANG", "ch")),
            )

    @staticmethod
    def _ocr_image(raw: bytes, *, image_name: str) -> tuple[str, str, float]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to import image interview knowledge") from exc
        ocr_engine, ocr_runner = InterviewKnowledgeLibrary._ocr_engine()
        language = os.environ.get("SMART_AGENT_OCR_LANGUAGE", "chi_sim+eng")
        try:
            if ocr_engine == "pytesseract":
                with Image.open(io.BytesIO(raw)) as image:
                    data = ocr_runner.image_to_data(
                        image,
                        lang=language,
                        output_type=ocr_runner.Output.DICT,
                    )
                content, confidence = InterviewKnowledgeLibrary._pytesseract_ocr_result(data)
            else:
                suffix = Path(image_name).suffix.casefold()
                if suffix not in _IMAGE_MEDIA_TYPES:
                    suffix = ".png"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                    handle.write(raw)
                    image_path = handle.name
                try:
                    if ocr_engine == "paddleocr-subprocess":
                        content, confidence = InterviewKnowledgeLibrary._paddle_ocr_subprocess(
                            str(ocr_runner), image_path
                        )
                    else:
                        result = (
                            ocr_runner.ocr(image_path, cls=True)
                            if hasattr(ocr_runner, "ocr")
                            else ocr_runner.predict(image_path)
                        )
                        content, confidence = InterviewKnowledgeLibrary._paddle_ocr_result(result)
                finally:
                    try:
                        os.unlink(image_path)
                    except OSError:
                        pass
        except Exception as exc:
            raise RuntimeError(
                f"Image OCR failed for {image_name}; check OCR packages, language packs, and OCR configuration"
            ) from exc
        return content.strip(), ocr_engine, confidence

    @staticmethod
    def _pytesseract_ocr_result(data: dict[str, Any]) -> tuple[str, float]:
        raw_texts = list(data.get("text") or ())
        raw_confidences = list(data.get("conf") or ())
        lines: dict[tuple[Any, Any, Any], list[str]] = {}
        confidences: list[float] = []

        def value_at(field: str, index: int) -> Any:
            values = data.get(field) or ()
            return values[index] if index < len(values) else 0

        for index, raw_text in enumerate(raw_texts):
            text = str(raw_text or "").strip()
            if not text:
                continue
            line_key = (
                value_at("block_num", index),
                value_at("par_num", index),
                value_at("line_num", index),
            )
            lines.setdefault(line_key, []).append(text)
            if index < len(raw_confidences):
                try:
                    confidence = float(raw_confidences[index])
                except (TypeError, ValueError):
                    continue
                if confidence >= 0:
                    confidences.append(max(0.0, min(1.0, confidence / 100.0)))
        content = "\n".join(" ".join(words) for words in lines.values() if words)
        average = sum(confidences) / len(confidences) if confidences else 0.0
        return content, round(average, 6)

    @staticmethod
    def _ocr_pdf_blocks(path: Path) -> list[tuple[str, dict[str, Any]]]:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("Scanned PDF interview knowledge requires PyMuPDF") from exc
        blocks: list[tuple[str, dict[str, Any]]] = []
        scale = float(os.environ.get("SMART_AGENT_PDF_OCR_SCALE", "2.0") or 2.0)
        try:
            with fitz.open(str(path)) as document:
                matrix = fitz.Matrix(scale, scale)
                for page_number, page in enumerate(document, start=1):
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    png_bytes = pixmap.tobytes("png")
                    InterviewKnowledgeLibrary._validate_ocr_image(png_bytes)
                    content, ocr_engine, confidence = InterviewKnowledgeLibrary._ocr_image(
                        png_bytes,
                        image_name=f"{path.name}#page-{page_number}.png",
                    )
                    if content:
                        blocks.append(
                            (
                                content,
                                {
                                    "page": page_number,
                                    "block_type": "ocr",
                                    "ocr_engine": ocr_engine,
                                    "ocr_confidence": confidence,
                                },
                            )
                        )
        except Exception as exc:
            raise RuntimeError(
                "Scanned PDF OCR failed; check OCR package, language packs, and SMART_AGENT_OCR_LANGUAGE"
            ) from exc
        if not blocks:
            raise ValueError("PDF interview knowledge contains no extractable text after OCR")
        return blocks

    @staticmethod
    def _paddle_ocr_text(result: Any) -> str:
        return InterviewKnowledgeLibrary._paddle_ocr_result(result)[0]

    @staticmethod
    def _paddle_ocr_result(result: Any) -> tuple[str, float]:
        entries: list[tuple[str, float | None]] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                rec_texts = value.get("rec_texts")
                if isinstance(rec_texts, list):
                    rec_scores = value.get("rec_scores")
                    for index, item in enumerate(rec_texts):
                        text = str(item or "").strip()
                        if not text:
                            continue
                        score: float | None = None
                        if isinstance(rec_scores, (list, tuple)) and index < len(rec_scores):
                            try:
                                score = float(rec_scores[index])
                            except (TypeError, ValueError):
                                pass
                        entries.append((text, score))
                for key, nested in value.items():
                    if key in {"rec_texts", "rec_scores"}:
                        continue
                    visit(nested)
                return
            if isinstance(value, (list, tuple)):
                if len(value) >= 2 and isinstance(value[1], (list, tuple)) and value[1] and isinstance(value[1][0], str):
                    score = None
                    if len(value[1]) >= 2:
                        try:
                            score = float(value[1][1])
                        except (TypeError, ValueError):
                            pass
                    entries.append((str(value[1][0]), score))
                    return
                for item in value:
                    visit(item)

        visit(result)
        unique_entries: list[tuple[str, float | None]] = []
        seen: set[str] = set()
        for text, score in entries:
            normalized = text.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_entries.append((normalized, score))
        scores = [max(0.0, min(1.0, score)) for _, score in unique_entries if score is not None]
        confidence = sum(scores) / len(scores) if scores else 0.0
        return "\n".join(text for text, _ in unique_entries), round(confidence, 6)

    @staticmethod
    def _paddle_ocr_subprocess(python_executable: str, image_path: str) -> tuple[str, float]:
        script = r'''
import json
import sys
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang=sys.argv[2])
result = ocr.ocr(sys.argv[1], cls=True) if hasattr(ocr, "ocr") else ocr.predict(sys.argv[1])
entries = []
def visit(value):
    if isinstance(value, dict):
        rec_texts = value.get("rec_texts")
        if isinstance(rec_texts, list):
            rec_scores = value.get("rec_scores")
            for index, item in enumerate(rec_texts):
                text = str(item or "").strip()
                if text:
                    score = None
                    if isinstance(rec_scores, (list, tuple)) and index < len(rec_scores):
                        try:
                            score = float(rec_scores[index])
                        except (TypeError, ValueError):
                            pass
                    entries.append((text, score))
        for key, nested in value.items():
            if key not in {"rec_texts", "rec_scores"}:
                visit(nested)
    elif isinstance(value, (list, tuple)):
        if len(value) >= 2 and isinstance(value[1], (list, tuple)) and value[1] and isinstance(value[1][0], str):
            score = None
            if len(value[1]) >= 2:
                try:
                    score = float(value[1][1])
                except (TypeError, ValueError):
                    pass
            entries.append((value[1][0], score))
        else:
            for item in value:
                visit(item)
visit(result)
unique = []
seen = set()
for text, score in entries:
    text = str(text or "").strip()
    if text and text not in seen:
        seen.add(text)
        unique.append((text, score))
scores = [max(0.0, min(1.0, score)) for _, score in unique if score is not None]
confidence = sum(scores) / len(scores) if scores else 0.0
print("SMART_AGENT_OCR_JSON:" + json.dumps({"text": "\n".join(text for text, _ in unique), "confidence": confidence}, ensure_ascii=False))
'''
        completed = subprocess.run(
            [
                python_executable,
                "-c",
                script,
                image_path,
                os.environ.get("SMART_AGENT_PADDLEOCR_LANG", "ch"),
            ],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("SMART_AGENT_PADDLEOCR_TIMEOUT", "180") or 180),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "PaddleOCR subprocess failed").strip()[:1000])
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith("SMART_AGENT_OCR_JSON:"):
                payload = json.loads(line.split(":", 1)[1])
                return (
                    str(payload.get("text") or ""),
                    round(max(0.0, min(1.0, float(payload.get("confidence") or 0.0))), 6),
                )
        raise RuntimeError("PaddleOCR subprocess did not return OCR JSON")

    @staticmethod
    def _notebook_blocks(raw: bytes) -> list[tuple[str, dict[str, Any]]]:
        try:
            notebook = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("IPYNB interview knowledge must be a valid UTF-8 Jupyter notebook") from exc
        cells = notebook.get("cells")
        if not isinstance(cells, list):
            raise ValueError("IPYNB interview knowledge contains no cells")
        blocks: list[tuple[str, dict[str, Any]]] = []
        chapter = ""
        for cell_number, cell in enumerate(cells, start=1):
            if not isinstance(cell, dict):
                continue
            cell_type = str(cell.get("cell_type") or "").strip().lower()
            if cell_type not in {"markdown", "code", "raw"}:
                continue
            source = cell.get("source") or ""
            content = "".join(str(part) for part in source) if isinstance(source, list) else str(source)
            content = content.strip()
            if not content:
                continue
            if cell_type == "markdown":
                first_line = content.splitlines()[0].strip()
                if first_line.startswith("#"):
                    chapter = first_line.lstrip("#").strip()
                block_type = "title" if first_line.startswith("#") else InterviewKnowledgeLibrary._block_type(content)
            elif cell_type == "code":
                block_type = "code"
                content = f"```python\n{content}\n```"
            else:
                block_type = InterviewKnowledgeLibrary._block_type(content)
            blocks.append(
                (
                    content,
                    {
                        "cell": cell_number,
                        "chapter": chapter,
                        "cell_type": cell_type,
                        "block_type": block_type,
                    },
                )
            )
        if not blocks:
            raise ValueError("IPYNB interview knowledge contains no text cells")
        return blocks

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(text or ""))
        normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
        normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", normalized)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        return normalized.strip()

    @staticmethod
    def _is_markdown_h2(line: str) -> bool:
        return bool(re.match(r"^##(?!#)\s+", line.strip()))

    @staticmethod
    def _is_heading_line(text: str) -> bool:
        stripped = InterviewKnowledgeLibrary._clean_extracted_text(text).strip()
        if not stripped:
            return False
        if stripped.startswith("#"):
            return True
        return bool(
            re.match(
                r"^(?:第\s*[一二三四五六七八九十百\d]+\s*[章节篇]"
                r"|\d+(?:\.\d+)+(?:[、.\s]|(?=[A-Za-z\u4e00-\u9fff]))"
                r"|[一二三四五六七八九十]+、)",
                stripped,
            )
        )

    @staticmethod
    def _is_noise_line(text: str) -> bool:
        stripped = InterviewKnowledgeLibrary._clean_extracted_text(text).strip()
        if not stripped:
            return True
        if stripped == "目录" or re.fullmatch(r"(?i)[ivxlcdm]{1,8}", stripped):
            return True
        if re.fullmatch(r"[\[\]{}(),;:，。；：、.\-—_+=*/\\|<>\"“”‘’]+", stripped):
            return True
        if re.fullmatch(r"\d{1,3}", stripped):
            return True
        if re.fullmatch(r"\d{1,3}\s*/\s*\d{1,3}", stripped):
            return True
        return False

    @staticmethod
    def _line_continues_previous(previous: str, current: str) -> bool:
        prev = previous.rstrip()
        curr = current.lstrip()
        if not prev or not curr:
            return False
        if InterviewKnowledgeLibrary._is_heading_line(prev):
            return False
        if prev.endswith((".", "。", "!", "！", "?", "？", ";", "；", ":", "：", ")", "）", "]", "】", '"', "”")):
            return False
        if curr.startswith(("-", "*", "+", "•", "|")):
            return False
        if InterviewKnowledgeLibrary._is_heading_line(curr):
            return False
        return True

    @staticmethod
    def _reflow_wrapped_lines(text: str) -> str:
        lines = [line.rstrip() for line in InterviewKnowledgeLibrary._clean_extracted_text(text).splitlines()]
        output: list[str] = []
        in_fence = False
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                if output and output[-1] != "":
                    output.append("")
                continue
            if line.startswith(chr(96) * 3):
                output.append(line)
                in_fence = not in_fence
                continue
            if in_fence:
                output.append(raw_line.rstrip())
                continue
            if InterviewKnowledgeLibrary._is_noise_line(line):
                continue
            if output and output[-1] and InterviewKnowledgeLibrary._line_continues_previous(output[-1], line):
                output[-1] = f"{output[-1].rstrip()}{line}"
            else:
                output.append(line)
        while output and not output[-1]:
            output.pop()
        return "\n".join(output).strip()

    @staticmethod
    def _markdown_blocks(text: str) -> list[tuple[str, dict[str, Any]]]:
        cleaned = InterviewKnowledgeLibrary._clean_extracted_text(text)
        lines = cleaned.splitlines()
        sections: list[tuple[str, int, int, str]] = []
        current: list[str] = []
        start_line = 1
        chapter = ""

        def chapter_from(content: str, fallback: str) -> str:
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    return stripped.lstrip("#").strip()
            return fallback

        def flush(end_line: int) -> None:
            nonlocal current, start_line, chapter
            content = "\n".join(current).strip()
            if content:
                next_chapter = chapter_from(content, chapter)
                has_body = any(line.strip() and not line.strip().startswith("#") for line in content.splitlines())
                if has_body:
                    sections.append((content, start_line, end_line, next_chapter))
                chapter = next_chapter
            current = []

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if InterviewKnowledgeLibrary._is_markdown_h2(stripped):
                flush(line_number - 1)
                start_line = line_number
                chapter = stripped.lstrip("#").strip()
            elif not current:
                start_line = line_number
                if stripped.startswith("#"):
                    chapter = stripped.lstrip("#").strip()
            current.append(line)
        flush(len(lines))
        blocks = [
            (
                content,
                {
                    "line": start,
                    "line_start": start,
                    "line_end": end,
                    "chapter": chapter,
                    "block_type": "markdown_section",
                },
            )
            for content, start, end, chapter in sections
            if content.strip()
        ]
        if not blocks:
            raise ValueError("Interview knowledge contains no text")
        return blocks

    @staticmethod
    def _pdf_text_blocks(
        pages: list[tuple[int, str]],
        *,
        pdf_mode: str = "normal",
    ) -> list[tuple[str, dict[str, Any]]]:
        pages = InterviewKnowledgeLibrary._prepare_pdf_pages(pages)
        if pdf_mode == "qa":
            qa_blocks = InterviewKnowledgeLibrary._qa_pdf_blocks(pages)
            if not qa_blocks:
                raise ValueError(
                    "Selected QA PDF contains no recognizable numbered question and answer pairs"
                )
            return qa_blocks
        blocks: list[tuple[str, dict[str, Any]]] = []
        chapter = ""
        for page_number, page_text in pages:
            page_blocks = InterviewKnowledgeLibrary._section_blocks_from_lines(
                page_text,
                base_locator={"page": page_number},
                chapter=chapter,
            )
            if pdf_mode != "qa":
                page_blocks = [
                    (
                        content,
                        {**locator, "block_type": "paragraph"}
                        if locator.get("block_type") == "qa"
                        else locator,
                    )
                    for content, locator in page_blocks
                ]
            if page_blocks:
                last_chapter = str(page_blocks[-1][1].get("chapter") or "").strip()
                if last_chapter:
                    chapter = last_chapter
            blocks.extend(page_blocks)
        return blocks

    @staticmethod
    def _prepare_pdf_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
        first_lines: list[str] = []
        for _page_number, page_text in pages:
            for line in InterviewKnowledgeLibrary._clean_extracted_text(page_text).splitlines():
                if line.strip():
                    first_lines.append(line.strip())
                    break
        repeated_headers = {
            line
            for line in first_lines
            if first_lines.count(line) >= 2 and line != "目录"
        }
        prepared: list[tuple[int, str]] = []
        for page_number, page_text in pages:
            cleaned = InterviewKnowledgeLibrary._clean_extracted_text(page_text)
            lines = [line for line in cleaned.splitlines() if line.strip()]
            dot_leaders = sum(
                bool(re.search(r"(?:\.\s*){5,}\d*\s*$", line))
                for line in lines
            )
            toc_headings = sum(
                bool(re.match(r"^(?:第\s*[一二三四五六七八九十百\d]+\s*章|\d+\.\d+)", line))
                for line in lines
            )
            if lines and lines[0].strip() == "目录" and (dot_leaders >= 3 or toc_headings >= 5):
                continue
            filtered = [
                line
                for index, line in enumerate(lines)
                if not InterviewKnowledgeLibrary._is_noise_line(line)
                and not (index == 0 and line.strip() in repeated_headers)
            ]
            content = "\n".join(filtered).strip()
            if content:
                prepared.append((page_number, content))
        return prepared

    @staticmethod
    def _qa_pdf_blocks(pages: list[tuple[int, str]]) -> list[tuple[str, dict[str, Any]]]:
        page_ranges: list[tuple[int, int, int]] = []
        parts: list[str] = []
        cursor = 0
        for page_number, page_text in pages:
            content = InterviewKnowledgeLibrary._reflow_wrapped_lines(page_text)
            if not content:
                continue
            if parts:
                parts.append("\n")
                cursor += 1
            start = cursor
            parts.append(content)
            cursor += len(content)
            page_ranges.append((start, cursor, page_number))
        joined = "".join(parts).strip()
        matches = list(_QA_QUESTION_PATTERN.finditer(joined))
        if not matches or not page_ranges:
            return []
        spans: list[tuple[re.Match[str], int, int]] = []
        answer_count = 0
        answer_chars = 0
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(joined)
            spans.append((match, match.start(), end))
            value = joined[match.start() : end]
            if _QA_ANSWER_PATTERN.search(value):
                answer_count += 1
                answer_chars += len(value)
        if answer_count == 0 or answer_chars == 0:
            return []

        def page_at(offset: int) -> int:
            for start, end, page_number in page_ranges:
                if start <= offset < end:
                    return page_number
            return page_ranges[-1][2]

        blocks: list[tuple[str, dict[str, Any]]] = []
        preamble = joined[: matches[0].start()].strip()
        if preamble and not InterviewKnowledgeLibrary._is_noise_line(preamble):
            blocks.append(
                (
                    preamble,
                    {
                        "page": page_at(0),
                        "page_start": page_at(0),
                        "page_end": page_at(max(0, matches[0].start() - 1)),
                        "block_type": "paragraph",
                        "document_layout": "qa_pdf",
                    },
                )
            )
        for ordinal, (match, start, end) in enumerate(spans, start=1):
            content = joined[start:end].strip()
            if not content:
                continue
            question_number = match.group("number")
            blocks.append(
                (
                    content,
                    {
                        "page": page_at(start),
                        "page_start": page_at(start),
                        "page_end": page_at(max(start, end - 1)),
                        "block_type": "qa",
                        "document_layout": "qa_pdf",
                        "qa_id": f"qa-{ordinal:04d}-{_slug(question_number)}",
                        "question_no": question_number,
                    },
                )
            )
        return blocks

    @staticmethod
    def _pdf_layout_blocks(
        path: Path,
        *,
        pdf_mode: str = "normal",
    ) -> list[tuple[str, dict[str, Any]]]:
        import fitz

        pages: list[tuple[int, str]] = []
        with fitz.open(str(path)) as document:
            for page_number, page in enumerate(document, start=1):
                pages.append((page_number, page.get_text("text", sort=True) or ""))
        return InterviewKnowledgeLibrary._pdf_text_blocks(pages, pdf_mode=pdf_mode)

    @staticmethod
    def _section_blocks_from_lines(
        text: str,
        *,
        base_locator: dict[str, Any] | None = None,
        chapter: str = "",
    ) -> list[tuple[str, dict[str, Any]]]:
        cleaned = InterviewKnowledgeLibrary._reflow_wrapped_lines(text)
        if not cleaned:
            return []
        base = dict(base_locator or {})
        blocks: list[tuple[str, dict[str, Any]]] = []
        current: list[str] = []
        current_start = 1
        current_chapter = chapter
        lines = cleaned.splitlines()

        def flush(end_line: int) -> None:
            nonlocal current, current_start
            content = "\n".join(current).strip()
            current = []
            if not content:
                return
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            if len(lines) == 1 and InterviewKnowledgeLibrary._is_heading_line(lines[0]):
                return
            blocks.append(
                (
                    content,
                    {
                        **base,
                        "line": current_start,
                        "line_start": current_start,
                        "line_end": end_line,
                        "chapter": current_chapter,
                        "block_type": InterviewKnowledgeLibrary._block_type(content),
                    },
                )
            )

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if InterviewKnowledgeLibrary._is_heading_line(stripped):
                flush(line_number - 1)
                current_start = line_number
                current_chapter = stripped.lstrip("#").strip()
                current.append(stripped)
                continue
            if not stripped:
                if current:
                    current.append("")
                continue
            if not current:
                current_start = line_number
            current.append(stripped)
        flush(len(lines))
        return blocks

    @staticmethod
    def _text_blocks(text: str, *, markdown: bool = False) -> list[tuple[str, dict[str, Any]]]:
        if markdown:
            return InterviewKnowledgeLibrary._markdown_blocks(text)
        blocks: list[tuple[str, dict[str, Any]]] = []
        chapter = ""
        paragraph: list[str] = []
        paragraph_start = 0

        def flush(end_line: int) -> None:
            nonlocal paragraph, paragraph_start
            if not paragraph:
                return
            content = "\n".join(paragraph).strip()
            block_type = InterviewKnowledgeLibrary._block_type(content)
            blocks.append(
                (
                    content,
                    {
                        "line": paragraph_start,
                        "line_start": paragraph_start,
                        "line_end": end_line,
                        "chapter": chapter,
                        "block_type": block_type,
                    },
                )
            )
            paragraph = []
            paragraph_start = 0

        lines = text.splitlines()
        for line_number, raw_line in enumerate(lines, start=1):
            content = raw_line.strip()
            if not content:
                flush(line_number - 1)
                continue
            if markdown and content.startswith("#"):
                flush(line_number - 1)
                chapter = content.lstrip("#").strip()
                continue
            if paragraph and InterviewKnowledgeLibrary._looks_like_table_row(content) != all(
                InterviewKnowledgeLibrary._looks_like_table_row(line) for line in paragraph
            ):
                flush(line_number - 1)
            if not paragraph:
                paragraph_start = line_number
            paragraph.append(content)
        flush(len(lines))
        if not blocks:
            raise ValueError("Interview knowledge contains no text")
        return blocks

    @staticmethod
    def _looks_like_table_row(text: str) -> bool:
        stripped = text.strip()
        return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3

    @staticmethod
    def _block_type(text: str) -> str:
        stripped = text.strip()
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if stripped.startswith(chr(96) * 3):
            return "code"
        if len(lines) >= 2 and all(InterviewKnowledgeLibrary._looks_like_table_row(line) for line in lines):
            return "table"
        if len(lines) >= 2 and (
            any(re.match(r'^\s*"[^"]+"\s*:', line) for line in lines)
            or stripped.startswith(("{", "["))
        ):
            return "code"
        if len(lines) == 1 and re.match(
            r"^(?:#{1,6}\s+|\u7b2c\s*[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\d]+\s*[\u7ae0\u8282\u7bc7]"
            r"|\d+(?:\.\d+)+(?:[\u3001.\s]|(?=[A-Za-z\u4e00-\u9fff]))"
            r"|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+\u3001)",
            stripped,
        ):
            return "title"
        if (
            (_QA_QUESTION_PATTERN.search(stripped) or re.search(r"(?im)^(?:Q|Question|问题|问)\s*[:：]", stripped))
            and _QA_ANSWER_PATTERN.search(stripped)
        ):
            return "qa"
        return "paragraph"

    def _split_sentences(self, text: str) -> list[str]:
        normalized = self._reflow_wrapped_lines(text)
        if not normalized:
            return []
        if len(normalized) <= self.max_chunk_chars:
            return [normalized]
        overlap = min(80, max(20, self.max_chunk_chars // 12))
        paragraphs = [
            value.strip()
            for value in re.split(r"\n{2,}", normalized)
            if value.strip()
        ]
        units: list[str] = []
        for paragraph in paragraphs or [normalized]:
            if len(paragraph) <= self.max_chunk_chars:
                units.append(paragraph)
                continue
            sentences = [
                value.strip()
                for value in re.findall(r".+?(?:[。！？!?；;](?:\s+|$)|$)", paragraph, flags=re.S)
                if value.strip()
            ] or [paragraph]
            for sentence in sentences:
                if len(sentence) <= self.max_chunk_chars:
                    units.append(sentence)
                    continue
                step = max(1, self.max_chunk_chars - overlap)
                units.extend(sentence[index : index + self.max_chunk_chars] for index in range(0, len(sentence), step))
        if not units:
            return []
        units = [unit for unit in units if not self._is_noise_line(unit)]
        if not units:
            return []
        chunks: list[str] = []
        current = ""
        for unit in units:
            separator = "\n\n" if "\n" in unit or "\n" in current else "\n"
            if current and len(current) + len(separator) + len(unit) > self.max_chunk_chars:
                chunks.append(current)
                carry = current[-overlap:].lstrip()
                current = f"{carry}\n{unit}" if carry and not unit.startswith(carry) else unit
            else:
                current = f"{current}{separator}{unit}" if current else unit
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _split_without_overlap(text: str, max_chars: int) -> list[str]:
        normalized = InterviewKnowledgeLibrary._reflow_wrapped_lines(text)
        if not normalized:
            return []
        if len(normalized) <= max_chars:
            return [normalized]
        units = [
            value.strip()
            for value in re.findall(r".+?(?:[\u3002\uff01\uff1f!?\uff1b;](?:\s+|$)|$)", normalized, flags=re.S)
            if value.strip()
        ] or [normalized]
        expanded: list[str] = []
        for unit in units:
            if len(unit) <= max_chars:
                expanded.append(unit)
            else:
                expanded.extend(unit[index : index + max_chars] for index in range(0, len(unit), max_chars))
        chunks: list[str] = []
        current = ""
        for unit in expanded:
            separator = "\n" if current else ""
            if current and len(current) + len(separator) + len(unit) > max_chars:
                chunks.append(current)
                current = unit
            else:
                current = f"{current}{separator}{unit}" if current else unit
        if current:
            chunks.append(current)
        return chunks

    def _split_qa(self, text: str) -> list[str]:
        normalized = self._reflow_wrapped_lines(text)
        if not normalized or len(normalized) <= self.max_chunk_chars:
            return [normalized] if normalized else []
        answer = _QA_ANSWER_PATTERN.search(normalized)
        if answer is None:
            return self._split_without_overlap(normalized, self.max_chunk_chars)
        question = normalized[: answer.start()].strip()
        answer_label = answer.group(0).strip()
        answer_body = normalized[answer.end() :].strip()
        prefix = f"{question}\n{answer_label}".strip()
        available = self.max_chunk_chars - len(prefix) - 1
        if available < 80:
            return self._split_without_overlap(normalized, self.max_chunk_chars)
        answer_parts = self._split_without_overlap(answer_body, available)
        return [f"{prefix}\n{part}".strip() for part in answer_parts] or [prefix]

    def _split_table(self, text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(text) <= self.max_chunk_chars or len(lines) < 3:
            return [text.strip()]
        header_count = 2 if re.fullmatch(r"\|?[\s:|-]+\|?", lines[1]) else 1
        header = lines[:header_count]
        chunks: list[str] = []
        current = list(header)
        for row in lines[header_count:]:
            proposed = "\n".join([*current, row])
            if len(proposed) > self.max_chunk_chars and len(current) > header_count:
                chunks.append("\n".join(current))
                current = [*header, row]
            else:
                current.append(row)
        if len(current) > header_count:
            chunks.append("\n".join(current))
        return chunks or [text.strip()]

    def _chunks(
        self,
        *,
        source_id: str,
        corpus_id: str,
        tenant_id: str,
        blocks: list[tuple[str, dict[str, Any]]],
        authority_level: int,
        domains: tuple[str, ...],
        subdomains: tuple[str, ...],
        topics: tuple[str, ...],
        difficulty: int,
        source_type: InterviewKnowledgeSourceType,
        security_flags: tuple[str, ...],
    ) -> tuple[InterviewKnowledgeChunk, ...]:
        records: list[tuple[str, dict[str, Any], int, int]] = []
        for block_number, (content, locator) in enumerate(blocks, start=1):
            block_type = str(locator.get("block_type") or self._block_type(content))
            chapter = str(locator.get("chapter") or "").strip()
            if block_type == "table":
                segments = self._split_table(content)
            elif block_type == "qa":
                segments = self._split_qa(content)
            else:
                segments = self._split_sentences(content)
            part_total = len(segments)
            for segment_number, segment in enumerate(segments, start=1):
                cleaned_segment = self._clean_extracted_text(segment)
                if (
                    not cleaned_segment
                    or len(cleaned_segment.strip()) < MIN_KNOWLEDGE_CHUNK_CHARS
                    or self._is_noise_line(cleaned_segment)
                ):
                    continue
                segment_locator = dict(locator)
                if block_type == "qa":
                    segment_locator.update(
                        {
                            "part_index": segment_number,
                            "part_total": part_total,
                        }
                    )
                records.append((cleaned_segment, segment_locator, block_number, segment_number))
        identities = [
            "knowledge-"
            + hashlib.sha256(
                f"{source_id}\0{KNOWLEDGE_CHUNKER_VERSION}\0{block_number}\0{segment_number}\0{text}".encode("utf-8")
            ).hexdigest()[:20]
            for text, _locator, block_number, segment_number in records
        ]
        chunks: list[InterviewKnowledgeChunk] = []
        total_chunks = len(records)
        for index, ((segment, locator, block_number, segment_number), chunk_id) in enumerate(
            zip(records, identities, strict=True)
        ):
            enriched_locator = {
                **locator,
                "block": block_number,
                "segment": segment_number,
                "block_type": str(locator.get("block_type") or self._block_type(segment)),
                "chunk_index": index,
                "total_chunks": total_chunks,
                "prev_chunk_id": identities[index - 1] if index > 0 else None,
                "next_chunk_id": identities[index + 1] if index + 1 < total_chunks else None,
                "chunker_version": KNOWLEDGE_CHUNKER_VERSION,
                "tenant_id": tenant_id,
                "authority_level": authority_level,
                "source_type": source_type.value,
                "domains": list(domains),
                "subdomains": list(subdomains),
                "topics": list(topics),
                "security_flags": list(security_flags),
            }
            chunks.append(
                InterviewKnowledgeChunk(
                    chunk_id=chunk_id,
                    corpus_id=corpus_id,
                    domain=domains[0],
                    subdomain=subdomains[0] if subdomains else "",
                    topic=topics[0],
                    text=segment,
                    source_id=source_id,
                    locator=enriched_locator,
                    difficulty=difficulty,
                    knowledge_type=(
                        InterviewKnowledgeType.USER_NOTE
                        if source_type in {InterviewKnowledgeSourceType.USER_NOTE, InterviewKnowledgeSourceType.INTERVIEW_NOTE}
                        else InterviewKnowledgeType.DOMAIN
                    ),
                )
            )
        return tuple(chunks)


@dataclass(frozen=True, slots=True)
class JDCapabilityNode:
    node_id: str
    parent_id: str | None
    label: str
    query: str
    requirement_ids: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    subdomains: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "label": self.label,
            "query": self.query,
            "requirement_ids": list(self.requirement_ids),
            "domains": list(self.domains),
            "subdomains": list(self.subdomains),
            "topics": list(self.topics),
        }


@dataclass(frozen=True, slots=True)
class JDCapabilityTree:
    jd_id: str
    root_id: str
    nodes: tuple[JDCapabilityNode, ...]
    builder: str
    taxonomy_version: str

    @property
    def leaves(self) -> tuple[JDCapabilityNode, ...]:
        parent_ids = {node.parent_id for node in self.nodes if node.parent_id}
        return tuple(node for node in self.nodes if node.node_id not in parent_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "jd_id": self.jd_id,
            "root_id": self.root_id,
            "builder": self.builder,
            "taxonomy_version": self.taxonomy_version,
            "nodes": [node.to_dict() for node in self.nodes],
        }


class HeuristicJDCapabilityTreeBuilder:
    name = "heuristic-jd-capability-tree-v1"
    taxonomy_version = "career-taxonomy-2026-08"
    _TAXONOMY = (
        (("python", "java", "fastapi", "backend", "api", "\u540e\u7aef"), "computer_science", "software_engineering", "backend engineering"),
        (("mysql", "postgres", "redis", "database", "sql", "\u6570\u636e\u5e93"), "computer_science", "data_systems", "database systems"),
        (("machine learning", "deep learning", "llm", "rag", "model", "\u673a\u5668\u5b66\u4e60", "\u6df1\u5ea6\u5b66\u4e60", "\u6a21\u578b"), "computer_science", "artificial_intelligence", "machine learning"),
        (("distributed", "kafka", "microservice", "consistency", "\u5206\u5e03\u5f0f", "\u5fae\u670d\u52a1", "\u4e00\u81f4\u6027"), "computer_science", "distributed_systems", "distributed systems"),
        (("product", "user research", "growth", "\u4ea7\u54c1", "\u7528\u6237\u7814\u7a76", "\u589e\u957f"), "product", "product_management", "product management"),
        (("finance", "risk", "quant", "\u91d1\u878d", "\u98ce\u63a7", "\u91cf\u5316"), "finance", "financial_engineering", "financial analysis"),
    )

    def build(self, jd: JDDocument, requirements: tuple[JDRequirement, ...]) -> JDCapabilityTree:
        root_id = f"capability-root-{jd.jd_id}"
        root = JDCapabilityNode(root_id, None, jd.role_title or "job capabilities", jd.raw_text)
        nodes: list[JDCapabilityNode] = [root]
        category_nodes: dict[str, str] = {}
        for requirement in requirements:
            category = requirement.category.value
            parent_id = category_nodes.get(category)
            if parent_id is None:
                parent_id = "capability-category-" + hashlib.sha256(f"{jd.jd_id}:{category}".encode()).hexdigest()[:12]
                category_nodes[category] = parent_id
                nodes.append(JDCapabilityNode(parent_id, root_id, category.replace("_", " "), category))
            domains: list[str] = []
            subdomains: list[str] = []
            topics: list[str] = []
            haystack = f"{jd.role_title} {requirement.text}".casefold()
            for keywords, domain, subdomain, topic in self._TAXONOMY:
                if any(keyword in haystack for keyword in keywords):
                    domains.append(domain)
                    subdomains.append(subdomain)
                    topics.append(topic)
            identity = hashlib.sha256(f"{jd.jd_id}:{requirement.requirement_id}".encode()).hexdigest()[:14]
            nodes.append(
                JDCapabilityNode(
                    node_id=f"capability-{identity}",
                    parent_id=parent_id,
                    label=requirement.text,
                    query=requirement.text,
                    requirement_ids=(requirement.requirement_id,),
                    domains=_stable_values(domains),
                    subdomains=_stable_values(subdomains),
                    topics=_stable_values(topics),
                )
            )
        return JDCapabilityTree(jd.jd_id, root_id, tuple(nodes), self.name, self.taxonomy_version)


@dataclass(frozen=True, slots=True)
class InterviewRetrievalFilters:
    tenant_id: str
    domains: tuple[str, ...] = ()
    subdomains: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    minimum_authority: int = 2
    maximum_difficulty: int = 5
    corpus_ids: tuple[str, ...] = ()
    include_security_flagged: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _required_text(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "domains", _stable_values(self.domains))
        object.__setattr__(self, "subdomains", _stable_values(self.subdomains))
        object.__setattr__(self, "topics", _stable_values(self.topics))
        object.__setattr__(self, "corpus_ids", _stable_values(self.corpus_ids))
        if not 1 <= self.minimum_authority <= 5 or not 1 <= self.maximum_difficulty <= 5:
            raise ValueError("Retrieval authority and difficulty filters must be between 1 and 5")


class InterviewEmbeddingProvider(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingInterviewEmbedding:
    def __init__(self, dimension: int = 256) -> None:
        self.dimension = int(dimension)
        if self.dimension < 32:
            raise ValueError("dimension must be at least 32")
        self.name = f"hashing-interview-{self.dimension}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in tokenize_knowledge(text):
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                vector[index] += -1.0 if digest[4] & 1 else 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class InterviewReranker(Protocol):
    name: str

    def rerank(
        self,
        query: str,
        rows: list[tuple[InterviewKnowledgeChunk, InterviewKnowledgeSource, float]],
    ) -> list[tuple[InterviewKnowledgeChunk, InterviewKnowledgeSource, float]]: ...


class AuthorityAwareReranker:
    name = "authority-topic-reranker-v1"

    def rerank(
        self,
        query: str,
        rows: list[tuple[InterviewKnowledgeChunk, InterviewKnowledgeSource, float]],
    ) -> list[tuple[InterviewKnowledgeChunk, InterviewKnowledgeSource, float]]:
        query_tokens = set(tokenize_knowledge(query))
        reranked = []
        for chunk, source, score in rows:
            topic_tokens = set(tokenize_knowledge(" ".join(source.topics)))
            topic_overlap = len(query_tokens & topic_tokens) / max(1, len(query_tokens))
            adjusted = min(1.0, score * 0.82 + source.authority_level / 5 * 0.12 + topic_overlap * 0.06)
            reranked.append((chunk, source, adjusted))
        return sorted(reranked, key=lambda row: (-row[2], row[0].chunk_id))


@dataclass(frozen=True, slots=True)
class InterviewRerankerCall:
    provider: str
    model: str
    api_id: str
    input_hash: str
    output_hash: str
    prompt_tokens: int
    latency_seconds: float
    cost_usd: float
    attempts: int
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_id": self.api_id,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "prompt_tokens": self.prompt_tokens,
            "latency_seconds": self.latency_seconds,
            "cost_usd": self.cost_usd,
            "attempts": self.attempts,
            "created_at": self.created_at,
        }


class OpenAICompatibleInterviewReranker:
    """Bounded `/rerank` client with source-free audit telemetry."""

    _RETRYABLE = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        input_price_per_million: float = 0.0,
        audit_path: str | Path | None = None,
        sleep=time.sleep,
    ) -> None:
        self.provider = _required_text(provider, "provider").lower()
        self.base_url = _required_text(base_url, "base_url").rstrip("/")
        self.api_key = _required_text(api_key, "api_key")
        self.model = _required_text(model, "model")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.input_price_per_million = float(input_price_per_million)
        self.audit_path = Path(audit_path).resolve() if audit_path else None
        self.sleep = sleep
        self.name = f"{self.provider}-reranker:{self.model}"
        self.last_call: InterviewRerankerCall | None = None
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Reranker base_url must be an http(s) URL")
        if self.timeout_seconds <= 0 or not 0 <= self.max_retries <= 5:
            raise ValueError("Invalid reranker timeout or retry count")
        if self.input_price_per_million < 0:
            raise ValueError("Reranker price cannot be negative")

    @classmethod
    def from_runtime(
        cls,
        runtime: dict[str, Any],
        *,
        audit_path: str | Path | None = None,
    ) -> "OpenAICompatibleInterviewReranker | None":
        values = {
            "provider": runtime.get("careerRerankerProvider") or os.environ.get("CAREER_RERANKER_PROVIDER"),
            "base_url": runtime.get("careerRerankerBaseUrl") or os.environ.get("CAREER_RERANKER_BASE_URL"),
            "api_key": runtime.get("careerRerankerApiKey") or os.environ.get("CAREER_RERANKER_API_KEY"),
            "model": runtime.get("careerRerankerModel") or os.environ.get("CAREER_RERANKER_MODEL"),
        }
        supplied = tuple(bool(str(value or "").strip()) for value in values.values())
        if not any(supplied):
            return None
        if not all(supplied):
            raise ValueError("Career reranker requires provider, base URL, API key, and model")
        return cls(
            **values,
            timeout_seconds=float(runtime.get("careerRerankerTimeoutSeconds") or 60),
            max_retries=int(runtime.get("careerRerankerMaxRetries") or 2),
            input_price_per_million=float(runtime.get("careerRerankerInputPricePerMillion") or 0),
            audit_path=audit_path,
        )

    def rerank(
        self,
        query: str,
        rows: list[tuple[InterviewKnowledgeChunk, InterviewKnowledgeSource, float]],
    ) -> list[tuple[InterviewKnowledgeChunk, InterviewKnowledgeSource, float]]:
        if not rows:
            return []
        endpoint = self.base_url if self.base_url.endswith("/rerank") else f"{self.base_url}/rerank"
        payload = {
            "model": self.model,
            "query": _required_text(query, "query"),
            "documents": [chunk.text for chunk, _source, _score in rows],
            "top_n": len(rows),
            "return_documents": False,
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.monotonic()
        response_payload: dict[str, Any] | None = None
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            request = urllib.request.Request(
                endpoint,
                data=encoded,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Smart-Agent-Career/1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    value = json.loads(response.read().decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("Reranker response must be an object")
                response_payload = value
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in self._RETRYABLE or attempt >= self.max_retries:
                    raise RuntimeError(f"Career reranker failed with HTTP {exc.code}") from exc
            except urllib.error.URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError("Career reranker failed due to a network error") from exc
            self.sleep(min(4.0, 0.5 * (2**attempt)))
        if response_payload is None:
            raise RuntimeError("Career reranker ended without a response")
        results = response_payload.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("Reranker response requires non-empty results")
        ranked: list[tuple[InterviewKnowledgeChunk, InterviewKnowledgeSource, float]] = []
        seen: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                raise ValueError("Reranker result must be an object")
            index = int(item.get("index", -1))
            score = float(item.get("relevance_score", item.get("score", -1)))
            if index < 0 or index >= len(rows) or index in seen or not 0 <= score <= 1:
                raise ValueError("Reranker returned an invalid index or score")
            seen.add(index)
            ranked.append((rows[index][0], rows[index][1], score))
        ranked.extend(row for index, row in enumerate(rows) if index not in seen)
        usage = response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
        output_hash = hashlib.sha256(
            json.dumps([(row[0].chunk_id, row[2]) for row in ranked], sort_keys=True).encode()
        ).hexdigest()
        api_id = "api:" + hashlib.sha256(self.base_url.casefold().encode()).hexdigest()[:16]
        call = InterviewRerankerCall(
            provider=self.provider,
            model=self.model,
            api_id=api_id,
            input_hash=hashlib.sha256(encoded).hexdigest(),
            output_hash=output_hash,
            prompt_tokens=prompt_tokens,
            latency_seconds=time.monotonic() - started,
            cost_usd=prompt_tokens * self.input_price_per_million / 1_000_000,
            attempts=attempts,
            created_at=time.time(),
        )
        self.last_call = call
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(call.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return ranked


@dataclass(frozen=True, slots=True)
class InterviewRetrievalIndexManifest:
    index_version: str
    backend: str
    embedding_backend: str
    reranker: str
    parameters: dict[str, Any]
    corpus_versions: dict[str, str]
    corpus_hash: str
    random_seed: int
    created_at: float
    schema_version: str = KNOWLEDGE_INDEX_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "index_version": self.index_version,
            "backend": self.backend,
            "embedding_backend": self.embedding_backend,
            "reranker": self.reranker,
            "parameters": self.parameters,
            "corpus_versions": self.corpus_versions,
            "corpus_hash": self.corpus_hash,
            "random_seed": self.random_seed,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class InterviewRetrievalHit:
    chunk: InterviewKnowledgeChunk
    source: InterviewKnowledgeSource
    bm25_score: float
    vector_score: float | None
    combined_score: float
    rank: int
    context_chunks: tuple[InterviewKnowledgeChunk, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk": self.chunk.to_dict(),
            "source": self.source.to_dict(),
            "bm25_score": round(self.bm25_score, 6),
            "vector_score": round(self.vector_score, 6) if self.vector_score is not None else None,
            "combined_score": round(self.combined_score, 6),
            "context_chunks": [chunk.to_dict() for chunk in self.context_chunks],
        }


@dataclass(frozen=True, slots=True)
class InterviewRetrievalResult:
    query: str
    hits: tuple[InterviewRetrievalHit, ...]
    manifest: InterviewRetrievalIndexManifest
    degraded: bool
    degradation_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "hits": [hit.to_dict() for hit in self.hits],
            "manifest": self.manifest.to_dict(),
            "degraded": self.degraded,
            "degradation_reason": self.degradation_reason,
        }


def _bm25_scores(query: tuple[str, ...], documents: list[tuple[str, ...]]) -> list[float]:
    if not documents:
        return []
    average_length = sum(len(document) for document in documents) / len(documents) or 1.0
    frequencies = [Counter(document) for document in documents]
    document_frequency = {term: sum(1 for frequency in frequencies if term in frequency) for term in set(query)}
    scores = []
    for document, frequency in zip(documents, frequencies, strict=True):
        score = 0.0
        for term in set(query):
            count = frequency.get(term, 0)
            if not count:
                continue
            appearances = document_frequency[term]
            inverse_frequency = math.log(1 + (len(documents) - appearances + 0.5) / (appearances + 0.5))
            denominator = count + 1.5 * (0.25 + 0.75 * len(document) / average_length)
            score += inverse_frequency * count * 2.5 / denominator
        scores.append(score)
    maximum = max(scores, default=0.0)
    return [score / maximum if maximum else 0.0 for score in scores]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have one dimension")
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))


class HybridInterviewKnowledgeRetriever:
    """Traceable BM25/vector retrieval with weighted or RRF fusion."""

    def __init__(
        self,
        library: InterviewKnowledgeLibrary,
        *,
        embedding_provider: InterviewEmbeddingProvider | None = None,
        reranker: InterviewReranker | None = None,
        variant: str = "hybrid_rerank",
        bm25_weight: float = 0.55,
        vector_weight: float = 0.45,
        rrf_k: int = 60,
        neighbor_window: int = 1,
        rerank_candidate_limit: int = 30,
        random_seed: int = 0,
    ) -> None:
        variants = {"bm25", "vector", "hybrid", "hybrid_rerank", "hybrid_rrf", "hybrid_rrf_rerank"}
        if variant not in variants:
            raise ValueError(f"variant must be one of: {', '.join(sorted(variants))}")
        if variant != "bm25" and embedding_provider is None:
            raise ValueError(f"{variant} retrieval requires an embedding provider")
        if variant.endswith("_rerank") and reranker is None:
            raise ValueError(f"{variant} retrieval requires a reranker")
        if bm25_weight < 0 or vector_weight < 0 or bm25_weight + vector_weight <= 0:
            raise ValueError("Retrieval weights must be non-negative and not both zero")
        if rrf_k < 1 or not 0 <= neighbor_window <= 3 or rerank_candidate_limit < 1:
            raise ValueError("rrf_k, neighbor_window, and rerank_candidate_limit are invalid")
        self.library = library
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.variant = variant
        self.bm25_weight = float(bm25_weight)
        self.vector_weight = float(vector_weight)
        self.rrf_k = int(rrf_k)
        self.neighbor_window = int(neighbor_window)
        self.rerank_candidate_limit = int(rerank_candidate_limit)
        self.random_seed = int(random_seed)
        self.name = f"interview-{variant}-v1"

    def retrieve(
        self,
        query: str,
        *,
        filters: InterviewRetrievalFilters,
        top_k: int = 5,
    ) -> InterviewRetrievalResult:
        query = _required_text(query, "query")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        candidates: list[tuple[InterviewKnowledgeChunk, InterviewKnowledgeSource]] = []
        corpus_versions: dict[str, str] = {}
        excluded_changed = False
        for source in self.library.list_sources(tenant_id=filters.tenant_id):
            if filters.corpus_ids and source.corpus_id not in filters.corpus_ids:
                continue
            if source.authority_level < filters.minimum_authority or source.difficulty > filters.maximum_difficulty:
                continue
            if source.security_flags and not filters.include_security_flagged:
                continue
            if filters.domains and not set(filters.domains).intersection(source.domains):
                continue
            if filters.subdomains and not set(filters.subdomains).intersection(source.subdomains):
                continue
            if filters.topics and not set(filters.topics).intersection(source.topics):
                continue
            if not self.library.verify_source_integrity(source.source_id):
                excluded_changed = True
                continue
            corpus = self.library.corpus_store.get(source.corpus_id)
            corpus_versions[corpus.corpus_id] = corpus.version
            candidates.extend((chunk, source) for chunk in self.library.source_chunks(source))
        manifest = self._manifest(corpus_versions)
        if not candidates:
            reason = "source_content_changed" if excluded_changed else "no_reliable_source"
            return InterviewRetrievalResult(query, (), manifest, True, reason)
        document_tokens = [tokenize_knowledge(chunk.text) for chunk, _source in candidates]
        bm25 = _bm25_scores(tokenize_knowledge(query), document_tokens)
        vectors: list[float] | None = None
        if self.embedding_provider is not None and self.variant != "bm25":
            embedded = self.embedding_provider.embed([query, *(chunk.text for chunk, _source in candidates)])
            if len(embedded) != len(candidates) + 1:
                raise ValueError("Interview embedding provider returned an unexpected vector count")
            vectors = [_cosine(embedded[0], vector) for vector in embedded[1:]]
        if self.variant.startswith("hybrid_rrf"):
            assert vectors is not None
            rows = self._rrf_rows(candidates, bm25, vectors)
        else:
            rows = []
            for index, (chunk, source) in enumerate(candidates):
                if self.variant == "bm25":
                    score = bm25[index]
                elif self.variant == "vector":
                    assert vectors is not None
                    score = vectors[index]
                else:
                    assert vectors is not None
                    score = (self.bm25_weight * bm25[index] + self.vector_weight * vectors[index]) / (
                        self.bm25_weight + self.vector_weight
                    )
                rows.append((chunk, source, score))
        rows.sort(key=lambda row: (-row[2], row[0].chunk_id))
        if self.variant.endswith("_rerank"):
            assert self.reranker is not None
            rows = self.reranker.rerank(query, rows[: self.rerank_candidate_limit])
        original_by_id = {chunk.chunk_id: index for index, (chunk, _source) in enumerate(candidates)}
        candidate_by_id = {chunk.chunk_id: chunk for chunk, _source in candidates}
        hits = tuple(
            InterviewRetrievalHit(
                chunk=chunk,
                source=source,
                bm25_score=bm25[original_by_id[chunk.chunk_id]],
                vector_score=vectors[original_by_id[chunk.chunk_id]] if vectors is not None else None,
                combined_score=score,
                rank=rank,
                context_chunks=self._neighbor_context(chunk, candidate_by_id),
            )
            for rank, (chunk, source, score) in enumerate(rows[:top_k], start=1)
            if score > 0
        )
        if not hits:
            return InterviewRetrievalResult(query, (), manifest, True, "no_relevant_source")
        return InterviewRetrievalResult(query, hits, manifest, False)

    def _rrf_rows(
        self,
        candidates: list[tuple[InterviewKnowledgeChunk, InterviewKnowledgeSource]],
        bm25: list[float],
        vectors: list[float],
    ) -> list[tuple[InterviewKnowledgeChunk, InterviewKnowledgeSource, float]]:
        scores: dict[int, float] = {}
        channels = ((bm25, self.bm25_weight), (vectors, self.vector_weight))
        for values, weight in channels:
            ranked = sorted(
                (index for index, value in enumerate(values) if value > 0),
                key=lambda index: (-values[index], candidates[index][0].chunk_id),
            )
            for rank, index in enumerate(ranked, start=1):
                scores[index] = scores.get(index, 0.0) + weight / (self.rrf_k + rank)
        if not scores:
            return []
        maximum = max(scores.values())
        return [
            (candidates[index][0], candidates[index][1], score / maximum)
            for index, score in scores.items()
        ]

    def _neighbor_context(
        self,
        anchor: InterviewKnowledgeChunk,
        candidate_by_id: dict[str, InterviewKnowledgeChunk],
    ) -> tuple[InterviewKnowledgeChunk, ...]:
        if self.neighbor_window == 0:
            return ()
        collected: list[InterviewKnowledgeChunk] = []
        seen = {anchor.chunk_id}
        for direction in ("prev_chunk_id", "next_chunk_id"):
            current = anchor
            chain: list[InterviewKnowledgeChunk] = []
            for _ in range(self.neighbor_window):
                neighbor_id = str(current.locator.get(direction) or "")
                neighbor = candidate_by_id.get(neighbor_id)
                if neighbor is None or neighbor.source_id != anchor.source_id or neighbor.chunk_id in seen:
                    break
                anchor_chapter = str(anchor.locator.get("chapter") or "").strip()
                neighbor_chapter = str(neighbor.locator.get("chapter") or "").strip()
                if anchor_chapter and neighbor_chapter and anchor_chapter != neighbor_chapter:
                    break
                seen.add(neighbor.chunk_id)
                chain.append(neighbor)
                current = neighbor
            if direction == "prev_chunk_id":
                chain.reverse()
            collected.extend(chain)
        return tuple(collected)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        topic: str | None = None,
        tenant_id: str = "default",
        corpus_id: str | None = None,
    ) -> tuple[InterviewKnowledgeChunk, ...]:
        result = self.retrieve(
            query,
            filters=InterviewRetrievalFilters(
                tenant_id=tenant_id,
                topics=(topic,) if topic else (),
                corpus_ids=(corpus_id,) if corpus_id else (),
            ),
            top_k=top_k,
        )
        return tuple(hit.chunk for hit in result.hits)

    def _manifest(self, corpus_versions: dict[str, str]) -> InterviewRetrievalIndexManifest:
        parameters = {
            "variant": self.variant,
            "bm25_weight": self.bm25_weight,
            "vector_weight": self.vector_weight,
            "rrf_k": self.rrf_k,
            "neighbor_window": self.neighbor_window,
            "rerank_candidate_limit": self.rerank_candidate_limit,
            "chunker_version": KNOWLEDGE_CHUNKER_VERSION,
        }
        corpus_hash = hashlib.sha256(
            json.dumps(corpus_versions, sort_keys=True).encode("utf-8")
        ).hexdigest()
        identity = hashlib.sha256(
            json.dumps(
                {
                    "backend": self.name,
                    "embedding": getattr(self.embedding_provider, "name", ""),
                    "reranker": getattr(self.reranker, "name", ""),
                    "parameters": parameters,
                    "corpus_hash": corpus_hash,
                    "seed": self.random_seed,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:20]
        manifest = InterviewRetrievalIndexManifest(
            index_version=f"interview-index-{identity}",
            backend=self.name,
            embedding_backend=str(getattr(self.embedding_provider, "name", "")),
            reranker=str(getattr(self.reranker, "name", "")),
            parameters=parameters,
            corpus_versions=dict(corpus_versions),
            corpus_hash=corpus_hash,
            random_seed=self.random_seed,
            created_at=time.time(),
        )
        _atomic_json(self.library.root / "indexes" / f"{manifest.index_version}.json", manifest.to_dict())
        return manifest

