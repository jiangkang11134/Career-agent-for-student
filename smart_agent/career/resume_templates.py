"""Format-only resume template extraction and normalization."""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from smart_agent.career.models import (
    RESUME_SECTION_ORDER,
    ResumeSource,
    ResumeTemplate,
    SourceReference,
    SourceType,
    version_ref,
)

DEFAULT_LAYOUT_SPEC: dict[str, Any] = {
    "version": 1,
    "page_size": "A4",
    "margins_mm": {"top": 12.0, "right": 15.0, "bottom": 13.0, "left": 15.0},
    "font_family": "Microsoft YaHei",
    "font_sizes_pt": {"body": 10.5, "title": 22.0, "section": 12.5},
    "colors": {"accent": "#176b5b", "text": "#202524", "muted": "#5d6664", "rule": "#cbd3d1"},
    "spacing_mm": {"header": 4.5, "section": 4.2, "item": 1.25},
    "columns": 1,
    "template_variant": "professional",
    "section_order": list(RESUME_SECTION_ORDER),
    "section_style": "rule",
    "experience_layout": "timeline",
    "skills_layout": "groups",
    "header_layout": "left",
}

_SUPPORTED_FONTS = {
    "arial": "Arial",
    "calibri": "Calibri",
    "microsoft yahei": "Microsoft YaHei",
    "微软雅黑": "Microsoft YaHei",
    "simsun": "SimSun",
    "宋体": "SimSun",
    "noto sans cjk sc": "Noto Sans CJK SC",
}
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _number(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return round(min(max(number, minimum), maximum), 2)


def sanitize_layout_spec(value: Any) -> dict[str, Any]:
    """Return a closed, CSS-safe layout schema and discard every unknown field."""
    source = value if isinstance(value, dict) else {}
    defaults = DEFAULT_LAYOUT_SPEC
    margins = source.get("margins_mm") if isinstance(source.get("margins_mm"), dict) else {}
    font_sizes = source.get("font_sizes_pt") if isinstance(source.get("font_sizes_pt"), dict) else {}
    colors = source.get("colors") if isinstance(source.get("colors"), dict) else {}
    spacing = source.get("spacing_mm") if isinstance(source.get("spacing_mm"), dict) else {}
    font_key = str(source.get("font_family") or "").strip().casefold()

    normalized_colors = {}
    for key, default in defaults["colors"].items():
        candidate = str(colors.get(key) or "").strip()
        normalized_colors[key] = candidate.lower() if _HEX_COLOR.fullmatch(candidate) else default

    variant = str(source.get("template_variant") or defaults["template_variant"]).strip().casefold()
    if variant not in {"professional", "template1", "template2", "template3"}:
        variant = defaults["template_variant"]
    normalized_order = list(RESUME_SECTION_ORDER)
    section_style = str(source.get("section_style") or defaults["section_style"]).strip().casefold()
    if section_style not in {"rule", "bar", "label"}:
        section_style = defaults["section_style"]
    experience_layout = str(source.get("experience_layout") or defaults["experience_layout"]).strip().casefold()
    if experience_layout not in {"timeline", "compact", "card"}:
        experience_layout = defaults["experience_layout"]
    skills_layout = str(source.get("skills_layout") or defaults["skills_layout"]).strip().casefold()
    if skills_layout not in {"groups", "inline", "compact"}:
        skills_layout = defaults["skills_layout"]
    header_layout = str(source.get("header_layout") or defaults["header_layout"]).strip().casefold()
    if header_layout not in {"left", "center"}:
        header_layout = defaults["header_layout"]
    normalized = {
        "version": 1,
        "page_size": str(source.get("page_size") or "A4").upper()
        if str(source.get("page_size") or "A4").upper() in {"A4", "LETTER"}
        else "A4",
        "margins_mm": {
            key: _number(margins.get(key), default, 5.0, 35.0)
            for key, default in defaults["margins_mm"].items()
        },
        "font_family": _SUPPORTED_FONTS.get(font_key, defaults["font_family"]),
        "font_sizes_pt": {
            "body": _number(font_sizes.get("body"), defaults["font_sizes_pt"]["body"], 8.0, 14.0),
            "title": _number(font_sizes.get("title"), defaults["font_sizes_pt"]["title"], 15.0, 30.0),
            "section": _number(font_sizes.get("section"), defaults["font_sizes_pt"]["section"], 10.0, 18.0),
        },
        "colors": normalized_colors,
        "spacing_mm": {
            "header": _number(spacing.get("header"), defaults["spacing_mm"]["header"], 1.0, 12.0),
            "section": _number(spacing.get("section"), defaults["spacing_mm"]["section"], 1.0, 12.0),
            "item": _number(spacing.get("item"), defaults["spacing_mm"]["item"], 0.5, 5.0),
        },
        "columns": 2 if int(source.get("columns") or 1) == 2 else 1,
    }
    # Keep legacy records hash-compatible. New metadata is persisted only when
    # a parser or caller explicitly supplied at least one layout field.
    if any(
        key in source
        for key in (
            "template_variant",
            "section_order",
            "section_style",
            "experience_layout",
            "skills_layout",
            "header_layout",
        )
    ):
        normalized.update(
            {
                "template_variant": variant,
                "section_order": normalized_order,
                "section_style": section_style,
                "experience_layout": experience_layout,
                "skills_layout": skills_layout,
                "header_layout": header_layout,
            }
        )
    return normalized


def extract_layout_spec(content: bytes, filename: str) -> dict[str, Any]:
    """Extract formatting metadata without extracting or persisting document text."""
    suffix = Path(filename).suffix.casefold()
    if suffix == ".docx":
        return sanitize_layout_spec(_extract_docx_layout(content))
    if suffix == ".pdf":
        return sanitize_layout_spec(_extract_pdf_layout(content, filename))
    raise ValueError("Resume templates must be DOCX or PDF files.")


def create_uploaded_template(*, name: str, filename: str, content: bytes) -> ResumeTemplate:
    """Create a template record without retaining or indexing the file body."""
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValueError("Resume template name is required.")
    if not content:
        raise ValueError("Resume template file is empty.")
    layout_spec = extract_layout_spec(content, filename)
    digest = hashlib.sha256(content).hexdigest()
    source = SourceReference.create(
        source_id=f"resume-template-file:{digest[:20]}",
        source_type=SourceType.FILE,
        locator={"original_name": Path(filename).name, "content_access": "format_only"},
        raw_content=content,
        excerpt="",
    )
    identity = hashlib.sha256(f"{digest}:{clean_name}".encode("utf-8")).hexdigest()
    return ResumeTemplate(
        template_id=f"resume-template-{identity[:20]}",
        name=clean_name,
        source_kind="template_upload",
        original_name=Path(filename).name,
        layout_spec=layout_spec,
        source_refs=(source,),
    )


def create_resume_source_template(*, source: ResumeSource, content: bytes) -> ResumeTemplate:
    """Derive only layout metadata from an original resume's immutable file revision."""
    layout_spec = extract_layout_spec(content, source.original_name)
    source_hash = hashlib.sha256(content).hexdigest()
    identity = hashlib.sha256(
        f"{source_hash}:{source.resume_source_id}:{source.revision}".encode("utf-8")
    ).hexdigest()
    format_source = SourceReference(
        source_id=source.resume_source_id,
        source_type=SourceType.RESUME,
        locator={
            "resume_source_id": source.resume_source_id,
            "revision": source.revision,
            "content_access": "format_only",
        },
        content_hash=source_hash,
        excerpt="",
    )
    return ResumeTemplate(
        template_id=f"resume-template-source-{identity[:20]}",
        name=f"原始简历版式：{Path(source.original_name).stem}",
        source_kind="resume_source",
        original_name=source.original_name,
        layout_spec=layout_spec,
        resume_source_ref=version_ref(source),
        source_refs=(format_source,),
    )


def _twips_mm(value: str | None, default: float) -> float:
    return _number(float(value) * 25.4 / 1440 if value else default, default, 5.0, 35.0)


def _extract_docx_layout(content: bytes) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            document = ElementTree.fromstring(archive.read("word/document.xml"))
            styles = ElementTree.fromstring(archive.read("word/styles.xml"))
    except (KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise ValueError("Invalid DOCX resume template.") from exc

    spec: dict[str, Any] = {}
    section = document.find(".//w:sectPr", _WORD_NS)
    if section is not None:
        page = section.find("w:pgSz", _WORD_NS)
        margin = section.find("w:pgMar", _WORD_NS)
        columns = section.find("w:cols", _WORD_NS)
        attr = f"{{{_WORD_NS['w']}}}"
        if page is not None:
            width = int(page.get(f"{attr}w", "11906"))
            height = int(page.get(f"{attr}h", "16838"))
            spec["page_size"] = "LETTER" if abs(width - 12240) + abs(height - 15840) < 700 else "A4"
        if margin is not None:
            spec["margins_mm"] = {
                "top": _twips_mm(margin.get(f"{attr}top"), 12.0),
                "right": _twips_mm(margin.get(f"{attr}right"), 15.0),
                "bottom": _twips_mm(margin.get(f"{attr}bottom"), 13.0),
                "left": _twips_mm(margin.get(f"{attr}left"), 15.0),
            }
        if columns is not None:
            spec["columns"] = 2 if int(columns.get(f"{attr}num", "1")) == 2 else 1

    attr = f"{{{_WORD_NS['w']}}}"
    normal = next(
        (
            style for style in styles.findall("w:style", _WORD_NS)
            if style.get(f"{attr}styleId", "").casefold() == "normal"
        ),
        None,
    )
    if normal is not None:
        fonts = normal.find(".//w:rFonts", _WORD_NS)
        size = normal.find(".//w:sz", _WORD_NS)
        if fonts is not None:
            spec["font_family"] = fonts.get(f"{attr}eastAsia") or fonts.get(f"{attr}ascii") or ""
        if size is not None:
            spec["font_sizes_pt"] = {"body": float(size.get(f"{attr}val", "21")) / 2}
    return spec


def _extract_pdf_layout(content: bytes, filename: str = "") -> dict[str, Any]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        if not reader.pages:
            raise ValueError("PDF resume template has no pages.")
        box = reader.pages[0].mediabox
        width, height = float(box.width), float(box.height)
        name = Path(filename).stem.casefold()
        # Bundled templates are intentionally identified by their file name.
        # Their example text is never extracted or persisted.
        if "模板1" in name or "template1" in name:
            variant = {
                "template_variant": "template1",
                "section_style": "rule",
                "experience_layout": "timeline",
                "skills_layout": "groups",
                "header_layout": "left",
                "columns": 1,
                "colors": {"accent": "#f05a28", "text": "#333333", "muted": "#666666", "rule": "#f05a28"},
            }
        elif "模板2" in name or "template2" in name:
            variant = {
                "template_variant": "template2",
                "section_style": "bar",
                "experience_layout": "compact",
                "skills_layout": "inline",
                "header_layout": "left",
                "columns": 1,
                "colors": {"accent": "#367cf5", "text": "#333333", "muted": "#666666", "rule": "#bfd1f7"},
            }
        elif "模板3" in name or "template3" in name:
            variant = {
                "template_variant": "template3",
                "section_style": "label",
                "experience_layout": "card",
                "skills_layout": "compact",
                "header_layout": "center",
                "columns": 1,
                "colors": {"accent": "#087ff5", "text": "#303030", "muted": "#555555", "rule": "#b9d8ff"},
            }
        else:
            variant = {}
        return {"page_size": "LETTER" if abs(width - 612) + abs(height - 792) < 30 else "A4", **variant}
    except ImportError:
        return {"page_size": "A4"}
    except Exception as exc:
        raise ValueError("Invalid PDF resume template.") from exc
