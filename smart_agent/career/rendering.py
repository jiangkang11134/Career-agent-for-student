"""Backend HTML/CSS and Chromium PDF rendering for verified resumes."""

from __future__ import annotations

import html
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_agent.career.models import (
    RESUME_SECTION_LABELS,
    RESUME_SECTION_ORDER,
    ResumeClaim,
    ResumeDocument,
    ResumeDocumentStatus,
    VerificationReport,
    VerificationStatus,
    normalize_resume_sections,
)
from smart_agent.career.resume_templates import DEFAULT_LAYOUT_SPEC, sanitize_layout_spec
from smart_agent.career.store import CareerStore


@dataclass(frozen=True, slots=True)
class ResumeTheme:
    theme_id: str
    accent: str
    text: str
    muted: str
    rule: str
    font_family: str
    page_size: str = "A4"
    margins_mm: dict[str, float] | None = None
    body_size_pt: float = 10.5
    title_size_pt: float = 22.0
    section_size_pt: float = 12.5
    header_spacing_mm: float = 4.5
    section_spacing_mm: float = 4.2
    item_spacing_mm: float = 1.25
    columns: int = 1
    template_variant: str = "professional"
    section_order: tuple[str, ...] = ()
    section_style: str = "rule"
    experience_layout: str = "timeline"
    skills_layout: str = "groups"
    header_layout: str = "left"

    @classmethod
    def from_layout_spec(cls, layout_spec: dict[str, Any], *, theme_id: str = "template") -> "ResumeTheme":
        spec = sanitize_layout_spec(layout_spec)
        return cls(
            theme_id=theme_id,
            accent=spec["colors"]["accent"],
            text=spec["colors"]["text"],
            muted=spec["colors"]["muted"],
            rule=spec["colors"]["rule"],
            font_family=(
                f'"{spec["font_family"]}", "Microsoft YaHei", "SimSun", Arial, sans-serif'
            ),
            page_size=spec["page_size"],
            margins_mm=spec["margins_mm"],
            body_size_pt=spec["font_sizes_pt"]["body"],
            title_size_pt=spec["font_sizes_pt"]["title"],
            section_size_pt=spec["font_sizes_pt"]["section"],
            header_spacing_mm=spec["spacing_mm"]["header"],
            section_spacing_mm=spec["spacing_mm"]["section"],
            item_spacing_mm=spec["spacing_mm"]["item"],
            columns=spec["columns"],
            template_variant=spec.get("template_variant", DEFAULT_LAYOUT_SPEC["template_variant"]),
            section_order=tuple(spec.get("section_order", DEFAULT_LAYOUT_SPEC["section_order"])),
            section_style=spec.get("section_style", DEFAULT_LAYOUT_SPEC["section_style"]),
            experience_layout=spec.get("experience_layout", DEFAULT_LAYOUT_SPEC["experience_layout"]),
            skills_layout=spec.get("skills_layout", DEFAULT_LAYOUT_SPEC["skills_layout"]),
            header_layout=spec.get("header_layout", DEFAULT_LAYOUT_SPEC["header_layout"]),
        )


THEMES = {
    "professional": ResumeTheme(
        theme_id="professional",
        accent="#176b5b",
        text="#202524",
        muted="#5d6664",
        rule="#cbd3d1",
        font_family='"Microsoft YaHei", "SimSun", "Noto Sans CJK SC", Arial, sans-serif',
    )
}


@dataclass(frozen=True, slots=True)
class ResumeRenderAudit:
    page_count: int
    max_pages: int
    text_layer_ok: bool
    horizontal_overflow: bool
    font_ready: bool
    computed_font_family: str
    extracted_text_characters: int

    @property
    def passed(self) -> bool:
        return (
            self.page_count <= self.max_pages
            and self.text_layer_ok
            and not self.horizontal_overflow
            and self.font_ready
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "page_count": self.page_count,
            "max_pages": self.max_pages,
            "text_layer_ok": self.text_layer_ok,
            "horizontal_overflow": self.horizontal_overflow,
            "font_ready": self.font_ready,
            "computed_font_family": self.computed_font_family,
            "extracted_text_characters": self.extracted_text_characters,
        }


@dataclass(frozen=True, slots=True)
class ResumeRenderResult:
    resume_document_id: str
    resume_revision: int
    html_path: Path
    pdf_path: Path
    trace_path: Path
    audit: ResumeRenderAudit

    def to_dict(self) -> dict[str, Any]:
        return {
            "resume_document_id": self.resume_document_id,
            "resume_revision": self.resume_revision,
            "html_path": str(self.html_path),
            "pdf_path": str(self.pdf_path),
            "trace_path": str(self.trace_path),
            "audit": self.audit.to_dict(),
        }


class ResumeHTMLRenderer:
    """Render the content protocol without owning planning or verification decisions."""

    def __init__(self, theme: ResumeTheme) -> None:
        self.theme = theme

    def render(self, document: ResumeDocument, claims: tuple[ResumeClaim, ...]) -> str:
        if document.header is None:
            raise ValueError("ResumeDocument requires a structured header before rendering.")
        by_id = {claim.claim_id: claim for claim in claims}
        contacts: list[str] = []
        if document.header.phone:
            contacts.append(f"电话：{document.header.phone}")
        if document.header.email:
            contacts.append(f"邮箱：{document.header.email}")
        if document.header.location:
            contacts.append(f"所在地：{document.header.location}")
        for link in document.header.links:
            contacts.append(f"作品链接：{link}")
        contact_html = "<span aria-hidden=\"true\"> | </span>".join(
            f"<span>{html.escape(value)}</span>" for value in contacts
        )
        headline = (
            f'<p class="headline">{html.escape(document.header.headline)}</p>'
            if document.header.headline
            else ""
        )
        sections: list[str] = []
        normalized_sections = normalize_resume_sections(document.sections)
        for section in RESUME_SECTION_ORDER:
            claim_ids = normalized_sections[section]
            if not claim_ids:
                continue
            label = RESUME_SECTION_LABELS[section]
            content = self._render_section(section, claim_ids, by_id)
            sections.append(
                f'<section class="resume-section section-{html.escape(section.casefold())} section-{self.theme.section_style}" '
                f'data-resume-section="{html.escape(section)}">'
                f"<h2><span>{html.escape(label)}</span></h2>{content}</section>"
            )
        margins = self.theme.margins_mm or DEFAULT_LAYOUT_SPEC["margins_mm"]
        page_margin = f'{margins["top"]}mm {margins["right"]}mm {margins["bottom"]}mm {margins["left"]}mm'
        columns_css = f'column-count: {self.theme.columns}; column-gap: 8mm;' if self.theme.columns > 1 else ''
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex,nofollow">
  <title>{html.escape(document.header.full_name)} - 简历</title>
  <style>
    @page {{ size: {self.theme.page_size}; margin: {page_margin}; }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; background: #ffffff; }}
    body {{ color: {self.theme.text}; font-family: {self.theme.font_family}; font-size: {self.theme.body_size_pt}pt; line-height: 1.46; letter-spacing: 0; }}
    main {{ width: 100%; max-width: 180mm; margin: 0 auto; }}
    header {{ padding-bottom: {self.theme.header_spacing_mm}mm; border-bottom: 1.5pt solid {self.theme.accent}; }}
    .header-center {{ text-align: center; }}
    .header-center .contact {{ justify-content: center; }}
    h1 {{ margin: 0; color: {self.theme.text}; font-size: {self.theme.title_size_pt}pt; line-height: 1.15; font-weight: 700; letter-spacing: 0; overflow-wrap: anywhere; }}
    .headline {{ margin: 1.5mm 0 0; color: {self.theme.accent}; font-size: 11pt; font-weight: 600; }}
    .contact {{ display: flex; flex-wrap: wrap; gap: 1mm 2mm; margin-top: 2mm; color: {self.theme.muted}; font-size: 9pt; overflow-wrap: anywhere; }}
    .sections {{ {columns_css} }}
    section {{ break-inside: avoid; margin-top: {self.theme.section_spacing_mm}mm; }}
    h2 {{ margin: 0 0 2mm; padding-bottom: 0.9mm; color: {self.theme.accent}; font-size: {self.theme.section_size_pt}pt; line-height: 1.25; font-weight: 700; letter-spacing: 0; }}
    .section-rule h2 {{ border-bottom: 0.6pt solid {self.theme.rule}; }}
    .section-bar h2 {{ border-bottom: 0.6pt solid {self.theme.rule}; }}
    .section-bar h2 span {{ display: inline-block; padding: 1mm 8mm 1mm 2.5mm; background: {self.theme.accent}; color: #ffffff; }}
    .section-label h2 {{ border-bottom: 1.2pt solid {self.theme.accent}; }}
    .entry {{ break-inside: avoid; margin: 0 0 {self.theme.item_spacing_mm}mm; overflow-wrap: anywhere; }}
    .entry-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 4mm; font-weight: 700; }}
    .timeline-head {{ display: grid; grid-template-columns: 26mm minmax(0, 1fr) minmax(42mm, 58mm); align-items: baseline; gap: 3mm; }}
    .timeline-head.with-department {{ grid-template-columns: 26mm minmax(0, 1fr) minmax(24mm, 34mm) minmax(36mm, 50mm); }}
    .timeline-head .entry-meta {{ justify-self: start; }}
    .timeline-head .entry-subtitle-inline {{ justify-self: end; min-width: 0; max-width: 58mm; color: {self.theme.text}; font-weight: 700; text-align: right; overflow-wrap: anywhere; }}
    .timeline-head.internship-head.no-department .entry-title {{ text-align: center; }}
    .entry-title {{ color: {self.theme.text}; }}
    .entry-meta {{ color: {self.theme.muted}; font-size: 9pt; font-weight: 400; white-space: nowrap; }}
    .entry-subtitle {{ margin-top: 0.4mm; color: {self.theme.accent}; font-weight: 600; }}
    .education-head {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(42mm, 62mm) 32mm; gap: 3mm; align-items: baseline; font-weight: 700; }}
    .education-major {{ text-align: center; }}
    .education-date {{ color: {self.theme.muted}; font-size: 9pt; font-weight: 400; text-align: right; white-space: nowrap; }}
    .education-courses {{ margin-top: 0.7mm; line-height: 1.42; }}
    .entry-body {{ margin-top: 0.7mm; line-height: 1.42; }}
    .entry-body .detail-inline + .detail-inline::before {{ content: "；"; color: {self.theme.muted}; }}
    .detail-block {{ display: block; margin-top: 0.35mm; }}
    .detail-label {{ color: {self.theme.muted}; font-weight: 600; }}
    .detail-value {{ color: {self.theme.text}; }}
    .project-design-items {{ margin-top: 0.25mm; }}
    .project-design-item {{ display: block; margin-left: 4mm; }}
    .skill-body {{ white-space: pre-wrap; }}
    .skill-group {{ break-inside: avoid; margin: 0 0 {self.theme.item_spacing_mm}mm; }}
    .skill-group-title {{ color: #000000; font-size: 1.08em; font-weight: 800; line-height: 1.3; }}
    .skill-group-content {{ color: {self.theme.text}; white-space: pre-wrap; margin-top: 0.45mm; }}
    .skill-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); column-gap: 7mm; row-gap: 1mm; }}
    .skill-line {{ break-inside: avoid; }}
    .skill-line .detail-label {{ color: {self.theme.accent}; }}
    .plain-list {{ margin: 0; padding: 0; list-style: none; }}
    .plain-list .entry-body {{ margin-top: 0; }}
    .template-template2 .entry {{ margin-bottom: 0.7mm; }}
    .template-template3 .entry {{ padding-left: 3mm; border-left: 1.2pt solid {self.theme.rule}; }}
    .template-template3 .entry-head {{ display: block; }}
    .template-template3 .entry-meta {{ display: block; margin-top: 0.3mm; }}
  </style>
</head>
<body>
  <main class="template-{html.escape(self.theme.template_variant)}">
    <header class="header-{html.escape(self.theme.header_layout)}">
      <h1>{html.escape(document.header.full_name)}</h1>
      {headline}
      <div class="contact">{contact_html}</div>
    </header>
    <div class="sections">{''.join(sections)}</div>
  </main>
</body>
</html>
"""

    @staticmethod
    def _parts(text: str) -> list[tuple[str, str]]:
        text = re.sub(r"^\s*已确认内容\s*[:：]\s*", "", text)
        text = re.sub(
            r"^\s*(教育背景|教育经历|实习经历|项目经历|技能/其他|个人技能/其他|技能证书|科研成果|科研经历|获奖经历|个人奖项)\s*[:：]\s*",
            "",
            text,
        )
        parts = []
        for raw in re.split(r"[；;]", text):
            value = " ".join(raw.split()).strip()
            if not value:
                continue
            if "：" in value:
                label, content = value.split("：", 1)
            elif ":" in value:
                label, content = value.split(":", 1)
            else:
                label, content = "", value
            parts.append((label.strip(), content.strip()))
        return parts

    @classmethod
    def _structured_parts(cls, text: str, labels: tuple[str, ...]) -> list[tuple[str, str]]:
        text = re.sub(r"^\s*已确认内容\s*[:：]\s*", "", text)
        escaped = "|".join(re.escape(label) for label in labels)
        matches = list(re.finditer(rf"(?<![\w\u4e00-\u9fff])({escaped})\s*[:：]", text))
        if not matches:
            return cls._parts(text)
        parts: list[tuple[str, str]] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            value = text[match.end():end].strip().strip("；;").strip()
            if value:
                parts.append((match.group(1), value))
        return parts

    @classmethod
    def _section_parts(cls, section: str, text: str) -> list[tuple[str, str]]:
        normalized = section.casefold()
        if normalized in {"project", "projects"}:
            return cls._structured_parts(text, ("项目名称", "项目简述", "技术栈", "项目设计"))
        if normalized in {"internship", "internships"}:
            return cls._structured_parts(
                text,
                ("实习公司名称", "公司", "项目部门", "实习岗位", "岗位", "实习简述", "实习内容", "实习日期"),
            )
        if normalized in {"education", "educations"}:
            return cls._structured_parts(
                text,
                ("学校名", "学校", "专业", "学科", "学位层次", "学历", "主修课程", "绩点", "开始时间", "结束时间", "教育日期"),
            )
        return cls._parts(text)

    @staticmethod
    def _part_map(parts: list[tuple[str, str]]) -> dict[str, str]:
        values: dict[str, str] = {}
        for label, value in parts:
            if label and value and label not in values:
                values[label] = value
        return values

    @staticmethod
    def _compact_month(value: str) -> str:
        text = str(value or "").strip()
        match = re.fullmatch(r"(\d{4})[-./年](\d{1,2})(?:月)?", text)
        if match:
            return f"{match.group(1)}.{int(match.group(2))}"
        return text.replace("-", ".")

    def _education_entry_html(self, claim_id: str, parts: list[tuple[str, str]]) -> str:
        values = self._education_visible_values(parts)
        school = values["school"]
        major_degree = values["major_degree"]
        date_value = values["date"]
        body_line = values["body"]
        fallback = values["fallback"]
        fallback_html = self._body_html("education", fallback) if fallback else ""
        return (
            f'<article class="entry" data-claim-id="{html.escape(claim_id)}">'
            '<div class="entry-head education-head">'
            f'<span class="entry-title">{html.escape(school)}</span>'
            f'<span class="education-major">{html.escape(major_degree)}</span>'
            f'<span class="education-date">{html.escape(date_value)}</span></div>'
            f'<div class="education-courses">{html.escape(body_line)}</div>'
            f'{fallback_html}</article>'
        )

    def _education_visible_values(self, parts: list[tuple[str, str]]) -> dict[str, Any]:
        fields = self._part_map(parts)
        school = fields.get("学校名") or fields.get("学校") or ""
        major = fields.get("专业") or fields.get("学科") or ""
        degree = fields.get("学位层次") or fields.get("学历") or ""
        start = fields.get("开始时间") or ""
        end = fields.get("结束时间") or ""
        date_value = fields.get("教育日期") or (
            f"{self._compact_month(start)}-{self._compact_month(end)}" if start or end else ""
        )
        major_degree = " · ".join(value for value in (major, degree) if value)
        courses = fields.get("主修课程") or ""
        gpa = fields.get("绩点") or ""
        body_line = "；".join(
            value
            for value in (
                f"主修课程：{courses}" if courses else "",
                f"绩点：{gpa}" if gpa else "",
            )
            if value
        )
        fallback = [
            (label, value)
            for label, value in parts
            if label
            not in {
                "学校名",
                "学校",
                "专业",
                "学科",
                "学位层次",
                "学历",
                "开始时间",
                "结束时间",
                "教育日期",
                "主修课程",
                "绩点",
            }
        ]
        return {
            "school": school,
            "major_degree": major_degree,
            "date": date_value,
            "body": body_line,
            "fallback": fallback,
        }

    def _visible_text_values(self, section: str, text: str) -> list[str]:
        normalized = section.casefold()
        parts = self._section_parts(normalized, text)
        if normalized in {"education", "educations"}:
            values = self._education_visible_values(parts)
            return [
                value
                for value in (
                    values["school"],
                    values["major_degree"],
                    values["date"],
                    values["body"],
                    *(item[1] for item in values["fallback"]),
                )
                if value
            ]
        return [value for _label, value in parts if value]

    @staticmethod
    def _project_design_items(value: str) -> list[str]:
        matches = list(re.finditer(r"(?<!\d)([1-5])、", value))
        if not matches:
            return []
        items: list[str] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
            content = value[match.end():end].strip()
            if content:
                items.append(f"{match.group(1)}、{content}")
        return items

    @staticmethod
    def _detail_html(label: str, value: str, *, block: bool = False) -> str:
        css_class = "detail detail-block" if block else "detail detail-inline"
        label_html = f'<span class="detail-label">{html.escape(label + "：")}</span>' if label else ""
        return (
            f'<span class="{css_class}">{label_html}'
            f'<span class="detail-value">{html.escape(value)}</span></span>'
        )

    def _body_html(self, section: str, body_parts: list[tuple[str, str]]) -> str:
        normalized = section.casefold()
        chunks: list[str] = []
        for label, value in body_parts:
            if normalized in {"project", "projects"} and label == "项目设计":
                design_items = self._project_design_items(value)
                if design_items:
                    chunks.append(
                        '<div class="detail detail-block project-design">'
                        '<span class="detail-label">项目设计：</span></div>'
                    )
                    chunks.append(
                        '<div class="project-design-items">'
                        + "".join(
                            f'<span class="project-design-item">{html.escape(item)}</span>'
                            for item in design_items
                        )
                        + "</div>"
                    )
                    continue
            block = (
                normalized in {"project", "projects"} and label in {"项目简述", "技术栈", "项目设计"}
            ) or (
                normalized in {"internship", "internships"} and label in {"实习简述", "实习内容"}
            )
            chunks.append(self._detail_html(label, value, block=block))
        return "".join(chunks)

    @staticmethod
    def _skill_groups(text: str) -> list[tuple[str, str]]:
        normalized = re.sub(
            r"^\s*(?:技能/其他|个人技能/其他)\s*[:：]\s*",
            "",
            str(text or "").strip(),
        )
        groups: list[tuple[str, str]] = []
        for paragraph in re.split(r"\n\s*\n", normalized):
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if not lines:
                continue
            match = re.match(r"^([^：:]{1,64})\s*[:：]\s*(.*)$", lines[0])
            if match:
                title = match.group(1).strip()
                content = "\n".join(item for item in (match.group(2).strip(), *lines[1:]) if item)
                if title and content:
                    groups.append((title, content))
                    continue
            if len(lines) > 1 and re.fullmatch(r"[^：:]{1,64}\s*[:：]?", lines[0]):
                groups.append((re.sub(r"\s*[:：]\s*$", "", lines[0]).strip(), "\n".join(lines[1:])))
                continue
            groups.append(("", "\n".join(lines)))
        return groups

    def _render_section(self, section: str, claim_ids: tuple[str, ...], by_id: dict[str, ResumeClaim]) -> str:
        normalized = section.casefold()
        if normalized == "skills":
            entries = []
            for claim_id in claim_ids:
                claim = by_id[claim_id]
                text = str(claim.text or "").strip()
                if not text:
                    continue
                skill_groups = self._skill_groups(text)
                if skill_groups and any(title for title, _content in skill_groups):
                    content_html = "".join(
                        '<div class="skill-group">'
                        + (f'<div class="skill-group-title">{html.escape(title)}</div>' if title else "")
                        + f'<div class="skill-group-content">{html.escape(content)}</div>'
                        + "</div>"
                        for title, content in skill_groups
                    )
                    entries.append(
                        f'<article class="entry" data-claim-id="{html.escape(claim_id)}">{content_html}</article>'
                    )
                    continue
                entries.append(
                    f'<article class="entry" data-claim-id="{html.escape(claim_id)}">'
                    f'<div class="entry-body skill-body">{html.escape(text)}</div></article>'
                )
            return "".join(entries)
        entries = []
        for claim_id in claim_ids:
            claim = by_id[claim_id]
            parts = self._section_parts(normalized, claim.text)
            if not parts:
                parts = [("", claim.text)]
            if normalized in {"education", "educations"}:
                entries.append(self._education_entry_html(claim_id, parts))
                continue
            is_project = normalized in {"project", "projects"}
            title_label, title_value = parts[0]
            date_indexes = [
                index for index, (label, _) in enumerate(parts)
                if any(token in label for token in ("日期", "时间", "开始", "结束"))
            ]
            date_value = ""
            if date_indexes:
                start = next((parts[index][1] for index in date_indexes if "开始" in parts[index][0]), "")
                end = next((parts[index][1] for index in date_indexes if "结束" in parts[index][0]), "")
                date_value = " - ".join(value for value in (start, end) if value) if (start or end) else parts[date_indexes[0]][1]
            structured_title = title_label in {"项目名称", "项目经历", "实习公司名称", "公司", "学校名", "学校"}
            title = title_value if structured_title else title_label
            subtitle_indexes = [
                index for index, (label, _) in enumerate(parts)
                if "岗位" in label or "专业" in label or "角色" in label or "学位层次" in label
            ]
            subtitle_index = subtitle_indexes[0] if subtitle_indexes else -1
            subtitle = " · ".join(parts[index][1] for index in subtitle_indexes if parts[index][1])
            department_indexes = [
                index for index, (label, _) in enumerate(parts)
                if "部门" in label or "项目部门" in label
            ]
            department = " · ".join(parts[index][1] for index in department_indexes if parts[index][1])
            consumed = {0}
            consumed.update(date_indexes)
            consumed.update(subtitle_indexes)
            consumed.update(department_indexes)
            body_parts = [
                (label, value) for index, (label, value) in enumerate(parts)
                if index not in consumed
            ]
            if title_label and not structured_title:
                body_parts.insert(0, ("", title_value))
            elif not title:
                title = title_value
                body_parts = parts[1:]
            timeline = self.theme.experience_layout in {"timeline", "compact"} and normalized in {
                "internship", "internships", "education", "educations"
            }
            if is_project:
                head_html = (
                    '<div class="entry-head project-head">'
                    f'<span class="entry-title">{html.escape(title)}</span></div>'
                )
                subtitle_html = f'<div class="entry-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
            elif timeline:
                internship_with_department = normalized in {"internship", "internships"} and bool(department)
                department_html = f'<span class="entry-department">{html.escape(department)}</span>' if internship_with_department else ""
                is_internship = normalized in {"internship", "internships"}
                head_class = (
                    f'internship-head {"with-department" if internship_with_department else "no-department"}'
                    if is_internship
                    else ""
                )
                head_html = (
                    f'<div class="entry-head timeline-head {head_class}">'
                    f'<span class="entry-meta">{html.escape(date_value)}</span>'
                    f'<span class="entry-title">{html.escape(title)}</span>'
                    f'{department_html}'
                    f'<span class="entry-subtitle-inline">{html.escape(subtitle)}</span></div>'
                )
                subtitle_html = ""
            else:
                head_html = (
                    f'<div class="entry-head"><span class="entry-title">{html.escape(title)}</span>'
                    f'<span class="entry-meta">{html.escape(date_value)}</span></div>'
                )
                subtitle_html = f'<div class="entry-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
            visible_body = self._body_html(normalized, body_parts)
            entries.append(
                f'<article class="entry" data-claim-id="{html.escape(claim_id)}">'
                f'{head_html}'
                f'{subtitle_html}'
                f'<div class="entry-body">{visible_body}</div></article>'
            )
        return "".join(entries)


class ResumePDFRenderer:
    """Render and audit a verified document using a real Chromium print pipeline."""

    def __init__(self, store: CareerStore) -> None:
        self.store = store

    def render(
        self,
        resume_document_id: str,
        *,
        output_dir: str | Path,
        max_pages: int = 1,
        theme_id: str = "professional",
    ) -> ResumeRenderResult:
        if max_pages < 1 or max_pages > 3:
            raise ValueError("max_pages must be between 1 and 3.")
        try:
            theme = THEMES[theme_id]
        except KeyError as exc:
            raise ValueError(f"Unknown resume theme: {theme_id}") from exc
        document = self.store.get(resume_document_id, expected_type=ResumeDocument)
        if document.layout_spec:
            theme = ResumeTheme.from_layout_spec(
                document.layout_spec,
                theme_id=f"resume:{document.resume_document_id}",
            )
        if document.status is not ResumeDocumentStatus.VERIFIED:
            raise ValueError("Only an independently verified ResumeDocument can be rendered to PDF.")
        if document.verification_report_ref is None:
            raise ValueError("Verified ResumeDocument is missing its independent verification report.")
        report = self.store.get(
            document.verification_report_ref.record_id,
            document.verification_report_ref.revision,
            expected_type=VerificationReport,
        )
        if report.status is not VerificationStatus.PASSED:
            raise ValueError("ResumeDocument does not reference a passed VerificationReport.")
        claims = tuple(
            self.store.get(ref.record_id, ref.revision, expected_type=ResumeClaim)
            for ref in document.claim_refs
        )
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        stem = f"{document.resume_document_id}-r{document.revision}"
        html_path = destination / f"{stem}.html"
        pdf_path = destination / f"{stem}.pdf"
        trace_path = destination / f"{stem}.trace.json"
        html_renderer = ResumeHTMLRenderer(theme)
        html_text = html_renderer.render(document, claims)
        _atomic_write_text(html_path, html_text)

        browser_metrics = self._print_pdf(html_path, pdf_path, theme)
        page_count, extracted_text = self._read_pdf(pdf_path)
        normalized_pdf = "".join(extracted_text.split())
        expected_text = [document.header.full_name]
        claim_sections = {
            claim_id: section
            for section, claim_ids in document.sections.items()
            for claim_id in claim_ids
        }
        for claim in claims:
            section = claim_sections.get(claim.claim_id, "")
            expected_text.extend(html_renderer._visible_text_values(section, claim.text))
        text_layer_ok = bool(normalized_pdf) and all(
            "".join(value.split()) in normalized_pdf for value in expected_text
        )
        audit = ResumeRenderAudit(
            page_count=page_count,
            max_pages=max_pages,
            text_layer_ok=text_layer_ok,
            horizontal_overflow=bool(browser_metrics["horizontal_overflow"]),
            font_ready=bool(browser_metrics["font_ready"]),
            computed_font_family=str(browser_metrics["font_family"]),
            extracted_text_characters=len(extracted_text),
        )
        trace_payload = self._trace_payload(document, claims, report, audit, theme)
        _atomic_write_text(
            trace_path,
            json.dumps(trace_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        if not audit.passed:
            raise RuntimeError(
                "Resume PDF failed render audit: "
                + json.dumps(audit.to_dict(), ensure_ascii=False, sort_keys=True)
            )
        return ResumeRenderResult(
            document.resume_document_id,
            document.revision,
            html_path,
            pdf_path,
            trace_path,
            audit,
        )

    @staticmethod
    def _print_pdf(html_path: Path, pdf_path: Path, theme: ResumeTheme) -> dict[str, Any]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for resume PDF rendering; install the career extra."
            ) from exc
        temporary_pdf = pdf_path.with_suffix(".pdf.tmp")
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=playwright.chromium.executable_path,
                )
                try:
                    page = browser.new_page(viewport={"width": 794, "height": 1123})
                    page.goto(html_path.as_uri(), wait_until="load")
                    metrics = page.evaluate(
                        """async ({ fontFamily }) => {
                          await document.fonts.ready;
                          const width = document.documentElement.clientWidth;
                          const overflow = [...document.querySelectorAll('body *')].some((element) => {
                            const rect = element.getBoundingClientRect();
                            return rect.left < -0.5 || rect.right > width + 0.5;
                          });
                          const requested = fontFamily.split(',')[0].replaceAll('"', '').trim();
                          const loaded = await document.fonts.load(`12px "${requested}"`, '中文简历');
                          return {
                            horizontal_overflow: overflow,
                            font_ready: loaded.length > 0 || document.fonts.check(`12px "${requested}"`, '中文简历'),
                            font_family: getComputedStyle(document.body).fontFamily,
                          };
                        }""",
                        {"fontFamily": theme.font_family},
                    )
                    page.emulate_media(media="print")
                    page.pdf(
                        path=str(temporary_pdf),
                        format="A4",
                        print_background=True,
                        prefer_css_page_size=True,
                    )
                finally:
                    browser.close()
            os.replace(temporary_pdf, pdf_path)
            return metrics
        finally:
            if temporary_pdf.exists():
                temporary_pdf.unlink()

    @staticmethod
    def _read_pdf(pdf_path: Path) -> tuple[int, str]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("pypdf is required for ATS text-layer verification.") from exc
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return len(reader.pages), text

    def _trace_payload(
        self,
        document: ResumeDocument,
        claims: tuple[ResumeClaim, ...],
        report: VerificationReport,
        audit: ResumeRenderAudit,
        theme: ResumeTheme,
    ) -> dict[str, Any]:
        return {
            "resume": [document.resume_document_id, document.revision],
            "verification_report": [report.verification_report_id, report.revision],
            "theme_id": theme.theme_id,
            "audit": audit.to_dict(),
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "claim_revision": claim.revision,
                    "text": claim.text,
                    "evidence_ids": list(claim.evidence_ids),
                    "evidence_fact_ids": list(claim.evidence_fact_ids),
                    "requirement_ids": list(claim.requirement_ids),
                    "sources": [
                        {
                            "source_id": source.source_id,
                            "source_type": source.source_type.value,
                            "locator": source.locator,
                            "content_hash": source.content_hash,
                            "excerpt": source.excerpt,
                        }
                        for source in self.store.trace_claim(claim.claim_id).source_refs
                    ],
                }
                for claim in claims
            ],
        }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
