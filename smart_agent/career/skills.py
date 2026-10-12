"""Deterministic skill atomization and presentation grouping."""

from __future__ import annotations

import hashlib
import re
from typing import Any

SKILL_CATEGORIES = (
    "专业能力",
    "工具与平台",
    "方法与流程",
    "数据与分析",
    "项目与交付",
    "语言与证书",
    "通用能力",
    "其他技能",
)

_SECTION_HEADING = re.compile(
    r"^\s*(?:技能/其他|技能证书|专业技能|个人技能|技能|其他|skills)\s*[:：]\s*",
    re.IGNORECASE,
)
_PROFICIENCY = re.compile(
    r"^\s*(熟练使用|实际使用过|精读并完整复现|系统学习|深入理解|"
    r"熟悉|熟练|掌握|了解|精通|具备|擅长|使用过|能够使用)\s*",
)
_LIST_SEPARATOR = re.compile(r"\s*(?:[、,，]+|\s+[和与]\s+|/)\s*")
_SENTENCE_SEPARATOR = re.compile(
    r"[；;。]+|，(?=\s*(?:熟练使用|实际使用过|精读并完整复现|系统学习|深入理解|"
    r"熟悉|熟练|掌握|了解|精通|具备|擅长|使用过|能够使用))"
)
_HEADING_PUNCTUATION = re.compile(r"[、,，；;。!?！？]")

_GENERIC_CATEGORY_RULES = (
    ("语言与证书", ("英语", "日语", "韩语", "法语", "德语", "雅思", "托福", "cet", "证书", "资质", "普通话")),
    ("通用能力", ("沟通", "协作", "管理", "组织", "表达", "学习能力", "解决问题", "抗压", "复盘", "汇报")),
    ("数据与分析", ("数据", "分析", "统计", "报表", "建模", "指标", "调研", "归因", "预算", "excel", "spss", "bi")),
    ("项目与交付", ("项目", "交付", "实施", "落地", "部署", "运营", "推进", "统筹", "验收", "复盘")),
    ("方法与流程", ("方法", "流程", "规范", "体系", "策略", "评估", "测试", "设计", "研究", "优化")),
    ("工具与平台", ("工具", "平台", "系统", "软件", "office", "cad", "erp", "crm", "ps", "ai", "git", "linux")),
)


def _clean_line(value: str) -> str:
    return re.sub(r"^[#*+\-•·\s]+", "", str(value or "")).strip()


def _category_for_text(text: str) -> str:
    normalized = re.sub(r"\s+", "", text).casefold()
    for category, keywords in _GENERIC_CATEGORY_RULES:
        if any(keyword.casefold() in normalized for keyword in keywords):
            return category
    if re.search(r"[A-Za-z][A-Za-z0-9+#._-]*", text):
        return "工具与平台"
    return "专业能力"


def _heading_category(text: str, *, paragraph_heading: bool = False) -> str | None:
    cleaned = _clean_line(text).rstrip("：:").strip()
    if not cleaned or _PROFICIENCY.match(cleaned) or _HEADING_PUNCTUATION.search(cleaned):
        return None
    if paragraph_heading and len(cleaned) <= 64:
        return cleaned
    return None


def _skill_id(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    return "skill-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _split_atoms(text: str) -> list[str]:
    atoms: list[str] = []
    for clause in _SENTENCE_SEPARATOR.split(str(text or "")):
        clause = clause.strip()
        if not clause:
            continue
        qualifier_match = _PROFICIENCY.match(clause)
        qualifier = qualifier_match.group(1) if qualifier_match else ""
        pieces = [piece.strip() for piece in _LIST_SEPARATOR.split(clause) if piece.strip()]
        for index, piece in enumerate(pieces):
            if index > 0 and qualifier and not _PROFICIENCY.match(piece):
                piece = f"{qualifier}{piece}"
            atoms.append(piece.rstrip("；;。 "))
    return atoms


def prepare_skill_block(text: str) -> dict[str, Any]:
    """Extract evidence-backed skill atoms and deterministic category suggestions."""
    normalized = str(text or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("Skill evidence block is empty")

    paragraphs = [part for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    facts: list[dict[str, str]] = []
    categories: list[str] = []
    seen: set[str] = set()

    def remember_category(category: str) -> None:
        if category and category not in categories:
            categories.append(category)

    for paragraph in paragraphs:
        lines = [_clean_line(line) for line in paragraph.splitlines() if _clean_line(line)]
        if not lines:
            continue
        lines[0] = _SECTION_HEADING.sub("", lines[0]).strip()
        lines = [line for line in lines if line]
        if not lines:
            continue

        current_category = ""
        first_category = _heading_category(lines[0], paragraph_heading=len(lines) > 1)
        if first_category:
            current_category = first_category
            remember_category(current_category)
            lines = lines[1:]

        for line in lines:
            heading = _heading_category(line)
            if heading:
                current_category = heading
                remember_category(current_category)
                continue
            heading_only = re.match(r"^([^：:]{1,64})[：:]\s*$", line)
            if heading_only and not _PROFICIENCY.match(heading_only.group(1).strip()):
                current_category = heading_only.group(1).strip()
                remember_category(current_category)
                continue
            heading_with_content = re.match(r"^([^：:]{1,32})[：:]\s*(.+)$", line)
            if heading_with_content:
                parsed_heading = _heading_category(heading_with_content.group(1), paragraph_heading=True)
                if parsed_heading:
                    current_category = parsed_heading
                    remember_category(current_category)
                    line = heading_with_content.group(2).strip()

            for atom in _split_atoms(line):
                key = re.sub(r"\s+", "", atom).casefold()
                if not key or key in seen:
                    continue
                seen.add(key)
                rule_category = _category_for_text(atom)
                suggested = current_category or rule_category
                remember_category(suggested)
                source = "existing" if current_category else ("fallback" if rule_category == "其他技能" else "rule")
                facts.append(
                    {
                        "skill_id": _skill_id(atom),
                        "text": atom,
                        "suggested_category": suggested,
                        "category_source": source,
                    }
                )

    if not facts:
        raise ValueError("Skill evidence block contains no usable skill facts")
    return {"skill_facts": facts, "allowed_categories": categories}


def render_skill_groups(skill_facts: list[dict[str, str]], groups: list[dict[str, Any]]) -> str:
    """Render validated skill IDs; model-authored ability text is never accepted."""
    by_id = {fact["skill_id"]: fact["text"] for fact in skill_facts}
    lines = []
    for group in groups:
        category = str(group["category"]).strip()
        values = [by_id[str(skill_id)] for skill_id in group["skill_ids"]]
        lines.append(f"{category}：{'；'.join(values)}")
    return "\n".join(lines)
