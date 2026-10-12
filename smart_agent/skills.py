"""技能文件发现、加载与管理系统。

支持 YAML front matter 元数据、按 agent type 过滤、always_load 自动预热。
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# ── 元数据 ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class SkillMetadata:
    """从 SKILL.md front matter 解析出的元数据。"""
    triggers: list[str] = field(default_factory=list)
    always_load: bool = False
    agent_types: list[str] = field(default_factory=lambda: ["general"])
    stages: list[str] = field(default_factory=list)
    priority: int = 0  # 数值越高越优先进入系统提示


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

_DEFAULT_AGENT_TYPES = ["general"]


def _parse_front_matter(content: str) -> tuple[dict, str]:
    """解析 YAML front matter，返回 (元数据字典, 正文)。

    参数:
        content: SKILL.md 完整内容。

    返回:
        (元数据字典, 去除 front matter 后的正文)。
    """
    m = _FRONT_MATTER_RE.match(content)
    if not m:
        return {}, content
    raw = m.group(1)
    body = content[m.end():]
    meta: dict = {}
    for line in raw.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip()

        if not val:
            continue

        # 解析 YAML 列表: [a, b, c]
        if val.startswith("[") and val.endswith("]"):
            val = [v.strip().strip("\"'") for v in val[1:-1].split(",") if v.strip()]
        # 布尔值
        elif val.lower() in ("true", "false"):
            val = val.lower() == "true"
        # 数字
        elif val.isdigit():
            val = int(val)

        meta[key] = val
    return meta, body


def _extract_description(markdown_body: str) -> str:
    """从 markdown 正文提取第一个非标题段落作为描述。"""
    normalized = markdown_body.replace("\r\n", "\n")
    paragraphs = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    for block in paragraphs:
        if block.startswith("#"):
            continue
        for line in [part.strip() for part in block.split("\n")]:
            if line and not line.startswith("#"):
                return line.replace("`", "")
    return "No description provided."


def _clean_descriptions(descriptions_str: str) -> list[str]:
    """将 triggers 描述的字符串按逗号分隔转换为列表。"""
    if isinstance(descriptions_str, list):
        return descriptions_str
    if isinstance(descriptions_str, str):
        return [item.strip().strip("\"'") for item in descriptions_str.replace("，", ",").split(",") if item.strip()]
    return []


# ── 数据模型 ────────────────────────────────────────────────────────────

@dataclass(slots=True)
class SkillSummary:
    """技能摘要信息。"""
    name: str
    description: str
    path: str
    source: str
    triggers: list[str] = field(default_factory=list)
    always_load: bool = False
    agent_types: list[str] = field(default_factory=lambda: _DEFAULT_AGENT_TYPES)
    stages: list[str] = field(default_factory=list)
    priority: int = 0


@dataclass(slots=True)
class LoadedSkill(SkillSummary):
    """已加载的技能，包含完整内容。"""
    content: str = ""
    metadata: SkillMetadata = field(default_factory=SkillMetadata)


# ── 文件操作 ────────────────────────────────────────────────────────────

def _parse_skill_file(content: str) -> tuple[SkillMetadata, str, str]:
    """解析 SKILL.md 内容，返回 (元数据, 正文, 描述)。"""
    meta_dict, body = _parse_front_matter(content)
    description = _extract_description(body)
    triggers = meta_dict.get("triggers", [])
    if isinstance(triggers, str):
        triggers = _clean_descriptions(triggers)
    agent_types = meta_dict.get("agent_types", _DEFAULT_AGENT_TYPES)
    if isinstance(agent_types, str):
        agent_types = _clean_descriptions(agent_types)
    stages = meta_dict.get("stages", [])
    if isinstance(stages, str):
        stages = _clean_descriptions(stages)
    meta = SkillMetadata(
        triggers=triggers or [],
        always_load=meta_dict.get("always_load", False),
        agent_types=agent_types or _DEFAULT_AGENT_TYPES,
        stages=stages or [],
        priority=meta_dict.get("priority", 0),
    )
    return meta, body, description or "No description provided."


def _home_dir() -> Path:
    return Path.home()


def _skill_roots(cwd: str | Path) -> list[tuple[Path, str]]:
    """获取所有技能根目录（按优先级降序）。"""
    base = Path(cwd)
    home = _home_dir()
    builtin = Path(__file__).resolve().parent / "builtin_skills"
    return [
        (base / ".runtime" / "skills", "project"),
        (home / ".runtime" / "skills", "user"),
        (builtin, "builtin"),
        (base / ".claude" / "skills", "compat_project"),
        (home / ".claude" / "skills", "compat_user"),
    ]


def _list_skill_dirs(root: Path, source: str) -> list[LoadedSkill]:
    """扫描单根目录下的所有 SKILL.md。"""
    if not root.exists():
        return []
    results: list[LoadedSkill] = []
    for entry in root.iterdir():
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        skill_path = entry / "SKILL.md"
        if not skill_path.exists():
            continue
        try:
            content = skill_path.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, body, desc = _parse_skill_file(content)
        results.append(
            LoadedSkill(
                name=entry.name,
                description=desc,
                path=str(skill_path),
                source=source,
                content=content,
                metadata=meta,
                triggers=meta.triggers,
                always_load=meta.always_load,
                agent_types=meta.agent_types,
                stages=meta.stages,
                priority=meta.priority,
            )
        )
    return results


def _matches_agent_type(skill: LoadedSkill, agent_type: str | None) -> bool:
    """检查技能是否匹配指定 agent type。"""
    if agent_type is None:
        return True
    return agent_type in skill.agent_types or "general" in skill.agent_types


def _matches_stage(skill: LoadedSkill, stage: str | None) -> bool:
    if not skill.stages:
        return True
    if stage is None:
        return False
    selected = str(stage).strip().lower()
    return selected in {str(item).strip().lower() for item in skill.stages}


# ── 公开 API ────────────────────────────────────────────────────────────

def discover_skills(
    cwd: str | Path,
    agent_type: str | None = None,
    stage: str | None = None,
) -> list[SkillSummary]:
    """发现当前环境下所有可用的技能。

    参数:
        cwd: 当前工作目录。
        agent_type: 可选的 agent type 过滤（如 "review"、"test"、"explore"、"general"）。

    返回:
        SkillSummary 列表（不含完整内容）。
    """
    by_name: dict[str, LoadedSkill] = {}
    for root, source in _skill_roots(cwd):
        for skill in _list_skill_dirs(root, source):
            if by_name.setdefault(skill.name, skill) is skill:
                # 新技能（优先级最高）
                pass
    return [
        SkillSummary(
            name=s.name,
            description=s.description,
            path=s.path,
            source=s.source,
            triggers=s.triggers,
            always_load=s.always_load,
            agent_types=s.agent_types,
            stages=s.stages,
            priority=s.priority,
        )
        for s in by_name.values()
        if _matches_agent_type(s, agent_type) and _matches_stage(s, stage)
    ]


def load_skill(cwd: str | Path, name: str) -> LoadedSkill | None:
    """按名称加载指定技能，返回第一个匹配的。"""
    normalized_name = name.strip()
    if not normalized_name:
        return None
    for root, source in _skill_roots(cwd):
        skill_path = root / normalized_name / "SKILL.md"
        if skill_path.exists():
            try:
                content = skill_path.read_text(encoding="utf-8")
            except OSError:
                continue
            meta, body, desc = _parse_skill_file(content)
            return LoadedSkill(
                name=normalized_name,
                description=desc,
                path=str(skill_path),
                source=source,
                content=content,
                metadata=meta,
                triggers=meta.triggers,
                always_load=meta.always_load,
                agent_types=meta.agent_types,
                stages=meta.stages,
                priority=meta.priority,
            )
    return None


def load_always_load_skills(cwd: str | Path) -> list[LoadedSkill]:
    """加载所有标记为 always_load=True 的技能完整内容。

    用于启动时自动注入系统提示。
    """
    result: list[LoadedSkill] = []
    seen: set[str] = set()
    for root, source in _skill_roots(cwd):
        for skill in _list_skill_dirs(root, source):
            if skill.name in seen:
                continue
            seen.add(skill.name)
            if skill.always_load:
                result.append(skill)
    result.sort(key=lambda s: s.priority, reverse=True)
    return result


# ── 安装/移除 ───────────────────────────────────────────────────────────

def _managed_skill_root(scope: str, cwd: str | Path) -> Path:
    return (Path(cwd) / ".runtime" / "skills") if scope == "project" else (_home_dir() / ".runtime" / "skills")


def install_skill(cwd: str | Path, source_path: str, name: str | None = None, scope: str = "user") -> dict[str, str]:
    """安装一个技能到指定作用域。"""
    source = Path(source_path)
    if not source.is_absolute():
        source = Path(cwd) / source
    if source.is_dir():
        skill_file = source / "SKILL.md"
        inferred_name = source.name
    else:
        skill_file = source if source.name == "SKILL.md" else source / "SKILL.md"
        inferred_name = skill_file.parent.name
    if not skill_file.exists():
        raise RuntimeError(f"No SKILL.md found in {source}")

    skill_name = (name or inferred_name).strip()
    if not skill_name:
        raise RuntimeError("Skill name cannot be empty.")

    target_dir = _managed_skill_root(scope, cwd) / skill_name
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(skill_file, target_dir / "SKILL.md")
    return {"name": skill_name, "targetPath": str(target_dir / "SKILL.md")}


def remove_managed_skill(cwd: str | Path, name: str, scope: str = "user") -> dict[str, object]:
    """移除指定作用域中的托管技能。"""
    target_path = _managed_skill_root(scope, cwd) / name
    if not target_path.exists():
        return {"removed": False, "targetPath": str(target_path)}
    shutil.rmtree(target_path)
    return {"removed": True, "targetPath": str(target_path)}
