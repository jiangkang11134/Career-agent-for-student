"""审查系统钩子 — 3 个插入点，4 个职责分离。

职责分离：
  LooseReviewEngine    → 正则+AST 审查（毫秒级）
  StrictReviewEngine   → LLM 审查 + 熔断器（秒级）
  SandboxTestRunner    → Docker 沙箱测试（零 LLM）
  ReviewOrchestrator   → 编排 + 模式切换 + 3 个钩子
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from smart_agent.review.config import (
    IMPORT_MAP_DIR,
    IMPORT_MAP_FILE,
    SUB_AGENT_API_BASE,
    SUB_AGENT_API_KEY,
    SUB_AGENT_MODEL,
    get_review_mode,
)

logger = logging.getLogger("smart_agent.review")

# ════════════════════════════════════════════════════════════════
# 宽松审查引擎（写前预审查，正则+AST，毫秒级）
# ════════════════════════════════════════════════════════════════

_fp_cache: set[str] = set()
_fp_cache_lock = threading.Lock()

_FP_PREFIXES = ("test_", "example_", "mock_", "fixture_", "sample_", "fake_")

_CRITICAL_PATTERNS = [
    (r'(?i)(api[_-]?key|secret|password|token)\s*=\s*["\'][^"\']+["\']',
     "hardcoded-secret", "密钥硬编码"),
    (r'(?i)(execute|exec|raw)\(.*f["\']|\.format\(.*(?:request|input|user|get)',
     "sqli", "SQL 注入风险"),
    (r'\beval\s*\(', "unsafe-eval", "不安全的 eval 使用"),
]
_MINOR_PATTERNS = [
    (r'#\s*(TODO|FIXME|HACK|XXX)\b', "todo-left", "待办事项遗留"),
]
_MAJOR_PATTERNS = [
    (r'^\s*except\s*:\s*$', "bare-except", "裸 except 捕获所有异常"),
]


class LooseReviewEngine:
    """宽松审查引擎：写前执行正则+AST 检查，毫秒级阻断。"""

    @staticmethod
    def review(content: str, file_path: str) -> list[dict]:
        """执行审查，返回发现问题列表。"""
        return _pre_review_content(content, file_path)

    @staticmethod
    def has_issues(issues: list[dict]) -> tuple[list[dict], list[dict]]:
        """按严重等级分类。"""
        c = [i for i in issues if i["severity"] in ("critical", "major")]
        m = [i for i in issues if i["severity"] in ("minor", "suggestion")]
        return c, m


# 模块级函数（保持向下兼容）
def _pre_review_content(content: str, file_path: str = "") -> list[dict]:
    import ast
    findings = []
    lines = content.splitlines()
    fp_file = any(file_path.startswith(p) or f"/{p}" in file_path for p in _FP_PREFIXES)

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or fp_file:
            continue
        if any(stripped.startswith(p) for p in _FP_PREFIXES):
            continue
        if stripped.startswith(('#', '"', "'")):
            if stripped.startswith('"') or stripped.startswith("'"):
                continue

        for pattern, rule_id, message in _CRITICAL_PATTERNS:
            if _is_false_positive(file_path, rule_id, i):
                continue
            if re.search(pattern, stripped):
                findings.append({"line": i, "severity": "critical", "rule_id": rule_id, "message": message})

        for pattern, rule_id, message in _MINOR_PATTERNS:
            if re.search(pattern, stripped):
                findings.append({"line": i, "severity": "minor", "rule_id": rule_id, "message": message})

        for pattern, rule_id, message in _MAJOR_PATTERNS:
            if re.search(pattern, stripped):
                findings.append({"line": i, "severity": "major", "rule_id": rule_id, "message": message})

    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and any(
                isinstance(h.type, ast.Name) and h.type.id == "Exception" and not h.name
                for h in node.handlers
            ):
                findings.append({
                    "line": getattr(node, "lineno", 0), "severity": "major",
                    "rule_id": "bare-except", "message": "裸 except 捕获所有异常",
                })
    except SyntaxError:
        pass
    return findings


def _line_hash(file_path: str, line: str) -> str:
    import hashlib
    return hashlib.md5(f"{file_path}:{line.strip()[:60]}".encode()).hexdigest()[:12]


def _record_false_positive(file_path: str, issues: list[dict]) -> None:
    from smart_agent.review.memory import ReviewFinding, ReviewMemoryStore
    store = ReviewMemoryStore()
    for issue in issues:
        fp_id = f"fp-{issue['rule_id']}-{_line_hash(file_path, issue.get('line_content', ''))}"
        store.add_finding(ReviewFinding(id=fp_id, severity=issue["severity"], file_path=file_path, rule_id=issue["rule_id"], status="false_positive"))
    store.save()
    with _fp_cache_lock:
        _fp_cache.add(fp_id)


def _is_false_positive(file_path: str, rule_id: str, line: int) -> bool:
    key = f"fp-{rule_id}-{_line_hash(file_path, '')}"
    with _fp_cache_lock:
        if key in _fp_cache:
            return True
    from smart_agent.review.memory import ReviewMemoryStore
    store = ReviewMemoryStore()
    for f in store.find_by_file(file_path):
        if f.rule_id == rule_id and f.status == "false_positive":
            with _fp_cache_lock:
                _fp_cache.add(key)
            return True
    return False


# ════════════════════════════════════════════════════════════════
# 严格审查引擎（收集上下文 + 1 次 LLM + 熔断器保护）
# ════════════════════════════════════════════════════════════════

_strict_failures: list[int] = [0]
_strict_max_failures: int = 3
_strict_lock: threading.Lock = threading.Lock()


class StrictReviewEngine:
    """严格审查引擎：收集上下文 → 1 次 LLM 调用 → 返回裁决。"""

    def __init__(self, cwd: str, tools, tool_context):
        self.cwd = cwd
        self.tools = tools
        self.tool_context = tool_context

    def review(self, file_path: str, content: str, reason: str = "写入前审查") -> dict | None:
        """执行严格审查。返回 {"severity", "summary", "detail"} 或 None。"""
        from smart_agent.agents import (
            CODE_REVIEW_SPEC,
            AgentArtifact,
            AgentTask,
            CodeAgent,
            CodeReviewAgent,
            SubagentRunner,
        )
        from smart_agent.agent_collection import build_default_registry

        if self.tools is None:
            return None

        parent_session_id = str(
            getattr(getattr(self.tool_context, "session", None), "session_id", "")
            or f"code-review-{int(time.time() * 1000)}"
        )
        artifacts = [
            AgentArtifact.create(
                "task_goal",
                domain="code",
                session_id=parent_session_id,
                created_by="code_agent",
                content={"file_path": file_path, "reason": reason},
            )
        ]

        previous_content = ""
        try:
            from smart_agent.tooling import ToolContext
            from smart_agent.workspace import resolve_tool_path

            target_path = resolve_tool_path(
                ToolContext(cwd=self.cwd),
                file_path,
                "read for code review",
            )
            if target_path.exists() and target_path.is_file():
                previous_content = target_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            previous_content = ""
        code_diff = "\n".join(
            difflib.unified_diff(
                previous_content.splitlines(),
                content.splitlines(),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm="",
            )
        )
        artifacts.append(
            AgentArtifact.create(
                "code_diff",
                domain="code",
                session_id=parent_session_id,
                created_by="code_agent",
                content={"file_path": file_path, "diff": code_diff[:40_000]},
            )
        )

        # import map → 受影响文件
        im_path = Path(self.cwd) / IMPORT_MAP_DIR / IMPORT_MAP_FILE
        affected = []
        if im_path.exists():
            try:
                data = json.loads(im_path.read_text(encoding="utf-8"))
                for sym_name, sym_data in data.get("symbols", {}).items():
                    if sym_data.get("file") == file_path:
                        for ref in sym_data.get("referenced_by", []):
                            if ref != file_path and ref not in affected:
                                affected.append(ref)
                if affected:
                    for af in affected[:5]:
                        try:
                            af_path = resolve_tool_path(
                                ToolContext(cwd=self.cwd),
                                af,
                                "read related file for code review",
                            )
                        except (OSError, PermissionError):
                            continue
                        if af_path.exists() and af_path.is_file():
                            artifacts.append(
                                AgentArtifact.create(
                                    "related_file",
                                    domain="code",
                                    session_id=parent_session_id,
                                    created_by="code_agent",
                                    content={
                                        "file_path": af,
                                        "content": af_path.read_text(
                                            encoding="utf-8",
                                            errors="replace",
                                        )[:20_000],
                                    },
                                )
                            )
            except Exception:
                pass

        # code_review 结果
        try:
            cr = self.tools.execute("code_review", {"path": str(Path(self.cwd) / Path(file_path).parent)}, self.tool_context)
            if cr and cr.ok:
                artifacts.append(
                    AgentArtifact.create(
                        "verification_result",
                        domain="code",
                        session_id=parent_session_id,
                        created_by="code_agent",
                        content={"tool": "code_review", "result": cr.output[:4_000]},
                    )
                )
        except Exception:
            pass

        runtime = dict(getattr(self.tool_context, "_runtime", None) or {})
        if not runtime:
            try:
                from smart_agent.config import load_runtime_config

                runtime = dict(load_runtime_config(self.cwd))
            except Exception:
                return None
        runtime["activeDomain"] = "code"
        runtime.setdefault("domainStage", "default")
        if SUB_AGENT_MODEL:
            runtime["model"] = SUB_AGENT_MODEL
        if SUB_AGENT_API_KEY:
            runtime["apiKey"] = SUB_AGENT_API_KEY
            runtime["authToken"] = SUB_AGENT_API_KEY
        if SUB_AGENT_API_BASE:
            runtime["baseUrl"] = SUB_AGENT_API_BASE

        task = AgentTask.create(
            "code_review",
            domain="code",
            description=f"Review {file_path}",
            parent_session_id=parent_session_id,
            assigned_agent="code_review",
            input_artifact_ids=[artifact.artifact_id for artifact in artifacts],
            allowed_tools=CODE_REVIEW_SPEC.allowed_tools,
            timeout_seconds=CODE_REVIEW_SPEC.timeout_seconds,
            max_steps=CODE_REVIEW_SPEC.max_steps,
        )

        try:
            result = build_default_registry().create("code", 
                SubagentRunner(
                    cwd=self.cwd,
                    runtime=runtime,
                    tools=self.tools,
                    permissions=getattr(self.tool_context, "permissions", None),
                )
            ).run(task, artifacts, mode="code_review")
            if result.status != "completed" or result.output_artifact is None:
                logger.warning("Code Review Agent failed: %s", result.error)
                return None

            severity_order = {"critical": 3, "major": 2, "minor": 1, "info": 0}
            severity = max(
                (finding.severity for finding in result.findings),
                key=lambda item: severity_order[item],
                default="pass",
            )
            report_content = result.output_artifact.content
            summary = str(report_content.get("summary") or "审查通过")
            return {
                "severity": severity,
                "summary": summary,
                "detail": json.dumps(report_content, ensure_ascii=False),
                "artifact_id": result.output_artifact.artifact_id,
                "child_session_id": task.child_session_id,
            }
        except Exception as exc:
            logger.warning("Code Review Agent failed: %s", exc)
            return None

    @staticmethod
    def is_circuit_open() -> bool:
        with _strict_lock:
            return _strict_failures[0] >= _strict_max_failures

    @staticmethod
    def record_failure() -> int:
        with _strict_lock:
            _strict_failures[0] += 1
            return _strict_failures[0]

    @staticmethod
    def reset_circuit() -> None:
        with _strict_lock:
            _strict_failures[0] = 0


# ════════════════════════════════════════════════════════════════
# Docker 沙箱测试引擎
# ════════════════════════════════════════════════════════════════

class SandboxTestRunner:
    """沙箱测试：Docker 容器跑测试，返回结构化结果（零 LLM）。"""

    def __init__(self, cwd: str, tools, tool_context):
        self.cwd = cwd
        self.tools = tools
        self.tool_context = tool_context

    def run(self, file_path: str) -> dict | None:
        """在 Docker 沙箱中跑测试，截断输出后返回。

        完全零 LLM，直接从 pytest 输出中提取失败信息。

        返回:
            {"passed": bool, "detail": str} 或 None（异常时）
        """
        try:
            result = self.tools.execute("sandbox_test", {"changed_files": [file_path]}, self.tool_context)
            if not result:
                return None
            output = result.output or ""
            if "[SANDBOX_RESULT: PASS]" in output:
                return {"passed": True, "detail": ""}

            # 直接截取 pytest 错误输出，不经过 LLM
            detail = output[:2000]
            logger.info("Tests FAILED for %s", file_path)
            return {"passed": False, "detail": detail}
        except Exception as exc:
            logger.warning("Test failed: %s", exc)
            return None


# ════════════════════════════════════════════════════════════════
# 编排层：3 个钩子 + 模式切换 + import map
# ════════════════════════════════════════════════════════════════

def get_review_hooks(cwd: str, tools=None, tool_context=None):
    """工厂函数，始终返回 ReviewOrchestrator 实例。"""
    return ReviewOrchestrator(cwd, tools, tool_context)


class ReviewOrchestrator:
    """审查编排器：管理 3 个钩子、模式切换、coda 注入。

    职责：
      - on_before_write → 宽松审查 + 严格审查（委派给引擎）
      - on_file_written → import map 更新 + test 触发
      - on_turn_end → 注入 + 沉淀
      - _check_transition → 模式切换检测
    """

    def __init__(self, cwd: str, tools=None, tool_context=None):
        self.cwd = cwd
        self.tools = tools
        self.tool_context = tool_context
        self._coda_findings: list[dict[str, Any]] = []
        self._coda_lock = threading.Lock()
        self._bg_threads: list[threading.Thread] = []
        self._import_map_thread: threading.Thread | None = None
        self._prev_mode: str | None = None
        self._retro_scan_done: bool = False
        self._new_files: set[str] = set()
        self._turn_files: set[str] = set()  # 当前 turn 已写入的文件，用于 touched_files 统计

        self._session_rule_counts: dict[str, int] = {}

        # 引擎实例
        self._strict = StrictReviewEngine(cwd, tools, tool_context)
        self._sandbox = SandboxTestRunner(cwd, tools, tool_context)

        self._init_import_map()

    # ── 模式切换 ──

    def _check_transition(self) -> bool:
        current = get_review_mode()
        if self._prev_mode is None:
            self._prev_mode = current
            return False
        if self._prev_mode == "off" and current in ("loose", "strict") and not self._retro_scan_done:
            self._retro_scan_done = True
            self._prev_mode = current
            logger.info("=== 审查模式切换: off → %s，启动全面回补扫描 ===", current)
            threading.Thread(target=self._retroactive_scan, daemon=True).start()
            return True
        if self._prev_mode != current:
            logger.info("审查模式切换: %s → %s", self._prev_mode, current)
            self._prev_mode = current
        return False

    def _retroactive_scan(self) -> None:
        import glob as py_glob
        logger.info("全面回补扫描开始...")
        im_path = Path(self.cwd) / IMPORT_MAP_DIR / IMPORT_MAP_FILE
        if not im_path.exists():
            try:
                from smart_agent.tools.import_map import build_import_map
                build_import_map(self.cwd)
            except Exception as exc:
                logger.warning("回补扫描：import map 建表失败: %s", exc)

        all_findings = []
        scanned = 0
        for py_file in py_glob.glob(f"{self.cwd}/**/*.py", recursive=True):
            try:
                content = Path(py_file).read_text(encoding="utf-8", errors="replace")
                rel_path = str(Path(py_file).relative_to(self.cwd))
                issues = _pre_review_content(content, rel_path)
                if issues:
                    all_findings.extend(issues)
                scanned += 1
            except (OSError, UnicodeDecodeError):
                continue

        logger.info("回补扫描完成：%d 文件，%d 个问题", scanned, len(all_findings))
        if all_findings:
            critical = [f for f in all_findings if f["severity"] in ("critical", "major")]
            minor = [f for f in all_findings if f["severity"] in ("minor", "suggestion")]
            report = [f"[Retroactive Scan] 扫描 {scanned} 文件，发现 {len(critical)} 个严重问题，{len(minor)} 个建议。"]
            if critical:
                report.append("\n严重问题：")
                for f in critical[:30]:
                    report.append(f"  {f['file_path']}:L{f['line']} [{f['severity']}] {f['message']}")
            if minor:
                report.append("\n建议：")
                for f in minor[:10]:
                    report.append(f"  {f['file_path']}:L{f['line']} {f['message']}")
            self._coda_findings.append({"role": "system", "content": "\n".join(report)})
            if critical:
                try:
                    from smart_agent.review.promotion import promote_findings
                    promote_findings(critical, self.cwd)
                except Exception:
                    pass

    # ── import map ──

    def _init_import_map(self) -> None:
        if get_review_mode() == "off":
            return
        im_path = Path(self.cwd) / IMPORT_MAP_DIR / IMPORT_MAP_FILE
        if im_path.exists():
            return

        def _build():
            try:
                from smart_agent.tools.import_map import build_import_map
                t0 = time.time()
                build_import_map(self.cwd)
                logger.info("Import map built in %d ms", int((time.time() - t0) * 1000))
            except Exception as exc:
                logger.warning("Import map build failed: %s", exc)

        threading.Thread(target=_build, daemon=True).start()

    def _async_update_import_map(self, file_path: str) -> None:
        try:
            from smart_agent.tools.import_map import update_import_map_for_file
            update_import_map_for_file(self.cwd, file_path)
        except Exception as exc:
            logger.debug("Import map update failed: %s", exc)

    def _compute_diff_stat(self, file_path: str) -> dict[str, Any]:
        """计算文件变更统计：行数、触及文件数、public API 变更。"""
        stat: dict[str, Any] = {
            "changed_lines": 0,
            "touched_files": len(self._turn_files),
            "public_api_changed": False,
        }
        try:
            import subprocess
            r = subprocess.run(
                ["git", "diff", "--numstat", "HEAD", "--", file_path],
                capture_output=True, text=True, cwd=self.cwd, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                parts = r.stdout.strip().split("\t")
                if len(parts) >= 2:
                    added = int(parts[0]) if parts[0].isdigit() else 0
                    deleted = int(parts[1]) if parts[1].isdigit() else 0
                    stat["changed_lines"] = added + deleted

            r2 = subprocess.run(
                ["git", "diff", "HEAD", "--", file_path],
                capture_output=True, text=True, cwd=self.cwd, timeout=5,
            )
            if r2.returncode == 0 and r2.stdout.strip():
                for line in r2.stdout.splitlines():
                    if line.startswith("@@"):
                        continue
                    if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                        import re
                        if re.search(r'^\s*(?:async\s+)?(?:def |class )', line[1:]):
                            stat["public_api_changed"] = True
                            break
        except Exception:
            pass
        return stat

    def _promote_finding_immediately(self, finding: dict) -> None:
        """发现即沉淀：增量计数 + 即时提升。

        替代原来的收尾批处理，每次发现直接判断是否需要沉淀为 skill 或全局记忆。
        """
        try:
            from smart_agent.review.promotion import is_important_finding, promote_to_global_memory, promote_to_skill

            rule_id = finding.get("rule_id", "unknown")
            self._session_rule_counts[rule_id] = self._session_rule_counts.get(rule_id, 0) + 1
            action = is_important_finding(finding, self._session_rule_counts)
            if action == "skill":
                promote_to_skill(finding)
                logger.info("Immediate promote to skill: %s", rule_id)
            elif action == "global_memory":
                promote_to_global_memory(finding, self.cwd)
                logger.info("Immediate promote to global memory: %s", rule_id)
        except Exception as exc:
            logger.debug("Immediate promotion failed: %s", exc)

    # ── 钩子 1：写前审查 ──

    def on_before_write(self, tool_name: str, tool_input: dict[str, Any]):
        """写前审查：宽松 + 严格（按需），返回 ToolResult 阻断 or None 放行。"""
        self._check_transition()
        if get_review_mode() == "off":
            return None
        from smart_agent.tooling import ToolResult

        content = tool_input.get("content") or tool_input.get("new_string", "")
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        if file_path and not Path(file_path).exists():
            self._new_files.add(file_path)
        if not content.strip():
            return None

        # 宽松审查
        issues = LooseReviewEngine.review(content, file_path)
        critical, minor = LooseReviewEngine.has_issues(issues)
        force = tool_input.get("force", False) or tool_input.get("force_write", False)

        # 发现即沉淀
        for issue in issues:
            self._promote_finding_immediately(issue)

        if critical:
            if force:
                _record_false_positive(file_path, critical)
                return None
            detail = "\n".join(f"  L{i['line']} [{i['severity']}] {i['message']}" for i in critical[:5])
            return ToolResult(ok=False, output=f"[Pre-review Blocked] 已阻止写入 {file_path}：\n{detail}\n\n如需强制写入，添加 force=true 参数。")

        # 严格审查
        if get_review_mode() == "strict":
            if self._strict.is_circuit_open():
                logger.warning("Strict review circuit breaker OPEN")
                self._coda_findings.append({"role": "system", "content": "[审查熔断] 严格审查已临时关闭，仅执行了基础安全检查。"})
            else:
                verdict = self._strict.review(file_path, content)
                if verdict is None:
                    fs = self._strict.record_failure()
                    self._coda_findings.append({"role": "system", "content": f"[审查退化] API 异常（{fs}/{_strict_max_failures}），仅执行基础检查。"})
                else:
                    self._strict.reset_circuit()
                    sev = verdict.get("severity", "pass")
                    if sev == "critical" and not force:
                        return ToolResult(ok=False, output=f"[审查阻断: 安全风险] {file_path}\n问题: {verdict['summary']}\n{verdict.get('detail','')}\n如需强制写入，添加 force=true 参数。")
                    elif sev == "major" and not force:
                        return ToolResult(ok=False, output=f"[审查阻断: 兼容性影响] {file_path}\n问题: {verdict['summary']}\n{verdict.get('detail','')}\n如需强制写入，添加 force=true 参数。")
                    elif sev == "minor":
                        self._coda_findings.append({"role": "system", "content": f"[审查报告] {file_path}\n  安全审查: PASS（建议）\n{verdict.get('detail', verdict.get('summary', ''))}"})
                    else:
                        self._coda_findings.append({"role": "system", "content": f"[审查报告] {file_path}\n  安全审查: PASS\n  兼容性: 无影响"})

        if minor:
            return ToolResult(ok=True, output="[Pre-review OK] 写入已放行。建议关注：\n" + "\n".join(f"  L{i['line']} {i['message']}" for i in minor[:3]))
        return None

    # ── 钩子 2：写后处理 ──

    def on_file_written(self, file_path: str, **kwargs):
        """写后：更新 import map + 触发 test agent。

        测试结果会合并到已有的审查报告中（如果存在），
        主 Agent 只看到一条统一的 [审查报告] 消息。
        """
        self._turn_files.add(file_path)
        self._import_map_thread = threading.Thread(target=self._async_update_import_map, args=(file_path,), daemon=True)
        self._import_map_thread.start()

        if get_review_mode() == "strict":
            from smart_agent.review.mode_engine import should_trigger_strict
            diff_stat = self._compute_diff_stat(file_path)
            should, reason = should_trigger_strict(file_path, diff_stat, cwd=self.cwd, is_new_file=file_path in self._new_files)
            if should:
                # 后台跑测试，结果会合并到 _coda_findings 中
                t = threading.Thread(target=self._merge_test_into_report, args=(file_path,), daemon=True)
                self._bg_threads.append(t)
                t.start()

    def _has_existing_tests(self, file_path: str) -> bool:
        """检查项目是否存在对应改动的测试文件。"""
        from smart_agent.tools.sandbox_test import resolve_test_candidates
        root = Path(self.cwd)
        return any(
            (root / c).exists()
            for c in resolve_test_candidates(self.cwd, file_path)
        )

    def _merge_test_into_report(self, file_path: str) -> None:
        """跑测试并合并结果到已有的审查报告中。

        条件：
          - 审查未通过 → 不测试（先修代码）
          - 无对应测试文件 → 不测试（测了也是空的）
          - 审查通过 + 有测试 → 沙箱跑测
        """
        if not self._has_existing_tests(file_path):
            logger.info("跳过测试：%s 无对应测试文件", file_path)
            return

        test_result = self._sandbox.run(file_path)
        if test_result is None:
            return

        status = "通过" if test_result["passed"] else "失败"
        test_line = f"\n  Docker 沙箱测试: {status}"
        if not test_result["passed"] and test_result.get("detail"):
            test_line += f"\n  失败分析:\n{test_result['detail'][:1500]}"

        with self._coda_lock:
            # 查找已有的审查报告，合并进去
            found = False
            for msg in self._coda_findings:
                if msg.get("role") == "system" and "[审查报告]" in msg.get("content", "") and file_path in msg.get("content", ""):
                    msg["content"] += "\n" + test_line
                    found = True
                    break
            # 没有已有的审查报告，单独追加
            if not found:
                self._coda_findings.append({
                    "role": "system",
                    "content": f"[审查报告] {file_path}\n  测试结果: {status}\n{test_line}",
                })

    # ── 钩子 3：Coda 阶段 ──

    def on_turn_end(self, current_messages: list[dict]) -> list[dict[str, Any]]:
        self._check_transition()
        if self._import_map_thread and self._import_map_thread.is_alive():
            self._import_map_thread.join(timeout=5)
            self._import_map_thread = None
        for t in self._bg_threads:
            if t.is_alive():
                t.join(timeout=30)
        self._bg_threads.clear()

        self._turn_files.clear()
        self._session_rule_counts.clear()

        with self._coda_lock:
            if not self._coda_findings:
                return []
            for msg in self._coda_findings:
                current_messages.append(msg)
            findings = [{"content": m.get("content", ""), "role": "system"} for m in self._coda_findings]
            self._coda_findings.clear()
            return findings
