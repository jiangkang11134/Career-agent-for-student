"""Generate a read-only automatic Project Analyst evaluation report.

Example:
    python scripts/evaluate_project_analyst.py
    python scripts/evaluate_project_analyst.py --career-root .runtime/career --format markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow direct execution from the repository root or from another cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_agent.career.project_evaluation import evaluate_project_analysis


def _markdown(report: dict[str, Any]) -> str:
    rows = (
        ("运行数", report["run_count"]),
        ("项目数", report["project_count"]),
        ("完成率", report["completion_rate"]),
        ("工具轨迹可用率", report["trace_available_rate"]),
        ("缺失工具轨迹数", report["trace_missing_count"]),
        ("README 首读率", report["readme_first_rate"]),
        ("核心模块读取率", report["core_module_read_rate"]),
        ("引用路径有效率", report["citation_valid_path_rate"]),
        ("引用行号有效率", report["citation_valid_line_rate"]),
        ("引用片段支持率", report["citation_excerpt_support_rate"]),
        ("禁止目录访问率", report["forbidden_path_rate"]),
        ("已登记路径率", report["registered_path_rate"]),
        ("缓存命中率", report["cache_hit_rate"]),
        ("未缓存平均耗时(ms)", report["average_elapsed_ms_uncached"]),
        ("工具调用总数", report["total_tool_call_count"]),
        ("Token 总数", report["total_token_count"]),
    )
    lines = [
        "# Project Analyst 自动评测",
        "",
        "范围：仅 Project Analyst；没有人工金标准，不计算语义 Precision/Recall/F1。",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines.extend(("", "## 运行明细", "", "| 项目 | 状态 | README 首读 | 核心模块读取率 | 引用支持率 | 禁止访问 |", "| --- | --- | ---: | ---: | ---: | ---: |"))
    for item in report["runs"]:
        lines.append(
            f"| {item['project_id']} | {item['status']} | "
            f"{'是' if item['readme_first'] else '否'} | {item['core_module_read_rate']} | "
            f"{item['citation_excerpt_support_rate']} | {item['forbidden_path_count']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="只读生成 Project Analyst 自动评测报告")
    parser.add_argument(
        "--career-root",
        default=str(Path(".runtime") / "career"),
        help="Career 数据根目录，默认 .runtime/career",
    )
    parser.add_argument("--project-id", action="append", dest="project_ids", default=[], help="只评测指定项目，可重复")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path, help="可选报告输出文件；不指定则打印到标准输出")
    args = parser.parse_args()
    report = evaluate_project_analysis(args.career_root, project_ids=args.project_ids)
    content = _markdown(report) if args.format == "markdown" else json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
