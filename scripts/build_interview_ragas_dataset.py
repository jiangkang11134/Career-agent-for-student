from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(r"E:\Smart Agent")
SOURCE = ROOT / "竞赛文件" / "面试助手rag测试" / "八股知识库100题.md"
OUTPUT = ROOT / "evaluation" / "interview_ragas_100_cases.jsonl"


def main() -> None:
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"###\s+(interview-foundation-\d+)｜(.+)$", line.strip())
        if heading:
            if current is not None:
                records.append(current)
            current = {"id": heading.group(1), "topic": heading.group(2).strip()}
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("**问题**："):
            current["query"] = stripped.removeprefix("**问题**：").strip()
        elif stripped.startswith("- 文件："):
            current["source_file"] = stripped.removeprefix("- 文件：").strip().strip("`")
        elif stripped.startswith("- 行号："):
            current["source_line"] = stripped.removeprefix("- 行号：").strip()
        elif stripped.startswith("- corpus_id："):
            current["source_corpus_id"] = stripped.removeprefix("- corpus_id：").strip().strip("`")
        elif stripped.startswith("- chunk："):
            chunk_values = re.findall(r"`([^`]+)`", stripped)
            if chunk_values:
                current["source_chunk_number"] = chunk_values[0]
                if len(chunk_values) > 1:
                    current["source_chunk_id"] = chunk_values[1]
        elif stripped.startswith("**参考上下文**："):
            current["ground_truth"] = stripped.removeprefix("**参考上下文**：").strip()
    if current is not None:
        records.append(current)

    if len(records) != 100:
        raise AssertionError(f"解析到 {len(records)} 道题，预期 100")
    required = ("id", "topic", "query", "source_file", "source_line", "source_corpus_id", "source_chunk_id", "ground_truth")
    if any(not all(record.get(key) for key in required) for record in records):
        raise AssertionError("存在缺少字段的题目")
    if len({record["id"] for record in records}) != 100:
        raise AssertionError("题目 ID 不唯一")
    if len({record["query"] for record in records}) != 100:
        raise AssertionError("问题不唯一")

    lines: list[str] = []
    for record in records:
        payload = {
            "id": record["id"],
            "rag": "interview_knowledge",
            "query": record["query"],
            "top_k": 5,
            "relevance": {"groups": [{"chunk_ids": [record["source_chunk_id"]]}]},
            "ground_truth": record["ground_truth"],
            "topic": record["topic"],
            "source_file": record["source_file"],
            "source_line": record["source_line"],
            "source_corpus_id": record["source_corpus_id"],
            "source_chunk_number": record["source_chunk_number"],
            "source_chunk_id": record["source_chunk_id"],
        }
        lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written={OUTPUT}")
    print(f"count={len(records)}")
    print(f"topics={len({record['topic'] for record in records})}")


if __name__ == "__main__":
    main()
