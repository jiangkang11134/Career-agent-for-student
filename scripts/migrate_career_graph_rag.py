"""Explicitly migrate persisted Career interview knowledge into Graph RAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_agent.career import CareerGraphRAGIndex, InterviewKnowledgeLibrary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--career-root",
        type=Path,
        default=Path(".runtime/career"),
        help="Career persistence root (default: .runtime/career)",
    )
    parser.add_argument(
        "--tenant-id",
        default=None,
        help="Only migrate one tenant; omit to migrate every persisted source",
    )
    args = parser.parse_args()

    career_root = args.career_root.expanduser().resolve()
    library = InterviewKnowledgeLibrary(career_root / "interview-knowledge")
    index = CareerGraphRAGIndex(career_root / "graph-rag")
    sources = library.list_sources(tenant_id=args.tenant_id)
    if not sources:
        raise ValueError("No Career interview knowledge sources matched the migration scope")

    results = index.sync_library(library, tenant_id=args.tenant_id)
    summary = library.metadata_summary(tenant_id=args.tenant_id)
    print(
        json.dumps(
            {
                "career_root": str(career_root),
                "source_count": len(sources),
                "node_count": sum(item.node_count for item in results),
                "edge_count": sum(item.edge_count for item in results),
                "deduplicated_count": sum(item.deduplicated for item in results),
                "ready_source_count": summary["ready_source_count"],
                "pending_source_count": summary["pending_source_count"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
