from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smart_agent.evaluation import run_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Smart Agent ablations.")
    parser.add_argument("--dataset", default="fixtures/ablation/competition_v1.json")
    parser.add_argument("--output", default="docs/evaluation/results/ablation-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--prompt-version", default="offline-baseline-v1")
    parser.add_argument("--model-id", default="deterministic-offline")
    args = parser.parse_args()

    report = run_ablation(
        Path(args.dataset),
        Path(args.output),
        seed=args.seed,
        prompt_version=args.prompt_version,
        model_id=args.model_id,
    )
    print(json.dumps(report.aggregate, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps({"cross_cutting": report.cross_cutting}, ensure_ascii=False, indent=2, sort_keys=True))
    if report.failures:
        print(f"Ablation completed with {len(report.failures)} failed variant cases.")
        return 1
    print(f"Ablation completed: {len(report.records)} variant cases, output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
