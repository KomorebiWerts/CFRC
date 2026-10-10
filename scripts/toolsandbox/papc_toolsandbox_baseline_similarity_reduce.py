#!/usr/bin/env python3
"""Score recorded ToolSandbox baseline trajectories with the official evaluator.

The baseline accounting runner records each episode's execution context but does
not score it, so a completed run reports token cost and milestone counts without
the official similarity metric. This reducer replays those recorded contexts
through the official evaluator. It reads only what the run already wrote and
makes zero provider calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from papc.certspec.toolsandbox.method_runtime import (  # noqa: E402
    _patch_numpy_compat,
    _prepare_toolsandbox_imports,
    named_scenarios_for_checkout,
    official_evaluation_summary,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.toolsandbox.baseline_similarity_reduction.v1"
DEFAULT_CHECKOUT = Path(f"{DATA_ROOT}/benchmarks/toolsandbox/repo")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--toolsandbox-checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--label", default="")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _prepare_toolsandbox_imports(args.toolsandbox_checkout)
    _patch_numpy_compat()
    from tool_sandbox.common.execution_context import (  # noqa: WPS433
        ExecutionContext,
        set_current_context,
    )
    from tool_sandbox.common.tool_discovery import ToolBackend  # noqa: WPS433

    scenarios = named_scenarios_for_checkout(
        args.toolsandbox_checkout,
        preferred_tool_backend=ToolBackend.DEFAULT,
    )
    rows: list[dict[str, Any]] = []
    for run_root in args.run_root:
        trajectories = Path(run_root) / "raw_tool_sandbox" / "trajectories"
        for episode in sorted(p for p in trajectories.iterdir() if p.is_dir()):
            context_path = episode / "execution_context.json"
            if not context_path.exists():
                continue
            name = episode.name
            scenario = scenarios.get(name)
            if scenario is None:
                rows.append({"scenario": name, "scored": False, "reason": "unknown_scenario"})
                continue
            context = ExecutionContext.from_dict(
                json.loads(context_path.read_text(encoding="utf-8"))
            )
            set_current_context(context)
            result = scenario.evaluation.evaluate(
                execution_context=context,
                max_turn_count=scenario.max_messages,
            )
            summary = official_evaluation_summary(result)
            rows.append({"scenario": name, "scored": True, **dict(summary)})

    scored = [r for r in rows if r.get("scored")]
    def mean(key: str) -> float | None:
        vals = [float(r[key]) for r in scored if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    report = {
        "schema": SCHEMA,
        "label": args.label,
        "provider_calls": 0,
        "episode_count": len(rows),
        "scored_episode_count": len(scored),
        "official_primary_metric": "average_similarity_score",
        "official_avg_similarity_score": mean("similarity"),
        "official_avg_milestone_similarity_score": mean("milestone_similarity"),
        "official_avg_minefield_similarity_score": mean("minefield_similarity"),
        "milestone_complete_count": sum(
            1 for r in scored if float(r.get("milestone_similarity") or 0.0) >= 1.0
        ),
        "strict_complete_count": sum(
            1 for r in scored if float(r.get("similarity") or 0.0) >= 1.0
        ),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({**report, "rows": rows}, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
