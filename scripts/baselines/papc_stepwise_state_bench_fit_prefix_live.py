#!/usr/bin/env python3
"""Stop the shared STATE cheap-fit collector at the StepWise prefix."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from papc.baselines.external_router_fit_dev import StageItemExecutor  # noqa: E402
from papc.baselines.external_router_fit_dev_native import (  # noqa: E402
    build_native_shared_endpoint_executor,
)
from scripts.baselines.papc_external_router_fit_dev_live import (  # noqa: E402
    _write_report_append_only,
    parse_args,
    run,
)
from scripts.baselines.papc_stepwise_state_bench_pilot_gate_noapi import (  # noqa: E402
    DEFAULT_CHEAP_FIT_ITEMS,
    FIT_SELECTED_ROWS,
)

SCHEMA = "papc.stepwise_state_bench_fit_prefix_live.v1"
SHARED_STAGE = "shared_paired_endpoint_fit_dev_and_route_embeddings"
CHEAP_FIT_JOB = (
    "shared_paired_endpoint_fit_dev_and_route_embeddings."
    "shared_endpoint_episode.shared_cheap_endpoint.state_bench.router_fit"
)
# Four fit rows per each of the three STATE domains occur within the first 16
# rows of the frozen registry.  The gate labels only those 12 selected rows.
COLLECT_PREFIX_ROWS = 16

NativeBuilder = Callable[..., StageItemExecutor]


class StatePilotPrefixComplete(RuntimeError):
    """Internal clean stop before the first out-of-prefix provider attempt."""


def bounded_native_builder(
    *,
    inner_builder: NativeBuilder = build_native_shared_endpoint_executor,
    **kwargs: Any,
) -> StageItemExecutor:
    """Wrap the production executor and stop before ordinal 17."""

    inner = inner_builder(**kwargs)

    def execute(job, item, broker):
        if item.ordinal > COLLECT_PREFIX_ROWS:
            raise StatePilotPrefixComplete(
                f"stepwise_state_fit_prefix_complete:{COLLECT_PREFIX_ROWS}"
            )
        return inner(job, item, broker)

    return execute


def run_bounded(argv: list[str] | None = None) -> tuple[int, dict[str, Any]]:
    args = parse_args(argv)
    if (
        args.stage != SHARED_STAGE
        or args.job_id != [CHEAP_FIT_JOB]
        or args.max_concurrency != 1
    ):
        return 2, {
            "schema": SCHEMA,
            "decision": "blocked_before_secret_read",
            "provider_calls": 0,
            "secret_reads": 0,
            "api_spend_performed": False,
            "blockers": ["stepwise_state_fit_prefix_exact_job_required"],
        }
    code, inner = run(
        argv,
        native_executor_builder=bounded_native_builder,
    )
    prefix_count = _checkpoint_item_count(DEFAULT_CHEAP_FIT_ITEMS)
    sentinel = f"stepwise_state_fit_prefix_complete:{COLLECT_PREFIX_ROWS}"
    stopped_cleanly = (
        code == 2
        and prefix_count >= COLLECT_PREFIX_ROWS
        and sentinel in inner.get("blockers", [])
    )
    report = {
        **inner,
        "schema": SCHEMA,
        "inner_schema": inner.get("schema"),
        "selected_fit_rows": FIT_SELECTED_ROWS,
        "required_collection_prefix_rows": COLLECT_PREFIX_ROWS,
        "observed_checkpoint_rows": prefix_count,
        "out_of_prefix_provider_attempts": 0 if stopped_cleanly else None,
        "decision": (
            "state_pilot_prefix_complete"
            if stopped_cleanly
            else inner.get("decision", "blocked")
        ),
        "blockers": [] if stopped_cleanly else inner.get("blockers", []),
    }
    return (0 if stopped_cleanly else code), report


def _checkpoint_item_count(path: Path) -> int:
    if not Path(path).is_file():
        return 0
    records = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records or records[0].get("record_type") != "checkpoint_header":
        raise ValueError("stepwise_state_fit_checkpoint_header_invalid")
    return len(records) - 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code, report = run_bounded(argv)
    if args.report_json is not None:
        _write_report_append_only(args.report_json, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
