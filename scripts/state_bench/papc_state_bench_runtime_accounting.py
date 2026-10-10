#!/usr/bin/env python3
"""Public-safe STATE-Bench runtime accounting for baselines and true-live CFRC."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from papc.certspec.state_bench import runtime_accounting  # noqa: E402
from papc.certspec.state_bench.noapi_readiness import (  # noqa: E402
    load_json_object,
    public_leakage_audit,
    write_json_append_only,
)

MODE_BASELINE = "baseline"
MODE_TRUE_LIVE = "true-live"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="accounting_mode", required=True)

    baseline = subparsers.add_parser(MODE_BASELINE)
    baseline.add_argument("--run-name", required=True)
    baseline.add_argument("--model", required=True)
    baseline.add_argument("--reasoning-effort", required=True)
    baseline.add_argument("--trajectory-root", action="append", required=True)
    baseline.add_argument("--run-batch-log", action="append", default=[])
    baseline.add_argument("--price-table-json", type=Path)
    baseline.add_argument("--output-json", type=Path, required=True)

    true_live = subparsers.add_parser(MODE_TRUE_LIVE)
    true_live.add_argument("--true-live-public-report-json", type=Path, required=True)
    true_live.add_argument("--receipt-root", type=Path)
    true_live.add_argument("--method-wall-clock-seconds", type=float)
    true_live.add_argument("--max-concurrency", type=int)
    true_live.add_argument("--price-table-json", type=Path)
    true_live.add_argument("--output-json", type=Path, required=True)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.accounting_mode == MODE_BASELINE:
        report = build_baseline_report(args)
    elif args.accounting_mode == MODE_TRUE_LIVE:
        report = build_true_live_report(args)
    else:  # pragma: no cover - argparse prevents this
        raise SystemExit(f"unsupported accounting mode: {args.accounting_mode}")
    write_json_append_only(args.output_json, report)


def build_baseline_report(args: argparse.Namespace) -> dict[str, Any]:
    return runtime_accounting.build_baseline_accounting_report(
        run_name=args.run_name,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        trajectory_roots=_parse_domain_paths(args.trajectory_root),
        run_batch_logs=_parse_domain_paths(args.run_batch_log),
        price_table=_load_price_table(args.price_table_json),
    )


def build_true_live_report(args: argparse.Namespace) -> dict[str, Any]:
    public_report = load_json_object(args.true_live_public_report_json)
    runtime_ledger = _mapping(_nested(public_report, "summary", "runtime_ledger"))
    max_concurrency = args.max_concurrency or _optional_int(
        _nested(public_report, "parallelism", "max_concurrency")
    )
    row_wall = _load_receipt_wall_seconds(args.receipt_root)
    breakdown = runtime_accounting.true_live_runtime_breakdown(
        runtime_ledger,
        method_wall_clock_seconds=args.method_wall_clock_seconds,
        max_concurrency=max_concurrency,
        per_row_wall_seconds=row_wall,
        role_models=_role_models_from_public_report(public_report),
        price_table=_load_price_table(args.price_table_json),
    )
    report: dict[str, Any] = {
        "schema": runtime_accounting.SCHEMA_TRUE_LIVE,
        "mode": "state_bench_true_live_runtime_accounting",
        "surface": "state_bench",
        "provider_calls": 0,
        "api_spend_performed": False,
        "runtime_breakdown": breakdown,
        "source": {
            "true_live_public_report_read": True,
            "receipt_wall_seconds_rows": len(row_wall),
            "raw_receipts_publicly_disclosed": False,
        },
        "claim_boundary": [
            "Cheap and residual strong model accounting are reported separately.",
            "Agent-side accounting is cheap plus residual strong.",
            "Method total run latency uses per-row receipt wall-clock sum when available; otherwise it falls back to elapsed wall-clock times max concurrency when supplied.",
            "Raw receipts, task ids, provider payloads, and private paths are not public.",
        ],
        "blockers": [],
    }
    report["public_leakage_audit"] = public_leakage_audit(report)
    if report["public_leakage_audit"]["pass"] is not True:
        report["blockers"].append("public_leakage_marker_present")
    return report


def _load_receipt_wall_seconds(root: Path | None) -> list[float]:
    if root is None or not root.is_dir():
        return []
    values: list[float] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(
            payload.get("method_wall_clock_seconds"), int | float
        ):
            values.append(max(0.0, float(payload["method_wall_clock_seconds"])))
    return values


def _load_price_table(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = load_json_object(path)
    return payload if isinstance(payload, dict) else None


def _role_models_from_public_report(public_report: dict[str, Any]) -> dict[str, str]:
    plan = _mapping(public_report.get("model_plan"))
    role_models = {
        "cheap_agent": str(plan.get("cheap_model") or ""),
        "residual_strong": str(plan.get("residual_strong_model") or ""),
        "user_simulator": str(plan.get("user_simulator_model") or ""),
        "evaluator": str(plan.get("evaluator_model") or ""),
    }
    return {role: model for role, model in role_models.items() if model}


def _parse_domain_paths(values: Sequence[str]) -> dict[str, list[Path]]:
    parsed: dict[str, list[Path]] = {}
    for value in values:
        domain, sep, path = str(value).partition(",")
        if not sep:
            raise ValueError(f"expected domain,path: {value}")
        parsed.setdefault(domain.strip(), []).append(Path(path.strip()))
    return parsed


def _nested(mapping: Any, *keys: str) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":  # pragma: no cover
    main()
