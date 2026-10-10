"""Public-safe aggregate for a unified true-live STATE-Bench CFRC run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from papc.certspec.runtime_ledger import RuntimeLedger
from papc.certspec.state_bench import runtime_accounting
from papc.certspec.state_bench.noapi_readiness import (
    load_json_object,
    public_leakage_audit,
    write_json_append_only,
)

SCHEMA = "papc.state_bench_true_live_score_aggregate.v1"
MODE = "state_bench_true_live_score_aggregate"
DECISION_COMPLETE = "state_bench_true_live_score_aggregate_complete"
DECISION_BLOCKED = "blocked_state_bench_true_live_score_aggregate"
CHEAP_HIGH_TARGET_COUNT = 70
STRONG_HIGH_TARGET_COUNT = 92


def build_report(
    *,
    true_live_public_report: Mapping[str, Any] | None = None,
    true_live_public_reports: Sequence[Mapping[str, Any]] | None = None,
    scored_roots: Mapping[str, Sequence[Path | str]],
    output_json: Path | str | None = None,
    price_table: Mapping[str, Any] | None = None,
    cheap_high_target_count: int = CHEAP_HIGH_TARGET_COUNT,
    strong_high_target_count: int = STRONG_HIGH_TARGET_COUNT,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate official scorer outputs for one unified true-live run.

    This is intentionally stricter than overlay accounting: every scored row is
    counted exactly once as the final true-live trajectory, with no replacement
    or selected-row credit.
    """

    blockers: list[str] = []
    source = _summarize_public_reports(
        true_live_public_report=true_live_public_report,
        true_live_public_reports=true_live_public_reports,
    )
    blockers.extend(source["blockers"])
    source_rows = source["source_rows"]
    completed_rows = source["completed_rows"]
    frontier_found_count = source["frontier_found_count"]
    source_provider_calls = source["source_provider_calls"]
    per_role = source["per_role"]
    runtime_breakdown = runtime_accounting.true_live_runtime_breakdown(
        source["runtime_summary"],
        frontier_found_count=frontier_found_count,
        role_models=source["role_models"],
        price_table=price_table,
    )
    method_wall_clock = _map(source.get("method_wall_clock"))
    if method_wall_clock:
        runtime_breakdown["method_wall_clock"] = method_wall_clock

    if source_rows <= 0:
        blockers.append("source_rows_missing")
    if completed_rows and source_rows and completed_rows != source_rows:
        blockers.append("completed_rows_mismatch_source_rows")
    if not per_role:
        blockers.append("runtime_ledger_per_role_missing")

    scored = _collect_scored_roots(scored_roots)
    blockers.extend(scored["blockers"])
    if source_rows and scored["scored_rows"] != source_rows:
        blockers.append("scored_rows_mismatch_source_rows")

    task_pass_count = scored["pass_count"]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "mode": MODE,
        "surface": "state_bench",
        "api_spend_performed": False,
        "no_api_calls": True,
        "provider_calls": 0,
        "source_api_spend_performed": source_provider_calls > 0,
        "summary": {
            "source_rows": source_rows,
            "completed_rows": completed_rows,
            "scored_rows": scored["scored_rows"],
            "unscored_rows": scored["unscored_rows"],
            "task_completion_pass_count": task_pass_count,
            "task_completion_fail_count": scored["fail_count"],
            "state_pass_count": scored["state_pass_count"],
            "task_requirement_pass_count": scored["task_requirement_pass_count"],
            "frontier_found_count": frontier_found_count,
            "cheap_high_target_count": int(cheap_high_target_count),
            "strong_high_target_count": int(strong_high_target_count),
            "delta_vs_cheap_high_target_count": task_pass_count
            - int(cheap_high_target_count),
            "delta_vs_strong_high_target_count": task_pass_count
            - int(strong_high_target_count),
            "reaches_90_task_threshold": task_pass_count >= 90,
            "exceeds_strong_high_target_count": task_pass_count
            > int(strong_high_target_count),
            "source_provider_calls": source_provider_calls,
            "hybrid_replacement_count": 0,
            "hybrid_replacement_supported": False,
            "selected_row_credit_supported": False,
            "mean_ux": scored["mean_ux"],
            "failure_shape_counts": scored["failure_shape_counts"],
            "frontier_outcome_counts": scored["frontier_outcome_counts"],
            "failed_frontier_outcome_counts": scored["failed_frontier_outcome_counts"],
            "failed_candidate_class_counts": scored["failed_candidate_class_counts"],
            "failed_task_requirement_category_counts": scored[
                "failed_task_requirement_category_counts"
            ],
            "staged_precommit_counts": scored["staged_precommit_counts"],
        },
        "domain_summaries": scored["domain_summaries"],
        "runtime": {
            "per_role": per_role,
            "source_provider_calls": source_provider_calls,
        },
        "runtime_breakdown": runtime_breakdown,
        "claim_boundary": [
            "This aggregate summarizes one unified true-live STATE-Bench CFRC run.",
            "Every counted row is an official-scored final trajectory from that run.",
            "No overlay, replacement, receipt swap, or selected-row credit is supported.",
            "Method cost is agent-side only: cheap agent plus residual strong.",
            "User simulator and evaluator costs are benchmark harness accounting, not method cost.",
            "No task ids, private paths, raw provider payloads, or scorer details are public.",
        ],
        "authorizations": {
            "authorizes_unified_true_live_result": not blockers,
            "authorizes_gt90_statement": not blockers and task_pass_count >= 90,
            "authorizes_strong_high_baseline_exceeded_statement": (
                not blockers and task_pass_count > int(strong_high_target_count)
            ),
            "authorizes_protocol_submission_claim": False,
            "authorizes_cost_or_latency_claim": False,
        },
        "blockers": sorted(set(blockers)),
        "output_path": _display_path(Path(output_json)) if output_json else None,
    }
    report["decision"] = (
        DECISION_COMPLETE if not report["blockers"] else DECISION_BLOCKED
    )
    leakage = public_leakage_audit(report)
    report["public_leakage_audit"] = leakage
    if leakage["pass"] is not True:
        report["blockers"] = sorted(
            {*report["blockers"], "public_leakage_marker_present"}
        )
        report["authorizations"]["authorizes_unified_true_live_result"] = False
        report["authorizations"]["authorizes_gt90_statement"] = False
        report["authorizations"][
            "authorizes_strong_high_baseline_exceeded_statement"
        ] = False
        report["decision"] = DECISION_BLOCKED
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--true-live-public-report-json",
        type=Path,
        action="append",
        required=True,
        help="One or more public true-live run reports to combine.",
    )
    parser.add_argument(
        "--scored-root",
        action="append",
        required=True,
        help="domain,official_scored_root",
    )
    parser.add_argument("--price-table-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    public_reports = [
        load_json_object(path) for path in args.true_live_public_report_json
    ]
    report = build_report(
        true_live_public_report=public_reports[0] if len(public_reports) == 1 else None,
        true_live_public_reports=public_reports if len(public_reports) != 1 else None,
        scored_roots=_parse_domain_roots(args.scored_root),
        output_json=args.output_json,
        price_table=(
            load_json_object(args.price_table_json)
            if args.price_table_json is not None
            else None
        ),
    )
    write_json_append_only(args.output_json, report)


def _summarize_public_reports(
    *,
    true_live_public_report: Mapping[str, Any] | None,
    true_live_public_reports: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    reports: list[Mapping[str, Any]] = []
    if true_live_public_reports:
        reports.extend(report for report in true_live_public_reports if report)
    elif true_live_public_report:
        reports.append(true_live_public_report)

    blockers: list[str] = []
    if not reports:
        blockers.append("true_live_public_report_missing")
    source_rows = 0
    completed_rows = 0
    frontier_found_count = 0
    source_provider_calls = 0
    runtime_ledger = RuntimeLedger()
    phase_ledgers: list[Mapping[str, Any]] = []
    method_wall_clocks: list[Mapping[str, Any]] = []
    role_models: dict[str, str] = {}
    for report in reports:
        run_summary = _map(report.get("summary"))
        task_plan = _map(report.get("task_plan"))
        source_rows += _int(task_plan.get("planned_rows")) or _int(
            run_summary.get("completed_rows")
        )
        completed_rows += _int(run_summary.get("completed_rows"))
        frontier_found_count += _int(run_summary.get("frontier_found_count"))
        source_provider_calls += _int(report.get("provider_calls"))
        runtime_public = _map(run_summary.get("runtime_ledger"))
        runtime_ledger.merge_public_summary(runtime_public)
        phase_ledger = _map(runtime_public.get("phase_ledger"))
        if phase_ledger:
            phase_ledgers.append(phase_ledger)
        run_breakdown = _map(run_summary.get("runtime_breakdown"))
        method_wall_clock = _map(run_breakdown.get("method_wall_clock"))
        if method_wall_clock:
            method_wall_clocks.append(method_wall_clock)
        if _list(report.get("blockers")):
            blockers.append("true_live_public_report_blockers_present")
        if _nested(report, "public_leakage_audit", "pass") is not True:
            blockers.append("true_live_public_report_public_leakage_audit_must_pass")
        model_plan = _map(report.get("model_plan"))
        for role, key in (
            ("cheap_agent", "cheap_model"),
            ("residual_strong", "residual_strong_model"),
            ("user_simulator", "user_simulator_model"),
            ("evaluator", "evaluator_model"),
        ):
            model = str(model_plan.get(key) or "").strip()
            if not model:
                continue
            if role in role_models and role_models[role] != model:
                blockers.append(f"source_model_plan_mismatch:{role}")
                continue
            role_models[role] = model
    runtime_summary = runtime_ledger.public_summary()
    runtime_summary["phase_ledger"] = runtime_accounting.merge_phase_ledgers(
        phase_ledgers
    )
    return {
        "blockers": sorted(set(blockers)),
        "completed_rows": completed_rows,
        "frontier_found_count": frontier_found_count,
        "per_role": _map(runtime_summary.get("per_role")),
        "runtime_summary": runtime_summary,
        "role_models": role_models,
        "method_wall_clock": _merge_method_wall_clocks(method_wall_clocks),
        "source_provider_calls": source_provider_calls,
        "source_rows": source_rows,
    }


def _collect_scored_roots(
    roots_by_domain: Mapping[str, Sequence[Path | str]],
) -> dict[str, Any]:
    blockers: list[str] = []
    seen_private_ids: set[str] = set()
    duplicate_private_ids: set[str] = set()
    domain_summaries: list[dict[str, Any]] = []
    scored_rows = 0
    unscored_rows = 0
    pass_count = 0
    fail_count = 0
    state_pass_count = 0
    task_requirement_pass_count = 0
    ux_values: list[float] = []
    failure_shapes: Counter[str] = Counter()
    frontier_outcomes: Counter[str] = Counter()
    failed_frontier_outcomes: Counter[str] = Counter()
    failed_candidate_classes: Counter[str] = Counter()
    failed_requirement_categories: Counter[str] = Counter()
    staged_precommit_counts: Counter[str] = Counter()
    if not roots_by_domain:
        blockers.append("scored_roots_missing")
    for domain, roots in sorted(roots_by_domain.items()):
        domain_scored = 0
        domain_unscored = 0
        domain_pass = 0
        domain_fail = 0
        domain_state_pass = 0
        domain_task_pass = 0
        domain_failure_shapes: Counter[str] = Counter()
        domain_frontier_outcomes: Counter[str] = Counter()
        domain_failed_frontier_outcomes: Counter[str] = Counter()
        domain_failed_requirement_categories: Counter[str] = Counter()
        domain_staged_precommit_counts: Counter[str] = Counter()
        for root in roots:
            files = _iter_scored_files(Path(root))
            if not files:
                blockers.append(f"scored_root_empty:{domain}")
            for path in files:
                payload = _load_json(path)
                private_id = str(payload.get("task_id") or path.stem)
                if private_id in seen_private_ids:
                    duplicate_private_ids.add(private_id)
                seen_private_ids.add(private_id)
                value = payload.get("task_completion_pass")
                if value is None:
                    unscored_rows += 1
                    domain_unscored += 1
                    continue
                scored_rows += 1
                domain_scored += 1
                state_passed = payload.get("state_requirements_met") == 1
                task_requirement_passed = payload.get("task_requirements_met") == 1
                row_staged_counts = _staged_precommit_counts(payload)
                staged_precommit_counts.update(row_staged_counts)
                domain_staged_precommit_counts.update(row_staged_counts)
                frontier_outcome = _frontier_outcome_key(payload)
                frontier_outcomes[frontier_outcome] += 1
                domain_frontier_outcomes[frontier_outcome] += 1
                if value == 1:
                    pass_count += 1
                    domain_pass += 1
                else:
                    fail_count += 1
                    domain_fail += 1
                    shape = _failure_shape(
                        state_passed=state_passed,
                        task_requirement_passed=task_requirement_passed,
                    )
                    failure_shapes[shape] += 1
                    domain_failure_shapes[shape] += 1
                    failed_frontier_outcomes[frontier_outcome] += 1
                    domain_failed_frontier_outcomes[frontier_outcome] += 1
                    candidate_class = _candidate_class(payload)
                    if candidate_class:
                        failed_candidate_classes[candidate_class] += 1
                    for category in _failed_task_requirement_categories(payload):
                        failed_requirement_categories[category] += 1
                        domain_failed_requirement_categories[category] += 1
                if state_passed:
                    state_pass_count += 1
                    domain_state_pass += 1
                if task_requirement_passed:
                    task_requirement_pass_count += 1
                    domain_task_pass += 1
                ux = payload.get("ux_score")
                if isinstance(ux, int | float):
                    ux_values.append(float(ux))
        domain_summaries.append(
            {
                "domain": str(domain),
                "scored_rows": domain_scored,
                "unscored_rows": domain_unscored,
                "pass_count": domain_pass,
                "fail_count": domain_fail,
                "state_pass_count": domain_state_pass,
                "task_requirement_pass_count": domain_task_pass,
                "failure_shape_counts": _counter_json(domain_failure_shapes),
                "frontier_outcome_counts": _counter_json(domain_frontier_outcomes),
                "failed_frontier_outcome_counts": _counter_json(
                    domain_failed_frontier_outcomes
                ),
                "failed_task_requirement_category_counts": _counter_json(
                    domain_failed_requirement_categories
                ),
                "staged_precommit_counts": _counter_json(
                    domain_staged_precommit_counts
                ),
            }
        )
    if unscored_rows:
        blockers.append("scored_root_contains_unscored_rows")
    if duplicate_private_ids:
        blockers.append("duplicate_scored_task_id")
    return {
        "blockers": blockers,
        "domain_summaries": domain_summaries,
        "fail_count": fail_count,
        "mean_ux": round(sum(ux_values) / len(ux_values), 4) if ux_values else None,
        "pass_count": pass_count,
        "failure_shape_counts": _counter_json(failure_shapes),
        "frontier_outcome_counts": _counter_json(frontier_outcomes),
        "failed_frontier_outcome_counts": _counter_json(failed_frontier_outcomes),
        "failed_candidate_class_counts": _counter_json(failed_candidate_classes),
        "failed_task_requirement_category_counts": _counter_json(
            failed_requirement_categories
        ),
        "staged_precommit_counts": _counter_json(staged_precommit_counts),
        "scored_rows": scored_rows,
        "state_pass_count": state_pass_count,
        "task_requirement_pass_count": task_requirement_pass_count,
        "unscored_rows": unscored_rows,
    }


def _failure_shape(*, state_passed: bool, task_requirement_passed: bool) -> str:
    if state_passed and not task_requirement_passed:
        return "task_fail_state_pass_requirement_fail"
    if not state_passed and task_requirement_passed:
        return "task_fail_state_fail_requirement_pass"
    if not state_passed and not task_requirement_passed:
        return "task_fail_state_fail_requirement_fail"
    return "task_fail_state_pass_requirement_pass"


def _staged_precommit_counts(payload: Mapping[str, Any]) -> Counter[str]:
    cfrc = payload.get("papc_true_live_cfrc")
    if not isinstance(cfrc, Mapping):
        return Counter()
    return Counter(
        {
            "preview": int(cfrc.get("staged_precommit_preview_count") or 0),
            "execute": int(cfrc.get("staged_precommit_execute_count") or 0),
            "cancel": int(cfrc.get("staged_precommit_cancel_count") or 0),
        }
    )


def _frontier_outcome_key(payload: Mapping[str, Any]) -> str:
    cfrc = _map(payload.get("papc_true_live_cfrc"))
    if cfrc.get("frontier_found") is not True:
        return "no_frontier"
    compiler_results = _list(cfrc.get("compiler_results"))
    if not compiler_results:
        return "frontier_no_compiler_result"
    last = _map(compiler_results[-1])
    decision = str(last.get("decision") or "unknown")
    errors = [str(item) for item in _list(last.get("compile_errors")) if str(item)]
    if not errors:
        return f"frontier_{decision}"
    return f"frontier_{decision}:{'+'.join(sorted(set(errors)))}"


def _candidate_class(payload: Mapping[str, Any]) -> str:
    cfrc = _map(payload.get("papc_true_live_cfrc"))
    frontier = _map(cfrc.get("frontier"))
    return str(frontier.get("candidate_class") or "").strip()


def _failed_task_requirement_categories(payload: Mapping[str, Any]) -> list[str]:
    if payload.get("task_requirements_met") == 1:
        return []
    details = [_map(item) for item in _list(payload.get("task_requirements_details"))]
    categories: list[str] = []
    for detail in details:
        if detail.get("passed") is not False:
            continue
        categories.append(_task_requirement_category(detail))
    if not categories and payload.get("task_requirements_met") == 0:
        categories.append("task_requirement_fail_no_detail")
    return categories


def _task_requirement_category(detail: Mapping[str, Any]) -> str:
    kind = str(detail.get("kind") or "").lower()
    text = " ".join(
        str(detail.get(key) or "") for key in ("requirement", "reasoning", "evidence")
    ).lower()
    if _looks_pre_action_ordering_failure(text):
        return "pre_action_ordering_or_consent"
    if kind == "must_not" or " must not " in f" {text} ":
        return "forbidden_or_unauthorized_suggestion"
    if any(
        marker in text
        for marker in (
            "insufficient evidence",
            "system record",
            "system-record",
            "verify from system",
            "verified from system",
            "tool calls do not include",
            "does not include a clear",
            "no post-application",
            "does not establish",
        )
    ):
        return "missing_visible_verification"
    if any(
        marker in text
        for marker in (
            "explicitly tell",
            "explicitly told",
            "must explain",
            "explain that",
            "stated",
            "disclose",
            "disclosed",
            "told the customer",
            "did not explicitly",
        )
    ):
        return "missing_required_disclosure"
    if any(
        marker in text
        for marker in (
            "wrong",
            "incorrect",
            "never added",
            "did not add",
            "different",
            "not the requested",
            "not satisfied",
        )
    ):
        return "action_or_value_mismatch"
    return "other_requirement_failure"


def _looks_pre_action_ordering_failure(text: str) -> bool:
    if "before" not in text and "prior to" not in text:
        return False
    ordering_markers = (
        "before executing",
        "before recommending",
        "before issuing",
        "before refunding",
        "before processing",
        "before booking",
        "before cancelling",
        "before canceling",
        "before creating",
        "before updating",
        "before adding",
        "before making",
        "before any prior",
        "before the customer",
        "before the user",
        "asked for consent before",
        "consent before",
        "compare ",
        "comparing ",
    )
    if any(marker in text for marker in ordering_markers):
        return True
    return " after " in text and " before " in text


def _counter_json(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _iter_scored_files(root: Path) -> list[Path]:
    run_root = root / "run1"
    if run_root.is_dir():
        return sorted(run_root.glob("*.json"))
    return sorted(root.glob("*.json"))


def _parse_domain_roots(values: Sequence[str]) -> dict[str, list[Path]]:
    parsed: dict[str, list[Path]] = {}
    for value in values:
        domain, sep, path = str(value).partition(",")
        if not sep:
            raise ValueError(f"expected domain,path: {value}")
        parsed.setdefault(domain.strip(), []).append(Path(path.strip()))
    return parsed


def _merge_method_wall_clocks(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [_map(value) for value in values if _map(value)]
    if not valid:
        return {}

    elapsed = _sum_optional(value.get("elapsed_seconds") for value in valid)
    adjusted = _sum_optional(
        value.get("parallelism_adjusted_seconds") for value in valid
    )
    per_row = _sum_optional(value.get("per_row_wall_seconds_sum") for value in valid)
    max_concurrency_values = [
        _int(value.get("max_concurrency"))
        for value in valid
        if value.get("max_concurrency") is not None
    ]
    max_concurrency = max(max_concurrency_values) if max_concurrency_values else None
    total = (
        per_row
        if per_row is not None
        else adjusted
        if adjusted is not None
        else elapsed
    )
    source = (
        "per_row_wall_seconds_sum"
        if per_row is not None
        else "parallelism_adjusted_wall_clock"
        if adjusted is not None
        else "elapsed_wall_clock"
        if elapsed is not None
        else "unavailable"
    )
    return {
        "elapsed_seconds": elapsed,
        "max_concurrency": max_concurrency,
        "parallelism_adjusted_seconds": adjusted,
        "per_row_wall_seconds_sum": per_row,
        "total_run_latency_seconds": total,
        "total_run_latency_source": source,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _map(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return (
        list(value)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes)
        else []
    )


def _nested(mapping: Any, *keys: str) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _sum_optional(values: Sequence[Any]) -> float | None:
    total = 0.0
    observed = False
    for value in values:
        if value is None:
            continue
        try:
            total += max(0.0, float(value))
        except (TypeError, ValueError):
            continue
        observed = True
    return round(total, 6) if observed else None


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return path.name


if __name__ == "__main__":  # pragma: no cover
    main()
