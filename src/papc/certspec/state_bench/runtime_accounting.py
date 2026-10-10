"""STATE-Bench public-safe runtime accounting helpers.

This module keeps three latency notions separate:

* provider latency: sum of model-provider call durations;
* run wall clock: elapsed batch wall time from run logs or runner reports;
* parallelism-adjusted wall clock: batch wall clock multiplied by workers.

It never treats missing cost or latency as zero.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from papc.certspec.costing import (
    PUBLIC_MODEL_PRICES_PER_1M,
    public_pricing_metadata_for_model,
    reasoning_tokens_included_in_output,
)
from papc.certspec.provider import reasoning_tokens_from_usage
from papc.certspec.state_bench.noapi_readiness import public_leakage_audit

SCHEMA_BASELINE = "papc.state_bench_baseline_runtime_accounting.v1"
SCHEMA_TRUE_LIVE = "papc.state_bench_true_live_runtime_accounting.v1"

_RUN_WALL_RE = re.compile(r"All runs complete in\s+([0-9]+(?:\.[0-9]+)?)s")
_WORKERS_RE = re.compile(r"([0-9]+)\s+workers\b")
_DEFAULT_ROLE_MODELS = {
    "cheap_agent": "gpt-5.4-mini",
    "cheap_full_agent": "gpt-5.4-mini",
    "residual_strong": "gpt-5.5",
    "strong_full_agent": "gpt-5.5",
}


@dataclass
class RuntimeTotals:
    rows: int = 0
    passed: int = 0
    failed: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    token_rows: int = 0
    provider_calls: int = 0
    provider_call_rows: int = 0
    provider_latency_seconds: float = 0.0
    provider_latency_rows: int = 0
    agent_wall_seconds: float = 0.0
    agent_wall_rows: int = 0
    cost_usd: float = 0.0
    cost_rows: int = 0

    def add_trajectory(self, payload: Mapping[str, Any]) -> None:
        self.rows += 1
        passed = payload.get("task_completion_pass")
        if passed == 1:
            self.passed += 1
        elif passed is not None:
            self.failed += 1

        ledger = _map(payload.get("papc_runtime_ledger"))
        usage = _map(ledger.get("token_usage"))
        if usage:
            self._add_tokens(
                input_tokens=_optional_int(usage.get("input_tokens")),
                cached_input_tokens=_optional_int(usage.get("cached_input_tokens")),
                output_tokens=_optional_int(usage.get("output_tokens")),
                reasoning_tokens=reasoning_tokens_from_usage(usage),
                total_tokens=_optional_int(usage.get("total_tokens")),
            )
        else:
            self._add_tokens(
                input_tokens=_optional_int(payload.get("input_tokens")),
                cached_input_tokens=_optional_int(payload.get("cached_input_tokens")),
                output_tokens=_optional_int(payload.get("output_tokens")),
                reasoning_tokens=reasoning_tokens_from_usage(payload),
                total_tokens=_optional_int(payload.get("total_tokens")),
            )

        calls = _optional_int(ledger.get("calls"))
        if calls is not None:
            self.provider_calls += max(0, calls)
            self.provider_call_rows += 1
        provider_latency = _optional_float(ledger.get("provider_latency_seconds"))
        if provider_latency is not None:
            self.provider_latency_seconds += max(0.0, provider_latency)
            self.provider_latency_rows += 1
        agent_wall = _optional_float(ledger.get("agent_wall_seconds"))
        if agent_wall is not None:
            self.agent_wall_seconds += max(0.0, agent_wall)
            self.agent_wall_rows += 1

        if payload.get("agent_pricing") and isinstance(
            payload.get("cost_usd"), int | float
        ):
            self.cost_usd += max(0.0, float(payload["cost_usd"]))
            self.cost_rows += 1

    def _add_tokens(
        self,
        *,
        input_tokens: int | None,
        cached_input_tokens: int | None,
        output_tokens: int | None,
        reasoning_tokens: int | None,
        total_tokens: int | None,
    ) -> None:
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return
        input_value = max(0, int(input_tokens or 0))
        output_value = max(0, int(output_tokens or 0))
        self.input_tokens += input_value
        self.cached_input_tokens += max(0, int(cached_input_tokens or 0))
        self.output_tokens += output_value
        self.reasoning_tokens += max(0, int(reasoning_tokens or 0))
        self.total_tokens += max(
            0,
            int(
                total_tokens if total_tokens is not None else input_value + output_value
            ),
        )
        self.token_rows += 1

    def public(self) -> dict[str, Any]:
        has_tokens = self.token_rows > 0
        has_calls = self.provider_call_rows > 0
        has_provider_latency = self.provider_latency_rows > 0
        has_agent_wall = self.agent_wall_rows > 0
        has_cost = self.cost_rows > 0
        return {
            "rows": self.rows,
            "task_completion_pass_count": self.passed,
            "task_completion_fail_count": self.failed,
            "calls": self.provider_calls if has_calls else None,
            "provider_calls_available": has_calls
            and self.provider_call_rows == self.rows,
            "provider_call_rows": self.provider_call_rows,
            "input_tokens": self.input_tokens if has_tokens else None,
            "cached_input_tokens": self.cached_input_tokens if has_tokens else None,
            "billable_input_tokens": (
                max(0, self.input_tokens - self.cached_input_tokens)
                if has_tokens
                else None
            ),
            "output_tokens": self.output_tokens if has_tokens else None,
            "reasoning_tokens": self.reasoning_tokens if has_tokens else None,
            "total_tokens": self.total_tokens if has_tokens else None,
            "token_usage_available": has_tokens and self.token_rows == self.rows,
            "token_rows": self.token_rows,
            "provider_latency_seconds": (
                round(self.provider_latency_seconds, 6)
                if has_provider_latency
                else None
            ),
            "provider_latency_available": (
                has_provider_latency and self.provider_latency_rows == self.rows
            ),
            "provider_latency_rows": self.provider_latency_rows,
            "agent_wall_seconds": round(self.agent_wall_seconds, 6)
            if has_agent_wall
            else None,
            "agent_wall_available": has_agent_wall
            and self.agent_wall_rows == self.rows,
            "agent_wall_rows": self.agent_wall_rows,
            "cost_usd": round(self.cost_usd, 6) if has_cost else None,
            "cost_available": has_cost and self.cost_rows == self.rows,
            "cost_rows": self.cost_rows,
            "cost_unavailable_count": self.rows - self.cost_rows,
        }


def build_baseline_accounting_report(
    *,
    run_name: str,
    model: str,
    reasoning_effort: str,
    trajectory_roots: Mapping[str, Sequence[Path | str]],
    run_batch_logs: Mapping[str, Sequence[Path | str]] | None = None,
    price_table: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    totals = RuntimeTotals()
    domain_summaries: list[dict[str, Any]] = []
    blockers: list[str] = []
    for domain, roots in sorted(trajectory_roots.items()):
        domain_totals = RuntimeTotals()
        for root in roots:
            files = _iter_json_files(Path(root))
            if not files:
                blockers.append(f"trajectory_root_empty:{domain}")
            for path in files:
                payload = _load_json(path)
                if not payload:
                    blockers.append(f"trajectory_unreadable:{domain}")
                    continue
                domain_totals.add_trajectory(payload)
                totals.add_trajectory(payload)
        domain_summaries.append(
            {"domain": domain, "agent_runtime": domain_totals.public()}
        )

    wall_clock = summarize_run_batch_logs(run_batch_logs or {})
    totals_public = totals.public()
    baseline_role = _baseline_role_for_model(model)
    report: dict[str, Any] = {
        "schema": SCHEMA_BASELINE,
        "mode": "state_bench_baseline_runtime_accounting",
        "surface": "state_bench",
        "run_name": str(run_name),
        "model_plan": {
            "agent_model": str(model),
            "agent_reasoning_effort": str(reasoning_effort),
        },
        "summary": {
            "rows": totals.rows,
            "task_completion_pass_count": totals.passed,
            "task_completion_fail_count": totals.failed,
        },
        "agent_runtime": {
            "totals": totals_public,
            "by_domain": domain_summaries,
        },
        "primary_resource_comparison": _primary_resource_comparison(
            {
                baseline_role: totals_public,
            },
            baseline_role=baseline_role,
            role_models={baseline_role: str(model)},
            price_table=price_table,
        ),
        "run_wall_clock": wall_clock,
        "latency_notes": {
            "provider_latency_is_sum_of_model_calls": True,
            "run_wall_clock_is_parallel_batch_elapsed": True,
            "parallelism_adjusted_seconds_is_wall_clock_times_workers": True,
            "provider_latency_and_run_wall_clock_are_not_interchangeable": True,
        },
        "claim_boundary": [
            "Provider latency is summed over model calls and may exceed elapsed wall-clock under parallel execution.",
            "Run wall-clock is parsed from batch logs when available.",
            "Parallelism-adjusted wall time is wall-clock multiplied by reported workers; it is a worker-slot estimate, not provider latency.",
            "Raw agent_runtime cost is unavailable unless rows include explicit pricing metadata; primary_resource_comparison may compute a public-pricing estimate from token counts.",
            "A row-level cost_usd=0 without pricing metadata is not treated as a free model call.",
        ],
        "blockers": sorted(set(blockers)),
    }
    report["public_leakage_audit"] = public_leakage_audit(report)
    if report["public_leakage_audit"]["pass"] is not True:
        report["blockers"] = sorted(
            {*report["blockers"], "public_leakage_marker_present"}
        )
    return report


def summarize_run_batch_logs(
    logs_by_domain: Mapping[str, Sequence[Path | str]],
) -> dict[str, Any]:
    seconds = 0.0
    adjusted = 0.0
    log_count = 0
    max_workers: int | None = None
    missing_count = 0
    domain_summaries: list[dict[str, Any]] = []
    for domain, logs in sorted(logs_by_domain.items()):
        domain_seconds = 0.0
        domain_adjusted = 0.0
        domain_logs = 0
        for log in logs:
            path = Path(log)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                missing_count += 1
                continue
            wall = _parse_run_wall_seconds(text)
            workers = _parse_worker_count(text)
            if wall is None:
                missing_count += 1
                continue
            worker_count = workers or 1
            seconds += wall
            adjusted += wall * worker_count
            domain_seconds += wall
            domain_adjusted += wall * worker_count
            domain_logs += 1
            log_count += 1
            max_workers = (
                worker_count if max_workers is None else max(max_workers, worker_count)
            )
        domain_summaries.append(
            {
                "domain": domain,
                "log_count": domain_logs,
                "seconds": round(domain_seconds, 6) if domain_logs else None,
                "parallelism_adjusted_seconds": (
                    round(domain_adjusted, 6) if domain_logs else None
                ),
            }
        )
    return {
        "seconds": round(seconds, 6) if log_count else None,
        "max_workers": max_workers,
        "parallelism_adjusted_seconds": round(adjusted, 6) if log_count else None,
        "log_count": log_count,
        "missing_or_unparsed_log_count": missing_count,
        "by_domain": domain_summaries,
    }


def true_live_runtime_breakdown(
    runtime_ledger: Mapping[str, Any],
    *,
    method_wall_clock_seconds: float | None = None,
    max_concurrency: int | None = None,
    per_row_wall_seconds: Sequence[float] | None = None,
    frontier_found_count: int | None = None,
    role_models: Mapping[str, str] | None = None,
    price_table: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    per_role = _map(runtime_ledger.get("per_role"))
    cheap = _role_summary(per_role.get("cheap"))
    strong = _role_summary(per_role.get("strong"))
    simulator = _role_summary(per_role.get("user_simulator"))
    evaluator = _role_summary(per_role.get("evaluator"))
    full_runtime = _role_summary(runtime_ledger.get("totals"))
    if not full_runtime["observed"]:
        full_runtime = _sum_roles(cheap, strong, simulator, evaluator)
    agent_side = _sum_roles(cheap, strong)
    row_wall = [max(0.0, float(value)) for value in (per_row_wall_seconds or [])]
    row_wall_total = round(sum(row_wall), 6) if row_wall else None
    concurrency = max(1, int(max_concurrency)) if max_concurrency else None
    adjusted = (
        round(float(method_wall_clock_seconds) * concurrency, 6)
        if method_wall_clock_seconds is not None and concurrency is not None
        else None
    )
    method_total_latency = row_wall_total or adjusted or method_wall_clock_seconds
    strong_phase_breakdown = _strong_phase_breakdown(runtime_ledger.get("phase_ledger"))
    primary_resource_comparison = _primary_resource_comparison(
        {
            "cheap_agent": cheap,
            "residual_strong": strong,
            "user_simulator": simulator,
            "evaluator": evaluator,
            "agent_side": agent_side,
            "full_runtime": full_runtime,
        },
        role_models=role_models,
        price_table=price_table,
    )
    _fill_aggregate_resource_costs(primary_resource_comparison)
    return {
        "schema": SCHEMA_TRUE_LIVE,
        "cheap": cheap,
        "strong": strong,
        "user_simulator": simulator,
        "evaluator": evaluator,
        "agent_side": agent_side,
        "full_runtime": full_runtime,
        "strong_phase_breakdown": strong_phase_breakdown,
        "resource_efficiency": _resource_efficiency(
            strong=strong,
            frontier_found_count=frontier_found_count,
        ),
        "primary_resource_comparison": primary_resource_comparison,
        "model_provider_latency": {
            "cheap_seconds": cheap["latency_seconds"],
            "strong_seconds": strong["latency_seconds"],
            "agent_side_seconds": agent_side["latency_seconds"],
            "user_simulator_seconds": simulator["latency_seconds"],
            "full_runtime_seconds": full_runtime["latency_seconds"],
        },
        "method_wall_clock": {
            "elapsed_seconds": method_wall_clock_seconds,
            "max_concurrency": concurrency,
            "parallelism_adjusted_seconds": adjusted,
            "per_row_wall_seconds_sum": row_wall_total,
            "total_run_latency_seconds": method_total_latency,
            "total_run_latency_source": (
                "per_row_wall_seconds_sum"
                if row_wall_total is not None
                else "parallelism_adjusted_wall_clock"
                if adjusted is not None
                else "elapsed_wall_clock"
                if method_wall_clock_seconds is not None
                else "unavailable"
            ),
        },
        "claim_boundary": [
            "Method cost is agent-side accounting only: cheap agent plus residual strong.",
            "User simulator and evaluator are benchmark harness costs, not method costs.",
            "Full runtime accounting is harness-inclusive and is reported only as secondary execution accounting.",
            "Model-provider latency is not the same as elapsed method wall-clock.",
            "Parallelism-adjusted wall-clock multiplies elapsed wall-clock by max concurrency when per-row wall-clock is unavailable.",
        ],
    }


def _resource_efficiency(
    *,
    strong: Mapping[str, Any],
    frontier_found_count: int | None,
) -> dict[str, Any]:
    strong_calls = _int(strong.get("calls"))
    frontiers = _int(frontier_found_count)
    return {
        "frontier_found_count": frontiers if frontier_found_count is not None else None,
        "residual_strong_input_tokens_per_call": _divide_optional(
            _optional_int(strong.get("input_tokens")),
            strong_calls,
        ),
        "residual_strong_total_tokens_per_call": _divide_optional(
            _optional_int(strong.get("total_tokens")),
            strong_calls,
        ),
        "residual_strong_output_tokens_per_call": _divide_optional(
            _optional_int(strong.get("output_tokens")),
            strong_calls,
        ),
        "residual_strong_input_tokens_per_frontier": _divide_optional(
            _optional_int(strong.get("input_tokens")),
            frontiers,
        ),
        "residual_strong_total_tokens_per_frontier": _divide_optional(
            _optional_int(strong.get("total_tokens")),
            frontiers,
        ),
    }


def _sum_roles(*roles: Mapping[str, Any]) -> dict[str, Any]:
    observed_roles = [role for role in roles if bool(role.get("observed"))]
    if not observed_roles:
        return _empty_role()
    latency_values = [role.get("latency_seconds") for role in observed_roles]
    cost_values = [role.get("cost_usd") for role in observed_roles]
    cached_input_values = [
        _optional_int(role.get("cached_input_tokens")) for role in observed_roles
    ]
    billable_input_values = [
        _optional_int(role.get("billable_input_tokens")) for role in observed_roles
    ]
    cost_available = all(bool(role.get("cost_available")) for role in observed_roles)
    return {
        "observed": True,
        "calls": sum(_int(role.get("calls")) for role in observed_roles),
        "input_tokens": sum(_int(role.get("input_tokens")) for role in observed_roles),
        "cached_input_tokens": (
            sum(int(value) for value in cached_input_values if value is not None)
            if all(value is not None for value in cached_input_values)
            else None
        ),
        "billable_input_tokens": (
            sum(int(value) for value in billable_input_values if value is not None)
            if all(value is not None for value in billable_input_values)
            else None
        ),
        "output_tokens": sum(
            _int(role.get("output_tokens")) for role in observed_roles
        ),
        "reasoning_tokens": sum(
            _int(role.get("reasoning_tokens")) for role in observed_roles
        ),
        "total_tokens": sum(_int(role.get("total_tokens")) for role in observed_roles),
        "latency_seconds": (
            round(
                sum(
                    float(value)
                    for value in latency_values
                    if isinstance(value, int | float)
                ),
                6,
            )
            if any(isinstance(value, int | float) for value in latency_values)
            else None
        ),
        "cost_usd": (
            round(
                sum(
                    float(value)
                    for value in cost_values
                    if isinstance(value, int | float)
                ),
                6,
            )
            if all(isinstance(value, int | float) for value in cost_values)
            else None
        ),
        "cost_available": cost_available,
    }


def _role_summary(value: Any) -> dict[str, Any]:
    mapping = _map(value)
    if not mapping or _int(mapping.get("calls")) <= 0:
        return _empty_role()
    cached_input_tokens = _optional_int(mapping.get("cached_input_tokens"))
    billable_input_tokens = _optional_int(mapping.get("billable_input_tokens"))
    if billable_input_tokens is None and cached_input_tokens is not None:
        billable_input_tokens = _billable_input_tokens(mapping)
    return {
        "observed": True,
        "calls": _int(mapping.get("calls")),
        "input_tokens": _optional_int(mapping.get("input_tokens")),
        "cached_input_tokens": cached_input_tokens,
        "billable_input_tokens": billable_input_tokens,
        "output_tokens": _optional_int(mapping.get("output_tokens")),
        "reasoning_tokens": _optional_int(mapping.get("reasoning_tokens")),
        "total_tokens": _optional_int(mapping.get("total_tokens")),
        "latency_seconds": _optional_float(mapping.get("latency_seconds")),
        "cost_usd": _optional_float(mapping.get("cost_usd")),
        "cost_available": bool(mapping.get("cost_available")),
    }


def _empty_role() -> dict[str, Any]:
    return {
        "observed": False,
        "calls": 0,
        "input_tokens": None,
        "cached_input_tokens": None,
        "billable_input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "latency_seconds": None,
        "cost_usd": None,
        "cost_available": False,
    }


def merge_phase_ledgers(
    phase_ledgers: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any]:
    totals: dict[str, dict[str, Any]] = {}
    for ledger in phase_ledgers:
        for phase, summary in _map(ledger).items():
            summary_map = _map(summary)
            if not summary_map:
                continue
            bucket = totals.setdefault(
                str(phase),
                {
                    "calls": 0,
                    "ok_calls": 0,
                    "error_calls": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "cached_input_usage_call_count": 0,
                    "cached_input_usage_unavailable_count": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                    "latency_seconds": 0.0,
                    "cost_usd": 0.0,
                    "cost_available": True,
                    "_has_latency": False,
                    "_has_cost": False,
                },
            )
            calls = _int(summary_map.get("calls"))
            error_calls = _optional_int(summary_map.get("error_calls"))
            ok_calls = _optional_int(summary_map.get("ok_calls"))
            if error_calls is None and ok_calls is None:
                error_calls = 0
                ok_calls = calls
            elif error_calls is None:
                error_calls = max(0, calls - int(ok_calls or 0))
            elif ok_calls is None:
                ok_calls = max(0, calls - int(error_calls or 0))
            bucket["calls"] += calls
            bucket["ok_calls"] += max(0, int(ok_calls or 0))
            bucket["error_calls"] += max(0, int(error_calls or 0))
            bucket["input_tokens"] += _int(summary_map.get("input_tokens"))
            bucket["cached_input_tokens"] += _int(
                summary_map.get("cached_input_tokens")
            )
            if "cached_input_usage_call_count" in summary_map:
                bucket["cached_input_usage_call_count"] += _int(
                    summary_map.get("cached_input_usage_call_count")
                )
                bucket["cached_input_usage_unavailable_count"] += _int(
                    summary_map.get("cached_input_usage_unavailable_count")
                )
            elif _int(summary_map.get("cached_input_tokens")) > 0:
                bucket["cached_input_usage_call_count"] += calls
            else:
                bucket["cached_input_usage_unavailable_count"] += calls
            bucket["output_tokens"] += _int(summary_map.get("output_tokens"))
            bucket["reasoning_tokens"] += _int(summary_map.get("reasoning_tokens"))
            bucket["total_tokens"] += _int(summary_map.get("total_tokens"))
            latency = _optional_float(summary_map.get("latency_seconds"))
            if latency is not None:
                bucket["latency_seconds"] += latency
                bucket["_has_latency"] = True
            cost = _optional_float(summary_map.get("cost_usd"))
            if cost is not None:
                bucket["cost_usd"] += cost
                bucket["_has_cost"] = True
            else:
                bucket["cost_available"] = False
    return {
        phase: _phase_public_summary(summary)
        for phase, summary in sorted(totals.items())
        if _int(summary.get("calls")) > 0
    }


def _phase_public_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    input_tokens = _int(summary.get("input_tokens"))
    cached_input_usage_call_count = _int(summary.get("cached_input_usage_call_count"))
    cached_input_usage_unavailable_count = _int(
        summary.get("cached_input_usage_unavailable_count")
    )
    cached_input_usage_available = (
        cached_input_usage_call_count > 0 and cached_input_usage_unavailable_count == 0
    )
    cached_input_tokens = (
        min(input_tokens, _int(summary.get("cached_input_tokens")))
        if cached_input_usage_available
        else None
    )
    output_tokens = _int(summary.get("output_tokens"))
    total_tokens = _int(summary.get("total_tokens")) or input_tokens + output_tokens
    has_latency = bool(summary.get("_has_latency"))
    has_cost = bool(summary.get("_has_cost"))
    return {
        "calls": _int(summary.get("calls")),
        "ok_calls": _int(summary.get("ok_calls")),
        "error_calls": _int(summary.get("error_calls")),
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "billable_input_tokens": (
            max(0, input_tokens - cached_input_tokens)
            if cached_input_tokens is not None
            else None
        ),
        "cached_input_usage_available": cached_input_usage_available,
        "cached_input_usage_call_count": cached_input_usage_call_count,
        "cached_input_usage_unavailable_count": (cached_input_usage_unavailable_count),
        "output_tokens": output_tokens,
        "reasoning_tokens": _int(summary.get("reasoning_tokens")),
        "total_tokens": total_tokens,
        "latency_seconds": (
            round(float(summary.get("latency_seconds") or 0.0), 6)
            if has_latency
            else None
        ),
        "cost_usd": (
            round(float(summary.get("cost_usd") or 0.0), 6) if has_cost else None
        ),
        "cost_available": bool(summary.get("cost_available")) and has_cost,
    }


def _strong_phase_breakdown(value: Any) -> dict[str, Any]:
    merged = merge_phase_ledgers([_map(value)])
    return {
        phase: summary
        for phase, summary in merged.items()
        if phase.startswith("strong_")
    }


def _primary_resource_comparison(
    roles: Mapping[str, Mapping[str, Any]],
    *,
    baseline_role: str | None = None,
    role_models: Mapping[str, str] | None = None,
    price_table: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    explicit_price_table = price_table is not None
    effective_price_table = _effective_price_table(price_table)
    default_pricing_sources = (
        _default_pricing_sources(roles, role_models)
        if not explicit_price_table
        else []
    )
    return {
        "schema": "papc.state_bench_primary_resource_comparison.v1",
        "baseline_role": baseline_role,
        "method_cost_role": "agent_side",
        "method_cost_roles": ["cheap_agent", "residual_strong"],
        "harness_runtime_roles": ["user_simulator", "evaluator"],
        "harness_inclusive_role": "full_runtime",
        "user_simulator_is_method_cost": False,
        "evaluator_is_method_cost": False,
        **_reasoning_output_containment(roles, role_models),
        "cache_discount_applied": False,
        "cached_input_metadata_priced_at_standard_input_rate": True,
        "price_table_supplied": explicit_price_table,
        "default_price_table_used": not explicit_price_table,
        "default_price_table_source": (
            default_pricing_sources[0]["source"]
            if len(default_pricing_sources) == 1
            else None
        ),
        "default_price_table_snapshot_date": (
            default_pricing_sources[0]["snapshot_date"]
            if len(default_pricing_sources) == 1
            else None
        ),
        "default_price_table_sources": default_pricing_sources,
        "roles": {
            name: _resource_row(
                summary,
                model=_resource_model_for_role(name, role_models),
                price_table=effective_price_table,
                default_public_pricing=not explicit_price_table,
            )
            for name, summary in sorted(roles.items())
        },
    }


def _resource_row(
    summary: Mapping[str, Any],
    *,
    model: str = "",
    price_table: Mapping[str, Any] | None = None,
    default_public_pricing: bool = False,
) -> dict[str, Any]:
    input_tokens = _optional_int(summary.get("input_tokens"))
    cached_input_tokens = _optional_int(summary.get("cached_input_tokens"))
    billable_input_tokens = _optional_int(summary.get("billable_input_tokens"))
    if (
        billable_input_tokens is None
        and input_tokens is not None
        and cached_input_tokens is not None
    ):
        billable_input_tokens = max(0, input_tokens - cached_input_tokens)
    computed_cost = _cost_from_price_table(
        model=model,
        input_tokens=input_tokens,
        output_tokens=_optional_int(summary.get("output_tokens")),
        reasoning_tokens=_optional_int(summary.get("reasoning_tokens")),
        price_table=price_table,
    )
    pricing_source, pricing_snapshot_date = (
        public_pricing_metadata_for_model(model)
        if default_public_pricing
        else (None, None)
    )
    return {
        "model": str(model or "") or None,
        "calls": _optional_int(summary.get("calls")),
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "billable_input_tokens": billable_input_tokens,
        "output_tokens": _optional_int(summary.get("output_tokens")),
        "reasoning_tokens": _optional_int(summary.get("reasoning_tokens")),
        "total_tokens": _optional_int(summary.get("total_tokens")),
        "provider_call_latency_seconds": _latency_from_summary(summary),
        # The primary comparison prices every input token at the standard input
        # rate. Cache metadata is retained for audit only and earns no discount.
        "cost_usd": computed_cost,
        "cost_available": computed_cost is not None,
        "pricing_source": pricing_source,
        "pricing_snapshot_date": pricing_snapshot_date,
    }


def compute_role_cost_usd(
    summary: Mapping[str, Any],
    *,
    model: str,
    price_table: Mapping[str, Any] | None = None,
) -> float | None:
    """Role cost using total input and output with no cache discount.

    Reasoning tokens are priced only for vendors that report them outside the
    output count and bill them at the output rate; where a vendor's output
    already contains reasoning, that count is used as is. Every input token
    uses the ordinary input price; cached-input metadata is reported separately
    and never changes the primary comparison cost. Falls back to the shared
    public price snapshot when no explicit price table is supplied.
    """

    input_tokens = _optional_int(summary.get("input_tokens"))
    return _cost_from_price_table(
        model=model,
        input_tokens=input_tokens,
        output_tokens=_optional_int(summary.get("output_tokens")),
        reasoning_tokens=_optional_int(summary.get("reasoning_tokens")),
        price_table=_effective_price_table(price_table),
    )


def _effective_price_table(price_table: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if price_table is not None:
        return price_table
    return PUBLIC_MODEL_PRICES_PER_1M


def _reasoning_output_containment(
    roles: Mapping[str, Mapping[str, Any]],
    role_models: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Report whether reasoning is already inside each role's output count.

    This used to be a hardcoded ``True``. It holds for OpenAI-style reporting
    but not for Gemini, which counts reasoning outside its completion total and
    bills it at the output rate, so asserting containment priced those arms as
    if the model had produced no reasoning at all.
    """

    billed_on_top: list[str] = []
    undeclared: list[str] = []
    for role_name in sorted(roles):
        model = _resource_model_for_role(role_name, role_models)
        if not model:
            # Aggregate rows and roles the caller did not name a model for are
            # priced through their constituents, so they declare nothing here.
            continue
        included = reasoning_tokens_included_in_output(model)
        if included is None:
            undeclared.append(role_name)
        elif not included:
            billed_on_top.append(role_name)
    subset = None if undeclared else not billed_on_top
    return {
        "reasoning_tokens_are_output_subset": subset,
        "reasoning_tokens_billed_on_top_of_output_roles": billed_on_top,
        "reasoning_containment_undeclared_roles": undeclared,
        "cost_basis": (
            "total_input_plus_output_no_cache_discount"
            if subset is True
            else "total_input_plus_output_and_uncontained_reasoning_no_cache_discount"
        ),
    }


def _default_pricing_sources(
    roles: Mapping[str, Mapping[str, Any]],
    role_models: Mapping[str, str] | None,
) -> list[dict[str, str]]:
    sources: set[tuple[str, str]] = set()
    for role_name in roles:
        model = _resource_model_for_role(role_name, role_models)
        source, snapshot_date = public_pricing_metadata_for_model(model)
        if source is not None and snapshot_date is not None:
            sources.add((source, snapshot_date))
    return [
        {"source": source, "snapshot_date": snapshot_date}
        for source, snapshot_date in sorted(sources)
    ]


def _resource_model_for_role(
    role_name: str,
    role_models: Mapping[str, str] | None,
) -> str:
    explicit = str(_map(role_models).get(role_name) or "").strip()
    if explicit:
        return explicit
    return _DEFAULT_ROLE_MODELS.get(str(role_name), "")


def _cost_from_price_table(
    *,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    price_table: Mapping[str, Any] | None,
    reasoning_tokens: int | None = None,
) -> float | None:
    table = _map(price_table)
    prices = _map(table.get(model))
    if not prices:
        normalized = model.rsplit("/", 1)[-1].lower()
        for name, candidate in sorted(
            table.items(),
            key=lambda item: len(str(item[0])),
            reverse=True,
        ):
            if normalized.startswith(str(name).lower()):
                prices = _map(candidate)
                break
    if not prices:
        return None
    input_price = _optional_float(
        prices.get("input_per_million") or prices.get("input_price_per_1m")
    )
    output_price = _optional_float(
        prices.get("output_per_million") or prices.get("output_price_per_1m")
    )
    if input_price is None or output_price is None:
        return None
    if input_tokens is None or output_tokens is None:
        return None
    input_value = int(input_tokens or 0)
    output = int(output_tokens or 0)
    # Gemini reports reasoning outside its output count and bills it at the
    # output rate, so that part has to be added back before pricing. Vendors
    # whose output already contains reasoning add nothing here.
    if reasoning_tokens is not None and (
        reasoning_tokens_included_in_output(model) is False
    ):
        output += max(0, int(reasoning_tokens))
    cost = (input_value * input_price + output * output_price) / 1_000_000.0
    return round(cost, 6)


def _fill_aggregate_resource_costs(comparison: Mapping[str, Any]) -> None:
    roles = comparison.get("roles") if isinstance(comparison, Mapping) else None
    if not isinstance(roles, dict):
        return
    _sum_resource_cost(roles, "agent_side", ("cheap_agent", "residual_strong"))
    _sum_resource_cost(
        roles,
        "full_runtime",
        ("cheap_agent", "residual_strong", "user_simulator", "evaluator"),
    )


def _sum_resource_cost(
    roles: dict[str, Any],
    target: str,
    sources: Sequence[str],
) -> None:
    target_row = roles.get(target)
    if not isinstance(target_row, dict):
        return
    source_rows = [roles.get(source) for source in sources]
    observed = [
        row
        for row in source_rows
        if isinstance(row, Mapping) and _int(row.get("calls")) > 0
    ]
    if not observed:
        return
    if not all(bool(row.get("cost_available")) for row in observed):
        return
    target_row["cost_usd"] = round(
        sum(float(row.get("cost_usd") or 0.0) for row in observed),
        6,
    )
    target_row["cost_available"] = True


def _latency_from_summary(summary: Mapping[str, Any]) -> float | None:
    return _optional_float(
        summary.get("provider_latency_seconds")
        if summary.get("provider_latency_seconds") is not None
        else summary.get("latency_seconds")
    )


def _billable_input_tokens(mapping: Mapping[str, Any]) -> int | None:
    input_tokens = _optional_int(mapping.get("input_tokens"))
    cached_input_tokens = _optional_int(mapping.get("cached_input_tokens"))
    if input_tokens is None or cached_input_tokens is None:
        return None
    return max(0, input_tokens - cached_input_tokens)


def _baseline_role_for_model(model: str) -> str:
    normalized = str(model or "").lower()
    if "5.5" in normalized or "55" in normalized:
        return "strong_full_agent"
    return "cheap_full_agent"


def _iter_json_files(root: Path) -> list[Path]:
    run_root = root / "run1"
    if run_root.is_dir():
        return sorted(path for path in run_root.glob("*.json") if path.is_file())
    return sorted(path for path in root.glob("*.json") if path.is_file())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _parse_run_wall_seconds(text: str) -> float | None:
    match = _RUN_WALL_RE.search(text)
    return float(match.group(1)) if match else None


def _parse_worker_count(text: str) -> int | None:
    match = _WORKERS_RE.search(text)
    return int(match.group(1)) if match else None


def _map(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _divide_optional(numerator: int | None, denominator: int) -> float | None:
    if numerator is None or denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
