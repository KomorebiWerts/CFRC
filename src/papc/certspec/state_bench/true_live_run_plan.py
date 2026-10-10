"""STATE-Bench true-live run-plan guards.

This module is deliberately no-API. It keeps the paid-run control plane in one
place: the canonical model plan, model-plan drift checks, and the execute-time
comparison against the no-API gate artifact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

NONCANONICAL_MODEL_PLAN_BLOCKER = "noncanonical_model_plan_requires_explicit_override"
EXPECTED_NOAPI_REPORT_REQUIRED_BLOCKER = "expected_noapi_report_required_for_execute"
EXPECTED_NOAPI_REPORT_LOAD_BLOCKER = "expected_noapi_report_load_failed"
EXPECTED_NOAPI_REPORT_INVALID_BLOCKER = "expected_noapi_report_invalid"
EXPECTED_NOAPI_REPORT_MISMATCH_BLOCKER = "expected_noapi_report_mismatch"

CANONICAL_TRUE_LIVE_MODEL_PLAN = {
    "cheap_model": "gpt-5.4-mini",
    "cheap_reasoning_effort": "high",
    "strong_model": "gpt-5.5",
    "strong_reasoning_effort": "high",
    "user_simulator_model": "gpt-5.5",
    "user_simulator_reasoning_effort": "high",
}

# All registered model pools are canonical. Freezing only the first one made
# the second pool mismatch by construction, and the only way to run it was an
# override flag that the shared entry point appended behind the caller's back:
# the two pools then cleared different preflight gates, and the registry did not
# say so. A pool that this project registered as an operating point is an
# intended plan, not drift, so the guard admits either one and keeps failing
# closed on anything else.
CANONICAL_TRUE_LIVE_MODEL_PLANS: tuple[dict[str, str], ...] = (
    CANONICAL_TRUE_LIVE_MODEL_PLAN,
    {
        "cheap_model": "gpt-5.6-luna",
        "cheap_reasoning_effort": "high",
        "strong_model": "gpt-5.6-sol",
        "strong_reasoning_effort": "high",
        "user_simulator_model": "gpt-5.5",
        "user_simulator_reasoning_effort": "high",
    },
    {
        "cheap_model": "gemini-3.6-flash-high",
        "cheap_reasoning_effort": "high",
        "strong_model": "gpt-5.6-sol",
        "strong_reasoning_effort": "high",
        "user_simulator_model": "gpt-5.5",
        "user_simulator_reasoning_effort": "high",
    },
    {
        "cheap_model": "gemini-3.5-flash-high",
        "cheap_reasoning_effort": "high",
        "strong_model": "gpt-5.6-sol",
        "strong_reasoning_effort": "high",
        "user_simulator_model": "gpt-5.5",
        "user_simulator_reasoning_effort": "high",
    },
    {
        "cheap_model": "gemini-3.5-flash-high",
        "cheap_reasoning_effort": "high",
        "strong_model": "gpt-5.5",
        "strong_reasoning_effort": "high",
        "user_simulator_model": "gpt-5.5",
        "user_simulator_reasoning_effort": "high",
    },
    {
        "cheap_model": "gpt-5.4-mini",
        "cheap_reasoning_effort": "high",
        "strong_model": "gemini-3.5-flash-high",
        "strong_reasoning_effort": "high",
        "user_simulator_model": "gpt-5.5",
        "user_simulator_reasoning_effort": "high",
    },
    {
        "cheap_model": "gpt-5.6-luna",
        "cheap_reasoning_effort": "high",
        "strong_model": "gemini-3.7-flash-high",
        "strong_reasoning_effort": "high",
        "user_simulator_model": "gpt-5.5",
        "user_simulator_reasoning_effort": "high",
    },
)

_MODEL_PLAN_FIELDS = (
    "cheap_model",
    "cheap_reasoning_effort",
    "strong_model",
    "strong_reasoning_effort",
    "user_simulator_model",
    "user_simulator_reasoning_effort",
)

EXPECTED_NOAPI_COMPARE_FIELDS = (
    "surface",
    "source_bundle",
    "execution_contract",
    "model_plan",
    "frontier_policy",
    "parallelism",
    "task_plan",
)


def build_model_plan_report(source: Any) -> dict[str, Any]:
    return {
        "cheap_model": _value(source, "cheap_model"),
        "cheap_reasoning_effort": _value(source, "cheap_reasoning_effort"),
        "residual_strong_model": _value(source, "strong_model"),
        "residual_strong_reasoning_effort": _value(source, "strong_reasoning_effort"),
        "user_simulator_model": _value(source, "user_simulator_model"),
        "user_simulator_reasoning_effort": _value(
            source, "user_simulator_reasoning_effort"
        ),
        "residual_final_repair_reasoning_effort": _value(
            source, "final_repair_reasoning_effort", default=None
        ),
        "allow_noncanonical_model_plan": bool(
            _value(source, "allow_noncanonical_model_plan", default=False)
        ),
    }


def build_model_plan_consistency(source: Any) -> dict[str, Any]:
    mismatches = model_plan_mismatches(source)
    override_allowed = bool(
        _value(source, "allow_noncanonical_model_plan", default=False)
    )
    return {
        "schema": "papc.state_bench_true_live_model_plan_consistency.v1",
        "status": (
            "canonical"
            if not mismatches
            else "noncanonical_override"
            if override_allowed
            else "noncanonical_blocked"
        ),
        "canonical_model_plan": canonical_model_plan_report(source),
        "mismatches": mismatches,
        "allow_noncanonical_model_plan": override_allowed,
        "fail_closed_blocker": (
            None
            if not mismatches or override_allowed
            else NONCANONICAL_MODEL_PLAN_BLOCKER
        ),
    }


def canonical_model_plan_report(source: Any = None) -> dict[str, str]:
    """Report the registered plan this run is measured against.

    With no source, or with one that matches no registered pool, this is the
    first registered plan. Reporting that one against a run of the second pool
    would print six mismatched model names beside a canonical verdict, so a run
    that matches a registered pool is reported against the pool it matched.
    """

    plan = _matching_canonical_plan(source) or CANONICAL_TRUE_LIVE_MODEL_PLAN
    return {
        "cheap_model": plan["cheap_model"],
        "cheap_reasoning_effort": plan["cheap_reasoning_effort"],
        "residual_strong_model": plan["strong_model"],
        "residual_strong_reasoning_effort": plan["strong_reasoning_effort"],
        "user_simulator_model": plan["user_simulator_model"],
        "user_simulator_reasoning_effort": plan["user_simulator_reasoning_effort"],
    }


def model_plan_mismatches(source: Any) -> dict[str, dict[str, str]]:
    """Mismatches against the nearest registered plan, empty if any one matches.

    Reported against the first registered plan when none matches, because a
    plan that matches nothing is drift and the reader needs one reference to
    read the mismatch against rather than one per registered pool.
    """

    if _matching_canonical_plan(source) is not None:
        return {}
    mismatches: dict[str, dict[str, str]] = {}
    for field in _MODEL_PLAN_FIELDS:
        actual = str(_value(source, field) or "").strip()
        expected = CANONICAL_TRUE_LIVE_MODEL_PLAN[field]
        if actual != expected:
            report_field = _public_model_plan_field(field)
            mismatches[report_field] = {
                "expected": expected,
                "actual": actual,
            }
    return mismatches


def _matching_canonical_plan(source: Any) -> dict[str, str] | None:
    if source is None:
        return None
    for plan in CANONICAL_TRUE_LIVE_MODEL_PLANS:
        if all(
            str(_value(source, field) or "").strip() == plan[field]
            for field in _MODEL_PLAN_FIELDS
        ):
            return plan
    return None


def expected_noapi_report_consistency(
    source: Any,
    *,
    current_plan: Mapping[str, Any],
) -> dict[str, Any]:
    execute = bool(_value(source, "execute", default=False))
    path = expected_noapi_report_path(source)
    if not execute:
        return _expected_noapi_report(
            status="not_required",
            path=path,
            blockers=[],
        )
    if path is None:
        return _expected_noapi_report(
            status="not_supplied",
            path=None,
            blockers=[EXPECTED_NOAPI_REPORT_REQUIRED_BLOCKER],
        )
    try:
        expected = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _expected_noapi_report(
            status="load_failed",
            path=path,
            blockers=[EXPECTED_NOAPI_REPORT_LOAD_BLOCKER],
            load_error=type(exc).__name__,
        )
    if not isinstance(expected, Mapping):
        return _expected_noapi_report(
            status="invalid",
            path=path,
            blockers=[EXPECTED_NOAPI_REPORT_INVALID_BLOCKER],
            invalid_reasons=["expected_noapi_report_must_be_json_object"],
        )
    invalid_reasons = expected_noapi_report_invalid_reasons(expected)
    mismatches = expected_noapi_plan_mismatches(expected, current_plan)
    blockers: list[str] = []
    if invalid_reasons:
        blockers.append(EXPECTED_NOAPI_REPORT_INVALID_BLOCKER)
    if mismatches:
        blockers.append(EXPECTED_NOAPI_REPORT_MISMATCH_BLOCKER)
    status = "match"
    if invalid_reasons and mismatches:
        status = "invalid_and_mismatch"
    elif invalid_reasons:
        status = "invalid"
    elif mismatches:
        status = "mismatch"
    return _expected_noapi_report(
        status=status,
        path=path,
        blockers=blockers,
        invalid_reasons=invalid_reasons,
        mismatches=mismatches,
    )


def expected_noapi_report_path(source: Any) -> Path | None:
    raw = _value(source, "expected_noapi_report_json", default=None)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return Path(text)


def expected_noapi_report_invalid_reasons(
    expected: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if bool(expected.get("execute")):
        reasons.append("expected_noapi_report_must_have_execute_false")
    if int(expected.get("provider_calls") or 0) != 0:
        reasons.append("expected_noapi_report_must_have_zero_provider_calls")
    blockers = expected.get("blockers")
    if blockers != []:
        reasons.append("expected_noapi_report_must_have_no_blockers")
    leakage = expected.get("public_leakage_audit")
    if isinstance(leakage, Mapping) and leakage.get("pass") is not True:
        reasons.append("expected_noapi_report_must_pass_public_leakage_audit")
    return reasons


def expected_noapi_plan_mismatches(
    expected: Mapping[str, Any],
    current_plan: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    mismatches: dict[str, dict[str, Any]] = {}
    for field in EXPECTED_NOAPI_COMPARE_FIELDS:
        expected_value = expected.get(field)
        current_value = current_plan.get(field)
        if expected_value != current_value:
            mismatches[field] = {
                "expected_noapi": expected_value,
                "current_execute": current_value,
            }
    return mismatches


def _expected_noapi_report(
    *,
    status: str,
    path: Path | None,
    blockers: list[str],
    load_error: str | None = None,
    invalid_reasons: list[str] | None = None,
    mismatches: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "papc.state_bench_expected_noapi_report_consistency.v1",
        "status": status,
        "expected_noapi_report_json": str(path) if path is not None else None,
        "compared_fields": list(EXPECTED_NOAPI_COMPARE_FIELDS),
        "mismatches": mismatches or {},
        "invalid_reasons": invalid_reasons or [],
        "fail_closed_blockers": sorted(set(blockers)),
    }
    if load_error:
        report["load_error_type"] = load_error
    return report


def _public_model_plan_field(field: str) -> str:
    if field == "strong_model":
        return "residual_strong_model"
    if field == "strong_reasoning_effort":
        return "residual_strong_reasoning_effort"
    return field


def _value(source: Any, key: str, default: Any = "") -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)
