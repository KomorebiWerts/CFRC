#!/usr/bin/env python3
"""Prepare/run ToolSandbox base129 same-accounting official baselines."""

from __future__ import annotations

import argparse
import code
import copy
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from papc.certspec.runtime_ledger import RuntimeLedger  # noqa: E402
from papc.certspec.toolsandbox import role_bridge  # noqa: E402
from papc.certspec.toolsandbox.base129_true_online_run import (  # noqa: E402
    EXPECTED_OFFICIAL_ENGINE,
    EXPECTED_SCENARIO_SCOPE,
    EXPECTED_SURFACE,
    EXPECTED_TASK_COUNT,
    Base129ScenarioSpec,
    scenario_specs_from_lower_receipts,
)
from papc.certspec.toolsandbox.method_runtime import (  # noqa: E402
    _patch_numpy_compat,
    _prepare_toolsandbox_imports,
    named_scenarios_for_checkout,
    official_evaluation_summary,
)
from papc.certspec.toolsandbox.public_hygiene import public_leakage_audit  # noqa: E402
from papc.certspec.toolsandbox.runtime_accounting import (  # noqa: E402
    ToolSandboxRuntimeAccountingRecorder,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.toolsandbox.base129_baseline_accounting.v1"
PRIVATE_SCHEMA = "papc.toolsandbox.base129_baseline_accounting.private.v1"
READY_NOAPI_DECISION = "toolsandbox_base129_baseline_accounting_ready_noapi"
READY_DECISION = "toolsandbox_base129_baseline_accounting_ready"
BLOCKED_DECISION = "blocked_toolsandbox_base129_baseline_accounting"
APPROVAL_SCOPE = "toolsandbox_base129_official_baseline_accounting"
REQUIRED_TOOL_ENV_VARS = ("RAPID_API_KEY",)
DEFAULT_LOWER_RECEIPT_ROOT = Path(
    f"{DATA_ROOT}/papc/toolsandbox_base129_official_bounds_private_20260627_codex/lower/receipts"
)
DEFAULT_TOOLSANDBOX_CHECKOUT = Path(f"{DATA_ROOT}/benchmarks/toolsandbox/repo")
DEFAULT_PRIVATE_RUN_ROOT = Path(
    f"{DATA_ROOT}/papc/toolsandbox_base129_baseline_accounting_private_20260628_codex_v1"
)
DEFAULT_PRIVATE_OUTPUT_JSON = Path(
    f"{DATA_ROOT}/papc/toolsandbox_base129_baseline_accounting_private_20260628_codex_v1.json"
)
DEFAULT_PUBLIC_OUTPUT_JSON = (
    _REPO / "outputs" / "papc" / "toolsandbox_base129_baseline_accounting_20260628_codex_v1.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fake-execute", action="store_true")
    parser.add_argument("--approval-receipt-json", type=Path)
    parser.add_argument("--api-base")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--arm", choices=("cheap_full", "strong_full"), required=True)
    parser.add_argument("--agent-model")
    parser.add_argument("--agent-reasoning-effort", default="high")
    parser.add_argument("--user-model", default="gpt-5.4")
    parser.add_argument("--user-reasoning-effort", default="high")
    parser.add_argument("--cap", type=int, default=EXPECTED_TASK_COUNT)
    parser.add_argument(
        "--public-ordinals",
        default="",
        help="Comma-separated public ordinals for a sharded run; filter before cap.",
    )
    parser.add_argument(
        "--targeted-recovery",
        action="store_true",
        help="Allow an append-only rerun of explicitly selected ordinals.",
    )
    parser.add_argument("--lower-receipt-root", type=Path, default=DEFAULT_LOWER_RECEIPT_ROOT)
    parser.add_argument("--toolsandbox-checkout", type=Path, default=DEFAULT_TOOLSANDBOX_CHECKOUT)
    parser.add_argument("--private-run-root", type=Path, default=DEFAULT_PRIVATE_RUN_ROOT)
    parser.add_argument("--private-output-json", type=Path, default=DEFAULT_PRIVATE_OUTPUT_JSON)
    parser.add_argument("--public-output-json", type=Path, default=DEFAULT_PUBLIC_OUTPUT_JSON)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.agent_model:
        args.agent_model = "gpt-5.4-mini" if args.arm == "cheap_full" else "gpt-5.5"
    approval_receipt = (
        _read_json_object(args.approval_receipt_json)
        if args.approval_receipt_json is not None
        else {}
    )
    specs = _filter_public_ordinals(
        scenario_specs_from_lower_receipts(args.lower_receipt_root),
        public_ordinals=args.public_ordinals,
    )[: max(0, args.cap)]
    private_report, public_report = build_report(
        specs=specs,
        approval_receipt=approval_receipt,
        execute_requested=args.execute,
        fake_execute=args.fake_execute,
        arm=args.arm,
        agent_model=args.agent_model,
        agent_reasoning_effort=args.agent_reasoning_effort,
        user_model=args.user_model,
        user_reasoning_effort=args.user_reasoning_effort,
        api_base=args.api_base or "",
        api_key_env=args.api_key_env,
        checkout=args.toolsandbox_checkout,
        private_run_root=args.private_run_root,
        cap=args.cap,
        targeted_recovery=args.targeted_recovery,
    )
    _write_pair_append_only(
        private_path=args.private_output_json,
        public_path=args.public_output_json,
        private_report=private_report,
        public_report=public_report,
    )
    print(
        json.dumps(
            {
                "decision": public_report["decision"],
                "provider_calls": public_report["provider_calls"],
                "arm": args.arm,
                "blockers": public_report["blockers"],
                "public_output": str(args.public_output_json),
                "private_output": str(args.private_output_json),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if public_report["decision"] in {READY_NOAPI_DECISION, READY_DECISION} else 1


def build_report(
    *,
    specs: Sequence[Base129ScenarioSpec],
    approval_receipt: Mapping[str, Any],
    execute_requested: bool,
    fake_execute: bool,
    arm: str,
    agent_model: str,
    agent_reasoning_effort: str,
    user_model: str,
    user_reasoning_effort: str,
    api_base: str,
    api_key_env: str,
    checkout: Path,
    private_run_root: Path,
    cap: int,
    targeted_recovery: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preflight_blockers = _blockers(
        specs=specs,
        approval_receipt=approval_receipt,
        execute_requested=execute_requested,
        arm=arm,
        agent_model=agent_model,
        agent_reasoning_effort=agent_reasoning_effort,
        user_model=user_model,
        user_reasoning_effort=user_reasoning_effort,
        api_base=api_base,
        cap=cap,
        fake_execute=fake_execute,
        targeted_recovery=targeted_recovery,
    )
    execution_started = bool(execute_requested and not preflight_blockers)
    receipts: list[dict[str, Any]] = []
    if execution_started:
        receipts = (
            _fake_execute(specs=specs, arm=arm)
            if fake_execute
            else _execute_official(
                specs=specs,
                arm=arm,
                agent_model=agent_model,
                agent_reasoning_effort=agent_reasoning_effort,
                user_model=user_model,
                user_reasoning_effort=user_reasoning_effort,
                api_base=api_base,
                api_key_env=api_key_env,
                checkout=checkout,
                private_run_root=private_run_root,
            )
        )
    runtime_ledger = _aggregate_runtime_ledger(receipts)
    wall_seconds = _wall_seconds(receipts)
    complete = sum(1 for receipt in receipts if _milestone_complete(receipt))
    top_level_failures = sum(
        1 for receipt in receipts if _private(receipt).get("exception_type")
    )
    execution_response_count = sum(
        _int(_execution_integrity(receipt).get("execution_response_count"))
        for receipt in receipts
    )
    tool_call_exception_count = sum(
        _int(_execution_integrity(receipt).get("tool_call_exception_count"))
        for receipt in receipts
    )
    compile_failure_count = sum(
        _int(
            _execution_integrity(receipt).get(
                "compile_level_tool_call_exception_count"
            )
        )
        for receipt in receipts
    )
    compile_failure_episodes = sum(
        1
        for receipt in receipts
        if _int(
            _execution_integrity(receipt).get(
                "compile_level_tool_call_exception_count"
            )
        )
        > 0
    )
    all_execution_responses_compile_failed_episodes = sum(
        1
        for receipt in receipts
        if _execution_integrity(receipt).get(
            "all_execution_responses_compile_failed"
        )
        is True
    )
    runtime_failures = sum(
        1
        for receipt in receipts
        if _private(receipt).get("exception_type")
        or _int(
            _execution_integrity(receipt).get(
                "compile_level_tool_call_exception_count"
            )
        )
        > 0
    )
    report_blockers = set(preflight_blockers)
    if execution_started and top_level_failures:
        report_blockers.add("runtime_failure_present")
    if execution_started and compile_failure_episodes:
        report_blockers.add("compile_level_tool_call_exception_present")
    generated_at = datetime.now(timezone.utc).isoformat()
    private_report = {
        "schema": PRIVATE_SCHEMA,
        "surface": EXPECTED_SURFACE,
        "mode": "toolsandbox_base129_baseline_accounting_private",
        "generated_at": generated_at,
        "execute_requested": bool(execute_requested),
        "fake_execute": bool(fake_execute),
        "arm": arm,
        "receipt_count": len(receipts),
        "receipts": receipts,
        "blockers": sorted(report_blockers),
        "targeted_recovery": bool(targeted_recovery),
    }
    public_report: dict[str, Any] = {
        "schema": SCHEMA,
        "surface": EXPECTED_SURFACE,
        "mode": "toolsandbox_base129_official_baseline_accounting",
        "generated_at": generated_at,
        "execute_requested": bool(execute_requested),
        "api_spend_performed": bool(execution_started and not fake_execute),
        "fake_execute": bool(fake_execute),
        "arm": arm,
        "official_engine": EXPECTED_OFFICIAL_ENGINE,
        "provider_calls": _runtime_provider_calls(runtime_ledger) if execute_requested else 0,
        "model_plan": {
            "agent_model": agent_model,
            "agent_reasoning_effort": agent_reasoning_effort,
            "user_model": user_model,
            "user_reasoning_effort": user_reasoning_effort,
        },
        "execution_environment": {
            "required_tool_env_vars": list(REQUIRED_TOOL_ENV_VARS),
            "missing_required_tool_env_vars": _missing_required_tool_env_vars(
                REQUIRED_TOOL_ENV_VARS
            )
            if execute_requested and not fake_execute
            else [],
            "tool_api_values_excluded": True,
        },
        "summary": {
            "scenario_scope": EXPECTED_SCENARIO_SCOPE,
            "cap": int(cap),
            "scenario_spec_count": len(specs),
            "attempted_episode_count": len(receipts),
            "milestone_complete_count": complete,
            "execution_response_count": execution_response_count,
            "tool_call_exception_count": tool_call_exception_count,
            "compile_level_tool_call_exception_count": compile_failure_count,
            "compile_level_tool_call_exception_episode_count": compile_failure_episodes,
            "all_execution_responses_compile_failed_episode_count": (
                all_execution_responses_compile_failed_episodes
            ),
            "top_level_runtime_failure_count": top_level_failures,
            "runtime_failure_count": runtime_failures,
        },
        "resource_accounting": _resource_accounting(runtime_ledger, wall_seconds),
        "claim_boundary": [
            "This report prepares same-accounting ToolSandbox official baselines.",
            "Dry-run performs no provider calls.",
            "Fake execution is a test fixture and is not a baseline result.",
        ],
        "blockers": sorted(report_blockers),
        "targeted_recovery": bool(targeted_recovery),
    }
    public_report["decision"] = (
        READY_DECISION
        if execute_requested and not public_report["blockers"]
        else READY_NOAPI_DECISION
        if not execute_requested and not public_report["blockers"]
        else BLOCKED_DECISION
    )
    leakage = public_leakage_audit(public_report)
    public_report["public_leakage_audit"] = leakage
    if leakage["pass"] is not True:
        public_report["blockers"] = sorted(
            {*public_report["blockers"], "public_leakage_marker_present"}
        )
        public_report["decision"] = BLOCKED_DECISION
    return private_report, public_report


def _execute_official(
    *,
    specs: Sequence[Base129ScenarioSpec],
    arm: str,
    agent_model: str,
    agent_reasoning_effort: str,
    user_model: str,
    user_reasoning_effort: str,
    api_base: str,
    api_key_env: str,
    checkout: Path,
    private_run_root: Path,
) -> list[dict[str, Any]]:
    _prepare_toolsandbox_imports(checkout)
    _patch_numpy_compat()
    from tool_sandbox.common.execution_context import RoleType  # type: ignore
    from tool_sandbox.common.tool_discovery import ToolBackend  # type: ignore
    from tool_sandbox.roles.execution_environment import (
        ExecutionEnvironment,  # type: ignore
    )

    scenarios = named_scenarios_for_checkout(
        checkout,
        preferred_tool_backend=ToolBackend.DEFAULT,
    )
    factories = role_bridge.openai_compatible_role_factories(
        role_bridge.ToolSandboxRoleBridgeConfig(
            agent_model=agent_model,
            user_model=user_model,
            api_base=api_base,
            api_key_env=api_key_env,
            agent_reasoning_effort=agent_reasoning_effort,
            user_reasoning_effort=user_reasoning_effort,
        )
    )
    private_run_root.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    agent_runtime_role = "cheap" if arm == "cheap_full" else "strong"
    for spec in specs:
        recorder = ToolSandboxRuntimeAccountingRecorder()
        roles = {
            RoleType.USER: factories["user"](),
            RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
            RoleType.AGENT: factories["agent"](),
        }
        recorder.wrap_role_model_inference(
            roles[RoleType.AGENT],
            role_name="agent",
            runtime_role=agent_runtime_role,
        )
        recorder.wrap_role_model_inference(
            roles[RoleType.USER],
            role_name="user",
            runtime_role="user_simulator",
        )
        started = time.monotonic()
        exception_type: str | None = None
        evaluation: dict[str, Any]
        execution_integrity = _empty_execution_integrity()
        try:
            result = scenarios[spec.scenario_name].play_and_evaluate(
                roles=roles,
                output_directory=private_run_root / "raw_tool_sandbox",
                scenario_name=spec.scenario_name,
            )
            evaluation = dict(official_evaluation_summary(result.evaluation_result))
            execution_integrity = _execution_context_tool_call_exception_summary(
                result.ending_context.to_dict(serialize_console=False)
            )
        except Exception as exc:  # noqa: BLE001 - receipt records provider/runtime failures.
            exception_type = type(exc).__name__
            evaluation = {
                "milestone_similarity": 0.0,
                "minefield_similarity": 0.0,
                "similarity": 0.0,
            }
        finally:
            for role in roles.values():
                teardown = getattr(role, "teardown", None)
                if callable(teardown):
                    teardown()
        receipts.append(
            _receipt(
                spec=spec,
                arm=arm,
                evaluation=evaluation,
                runtime_ledger=recorder.public_summary(),
                wall_seconds=round(time.monotonic() - started, 6),
                exception_type=exception_type,
                execution_integrity=execution_integrity,
            )
        )
    return receipts


def _fake_execute(
    *,
    specs: Sequence[Base129ScenarioSpec],
    arm: str,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    agent_runtime_role = "cheap" if arm == "cheap_full" else "strong"
    for spec in specs:
        ledger = RuntimeLedger()
        ledger.add_usage(
            agent_runtime_role,
            {"prompt_tokens": 10, "completion_tokens": 2},
            latency_seconds=0.5,
        )
        ledger.add_usage(
            "user_simulator",
            {"prompt_tokens": 5, "completion_tokens": 2},
            latency_seconds=0.25,
        )
        receipts.append(
            _receipt(
                spec=spec,
                arm=arm,
                evaluation={
                    "milestone_similarity": 1.0,
                    "minefield_similarity": 0.0,
                    "similarity": 1.0,
                },
                runtime_ledger=ledger.public_summary(),
                wall_seconds=1.0,
                exception_type=None,
            )
        )
    return receipts


def _receipt(
    *,
    spec: Base129ScenarioSpec,
    arm: str,
    evaluation: Mapping[str, Any],
    runtime_ledger: Mapping[str, Any],
    wall_seconds: float,
    exception_type: str | None,
    execution_integrity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "papc.toolsandbox.base129_baseline_accounting_receipt.v1",
        "surface": EXPECTED_SURFACE,
        "public_ordinal": spec.public_ordinal,
        "arm": arm,
        "official_evaluator_used": exception_type is None,
        "evaluation": {
            key: value
            for key, value in evaluation.items()
            if isinstance(value, (int, float, bool))
        },
        "private": {
            "scenario_name": spec.scenario_name,
            "exception_type": exception_type,
            "method_wall_clock_seconds": float(wall_seconds),
            "runtime_ledger": copy.deepcopy(dict(runtime_ledger)),
            "execution_integrity": copy.deepcopy(
                dict(execution_integrity or _empty_execution_integrity())
            ),
        },
    }


def _blockers(
    *,
    specs: Sequence[Base129ScenarioSpec],
    approval_receipt: Mapping[str, Any],
    execute_requested: bool,
    arm: str,
    agent_model: str,
    agent_reasoning_effort: str,
    user_model: str,
    user_reasoning_effort: str,
    api_base: str,
    cap: int,
    fake_execute: bool,
    targeted_recovery: bool,
) -> list[str]:
    blockers: list[str] = []
    if not targeted_recovery and len(specs) != min(int(cap), EXPECTED_TASK_COUNT):
        blockers.append("scenario_spec_count_mismatch")
    expected = {
        "approval_scope": APPROVAL_SCOPE,
        "surface": EXPECTED_SURFACE,
        "scenario_scope": EXPECTED_SCENARIO_SCOPE,
        "arm": arm,
        "max_episodes": min(int(cap), EXPECTED_TASK_COUNT),
        "agent_model": agent_model,
        "agent_reasoning_effort": agent_reasoning_effort,
        "user_model": user_model,
        "user_reasoning_effort": user_reasoning_effort,
        "resource_accounting_required": True,
        "method_fallback_allowed": False,
    }
    if approval_receipt:
        for key, expected_value in expected.items():
            if approval_receipt.get(key) != expected_value:
                blockers.append(f"approval_{key}_mismatch")
    if not execute_requested:
        return blockers
    if not api_base:
        blockers.append("api_base_missing")
    if not fake_execute:
        blockers.extend(
            f"required_tool_env_missing:{name}"
            for name in _missing_required_tool_env_vars(REQUIRED_TOOL_ENV_VARS)
        )
    if not approval_receipt:
        blockers.append("approval_receipt_missing")
    return blockers


def _filter_public_ordinals(
    scenario_specs: list[Base129ScenarioSpec],
    *,
    public_ordinals: str,
) -> list[Base129ScenarioSpec]:
    """Keep only the rows named by a shard, preserving lower-receipt order."""

    selected: set[int] = set()
    for token in str(public_ordinals or "").split(","):
        token = token.strip()
        if not token:
            continue
        selected.add(int(token))
    if not selected:
        return scenario_specs
    return [spec for spec in scenario_specs if int(spec.public_ordinal) in selected]


def _execution_context_tool_call_exception_summary(
    serialized_context: Mapping[str, Any],
) -> dict[str, Any]:
    dbs = serialized_context.get("_dbs")
    if not isinstance(dbs, Mapping):
        return _empty_execution_integrity()
    sandbox_rows: Any = None
    for namespace, rows in dbs.items():
        if _role_name(namespace) == "SANDBOX":
            sandbox_rows = rows
            break
    if not isinstance(sandbox_rows, Sequence) or isinstance(
        sandbox_rows, (str, bytes, bytearray)
    ):
        return _empty_execution_integrity()

    execution_response_count = 0
    tool_call_exception_count = 0
    compile_failure_count = 0
    unclassified_exception_count = 0
    for row in sandbox_rows:
        if not isinstance(row, Mapping):
            continue
        sender = _role_name(row.get("sender"))
        recipient = _role_name(row.get("recipient"))
        if (
            sender in {"AGENT", "USER"}
            and recipient == "EXECUTION_ENVIRONMENT"
        ):
            if _python_source_has_compile_failure(str(row.get("content") or "")):
                compile_failure_count += 1
            continue
        if sender != "EXECUTION_ENVIRONMENT" or recipient not in {"AGENT", "USER"}:
            continue
        execution_response_count += 1
        if not row.get("tool_call_exception"):
            continue
        tool_call_exception_count += 1
        if not row.get("openai_tool_call_id"):
            unclassified_exception_count += 1

    return {
        "execution_response_count": execution_response_count,
        "tool_call_exception_count": tool_call_exception_count,
        "compile_level_tool_call_exception_count": compile_failure_count,
        "unclassified_tool_call_exception_count": unclassified_exception_count,
        "all_execution_responses_compile_failed": bool(
            execution_response_count > 0
            and compile_failure_count == execution_response_count
        ),
    }


def _python_source_has_compile_failure(source: str) -> bool:
    try:
        command = code.compile_command(source, symbol="exec")
    except (OverflowError, SyntaxError, ValueError):
        return True
    return command is None


def _role_name(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").rsplit(".", 1)[-1].upper()


def _empty_execution_integrity() -> dict[str, Any]:
    return {
        "execution_response_count": 0,
        "tool_call_exception_count": 0,
        "compile_level_tool_call_exception_count": 0,
        "unclassified_tool_call_exception_count": 0,
        "all_execution_responses_compile_failed": False,
    }


def _missing_required_tool_env_vars(env_vars: tuple[str, ...]) -> list[str]:
    return [name for name in env_vars if not os.environ.get(name)]


def _resource_accounting(
    runtime_ledger: Mapping[str, Any],
    wall_seconds: Sequence[float],
) -> dict[str, Any]:
    if not wall_seconds and _runtime_provider_calls(runtime_ledger) == 0:
        return {
            "schema": "papc.toolsandbox.baseline_resource_accounting.v1",
            "status": "pending_execute",
            "runtime_breakdown": {},
            "missing_fields_are_unavailable_not_zero": True,
        }
    wall_sum = round(sum(float(value) for value in wall_seconds), 6)
    return {
        "schema": "papc.toolsandbox.baseline_resource_accounting.v1",
        "status": "observed",
        "runtime_ledger": runtime_ledger,
        "runtime_breakdown": {
            "cheap": _role(runtime_ledger, "cheap"),
            "strong": _role(runtime_ledger, "strong"),
            "user_simulator": _role(runtime_ledger, "user_simulator"),
            "method_wall_clock": {
                "row_count": len(wall_seconds),
                "per_row_wall_seconds_sum": wall_sum,
                "total_run_latency_seconds": wall_sum,
            },
        },
        "missing_fields_are_unavailable_not_zero": True,
    }


def _aggregate_runtime_ledger(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ledger = RuntimeLedger()
    for receipt in receipts:
        runtime_ledger = _private(receipt).get("runtime_ledger")
        if isinstance(runtime_ledger, Mapping):
            ledger.merge_public_summary(runtime_ledger)
    return ledger.public_summary()


def _wall_seconds(receipts: Sequence[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for receipt in receipts:
        value = _private(receipt).get("method_wall_clock_seconds")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(max(0.0, float(value)))
    return values


def _role(runtime_ledger: Mapping[str, Any], role: str) -> dict[str, Any]:
    per_role = runtime_ledger.get("per_role")
    if not isinstance(per_role, Mapping):
        return {}
    value = per_role.get(role)
    return dict(value) if isinstance(value, Mapping) else {}


def _runtime_provider_calls(runtime_ledger: Mapping[str, Any]) -> int:
    totals = runtime_ledger.get("totals")
    if not isinstance(totals, Mapping):
        return 0
    return _int(totals.get("calls"))


def _milestone_complete(receipt: Mapping[str, Any]) -> bool:
    evaluation = receipt.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return False
    try:
        return float(evaluation.get("milestone_similarity") or 0.0) >= 1.0
    except (TypeError, ValueError):
        return False


def _execution_integrity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = _private(receipt).get("execution_integrity")
    return dict(value) if isinstance(value, Mapping) else _empty_execution_integrity()


def _private(receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = receipt.get("private")
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object at {path}")
    return dict(payload)


def _write_pair_append_only(
    *,
    private_path: Path,
    public_path: Path,
    private_report: Mapping[str, Any],
    public_report: Mapping[str, Any],
) -> None:
    if private_path.exists():
        raise FileExistsError(private_path)
    if public_path.exists():
        raise FileExistsError(public_path)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        json.dumps(private_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    public_path.write_text(
        json.dumps(public_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
