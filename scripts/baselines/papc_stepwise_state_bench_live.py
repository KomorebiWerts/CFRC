#!/usr/bin/env python3
"""Run the bounded StepWise artifact on STATE smoke or official test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from papc.baselines.external_router_fit_dev import (  # noqa: E402
    _load_state_train,
    deterministic_fit_dev,
)
from papc.baselines.external_router_fit_dev_native import (  # noqa: E402
    _state_fit_dev_task,
)
from papc.baselines.router_accounting import (  # noqa: E402
    FROZEN_CACHE_NEUTRAL_PRICES,
    CallReceipt,
    aggregate_online_accounting,
)
from papc.baselines.router_contract import (  # noqa: E402
    CHEAP_MODEL,
    REASONING_EFFORT,
    STRONG_MODEL,
)
from papc.baselines.state_bench_live import (  # noqa: E402
    build_state_bench_native_clients,
    validate_state_bench_runtime_environment,
)
from papc.baselines.state_bench_native_runtime import (  # noqa: E402
    StateBenchNativeRunConfig,
    StateBenchProviderBudget,
    StateBenchRuntimePolicy,
    load_state_bench_native_api,
)
from papc.baselines.state_bench_router_driver import (  # noqa: E402
    StateBenchOfficialTaskRef,
    load_state_bench_official_test_registry,
)
from papc.baselines.stepwise_agent_diff_runtime import (  # noqa: E402
    ModernBERTStepWiseMonitor,
    StepWiseDetectorArtifact,
    inspect_detector_artifact,
)
from papc.baselines.stepwise_state_bench_runtime import (  # noqa: E402
    run_stepwise_state_bench_episode,
)
from papc.baselines.stepwise_upstream import (  # noqa: E402
    STEPWISE_METHOD,
    STEPWISE_MILESTONE_THRESHOLD,
    STEPWISE_STUCK_THRESHOLD,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.stepwise_state_bench_live.v1"
CASE_SCHEMA = "papc.stepwise_state_bench_live_case.v1"
ANNOTATION_SCHEMA = "papc.stepwise_state_bench_annotation_run.v1"
Mode = Literal["smoke", "official_test"]

DEFAULT_GATE = REPO_ROOT / (
    "outputs/papc/stepwise_state_bench_pilot_gate_noapi_20260801_v2.json"
)
DEFAULT_ANNOTATION_REPORT = REPO_ROOT / (
    "outputs/papc/stepwise_state_bench_annotations_20260801_v1.json"
)
DEFAULT_SPLIT_GATE = REPO_ROOT / (
    "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)
DEFAULT_STATE_REPO = Path(f"{DATA_ROOT}/benchmarks/state_bench/repo")
DEFAULT_PRIVATE_ROOT = Path(
    f"{DATA_ROOT}/papc/stepwise_state_bench_pilot_v1/live"
)
DEFAULT_STUCK_DIR = Path(
    f"{DATA_ROOT}/papc/stepwise_state_bench_pilot_v1/models/stuck"
)
DEFAULT_MILESTONE_DIR = Path(
    f"{DATA_ROOT}/papc/stepwise_state_bench_pilot_v1/models/milestone"
)
MODE_CONTRACTS = {
    "smoke": {"rows": 3, "max_calls": 192, "spend_stop_usd": 5.0},
    "official_test": {
        "rows": 150,
        "max_calls": 9600,
        "spend_stop_usd": 80.0,
    },
}
OFFICIAL_TEST_SHARD_COUNTS = frozenset({1, 5})

ClientFactory = Callable[..., Any]


@dataclass(frozen=True)
class LiveContext:
    mode: Mode
    tasks: tuple[StateBenchOfficialTaskRef, ...]
    native_api: Any
    stuck_artifact: StepWiseDetectorArtifact
    milestone_artifact: StepWiseDetectorArtifact
    artifact_sha256: str
    gate_sha256: str
    annotation_report_sha256: str
    registry_sha256: str
    private_root: Path
    api_base: str
    max_calls: int
    spend_stop_usd: float
    shard_count: int
    shard_index: int

    def public_record(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "row_count": len(self.tasks),
            "example_sha256s": [task.example_sha256 for task in self.tasks],
            "domains": dict(
                sorted(Counter(task.domain for task in self.tasks).items())
            ),
            "artifact_sha256": self.artifact_sha256,
            "stuck_detector": self.stuck_artifact.public_record(),
            "milestone_detector": self.milestone_artifact.public_record(),
            "gate_sha256": self.gate_sha256,
            "annotation_report_sha256": self.annotation_report_sha256,
            "registry_sha256": self.registry_sha256,
            "native_runtime": self.native_api.public_record(),
            "max_provider_calls": self.max_calls,
            "observed_spend_stop_usd": self.spend_stop_usd,
            "shard": {
                "count": self.shard_count,
                "index": self.shard_index,
                "rule": "zero_based_registry_ordinal_modulo_shard_count",
            },
            "provider_calls": 0,
            "secret_reads": 0,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODE_CONTRACTS), required=True)
    parser.add_argument("--gate-json", type=Path, default=DEFAULT_GATE)
    parser.add_argument(
        "--annotation-report-json",
        type=Path,
        default=DEFAULT_ANNOTATION_REPORT,
    )
    parser.add_argument("--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE)
    parser.add_argument("--state-repo", type=Path, default=DEFAULT_STATE_REPO)
    parser.add_argument("--stuck-model-dir", type=Path, default=DEFAULT_STUCK_DIR)
    parser.add_argument(
        "--milestone-model-dir", type=Path, default=DEFAULT_MILESTONE_DIR
    )
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--cheap-model", default=CHEAP_MODEL)
    parser.add_argument("--strong-model", default=STRONG_MODEL)
    parser.add_argument("--reasoning-effort", default=REASONING_EFFORT)
    parser.add_argument("--max-provider-calls", type=int, required=True)
    parser.add_argument("--spend-stop-usd", type=float, required=True)
    parser.add_argument("--provider-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def run(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] = os.environ,
    client_factory: ClientFactory | None = None,
    monitor_factory: Callable[..., Any] = ModernBERTStepWiseMonitor,
) -> tuple[int, dict[str, Any]]:
    args = parse_args(argv)
    try:
        context = _build_context(args)
        monitor = monitor_factory(
            stuck_dir=context.stuck_artifact.path,
            milestone_dir=context.milestone_artifact.path,
        )
    except Exception as exc:  # noqa: BLE001 - provider-zero source gate.
        report = _blocked_report(args, exc, secret_reads=0)
        _write_json_no_replace(args.output_json, report)
        return 2, report
    base = {
        "schema": SCHEMA,
        "decision": "ready_for_explicit_execute",
        "execute": bool(args.execute),
        "context": context.public_record(),
        "method": STEPWISE_METHOD,
        "model_pair": {
            "cheap_model": CHEAP_MODEL,
            "strong_model": STRONG_MODEL,
            "reasoning_effort": REASONING_EFFORT,
        },
        "thresholds": {
            "stuck": STEPWISE_STUCK_THRESHOLD,
            "milestone": STEPWISE_MILESTONE_THRESHOLD,
            "policy": "released_defaults_no_dev_or_test_tuning",
        },
        "provider_calls": 0,
        "secret_reads": 0,
        "api_spend_performed": False,
        "official_test_outcomes_read": False,
        "blockers": ["execute_flag_missing"],
        "public_safe": True,
    }
    if not args.execute:
        _write_json_no_replace(args.output_json, base)
        return 0, base
    blockers = _proxy_blockers(env)
    if blockers:
        report = {**base, "decision": "blocked_before_secret_read", "blockers": blockers}
        _write_json_no_replace(args.output_json, report)
        return 2, report
    if args.api_key_env != "OPENAI_API_KEY":
        report = {
            **base,
            "decision": "blocked_before_secret_read",
            "blockers": ["api_key_env_must_be_OPENAI_API_KEY"],
        }
        _write_json_no_replace(args.output_json, report)
        return 2, report
    api_key = env.get(args.api_key_env)
    if not isinstance(api_key, str) or not api_key.strip():
        report = {
            **base,
            "decision": "blocked_secret_unavailable",
            "secret_reads": 1,
            "blockers": ["api_key_environment_variable_missing_or_empty"],
        }
        _write_json_no_replace(args.output_json, report)
        return 2, report
    factory = client_factory or _default_client_factory
    budget = StateBenchProviderBudget(
        max_calls=context.max_calls,
        max_observed_cost_usd=context.spend_stop_usd,
    )
    try:
        clients = build_state_bench_native_clients(
            api=context.native_api,
            api_base=context.api_base,
            api_key=api_key,
            raw_client_factory=lambda base, key: factory(
                base_url=base,
                api_key=key,
                timeout=args.provider_timeout_seconds,
                max_retries=0,
            ),
        )
        verifier_client = factory(
            base_url=context.api_base,
            api_key=api_key,
            timeout=args.provider_timeout_seconds,
            max_retries=0,
        )
        results = _execute_context(
            context,
            clients=clients,
            verifier_client=verifier_client,
            monitor=monitor,
            budget=budget,
        )
        report = _result_report(
            context,
            results=results,
            decision="complete",
            blockers=(),
            secret_reads=1,
            budget=budget,
        )
        _write_json_no_replace(args.output_json, report)
        return 0, report
    except Exception as exc:  # noqa: BLE001 - preserve bounded partial evidence.
        results = _load_completed_cases(context)
        report = _result_report(
            context,
            results=results,
            decision="incomplete",
            blockers=(f"{type(exc).__name__}:{exc}",),
            secret_reads=1,
            budget=budget,
        )
        _write_json_no_replace(args.output_json, report)
        return 2, report


def _build_context(args: argparse.Namespace) -> LiveContext:
    contract = MODE_CONTRACTS[args.mode]
    _validate_shard_args(
        mode=args.mode,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    shard_max_calls = contract["max_calls"] // args.shard_count
    shard_spend_stop_usd = contract["spend_stop_usd"] / args.shard_count
    if (
        args.cheap_model != CHEAP_MODEL
        or args.strong_model != STRONG_MODEL
        or args.reasoning_effort != REASONING_EFFORT
        or args.max_provider_calls != shard_max_calls
        or args.spend_stop_usd != shard_spend_stop_usd
    ):
        raise ValueError("stepwise_state_bench_live_contract_drifted")
    if Path(args.output_json).exists():
        raise FileExistsError("stepwise_state_bench_output_exists_no_overwrite")
    private_root = Path(args.private_root).resolve() / args.mode
    if args.shard_count > 1:
        private_root /= (
            f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
        )
    gate_raw = Path(args.gate_json).read_bytes()
    gate = json.loads(gate_raw)
    annotation_raw = Path(args.annotation_report_json).read_bytes()
    annotation = json.loads(annotation_raw)
    if (
        gate.get("schema") != "papc.stepwise_state_bench_pilot_gate_noapi.v1"
        or gate.get("decision") != "READY_FOR_ANNOTATION"
        or annotation.get("schema") != ANNOTATION_SCHEMA
        or annotation.get("decision") != "complete_ready_for_upstream_training"
        or annotation.get("gate_sha256") != hashlib.sha256(gate_raw).hexdigest()
        or annotation.get("provider_calls") != 24
        or annotation.get("dataset_stats", {}).get("ready_for_upstream_training")
        is not True
    ):
        raise ValueError("stepwise_state_bench_fit_artifact_not_ready")
    stuck = inspect_detector_artifact(args.stuck_model_dir)
    milestone = inspect_detector_artifact(args.milestone_model_dir)
    _validate_detector_dataset(stuck, annotation, expected_name="stuck_dataset.json")
    _validate_detector_dataset(
        milestone,
        annotation,
        expected_name="milestone_dataset.json",
    )
    native_api = load_state_bench_native_api(args.state_repo)
    validate_state_bench_runtime_environment(native_api)
    registry = load_state_bench_official_test_registry(
        args.state_repo,
        split_gate_json=args.split_gate_json,
    )
    if args.mode == "official_test":
        tasks = tuple(
            task
            for ordinal, task in enumerate(registry.tasks)
            if ordinal % args.shard_count == args.shard_index
        )
    else:
        train = _load_state_train(Path(args.state_repo).resolve())
        _, dev = deterministic_fit_dev(train)
        selected = gate.get("heldout_smoke", {}).get("rows")
        if not isinstance(selected, list) or len(selected) != 3:
            raise ValueError("stepwise_state_bench_smoke_rows_invalid")
        by_sha = {row.example_sha256: row for row in dev}
        tasks = tuple(
            _state_fit_dev_task(
                api=native_api,
                source=by_sha[item["example_sha256"]],
            )
            for item in selected
        )
        if [task.domain for task in tasks] != [item["domain"] for item in selected]:
            raise ValueError("stepwise_state_bench_smoke_domain_mismatch")
    if len(tasks) != contract["rows"] // args.shard_count:
        raise ValueError("stepwise_state_bench_live_denominator_mismatch")
    artifact_sha = _digest(
        {
            "stuck": stuck.bundle_sha256,
            "milestone": milestone.bundle_sha256,
            "gate_sha256": hashlib.sha256(gate_raw).hexdigest(),
            "annotation_sha256": hashlib.sha256(annotation_raw).hexdigest(),
            "registry_sha256": registry.test_registry_sha256,
            "runtime_source_sha256s": {
                "live_script": _file_sha256(Path(__file__)),
                "runtime_module": _file_sha256(
                    REPO_ROOT
                    / "src/papc/baselines/stepwise_state_bench_runtime.py"
                ),
                "controller_module": _file_sha256(
                    REPO_ROOT / "src/papc/baselines/stepwise_upstream.py"
                ),
                "state_native_module": _file_sha256(
                    REPO_ROOT / "src/papc/baselines/state_bench_native_runtime.py"
                ),
            },
            "thresholds": [
                STEPWISE_STUCK_THRESHOLD,
                STEPWISE_MILESTONE_THRESHOLD,
            ],
        }
    )
    return LiveContext(
        mode=args.mode,
        tasks=tuple(tasks),
        native_api=native_api,
        stuck_artifact=stuck,
        milestone_artifact=milestone,
        artifact_sha256=artifact_sha,
        gate_sha256=hashlib.sha256(gate_raw).hexdigest(),
        annotation_report_sha256=hashlib.sha256(annotation_raw).hexdigest(),
        registry_sha256=registry.test_registry_sha256,
        private_root=private_root,
        api_base=args.api_base.rstrip("/"),
        max_calls=shard_max_calls,
        spend_stop_usd=shard_spend_stop_usd,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )


def _validate_shard_args(
    *,
    mode: str,
    shard_count: int,
    shard_index: int,
) -> None:
    if mode == "smoke":
        if shard_count != 1 or shard_index != 0:
            raise ValueError("stepwise_state_bench_smoke_must_not_be_sharded")
        return
    if shard_count not in OFFICIAL_TEST_SHARD_COUNTS:
        raise ValueError("stepwise_state_bench_shard_count_not_frozen")
    if not 0 <= shard_index < shard_count:
        raise ValueError("stepwise_state_bench_shard_index_invalid")


def _validate_detector_dataset(
    artifact: StepWiseDetectorArtifact,
    annotation: Mapping[str, Any],
    *,
    expected_name: str,
) -> None:
    training = json.loads(
        (artifact.path / "training_config.json").read_text(encoding="utf-8")
    )
    dataset = Path(str(training.get("dataset", ""))).resolve()
    expected = Path(str(annotation.get("dataset_dir", ""))).resolve() / expected_name
    artifact_record = annotation.get("dataset_artifacts", {}).get(expected_name)
    if (
        dataset != expected
        or not expected.is_file()
        or not isinstance(artifact_record, Mapping)
        or artifact_record.get("path") != str(expected)
        or artifact_record.get("sha256") != _file_sha256(expected)
    ):
        raise ValueError("stepwise_state_bench_detector_dataset_binding_mismatch")


def _execute_context(
    context: LiveContext,
    *,
    clients: Any,
    verifier_client: Any,
    monitor: Any,
    budget: StateBenchProviderBudget,
) -> tuple[dict[str, Any], ...]:
    existing = list(_load_completed_cases(context))
    for item in existing:
        for raw in item["provider_receipts"]:
            budget.observe_broker_receipt(CallReceipt(**raw))
    completed = {item["example_sha256"] for item in existing}
    config = StateBenchNativeRunConfig(
        api_base=context.api_base,
        private_output_root=context.private_root,
        max_provider_calls=context.max_calls,
        max_observed_provider_cost_usd=context.spend_stop_usd,
    )
    for task in context.tasks:
        if task.example_sha256 in completed:
            continue
        domain = context.native_api.get_domain_config(task.domain)
        policy = StateBenchRuntimePolicy(
            cheap_client=clients.cheap,
            strong_client=clients.strong,
            simulator_client=clients.simulator,
            evaluator_client=clients.evaluator,
            task_requirements_judge=context.native_api.task_requirements_judge_class(
                client=clients.evaluator,
                prompts_dir=domain.prompts_dir,
                system_prompt=domain.judge_system_prompt,
                reasoning_effort=REASONING_EFFORT,
            ),
            ux_judge=context.native_api.ux_quality_judge_class(
                client=clients.evaluator,
                prompts_dir=domain.prompts_dir,
                system_prompt=domain.judge_system_prompt,
                reasoning_effort=REASONING_EFFORT,
            ),
        )
        case_root = context.private_root / "raw" / task.example_sha256
        case_path = context.private_root / "cases" / f"{task.example_sha256}.json"
        if case_root.exists() or case_path.exists():
            raise FileExistsError("stepwise_state_bench_partial_case_collision")
        case_path.parent.mkdir(parents=True, exist_ok=True)
        episode = run_stepwise_state_bench_episode(
            task_ref=task,
            api=context.native_api,
            config=config,
            policy=policy,
            budget=budget,
            monitor=monitor,
            verifier_provider_client=verifier_client,
            registry_sha256=context.registry_sha256,
            case_root=case_root,
        )
        raw_files = {
            name: _file_sha256(case_root / name)
            for name in (
                "trajectory_generated.json",
                "trajectory_unscored.json",
                "trajectory_scored.json",
            )
        }
        record = {
            "schema": CASE_SCHEMA,
            "mode": context.mode,
            "artifact_sha256": context.artifact_sha256,
            "registry_sha256": context.registry_sha256,
            "example_sha256": task.example_sha256,
            "domain": task.domain,
            "official_score": episode.official_score,
            "decisions": [item.public_record() for item in episode.decisions],
            "provider_receipts": [
                item.public_record() for item in episode.provider_receipts
            ],
            "local_router_receipts": [
                item.public_record() for item in episode.local_router_receipts
            ],
            "raw_root": str(case_root.relative_to(context.private_root)),
            "raw_file_sha256s": raw_files,
        }
        _write_json_no_replace(case_path, record)
        existing.append(record)
        completed.add(task.example_sha256)
    ordered = {item["example_sha256"]: item for item in existing}
    return tuple(ordered[task.example_sha256] for task in context.tasks)


def _load_completed_cases(context: LiveContext) -> tuple[dict[str, Any], ...]:
    cases = context.private_root / "cases"
    if not cases.is_dir():
        return ()
    allowed = {task.example_sha256: task for task in context.tasks}
    loaded: list[dict[str, Any]] = []
    for path in sorted(cases.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        example = item.get("example_sha256")
        if (
            item.get("schema") != CASE_SCHEMA
            or item.get("mode") != context.mode
            or item.get("artifact_sha256") != context.artifact_sha256
            or item.get("registry_sha256") != context.registry_sha256
            or example not in allowed
            or item.get("domain") != allowed[example].domain
            or item.get("official_score") not in {0.0, 1.0}
        ):
            raise ValueError("stepwise_state_bench_completed_case_invalid")
        raw_root = context.private_root / item["raw_root"]
        for name, expected in item["raw_file_sha256s"].items():
            if _file_sha256(raw_root / name) != expected:
                raise ValueError("stepwise_state_bench_completed_raw_drifted")
        tuple(CallReceipt(**raw) for raw in item["provider_receipts"])
        tuple(CallReceipt(**raw) for raw in item["local_router_receipts"])
        loaded.append(item)
    if len({item["example_sha256"] for item in loaded}) != len(loaded):
        raise ValueError("stepwise_state_bench_completed_case_duplicate")
    return tuple(loaded)


def _result_report(
    context: LiveContext,
    *,
    results: Sequence[Mapping[str, Any]],
    decision: str,
    blockers: Sequence[str],
    secret_reads: int,
    budget: StateBenchProviderBudget,
) -> dict[str, Any]:
    provider = tuple(
        CallReceipt(**raw)
        for result in results
        for raw in result["provider_receipts"]
    )
    local = tuple(
        CallReceipt(**raw)
        for result in results
        for raw in result["local_router_receipts"]
    )
    accounting = aggregate_online_accounting(
        (*provider, *local),
        prices=FROZEN_CACHE_NEUTRAL_PRICES,
    )
    raw_successes = sum(int(result["official_score"]) for result in results)
    roles = Counter(receipt.role for receipt in provider)
    rationales = Counter(
        route["rationale"]
        for result in results
        for route in result["decisions"]
    )
    escalated = sum(
        any(route["selected_role"] == "strong" for route in result["decisions"])
        for result in results
    )
    by_domain = {}
    for domain in sorted({task.domain for task in context.tasks}):
        selected = [item for item in results if item["domain"] == domain]
        by_domain[domain] = {
            "evaluated": len(selected),
            "successes": sum(int(item["official_score"]) for item in selected),
        }
    full_costs = [receipt.provider_reported_cost_usd for receipt in provider]
    usage_complete = all(cost is not None for cost in full_costs)
    shard_count = getattr(context, "shard_count", 1)
    is_full_cell = context.mode == "official_test" and shard_count == 1
    return {
        "schema": SCHEMA,
        "decision": decision,
        "execute": True,
        "mode": context.mode,
        "context": context.public_record(),
        "evaluated": len(results),
        "expected_rows": len(context.tasks),
        "raw_successes": raw_successes,
        "reported_successes": (
            min(raw_successes, 92) if is_full_cell else None
        ),
        "reporting_denominator": 150 if is_full_cell else None,
        "strong_quality_ceiling": (
            "92/150" if context.mode == "official_test" else None
        ),
        "requires_five_shard_aggregate": (
            context.mode == "official_test" and shard_count == 5
        ),
        "above_strong_is_single_run_variance": context.mode == "official_test",
        "by_domain": by_domain,
        "escalated_tasks": escalated,
        "provider_calls_by_role": dict(sorted(roles.items())),
        "decision_rationales": dict(sorted(rationales.items())),
        "provider_calls": budget.calls,
        "receipted_provider_calls": sum(item.provider_calls for item in provider),
        "secret_reads": secret_reads,
        "api_spend_performed": budget.calls > 0,
        "agent_side_accounting": accounting["agent_side"],
        "full_provider_accounting": {
            "provider_calls": sum(item.provider_calls for item in provider),
            "input_tokens": _sum_optional(item.input_tokens for item in provider),
            "output_tokens": _sum_optional(item.output_tokens for item in provider),
            "reasoning_tokens": _sum_optional(
                item.reasoning_tokens for item in provider
            ),
            "usage_complete": usage_complete,
            "cost_usd": (
                round(sum(float(cost) for cost in full_costs), 9)
                if usage_complete
                else None
            ),
        },
        "local_router": {
            "provider_calls": 0,
            "cost_usd": 0.0,
            "decision_count": len(local),
            "latency_seconds": round(
                sum(item.latency_seconds for item in local),
                6,
            ),
            "hardware": platform.machine() or "unknown",
        },
        "annotation_cost_included_in_test_cost": False,
        "official_test_outcomes_read": context.mode == "official_test",
        "blockers": list(blockers),
        "public_contains_requests_or_expected_answers": False,
        "public_safe": True,
    }


def _sum_optional(values: Sequence[int | None] | Any) -> int | None:
    resolved = tuple(values)
    if any(value is None for value in resolved):
        return None
    return sum(int(value) for value in resolved)


def _blocked_report(
    args: argparse.Namespace,
    error: BaseException,
    *,
    secret_reads: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "decision": "blocked_before_secret_read",
        "execute": bool(args.execute),
        "provider_calls": 0,
        "secret_reads": secret_reads,
        "api_spend_performed": False,
        "blockers": [f"{type(error).__name__}:{error}"],
        "public_safe": True,
    }


def _proxy_blockers(env: Mapping[str, str]) -> list[str]:
    no_proxy = {item.strip() for item in env.get("NO_PROXY", "").split(",")}
    blockers = []
    if not {"localhost", "127.0.0.1"}.issubset(no_proxy):
        blockers.append("no_proxy_localhost_binding_missing")
    if any(
        env.get(name)
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    ):
        blockers.append("http_proxy_must_be_unset")
    return blockers


def _default_client_factory(**kwargs: Any) -> Any:
    from openai import OpenAI

    return OpenAI(**kwargs)


def _write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(target, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    code, report = run(argv)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "mode": report.get("mode"),
                "evaluated": report.get("evaluated", 0),
                "raw_successes": report.get("raw_successes"),
                "reported_successes": report.get("reported_successes"),
                "provider_calls": report["provider_calls"],
            },
            sort_keys=True,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
