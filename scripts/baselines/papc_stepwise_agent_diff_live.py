#!/usr/bin/env python3
"""Run the bounded StepWise artifact on Agent-Diff smoke or official test."""

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

from papc.baselines.agent_diff_live import (  # noqa: E402
    AgentDiffPrivateTask,
    load_agent_diff_private_tasks,
)
from papc.baselines.agent_diff_router_driver import (  # noqa: E402
    AgentDiffNativeRunConfig,
    AgentDiffProviderBudget,
    load_agent_diff_native_api,
    load_agent_diff_official_registry,
)
from papc.baselines.external_router_fit_dev import (  # noqa: E402
    _load_agent_diff_train,
    deterministic_fit_dev,
)
from papc.baselines.external_router_fit_dev_native import (  # noqa: E402
    _load_agent_diff_train_task,
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
from papc.baselines.stepwise_agent_diff_runtime import (  # noqa: E402
    ModernBERTStepWiseMonitor,
    StepWiseDetectorArtifact,
    inspect_detector_artifact,
    run_stepwise_agent_diff_episode,
)
from papc.baselines.stepwise_upstream import (  # noqa: E402
    STEPWISE_METHOD,
    STEPWISE_MILESTONE_THRESHOLD,
    STEPWISE_STUCK_THRESHOLD,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.stepwise_agent_diff_live.v1"
CASE_SCHEMA = "papc.stepwise_agent_diff_live_case.v1"
Mode = Literal["smoke", "official_test"]

DEFAULT_GATE = (
    REPO_ROOT
    / "outputs/papc/stepwise_agent_diff_pilot_gate_noapi_20260731_v2.json"
)
DEFAULT_ANNOTATION_REPORT = (
    REPO_ROOT
    / "outputs/papc/stepwise_agent_diff_annotations_20260731_v1.json"
)
DEFAULT_SPLIT_GATE = (
    REPO_ROOT
    / "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)
DEFAULT_PRIVATE_ROOT = Path(
    f"{DATA_ROOT}/papc/stepwise_agent_diff_pilot_v1/live"
)
DEFAULT_DATASET = Path(
    f"{DATA_ROOT}/benchmarks/agent_diff/repo/datasets/agent-diff-bench"
)
DEFAULT_AGENT_DIFF_SDK = Path(
    f"{DATA_ROOT}/benchmarks/agent_diff/repo/sdk/agent-diff-python"
)
DEFAULT_SCORER = REPO_ROOT / "scripts/agent_diff/papc_agent_diff_scoring.py"
DEFAULT_BASELINE_AUTHORITY = REPO_ROOT / "scripts/agent_diff/papc_agent_diff_hf_test_baseline_run.py"
DEFAULT_STUCK_DIR = Path(
    f"{DATA_ROOT}/papc/stepwise_agent_diff_pilot_v1/models/stuck"
)
DEFAULT_MILESTONE_DIR = Path(
    f"{DATA_ROOT}/papc/stepwise_agent_diff_pilot_v1/models/milestone"
)
MODE_CONTRACTS = {
    "smoke": {"rows": 4, "max_calls": 320, "spend_stop_usd": 3.0},
    "official_test": {
        "rows": 45,
        "max_calls": 3600,
        "spend_stop_usd": 25.0,
    },
}

ClientFactory = Callable[..., Any]


@dataclass(frozen=True)
class LiveContext:
    mode: Mode
    tasks: tuple[AgentDiffPrivateTask, ...]
    native_api: Any
    stuck_artifact: StepWiseDetectorArtifact
    milestone_artifact: StepWiseDetectorArtifact
    artifact_sha256: str
    gate_sha256: str
    annotation_report_sha256: str
    test_registry_sha256: str
    private_root: Path
    api_base: str
    agent_diff_base_url: str
    max_calls: int
    spend_stop_usd: float

    def public_record(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "row_count": len(self.tasks),
            "services": dict(
                sorted(Counter(task.ref.service for task in self.tasks).items())
            ),
            "artifact_sha256": self.artifact_sha256,
            "stuck_detector": self.stuck_artifact.public_record(),
            "milestone_detector": self.milestone_artifact.public_record(),
            "gate_sha256": self.gate_sha256,
            "annotation_report_sha256": self.annotation_report_sha256,
            "test_registry_sha256": self.test_registry_sha256,
            "native_runtime": self.native_api.public_record(),
            "max_provider_calls": self.max_calls,
            "observed_spend_stop_usd": self.spend_stop_usd,
            "provider_calls": 0,
            "secret_reads": 0,
        }


class _BudgetBroker:
    def __init__(self, budget: AgentDiffProviderBudget) -> None:
        self.budget = budget

    def perform(self, raw_attempt: Callable[[], CallReceipt]) -> CallReceipt:
        self.budget.ensure_call_capacity(1)
        receipt = raw_attempt()
        self.budget.charge_receipts((receipt,))
        return receipt


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
    parser.add_argument("--agent-diff-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--agent-diff-sdk-path", type=Path, default=DEFAULT_AGENT_DIFF_SDK
    )
    parser.add_argument("--agent-diff-scorer-path", type=Path, default=DEFAULT_SCORER)
    parser.add_argument(
        "--agent-diff-baseline-authority-path",
        type=Path,
        default=DEFAULT_BASELINE_AUTHORITY,
    )
    parser.add_argument("--stuck-model-dir", type=Path, default=DEFAULT_STUCK_DIR)
    parser.add_argument(
        "--milestone-model-dir", type=Path, default=DEFAULT_MILESTONE_DIR
    )
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--agent-diff-base-url", default="http://localhost:8020")
    parser.add_argument("--cheap-model", default=CHEAP_MODEL)
    parser.add_argument("--strong-model", default=STRONG_MODEL)
    parser.add_argument("--reasoning-effort", default=REASONING_EFFORT)
    parser.add_argument("--max-provider-calls", type=int, required=True)
    parser.add_argument("--spend-stop-usd", type=float, required=True)
    parser.add_argument("--provider-timeout-seconds", type=float, default=180.0)
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
    secret_reads = 0
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
    secret_reads = 1
    if not isinstance(api_key, str) or not api_key.strip():
        report = {
            **base,
            "decision": "blocked_secret_unavailable",
            "secret_reads": secret_reads,
            "blockers": ["api_key_environment_variable_missing_or_empty"],
        }
        _write_json_no_replace(args.output_json, report)
        return 2, report
    factory = client_factory or _default_client_factory
    try:
        provider_client = factory(
            base_url=context.api_base,
            api_key=api_key,
            timeout=args.provider_timeout_seconds,
            max_retries=0,
        )
        results = _execute_context(
            context,
            provider_client=provider_client,
            monitor=monitor,
            api_key_env=args.api_key_env,
        )
        report = _complete_report(context, results=results, secret_reads=secret_reads)
        _write_json_no_replace(args.output_json, report)
        return 0, report
    except Exception as exc:  # noqa: BLE001 - preserve bounded partial evidence.
        results = _load_completed_cases(context)
        report = _incomplete_report(
            context,
            results=results,
            error=exc,
            secret_reads=secret_reads,
        )
        _write_json_no_replace(args.output_json, report)
        return 2, report


def _build_context(args: argparse.Namespace) -> LiveContext:
    contract = MODE_CONTRACTS[args.mode]
    if (
        args.cheap_model != CHEAP_MODEL
        or args.strong_model != STRONG_MODEL
        or args.reasoning_effort != REASONING_EFFORT
        or args.max_provider_calls != contract["max_calls"]
        or args.spend_stop_usd != contract["spend_stop_usd"]
    ):
        raise ValueError("stepwise_agent_diff_live_contract_drifted")
    private_root = Path(args.private_root).resolve() / args.mode
    if not private_root.is_absolute():
        raise ValueError("stepwise_agent_diff_private_root_not_absolute")
    if Path(args.output_json).exists():
        raise FileExistsError("stepwise_agent_diff_output_exists_no_overwrite")
    gate_raw = Path(args.gate_json).read_bytes()
    gate = json.loads(gate_raw)
    annotation_raw = Path(args.annotation_report_json).read_bytes()
    annotation = json.loads(annotation_raw)
    if (
        gate.get("schema") != "papc.stepwise_agent_diff_pilot_gate_noapi.v1"
        or gate.get("decision") != "READY_FOR_ANNOTATION"
        or annotation.get("schema")
        != "papc.stepwise_agent_diff_annotation_run.v1"
        or annotation.get("decision") != "complete_ready_for_upstream_training"
        or annotation.get("gate_sha256") != hashlib.sha256(gate_raw).hexdigest()
        or annotation.get("provider_calls") != 24
        or annotation.get("dataset_stats", {}).get(
            "ready_for_upstream_training"
        )
        is not True
    ):
        raise ValueError("stepwise_agent_diff_fit_artifact_not_ready")
    stuck = inspect_detector_artifact(args.stuck_model_dir)
    milestone = inspect_detector_artifact(args.milestone_model_dir)
    _validate_detector_dataset(stuck, annotation, expected_name="stuck_dataset.json")
    _validate_detector_dataset(
        milestone,
        annotation,
        expected_name="milestone_dataset.json",
    )
    registry = load_agent_diff_official_registry(
        args.agent_diff_dataset,
        split_gate_json=args.split_gate_json,
    )
    if args.mode == "official_test":
        tasks = load_agent_diff_private_tasks(registry)
    else:
        train = _load_agent_diff_train(Path(args.agent_diff_dataset).resolve())
        _, dev = deterministic_fit_dev(train)
        selected = gate.get("heldout_smoke", {}).get("rows")
        if not isinstance(selected, list) or len(selected) != 4:
            raise ValueError("stepwise_agent_diff_smoke_rows_invalid")
        by_sha = {row.example_sha256: row for row in dev}
        tasks = tuple(
            _load_agent_diff_train_task(
                Path(args.agent_diff_dataset).resolve(),
                by_sha[item["example_sha256"]],
            )
            for item in selected
        )
        if [task.ref.service for task in tasks] != [
            item["service"] for item in selected
        ]:
            raise ValueError("stepwise_agent_diff_smoke_service_mismatch")
    if len(tasks) != contract["rows"]:
        raise ValueError("stepwise_agent_diff_live_denominator_mismatch")
    native_api = load_agent_diff_native_api(
        sdk_path=args.agent_diff_sdk_path,
        scorer_path=args.agent_diff_scorer_path,
        baseline_authority_path=args.agent_diff_baseline_authority_path,
    )
    artifact_sha = _digest(
        {
            "stuck": stuck.bundle_sha256,
            "milestone": milestone.bundle_sha256,
            "gate_sha256": hashlib.sha256(gate_raw).hexdigest(),
            "annotation_sha256": hashlib.sha256(annotation_raw).hexdigest(),
            "runtime_source_sha256s": {
                "live_script": _file_sha256(Path(__file__)),
                "runtime_module": _file_sha256(
                    REPO_ROOT
                    / "src/papc/baselines/stepwise_agent_diff_runtime.py"
                ),
                "controller_module": _file_sha256(
                    REPO_ROOT / "src/papc/baselines/stepwise_upstream.py"
                ),
                "agent_diff_native_module": _file_sha256(
                    REPO_ROOT / "src/papc/baselines/agent_diff_live.py"
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
        tasks=tasks,
        native_api=native_api,
        stuck_artifact=stuck,
        milestone_artifact=milestone,
        artifact_sha256=artifact_sha,
        gate_sha256=hashlib.sha256(gate_raw).hexdigest(),
        annotation_report_sha256=hashlib.sha256(annotation_raw).hexdigest(),
        test_registry_sha256=registry.split.test_registry_sha256,
        private_root=private_root,
        api_base=args.api_base.rstrip("/"),
        agent_diff_base_url=args.agent_diff_base_url.rstrip("/"),
        max_calls=contract["max_calls"],
        spend_stop_usd=contract["spend_stop_usd"],
    )


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
        raise ValueError("stepwise_detector_dataset_binding_mismatch")


def _execute_context(
    context: LiveContext,
    *,
    provider_client: Any,
    monitor: Any,
    api_key_env: str,
) -> tuple[dict[str, Any], ...]:
    existing = list(_load_completed_cases(context))
    completed = {item["example_sha256"] for item in existing}
    budget = AgentDiffProviderBudget(
        max_calls=context.max_calls,
        max_observed_cost_usd=context.spend_stop_usd,
    )
    for item in existing:
        budget.charge_receipts(
            tuple(CallReceipt(**receipt) for receipt in item["provider_receipts"])
        )
    broker = _BudgetBroker(budget)
    config = AgentDiffNativeRunConfig(
        api_base=context.api_base,
        agent_diff_base_url=context.agent_diff_base_url,
        private_output_root=context.private_root,
        max_provider_calls=context.max_calls,
        max_observed_provider_cost_usd=context.spend_stop_usd,
    )
    for task in context.tasks:
        example = task.ref.example_sha256
        if example in completed:
            continue
        raw_path = context.private_root / "raw" / f"{example}.json"
        case_path = context.private_root / "cases" / f"{example}.json"
        if raw_path.exists() or case_path.exists():
            raise FileExistsError("stepwise_agent_diff_partial_case_collision")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        case_path.parent.mkdir(parents=True, exist_ok=True)
        episode = run_stepwise_agent_diff_episode(
            task=task,
            api=context.native_api,
            config=config,
            provider_client=provider_client,
            broker=broker,
            raw_path=raw_path,
            api_key_env=api_key_env,
            monitor=monitor,
        )
        record = {
            "schema": CASE_SCHEMA,
            "mode": context.mode,
            "artifact_sha256": context.artifact_sha256,
            "example_sha256": example,
            "service": task.ref.service,
            "official_score": episode.official_score,
            "decisions": [item.public_record() for item in episode.decisions],
            "provider_receipts": [
                item.public_record() for item in episode.provider_receipts
            ],
            "local_router_receipts": [
                item.public_record() for item in episode.local_router_receipts
            ],
            "raw_path": str(raw_path.relative_to(context.private_root)),
            "raw_sha256": _file_sha256(raw_path),
        }
        _write_json_no_replace(case_path, record)
        existing.append(record)
        completed.add(example)
    ordered = {item["example_sha256"]: item for item in existing}
    return tuple(ordered[task.ref.example_sha256] for task in context.tasks)


def _load_completed_cases(context: LiveContext) -> tuple[dict[str, Any], ...]:
    cases = context.private_root / "cases"
    if not cases.is_dir():
        return ()
    allowed = {task.ref.example_sha256: task for task in context.tasks}
    loaded: list[dict[str, Any]] = []
    for path in sorted(cases.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        example = item.get("example_sha256")
        if (
            item.get("schema") != CASE_SCHEMA
            or item.get("mode") != context.mode
            or item.get("artifact_sha256") != context.artifact_sha256
            or example not in allowed
            or item.get("service") != allowed[example].ref.service
            or item.get("official_score") not in {0.0, 1.0}
        ):
            raise ValueError("stepwise_agent_diff_completed_case_invalid")
        raw_path = context.private_root / item["raw_path"]
        if _file_sha256(raw_path) != item.get("raw_sha256"):
            raise ValueError("stepwise_agent_diff_completed_raw_drifted")
        tuple(CallReceipt(**receipt) for receipt in item["provider_receipts"])
        tuple(CallReceipt(**receipt) for receipt in item["local_router_receipts"])
        loaded.append(item)
    if len({item["example_sha256"] for item in loaded}) != len(loaded):
        raise ValueError("stepwise_agent_diff_completed_case_duplicate")
    return tuple(loaded)


def _complete_report(
    context: LiveContext,
    *,
    results: Sequence[Mapping[str, Any]],
    secret_reads: int,
) -> dict[str, Any]:
    if len(results) != len(context.tasks):
        raise ValueError("stepwise_agent_diff_result_denominator_incomplete")
    return _result_report(
        context,
        results=results,
        decision="complete",
        blockers=(),
        secret_reads=secret_reads,
    )


def _incomplete_report(
    context: LiveContext,
    *,
    results: Sequence[Mapping[str, Any]],
    error: BaseException,
    secret_reads: int,
) -> dict[str, Any]:
    return _result_report(
        context,
        results=results,
        decision="incomplete",
        blockers=(f"{type(error).__name__}:{error}",),
        secret_reads=secret_reads,
    )


def _result_report(
    context: LiveContext,
    *,
    results: Sequence[Mapping[str, Any]],
    decision: str,
    blockers: Sequence[str],
    secret_reads: int,
) -> dict[str, Any]:
    provider = tuple(
        CallReceipt(**receipt)
        for result in results
        for receipt in result["provider_receipts"]
    )
    local = tuple(
        CallReceipt(**receipt)
        for result in results
        for receipt in result["local_router_receipts"]
    )
    accounting = aggregate_online_accounting(
        (*provider, *local),
        prices=FROZEN_CACHE_NEUTRAL_PRICES,
    )
    raw_successes = sum(int(result["official_score"]) for result in results)
    roles = Counter(receipt.role for receipt in provider)
    rationales = Counter(
        decision_record["rationale"]
        for result in results
        for decision_record in result["decisions"]
    )
    escalated = sum(
        any(item["selected_role"] == "strong" for item in result["decisions"])
        for result in results
    )
    by_service = {}
    for service in sorted({task.ref.service for task in context.tasks}):
        selected = [item for item in results if item["service"] == service]
        by_service[service] = {
            "evaluated": len(selected),
            "successes": sum(int(item["official_score"]) for item in selected),
        }
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
            min(raw_successes, 40)
            if context.mode == "official_test" and len(results) == 45
            else None
        ),
        "reporting_denominator": 45 if context.mode == "official_test" else None,
        "strong_quality_ceiling": "40/45" if context.mode == "official_test" else None,
        "above_strong_is_single_run_variance": context.mode == "official_test",
        "by_service": by_service,
        "escalated_tasks": escalated,
        "provider_calls_by_role": dict(sorted(roles.items())),
        "decision_rationales": dict(sorted(rationales.items())),
        "provider_calls": accounting["agent_side"]["provider_calls"],
        "secret_reads": secret_reads,
        "api_spend_performed": bool(accounting["agent_side"]["provider_calls"]),
        "agent_side_accounting": accounting["agent_side"],
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
        "blockers": list(blockers),
        "public_contains_requests_or_expected_answers": False,
        "public_safe": True,
    }


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
