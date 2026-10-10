#!/usr/bin/env python3
"""Run AgServe-QAC on frozen Tau2-Retail dev smoke or official test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Mapping
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.agserve_qac import (  # noqa: E402
    AGSERVE_QAC_METHOD,
    AgServeArtifact,
    TransformerAgServeJudge,
    inspect_agserve_artifact,
)
from papc.baselines.agserve_qac_tau2_runtime import (  # noqa: E402
    register_agserve_tau2_agent,
)
from papc.baselines.external_router_execution import (  # noqa: E402
    ProviderAttemptBroker,
    ProviderAttemptLedger,
)
from papc.baselines.external_router_fit_dev import (  # noqa: E402
    FitDevRegistry,
    _load_tau2_train,
    deterministic_fit_dev,
)
from papc.baselines.external_router_fit_dev_native import (  # noqa: E402
    activate_tau2_dependency_environment,
)
from papc.baselines.router_accounting import (  # noqa: E402
    FROZEN_CACHE_NEUTRAL_PRICES,
    aggregate_online_accounting,
)
from papc.baselines.router_contract import (  # noqa: E402
    CHEAP_MODEL,
    REASONING_EFFORT,
    STRONG_MODEL,
)
from papc.baselines.router_runtime_environment import (  # noqa: E402
    validate_router_runtime_environment,
)
from papc.baselines.tau2_native_runtime import (  # noqa: E402
    Tau2NativeExecutor,
    Tau2NativeRunConfig,
    load_tau2_native_api,
)
from papc.baselines.tau2_router_driver import (  # noqa: E402
    Tau2OfficialTaskRef,
    Tau2OfficialTestRegistry,
    load_tau2_official_test_registry,
)
from papc.paths import DATA_ROOT

Mode = Literal["smoke", "official_test"]
SCHEMA = "papc.agserve_qac_tau2_retail_live.v1"
DEFAULT_SPLIT_GATE = REPO_ROOT / (
    "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)
DEFAULT_DATA_REPO = Path(f"{DATA_ROOT}/benchmarks/tau2-bench-revised/repo")
DEFAULT_RUNTIME_REPO = Path(f"{DATA_ROOT}/benchmarks/tau_bench/repo")
DEFAULT_ARTIFACT_ROOT = Path(
    f"{DATA_ROOT}/papc/agserve_qac_v1/models_tau2_retail_v2"
)
DEFAULT_PRIVATE_ROOT = Path(f"{DATA_ROOT}/papc/agserve_qac_v1/tau2_retail")
MODE_CONTRACTS = {
    "smoke": {"rows": 3, "max_calls": 240, "spend_stop_usd": 3.0},
    "official_test": {"rows": 40, "max_calls": 3200, "spend_stop_usd": 40.0},
}
OFFICIAL_SHARD_COUNTS = frozenset({1, 5})


@dataclass(frozen=True)
class LiveContext:
    mode: Mode
    tasks: tuple[Tau2OfficialTaskRef, ...]
    registry: Tau2OfficialTestRegistry
    native_api: Any
    artifact: AgServeArtifact
    artifact_sha256: str
    fit_dev_registry_sha256: str
    dependency_environment_sha256: str
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
            "artifact_sha256": self.artifact_sha256,
            "artifact": self.artifact.public_record(),
            "official_test_registry_sha256": self.registry.test_registry_sha256,
            "fit_dev_registry_sha256": self.fit_dev_registry_sha256,
            "dependency_environment_sha256": self.dependency_environment_sha256,
            "native_runtime": self.native_api.public_record(),
            "max_provider_calls": self.max_calls,
            "observed_spend_stop_usd": self.spend_stop_usd,
            "shard": {"count": self.shard_count, "index": self.shard_index},
            "provider_calls": 0,
            "secret_reads": 0,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODE_CONTRACTS), required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--local-judge-device", default="cpu")
    parser.add_argument("--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE)
    parser.add_argument("--tau2-repo", type=Path, default=DEFAULT_DATA_REPO)
    parser.add_argument("--tau2-runtime-repo", type=Path, default=DEFAULT_RUNTIME_REPO)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--cheap-model", default=CHEAP_MODEL)
    parser.add_argument("--strong-model", default=STRONG_MODEL)
    parser.add_argument("--reasoning-effort", default=REASONING_EFFORT)
    parser.add_argument("--max-provider-calls", type=int, required=True)
    parser.add_argument("--spend-stop-usd", type=float, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def run(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] = os.environ,
) -> tuple[int, dict[str, Any]]:
    args = parse_args(argv)
    try:
        context = _build_context(args)
        judge = TransformerAgServeJudge(
            context.artifact,
            device=args.local_judge_device,
        )
    except Exception as exc:  # noqa: BLE001 - provider-zero gate.
        report = _blocked_report(args, exc, secret_reads=0)
        _write_json_no_replace(args.output_json, report)
        return 2, report

    base = {
        "schema": SCHEMA,
        "decision": "ready_for_explicit_execute",
        "execute": bool(args.execute),
        "context": context.public_record(),
        "method": AGSERVE_QAC_METHOD,
        "scope": "agserve_quality_aware_client_only",
        "model_pair": {
            "cheap_model": CHEAP_MODEL,
            "strong_model": STRONG_MODEL,
            "reasoning_effort": REASONING_EFFORT,
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

    try:
        results, ledger = _execute_context(context, judge=judge)
        report = _result_report(
            context,
            results=results,
            ledger=ledger,
            decision="complete",
            blockers=(),
            secret_reads=1,
        )
        _write_json_no_replace(args.output_json, report)
        return 0, report
    except Exception as exc:  # noqa: BLE001 - preserve bounded partial evidence.
        results = _load_completed_cases(context.private_root)
        report = _result_report(
            context,
            results=results,
            ledger=None,
            decision="incomplete",
            blockers=(f"{type(exc).__name__}:{exc}",),
            secret_reads=1,
        )
        _write_json_no_replace(args.output_json, report)
        return 2, report


def _build_context(args: argparse.Namespace) -> LiveContext:
    contract = MODE_CONTRACTS[args.mode]
    _validate_shard(args.mode, args.shard_count, args.shard_index)
    max_calls = contract["max_calls"] // args.shard_count
    spend = contract["spend_stop_usd"] / args.shard_count
    if (
        args.cheap_model != CHEAP_MODEL
        or args.strong_model != STRONG_MODEL
        or args.reasoning_effort != REASONING_EFFORT
        or args.max_provider_calls != max_calls
        or args.spend_stop_usd != spend
        or args.max_steps != 100
    ):
        raise ValueError("agserve_tau2_retail_live_contract_drifted")
    if Path(args.output_json).exists():
        raise FileExistsError("agserve_tau2_retail_output_exists_no_overwrite")

    runtime_environment_sha256 = validate_router_runtime_environment()
    dependency_sha = activate_tau2_dependency_environment(args.tau2_runtime_repo)
    artifact = inspect_agserve_artifact(args.artifact_root)
    if (
        artifact.metadata.get("training_profile") != "tau2_retail"
        or artifact.metadata.get("check_frequencies", {}).get("tau2_retail") != 2
    ):
        raise ValueError("agserve_tau2_retail_artifact_profile_mismatch")
    native_api = load_tau2_native_api(args.tau2_runtime_repo)
    registry = load_tau2_official_test_registry(args.tau2_repo, domain="retail")
    gate = json.loads(Path(args.split_gate_json).read_text(encoding="utf-8"))
    gate_entry = gate["surface_splits"]["tau2_retail"]
    if (
        gate_entry.get("official_test_count") != 40
        or gate_entry.get("test_registry_sha256") != registry.test_registry_sha256
        or gate_entry.get("source_binding_sha256") != registry.source_binding_sha256
    ):
        raise ValueError("agserve_tau2_retail_split_gate_mismatch")

    fit_dev_registry_sha256 = str(gate_entry["dev_registry_sha256"])
    if args.mode == "official_test":
        tasks = tuple(
            task
            for ordinal, task in enumerate(registry.tasks)
            if ordinal % args.shard_count == args.shard_index
        )
    else:
        train_rows = _load_tau2_train(Path(args.tau2_repo).resolve(), domain="retail")
        fit, dev = deterministic_fit_dev(train_rows)
        FitDevRegistry(
            surface="tau2_retail",
            split_role="router_fit",
            source_binding_sha256=registry.source_binding_sha256,
            registry_sha256=str(gate_entry["fit_registry_sha256"]),
            rows=fit,
        )
        dev_registry = FitDevRegistry(
            surface="tau2_retail",
            split_role="router_dev",
            source_binding_sha256=registry.source_binding_sha256,
            registry_sha256=fit_dev_registry_sha256,
            rows=dev,
        )
        tasks = tuple(
            Tau2OfficialTaskRef(
                native_task_locator=row.native_locator["task_id"],
                payload_sha256=row.payload_sha256,
                example_sha256=row.example_sha256,
            )
            for row in dev_registry.rows[:3]
        )
    expected_rows = contract["rows"] // args.shard_count
    if len(tasks) != expected_rows:
        raise ValueError("agserve_tau2_retail_denominator_mismatch")

    private_root = Path(args.private_root).resolve() / args.mode
    if args.shard_count > 1:
        private_root /= f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
    artifact_sha = _digest(
        {
            "judge_bundle": artifact.bundle_sha256,
            "test_registry_sha256": registry.test_registry_sha256,
            "fit_dev_registry_sha256": fit_dev_registry_sha256,
            "runtime_environment_sha256": runtime_environment_sha256,
            "dependency_environment_sha256": dependency_sha,
            "source_sha256s": {
                "live_script": _file_sha256(Path(__file__)),
                "controller": _file_sha256(
                    REPO_ROOT / "src/papc/baselines/agserve_qac.py"
                ),
                "runtime": _file_sha256(
                    REPO_ROOT / "src/papc/baselines/agserve_qac_runtime.py"
                ),
                "tau2_native": _file_sha256(
                    REPO_ROOT / "src/papc/baselines/tau2_native_runtime.py"
                ),
            },
        }
    )
    return LiveContext(
        mode=args.mode,
        tasks=tasks,
        registry=registry,
        native_api=native_api,
        artifact=artifact,
        artifact_sha256=artifact_sha,
        fit_dev_registry_sha256=fit_dev_registry_sha256,
        dependency_environment_sha256=dependency_sha,
        private_root=private_root,
        api_base=args.api_base.rstrip("/"),
        max_calls=max_calls,
        spend_stop_usd=spend,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )


def _execute_context(
    context: LiveContext,
    *,
    judge: TransformerAgServeJudge,
) -> tuple[list[dict[str, Any]], ProviderAttemptLedger]:
    context.private_root.mkdir(parents=True, exist_ok=False)
    (context.private_root / "cases").mkdir()
    (context.private_root / "raw").mkdir()
    stage = "agserve_tau2_retail"
    job_id = f"{context.mode}_shard_{context.shard_index}"
    ledger = ProviderAttemptLedger(
        global_attempt_cap=context.max_calls,
        global_observed_usd_stop=Decimal(str(context.spend_stop_usd)),
        stage_attempt_caps={stage: context.max_calls},
        stage_observed_usd_stops={stage: Decimal(str(context.spend_stop_usd))},
        job_attempt_caps={job_id: (stage, context.max_calls)},
    )
    broker = ProviderAttemptBroker(ledger=ledger, job_id=job_id)
    results: list[dict[str, Any]] = []
    for task in context.tasks:
        raw_root = context.private_root / "raw" / task.example_sha256
        raw_root.mkdir()
        config = Tau2NativeRunConfig(
            api_base=context.api_base,
            private_output_root=raw_root,
            max_provider_calls=context.max_calls,
            max_observed_provider_cost_usd=context.spend_stop_usd,
            max_infra_retries=0,
        )
        if context.mode == "official_test":
            executor = Tau2NativeExecutor.for_custom_official_episode(
                api=context.native_api,
                task=task,
                registry=context.registry,
                config=config,
                attempt_broker=broker,
                runtime_label=AGSERVE_QAC_METHOD,
            )
        else:
            executor = Tau2NativeExecutor.for_fit_dev_pristine_episode(
                api=context.native_api,
                task=task,
                domain="retail",
                config=config,
                source_binding_sha256=context.registry.source_binding_sha256,
                attempt_broker=broker,
            )
        decisions = []
        local_receipts = []

        def register(generate_callback: Any) -> str:
            return register_agserve_tau2_agent(
                api=context.native_api,
                agent_name=(
                    f"papc_agserve_qac_retail_{task.example_sha256[:10]}_"
                    f"{uuid4().hex[:10]}"
                ),
                judge=judge,
                artifact=context.artifact,
                generate_callback=generate_callback,
                decisions=decisions,
                local_receipts=local_receipts,
                api_base=context.api_base,
                max_steps=config.max_steps,
            )

        episode = executor.run_registered_mixed_model_agent(
            register_agent=register,
            agent_model=CHEAP_MODEL,
        )
        receipts = (*local_receipts, *episode.receipts)
        accounting = aggregate_online_accounting(
            receipts,
            prices=FROZEN_CACHE_NEUTRAL_PRICES,
        )
        agent_receipts = tuple(
            item
            for item in episode.receipts
            if item.role in {"cheap_agent", "strong_agent"}
        )
        if len(agent_receipts) != len(decisions):
            raise RuntimeError("agserve_tau2_retail_decision_receipt_mismatch")
        case = {
            "schema": "papc.agserve_qac_tau2_retail_case.v1",
            "mode": context.mode,
            "artifact_sha256": context.artifact_sha256,
            "example_sha256": task.example_sha256,
            "official_score": episode.official_score,
            "termination_reason": episode.termination_reason,
            "decisions": [item.public_record() for item in decisions],
            "receipts": [item.public_record() for item in receipts],
            "accounting": accounting,
            "public_transcript": [dict(item) for item in episode.public_transcript],
            "raw_manifest": list(executor.private_simulation_manifest),
        }
        case_path = context.private_root / "cases" / f"{task.example_sha256}.json"
        _write_json_no_replace(case_path, case)
        results.append(case)
    ledger.assert_job_quiescent(job_id)
    return results, ledger


def _load_completed_cases(root: Path) -> list[dict[str, Any]]:
    cases = Path(root) / "cases"
    if not cases.is_dir():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(cases.glob("*.json"))
    ]


def _result_report(
    context: LiveContext,
    *,
    results: list[dict[str, Any]],
    ledger: ProviderAttemptLedger | None,
    decision: str,
    blockers: tuple[str, ...],
    secret_reads: int,
) -> dict[str, Any]:
    receipts = [
        receipt
        for case in results
        for receipt in case.get("receipts", [])
    ]
    roles = Counter(
        item["selected_role"]
        for case in results
        for item in case.get("decisions", [])
    )
    successes = int(sum(float(item["official_score"]) for item in results))
    agent_cost = sum(
        float(case["accounting"]["agent_side"]["cost_usd"])
        for case in results
    )
    provider_calls = sum(int(item.get("provider_calls", 0)) for item in receipts)
    return {
        "schema": SCHEMA,
        "decision": decision,
        "execute": True,
        "mode": context.mode,
        "method": AGSERVE_QAC_METHOD,
        "context": context.public_record(),
        "evaluated": len(results),
        "required_denominator": len(context.tasks),
        "raw_successes": successes,
        "accuracy_percent": (
            round(100.0 * successes / len(results), 2) if results else None
        ),
        "agent_side_cost_usd": round(agent_cost, 9),
        "provider_calls": provider_calls,
        "router_role_turns": dict(sorted(roles.items())),
        "ledger": ledger.snapshot() if ledger is not None else None,
        "secret_reads": secret_reads,
        "api_spend_performed": provider_calls > 0,
        "official_test_outcomes_read": context.mode == "official_test",
        "point_estimate_is_observed_run": context.mode == "official_test",
        "complete_full_denominator": (
            context.mode == "official_test" and len(results) == len(context.tasks)
        ),
        "blockers": list(blockers),
        "public_safe": True,
    }


def _blocked_report(
    args: argparse.Namespace,
    exc: Exception,
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
        "blockers": [f"{type(exc).__name__}:{exc}"],
        "public_safe": True,
    }


def _validate_shard(mode: str, count: int, index: int) -> None:
    if mode == "smoke":
        if count != 1 or index != 0:
            raise ValueError("agserve_tau2_retail_smoke_must_not_be_sharded")
        return
    if count not in OFFICIAL_SHARD_COUNTS or not 0 <= index < count:
        raise ValueError("agserve_tau2_retail_shard_invalid")


def _proxy_blockers(env: Mapping[str, str]) -> list[str]:
    blockers = []
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        if env.get(key):
            blockers.append(f"proxy_environment_must_be_unset:{key}")
    no_proxy = env.get("NO_PROXY", "")
    if "localhost" not in no_proxy or "127.0.0.1" not in no_proxy:
        blockers.append("NO_PROXY_must_include_localhost_and_127.0.0.1")
    return blockers


def _write_json_no_replace(path: Path, payload: dict[str, Any]) -> None:
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
                "decision": report.get("decision"),
                "mode": report.get("mode"),
                "evaluated": report.get("evaluated", 0),
                "raw_successes": report.get("raw_successes"),
                "provider_calls": report.get("provider_calls", 0),
                "agent_side_cost_usd": report.get("agent_side_cost_usd"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
