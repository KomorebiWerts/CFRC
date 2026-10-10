#!/usr/bin/env python3
"""Freeze and collect method-neutral paired endpoints on first-16 dev rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import papc.baselines.agent_diff_router_driver as agent_driver  # noqa: E402
from papc.baselines.external_router_execution import (  # noqa: E402
    AGENT_DIFF,
    SHARED_STAGE,
    STATE_BENCH,
    ExecutionJob,
    ProviderAttemptBroker,
    ProviderAttemptLedger,
    load_execution_plan,
)
from papc.baselines.external_router_fit_dev import (  # noqa: E402
    FitDevSourceRow,
    StageItemInput,
    load_fit_dev_registry,
)
from papc.baselines.external_router_fit_dev_native import (  # noqa: E402
    NativeFitDevAuthority,
    build_native_shared_endpoint_executor,
    load_native_fit_dev_authority,
)
from papc.baselines.router_accounting import (  # noqa: E402
    FROZEN_CACHE_NEUTRAL_PRICES,
    aggregate_online_accounting,
)
from papc.baselines.router_contract import (  # noqa: E402
    CHEAP_MODEL,
    STRONG_MODEL,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.shared_dev16_endpoint_gate.v1"
ROW_SCHEMA = "papc.shared_dev16_endpoint_row.v1"
RESULT_SCHEMA = "papc.shared_dev16_endpoint_result.v1"
DEV_COUNT = 16
SURFACES = (STATE_BENCH, AGENT_DIFF)
ENDPOINTS = ("cheap", "strong")

DEFAULT_SPLIT_GATE = (
    REPO_ROOT / "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)
DEFAULT_BUDGET_GATE = (
    REPO_ROOT / "outputs/papc/external_router_budget_gate_noapi_20260730_v10.json"
)
DEFAULT_AUTHORITY = (
    REPO_ROOT / "outputs/papc/table1_authority_manifest_noapi_20260730_v2.json"
)
DEFAULT_PRIVATE_ROOT = Path(
    f"{DATA_ROOT}/papc/shared_endpoint_dev16_calibration_v1"
)
DEFAULT_STATE_REPO = Path(f"{DATA_ROOT}/benchmarks/state_bench/repo")
DEFAULT_TAU2_REPO = Path(
    f"{DATA_ROOT}/benchmarks/tau2-bench-revised/repo"
)
DEFAULT_TAU2_RUNTIME_REPO = Path(
    f"{DATA_ROOT}/benchmarks/tau_bench/repo"
)
DEFAULT_AGENT_DIFF_REPO = Path(
    f"{DATA_ROOT}/benchmarks/agent_diff/repo"
)
DEFAULT_AGENT_DIFF_DATASET = (
    DEFAULT_AGENT_DIFF_REPO / "datasets/agent-diff-bench"
)
DEFAULT_AGENT_DIFF_SDK = DEFAULT_AGENT_DIFF_REPO / "sdk/agent-diff-python"
DEFAULT_AGENT_DIFF_SCORER = (
    REPO_ROOT / "scripts/agent_diff/papc_agent_diff_scoring.py"
)
DEFAULT_AGENT_DIFF_AUTHORITY = (
    REPO_ROOT / "scripts/agent_diff/papc_agent_diff_hf_test_baseline_run.py"
)
CURRENT_AGENT_DIFF_AUTHORITY_SHA256 = (
    "639bca2b9749f67dc039c0eb6e4afbc252ba767a21a7f168b2faa7cc2d5a4bcf"
)
CURRENT_AGENT_DIFF_AUTHORITY_DIFF_SHA256 = (
    "7a84ccd4eb0f9c7bc508ce0ceb698ab163dfeaad62e7418221a71672bcc76e36"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", choices=SURFACES, required=True)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE)
    parser.add_argument("--budget-gate-json", type=Path, default=DEFAULT_BUDGET_GATE)
    parser.add_argument("--authority-json", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--state-repo", type=Path, default=DEFAULT_STATE_REPO)
    parser.add_argument("--tau2-repo", type=Path, default=DEFAULT_TAU2_REPO)
    parser.add_argument(
        "--tau2-runtime-repo", type=Path, default=DEFAULT_TAU2_RUNTIME_REPO
    )
    parser.add_argument(
        "--agent-diff-repo", type=Path, default=DEFAULT_AGENT_DIFF_REPO
    )
    parser.add_argument(
        "--agent-diff-dataset", type=Path, default=DEFAULT_AGENT_DIFF_DATASET
    )
    parser.add_argument(
        "--agent-diff-sdk", type=Path, default=DEFAULT_AGENT_DIFF_SDK
    )
    parser.add_argument(
        "--agent-diff-scorer", type=Path, default=DEFAULT_AGENT_DIFF_SCORER
    )
    parser.add_argument(
        "--agent-diff-authority", type=Path, default=DEFAULT_AGENT_DIFF_AUTHORITY
    )
    parser.add_argument("--agent-diff-base-url", default="http://localhost:8020")
    parser.add_argument("--api-base", default="provider-zero.invalid")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_frozen_write(path: Path, payload: Any) -> str:
    raw = _canonical_bytes(payload)
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f"frozen_output_differs:{path}")
        return hashlib.sha256(raw).hexdigest()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(raw)
        temporary = Path(handle.name)
    temporary.chmod(0o600 if str(path).startswith(f"{DATA_ROOT}/") else 0o644)
    temporary.replace(path)
    return hashlib.sha256(raw).hexdigest()


def _authority_diff_sha256(path: Path) -> str:
    relative = Path(path).resolve().relative_to(REPO_ROOT)
    completed = subprocess.run(
        ["git", "diff", "HEAD", "--binary", "--", str(relative)],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _load_current_agent_diff_api(args: argparse.Namespace) -> Any:
    sdk = args.agent_diff_sdk.resolve()
    scorer = args.agent_diff_scorer.resolve()
    authority = args.agent_diff_authority.resolve()
    authority_sha = _file_sha256(authority)
    diff_sha = _authority_diff_sha256(authority)
    if authority_sha != CURRENT_AGENT_DIFF_AUTHORITY_SHA256:
        raise ValueError("agent_diff_current_authority_sha256_mismatch")
    if diff_sha != CURRENT_AGENT_DIFF_AUTHORITY_DIFF_SHA256:
        raise ValueError("agent_diff_current_authority_diff_sha256_mismatch")
    revision = agent_driver._git_output(sdk, "rev-parse", "HEAD")
    sdk_sha = agent_driver._python_tree_sha256(sdk)
    scorer_sha = agent_driver._file_sha256(scorer)
    if revision != agent_driver.FROZEN_AGENT_DIFF_REPO_REVISION:
        raise ValueError("agent_diff_repo_revision_not_frozen")
    if sdk_sha != agent_driver.FROZEN_AGENT_DIFF_SDK_SOURCE_SHA256:
        raise ValueError("agent_diff_sdk_source_not_frozen")
    if scorer_sha != agent_driver.FROZEN_AGENT_DIFF_SCORER_SHA256:
        raise ValueError("agent_diff_scorer_source_not_frozen")
    if str(sdk) not in sys.path:
        sys.path.insert(0, str(sdk))
    import agent_diff  # type: ignore

    scoring = agent_driver._load_module_from_path(
        "_papc_shared_dev16_agent_diff_scoring", scorer
    )
    if getattr(scoring, "SCHEMA", None) != agent_driver.AGENT_DIFF_SCORER_CORRECTION_SCHEMA:
        raise ValueError("agent_diff_scorer_schema_mismatch")
    if getattr(scoring, "SCORER_NAME", None) != agent_driver.AGENT_DIFF_SCORER_NAME:
        raise ValueError("agent_diff_scorer_name_mismatch")
    apply_corrections = getattr(
        scoring, "apply_registered_scorer_corrections", None
    )
    agent_class = getattr(agent_diff, "AgentDiff", None)
    bash_class = getattr(agent_diff, "BashExecutorProxy", None)
    if not callable(apply_corrections):
        raise RuntimeError("agent_diff_registered_scorer_api_missing")
    if not isinstance(agent_class, type) or not isinstance(bash_class, type):
        raise RuntimeError("agent_diff_native_sdk_api_missing")
    return agent_driver.AgentDiffNativeAPI(
        sdk_path=sdk,
        repo_revision=revision,
        sdk_source_sha256=sdk_sha,
        scorer_sha256=scorer_sha,
        baseline_authority_sha256=authority_sha,
        agent_diff_class=agent_class,
        bash_executor_class=bash_class,
        apply_scorer_corrections=apply_corrections,
    )


def _load_authority(args: argparse.Namespace) -> NativeFitDevAuthority:
    if args.surface == STATE_BENCH:
        return load_native_fit_dev_authority(
            surfaces=(STATE_BENCH,),
            private_root=args.private_root,
            state_repo=args.state_repo,
            tau2_data_repo=args.tau2_repo,
            tau2_runtime_repo=args.tau2_runtime_repo,
            agent_diff_repo=args.agent_diff_repo,
            agent_diff_dataset=args.agent_diff_dataset,
            agent_diff_sdk_path=args.agent_diff_sdk,
            agent_diff_scorer_path=args.agent_diff_scorer,
            agent_diff_baseline_authority_path=args.agent_diff_authority,
            agent_diff_base_url=args.agent_diff_base_url,
        )
    agent_api = _load_current_agent_diff_api(args)
    return NativeFitDevAuthority(
        surfaces=(AGENT_DIFF,),
        private_root=args.private_root.resolve(),
        agent_diff_dataset=args.agent_diff_dataset.resolve(),
        agent_diff_base_url=args.agent_diff_base_url.rstrip("/"),
        agent_diff_api=agent_api,
    )


def _find_full_job(plan: Any, surface: str, endpoint: str) -> ExecutionJob:
    method = f"shared_{endpoint}_endpoint"
    matches = tuple(
        job
        for job in plan.jobs
        if job.stage == SHARED_STAGE
        and job.kind == "shared_endpoint_episode"
        and job.surface == surface
        and job.split_role == "router_dev"
        and job.method == method
    )
    if len(matches) != 1:
        raise ValueError(f"dev16_full_job_not_unique:{surface}:{endpoint}")
    return matches[0]


def _selected_rows(args: argparse.Namespace, job: ExecutionJob) -> tuple[FitDevSourceRow, ...]:
    registry = load_fit_dev_registry(
        job=job,
        split_gate_json=args.split_gate_json,
        state_repo=args.state_repo,
        tau2_repo=args.tau2_repo,
        agent_diff_dataset=args.agent_diff_dataset,
    )
    if len(registry.rows) < DEV_COUNT:
        raise ValueError("dev16_registry_too_small")
    return tuple(registry.rows[:DEV_COUNT])


def _gate_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    public = (
        REPO_ROOT
        / f"outputs/papc/shared_dev16_{args.surface}_gate_noapi_20260802_v1.json"
    )
    private = args.private_root / f"{args.surface}_dev16_manifest_private_v1.json"
    return public, private


def _build_gate(args: argparse.Namespace) -> tuple[dict[str, Any], tuple[FitDevSourceRow, ...], Any, NativeFitDevAuthority]:
    plan = load_execution_plan(
        split_gate_json=args.split_gate_json,
        budget_gate_json=args.budget_gate_json,
        authority_manifest_json=args.authority_json,
        private_root=args.private_root,
    )
    cheap_job = _find_full_job(plan, args.surface, "cheap")
    strong_job = _find_full_job(plan, args.surface, "strong")
    cheap_rows = _selected_rows(args, cheap_job)
    strong_rows = _selected_rows(args, strong_job)
    if tuple(row.example_sha256 for row in cheap_rows) != tuple(
        row.example_sha256 for row in strong_rows
    ):
        raise ValueError("dev16_paired_membership_mismatch")
    selected_record = [
        {
            "ordinal": ordinal,
            "surface": row.surface,
            "stratum": row.stratum,
            "payload_sha256": row.payload_sha256,
            "example_sha256": row.example_sha256,
        }
        for ordinal, row in enumerate(cheap_rows, start=1)
    ]
    selection_sha = _digest(selected_record)
    public_path, private_path = _gate_paths(args)
    private_payload = {
        "schema": "papc.shared_dev16_private_manifest.v1",
        "surface": args.surface,
        "selection_rule": "first_16_rows_in_frozen_router_dev_registry_order",
        "selection_sha256": selection_sha,
        "rows": [row.private_record() for row in cheap_rows],
        "official_test_outcomes_read": False,
    }
    private_sha = _atomic_frozen_write(private_path, private_payload)
    authority = _load_authority(args)
    authority_record = authority.public_record()
    source_authority = {
        "split_gate_sha256": _file_sha256(args.split_gate_json),
        "budget_gate_sha256": _file_sha256(args.budget_gate_json),
        "authority_manifest_sha256": _file_sha256(args.authority_json),
        "dev_registry_sha256": cheap_job.registry_sha256,
        "source_binding_sha256": cheap_job.source_binding_sha256,
        "selection_sha256": selection_sha,
        "private_manifest_sha256": private_sha,
    }
    if args.surface == AGENT_DIFF:
        source_authority["agent_diff_baseline_authority_sha256"] = _file_sha256(
            args.agent_diff_authority
        )
        source_authority["agent_diff_baseline_authority_diff_sha256"] = (
            _authority_diff_sha256(args.agent_diff_authority)
        )
    gate = {
        "schema": SCHEMA,
        "decision": "ready_for_explicit_execute",
        "surface": args.surface,
        "split_role": "router_dev",
        "endpoint_roles": list(ENDPOINTS),
        "models": {"cheap": CHEAP_MODEL, "strong": STRONG_MODEL},
        "selected_count": DEV_COUNT,
        "selection_rule": "first_16_rows_in_frozen_router_dev_registry_order",
        "selection_is_prefrozen_membership_not_runtime_routing": True,
        "selected_rows": selected_record,
        "source_authority": source_authority,
        "native_runtime_authority": authority_record,
        "method_neutral_endpoint_data": True,
        "scorer_fields_in_endpoint_input": False,
        "dev_labels_use": "threshold_calibration_only",
        "official_test_rows_read": False,
        "official_test_outcomes_read": False,
        "runtime_routing_uses_task_or_example_id": False,
        "runtime_tie_break_uses_task_or_example_id": False,
        "max_concurrency_per_surface": 1,
        "max_retries": 0,
        "first_infrastructure_error_action": "stop_surface_and_peer",
        "execution_requested": False,
        "provider_calls": 0,
        "secret_reads": 0,
        "api_spend_performed": False,
        "authorizes_paid_run": False,
        "blockers": ["execute_flag_missing"],
    }
    _atomic_frozen_write(public_path, gate)
    return gate, cheap_rows, plan, authority


def _synthetic_job(
    *,
    full_job: ExecutionJob,
    endpoint: str,
    selection_sha256: str,
    private_root: Path,
) -> ExecutionJob:
    relative = f"{full_job.surface}/{endpoint}/dev16_rows.jsonl"
    return replace(
        full_job,
        job_id=f"shared_dev16.{full_job.surface}.{endpoint}",
        item_count=DEV_COUNT,
        provider_attempt_hard_cap=DEV_COUNT * full_job.attempts_per_item,
        registry_sha256=selection_sha256,
        relative_output_path=relative,
        output_path=(private_root / relative).resolve(),
    )


def _proxy_blockers() -> list[str]:
    blockers = []
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if os.environ.get(name):
            blockers.append(f"proxy_must_be_unset:{name}")
    no_proxy = os.environ.get("NO_PROXY", "")
    if "localhost" not in no_proxy or "127.0.0.1" not in no_proxy:
        blockers.append("NO_PROXY_must_include_localhost_and_127.0.0.1")
    return blockers


def _execute(
    args: argparse.Namespace,
    *,
    rows: tuple[FitDevSourceRow, ...],
    plan: Any,
    authority: NativeFitDevAuthority,
    gate: dict[str, Any],
) -> dict[str, Any]:
    blockers = _proxy_blockers()
    if blockers:
        raise RuntimeError(";".join(blockers))
    if args.api_key_env != "OPENAI_API_KEY":
        raise ValueError("api_key_env_not_frozen")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError("api_key_missing")
    if not args.api_base.startswith(("http://", "https://")):
        raise ValueError("api_base_invalid")
    selection_sha = gate["source_authority"]["selection_sha256"]
    jobs = tuple(
        _synthetic_job(
            full_job=_find_full_job(plan, args.surface, endpoint),
            endpoint=endpoint,
            selection_sha256=selection_sha,
            private_root=args.private_root,
        )
        for endpoint in ENDPOINTS
    )
    caps = {job.job_id: (SHARED_STAGE, job.provider_attempt_hard_cap) for job in jobs}
    total_cap = sum(job.provider_attempt_hard_cap for job in jobs)
    ledger = ProviderAttemptLedger(
        global_attempt_cap=total_cap,
        global_observed_usd_stop=Decimal("75"),
        stage_attempt_caps={SHARED_STAGE: total_cap},
        stage_observed_usd_stops={SHARED_STAGE: Decimal("75")},
        job_attempt_caps=caps,
    )
    executor = build_native_shared_endpoint_executor(
        authority=authority,
        api_base=args.api_base.rstrip("/"),
        api_key=api_key,
        max_retries=0,
        max_observed_provider_cost_usd=75.0,
        api_key_env=args.api_key_env,
    )
    completed_endpoints: list[dict[str, Any]] = []
    for job, endpoint in zip(jobs, ENDPOINTS, strict=True):
        row_dir = job.output_path.parent / "rows"
        if job.output_path.exists() or row_dir.exists():
            raise FileExistsError(f"dev16_endpoint_output_exists:{endpoint}")
        row_dir.mkdir(parents=True, exist_ok=False)
        broker = ProviderAttemptBroker(ledger=ledger, job_id=job.job_id)
        private_rows: list[dict[str, Any]] = []
        all_receipts = []
        for ordinal, source in enumerate(rows, start=1):
            result = executor(
                job,
                StageItemInput(ordinal=ordinal, source=source),
                broker,
            )
            row = {
                "schema": ROW_SCHEMA,
                "surface": args.surface,
                "split_role": "router_dev16",
                "endpoint_role": endpoint,
                "model": CHEAP_MODEL if endpoint == "cheap" else STRONG_MODEL,
                "ordinal": ordinal,
                "payload_sha256": source.payload_sha256,
                "example_sha256": source.example_sha256,
                "selection_sha256": selection_sha,
                **dict(result.output),
                "receipts": [receipt.public_record() for receipt in result.receipts],
            }
            _atomic_frozen_write(row_dir / f"row_{ordinal:02d}.json", row)
            private_rows.append(row)
            all_receipts.extend(result.receipts)
        payload = b"".join(_canonical_bytes(row) for row in private_rows)
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        if job.output_path.exists():
            raise FileExistsError(f"dev16_endpoint_summary_exists:{endpoint}")
        job.output_path.write_bytes(payload)
        job.output_path.chmod(0o600)
        accounting = aggregate_online_accounting(
            all_receipts, prices=FROZEN_CACHE_NEUTRAL_PRICES
        )
        public_result = {
            "schema": RESULT_SCHEMA,
            "surface": args.surface,
            "split_role": "router_dev16",
            "endpoint_role": endpoint,
            "model": CHEAP_MODEL if endpoint == "cheap" else STRONG_MODEL,
            "completed_count": len(private_rows),
            "official_score_sum": sum(
                float(row["official_score"]) for row in private_rows
            ),
            "selection_sha256": selection_sha,
            "private_rows_sha256": hashlib.sha256(payload).hexdigest(),
            "private_rows_path": str(job.output_path),
            "accounting": accounting,
            "provider_calls": accounting["agent_side"]["provider_calls"]
            + accounting["excluded_harness"]["provider_calls"],
            "infra_failure_count": 0,
            "official_test_outcomes_read": False,
            "runtime_routing_uses_task_or_example_id": False,
            "max_concurrency": 1,
            "max_retries": 0,
        }
        public_path = (
            REPO_ROOT
            / f"outputs/papc/shared_dev16_{args.surface}_{endpoint}_result_20260802_v1.json"
        )
        _atomic_frozen_write(public_path, public_result)
        completed_endpoints.append(public_result)
    return {
        "schema": "papc.shared_dev16_surface_dispatch.v1",
        "decision": "complete",
        "surface": args.surface,
        "completed_endpoints": completed_endpoints,
        "ledger": ledger.snapshot(),
        "first_infrastructure_error_count": 0,
        "official_test_outcomes_read": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        gate, rows, plan, authority = _build_gate(args)
        if not args.execute:
            print(json.dumps(gate, sort_keys=True))
            return 0
        report = _execute(
            args,
            rows=rows,
            plan=plan,
            authority=authority,
            gate=gate,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - exact first failure is the stop signal.
        report = {
            "schema": "papc.shared_dev16_surface_dispatch.v1",
            "decision": "blocked" if not args.execute else "execution_failed_stop",
            "surface": args.surface,
            "execution_requested": bool(args.execute),
            "error_type": type(exc).__name__,
            "error": str(exc) or type(exc).__name__,
            "first_infrastructure_error_action": "stop_surface_and_peer",
        }
        print(json.dumps(report, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
