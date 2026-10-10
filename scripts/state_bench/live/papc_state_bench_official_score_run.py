#!/usr/bin/env python3
"""Safely run one append-only phase of the official STATE-Bench scorer.

The provider-zero mode binds the source trajectories, online run report,
official protocol, judge model, phase, and implementation.  Online mode copies
the bound trajectories to a fresh private root and calls the benchmark-owned
``score_one`` function without changing its prompts or scoring semantics.
Task completion and UX are separate phases so the primary quality gate can be
evaluated before paying for secondary UX judgments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Literal

_REPO = Path(__file__).resolve().parents[3]
for _path in (_REPO, _REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from papc.certspec.state_bench.true_live_online import (  # noqa: E402
    ensure_optional_azure_identity_importable,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.state_bench.official_score_run.v1"
PHASES = ("task_completion", "ux")
ScorePhase = Literal["task_completion", "ux"]
DOMAINS = ("shopping_assistant", "travel", "customer_support")
MAX_SCORE_CONCURRENCY = 10
CFRC_SOURCE_GROUP = "cfrc_method"
CHEAP_ANCHOR_SOURCE_GROUP = "cheap_anchor"
CFRC_SOURCE_SCHEMA = "papc.state_bench_true_live_cfrc_run.v1"
CFRC_SOURCE_DECISION = "state_bench_true_live_cfrc_complete"
DEFAULT_STATE_BENCH_ROOT = Path(f"{DATA_ROOT}/benchmarks/state_bench/repo")
_PROVIDER_ENV = (
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "STATE_BENCH_EVAL_API_KEY",
    "STATE_BENCH_EVAL_BASE_URL",
    "STATE_BENCH_EVAL_MODEL",
    "STATE_BENCH_EVAL_PROVIDER",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trajectory_execution_projection(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """Fields that official scoring must not change."""

    stable_keys = {
        "agent_name",
        "conversation",
        "state_diff",
        "task_id",
        "tool_calls",
        "tool_errors",
        "turns",
        "user_id",
    }
    return {
        key: value
        for key, value in trajectory.items()
        if key in stable_keys or key.startswith("papc_")
    }


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _trajectory_files(root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for domain in DOMAINS:
        domain_rows = sorted(
            (root / domain / "run1").glob("row_*.json"),
            key=_row_index,
        )
        rows.extend((domain, path) for path in domain_rows)
    identities = [(domain, _row_index(path)) for domain, path in rows]
    if not rows or len(identities) != len(set(identities)):
        raise RuntimeError("STATE score source is empty or has duplicate identities")
    return rows


def _row_index(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("row_") or not stem[4:].isdigit():
        raise RuntimeError(f"invalid STATE trajectory filename: {path.name}")
    return int(stem[4:])


def _source_manifest(root: Path) -> dict[str, Any]:
    files = []
    for domain, path in _trajectory_files(root):
        trajectory = _read_json(path, "STATE score source trajectory")
        files.append(
            {
                "domain": domain,
                "row_index": _row_index(path),
                "execution_sha256": _digest(
                    _trajectory_execution_projection(trajectory)
                ),
            }
        )
    return {
        "row_count": len(files),
        "files": files,
        "bundle_sha256": _digest(files),
    }


RESCUABLE_PRIMARY_BLOCKERS = frozenset(
    {"non_executed_receipts_present", "provider_failure_receipts_present"}
)


def _cfrc_source_eligibility(
    report: Mapping[str, Any],
    *,
    source_root: Path,
    rescue_reports: tuple[Mapping[str, Any], ...] = (),
) -> dict[str, Any]:
    """Bind a unified true-live CFRC method run for official scoring.

    The CFRC run report carries no per-row trajectory digests, so the binding
    is the source-root inventory itself: the trajectory count must equal the
    completed and written row counts declared by the online report, and the
    gate-versus-execute equality of ``observed_bindings`` (which embeds the
    per-file execution digests via ``_source_manifest``) freezes the files
    between the provider-zero gate and the scoring run.

    A primary run whose only defect is provider-failure rows may be completed
    by rescue runs. The salvage binding stays fail-closed: the primary may be
    blocked only by the provider-failure receipt blockers, every rescue run
    must itself be a complete unblocked method run, the primary
    provider-failure row count must equal the total rescued rows, and the
    written trajectories of the primary and rescue runs together must equal
    the source-root inventory.
    """

    summary = report.get("summary")
    leakage = report.get("public_leakage_audit")
    files = _trajectory_files(source_root)
    summary = summary if isinstance(summary, Mapping) else {}
    common_online = (
        report.get("schema") == CFRC_SOURCE_SCHEMA
        and report.get("execute") is True
        and report.get("api_spend_performed") is True
        and bool(summary)
        and int(summary.get("completed_rows") or 0) == len(files)
        and len(files) > 0
        and isinstance(leakage, Mapping)
        and leakage.get("pass") is True
    )
    if not rescue_reports:
        complete_online = (
            common_online
            and report.get("decision") == CFRC_SOURCE_DECISION
            and report.get("blockers") == []
            and int(summary.get("trajectory_written_count") or 0) == len(files)
        )
        if not complete_online:
            raise RuntimeError(
                "STATE cfrc score source is not a complete online method run"
            )
        return {
            "diagnostic_only": False,
            "official_itt_readout_allowed": True,
            "eligible_for_official_scoring": True,
            "rescued_row_count": 0,
        }

    blockers = report.get("blockers")
    label_counts = summary.get("label_counts")
    failure_rows = (
        int(label_counts.get("provider_failure") or 0)
        if isinstance(label_counts, Mapping)
        else -1
    )
    primary_written = int(summary.get("trajectory_written_count") or 0)
    rescue_written = 0
    rescues_complete = True
    for rescue in rescue_reports:
        rescue_summary = rescue.get("summary")
        rescue_summary = rescue_summary if isinstance(rescue_summary, Mapping) else {}
        rescue_leakage = rescue.get("public_leakage_audit")
        rescue_row_count = int(rescue_summary.get("trajectory_written_count") or 0)
        rescues_complete = (
            rescues_complete
            and rescue.get("schema") == CFRC_SOURCE_SCHEMA
            and rescue.get("execute") is True
            and rescue.get("decision") == CFRC_SOURCE_DECISION
            and rescue.get("api_spend_performed") is True
            and rescue.get("blockers") == []
            and rescue_row_count > 0
            and int(rescue_summary.get("completed_rows") or 0) == rescue_row_count
            and isinstance(rescue_leakage, Mapping)
            and rescue_leakage.get("pass") is True
        )
        rescue_written += rescue_row_count
    salvage_bound = (
        common_online
        and rescues_complete
        and isinstance(blockers, list)
        and bool(blockers)
        and set(blockers) <= RESCUABLE_PRIMARY_BLOCKERS
        and failure_rows > 0
        and failure_rows == rescue_written
        and primary_written + rescue_written == len(files)
    )
    if not salvage_bound:
        raise RuntimeError(
            "STATE cfrc score source with rescue reports is not a bound "
            "complete set"
        )
    return {
        "diagnostic_only": False,
        "official_itt_readout_allowed": True,
        "eligible_for_official_scoring": True,
        "rescued_row_count": rescue_written,
    }


def _source_report_eligibility(
    path: Path,
    *,
    source_root: Path,
    source_group: str,
    rescue_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if source_group not in {CFRC_SOURCE_GROUP, CHEAP_ANCHOR_SOURCE_GROUP}:
        raise RuntimeError("STATE score source group is unsupported")
    report = _read_json(path, "STATE score source report")
    if source_group == CHEAP_ANCHOR_SOURCE_GROUP:
        frontier_policy = report.get("frontier_policy")
        if (
            not isinstance(frontier_policy, Mapping)
            or frontier_policy.get("arm") != CHEAP_ANCHOR_SOURCE_GROUP
        ):
            raise RuntimeError("STATE cheap-anchor score source arm is invalid")
    rescue_reports = tuple(
        _read_json(rescue_path, "STATE score rescue report")
        for rescue_path in rescue_paths
    )
    return _cfrc_source_eligibility(
        report,
        source_root=source_root,
        rescue_reports=rescue_reports,
    )

def _protocol_binding(state_bench_root: Path) -> dict[str, Any]:
    root = state_bench_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from state_bench.protocol import load_default_protocol

    protocol = load_default_protocol()
    errors = list(protocol.validate_prompt_hashes())
    if errors:
        raise RuntimeError("official STATE judge prompt hash validation failed")
    return {
        "protocol_id": protocol.protocol_id,
        "official_model": protocol.official_model,
        "judge_reasoning_effort": protocol.judge_reasoning_effort,
        "prompt_hashes_verified": True,
    }


def _observed_bindings(args: argparse.Namespace) -> dict[str, Any]:
    source_eligibility = _source_report_eligibility(
        args.source_report_json,
        source_root=args.source_root,
        source_group=args.source_group,
        rescue_paths=tuple(args.rescue_report_json or ()),
    )
    protocol = _protocol_binding(args.state_bench_root)
    if (
        args.judge_model != protocol["official_model"]
        or args.reasoning_effort != protocol["judge_reasoning_effort"]
    ):
        raise RuntimeError("STATE score model plan differs from official protocol")
    return {
        "source_report_sha256": _file_sha256(args.source_report_json),
        "rescue_report_sha256s": [
            _file_sha256(path) for path in (args.rescue_report_json or [])
        ],
        "source_group": args.source_group,
        "source_eligibility": source_eligibility,
        "source_manifest": _source_manifest(args.source_root),
        "protocol": protocol,
        "score_phase": args.phase,
        "judge_model": args.judge_model,
        "reasoning_effort": args.reasoning_effort,
        "max_concurrency": args.max_concurrency,
        "implementation_sha256": _file_sha256(Path(__file__)),
    }


def _copy_sources(source_root: Path, output_root: Path) -> list[tuple[str, Path]]:
    if output_root.exists():
        raise RuntimeError("refusing to overwrite existing STATE scored root")
    copied: list[tuple[str, Path]] = []
    for domain, source in _trajectory_files(source_root):
        target = output_root / domain / "run1" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append((domain, target))
    return copied


def _build_official_scorers(
    *,
    state_bench_root: Path,
    phase: ScorePhase,
    reasoning_effort: str,
) -> tuple[Any, dict[str, tuple[Path, Any, Any]], Any]:
    root = state_bench_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    ensure_optional_azure_identity_importable()
    from state_bench.client import build_locked_judge_client
    from state_bench.domain import get_domain_config
    from state_bench.paths import domain_tasks_dir
    from state_bench.protocol import load_default_protocol
    from state_bench.scoring import TaskRequirementsJudge, UXQualityJudge
    from state_bench.scripts.score import score_one

    client = build_locked_judge_client()
    scorers: dict[str, tuple[Path, Any, Any]] = {}
    for domain_name in DOMAINS:
        domain = get_domain_config(domain_name)
        task_judge = (
            TaskRequirementsJudge(
                client=client,
                prompts_dir=domain.prompts_dir,
                system_prompt=domain.judge_system_prompt,
                reasoning_effort=reasoning_effort,
            )
            if phase == "task_completion"
            else None
        )
        ux_judge = (
            UXQualityJudge(
                client=client,
                prompts_dir=domain.prompts_dir,
                system_prompt=domain.judge_system_prompt,
                reasoning_effort=reasoning_effort,
            )
            if phase == "ux"
            else None
        )
        scorers[domain_name] = (
            domain_tasks_dir(domain_name),
            task_judge,
            ux_judge,
        )
    return score_one, scorers, load_default_protocol()


def _score_job(
    *,
    domain: str,
    path: Path,
    score_one: Any,
    scorers: Mapping[str, tuple[Path, Any, Any]],
    protocol: Any,
    phase: ScorePhase,
) -> dict[str, Any]:
    tasks_dir, task_judge, ux_judge = scorers[domain]
    try:
        result = score_one(
            path,
            tasks_dir,
            task_judge,
            ux_judge,
            path,
            protocol,
            domain,
        )
        if result.get("status") != "OK":
            return {
                "domain": domain,
                "row_index": _row_index(path),
                "status": "failed",
                "error_type": "OfficialScoreError",
                "private_error": str(result.get("error") or "official scorer error"),
            }
        scored = _read_json(path, "officially scored STATE trajectory")
        required = (
            ("state_requirements_met", "task_requirements_met", "task_completion_pass")
            if phase == "task_completion"
            else ("ux_score",)
        )
        if any(scored.get(key) is None for key in required):
            raise RuntimeError("official scorer omitted required phase fields")
        return {
            "domain": domain,
            "row_index": _row_index(path),
            "status": "complete",
            "execution_sha256": _digest(_trajectory_execution_projection(scored)),
            "task_completion_pass": (
                bool(scored["task_completion_pass"])
                if phase == "task_completion"
                else None
            ),
        }
    except Exception as exc:
        return {
            "domain": domain,
            "row_index": _row_index(path),
            "status": "failed",
            "error_type": type(exc).__name__,
            "private_error": str(exc),
            "trace": traceback.format_exc(),
        }


def _execute_jobs(
    jobs: Sequence[tuple[str, Path]],
    *,
    max_concurrency: int,
    score_one: Any,
    scorers: Mapping[str, tuple[Path, Any, Any]],
    protocol: Any,
    phase: ScorePhase,
) -> tuple[list[dict[str, Any]], bool]:
    iterator = iter(jobs)
    receipts: list[dict[str, Any]] = []
    futures: dict[Future[Any], tuple[str, Path]] = {}
    admission_stopped = False

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            domain, path = next(iterator)
        except StopIteration:
            return False
        future = executor.submit(
            _score_job,
            domain=domain,
            path=path,
            score_one=score_one,
            scorers=scorers,
            protocol=protocol,
            phase=phase,
        )
        futures[future] = (domain, path)
        return True

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        for _ in range(min(max_concurrency, len(jobs))):
            submit_next(executor)
        while futures:
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            completed_batch: list[dict[str, Any]] = []
            for future in done:
                receipt = future.result()
                futures.pop(future)
                completed_batch.append(receipt)
            receipts.extend(completed_batch)
            if any(row.get("status") != "complete" for row in completed_batch):
                admission_stopped = True
            while (
                not admission_stopped
                and len(futures) < max_concurrency
                and submit_next(executor)
            ):
                pass
    receipts.sort(key=lambda row: (str(row["domain"]), int(row["row_index"])))
    return receipts, admission_stopped


def _public_leakage(payload: Mapping[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(payload, sort_keys=True).lower()
    markers = [
        marker
        for marker in ("/home/", "api_key", "private_error", "trace", "task_id")
        if marker in serialized
    ]
    return {"pass": not markers, "markers": markers}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--state-bench-root", type=Path, default=DEFAULT_STATE_BENCH_ROOT
    )
    parser.add_argument("--source-report-json", type=Path, required=True)
    parser.add_argument(
        "--rescue-report-json",
        type=Path,
        action="append",
        default=None,
        help=(
            "Public report of a complete rescue run that re-executed the "
            "provider-failure rows of the primary run. Repeatable; the "
            "primary and rescue reports are bound together fail-closed."
        ),
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-group",
        choices=[CFRC_SOURCE_GROUP, CHEAP_ANCHOR_SOURCE_GROUP],
        default=CFRC_SOURCE_GROUP,
    )
    parser.add_argument("--public-output-json", type=Path, required=True)
    parser.add_argument("--private-output-root", type=Path, required=True)
    parser.add_argument("--private-output-root-label", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--max-concurrency", type=int, default=MAX_SCORE_CONCURRENCY
    )
    parser.add_argument("--api-base")
    parser.add_argument("--api-key-env")
    parser.add_argument("--gate-artifact-json", type=Path)
    parser.add_argument("--expected-gate-artifact-sha256")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.public_output_json.exists() or args.private_output_root.exists():
        raise RuntimeError("refusing to overwrite existing STATE score output")
    if not 1 <= args.max_concurrency <= MAX_SCORE_CONCURRENCY:
        raise RuntimeError("STATE score concurrency is outside the authorized range")
    if not args.execute:
        for key in _PROVIDER_ENV:
            os.environ.pop(key, None)

    observed = _observed_bindings(args)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "mode": "online" if args.execute else "provider_zero_preflight",
        "decision": "running" if args.execute else "ready",
        "provider_calls": None if args.execute else 0,
        "api_spend_performed": bool(args.execute),
        "observed_bindings": observed,
        "private_output_root_label": args.private_output_root_label,
    }
    report["public_leakage_audit"] = _public_leakage(report)
    if report["public_leakage_audit"]["pass"] is not True:
        raise RuntimeError("STATE score public preflight leakage audit failed")
    if not args.execute:
        _write_json_new(args.public_output_json, report)
        return 0

    if (
        not args.api_base
        or args.api_key_env != "OPENAI_API_KEY"
        or not os.environ.get(args.api_key_env)
        or args.gate_artifact_json is None
        or not args.expected_gate_artifact_sha256
    ):
        raise RuntimeError("STATE score online provider configuration is invalid")
    gate = _read_json(args.gate_artifact_json, "STATE score provider-zero gate")
    if (
        _file_sha256(args.gate_artifact_json)
        != args.expected_gate_artifact_sha256
        or gate.get("decision") != "ready"
        or gate.get("provider_calls") != 0
        or gate.get("observed_bindings") != observed
    ):
        raise RuntimeError("STATE score provider-zero gate mismatch")

    os.environ["STATE_BENCH_EVAL_PROVIDER"] = "openai"
    os.environ["STATE_BENCH_EVAL_MODEL"] = args.judge_model
    os.environ["STATE_BENCH_EVAL_BASE_URL"] = args.api_base
    copied = _copy_sources(args.source_root, args.private_output_root)
    score_one, scorers, protocol = _build_official_scorers(
        state_bench_root=args.state_bench_root,
        phase=args.phase,
        reasoning_effort=args.reasoning_effort,
    )
    receipts, admission_stopped = _execute_jobs(
        copied,
        max_concurrency=args.max_concurrency,
        score_one=score_one,
        scorers=scorers,
        protocol=protocol,
        phase=args.phase,
    )
    _write_json_new(
        args.private_output_root / "score_receipts.json", {"receipts": receipts}
    )
    failures = [row for row in receipts if row.get("status") != "complete"]
    public_receipts = [
        {
            key: value
            for key, value in row.items()
            if key not in {"private_error", "trace", "task_completion_pass"}
        }
        for row in receipts
    ]
    report.update(
        {
            "decision": "complete" if not failures else "blocked",
            "provider_calls": None,
            "api_spend_performed": bool(receipts),
            "online_summary": {
                "planned_rows": len(copied),
                "completed_rows": len(receipts) - len(failures),
                "failed_rows": len(failures),
                "logical_judge_invocations": len(receipts),
                "physical_provider_calls_unavailable": True,
                "admission_stopped": admission_stopped,
                "not_admitted_rows": len(copied) - len(receipts),
                "task_completion_pass_count": (
                    sum(bool(row.get("task_completion_pass")) for row in receipts)
                    if args.phase == "task_completion" and not failures
                    else None
                ),
            },
            "online_receipts": public_receipts,
        }
    )
    report["public_leakage_audit"] = _public_leakage(report)
    if report["public_leakage_audit"]["pass"] is not True:
        raise RuntimeError("STATE score public report leakage audit failed")
    _write_json_new(args.public_output_json, report)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
