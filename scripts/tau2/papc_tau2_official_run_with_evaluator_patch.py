#!/usr/bin/env python3
"""Run the official tau2 CLI with an explicit evaluator model patch.

The wrapper keeps tau2's official CLI, tasks, tools, trajectories, and metrics.
It only replaces tau2's default evaluator/interface LLM settings, because the
upstream defaults may point at unavailable models in our OpenAI-compatible
runtime. Use it when an official tau2 baseline must still run through
``tau2 run`` but evaluator models must be explicit and reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from papc.certspec.tau2.noapi_readiness import (  # noqa: E402
    BLOCKED_DECISION,
    EXPECTED_APPROVAL_SCOPE,
    SUPPORTED_LAUNCH_DOMAINS,
    build_official_launch_report,
    public_leakage_audit,
    write_json_append_only,
)
from papc.paths import DATA_ROOT

DEFAULT_TAU_REPO = Path(f"{DATA_ROOT}/benchmarks/tau_bench/repo")
DEFAULT_DRY_RUN_OUTPUT = Path(
    "outputs/papc/tau2_retail_official_launch_guard_noapi_20260620.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-repo", type=Path, default=DEFAULT_TAU_REPO)
    parser.add_argument("--evaluator-model", required=True)
    parser.add_argument("--evaluator-reasoning-effort", default=None)
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run-launch-report", action="store_true")
    parser.add_argument("--launch-arm", choices=("official_baseline", "final_papc"))
    parser.add_argument(
        "--launch-domain", choices=SUPPORTED_LAUNCH_DOMAINS, default="retail"
    )
    parser.add_argument("--fixed-manifest-json", type=Path)
    parser.add_argument("--cost-ledger-json", type=Path)
    parser.add_argument("--public-output-root", type=Path)
    parser.add_argument("--approval-scope")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_DRY_RUN_OUTPUT)
    parser.add_argument("--summarize-results-json", type=Path)
    parser.add_argument("--summary-arm-name", choices=("cheap_full", "strong_full"))
    parser.add_argument("--summary-max-concurrency", type=int)
    parser.add_argument(
        "tau2_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the official tau2 CLI. Prefix with --, e.g. -- run --domain retail ...",
    )
    args = parser.parse_args(argv)
    if args.tau2_args and args.tau2_args[0] == "--":
        args.tau2_args = args.tau2_args[1:]
    if not args.tau2_args and args.summarize_results_json is None:
        parser.error("missing tau2 CLI arguments after --")
    return args


def evaluator_kwargs(*, api_base: str, reasoning_effort: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "drop_params": True,
    }
    if api_base:
        kwargs["api_base"] = api_base
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return kwargs


def import_tau2(tau_repo: Path) -> None:
    tau_src = tau_repo / "src"
    if str(tau_src) not in sys.path:
        sys.path.insert(0, str(tau_src))


def apply_evaluator_patch(
    *,
    evaluator_model: str,
    evaluator_args: dict[str, Any],
    tau_config: Any | None = None,
    nl_assertions: Any | None = None,
    interface_agent: Any | None = None,
) -> None:
    if tau_config is None:
        import tau2.config as tau_config
    if nl_assertions is None:
        import tau2.evaluator.evaluator_nl_assertions as nl_assertions
    if interface_agent is None:
        import tau2.environment.utils.interface_agent as interface_agent

    tau_config.DEFAULT_LLM_NL_ASSERTIONS = evaluator_model
    tau_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(evaluator_args)
    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = evaluator_model
    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(evaluator_args)

    tau_config.DEFAULT_LLM_ENV_INTERFACE = evaluator_model
    tau_config.DEFAULT_LLM_ENV_INTERFACE_ARGS = dict(evaluator_args)
    interface_agent.DEFAULT_LLM_ENV_INTERFACE = evaluator_model
    interface_agent.DEFAULT_LLM_ENV_INTERFACE_ARGS = dict(evaluator_args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarize_results_json is not None:
        if args.summary_arm_name is None:
            raise SystemExit(
                "--summary-arm-name is required with --summarize-results-json"
            )
        payload = json.loads(args.summarize_results_json.read_text(encoding="utf-8"))
        report = build_official_baseline_public_summary(
            run_name=args.summarize_results_json.parent.name,
            results_payload=payload,
            arm_name=args.summary_arm_name,
            max_concurrency=args.summary_max_concurrency,
            evaluator_model=args.evaluator_model,
            evaluator_reasoning_effort=args.evaluator_reasoning_effort,
        )
        write_json_append_only(args.output_json, report)
        print(
            json.dumps(
                {
                    "decision": "summarized_tau2_official_baseline_public",
                    "provider_calls": 0,
                    "api_spend_performed": False,
                    "output": str(args.output_json),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.dry_run_launch_report:
        report = build_official_launch_report(
            tau2_args=args.tau2_args,
            launch_arm=args.launch_arm,
            evaluator_model=args.evaluator_model,
            evaluator_reasoning_effort=args.evaluator_reasoning_effort,
            api_base_supplied=bool(args.api_base),
            fixed_manifest_json=args.fixed_manifest_json,
            cost_ledger_json=args.cost_ledger_json,
            public_output_root=args.public_output_root,
            approval_scope=args.approval_scope,
            launch_domain=args.launch_domain,
            tau_repo=args.tau_repo,
        )
        write_json_append_only(args.output_json, report)
        _print_dry_run_summary(report, args.output_json)
        return 1 if report["decision"] == BLOCKED_DECISION else 0

    execute_blockers = execution_preflight_blockers(
        execute=args.execute,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        environ=os.environ,
        launch_domain=args.launch_domain,
        launch_arm=args.launch_arm,
        approval_scope=args.approval_scope,
    )
    if execute_blockers:
        _print_execution_blocked_summary(execute_blockers)
        return 2

    import_tau2(args.tau_repo)
    evaluator_args = evaluator_kwargs(
        api_base=str(args.api_base or ""),
        reasoning_effort=args.evaluator_reasoning_effort,
    )
    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = str(args.api_base)
    if args.api_key_env != "OPENAI_API_KEY":
        os.environ["OPENAI_API_KEY"] = os.environ[args.api_key_env]
    apply_evaluator_patch(
        evaluator_model=args.evaluator_model,
        evaluator_args=evaluator_args,
    )

    import tau2.cli as tau_cli

    sys.argv = ["tau2", *args.tau2_args]
    tau_cli.main()
    return 0


def build_official_baseline_public_summary(
    *,
    run_name: str,
    results_payload: dict[str, Any],
    arm_name: str,
    max_concurrency: int | None,
    evaluator_model: str,
    evaluator_reasoning_effort: str | None,
) -> dict[str, Any]:
    from scripts.tau2 import papc_tau2_cfrc_clean_run as method_runner

    info = _dict_value(results_payload.get("info"))
    agent_info = _dict_value(info.get("agent_info"))
    user_info = _dict_value(info.get("user_info"))
    environment_info = _dict_value(info.get("environment_info"))
    agent_model = str(agent_info.get("llm") or "")
    user_model = str(user_info.get("llm") or "")
    agent_reasoning_effort = _reasoning_effort(agent_info)
    user_reasoning_effort = _reasoning_effort(user_info)
    if arm_name == "cheap_full":
        cheap_model = agent_model
        strong_model = ""
    elif arm_name == "strong_full":
        cheap_model = ""
        strong_model = agent_model
    else:
        raise ValueError(f"unsupported tau2 official baseline arm: {arm_name}")
    summary_args = SimpleNamespace(
        cheap_model=cheap_model,
        strong_model=strong_model,
    )
    arm = method_runner.arm_public_summary(
        summary_args,
        arm_name=arm_name,
        results_payload=results_payload,
        sidecar_rows=[],
    )
    arms = {arm_name: arm}
    external_time = method_runner.external_time_accounting(arms)
    parallel_elapsed = simulation_parallel_elapsed_seconds(results_payload)
    for row in external_time.values():
        if isinstance(row, dict):
            row["parallel_elapsed_wall_seconds"] = parallel_elapsed
            method_runner.attach_parallel_capacity_accounting(
                row,
                parallel_elapsed_wall_seconds=parallel_elapsed,
                max_concurrency=max_concurrency,
            )
    report = {
        "schema": "papc.tau2.official_baseline_public.v1",
        "run_name": run_name,
        "surface": "tau2",
        "domain": str(environment_info.get("domain_name") or "retail"),
        "baseline_arm": arm_name,
        "task_count": arm["task_count"],
        "reward_evaluable_count": arm["reward_evaluable_count"],
        "infrastructure_excluded_count": arm["infrastructure_excluded_count"],
        "reward_sum": arm["reward_sum"],
        "reward_mean": arm["reward_mean"],
        "arms": arms,
        "model_resource_summary": method_runner.model_resource_summary(arms),
        "external_time_accounting": external_time,
        "primary_resource_comparison": method_runner.primary_resource_comparison_report(),
        "models": {
            "agent": agent_model,
            "agent_role": "cheap" if arm_name == "cheap_full" else "strong",
            "agent_reasoning_effort": agent_reasoning_effort,
            "user_simulator": user_model,
            "user_reasoning_effort": user_reasoning_effort,
            "evaluator": evaluator_model,
            "evaluator_reasoning_effort": evaluator_reasoning_effort,
        },
        "baseline_accounting": {
            "same_accounting_as_method_public_summary": True,
            "primary_comparison_is_model_call_latency_and_tokens": True,
            "total_runtime_is_secondary_accounting": True,
            "role_model_latency_and_tokens_field": "arms.*.runtime_ledger",
            "aggregate_external_time_field": (
                "external_time_accounting.*.aggregate_external_time_seconds"
            ),
            "parallel_elapsed_field": (
                "external_time_accounting.*.parallel_elapsed_wall_seconds"
            ),
            "parallel_capacity_field": (
                "external_time_accounting.*.parallel_capacity_seconds"
            ),
            "parallel_capacity_is_elapsed_times_max_concurrency": (
                max_concurrency is not None and parallel_elapsed is not None
            ),
            "aggregate_external_time_prefers_episode_sum_when_available": True,
            "missing_latency_fields_are_unavailable_not_zero": True,
        },
        "claim_boundary": [
            "official tau2 full-agent baseline public aggregate",
            "not a method-arm result",
            "task ids and raw trajectories are excluded from public output",
            "parallel capacity is wall-clock times max_concurrency, not an exact duration sum when episode durations are available",
        ],
        "provider_calls": 0,
        "api_spend_performed": False,
        "raw_private_paths_excluded": True,
        "raw_task_ids_excluded": True,
        "public_safe": True,
    }
    report["public_leakage_audit"] = public_leakage_audit(report)
    return report


def simulation_parallel_elapsed_seconds(
    results_payload: dict[str, Any],
) -> float | None:
    starts: list[float] = []
    ends: list[float] = []
    simulations = results_payload.get("simulations")
    if not isinstance(simulations, list):
        return None
    for simulation in simulations:
        if not isinstance(simulation, dict):
            continue
        start = _timestamp_seconds(simulation.get("start_time"))
        end = _timestamp_seconds(simulation.get("end_time"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    if not starts or not ends:
        return None
    return round(max(ends) - min(starts), 6)


def _timestamp_seconds(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _reasoning_effort(model_info: dict[str, Any]) -> str | None:
    llm_args = _dict_value(model_info.get("llm_args"))
    value = llm_args.get("reasoning_effort")
    return str(value) if value is not None else None


def execution_preflight_blockers(
    *,
    execute: bool,
    api_base: str | None,
    api_key_env: str | None,
    environ: dict[str, str] | os._Environ[str],
    launch_domain: str = "retail",
    launch_arm: str | None = None,
    approval_scope: str | None = None,
) -> list[str]:
    blockers: list[str] = []
    if not execute:
        blockers.append("execute_flag_missing")
    if not api_base:
        blockers.append("api_base_missing")
    if not api_key_env:
        blockers.append("api_key_env_missing")
    elif not environ.get(api_key_env):
        blockers.append("api_key_env_not_present")
    if launch_domain == "retail":
        blockers.extend(
            retail_execution_preflight_blockers(
                launch_arm=launch_arm,
                approval_scope=approval_scope,
            )
        )
    else:
        blockers.append("launch_domain_must_be_retail")
    return blockers


def retail_execution_preflight_blockers(
    *,
    launch_arm: str | None,
    approval_scope: str | None,
) -> list[str]:
    blockers: list[str] = []
    if launch_arm not in {"official_baseline", "final_papc"}:
        blockers.append("retail_launch_arm_must_be_official_baseline_or_final_papc")
    if approval_scope != EXPECTED_APPROVAL_SCOPE:
        blockers.append("approval_scope_must_match_tau2_retail_execution")
    return blockers


def _print_execution_blocked_summary(blockers: list[str]) -> None:
    print(
        json.dumps(
            {
                "decision": "blocked_tau2_official_execute_preflight",
                "provider_calls": 0,
                "api_spend_performed": False,
                "execution_blockers": sorted(blockers),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _print_dry_run_summary(report: dict[str, Any], output_json: Path) -> None:
    print(
        json.dumps(
            {
                "gate_id": report["gate_id"],
                "decision": report["decision"],
                "ready_for_explicit_approval_request": report["summary"][
                    "ready_for_explicit_approval_request"
                ],
                "online_experiment_ready": report["summary"]["online_experiment_ready"],
                "provider_calls": report["provider_calls"],
                "api_spend_performed": report["api_spend_performed"],
                "blockers": report["settings_blockers"],
                "execution_blockers": report["execution_blockers"],
                "output": str(output_json),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
