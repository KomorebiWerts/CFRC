#!/usr/bin/env python3
"""Run one official tau2 Airline single-model baseline arm.

The arm is a plain tau2 run: tau2's own default single-model agent, no reached
state contract, no residual graph, no checked execution. It exists so the
cheap-only and strong-only rows are matched field by field with the CFRC method
arm in ``scripts/tau2/papc_tau2_airline_residual_run.py``. The matching is
achieved by reusing that runner's split registry and the same tau2 config
plumbing rather than by driving the tau2 CLI by hand, because a hand-written CLI
line can silently diverge in an LLM setting that no output would reveal.
"""
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from papc.paths import DATA_ROOT
from papc.certspec.costing import (  # noqa: E402
    cache_neutral_cost_usd,
    public_openai_price_for_model,
    public_pricing_metadata_for_model,
)
from papc.certspec.tau2.noapi_readiness import (  # noqa: E402
    public_leakage_audit,
    write_json_append_only,
)
from papc.certspec.tau2.residual_support import (  # noqa: E402
    llm_args_for_model,
    litellm_model_name,
)
from scripts.tau2 import papc_tau2_airline_residual_run as airline_run  # noqa: E402
from scripts.tau2 import papc_tau2_official_run_with_evaluator_patch as official  # noqa: E402

DOMAIN = "airline"
DEFAULT_TAU_REPO = Path(f"{DATA_ROOT}/benchmarks/tau_bench/repo")
DEFAULT_OUTPUT_ROOT = Path(f"{DATA_ROOT}/papc")
MAX_ONLINE_CONCURRENCY = 10
# tau2's own default single-model agent. The baseline must not be a PAPC agent,
# so the name is fixed here rather than exposed as a flag.
BASELINE_AGENT = "llm_agent"
BASELINE_ARMS = ("cheap_full", "strong_full")
# The shared baseline summary builder reports the single agent under the ledger
# role that matches the arm it is standing in for, so the cost accounting has to
# read the same role back out.
AGENT_LEDGER_ROLE = {"cheap_full": "cheap", "strong_full": "strong"}
LAUNCH_REPORT_SCHEMA = "papc.tau2.airline_official_baseline_launch_report.v1"
PRIVATE_RESULTS_SCHEMA = "papc.tau2.airline_official_baseline_private_results.v1"
COST_ACCOUNTING_SCHEMA = "papc.tau2.airline_official_baseline_agent_side_cost.v1"
READY_DECISION = "ready_for_authorized_baseline_run"
BLOCKED_DECISION = "blocked"


def _task_split(args: argparse.Namespace) -> str:
    return str(
        getattr(args, "task_split", None) or airline_run.DEFAULT_TASK_SPLIT
    )


def split_denominator(task_split: str) -> int:
    """Return the reporting denominator the method runner registered for a split.

    The method arm owns the split registry, so the baseline reads the number
    from there instead of restating it. A baseline row counted against a
    different denominator than the method row is not a matched comparison.
    """

    return int(airline_run.SPLIT_OPERATING_POINTS[task_split]["denominator"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=BASELINE_ARMS, required=True)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--agent-reasoning-effort", required=True)
    parser.add_argument(
        "--task-split",
        choices=sorted(airline_run.SPLIT_OPERATING_POINTS),
        default=airline_run.DEFAULT_TASK_SPLIT,
        help=(
            "Official tau2 task split to report on. The denominator comes from "
            "the method runner's split registry."
        ),
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=310)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--user-reasoning-effort", required=True)
    parser.add_argument("--evaluator-model", required=True)
    parser.add_argument("--evaluator-reasoning-effort", required=True)
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--tau-repo", type=Path, default=DEFAULT_TAU_REPO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--public-output-json", type=Path, required=True)
    parser.add_argument("--dry-run-launch-report", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.domain = DOMAIN
    for field in ("agent_model", "user_model", "evaluator_model"):
        setattr(args, field, litellm_model_name(getattr(args, field)))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_name = args.run_name or f"tau2_airline_official_baseline_{timestamp()}"

    if args.dry_run_launch_report:
        report = build_launch_report(args, run_name=run_name)
        try:
            write_json_append_only(args.public_output_json, report)
        except (FileExistsError, FileNotFoundError, OSError) as exc:
            print(json.dumps({"blocked": str(exc)}))
            return 2
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "provider_calls": 0,
                    "api_spend_performed": False,
                    "blockers": report["blockers"],
                    "output": str(args.public_output_json),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if not report["blockers"] else 2

    blockers = execute_blockers(args)
    if blockers:
        print(
            json.dumps(
                {
                    "decision": "blocked_tau2_airline_official_baseline_execute",
                    "provider_calls": 0,
                    "api_spend_performed": False,
                    "blockers": blockers,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    official.import_tau2(args.tau_repo)
    evaluator_args = official.evaluator_kwargs(
        api_base=str(args.api_base or ""),
        reasoning_effort=args.evaluator_reasoning_effort,
    )
    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = str(args.api_base)
    if args.api_key_env != "OPENAI_API_KEY":
        os.environ["OPENAI_API_KEY"] = os.environ[args.api_key_env]
    official.apply_evaluator_patch(
        evaluator_model=args.evaluator_model,
        evaluator_args=evaluator_args,
    )

    run_dir = args.output_root / run_name
    arm_dir = run_dir / args.arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    results = run_official_baseline(args, run_dir=run_dir)
    results_payload = results.model_dump(mode="json")
    results_payload["schema"] = PRIVATE_RESULTS_SCHEMA
    results_payload["run_name"] = run_name
    results_payload["baseline_arm"] = args.arm
    results_payload["task_split"] = _task_split(args)
    results_payload["raw_task_ids_present_private"] = True
    (arm_dir / "airline_official_baseline_results.json").write_text(
        json.dumps(results_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    public_summary = build_execute_public_summary(
        args,
        run_name=run_name,
        results_payload=results_payload,
    )
    try:
        write_json_append_only(args.public_output_json, public_summary)
    except FileExistsError:
        # The spend already happened and the private results are on disk, so the
        # summary is recoverable. Say where from instead of losing the run to a
        # traceback.
        print(
            json.dumps(
                {
                    "decision": "completed_run_public_summary_write_blocked",
                    "private_run_dir": str(run_dir),
                    "output": str(args.public_output_json),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 3

    accounting = public_summary["agent_side_cache_neutral_cost_accounting"]
    print(
        json.dumps(
            {
                "run_name": run_name,
                "arm": args.arm,
                "task_count": public_summary["task_count"],
                "reward_sum": public_summary["reward_sum"],
                "reward_mean": public_summary["reward_mean"],
                "cache_neutral_cost_usd": accounting["cache_neutral_cost_usd"],
                "output": str(args.public_output_json),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_official_baseline(args: argparse.Namespace, *, run_dir: Path):
    """Run the split through tau2 with its own default single-model agent."""

    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain

    task_split = _task_split(args)
    save_dir = run_dir / "tau_runs" / DOMAIN / f"{args.arm}.r0"
    config = TextRunConfig(
        domain=DOMAIN,
        task_split_name=task_split,
        task_ids=None,
        num_tasks=split_denominator(task_split),
        llm_user=args.user_model,
        llm_args_user=llm_args_for_model(
            args.api_base,
            args.user_reasoning_effort,
            args.user_model,
        ),
        num_trials=args.num_trials,
        max_retries=args.max_retries,
        retry_delay=0.0,
        save_to=str(save_dir.resolve()),
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        log_level=args.log_level,
        agent=BASELINE_AGENT,
        llm_agent=args.agent_model,
        llm_args_agent=llm_args_for_model(
            args.api_base,
            args.agent_reasoning_effort,
            args.agent_model,
        ),
        user="user_simulator",
        max_steps=args.max_steps,
    )
    return run_domain(config)


def build_launch_report(
    args: argparse.Namespace,
    *,
    run_name: str,
) -> dict[str, Any]:
    blockers = launch_blockers(args)
    task_split = _task_split(args)
    report: dict[str, Any] = {
        "schema": LAUNCH_REPORT_SCHEMA,
        "run_name": run_name,
        "mode": "dry_run_launch_report",
        "decision": BLOCKED_DECISION if blockers else READY_DECISION,
        "blockers": blockers,
        "execution_preflight_blockers": execute_blockers(args),
        "surface": f"tau2_{DOMAIN}",
        "domain": DOMAIN,
        "domain_split": task_split,
        "task_split": task_split,
        "baseline_arm": args.arm,
        "method_family": "official_single_model_baseline",
        "cap": {
            "target_denominator": (
                split_denominator(task_split)
                if task_split in airline_run.SPLIT_OPERATING_POINTS
                else None
            ),
            "num_trials": int(args.num_trials),
            "task_id_selection_supported": False,
            "selected_task_count": 0,
        },
        "max_concurrency": int(args.max_concurrency),
        "models": {
            "agent": args.agent_model,
            "agent_role": AGENT_LEDGER_ROLE.get(args.arm),
            "agent_reasoning_effort": args.agent_reasoning_effort,
            "user_simulator": args.user_model,
            "user_reasoning_effort": args.user_reasoning_effort,
            "evaluator": args.evaluator_model,
            "evaluator_reasoning_effort": args.evaluator_reasoning_effort,
        },
        "baseline_contract": {
            "tau2_default_single_model_agent": BASELINE_AGENT,
            "reached_state_contract_absent": True,
            "residual_graph_absent": True,
            "checked_execution_absent": True,
            "task_ids_used_at_runtime": False,
            "gold_or_direct_outcome_used_at_runtime": False,
            "fixed_tool_args_used_at_runtime": False,
            "denominator_source": "method_runner_split_operating_points",
            "config_plumbing_shared_with_method_arm": True,
            "matched_fields": [
                "task_split",
                "num_tasks",
                "num_trials",
                "seed",
                "max_steps",
                "max_retries",
                "retry_delay",
                "max_concurrency",
                "log_level",
                "user_simulator_model",
                "user_reasoning_effort",
                "evaluator_model",
                "evaluator_reasoning_effort",
            ],
        },
        "cost_accounting_policy": cost_accounting_policy_report(),
        "public_paths": {
            "public_output_json": _public_path_label(args.public_output_json),
        },
        "provider_calls": 0,
        "api_spend_performed": False,
        "public_safe": True,
    }
    report["public_leakage_audit"] = public_leakage_audit(report)
    if report["public_leakage_audit"]["pass"] is not True:
        report["decision"] = BLOCKED_DECISION
        report["blockers"] = sorted(
            {*report["blockers"], "public_leakage_audit_failed"}
        )
    return report


def build_execute_public_summary(
    args: argparse.Namespace,
    *,
    run_name: str,
    results_payload: dict[str, Any],
) -> dict[str, Any]:
    task_split = _task_split(args)
    summary = official.build_official_baseline_public_summary(
        run_name=run_name,
        results_payload=results_payload,
        arm_name=args.arm,
        max_concurrency=args.max_concurrency,
        evaluator_model=args.evaluator_model,
        evaluator_reasoning_effort=args.evaluator_reasoning_effort,
    )
    summary.pop("public_leakage_audit", None)
    summary["domain"] = DOMAIN
    summary["domain_split"] = task_split
    summary["task_split"] = task_split
    summary["target_denominator"] = split_denominator(task_split)
    summary["agent_model"] = args.agent_model
    summary["agent_reasoning_effort"] = args.agent_reasoning_effort
    summary["agent_side_cache_neutral_cost_accounting"] = (
        agent_side_cache_neutral_cost_accounting(
            summary=summary,
            arm_name=args.arm,
            agent_model=args.agent_model,
        )
    )
    summary["public_leakage_audit"] = public_leakage_audit(summary)
    if summary["public_leakage_audit"]["pass"] is not True:
        raise ValueError(
            "tau2 airline official baseline public summary failed the leakage "
            "audit; the public summary was not written and must be rebuilt from "
            "the private root"
        )
    return summary


def agent_side_cache_neutral_cost_accounting(
    *,
    summary: Mapping[str, Any],
    arm_name: str,
    agent_model: str,
) -> dict[str, Any]:
    """Price the single baseline agent with no provider cache discount.

    Every logical input token is charged at the full input rate, including any
    subset the provider served from its cache. The cached rate is published
    alongside only so a reader can see it was available and not applied. A
    missing token ledger yields no cost rather than a guessed one, because a
    silently zeroed role reads as a cheaper arm.
    """

    role = AGENT_LEDGER_ROLE.get(arm_name)
    if role is None:
        raise ValueError(f"unsupported tau2 official baseline arm: {arm_name}")
    prices = public_openai_price_for_model(agent_model)
    pricing_source, pricing_snapshot_date = public_pricing_metadata_for_model(
        agent_model
    )
    row = _agent_ledger_row(summary, arm_name=arm_name, role=role)
    input_tokens = row.get("input_tokens") if row is not None else None
    output_tokens = row.get("output_tokens") if row is not None else None

    reason: str | None = None
    if row is None:
        reason = "agent_role_ledger_missing"
    elif row.get("observed") is not True:
        reason = "agent_role_not_observed"
    elif row.get("token_usage_available") is not True:
        reason = "agent_role_token_usage_unavailable"
    elif input_tokens is None or output_tokens is None:
        reason = "agent_role_token_counts_missing"
    elif prices is None:
        reason = "no_frozen_public_price_for_agent_model"

    cost: float | None = None
    if reason is None:
        cost = round(
            cache_neutral_cost_usd(
                model=agent_model,
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
            ),
            6,
        )

    accounting: dict[str, Any] = {
        "schema": COST_ACCOUNTING_SCHEMA,
        "primary": "cache_neutral_frozen_public_prices",
        "arm": arm_name,
        "agent_model": agent_model,
        "ledger_role": role,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_price_per_1m_usd": (
            prices.get("input_price_per_1m") if prices else None
        ),
        "output_price_per_1m_usd": (
            prices.get("output_price_per_1m") if prices else None
        ),
        "cached_input_price_per_1m_usd_not_applied": (
            prices.get("cached_input_price_per_1m") if prices else None
        ),
        "cache_neutral_cost_usd": cost,
        "all_input_tokens_priced_at_full_input_rate": True,
        "cache_discount_applied": False,
        "pricing_source": pricing_source,
        "pricing_snapshot_date": pricing_snapshot_date,
        "roles": ["agent"],
        "user_simulator_and_evaluator_excluded": True,
        "public_safe": True,
    }
    if reason is not None:
        accounting["cost_unavailable_reason"] = reason
    return accounting


def cost_accounting_policy_report() -> dict[str, Any]:
    return {
        "schema": "papc.tau2.airline_official_baseline_cost_policy.v1",
        "primary": "cache_neutral_frozen_public_prices",
        "priced_ledger_fields": ["input_tokens", "output_tokens"],
        "cache_neutral_formula": (
            "input_tokens * input_price_per_1m + output_tokens * "
            "output_price_per_1m, divided by 1e6"
        ),
        "cache_discount_applied": False,
        "billable_input_tokens_not_used_for_primary_cost": True,
        "missing_token_usage_is_unavailable_not_zero": True,
        "roles": ["agent"],
        "user_simulator_and_evaluator_excluded": True,
        "raw_provider_payloads_excluded": True,
        "public_safe": True,
    }


def launch_blockers(args: argparse.Namespace) -> list[str]:
    """Return provider-zero configuration blockers for this baseline arm."""

    blockers: list[str] = []
    if _task_split(args) not in airline_run.SPLIT_OPERATING_POINTS:
        blockers.append("task_split_not_registered_for_airline")
    if str(getattr(args, "arm", "") or "") not in BASELINE_ARMS:
        blockers.append("baseline_arm_must_be_cheap_full_or_strong_full")
    if not str(getattr(args, "agent_model", "") or ""):
        blockers.append("agent_model_missing")
    if not str(getattr(args, "agent_reasoning_effort", "") or ""):
        blockers.append("agent_reasoning_effort_missing")
    if not str(getattr(args, "user_model", "") or ""):
        blockers.append("user_model_missing")
    if not str(getattr(args, "evaluator_model", "") or ""):
        blockers.append("evaluator_model_missing")
    if int(args.num_trials) != 1:
        blockers.append("single_trial_required")
    if int(args.max_concurrency) < 1:
        blockers.append("max_concurrency_must_be_at_least_1")
    if int(args.max_concurrency) > MAX_ONLINE_CONCURRENCY:
        blockers.append(f"max_concurrency_exceeds_{MAX_ONLINE_CONCURRENCY}")
    return sorted(set(blockers))


def execute_blockers(args: argparse.Namespace) -> list[str]:
    blockers = set(launch_blockers(args))
    if not args.execute:
        blockers.add("execute_flag_missing")
    if not args.api_base:
        blockers.add("api_base_missing")
    if not args.api_key_env:
        blockers.add("api_key_env_missing")
    elif not os.environ.get(args.api_key_env):
        blockers.add("api_key_env_not_present")
    if Path(args.public_output_json).exists():
        # The summary writer is append-only, so an occupied path would cost the
        # run its whole summary after the spend rather than before it.
        blockers.add("public_output_json_already_exists")
    return sorted(blockers)


def _agent_ledger_row(
    summary: Mapping[str, Any],
    *,
    arm_name: str,
    role: str,
) -> dict[str, Any] | None:
    arms = summary.get("arms")
    if not isinstance(arms, Mapping):
        return None
    arm = arms.get(arm_name)
    if not isinstance(arm, Mapping):
        return None
    ledger = arm.get("runtime_ledger")
    if not isinstance(ledger, Mapping):
        return None
    per_role = ledger.get("per_role")
    if not isinstance(per_role, Mapping):
        return None
    row = per_role.get(role)
    return dict(row) if isinstance(row, Mapping) else None


def _public_path_label(path: Path) -> str:
    text = Path(path).as_posix()
    if text.startswith("outputs/papc/"):
        return text
    return Path(path).name


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
