#!/usr/bin/env python3
"""Run one official tau2 full-agent baseline through Claude subscription auth."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from papc.certspec.tau2.noapi_readiness import (  # noqa: E402
    public_leakage_audit,
    write_json_append_only,
)
from papc.paths import DATA_ROOT
from scripts.tau2 import (  # noqa: E402
    papc_tau2_official_run_with_evaluator_patch as official,
)

DEFAULT_TAU_REPO = Path(f"{DATA_ROOT}/benchmarks/tau_bench/repo")
DEFAULT_PRIVATE_ROOT = Path(f"{DATA_ROOT}/papc")
EXPECTED_TASK_COUNTS = {"retail": 114, "airline": 50}
# The split each surface is reported on. Retail is evaluated on its forty-task
# test split and Airline on its fifty-episode base, so a baseline that ran the
# whole retail base would not share a denominator with the cells it is read
# against.
SPLIT_TASK_COUNTS = {
    ("retail", "base"): 114,
    ("retail", "test"): 40,
    ("airline", "base"): 50,
    ("airline", "test"): 20,
}
EXPECTED_MAX_CONCURRENCY = 5
CLAUDE_MODELS: dict[str, dict[str, Any]] = {
    "claude-sonnet-5": {
        "reasoning_effort": "high",
        "input_price_per_1m": 2.0,
        "output_price_per_1m": 10.0,
        "cache_write_price_per_1m": 2.5,
        "cache_read_price_per_1m": 0.2,
        "pricing_note": (
            "introductory pricing through 2026-08-31; standard pricing thereafter "
            "is $3/M input and $15/M output"
        ),
    },
    "claude-opus-5": {
        "reasoning_effort": "high",
        "input_price_per_1m": 5.0,
        "output_price_per_1m": 25.0,
        "cache_write_price_per_1m": 6.25,
        "cache_read_price_per_1m": 0.5,
        "pricing_note": "current public list pricing",
    },
    # Measured through the relay on 2026-08-24: this model does take an effort
    # setting there and the setting changes what it does, none returning two
    # hundred and seventy-five completion tokens of direct working where high
    # returned six hundred and seventy-nine and a settled answer. The entry
    # read none because the subscription SDK path rejects the parameter
    # outright, which is a property of that transport rather than of the model,
    # and pinning it here refused a baseline the operator had asked for.
    "claude-haiku-4-5-20251001": {
        "reasoning_effort": "high",
        "input_price_per_1m": 1.0,
        "output_price_per_1m": 5.0,
        "cache_write_price_per_1m": 1.25,
        "cache_read_price_per_1m": 0.1,
        "pricing_note": "current public list pricing",
    },
}
ANTHROPIC_PRICING_SOURCE = "https://www.anthropic.com/pricing"
ANTHROPIC_PRICING_SNAPSHOT_DATE = "2026-07-17"
GPT55_PRICES = {"input_price_per_1m": 5.0, "output_price_per_1m": 30.0}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(CLAUDE_MODELS), required=True)
    parser.add_argument("--reasoning-effort", choices=("none", "high"), required=True)
    parser.add_argument("--domain", choices=sorted(EXPECTED_TASK_COUNTS), required=True)
    parser.add_argument("--task-split-name", choices=("base", "test"), default="base")
    parser.add_argument("--max-concurrency", type=int, default=EXPECTED_MAX_CONCURRENCY)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user-model", default="gpt-5.5")
    parser.add_argument("--user-reasoning-effort", default="high")
    parser.add_argument("--evaluator-model", default="gpt-5.5")
    parser.add_argument("--evaluator-reasoning-effort", default="high")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--tau-repo", type=Path, default=DEFAULT_TAU_REPO)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--public-output-json", type=Path, required=True)
    parser.add_argument(
        "--patch-infra-from",
        type=Path,
        default=None,
        help=(
            "Private run directory of a base run. Re-runs exactly the episodes "
            "that run lost to provider infrastructure and merges them back, so "
            "a transport hiccup costs a few episodes instead of the split."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> list[str]:
    blockers: list[str] = []
    expected_effort = str(CLAUDE_MODELS[args.model]["reasoning_effort"])
    if args.reasoning_effort != expected_effort:
        blockers.append(
            f"reasoning_effort_must_be_{expected_effort}_for_{args.model}"
        )
    if args.max_concurrency != EXPECTED_MAX_CONCURRENCY:
        blockers.append("max_concurrency_must_equal_5")
    if args.max_steps != 100:
        blockers.append("max_steps_must_equal_100")
    if args.max_retries != 1:
        blockers.append("max_retries_must_equal_1")
    if args.user_model != "gpt-5.5" or args.user_reasoning_effort != "high":
        blockers.append("user_simulator_must_equal_gpt_5_5_high")
    if (
        args.evaluator_model != "gpt-5.5"
        or args.evaluator_reasoning_effort != "high"
    ):
        blockers.append("evaluator_must_equal_gpt_5_5_high")
    if not args.api_base:
        blockers.append("api_base_missing")
    if not args.api_key_env or not os.environ.get(args.api_key_env):
        blockers.append("api_key_env_not_present")
    if not (args.tau_repo / "src").is_dir():
        blockers.append("tau2_source_missing")
    if (args.domain, args.task_split_name) not in SPLIT_TASK_COUNTS:
        blockers.append("unknown_domain_split_pair")
    if getattr(args, "patch_infra_from", None) is not None:
        base_payload = base_results_payload(args.patch_infra_from)
        if base_payload is None:
            blockers.append("patch_base_results_missing")
        elif not infrastructure_task_ids(base_payload):
            blockers.append("patch_base_has_no_infrastructure_rows")
    if not args.run_id.strip():
        blockers.append("run_id_missing")
    if args.public_output_json.exists():
        blockers.append("public_output_already_exists")
    private_run_dir = private_run_directory(args)
    if private_run_dir.exists():
        blockers.append("private_run_directory_already_exists")
    return sorted(set(blockers))


INFRASTRUCTURE_PATCH_SCHEMA = "papc.tau2.claude_baseline_infrastructure_patch.v1"


def base_results_payload(base_run_dir: Path) -> dict[str, Any] | None:
    """The tau2 results a base run left behind, if they are there."""

    path = base_run_dir / "results.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def infrastructure_task_ids(payload: Mapping[str, Any]) -> list[str]:
    """Episodes the base run never scored because the transport failed.

    Only these may be re-run. An episode that produced a reward was measured,
    and re-running it because the reward was low would select on outcome.
    """

    return sorted(
        {
            str(simulation.get("task_id"))
            for simulation in _list(payload.get("simulations"))
            if isinstance(simulation, Mapping)
            and str(simulation.get("termination_reason")) == "infrastructure_error"
        }
    )


def merge_patch_results(
    *,
    base_payload: Mapping[str, Any],
    patch_payload: Mapping[str, Any],
    patched_task_ids: Sequence[str],
) -> dict[str, Any]:
    """Return the base run with its infrastructure episodes substituted."""

    patched = set(patched_task_ids)
    replacement = {
        str(simulation.get("task_id")): simulation
        for simulation in _list(patch_payload.get("simulations"))
        if isinstance(simulation, Mapping)
    }
    merged = [
        replacement.get(str(simulation.get("task_id")), simulation)
        if str(simulation.get("task_id")) in patched
        else simulation
        for simulation in _list(base_payload.get("simulations"))
    ]
    payload = dict(patch_payload)
    payload["simulations"] = merged
    return payload


def private_run_directory(args: argparse.Namespace) -> Path:
    return (
        args.private_root
        / args.run_id
        / "tau2"
        / args.domain
        / "cheap_full.r0"
    )


def configure_runtime(args: argparse.Namespace) -> tuple[Any, dict[str, Any], int]:
    _sanitize_proxy_environment()
    tau_src = args.tau_repo / "src"
    if str(tau_src) not in sys.path:
        sys.path.insert(0, str(tau_src))
    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = str(args.api_base)
    if args.api_key_env != "OPENAI_API_KEY":
        os.environ["OPENAI_API_KEY"] = os.environ[args.api_key_env]

    from papc.certspec.claude_subscription import (
        ClaudeSubscriptionClient,
        ClaudeSubscriptionConfig,
        subscription_preflight,
    )
    from papc.certspec.tau2.claude_subscription_generate import (
        install_tau2_claude_generate_bridge,
    )

    config = ClaudeSubscriptionConfig.from_env().for_model(args.model)
    preflight = subscription_preflight(config)
    official.import_tau2(args.tau_repo)
    official.apply_evaluator_patch(
        evaluator_model=args.evaluator_model,
        evaluator_args=official.evaluator_kwargs(
            api_base=args.api_base,
            reasoning_effort=args.evaluator_reasoning_effort,
        ),
    )
    install_tau2_claude_generate_bridge(
        claude_model=args.model,
        claude_client=ClaudeSubscriptionClient(config),
    )

    from tau2.registry import registry
    from tau2.runner.build import build_agent

    tasks = registry.get_tasks_loader(args.domain)(args.task_split_name)
    expected_count = SPLIT_TASK_COUNTS[(args.domain, args.task_split_name)]
    if len(tasks) != expected_count:
        raise RuntimeError(
            f"tau2 {args.domain}/{args.task_split_name} task count drift: "
            f"{len(tasks)} != {expected_count}"
        )
    environment = registry.get_env_constructor(args.domain)()
    build_agent(
        "llm_agent",
        environment,
        llm=args.model,
        llm_args=agent_llm_args(args),
        task=tasks[0],
    )
    return config, preflight, len(tasks)


def agent_llm_args(args: argparse.Namespace) -> dict[str, Any]:
    # The endpoint travels with the agent model as it already does with the
    # user simulator below. Only the simulator carried it, so the agent was
    # left to whatever the client routes a bare Claude name to, which is the
    # vendor's own endpoint and a credential this project keeps unset.
    return {
        "temperature": 0.0,
        "drop_params": True,
        "reasoning_effort": args.reasoning_effort,
        "api_base": args.api_base,
    }


def agent_llm_name(args: argparse.Namespace) -> str:
    """The agent model as the OpenAI-compatible client should be asked for it.

    A bare Claude name selects the vendor's own route inside the client, which
    then wants a vendor credential this project deliberately leaves unset. The
    prefix selects the OpenAI-compatible route instead and is removed before
    the request leaves, so the endpoint still receives the plain model id.
    """

    model = str(args.model)
    return model if "/" in model else f"openai/{model}"


def user_llm_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "drop_params": True,
        "reasoning_effort": args.user_reasoning_effort,
        "api_base": args.api_base,
    }


def tau2_cli_args(
    args: argparse.Namespace, *, task_ids: Sequence[str] | None = None
) -> list[str]:
    selection = ["--task-ids", *task_ids] if task_ids else []
    return [
        "run",
        "--domain",
        args.domain,
        "--task-split-name",
        args.task_split_name,
        *selection,
        "--num-trials",
        "1",
        "--agent",
        "llm_agent",
        "--agent-llm",
        agent_llm_name(args),
        "--agent-llm-args",
        json.dumps(agent_llm_args(args), separators=(",", ":")),
        "--user",
        "user_simulator",
        "--user-llm",
        args.user_model,
        "--user-llm-args",
        json.dumps(user_llm_args(args), separators=(",", ":")),
        "--max-steps",
        str(args.max_steps),
        "--max-errors",
        "10",
        "--save-to",
        str(private_run_directory(args)),
        "--max-concurrency",
        str(args.max_concurrency),
        "--seed",
        str(args.seed),
        "--max-retries",
        str(args.max_retries),
        "--retry-delay",
        "1",
        "--log-level",
        "INFO",
    ]


def build_gate_report(
    args: argparse.Namespace,
    *,
    blockers: Sequence[str],
    subscription_preflight: Mapping[str, Any] | None,
    task_count: int | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "papc.tau2.claude_official_baseline_gate.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "decision": "ready" if not blockers else "blocked",
        "surface": "tau2",
        "domain": args.domain,
        "split": args.task_split_name,
        "task_count": task_count,
        "expected_task_count": SPLIT_TASK_COUNTS[
            (args.domain, args.task_split_name)
        ],
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "user_simulator": "gpt-5.5/high",
        "evaluator": "gpt-5.5/high",
        "max_concurrency": args.max_concurrency,
        "max_steps": args.max_steps,
        "num_trials": 1,
        "execute_requested": bool(args.execute),
        "provider_calls": 0,
        "api_spend_performed": False,
        "subscription_auth": {
            "ready": bool(subscription_preflight),
            "auth_method": (subscription_preflight or {}).get("auth_method"),
            "api_provider": (subscription_preflight or {}).get("api_provider"),
            "subscription_type": (subscription_preflight or {}).get(
                "subscription_type"
            ),
        },
        "blockers": sorted(set(blockers)),
        "private_run_label": f"{args.run_id}/tau2/{args.domain}/cheap_full.r0",
        "public_safe": True,
    }
    report["public_leakage_audit"] = public_leakage_audit(report)
    return report


def execute_tau2(
    args: argparse.Namespace, *, task_ids: Sequence[str] | None = None
) -> Path:
    import tau2.cli as tau_cli

    sys.argv = ["tau2", *tau2_cli_args(args, task_ids=task_ids)]
    tau_cli.main()
    results_path = private_run_directory(args) / "results.json"
    if not results_path.is_file():
        raise RuntimeError(f"tau2 results missing after execution: {results_path}")
    return results_path


def build_public_summary(
    args: argparse.Namespace, *, results_payload: dict[str, Any]
) -> dict[str, Any]:
    expected_count = SPLIT_TASK_COUNTS[(args.domain, args.task_split_name)]
    simulations = [
        row for row in results_payload.get("simulations", []) if isinstance(row, dict)
    ]
    if len(simulations) != expected_count:
        raise RuntimeError(
            f"incomplete tau2 result: {len(simulations)} != {expected_count}"
        )
    report = official.build_official_baseline_public_summary(
        run_name=args.run_id,
        results_payload=results_payload,
        arm_name="cheap_full",
        max_concurrency=args.max_concurrency,
        evaluator_model=args.evaluator_model,
        evaluator_reasoning_effort=args.evaluator_reasoning_effort,
    )
    report["schema"] = "papc.tau2.claude_official_baseline_public.v1"
    report["source_execution"] = {
        "provider_bearing_execution_performed": True,
        "claude_auth": "first_party_subscription",
        "private_run_label": f"{args.run_id}/tau2/{args.domain}/cheap_full.r0",
        "complete_task_count_required": expected_count,
        "complete_task_count_observed": len(simulations),
    }
    report["cost_accounting"] = build_cost_accounting(
        model=args.model,
        results_payload=results_payload,
        runtime_ledger=report["arms"]["cheap_full"]["runtime_ledger"],
    )
    report["provider_calls"] = report["cost_accounting"]["observed_provider_calls"]
    report["api_spend_performed"] = True
    report["claim_boundary"].extend(
        [
            "Claude agent calls used a fixed subscription; API-equivalent token cost is not a per-call subscription invoice",
            "cache-neutral cost prices all logical input tokens at the ordinary input rate",
        ]
    )
    report["public_leakage_audit"] = public_leakage_audit(report)
    return report


def build_cost_accounting(
    *,
    model: str,
    results_payload: Mapping[str, Any],
    runtime_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    price = CLAUDE_MODELS[model]
    claude = aggregate_claude_usage(results_payload, model=model)
    logical_input_tokens = (
        claude["uncached_input_tokens"]
        + claude["cache_creation_input_tokens"]
        + claude["cache_read_input_tokens"]
    )
    cache_neutral_input_cost = _token_cost(
        logical_input_tokens, price["input_price_per_1m"]
    )
    output_cost = _token_cost(claude["output_tokens"], price["output_price_per_1m"])
    cache_aware_input_cost = sum(
        (
            _token_cost(
                claude["uncached_input_tokens"], price["input_price_per_1m"]
            ),
            _token_cost(
                claude["cache_creation_input_tokens"],
                price["cache_write_price_per_1m"],
            ),
            _token_cost(
                claude["cache_read_input_tokens"],
                price["cache_read_price_per_1m"],
            ),
        )
    )

    per_role = _mapping(_mapping(runtime_ledger).get("per_role"))
    user = _mapping(per_role.get("user_simulator"))
    user_input_tokens = _optional_int(user.get("input_tokens"))
    user_output_tokens = _optional_int(user.get("output_tokens"))
    user_input_cost = (
        _token_cost(user_input_tokens, GPT55_PRICES["input_price_per_1m"])
        if user_input_tokens is not None
        else None
    )
    user_output_cost = (
        _token_cost(user_output_tokens, GPT55_PRICES["output_price_per_1m"])
        if user_output_tokens is not None
        else None
    )
    observed_provider_calls = claude["calls"] + int(user.get("calls") or 0)
    return {
        "schema": "papc.tau2.claude_baseline_cost_accounting.v1",
        "pricing_snapshot": {
            "date": ANTHROPIC_PRICING_SNAPSHOT_DATE,
            "source": ANTHROPIC_PRICING_SOURCE,
            "model": model,
            "input_price_per_1m": price["input_price_per_1m"],
            "output_price_per_1m": price["output_price_per_1m"],
            "cache_write_price_per_1m": price["cache_write_price_per_1m"],
            "cache_read_price_per_1m": price["cache_read_price_per_1m"],
            "note": price["pricing_note"],
        },
        "agent": {
            **claude,
            "logical_input_tokens": logical_input_tokens,
            "cache_neutral": {
                "input_cost_usd": cache_neutral_input_cost,
                "output_cost_usd": output_cost,
                "total_cost_usd": round(cache_neutral_input_cost + output_cost, 6),
            },
            "cache_aware_api_equivalent": {
                "input_cost_usd": round(cache_aware_input_cost, 6),
                "output_cost_usd": output_cost,
                "total_cost_usd": round(cache_aware_input_cost + output_cost, 6),
            },
            "subscription_billing": {
                "per_call_invoice_observed": False,
                "fixed_plan_marginal_spend_not_inferred_from_sdk_estimate": True,
            },
        },
        "user_simulator": {
            "model": "gpt-5.5",
            "input_tokens": user_input_tokens,
            "output_tokens": user_output_tokens,
            "input_cost_usd": user_input_cost,
            "output_cost_usd": user_output_cost,
            "cache_neutral_total_cost_usd": (
                round(user_input_cost + user_output_cost, 6)
                if user_input_cost is not None and user_output_cost is not None
                else None
            ),
            "excluded_from_agent_side_baseline_cost": True,
        },
        "evaluator": {
            "model": "gpt-5.5",
            "input_tokens": None,
            "output_tokens": None,
            "input_cost_usd": None,
            "output_cost_usd": None,
            "unavailable_reason": "official_tau2_results_do_not_record_evaluator_token_usage",
            "excluded_from_agent_side_baseline_cost": True,
        },
        "observed_provider_calls": observed_provider_calls,
        "provider_calls_are_minimum_without_evaluator_calls": True,
        "primary_agent_side_cost_field": "agent.cache_neutral.total_cost_usd",
        "public_safe": True,
    }


def aggregate_claude_usage(
    results_payload: Mapping[str, Any], *, model: str
) -> dict[str, Any]:
    totals = {
        "calls": 0,
        "uncached_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 0,
        "sdk_estimated_total_cost_usd": 0.0,
        "sdk_estimated_cost_call_count": 0,
    }
    observed_models: set[str] = set()
    for simulation in _list(results_payload.get("simulations")):
        if not isinstance(simulation, Mapping):
            continue
        for message in _list(simulation.get("messages")):
            if not isinstance(message, Mapping):
                continue
            raw = _mapping(message.get("raw_data"))
            if raw.get("papc_provider") != "anthropic_subscription_agent_sdk":
                continue
            observed_model = str(raw.get("papc_model") or "")
            if observed_model:
                observed_models.add(observed_model)
            usage = _mapping(message.get("usage"))
            cache = _mapping(raw.get("papc_cache_usage_diagnostic"))
            totals["calls"] += 1
            totals["uncached_input_tokens"] += int(usage.get("prompt_tokens") or 0)
            totals["output_tokens"] += int(usage.get("completion_tokens") or 0)
            totals["cache_creation_input_tokens"] += int(
                cache.get("cache_creation_input_tokens") or 0
            )
            totals["cache_read_input_tokens"] += int(
                cache.get("cache_read_input_tokens") or 0
            )
            cost = _optional_float(message.get("cost"))
            if cost is not None:
                totals["sdk_estimated_total_cost_usd"] += cost
                totals["sdk_estimated_cost_call_count"] += 1
    if observed_models and observed_models != {model}:
        raise RuntimeError(
            f"Claude model identity drift: expected {model}, observed {sorted(observed_models)}"
        )
    totals["sdk_estimated_total_cost_usd"] = round(
        float(totals["sdk_estimated_total_cost_usd"]), 6
    )
    totals["observed_models"] = sorted(observed_models)
    totals["all_calls_have_sdk_estimated_cost"] = (
        totals["calls"] > 0
        and totals["sdk_estimated_cost_call_count"] == totals["calls"]
    )
    return totals


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(_REPO / ".env", override=False)
    args = parse_args(argv)
    blockers = validate_args(args)
    preflight: dict[str, Any] | None = None
    task_count: int | None = None
    if not blockers:
        try:
            _, preflight, task_count = configure_runtime(args)
        except Exception as exc:
            blockers.append(f"runtime_preflight_failed:{type(exc).__name__}:{exc}")

    if not args.execute or blockers:
        report = build_gate_report(
            args,
            blockers=blockers,
            subscription_preflight=preflight,
            task_count=task_count,
        )
        write_json_append_only(args.public_output_json, report)
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "provider_calls": 0,
                    "domain": args.domain,
                    "model": args.model,
                    "output": str(args.public_output_json),
                    "blockers": report["blockers"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if not blockers else 2

    base_payload = (
        base_results_payload(args.patch_infra_from)
        if args.patch_infra_from is not None
        else None
    )
    patched_task_ids = (
        infrastructure_task_ids(base_payload) if base_payload is not None else []
    )
    results_path = execute_tau2(args, task_ids=patched_task_ids or None)
    results_payload = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(results_payload, dict):
        raise RuntimeError("tau2 results payload must be an object")
    patch_provenance: dict[str, Any] | None = None
    if base_payload is not None:
        patch_provenance = {
            "schema": INFRASTRUCTURE_PATCH_SCHEMA,
            "base_private_run_dir": str(args.patch_infra_from),
            "patched_task_count": len(patched_task_ids),
            "patch_reason": "infrastructure_error_only",
        }
        results_payload = merge_patch_results(
            base_payload=base_payload,
            patch_payload=results_payload,
            patched_task_ids=patched_task_ids,
        )
        patch_provenance["episodes_still_infrastructure_after_patch"] = len(
            infrastructure_task_ids(results_payload)
        )
    report = build_public_summary(args, results_payload=results_payload)
    if patch_provenance is not None:
        report["patch_provenance"] = patch_provenance
    write_json_append_only(args.public_output_json, report)
    print(
        json.dumps(
            {
                "decision": "completed",
                "domain": args.domain,
                "model": args.model,
                "task_count": report["task_count"],
                "reward_sum": report["reward_sum"],
                "agent_cost": report["cost_accounting"]["agent"]["cache_neutral"],
                "output": str(args.public_output_json),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _sanitize_proxy_environment() -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(name, None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    os.environ["no_proxy"] = "localhost,127.0.0.1"


def _token_cost(tokens: int, price_per_1m: float) -> float:
    return round(max(0, int(tokens)) * float(price_per_1m) / 1_000_000.0, 6)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
