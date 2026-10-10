#!/usr/bin/env python3
"""Run released RouteLLM-BERT on fresh Tau2 Airline episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.external_router_execution import (  # noqa: E402
    ProviderAttemptBroker,
    ProviderAttemptLedger,
)
from papc.baselines.external_router_fit_dev_native import (  # noqa: E402
    activate_tau2_dependency_environment,
)
from papc.baselines.routellm_bert_upstream import (  # noqa: E402
    ROUTELLM_BERT_BASE_BUNDLE_SHA256,
    ROUTELLM_BERT_BASE_REPO,
    ROUTELLM_BERT_BASE_REVISION,
)
from papc.baselines.router_accounting import (  # noqa: E402
    FROZEN_CACHE_NEUTRAL_PRICES,
    CallReceipt,
    aggregate_online_accounting,
)
from papc.baselines.router_contract import (  # noqa: E402
    CHEAP_MODEL,
    REASONING_EFFORT,
    ROUTELLM,
    STRONG_MODEL,
    UPSTREAM_REVISIONS,
    RequestRouterView,
    RouterArtifactBinding,
)
from papc.baselines.router_policies import decide_routellm  # noqa: E402
from papc.baselines.router_runtime_environment import (  # noqa: E402
    FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256,
    validate_router_runtime_environment,
)
from papc.baselines.sealed_router_policy import (  # noqa: E402
    load_released_routellm_bert_sealed_policy,
)
from papc.baselines.tau2_native_runtime import (  # noqa: E402
    TAU2_NATIVE_USER,
    TAU2_OFFICIAL_EVALUATION,
    Tau2ProviderBudget,
    _instrument_provider_attempts,
    _official_evaluator_config,
    _write_private_simulation,
    load_tau2_native_api,
    tau2_public_history,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.routellm.airline.offtheshelf_live.v1"
RESULT_SCHEMA = "papc.routellm.airline.offtheshelf_result.v1"
PRIVATE_SCHEMA = "papc.routellm.airline.offtheshelf_private.v1"
JOB_ID = "routellm_released_airline_fresh_full50"
EXPECTED_DENOMINATOR = 50
EXPECTED_MAX_CONCURRENCY = 2
EXPECTED_THRESHOLD = 0.21907290816307068
EXPECTED_STATE_POLICY_SHA256 = (
    "aeb71e2d1b78ba726ad9c8086f586680b8a3dacd7c148147ca01dd639d68ac90"
)
EXPECTED_BASE50_TASK_PLAN_SHA256 = (
    "109e16af0edede18d47d21d85f660e9af88add1cef54d86f9bf067b7b46e47a9"
)
EXPECTED_AIRLINE_SPLIT_FILE_SHA256 = (
    "b22ced4d9a9850ac9aea31c53bdcb6d6009058140bd9acc7db37c1d36222ba8b"
)
EXPECTED_AIRLINE_TASKS_FILE_SHA256 = (
    "ccd8ba737b4cc371415af70151187788f728d6108d0916e73bb4317b40542052"
)
DEFAULT_STATE_POLICY = (
    REPO_ROOT
    / "outputs/papc/routellm_state_offtheshelf_policy_noapi_20260802_v3.json"
)
DEFAULT_BASE_MODEL = Path(
    f"{DATA_ROOT}/papc/upstream_models/"
    "routellm-bert-gpt4-augmented-f625a9c"
)
DEFAULT_TAU2_DATA_REPO = Path(f"{DATA_ROOT}/benchmarks/tau2-bench-revised/repo")
DEFAULT_TAU2_RUNTIME_REPO = Path(f"{DATA_ROOT}/benchmarks/tau_bench/repo")


@dataclass(frozen=True)
class Prepared:
    native_api: Any
    native_tasks: tuple[Any, ...]
    example_sha256s: tuple[str, ...]
    test_registry_sha256: str
    source_binding_sha256: str
    policy: Any
    binding: RouterArtifactBinding
    runtime_environment_sha256: str
    tau2_dependency_environment_sha256: str
    state_policy_sha256: str
    state_policy: dict[str, Any]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"json_object_required:{path}")
    return payload


def _write_new(path: Path, payload: Mapping[str, Any], *, private: bool) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(target, flags, 0o600 if private else 0o644), "wb") as handle:
        handle.write(_canonical_bytes(dict(payload)) + b"\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-policy-json", type=Path, default=DEFAULT_STATE_POLICY)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--tau2-repo", type=Path, default=DEFAULT_TAU2_DATA_REPO)
    parser.add_argument(
        "--tau2-runtime-repo", type=Path, default=DEFAULT_TAU2_RUNTIME_REPO
    )
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--public-result-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--max-concurrency", type=int, default=EXPECTED_MAX_CONCURRENCY)
    parser.add_argument("--provider-attempt-hard-cap", type=int, default=3000)
    parser.add_argument("--observed-spend-stop-usd", type=Decimal, default=Decimal("80"))
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _prepare(args: argparse.Namespace) -> Prepared:
    if args.max_concurrency != EXPECTED_MAX_CONCURRENCY:
        raise ValueError("airline_max_concurrency_must_equal_two")
    if args.provider_attempt_hard_cap <= 0 or args.observed_spend_stop_usd <= 0:
        raise ValueError("airline_online_budget_invalid")
    if not args.api_base.startswith(("http://", "https://")):
        raise ValueError("airline_api_base_invalid")
    if args.api_key_env != "OPENAI_API_KEY":
        raise ValueError("api_key_env_must_be_OPENAI_API_KEY")
    runtime_sha = validate_router_runtime_environment()
    if runtime_sha != FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256:
        raise ValueError("router_runtime_environment_sha256_mismatch")
    dependency_sha = activate_tau2_dependency_environment(args.tau2_runtime_repo)
    state_policy_sha = _file_sha256(args.state_policy_json)
    if state_policy_sha != EXPECTED_STATE_POLICY_SHA256:
        raise ValueError("state_ood_threshold_policy_file_mismatch")
    state_policy = _read_json(args.state_policy_json)
    routing = state_policy.get("routing")
    if (
        state_policy.get("surface") != "state_bench"
        or state_policy.get("endpoint_outcomes_read") is not False
        or state_policy.get("test_fit_or_threshold_selection") is not False
        or not isinstance(routing, Mapping)
        or routing.get("strong_threshold") != EXPECTED_THRESHOLD
        or routing.get("strong_budget_fraction") != 0.9
    ):
        raise ValueError("state_ood_threshold_policy_semantics_mismatch")
    native_api = load_tau2_native_api(args.tau2_runtime_repo)
    split_path = (
        native_api.runtime_repo
        / "data/tau2/domains/airline/split_tasks.json"
    )
    tasks_path = native_api.runtime_repo / "data/tau2/domains/airline/tasks.json"
    data_split_path = args.tau2_repo / "data/tau2/domains/airline/split_tasks.json"
    data_tasks_path = args.tau2_repo / "data/tau2/domains/airline/tasks.json"
    if (
        _file_sha256(split_path) != EXPECTED_AIRLINE_SPLIT_FILE_SHA256
        or _file_sha256(tasks_path) != EXPECTED_AIRLINE_TASKS_FILE_SHA256
        or _file_sha256(data_split_path) != EXPECTED_AIRLINE_SPLIT_FILE_SHA256
        or _file_sha256(data_tasks_path) != EXPECTED_AIRLINE_TASKS_FILE_SHA256
    ):
        raise ValueError("airline_base50_source_file_mismatch")
    native_tasks = tuple(
        native_api.get_tasks(
            task_set_name="airline",
            task_split_name="base",
            task_ids=None,
            num_tasks=EXPECTED_DENOMINATOR,
        )
    )
    if len(native_tasks) != EXPECTED_DENOMINATOR:
        raise ValueError("airline_base50_denominator_mismatch")
    task_records = [task.model_dump(mode="json") for task in native_tasks]
    task_plan_sha256 = _digest(task_records)
    if task_plan_sha256 != EXPECTED_BASE50_TASK_PLAN_SHA256:
        raise ValueError("airline_base50_task_plan_mismatch")
    task_ids = tuple(str(task.id) for task in native_tasks)
    if task_ids != tuple(str(index) for index in range(EXPECTED_DENOMINATOR)):
        raise ValueError("airline_base50_task_order_mismatch")
    example_sha256s = tuple(
        _digest(
            {
                "surface": "tau2_airline",
                "split": "base",
                "ordinal": index,
                "task_payload_sha256": _digest(record),
            }
        )
        for index, record in enumerate(task_records)
    )
    source_binding_sha256 = _digest(
        {
            "runtime_revision": native_api.runtime_revision,
            "split_file_sha256": EXPECTED_AIRLINE_SPLIT_FILE_SHA256,
            "tasks_file_sha256": EXPECTED_AIRLINE_TASKS_FILE_SHA256,
            "task_plan_sha256": task_plan_sha256,
        }
    )
    threshold_binding = {
        "source_surface": "state_bench",
        "source_policy_file_sha256": state_policy_sha,
        "source_policy_canonical_sha256": _digest(state_policy),
        "threshold": EXPECTED_THRESHOLD,
        "airline_fit_rows": 0,
        "airline_dev_rows": 0,
        "airline_threshold_selection": False,
    }
    artifact_record = {
        "method": ROUTELLM,
        "variant": "released_bert_cross_domain_threshold",
        "surface": "tau2_airline",
        "base_repo": ROUTELLM_BERT_BASE_REPO,
        "base_revision": ROUTELLM_BERT_BASE_REVISION,
        "base_bundle_sha256": ROUTELLM_BERT_BASE_BUNDLE_SHA256,
        "test_registry_sha256": task_plan_sha256,
        "threshold_binding": threshold_binding,
    }
    no_fit_sha256 = _digest(
        {
            "role": "no_airline_fit",
            "rows": 0,
            "threshold_source": "state_v3_cross_domain_policy",
        }
    )
    no_dev_sha256 = _digest(
        {
            "role": "no_airline_dev",
            "rows": 0,
            "threshold_source": "state_v3_cross_domain_policy",
        }
    )
    binding = RouterArtifactBinding(
        method=ROUTELLM,
        surface="tau2_airline",
        upstream_revision=UPSTREAM_REVISIONS[ROUTELLM],
        artifact_sha256=_digest(artifact_record),
        fit_registry_sha256=no_fit_sha256,
        dev_registry_sha256=no_dev_sha256,
        test_registry_sha256=task_plan_sha256,
        calibration_sha256=_digest(threshold_binding),
    )
    policy = load_released_routellm_bert_sealed_policy(
        binding=binding,
        base_model_dir=args.base_model_dir,
        threshold=EXPECTED_THRESHOLD,
    )
    return Prepared(
        native_api=native_api,
        native_tasks=native_tasks,
        example_sha256s=example_sha256s,
        test_registry_sha256=task_plan_sha256,
        source_binding_sha256=source_binding_sha256,
        policy=policy,
        binding=binding,
        runtime_environment_sha256=runtime_sha,
        tau2_dependency_environment_sha256=dependency_sha,
        state_policy_sha256=state_policy_sha,
        state_policy=state_policy,
    )


def _base_report(args: argparse.Namespace, prepared: Prepared) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "scope": "fresh_tau2_airline_request_routed_full50",
        "method": "RouteLLM-BERT (released, STATE-frozen OOD threshold)",
        "surface": "tau2_airline",
        "task_split": "base",
        "evaluation_role": "base50_all_final_test",
        "official_test_count": EXPECTED_DENOMINATOR,
        "cheap_model": CHEAP_MODEL,
        "strong_model": STRONG_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "router_input": "actual_episode_first_public_user_message_only",
        "threshold": EXPECTED_THRESHOLD,
        "threshold_source": {
            "surface": "state_bench",
            "path_label": args.state_policy_json.name,
            "file_sha256": prepared.state_policy_sha256,
            "airline_fit_rows": 0,
            "airline_dev_rows": 0,
            "airline_threshold_selection": False,
        },
        "airline_split_sentinels": {
            "fit_registry_sha256": prepared.binding.fit_registry_sha256,
            "dev_registry_sha256": prepared.binding.dev_registry_sha256,
            "fit_rows": 0,
            "dev_rows": 0,
        },
        "router": {
            "base_repo": ROUTELLM_BERT_BASE_REPO,
            "base_revision": ROUTELLM_BERT_BASE_REVISION,
            "base_bundle_sha256": ROUTELLM_BERT_BASE_BUNDLE_SHA256,
            "artifact_binding_sha256": prepared.binding.artifact_sha256,
        },
        "test_registry_sha256": prepared.test_registry_sha256,
        "source_binding_sha256": prepared.source_binding_sha256,
        "base50_source": {
            "task_plan_sha256": EXPECTED_BASE50_TASK_PLAN_SHA256,
            "split_file_sha256": EXPECTED_AIRLINE_SPLIT_FILE_SHA256,
            "tasks_file_sha256": EXPECTED_AIRLINE_TASKS_FILE_SHA256,
        },
        "runtime_environment_sha256": prepared.runtime_environment_sha256,
        "tau2_dependency_environment_sha256": (
            prepared.tau2_dependency_environment_sha256
        ),
        "runtime_revision": prepared.native_api.runtime_revision,
        "max_concurrency": args.max_concurrency,
        "max_infra_retries": 0,
        "provider_attempt_hard_cap": args.provider_attempt_hard_cap,
        "observed_spend_stop_usd": float(args.observed_spend_stop_usd),
        "full_denominator_required": EXPECTED_DENOMINATOR,
        "partial_resume_or_cross_run_merge_allowed": False,
        "cap_or_score_correction": False,
        "row_splicing": False,
    }


def _validate_gate(args: argparse.Namespace, base: Mapping[str, Any]) -> str:
    if args.gate_json is None:
        raise ValueError("explicit_provider_zero_gate_missing")
    gate = _read_json(args.gate_json)
    if (
        gate.get("schema") != SCHEMA
        or gate.get("decision") != "ready_for_explicit_execution"
        or gate.get("provider_calls") != 0
        or gate.get("secret_reads") != 0
        or gate.get("api_spend_performed") is not False
    ):
        raise ValueError("provider_zero_gate_not_ready")
    for key in (
        "test_registry_sha256",
        "source_binding_sha256",
        "runtime_environment_sha256",
        "tau2_dependency_environment_sha256",
        "threshold",
        "max_concurrency",
        "provider_attempt_hard_cap",
        "observed_spend_stop_usd",
    ):
        if gate.get(key) != base.get(key):
            raise ValueError(f"provider_zero_gate_{key}_mismatch")
    return _file_sha256(args.gate_json)


def _first_user_text(simulation: Any) -> str:
    for item in tau2_public_history(tuple(getattr(simulation, "messages", ()) or ())):
        if item.get("role") == "user" and isinstance(item.get("content"), str):
            text = item["content"].strip()
            if text:
                return text
    raise RuntimeError("airline_actual_opening_request_missing")


def _official_score(simulation: Any) -> float:
    reward = getattr(getattr(simulation, "reward_info", None), "reward", None)
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TypeError("airline_official_reward_missing")
    value = float(reward)
    if value not in {0.0, 1.0}:
        raise ValueError("airline_accuracy_reward_not_binary")
    return value


def _execute(args: argparse.Namespace, prepared: Prepared, base: dict[str, Any]) -> dict[str, Any]:
    gate_sha = _validate_gate(args, base)
    if not os.environ.get(args.api_key_env, "").strip():
        raise ValueError("api_key_environment_variable_missing_or_empty")
    no_proxy = {
        item.strip() for item in os.environ.get("NO_PROXY", "").split(",") if item.strip()
    }
    if not {"localhost", "127.0.0.1"}.issubset(no_proxy):
        raise ValueError("no_proxy_localhost_binding_missing")
    if any(
        os.environ.get(name)
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    ):
        raise ValueError("http_proxy_must_be_unset")
    private_root = args.private_root.resolve()
    if private_root.exists() or args.public_result_json.exists():
        raise FileExistsError("airline_live_output_exists_no_overwrite")
    private_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    simulations_root = private_root / "simulations"
    simulations_root.mkdir(mode=0o700)
    policy_path = private_root / "request_route_policy.jsonl"
    policy_handle = os.fdopen(
        os.open(policy_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
        "w",
        encoding="utf-8",
    )
    policy_header = {
        "schema": "papc.routellm.airline.request_route_policy.v1",
        "decision": "threshold_frozen_before_airline_provider_calls",
        "provider_zero_gate_sha256": gate_sha,
        "threshold": EXPECTED_THRESHOLD,
        "threshold_source_file_sha256": prepared.state_policy_sha256,
        "airline_fit_rows": 0,
        "airline_dev_rows": 0,
        "airline_threshold_selection": False,
        "router_input": "actual_episode_first_public_user_message_only",
        "endpoint_outcomes_read_before_route": False,
    }
    policy_handle.write(json.dumps(policy_header, sort_keys=True) + "\n")
    policy_handle.flush()
    os.fsync(policy_handle.fileno())

    ledger = ProviderAttemptLedger(
        global_attempt_cap=args.provider_attempt_hard_cap,
        global_observed_usd_stop=args.observed_spend_stop_usd,
        stage_attempt_caps={"official_test": args.provider_attempt_hard_cap},
        stage_observed_usd_stops={"official_test": args.observed_spend_stop_usd},
        job_attempt_caps={JOB_ID: ("official_test", args.provider_attempt_hard_cap)},
    )
    broker = ProviderAttemptBroker(ledger=ledger, job_id=JOB_ID)
    budget = Tau2ProviderBudget(
        max_calls=args.provider_attempt_hard_cap,
        max_observed_cost_usd=float(args.observed_spend_stop_usd),
    )
    receipts: list[CallReceipt] = []
    decision_records: list[dict[str, Any]] = []
    decision_lock = threading.Lock()
    api = prepared.native_api
    policy = prepared.policy
    binding = prepared.binding

    class FreshRequestRoutedAgent(api.llm_agent_class):
        def __init__(self, *agent_args: Any, **agent_kwargs: Any) -> None:
            super().__init__(*agent_args, **agent_kwargs)
            self._papc_decision: Any = None
            self._papc_seed: int | None = None

        def set_seed(self, seed: int) -> None:
            self._papc_seed = seed
            super().set_seed(seed)

        def generate_next_message(self, message: Any, state: Any) -> tuple[Any, Any]:
            if self._papc_decision is None:
                history = tau2_public_history((*tuple(state.messages), message))
                opening = next(
                    (
                        str(item["content"]).strip()
                        for item in history
                        if item.get("role") == "user"
                        and isinstance(item.get("content"), str)
                        and str(item["content"]).strip()
                    ),
                    None,
                )
                if opening is None:
                    raise RuntimeError("airline_router_opening_request_missing")
                view = RequestRouterView(opening)
                with decision_lock:
                    prediction = policy.predict_routellm(view)
                    decision = decide_routellm(
                        binding=binding,
                        view=view,
                        predicted_strong_win_rate=prediction.value,
                        threshold=policy.routellm_threshold,
                    )
                    record = {
                        "decision_index": len(decision_records),
                        "request": opening,
                        "request_sha256": hashlib.sha256(
                            opening.encode("utf-8")
                        ).hexdigest(),
                        "router_view_sha256": view.sha256,
                        "strong_win_score": prediction.value,
                        "threshold": policy.routellm_threshold,
                        "selected_role": decision.selected_role,
                        "selected_model": decision.selected_model,
                        "outcome_available": False,
                    }
                    policy_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    policy_handle.flush()
                    os.fsync(policy_handle.fileno())
                    decision_records.append(record)
                    receipts.extend(prediction.receipts)
                self._papc_decision = decision
            self.llm = self._papc_decision.selected_model
            self.llm_args = {
                "api_base": args.api_base.rstrip("/"),
                "reasoning_effort": REASONING_EFFORT,
            }
            if self._papc_seed is not None:
                self.llm_args["seed"] = self._papc_seed
            return super().generate_next_message(message, state)

    def factory(
        tools: Any,
        domain_policy: str,
        llm: str,
        llm_args: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> Any:
        return FreshRequestRoutedAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=dict(llm_args or {}),
        )

    agent_name = f"papc_routellm_released_airline_{binding.artifact_sha256[:12]}"
    api.registry.register_agent_factory(factory, agent_name)
    ordered_tasks = prepared.native_tasks
    run_config = api.text_run_config(
        domain="airline",
        task_split_name="base",
        task_ids=None,
        num_tasks=EXPECTED_DENOMINATOR,
        num_trials=1,
        max_retries=0,
        retry_delay=0.0,
        max_concurrency=EXPECTED_MAX_CONCURRENCY,
        seed=310,
        log_level="INFO",
        agent=agent_name,
        llm_agent=CHEAP_MODEL,
        llm_args_agent={
            "api_base": args.api_base.rstrip("/"),
            "reasoning_effort": REASONING_EFFORT,
        },
        user=TAU2_NATIVE_USER,
        llm_user=STRONG_MODEL,
        llm_args_user={
            "api_base": args.api_base.rstrip("/"),
            "reasoning_effort": REASONING_EFFORT,
        },
        max_steps=args.max_steps,
        auto_review=False,
        hallucination_retries=0,
    )

    def run_one(index: int) -> dict[str, Any]:
        simulation = api.run_single_task(
            run_config,
            ordered_tasks[index],
            seed=310,
            evaluation_type=getattr(api.evaluation_type, TAU2_OFFICIAL_EVALUATION),
            save_dir=None,
            verbose_logs=False,
            auto_review=False,
        )
        opening = _first_user_text(simulation)
        run_dir = simulations_root / f"{index:03d}_{uuid4().hex}"
        private_path, private_sha = _write_private_simulation(run_dir, simulation)
        return {
            "registry_index": index,
            "example_sha256": prepared.example_sha256s[index],
            "request_sha256": hashlib.sha256(opening.encode("utf-8")).hexdigest(),
            "official_score": _official_score(simulation),
            "private_simulation_relative_path": private_path.relative_to(
                private_root
            ).as_posix(),
            "private_simulation_sha256": private_sha,
        }

    completed: dict[int, dict[str, Any]] = {}
    failure: BaseException | None = None
    try:
        with _official_evaluator_config(api, type("Config", (), {
            "evaluator_model": STRONG_MODEL,
            "llm_args": lambda self: {
                "api_base": args.api_base.rstrip("/"),
                "reasoning_effort": REASONING_EFFORT,
            },
        })()):
            with _instrument_provider_attempts(
                api,
                budget=budget,
                broker=broker,
                receipts=receipts,
                max_infra_retries=0,
            ):
                executor = ThreadPoolExecutor(max_workers=EXPECTED_MAX_CONCURRENCY)
                pending: dict[Future[dict[str, Any]], int] = {}
                next_index = 0
                try:
                    while next_index < min(EXPECTED_MAX_CONCURRENCY, EXPECTED_DENOMINATOR):
                        pending[executor.submit(run_one, next_index)] = next_index
                        next_index += 1
                    while pending:
                        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                        for future in done:
                            index = pending.pop(future)
                            try:
                                completed[index] = future.result()
                            except BaseException as exc:  # fail closed on first error
                                failure = exc
                                break
                            if next_index < EXPECTED_DENOMINATOR:
                                pending[executor.submit(run_one, next_index)] = next_index
                                next_index += 1
                        if failure is not None:
                            for future in pending:
                                future.cancel()
                            break
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)
    finally:
        policy_handle.close()
    if failure is not None:
        raise RuntimeError(
            f"airline_first_execution_failure:{type(failure).__name__}:{failure}"
        ) from failure
    if set(completed) != set(range(EXPECTED_DENOMINATOR)):
        raise RuntimeError("airline_full_denominator_incomplete")
    if len(decision_records) != EXPECTED_DENOMINATOR:
        raise RuntimeError("airline_route_decision_count_incomplete")
    decisions_by_request = {item["request_sha256"]: item for item in decision_records}
    if len(decisions_by_request) != EXPECTED_DENOMINATOR:
        raise RuntimeError("airline_opening_request_sha256_not_unique")
    joined: list[dict[str, Any]] = []
    for index in range(EXPECTED_DENOMINATOR):
        row = completed[index]
        decision = decisions_by_request.get(row["request_sha256"])
        if decision is None:
            raise RuntimeError("airline_route_outcome_join_missing")
        joined.append(
            {
                **row,
                "strong_win_score": decision["strong_win_score"],
                "selected_role": decision["selected_role"],
                "selected_model": decision["selected_model"],
            }
        )
    route_counts = Counter(item["selected_role"] for item in joined)
    scores = [float(item["official_score"]) for item in joined]
    accounting = aggregate_online_accounting(
        receipts, prices=FROZEN_CACHE_NEUTRAL_PRICES
    )
    snapshot = ledger.snapshot()
    ledger_job = snapshot["jobs"][JOB_ID]
    if (
        snapshot["blocked"]
        or snapshot["open_reservation_count"] != 0
        or accounting["agent_side"]["retry_count"] != 0
        or accounting["excluded_harness"]["retry_count"] != 0
        or ledger_job["receipts_settled"]
        != accounting["agent_side"]["provider_calls"]
        + accounting["excluded_harness"]["provider_calls"]
    ):
        raise RuntimeError("airline_provider_ledger_not_clean")
    private_payload = {
        "schema": PRIVATE_SCHEMA,
        "provider_zero_gate_sha256": gate_sha,
        "test_registry_sha256": prepared.test_registry_sha256,
        "threshold_source_file_sha256": prepared.state_policy_sha256,
        "route_policy_file_sha256": _file_sha256(policy_path),
        "cases": joined,
        "accounting": accounting,
        "ledger": snapshot,
        "infra_failure_count": 0,
        "retry_count": 0,
        "merged_from_prior_runs": False,
        "row_splicing": False,
        "cap_or_score_correction": False,
    }
    private_result = private_root / "private_result.json"
    _write_new(private_result, private_payload, private=True)
    public_payload = {
        "schema": RESULT_SCHEMA,
        "decision": "completed_fresh_full50",
        "method": "RouteLLM-BERT (released, STATE-frozen OOD threshold)",
        "surface": "tau2_airline",
        "split": "base50_all_final_test",
        "official_metric": "accuracy",
        "result": {
            "passed": int(sum(scores)),
            "total": EXPECTED_DENOMINATOR,
            "accuracy": sum(scores) / EXPECTED_DENOMINATOR,
        },
        "routes": {
            "cheap": route_counts["cheap"],
            "strong": route_counts["strong"],
            "threshold": EXPECTED_THRESHOLD,
            "actual_episode_first_request": True,
        },
        "agent_side": accounting["agent_side"],
        "excluded_harness": accounting["excluded_harness"],
        "local_router": {
            "provider_calls": 0,
            "provider_cost_usd": 0.0,
            "base_bundle_sha256": ROUTELLM_BERT_BASE_BUNDLE_SHA256,
        },
        "source_bindings": {
            "provider_zero_gate_sha256": gate_sha,
            "threshold_source_file_sha256": prepared.state_policy_sha256,
            "test_registry_sha256": prepared.test_registry_sha256,
            "source_binding_sha256": prepared.source_binding_sha256,
            "private_result_sha256": _file_sha256(private_result),
            "route_policy_sha256": _file_sha256(policy_path),
            "private_simulation_sha256s": sorted(
                item["private_simulation_sha256"] for item in joined
            ),
        },
        "protocol": {
            "airline_fit_rows": 0,
            "airline_dev_rows": 0,
            "airline_threshold_selection": False,
            "endpoint_outcome_available_before_route": False,
            "max_concurrency": EXPECTED_MAX_CONCURRENCY,
            "max_infra_retries": 0,
            "infra_failure_count": 0,
            "merged_from_prior_runs": False,
            "row_splicing": False,
            "cap_or_score_correction": False,
        },
        "provider_calls_added_by_router": 0,
        "api_spend_performed": True,
        "usage_complete": accounting["agent_side"]["usage_complete"],
        "public_safe": True,
    }
    _write_new(args.public_result_json, public_payload, private=False)
    return {
        **base,
        "decision": "completed",
        "provider_calls": ledger_job["receipts_settled"],
        "secret_reads": 1,
        "api_spend_performed": True,
        "provider_zero_gate_sha256": gate_sha,
        "public_result": str(args.public_result_json.resolve()),
        "public_result_sha256": _file_sha256(args.public_result_json),
        "private_root": str(private_root),
        "private_result_sha256": _file_sha256(private_result),
        "result": public_payload["result"],
        "routes": public_payload["routes"],
        "agent_side": public_payload["agent_side"],
        "infra_failure_count": 0,
        "retry_count": 0,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepared = _prepare(args)
        base = _base_report(args, prepared)
    except Exception as exc:  # provider-zero preparation must fail closed
        report = {
            "schema": SCHEMA,
            "decision": "blocked",
            "execution_requested": bool(args.execute),
            "execution_started": False,
            "provider_calls": 0,
            "secret_reads": 0,
            "api_spend_performed": False,
            "error": f"{type(exc).__name__}:{exc}",
        }
        _write_new(args.report_json, report, private=False)
        print(json.dumps(report, sort_keys=True))
        return 2
    if not args.execute:
        report = {
            **base,
            "decision": "ready_for_explicit_execution",
            "execution_requested": False,
            "execution_started": False,
            "provider_calls": 0,
            "secret_reads": 0,
            "api_spend_performed": False,
            "authorizes_paid_run": False,
        }
        _write_new(args.report_json, report, private=False)
        print(json.dumps(report, sort_keys=True))
        return 0
    try:
        report = _execute(args, prepared, base)
    except Exception as exc:  # post-secret failures remain diagnostic only
        report = {
            **base,
            "decision": "blocked_after_execution_started",
            "execution_requested": True,
            "execution_started": True,
            "provider_calls": None,
            "secret_reads": 1,
            "api_spend_performed": None,
            "partial_output_is_diagnostic_only": True,
            "no_resume_or_row_splicing": True,
            "error": f"{type(exc).__name__}:{exc}",
        }
        _write_new(args.report_json, report, private=False)
        print(json.dumps(report, sort_keys=True))
        return 2
    _write_new(args.report_json, report, private=False)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
