#!/usr/bin/env python3
"""Run the frozen STATE-fit MTRouter learned candidate on Airline base50."""

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
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from papc.baselines.external_router_execution import (  # noqa: E402
    ProviderAttemptBroker,
    ProviderAttemptLedger,
)
from papc.baselines.external_router_fit_dev_native import (  # noqa: E402
    activate_tau2_dependency_environment,
)
from papc.baselines.mtrouter_upstream import (  # noqa: E402
    MTROUTER_SERIALIZER_SHA256,
    MTROUTER_TRAINING_SOURCE_SHA256S,
    MTROUTER_UPSTREAM_REVISION,
    inspect_mtrouter_training_readiness,
)
from papc.baselines.router_accounting import (  # noqa: E402
    FROZEN_CACHE_NEUTRAL_PRICES,
    CallReceipt,
    aggregate_online_accounting,
)
from papc.baselines.router_contract import (  # noqa: E402
    CHEAP_MODEL,
    MTROUTER,
    REASONING_EFFORT,
    STRONG_MODEL,
    HistoryRouterView,
    RouterArtifactBinding,
)
from papc.baselines.router_runtime_environment import (  # noqa: E402
    FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256,
    validate_router_runtime_environment,
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
from scripts.baselines.papc_mtrouter_state_bench_minimal_live import (  # noqa: E402
    _LearnedCandidate,
)
from scripts.baselines.papc_routellm_airline_offtheshelf_live import (  # noqa: E402
    DEFAULT_TAU2_DATA_REPO,
    DEFAULT_TAU2_RUNTIME_REPO,
    EXPECTED_AIRLINE_SPLIT_FILE_SHA256,
    EXPECTED_AIRLINE_TASKS_FILE_SHA256,
    EXPECTED_BASE50_TASK_PLAN_SHA256,
    _digest,
    _file_sha256,
    _first_user_text,
    _official_score,
    _read_json,
    _write_new,
)

SCHEMA = "papc.mtrouter.airline.cross_domain_live.v1"
RESULT_SCHEMA = "papc.mtrouter.airline.cross_domain_result.v1"
PRIVATE_SCHEMA = "papc.mtrouter.airline.cross_domain_private.v1"
JOB_ID = "mtrouter_state_fit8_learned_airline_fresh_full50"
EXPECTED_DENOMINATOR = 50
EXPECTED_MAX_CONCURRENCY = 2
EXPECTED_ATTEMPT_CAP = 3000
EXPECTED_SPEND_STOP_USD = Decimal("80")
EXPECTED_CHECKPOINT_SHA256 = (
    "386374c7337533644838acf3bbb2ee92a3e2d44fd9bd73ce576d82c4e24bd6e0"
)
EXPECTED_QWEN_SOURCE_SHA256 = (
    "080f2af590661ff73cea3f2cde4e7843a426985e08e2f121f9d37348f33afad6"
)
EXPECTED_STATE_GATE_SHA256 = (
    "eb02e0d52e02f41d96fdfb7444e13af5bb2d977d51c20f9d26e48e6950323c5f"
)
EXPECTED_STATE_EVIDENCE_SHA256 = (
    "8eb78a45989507df27cab6b41cc66f5f4d7c20f2fd3e640c973c292e80cb5b30"
)
EXPECTED_STATE_FIT_REGISTRY_SHA256 = (
    "04aace958adc0a9b47b24c32e729fed81da10e516cb9d072546c94e562916aea"
)
DEFAULT_STATE_GATE = (
    REPO_ROOT
    / "outputs/papc/mtrouter_state_minimal_fit_dev_gate_noapi_20260801_v1.json"
)
DEFAULT_STATE_EVIDENCE = Path(
    f"{DATA_ROOT}/papc/mtrouter_state_minimal_20260801_v1/"
    "mtrouter_state_minimal_fit_dev_evidence.json"
)
DEFAULT_CHECKPOINT = Path(
    f"{DATA_ROOT}/papc/mtrouter_state_minimal_20260801_v1/"
    "training/q_function_best.pt"
)
DEFAULT_QWEN = Path(
    f"{DATA_ROOT}/papc/upstream_models/qwen3-embedding-0.6b-97b0c614"
)
DEFAULT_UPSTREAM = Path(f"{DATA_ROOT}/papc/upstream_sources/mtrouter-ee9e4573")


@dataclass(frozen=True)
class Prepared:
    native_api: Any
    native_tasks: tuple[Any, ...]
    example_sha256s: tuple[str, ...]
    test_registry_sha256: str
    source_binding_sha256: str
    candidate: Any
    binding: RouterArtifactBinding
    runtime_environment_sha256: str
    tau2_dependency_environment_sha256: str
    runtime_assets: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-gate-json", type=Path, default=DEFAULT_STATE_GATE)
    parser.add_argument(
        "--state-fit-dev-evidence-json", type=Path, default=DEFAULT_STATE_EVIDENCE
    )
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mtrouter-qwen-model-path", type=Path, default=DEFAULT_QWEN)
    parser.add_argument(
        "--mtrouter-upstream-checkout", type=Path, default=DEFAULT_UPSTREAM
    )
    parser.add_argument("--mtrouter-device", default="cpu")
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
    parser.add_argument(
        "--provider-attempt-hard-cap", type=int, default=EXPECTED_ATTEMPT_CAP
    )
    parser.add_argument(
        "--observed-spend-stop-usd", type=Decimal, default=EXPECTED_SPEND_STOP_USD
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _prepare(args: argparse.Namespace) -> Prepared:
    if args.max_concurrency != EXPECTED_MAX_CONCURRENCY:
        raise ValueError("mtrouter_airline_max_concurrency_must_equal_two")
    if args.provider_attempt_hard_cap != EXPECTED_ATTEMPT_CAP:
        raise ValueError("mtrouter_airline_attempt_cap_must_equal_3000")
    if args.observed_spend_stop_usd != EXPECTED_SPEND_STOP_USD:
        raise ValueError("mtrouter_airline_spend_stop_must_equal_80")
    if not args.api_base.startswith(("http://", "https://")):
        raise ValueError("mtrouter_airline_api_base_invalid")
    if args.api_key_env != "OPENAI_API_KEY":
        raise ValueError("api_key_env_must_be_OPENAI_API_KEY")
    if args.max_steps <= 0:
        raise ValueError("mtrouter_airline_max_steps_invalid")

    runtime_sha = validate_router_runtime_environment()
    if runtime_sha != FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256:
        raise ValueError("router_runtime_environment_sha256_mismatch")
    dependency_sha = activate_tau2_dependency_environment(args.tau2_runtime_repo)

    if _file_sha256(args.state_gate_json) != EXPECTED_STATE_GATE_SHA256:
        raise ValueError("mtrouter_state_gate_file_mismatch")
    if _file_sha256(args.state_fit_dev_evidence_json) != EXPECTED_STATE_EVIDENCE_SHA256:
        raise ValueError("mtrouter_state_fit_dev_evidence_file_mismatch")
    if _file_sha256(args.checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("mtrouter_state_fit8_checkpoint_mismatch")
    state_gate = _read_json(args.state_gate_json)
    state_evidence = _read_json(args.state_fit_dev_evidence_json)
    frozen = state_gate.get("frozen")
    training = state_evidence.get("training")
    if (
        state_gate.get("decision") != "ready_for_authorized_fit_dev"
        or not isinstance(frozen, Mapping)
        or frozen.get("fit_protocol")
        != "8_official_train_tasks_one_uniform_roulette_trajectory_each"
        or frozen.get("mtrouter_upstream_revision") != MTROUTER_UPSTREAM_REVISION
        or frozen.get("qwen_source_sha256") != EXPECTED_QWEN_SOURCE_SHA256
        or not isinstance(training, Mapping)
        or training.get("fit_task_count") != 8
        or training.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
        or training.get("fit_subset_registry_sha256")
        != EXPECTED_STATE_FIT_REGISTRY_SHA256
        or training.get("qwen_source_sha256") != EXPECTED_QWEN_SOURCE_SHA256
    ):
        raise ValueError("mtrouter_state_fit8_semantics_mismatch")
    if dict(frozen.get("mtrouter_upstream_source_sha256s", {})) != dict(
        MTROUTER_TRAINING_SOURCE_SHA256S
    ):
        raise ValueError("mtrouter_state_upstream_source_hashes_mismatch")

    readiness = inspect_mtrouter_training_readiness(
        upstream_checkout=args.mtrouter_upstream_checkout,
        qwen_model_path=args.mtrouter_qwen_model_path,
    )
    if (
        not readiness.ready
        or readiness.qwen_source_sha256 != EXPECTED_QWEN_SOURCE_SHA256
    ):
        raise ValueError("mtrouter_local_runtime_assets_not_ready")

    native_api = load_tau2_native_api(args.tau2_runtime_repo)
    split_path = native_api.runtime_repo / "data/tau2/domains/airline/split_tasks.json"
    tasks_path = native_api.runtime_repo / "data/tau2/domains/airline/tasks.json"
    data_split_path = args.tau2_repo / "data/tau2/domains/airline/split_tasks.json"
    data_tasks_path = args.tau2_repo / "data/tau2/domains/airline/tasks.json"
    if any(
        _file_sha256(path) != expected
        for path, expected in (
            (split_path, EXPECTED_AIRLINE_SPLIT_FILE_SHA256),
            (tasks_path, EXPECTED_AIRLINE_TASKS_FILE_SHA256),
            (data_split_path, EXPECTED_AIRLINE_SPLIT_FILE_SHA256),
            (data_tasks_path, EXPECTED_AIRLINE_TASKS_FILE_SHA256),
        )
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
    if tuple(str(task.id) for task in native_tasks) != tuple(
        str(index) for index in range(EXPECTED_DENOMINATOR)
    ):
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
    no_airline_dev_sha256 = _digest(
        {"role": "no_airline_dev", "rows": 0, "candidate": "state_fit8_learned"}
    )
    calibration_sha256 = _digest(
        {
            "source_surface": "state_bench",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "candidate": "fit_internal_best_learned",
            "airline_fit_rows": 0,
            "airline_dev_rows": 0,
            "airline_threshold_selection_rows": 0,
        }
    )
    artifact_sha256 = _digest(
        {
            "method": MTROUTER,
            "variant": "state_fit8_frozen_learned_cross_domain_candidate",
            "surface": "tau2_airline",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "qwen_source_sha256": EXPECTED_QWEN_SOURCE_SHA256,
            "serializer_sha256": MTROUTER_SERIALIZER_SHA256,
            "test_registry_sha256": task_plan_sha256,
            "calibration_sha256": calibration_sha256,
        }
    )
    binding = RouterArtifactBinding(
        method=MTROUTER,
        surface="tau2_airline",
        upstream_revision=MTROUTER_UPSTREAM_REVISION,
        artifact_sha256=artifact_sha256,
        fit_registry_sha256=EXPECTED_STATE_FIT_REGISTRY_SHA256,
        dev_registry_sha256=no_airline_dev_sha256,
        test_registry_sha256=task_plan_sha256,
        calibration_sha256=calibration_sha256,
    )
    candidate = _LearnedCandidate(
        checkpoint_path=args.checkpoint_path,
        model_path=args.mtrouter_qwen_model_path,
        device=args.mtrouter_device,
    )
    smoke_view = HistoryRouterView(
        original_request="Book a public airline itinerary.",
        public_history=(
            {"role": "user", "content": "Book a public airline itinerary."},
        ),
        turn_index=0,
        turn_cap=args.max_steps,
    )
    smoke_decision = candidate.decide(binding, smoke_view)
    runtime_assets = {
        "state_gate_file_sha256": EXPECTED_STATE_GATE_SHA256,
        "state_fit_dev_evidence_file_sha256": EXPECTED_STATE_EVIDENCE_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "qwen_source_sha256": EXPECTED_QWEN_SOURCE_SHA256,
        "mtrouter_upstream_revision": MTROUTER_UPSTREAM_REVISION,
        "mtrouter_upstream_source_sha256s": dict(MTROUTER_TRAINING_SOURCE_SHA256S),
        "mtrouter_serializer_sha256": MTROUTER_SERIALIZER_SHA256,
        "state_candidate_runner_sha256": _file_sha256(
            REPO_ROOT / "scripts/baselines/papc_mtrouter_state_bench_minimal_live.py"
        ),
        "mtrouter_runtime_source_sha256": _file_sha256(
            REPO_ROOT / "src/papc/baselines/mtrouter_upstream.py"
        ),
        "smoke_view_sha256": smoke_view.sha256,
        "smoke_selected_role": smoke_decision.selected_role,
        "smoke_scores": dict(smoke_decision.scores),
    }
    return Prepared(
        native_api=native_api,
        native_tasks=native_tasks,
        example_sha256s=example_sha256s,
        test_registry_sha256=task_plan_sha256,
        source_binding_sha256=source_binding_sha256,
        candidate=candidate,
        binding=binding,
        runtime_environment_sha256=runtime_sha,
        tau2_dependency_environment_sha256=dependency_sha,
        runtime_assets=runtime_assets,
    )


def _base_report(args: argparse.Namespace, prepared: Prepared) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "scope": "fresh_tau2_airline_history_routed_full50",
        "method": "MTRouter (STATE-fit8 frozen learned candidate)",
        "surface": "tau2_airline",
        "task_split": "base",
        "evaluation_role": "base50_all_final_test",
        "official_test_count": EXPECTED_DENOMINATOR,
        "cheap_model": CHEAP_MODEL,
        "strong_model": STRONG_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "router_input": "actual_episode_original_request_and_native_public_history_only",
        "cross_domain_policy": {
            "source_surface": "state_bench",
            "candidate": "fit_internal_best_learned",
            "source_dev_winner": "all_strong_not_used_for_this_cross_domain_candidate_test",
            "airline_fit_rows": 0,
            "airline_dev_rows": 0,
            "airline_threshold_selection_rows": 0,
            "cross_episode_learning": False,
        },
        "runtime_assets": prepared.runtime_assets,
        "method_artifact_sha256": prepared.binding.artifact_sha256,
        "fit_registry_sha256": prepared.binding.fit_registry_sha256,
        "dev_registry_sha256": prepared.binding.dev_registry_sha256,
        "calibration_sha256": prepared.binding.calibration_sha256,
        "test_registry_sha256": prepared.test_registry_sha256,
        "source_binding_sha256": prepared.source_binding_sha256,
        "base50_source": {
            "task_plan_sha256": EXPECTED_BASE50_TASK_PLAN_SHA256,
            "split_file_sha256": EXPECTED_AIRLINE_SPLIT_FILE_SHA256,
            "tasks_file_sha256": EXPECTED_AIRLINE_TASKS_FILE_SHA256,
        },
        "runtime_environment_sha256": prepared.runtime_environment_sha256,
        "tau2_dependency_environment_sha256": prepared.tau2_dependency_environment_sha256,
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
        "method_artifact_sha256",
        "fit_registry_sha256",
        "dev_registry_sha256",
        "calibration_sha256",
        "test_registry_sha256",
        "source_binding_sha256",
        "runtime_environment_sha256",
        "tau2_dependency_environment_sha256",
        "runtime_assets",
        "max_concurrency",
        "provider_attempt_hard_cap",
        "observed_spend_stop_usd",
    ):
        if gate.get(key) != base.get(key):
            raise ValueError(f"provider_zero_gate_{key}_mismatch")
    return _file_sha256(args.gate_json)


def _execute(
    args: argparse.Namespace,
    prepared: Prepared,
    base: dict[str, Any],
) -> dict[str, Any]:
    gate_sha = _validate_gate(args, base)
    if not os.environ.get(args.api_key_env, "").strip():
        raise ValueError("api_key_environment_variable_missing_or_empty")
    no_proxy = {
        item.strip()
        for item in os.environ.get("NO_PROXY", "").split(",")
        if item.strip()
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
        raise FileExistsError("mtrouter_airline_live_output_exists_no_overwrite")
    private_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    simulations_root = private_root / "simulations"
    simulations_root.mkdir(mode=0o700)
    policy_path = private_root / "history_route_policy.jsonl"
    policy_handle = os.fdopen(
        os.open(policy_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
        "w",
        encoding="utf-8",
    )
    policy_header = {
        "schema": "papc.mtrouter.airline.history_route_policy.v1",
        "decision": "learned_candidate_frozen_before_airline_provider_calls",
        "provider_zero_gate_sha256": gate_sha,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "qwen_source_sha256": EXPECTED_QWEN_SOURCE_SHA256,
        "airline_fit_rows": 0,
        "airline_dev_rows": 0,
        "airline_threshold_selection_rows": 0,
        "router_input": "actual_episode_original_request_and_native_public_history_only",
        "task_id_gold_outcome_cfrc_available_to_router": False,
        "cross_episode_learning": False,
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
    candidate = prepared.candidate
    binding = prepared.binding

    class FreshHistoryRoutedAgent(api.llm_agent_class):
        def __init__(self, *agent_args: Any, **agent_kwargs: Any) -> None:
            super().__init__(*agent_args, **agent_kwargs)
            self._papc_seed: int | None = None
            self._papc_turn_index = 0

        def set_seed(self, seed: int) -> None:
            self._papc_seed = seed
            super().set_seed(seed)

        def generate_next_message(self, message: Any, state: Any) -> tuple[Any, Any]:
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
            view = HistoryRouterView(
                original_request=opening,
                public_history=history,
                turn_index=self._papc_turn_index,
                turn_cap=args.max_steps,
            )
            with decision_lock:
                decision = candidate.decide(binding, view)
                score_map = dict(decision.scores)
                record = {
                    "decision_index": len(decision_records),
                    "request_sha256": hashlib.sha256(
                        opening.encode("utf-8")
                    ).hexdigest(),
                    "router_view_sha256": view.sha256,
                    "public_history_sha256": _digest(history),
                    "turn_index": self._papc_turn_index,
                    "cheap_predicted_terminal_value": score_map[
                        "cheap_predicted_terminal_value"
                    ],
                    "strong_predicted_terminal_value": score_map[
                        "strong_predicted_terminal_value"
                    ],
                    "selected_role": decision.selected_role,
                    "selected_model": decision.selected_model,
                    "outcome_available": False,
                    "task_identifier_available": False,
                    "cfrc_evidence_available": False,
                }
                policy_handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                policy_handle.flush()
                os.fsync(policy_handle.fileno())
                decision_records.append(record)
            self._papc_turn_index += 1
            self.llm = decision.selected_model
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
        return FreshHistoryRoutedAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=dict(llm_args or {}),
        )

    agent_name = f"papc_mtrouter_state_fit8_airline_{binding.artifact_sha256[:12]}"
    api.registry.register_agent_factory(factory, agent_name)
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
            prepared.native_tasks[index],
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
        evaluator_config = type(
            "Config",
            (),
            {
                "evaluator_model": STRONG_MODEL,
                "llm_args": lambda self: {
                    "api_base": args.api_base.rstrip("/"),
                    "reasoning_effort": REASONING_EFFORT,
                },
            },
        )()
        with _official_evaluator_config(api, evaluator_config):
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
                    while next_index < EXPECTED_MAX_CONCURRENCY:
                        pending[executor.submit(run_one, next_index)] = next_index
                        next_index += 1
                    while pending:
                        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                        for future in done:
                            index = pending.pop(future)
                            try:
                                completed[index] = future.result()
                            except BaseException as exc:
                                failure = exc
                                break
                            if next_index < EXPECTED_DENOMINATOR:
                                pending[executor.submit(run_one, next_index)] = (
                                    next_index
                                )
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
    if len(decision_records) < EXPECTED_DENOMINATOR:
        raise RuntimeError("airline_history_route_decisions_incomplete")
    decisions_by_request: dict[str, list[dict[str, Any]]] = {}
    for item in decision_records:
        decisions_by_request.setdefault(item["request_sha256"], []).append(item)
    if len(decisions_by_request) != EXPECTED_DENOMINATOR:
        raise RuntimeError("airline_opening_request_sha256_not_unique")

    joined: list[dict[str, Any]] = []
    for index in range(EXPECTED_DENOMINATOR):
        row = completed[index]
        route_rows = decisions_by_request.get(row["request_sha256"])
        if not route_rows:
            raise RuntimeError("airline_route_outcome_join_missing")
        route_rows = sorted(route_rows, key=lambda item: item["turn_index"])
        if [item["turn_index"] for item in route_rows] != list(range(len(route_rows))):
            raise RuntimeError("airline_route_turn_sequence_invalid")
        joined.append(
            {
                **row,
                "decision_count": len(route_rows),
                "selected_roles": [item["selected_role"] for item in route_rows],
                "route_decision_sha256s": [_digest(item) for item in route_rows],
            }
        )
    route_counts = Counter(item["selected_role"] for item in decision_records)
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
        "method": "MTRouter (STATE-fit8 frozen learned candidate)",
        "surface": "tau2_airline",
        "split": "base50_all_final_test",
        "official_metric": "accuracy",
        "result": {
            "passed": int(sum(scores)),
            "total": EXPECTED_DENOMINATOR,
            "accuracy": sum(scores) / EXPECTED_DENOMINATOR,
        },
        "routes": {
            "cheap_turns": route_counts["cheap"],
            "strong_turns": route_counts["strong"],
            "decision_count": len(decision_records),
            "actual_episode_public_history_each_turn": True,
        },
        "agent_side": accounting["agent_side"],
        "excluded_harness": accounting["excluded_harness"],
        "local_router": {
            "provider_calls": 0,
            "provider_cost_usd": 0.0,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "qwen_source_sha256": EXPECTED_QWEN_SOURCE_SHA256,
            "serializer_sha256": MTROUTER_SERIALIZER_SHA256,
        },
        "source_bindings": {
            "provider_zero_gate_sha256": gate_sha,
            "state_gate_file_sha256": EXPECTED_STATE_GATE_SHA256,
            "state_fit_dev_evidence_file_sha256": EXPECTED_STATE_EVIDENCE_SHA256,
            "test_registry_sha256": prepared.test_registry_sha256,
            "source_binding_sha256": prepared.source_binding_sha256,
            "method_artifact_sha256": prepared.binding.artifact_sha256,
            "private_result_sha256": _file_sha256(private_result),
            "route_policy_sha256": _file_sha256(policy_path),
            "private_simulation_sha256s": sorted(
                item["private_simulation_sha256"] for item in joined
            ),
        },
        "protocol": {
            "airline_fit_rows": 0,
            "airline_dev_rows": 0,
            "airline_threshold_selection_rows": 0,
            "endpoint_outcome_available_before_route": False,
            "task_id_gold_or_cfrc_available_to_router": False,
            "cross_episode_learning": False,
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
    except Exception as exc:
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
    except Exception as exc:
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
