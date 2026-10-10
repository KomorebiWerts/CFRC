#!/usr/bin/env python3
"""Bounded, method-faithful MTRouter fit/dev on official STATE train rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from papc.baselines.external_router_execution import (  # noqa: E402
    EXTERNAL_AGENT_COST_ROLES,
    SHARED_STAGE,
    TURN_STAGE,
    ProviderAttemptBroker,
    ProviderAttemptLedger,
    load_execution_plan,
)
from papc.baselines.external_router_fit_dev import (  # noqa: E402
    FitDevRegistry,
    FitDevSourceRow,
    load_fit_dev_registry,
)
from papc.baselines.external_router_fit_dev_native import (  # noqa: E402
    build_native_fit_dev_history_executor,
    load_native_fit_dev_authority,
)
from papc.baselines.mtrouter_upstream import (  # noqa: E402
    HISTORY_ENCODER_DIM,
    HISTORY_ENCODER_MODEL,
    HISTORY_MAX_TOKENS,
    MTROUTER_TRAINING_SOURCE_SHA256S,
    MTROUTER_UPSTREAM_REVISION,
    PUBLISHED_TRAINING_CONFIG,
    TRAINING_HISTORY_BACKEND,
    TRAINING_SCHEMA_VERSION,
    MTRouterRouletteCollector,
    MTRouterTrainingTrajectory,
    _build_torch_modules,
    _fit_internal_partition,
    _TorchLocalMTRouterTrainingBackend,
    inspect_mtrouter_training_readiness,
    mtrouter_messages_from_view,
    segment_mtrouter_history,
    validate_mtrouter_checkpoint_metadata,
)
from papc.baselines.router_accounting import CallReceipt  # noqa: E402
from papc.baselines.router_contract import (  # noqa: E402
    CHEAP_MODEL,
    MTROUTER,
    STATE_BENCH,
    STRONG_MODEL,
    UPSTREAM_REVISIONS,
    HistoryRouterView,
    RouterArtifactBinding,
)
from papc.baselines.router_policies import (  # noqa: E402
    RouterDecision,
    decide_mtrouter,
    decide_mtrouter_all_strong,
)
from papc.baselines.state_bench_router_driver import (  # noqa: E402
    load_state_bench_official_test_registry,
)
from papc.baselines.turn_router_fit_dispatch import (  # noqa: E402
    RecordingProviderAttemptBroker,
)
from papc.paths import DATA_ROOT

SCHEMA = "papc.mtrouter.state_bench_minimal_fit_dev.v1"
FIT_COUNT = 8
DEV_COUNT = 3
OBSERVED_SPEND_STOP_USD = Decimal("65")
EPISODE_ATTEMPT_CAP = 64

DEFAULT_SPLIT_GATE = (
    REPO_ROOT / "outputs/papc/external_router_split_gate_noapi_20260730_v2.json"
)
DEFAULT_BUDGET_GATE = (
    REPO_ROOT / "outputs/papc/external_router_budget_gate_noapi_20260730_v10.json"
)
DEFAULT_AUTHORITY = (
    REPO_ROOT / "outputs/papc/table1_authority_manifest_noapi_20260730_v2.json"
)
DEFAULT_STATE_REPO = Path(f"{DATA_ROOT}/benchmarks/state_bench/repo")
DEFAULT_TAU2_REPO = Path(f"{DATA_ROOT}/benchmarks/tau2-bench-revised/repo")
DEFAULT_AGENT_DIFF_REPO = Path(f"{DATA_ROOT}/benchmarks/agent_diff/repo")
DEFAULT_AGENT_DIFF_DATASET = DEFAULT_AGENT_DIFF_REPO / "datasets/agent-diff-bench"
DEFAULT_AGENT_DIFF_SDK = DEFAULT_AGENT_DIFF_REPO / "sdk/agent-diff-python"
DEFAULT_AGENT_DIFF_SCORER = REPO_ROOT / "scripts/agent_diff/papc_agent_diff_scoring.py"
DEFAULT_AGENT_DIFF_BASELINE = (
    REPO_ROOT / "scripts/agent_diff/papc_agent_diff_hf_test_baseline_run.py"
)
DEFAULT_UPSTREAM = Path(f"{DATA_ROOT}/papc/upstream_sources/mtrouter-ee9e4573")
DEFAULT_QWEN = Path(
    f"{DATA_ROOT}/papc/upstream_models/qwen3-embedding-0.6b-97b0c614"
)


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
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_append_only(path: Path, record: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(dict(record)) + b"\n")
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--split-gate-json", type=Path, default=DEFAULT_SPLIT_GATE)
    parser.add_argument("--budget-gate-json", type=Path, default=DEFAULT_BUDGET_GATE)
    parser.add_argument(
        "--authority-manifest-json", type=Path, default=DEFAULT_AUTHORITY
    )
    parser.add_argument("--state-repo", type=Path, default=DEFAULT_STATE_REPO)
    parser.add_argument("--tau2-repo", type=Path, default=DEFAULT_TAU2_REPO)
    parser.add_argument("--agent-diff-repo", type=Path, default=DEFAULT_AGENT_DIFF_REPO)
    parser.add_argument(
        "--agent-diff-dataset", type=Path, default=DEFAULT_AGENT_DIFF_DATASET
    )
    parser.add_argument(
        "--agent-diff-sdk-path", type=Path, default=DEFAULT_AGENT_DIFF_SDK
    )
    parser.add_argument(
        "--agent-diff-scorer-path", type=Path, default=DEFAULT_AGENT_DIFF_SCORER
    )
    parser.add_argument(
        "--agent-diff-baseline-authority-path",
        type=Path,
        default=DEFAULT_AGENT_DIFF_BASELINE,
    )
    parser.add_argument(
        "--mtrouter-upstream-checkout", type=Path, default=DEFAULT_UPSTREAM
    )
    parser.add_argument("--mtrouter-qwen-model-path", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--api-base")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--authorization-id")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _load_registries(
    args: argparse.Namespace, plan: Any
) -> tuple[FitDevRegistry, FitDevRegistry]:
    registries = []
    for role in ("router_fit", "router_dev"):
        matches = tuple(
            job
            for job in plan.jobs
            if job.stage == SHARED_STAGE
            and job.kind == "shared_endpoint_episode"
            and job.method == "shared_cheap_endpoint"
            and job.surface == STATE_BENCH
            and job.split_role == role
        )
        if len(matches) != 1:
            raise ValueError(f"minimal_state_shared_job_not_unique:{role}")
        registries.append(
            load_fit_dev_registry(
                job=matches[0],
                split_gate_json=args.split_gate_json,
                state_repo=args.state_repo,
                tau2_repo=args.tau2_repo,
                agent_diff_dataset=args.agent_diff_dataset,
            )
        )
    return registries[0], registries[1]


def _balanced_subset(
    rows: Sequence[FitDevSourceRow], count: int
) -> tuple[FitDevSourceRow, ...]:
    grouped: dict[str, list[FitDevSourceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.stratum].append(row)
    if len(grouped) != 3:
        raise ValueError("minimal_state_expected_three_domains")
    for values in grouped.values():
        values.sort(key=lambda item: item.example_sha256)
    selected: list[FitDevSourceRow] = []
    offset = 0
    strata = sorted(grouped)
    while len(selected) < count:
        for stratum in strata:
            if len(selected) == count:
                break
            selected.append(grouped[stratum][offset])
        offset += 1
    return tuple(selected)


def _turn_job(plan: Any, *, split_role: str) -> Any:
    kind = (
        "mtrouter_roulette_trajectories"
        if split_role == "router_fit"
        else "mtrouter_learned_dev_candidates"
    )
    matches = tuple(
        job
        for job in plan.jobs
        if job.stage == TURN_STAGE
        and job.kind == kind
        and job.method == MTROUTER
        and job.surface == STATE_BENCH
        and job.split_role == split_role
    )
    if len(matches) != 1:
        raise ValueError(f"minimal_state_turn_job_not_unique:{split_role}")
    return matches[0]


def _binding(
    *,
    fit_registry: FitDevRegistry,
    dev_registry: FitDevRegistry,
    test_registry_sha256: str,
    artifact_sha256: str,
    calibration_sha256: str,
) -> RouterArtifactBinding:
    return RouterArtifactBinding(
        method=MTROUTER,
        surface=STATE_BENCH,
        upstream_revision=UPSTREAM_REVISIONS[MTROUTER],
        artifact_sha256=artifact_sha256,
        fit_registry_sha256=fit_registry.registry_sha256,
        dev_registry_sha256=dev_registry.registry_sha256,
        test_registry_sha256=test_registry_sha256,
        calibration_sha256=calibration_sha256,
    )


def _source_record(row: FitDevSourceRow) -> dict[str, Any]:
    return {
        "stratum": row.stratum,
        "example_sha256": row.example_sha256,
        "payload_sha256": row.payload_sha256,
    }


def _preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    private_root = args.private_root.resolve()
    plan = load_execution_plan(
        split_gate_json=args.split_gate_json,
        budget_gate_json=args.budget_gate_json,
        authority_manifest_json=args.authority_manifest_json,
        private_root=private_root,
    )
    fit_registry, dev_registry = _load_registries(args, plan)
    test_registry = load_state_bench_official_test_registry(
        args.state_repo,
        split_gate_json=args.split_gate_json,
    )
    fit_rows = _balanced_subset(fit_registry.rows, FIT_COUNT)
    dev_rows = _balanced_subset(dev_registry.rows, DEV_COUNT)
    readiness = inspect_mtrouter_training_readiness(
        upstream_checkout=args.mtrouter_upstream_checkout,
        qwen_model_path=args.mtrouter_qwen_model_path,
    )
    blockers = list(readiness.blockers)
    if args.api_key_env != "OPENAI_API_KEY":
        blockers.append("api_key_env_not_frozen")
    frozen = {
        "plan_sha256": plan.sha256,
        "fit_registry_sha256": fit_registry.registry_sha256,
        "dev_registry_sha256": dev_registry.registry_sha256,
        "test_registry_sha256": test_registry.test_registry_sha256,
        "source_binding_sha256": fit_registry.source_binding_sha256,
        "fit_rows": [_source_record(row) for row in fit_rows],
        "dev_rows": [_source_record(row) for row in dev_rows],
        "mtrouter_upstream_revision": readiness.observed_upstream_revision,
        "mtrouter_upstream_source_sha256s": dict(readiness.source_sha256s),
        "qwen_source_sha256": readiness.qwen_source_sha256,
        "fit_protocol": "8_official_train_tasks_one_uniform_roulette_trajectory_each",
        "dev_protocol": "3_held_out_official_train_tasks_one_per_domain_learned_vs_all_strong",
        "selection_rule": "meet_all_strong_dev_quality_then_minimize_agent_cost_and_strong_calls",
        "max_online_episodes": FIT_COUNT + 2 * DEV_COUNT,
        "max_provider_attempts": (FIT_COUNT + 2 * DEV_COUNT) * EPISODE_ATTEMPT_CAP,
        "observed_spend_stop_usd": float(OBSERVED_SPEND_STOP_USD),
    }
    gate = {
        "schema": SCHEMA,
        "mode": "provider_zero_gate",
        "decision": "ready_for_authorized_fit_dev" if not blockers else "blocked",
        "blockers": blockers,
        "provider_calls": 0,
        "secret_reads": 0,
        "api_spend_performed": False,
        "execute": False,
        "frozen": frozen,
        "frozen_sha256": _digest(frozen),
    }
    runtime = {
        "plan": plan,
        "fit_registry": fit_registry,
        "dev_registry": dev_registry,
        "test_registry": test_registry,
        "fit_rows": fit_rows,
        "dev_rows": dev_rows,
        "readiness": readiness,
    }
    return gate, runtime


def _receipt_summary(receipts: Sequence[CallReceipt]) -> dict[str, Any]:
    by_role: dict[str, dict[str, float | int]] = {}
    for receipt in receipts:
        role = by_role.setdefault(receipt.role, {"calls": 0, "cost_usd": 0.0})
        role["calls"] = int(role["calls"]) + receipt.provider_calls
        role["cost_usd"] = float(role["cost_usd"]) + float(
            receipt.provider_reported_cost_usd or 0.0
        )
    return {
        "provider_calls": sum(item.provider_calls for item in receipts),
        "observed_provider_cost_usd": sum(
            float(item.provider_reported_cost_usd or 0.0) for item in receipts
        ),
        "agent_side_provider_api_cost_usd": sum(
            float(item.provider_reported_cost_usd or 0.0)
            for item in receipts
            if item.role in EXTERNAL_AGENT_COST_ROLES
        ),
        "strong_agent_calls": sum(
            item.provider_calls for item in receipts if item.role == "strong_agent"
        ),
        "cheap_agent_calls": sum(
            item.provider_calls for item in receipts if item.role == "cheap_agent"
        ),
        "by_role": by_role,
    }


def _roulette_episode(
    *,
    native_executor: Any,
    job: Any,
    row: FitDevSourceRow,
    binding: RouterArtifactBinding,
    broker: RecordingProviderAttemptBroker,
) -> tuple[MTRouterTrainingTrajectory, tuple[CallReceipt, ...]]:
    collector = MTRouterRouletteCollector(
        surface=STATE_BENCH,
        example_sha256=row.example_sha256,
        split="fit",
        trajectory_index=0,
    )
    before = len(broker.settled_receipts)

    def decide(view: HistoryRouterView) -> RouterDecision:
        turn = collector.select(view)
        return RouterDecision(
            method=MTROUTER,
            surface=STATE_BENCH,
            selected_role=turn.selected_role,
            selected_model=turn.selected_model,
            decision_stage="before_native_agent_turn",
            execution_semantics="continue_current_episode_with_selected_model",
            public_input_sha256=view.sha256,
            artifact_sha256=binding.artifact_sha256,
            scores=(("roulette_draw", turn.selection_draw),),
            threshold=0.5,
            rationale="frozen_uniform_half_roulette",
        )

    native = native_executor(job, row, binding, decide, broker)
    receipts = broker.settled_receipts[before:]
    trajectory = collector.finalize(terminal_official_reward=native.official_score)
    return trajectory, receipts


class _LearnedCandidate:
    def __init__(self, *, checkpoint_path: Path, model_path: Path, device: str) -> None:
        import torch
        import transformers

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        validate_mtrouter_checkpoint_metadata(checkpoint)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            padding_side="left",
            local_files_only=True,
        )
        encoder = transformers.AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            local_files_only=True,
        )
        if getattr(encoder.config, "hidden_size", None) != HISTORY_ENCODER_DIM:
            raise ValueError("minimal_mtrouter_history_encoder_dimension_mismatch")
        model_encoder, q_network = _build_torch_modules(torch)
        state_dict = checkpoint["state_dict"]
        model_encoder.load_state_dict(
            {
                key.removeprefix("model_encoder."): value
                for key, value in state_dict.items()
                if key.startswith("model_encoder.")
            },
            strict=True,
        )
        q_network.load_state_dict(
            {
                key.removeprefix("q_network."): value
                for key, value in state_dict.items()
                if key.startswith("q_network.")
            },
            strict=True,
        )
        self.torch = torch
        self.tokenizer = tokenizer
        self.encoder = encoder.to(device).eval()
        self.model_encoder = model_encoder.to(device).eval()
        self.q_network = q_network.to(device).eval()
        self.device = device

    def decide(
        self, binding: RouterArtifactBinding, view: HistoryRouterView
    ) -> RouterDecision:
        torch = self.torch
        text = segment_mtrouter_history(
            mtrouter_messages_from_view(view),
            tokenizer=self.tokenizer,
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=HISTORY_MAX_TOKENS,
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            hidden = self.encoder(**inputs).last_hidden_state
            attention = inputs["attention_mask"]
            if attention[:, -1].sum() == attention.shape[0]:
                state = hidden[:, -1]
            else:
                lengths = attention.sum(dim=1) - 1
                state = hidden[
                    torch.arange(hidden.shape[0], device=hidden.device),
                    lengths,
                ]
            state = torch.nn.functional.normalize(state, p=2, dim=-1)
            model_embeddings = self.model_encoder.forward_all()
            values = self.q_network(
                state.expand(2, -1),
                model_embeddings,
            ).squeeze(-1)
        cheap_value, strong_value = (
            float(value) for value in values.detach().cpu().tolist()
        )
        return decide_mtrouter(
            binding=binding,
            view=view,
            cheap_predicted_terminal_value=cheap_value,
            strong_predicted_terminal_value=strong_value,
        )


def _dev_episode(
    *,
    native_executor: Any,
    job: Any,
    row: FitDevSourceRow,
    binding: RouterArtifactBinding,
    broker: RecordingProviderAttemptBroker,
    learned: _LearnedCandidate | None,
) -> dict[str, Any]:
    before = len(broker.settled_receipts)
    candidate = "all_strong" if learned is None else "learned"

    def decide(view: HistoryRouterView) -> RouterDecision:
        if learned is None:
            return decide_mtrouter_all_strong(binding=binding, view=view)
        return learned.decide(binding, view)

    native = native_executor(job, row, binding, decide, broker)
    receipts = broker.settled_receipts[before:]
    return {
        "candidate": candidate,
        "source": _source_record(row),
        "official_score": native.official_score,
        "turn_count": len(native.decisions),
        "selected_roles": [item.selected_role for item in native.decisions],
        "receipt_summary": _receipt_summary(receipts),
        "receipts": [item.public_record() for item in receipts],
    }


def _train_candidate(
    *,
    args: argparse.Namespace,
    trajectories: Sequence[MTRouterTrainingTrajectory],
    readiness: Any,
    fit_registry: FitDevRegistry,
    fit_rows: Sequence[FitDevSourceRow],
) -> tuple[Path, dict[str, Any]]:
    import torch
    import transformers

    rows = tuple(row for trajectory in trajectories for row in trajectory.rows)
    row_hashes = tuple(_digest(row.public_record()) for row in rows)
    train_indices, val_indices = _fit_internal_partition(len(rows))
    subset_registry_sha256 = _digest(
        {
            "full_fit_registry_sha256": fit_registry.registry_sha256,
            "fit_example_sha256s": [row.example_sha256 for row in fit_rows],
            "protocol": "minimal_fit8_one_roulette_trajectory",
        }
    )
    fit_membership_sha256 = _digest([row.example_sha256 for row in fit_rows])
    fit_rows_sha256 = _digest(
        [trajectory.public_record() for trajectory in trajectories]
    )
    internal_split = {
        "algorithm": "deterministic_shuffle_then_tail_validation",
        "fit_train_row_sha256s": [row_hashes[index] for index in train_indices],
        "fit_internal_val_row_sha256s": [row_hashes[index] for index in val_indices],
    }
    receipt = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "surface": STATE_BENCH,
        "upstream_revision": MTROUTER_UPSTREAM_REVISION,
        "upstream_source_sha256s": dict(
            sorted(MTROUTER_TRAINING_SOURCE_SHA256S.items())
        ),
        "training_config": asdict(PUBLISHED_TRAINING_CONFIG),
        "training_config_sha256": PUBLISHED_TRAINING_CONFIG.sha256,
        "training_scope": "router_fit_only",
        "fit_registry_sha256": subset_registry_sha256,
        "fit_membership_sha256": fit_membership_sha256,
        "fit_rows_sha256": fit_rows_sha256,
        "fit_internal_split_sha256": _digest(internal_split),
        "qwen_source_sha256": readiness.qwen_source_sha256,
        "qwen_encoder_model": HISTORY_ENCODER_MODEL,
        "qwen_encoder_frozen": True,
        "history_backend": TRAINING_HISTORY_BACKEND,
        "local_files_only": True,
        "backend_name": "torch_local_qwen_single_head",
        "provider_calls": 0,
        "minimal_fit_protocol": "8_tasks_one_trajectory_each",
    }
    output_dir = args.private_root.resolve() / "training"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("minimal_mtrouter_training_output_not_empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = _TorchLocalMTRouterTrainingBackend(
        torch_module=torch,
        transformers_module=transformers,
        qwen_snapshot_path=args.mtrouter_qwen_model_path,
        qwen_source_sha256=readiness.qwen_source_sha256,
        device=args.device,
    )
    result = backend.train_and_export(
        rows=rows,
        row_sha256s=row_hashes,
        fit_train_indices=train_indices,
        fit_internal_val_indices=val_indices,
        output_dir=output_dir,
        checkpoint_receipt=receipt,
    )
    candidates = tuple(result.checkpoints)
    best = next(
        (
            candidate
            for candidate in candidates
            if candidate.origin == "fit_internal_best"
        ),
        next(candidate for candidate in candidates if candidate.origin == "final"),
    )
    validate_mtrouter_checkpoint_metadata(best.checkpoint_metadata)
    return best.checkpoint_path, {
        "checkpoint_path": str(best.checkpoint_path.resolve()),
        "checkpoint_sha256": _file_sha256(best.checkpoint_path),
        "checkpoint_epoch": best.epoch,
        "checkpoint_origin": best.origin,
        "fit_task_count": len(fit_rows),
        "fit_trajectory_count": len(trajectories),
        "fit_turn_row_count": len(rows),
        "fit_train_turn_row_count": len(train_indices),
        "fit_internal_val_turn_row_count": len(val_indices),
        "fit_subset_registry_sha256": subset_registry_sha256,
        "fit_rows_sha256": fit_rows_sha256,
        "qwen_source_sha256": readiness.qwen_source_sha256,
        "epochs_completed": result.epochs_completed,
    }


def _execute(
    args: argparse.Namespace, gate: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    if not args.authorization_id:
        raise ValueError("minimal_mtrouter_authorization_id_required")
    if gate.get("decision") != "ready_for_authorized_fit_dev":
        raise ValueError("minimal_mtrouter_gate_not_ready")
    frozen_gate = json.loads(args.gate_json.read_text(encoding="utf-8"))
    if frozen_gate != gate:
        raise ValueError("minimal_mtrouter_gate_binding_mismatch")
    api_base = args.api_base
    if not isinstance(api_base, str) or not api_base.startswith(
        ("http://", "https://")
    ):
        raise ValueError("minimal_mtrouter_api_base_invalid")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError("minimal_mtrouter_api_key_missing")

    plan = runtime["plan"]
    fit_registry = runtime["fit_registry"]
    dev_registry = runtime["dev_registry"]
    test_registry = runtime["test_registry"]
    fit_rows = runtime["fit_rows"]
    dev_rows = runtime["dev_rows"]
    fit_job = _turn_job(plan, split_role="router_fit")
    dev_job = _turn_job(plan, split_role="router_dev")
    total_cap = (FIT_COUNT + 2 * DEV_COUNT) * EPISODE_ATTEMPT_CAP
    ledger = ProviderAttemptLedger(
        global_attempt_cap=total_cap,
        global_observed_usd_stop=OBSERVED_SPEND_STOP_USD,
        stage_attempt_caps={TURN_STAGE: total_cap},
        stage_observed_usd_stops={TURN_STAGE: OBSERVED_SPEND_STOP_USD},
        job_attempt_caps={
            fit_job.job_id: (TURN_STAGE, FIT_COUNT * EPISODE_ATTEMPT_CAP),
            dev_job.job_id: (TURN_STAGE, 2 * DEV_COUNT * EPISODE_ATTEMPT_CAP),
        },
    )
    fit_broker = RecordingProviderAttemptBroker(
        ProviderAttemptBroker(ledger=ledger, job_id=fit_job.job_id)
    )
    dev_broker = RecordingProviderAttemptBroker(
        ProviderAttemptBroker(ledger=ledger, job_id=dev_job.job_id)
    )
    authority = load_native_fit_dev_authority(
        surfaces=[STATE_BENCH],
        private_root=args.private_root,
        state_repo=args.state_repo,
        tau2_data_repo=args.tau2_repo,
        tau2_runtime_repo=args.tau2_repo,
        agent_diff_repo=args.agent_diff_repo,
        agent_diff_dataset=args.agent_diff_dataset,
        agent_diff_sdk_path=args.agent_diff_sdk_path,
        agent_diff_scorer_path=args.agent_diff_scorer_path,
        agent_diff_baseline_authority_path=args.agent_diff_baseline_authority_path,
        agent_diff_base_url="http://localhost:8020",
    )
    native_executor = build_native_fit_dev_history_executor(
        authority=authority,
        api_base=api_base,
        api_key=api_key,
        max_retries=0,
        max_observed_provider_cost_usd=float(OBSERVED_SPEND_STOP_USD),
        api_key_env=args.api_key_env,
    )
    calibration_sha256 = _digest(
        {
            "fit_examples": [row.example_sha256 for row in fit_rows],
            "dev_examples": [row.example_sha256 for row in dev_rows],
            "protocol": "minimal_mtrouter_state_fit8_dev3",
        }
    )
    exploration_binding = _binding(
        fit_registry=fit_registry,
        dev_registry=dev_registry,
        test_registry_sha256=test_registry.test_registry_sha256,
        artifact_sha256=_digest(
            {"mode": "uniform_roulette", "calibration": calibration_sha256}
        ),
        calibration_sha256=calibration_sha256,
    )
    trajectories: list[MTRouterTrainingTrajectory] = []
    fit_evidence: list[dict[str, Any]] = []
    for index, row in enumerate(fit_rows, start=1):
        trajectory, receipts = _roulette_episode(
            native_executor=native_executor,
            job=fit_job,
            row=row,
            binding=exploration_binding,
            broker=fit_broker,
        )
        trajectories.append(trajectory)
        fit_evidence.append(
            {
                "source": _source_record(row),
                "trajectory": trajectory.public_record(),
                "receipt_summary": _receipt_summary(receipts),
                "receipts": [item.public_record() for item in receipts],
            }
        )
        print(
            json.dumps({"stage": "fit", "completed": index, "total": FIT_COUNT}),
            flush=True,
        )

    checkpoint_path, training = _train_candidate(
        args=args,
        trajectories=trajectories,
        readiness=runtime["readiness"],
        fit_registry=fit_registry,
        fit_rows=fit_rows,
    )
    checkpoint_sha256 = training["checkpoint_sha256"]
    learned_binding = _binding(
        fit_registry=fit_registry,
        dev_registry=dev_registry,
        test_registry_sha256=test_registry.test_registry_sha256,
        artifact_sha256=_digest(
            {"checkpoint_sha256": checkpoint_sha256, "calibration": calibration_sha256}
        ),
        calibration_sha256=calibration_sha256,
    )
    all_strong_binding = _binding(
        fit_registry=fit_registry,
        dev_registry=dev_registry,
        test_registry_sha256=test_registry.test_registry_sha256,
        artifact_sha256=_digest(
            {"mode": "all_strong", "calibration": calibration_sha256}
        ),
        calibration_sha256=calibration_sha256,
    )
    learned = _LearnedCandidate(
        checkpoint_path=checkpoint_path,
        model_path=args.mtrouter_qwen_model_path,
        device=args.device,
    )
    dev_evidence: list[dict[str, Any]] = []
    for index, row in enumerate(dev_rows, start=1):
        dev_evidence.append(
            _dev_episode(
                native_executor=native_executor,
                job=dev_job,
                row=row,
                binding=learned_binding,
                broker=dev_broker,
                learned=learned,
            )
        )
        print(
            json.dumps(
                {"stage": "dev_learned", "completed": index, "total": DEV_COUNT}
            ),
            flush=True,
        )
    for index, row in enumerate(dev_rows, start=1):
        dev_evidence.append(
            _dev_episode(
                native_executor=native_executor,
                job=dev_job,
                row=row,
                binding=all_strong_binding,
                broker=dev_broker,
                learned=None,
            )
        )
        print(
            json.dumps(
                {"stage": "dev_all_strong", "completed": index, "total": DEV_COUNT}
            ),
            flush=True,
        )

    aggregates: dict[str, dict[str, Any]] = {}
    for candidate in ("learned", "all_strong"):
        rows = [item for item in dev_evidence if item["candidate"] == candidate]
        aggregates[candidate] = {
            "correct": sum(item["official_score"] for item in rows),
            "denominator": len(rows),
            "accuracy": sum(item["official_score"] for item in rows) / len(rows),
            "agent_side_provider_api_cost_usd": sum(
                item["receipt_summary"]["agent_side_provider_api_cost_usd"]
                for item in rows
            ),
            "observed_provider_cost_usd": sum(
                item["receipt_summary"]["observed_provider_cost_usd"] for item in rows
            ),
            "provider_calls": sum(
                item["receipt_summary"]["provider_calls"] for item in rows
            ),
            "strong_agent_calls": sum(
                item["receipt_summary"]["strong_agent_calls"] for item in rows
            ),
        }
    learned_dev = aggregates["learned"]
    strong_dev = aggregates["all_strong"]
    learned_feasible = learned_dev["accuracy"] + 1e-12 >= strong_dev["accuracy"]
    selected = (
        "learned"
        if learned_feasible
        and (
            learned_dev["agent_side_provider_api_cost_usd"],
            learned_dev["strong_agent_calls"],
        )
        < (
            strong_dev["agent_side_provider_api_cost_usd"],
            strong_dev["strong_agent_calls"],
        )
        else "all_strong"
    )

    private_evidence = {
        "schema": SCHEMA,
        "authorization_id": args.authorization_id,
        "gate_sha256": _file_sha256(args.gate_json),
        "frozen_sha256": gate["frozen_sha256"],
        "fit_evidence": fit_evidence,
        "training": training,
        "dev_evidence": dev_evidence,
        "dev_aggregates": aggregates,
        "selected_candidate": selected,
        "ledger": ledger.snapshot(),
    }
    private_path = (
        args.private_root.resolve() / "mtrouter_state_minimal_fit_dev_evidence.json"
    )
    _write_json_append_only(private_path, private_evidence)
    report = {
        "schema": SCHEMA,
        "mode": "authorized_fit_dev_result",
        "decision": "fit_dev_complete",
        "authorization_id": args.authorization_id,
        "provider_error_count": 0,
        "retry_count": 0,
        "fit": {
            "task_count": FIT_COUNT,
            "trajectory_count": FIT_COUNT,
            "official_rewards": [
                trajectory.rows[0].terminal_official_reward
                for trajectory in trajectories
            ],
            "turn_row_count": sum(len(trajectory.rows) for trajectory in trajectories),
            "receipt_summary": _receipt_summary(fit_broker.settled_receipts),
        },
        "training": training,
        "dev": aggregates,
        "selected_candidate": selected,
        "selected_all_strong": selected == "all_strong",
        "selection_rule": gate["frozen"]["selection_rule"],
        "fit_examples": [_source_record(row) for row in fit_rows],
        "dev_examples": [_source_record(row) for row in dev_rows],
        "test_registry_sha256": test_registry.test_registry_sha256,
        "source_binding_sha256": fit_registry.source_binding_sha256,
        "cheap_model": CHEAP_MODEL,
        "strong_model": STRONG_MODEL,
        "reasoning_effort": "high",
        "private_evidence_path": str(private_path),
        "private_evidence_sha256": _file_sha256(private_path),
        "provider_calls": ledger.snapshot()["global"]["attempts_reserved"],
        "observed_provider_cost_usd": ledger.snapshot()["global"][
            "all_role_observed_provider_api_cost_usd"
        ],
        "agent_side_provider_api_cost_usd": ledger.snapshot()["global"][
            "agent_side_provider_api_cost_usd"
        ],
        "api_spend_performed": True,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate, runtime = _preflight(args)
    if not args.execute:
        _write_json_append_only(args.gate_json, gate)
        print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
        return 0 if gate["decision"] == "ready_for_authorized_fit_dev" else 2
    if args.report_json is None:
        raise ValueError("minimal_mtrouter_report_json_required")
    try:
        report = _execute(args, gate, runtime)
    except BaseException as exc:
        failure = {
            "schema": SCHEMA,
            "mode": "authorized_fit_dev_result",
            "decision": "stopped_after_first_error",
            "authorization_id": args.authorization_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "retry_count": 0,
            "no_retry_performed": True,
            "api_spend_performed": True,
        }
        _write_json_append_only(args.report_json, failure)
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
        return 2
    _write_json_append_only(args.report_json, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
