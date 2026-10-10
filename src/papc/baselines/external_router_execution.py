"""Provider-zero execution contracts for the external-router tranche.

This module binds the frozen split and budget gates, expands their aggregate
caps into source-bound jobs, and provides the only supported raw-attempt
accounting interface for later native live wiring.  It does not import any
CFRC/certspec implementation and it does not itself contact a provider.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from papc.baselines.automix_upstream import (
    VERIFIER_TOKENIZER_BUNDLE_SHA256,
    VERIFIER_TOKENIZER_FILE_SHA256S,
    VERIFIER_TOKENIZER_ID,
    VERIFIER_TOKENIZER_SOURCE_REPO,
    VERIFIER_TOKENIZER_SOURCE_REVISION,
)
from papc.baselines.mtrouter_upstream import (
    TRAINING_CANDIDATE_POLICY,
    TRAINING_MAX_DEV_CANDIDATES,
    TRAINING_MAX_LEARNED_CANDIDATES,
    TRAJECTORIES_PER_TASK,
)
from papc.baselines.router_accounting import (
    HARNESS_ROLES,
    CallReceipt,
)
from papc.baselines.router_contract import (
    AGENT_DIFF,
    AUTOMIX,
    FRUGALGPT,
    MTROUTER,
    OPENSQUILLA_G0,
    ROUTELLM,
    STATE_BENCH,
    TAU2_AIRLINE,
    TAU2_RETAIL,
    TOOL_SANDBOX,
)
from papc.baselines.router_runtime_environment import (
    FROZEN_ROUTER_RUNTIME_ENVIRONMENT,
    FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256,
)
from papc.baselines.table1_results import (
    SurfaceAuthority,
    Table1AuthorityManifest,
    load_table1_authority,
)

SPLIT_GATE_SCHEMA = "papc.external_router_split_gate_noapi.v2"
BUDGET_GATE_SCHEMA = "papc.external_router_budget_gate_noapi.v10"
EXECUTION_PLAN_SCHEMA = "papc.external_router_execution_plan.v1"
EXECUTION_PREFLIGHT_SCHEMA = "papc.external_router_execution_gate_noapi.v10"
TABLE1_AUTHORITY_PATH_LABEL = (
    "table1_authority_manifest_noapi_20260730_v2.json"
)
TABLE1_AUTHORITY_FILE_SHA256 = (
    "eb2d8220b480f78ecb4d1fbc01c9bc542e7dfbce23024581d101f85da3dd9038"
)
TABLE1_AUTHORITY_SEMANTIC_SHA256 = (
    "a23f0614f71270ed6b9870880389817d48e6cdd82d5de82b4dfcbb7e20acb101"
)
AUTOMIX_ADAPTER_SOURCE_SHA256 = (
    "36456fb4fec03c43dc43ff74e3bf254e4ee0f8598e740bb0ed4b463adb1885e2"
)
OPENSQUILLA_TRAINER_SMOKE_EXPORT_SHA256 = (
    "52aaceb171e63b2b46d4e0a307be2552f6a2872fefd8219c8eba21f2f2c1706f"
)
OPENSQUILLA_TRAINER_SMOKE_ARTIFACT_SHA256 = (
    "2de12fc56747a82f04e4598f1ea324112fe28e8f9bbdc6d3287840c8b45e1c95"
)
OPENSQUILLA_TRAINER_SMOKE_SUMMARY_SHA256 = (
    "048d16af13ca5dc794968ee295b2af5ad3f4285f433a9f380ccd3319836c3c0c"
)

INCLUDED = "included"
COMPUTE_SUBSET = "compute_subset_not_ineligible"
INELIGIBLE = "ineligible_no_disjoint_official_train_test"
TABLE_COMPUTE_OMITTED = "C/O"
TABLE_NOT_APPLICABLE = "N/A-D"

ELIGIBLE_SURFACES = (
    STATE_BENCH,
    TAU2_RETAIL,
    TAU2_AIRLINE,
    AGENT_DIFF,
)
SURFACE_ORDER = (*ELIGIBLE_SURFACES, TOOL_SANDBOX)
METHOD_ORDER = (
    ROUTELLM,
    FRUGALGPT,
    AUTOMIX,
    MTROUTER,
    OPENSQUILLA_G0,
)

EXPECTED_SPLIT_COUNTS: Mapping[str, Mapping[str, int]] = {
    STATE_BENCH: {
        "official_train_count": 300,
        "router_fit_count": 240,
        "router_dev_count": 60,
        "official_test_count": 150,
    },
    TAU2_RETAIL: {
        "official_train_count": 74,
        "router_fit_count": 59,
        "router_dev_count": 15,
        "official_test_count": 40,
    },
    TAU2_AIRLINE: {
        "official_train_count": 30,
        "router_fit_count": 24,
        "router_dev_count": 6,
        "official_test_count": 20,
    },
    AGENT_DIFF: {
        "official_train_count": 179,
        "router_fit_count": 141,
        "router_dev_count": 38,
        "official_test_count": 45,
    },
}
EPISODE_ATTEMPT_CAPS = {
    STATE_BENCH: 64,
    TAU2_RETAIL: 104,
    TAU2_AIRLINE: 104,
    AGENT_DIFF: 40,
}
PUBLICATION_INCLUDED = {
    ROUTELLM: frozenset(ELIGIBLE_SURFACES),
    FRUGALGPT: frozenset(ELIGIBLE_SURFACES),
    AUTOMIX: frozenset({STATE_BENCH, TAU2_RETAIL, TAU2_AIRLINE}),
    MTROUTER: frozenset({STATE_BENCH, TAU2_AIRLINE, AGENT_DIFF}),
    OPENSQUILLA_G0: frozenset({STATE_BENCH, TAU2_RETAIL}),
}


def _publication_surfaces(method: str) -> tuple[str, ...]:
    return tuple(
        surface
        for surface in ELIGIBLE_SURFACES
        if surface in PUBLICATION_INCLUDED[method]
    )


SHARED_STAGE = "shared_paired_endpoint_fit_dev_and_route_embeddings"
AUTOMIX_STAGE = "automix_fit_dev_verifier_batches"
TURN_STAGE = "turn_router_fit_and_dev"
TEST_STAGE = "publication_subset16_official_test"
STAGE_ORDER = (
    SHARED_STAGE,
    AUTOMIX_STAGE,
    TURN_STAGE,
    TEST_STAGE,
)
EXPECTED_STAGE_ATTEMPT_CAPS = {
    SHARED_STAGE: 74_935,
    AUTOMIX_STAGE: 404,
    TURN_STAGE: 151_928,
    TEST_STAGE: 112_305,
}
EXPECTED_STAGE_CUMULATIVE_CAPS = {
    SHARED_STAGE: 74_935,
    AUTOMIX_STAGE: 75_339,
    TURN_STAGE: 227_267,
    TEST_STAGE: 339_572,
}
EXPECTED_GLOBAL_ATTEMPT_CAP = 339_572

_OBSERVED_USD_STAGE_BINDINGS = {
    SHARED_STAGE: (
        "shared_endpoint_fit_dev_collection",
        Decimal("260"),
    ),
    AUTOMIX_STAGE: ("automix_fit_dev_verifier_batches", Decimal("30")),
    TURN_STAGE: ("turn_router_fit_and_dev", Decimal("450")),
    TEST_STAGE: ("publication_subset16_official_test", Decimal("230")),
}
EXPECTED_GLOBAL_OBSERVED_USD_STOP = Decimal("970")

ROUTE_EMBEDDING_ATTEMPTS_PER_EXAMPLE = 1
AUTOMIX_N32_BATCH_ATTEMPTS_PER_EXAMPLE = 1
MTROUTER_TRAJECTORIES_PER_EXAMPLE = TRAJECTORIES_PER_TASK
MTROUTER_MAX_LEARNED_DEV_CANDIDATES = TRAINING_MAX_LEARNED_CANDIDATES
MTROUTER_ONLINE_DEV_CANDIDATES = TRAINING_MAX_DEV_CANDIDATES
OPENSQUILLA_DEV_FLOORS = 5
OFFICIAL_REPLICATES_PER_TASK = 1
EXTERNAL_AGENT_COST_ROLES = frozenset(
    {"router", "cheap_agent", "strong_agent", "verifier"}
)

JobKind = Literal[
    "shared_endpoint_episode",
    "route_embedding",
    "automix_n32_verifier",
    "mtrouter_roulette_trajectories",
    "mtrouter_learned_dev_candidates",
    "opensquilla_extra_dev_floors",
    "official_test_cell",
]


class ExecutionContractError(ValueError):
    """A frozen split, budget, job, or result contract is invalid."""


class LedgerError(RuntimeError):
    """A raw provider attempt violated a hard accounting boundary."""


class OfficialCellRerunRequired(RuntimeError):
    """A failed official cell must be rerun whole from the same manifest."""


@dataclass(frozen=True)
class StageLimit:
    stage: str
    provider_attempt_hard_cap: int
    cumulative_provider_attempt_hard_cap: int
    observed_spend_stop_usd: Decimal
    cumulative_observed_spend_stop_usd: Decimal

    def public_record(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "provider_attempt_hard_cap": self.provider_attempt_hard_cap,
            "cumulative_provider_attempt_hard_cap": (
                self.cumulative_provider_attempt_hard_cap
            ),
            "observed_spend_stop_usd": float(
                self.observed_spend_stop_usd
            ),
            "cumulative_observed_spend_stop_usd": float(
                self.cumulative_observed_spend_stop_usd
            ),
        }


@dataclass(frozen=True)
class ExecutionSurfaceAuthority:
    surface: str
    denominator: int
    replicates_per_task: int
    trial_policy_sha256: str
    test_registry_sha256: str
    fit_registry_sha256: str | None
    dev_registry_sha256: str | None
    official_scorer: str
    official_metric: str
    scorer_sha256: str
    runtime_name: str
    runtime_sha256: str
    source_binding_sha256: str | None

    def public_record(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "denominator": self.denominator,
            "replicates_per_task": self.replicates_per_task,
            "trial_policy_sha256": self.trial_policy_sha256,
            "test_registry_sha256": self.test_registry_sha256,
            "fit_registry_sha256": self.fit_registry_sha256,
            "dev_registry_sha256": self.dev_registry_sha256,
            "official_scorer": self.official_scorer,
            "official_metric": self.official_metric,
            "scorer_sha256": self.scorer_sha256,
            "runtime_name": self.runtime_name,
            "runtime_sha256": self.runtime_sha256,
            "source_binding_sha256": self.source_binding_sha256,
        }


@dataclass(frozen=True)
class ExecutionJob:
    job_id: str
    stage: str
    kind: JobKind
    method: str
    surface: str
    split_role: str
    item_count: int
    attempts_per_item: int
    provider_attempt_hard_cap: int
    registry_sha256: str
    source_binding_sha256: str
    split_gate_sha256: str
    relative_output_path: str
    output_path: Path
    requires_router_artifact: bool = False
    replicates_per_task: int | None = None
    trial_policy_sha256: str | None = None
    authority_test_registry_sha256: str | None = None
    authority_manifest_file_sha256: str | None = None
    authority_manifest_semantic_sha256: str | None = None
    official_scorer: str | None = None
    official_metric: str | None = None
    scorer_sha256: str | None = None
    runtime_name: str | None = None
    runtime_sha256: str | None = None
    details: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in STAGE_ORDER:
            raise ExecutionContractError("execution_job_stage_invalid")
        if self.item_count <= 0 or self.attempts_per_item <= 0:
            raise ExecutionContractError("execution_job_count_invalid")
        if self.provider_attempt_hard_cap != (
            self.item_count * self.attempts_per_item
        ):
            raise ExecutionContractError("execution_job_cap_mismatch")
        for field_name in (
            "registry_sha256",
            "source_binding_sha256",
            "split_gate_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.kind == "official_test_cell":
            if self.replicates_per_task != OFFICIAL_REPLICATES_PER_TASK:
                raise ExecutionContractError(
                    "official_test_job_not_single_replicate"
                )
            for field_name in (
                "trial_policy_sha256",
                "authority_test_registry_sha256",
                "authority_manifest_file_sha256",
                "authority_manifest_semantic_sha256",
                "scorer_sha256",
                "runtime_sha256",
            ):
                _require_sha256(getattr(self, field_name), field_name)
            for field_name in (
                "official_scorer",
                "official_metric",
                "runtime_name",
            ):
                value = getattr(self, field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ExecutionContractError(
                        f"official_test_job_{field_name}_invalid"
                    )
            if self.authority_test_registry_sha256 != self.registry_sha256:
                raise ExecutionContractError(
                    "official_test_job_registry_authority_mismatch"
                )
        elif any(
            getattr(self, field_name) is not None
            for field_name in (
                "replicates_per_task",
                "trial_policy_sha256",
                "authority_test_registry_sha256",
                "authority_manifest_file_sha256",
                "authority_manifest_semantic_sha256",
                "official_scorer",
                "official_metric",
                "scorer_sha256",
                "runtime_name",
                "runtime_sha256",
            )
        ):
            raise ExecutionContractError(
                "non_test_job_has_trial_authority"
            )

    def public_record(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "kind": self.kind,
            "method": self.method,
            "surface": self.surface,
            "split_role": self.split_role,
            "item_count": self.item_count,
            "attempts_per_item": self.attempts_per_item,
            "provider_attempt_hard_cap": self.provider_attempt_hard_cap,
            "registry_sha256": self.registry_sha256,
            "source_binding_sha256": self.source_binding_sha256,
            "split_gate_sha256": self.split_gate_sha256,
            "relative_output_path": self.relative_output_path,
            "requires_router_artifact": self.requires_router_artifact,
            "replicates_per_task": self.replicates_per_task,
            "trial_policy_sha256": self.trial_policy_sha256,
            "authority_test_registry_sha256": (
                self.authority_test_registry_sha256
            ),
            "authority_manifest_file_sha256": (
                self.authority_manifest_file_sha256
            ),
            "authority_manifest_semantic_sha256": (
                self.authority_manifest_semantic_sha256
            ),
            "official_scorer": self.official_scorer,
            "official_metric": self.official_metric,
            "scorer_sha256": self.scorer_sha256,
            "runtime_name": self.runtime_name,
            "runtime_sha256": self.runtime_sha256,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ExecutionPlan:
    split_gate_path: Path
    budget_gate_path: Path
    authority_manifest_path: Path
    private_root: Path
    split_gate_sha256: str
    budget_gate_sha256: str
    authority_manifest_file_sha256: str
    authority_manifest_semantic_sha256: str
    stages: tuple[StageLimit, ...]
    authorities: tuple[ExecutionSurfaceAuthority, ...]
    jobs: tuple[ExecutionJob, ...]
    coverage_cells: tuple[dict[str, Any], ...]

    @property
    def sha256(self) -> str:
        """Backward-compatible alias for the exact run-specific plan hash."""

        return self.run_specific_sha256

    @property
    def run_specific_sha256(self) -> str:
        return _digest(self.public_record())

    @property
    def protocol_semantic_sha256(self) -> str:
        return _digest(self.protocol_semantic_record())

    def stage_limit(self, stage: str) -> StageLimit:
        for item in self.stages:
            if item.stage == stage:
                return item
        raise KeyError(stage)

    def job(self, job_id: str) -> ExecutionJob:
        for item in self.jobs:
            if item.job_id == job_id:
                return item
        raise KeyError(job_id)

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": EXECUTION_PLAN_SCHEMA,
            "split_gate_binding": {
                "path_label": self.split_gate_path.name,
                "sha256": self.split_gate_sha256,
            },
            "budget_gate_binding": {
                "path_label": self.budget_gate_path.name,
                "sha256": self.budget_gate_sha256,
            },
            "table1_authority_binding": {
                "path_label": self.authority_manifest_path.name,
                "file_sha256": self.authority_manifest_file_sha256,
                "semantic_sha256": (
                    self.authority_manifest_semantic_sha256
                ),
            },
            "private_root_label": self.private_root.name,
            "stages": [item.public_record() for item in self.stages],
            "surface_authorities": [
                item.public_record() for item in self.authorities
            ],
            "jobs": [item.public_record() for item in self.jobs],
            "coverage_cells": list(self.coverage_cells),
            "failure_policy": {
                "no_partial_resume_or_cross_run_merge": True,
                "official_test_failure_action": (
                    "rerun_entire_cell_from_same_frozen_manifest"
                ),
                "failed_attempts_are_diagnostic_only": True,
                "table1_requires_full_single_run_denominator": True,
            },
            "cost_scope": {
                "pre_call_attempt_ledger_roles": "all_raw_provider_roles",
                "observed_spend_guard_roles": "all_raw_provider_roles",
                "dev_selection_provider_api_cost_usd_roles": (
                    "agent_side_only"
                ),
                "table1_provider_api_cost_usd_roles": "agent_side_only",
                "harness_provider_api_cost_usd": "reported_separately",
                "dev_selection_cost_source_stages": [
                    SHARED_STAGE,
                    AUTOMIX_STAGE,
                    TURN_STAGE,
                ],
                "table1_cost_source_stage": TEST_STAGE,
            },
            "attempt_cap_semantics": {
                STATE_BENCH: {
                    "per_full_episode_provider_attempt_cap": 64,
                    "semantics": "pre_call_safety_envelope",
                    "protocol_derived_bound": False,
                }
            },
        }

    def protocol_semantic_record(self) -> dict[str, Any]:
        """Return protocol identity without the private output-root label."""

        record = self.public_record()
        del record["private_root_label"]
        return record


def load_execution_plan(
    *,
    split_gate_json: Path,
    budget_gate_json: Path,
    authority_manifest_json: Path,
    private_root: Path,
) -> ExecutionPlan:
    """Load and expand the frozen split, budget, and Table 1 authority."""

    split_path = Path(split_gate_json).resolve()
    budget_path = Path(budget_gate_json).resolve()
    authority_path = Path(authority_manifest_json).resolve()
    root = Path(private_root).resolve()
    split, split_raw = _read_mapping(split_path, "split_gate")
    budget, budget_raw = _read_mapping(budget_path, "budget_gate")
    split_sha = hashlib.sha256(split_raw).hexdigest()
    budget_sha = hashlib.sha256(budget_raw).hexdigest()
    authority_manifest, authority_raw = _load_authority_manifest(
        authority_path
    )
    authority_file_sha = hashlib.sha256(authority_raw).hexdigest()
    authority_semantic_sha = authority_manifest.sha256

    _validate_split_gate(split)
    _validate_budget_gate(
        budget,
        split=split,
        split_path=split_path,
        split_sha256=split_sha,
    )
    stages = _load_stage_limits(budget)
    authorities = _build_execution_authorities(
        split=split,
        manifest=authority_manifest,
    )
    jobs = _build_jobs(
        split=split,
        split_sha256=split_sha,
        private_root=root,
        authorities=authorities,
        authority_manifest_file_sha256=authority_file_sha,
        authority_manifest_semantic_sha256=authority_semantic_sha,
    )
    _validate_job_caps(jobs, stages=stages, budget=budget)
    coverage_cells = _build_coverage_cells(jobs)
    return ExecutionPlan(
        split_gate_path=split_path,
        budget_gate_path=budget_path,
        authority_manifest_path=authority_path,
        private_root=root,
        split_gate_sha256=split_sha,
        budget_gate_sha256=budget_sha,
        authority_manifest_file_sha256=authority_file_sha,
        authority_manifest_semantic_sha256=authority_semantic_sha,
        stages=stages,
        authorities=authorities,
        jobs=jobs,
        coverage_cells=coverage_cells,
    )


def output_path_preflight(plan: ExecutionPlan) -> dict[str, Any]:
    """Check that every future private output is unique and untouched."""

    paths = tuple(job.output_path for job in plan.jobs)
    if len(set(paths)) != len(paths):
        raise ExecutionContractError("execution_output_path_duplicate")
    for path in paths:
        try:
            path.relative_to(plan.private_root)
        except ValueError as exc:
            raise ExecutionContractError(
                "execution_output_escapes_private_root"
            ) from exc
    existing = tuple(
        job.relative_output_path for job in plan.jobs if job.output_path.exists()
    )
    return {
        "private_root_label": plan.private_root.name,
        "job_output_path_count": len(paths),
        "unique_output_paths": True,
        "existing_target_count": len(existing),
        "existing_target_labels": list(existing),
        "safe_for_fresh_run": not existing,
        "no_paid_output_overwrite": not existing,
    }


@dataclass(frozen=True)
class AttemptReservation:
    reservation_id: int
    job_id: str
    stage: str


@dataclass
class _LedgerCounter:
    cap: int
    observed_usd_stop: Decimal | None
    attempts_reserved: int = 0
    receipts_settled: int = 0
    observed_usd: Decimal = Decimal("0")
    agent_side_usd: Decimal = Decimal("0")
    harness_usd: Decimal = Decimal("0")


class ProviderAttemptLedger:
    """Atomically enforce global, stage, and job raw-attempt caps."""

    def __init__(
        self,
        *,
        global_attempt_cap: int,
        global_observed_usd_stop: Decimal,
        stage_attempt_caps: Mapping[str, int],
        stage_observed_usd_stops: Mapping[str, Decimal],
        job_attempt_caps: Mapping[str, tuple[str, int]],
    ) -> None:
        if global_attempt_cap <= 0:
            raise ValueError("ledger_global_attempt_cap_invalid")
        self._global = _LedgerCounter(
            cap=global_attempt_cap,
            observed_usd_stop=Decimal(global_observed_usd_stop),
        )
        self._stages = {
            stage: _LedgerCounter(
                cap=cap,
                observed_usd_stop=Decimal(
                    stage_observed_usd_stops[stage]
                ),
            )
            for stage, cap in stage_attempt_caps.items()
        }
        self._jobs = {
            job_id: _LedgerCounter(cap=cap, observed_usd_stop=None)
            for job_id, (_, cap) in job_attempt_caps.items()
        }
        self._job_stages = {
            job_id: stage
            for job_id, (stage, _) in job_attempt_caps.items()
        }
        if set(self._stages) != set(stage_observed_usd_stops):
            raise ValueError("ledger_stage_usd_scope_mismatch")
        if any(stage not in self._stages for stage in self._job_stages.values()):
            raise ValueError("ledger_job_stage_missing")
        self._next_reservation_id = 1
        self._open: dict[int, AttemptReservation] = {}
        self._blocked_reason: str | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_plan(cls, plan: ExecutionPlan) -> ProviderAttemptLedger:
        return cls(
            global_attempt_cap=EXPECTED_GLOBAL_ATTEMPT_CAP,
            global_observed_usd_stop=EXPECTED_GLOBAL_OBSERVED_USD_STOP,
            stage_attempt_caps={
                item.stage: item.provider_attempt_hard_cap
                for item in plan.stages
            },
            stage_observed_usd_stops={
                item.stage: item.observed_spend_stop_usd
                for item in plan.stages
            },
            job_attempt_caps={
                job.job_id: (job.stage, job.provider_attempt_hard_cap)
                for job in plan.jobs
            },
        )

    def reserve(self, job_id: str, count: int = 1) -> AttemptReservation:
        """Reserve exactly one raw attempt before invoking a provider."""

        if count != 1 or isinstance(count, bool):
            raise LedgerError("raw_provider_attempt_reservation_must_equal_one")
        with self._lock:
            self._require_open()
            if job_id not in self._jobs:
                raise LedgerError("ledger_job_unknown")
            stage = self._job_stages[job_id]
            counters = (
                self._global,
                self._stages[stage],
                self._jobs[job_id],
            )
            if any(
                counter.attempts_reserved + 1 > counter.cap
                for counter in counters
            ):
                self._blocked_reason = (
                    f"provider_attempt_hard_cap_exceeded:{job_id}"
                )
                raise LedgerError(self._blocked_reason)
            for counter in counters:
                counter.attempts_reserved += 1
            reservation = AttemptReservation(
                reservation_id=self._next_reservation_id,
                job_id=job_id,
                stage=stage,
            )
            self._next_reservation_id += 1
            self._open[reservation.reservation_id] = reservation
            return reservation

    def settle(
        self,
        reservation: AttemptReservation,
        receipt: CallReceipt,
    ) -> None:
        """Settle one reserved raw attempt with complete observed USD."""

        with self._lock:
            resolved = self._require_open_reservation(reservation)
            if not isinstance(receipt, CallReceipt):
                self._fail_locked(
                    resolved,
                    "raw_provider_attempt_receipt_invalid",
                )
            if receipt.provider_calls != 1 or receipt.retry_count != 0:
                self._fail_locked(
                    resolved,
                    "raw_provider_attempt_receipt_not_atomic",
                )
            cost = receipt.provider_reported_cost_usd
            if cost is None:
                self._fail_locked(
                    resolved,
                    "raw_provider_attempt_cost_missing",
                )
            assert cost is not None
            observed = Decimal(str(cost))
            if (
                receipt.role not in EXTERNAL_AGENT_COST_ROLES
                and receipt.role not in HARNESS_ROLES
            ):
                self._fail_locked(
                    resolved,
                    "raw_provider_attempt_role_outside_external_tranche",
                )
            stage_counter = self._stages[resolved.stage]
            job_counter = self._jobs[resolved.job_id]
            counters = (self._global, stage_counter, job_counter)
            for counter in counters:
                counter.receipts_settled += 1
                counter.observed_usd += observed
                if receipt.role in EXTERNAL_AGENT_COST_ROLES:
                    counter.agent_side_usd += observed
                elif receipt.role in HARNESS_ROLES:
                    counter.harness_usd += observed
            del self._open[resolved.reservation_id]
            exceeded = next(
                (
                    name
                    for name, counter in (
                        ("global", self._global),
                        (resolved.stage, stage_counter),
                    )
                    if counter.observed_usd_stop is not None
                    and counter.observed_usd > counter.observed_usd_stop
                ),
                None,
            )
            if exceeded is not None:
                self._blocked_reason = (
                    f"observed_spend_stop_exceeded:{exceeded}"
                )
                raise LedgerError(self._blocked_reason)

    def fail(
        self,
        reservation: AttemptReservation,
        reason: str,
    ) -> None:
        with self._lock:
            resolved = self._require_open_reservation(reservation)
            self._fail_locked(resolved, f"raw_attempt_failed:{reason}")

    def assert_job_quiescent(self, job_id: str) -> None:
        with self._lock:
            if any(item.job_id == job_id for item in self._open.values()):
                raise LedgerError("job_has_unsettled_provider_attempt")
            self._require_open()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "blocked": self._blocked_reason is not None,
                "blocked_reason": self._blocked_reason,
                "open_reservation_count": len(self._open),
                "global": self._counter_record(self._global),
                "stages": {
                    stage: self._counter_record(counter)
                    for stage, counter in self._stages.items()
                },
                "jobs": {
                    job_id: self._counter_record(counter)
                    for job_id, counter in self._jobs.items()
                },
                "cost_scope": {
                    "attempts_and_observed_spend_guard": (
                        "all_raw_provider_roles"
                    ),
                    "agent_side_provider_api_cost_usd": (
                        "cheap_strong_router_verifier_only"
                    ),
                    "harness_provider_api_cost_usd": (
                        "user_simulator_and_evaluator_reported_separately"
                    ),
                    "dev_selection_cost_uses_agent_side_only": True,
                    "table1_cost_uses_agent_side_only": True,
                },
            }

    @staticmethod
    def _counter_record(counter: _LedgerCounter) -> dict[str, Any]:
        return {
            "provider_attempt_hard_cap": counter.cap,
            "attempts_reserved": counter.attempts_reserved,
            "receipts_settled": counter.receipts_settled,
            "observed_spend_stop_usd": (
                float(counter.observed_usd_stop)
                if counter.observed_usd_stop is not None
                else None
            ),
            "all_role_observed_provider_api_cost_usd": float(
                counter.observed_usd
            ),
            "agent_side_provider_api_cost_usd": float(
                counter.agent_side_usd
            ),
            "harness_provider_api_cost_usd": float(counter.harness_usd),
        }

    def _require_open(self) -> None:
        if self._blocked_reason is not None:
            raise LedgerError(f"ledger_blocked:{self._blocked_reason}")

    def _require_open_reservation(
        self,
        reservation: AttemptReservation,
    ) -> AttemptReservation:
        if not isinstance(reservation, AttemptReservation):
            raise LedgerError("ledger_reservation_invalid")
        resolved = self._open.get(reservation.reservation_id)
        if resolved != reservation:
            raise LedgerError("ledger_reservation_not_open")
        return resolved

    def _fail_locked(
        self,
        reservation: AttemptReservation,
        reason: str,
    ) -> None:
        self._blocked_reason = reason
        self._open.pop(reservation.reservation_id, None)
        raise LedgerError(reason)


class ProviderAttemptBroker:
    """The injected executor's only raw-provider call interface."""

    def __init__(
        self,
        *,
        ledger: ProviderAttemptLedger,
        job_id: str,
    ) -> None:
        self._ledger = ledger
        self._job_id = job_id

    def perform(
        self,
        raw_attempt: Callable[[], CallReceipt],
    ) -> CallReceipt:
        reservation = self._ledger.reserve(self._job_id, 1)
        try:
            receipt = raw_attempt()
        except BaseException as exc:
            try:
                self._ledger.fail(
                    reservation,
                    type(exc).__name__,
                )
            except LedgerError as ledger_error:
                raise exc from ledger_error
            raise AssertionError(
                "ledger_fail_must_raise"
            ) from exc
        self._ledger.settle(reservation, receipt)
        return receipt


@dataclass(frozen=True)
class StageExecutionAuthorization:
    """Explicit live authority supplied outside the provider-zero gate."""

    authorization_id: str
    plan_sha256: str
    stage: str
    provider_attempt_hard_cap: int
    observed_spend_stop_usd: Decimal

    def __post_init__(self) -> None:
        if not self.authorization_id.strip():
            raise ValueError("stage_authorization_id_empty")
        _require_sha256(self.plan_sha256, "authorization_plan_sha256")


@dataclass(frozen=True)
class JobExecutionResult:
    completed_item_count: int
    registry_sha256: str
    output_payload_sha256: str
    infra_failure_count: int = 0
    replicates_per_task: int | None = None
    trial_policy_sha256: str | None = None
    authority_manifest_file_sha256: str | None = None
    authority_manifest_semantic_sha256: str | None = None
    scorer_sha256: str | None = None
    runtime_sha256: str | None = None
    merged_from_prior_runs: bool = False


JobExecutor = Callable[
    [ExecutionJob, ProviderAttemptBroker],
    JobExecutionResult,
]


class BatchCoordinator:
    """Run whole jobs through an injected native executor after authority."""

    def __init__(
        self,
        *,
        plan: ExecutionPlan,
        authorization: StageExecutionAuthorization,
        ledger: ProviderAttemptLedger | None = None,
    ) -> None:
        stage_limit = plan.stage_limit(authorization.stage)
        if authorization.plan_sha256 != plan.sha256:
            raise ExecutionContractError("stage_authorization_plan_mismatch")
        if (
            authorization.provider_attempt_hard_cap
            != stage_limit.provider_attempt_hard_cap
        ):
            raise ExecutionContractError("stage_authorization_cap_mismatch")
        if (
            Decimal(authorization.observed_spend_stop_usd)
            != stage_limit.observed_spend_stop_usd
        ):
            raise ExecutionContractError("stage_authorization_usd_mismatch")
        self.plan = plan
        self.authorization = authorization
        self.ledger = ledger or ProviderAttemptLedger.from_plan(plan)
        self._states: dict[str, str] = {}
        self._diagnostics: list[dict[str, Any]] = []

    def execute_job(
        self,
        job_id: str,
        executor: JobExecutor,
    ) -> JobExecutionResult:
        job = self.plan.job(job_id)
        if job.stage != self.authorization.stage:
            raise ExecutionContractError("job_outside_authorized_stage")
        if job_id in self._states:
            raise ExecutionContractError(
                "job_already_attempted_no_partial_resume"
            )
        if job.output_path.exists():
            raise FileExistsError("job_output_exists_no_overwrite")
        self._states[job_id] = "running"
        broker = ProviderAttemptBroker(ledger=self.ledger, job_id=job_id)
        before = self.ledger.snapshot()["jobs"][job_id][
            "attempts_reserved"
        ]
        try:
            result = executor(job, broker)
            self.ledger.assert_job_quiescent(job_id)
            after = self.ledger.snapshot()["jobs"][job_id][
                "attempts_reserved"
            ]
            if after <= before:
                raise LedgerError(
                    "job_result_has_no_broker_accounted_provider_attempt"
                )
            _validate_job_result(job, result)
        except BaseException as exc:
            self._states[job_id] = "diagnostic_only_failed"
            diagnostic = {
                "job_id": job_id,
                "error_type": type(exc).__name__,
                "no_partial_resume_or_cross_run_merge": True,
                "required_action": (
                    "rerun_entire_cell_from_same_frozen_manifest"
                    if job.kind == "official_test_cell"
                    else "rerun_entire_job_from_same_frozen_manifest"
                ),
            }
            self._diagnostics.append(diagnostic)
            if job.kind == "official_test_cell":
                raise OfficialCellRerunRequired(
                    diagnostic["required_action"]
                ) from exc
            raise
        self._states[job_id] = "complete"
        return result

    def public_report(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan.sha256,
            "authorized_stage": self.authorization.stage,
            "job_states": dict(self._states),
            "failure_diagnostics": list(self._diagnostics),
            "ledger": self.ledger.snapshot(),
            "no_partial_resume_or_cross_run_merge": True,
        }


def _validate_job_result(
    job: ExecutionJob,
    result: JobExecutionResult,
) -> None:
    if not isinstance(result, JobExecutionResult):
        raise ExecutionContractError("job_result_type_invalid")
    if result.completed_item_count != job.item_count:
        raise ExecutionContractError("job_result_denominator_incomplete")
    if result.registry_sha256 != job.registry_sha256:
        raise ExecutionContractError("job_result_registry_mismatch")
    _require_sha256(result.output_payload_sha256, "output_payload_sha256")
    if result.infra_failure_count != 0:
        raise ExecutionContractError("job_result_infra_failure")
    if result.merged_from_prior_runs:
        raise ExecutionContractError("job_result_cross_run_merge_forbidden")
    if job.kind == "official_test_cell":
        if result.replicates_per_task != OFFICIAL_REPLICATES_PER_TASK:
            raise ExecutionContractError(
                "job_result_not_single_replicate"
            )
        if result.trial_policy_sha256 != job.trial_policy_sha256:
            raise ExecutionContractError(
                "job_result_trial_policy_mismatch"
            )
        for field_name in (
            "authority_manifest_file_sha256",
            "authority_manifest_semantic_sha256",
            "scorer_sha256",
            "runtime_sha256",
        ):
            if getattr(result, field_name) != getattr(job, field_name):
                raise ExecutionContractError(
                    f"job_result_{field_name}_mismatch"
                )


def _validate_split_gate(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SPLIT_GATE_SCHEMA:
        raise ExecutionContractError("split_gate_schema_not_v2")
    if payload.get("decision") != "ready_for_training_collection_plan":
        raise ExecutionContractError("split_gate_not_ready")
    _require_provider_zero(payload, "split_gate")
    checks = payload.get("checks")
    required_checks = {
        "official_counts_match",
        "fit_dev_test_disjoint",
        "state_task_and_environment_payloads_bound",
        "five_methods_bound",
        "each_method_has_four_eligible_surfaces",
        "toolsandbox_never_used_for_fit_or_dev",
    }
    if not isinstance(checks, Mapping) or any(
        checks.get(key) is not True for key in required_checks
    ):
        raise ExecutionContractError("split_gate_checks_incomplete")
    splits = payload.get("surface_splits")
    if not isinstance(splits, Mapping):
        raise ExecutionContractError("split_gate_surface_splits_missing")
    for surface, expected in EXPECTED_SPLIT_COUNTS.items():
        observed = splits.get(surface)
        if not isinstance(observed, Mapping):
            raise ExecutionContractError(
                f"split_gate_surface_missing:{surface}"
            )
        if any(observed.get(key) != value for key, value in expected.items()):
            raise ExecutionContractError(
                f"split_gate_count_mismatch:{surface}"
            )
        for field_name in (
            "fit_registry_sha256",
            "dev_registry_sha256",
            "test_registry_sha256",
            "source_binding_sha256",
        ):
            _require_sha256(
                observed.get(field_name),
                f"{surface}_{field_name}",
            )
    toolsandbox = splits.get(TOOL_SANDBOX)
    if (
        not isinstance(toolsandbox, Mapping)
        or toolsandbox.get("eligible_for_in_domain_router") is not False
        or toolsandbox.get("official_train_count") != 0
        or toolsandbox.get("official_test_count") != 129
    ):
        raise ExecutionContractError("split_gate_toolsandbox_invalid")
    _require_all_authorizations_false(payload, "split_gate")


def _validate_budget_gate(
    payload: Mapping[str, Any],
    *,
    split: Mapping[str, Any],
    split_path: Path,
    split_sha256: str,
) -> None:
    if payload.get("schema") != BUDGET_GATE_SCHEMA:
        raise ExecutionContractError("budget_gate_schema_not_v10")
    if payload.get("decision") != (
        "ready_for_selected_scope_implementation_noapi"
    ):
        raise ExecutionContractError("budget_gate_not_ready")
    _require_provider_zero(payload, "budget_gate")
    if payload.get("blockers") != []:
        raise ExecutionContractError("budget_gate_has_blockers")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or not checks or any(
        value is not True for value in checks.values()
    ):
        raise ExecutionContractError("budget_gate_checks_incomplete")
    if payload.get("surface_execution_ready") is not False:
        raise ExecutionContractError(
            "budget_gate_surface_execution_state_invalid"
        )
    _require_all_authorizations_false(payload, "budget_gate")
    environment = payload.get("external_router_environment")
    automix = payload.get("automix_source_authority")
    opensquilla_plan = payload.get("opensquilla_main_plan")
    opensquilla_reference = payload.get(
        "opensquilla_trainer_smoke_binding"
    )
    if (
        not isinstance(environment, Mapping)
        or environment.get("validated") is not True
        or environment.get("sha256")
        != FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256
        or environment.get("record")
        != FROZEN_ROUTER_RUNTIME_ENVIRONMENT
        or environment.get("numpy_version")
        != FROZEN_ROUTER_RUNTIME_ENVIRONMENT["packages"]["numpy"]
    ):
        raise ExecutionContractError(
            "budget_external_router_environment_invalid"
        )
    if (
        not isinstance(automix, Mapping)
        or automix.get("validated") is not True
        or automix.get("observed_revision")
        != "531af3ee3c4e4f4a53da1ffbc91dec4d6bb73781"
        or not isinstance(automix.get("pomdp_smoke"), Mapping)
        or automix["pomdp_smoke"].get("numpy_version") != "1.26.4"
        or automix["pomdp_smoke"].get("candidate_count") != 50
        or automix["pomdp_smoke"].get("candidate_sets_equal") is not True
        or automix["pomdp_smoke"].get("all_decisions_equal") is not True
        or automix["pomdp_smoke"].get(
            "runtime_environment_sha256"
        )
        != FROZEN_ROUTER_RUNTIME_ENVIRONMENT_SHA256
        or automix.get("reimplementation_source_sha256")
        != AUTOMIX_ADAPTER_SOURCE_SHA256
        or automix.get("verifier_tokenizer")
        != {
            "tokenizer_id": VERIFIER_TOKENIZER_ID,
            "source_repo": VERIFIER_TOKENIZER_SOURCE_REPO,
            "source_revision": VERIFIER_TOKENIZER_SOURCE_REVISION,
            "file_sha256s": dict(
                sorted(VERIFIER_TOKENIZER_FILE_SHA256S.items())
            ),
            "bundle_sha256": VERIFIER_TOKENIZER_BUNDLE_SHA256,
            "local_files_only": True,
        }
    ):
        raise ExecutionContractError(
            "budget_automix_source_authority_invalid"
        )
    if (
        not isinstance(opensquilla_plan, Mapping)
        or opensquilla_plan.get("runtime_artifact_schema")
        != "papc.opensquilla.runtime_artifact.v2"
        or opensquilla_plan.get("training_export_schema")
        != "papc.opensquilla.fit_training_export.v1"
        or opensquilla_plan.get("fit_consumption")
        != "in_domain_fit_only_paired_official_outcomes"
        or opensquilla_plan.get("used_init_model_required") is not True
        or opensquilla_plan.get(
            "new_provider_attempts_for_local_training"
        )
        != 0
        or not isinstance(
            opensquilla_plan.get("execution_resource_contract"),
            Mapping,
        )
        or opensquilla_plan["execution_resource_contract"].get(
            "checkout_and_bundle_are_distinct_paths"
        )
        is not True
        or opensquilla_plan["execution_resource_contract"].get(
            "cold_start_checkout_bundle_forbidden_for_main_table"
        )
        is not True
        or opensquilla_plan.get(
            "provider_zero_trainer_smoke_is_not_surface_artifact"
        )
        is not True
        or opensquilla_plan.get("official_surface_artifacts_ready")
        is not False
    ):
        raise ExecutionContractError(
            "budget_opensquilla_main_plan_invalid"
        )
    opensquilla_surfaces = opensquilla_plan.get("surfaces")
    if not isinstance(opensquilla_surfaces, Mapping):
        raise ExecutionContractError(
            "budget_opensquilla_surface_plan_invalid"
        )
    for surface in _publication_surfaces(OPENSQUILLA_G0):
        surface_plan = opensquilla_surfaces.get(surface)
        split_surface = split["surface_splits"][surface]
        if (
            not isinstance(surface_plan, Mapping)
            or surface_plan.get("fit_registry_sha256")
            != split_surface["fit_registry_sha256"]
            or surface_plan.get("dev_registry_sha256")
            != split_surface["dev_registry_sha256"]
            or surface_plan.get("test_registry_sha256")
            != split_surface["test_registry_sha256"]
            or surface_plan.get("expected_in_domain_fit_log_count")
            != split_surface["router_fit_count"]
            or surface_plan.get("official_surface_artifact_ready")
            is not False
            or surface_plan.get(
                "runtime_artifact_required_after_official_fit"
            )
            is not True
            or surface_plan.get(
                "runtime_artifact_must_bind_exact_surface_split"
            )
            is not True
        ):
            raise ExecutionContractError(
                f"budget_opensquilla_surface_plan_invalid:{surface}"
            )
    if (
        not isinstance(opensquilla_reference, Mapping)
        or opensquilla_reference.get("validated") is not True
        or opensquilla_reference.get("role")
        != (
            "provider_zero_trainer_recipe_smoke_not_table1_surface_artifact"
        )
        or opensquilla_reference.get("official_surface_artifact_ready")
        is not False
        or opensquilla_reference.get(
            "training_export_semantic_sha256"
        )
        != OPENSQUILLA_TRAINER_SMOKE_EXPORT_SHA256
        or opensquilla_reference.get(
            "runtime_artifact_semantic_sha256"
        )
        != OPENSQUILLA_TRAINER_SMOKE_ARTIFACT_SHA256
        or opensquilla_reference.get("coherent_smoke_file_sha256")
        != OPENSQUILLA_TRAINER_SMOKE_SUMMARY_SHA256
        or not isinstance(
            opensquilla_reference.get("smoke_resources"),
            Mapping,
        )
        or opensquilla_reference["smoke_resources"].get(
            "distinct_paths"
        )
        is not True
        or opensquilla_reference["smoke_resources"].get(
            "cold_start_bundle_used"
        )
        is not False
        or opensquilla_reference["smoke_resources"].get(
            "eligible_as_official_surface_execution_resource"
        )
        is not False
    ):
        raise ExecutionContractError(
            "budget_opensquilla_fit_reference_invalid"
        )

    binding = payload.get("split_gate_binding")
    if not isinstance(binding, Mapping):
        raise ExecutionContractError("budget_split_binding_missing")
    if (
        binding.get("validated") is not True
        or binding.get("required_schema") != SPLIT_GATE_SCHEMA
        or binding.get("schema") != SPLIT_GATE_SCHEMA
        or binding.get("path_label") != split_path.name
        or binding.get("sha256") != split_sha256
    ):
        raise ExecutionContractError("budget_split_binding_mismatch")
    expected_registry_bindings = {
        surface: {
            key: split["surface_splits"][surface][key]
            for key in (
                "fit_registry_sha256",
                "dev_registry_sha256",
                "test_registry_sha256",
                "source_binding_sha256",
            )
        }
        for surface in ELIGIBLE_SURFACES
    }
    if binding.get("surface_registry_sha256s") != (
        expected_registry_bindings
    ):
        raise ExecutionContractError(
            "budget_split_registry_binding_mismatch"
        )

    frozen = payload.get("frozen_publication_scope")
    if not isinstance(frozen, Mapping):
        raise ExecutionContractError("budget_publication_scope_missing")
    if (
        frozen.get("name") != "publication_subset16"
        or frozen.get("included_cell_count") != 16
        or frozen.get("compute_subset_cell_count") != 4
        or frozen.get("selection_is_result_blind") is not True
        or frozen.get("selected_before_official_test_execution") is not True
        or frozen.get(
            "retain_every_executed_result_including_baseline_wins"
        )
        is not True
        or frozen.get("table_display_label_for_omitted_eligible_cells")
        != TABLE_COMPUTE_OMITTED
    ):
        raise ExecutionContractError("budget_publication_scope_invalid")
    if frozen.get("coverage_matrix") != _expected_coverage_matrix():
        raise ExecutionContractError("budget_coverage_matrix_drift")

    call_caps = payload.get("provider_call_cap_plan")
    if not isinstance(call_caps, Mapping):
        raise ExecutionContractError("budget_call_cap_plan_missing")
    exact_scalars = {
        "route_embedding_attempts_per_example": (
            ROUTE_EMBEDDING_ATTEMPTS_PER_EXAMPLE
        ),
        "automix_n32_batch_attempts_per_example": (
            AUTOMIX_N32_BATCH_ATTEMPTS_PER_EXAMPLE
        ),
        "mtrouter_roulette_trajectories_per_fit_dev_example": (
            MTROUTER_TRAJECTORIES_PER_EXAMPLE
        ),
        "mtrouter_online_dev_candidates": MTROUTER_ONLINE_DEV_CANDIDATES,
        "mtrouter_learned_checkpoint_candidates_max": (
            MTROUTER_MAX_LEARNED_DEV_CANDIDATES
        ),
        "mtrouter_new_dev_candidates_after_shared_all_strong": (
            MTROUTER_MAX_LEARNED_DEV_CANDIDATES
        ),
        "mtrouter_training_candidate_policy": TRAINING_CANDIDATE_POLICY,
        "opensquilla_dev_projection_candidates": OPENSQUILLA_DEV_FLOORS,
        "cumulative_provider_attempt_hard_cap": (
            EXPECTED_GLOBAL_ATTEMPT_CAP
        ),
    }
    if any(call_caps.get(key) != value for key, value in exact_scalars.items()):
        raise ExecutionContractError("budget_call_cap_scalar_drift")
    if call_caps.get("per_full_episode_provider_attempt_cap") != (
        EPISODE_ATTEMPT_CAPS
    ):
        raise ExecutionContractError("budget_episode_cap_drift")
    if (
        call_caps.get("authorizes_provider_calls") is not False
        or call_caps.get("live_global_and_per_job_ledger_ready") is not False
    ):
        raise ExecutionContractError("budget_call_authority_invalid")
    state_policy = call_caps.get("state_bench_trial_policy")
    if (
        not isinstance(state_policy, Mapping)
        or state_policy.get("replicates_per_official_test_task") != 1
        or state_policy.get("reported_denominator") != 150
        or state_policy.get("official_five_run_protocol_claimed") is not False
    ):
        raise ExecutionContractError("budget_state_trial_policy_invalid")


def _load_stage_limits(
    budget: Mapping[str, Any],
) -> tuple[StageLimit, ...]:
    call_caps = budget["provider_call_cap_plan"]
    raw_attempt_stages = call_caps.get("stages")
    frozen_stages = budget["frozen_publication_scope"].get("stages")
    if not isinstance(raw_attempt_stages, list) or not isinstance(
        frozen_stages,
        list,
    ):
        raise ExecutionContractError("budget_stage_plan_missing")
    attempt_by_name = {
        item.get("stage"): item
        for item in raw_attempt_stages
        if isinstance(item, Mapping)
    }
    usd_by_name = {
        item.get("stage"): item
        for item in frozen_stages
        if isinstance(item, Mapping)
    }
    stages: list[StageLimit] = []
    cumulative_usd = Decimal("0")
    for stage in STAGE_ORDER:
        attempt = attempt_by_name.get(stage)
        usd_name, expected_usd = _OBSERVED_USD_STAGE_BINDINGS[stage]
        usd = usd_by_name.get(usd_name)
        if not isinstance(attempt, Mapping) or not isinstance(usd, Mapping):
            raise ExecutionContractError(f"budget_stage_missing:{stage}")
        expected_cap = EXPECTED_STAGE_ATTEMPT_CAPS[stage]
        expected_cumulative = EXPECTED_STAGE_CUMULATIVE_CAPS[stage]
        if (
            attempt.get("stage_provider_attempt_hard_cap") != expected_cap
            or attempt.get("cumulative_provider_attempt_hard_cap")
            != expected_cumulative
            or attempt.get("provider_calls_before_explicit_authorization")
            != 0
        ):
            raise ExecutionContractError(
                f"budget_stage_attempt_cap_drift:{stage}"
            )
        observed_usd = Decimal(
            str(usd.get("stage_observed_spend_stop_usd"))
        )
        cumulative_usd += expected_usd
        if (
            observed_usd != expected_usd
            or Decimal(
                str(usd.get("cumulative_observed_spend_stop_usd"))
            )
            != cumulative_usd
            or usd.get("provider_calls_before_explicit_authorization")
            != 0
            or usd.get("stop_when_guard_reached") is not True
        ):
            raise ExecutionContractError(
                f"budget_stage_usd_stop_drift:{stage}"
            )
        stages.append(
            StageLimit(
                stage=stage,
                provider_attempt_hard_cap=expected_cap,
                cumulative_provider_attempt_hard_cap=expected_cumulative,
                observed_spend_stop_usd=expected_usd,
                cumulative_observed_spend_stop_usd=cumulative_usd,
            )
        )
    if cumulative_usd != EXPECTED_GLOBAL_OBSERVED_USD_STOP:
        raise ExecutionContractError("budget_global_usd_stop_drift")
    return tuple(stages)


def _load_authority_manifest(
    path: Path,
) -> tuple[Table1AuthorityManifest, bytes]:
    if path.name != TABLE1_AUTHORITY_PATH_LABEL:
        raise ExecutionContractError(
            "table1_authority_path_label_mismatch"
        )
    _, raw = _read_mapping(path, "table1_authority")
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != TABLE1_AUTHORITY_FILE_SHA256:
        raise ExecutionContractError(
            "table1_authority_file_sha256_mismatch"
        )
    try:
        manifest = load_table1_authority(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExecutionContractError(
            "table1_authority_manifest_invalid"
        ) from exc
    if manifest.sha256 != TABLE1_AUTHORITY_SEMANTIC_SHA256:
        raise ExecutionContractError(
            "table1_authority_semantic_sha256_mismatch"
        )
    return manifest, raw


def _build_execution_authorities(
    *,
    split: Mapping[str, Any],
    manifest: Table1AuthorityManifest,
) -> tuple[ExecutionSurfaceAuthority, ...]:
    authorities: list[ExecutionSurfaceAuthority] = []
    for authority in manifest.authorities:
        if not isinstance(authority, SurfaceAuthority):
            raise ExecutionContractError(
                "table1_surface_authority_invalid"
            )
        surface = authority.surface
        split_entry = split["surface_splits"][surface]
        if surface in ELIGIBLE_SURFACES:
            if (
                authority.denominator
                != split_entry["official_test_count"]
                or authority.test_registry_sha256
                != split_entry["test_registry_sha256"]
                or authority.fit_registry_sha256
                != split_entry["fit_registry_sha256"]
                or authority.dev_registry_sha256
                != split_entry["dev_registry_sha256"]
            ):
                raise ExecutionContractError(
                    f"table1_authority_split_mismatch:{surface}"
                )
            source_binding_sha256 = split_entry[
                "source_binding_sha256"
            ]
        else:
            if (
                surface != TOOL_SANDBOX
                or authority.denominator
                != split_entry["official_test_count"]
                or authority.fit_registry_sha256 is not None
                or authority.dev_registry_sha256 is not None
            ):
                raise ExecutionContractError(
                    "table1_toolsandbox_authority_mismatch"
                )
            source_binding_sha256 = None
        authorities.append(
            ExecutionSurfaceAuthority(
                surface=surface,
                denominator=authority.denominator,
                replicates_per_task=authority.replicates_per_task,
                trial_policy_sha256=authority.trial_policy_sha256,
                test_registry_sha256=authority.test_registry_sha256,
                fit_registry_sha256=authority.fit_registry_sha256,
                dev_registry_sha256=authority.dev_registry_sha256,
                official_scorer=authority.official_scorer,
                official_metric=authority.official_metric,
                scorer_sha256=authority.scorer_sha256,
                runtime_name=authority.runtime_name,
                runtime_sha256=authority.runtime_sha256,
                source_binding_sha256=source_binding_sha256,
            )
        )
    return tuple(authorities)


def _build_jobs(
    *,
    split: Mapping[str, Any],
    split_sha256: str,
    private_root: Path,
    authorities: Sequence[ExecutionSurfaceAuthority],
    authority_manifest_file_sha256: str,
    authority_manifest_semantic_sha256: str,
) -> tuple[ExecutionJob, ...]:
    jobs: list[ExecutionJob] = []
    authority_by_surface = {item.surface: item for item in authorities}

    def add(
        *,
        stage: str,
        kind: JobKind,
        method: str,
        surface: str,
        split_role: str,
        item_count: int,
        attempts_per_item: int,
        registry_sha256: str,
        source_binding_sha256: str,
        requires_router_artifact: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        tokens = (stage, kind, method, surface, split_role)
        job_id = ".".join(tokens)
        relative = str(Path(stage) / f"{job_id}.jsonl")
        authority = authority_by_surface.get(surface)
        is_test = kind == "official_test_cell"
        jobs.append(
            ExecutionJob(
                job_id=job_id,
                stage=stage,
                kind=kind,
                method=method,
                surface=surface,
                split_role=split_role,
                item_count=item_count,
                attempts_per_item=attempts_per_item,
                provider_attempt_hard_cap=(
                    item_count * attempts_per_item
                ),
                registry_sha256=registry_sha256,
                source_binding_sha256=source_binding_sha256,
                split_gate_sha256=split_sha256,
                relative_output_path=relative,
                output_path=private_root / relative,
                requires_router_artifact=requires_router_artifact,
                replicates_per_task=(
                    authority.replicates_per_task if is_test else None
                ),
                trial_policy_sha256=(
                    authority.trial_policy_sha256 if is_test else None
                ),
                authority_test_registry_sha256=(
                    authority.test_registry_sha256 if is_test else None
                ),
                authority_manifest_file_sha256=(
                    authority_manifest_file_sha256 if is_test else None
                ),
                authority_manifest_semantic_sha256=(
                    authority_manifest_semantic_sha256
                    if is_test
                    else None
                ),
                official_scorer=(
                    authority.official_scorer if is_test else None
                ),
                official_metric=(
                    authority.official_metric if is_test else None
                ),
                scorer_sha256=(
                    authority.scorer_sha256 if is_test else None
                ),
                runtime_name=(
                    authority.runtime_name if is_test else None
                ),
                runtime_sha256=(
                    authority.runtime_sha256 if is_test else None
                ),
                details=tuple(sorted((details or {}).items())),
            )
        )

    for surface in ELIGIBLE_SURFACES:
        entry = split["surface_splits"][surface]
        episode_cap = EPISODE_ATTEMPT_CAPS[surface]
        for split_role, count_key, registry_key in (
            ("router_fit", "router_fit_count", "fit_registry_sha256"),
            ("router_dev", "router_dev_count", "dev_registry_sha256"),
        ):
            count = entry[count_key]
            for role in ("cheap", "strong"):
                add(
                    stage=SHARED_STAGE,
                    kind="shared_endpoint_episode",
                    method=f"shared_{role}_endpoint",
                    surface=surface,
                    split_role=split_role,
                    item_count=count,
                    attempts_per_item=episode_cap,
                    registry_sha256=entry[registry_key],
                    source_binding_sha256=entry[
                        "source_binding_sha256"
                    ],
                    details={
                        "model_role": role,
                        "attempt_cap_semantics": (
                            "pre_call_safety_envelope"
                        ),
                    },
                )
            add(
                stage=SHARED_STAGE,
                kind="route_embedding",
                method=ROUTELLM,
                surface=surface,
                split_role=split_role,
                item_count=count,
                attempts_per_item=(
                    ROUTE_EMBEDDING_ATTEMPTS_PER_EXAMPLE
                ),
                registry_sha256=entry[registry_key],
                source_binding_sha256=entry["source_binding_sha256"],
                details={
                    "embedding_attempts_per_example": 1,
                },
            )

    for surface in _publication_surfaces(AUTOMIX):
        entry = split["surface_splits"][surface]
        for split_role, count_key, registry_key in (
            ("router_fit", "router_fit_count", "fit_registry_sha256"),
            ("router_dev", "router_dev_count", "dev_registry_sha256"),
        ):
            add(
                stage=AUTOMIX_STAGE,
                kind="automix_n32_verifier",
                method=AUTOMIX,
                surface=surface,
                split_role=split_role,
                item_count=entry[count_key],
                attempts_per_item=(
                    AUTOMIX_N32_BATCH_ATTEMPTS_PER_EXAMPLE
                ),
                registry_sha256=entry[registry_key],
                source_binding_sha256=entry["source_binding_sha256"],
                details={
                    "self_verification_samples_per_batch": 32,
                    "provider_batches_per_example": 1,
                },
            )

    for surface in _publication_surfaces(MTROUTER):
        entry = split["surface_splits"][surface]
        episode_cap = EPISODE_ATTEMPT_CAPS[surface]
        for split_role, count_key, registry_key in (
            ("router_fit", "router_fit_count", "fit_registry_sha256"),
            ("router_dev", "router_dev_count", "dev_registry_sha256"),
        ):
            add(
                stage=TURN_STAGE,
                kind="mtrouter_roulette_trajectories",
                method=MTROUTER,
                surface=surface,
                split_role=split_role,
                item_count=entry[count_key],
                attempts_per_item=(
                    MTROUTER_TRAJECTORIES_PER_EXAMPLE * episode_cap
                ),
                registry_sha256=entry[registry_key],
                source_binding_sha256=entry["source_binding_sha256"],
                details={
                    "trajectory_count": (
                        MTROUTER_TRAJECTORIES_PER_EXAMPLE
                    ),
                    "attempt_cap_semantics": (
                        "pre_call_safety_envelope"
                    ),
                },
            )
        add(
            stage=TURN_STAGE,
            kind="mtrouter_learned_dev_candidates",
            method=MTROUTER,
            surface=surface,
            split_role="router_dev",
            item_count=entry["router_dev_count"],
            attempts_per_item=(
                MTROUTER_MAX_LEARNED_DEV_CANDIDATES * episode_cap
            ),
            registry_sha256=entry["dev_registry_sha256"],
            source_binding_sha256=entry["source_binding_sha256"],
            details={
                "online_dev_candidates_total": (
                    MTROUTER_ONLINE_DEV_CANDIDATES
                ),
                "learned_checkpoint_candidates_max": (
                    MTROUTER_MAX_LEARNED_DEV_CANDIDATES
                ),
                "training_candidate_policy": TRAINING_CANDIDATE_POLICY,
                "shared_strong_candidate_reused": 1,
                "new_online_candidate_count": (
                    MTROUTER_MAX_LEARNED_DEV_CANDIDATES
                ),
            },
        )

    for surface in _publication_surfaces(OPENSQUILLA_G0):
        entry = split["surface_splits"][surface]
        add(
            stage=TURN_STAGE,
            kind="opensquilla_extra_dev_floors",
            method=OPENSQUILLA_G0,
            surface=surface,
            split_role="router_dev",
            item_count=entry["router_dev_count"],
            attempts_per_item=(
                (OPENSQUILLA_DEV_FLOORS - 1)
                * EPISODE_ATTEMPT_CAPS[surface]
            ),
            registry_sha256=entry["dev_registry_sha256"],
            source_binding_sha256=entry["source_binding_sha256"],
            details={
                "projection_floors_total": OPENSQUILLA_DEV_FLOORS,
                "shared_floor_zero_reused": 1,
                "new_projection_floor_count": (
                    OPENSQUILLA_DEV_FLOORS - 1
                ),
            },
        )

    for method in METHOD_ORDER:
        for surface in ELIGIBLE_SURFACES:
            if surface not in PUBLICATION_INCLUDED[method]:
                continue
            entry = split["surface_splits"][surface]
            episode_cap = EPISODE_ATTEMPT_CAPS[surface]
            if method == ROUTELLM:
                attempts = episode_cap + 1
            elif method == FRUGALGPT:
                attempts = 2 * episode_cap
            elif method == AUTOMIX:
                attempts = 2 * episode_cap + 1
            else:
                attempts = episode_cap
            add(
                stage=TEST_STAGE,
                kind="official_test_cell",
                method=method,
                surface=surface,
                split_role="official_test",
                item_count=entry["official_test_count"],
                attempts_per_item=attempts,
                registry_sha256=entry["test_registry_sha256"],
                source_binding_sha256=entry["source_binding_sha256"],
                requires_router_artifact=True,
                details={
                    "attempt_cap_semantics": (
                        "pre_call_safety_envelope"
                    ),
                    "state_episode_cap_64_is_protocol_derived": False,
                    "no_partial_resume_or_cross_run_merge": True,
                },
            )
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ExecutionContractError("execution_job_id_duplicate")
    return tuple(jobs)


def _validate_job_caps(
    jobs: Sequence[ExecutionJob],
    *,
    stages: Sequence[StageLimit],
    budget: Mapping[str, Any],
) -> None:
    observed = {
        stage: sum(
            job.provider_attempt_hard_cap
            for job in jobs
            if job.stage == stage
        )
        for stage in STAGE_ORDER
    }
    if observed != EXPECTED_STAGE_ATTEMPT_CAPS:
        raise ExecutionContractError(
            f"execution_job_stage_cap_mismatch:{observed}"
        )
    if sum(observed.values()) != EXPECTED_GLOBAL_ATTEMPT_CAP:
        raise ExecutionContractError("execution_job_global_cap_mismatch")
    if {
        item.stage: item.provider_attempt_hard_cap for item in stages
    } != observed:
        raise ExecutionContractError("execution_stage_plan_job_mismatch")

    official = {
        method: {
            surface: next(
                job.provider_attempt_hard_cap
                for job in jobs
                if job.kind == "official_test_cell"
                and job.method == method
                and job.surface == surface
            )
            for surface in _publication_surfaces(method)
        }
        for method in METHOD_ORDER
    }
    if budget["provider_call_cap_plan"].get(
        "publication_test_attempt_caps"
    ) != official:
        raise ExecutionContractError(
            "execution_official_test_caps_budget_mismatch"
        )


def _build_coverage_cells(
    jobs: Sequence[ExecutionJob],
) -> tuple[dict[str, Any], ...]:
    official_job_ids = {
        (job.method, job.surface): job.job_id
        for job in jobs
        if job.kind == "official_test_cell"
    }
    cells: list[dict[str, Any]] = []
    expected = _expected_coverage_matrix()
    for method in METHOD_ORDER:
        for surface in SURFACE_ORDER:
            status = expected[method][surface]
            if status == INCLUDED:
                label = "REQUIRED"
                job_id = official_job_ids[(method, surface)]
            elif status == COMPUTE_SUBSET:
                label = TABLE_COMPUTE_OMITTED
                job_id = None
            else:
                label = TABLE_NOT_APPLICABLE
                job_id = None
            cells.append(
                {
                    "method": method,
                    "surface": surface,
                    "coverage_status": status,
                    "table_label": label,
                    "official_test_job_id": job_id,
                }
            )
    if (
        sum(item["coverage_status"] == INCLUDED for item in cells) != 16
        or sum(item["coverage_status"] == COMPUTE_SUBSET for item in cells)
        != 4
        or sum(item["coverage_status"] == INELIGIBLE for item in cells) != 5
    ):
        raise ExecutionContractError("execution_coverage_count_mismatch")
    return tuple(cells)


def _expected_coverage_matrix() -> dict[str, dict[str, str]]:
    return {
        method: {
            surface: (
                INELIGIBLE
                if surface == TOOL_SANDBOX
                else (
                    INCLUDED
                    if surface in PUBLICATION_INCLUDED[method]
                    else COMPUTE_SUBSET
                )
            )
            for surface in SURFACE_ORDER
        }
        for method in METHOD_ORDER
    }


def _require_provider_zero(
    payload: Mapping[str, Any],
    label: str,
) -> None:
    if (
        payload.get("provider_calls") != 0
        or payload.get("secret_reads") != 0
        or payload.get("execute") is not False
        or payload.get("api_spend_performed") is not False
    ):
        raise ExecutionContractError(f"{label}_not_provider_zero")


def _require_all_authorizations_false(
    payload: Mapping[str, Any],
    label: str,
) -> None:
    authorizations = payload.get("authorizations")
    if not isinstance(authorizations, Mapping) or any(
        value is not False for value in authorizations.values()
    ):
        raise ExecutionContractError(f"{label}_authorization_invalid")


def _read_mapping(
    path: Path,
    label: str,
) -> tuple[Mapping[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionContractError(f"{label}_unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ExecutionContractError(f"{label}_not_mapping")
    return payload, raw


def _require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ExecutionContractError(f"{field_name}_invalid")
    return value


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
