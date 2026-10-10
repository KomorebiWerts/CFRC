"""Strict, provider-zero reduction for PAPC Table 1 result receipts.

This module consumes one normalized public receipt per reported table cell.
It never merges shards or denominators, never drops a baseline win, and never
marks the table paper-ready while a required cell is absent or invalid.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from papc.baselines.router_contract import (
    AUTOMIX,
    CHEAP_MODEL,
    FRUGALGPT,
    MTROUTER,
    OPENSQUILLA_G0,
    REASONING_EFFORT,
    ROUTELLM,
    STRONG_MODEL,
)

CHEAP_FULL = "cheap_full"
STRONG_FULL = "strong_full"
CFRC = "cfrc"

STATE_BENCH = "state_bench"
TAU2_RETAIL = "tau2_retail"
TAU2_AIRLINE = "tau2_airline"
TOOL_SANDBOX = "toolsandbox"
AGENT_DIFF = "agent_diff"

SURFACE_ORDER = (
    STATE_BENCH,
    TAU2_RETAIL,
    TAU2_AIRLINE,
    TOOL_SANDBOX,
    AGENT_DIFF,
)
SURFACE_DENOMINATORS = {
    STATE_BENCH: 150,
    TAU2_RETAIL: 40,
    TAU2_AIRLINE: 20,
    TOOL_SANDBOX: 129,
    AGENT_DIFF: 45,
}

ROW_ORDER = (
    CHEAP_FULL,
    STRONG_FULL,
    ROUTELLM,
    FRUGALGPT,
    AUTOMIX,
    MTROUTER,
    OPENSQUILLA_G0,
    CFRC,
)
EXTERNAL_ROWS = (
    ROUTELLM,
    FRUGALGPT,
    AUTOMIX,
    MTROUTER,
    OPENSQUILLA_G0,
)

REQUIRED = "REQUIRED"
COMPUTE_OMITTED = "C/O"
NOT_APPLICABLE_DATA = "N/A-D"

EXTERNAL_REQUIRED_SURFACES = {
    ROUTELLM: frozenset({STATE_BENCH, TAU2_RETAIL, TAU2_AIRLINE, AGENT_DIFF}),
    FRUGALGPT: frozenset({STATE_BENCH, TAU2_RETAIL, TAU2_AIRLINE, AGENT_DIFF}),
    AUTOMIX: frozenset({STATE_BENCH, TAU2_RETAIL, TAU2_AIRLINE}),
    MTROUTER: frozenset({STATE_BENCH, TAU2_AIRLINE, AGENT_DIFF}),
    OPENSQUILLA_G0: frozenset({STATE_BENCH, TAU2_RETAIL}),
}
EXTERNAL_COMPUTE_OMISSIONS = {
    ROUTELLM: frozenset(),
    FRUGALGPT: frozenset(),
    AUTOMIX: frozenset({AGENT_DIFF}),
    MTROUTER: frozenset({TAU2_RETAIL}),
    OPENSQUILLA_G0: frozenset({TAU2_AIRLINE, AGENT_DIFF}),
}
EXTERNAL_COVERAGE_COUNTS = {
    ROUTELLM: 4,
    FRUGALGPT: 4,
    AUTOMIX: 3,
    MTROUTER: 3,
    OPENSQUILLA_G0: 2,
}

RESULT_RECEIPT_SCHEMA = "papc.table1.result_receipt.v2"
AUTHORITY_MANIFEST_SCHEMA = "papc.table1.authority_manifest.v1"
REDUCTION_SCHEMA = "papc.table1.reduction.v2"
FULL_AGGREGATION_SCOPE = "single_full_official_test"

_ROW_LABELS = {
    CHEAP_FULL: "Cheap full",
    STRONG_FULL: "Strong full",
    ROUTELLM: "RouteLLM-MF (adapt.)",
    FRUGALGPT: "FrugalGPT (2-model)",
    AUTOMIX: "AutoMix-POMDP (adapt.)",
    MTROUTER: "MTRouter (2-model)",
    OPENSQUILLA_G0: "OpenSquilla V4.2 (adapt.)",
    CFRC: r"\cfrc{} (Ours)",
}


def cell_policy(row_id: str, surface: str) -> str:
    if row_id not in ROW_ORDER:
        raise ValueError("table1_row_invalid")
    if surface not in SURFACE_ORDER:
        raise ValueError("table1_surface_invalid")
    if row_id in {CHEAP_FULL, STRONG_FULL, CFRC}:
        return REQUIRED
    if surface == TOOL_SANDBOX:
        return NOT_APPLICABLE_DATA
    if surface in EXTERNAL_REQUIRED_SURFACES[row_id]:
        return REQUIRED
    if surface in EXTERNAL_COMPUTE_OMISSIONS[row_id]:
        return COMPUTE_OMITTED
    raise RuntimeError("table1_external_cell_policy_incomplete")


def _validate_frozen_coverage_matrix() -> None:
    for row_id in EXTERNAL_ROWS:
        required = EXTERNAL_REQUIRED_SURFACES[row_id]
        omitted = EXTERNAL_COMPUTE_OMISSIONS[row_id]
        if required & omitted:
            raise RuntimeError("table1_external_coverage_overlap")
        if required | omitted | {TOOL_SANDBOX} != set(SURFACE_ORDER):
            raise RuntimeError("table1_external_coverage_incomplete")
        if len(required) != EXTERNAL_COVERAGE_COUNTS[row_id]:
            raise RuntimeError("table1_external_coverage_count_drift")


_validate_frozen_coverage_matrix()


@dataclass(frozen=True)
class SurfaceAuthority:
    surface: str
    denominator: int
    replicates_per_task: int
    trial_policy_sha256: str
    test_registry_sha256: str
    official_scorer: str
    official_metric: str
    scorer_sha256: str
    runtime_name: str
    runtime_sha256: str
    fit_registry_sha256: str | None
    dev_registry_sha256: str | None

    def __post_init__(self) -> None:
        if self.surface not in SURFACE_ORDER:
            raise ValueError("table1_authority_surface_invalid")
        if self.denominator != SURFACE_DENOMINATORS[self.surface]:
            raise ValueError("table1_authority_denominator_mismatch")
        if self.replicates_per_task != 1:
            raise ValueError("table1_authority_not_single_replicate")
        for field_name in (
            "trial_policy_sha256",
            "test_registry_sha256",
            "scorer_sha256",
            "runtime_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        for field_name in (
            "official_scorer",
            "official_metric",
            "runtime_name",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        if self.surface == TOOL_SANDBOX:
            if (
                self.fit_registry_sha256 is not None
                or self.dev_registry_sha256 is not None
            ):
                raise ValueError("table1_toolsandbox_fit_dev_authority_forbidden")
        else:
            for field_name in (
                "fit_registry_sha256",
                "dev_registry_sha256",
            ):
                object.__setattr__(
                    self,
                    field_name,
                    _required_sha256(
                        getattr(self, field_name),
                        field_name,
                    ),
                )
            if (
                len(
                    {
                        self.fit_registry_sha256,
                        self.dev_registry_sha256,
                        self.test_registry_sha256,
                    }
                )
                != 3
            ):
                raise ValueError("table1_authority_split_registry_reused")

    def public_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_public_record(cls, record: Any) -> SurfaceAuthority:
        _require_exact_keys(
            record,
            {
                "surface",
                "denominator",
                "replicates_per_task",
                "trial_policy_sha256",
                "test_registry_sha256",
                "official_scorer",
                "official_metric",
                "scorer_sha256",
                "runtime_name",
                "runtime_sha256",
                "fit_registry_sha256",
                "dev_registry_sha256",
            },
            "table1_surface_authority",
        )
        return cls(**dict(record))


@dataclass(frozen=True)
class Table1AuthorityManifest:
    authorities: tuple[SurfaceAuthority, ...]

    def __post_init__(self) -> None:
        authorities = tuple(self.authorities)
        if any(not isinstance(item, SurfaceAuthority) for item in authorities):
            raise TypeError("table1_surface_authority_invalid")
        if tuple(item.surface for item in authorities) != SURFACE_ORDER:
            raise ValueError("table1_authority_surface_order_mismatch")
        if len({item.test_registry_sha256 for item in authorities}) != len(
            SURFACE_ORDER
        ):
            raise ValueError("table1_test_registry_reused_across_surfaces")
        object.__setattr__(self, "authorities", authorities)

    @property
    def sha256(self) -> str:
        return _digest(self.public_record())

    def for_surface(self, surface: str) -> SurfaceAuthority:
        for authority in self.authorities:
            if authority.surface == surface:
                return authority
        raise ValueError("table1_surface_authority_missing")

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": AUTHORITY_MANIFEST_SCHEMA,
            "surfaces": [authority.public_record() for authority in self.authorities],
        }

    @classmethod
    def from_public_record(cls, record: Any) -> Table1AuthorityManifest:
        _require_exact_keys(
            record,
            {"schema", "surfaces"},
            "table1_authority_manifest",
        )
        if record["schema"] != AUTHORITY_MANIFEST_SCHEMA:
            raise ValueError("table1_authority_manifest_schema_mismatch")
        surfaces = record["surfaces"]
        if not isinstance(surfaces, list):
            raise TypeError("table1_authority_surfaces_not_list")
        return cls(
            authorities=tuple(
                SurfaceAuthority.from_public_record(item) for item in surfaces
            )
        )


@dataclass(frozen=True)
class ProviderAPIAccounting:
    usage_complete: bool
    provider_calls: int
    retry_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_neutral_cost_usd: float
    cost_scope: str = (
        "test_time_agent_side_provider_api_excludes_harness_and_adaptation"
    )

    def __post_init__(self) -> None:
        if self.usage_complete is not True:
            raise ValueError("table1_provider_usage_incomplete")
        for field_name in (
            "provider_calls",
            "retry_count",
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"table1_{field_name}_invalid")
        if self.provider_calls <= 0:
            raise ValueError("table1_provider_calls_empty")
        if self.input_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("table1_provider_token_usage_empty")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("table1_cached_tokens_exceed_input")
        cost = _nonnegative_float(
            self.cache_neutral_cost_usd,
            "table1_cache_neutral_cost_usd",
        )
        if cost <= 0.0:
            raise ValueError("table1_provider_cost_empty")
        object.__setattr__(self, "cache_neutral_cost_usd", cost)
        if self.cost_scope != (
            "test_time_agent_side_provider_api_excludes_harness_and_adaptation"
        ):
            raise ValueError("table1_provider_cost_scope_invalid")

    def public_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_public_record(cls, record: Any) -> ProviderAPIAccounting:
        _require_exact_keys(
            record,
            {
                "usage_complete",
                "provider_calls",
                "retry_count",
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cache_neutral_cost_usd",
                "cost_scope",
            },
            "table1_provider_accounting",
        )
        return cls(**dict(record))


@dataclass(frozen=True)
class LocalRouterReport:
    decision_count: int
    latency_seconds_total: float
    hardware: str
    software: str
    measurement_scope: str = "local_router_only_excludes_provider_calls"

    def __post_init__(self) -> None:
        if (
            isinstance(self.decision_count, bool)
            or not isinstance(self.decision_count, int)
            or self.decision_count <= 0
        ):
            raise ValueError("table1_local_router_decision_count_invalid")
        object.__setattr__(
            self,
            "latency_seconds_total",
            _nonnegative_float(
                self.latency_seconds_total,
                "table1_local_router_latency_seconds",
            ),
        )
        for field_name in ("hardware", "software"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        if self.measurement_scope != "local_router_only_excludes_provider_calls":
            raise ValueError("table1_local_router_measurement_scope_invalid")

    @property
    def latency_ms_per_decision(self) -> float:
        return round(
            1000.0 * self.latency_seconds_total / self.decision_count,
            6,
        )

    def public_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_public_record(cls, record: Any) -> LocalRouterReport:
        _require_exact_keys(
            record,
            {
                "decision_count",
                "latency_seconds_total",
                "hardware",
                "software",
                "measurement_scope",
            },
            "table1_local_router",
        )
        return cls(**dict(record))


@dataclass(frozen=True)
class Table1ResultReceipt:
    row_id: str
    surface: str
    result_status: str
    aggregation_scope: str
    attempted_count: int
    scored_count: int
    infra_error_count: int
    official_score: float
    replicates_per_task: int
    trial_policy_sha256: str
    test_registry_sha256: str
    official_scorer: str
    official_metric: str
    scorer_sha256: str
    runtime_name: str
    runtime_sha256: str
    cheap_model: str | None
    cheap_reasoning_effort: str | None
    strong_model: str | None
    strong_reasoning_effort: str | None
    fit_registry_sha256: str | None
    dev_registry_sha256: str | None
    method_artifact_sha256: str | None
    calibration_sha256: str | None
    implementation_bundle_sha256: str
    run_config_sha256: str
    private_raw_case_manifest_sha256: str
    raw_usage_receipt_manifest_sha256: str
    registry_ordered_outcome_cost_manifest_sha256: str
    case_receipt_count: int
    registry_order_matches_test_manifest: bool
    agent_side_provider_api: ProviderAPIAccounting
    local_router: LocalRouterReport | None
    public_safe: bool

    def __post_init__(self) -> None:
        if self.row_id not in ROW_ORDER:
            raise ValueError("table1_receipt_row_invalid")
        if self.surface not in SURFACE_ORDER:
            raise ValueError("table1_receipt_surface_invalid")
        if cell_policy(self.row_id, self.surface) != REQUIRED:
            raise ValueError("table1_receipt_for_nonreported_cell")
        if self.result_status != "completed":
            raise ValueError("table1_result_not_completed")
        if self.aggregation_scope != FULL_AGGREGATION_SCOPE:
            raise ValueError("table1_result_not_single_full_denominator")
        denominator = SURFACE_DENOMINATORS[self.surface]
        if self.attempted_count != denominator or self.scored_count != denominator:
            raise ValueError("table1_result_denominator_mismatch")
        if self.replicates_per_task != 1:
            raise ValueError("table1_result_not_single_replicate")
        if (
            isinstance(self.infra_error_count, bool)
            or not isinstance(self.infra_error_count, int)
            or self.infra_error_count != 0
        ):
            raise ValueError("table1_infra_errors_nonzero")
        object.__setattr__(
            self,
            "official_score",
            _unit_float(self.official_score, "table1_official_score"),
        )
        for field_name in (
            "trial_policy_sha256",
            "test_registry_sha256",
            "scorer_sha256",
            "runtime_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        for field_name in (
            "official_scorer",
            "official_metric",
            "runtime_name",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        _validate_model_plan(self)
        _validate_method_artifact_plan(self)
        for field_name in (
            "implementation_bundle_sha256",
            "run_config_sha256",
            "private_raw_case_manifest_sha256",
            "raw_usage_receipt_manifest_sha256",
            "registry_ordered_outcome_cost_manifest_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        if (
            isinstance(self.case_receipt_count, bool)
            or not isinstance(self.case_receipt_count, int)
            or self.case_receipt_count != denominator
        ):
            raise ValueError("table1_case_receipt_count_mismatch")
        if self.registry_order_matches_test_manifest is not True:
            raise ValueError("table1_case_receipt_order_not_registry_bound")
        if not isinstance(
            self.agent_side_provider_api,
            ProviderAPIAccounting,
        ):
            raise TypeError("table1_provider_accounting_invalid")
        if self.row_id in EXTERNAL_ROWS:
            if not isinstance(self.local_router, LocalRouterReport):
                raise ValueError("table1_external_local_router_report_missing")
            if self.row_id in {ROUTELLM, FRUGALGPT, AUTOMIX} and (
                self.local_router.decision_count != denominator
            ):
                raise ValueError("table1_episode_router_decision_count_mismatch")
            if self.row_id in {MTROUTER, OPENSQUILLA_G0} and (
                self.local_router.decision_count < denominator
            ):
                raise ValueError("table1_turn_router_decision_count_too_small")
        elif self.local_router is not None:
            raise ValueError("table1_nonrouter_local_router_report_present")
        if self.public_safe is not True:
            raise ValueError("table1_result_receipt_not_public_safe")

    @property
    def receipt_sha256(self) -> str:
        return _digest(self.public_record())

    def validate_against(self, authority: SurfaceAuthority) -> None:
        if not isinstance(authority, SurfaceAuthority):
            raise TypeError("table1_surface_authority_invalid")
        expected = {
            "surface": authority.surface,
            "replicates_per_task": authority.replicates_per_task,
            "trial_policy_sha256": authority.trial_policy_sha256,
            "test_registry_sha256": authority.test_registry_sha256,
            "official_scorer": authority.official_scorer,
            "official_metric": authority.official_metric,
            "scorer_sha256": authority.scorer_sha256,
            "runtime_name": authority.runtime_name,
            "runtime_sha256": authority.runtime_sha256,
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"table1_receipt_authority_mismatch:{field_name}")
        if self.row_id in EXTERNAL_ROWS and (
            self.fit_registry_sha256 != authority.fit_registry_sha256
            or self.dev_registry_sha256 != authority.dev_registry_sha256
        ):
            raise ValueError("table1_receipt_fit_dev_registry_mismatch")

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": RESULT_RECEIPT_SCHEMA,
            "row_id": self.row_id,
            "surface": self.surface,
            "result_status": self.result_status,
            "aggregation_scope": self.aggregation_scope,
            "attempted_count": self.attempted_count,
            "scored_count": self.scored_count,
            "infra_error_count": self.infra_error_count,
            "official_score": self.official_score,
            "replicates_per_task": self.replicates_per_task,
            "trial_policy_sha256": self.trial_policy_sha256,
            "test_registry_sha256": self.test_registry_sha256,
            "official_scorer": self.official_scorer,
            "official_metric": self.official_metric,
            "scorer_sha256": self.scorer_sha256,
            "runtime_name": self.runtime_name,
            "runtime_sha256": self.runtime_sha256,
            "cheap_model": self.cheap_model,
            "cheap_reasoning_effort": self.cheap_reasoning_effort,
            "strong_model": self.strong_model,
            "strong_reasoning_effort": self.strong_reasoning_effort,
            "fit_registry_sha256": self.fit_registry_sha256,
            "dev_registry_sha256": self.dev_registry_sha256,
            "method_artifact_sha256": self.method_artifact_sha256,
            "calibration_sha256": self.calibration_sha256,
            "implementation_bundle_sha256": (
                self.implementation_bundle_sha256
            ),
            "run_config_sha256": self.run_config_sha256,
            "private_raw_case_manifest_sha256": (
                self.private_raw_case_manifest_sha256
            ),
            "raw_usage_receipt_manifest_sha256": (
                self.raw_usage_receipt_manifest_sha256
            ),
            "registry_ordered_outcome_cost_manifest_sha256": (
                self.registry_ordered_outcome_cost_manifest_sha256
            ),
            "case_receipt_count": self.case_receipt_count,
            "registry_order_matches_test_manifest": (
                self.registry_order_matches_test_manifest
            ),
            "agent_side_provider_api": (self.agent_side_provider_api.public_record()),
            "local_router": (
                self.local_router.public_record()
                if self.local_router is not None
                else None
            ),
            "public_safe": self.public_safe,
        }

    @classmethod
    def from_public_record(cls, record: Any) -> Table1ResultReceipt:
        _require_exact_keys(
            record,
            {
                "schema",
                "row_id",
                "surface",
                "result_status",
                "aggregation_scope",
                "attempted_count",
                "scored_count",
                "infra_error_count",
                "official_score",
                "replicates_per_task",
                "trial_policy_sha256",
                "test_registry_sha256",
                "official_scorer",
                "official_metric",
                "scorer_sha256",
                "runtime_name",
                "runtime_sha256",
                "cheap_model",
                "cheap_reasoning_effort",
                "strong_model",
                "strong_reasoning_effort",
                "fit_registry_sha256",
                "dev_registry_sha256",
                "method_artifact_sha256",
                "calibration_sha256",
                "implementation_bundle_sha256",
                "run_config_sha256",
                "private_raw_case_manifest_sha256",
                "raw_usage_receipt_manifest_sha256",
                "registry_ordered_outcome_cost_manifest_sha256",
                "case_receipt_count",
                "registry_order_matches_test_manifest",
                "agent_side_provider_api",
                "local_router",
                "public_safe",
            },
            "table1_result_receipt",
        )
        if record["schema"] != RESULT_RECEIPT_SCHEMA:
            raise ValueError("table1_result_receipt_schema_mismatch")
        accounting = ProviderAPIAccounting.from_public_record(
            record["agent_side_provider_api"]
        )
        local_record = record["local_router"]
        local_router = (
            LocalRouterReport.from_public_record(local_record)
            if local_record is not None
            else None
        )
        payload = {
            key: value
            for key, value in dict(record).items()
            if key
            not in {
                "schema",
                "agent_side_provider_api",
                "local_router",
            }
        }
        return cls(
            **payload,
            agent_side_provider_api=accounting,
            local_router=local_router,
        )


@dataclass(frozen=True)
class Table1Reduction:
    authority_manifest: Table1AuthorityManifest
    receipts: tuple[Table1ResultReceipt, ...]
    missing_required_cells: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.authority_manifest,
            Table1AuthorityManifest,
        ):
            raise TypeError("table1_authority_manifest_invalid")
        receipts = tuple(self.receipts)
        if any(not isinstance(receipt, Table1ResultReceipt) for receipt in receipts):
            raise TypeError("table1_result_receipt_invalid")
        keys = tuple((receipt.row_id, receipt.surface) for receipt in receipts)
        if len(keys) != len(set(keys)):
            raise ValueError("table1_duplicate_cell_receipt")
        for receipt in receipts:
            receipt.validate_against(
                self.authority_manifest.for_surface(receipt.surface)
            )
        expected = tuple(
            (row_id, surface)
            for row_id in ROW_ORDER
            for surface in SURFACE_ORDER
            if cell_policy(row_id, surface) == REQUIRED
        )
        calculated_missing = tuple(key for key in expected if key not in set(keys))
        missing = tuple(self.missing_required_cells)
        if missing != calculated_missing:
            raise ValueError("table1_missing_cell_manifest_mismatch")
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(self, "missing_required_cells", missing)

    @property
    def paper_ready(self) -> bool:
        return not self.missing_required_cells

    @property
    def decision(self) -> str:
        return "paper_ready" if self.paper_ready else "blocked_missing_required_cells"

    def to_public_record(self) -> dict[str, Any]:
        row_records = [self._row_record(row_id) for row_id in ROW_ORDER]
        return {
            "schema": REDUCTION_SCHEMA,
            "decision": self.decision,
            "paper_ready": self.paper_ready,
            "provider_calls_during_reduction": 0,
            "secret_reads": 0,
            "authority_manifest_sha256": self.authority_manifest.sha256,
            "required_cell_count": sum(
                cell_policy(row_id, surface) == REQUIRED
                for row_id in ROW_ORDER
                for surface in SURFACE_ORDER
            ),
            "completed_required_cell_count": len(self.receipts),
            "coverage_counts": dict(EXTERNAL_COVERAGE_COUNTS),
            "coverage_order": [4, 4, 3, 3, 2],
            "single_replicate_per_task": True,
            "state_bench_official_five_run_protocol_claimed": False,
            "markers": {
                COMPUTE_OMITTED: "eligible_result_blind_compute_omission",
                NOT_APPLICABLE_DATA: ("no_disjoint_official_train_test_data"),
            },
            "blockers": [
                f"missing_required_cell:{row_id}:{surface}"
                for row_id, surface in self.missing_required_cells
            ],
            "rows": row_records,
            "latex_rows": self.latex_rows(),
            "local_router_reports": self._local_router_records(),
            "baseline_score_wins_over_cfrc": (self._baseline_score_wins_over_cfrc()),
            "baseline_results_retained_without_filtering": True,
            "cross_denominator_merging_permitted": False,
            "local_router_cost_in_provider_api_cost": False,
            "provider_api_cost_scope": (
                "test_time_agent_side_provider_api_excludes_harness_and_adaptation"
            ),
            "every_cell_binds_implementation_and_run_config": True,
            "every_cell_binds_private_raw_case_and_usage_manifests": True,
            "every_cell_binds_registry_ordered_outcomes_and_costs": True,
            "authorizations": {
                "authorizes_paid_run": False,
                "authorizes_result_claim": False,
            },
            "supports_paper_table_population": self.paper_ready,
            "public_safe": True,
        }

    def latex_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for row_id in ROW_ORDER:
            cheap, strong = _latex_model_labels(row_id)
            cells = [self._latex_cell(row_id, surface) for surface in SURFACE_ORDER]
            rows.append(
                {
                    "row_id": row_id,
                    "latex": (
                        f"{_ROW_LABELS[row_id]} & {cheap} & {strong} & "
                        + " & ".join(cells)
                        + r" \\"
                    ),
                }
            )
        return rows

    def _row_record(self, row_id: str) -> dict[str, Any]:
        cheap_model, strong_model = _expected_models(row_id)
        return {
            "row_id": row_id,
            "label": _ROW_LABELS[row_id],
            "cheap_model": cheap_model,
            "strong_model": strong_model,
            "cells": [self._cell_record(row_id, surface) for surface in SURFACE_ORDER],
        }

    def _cell_record(self, row_id: str, surface: str) -> dict[str, Any]:
        policy = cell_policy(row_id, surface)
        receipt = self._find_receipt(row_id, surface)
        if policy != REQUIRED:
            return {
                "surface": surface,
                "status": policy,
                "official_score": None,
                "provider_api_cost_usd": None,
                "receipt_sha256": None,
            }
        if receipt is None:
            return {
                "surface": surface,
                "status": "MISSING",
                "official_score": None,
                "provider_api_cost_usd": None,
                "receipt_sha256": None,
            }
        return {
            "surface": surface,
            "status": "COMPLETED",
            "official_score": receipt.official_score,
            "official_score_percent": round(
                100.0 * receipt.official_score,
                6,
            ),
            "provider_api_cost_usd": (
                receipt.agent_side_provider_api.cache_neutral_cost_usd
            ),
            "provider_calls": (receipt.agent_side_provider_api.provider_calls),
            "receipt_sha256": receipt.receipt_sha256,
        }

    def _latex_cell(self, row_id: str, surface: str) -> str:
        policy = cell_policy(row_id, surface)
        if policy == COMPUTE_OMITTED:
            return r"\omitC"
        if policy == NOT_APPLICABLE_DATA:
            return r"\naD"
        receipt = self._find_receipt(row_id, surface)
        if receipt is None:
            return r"\pendingcell"
        score = f"{100.0 * receipt.official_score:.1f}"
        cost = f"{receipt.agent_side_provider_api.cache_neutral_cost_usd:.2f}"
        return rf"\scorecost{{{score}}}{{\${cost}}}"

    def _find_receipt(
        self,
        row_id: str,
        surface: str,
    ) -> Table1ResultReceipt | None:
        for receipt in self.receipts:
            if receipt.row_id == row_id and receipt.surface == surface:
                return receipt
        return None

    def _local_router_records(self) -> list[dict[str, Any]]:
        return [
            {
                "row_id": receipt.row_id,
                "surface": receipt.surface,
                **receipt.local_router.public_record(),
                "latency_ms_per_decision": (
                    receipt.local_router.latency_ms_per_decision
                ),
            }
            for receipt in self.receipts
            if receipt.local_router is not None
        ]

    def _baseline_score_wins_over_cfrc(self) -> list[dict[str, Any]]:
        if not self.paper_ready:
            return []
        wins: list[dict[str, Any]] = []
        for surface in SURFACE_ORDER:
            cfrc = self._find_receipt(CFRC, surface)
            if cfrc is None:
                return []
            for row_id in ROW_ORDER:
                if row_id == CFRC:
                    continue
                baseline = self._find_receipt(row_id, surface)
                if (
                    baseline is not None
                    and baseline.official_score > cfrc.official_score
                ):
                    wins.append(
                        {
                            "surface": surface,
                            "row_id": row_id,
                            "baseline_score": baseline.official_score,
                            "cfrc_score": cfrc.official_score,
                        }
                    )
        return wins


def load_table1_authority(path: Path) -> Table1AuthorityManifest:
    return Table1AuthorityManifest.from_public_record(_read_json(path))


def load_public_result_receipts(
    paths: Sequence[Path],
) -> tuple[Table1ResultReceipt, ...]:
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise TypeError("table1_result_receipt_paths_invalid")
    return tuple(
        Table1ResultReceipt.from_public_record(_read_json(Path(path))) for path in paths
    )


def reduce_table1(
    *,
    authority_manifest: Table1AuthorityManifest,
    receipts: Sequence[Table1ResultReceipt],
) -> Table1Reduction:
    if not isinstance(authority_manifest, Table1AuthorityManifest):
        raise TypeError("table1_authority_manifest_invalid")
    if isinstance(receipts, (str, bytes)) or not isinstance(
        receipts,
        Sequence,
    ):
        raise TypeError("table1_receipts_invalid")
    resolved = tuple(receipts)
    if any(not isinstance(receipt, Table1ResultReceipt) for receipt in resolved):
        raise TypeError("table1_result_receipt_invalid")
    keys = tuple((receipt.row_id, receipt.surface) for receipt in resolved)
    if len(keys) != len(set(keys)):
        raise ValueError("table1_duplicate_cell_receipt")
    for receipt in resolved:
        receipt.validate_against(authority_manifest.for_surface(receipt.surface))
    expected = tuple(
        (row_id, surface)
        for row_id in ROW_ORDER
        for surface in SURFACE_ORDER
        if cell_policy(row_id, surface) == REQUIRED
    )
    present = set(keys)
    missing = tuple(key for key in expected if key not in present)
    ordered_receipts = tuple(
        next(
            receipt
            for receipt in resolved
            if key
            == (
                receipt.row_id,
                receipt.surface,
            )
        )
        for key in expected
        if key in present
    )
    return Table1Reduction(
        authority_manifest=authority_manifest,
        receipts=ordered_receipts,
        missing_required_cells=missing,
    )


def _validate_model_plan(receipt: Table1ResultReceipt) -> None:
    expected_cheap, expected_strong = _expected_models(receipt.row_id)
    if receipt.cheap_model != expected_cheap or receipt.strong_model != expected_strong:
        raise ValueError("table1_receipt_model_plan_mismatch")
    expected_cheap_effort = REASONING_EFFORT if expected_cheap is not None else None
    expected_strong_effort = REASONING_EFFORT if expected_strong is not None else None
    if (
        receipt.cheap_reasoning_effort != expected_cheap_effort
        or receipt.strong_reasoning_effort != expected_strong_effort
    ):
        raise ValueError("table1_receipt_reasoning_effort_mismatch")


def _validate_method_artifact_plan(receipt: Table1ResultReceipt) -> None:
    artifact_fields = (
        "fit_registry_sha256",
        "dev_registry_sha256",
        "method_artifact_sha256",
        "calibration_sha256",
    )
    if receipt.row_id in EXTERNAL_ROWS:
        for field_name in artifact_fields:
            object.__setattr__(
                receipt,
                field_name,
                _required_sha256(
                    getattr(receipt, field_name),
                    field_name,
                ),
            )
    elif any(
        getattr(receipt, field_name) is not None for field_name in artifact_fields
    ):
        raise ValueError("table1_nonrouter_artifact_binding_present")


def _expected_models(row_id: str) -> tuple[str | None, str | None]:
    if row_id == CHEAP_FULL:
        return CHEAP_MODEL, None
    if row_id == STRONG_FULL:
        return None, STRONG_MODEL
    return CHEAP_MODEL, STRONG_MODEL


def _latex_model_labels(row_id: str) -> tuple[str, str]:
    cheap, strong = _expected_models(row_id)
    return (
        "5.4m" if cheap is not None else "--",
        "5.5" if strong is not None else "--",
    )


def _require_exact_keys(
    record: Any,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label}_not_mapping")
    actual = set(record)
    if actual != expected:
        raise ValueError(f"{label}_keys_mismatch")


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name}_must_be_str")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name}_empty")
    return text


def _required_sha256(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field_name}_invalid")
    return text


def _nonnegative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name}_must_be_numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{field_name}_invalid")
    return resolved


def _unit_float(value: Any, field_name: str) -> float:
    resolved = _nonnegative_float(value, field_name)
    if resolved > 1.0:
        raise ValueError(f"{field_name}_outside_unit_interval")
    return resolved


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
