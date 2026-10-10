"""Provider-zero paired uncertainty for the frozen PAPC Table 1.

The input manifests are private, registry-ordered case records.  The emitted
summary contains only aggregate paired statistics and content hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from papc.baselines.table1_results import (
    AGENT_DIFF,
    CFRC,
    REQUIRED,
    ROW_ORDER,
    STATE_BENCH,
    SURFACE_DENOMINATORS,
    SURFACE_ORDER,
    TAU2_AIRLINE,
    TAU2_RETAIL,
    ProviderAPIAccounting,
    Table1AuthorityManifest,
    Table1ResultReceipt,
    cell_policy,
)

CASE_MANIFEST_SCHEMA = "papc.table1.registry_ordered_outcome_cost_manifest.v1"
PAIRED_STATS_SCHEMA = "papc.table1.paired_stats.v1"
PRIVATE_MANIFEST_SCOPE = "private_registry_ordered_full_official_test"
BINARY_SURFACES = frozenset(
    {STATE_BENCH, TAU2_RETAIL, TAU2_AIRLINE, AGENT_DIFF}
)
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class CaseOutcomeCost:
    """One official test case in frozen registry order."""

    registry_index: int
    case_id_sha256: str
    official_outcome: float
    provider_calls: int
    retry_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_neutral_cost_usd: float
    private_raw_case_receipt_sha256: str
    raw_usage_receipt_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.registry_index, bool)
            or not isinstance(self.registry_index, int)
            or self.registry_index < 0
        ):
            raise ValueError("table1_case_registry_index_invalid")
        for field_name in (
            "case_id_sha256",
            "private_raw_case_receipt_sha256",
            "raw_usage_receipt_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "official_outcome",
            _unit_float(self.official_outcome, "table1_case_official_outcome"),
        )
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
                raise ValueError(f"table1_case_{field_name}_invalid")
        if self.provider_calls <= 0:
            raise ValueError("table1_case_provider_calls_empty")
        if self.input_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("table1_case_provider_tokens_empty")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("table1_case_cached_tokens_exceed_input")
        object.__setattr__(
            self,
            "cache_neutral_cost_usd",
            _positive_float(
                self.cache_neutral_cost_usd,
                "table1_case_cache_neutral_cost_usd",
            ),
        )

    def private_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_private_record(cls, record: Any) -> CaseOutcomeCost:
        _require_exact_keys(
            record,
            {
                "registry_index",
                "case_id_sha256",
                "official_outcome",
                "provider_calls",
                "retry_count",
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cache_neutral_cost_usd",
                "private_raw_case_receipt_sha256",
                "raw_usage_receipt_sha256",
            },
            "table1_case_outcome_cost",
        )
        return cls(**dict(record))


@dataclass(frozen=True)
class RegistryOrderedOutcomeCostManifest:
    """Private per-cell data used only for aggregate paired inference."""

    row_id: str
    surface: str
    test_registry_sha256: str
    trial_policy_sha256: str
    official_metric: str
    cases: tuple[CaseOutcomeCost, ...]
    registry_order_matches_test_manifest: bool
    scope: str = PRIVATE_MANIFEST_SCOPE
    public_safe: bool = False

    def __post_init__(self) -> None:
        if self.row_id not in ROW_ORDER:
            raise ValueError("table1_case_manifest_row_invalid")
        if self.surface not in SURFACE_ORDER:
            raise ValueError("table1_case_manifest_surface_invalid")
        if cell_policy(self.row_id, self.surface) != REQUIRED:
            raise ValueError("table1_case_manifest_for_nonreported_cell")
        object.__setattr__(
            self,
            "test_registry_sha256",
            _required_sha256(
                self.test_registry_sha256,
                "table1_case_manifest_test_registry_sha256",
            ),
        )
        object.__setattr__(
            self,
            "trial_policy_sha256",
            _required_sha256(
                self.trial_policy_sha256,
                "table1_case_manifest_trial_policy_sha256",
            ),
        )
        object.__setattr__(
            self,
            "official_metric",
            _required_text(
                self.official_metric,
                "table1_case_manifest_official_metric",
            ),
        )
        cases = tuple(self.cases)
        if any(not isinstance(case, CaseOutcomeCost) for case in cases):
            raise TypeError("table1_case_manifest_case_invalid")
        denominator = SURFACE_DENOMINATORS[self.surface]
        if len(cases) != denominator:
            raise ValueError("table1_case_manifest_denominator_mismatch")
        if tuple(case.registry_index for case in cases) != tuple(
            range(denominator)
        ):
            raise ValueError("table1_case_manifest_registry_order_invalid")
        if len({case.case_id_sha256 for case in cases}) != denominator:
            raise ValueError("table1_case_manifest_case_id_reused")
        if self.surface in BINARY_SURFACES and any(
            case.official_outcome not in {0.0, 1.0} for case in cases
        ):
            raise ValueError("table1_case_manifest_binary_outcome_invalid")
        if self.registry_order_matches_test_manifest is not True:
            raise ValueError("table1_case_manifest_registry_binding_missing")
        if self.scope != PRIVATE_MANIFEST_SCOPE:
            raise ValueError("table1_case_manifest_scope_invalid")
        if self.public_safe is not False:
            raise ValueError("table1_case_manifest_must_remain_private")
        object.__setattr__(self, "cases", cases)

    @property
    def manifest_sha256(self) -> str:
        return _digest(self.private_record())

    def private_record(self) -> dict[str, Any]:
        return {
            "schema": CASE_MANIFEST_SCHEMA,
            "row_id": self.row_id,
            "surface": self.surface,
            "test_registry_sha256": self.test_registry_sha256,
            "trial_policy_sha256": self.trial_policy_sha256,
            "official_metric": self.official_metric,
            "cases": [case.private_record() for case in self.cases],
            "registry_order_matches_test_manifest": (
                self.registry_order_matches_test_manifest
            ),
            "scope": self.scope,
            "public_safe": self.public_safe,
        }

    def validate_against(self, receipt: Table1ResultReceipt) -> None:
        if not isinstance(receipt, Table1ResultReceipt):
            raise TypeError("table1_result_receipt_invalid")
        expected = {
            "row_id": receipt.row_id,
            "surface": receipt.surface,
            "test_registry_sha256": receipt.test_registry_sha256,
            "trial_policy_sha256": receipt.trial_policy_sha256,
            "official_metric": receipt.official_metric,
            "registry_order_matches_test_manifest": (
                receipt.registry_order_matches_test_manifest
            ),
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(
                    f"table1_case_manifest_receipt_mismatch:{field_name}"
                )
        if (
            self.manifest_sha256
            != receipt.registry_ordered_outcome_cost_manifest_sha256
        ):
            raise ValueError("table1_case_manifest_receipt_hash_mismatch")
        if len(self.cases) != receipt.case_receipt_count:
            raise ValueError("table1_case_manifest_receipt_count_mismatch")
        observed_score = math.fsum(
            case.official_outcome for case in self.cases
        ) / len(self.cases)
        if not math.isclose(
            observed_score,
            receipt.official_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("table1_case_manifest_score_mismatch")
        _validate_accounting_sum(
            cases=self.cases,
            accounting=receipt.agent_side_provider_api,
        )

    @classmethod
    def from_private_record(
        cls,
        record: Any,
    ) -> RegistryOrderedOutcomeCostManifest:
        _require_exact_keys(
            record,
            {
                "schema",
                "row_id",
                "surface",
                "test_registry_sha256",
                "trial_policy_sha256",
                "official_metric",
                "cases",
                "registry_order_matches_test_manifest",
                "scope",
                "public_safe",
            },
            "table1_case_manifest",
        )
        if record["schema"] != CASE_MANIFEST_SCHEMA:
            raise ValueError("table1_case_manifest_schema_mismatch")
        case_records = record["cases"]
        if not isinstance(case_records, list):
            raise TypeError("table1_case_manifest_cases_not_list")
        return cls(
            row_id=record["row_id"],
            surface=record["surface"],
            test_registry_sha256=record["test_registry_sha256"],
            trial_policy_sha256=record["trial_policy_sha256"],
            official_metric=record["official_metric"],
            cases=tuple(
                CaseOutcomeCost.from_private_record(case)
                for case in case_records
            ),
            registry_order_matches_test_manifest=record[
                "registry_order_matches_test_manifest"
            ],
            scope=record["scope"],
            public_safe=record["public_safe"],
        )


@dataclass(frozen=True)
class PairedComparison:
    baseline_row_id: str
    surface: str
    denominator: int
    outcome_type: str
    baseline_score: float
    cfrc_score: float
    score_delta_cfrc_minus_baseline: float
    score_delta_ci_low: float
    score_delta_ci_high: float
    baseline_cost_usd: float
    cfrc_cost_usd: float
    cost_saving_usd_baseline_minus_cfrc: float
    cost_saving_ci_low: float
    cost_saving_ci_high: float
    cfrc_case_wins: int | None
    baseline_case_wins: int | None
    case_ties: int | None
    exact_mcnemar_two_sided_p: float | None
    bootstrap_replicates: int
    confidence_level: float
    deterministic_seed_sha256: str
    baseline_manifest_sha256: str
    cfrc_manifest_sha256: str

    def public_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Table1PairedStats:
    authority_manifest_sha256: str
    comparisons: tuple[PairedComparison, ...]
    expected_comparison_count: int
    missing_comparisons: tuple[tuple[str, str], ...]
    bootstrap_replicates: int
    confidence_level: float

    @property
    def paper_ready(self) -> bool:
        return (
            not self.missing_comparisons
            and len(self.comparisons) == self.expected_comparison_count
        )

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": PAIRED_STATS_SCHEMA,
            "decision": (
                "paper_ready_paired_statistics"
                if self.paper_ready
                else "blocked_missing_paired_comparisons"
            ),
            "paper_ready": self.paper_ready,
            "provider_calls_during_analysis": 0,
            "secret_reads": 0,
            "authority_manifest_sha256": self.authority_manifest_sha256,
            "expected_comparison_count": self.expected_comparison_count,
            "completed_comparison_count": len(self.comparisons),
            "bootstrap_replicates": self.bootstrap_replicates,
            "confidence_level": self.confidence_level,
            "comparison_scope": (
                "cfrc_vs_every_reported_non_cfrc_cell_on_identical_test_cases"
            ),
            "binary_test": "exact_two_sided_mcnemar",
            "continuous_test": "paired_nonparametric_bootstrap",
            "multiplicity_adjusted_superiority_claim": False,
            "baseline_results_retained_without_filtering": True,
            "private_case_records_emitted": False,
            "missing_comparisons": [
                {"baseline_row_id": row_id, "surface": surface}
                for row_id, surface in self.missing_comparisons
            ],
            "comparisons": [
                comparison.public_record()
                for comparison in self.comparisons
            ],
            "authorizations": {
                "authorizes_paid_run": False,
                "authorizes_result_claim": False,
            },
            "supports_paper_uncertainty_reporting": self.paper_ready,
            "public_safe": True,
        }


def load_private_case_manifests(
    paths: Sequence[Path],
) -> tuple[RegistryOrderedOutcomeCostManifest, ...]:
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise TypeError("table1_case_manifest_paths_invalid")
    return tuple(
        RegistryOrderedOutcomeCostManifest.from_private_record(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
        for path in paths
    )


def build_table1_paired_stats(
    *,
    authority_manifest: Table1AuthorityManifest,
    receipts: Sequence[Table1ResultReceipt],
    case_manifests: Sequence[RegistryOrderedOutcomeCostManifest],
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Table1PairedStats:
    if not isinstance(authority_manifest, Table1AuthorityManifest):
        raise TypeError("table1_authority_manifest_invalid")
    resolved_receipts = _validated_receipts(
        authority_manifest=authority_manifest,
        receipts=receipts,
    )
    resolved_manifests = _validated_manifests(
        receipts=resolved_receipts,
        manifests=case_manifests,
    )
    replicates = _positive_int(
        bootstrap_replicates,
        "table1_bootstrap_replicates",
    )
    level = _confidence_level(confidence_level)
    receipt_by_key = {
        (receipt.row_id, receipt.surface): receipt
        for receipt in resolved_receipts
    }
    manifest_by_key = {
        (manifest.row_id, manifest.surface): manifest
        for manifest in resolved_manifests
    }
    expected = tuple(
        (row_id, surface)
        for row_id in ROW_ORDER
        if row_id != CFRC
        for surface in SURFACE_ORDER
        if cell_policy(row_id, surface) == REQUIRED
    )
    comparisons: list[PairedComparison] = []
    missing: list[tuple[str, str]] = []
    for baseline_key in expected:
        cfrc_key = (CFRC, baseline_key[1])
        if (
            baseline_key not in receipt_by_key
            or cfrc_key not in receipt_by_key
            or baseline_key not in manifest_by_key
            or cfrc_key not in manifest_by_key
        ):
            missing.append(baseline_key)
            continue
        comparisons.append(
            _paired_comparison(
                baseline_receipt=receipt_by_key[baseline_key],
                cfrc_receipt=receipt_by_key[cfrc_key],
                baseline_manifest=manifest_by_key[baseline_key],
                cfrc_manifest=manifest_by_key[cfrc_key],
                authority_manifest_sha256=authority_manifest.sha256,
                bootstrap_replicates=replicates,
                confidence_level=level,
            )
        )
    return Table1PairedStats(
        authority_manifest_sha256=authority_manifest.sha256,
        comparisons=tuple(comparisons),
        expected_comparison_count=len(expected),
        missing_comparisons=tuple(missing),
        bootstrap_replicates=replicates,
        confidence_level=level,
    )


def exact_mcnemar_two_sided(*, cfrc_wins: int, baseline_wins: int) -> float:
    left = _nonnegative_int(cfrc_wins, "table1_mcnemar_cfrc_wins")
    right = _nonnegative_int(
        baseline_wins,
        "table1_mcnemar_baseline_wins",
    )
    discordant = left + right
    if discordant == 0:
        return 1.0
    tail = min(left, right)
    probability = math.fsum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def _paired_comparison(
    *,
    baseline_receipt: Table1ResultReceipt,
    cfrc_receipt: Table1ResultReceipt,
    baseline_manifest: RegistryOrderedOutcomeCostManifest,
    cfrc_manifest: RegistryOrderedOutcomeCostManifest,
    authority_manifest_sha256: str,
    bootstrap_replicates: int,
    confidence_level: float,
) -> PairedComparison:
    if baseline_receipt.surface != cfrc_receipt.surface:
        raise ValueError("table1_paired_surface_mismatch")
    if cfrc_receipt.row_id != CFRC:
        raise ValueError("table1_paired_cfrc_receipt_invalid")
    if baseline_receipt.row_id == CFRC:
        raise ValueError("table1_paired_baseline_receipt_invalid")
    baseline_cases = baseline_manifest.cases
    cfrc_cases = cfrc_manifest.cases
    if tuple(case.case_id_sha256 for case in baseline_cases) != tuple(
        case.case_id_sha256 for case in cfrc_cases
    ):
        raise ValueError("table1_paired_case_identity_or_order_mismatch")
    score_deltas = tuple(
        cfrc_case.official_outcome - baseline_case.official_outcome
        for baseline_case, cfrc_case in zip(
            baseline_cases,
            cfrc_cases,
            strict=True,
        )
    )
    cost_savings = tuple(
        baseline_case.cache_neutral_cost_usd
        - cfrc_case.cache_neutral_cost_usd
        for baseline_case, cfrc_case in zip(
            baseline_cases,
            cfrc_cases,
            strict=True,
        )
    )
    seed = _digest(
        {
            "schema": PAIRED_STATS_SCHEMA,
            "authority_manifest_sha256": authority_manifest_sha256,
            "baseline_manifest_sha256": baseline_manifest.manifest_sha256,
            "cfrc_manifest_sha256": cfrc_manifest.manifest_sha256,
            "bootstrap_replicates": bootstrap_replicates,
            "confidence_level": confidence_level,
        }
    )
    score_ci, cost_ci = _paired_bootstrap_intervals(
        score_deltas=score_deltas,
        cost_savings=cost_savings,
        seed_sha256=seed,
        replicates=bootstrap_replicates,
        confidence_level=confidence_level,
    )
    binary = baseline_receipt.surface in BINARY_SURFACES
    cfrc_wins = (
        sum(delta > 0.0 for delta in score_deltas) if binary else None
    )
    baseline_wins = (
        sum(delta < 0.0 for delta in score_deltas) if binary else None
    )
    ties = (
        len(score_deltas) - int(cfrc_wins) - int(baseline_wins)
        if binary
        else None
    )
    return PairedComparison(
        baseline_row_id=baseline_receipt.row_id,
        surface=baseline_receipt.surface,
        denominator=len(score_deltas),
        outcome_type="binary" if binary else "continuous",
        baseline_score=baseline_receipt.official_score,
        cfrc_score=cfrc_receipt.official_score,
        score_delta_cfrc_minus_baseline=math.fsum(score_deltas)
        / len(score_deltas),
        score_delta_ci_low=score_ci[0],
        score_delta_ci_high=score_ci[1],
        baseline_cost_usd=(
            baseline_receipt.agent_side_provider_api.cache_neutral_cost_usd
        ),
        cfrc_cost_usd=(
            cfrc_receipt.agent_side_provider_api.cache_neutral_cost_usd
        ),
        cost_saving_usd_baseline_minus_cfrc=math.fsum(cost_savings),
        cost_saving_ci_low=cost_ci[0],
        cost_saving_ci_high=cost_ci[1],
        cfrc_case_wins=cfrc_wins,
        baseline_case_wins=baseline_wins,
        case_ties=ties,
        exact_mcnemar_two_sided_p=(
            exact_mcnemar_two_sided(
                cfrc_wins=int(cfrc_wins),
                baseline_wins=int(baseline_wins),
            )
            if binary
            else None
        ),
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        deterministic_seed_sha256=seed,
        baseline_manifest_sha256=baseline_manifest.manifest_sha256,
        cfrc_manifest_sha256=cfrc_manifest.manifest_sha256,
    )


def _paired_bootstrap_intervals(
    *,
    score_deltas: Sequence[float],
    cost_savings: Sequence[float],
    seed_sha256: str,
    replicates: int,
    confidence_level: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(score_deltas) != len(cost_savings) or not score_deltas:
        raise ValueError("table1_paired_bootstrap_input_mismatch")
    rng = random.Random(int(seed_sha256, 16))
    count = len(score_deltas)
    score_samples: list[float] = []
    cost_samples: list[float] = []
    for _ in range(replicates):
        indices = [rng.randrange(count) for _ in range(count)]
        score_samples.append(
            math.fsum(score_deltas[index] for index in indices) / count
        )
        cost_samples.append(
            math.fsum(cost_savings[index] for index in indices)
        )
    alpha = 1.0 - confidence_level
    return (
        _percentile_interval(score_samples, alpha=alpha),
        _percentile_interval(cost_samples, alpha=alpha),
    )


def _percentile_interval(
    values: Sequence[float],
    *,
    alpha: float,
) -> tuple[float, float]:
    ordered = sorted(values)
    lower = _linear_percentile(ordered, alpha / 2.0)
    upper = _linear_percentile(ordered, 1.0 - alpha / 2.0)
    return (lower, upper)


def _linear_percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("table1_percentile_values_empty")
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _validated_receipts(
    *,
    authority_manifest: Table1AuthorityManifest,
    receipts: Sequence[Table1ResultReceipt],
) -> tuple[Table1ResultReceipt, ...]:
    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise TypeError("table1_receipts_invalid")
    resolved = tuple(receipts)
    if any(not isinstance(receipt, Table1ResultReceipt) for receipt in resolved):
        raise TypeError("table1_result_receipt_invalid")
    keys = tuple((receipt.row_id, receipt.surface) for receipt in resolved)
    if len(keys) != len(set(keys)):
        raise ValueError("table1_duplicate_cell_receipt")
    for receipt in resolved:
        receipt.validate_against(
            authority_manifest.for_surface(receipt.surface)
        )
    return resolved


def _validated_manifests(
    *,
    receipts: Sequence[Table1ResultReceipt],
    manifests: Sequence[RegistryOrderedOutcomeCostManifest],
) -> tuple[RegistryOrderedOutcomeCostManifest, ...]:
    if isinstance(manifests, (str, bytes)) or not isinstance(
        manifests,
        Sequence,
    ):
        raise TypeError("table1_case_manifests_invalid")
    resolved = tuple(manifests)
    if any(
        not isinstance(manifest, RegistryOrderedOutcomeCostManifest)
        for manifest in resolved
    ):
        raise TypeError("table1_case_manifest_invalid")
    keys = tuple((manifest.row_id, manifest.surface) for manifest in resolved)
    if len(keys) != len(set(keys)):
        raise ValueError("table1_duplicate_case_manifest")
    receipt_by_key = {
        (receipt.row_id, receipt.surface): receipt for receipt in receipts
    }
    for manifest in resolved:
        key = (manifest.row_id, manifest.surface)
        if key not in receipt_by_key:
            raise ValueError("table1_case_manifest_without_receipt")
        manifest.validate_against(receipt_by_key[key])
    return resolved


def _validate_accounting_sum(
    *,
    cases: Sequence[CaseOutcomeCost],
    accounting: ProviderAPIAccounting,
) -> None:
    for field_name in (
        "provider_calls",
        "retry_count",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
    ):
        if sum(getattr(case, field_name) for case in cases) != getattr(
            accounting,
            field_name,
        ):
            raise ValueError(
                f"table1_case_manifest_accounting_mismatch:{field_name}"
            )
    observed_cost = math.fsum(
        case.cache_neutral_cost_usd for case in cases
    )
    if not math.isclose(
        observed_cost,
        accounting.cache_neutral_cost_usd,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("table1_case_manifest_accounting_mismatch:cost")


def _require_exact_keys(
    record: Any,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label}_not_mapping")
    if set(record) != expected:
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


def _unit_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name}_must_be_numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{field_name}_invalid")
    return resolved


def _positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name}_must_be_numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{field_name}_invalid")
    return resolved


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name}_invalid")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    resolved = _nonnegative_int(value, field_name)
    if resolved <= 0:
        raise ValueError(f"{field_name}_invalid")
    return resolved


def _confidence_level(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("table1_confidence_level_must_be_numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 < resolved < 1.0:
        raise ValueError("table1_confidence_level_invalid")
    return resolved


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
