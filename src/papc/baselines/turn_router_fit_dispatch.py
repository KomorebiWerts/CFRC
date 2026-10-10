"""Fail-closed TURN_STAGE dispatch and offline typed-artifact compilation.

The live seam is deliberately injected: this module owns the frozen execution
plan, registry membership, raw-attempt broker, and append-only job output, but
it has no provider client and never reads a secret.  Native surface collectors
may only return item payloads after making every raw attempt through the
provided broker.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from papc.baselines.external_router_execution import (
    EXECUTION_PREFLIGHT_SCHEMA,
    EXPECTED_SPLIT_COUNTS,
    SHARED_STAGE,
    TURN_STAGE,
    BatchCoordinator,
    ExecutionContractError,
    ExecutionJob,
    ExecutionPlan,
    JobExecutionResult,
    ProviderAttemptBroker,
    StageExecutionAuthorization,
    load_execution_plan,
)
from papc.baselines.mtrouter_upstream import (
    MTROUTER_ALL_STRONG_CONFIG_SHA256,
    TRAJECTORIES_PER_TASK,
    VALUE_LAMBDA,
    MTRouterArtifact,
    MTRouterDevEvaluation,
    MTRouterExplorationTurn,
    MTRouterTrainingExport,
    MTRouterTrainingRow,
    MTRouterTrainingTrajectory,
    build_mtrouter_artifact,
    load_mtrouter_training_export,
    select_trained_mtrouter_dev_candidate,
)
from papc.baselines.opensquilla_upstream import (
    OpenSquillaDevEvaluation,
    OpenSquillaDevProjection,
    OpenSquillaRuntimeArtifact,
    OpenSquillaTrainingExport,
    bind_opensquilla_runtime_artifact,
    load_opensquilla_training_export,
    select_opensquilla_dev_projection,
)
from papc.baselines.router_accounting import CallReceipt
from papc.baselines.router_contract import (
    CHEAP_MODEL,
    STATE_BENCH,
    STRONG_MODEL,
    TAU2_RETAIL,
    SplitManifest,
    assert_disjoint_manifests,
)

FINAL_EXECUTION_GATE_DECISION = (
    "provider_zero_plan_preflight_passed_"
    "artifacts_and_live_authorization_required"
)
TURN_STAGE_JOB_OUTPUT_SCHEMA = "papc.turn_router_fit.job_output.v1"
TURN_STAGE_ITEM_SCHEMA = "papc.turn_router_fit.item.v1"
TURN_STAGE_FAILURE_SCHEMA = "papc.turn_router_fit.job_failure.v1"
MTROUTER_ROULETTE_ITEM_SCHEMA = "papc.mtrouter.roulette_item.v1"
MTROUTER_DEV_ITEM_SCHEMA = "papc.mtrouter.learned_dev_item.v1"
OPENSQUILLA_DEV_ITEM_SCHEMA = "papc.opensquilla.extra_dev_floors_item.v1"
TURN_JOB_KINDS = frozenset(
    {
        "mtrouter_roulette_trajectories",
        "mtrouter_learned_dev_candidates",
        "opensquilla_extra_dev_floors",
    }
)
OPENSQUILLA_OFFICIAL_COUNTS = {
    STATE_BENCH: (240, 60),
    TAU2_RETAIL: (59, 15),
}


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("turn_stage_payload_not_canonical_json") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name}_invalid")
    return value


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name}_must_be_mapping")
    return value


@dataclass(frozen=True)
class FinalTurnStagePlan:
    """A plan whose exact public record was rebound to the current gate."""

    plan: ExecutionPlan
    execution_gate_file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ExecutionPlan):
            raise TypeError("turn_stage_execution_plan_required")
        _require_sha256(
            self.execution_gate_file_sha256,
            "turn_stage_execution_gate_file_sha256",
        )

    @property
    def sha256(self) -> str:
        return self.plan.sha256


def load_final_turn_stage_plan(
    *,
    execution_gate_json: Path,
    split_gate_json: Path,
    budget_gate_json: Path,
    authority_manifest_json: Path,
    private_root: Path,
) -> FinalTurnStagePlan:
    """Rebuild the plan and require equality with the current gate."""

    gate_path = Path(execution_gate_json).resolve()
    raw = gate_path.read_bytes()
    gate = _mapping(json.loads(raw), "turn_stage_execution_gate")
    if gate.get("schema") != EXECUTION_PREFLIGHT_SCHEMA:
        raise ExecutionContractError(
            "turn_stage_execution_gate_schema_not_current"
        )
    if gate.get("decision") != FINAL_EXECUTION_GATE_DECISION:
        raise ExecutionContractError("turn_stage_execution_gate_not_ready")
    if (
        gate.get("provider_calls") != 0
        or gate.get("secret_reads") != 0
        or gate.get("api_spend_performed") is not False
        or gate.get("execute") is not False
    ):
        raise ExecutionContractError(
            "turn_stage_execution_gate_not_provider_zero"
        )
    authorizations = _mapping(
        gate.get("authorizations"),
        "turn_stage_execution_gate_authorizations",
    )
    if not authorizations or any(
        value is not False for value in authorizations.values()
    ):
        raise ExecutionContractError(
            "turn_stage_execution_gate_authorizes_execution"
        )

    plan = load_execution_plan(
        split_gate_json=split_gate_json,
        budget_gate_json=budget_gate_json,
        authority_manifest_json=authority_manifest_json,
        private_root=private_root,
    )
    if gate.get("execution_plan_sha256") != plan.sha256:
        raise ExecutionContractError(
            "turn_stage_execution_gate_plan_sha256_mismatch"
        )
    if gate.get("execution_plan") != plan.public_record():
        raise ExecutionContractError(
            "turn_stage_execution_gate_plan_record_mismatch"
        )
    return FinalTurnStagePlan(
        plan=plan,
        execution_gate_file_sha256=hashlib.sha256(raw).hexdigest(),
    )


@dataclass(frozen=True)
class TurnStageRegistry:
    """Exact ordered registry and source binding for one fit/dev job."""

    surface: str
    manifest: SplitManifest
    source_binding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface:
            raise ValueError("turn_stage_registry_surface_invalid")
        if not isinstance(self.manifest, SplitManifest):
            raise TypeError("turn_stage_split_manifest_required")
        if self.manifest.role not in {"router_fit", "router_dev"}:
            raise ValueError("turn_stage_registry_role_invalid")
        _require_sha256(
            self.source_binding_sha256,
            "turn_stage_source_binding_sha256",
        )

    def bind_job(self, job: ExecutionJob) -> None:
        if (
            job.stage != TURN_STAGE
            or job.kind not in TURN_JOB_KINDS
            or job.surface != self.surface
            or job.split_role != self.manifest.role
            or job.registry_sha256 != self.manifest.registry_sha256
            or job.source_binding_sha256 != self.source_binding_sha256
            or job.item_count != len(self.manifest.example_sha256s)
        ):
            raise ExecutionContractError(
                "turn_stage_registry_job_binding_mismatch"
            )


@dataclass(frozen=True)
class TurnStageItem:
    """One registry member and its private method-specific payload."""

    example_sha256: str
    payload: Mapping[str, Any] = field(repr=False)
    receipts: tuple[CallReceipt, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require_sha256(self.example_sha256, "turn_stage_example_sha256")
        payload = json.loads(_canonical_bytes(_mapping(self.payload, "payload")))
        receipts = tuple(self.receipts)
        if not receipts or any(
            not isinstance(receipt, CallReceipt)
            or receipt.provider_calls != 1
            or receipt.retry_count != 0
            or receipt.provider_reported_cost_usd is None
            for receipt in receipts
        ):
            raise ValueError("turn_stage_item_atomic_receipts_invalid")
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "receipts", receipts)

    @property
    def payload_sha256(self) -> str:
        return _digest(self.payload)

    @property
    def receipts_sha256(self) -> str:
        return _digest(
            [receipt.public_record() for receipt in self.receipts]
        )


@dataclass(frozen=True)
class TurnStageNativeResult:
    """Whole-job result returned by one injected native collector."""

    items: tuple[TurnStageItem, ...]
    infra_failure_count: int = 0

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if not items or any(not isinstance(item, TurnStageItem) for item in items):
            raise ValueError("turn_stage_native_items_invalid")
        if self.infra_failure_count != 0:
            raise ValueError("turn_stage_native_infra_failure")
        object.__setattr__(self, "items", items)


@dataclass(frozen=True)
class TurnStageJobOutput:
    """Canonical full-job output used by offline artifact compilers."""

    plan_sha256: str
    split_gate_sha256: str
    job_id: str
    kind: str
    method: str
    surface: str
    split_role: str
    registry_sha256: str
    source_binding_sha256: str
    items: tuple[TurnStageItem, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "plan_sha256",
            "split_gate_sha256",
            "registry_sha256",
            "source_binding_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.kind not in TURN_JOB_KINDS:
            raise ValueError("turn_stage_output_kind_invalid")
        items = tuple(self.items)
        if not items or any(not isinstance(item, TurnStageItem) for item in items):
            raise ValueError("turn_stage_output_items_invalid")
        object.__setattr__(self, "items", items)

    @classmethod
    def bind(
        cls,
        *,
        final_plan: FinalTurnStagePlan,
        job: ExecutionJob,
        registry: TurnStageRegistry,
        result: TurnStageNativeResult,
    ) -> TurnStageJobOutput:
        registry.bind_job(job)
        observed = tuple(item.example_sha256 for item in result.items)
        expected = registry.manifest.example_sha256s
        if observed != expected:
            raise ExecutionContractError(
                "turn_stage_output_registry_order_or_membership_mismatch"
            )
        return cls(
            plan_sha256=final_plan.sha256,
            split_gate_sha256=job.split_gate_sha256,
            job_id=job.job_id,
            kind=job.kind,
            method=job.method,
            surface=job.surface,
            split_role=job.split_role,
            registry_sha256=job.registry_sha256,
            source_binding_sha256=job.source_binding_sha256,
            items=result.items,
        )

    def validate_binding(
        self,
        *,
        final_plan: FinalTurnStagePlan,
        job: ExecutionJob,
        registry: TurnStageRegistry,
    ) -> None:
        registry.bind_job(job)
        if (
            self.plan_sha256 != final_plan.sha256
            or self.split_gate_sha256 != job.split_gate_sha256
            or self.job_id != job.job_id
            or self.kind != job.kind
            or self.method != job.method
            or self.surface != job.surface
            or self.split_role != job.split_role
            or self.registry_sha256 != job.registry_sha256
            or self.source_binding_sha256 != job.source_binding_sha256
            or tuple(item.example_sha256 for item in self.items)
            != registry.manifest.example_sha256s
        ):
            raise ExecutionContractError(
                "turn_stage_output_job_binding_mismatch"
            )

    @property
    def item_payloads_sha256(self) -> str:
        return _digest(
            [
                {
                    "example_sha256": item.example_sha256,
                    "payload_sha256": item.payload_sha256,
                    "receipts_sha256": item.receipts_sha256,
                }
                for item in self.items
            ]
        )

    @property
    def output_payload_sha256(self) -> str:
        return hashlib.sha256(self.to_jsonl_bytes()).hexdigest()

    def _header(self) -> dict[str, Any]:
        return {
            "schema": TURN_STAGE_JOB_OUTPUT_SCHEMA,
            "record_type": "job_header",
            "plan_sha256": self.plan_sha256,
            "split_gate_sha256": self.split_gate_sha256,
            "job_id": self.job_id,
            "stage": TURN_STAGE,
            "kind": self.kind,
            "method": self.method,
            "surface": self.surface,
            "split_role": self.split_role,
            "item_count": len(self.items),
            "registry_sha256": self.registry_sha256,
            "source_binding_sha256": self.source_binding_sha256,
            "example_membership_sha256": _digest(
                [item.example_sha256 for item in self.items]
            ),
            "item_payloads_sha256": self.item_payloads_sha256,
            "merged_from_prior_runs": False,
        }

    def to_jsonl_bytes(self) -> bytes:
        records: list[dict[str, Any]] = [self._header()]
        records.extend(
            {
                "schema": TURN_STAGE_ITEM_SCHEMA,
                "record_type": "item",
                "ordinal": ordinal,
                "job_id": self.job_id,
                "example_sha256": item.example_sha256,
                "payload_sha256": item.payload_sha256,
                "payload": item.payload,
                "receipts_sha256": item.receipts_sha256,
                "receipts": [
                    receipt.public_record() for receipt in item.receipts
                ],
            }
            for ordinal, item in enumerate(self.items)
        )
        return b"".join(_canonical_bytes(record) + b"\n" for record in records)


class RecordingProviderAttemptBroker(ProviderAttemptBroker):
    """Record the exact receipt objects settled by the shared ledger."""

    def __init__(self, broker: ProviderAttemptBroker) -> None:
        self._broker = broker
        self._settled: list[CallReceipt] = []

    @property
    def settled_receipts(self) -> tuple[CallReceipt, ...]:
        return tuple(self._settled)

    def perform(
        self,
        raw_attempt: Callable[[], CallReceipt],
    ) -> CallReceipt:
        receipt = self._broker.perform(raw_attempt)
        self._settled.append(receipt)
        return receipt


NativeTurnExecutor = Callable[
    [ExecutionJob, RecordingProviderAttemptBroker, TurnStageRegistry],
    TurnStageNativeResult,
]


class TurnStageDispatcher:
    """Execute exact TURN_STAGE jobs with O_EXCL output ownership."""

    def __init__(
        self,
        *,
        final_plan: FinalTurnStagePlan,
        authorization: StageExecutionAuthorization,
        registries: Mapping[tuple[str, str], TurnStageRegistry],
    ) -> None:
        if authorization.stage != TURN_STAGE:
            raise ExecutionContractError(
                "turn_stage_dispatch_authorization_stage_mismatch"
            )
        self.final_plan = final_plan
        self.registries = dict(registries)
        self.coordinator = BatchCoordinator(
            plan=final_plan.plan,
            authorization=authorization,
        )

    def execute_job(
        self,
        job_id: str,
        executor: NativeTurnExecutor,
    ) -> JobExecutionResult:
        try:
            job = self.final_plan.plan.job(job_id)
        except KeyError as exc:
            raise ExecutionContractError(
                "turn_stage_dispatch_job_unknown"
            ) from exc
        if job.stage != TURN_STAGE or job.kind not in TURN_JOB_KINDS:
            raise ExecutionContractError(
                "turn_stage_dispatch_job_outside_scope"
            )
        try:
            registry = self.registries[(job.surface, job.split_role)]
        except KeyError as exc:
            raise ExecutionContractError(
                "turn_stage_dispatch_registry_missing"
            ) from exc
        registry.bind_job(job)

        def run(
            exact_job: ExecutionJob,
            broker: ProviderAttemptBroker,
        ) -> JobExecutionResult:
            exact_job.output_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                exact_job.output_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                recording_broker = RecordingProviderAttemptBroker(broker)
                result = executor(exact_job, recording_broker, registry)
                if not isinstance(result, TurnStageNativeResult):
                    raise TypeError("turn_stage_native_result_required")
                output_receipts = tuple(
                    receipt
                    for item in result.items
                    for receipt in item.receipts
                )
                if output_receipts != recording_broker.settled_receipts:
                    raise ExecutionContractError(
                        "turn_stage_item_receipts_not_exact_ledger_settlements"
                    )
                output = TurnStageJobOutput.bind(
                    final_plan=self.final_plan,
                    job=exact_job,
                    registry=registry,
                    result=result,
                )
                raw = output.to_jsonl_bytes()
                owned_descriptor = descriptor
                descriptor = -1
                _write_descriptor(owned_descriptor, raw)
                return JobExecutionResult(
                    completed_item_count=len(output.items),
                    registry_sha256=output.registry_sha256,
                    output_payload_sha256=hashlib.sha256(raw).hexdigest(),
                )
            except BaseException as exc:
                if descriptor >= 0:
                    failure = {
                        "schema": TURN_STAGE_FAILURE_SCHEMA,
                        "plan_sha256": self.final_plan.sha256,
                        "job_id": exact_job.job_id,
                        "registry_sha256": exact_job.registry_sha256,
                        "source_binding_sha256": (
                            exact_job.source_binding_sha256
                        ),
                        "error_type": type(exc).__name__,
                        "diagnostic_only": True,
                        "no_partial_resume_or_cross_run_merge": True,
                    }
                    owned_descriptor = descriptor
                    descriptor = -1
                    _write_descriptor(
                        owned_descriptor,
                        _canonical_bytes(failure) + b"\n",
                    )
                raise
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

        return self.coordinator.execute_job(job_id, run)


def _write_descriptor(descriptor: int, raw: bytes) -> None:
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def load_turn_stage_job_output(
    path: Path,
    *,
    final_plan: FinalTurnStagePlan,
    job_id: str,
    registry: TurnStageRegistry,
) -> TurnStageJobOutput:
    """Load and fully revalidate one immutable TURN_STAGE JSONL output."""

    records = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(records) < 2 or not all(
        isinstance(record, Mapping) for record in records
    ):
        raise ValueError("turn_stage_output_jsonl_invalid")
    job = final_plan.plan.job(job_id)
    header = records[0]
    items: list[TurnStageItem] = []
    for ordinal, record in enumerate(records[1:]):
        if (
            record.get("schema") != TURN_STAGE_ITEM_SCHEMA
            or record.get("record_type") != "item"
            or record.get("ordinal") != ordinal
            or record.get("job_id") != job_id
        ):
            raise ValueError("turn_stage_output_item_envelope_invalid")
        item = TurnStageItem(
            example_sha256=record.get("example_sha256"),
            payload=_mapping(
                record.get("payload"),
                "turn_stage_output_item_payload",
            ),
            receipts=tuple(
                CallReceipt(**_mapping(receipt, "turn_stage_receipt"))
                for receipt in record.get("receipts", ())
            ),
        )
        if record.get("payload_sha256") != item.payload_sha256:
            raise ValueError("turn_stage_output_item_payload_digest_mismatch")
        if record.get("receipts_sha256") != item.receipts_sha256:
            raise ValueError("turn_stage_output_item_receipts_digest_mismatch")
        items.append(item)
    output = TurnStageJobOutput(
        plan_sha256=header.get("plan_sha256"),
        split_gate_sha256=header.get("split_gate_sha256"),
        job_id=header.get("job_id"),
        kind=header.get("kind"),
        method=header.get("method"),
        surface=header.get("surface"),
        split_role=header.get("split_role"),
        registry_sha256=header.get("registry_sha256"),
        source_binding_sha256=header.get("source_binding_sha256"),
        items=tuple(items),
    )
    output.validate_binding(
        final_plan=final_plan,
        job=job,
        registry=registry,
    )
    if header != output._header():
        raise ValueError("turn_stage_output_header_mismatch")
    return output


class TrainingExportLoader(Protocol):
    def __call__(self, path: Path) -> Any: ...


@dataclass(frozen=True)
class SurfaceArtifactContext:
    """Exact official fit/dev/test and source authority for one surface."""

    final_plan: FinalTurnStagePlan
    surface: str
    fit_manifest: SplitManifest
    dev_manifest: SplitManifest
    test_manifest: SplitManifest
    source_binding_sha256: str

    def __post_init__(self) -> None:
        assert_disjoint_manifests(
            self.fit_manifest,
            self.dev_manifest,
            self.test_manifest,
        )
        _require_sha256(
            self.source_binding_sha256,
            "surface_artifact_source_binding_sha256",
        )
        expected = EXPECTED_SPLIT_COUNTS.get(self.surface)
        if expected is None:
            raise ValueError("surface_artifact_surface_invalid")
        if (
            len(self.fit_manifest.example_sha256s)
            != expected["router_fit_count"]
            or len(self.dev_manifest.example_sha256s)
            != expected["router_dev_count"]
            or len(self.test_manifest.example_sha256s)
            != expected["official_test_count"]
        ):
            raise ValueError("surface_artifact_split_count_mismatch")
        authority = next(
            (
                item
                for item in self.final_plan.plan.authorities
                if item.surface == self.surface
            ),
            None,
        )
        if (
            authority is None
            or authority.fit_registry_sha256
            != self.fit_manifest.registry_sha256
            or authority.dev_registry_sha256
            != self.dev_manifest.registry_sha256
            or authority.test_registry_sha256
            != self.test_manifest.registry_sha256
            or authority.source_binding_sha256
            != self.source_binding_sha256
        ):
            raise ExecutionContractError(
                "surface_artifact_execution_authority_mismatch"
            )

    def registry(self, role: str) -> TurnStageRegistry:
        manifest = (
            self.fit_manifest if role == "router_fit" else self.dev_manifest
        )
        return TurnStageRegistry(
            surface=self.surface,
            manifest=manifest,
            source_binding_sha256=self.source_binding_sha256,
        )

    def turn_job(self, kind: str, split_role: str) -> ExecutionJob:
        matches = tuple(
            job
            for job in self.final_plan.plan.jobs
            if job.stage == TURN_STAGE
            and job.kind == kind
            and job.surface == self.surface
            and job.split_role == split_role
        )
        if len(matches) != 1:
            raise ExecutionContractError(
                "surface_artifact_turn_job_not_unique"
            )
        return matches[0]


@dataclass(frozen=True)
class ReusedStrongDevEvaluation:
    """One all-strong dev result rebound from the shared collection stage."""

    plan_sha256: str
    surface: str
    dev_registry_sha256: str
    source_binding_sha256: str
    source_job_output_sha256: str
    example_sha256: str
    official_quality: float
    provider_api_cost_usd: float
    strong_call_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "plan_sha256",
            "dev_registry_sha256",
            "source_binding_sha256",
            "source_job_output_sha256",
            "example_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        quality = float(self.official_quality)
        cost = float(self.provider_api_cost_usd)
        if not 0.0 <= quality <= 1.0:
            raise ValueError("reused_strong_quality_invalid")
        if cost < 0.0:
            raise ValueError("reused_strong_cost_invalid")
        if (
            isinstance(self.strong_call_count, bool)
            or not isinstance(self.strong_call_count, int)
            or self.strong_call_count <= 0
        ):
            raise ValueError("reused_strong_call_count_invalid")
        object.__setattr__(self, "official_quality", quality)
        object.__setattr__(self, "provider_api_cost_usd", cost)


@dataclass(frozen=True)
class OpenSquillaSharedFitReceiptBinding:
    """Exact shared cheap/strong fit outputs consumed by local adaptation."""

    plan_sha256: str
    surface: str
    fit_registry_sha256: str
    source_binding_sha256: str
    cheap_fit_job_output_sha256: str
    strong_fit_job_output_sha256: str
    cheap_fit_receipts_sha256: str
    strong_fit_receipts_sha256: str
    paired_source_episode_sha256: str
    fit_example_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "plan_sha256",
            "fit_registry_sha256",
            "source_binding_sha256",
            "cheap_fit_job_output_sha256",
            "strong_fit_job_output_sha256",
            "cheap_fit_receipts_sha256",
            "strong_fit_receipts_sha256",
            "paired_source_episode_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        examples = tuple(self.fit_example_sha256s)
        if (
            not examples
            or len(examples) != len(set(examples))
            or any(
                _require_sha256(example, "opensquilla_fit_example_sha256")
                != example
                for example in examples
            )
        ):
            raise ValueError("opensquilla_fit_receipt_examples_invalid")
        if (
            self.cheap_fit_job_output_sha256
            == self.strong_fit_job_output_sha256
        ):
            raise ValueError("opensquilla_fit_endpoint_output_reused")
        object.__setattr__(self, "fit_example_sha256s", examples)

    @classmethod
    def from_shared_job_outputs(
        cls,
        *,
        context: SurfaceArtifactContext,
        cheap_fit_output: Path,
        strong_fit_output: Path,
        paired_source_episode_sha256: str,
    ) -> OpenSquillaSharedFitReceiptBinding:
        """Verify both complete shared fit jobs and bind their exact receipts."""

        from papc.baselines.external_router_fit_dev import (
            read_fit_dev_job_output,
        )

        endpoint_records: dict[str, tuple[dict[str, Any], ...]] = {}
        endpoint_hashes: dict[str, str] = {}
        receipt_hashes: dict[str, str] = {}
        for role, path in (
            ("cheap", cheap_fit_output),
            ("strong", strong_fit_output),
        ):
            job_id = ".".join(
                (
                    SHARED_STAGE,
                    "shared_endpoint_episode",
                    f"shared_{role}_endpoint",
                    context.surface,
                    "router_fit",
                )
            )
            job = context.final_plan.plan.job(job_id)
            raw = Path(path).read_bytes()
            records = read_fit_dev_job_output(path)
            header = records[0]
            if (
                header.get("plan_sha256") != context.final_plan.sha256
                or header.get("job_id") != job.job_id
                or header.get("stage") != SHARED_STAGE
                or header.get("kind") != "shared_endpoint_episode"
                or header.get("method") != f"shared_{role}_endpoint"
                or header.get("surface") != context.surface
                or header.get("split_role") != "router_fit"
                or header.get("item_count")
                != len(context.fit_manifest.example_sha256s)
                or header.get("registry_sha256")
                != context.fit_manifest.registry_sha256
                or header.get("source_binding_sha256")
                != context.source_binding_sha256
                or header.get("split_gate_sha256")
                != job.split_gate_sha256
                or header.get("official_test_outcomes_read") is not False
            ):
                raise ExecutionContractError(
                    "opensquilla_shared_fit_header_mismatch"
                )
            ordered_receipts: list[dict[str, Any]] = []
            for example_sha256, record in zip(
                context.fit_manifest.example_sha256s,
                records[1:],
                strict=True,
            ):
                output = _mapping(
                    record.get("output"),
                    "opensquilla_shared_fit_output",
                )
                raw_receipts = record.get("receipts")
                if (
                    record.get("example_sha256") != example_sha256
                    or output.get("model_role") != role
                    or not isinstance(raw_receipts, list)
                    or not raw_receipts
                ):
                    raise ExecutionContractError(
                        "opensquilla_shared_fit_item_mismatch"
                    )
                for raw_receipt in raw_receipts:
                    receipt = CallReceipt(
                        **_mapping(
                            raw_receipt,
                            "opensquilla_shared_fit_receipt",
                        )
                    )
                    if (
                        receipt.provider_calls != 1
                        or receipt.retry_count != 0
                        or receipt.provider_reported_cost_usd is None
                        or receipt.role != f"{role}_agent"
                        or receipt.model
                        != (
                            CHEAP_MODEL
                            if role == "cheap"
                            else STRONG_MODEL
                        )
                    ):
                        raise ExecutionContractError(
                            "opensquilla_shared_fit_receipt_not_atomic"
                        )
                    ordered_receipts.append(receipt.public_record())
            endpoint_records[role] = records
            endpoint_hashes[role] = hashlib.sha256(raw).hexdigest()
            receipt_hashes[role] = _digest(ordered_receipts)
        if len(endpoint_records) != 2:
            raise AssertionError("opensquilla_endpoint_records_incomplete")
        return cls(
            plan_sha256=context.final_plan.sha256,
            surface=context.surface,
            fit_registry_sha256=context.fit_manifest.registry_sha256,
            source_binding_sha256=context.source_binding_sha256,
            cheap_fit_job_output_sha256=endpoint_hashes["cheap"],
            strong_fit_job_output_sha256=endpoint_hashes["strong"],
            cheap_fit_receipts_sha256=receipt_hashes["cheap"],
            strong_fit_receipts_sha256=receipt_hashes["strong"],
            paired_source_episode_sha256=paired_source_episode_sha256,
            fit_example_sha256s=context.fit_manifest.example_sha256s,
        )

    def validate(
        self,
        *,
        context: SurfaceArtifactContext,
        training_export: OpenSquillaTrainingExport,
    ) -> None:
        if (
            self.plan_sha256 != context.final_plan.sha256
            or self.surface != context.surface
            or self.fit_registry_sha256
            != context.fit_manifest.registry_sha256
            or self.source_binding_sha256
            != context.source_binding_sha256
            or self.fit_example_sha256s
            != context.fit_manifest.example_sha256s
            or self.paired_source_episode_sha256
            != training_export.source_episode_sha256
        ):
            raise ExecutionContractError(
                "opensquilla_shared_fit_receipt_binding_mismatch"
            )


def _validate_reused_strong(
    rows: Sequence[ReusedStrongDevEvaluation],
    *,
    context: SurfaceArtifactContext,
) -> tuple[ReusedStrongDevEvaluation, ...]:
    resolved = tuple(rows)
    if (
        tuple(item.example_sha256 for item in resolved)
        != context.dev_manifest.example_sha256s
        or any(
            item.plan_sha256 != context.final_plan.sha256
            or item.surface != context.surface
            or item.dev_registry_sha256
            != context.dev_manifest.registry_sha256
            or item.source_binding_sha256
            != context.source_binding_sha256
            for item in resolved
        )
    ):
        raise ExecutionContractError(
            "reused_strong_dev_binding_or_membership_mismatch"
        )
    return resolved


def compile_mtrouter_surface_artifact(
    *,
    context: SurfaceArtifactContext,
    fit_roulette: TurnStageJobOutput,
    dev_roulette: TurnStageJobOutput,
    learned_dev: TurnStageJobOutput,
    reused_all_strong: Sequence[ReusedStrongDevEvaluation],
    training_export_json: Path,
    training_export_loader: TrainingExportLoader = (
        load_mtrouter_training_export
    ),
) -> MTRouterArtifact:
    """Compile one learned MTRouter artifact from complete fit/dev receipts."""

    fit_job = context.turn_job(
        "mtrouter_roulette_trajectories",
        "router_fit",
    )
    dev_job = context.turn_job(
        "mtrouter_roulette_trajectories",
        "router_dev",
    )
    candidate_job = context.turn_job(
        "mtrouter_learned_dev_candidates",
        "router_dev",
    )
    fit_roulette.validate_binding(
        final_plan=context.final_plan,
        job=fit_job,
        registry=context.registry("router_fit"),
    )
    for output, job in (
        (dev_roulette, dev_job),
        (learned_dev, candidate_job),
    ):
        output.validate_binding(
            final_plan=context.final_plan,
            job=job,
            registry=context.registry("router_dev"),
        )

    fit_trajectories = _parse_mtrouter_roulette(fit_roulette)
    dev_trajectories = _parse_mtrouter_roulette(dev_roulette)
    training_export = training_export_loader(Path(training_export_json))
    if not isinstance(training_export, MTRouterTrainingExport):
        raise TypeError("mtrouter_training_export_required")
    if (
        training_export.surface != context.surface
        or training_export.fit_registry_sha256
        != context.fit_manifest.registry_sha256
        or training_export.fit_example_sha256s
        != context.fit_manifest.example_sha256s
        or training_export.fit_rows_sha256
        != _digest(
            [trajectory.public_record() for trajectory in fit_trajectories]
        )
    ):
        raise ExecutionContractError(
            "mtrouter_training_export_official_fit_mismatch"
        )

    evaluations = list(
        _parse_mtrouter_learned_dev(
            learned_dev,
            training_export=training_export,
        )
    )
    evaluations.extend(
        MTRouterDevEvaluation(
            example_sha256=row.example_sha256,
            candidate_config_sha256=MTROUTER_ALL_STRONG_CONFIG_SHA256,
            official_quality=row.official_quality,
            provider_api_cost_usd=row.provider_api_cost_usd,
            strong_call_count=row.strong_call_count,
            all_strong=True,
        )
        for row in _validate_reused_strong(
            reused_all_strong,
            context=context,
        )
    )
    selection = select_trained_mtrouter_dev_candidate(
        evaluations,
        dev_manifest=context.dev_manifest,
        training_export=training_export,
    )
    selected_checkpoint: Path | None = None
    if not selection.selected_all_strong:
        selected = next(
            candidate
            for candidate in training_export.candidates
            if candidate.checkpoint_sha256
            == selection.selected_candidate_config_sha256
        )
        selected_checkpoint = Path(selected.checkpoint_path)
    return build_mtrouter_artifact(
        surface=context.surface,
        checkpoint_path=selected_checkpoint,
        fit_manifest=context.fit_manifest,
        dev_manifest=context.dev_manifest,
        test_manifest=context.test_manifest,
        fit_trajectories=fit_trajectories,
        dev_trajectories=dev_trajectories,
        dev_selection=selection,
    )


def compile_opensquilla_surface_artifact(
    *,
    context: SurfaceArtifactContext,
    extra_dev_floors: TurnStageJobOutput,
    reused_floor_zero: Sequence[ReusedStrongDevEvaluation],
    shared_fit_receipts: OpenSquillaSharedFitReceiptBinding,
    training_export_json: Path,
    training_export_loader: TrainingExportLoader = (
        load_opensquilla_training_export
    ),
) -> OpenSquillaRuntimeArtifact:
    """Compile an official fit-adapted OpenSquilla bundle and projection."""

    expected_counts = OPENSQUILLA_OFFICIAL_COUNTS.get(context.surface)
    if expected_counts is None:
        raise ExecutionContractError(
            "opensquilla_surface_outside_publication_scope"
        )
    training_export = training_export_loader(Path(training_export_json))
    if not isinstance(training_export, OpenSquillaTrainingExport):
        raise TypeError("opensquilla_training_export_required")
    if not isinstance(
        shared_fit_receipts,
        OpenSquillaSharedFitReceiptBinding,
    ):
        raise TypeError("opensquilla_shared_fit_receipt_binding_required")
    shared_fit_receipts.validate(
        context=context,
        training_export=training_export,
    )
    fit_count, dev_count = expected_counts
    if (
        len(context.fit_manifest.example_sha256s) != fit_count
        or len(context.dev_manifest.example_sha256s) != dev_count
        or training_export.surface != context.surface
        or training_export.fit_registry_sha256
        != context.fit_manifest.registry_sha256
        or training_export.in_domain_fit_log_count != fit_count
        or training_export.fit_example_sha256s
        != context.fit_manifest.example_sha256s
    ):
        raise ExecutionContractError(
            "opensquilla_official_fit_log_count_or_registry_mismatch"
        )

    job = context.turn_job(
        "opensquilla_extra_dev_floors",
        "router_dev",
    )
    extra_dev_floors.validate_binding(
        final_plan=context.final_plan,
        job=job,
        registry=context.registry("router_dev"),
    )
    evaluations = [
        OpenSquillaDevEvaluation(
            example_sha256=row.example_sha256,
            strong_class_floor=0,
            official_quality=row.official_quality,
            provider_api_cost_usd=row.provider_api_cost_usd,
            strong_call_count=row.strong_call_count,
        )
        for row in _validate_reused_strong(
            reused_floor_zero,
            context=context,
        )
    ]
    evaluations.extend(_parse_opensquilla_extra_dev(extra_dev_floors))
    selection = select_opensquilla_dev_projection(
        tuple(evaluations),
        dev_manifest=context.dev_manifest,
    )
    projection = OpenSquillaDevProjection.freeze_from_dev(
        dev_manifest=context.dev_manifest,
        selection_report=selection,
    )
    return bind_opensquilla_runtime_artifact(
        surface=context.surface,
        training_export=training_export,
        projection=projection,
        fit_manifest=context.fit_manifest,
        dev_manifest=context.dev_manifest,
        test_manifest=context.test_manifest,
    )


def _parse_mtrouter_roulette(
    output: TurnStageJobOutput,
) -> tuple[MTRouterTrainingTrajectory, ...]:
    trajectories: list[MTRouterTrainingTrajectory] = []
    for item in output.items:
        payload = item.payload
        if payload.get("schema") != MTROUTER_ROULETTE_ITEM_SCHEMA:
            raise ValueError("mtrouter_roulette_item_schema_invalid")
        raw_trajectories = payload.get("trajectories")
        if (
            not isinstance(raw_trajectories, list)
            or len(raw_trajectories) != TRAJECTORIES_PER_TASK
        ):
            raise ValueError("mtrouter_roulette_trajectory_count_invalid")
        parsed = tuple(
            _mtrouter_trajectory_from_record(record)
            for record in raw_trajectories
        )
        if any(
            trajectory.example_sha256 != item.example_sha256
            for trajectory in parsed
        ):
            raise ValueError("mtrouter_roulette_item_example_mismatch")
        trajectories.extend(parsed)
    return tuple(trajectories)


def parse_mtrouter_roulette_output(
    output: TurnStageJobOutput,
) -> tuple[MTRouterTrainingTrajectory, ...]:
    """Public validated parser used by the local training live entrypoint."""

    return _parse_mtrouter_roulette(output)


def _mtrouter_trajectory_from_record(
    value: Any,
) -> MTRouterTrainingTrajectory:
    record = _mapping(value, "mtrouter_trajectory")
    raw_rows = record.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("mtrouter_trajectory_rows_invalid")
    rows: list[MTRouterTrainingRow] = []
    for raw_row in raw_rows:
        row = _mapping(raw_row, "mtrouter_trajectory_row")
        if row.get("lambda") != VALUE_LAMBDA:
            raise ValueError("mtrouter_trajectory_lambda_mismatch")
        exploration = MTRouterExplorationTurn(
            example_sha256=row.get("example_sha256"),
            episode_sha256=row.get("episode_sha256"),
            public_input_sha256=row.get("public_input_sha256"),
            turn_index=row.get("turn_index"),
            messages=tuple(row.get("messages", ())),
            selection_draw=row.get("selection_draw"),
            trajectory_seed=row.get("trajectory_seed"),
            selected_role=row.get("selected_role"),
            selected_model=row.get("selected_model"),
            propensity=row.get("propensity"),
            collection_policy=row.get("collection_policy"),
        )
        rows.append(
            MTRouterTrainingRow(
                exploration=exploration,
                terminal_official_reward=row.get(
                    "terminal_official_reward"
                ),
                target_value=row.get("target_value"),
            )
        )
    trajectory = MTRouterTrainingTrajectory(rows=tuple(rows))
    if (
        record.get("episode_sha256") != trajectory.episode_sha256
        or record.get("example_sha256") != trajectory.example_sha256
        or record.get("trajectory_seed") != trajectory.trajectory_seed
    ):
        raise ValueError("mtrouter_trajectory_envelope_mismatch")
    return trajectory


def _parse_mtrouter_learned_dev(
    output: TurnStageJobOutput,
    *,
    training_export: MTRouterTrainingExport,
) -> tuple[MTRouterDevEvaluation, ...]:
    expected_candidates = set(training_export.candidate_config_sha256s)
    evaluations: list[MTRouterDevEvaluation] = []
    for item in output.items:
        payload = item.payload
        if payload.get("schema") != MTROUTER_DEV_ITEM_SCHEMA:
            raise ValueError("mtrouter_dev_item_schema_invalid")
        raw_evaluations = payload.get("evaluations")
        if not isinstance(raw_evaluations, list):
            raise ValueError("mtrouter_dev_item_evaluations_invalid")
        parsed = tuple(
            MTRouterDevEvaluation(**_mapping(record, "mtrouter_dev_row"))
            for record in raw_evaluations
        )
        if (
            {row.candidate_config_sha256 for row in parsed}
            != expected_candidates
            or any(
                row.example_sha256 != item.example_sha256
                or row.all_strong
                for row in parsed
            )
        ):
            raise ValueError(
                "mtrouter_dev_item_candidate_family_or_example_mismatch"
            )
        evaluations.extend(parsed)
    return tuple(evaluations)


def _parse_opensquilla_extra_dev(
    output: TurnStageJobOutput,
) -> tuple[OpenSquillaDevEvaluation, ...]:
    evaluations: list[OpenSquillaDevEvaluation] = []
    for item in output.items:
        payload = item.payload
        if payload.get("schema") != OPENSQUILLA_DEV_ITEM_SCHEMA:
            raise ValueError("opensquilla_dev_item_schema_invalid")
        raw_evaluations = payload.get("evaluations")
        if not isinstance(raw_evaluations, list):
            raise ValueError("opensquilla_dev_item_evaluations_invalid")
        parsed = tuple(
            OpenSquillaDevEvaluation(
                **_mapping(record, "opensquilla_dev_row")
            )
            for record in raw_evaluations
        )
        if (
            tuple(row.strong_class_floor for row in parsed)
            != (1, 2, 3, 4)
            or any(row.example_sha256 != item.example_sha256 for row in parsed)
        ):
            raise ValueError(
                "opensquilla_dev_item_floor_family_or_example_mismatch"
            )
        evaluations.extend(parsed)
    return tuple(evaluations)
