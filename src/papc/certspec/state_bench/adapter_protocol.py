"""Shared adapter protocol for STATE-Bench residual pilots.

This file is STATE-Bench-shared adapter code, not benchmark-independent PAPC
core.  It defines the small surface that per-domain adapters expose to runners:
visible digests, dialogue triage, evidence tokens, precheck hooks, and optional
deterministic seeds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from papc.certspec.state_write_contract import (
    ActionProjection,
    ContractJudgment,
    ProposedTargetAction,
    StateWriteContract,
    idempotency_precheck,
    judge_target_action,
    project_admissible_action,
)


@dataclass(frozen=True)
class ResidualProposal:
    action: str
    assistant_message: str = ""
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    raw_json: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, int | None] = field(default_factory=dict)
    source: str = "adapter_seed"


@dataclass(frozen=True)
class DialogueTriage:
    open: tuple[str, ...] = ()
    satisfied: tuple[str, ...] = ()
    violated: tuple[str, ...] = ()

    @property
    def has_open(self) -> bool:
        return bool(self.open)

    @property
    def has_violated(self) -> bool:
        return bool(self.violated)


@dataclass(frozen=True)
class KernelVerdict:
    """Shared certspec verdict for one residual proposal.

    Domain adapters own the projection from visible trajectory data to
    ``StateWriteContract`` / ``ProposedTargetAction``. Runners should use this
    object instead of re-implementing the state-write mask or projection logic.
    """

    contract: StateWriteContract
    proposed_action: ProposedTargetAction
    idempotency: ContractJudgment
    judgment: ContractJudgment
    projection: ActionProjection
    restoration_cost_mode: str = "uniform_mu"
    dialogue_progress_units: tuple[str, ...] = ()

    @property
    def blocks_target(self) -> bool:
        return self.idempotency.blocks_target or self.judgment.blocks_target

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.state_bench.kernel_verdict.v1",
            "surface": self.contract.surface,
            "restoration_cost_mode": self.restoration_cost_mode,
            "blocks_target": self.blocks_target,
            "idempotency_precheck": self.idempotency.to_json(),
            "contract_judgment": self.judgment.to_json(),
            "action_projection": self.projection.to_json(),
            "state_write_contract": self.contract.to_json(),
            "proposed_target_action": self.proposed_action.to_json(),
            "dialogue_progress_units": list(self.dialogue_progress_units),
            "dialogue_progress_count": len(self.dialogue_progress_units),
            "makes_dialogue_progress": bool(self.dialogue_progress_units),
        }


@runtime_checkable
class ResidualDomainAdapter(Protocol):
    surface: str
    write_tool_names: frozenset[str]
    cost_unit_families: frozenset[str]
    derived_unit_families: frozenset[str]

    def build_obligation_digest(
        self,
        payload: Mapping[str, Any],
        tool_schemas: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def extract_dialogue_obligations(
        self,
        payload: Mapping[str, Any],
    ) -> DialogueTriage: ...

    def dialogue_progress_units(
        self,
        payload: Mapping[str, Any],
        proposal: Any,
    ) -> tuple[str, ...]: ...

    def visible_evidence_tokens(self, payload: Mapping[str, Any]) -> set[str]: ...

    def visible_conversation(
        self, payload: Mapping[str, Any]
    ) -> list[dict[str, Any]]: ...

    def q_precheck_errors(
        self,
        payload: Mapping[str, Any],
        proposal: Any,
    ) -> list[str]: ...

    def q_precheck_judgment(
        self,
        payload: Mapping[str, Any],
        proposal: Any,
    ) -> ContractJudgment: ...

    def deterministic_seeds(
        self,
        payload: Mapping[str, Any],
    ) -> list[ResidualProposal]: ...

    def state_write_contract(
        self,
        payload: Mapping[str, Any],
    ) -> StateWriteContract: ...

    def to_target_action(
        self,
        proposal: Any,
        payload: Mapping[str, Any],
    ) -> ProposedTargetAction: ...

    def restoration_cost_vector(
        self,
        payload: Mapping[str, Any],
        contract: StateWriteContract,
        proposed_action: ProposedTargetAction,
    ) -> Mapping[str, float] | None: ...


def judge_residual_proposal_with_kernel(
    adapter: ResidualDomainAdapter,
    payload: Mapping[str, Any],
    proposal: Any,
    *,
    collision_budget: float = 0.0,
    require_full_discharge: bool = False,
) -> KernelVerdict:
    """Run the fixed STATE-Bench residual verdict chain through certspec.

    The chain is deliberately dataset-shared:
    ``idempotency_precheck -> judge_target_action -> project_admissible_action``.
    Dataset-specific code is limited to producing the contract, target action,
    and optional schema-derived restoration-cost vector.
    """

    contract = adapter.state_write_contract(payload)
    proposed_action = adapter.to_target_action(proposal, payload)
    restoration_cost = adapter.restoration_cost_vector(
        payload, contract, proposed_action
    )
    return KernelVerdict(
        contract=contract,
        proposed_action=proposed_action,
        idempotency=idempotency_precheck(contract, proposed_action),
        judgment=judge_target_action(contract, proposed_action),
        projection=project_admissible_action(
            contract,
            proposed_action,
            restoration_cost=restoration_cost,
            collision_budget=collision_budget,
            require_full_discharge=require_full_discharge,
        ),
        restoration_cost_mode=(
            "uniform_mu" if restoration_cost is None else "schema_restoration_cost"
        ),
        dialogue_progress_units=adapter.dialogue_progress_units(payload, proposal),
    )


def q_precheck_judgment_from_errors(errors: list[str]) -> ContractJudgment:
    """Compile legacy terminal/protocol precheck labels into ContractJudgment."""

    if not errors:
        return ContractJudgment(
            decision="allow",
            reason="q_precheck_pass",
            enforcement_point="q_precheck",
        )
    return ContractJudgment(
        decision="block_q_precheck",
        reason=";".join(sorted(set(str(error) for error in errors))),
        enforcement_point="q_precheck",
    )


def judge_q_precheck_with_adapter(
    adapter: ResidualDomainAdapter,
    payload: Mapping[str, Any],
    proposal: Any,
) -> ContractJudgment:
    """Run the adapter-owned terminal/protocol precheck as a shared judgment."""

    return adapter.q_precheck_judgment(payload, proposal)
