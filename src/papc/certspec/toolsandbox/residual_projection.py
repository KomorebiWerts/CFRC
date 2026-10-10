"""ToolSandbox residual projection helpers.

These helpers wrap the shared PAPC state-write kernel for ToolSandbox's small
offline certificate surface: dependency completion is the legal residual effect,
and minefield side effects are projected away or blocked.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from papc.certspec.cfrc_summary import summarize_cfrc_decision
from papc.certspec.state_write_contract import (
    judge_target_action,
    project_admissible_action,
)
from papc.certspec.toolsandbox.state_write_contract import (
    kernel_verdict_json,
    proposed_target_action_from_probe,
    q_precheck_judgment_from_runtime_digest,
    restoration_cost_vector,
    state_write_contract_from_runtime_digest,
)


def project_residual_suffix(
    *,
    digest: Mapping[str, Any],
    proposal: Mapping[str, Any],
    require_full_discharge: bool = True,
) -> dict[str, Any]:
    """Project a ToolSandbox proposal into the admissible residual suffix."""

    contract = state_write_contract_from_runtime_digest(digest)
    action = proposed_target_action_from_probe(proposal)
    q_precheck = q_precheck_judgment_from_runtime_digest(digest, proposal)
    judgment = judge_target_action(contract, action)
    projection = project_admissible_action(
        contract,
        action,
        restoration_cost=restoration_cost_vector(sorted(contract.touched_units | action.effects())),
        require_full_discharge=require_full_discharge,
    )
    # Execute the projected sub-action (Pi_K), matching the VitaBench wrapper.
    # The all-or-nothing mask remains reported as the baseline judgment, but it is
    # not the execution gate once Pi_K has trimmed every forbidden unit.
    execute = (
        not q_precheck.blocks_target
        and projection.makes_progress
        and (projection.fully_discharges_obligation or not require_full_discharge)
    )
    decision = "execute_projected_suffix" if execute else "abstain"
    return {
        "schema": "papc.toolsandbox.residual_projection.v1",
        "surface": "toolsandbox",
        "q_precheck": q_precheck.to_json(),
        "contract_judgment": judgment.to_json(),
        "mask_baseline_blocks": judgment.blocks_target,
        "action_projection": projection.to_json(),
        "cfrc_summary": summarize_cfrc_decision(
            contract=contract,
            action=action,
            projection=projection,
            route_family=decision,
            provider_calls=0,
            b_call_total=1,
            b_call_spent=0,
            surface="toolsandbox",
        ),
        "execute": execute,
        "decision": decision,
        "projection_beats_mask": bool(execute and judgment.blocks_target),
        "provider_calls": 0,
    }


def kernel_receipt(
    *,
    digest: Mapping[str, Any],
    proposal: Mapping[str, Any],
    require_full_discharge: bool = True,
) -> dict[str, Any]:
    """Return the existing ToolSandbox kernel receipt shape."""

    return kernel_verdict_json(
        digest=digest,
        proposal=proposal,
        require_full_discharge=require_full_discharge,
    )


__all__ = ["project_residual_suffix", "kernel_receipt"]
