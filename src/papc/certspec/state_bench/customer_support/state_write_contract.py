"""STATE-Bench customer_support projection into shared RCBE contracts.

This module owns the dataset-specific state units for customer_support:
request-derived item scope, anonymized refund-slot footprints, and the
``pi_C`` projection from over-broad item proposals to request-scoped residual
authority. Scripts may load cases and publish artifacts, but they should not
re-implement these adapter rules inline.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from papc.certspec.state_write_contract import (
    Obligation,
    ProposedTargetAction,
    StateWriteContract,
    project_admissible_action,
)

SURFACE_FOOTPRINT = "state_bench_customer_support_footprint_projection"
SURFACE_RESIDUAL = "state_bench_customer_support_cheap_to_strong_residual"
DOMAIN = "customer_support"

READ_TOOL_NAMES = frozenset(
    {
        "get_customer",
        "get_order",
        "get_policies",
        "get_product_details",
        "get_warranty_status",
        "search_products",
    }
)
WRITE_TOOL_NAMES = frozenset(
    {
        "process_return",
        "process_refund",
        "cancel_order",
        "process_exchange",
        "process_warranty_claim",
    }
)
TERMINAL_ITEM_STATES = ("returned", "cancelled", "exchanged")

# Generic product-name tokens that do not distinguish one item from another in
# the same order. Request scope should be driven by nouns such as "headphones",
# not by adjectives like "wireless" or "premium".
GENERIC_NAME_TOKENS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "wireless",
        "premium",
        "pro",
        "plus",
        "max",
        "mini",
        "inch",
        "port",
        "hardcover",
        "paperback",
        "edition",
        "set",
        "pack",
        "kit",
        "new",
        "classic",
        "deluxe",
        "standard",
        "black",
        "white",
        "blue",
        "red",
        "green",
        "large",
        "small",
        "medium",
    }
)


@dataclass(frozen=True)
class RuntimeCase:
    row_index: int
    task_path: Path
    task: Any
    env_data: Any
    order_item_ids: tuple[str, ...]
    request_scope_item_ids: tuple[str, ...]
    gold_item_ids: tuple[str, ...]


def request_scoped_items(
    task: Any, env_data: Any, order_items: Sequence[str]
) -> list[str]:
    """Return order items scoped by observable request text plus catalog only.

    This is the non-circular customer_support scope contract. It never reads
    gold record keys; gold is used later only by scoring audits.
    """

    text = request_text(task)
    product_id_by_item = {i.item_id: i.product_id for i in env_data.order_items}
    name_by_product = {p.product_id: p.name for p in env_data.products}
    category_by_product = {
        p.product_id: str(getattr(p, "category", "")) for p in env_data.products
    }
    scoped: list[str] = []
    for item_id in order_items:
        product_id = product_id_by_item.get(item_id)
        tokens = distinctive_product_tokens(name_by_product.get(product_id, ""))
        category = category_by_product.get(product_id, "").lower()
        if (tokens and any(tok in text for tok in tokens)) or (
            len(category) >= 4 and category in text
        ):
            scoped.append(item_id)
    return scoped


def request_text(task: Any) -> str:
    sim = getattr(task, "user_simulator", {}) or {}
    known = " ".join(sim.get("known_info", [])) if isinstance(sim, Mapping) else ""
    return (str(getattr(task, "opening_message", "") or "") + " " + known).lower()


def distinctive_product_tokens(name: str) -> set[str]:
    return {
        tok
        for tok in re.split(r"[^a-z0-9]+", name.lower())
        if len(tok) >= 3 and not tok.isdigit() and tok not in GENERIC_NAME_TOKENS
    }


def install_state_bench_path(state_bench_repo: Path) -> None:
    path = str(state_bench_repo)
    if path not in sys.path:
        sys.path.insert(0, path)
    expected_package = str(state_bench_repo / "state_bench")
    module = sys.modules.get("state_bench")
    module_file = str(getattr(module, "__file__", "") or "") if module else ""
    module_paths = (
        [str(item) for item in getattr(module, "__path__", [])] if module else []
    )
    if module is not None and not (
        module_file.startswith(expected_package) or expected_package in module_paths
    ):
        for name in [
            name
            for name in sys.modules
            if name == "state_bench" or name.startswith("state_bench.")
        ]:
            del sys.modules[name]


def load_runtime_cases(
    *, state_bench_repo: Path, domain_name: str = DOMAIN, limit: int
) -> list[RuntimeCase]:
    """Load the strict-subset customer_support slice used by G8.205/G8.207."""

    install_state_bench_path(state_bench_repo)
    from state_bench.domain import get_domain_config
    from state_bench.env_loader import load_task_environment
    from state_bench.schemas import TaskDefinition

    domain = get_domain_config(domain_name)
    task_paths = sorted(
        (state_bench_repo / "state_bench" / "domains" / domain_name / "tasks").glob(
            "*.json"
        )
    )
    cases: list[RuntimeCase] = []
    for task_path in task_paths:
        if len(cases) >= limit:
            break
        task = TaskDefinition.load(task_path)
        if not task.state_requirements:
            continue
        env_data, _env_path = load_task_environment(domain, task)
        gold_items = sorted(
            {
                str(req.get("record_key"))
                for req in task.state_requirements
                if str(req.get("record_key", "")).startswith("ITEM-")
            }
        )
        order_ids = {
            item.order_id for item in env_data.order_items if item.item_id in gold_items
        }
        order_items = sorted(
            {
                item.item_id
                for item in env_data.order_items
                if item.order_id in order_ids
            }
        )
        if (
            not gold_items
            or len(order_items) <= len(gold_items)
            or len(order_items) <= 1
        ):
            continue
        request_scope = request_scoped_items(task, env_data, order_items)
        cases.append(
            RuntimeCase(
                row_index=len(cases),
                task_path=task_path,
                task=task,
                env_data=env_data,
                order_item_ids=tuple(order_items),
                request_scope_item_ids=tuple(request_scope or order_items),
                gold_item_ids=tuple(gold_items),
            )
        )
    return cases


def audit_loaded_cases(cases: Sequence[RuntimeCase]) -> dict[str, Any]:
    request_scope_sizes = [len(case.request_scope_item_ids) for case in cases]
    gold_sizes = [len(case.gold_item_ids) for case in cases]
    order_sizes = [len(case.order_item_ids) for case in cases]
    request_missing_gold_rows = sum(
        1
        for case in cases
        if set(case.gold_item_ids) - set(case.request_scope_item_ids)
    )
    request_extra_gold_rows = sum(
        1
        for case in cases
        if set(case.request_scope_item_ids) - set(case.gold_item_ids)
    )
    strict_request_subset_rows = sum(
        1
        for case in cases
        if 0 < len(case.request_scope_item_ids) < len(case.order_item_ids)
    )
    return {
        "row_count": len(cases),
        "request_scope_under_recall_rows_scoring_only": request_missing_gold_rows,
        "request_scope_extra_vs_gold_rows_scoring_only": request_extra_gold_rows,
        "strict_request_subset_rows": strict_request_subset_rows,
        "request_scope_equals_gold_rows_scoring_only": sum(
            1
            for case in cases
            if set(case.request_scope_item_ids) == set(case.gold_item_ids)
        ),
        "request_scope_item_count_sum": sum(request_scope_sizes),
        "gold_item_count_sum_scoring_only": sum(gold_sizes),
        "order_item_count_sum": sum(order_sizes),
    }


def item_slot_units(case: RuntimeCase) -> dict[str, str]:
    return {
        item_id: f"refund.slot_{index}"
        for index, item_id in enumerate(case.order_item_ids)
    }


def project_items(
    *, case: RuntimeCase, proposed_items: Sequence[str]
) -> dict[str, Any]:
    """Project a same-proposal item footprint to request-scoped residual items."""

    units_by_item = item_slot_units(case)
    request_scope = set(case.request_scope_item_ids)
    residual_units = frozenset(units_by_item[item_id] for item_id in request_scope)
    committed_units = frozenset(
        unit for item_id, unit in units_by_item.items() if item_id not in request_scope
    )
    proposed_units = frozenset(
        units_by_item[item_id] for item_id in proposed_items if item_id in units_by_item
    )
    contract = StateWriteContract(
        surface=SURFACE_FOOTPRINT,
        derivation_timing="runtime_progressive",
        closed=frozenset(
            {
                Obligation(
                    obligation_id="closed_outside_request_scope",
                    write_footprint=committed_units,
                    status="closed",
                )
            }
        )
        if committed_units
        else frozenset(),
        residual=frozenset(
            {
                Obligation(
                    obligation_id="request_scoped_items",
                    write_footprint=residual_units,
                    status="open_bounded",
                )
            }
        )
        if residual_units
        else frozenset(),
        forbidden_writes=committed_units,
        legal_actions=frozenset({"bounded_repair", "abstain"}),
        state_change_required=bool(residual_units),
    )
    action = ProposedTargetAction(
        surface=contract.surface,
        action_id="same_proposal_item_footprint",
        write_footprint=proposed_units,
    )
    projection = project_admissible_action(contract, action, collision_budget=0.0)
    return _projection_payload(
        contract=contract,
        units_by_item=units_by_item,
        projection=projection,
        committed_units=committed_units,
        residual_units=residual_units,
        proposed_units=proposed_units,
        extra={},
    )


def project_residual_items(
    *,
    case: RuntimeCase,
    proposed_items: Sequence[str],
    cheap_items: Sequence[str],
) -> dict[str, Any]:
    """Project a residual proposal after cheap-written items have committed."""

    units_by_item = item_slot_units(case)
    request_scope = set(case.request_scope_item_ids)
    cheap_item_set = {item_id for item_id in cheap_items if item_id in units_by_item}
    residual_item_set = request_scope - cheap_item_set
    residual_units = frozenset(units_by_item[item_id] for item_id in residual_item_set)
    cheap_written_units = frozenset(
        units_by_item[item_id] for item_id in cheap_item_set
    )
    outside_request_units = frozenset(
        unit for item_id, unit in units_by_item.items() if item_id not in request_scope
    )
    committed_units = frozenset(outside_request_units | cheap_written_units)
    proposed_units = frozenset(
        units_by_item[item_id] for item_id in proposed_items if item_id in units_by_item
    )
    contract = StateWriteContract(
        surface=SURFACE_RESIDUAL,
        derivation_timing="runtime_progressive",
        closed=frozenset(
            {
                Obligation(
                    obligation_id="formed_state_committed_units",
                    write_footprint=committed_units,
                    status="closed",
                )
            }
        )
        if committed_units
        else frozenset(),
        residual=frozenset(
            {
                Obligation(
                    obligation_id="request_scoped_unwritten_residual_items",
                    write_footprint=residual_units,
                    status="open_bounded",
                )
            }
        )
        if residual_units
        else frozenset(),
        forbidden_writes=committed_units,
        legal_actions=frozenset({"bounded_repair", "abstain"}),
        state_change_required=bool(residual_units),
    )
    action = ProposedTargetAction(
        surface=contract.surface,
        action_id="residual_item_footprint_after_cheap_state",
        write_footprint=proposed_units,
    )
    projection = project_admissible_action(contract, action, collision_budget=0.0)
    return _projection_payload(
        contract=contract,
        units_by_item=units_by_item,
        projection=projection,
        committed_units=committed_units,
        residual_units=residual_units,
        proposed_units=proposed_units,
        extra={
            "cheap_written_units": sorted(cheap_written_units),
            "outside_request_units": sorted(outside_request_units),
        },
    )


def _projection_payload(
    *,
    contract: StateWriteContract,
    units_by_item: Mapping[str, str],
    projection: Any,
    committed_units: frozenset[str],
    residual_units: frozenset[str],
    proposed_units: frozenset[str],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    unit_to_item = {unit: item_id for item_id, unit in units_by_item.items()}
    projected_item_ids = [
        unit_to_item[unit]
        for unit in sorted(projection.projected_footprint)
        if unit in unit_to_item
    ]
    return {
        "contract": contract.to_json(),
        "projection": projection.to_json(),
        "committed_units": sorted(committed_units),
        "residual_units": sorted(residual_units),
        "proposal_writable_units": sorted(proposed_units),
        "projected_units": sorted(projection.projected_footprint),
        "declared_collision_units": sorted(committed_units & proposed_units),
        "projected_item_ids": projected_item_ids,
        **dict(extra),
    }


def execute_tool_calls(
    handlers: Mapping[str, Any],
    tool_calls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for call in tool_calls:
        name = str(call.get("name") or "")
        args = dict(mapping(call.get("arguments")))
        if name not in handlers:
            records.append(
                {
                    "name": name,
                    "arguments": args,
                    "result": {"error": "unavailable_tool"},
                }
            )
            continue
        try:
            result = handlers[name](args)
        except Exception as exc:  # pragma: no cover - domain-specific invalid args.
            result = {"error": type(exc).__name__, "message": str(exc)[:300]}
        records.append({"name": name, "arguments": args, "result": result})
    return records


def infer_return_reason(task: Any) -> str:
    """Infer the return reason from observable request text only."""

    text = request_text(task)
    if any(
        marker in text
        for marker in (
            "wrong item",
            "wrong product",
            "sent the wrong",
            "received the wrong",
        )
    ):
        return "wrong_item"
    if any(
        marker in text
        for marker in (
            "damaged in transit",
            "arrived damaged",
            "shipping damage",
            "damaged during shipping",
        )
    ):
        return "damaged_in_transit"
    if any(
        marker in text
        for marker in (
            "missing item",
            "item is missing",
            "not received",
            "never arrived",
            "didn't receive",
            "did not receive",
        )
    ):
        return "missing"
    if any(
        marker in text
        for marker in (
            "defective",
            "defect",
            "won't turn on",
            "will not turn on",
            "doesn't turn on",
            "does not turn on",
            "not working",
            "doesn't work",
            "does not work",
            "broken",
        )
    ):
        return "defective"
    return "changed_mind"


def return_item(
    *, env: Any, item_id: str, reason: str = "changed_mind"
) -> list[dict[str, Any]]:
    records = execute_tool_calls(
        env.tool_handlers, [{"name": "get_policies", "arguments": {"topic": "return"}}]
    )
    preview = execute_tool_calls(
        env.tool_handlers,
        [
            {
                "name": "process_return",
                "arguments": {"item_id": item_id, "reason": reason},
            }
        ],
    )
    records.extend(preview)
    result = mapping(preview[0].get("result")) if preview else {}
    amount = result.get("refund_amount")
    if amount is None:
        return records
    confirm = execute_tool_calls(
        env.tool_handlers,
        [
            {
                "name": "process_return",
                "arguments": {
                    "item_id": item_id,
                    "reason": reason,
                    "amount": int(amount),
                    "confirm": True,
                },
            }
        ],
    )
    records.extend(confirm)
    return records


def score_after_returns(
    *, case: RuntimeCase, item_ids: Sequence[str]
) -> dict[str, Any]:
    return score_task_after_returns(
        task=case.task, env_data=case.env_data, item_ids=item_ids
    )


def score_sequence_after_returns(
    *,
    case: RuntimeCase,
    cheap_item_ids: Sequence[str],
    residual_item_ids: Sequence[str],
) -> dict[str, Any]:
    seen: set[str] = set()
    item_ids: list[str] = []
    for item_id in [*cheap_item_ids, *residual_item_ids]:
        if item_id in seen:
            continue
        seen.add(item_id)
        item_ids.append(item_id)
    return score_task_after_returns(
        task=case.task, env_data=case.env_data, item_ids=item_ids
    )


def score_task_after_returns(
    *, task: Any, env_data: Any, item_ids: Sequence[str]
) -> dict[str, Any]:
    from state_bench.domain import get_domain_config
    from state_bench.schemas import StateDiff
    from state_bench.scoring import evaluate_state_requirements

    domain = get_domain_config(DOMAIN)
    env = domain.environment_class(env_data.deep_copy(), now=task.now)
    before = env.get_full_snapshot()
    records: list[dict[str, Any]] = []
    reason = infer_return_reason(task)
    for item_id in item_ids:
        records.extend(return_item(env=env, item_id=item_id, reason=reason))
    after = env.get_full_snapshot()
    diff = StateDiff.compute(before, after)
    state_score = evaluate_state_requirements(task, diff)
    details = mapping(getattr(state_score, "details", None))
    return {
        "score": getattr(state_score, "score", None),
        "negative_delta": int(len(sequence(details.get("unexpected_assertions"))) > 0),
        "unexpected_assertion_count": len(
            sequence(details.get("unexpected_assertions"))
        ),
        "missing_assertion_count": len(sequence(details.get("missing_assertions")))
        + len(sequence(details.get("unresolved_match_requirements"))),
        "tool_error_count": sum(
            1 for record in records if is_tool_error(record.get("result"))
        ),
        "changed_state_unit_count": count_state_units(diff.to_dict()),
        "records": records,
        "state_diff": diff.to_dict(),
        "score_details": details,
    }


def find_tool_call(
    records: Sequence[Mapping[str, Any]], *, name: str, confirm: bool
) -> dict[str, Any] | None:
    for record in records:
        if record.get("name") != name:
            continue
        has_confirm = bool(mapping(record.get("arguments")).get("confirm"))
        if has_confirm == confirm:
            return dict(record)
    return None


def preview_refund_amount(result: Any) -> int | None:
    if isinstance(result, Mapping) and result.get("status") == "preview":
        value = result.get("refund_amount")
        if value is not None:
            return int(value)
    return None


def is_tool_error(result: Any) -> bool:
    return isinstance(result, Mapping) and (
        "error" in result or result.get("status") == "error"
    )


def count_state_units(diff: Mapping[str, Any]) -> int:
    count = 0
    for record in mapping(diff.get("created")).values():
        for fields in mapping(record).values():
            count += len(mapping(fields))
    for record in mapping(diff.get("modified")).values():
        for fields in mapping(record).values():
            count += len(mapping(fields))
    for record in mapping(diff.get("deleted")).values():
        count += len(mapping(record))
    return count


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []
