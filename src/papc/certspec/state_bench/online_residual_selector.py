"""Scorer-blind online residual selector over visible STATE-Bench trajectories.

This module decides, per cheap-trajectory row, whether the online CFRC method
should open a strong residual at a visible frontier ``z_f``. It is the runtime
selector, so it obeys a hard observability boundary: it reads ONLY visible
evidence -- the conversation and its tool calls/results -- through the existing
per-domain visible-evidence verifiers (``state_write_contract`` and
``extract_dialogue_obligations``). It never reads scorer fields
(``task_completion_pass``, ``state_requirements_*``, ``task_requirements_*``,
``*_details``, ``*_gt``, ``*_reasoning``, ``ux_*``) or the task id.

The guarantee is structural, not advisory: :func:`project_visible_payload`
reduces the payload to a whitelist (``conversation``/``messages``) before any
classifier logic runs, and every adapter verifier reads only ``conversation``.
Scoring may COUNT a final trajectory's outcome (evaluation); it must never be
used to SELECT, preserve, or repair a row. This is the STATE-Bench analogue of
the tau2 clean runner's ``z -> H(a) -> Gamma_z/Pi_K -> suffix`` boundary.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from papc.certspec.state_bench.adapter_protocol import (
    DialogueTriage,
    ResidualDomainAdapter,
)
from papc.certspec.state_bench.customer_support import (
    residual_obligations as customer_support_obligations,
)
from papc.certspec.state_bench.shopping_assistant import (
    residual_obligations as shopping_obligations,
)
from papc.certspec.state_bench.travel import residual_obligations as travel_obligations
from papc.certspec.state_write_contract import Obligation, StateWriteContract

SCHEMA = "papc.state_bench_online_residual_selector.v1"

# Keys the selector is allowed to see. Everything else -- including every scorer
# field and the task id -- is dropped before classification.
VISIBLE_PAYLOAD_KEYS = frozenset({"conversation", "messages"})

# Scorer-derived / gold keys that must never reach selector logic. Used by the
# defense-in-depth guard and by the shared observability audit.
FORBIDDEN_SELECTOR_KEYS = frozenset(
    {
        "task_completion_pass",
        "state_requirements_met",
        "task_requirements_met",
        "state_requirements_details",
        "task_requirements_details",
        "state_requirements_gt",
        "state_requirements_reasoning",
        "task_requirements_reasoning",
        "ux_score",
        "ux_communication_quality",
        "ux_friction",
        "ux_intent_alignment",
        "ux_reasoning",
        "ux_situational_awareness",
        "ux_user_control",
    }
)

OnlineCandidateClass = Literal[
    "visible_state_write_residual",
    "committed_change_repair_residual",
    "visible_dialogue_residual",
    "no_visible_residual",
]
OnlineDecision = Literal["residual_candidate", "no_frontier"]

_AUTHORITY_BY_CLASS: dict[str, str] = {
    "visible_state_write_residual": "ell1_safe_state_write_residual",
    "committed_change_repair_residual": "ell2_temporal_or_irreversible_repair",
    "visible_dialogue_residual": "ell0_message_or_protocol_residual",
    "no_visible_residual": "",
}
_ACTIVATION_BY_CLASS: dict[str, str] = {
    "visible_state_write_residual": "state_write_pack",
    "committed_change_repair_residual": "committed_change_repair_targets",
    "visible_dialogue_residual": "message_candidate",
    "no_visible_residual": "",
}
_READ_BEFORE_MESSAGE_DIALOGUE_UNITS = frozenset(
    {
        "dialogue.customer_support.offer_return_refund_for_out_of_stock_exchange",
        "dialogue.shopping.catalog_price_not_promo",
        "dialogue.shopping.backorder_disclosure",
        "dialogue.shopping.current_cart_total_price_drop",
        "dialogue.shopping.product_verification_lookup",
        "dialogue.shopping.quantity_cap_stock_final_quantity",
        "dialogue.shopping.unsupported_save_for_later_lookup",
    }
)

_DOMAIN_ALIASES: dict[str, str] = {
    "shopping": "shopping_assistant",
    "shopping_assistant": "shopping_assistant",
    "customer_support": "customer_support",
    "travel": "travel",
}


class ScorerLeakageError(RuntimeError):
    """Raised when a value that would let the selector see scorer state appears."""


@dataclass(frozen=True)
class OnlineRowDecision:
    row: str
    decision: OnlineDecision
    candidate_class: OnlineCandidateClass
    authority_level: str
    activation_profile: str
    frontier_index: int
    committed_write_count: int
    open_residual_write_count: int
    open_dialogue_unit_count: int
    visible_tool_call_count: int
    state_write_families: tuple[str, ...]
    repair_target_families: tuple[str, ...]
    dialogue_families: tuple[str, ...]
    frontier_tool_call_index: int | None = None
    owes_state_write: bool = False

    @property
    def is_candidate(self) -> bool:
        return self.decision == "residual_candidate"

    def to_public_json(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "decision": self.decision,
            "candidate_class": self.candidate_class,
            "authority_level": self.authority_level,
            "activation_profile": self.activation_profile,
            "frontier_index": self.frontier_index,
            "committed_write_count": self.committed_write_count,
            "open_residual_write_count": self.open_residual_write_count,
            "open_dialogue_unit_count": self.open_dialogue_unit_count,
            "visible_tool_call_count": self.visible_tool_call_count,
            "state_write_families": list(self.state_write_families),
            "repair_target_families": list(self.repair_target_families),
            "dialogue_families": list(self.dialogue_families),
            "frontier_tool_call_index": self.frontier_tool_call_index,
        }


@dataclass(frozen=True)
class RowResidualEvidence:
    """The instance-level material one row decision was derived from.

    :class:`OnlineRowDecision` reduces this material to counts and write-family
    names, which is all the frontier predicate needs. A reached-state contract
    needs the structures themselves -- which obligation, backed by which visible
    witness, over which state units -- so this view hands back exactly what the
    classifier read. It is a read-only projection: nothing here participates in
    the decision, and :func:`classify_row` consumes the same object.
    """

    canonical_domain: str
    contract: StateWriteContract
    triage: DialogueTriage
    open_writes: tuple[Obligation, ...]
    protocol_dialogue_units: tuple[str, ...]
    open_dialogue_units: tuple[str, ...]
    repair_targets: tuple[str, ...]


def project_visible_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a trajectory payload to the keys the selector may read."""

    return {key: payload[key] for key in payload if key in VISIBLE_PAYLOAD_KEYS}


def assert_scorer_blind(view: Mapping[str, Any]) -> None:
    """Fail closed if a scorer-derived key reached the selector view."""

    leaked = sorted(key for key in view if key in FORBIDDEN_SELECTOR_KEYS)
    if leaked:
        raise ScorerLeakageError(
            "selector view contains forbidden scorer-derived keys: " + ", ".join(leaked)
        )


def _row_residual_evidence(
    adapter: ResidualDomainAdapter,
    view: Mapping[str, Any],
    *,
    domain: str,
) -> RowResidualEvidence:
    """Derive the visible residual material shared by the decision and contract."""

    contract = adapter.state_write_contract(view)
    triage = adapter.extract_dialogue_obligations(view)
    canonical_domain = _DOMAIN_ALIASES.get(str(domain).strip()) or str(domain).strip()

    open_writes = tuple(sorted(contract.residual, key=lambda ob: ob.obligation_id))
    protocol_dialogue_units = _adapter_protocol_dialogue_units(adapter, view)
    if not protocol_dialogue_units:
        protocol_dialogue_units = _selector_protocol_dialogue_units(
            canonical_domain=canonical_domain,
            view=view,
        )
    if protocol_dialogue_units:
        open_dialogue_units = tuple(sorted(protocol_dialogue_units))
    else:
        open_dialogue_units = tuple(sorted({*triage.open, *triage.violated}))
    repair_targets = _adapter_repair_targets(adapter, view)
    return RowResidualEvidence(
        canonical_domain=canonical_domain,
        contract=contract,
        triage=triage,
        open_writes=open_writes,
        protocol_dialogue_units=tuple(sorted(protocol_dialogue_units)),
        open_dialogue_units=open_dialogue_units,
        repair_targets=tuple(sorted(repair_targets)),
    )


def row_residual_evidence(
    domain: str,
    payload: Mapping[str, Any],
) -> RowResidualEvidence:
    """Read-only view of the instance-level material behind a row decision.

    Callers that need the obligations themselves rather than their family names
    use this instead of re-deriving them. It runs the same observability guard
    as :func:`classify_row` and never influences a frontier decision.
    """

    view = project_visible_payload(payload)
    assert_scorer_blind(view)
    return _row_residual_evidence(adapter_for_domain(domain), view, domain=domain)


def classify_row(
    domain: str,
    payload: Mapping[str, Any],
    *,
    row_index: int,
) -> OnlineRowDecision:
    """Classify one row using visible evidence only."""

    view = project_visible_payload(payload)
    assert_scorer_blind(view)
    adapter = adapter_for_domain(domain)

    evidence = _row_residual_evidence(adapter, view, domain=domain)
    contract = evidence.contract
    canonical_domain = evidence.canonical_domain
    open_writes = evidence.open_writes
    protocol_dialogue_units = evidence.protocol_dialogue_units
    open_dialogue_units = evidence.open_dialogue_units
    repair_targets = evidence.repair_targets
    committed_write_count = len(contract.closed)
    state_write_families = _state_write_families(open_writes)
    customer_support_terminal_oos_store_credit = (
        canonical_domain == "customer_support"
        and _customer_support_oos_exchange_store_credit_confirmed_by_user(view)
    )

    if (
        canonical_domain == "shopping_assistant"
        and "shopping.cart.price_refresh" in state_write_families
    ):
        candidate_class: OnlineCandidateClass = "visible_state_write_residual"
    elif customer_support_terminal_oos_store_credit and open_writes:
        candidate_class: OnlineCandidateClass = "visible_state_write_residual"
    elif protocol_dialogue_units:
        candidate_class: OnlineCandidateClass = "visible_dialogue_residual"
    elif open_writes:
        candidate_class = "visible_state_write_residual"
    elif repair_targets:
        candidate_class = "committed_change_repair_residual"
    elif open_dialogue_units:
        candidate_class = "visible_dialogue_residual"
    else:
        candidate_class = "no_visible_residual"

    decision: OnlineDecision = (
        "residual_candidate"
        if candidate_class != "no_visible_residual"
        else "no_frontier"
    )

    visible_calls = _visible_tool_calls(view)
    frontier = _adapter_frontier_position(
        adapter,
        view,
        open_dialogue_units=open_dialogue_units,
        candidate_class=candidate_class,
    )
    if frontier is None:
        frontier = _frontier_position_for_decision(
            canonical_domain=canonical_domain,
            view=view,
            write_tool_names=adapter.write_tool_names,
            open_dialogue_units=open_dialogue_units,
            candidate_class=candidate_class,
        )
    frontier_index, frontier_tool_call_index = frontier
    return OnlineRowDecision(
        row=f"row_{row_index:03d}",
        decision=decision,
        candidate_class=candidate_class,
        authority_level=_AUTHORITY_BY_CLASS[candidate_class],
        activation_profile=_activation_profile_for_decision(
            candidate_class,
            open_dialogue_units,
            state_write_families,
        ),
        frontier_index=frontier_index,
        frontier_tool_call_index=frontier_tool_call_index,
        committed_write_count=committed_write_count,
        open_residual_write_count=len(open_writes),
        open_dialogue_unit_count=len(open_dialogue_units),
        visible_tool_call_count=len(visible_calls),
        state_write_families=state_write_families,
        repair_target_families=tuple(sorted(repair_targets)),
        dialogue_families=open_dialogue_units,
        # ``owes_state_write`` is orthogonal to the candidate class: a row can be
        # labelled ``visible_dialogue_residual`` (protocol dialogue chosen first
        # in the class elif chain) while it still owes a state write. Two
        # INDEPENDENT write sources feed this flag and both must count: the
        # ordinary ell1 residual (``open_writes`` = ``contract.residual``) and the
        # ell2 committed-change repair writes (``repair_targets``), which are
        # excluded from the ordinary ell1 residual (they live in
        # ``forbidden_writes``) and so would otherwise be invisible here. This flag
        # is the runtime hard gate the min-authority shadow-loop skip consults so
        # it never drops a real state write of EITHER kind (ordinary or repair).
        owes_state_write=bool(open_writes) or bool(repair_targets),
    )


def _activation_profile_for_decision(
    candidate_class: OnlineCandidateClass,
    dialogue_units: Sequence[str],
    state_write_families: Sequence[str] = (),
) -> str:
    if (
        candidate_class == "visible_dialogue_residual"
        and _READ_BEFORE_MESSAGE_DIALOGUE_UNITS.intersection(dialogue_units)
    ):
        return "read_before_message_candidate"
    if (
        candidate_class == "visible_state_write_residual"
        and "shopping.cart.price_refresh" in set(state_write_families)
    ):
        return "read_before_message_candidate"
    return _ACTIVATION_BY_CLASS[candidate_class]


def classify_domain_rows(
    domain: str,
    payloads: Sequence[Mapping[str, Any]],
) -> list[OnlineRowDecision]:
    return [
        classify_row(domain, payload, row_index=index)
        for index, payload in enumerate(payloads)
    ]


def summarize_decisions(rows: Sequence[OnlineRowDecision]) -> dict[str, Any]:
    class_counts = Counter(row.candidate_class for row in rows)
    authority_counts = Counter(
        row.authority_level for row in rows if row.authority_level
    )
    activation_counts = Counter(
        row.activation_profile for row in rows if row.activation_profile
    )
    return {
        "schema": SCHEMA,
        "row_count": len(rows),
        "residual_candidate_count": sum(1 for row in rows if row.is_candidate),
        "no_frontier_count": sum(1 for row in rows if not row.is_candidate),
        "candidate_class_counts": dict(sorted(class_counts.items())),
        "authority_level_counts": dict(sorted(authority_counts.items())),
        "activation_profile_counts": dict(sorted(activation_counts.items())),
        "predicted_strong_calls": sum(1 for row in rows if row.is_candidate),
    }


def strong_call_candidate_decisions() -> frozenset[str]:
    return frozenset({"residual_candidate"})


def adapter_for_domain(domain: str) -> ResidualDomainAdapter:
    canonical = _DOMAIN_ALIASES.get(str(domain).strip())
    if canonical == "travel":
        return travel_obligations.ADAPTER
    if canonical == "shopping_assistant":
        return shopping_obligations.ADAPTER
    if canonical == "customer_support":
        return customer_support_obligations.ADAPTER
    raise ValueError(f"unknown STATE-Bench domain: {domain!r}")


def _visible_tool_calls(view: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    messages = view.get("conversation")
    if not isinstance(messages, list):
        messages = view.get("messages")
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, Mapping):
            continue
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for call in raw_calls:
            if isinstance(call, Mapping):
                calls.append({"name": str(call.get("name") or "")})
    return calls


def _adapter_repair_targets(
    adapter: ResidualDomainAdapter,
    view: Mapping[str, Any],
) -> frozenset[str]:
    hook = getattr(adapter, "repair_targets", None)
    if hook is None:
        return frozenset()
    targets = hook(view)
    if not isinstance(targets, frozenset):
        return frozenset(str(target) for target in targets)
    return targets


def _adapter_protocol_dialogue_units(
    adapter: ResidualDomainAdapter,
    view: Mapping[str, Any],
) -> frozenset[str]:
    hook = getattr(adapter, "selector_protocol_dialogue_units", None)
    if hook is None:
        return frozenset()
    units = hook(view)
    if units is None:
        return frozenset()
    if isinstance(units, frozenset):
        return units
    return frozenset(str(unit) for unit in units)


def _adapter_frontier_position(
    adapter: ResidualDomainAdapter,
    view: Mapping[str, Any],
    *,
    open_dialogue_units: Sequence[str],
    candidate_class: OnlineCandidateClass,
) -> tuple[int, int | None] | None:
    hook = getattr(adapter, "selector_frontier_position", None)
    if hook is None:
        return None
    frontier = hook(
        view,
        open_dialogue_units=tuple(open_dialogue_units),
        candidate_class=candidate_class,
    )
    if frontier is None:
        return None
    if (
        not isinstance(frontier, tuple)
        or len(frontier) != 2
        or not isinstance(frontier[0], int)
        or (frontier[1] is not None and not isinstance(frontier[1], int))
    ):
        raise TypeError(
            "selector_frontier_position must return tuple[int, int | None] or None"
        )
    return frontier


def _frontier_position(
    view: Mapping[str, Any],
    write_tool_names: frozenset[str],
) -> tuple[int, int | None]:
    messages = view.get("conversation")
    if not isinstance(messages, list):
        messages = view.get("messages")
    if not isinstance(messages, list):
        return 0, None
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call_index, call in enumerate(calls):
            if isinstance(call, Mapping) and _call_is_frontier_mutation(
                call,
                write_tool_names,
            ):
                return index, call_index
    return len(messages), None


def _frontier_position_for_decision(
    *,
    canonical_domain: str,
    view: Mapping[str, Any],
    write_tool_names: frozenset[str],
    open_dialogue_units: Sequence[str],
    candidate_class: OnlineCandidateClass,
) -> tuple[int, int | None]:
    if (
        canonical_domain == "travel"
        and "dialogue.travel.disclose_connection_risk" in set(open_dialogue_units)
    ):
        protocol_frontier = _travel_connection_protocol_frontier(
            view,
            write_tool_names,
        )
        if protocol_frontier is not None:
            return protocol_frontier
    if (
        canonical_domain == "travel"
        and candidate_class == "visible_dialogue_residual"
        and "dialogue.travel.final_confirmation_or_correction"
        in set(open_dialogue_units)
    ):
        return _terminal_frontier(view)
    if (
        canonical_domain == "customer_support"
        and candidate_class == "visible_state_write_residual"
        and _customer_support_oos_exchange_store_credit_confirmed_by_user(view)
    ):
        return _conversation_end_frontier(view)
    if (
        canonical_domain == "customer_support"
        and "dialogue.customer_support.offer_return_refund_for_out_of_stock_exchange"
        in set(open_dialogue_units)
    ):
        protocol_frontier = _customer_support_oos_exchange_protocol_frontier(
            view,
            write_tool_names,
        )
        if protocol_frontier is not None:
            return protocol_frontier
    return _frontier_position(view, write_tool_names)


def _terminal_frontier(view: Mapping[str, Any]) -> tuple[int, None]:
    messages = view.get("conversation")
    if not isinstance(messages, list):
        messages = view.get("messages")
    if not isinstance(messages, list):
        return 0, None
    task_done_index = _task_done_message_index(messages)
    if task_done_index is not None:
        for index in range(task_done_index - 1, -1, -1):
            message = messages[index]
            if (
                isinstance(message, Mapping)
                and str(message.get("role") or "") == "assistant"
            ):
                return index, None
        return task_done_index, None
    return len(messages), None


def _conversation_end_frontier(view: Mapping[str, Any]) -> tuple[int, None]:
    messages = view.get("conversation")
    if not isinstance(messages, list):
        messages = view.get("messages")
    if not isinstance(messages, list):
        return 0, None
    return len(messages), None


def _task_done_message_index(messages: Sequence[Any]) -> int | None:
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role") or "") != "user":
            continue
        if "[TASK_DONE]" in str(message.get("content") or ""):
            return index
    return None


def _selector_protocol_dialogue_units(
    *,
    canonical_domain: str,
    view: Mapping[str, Any],
) -> frozenset[str]:
    if (
        canonical_domain == "customer_support"
        and _customer_support_oos_exchange_store_credit_confirmed_by_user(view)
    ):
        return frozenset()
    if (
        canonical_domain == "customer_support"
        and _customer_support_oos_exchange_protocol_frontier(
            view,
            frozenset(),
        )
    ):
        return frozenset(
            {"dialogue.customer_support.offer_return_refund_for_out_of_stock_exchange"}
        )
    return frozenset()


def _customer_support_oos_exchange_protocol_frontier(
    view: Mapping[str, Any],
    write_tool_names: frozenset[str],
) -> tuple[int, int] | None:
    messages = _visible_messages(view)
    if not messages:
        return None
    prior_user_exchange_request = False
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "user" and "exchange" in content.casefold():
            prior_user_exchange_request = True
            continue
        if role != "assistant" or not prior_user_exchange_request:
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            continue
        tool_cut = len(calls)
        visible_read_calls: list[Mapping[str, Any]] = []
        for call_index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                continue
            if _call_is_frontier_mutation(call, write_tool_names):
                tool_cut = call_index
                break
            visible_read_calls.append(call)
        if tool_cut <= 0:
            continue
        if not _customer_support_calls_show_oos_exchange(visible_read_calls):
            continue
        if not _customer_support_message_discloses_oos_without_return(content):
            continue
        return index, tool_cut
    return None


def _customer_support_oos_exchange_store_credit_confirmed_by_user(
    view: Mapping[str, Any],
) -> bool:
    """Detect a user-confirmed out-of-stock exchange store-credit write.

    This is intentionally narrower than the OOS dialogue protocol gate. A
    policy question such as "can you confirm store credit is allowed?" should
    keep the residual at ell0. We lift to a terminal state-write frontier only
    after a visible OOS exchange preview and a later user utterance that accepts
    or asks the agent to proceed with store credit.
    """

    messages = _visible_messages(view)
    if not messages:
        return False
    oos_preview_seen = False
    confirmed_exchange_seen = False
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, Mapping):
                    continue
                if _customer_support_is_confirmed_oos_exchange(call):
                    confirmed_exchange_seen = True
                if _customer_support_is_oos_exchange_preview(call):
                    oos_preview_seen = True
        if confirmed_exchange_seen:
            return False
        if (
            oos_preview_seen
            and str(message.get("role") or "") == "user"
            and _customer_support_user_accepts_store_credit(message.get("content"))
        ):
            return True
    return False


def _customer_support_is_oos_exchange_preview(call: Mapping[str, Any]) -> bool:
    if str(call.get("name") or "") != "process_exchange":
        return False
    args = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
    if args.get("confirm") is True:
        return False
    result = call.get("result") if isinstance(call.get("result"), Mapping) else {}
    return (
        str(result.get("status") or "").casefold() == "preview"
        and result.get("out_of_stock") is True
        and (
            str(result.get("refund_method") or "").casefold() == "store_credit"
            or "store_credit_amount" in result
        )
    )


def _customer_support_is_confirmed_oos_exchange(call: Mapping[str, Any]) -> bool:
    if str(call.get("name") or "") != "process_exchange":
        return False
    args = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
    result = call.get("result") if isinstance(call.get("result"), Mapping) else {}
    return args.get("confirm") is True or (
        str(result.get("status") or "").casefold() == "store_credit_issued"
        and result.get("out_of_stock") is True
    )


def _customer_support_user_accepts_store_credit(content: Any) -> bool:
    text = str(content or "").casefold()
    if "store credit" not in text:
        return False
    if "original payment" in text or "refund me" in text:
        return False
    proceed_markers = (
        "go ahead",
        "please issue",
        "issue the",
        "process the",
        "proceed with",
        "that works",
        "works for me",
        "do that",
        "go with",
        "i accept",
        "accepted",
    )
    return any(marker in text for marker in proceed_markers)


def _visible_messages(view: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    messages = view.get("conversation")
    if not isinstance(messages, list):
        messages = view.get("messages")
    if not isinstance(messages, list):
        return []
    return [message for message in messages if isinstance(message, Mapping)]


def _customer_support_message_discloses_oos_without_return(content: str) -> bool:
    text = str(content or "").casefold()
    return "out of stock" in text and "return" not in text and "refund" not in text


def _customer_support_calls_show_oos_exchange(
    calls: Sequence[Mapping[str, Any]],
) -> bool:
    has_exchange_policy = False
    has_oos_product = False
    for call in calls:
        name = str(call.get("name") or "")
        args = (
            call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        )
        result = call.get("result")
        if (
            name == "get_policies"
            and str(args.get("topic") or "").casefold() == "exchange"
            and _nested_key_or_text_contains(result, "out_of_stock")
        ):
            has_exchange_policy = True
        if _contains_in_stock_false(result):
            has_oos_product = True
    return has_exchange_policy and has_oos_product


def _nested_key_or_text_contains(value: Any, needle: str) -> bool:
    folded = str(needle).casefold()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if folded in str(key).casefold():
                return True
            if _nested_key_or_text_contains(child, needle):
                return True
        return False
    if isinstance(value, list):
        return any(_nested_key_or_text_contains(child, needle) for child in value)
    return folded in str(value).casefold()


def _contains_in_stock_false(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) == "in_stock" and child is False:
                return True
            if _contains_in_stock_false(child):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_in_stock_false(child) for child in value)
    return False


def _travel_connection_protocol_frontier(
    view: Mapping[str, Any],
    write_tool_names: frozenset[str],
) -> tuple[int, int] | None:
    messages = view.get("conversation")
    if not isinstance(messages, list):
        messages = view.get("messages")
    if not isinstance(messages, list):
        return None
    visible_bookings: dict[str, Mapping[str, Any]] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        tool_cut = len(calls)
        for call_index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                continue
            if _call_is_frontier_mutation(call, write_tool_names):
                tool_cut = call_index
                break
            _accumulate_travel_booking_result(visible_bookings, call)
        if tool_cut > 0 and _has_same_day_onward_pair(visible_bookings):
            return index, tool_cut
    return None


def _accumulate_travel_booking_result(
    visible_bookings: dict[str, Mapping[str, Any]],
    call: Mapping[str, Any],
) -> None:
    result = call.get("result") if isinstance(call.get("result"), Mapping) else {}
    booking_id = str(result.get("booking_id") or "").strip()
    if not booking_id:
        return
    if not {"origin", "destination", "departure_time"} <= set(result):
        return
    visible_bookings[booking_id] = result


def _has_same_day_onward_pair(bookings: Mapping[str, Mapping[str, Any]]) -> bool:
    values = list(bookings.values())
    for first in values:
        for second in values:
            if first is second:
                continue
            if _same_day_onward(first, second):
                return True
    return False


def _same_day_onward(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    first_destination = str(first.get("destination") or "")
    second_origin = str(second.get("origin") or "")
    if not first_destination or first_destination != second_origin:
        return False
    first_departure = str(first.get("departure_time") or "")
    second_departure = str(second.get("departure_time") or "")
    if not first_departure or not second_departure:
        return False
    return (
        first_departure[:10] == second_departure[:10]
        and second_departure > first_departure
    )


def _call_is_frontier_mutation(
    call: Mapping[str, Any],
    write_tool_names: frozenset[str],
) -> bool:
    if str(call.get("name") or "") not in write_tool_names:
        return False
    args = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
    if args.get("confirm") is False:
        return False
    return True


def call_is_state_mutation(
    call: Mapping[str, Any],
    write_tool_names: frozenset[str],
) -> bool:
    """Whether one visible call landed a state effect.

    Read-only alias of the predicate the frontier position already uses, so a
    contract that reports committed effects and the selector that places the
    frontier cannot drift apart on what counts as a write.
    """

    return _call_is_frontier_mutation(call, write_tool_names)


def _state_write_families(open_writes: Sequence[Any]) -> tuple[str, ...]:
    families: set[str] = set()
    for obligation in open_writes:
        for unit in getattr(obligation, "write_footprint", frozenset()):
            families.add(_family(str(unit)))
    return tuple(sorted(families))


def _family(unit: str) -> str:
    return unit.split("#", 1)[0]
