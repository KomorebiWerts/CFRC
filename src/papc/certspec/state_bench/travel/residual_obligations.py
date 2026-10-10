"""Travel dialogue-obligation triage over visible STATE-Bench trajectories."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from papc.certspec.state_bench import request_cues
from papc.certspec.state_bench.adapter_protocol import (
    ContractJudgment,
    ResidualDomainAdapter,
    ResidualProposal,
    q_precheck_judgment_from_errors,
)
from papc.certspec.state_bench.adapter_protocol import (
    DialogueTriage as AdapterDialogueTriage,
)
from papc.certspec.state_write_contract import (
    Obligation,
    ProposedTargetAction,
    StateWriteContract,
)

from . import residual_gates
from .state_write_contract import IRREVERSIBLE_FIELDS
from .state_write_contract import SURFACE as STATE_WRITE_SURFACE
from .state_write_contract import (
    restoration_cost_vector as travel_restoration_cost_vector,
)

DialogueStatus = Literal["open", "satisfied", "violated"]

DIALOGUE_CONNECTION_RISK_UNIT = "dialogue.travel.disclose_connection_risk"
FINAL_CONFIRMATION_OR_CORRECTION_UNIT = (
    "dialogue.travel.final_confirmation_or_correction"
)
WRITE_TOOL_NAMES = frozenset(
    {
        "create_booking",
        "update_booking",
        "cancel_booking",
        "book_hotel",
        "cancel_hotel_reservation",
        "book_car_rental",
        "cancel_car_rental",
    }
)
COST_UNIT_FAMILIES = IRREVERSIBLE_FIELDS
DERIVED_UNIT_FAMILIES = frozenset(
    {DIALOGUE_CONNECTION_RISK_UNIT, FINAL_CONFIRMATION_OR_CORRECTION_UNIT}
)

# Maximal per-tool effect-unit sets for certified tool-surface projection,
# mirroring the fixed branches of ``_travel_call_units``. ``update_booking``
# is deliberately absent: its units depend on the argument fields, so its
# schema can never be dropped. Reads are absent and always kept.
CERTIFIED_PROJECTION_TOOL_UNITS: Mapping[str, frozenset[str]] = {
    "create_booking": frozenset({"bookings.status#residual1"}),
    "cancel_booking": frozenset({"bookings.status#residual1"}),
    "book_hotel": frozenset({"hotels.status#residual1"}),
    "cancel_hotel_reservation": frozenset({"hotels.status#residual1"}),
    "book_car_rental": frozenset({"car_rentals.status#residual1"}),
    "cancel_car_rental": frozenset({"car_rentals.status#residual1"}),
}
_PRICE_DISCLOSURE_WRITE_TOOLS = frozenset({"update_booking", "cancel_booking"})
_PRICE_DISCLOSURE_TERMS = ("price", "total", "paid")
_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])\$?\d+(?:,\d{3})*(?:\.\d{1,2})?(?![A-Za-z0-9_-])"
)


@dataclass(frozen=True)
class OrderingRule:
    message_unit: str
    write_family: str


@dataclass(frozen=True)
class DialogueObligation:
    unit: str
    status: DialogueStatus
    write_family: str
    evidence_family: str


@dataclass(frozen=True)
class DialogueTriage:
    open_obligations: tuple[DialogueObligation, ...] = ()
    satisfied_obligations: tuple[DialogueObligation, ...] = ()
    violated_obligations: tuple[DialogueObligation, ...] = ()

    @property
    def has_open(self) -> bool:
        return bool(self.open_obligations)

    @property
    def has_violated(self) -> bool:
        return bool(self.violated_obligations)

    def unit_family_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for obligation in (
            *self.open_obligations,
            *self.satisfied_obligations,
            *self.violated_obligations,
        ):
            counts[obligation.unit] = counts.get(obligation.unit, 0) + 1
        return counts


@dataclass(frozen=True)
class PrewritePriceDisclosure:
    """Visible booking price that must be disclosed before a bound write.

    The amount is private runtime evidence. Public reducers may count that an
    obligation exists, but must never emit ``amount_text`` or booking ids.
    """

    amount_text: str
    pending_write_tool: str
    evidence_family: str = "visible_bound_get_booking_price_paid"


DEFAULT_ORDERING_RULES: tuple[OrderingRule, ...] = (
    OrderingRule(
        message_unit=DIALOGUE_CONNECTION_RISK_UNIT,
        write_family="cancel_booking",
    ),
)
_DELAY_COMPENSATION_TERMS = (
    "compensation",
    "meal voucher",
    "voucher",
)
_DELAY_COMPENSATION_WRITE_CUES = (
    "record",
    "issue",
    "apply",
    "credit",
    "include",
    "add",
    "set",
    "update",
    "process",
    "give me",
)
_FLIGHT_CODE_RE = re.compile(r"\b[A-Z]{2}\d{2,4}\b")
_FINAL_CONFIRMATION_OR_CORRECTION_RE = re.compile(
    r"\b(?:please\s+)?(?:confirm|verify|double-check|double check)\b"
    r"|\bjust\s+to\s+(?:confirm|verify)\b"
    r"|\b(?:correct|right)\?\s*$"
    r"|\b(?:doesn['’]?t|does not)\s+sound\s+right\b"
    r"|\bshould\s+be\b"
    r"|\bnot\s+\$?\d"
    r"|\b(?:changed|different)\s+from\b"
    r"|\bfix\s+it\s+if\s+needed\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class TravelResidualAdapter:
    surface: str = "state_bench_travel"
    write_tool_names: frozenset[str] = WRITE_TOOL_NAMES
    cost_unit_families: frozenset[str] = COST_UNIT_FAMILIES
    derived_unit_families: frozenset[str] = DERIVED_UNIT_FAMILIES
    # Reads that ground traveler-entitlement state (loyalty points,
    # memberships, discounts) before pricing-bearing closure.
    account_read_tool_names: frozenset[str] = frozenset({"get_user_details"})
    # Maximal per-tool effect-unit sets for certified tool-surface projection;
    # tools absent from this mapping (reads, update_booking) are never dropped.
    certified_projection_tool_units: ClassVar[Mapping[str, frozenset[str]]] = (
        CERTIFIED_PROJECTION_TOOL_UNITS
    )

    def build_obligation_digest(
        self,
        payload: Mapping[str, Any],
        tool_schemas: Mapping[str, Any],
    ) -> dict[str, Any]:
        triage = extract_dialogue_obligations(payload)
        return {
            "dialogue_triage": _adapter_dialogue_triage(triage).__dict__,
            "visible_booking_ids": sorted(
                residual_gates.visible_booking_records(payload)
            ),
            "visible_booking_statuses": dict(
                sorted(residual_gates.visible_booking_statuses(payload).items())
            ),
            "frozen_visible_booking_ids": residual_gates.frozen_visible_booking_ids(
                payload
            ),
            "mentioned_booking_ids": residual_gates.mentioned_booking_ids(payload),
            "visibly_active_mentioned_booking_ids": (
                residual_gates.visibly_active_mentioned_booking_ids(payload)
            ),
            "visibly_frozen_mentioned_booking_ids": (
                residual_gates.visibly_frozen_mentioned_booking_ids(payload)
            ),
            "entity_freeze_rule": (
                "Bookings whose visible status is 'cancelled' are frozen: the "
                "environment rejects update_booking and cancel_booking on them. "
                "Do not propose writes that target a frozen booking."
            ),
            "committed_cancelled_booking_count": len(
                residual_gates.confirmed_cancelled_booking_ids(payload)
            ),
            "write_tool_names": sorted(str(name) for name in tool_schemas),
        }

    def extract_dialogue_obligations(
        self, payload: Mapping[str, Any]
    ) -> AdapterDialogueTriage:
        return _adapter_dialogue_triage(extract_dialogue_obligations(payload))

    def dialogue_progress_units(
        self,
        payload: Mapping[str, Any],
        proposal: Any,
    ) -> tuple[str, ...]:
        triage = extract_dialogue_obligations(payload)
        if residual_gates.proposal_tool_calls(proposal):
            return ()
        assistant_message = proposal_assistant_message(proposal)
        obligations = (*triage.open_obligations, *triage.violated_obligations)
        progress: list[str] = []
        for obligation in obligations:
            if obligation.unit == DIALOGUE_CONNECTION_RISK_UNIT:
                if residual_gates.mentions_connection_risk(assistant_message):
                    progress.append(obligation.unit)
            elif obligation.unit == FINAL_CONFIRMATION_OR_CORRECTION_UNIT:
                if _message_satisfies_final_confirmation_or_correction(
                    assistant_message
                ):
                    progress.append(obligation.unit)
        return tuple(progress)

    def visible_evidence_tokens(self, payload: Mapping[str, Any]) -> set[str]:
        return residual_gates.visible_evidence_tokens(payload)

    def visible_conversation(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        return residual_gates.visible_conversation(payload)

    def shadow_tool_names_for_frontier(
        self,
        payload: Mapping[str, Any],
        frontier: Mapping[str, Any],
        *,
        tool_names: Sequence[str],
    ) -> frozenset[str]:
        names = frozenset(str(name) for name in tool_names if str(name))
        if not _is_final_confirmation_frontier(frontier):
            return names
        if _latest_final_confirmation_user_allows_write(payload):
            return names
        return names - self.write_tool_names

    def q_precheck_errors(self, payload: Mapping[str, Any], proposal: Any) -> list[str]:
        errors = [
            *residual_gates.connected_cancel_q_precheck_errors(payload, proposal),
            *residual_gates.frozen_target_q_precheck_errors(payload, proposal),
            *protocol_prefix_errors(payload, proposal),
            *residual_gates.fee_scope_errors(proposal),
        ]
        return sorted(set(errors))

    def q_precheck_judgment(
        self,
        payload: Mapping[str, Any],
        proposal: Any,
    ) -> ContractJudgment:
        return q_precheck_judgment_from_errors(
            self.q_precheck_errors(payload, proposal)
        )

    def deterministic_seeds(self, payload: Mapping[str, Any]) -> list[ResidualProposal]:
        seed = residual_gates.deterministic_connected_cancel_seed(payload)
        if seed is None:
            return []
        return [
            ResidualProposal(
                action=seed.action,
                assistant_message=seed.assistant_message,
                tool_calls=seed.tool_calls,
                raw_json=seed.raw_json,
                usage=seed.usage,
                source=seed.source,
            )
        ]

    def state_write_contract(self, payload: Mapping[str, Any]) -> StateWriteContract:
        committed = frozenset(
            f"bookings.status#committed{index}"
            for index, _booking_id in enumerate(
                residual_gates.confirmed_cancelled_booking_ids(payload),
                start=1,
            )
        )
        residual_witnesses = _visible_residual_write_witnesses(payload)
        return StateWriteContract(
            surface=STATE_WRITE_SURFACE,
            derivation_timing="runtime_progressive",
            closed=frozenset(
                Obligation(
                    obligation_id=f"travel.committed.{unit}",
                    write_footprint=frozenset({unit}),
                    postcondition_witness="visible_confirmed_cancel_booking_write",
                    surface=STATE_WRITE_SURFACE,
                    status="closed",
                )
                for unit in sorted(committed)
            ),
            residual=frozenset(
                Obligation(
                    obligation_id=f"travel.residual.{unit}",
                    write_footprint=frozenset({unit}),
                    postcondition_witness=witness,
                    surface=STATE_WRITE_SURFACE,
                    status="open_bounded",
                )
                for unit, witness in sorted(residual_witnesses.items())
            ),
            forbidden_writes=committed,
            legal_actions=frozenset(
                {"bounded_repair", "abstain"} if residual_witnesses else {"abstain"}
            ),
            state_change_required=bool(residual_witnesses),
            commitment_call_count=len(committed),
            commitment_footprint_size=len(committed),
            commitment_app_count=1 if committed else 0,
            soundness_assumptions=(
                "travel_runtime_bridge_uses_visible_write_family_units_only",
                "booking_ids_are_not_part_of_contract_units",
                "visible_pending_write_cues_are_not_scorer_or_gold_fields",
            ),
            touched_units=committed | frozenset(residual_witnesses),
        )

    def to_target_action(
        self,
        proposal: Any,
        payload: Mapping[str, Any],
    ) -> ProposedTargetAction:
        del payload
        groups = tuple(
            group
            for group in (
                _travel_call_units(call)
                for call in residual_gates.proposal_tool_calls(proposal)
            )
            if group
        )
        write_footprint = frozenset(unit for group in groups for unit in group)
        return ProposedTargetAction(
            surface=STATE_WRITE_SURFACE,
            action_id=f"travel_residual:{_proposal_action(proposal)}",
            write_footprint=write_footprint,
            terminal_completion=bool(
                _proposal_action(proposal) == "execute" and not write_footprint
            ),
            effect_groups=groups,
            writable_scope=write_footprint,
        )

    def restoration_cost_vector(
        self,
        payload: Mapping[str, Any],
        contract: StateWriteContract,
        proposed_action: ProposedTargetAction,
    ) -> Mapping[str, float] | None:
        del payload
        return travel_restoration_cost_vector(
            sorted(contract.forbidden_writes | proposed_action.write_footprint)
        )


ADAPTER: ResidualDomainAdapter = TravelResidualAdapter()


def _visible_residual_write_witnesses(payload: Mapping[str, Any]) -> dict[str, str]:
    """Open residual write families: requested AND not yet discharged AND enabled.

    A family is a residual obligation only when all three hold from the visible
    trajectory: (pending) the request has no visible discharging write yet;
    (enabled) at least one visible target entity still admits the write -- the
    environment freezes cancelled bookings, so a row whose only bookings are
    cancelled exposes no bookings.* write scope; (requested) a visible user
    request or agent commitment cue exists. Keyword cues alone previously listed
    every discussed family as open (zero precision); the pending/enabled guards
    are what make ``open_bounded`` mean open.

    Obligations are limited to the DIRECTLY WRITABLE tool surface (lifecycle
    status via cancel/book, ``flight_id`` and ``delay_compensation`` via
    ``update_booking``). Fee/refund/fare families are environment-computed
    side effects of those calls, not independently writable units, so listing
    them as obligations would demand writes no proposal can ever discharge.
    """

    witnesses: dict[str, str] = {}
    cancel_quote = (
        first_user_cancel_request_quote(payload)
        or "visible_same_day_onward_cancel_candidate"
    )
    if residual_gates.deterministic_connected_cancel_seed(payload) is not None:
        witnesses["bookings.status#residual1"] = cancel_quote

    active_booking = residual_gates.has_visible_enabled_booking_target(payload)
    cancel_discharged = bool(residual_gates.confirmed_cancelled_booking_ids(payload))
    rebook_discharged = residual_gates.visible_successful_book_flight(
        payload
    ) or _visible_successful_flight_update(payload)
    if (
        active_booking
        and not cancel_discharged
        and residual_gates.user_cancel_request(payload)
    ):
        witnesses.setdefault("bookings.status#residual1", cancel_quote)
    if hotel_cancel_write_requested(payload):
        witnesses.setdefault("hotels.status#residual1", "visible_hotel_cancel_request")
    if active_booking and delay_compensation_write_requested(payload):
        witnesses.setdefault(
            "bookings.delay_compensation#residual1",
            "visible_delay_compensation_write_request",
        )
    if (
        active_booking
        and not rebook_discharged
        and flight_change_write_requested(payload)
    ):
        witnesses.setdefault(
            "bookings.flight_id#residual1",
            "visible_flight_change_or_rebook_request",
        )
    return witnesses


def _visible_text(payload: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    messages = payload.get("conversation", [])
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, Mapping):
            continue
        pieces.append(str(message.get("content") or ""))
        calls = message.get("tool_calls", [])
        for call in calls if isinstance(calls, list) else []:
            if not isinstance(call, Mapping):
                continue
            pieces.append(str(call.get("name") or ""))
            result = call.get("result")
            if isinstance(result, Mapping):
                pieces.extend(str(key) for key in result)
                pieces.extend(
                    str(value) for value in result.values() if isinstance(value, str)
                )
    return " ".join(pieces).lower()


def delay_compensation_write_requested(payload: Mapping[str, Any]) -> bool:
    """True only for user-visible requests to record/issue compensation.

    Delay policy lookups and "tell me what qualifies" are message obligations,
    not authority to mutate ``bookings.delay_compensation``. Tool results and
    assistant policy summaries are deliberately ignored here; only user text can
    open the write authority.
    """

    for message in residual_gates.user_messages(payload):
        for sentence in request_cues.sentences(message):
            lowered = sentence.lower()
            if not any(term in lowered for term in _DELAY_COMPENSATION_TERMS):
                continue
            if any(cue in lowered for cue in _DELAY_COMPENSATION_WRITE_CUES):
                return True
    return False


def hotel_cancel_write_requested(payload: Mapping[str, Any]) -> bool:
    for message in residual_gates.user_messages(payload):
        for sentence in request_cues.sentences(message):
            lowered = sentence.lower()
            if "hotel" in lowered and residual_gates._has_unnegated_cancel_clause(
                sentence
            ):
                return True
    return False


def flight_change_write_requested(payload: Mapping[str, Any]) -> bool:
    for message in residual_gates.user_messages(payload):
        for sentence in request_cues.sentences(message):
            lowered = sentence.lower()
            mentions_flight_target = any(
                term in lowered for term in ("flight", "booking")
            ) or bool(_FLIGHT_CODE_RE.search(sentence))
            if not mentions_flight_target:
                continue
            if "book " in lowered and _FLIGHT_CODE_RE.search(sentence):
                return True
            if any(
                cue in lowered
                for cue in (
                    "change my flight",
                    "change the flight",
                    "change this flight",
                    "change to",
                    "cancel+rebook",
                    "cancel and rebook",
                    "rebook my",
                    "rebook onto",
                    "please rebook",
                    "want to rebook",
                    "need to rebook",
                    "switch flight",
                    "new flight",
                )
            ):
                return True
    return False


def _visible_successful_flight_update(payload: Mapping[str, Any]) -> bool:
    for message in payload.get("conversation", []):
        if not isinstance(message, Mapping):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            if str(call.get("name") or "") != "update_booking":
                continue
            args = call.get("arguments")
            result = call.get("result")
            if not isinstance(args, Mapping) or "flight_id" not in args:
                continue
            if isinstance(result, Mapping) and str(result.get("status") or "") in {
                "confirmed",
                "updated",
            }:
                return True
    return False


def _travel_call_units(call: Mapping[str, Any]) -> frozenset[str]:
    """Map a proposed tool call to the units the CALLER directly writes.

    Fee/refund/fare families are environment-computed side effects of the
    lifecycle calls, not arguments the caller sets, so they stay out of the
    proposal footprint (mirroring the witness rule that keeps them out of
    residual obligations). Their cost shows up through the committed-unit
    restoration-cost vector, not as extra footprint units.
    """

    name = str(call.get("name") or "")
    args = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
    if name in {"create_booking", "cancel_booking"}:
        return frozenset({"bookings.status#residual1"})
    if name == "update_booking":
        units = {
            f"bookings.{field}#residual1"
            for field in args
            if field
            not in {
                "booking_id",
                "confirm",
                "change_reason",
            }
        }
        return frozenset(units)
    if name in {"book_hotel", "cancel_hotel_reservation"}:
        return frozenset({"hotels.status#residual1"})
    if name in {"book_car_rental", "cancel_car_rental"}:
        return frozenset({"car_rentals.status#residual1"})
    return frozenset()


def _proposal_action(proposal: Any) -> str:
    if isinstance(proposal, Mapping):
        return str(proposal.get("action") or "")
    return str(getattr(proposal, "action", "") or "")


def _adapter_dialogue_triage(triage: "DialogueTriage") -> AdapterDialogueTriage:
    return AdapterDialogueTriage(
        open=tuple(obligation.unit for obligation in triage.open_obligations),
        satisfied=tuple(obligation.unit for obligation in triage.satisfied_obligations),
        violated=tuple(obligation.unit for obligation in triage.violated_obligations),
    )


def extract_dialogue_obligations(
    payload: Mapping[str, object],
    rules: Sequence[OrderingRule] = DEFAULT_ORDERING_RULES,
) -> DialogueTriage:
    """Classify visible dialogue obligations without reading tasks or scorer state."""

    open_obligations: list[DialogueObligation] = []
    satisfied_obligations: list[DialogueObligation] = []
    violated_obligations: list[DialogueObligation] = []
    for rule in rules:
        if rule.message_unit != DIALOGUE_CONNECTION_RISK_UNIT:
            continue
        status = classify_connection_risk_obligation(payload)
        if status is None:
            continue
        obligation = DialogueObligation(
            unit=rule.message_unit,
            status=status,
            write_family=rule.write_family,
            evidence_family="same_day_onward_cancel",
        )
        if status == "open":
            open_obligations.append(obligation)
        elif status == "satisfied":
            satisfied_obligations.append(obligation)
        else:
            violated_obligations.append(obligation)
    final_status = classify_final_confirmation_or_correction_obligation(payload)
    if final_status == "open":
        open_obligations.append(
            DialogueObligation(
                unit=FINAL_CONFIRMATION_OR_CORRECTION_UNIT,
                status="open",
                write_family="",
                evidence_family="final_user_visible_confirmation_or_correction",
            )
        )
    assistant_final_status = classify_assistant_final_confirmation_prefix_obligation(
        payload
    )
    if assistant_final_status == "open":
        open_obligations.append(
            DialogueObligation(
                unit=FINAL_CONFIRMATION_OR_CORRECTION_UNIT,
                status="open",
                write_family="",
                evidence_family="assistant_visible_final_confirmation_gap",
            )
        )
    return DialogueTriage(
        open_obligations=tuple(open_obligations),
        satisfied_obligations=tuple(satisfied_obligations),
        violated_obligations=tuple(violated_obligations),
    )


def classify_final_confirmation_or_correction_obligation(
    payload: Mapping[str, object],
) -> DialogueStatus | None:
    """Open a message-only residual for final visible confirmation/correction asks.

    This is deliberately a dialogue unit, not a write unit. It fires only when a
    user's last visible turn asks for confirmation or flags a correction after a
    visible travel write, and no assistant message follows that turn.
    """

    messages = [
        message
        for message in payload.get("conversation", [])
        if isinstance(message, Mapping)
    ]
    if not messages or not _has_visible_travel_write(payload):
        return None
    last_user_index = None
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user_index = index
    if last_user_index is None:
        return None
    if any(
        message.get("role") == "assistant"
        for message in messages[last_user_index + 1 :]
    ):
        return "satisfied"
    content = str(messages[last_user_index].get("content") or "")
    visible_text = content.replace("[TASK_DONE]", " ")
    if not _FINAL_CONFIRMATION_OR_CORRECTION_RE.search(visible_text):
        return None
    return "open"


def classify_assistant_final_confirmation_prefix_obligation(
    payload: Mapping[str, object],
) -> DialogueStatus | None:
    """Open before terminal user correction when an assistant finalizes too little.

    This is the prefix-local counterpart of the final-user correction unit. It
    fires after a visible travel write when the latest assistant presents a
    final booking/change summary, but the visible user request carried detail
    constraints that commonly require explicit confirmation.
    """

    messages = [
        message
        for message in payload.get("conversation", [])
        if isinstance(message, Mapping)
    ]
    if not messages or not _has_visible_travel_write(payload):
        return None
    if messages[-1].get("role") != "assistant":
        return None
    assistant = str(messages[-1].get("content") or "")
    if not _assistant_message_is_final_travel_confirmation(assistant):
        return None
    user_text = " ".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
    ).lower()
    if not _user_text_requests_confirmation_sensitive_detail(user_text):
        return None
    return "open"


def _assistant_message_is_final_travel_confirmation(message: str) -> bool:
    lowered = str(message or "").lower()
    return any(
        term in lowered
        for term in (
            "booked",
            "changed successfully",
            "has been changed",
            "updated booking",
            "final summary",
            "final booking",
            "done",
        )
    )


def _user_text_requests_confirmation_sensitive_detail(text: str) -> bool:
    return any(
        term in text
        for term in (
            "medical",
            "reason",
            "preference",
            "preferences",
            "vegetarian",
            "window seat",
            "economy",
            "net",
            "fee",
            "cost",
            "refund",
            "summarize",
            "summary",
            "make sure",
            "confirm",
        )
    )


def _is_final_confirmation_frontier(frontier: Mapping[str, Any]) -> bool:
    families = frontier.get("class_relevant_families")
    if not isinstance(families, Sequence) or isinstance(families, (str, bytes)):
        families = frontier.get("dialogue_families")
    if not isinstance(families, Sequence) or isinstance(families, (str, bytes)):
        return False
    return FINAL_CONFIRMATION_OR_CORRECTION_UNIT in {str(family) for family in families}


def _latest_final_confirmation_user_allows_write(payload: Mapping[str, object]) -> bool:
    text = _latest_user_text(payload).lower()
    if not text:
        return False
    if _blocks_followup_mutation(text):
        return False
    return _requests_followup_mutation(text)


def _latest_user_text(payload: Mapping[str, object]) -> str:
    messages = [
        message
        for message in payload.get("conversation", [])
        if isinstance(message, Mapping)
    ]
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _blocks_followup_mutation(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:do\s+not|don['’]?t|dont|no)\s+(?:approve|execute|change|book|cancel|use|take\s+action)\b"
            r"|\bnot\s+(?:approve|execute|change|book|cancel|use|taking\s+action)\b"
            r"|\bno\s+action\s+yet\b"
            r"|\bbefore\s+(?:i\s+)?(?:approve|confirm|confirmation)\b"
            r"|\bwithout\s+(?:my\s+)?(?:approval|confirmation)\b"
            r"|\bsurprised\s+this\s+was\s+executed\b"
            r"|\bover\s+(?:my\s+)?budget\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _requests_followup_mutation(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:add|added|update|set|change|book|cancel|rebook|proceed|execute|finalize|finalise)\b"
            r"|\bgo\s+ahead\b"
            r"|\bmake\s+sure\b.*\b(?:set|added|included|on|off)\b"
            r"|\bif\s+[^.?!]{0,80}\b(?:free|eligible|available)\b[^.?!]{0,80}\badd\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_visible_travel_write(payload: Mapping[str, object]) -> bool:
    return any(
        call.get("name") in WRITE_TOOL_NAMES
        for call in residual_gates.extract_tool_calls_with_results(payload)
    )


def _message_satisfies_final_confirmation_or_correction(message: str) -> bool:
    lowered = str(message or "").lower()
    if not lowered.strip():
        return False
    return any(
        term in lowered
        for term in (
            "confirm",
            "confirmed",
            "correct",
            "exact",
            "fee",
            "fare difference",
            "refund",
            "booking",
            "flight",
        )
    )


def classify_connection_risk_obligation(
    payload: Mapping[str, object],
) -> DialogueStatus | None:
    """Return the connected-cancel dialogue status visible from the trajectory."""

    visible_bookings = residual_gates.visible_booking_records(payload)
    if not visible_bookings:
        return None
    same_day_pairs = same_day_connection_pairs(visible_bookings)
    if not same_day_pairs:
        return None
    cancelled_ids = residual_gates.confirmed_cancelled_booking_ids(payload)
    for cancelled_id, onward_id in same_day_pairs:
        if cancelled_id not in cancelled_ids:
            continue
        if residual_gates.connection_warning_before_cancel(
            payload, cancelled_id=cancelled_id
        ):
            return "satisfied"
        onward = visible_bookings.get(onward_id, {})
        if str(onward.get("status") or "") == "confirmed":
            return "violated"
        return None
    if first_user_cancel_request(payload):
        return "open"
    return None


def same_day_connection_pairs(
    visible_bookings: Mapping[str, Mapping[str, object]],
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for first_id, first in sorted(visible_bookings.items()):
        for second_id, second in sorted(visible_bookings.items()):
            if first_id == second_id:
                continue
            if residual_gates.same_day_onward(first, second):
                pairs.append((first_id, second_id))
    return tuple(pairs)


def first_user_cancel_request(payload: Mapping[str, object]) -> bool:
    return residual_gates.user_cancel_request(payload)


def first_user_cancel_request_quote(payload: Mapping[str, object]) -> str:
    for message in payload.get("conversation", []):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        for sentence in request_cues.sentences(str(message.get("content") or "")):
            if "cancel" in sentence.lower():
                return sentence
    return ""


def prewrite_price_disclosure(
    payload: Mapping[str, object],
    proposal: Any,
) -> PrewritePriceDisclosure | None:
    """Return a bound price duty before a proposed flight mutation.

    The predicate is scorer-blind. It requires a proposed ``update_booking`` or
    ``cancel_booking`` target, a prior successful ``get_booking`` result for the
    same visible booking, a non-zero ``price_paid``, and no assistant disclosure
    of that exact amount with a price label. Merely seeing another booking price
    is insufficient.
    """

    pending_call = next(
        (
            call
            for call in residual_gates.proposal_tool_calls(proposal)
            if str(call.get("name") or "") in _PRICE_DISCLOSURE_WRITE_TOOLS
        ),
        None,
    )
    if pending_call is None:
        return None
    pending_args = pending_call.get("arguments")
    if not isinstance(pending_args, Mapping):
        return None
    target_booking_id = str(pending_args.get("booking_id") or "").strip()
    if not target_booking_id:
        return None

    amount: object | None = None
    for message in payload.get("conversation", []):
        if not isinstance(message, Mapping):
            continue
        calls = message.get("tool_calls")
        for call in calls if isinstance(calls, list) else []:
            if (
                not isinstance(call, Mapping)
                or str(call.get("name") or "") != "get_booking"
            ):
                continue
            result = call.get("result")
            arguments = call.get("arguments")
            if not isinstance(result, Mapping):
                continue
            source_booking_id = str(
                (arguments.get("booking_id") if isinstance(arguments, Mapping) else "")
                or result.get("booking_id")
                or ""
            ).strip()
            if source_booking_id != target_booking_id:
                continue
            candidate = result.get("price_paid")
            if _nonzero_numeric_text(candidate):
                amount = candidate
    if amount is None:
        return None

    visible_assistant_text = " ".join(
        str(message.get("content") or "")
        for message in payload.get("conversation", [])
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    )
    proposal_text = proposal_assistant_message(proposal)
    if _message_discloses_price(
        visible_assistant_text, amount
    ) or _message_discloses_price(
        proposal_text,
        amount,
    ):
        return None
    return PrewritePriceDisclosure(
        amount_text=str(amount).strip(),
        pending_write_tool=str(pending_call.get("name") or ""),
    )


def _message_discloses_price(message: str, amount: object) -> bool:
    lowered = str(message or "").casefold()
    if not any(term in lowered for term in _PRICE_DISCLOSURE_TERMS):
        return False
    return bool(_numeric_variants(amount) & _numeric_tokens(lowered))


def _numeric_tokens(value: object) -> frozenset[str]:
    tokens: set[str] = set()
    for match in _NUMERIC_TOKEN_RE.findall(str(value or "")):
        tokens.update(_numeric_variants(match))
    return frozenset(tokens)


def _numeric_variants(value: object) -> frozenset[str]:
    raw = str(value or "").strip().casefold()
    if not raw:
        return frozenset()
    compact = raw.replace(",", "")
    plain = compact.removeprefix("$")
    variants = {compact, plain, f"${plain}"}
    if "." in plain:
        variants.add(plain.rstrip("0").rstrip("."))
        variants.add(f"${plain.rstrip('0').rstrip('.')}")
    return frozenset(value for value in variants if value)


def _nonzero_numeric_text(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    plain = str(value).strip().replace("$", "").replace(",", "")
    try:
        return bool(plain) and float(plain) != 0.0
    except ValueError:
        return False


def protocol_prefix_errors(payload: Mapping[str, object], proposal: Any) -> list[str]:
    """Require open dialogue obligations to be discharged before paired writes."""

    triage = extract_dialogue_obligations(payload)
    if not triage.has_open:
        return []
    assistant_message = proposal_assistant_message(proposal)
    errors: list[str] = []
    for obligation in triage.open_obligations:
        if not proposal_contains_write_family(proposal, obligation.write_family):
            continue
        if (
            obligation.unit == DIALOGUE_CONNECTION_RISK_UNIT
            and not residual_gates.mentions_connection_risk(assistant_message)
        ):
            errors.append("protocol_prefix_missing_dialogue_obligation")
    return sorted(set(errors))


def proposal_assistant_message(proposal: Any) -> str:
    if isinstance(proposal, Mapping):
        return str(proposal.get("assistant_message") or "")
    return str(getattr(proposal, "assistant_message", "") or "")


def proposal_contains_write_family(proposal: Any, write_family: str) -> bool:
    return any(
        str(call.get("name") or "") == write_family
        for call in residual_gates.proposal_tool_calls(proposal)
    )
