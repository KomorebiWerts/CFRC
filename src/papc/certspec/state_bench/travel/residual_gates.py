"""Visible-data Travel residual gates for STATE-Bench pilots.

This module is adapter-owned.  It accepts only trajectory-shaped visible data
and proposal-shaped objects.  It must not import STATE-Bench task definitions,
scorers, or gold annotations.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ID_LIKE_RE = re.compile(
    r"\b(?:BK|HR|CR|UA|DL|AA|SW|BA|AF|LH|AC|AS|B6|NK|F9|HA)-?"
    r"[A-Za-z0-9_]*\d[A-Za-z0-9_]*\b"
    r"|\buser_[A-Za-z0-9_]+\b",
    flags=re.IGNORECASE,
)
_CANCEL_CLAUSE_SPLIT_RE = re.compile(
    r"[.!?;]|\b(?:but|then)\b",
    flags=re.IGNORECASE,
)
_CANCEL_TOKEN_RE = re.compile(
    r"\bcancel(?:led|ing|lation|lations)?\b", flags=re.IGNORECASE
)
_NEGATED_CANCEL_RE = re.compile(
    r"\b(?:do not|don't|dont|never|not)\b.{0,40}\bcancel(?:led|ing)?\b"
    r"|\bwithout\b.{0,20}\bcancel(?:ing|ling)?\b",
    flags=re.IGNORECASE,
)
_CANCEL_WRITE_REQUEST_RE = re.compile(
    r"\b(?:please\s+)?cancel\s+(?:my|the|this|both|all|booking|trip|flight|hotel|reservation|bk-?)\b"
    r"|\b(?:go ahead|proceed|process|start)\b.{0,30}\bcancel(?:lation|ing)?\b"
    r"|\b(?:i\s+want|i\s+need|i'd\s+like|i would\s+like)\s+to\s+cancel\b",
    flags=re.IGNORECASE,
)
_CANCEL_INFO_ONLY_RE = re.compile(
    r"\b(?:thinking of|considering|before i decide|before deciding|if i cancel|"
    r"can you explain|explain|tell me|what would|what's|what is|policy|fee|refund|"
    r"alternative to cancel(?:ing|ling)?)\b"
    r"|\b(?:hold off|not taking any action|no action|only wanted|just needed)\b",
    flags=re.IGNORECASE,
)

_BOOKING_ID_RE = re.compile(
    r"\bBK-?[A-Za-z0-9_]*\d[A-Za-z0-9_]*\b", flags=re.IGNORECASE
)

FEE_BEARING_WRITE_TOOLS = frozenset(
    {
        "create_booking",
        "cancel_booking",
        "book_hotel",
        "cancel_hotel_reservation",
        "book_car_rental",
        "cancel_car_rental",
    }
)
FEE_BEARING_UPDATE_ARGUMENTS = frozenset({"flight_id", "cabin_class"})


@dataclass(frozen=True)
class TravelResidualSeed:
    action: str
    assistant_message: str
    tool_calls: tuple[dict[str, Any], ...]
    raw_json: Mapping[str, Any]
    usage: Mapping[str, int | None]
    source: str = "strong_model"


def deterministic_connected_cancel_seed(
    payload: Mapping[str, Any],
) -> TravelResidualSeed | None:
    """Visible-evidence seed for same-day connected-flight cancellation."""

    if not user_cancel_request(payload):
        return None
    visible_bookings = visible_booking_records(payload)
    cancelled_ids = confirmed_cancelled_booking_ids(payload)
    if not cancelled_ids:
        return None
    for cancelled_id in cancelled_ids:
        cancelled = visible_bookings.get(cancelled_id)
        if not cancelled:
            continue
        for booking_id, booking in sorted(visible_bookings.items()):
            if booking_id == cancelled_id:
                continue
            if str(booking.get("status") or "") != "confirmed":
                continue
            if not same_day_onward(cancelled, booking):
                continue
            return TravelResidualSeed(
                action="execute",
                assistant_message=(
                    "I also found a same-day onward connection tied to the cancelled leg, "
                    "so I am cancelling that connected booking as the remaining safe "
                    "residual action."
                ),
                tool_calls=(
                    {
                        "name": "cancel_booking",
                        "arguments": {"booking_id": booking_id, "confirm": True},
                    },
                ),
                raw_json={
                    "action": "execute",
                    "assistant_message": "deterministic_same_day_onward_cancel_seed",
                    "tool_calls": [
                        {
                            "name": "cancel_booking",
                            "arguments": {"booking_id": booking_id, "confirm": True},
                        }
                    ],
                    "cost_scope": {
                        "applies_to": ["cancel_booking"],
                        "visible_scope": (
                            "cancellation fee and refund effects for the visible "
                            "same-day onward booking only"
                        ),
                    },
                },
                usage={},
                source="deterministic_same_day_onward_cancel_seed",
            )
    return None


def first_user_message(payload: Mapping[str, Any]) -> str:
    for message in (
        payload.get("conversation", [])
        if isinstance(payload.get("conversation"), list)
        else []
    ):
        if isinstance(message, Mapping) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def user_messages(payload: Mapping[str, Any]) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in payload.get("conversation", [])
        if isinstance(message, Mapping) and message.get("role") == "user"
    ]


def user_cancel_request(payload: Mapping[str, Any]) -> bool:
    return any(
        _has_unnegated_cancel_clause(message) for message in user_messages(payload)
    )


def _has_unnegated_cancel_clause(message: str) -> bool:
    for clause in _CANCEL_CLAUSE_SPLIT_RE.split(message):
        if not _CANCEL_TOKEN_RE.search(clause):
            continue
        if _NEGATED_CANCEL_RE.search(clause):
            continue
        if not _CANCEL_WRITE_REQUEST_RE.search(clause):
            continue
        if _CANCEL_INFO_ONLY_RE.search(clause):
            continue
        return True
    return False


def visible_booking_records(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    bookings: dict[str, Mapping[str, Any]] = {}
    for call in extract_tool_calls_with_results(payload):
        if call.get("name") != "get_booking":
            continue
        result = call.get("result")
        if not isinstance(result, Mapping):
            continue
        booking_id = result.get("booking_id")
        if isinstance(booking_id, str) and booking_id:
            bookings[booking_id] = result
    return bookings


def confirmed_cancelled_booking_ids(payload: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for call in extract_tool_calls_with_results(payload):
        if call.get("name") != "cancel_booking":
            continue
        args = (
            call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        )
        result = call.get("result") if isinstance(call.get("result"), Mapping) else {}
        if args.get("confirm") is not True:
            continue
        if result.get("status") != "cancelled":
            continue
        booking_id = result.get("booking_id") or args.get("booking_id")
        if isinstance(booking_id, str) and booking_id:
            ids.append(booking_id)
    return ids


def visible_booking_statuses(payload: Mapping[str, Any]) -> dict[str, str]:
    """Latest visible status per booking id, read from visible tool results only.

    Walks the visible cheap trajectory in order and keeps the last status seen
    for each booking id, so a booking that was confirmed and later cancelled
    reports ``cancelled``. Records that carry no ``booking_id`` (e.g. rejected
    call envelopes) are skipped. No gold or scorer fields are consumed.
    """

    statuses: dict[str, str] = {}
    for call in extract_tool_calls_with_results(payload):
        result = call.get("result")
        if not isinstance(result, Mapping):
            continue
        records: list[Mapping[str, Any]] = [result]
        nested = result.get("bookings")
        if isinstance(nested, list):
            records.extend(item for item in nested if isinstance(item, Mapping))
        for record in records:
            booking_id = record.get("booking_id")
            status = record.get("status")
            if (
                isinstance(booking_id, str)
                and booking_id
                and isinstance(status, str)
                and status
            ):
                statuses[booking_id] = status
    return statuses


def frozen_visible_booking_ids(payload: Mapping[str, Any]) -> list[str]:
    """Booking ids whose latest visible status forbids further writes.

    The travel environment rejects ``update_booking``/``cancel_booking`` on a
    cancelled booking ("Booking ... is cancelled and cannot be updated"), so a
    visibly-cancelled booking is a frozen write target.
    """

    return sorted(
        booking_id
        for booking_id, status in visible_booking_statuses(payload).items()
        if status == "cancelled"
    )


def has_visible_active_booking(payload: Mapping[str, Any]) -> bool:
    """True iff at least one visible booking still admits in-place writes."""

    return any(
        status != "cancelled" for status in visible_booking_statuses(payload).values()
    )


def mentioned_booking_ids(payload: Mapping[str, Any]) -> list[str]:
    """Booking-like BK identifiers explicitly named in user-visible text."""

    return sorted(
        {
            match.group(0).upper()
            for message in user_messages(payload)
            for match in _BOOKING_ID_RE.finditer(message)
        }
    )


def visibly_active_mentioned_booking_ids(payload: Mapping[str, Any]) -> list[str]:
    """Visible mentioned booking ids whose latest status still admits writes.

    This binds residual booking-write authority to the user-visible target when
    the user names one. If no booking id is mentioned, callers can retain their
    broader visible-active-booking fallback to protect recall.
    """

    statuses = visible_booking_statuses(payload)
    if not statuses:
        return []
    mentioned = set(mentioned_booking_ids(payload))
    return sorted(
        booking_id
        for booking_id, status in statuses.items()
        if booking_id.upper() in mentioned and status != "cancelled"
    )


def visibly_frozen_mentioned_booking_ids(payload: Mapping[str, Any]) -> list[str]:
    """Visible mentioned booking ids whose latest status freezes writes."""

    statuses = visible_booking_statuses(payload)
    if not statuses:
        return []
    mentioned = set(mentioned_booking_ids(payload))
    return sorted(
        booking_id
        for booking_id, status in statuses.items()
        if booking_id.upper() in mentioned and status == "cancelled"
    )


def user_mentions_booking_id(payload: Mapping[str, Any]) -> bool:
    """True iff user-visible text names a booking-like BK identifier."""

    return bool(mentioned_booking_ids(payload))


def has_visible_enabled_booking_target(payload: Mapping[str, Any]) -> bool:
    """True iff booking writes have a visible active target.

    If the user explicitly names a booking id, the named target must be visibly
    active. If no booking id is named, fall back to the historical domain-wide
    active-booking guard so generic requests such as "rebook my flight" keep
    their residual recall.
    """

    if user_mentions_booking_id(payload):
        return bool(visibly_active_mentioned_booking_ids(payload))
    return has_visible_active_booking(payload)


def visible_successful_book_flight(payload: Mapping[str, Any]) -> bool:
    """True iff a visible flight-booking call returned a created booking."""

    for call in extract_tool_calls_with_results(payload):
        if call.get("name") not in {"book_flight", "create_booking"}:
            continue
        result = call.get("result")
        if (
            isinstance(result, Mapping)
            and isinstance(result.get("booking_id"), str)
            and result.get("booking_id")
        ):
            return True
    return False


def frozen_target_q_precheck_errors(
    payload: Mapping[str, Any], proposal: Any
) -> list[str]:
    """Block writes that target a visibly frozen (cancelled) booking.

    The environment would reject these calls anyway; vetoing them at q_precheck
    gives the receipt a typed reason instead of a runtime tool error, and keeps
    the single bounded execution from being spent on a write that cannot land.
    """

    frozen = set(frozen_visible_booking_ids(payload))
    if not frozen:
        return []
    errors: list[str] = []
    for call in proposal_tool_calls(proposal):
        if call.get("name") not in {"update_booking", "cancel_booking"}:
            continue
        args = (
            call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        )
        target_id = args.get("booking_id")
        if isinstance(target_id, str) and target_id in frozen:
            errors.append("q_precheck_fail_target_entity_frozen")
    return sorted(set(errors))


def same_day_onward(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    first_destination = str(first.get("destination") or "")
    second_origin = str(second.get("origin") or "")
    if not first_destination or first_destination != second_origin:
        return False
    first_departure = str(first.get("departure_time") or "")
    second_departure = str(second.get("departure_time") or "")
    if not first_departure or not second_departure:
        return False
    if first_departure[:10] != second_departure[:10]:
        return False
    return second_departure > first_departure


def extract_tool_calls_with_results(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for msg in (
        payload.get("conversation", [])
        if isinstance(payload.get("conversation"), list)
        else []
    ):
        if not isinstance(msg, Mapping) or not isinstance(msg.get("tool_calls"), list):
            continue
        for call in msg["tool_calls"]:
            if isinstance(call, Mapping):
                args = (
                    call.get("arguments")
                    if isinstance(call.get("arguments"), Mapping)
                    else {}
                )
                calls.append(
                    {
                        "name": str(call.get("name") or ""),
                        "arguments": dict(args),
                        "result": call.get("result"),
                    }
                )
    return calls


def visible_conversation(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for message in (
        payload.get("conversation", [])
        if isinstance(payload.get("conversation"), list)
        else []
    ):
        if not isinstance(message, Mapping):
            continue
        entry = {
            "role": message.get("role"),
            "content": message.get("content"),
        }
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            entry["tool_calls"] = [
                {
                    "name": call.get("name"),
                    "arguments": call.get("arguments"),
                    "result": call.get("result"),
                }
                for call in calls
                if isinstance(call, Mapping)
            ]
        visible.append(entry)
    return visible


def connected_cancel_q_precheck_errors(
    payload: Mapping[str, Any], proposal: Any
) -> list[str]:
    """Block state-only connection fixes after an irreversible dialogue miss."""

    visible_bookings = visible_booking_records(payload)
    cancelled_ids = confirmed_cancelled_booking_ids(payload)
    if not visible_bookings or not cancelled_ids:
        return []
    errors: list[str] = []
    for call in proposal_tool_calls(proposal):
        if call.get("name") != "cancel_booking":
            continue
        args = (
            call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        )
        target_id = args.get("booking_id")
        if not isinstance(target_id, str):
            continue
        target = visible_bookings.get(target_id)
        if not target:
            continue
        for cancelled_id in cancelled_ids:
            cancelled = visible_bookings.get(cancelled_id)
            if not cancelled or not same_day_onward(cancelled, target):
                continue
            if not connection_warning_before_cancel(payload, cancelled_id=cancelled_id):
                errors.append("q_precheck_fail_dialogue_order_irreversible")
    return errors


def connection_warning_before_cancel(
    payload: Mapping[str, Any], *, cancelled_id: str
) -> bool:
    warning_seen = False
    for message in (
        payload.get("conversation", [])
        if isinstance(payload.get("conversation"), list)
        else []
    ):
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "assistant" and mentions_connection_risk(
            str(message.get("content") or "")
        ):
            warning_seen = True
        for call in (
            message.get("tool_calls", [])
            if isinstance(message.get("tool_calls"), list)
            else []
        ):
            if not isinstance(call, Mapping) or call.get("name") != "cancel_booking":
                continue
            args = (
                call.get("arguments")
                if isinstance(call.get("arguments"), Mapping)
                else {}
            )
            result = (
                call.get("result") if isinstance(call.get("result"), Mapping) else {}
            )
            booking_id = result.get("booking_id") or args.get("booking_id")
            if booking_id == cancelled_id and args.get("confirm") is True:
                return warning_seen
    return warning_seen


def mentions_connection_risk(text: str) -> bool:
    lowered = text.lower()
    risk_terms = ("connection", "connected", "onward", "same-day", "strand", "orphan")
    if any(term in lowered for term in risk_terms):
        return True
    later_flight_terms = (
        "later flight may be affected",
        "later flight could be affected",
        "may affect your ability to use the later",
        "could affect your ability to use the later",
        "separate later flight",
        "later atl",
    )
    return any(term in lowered for term in later_flight_terms)


def visible_evidence_tokens(payload: Mapping[str, Any]) -> set[str]:
    blob = json.dumps(visible_conversation(payload), ensure_ascii=False)
    return {token.lower() for token in ID_LIKE_RE.findall(blob)}


def id_like_tokens(value: Any) -> set[str]:
    return {
        token for token in ID_LIKE_RE.findall(json.dumps(value, ensure_ascii=False))
    }


def proposal_raw_json(proposal: Any) -> Mapping[str, Any]:
    raw = getattr(proposal, "raw_json", None)
    if isinstance(raw, Mapping):
        return raw
    if isinstance(proposal, Mapping):
        return proposal
    return {}


def proposal_tool_calls(proposal: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(proposal, Mapping) and isinstance(proposal.get("tool_calls"), list):
        return tuple(
            call for call in proposal["tool_calls"] if isinstance(call, Mapping)
        )
    calls = getattr(proposal, "tool_calls", ())
    if isinstance(calls, tuple):
        return calls
    if isinstance(calls, list):
        return tuple(call for call in calls if isinstance(call, Mapping))
    return ()


def fee_scope_errors(proposal: Any) -> list[str]:
    """Require explicit cost-scope declarations for fee-bearing Travel writes."""

    errors: list[str] = []
    for call in proposal_tool_calls(proposal):
        if fee_bearing_call(call) and not cost_scope_declared_for_call(proposal, call):
            errors.append("fee_scope_missing_for_fee_bearing_action")
    return sorted(set(errors))


def fee_bearing_call(call: Mapping[str, Any]) -> bool:
    name = str(call.get("name") or "")
    args = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
    if name in FEE_BEARING_WRITE_TOOLS:
        return True
    if name == "update_booking":
        return any(key in args for key in FEE_BEARING_UPDATE_ARGUMENTS)
    return False


def cost_scope_declared_for_call(proposal: Any, call: Mapping[str, Any]) -> bool:
    raw = proposal_raw_json(proposal)
    scope = raw.get("cost_scope")
    call_name = str(call.get("name") or "")
    if isinstance(scope, Mapping):
        return _cost_scope_entry_applies(scope, call_name)
    if isinstance(scope, list):
        return any(
            isinstance(entry, Mapping) and _cost_scope_entry_applies(entry, call_name)
            for entry in scope
        )
    return False


def _cost_scope_entry_applies(scope: Mapping[str, Any], call_name: str) -> bool:
    visible_scope = scope.get("visible_scope") or scope.get("scope")
    if not isinstance(visible_scope, str) or not visible_scope.strip():
        return False
    applies_raw = (
        scope.get("applies_to")
        or scope.get("fee_bearing_actions")
        or scope.get("tool_calls")
    )
    applies = _scope_applies_to(applies_raw)
    return call_name in applies or "*" in applies


def _scope_applies_to(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list | tuple | set):
        return {str(item) for item in value if str(item)}
    return set()
