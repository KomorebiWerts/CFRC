"""Receipt-grounded residual authoring for ToolSandbox true-online CFRC.

ToolSandbox used to author the whole residual in one strong call and check only
the shape of what came back: tool order, argument key names, argument types.
Nothing tied an argument *value* to this request, so a reminder whose content and
target time were invented passed the same checks as one read off the visible
prefix, and a single rejection ended the episode.

This module puts the surface on the shared receipt-grounded protocol the other
surfaces already use.  The successor authors the residual one native call at a
time against a private replica of the reached state, and each receipt is fed back
before the next proposal.  The realized calls are then lifted into the shared
proposal graph by :func:`bridge_grounded_proposal`, so the compiler enforces that
a value produced during completion is consumed only through a receipt of a
selected predecessor.

Argument carriers cover values that execution produced.  They cannot speak for a
value the request supplied, so this module adds the surface-local half: every
argument of an effect-bearing call must be request-grounded (a contiguous span of
the visible request text) or receipt-derived.  A value that is neither is refused
and returned to the successor as typed feedback for a bounded re-proposal.
"""

from __future__ import annotations

import copy
import json
import re
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from papc.certspec.grounded_proposal import (
    GroundedDischargeDeclaration,
    GroundedNodeProjection,
    GroundedProposalBundle,
    GroundedProposalTrace,
    GroundedToolCall,
    bridge_grounded_proposal,
)
from papc.certspec.toolsandbox.online_frontier_monitor import (
    CELLULAR_OFF_EFFECT_UNIT,
    LOCATION_SERVICE_EFFECT_UNIT,
    LOW_BATTERY_EFFECT_UNIT,
    MESSAGE_ANSWER_EFFECT_UNIT,
    REMINDER_EFFECT_UNIT,
    STOCK_ANSWER_EFFECT_UNIT,
    WEATHER_TEMP_F_EFFECT_UNIT,
    WIFI_OFF_EFFECT_UNIT,
)
from papc.certspec.toolsandbox.online_official_runtime import (
    render_assistant_response_template,
)

SCHEMA = "papc.toolsandbox.grounded_residual_authoring.v1"
OBLIGATION_ID_PREFIX = "toolsandbox.residual."
GRAPH_ID = "toolsandbox_grounded_residual_graph"
AUTHORITY_EFFECT = "ell1"
AUTHORITY_EVIDENCE = "ell0"

ACTION_TOOL_CALL = "tool_call"
ACTION_TERMINAL = "terminal"
ACTION_DECLINE = "decline"

# Which semantic effect each native ToolSandbox write carries. A tool absent
# here is read-only for this surface and becomes an evidence-acquisition node.
EFFECT_UNIT_BY_TOOL: Mapping[str, str] = {
    "add_reminder": REMINDER_EFFECT_UNIT,
    "set_low_battery_mode_status": LOW_BATTERY_EFFECT_UNIT,
    "set_wifi_status": WIFI_OFF_EFFECT_UNIT,
    "set_location_service_status": LOCATION_SERVICE_EFFECT_UNIT,
    "set_cellular_service_status": CELLULAR_OFF_EFFECT_UNIT,
    "search_stock": STOCK_ANSWER_EFFECT_UNIT,
    "search_messages": MESSAGE_ANSWER_EFFECT_UNIT,
    "search_weather_around_lat_lon": WEATHER_TEMP_F_EFFECT_UNIT,
}

# What one authoring step may read from the typed row: the tool contract the
# residual must stay inside, and the surface rules that follow from it. The rest
# of the row is internal bookkeeping or a carrier for the single-shot graph
# protocol, and handing that over invites the successor to answer in a protocol
# this pass does not accept.
_AUTHORING_SUFFIX_KEYS: tuple[str, ...] = (
    "template_id",
    "mutating_tool_name",
    "mutating_tool_must_be_last",
    "mutating_arg_key_types",
    "optional_mutating_arg_key_types",
    "tool_arg_key_types",
)
# A step can be malformed without the pass being lost: the successor still has
# rounds left and now has the reason. Only a run of them ends the pass.
_MAX_MALFORMED_AUTHORING_STEPS = 3
_MAX_TERMINAL_RESPONSE_CHARACTERS = 8192
_NUMERIC_MATCH_TOLERANCE = 1e-6
_WORD_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def obligation_id_for_unit(unit: str) -> str:
    """Return the stable obligation id carrying one semantic effect unit."""

    suffix = str(unit)
    if suffix.startswith("toolsandbox."):
        suffix = suffix[len("toolsandbox.") :]
    return f"{OBLIGATION_ID_PREFIX}{suffix}"


def authoring_suffix_view(typed_suffix: Mapping[str, Any]) -> dict[str, Any]:
    """Project the typed suffix down to the tool contract one step may read.

    The surface rules travel separately, as ``domain_instructions`` on the step
    context, because the prompt states them once as instructions rather than
    twice per step.
    """

    return {
        key: copy.deepcopy(typed_suffix[key])
        for key in _AUTHORING_SUFFIX_KEYS
        if key in typed_suffix
    }


@dataclass(frozen=True)
class AuthoringStep:
    """One successor decision during receipt-grounded authoring."""

    action: str
    tool_name: str = ""
    arguments: Mapping[str, Any] = field(default_factory=dict)
    terminal_response: str = ""
    terminal_response_template: str = ""
    terminal_response_bindings: Mapping[str, Any] = field(default_factory=dict)
    obligation_id: str = ""
    mode: str = "decline"
    basis: str = ""

    @classmethod
    def from_response(cls, response: Mapping[str, Any]) -> "AuthoringStep":
        action = str(response.get("action") or "").strip()
        if action == ACTION_TERMINAL:
            return cls(
                action=ACTION_TERMINAL,
                terminal_response=str(response.get("terminal_response") or "").strip(),
            )
        if action == ACTION_DECLINE:
            mode = str(response.get("mode") or "decline").strip()
            return cls(
                action=ACTION_DECLINE,
                obligation_id=str(response.get("obligation_id") or "").strip(),
                mode=mode if mode in ("decline", "clarify") else "decline",
                basis=str(response.get("basis") or "").strip(),
            )
        arguments = response.get("arguments")
        terminal_bindings = response.get("terminal_response_bindings")
        return cls(
            action=ACTION_TOOL_CALL,
            tool_name=str(response.get("tool_name") or "").strip(),
            arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
            terminal_response=str(response.get("terminal_response") or "").strip(),
            terminal_response_template=str(
                response.get("terminal_response_template") or ""
            ).strip(),
            terminal_response_bindings=(
                dict(terminal_bindings)
                if isinstance(terminal_bindings, Mapping)
                else {}
            ),
        )


@dataclass(frozen=True)
class AuthoringOutcome:
    """Result of one bounded authoring pass over a private replica."""

    calls: tuple[GroundedToolCall, ...]
    declarations: tuple[GroundedDischargeDeclaration, ...]
    terminal_response: str
    blockers: tuple[str, ...]
    authoring_round_count: int
    executed_call_count: int
    failed_call_count: int
    terminal_with_last_call_enabled: bool = False
    terminal_with_last_call_offered_count: int = 0
    terminal_with_last_call_accepted_count: int = 0
    terminal_with_last_call_rejection_reasons: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.blockers and bool(self.terminal_response)

    def public_summary(self) -> dict[str, Any]:
        summary = {
            "schema": SCHEMA,
            "authoring_round_count": int(self.authoring_round_count),
            "replica_executed_call_count": int(self.executed_call_count),
            "replica_failed_call_count": int(self.failed_call_count),
            "declaration_count": len(self.declarations),
            "terminal_present": bool(self.terminal_response),
            "authoring_blockers": list(self.blockers),
            "concrete_arguments_public": False,
        }
        if self.terminal_with_last_call_enabled:
            summary.update(
                {
                    "terminal_with_last_call": True,
                    "terminal_with_last_call_offered_count": int(
                        self.terminal_with_last_call_offered_count
                    ),
                    "terminal_with_last_call_accepted_count": int(
                        self.terminal_with_last_call_accepted_count
                    ),
                    "terminal_with_last_call_rejected_count": max(
                        0,
                        int(self.terminal_with_last_call_offered_count)
                        - int(self.terminal_with_last_call_accepted_count),
                    ),
                    "terminal_with_last_call_rejection_reasons": sorted(
                        set(self.terminal_with_last_call_rejection_reasons)
                    ),
                }
            )
        return summary


NextStep = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def run_grounded_authoring(
    *,
    runtime: Any,
    typed_row: Mapping[str, Any],
    prefix_conversation: Sequence[Mapping[str, Any]],
    next_step: NextStep,
    max_authoring_rounds: int,
    feedback: Sequence[Mapping[str, Any]] = (),
    declarations_accepted: bool = True,
    terminal_with_last_call: bool = False,
) -> AuthoringOutcome:
    """Author the residual one native call at a time against the active replica.

    The caller owns the state: it must have restored the reached-state checkpoint
    before calling and must restore it again afterwards.  This function only
    advances the replica and records what each call returned.

    ``declarations_accepted`` states whether this surface's contract can discharge
    an obligation without a receipt.  When it cannot, a declaration is refused
    here rather than carried to a compiler that rejects every one of them, which
    would spend the pass on a route it could never complete.
    """

    allowed_sequence = _allowed_tool_sequence(typed_row)
    allowed_units = frozenset(_effect_units(typed_row))
    calls: list[GroundedToolCall] = []
    declarations: list[GroundedDischargeDeclaration] = []
    blockers: list[str] = []
    terminal_response = ""
    rounds = 0
    malformed = 0
    seen_receipt_refs: set[str] = set()
    piggyback_offered = 0
    piggyback_accepted = 0
    piggyback_rejection_reasons: list[str] = []

    while rounds < max(1, int(max_authoring_rounds)):
        rounds += 1
        context = _authoring_context(
            typed_row=typed_row,
            prefix_conversation=prefix_conversation,
            calls=calls,
            allowed_sequence=allowed_sequence,
            allowed_units=allowed_units,
            feedback=feedback,
            round_index=rounds,
            declarations=declarations,
            declarations_accepted=declarations_accepted,
            terminal_with_last_call=terminal_with_last_call,
        )
        step = AuthoringStep.from_response(next_step(context))
        if step.action == ACTION_TERMINAL:
            terminal_response = step.terminal_response
            if not terminal_response:
                blockers.append("terminal_response_missing")
            break
        if step.action == ACTION_DECLINE:
            if not declarations_accepted:
                # This contract discharges an obligation only through a receipt,
                # so a declaration can never complete the residual. Refuse it and
                # let the successor spend the round on a real call.
                malformed += 1
                if malformed > _MAX_MALFORMED_AUTHORING_STEPS:
                    blockers.append("declaration_not_accepted_by_contract")
                    break
                continue
            obligation_id = step.obligation_id
            if obligation_id not in {
                obligation_id_for_unit(unit) for unit in allowed_units
            }:
                malformed += 1
                if malformed > _MAX_MALFORMED_AUTHORING_STEPS:
                    blockers.append("declaration_unknown_obligation")
                    break
                continue
            if any(item.obligation_id == obligation_id for item in declarations):
                # The obligation is already recorded as declined. Repeating that
                # is redundant, not a failure, so the pass keeps its remaining
                # rounds to reach a terminal.
                malformed += 1
                if malformed > _MAX_MALFORMED_AUTHORING_STEPS:
                    blockers.append("declaration_repeats_obligation")
                    break
                continue
            declarations.append(
                GroundedDischargeDeclaration(
                    obligation_id=obligation_id,
                    mode=step.mode,
                    basis=step.basis or "successor_declined_open_obligation",
                )
            )
            continue
        if not step.tool_name:
            malformed += 1
            if malformed > _MAX_MALFORMED_AUTHORING_STEPS:
                blockers.append("authoring_step_missing_tool_name")
                break
            continue
        if step.tool_name not in allowed_sequence:
            blockers.append("authoring_step_tool_outside_allowed_sequence")
            break

        receipt = runtime.execute_grounded_call(
            tool_name=step.tool_name,
            arguments=dict(step.arguments),
        )
        call = GroundedToolCall(
            node_id=f"grounded_call_{len(calls) + 1:03d}",
            tool_name=step.tool_name,
            arguments=dict(step.arguments),
            result=_receipt_result(receipt),
            proposal_round=rounds,
            succeeded=bool(receipt.get("succeeded")),
            visible_receipt_refs=tuple(prior.receipt_ref for prior in calls),
            replay_payload={
                "kind": "tool_call",
                "tool_calls": [
                    {
                        "tool_name": step.tool_name,
                        "arguments": dict(step.arguments),
                    }
                ],
                # A call that failed on the replica produced no receipt and
                # discharged no obligation. Replaying it can only fail again.
                "replica_succeeded": bool(receipt.get("succeeded")),
            },
        )
        if call.receipt_ref in seen_receipt_refs:
            blockers.append("authoring_step_repeats_receipt")
            break
        seen_receipt_refs.add(call.receipt_ref)
        calls.append(call)
        if terminal_with_last_call and _terminal_with_last_call_offered(step):
            piggyback_offered += 1
            rendered_terminal, rejection_reason = _admit_terminal_with_last_call(
                step=step,
                calls=calls,
                declarations=declarations,
                allowed_units=allowed_units,
                typed_row=typed_row,
                prefix_conversation=prefix_conversation,
            )
            if rejection_reason:
                piggyback_rejection_reasons.append(rejection_reason)
                continue
            terminal_response = rendered_terminal
            piggyback_accepted += 1
            break
    else:
        blockers.append("authoring_round_budget_exhausted")

    if not terminal_response and "terminal_response_missing" not in blockers:
        blockers.append("terminal_response_missing")

    return AuthoringOutcome(
        calls=tuple(calls),
        declarations=tuple(declarations),
        terminal_response=terminal_response,
        blockers=tuple(sorted(set(blockers))),
        authoring_round_count=rounds,
        executed_call_count=len(calls),
        failed_call_count=sum(1 for call in calls if not call.succeeded),
        terminal_with_last_call_enabled=bool(terminal_with_last_call),
        terminal_with_last_call_offered_count=piggyback_offered,
        terminal_with_last_call_accepted_count=piggyback_accepted,
        terminal_with_last_call_rejection_reasons=tuple(
            piggyback_rejection_reasons
        ),
    )


def build_grounded_bundle(
    *,
    outcome: AuthoringOutcome,
    typed_row: Mapping[str, Any],
    prefix_conversation: Sequence[Mapping[str, Any]],
) -> tuple[GroundedProposalBundle | None, tuple[str, ...]]:
    """Lift a realized authoring pass into the shared IR, or report blockers."""

    if not outcome.ready:
        return None, outcome.blockers
    units = _effect_units(typed_row)
    obligation_ids = tuple(obligation_id_for_unit(unit) for unit in units)
    grounding_blockers = argument_grounding_blockers(
        calls=outcome.calls,
        prefix_conversation=prefix_conversation,
        allowed_units=frozenset(units),
        schema_defaults=schema_default_arguments(typed_row),
    )
    if grounding_blockers:
        return None, grounding_blockers
    trace = GroundedProposalTrace(
        calls=outcome.calls,
        terminal_response=outcome.terminal_response,
        terminal_visible_receipt_refs=tuple(
            call.receipt_ref for call in outcome.calls
        ),
    )
    try:
        bundle = bridge_grounded_proposal(
            trace=trace,
            graph_id=GRAPH_ID,
            residual_obligation_ids=obligation_ids,
            evidence_fact_ids=(),
            terminal_authority_label=AUTHORITY_EVIDENCE,
            project_call=_call_projector(allowed_units=frozenset(units)),
            bind_value_provenance=True,
            frontier_values=frontier_values_from_prefix(prefix_conversation),
            include_numeric_value_provenance=True,
            declarations=outcome.declarations,
            declaration_authority_label=AUTHORITY_EVIDENCE,
            coverage_unit_tokens_by_obligation={
                obligation_id_for_unit(unit): (unit,) for unit in units
            },
        )
    except ValueError as exc:
        return None, (_bridge_blocker(exc),)
    return bundle, ()


def schema_default_arguments(
    typed_row: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return each tool's argument defaults as transcribed from its signature.

    A value the tool signature itself supplies was not invented by the successor
    and was never in the request, so request grounding does not apply to it.
    """

    typed_suffix = typed_row.get("typed_suffix")
    if not isinstance(typed_suffix, Mapping):
        return {}
    arg_key_types = typed_suffix.get("tool_arg_key_types")
    if not isinstance(arg_key_types, Mapping):
        return {}
    defaults: dict[str, dict[str, Any]] = {}
    for tool_name, entries in arg_key_types.items():
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping) or "schema_default" not in entry:
                continue
            key = str(entry.get("key") or "")
            if key:
                defaults.setdefault(str(tool_name), {})[key] = entry["schema_default"]
    return defaults


def argument_grounding_blockers(
    *,
    calls: Sequence[GroundedToolCall],
    prefix_conversation: Sequence[Mapping[str, Any]],
    allowed_units: frozenset[str],
    schema_defaults: Mapping[str, Mapping[str, Any]] = {},
) -> tuple[str, ...]:
    """Refuse an effect-bearing argument that the request never supplied.

    Carriers already cover values execution produced.  The remaining gap is a
    value the successor invented: it matches no receipt and appears nowhere in
    the visible request.  Only effect-bearing calls are checked, because a wrong
    read argument cannot commit state and shows up as a wrong result downstream.
    """

    raw_request_text = _visible_request_text(prefix_conversation)
    request_text = _normalized_text(raw_request_text)
    blockers: list[str] = []
    produced_text: set[str] = set()
    produced_numbers: list[float] = []
    for call in calls:
        unit = EFFECT_UNIT_BY_TOOL.get(call.tool_name)
        if unit is not None and unit in allowed_units and call.succeeded:
            tool_defaults = schema_defaults.get(call.tool_name) or {}
            for key, value in call.arguments.items():
                if key in tool_defaults and _matches_schema_default(
                    value,
                    tool_defaults[key],
                ):
                    continue
                if isinstance(value, bool) or value is None:
                    continue
                if isinstance(value, (int, float)):
                    if not _number_is_grounded(
                        float(value),
                        produced_numbers=produced_numbers,
                        request_text=raw_request_text,
                    ):
                        blockers.append("ungrounded_numeric_write_argument")
                    continue
                if isinstance(value, str):
                    if not _text_is_grounded(
                        value,
                        produced_text=produced_text,
                        request_text=request_text,
                    ):
                        blockers.append("ungrounded_text_write_argument")
                    continue
                blockers.append("unsupported_write_argument_type")
        _absorb_produced_values(
            call,
            produced_text=produced_text,
            produced_numbers=produced_numbers,
        )
    return tuple(sorted(set(blockers)))


def frontier_values_from_prefix(
    conversation: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    """Return request-visible scalars so provenance never attributes them to a receipt."""

    values: set[str] = set()
    for message in conversation:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            values.add(content.strip())
    return frozenset(values)


def _call_projector(
    *,
    allowed_units: frozenset[str],
) -> Callable[[GroundedToolCall], GroundedNodeProjection]:
    def project(call: GroundedToolCall) -> GroundedNodeProjection:
        unit = EFFECT_UNIT_BY_TOOL.get(call.tool_name)
        # A failed write committed nothing, so it stays evidence and discharges
        # no obligation; the uncovered obligation is what ends the round.
        if unit is None or not call.succeeded:
            return GroundedNodeProjection(authority_label=AUTHORITY_EVIDENCE)
        if unit not in allowed_units:
            # Keep the effect visible so the compiler refuses it as outside the
            # frontier rather than silently accepting an unauthorized write.
            return GroundedNodeProjection(
                authority_label=AUTHORITY_EFFECT,
                semantic_effects=(unit,),
                effect_families=(_effect_family(unit),),
            )
        return GroundedNodeProjection(
            authority_label=AUTHORITY_EFFECT,
            semantic_effects=(unit,),
            effect_families=(_effect_family(unit),),
            covered_obligation_ids=(obligation_id_for_unit(unit),),
        )

    return project


def _terminal_with_last_call_offered(step: AuthoringStep) -> bool:
    return bool(
        step.terminal_response
        or step.terminal_response_template
        or step.terminal_response_bindings
    )


def _admit_terminal_with_last_call(
    *,
    step: AuthoringStep,
    calls: Sequence[GroundedToolCall],
    declarations: Sequence[GroundedDischargeDeclaration],
    allowed_units: frozenset[str],
    typed_row: Mapping[str, Any],
    prefix_conversation: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Admit a terminal piggyback only after its final receipt closes the residual."""

    if not calls or not calls[-1].succeeded:
        return "", "last_replica_call_failed"
    if _open_obligation_ids(
        allowed_units=allowed_units,
        calls=calls,
        declarations=declarations,
    ):
        return "", "open_obligations_remain"
    if argument_grounding_blockers(
        calls=calls,
        prefix_conversation=prefix_conversation,
        allowed_units=allowed_units,
        schema_defaults=schema_default_arguments(typed_row),
    ):
        return "", "terminal_graph_grounding_failed"

    static_response = step.terminal_response
    template = step.terminal_response_template
    bindings = step.terminal_response_bindings
    if static_response and template:
        return "", "terminal_response_mode_ambiguous"
    if static_response:
        if bindings:
            return "", "static_terminal_has_bindings"
        if len(static_response) > _MAX_TERMINAL_RESPONSE_CHARACTERS:
            return "", "terminal_response_too_long"
        return static_response, ""
    if not template:
        return "", "terminal_template_missing"
    if not _terminal_bindings_reference_selected_receipts(
        bindings=bindings,
        calls=calls,
    ):
        return "", "terminal_binding_invalid"
    if not _terminal_template_is_strict(template=template, bindings=bindings):
        return "", "terminal_template_invalid"

    tool_outputs = [
        call.result
        if isinstance(call.result, str)
        else json.dumps(
            call.result,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        for call in calls
    ]
    try:
        rendered = render_assistant_response_template(
            template=template,
            bindings=bindings,
            tool_outputs=tool_outputs,
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return "", "terminal_render_failed"
    if not rendered:
        return "", "terminal_render_empty"
    if len(rendered) > _MAX_TERMINAL_RESPONSE_CHARACTERS:
        return "", "terminal_response_too_long"
    return rendered, ""


def _terminal_bindings_reference_selected_receipts(
    *,
    bindings: Mapping[str, Any],
    calls: Sequence[GroundedToolCall],
) -> bool:
    """Accept only strict, non-fallback bindings to successful selected receipts."""

    if not isinstance(bindings, Mapping) or not bindings:
        return False
    allowed_keys = frozenset({"tool_output_index", "path", "round"})
    for name, binding in bindings.items():
        if not str(name) or not isinstance(binding, Mapping):
            return False
        if not set(binding).issubset(allowed_keys):
            return False
        index = binding.get("tool_output_index")
        if isinstance(index, bool) or not isinstance(index, int):
            return False
        if index < 0 or index >= len(calls) or not calls[index].succeeded:
            return False
        path = binding.get("path")
        if path is not None and (
            not isinstance(path, list)
            or any(not isinstance(part, (str, int)) for part in path)
        ):
            return False
        digits = binding.get("round")
        if digits is not None and (
            isinstance(digits, bool)
            or not isinstance(digits, int)
            or digits < 0
            or digits > 12
        ):
            return False
    return True


def _terminal_template_is_strict(
    *,
    template: str,
    bindings: Mapping[str, Any],
) -> bool:
    """Allow only simple named placeholders, each backed by one binding."""

    binding_names = {str(name) for name in bindings}
    if not binding_names or any(not name.isidentifier() for name in binding_names):
        return False
    seen: set[str] = set()
    try:
        parsed = string.Formatter().parse(template)
        for _literal, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if (
                not field_name.isidentifier()
                or field_name not in binding_names
                or format_spec
                or conversion is not None
            ):
                return False
            seen.add(field_name)
    except ValueError:
        return False
    return seen == binding_names


def _open_obligation_ids(
    *,
    allowed_units: frozenset[str],
    calls: Sequence[GroundedToolCall],
    declarations: Sequence[GroundedDischargeDeclaration],
) -> tuple[str, ...]:
    covered = {
        obligation_id_for_unit(unit)
        for call in calls
        if call.succeeded
        for unit in (EFFECT_UNIT_BY_TOOL.get(call.tool_name),)
        if unit is not None and unit in allowed_units
    }
    covered.update(item.obligation_id for item in declarations)
    return tuple(
        obligation_id_for_unit(unit)
        for unit in sorted(allowed_units)
        if obligation_id_for_unit(unit) not in covered
    )


def _authoring_context(
    *,
    typed_row: Mapping[str, Any],
    prefix_conversation: Sequence[Mapping[str, Any]],
    calls: Sequence[GroundedToolCall],
    allowed_sequence: tuple[str, ...],
    allowed_units: frozenset[str],
    feedback: Sequence[Mapping[str, Any]],
    round_index: int,
    declarations: Sequence[GroundedDischargeDeclaration] = (),
    declarations_accepted: bool = True,
    terminal_with_last_call: bool = False,
) -> dict[str, Any]:
    typed_suffix = typed_row.get("typed_suffix")
    context = {
        "surface": "toolsandbox",
        "round_index": int(round_index),
        "prefix_messages": [
            copy.deepcopy(dict(message))
            for message in prefix_conversation
            if isinstance(message, Mapping)
        ],
        "typed_suffix": authoring_suffix_view(typed_suffix)
        if isinstance(typed_suffix, Mapping)
        else {},
        "domain_instructions": [
            str(item)
            for item in (
                typed_suffix.get("domain_instructions")
                if isinstance(typed_suffix, Mapping)
                else ()
            )
            or ()
        ],
        "allowed_tool_sequence": list(allowed_sequence),
        "declaration_accepted": bool(declarations_accepted),
        "open_obligation_ids": list(
            _open_obligation_ids(
                allowed_units=allowed_units,
                calls=calls if terminal_with_last_call else (),
                declarations=declarations,
            )
        ),
        # Already declined; repeating one of these wastes a round.
        "declared_obligation_ids": [item.obligation_id for item in declarations],
        "observed_receipts": [
            {
                "node_id": call.node_id,
                "tool_name": call.tool_name,
                "arguments": dict(call.arguments),
                "result": call.result,
                "succeeded": call.succeeded,
            }
            for call in calls
        ],
        "verifier_feedback": [dict(item) for item in feedback],
    }
    if terminal_with_last_call:
        context["terminal_with_last_call"] = True
    return context


def _allowed_tool_sequence(typed_row: Mapping[str, Any]) -> tuple[str, ...]:
    typed_suffix = typed_row.get("typed_suffix")
    if not isinstance(typed_suffix, Mapping):
        return ()
    return tuple(
        str(name)
        for name in typed_suffix.get("suffix_tool_sequence") or ()
        if str(name)
    )


def _effect_units(typed_row: Mapping[str, Any]) -> tuple[str, ...]:
    binding = typed_row.get("frontier_binding")
    if not isinstance(binding, Mapping):
        return ()
    return tuple(
        dict.fromkeys(
            str(unit) for unit in binding.get("effect_units") or () if str(unit)
        )
    )


def _effect_family(unit: str) -> str:
    parts = unit.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else unit


def _receipt_result(receipt: Mapping[str, Any]) -> Any:
    if receipt.get("succeeded"):
        return receipt.get("result")
    return {
        "failed": True,
        "tool_call_exception": str(receipt.get("tool_call_exception") or ""),
    }


def _bridge_blocker(exc: ValueError) -> str:
    message = str(exc)
    if "leaves obligations uncovered" in message:
        return "grounded_obligation_uncovered"
    if "both realized and declared" in message:
        return "grounded_obligation_double_covered"
    if "unknown obligation" in message:
        return "grounded_declaration_unknown_obligation"
    return "grounded_bridge_rejected_proposal"


def _visible_request_text(conversation: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for message in conversation:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content)
    return "\n".join(parts)


def _normalized_text(text: str) -> str:
    return " ".join(_WORD_RE.findall(str(text).lower()))


def _text_is_grounded(
    value: str,
    *,
    produced_text: set[str],
    request_text: str,
) -> bool:
    candidate = _normalized_text(value)
    if not candidate:
        return True
    if candidate in request_text:
        return True
    return any(candidate in _normalized_text(item) for item in produced_text)


def _matches_schema_default(value: Any, default: Any) -> bool:
    if isinstance(value, bool) or isinstance(default, bool):
        return value is default
    if isinstance(value, (int, float)) and isinstance(default, (int, float)):
        return abs(float(value) - float(default)) <= _NUMERIC_MATCH_TOLERANCE
    return value == default


def _number_is_grounded(
    value: float,
    *,
    produced_numbers: Sequence[float],
    request_text: str,
) -> bool:
    if any(
        abs(value - produced) <= _NUMERIC_MATCH_TOLERANCE
        for produced in produced_numbers
    ):
        return True
    for token in _NUMBER_RE.findall(request_text):
        try:
            if abs(value - float(token)) <= _NUMERIC_MATCH_TOLERANCE:
                return True
        except ValueError:
            continue
    return False


def _absorb_produced_values(
    call: GroundedToolCall,
    *,
    produced_text: set[str],
    produced_numbers: list[float],
) -> None:
    """Record what this receipt makes available to later calls."""

    if not call.succeeded:
        return
    for value in _scalars(call.result):
        if isinstance(value, str):
            produced_text.add(value)
            produced_numbers.extend(
                float(token) for token in _NUMBER_RE.findall(value)
            )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            produced_numbers.append(float(value))


def _scalars(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _scalars(child)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _scalars(child)]
    if value is None or isinstance(value, bool):
        return []
    return [value]


__all__ = [
    "ACTION_DECLINE",
    "ACTION_TERMINAL",
    "ACTION_TOOL_CALL",
    "AUTHORITY_EFFECT",
    "AUTHORITY_EVIDENCE",
    "EFFECT_UNIT_BY_TOOL",
    "GRAPH_ID",
    "OBLIGATION_ID_PREFIX",
    "SCHEMA",
    "AuthoringOutcome",
    "AuthoringStep",
    "argument_grounding_blockers",
    "authoring_suffix_view",
    "build_grounded_bundle",
    "frontier_values_from_prefix",
    "obligation_id_for_unit",
    "run_grounded_authoring",
    "schema_default_arguments",
]
