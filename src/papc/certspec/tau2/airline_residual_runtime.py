from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from papc.certspec.cfrc_compiler import (
    CFRCCompileConfig,
    CFRCCompileResult,
    CFRCFrontier,
    compile_cfrc_suffix,
)
from papc.certspec.cfrc_ir import AuthorityLattice, EvidenceFact
from papc.certspec.contract_projection import (
    CompactTool,
    compact_successor_tools,
)
from papc.certspec.grounded_proposal import (
    GroundedNodeProjection,
    GroundedProposalTrace,
    GroundedToolCall,
    bridge_grounded_proposal,
)
from papc.certspec.runtime_ledger import RuntimeLedger
from papc.certspec.state_write_contract import Obligation, StateWriteContract
from papc.certspec.tau2.residual_support import (
    attach_metadata as _attach_metadata,
)
from papc.certspec.tau2.residual_support import (
    compact_visible_evidence,
    write_jsonl_append,
)
from papc.certspec.tau2.residual_support import (
    json_safe_value as _json_safe_tool_value,
)
from papc.certspec.tau2.residual_support import (
    mapping_or_none as _mapping_or_none,
)
from papc.certspec.tau2.residual_support import (
    message_attr as _message_attr,
)
from papc.certspec.tau2.residual_support import (
    message_attr_str as _message_attr_str,
)
from papc.certspec.tau2.residual_support import (
    message_tool_calls as _message_tool_calls,
)
from papc.certspec.tau2.residual_support import (
    optional_float as _optional_float,
)
from papc.certspec.tau2.residual_support import (
    runtime_ledger_for_messages as _runtime_ledger_for_messages,
)
from papc.certspec.tau2.residual_support import (
    tool_call_json as _tool_call_prompt_json,
)

AIRLINE_WRITE_EFFECTS = {
    "book_reservation": "booking",
    "cancel_reservation": "cancel",
    "send_certificate": "certificate",
    "update_reservation_baggages": "baggage",
    "update_reservation_flights": "flight",
    "update_reservation_passengers": "passenger",
}


TRANSFER_TOOL_NAME = "transfer_to_human_agents"
_TERMINAL_DECLARATION_TOOL = "cfrc_declare_completion"
_PREFIX_COMPLETION_INSTRUCTION = (
    "Fulfil every independently admissible part of the user's request. "
    "Policy permission, an entitlement, or a zero price does not establish "
    "that a requested state change has happened. Before saying no update is "
    "needed, check whether the visible state already satisfies that request. "
    "If it does not, propose the necessary allowed action after confirmation. "
    "Do not repeat an effect that the observed receipts already establish."
)
# The airline policy dictates this sentence word for word, so it is contract
# text rather than anything a model has to author. See the published domain
# policy: make the transfer call, then send exactly this to the user.
TRANSFER_POLICY_MESSAGE = (
    "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
)


# The published airline policy serialises actions by turn: "You should only make
# one tool call at a time, and if you make a tool call, you should not respond to
# the user simultaneously." The second clause is why a closing round exists at
# all; what a declared closure changes is who holds the message, not when the
# user reads it. The first clause is what makes the successor spend one
# round per action: measured over the base split, ninety-one percent of the
# actions it issues in a later round were already grounded before the graph
# opened, so those rounds buy a receipt nothing was waiting on.
_BATCHING_PROHIBITION = re.compile(
    r"You should only make one tool call at a time,\s*and\s*",
    re.IGNORECASE,
)
_BATCHING_RELAXATION = "You may make several tool calls at a time, and "


def relax_batching_prohibition(text: str) -> tuple[str, bool]:
    """Lift only the one-call-at-a-time clause, leaving the rest byte identical."""

    replaced, count = _BATCHING_PROHIBITION.subn(_BATCHING_RELAXATION, str(text or ""))
    return replaced, bool(count)


# The clause above forbids answering the user in the same turn as a tool call
# because the answer would precede the receipt. Under a declared closure the
# runtime holds any message written beside a call and delivers it only once
# Checked Execution has accepted every receipt that call produced, so what the
# user observes is unchanged: the acting turn still carries no message and the
# closing message still arrives after the receipts. Only the round that
# authored it is saved. The successor's copy of the clause is rewritten to say
# what the runtime actually does, and no other sentence moves.
# The prohibition is stated twice, once in each direction, and lifting only the
# first leaves the second forbidding exactly the behaviour the first now allows.
# Measured: with only the first rewritten, the successor wrote no message beside
# an action in any of the ninety-six graphs across both pools.
_SIMULTANEOUS_MESSAGE_PROHIBITION = re.compile(
    r"if you make a tool call, you should not respond to the user "
    r"simultaneously\.\s*If you respond to the user, you should not make a "
    r"tool call at the same time\.",
    re.IGNORECASE,
)
_HELD_MESSAGE_CLAUSE = (
    "if you make a tool call, any user-facing message you write beside it is "
    "held by the runtime and delivered only after the receipts of that call "
    "have been checked, so it never reaches the user simultaneously. You may "
    "therefore write the message that closes the request in the same turn as "
    "the calls that earn it."
)


def hold_simultaneous_message_clause(text: str) -> tuple[str, bool]:
    """Restate the simultaneity clause as the held-message rule the runtime obeys."""

    replaced, count = _SIMULTANEOUS_MESSAGE_PROHIBITION.subn(
        _HELD_MESSAGE_CLAUSE, str(text or "")
    )
    return replaced, bool(count)


# The callable-surface projection is shared core: one mechanism, one place.
# Airline keeps the historical private alias so its call sites stay unchanged.
_CompactTool = CompactTool


def count_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    return "4plus"


def read_count_bucket(value: int) -> str:
    if value <= 2:
        return "0_to_2"
    if value <= 5:
        return "3_to_5"
    return "6plus"


def effect_shape(nodes: Sequence[Mapping[str, str]]) -> str:
    if not nodes:
        return "none"
    return "+".join(str(node.get("effect") or "unknown") for node in nodes)


def extract_object_ref(args: Mapping[str, Any]) -> tuple[str, str]:
    for key, scope in (
        ("reservation_id", "reservation"),
        ("user_id", "user"),
        ("flight_number", "flight"),
    ):
        value = args.get(key)
        if value is not None and str(value).strip():
            return scope, str(value).strip()
    return "unknown", ""


RESIDUAL_AUTHORITY = "ell1"
RESIDUAL_OBLIGATION_ID = "airline_residual_completion"
REACHED_STATE_EVIDENCE_ID = "airline_reached_state"
DEFAULT_RESIDUAL_WRITE_BUDGET = 5
# The cap covers the whole interactive proposal, including clarification
# pauses. It is a safety bound, not a per-user-turn budget.
DEFAULT_MAX_SUCCESSOR_TURNS = 24
# The airline policy requires explicit user confirmation before a write, so a
# successor that reaches the frontier without it has to ask. Asking costs a
# user turn, and the open obligation is what the graph carries across it. The
# bound comes from the recorded fifty row run where the graph survived user
# turns: the deepest task that still completed needed four pauses, so a
# successor that has asked more often than that is not converging.
RESIDUAL_USER_TURN_BUDGET = 4
FRONTIER_REASON = "airline_first_write_or_transfer_graph_continuation"


@dataclass(frozen=True)
class AirlineResidualAdmission:
    execute: bool
    reason: str
    selected_tool_names: tuple[str, ...] = ()
    proposal_public: Mapping[str, Any] | None = None
    compiler_result: CFRCCompileResult | None = None
    diagnostics: Mapping[str, Any] | None = None

    def public_json(self) -> dict[str, Any]:
        selected_tool_names = list(self.selected_tool_names)
        selected_count = len(selected_tool_names)
        selected_write_names = [
            name for name in selected_tool_names if name in AIRLINE_WRITE_EFFECTS
        ]
        selected_read_names = [
            name
            for name in selected_tool_names
            if name not in AIRLINE_WRITE_EFFECTS and name != TRANSFER_TOOL_NAME
        ]
        selected_message_count = int(
            dict(self.proposal_public or {}).get("terminal_node_count") or 0
        )
        compiler = (
            self.compiler_result.to_json()
            if self.compiler_result is not None
            else {
                "schema": "papc.cfrc_compiler.result.v1",
                "decision": "abstain",
                "selected_node_ids": [],
                "selected_authority_label": "",
                "compile_errors": [],
            }
        )
        authority = str(compiler.get("selected_authority_label") or "")
        return {
            "schema": "papc.tau2.airline_residual_projection_result.v1",
            "execute": self.execute,
            "reason": self.reason,
            "selected_node_count": selected_count,
            "selected_node_id_hashes": [
                hashlib.sha256(f"airline_residual_{index}".encode("utf-8")).hexdigest()[:16]
                for index in range(selected_count)
            ],
            "selected_tool_names": selected_tool_names,
            "selected_read_tool_names": selected_read_names,
            "selected_message_count": selected_message_count,
            "selected_write_node_count": len(selected_write_names),
            "selected_read_node_count": len(selected_read_names),
            "selected_obligation_count": 1 if self.execute else 0,
            "abstain_reason_available": not self.execute,
            "certificate": {
                "schema": "papc.tau2.airline_residual_certificate_summary.v1",
                "authority_level": authority,
                "bounded_graph_continuation": True,
                "explicit_residual_obligation_count": 1,
                "public_safe": True,
            },
            "proposal": dict(self.proposal_public or {}),
            "projection": {
                "decision": "project_execute" if self.execute else "project_abstain",
                "selected_authority_label": authority if self.execute else "",
                "diagnostics": dict(self.diagnostics or {}),
                "compiler": {
                    **compiler,
                },
            },
        }


@dataclass(frozen=True)
class _PendingLiveCall:
    node_id: str
    call: Any
    tool_name: str
    arguments: Mapping[str, Any]
    proposal_round: int
    visible_receipt_refs: tuple[str, ...]


def register_tau2_airline_residual_agent(
    *,
    agent_name: str,
    strong_model: str,
    strong_llm_args: Mapping[str, Any],
    decision_log_path: Path,
    write_budget: int = DEFAULT_RESIDUAL_WRITE_BUDGET,
    max_strong_turns: int = DEFAULT_MAX_SUCCESSOR_TURNS,
    max_successor_rounds: int | None = None,
    frontier_activation_policy: str = "all_write_or_transfer",
    successor_batches_actions: bool = False,
    successor_tool_descriptions: bool = True,
    residual_recertifies_after_user_turn: bool = False,
    residual_frontier_rearms_after_close: bool = True,
    defer_frontier_until_confirmed: bool = False,
    successor_declares_closure: bool = False,
    request_scoped_obligations: bool = False,
    lean_residual_packet: bool = False,
    withhold_first_handoff: bool = False,
    prefix_completion_grounding: bool = False,
    generate_callback: Callable[..., Any] | None = None,
) -> str:
    if frontier_activation_policy not in FRONTIER_ACTIVATION_POLICIES:
        raise ValueError(
            "frontier_activation_policy must be one of "
            f"{FRONTIER_ACTIVATION_POLICIES}"
        )
    if int(write_budget) <= 0:
        raise ValueError("write_budget must be positive")
    if int(max_strong_turns) < 2:
        raise ValueError("max_strong_turns must allow a tool call and terminal")
    # How many rounds the successor gets, separately from how wide one round's
    # graph may be. The two were one number, so asking for fewer rounds also
    # narrowed the graph to that many tool calls, which is the opposite of what
    # a one-shot residual completion needs. Left unset the pair stays fused and
    # every registered result reproduces byte for byte.
    successor_rounds = (
        int(max_strong_turns) if max_successor_rounds is None else int(max_successor_rounds)
    )
    if successor_rounds < 1:
        raise ValueError("max_successor_rounds must allow at least one round")

    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        SystemMessage,
        UserMessage,
    )
    from tau2.registry import registry
    from tau2.utils.llm_utils import generate

    # A provider-zero gate drives the agent with a scripted callback. Online
    # the callback is absent, so this seam cannot change what a paid run does.
    # The OpenAI-compatible Gemini relay has one transport defect: on some
    # long tool transcripts it returns a zero-token ``stop`` response. tau2's
    # ``generate`` turns that response into a Pydantic validation error before
    # the agent can recover. Keep the ordinary request byte-identical, and only
    # retry that exact Gemini transport failure with a minimal validity reminder
    # and a non-zero temperature. No task data or answer sketch is introduced.
    if generate_callback is not None:
        generate_fn = generate_callback
    else:

        def generate_fn(
            *,
            model: str,
            messages: Sequence[Any],
            tools: Sequence[Any] | None = None,
            call_name: str | None = None,
            **kwargs: Any,
        ) -> Any:
            normalized_model = str(model or "").strip().lower()
            if normalized_model.startswith("openai/"):
                normalized_model = normalized_model[len("openai/") :]
            is_gemini = normalized_model.startswith("gemini-")

            class _EmptyGeminiAssistant(RuntimeError):
                """The relay returned a syntactically empty assistant turn."""

            def prepare_messages(
                source_messages: Sequence[Any],
            ) -> list[Any]:
                """Make prior tool turns explicit for the Gemini relay.

                The relay intermittently emits a zero-token stop when an
                assistant tool-call message carries ``content: null``. That is
                valid Chat Completions, but the relay's Gemini adapter is more
                reliable when the otherwise invisible turn has a short textual
                marker. The marker is sent only in the provider request; it is
                never appended to tau2's live transcript or shown to the user.
                """

                prepared = list(source_messages)
                for index, message_item in enumerate(prepared):
                    if _message_attr_str(message_item, "role") != "assistant":
                        continue
                    calls = _message_tool_calls(message_item)
                    if not calls or _message_attr_str(message_item, "content"):
                        continue
                    names = [
                        _message_attr_str(call, "name") or "tool"
                        for call in calls
                    ]
                    marker = "Tool call: " + ", ".join(names) + "."
                    if isinstance(message_item, Mapping):
                        replacement = dict(message_item)
                        replacement["content"] = marker
                    else:
                        replacement = deepcopy(message_item)
                        setattr(replacement, "content", marker)
                    prepared[index] = replacement
                return prepared

            def invoke(
                retry_messages: Sequence[Any],
                *,
                retry_name: str | None,
                retry_kwargs: Mapping[str, Any],
            ) -> Any:
                provider_generate = generate
                if normalized_model == "claude-haiku-4-5-20251001":
                    from papc.certspec.tau2.claude_native_generate import (
                        generate_claude_native,
                    )

                    provider_generate = generate_claude_native
                result = provider_generate(
                    model=model,
                    messages=list(retry_messages),
                    tools=list(tools) if tools is not None else None,
                    call_name=retry_name,
                    **dict(retry_kwargs),
                )
                if is_gemini and not (
                    _message_attr_str(result, "content")
                    or _message_tool_calls(result)
                ):
                    raise _EmptyGeminiAssistant(
                        "AssistantMessage must have either content or tool_calls"
                    )
                return result

            prepared_messages = prepare_messages(messages) if is_gemini else list(messages)

            try:
                return invoke(
                    prepared_messages,
                    retry_name=call_name,
                    retry_kwargs=kwargs,
                )
            except Exception as exc:
                if not is_gemini or (
                    "AssistantMessage must have either content or tool_calls"
                    not in str(exc)
                ):
                    raise

                last_error: Exception = exc
                retry_notes = (
                    "You must emit either a tool call or a user-facing message now.",
                    "Continue the tool-assisted conversation and emit one valid next response now.",
                )
                for attempt, note in enumerate(retry_notes, start=1):
                    retry_messages = list(prepared_messages)
                    for index, message_item in enumerate(retry_messages):
                        if _message_attr_str(message_item, "role") != "system":
                            continue
                        content = _message_attr_str(message_item, "content")
                        retry_messages[index] = SystemMessage(
                            role="system",
                            content=f"{content}\n{note}",
                        )
                        break
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["temperature"] = 0.5 + (0.2 * (attempt - 1))
                    seed = retry_kwargs.get("seed")
                    if isinstance(seed, int) and not isinstance(seed, bool):
                        retry_kwargs["seed"] = seed + attempt
                    retry_name = (
                        f"{call_name}_gemini_empty_retry{attempt}"
                        if call_name
                        else f"gemini_empty_retry{attempt}"
                    )
                    try:
                        return invoke(
                            retry_messages,
                            retry_name=retry_name,
                            retry_kwargs=retry_kwargs,
                        )
                    except Exception as retry_exc:
                        if (
                            "AssistantMessage must have either content or tool_calls"
                            not in str(retry_exc)
                        ):
                            raise
                        last_error = retry_exc
                raise last_error

    def _compact_successor_tools(tools: Sequence[Any]) -> list[Any]:
        return compact_successor_tools(
            tools, keep_tool_description=successor_tool_descriptions
        )

    def _successor_system_messages(messages: Sequence[Any]) -> list[Any]:
        """The successor's copy of the domain policy.

        Identical to the agent's own copy unless this arm rewrites the batching
        clause or the simultaneity clause, in which case exactly that clause
        moves and nothing else does. The cheap prefix and every baseline keep
        the published text.
        """

        if not (successor_batches_actions or successor_declares_closure):
            return list(messages)
        rewritten: list[Any] = []
        for message in messages:
            content = _message_attr_str(message, "content")
            replacement = content
            changed = False
            if successor_batches_actions:
                replacement, batched = relax_batching_prohibition(replacement)
                changed = changed or batched
            if successor_declares_closure:
                replacement, held = hold_simultaneous_message_clause(replacement)
                changed = changed or held
            if not changed:
                rewritten.append(message)
                continue
            rewritten.append(SystemMessage(role="system", content=replacement))
        return rewritten

    def _prefix_system_messages(messages: Sequence[Any]) -> list[Any]:
        # Keep completion claims tied to observed state before a write candidate
        # opens the frontier. The published policy and native tools stay intact.
        if not prefix_completion_grounding:
            return list(messages)
        return [*messages, SystemMessage(
            role="system", content=_PREFIX_COMPLETION_INSTRUCTION,
        )]

    class AirlineResidualAgent(LLMAgent):
        def __init__(self, *agent_args: Any, task: Any = None, **kwargs: Any):
            super().__init__(*agent_args, **kwargs)
            self._task = task
            self._residual_opened = False
            self._graph_active = False
            self._graph_frontier: CFRCFrontier | None = None
            self._graph_compile_config: CFRCCompileConfig | None = None
            self._frontier_cheap_message: Any | None = None
            self._frontier_candidate_families: dict[str, int] = {}
            self._grounded_calls: list[GroundedToolCall] = []
            self._pending_live_calls: list[_PendingLiveCall] = []
            self._intercepted_strong_messages: list[Any] = []
            self._graph_turn_count = 0
            self._successor_tool_cache: list[Any] | None = None
            self._graph_write_count = 0
            self._graph_receipt_error_count = 0
            self._declared_closure_text: str | None = None
            self._declared_closure_released = False
            self._residual_awaiting_user_turn = False
            self._residual_user_turn_count = 0
            self._residual_pause_total = 0
            self._frontier_cheap_accounted = False
            self._frontier_deferred_once = False
            self._handoff_withheld_once = False
            self._handoff_withheld_count = 0
            self._frontier_deferral_count = 0
            self._held_cheap_messages: list[Any] = []
            self._request_obligations: dict[str, Any] | None = None
            self._residual_contract_context: dict[str, Any] = {}
            self._last_successor_text_status = "none"
            self._cheap_call_count = 0
            self._strong_call_count = 0

        def _generate_next_message(self, message: Any, state: Any):
            if isinstance(message, UserMessage) and message.is_audio:
                raise ValueError(
                    "User message cannot be audio. Use VoiceLLMAgent instead."
                )
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)

            if self._residual_awaiting_user_turn and not isinstance(message, UserMessage):
                # A pause is only ever answered by a user turn. Anything else
                # means the pause will not be answered, and the decision row
                # for it is already written, so the graph is given up here.
                self._clear_incremental_graph()

            if self._residual_awaiting_user_turn and isinstance(message, UserMessage):
                # The obligation the successor paused on is still open, and the
                # user has now answered the question it paused to ask. The
                # frontier, the forbidden writes and the receipts already bound
                # into the graph are unchanged, so the successor resumes there
                # rather than handing an open residual back to the cheap model.
                self._residual_awaiting_user_turn = False
                self._graph_active = True
                return self._residual_next_message(
                    state,
                    resumed_after_user_turn=True,
                )

            if self._graph_active:
                if self._pending_live_calls:
                    receipt_error = self._bind_live_receipts(message)
                    if receipt_error:
                        return self._stop_incremental_graph(
                            reason=receipt_error,
                            user_message=(
                                "I could not safely verify the airline tool "
                                "result, so I stopped before continuing."
                            ),
                        )
                # The transfer sentence is contract text and wins over anything
                # the successor authored for the same turn.
                closed = self._close_on_policy_literal()
                if closed is None:
                    closed = self._close_on_declared_closure()
                if closed is not None:
                    return closed
                return self._residual_next_message(state)

            cheap_message = generate_fn(
                model=self.llm,
                tools=self.tools,
                messages=_prefix_system_messages(state.system_messages) + state.messages,
                call_name="airline_residual_cheap_prefix",
                **self.llm_args,
            )
            self._cheap_call_count += 1
            if (
                withhold_first_handoff
                and not self._handoff_withheld_once
                and not self._residual_opened
                and not _opens_frontier(
                    cheap_message, policy=frontier_activation_policy
                )
                and any(
                    _message_attr_str(call, "name") == TRANSFER_TOOL_NAME
                    for call in _message_tool_calls(cheap_message)
                )
            ):
                # Recheck every first non-commitment hand-off once. Its answer
                # is still a prefix candidate: it must pass the same budget,
                # confirmation and frontier checks as the original generation.
                self._handoff_withheld_once = True
                self._handoff_withheld_count += 1
                self._held_cheap_messages.append(cheap_message)
                cheap_message = generate_fn(
                    model=self.llm,
                    tools=self.tools,
                    messages=(
                        _prefix_system_messages(state.system_messages)
                        + state.messages
                        + [UserMessage.text(_WITHHELD_HANDOFF_INSTRUCTION)]
                    ),
                    call_name="airline_residual_cheap_handoff_recheck",
                    **self.llm_args,
                )
                self._cheap_call_count += 1

            opens_frontier = (
                not self._residual_opened
                and _opens_frontier(cheap_message, policy=frontier_activation_policy)
            )
            if (
                opens_frontier
                and self._strong_call_count >= successor_rounds
            ):
                # The episode has spent its successor budget and this is an
                # uncertified write. Returning the candidate would put it in
                # the domain with no contract, no graph and no checked
                # execution, which is the single thing the frontier exists to
                # stop; a budget is a reason to refuse, never a reason to let
                # something through unchecked. The method's failure return is
                # a refusal, so the request ends unsatisfied and says so.
                self._held_cheap_messages.append(cheap_message)
                self._settle_held_cheap_messages(
                    reason="successor_budget_spent_before_frontier",
                    candidate=cheap_message,
                )
                return AssistantMessage.text(
                    "I could not safely complete the airline request within "
                    "the bounded residual budget.",
                    raw_data={
                        "papc_cfrc": {
                            "schema": "papc.tau2.airline_frontier_refusal.v1",
                            "reason": "successor_budget_spent_before_frontier",
                            "fail_closed_before_next_live_write": True,
                            "public_safe": True,
                        }
                    },
                )
            if opens_frontier:
                if (
                    defer_frontier_until_confirmed
                    and not self._frontier_deferred_once
                    and _write_confirmation_is_open(state.messages)
                ):
                    # The candidate is a write the policy cannot admit yet, so
                    # the residual behind it is not executable and there is
                    # nothing for the successor to complete. The turn the state
                    # is missing is a conversational one, which the prefix model
                    # owns, so it is asked for here and the frontier is
                    # certified on the next write candidate instead. The
                    # deferral is taken at most once per episode, so a prefix
                    # that keeps proposing writes still reaches the successor.
                    self._frontier_deferred_once = True
                    self._frontier_deferral_count += 1
                    confirmation_message = generate_fn(
                        model=self.llm,
                        tools=self.tools,
                        messages=(
                            _prefix_system_messages(state.system_messages)
                            + state.messages
                            + [UserMessage.text(_DEFERRED_FRONTIER_INSTRUCTION)]
                        ),
                        call_name="airline_residual_cheap_confirmation",
                        **self.llm_args,
                    )
                    self._cheap_call_count += 1
                    if not _assistant_has_tool_calls(
                        confirmation_message
                    ) and _message_attr_str(confirmation_message, "content"):
                        self._held_cheap_messages.append(cheap_message)
                        self._settle_held_cheap_messages(
                            reason="frontier_waiting_for_confirmation",
                            candidate=cheap_message,
                        )
                        return confirmation_message
                    # The prefix did not produce the confirmation turn, so the
                    # candidate stands and the frontier is certified now. The
                    # message just paid for never reaches the transcript, so it
                    # is carried here to be billed with the decision row.
                    self._held_cheap_messages.append(confirmation_message)
                self._residual_opened = True
                return self._open_incremental_graph(
                    state=state,
                    cheap_frontier_message=cheap_message,
                )
            self._settle_held_cheap_messages(
                reason="handoff_withheld_no_frontier", candidate=cheap_message
            )
            return cheap_message

        def _open_incremental_graph(
            self,
            *,
            state: Any,
            cheap_frontier_message: Any,
        ):
            self._graph_active = True
            self._graph_frontier = _airline_frontier_from_prefix(state.messages)
            self._request_obligations = self._build_request_obligations(state)
            self._graph_compile_config = _airline_compile_config(
                graph_tool_call_budget=max_strong_turns - 1
            )
            self._frontier_cheap_message = cheap_frontier_message
            # What KIND of candidate opened this frontier. The summary is
            # already computed and already shipped to the successor, but it was
            # never recorded, and without it the decision log cannot say whether
            # a frontier was opened by a write candidate or by a terminal
            # escalation. That distinction decides a method question -- whether
            # a hand-off to a human is a commitment at all -- and today it can
            # only be answered by rerunning. The histogram carries one of four
            # labels per call and no arguments, object ids or task ids, so it is
            # public-safe on the same footing as the copy the model already
            # receives.
            self._frontier_candidate_families = dict(
                _cheap_frontier_candidate_summary(cheap_frontier_message).get(
                    "tool_family_counts"
                )
                or {}
            )
            self._grounded_calls = []
            self._pending_live_calls = []
            self._intercepted_strong_messages = []
            self._graph_turn_count = 0
            self._graph_write_count = 0
            self._graph_receipt_error_count = 0
            self._declared_closure_text: str | None = None
            self._residual_awaiting_user_turn = False
            self._residual_user_turn_count = 0
            self._frontier_cheap_accounted = False
            return self._residual_next_message(
                state,
                cheap_frontier_message=cheap_frontier_message,
            )

        def _build_request_obligations(self, state: Any) -> dict[str, Any] | None:
            """What the visible request still owes, enumerated once per frontier.

            Off by default, so every registered arm keeps the ledger it was
            measured on. It fails open: a builder call that does not parse
            leaves the successor with exactly the packet it had before.
            """

            if not request_scoped_obligations:
                return None
            report = airline_request_obligations(
                generate_fn=generate_fn,
                model=self.llm,
                llm_args=self.llm_args,
                messages=state.messages,
                system_messages=state.system_messages,
                tools=self.tools,
                visible_evidence=compact_visible_evidence(
                    state.messages,
                    # This builder gets no parallel copy of the transcript.
                    # Truncating a receipt can remove the very value needed to
                    # ground an obligation; keep the visible prefix in full.
                    max_events=len(state.messages),
                    max_text_chars=max(
                        (len(str(_message_attr(item, "content") or ""))
                         for item in state.messages),
                        default=0,
                    ),
                    include_assistant_text=True,
                    user_requests_carried_verbatim=True,
                ),
                system_message_cls=SystemMessage,
                user_message_cls=UserMessage,
            )
            self._cheap_call_count += 1
            builder_message = report.pop("builder_message", None)
            if builder_message is not None:
                # The builder answer never reaches the transcript, so this list
                # is the only carrier its tokens ever reach the ledger through.
                self._held_cheap_messages.append(builder_message)
            return report

        def _residual_next_message(
            self,
            state: Any,
            *,
            cheap_frontier_message: Any | None = None,
            resumed_after_user_turn: bool = False,
        ):
            if self._graph_turn_count >= successor_rounds:
                return self._stop_incremental_graph(
                    reason="incremental_graph_turn_cap_reached",
                    user_message=(
                        "I could not safely finish the airline request within "
                        "the bounded graph."
                    ),
                )
            prompt = (
                _residual_prompt_payload(
                    visible_evidence=compact_visible_evidence(
                        state.messages,
                        max_events=40,
                        exclude_tool_receipt_ids=_carried_tool_receipt_ids(
                            state.messages
                        ),
                        # This request replays the whole transcript below, so
                        # the user turns arrive verbatim and the packet keeps
                        # only their place in the ordering.
                        user_requests_carried_verbatim=True,
                    ),
                    cheap_frontier_message=cheap_frontier_message,
                    prefix_trace=_trace_from_messages(state.messages),
                    write_budget=write_budget,
                    used_write_count=self._graph_write_count,
                    pending_carrier_audit=True,
                    declared_closure=successor_declares_closure,
                    request_obligations=self._request_obligations,
                    lean_packet=lean_residual_packet,
                )
                if self._graph_turn_count == 0
                else _residual_step_reminder(
                    write_budget=write_budget,
                    used_write_count=self._graph_write_count,
                    resumed_after_user_turn=resumed_after_user_turn,
                    declared_closure=successor_declares_closure,
                )
            )
            if self._graph_turn_count == 0:
                # Provider calls are stateless and the frontier packet is not
                # part of the live transcript. Retain its constraints for every
                # extension, including one resumed after user confirmation.
                # Evidence and budgets stay live; replaying their initial values
                # here would substitute the frontier for the reached state.
                self._residual_contract_context = {
                    key: prompt[key]
                    for key in (
                        "compiler_rules",
                        "pending_carrier_audit",
                        "request_obligations",
                    )
                    if key in prompt
                }
            else:
                prompt.update(self._residual_contract_context)
            strong_message = generate_fn(
                model=strong_model,
                tools=self._successor_tools(),
                messages=[
                    SystemMessage(
                        role="system", content=_SUCCESSOR_SYSTEM_PROMPT
                    ),
                    *_successor_system_messages(state.system_messages),
                    *state.messages,
                    UserMessage.text(
                        json.dumps(
                            prompt,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                ],
                call_name="airline_residual_strong_next",
                **dict(strong_llm_args),
            )
            self._strong_call_count += 1
            self._graph_turn_count += 1
            # Every strong generation is accounted here, not only the ones a
            # stop path intercepts. A message the successor returns into the
            # live transcript arrives there without provider usage attached, so
            # this list is the only carrier its tokens and cost ever reach.
            self._intercepted_strong_messages.append(strong_message)
            calls = list(_message_tool_calls(strong_message))
            declarations = [
                call for call in calls
                if _message_attr_str(call, "name") == _TERMINAL_DECLARATION_TOOL
            ] if successor_declares_closure else []
            if declarations:
                ids = [_message_attr_str(call, "id") for call in calls]
                arguments = _tool_call_prompt_json(declarations[0])["arguments"]
                text = arguments.get("message")
                if (
                    len(declarations) != 1 or not all(ids) or len(set(ids)) != len(ids)
                    or set(arguments) != {"message"}
                    or not isinstance(text, str) or not text.strip()
                ):
                    return self._stop_incremental_graph(
                        reason="incremental_graph_terminal_declaration_invalid",
                        user_message="I could not safely interpret the completion declaration.",
                    )
                calls = [call for call in calls if call is not declarations[0]]
                # A terminal vertex is proposal data, never a domain call. It
                # has no fabricated receipt and never enters native history.
                strong_message = deepcopy(strong_message)
                strong_message.tool_calls = calls or None
                strong_message.content = json.dumps({
                    "message": text, "residual_status": "complete",
                }, ensure_ascii=False)
                if not calls and not any(
                    call.succeeded and call.tool_name in AIRLINE_WRITE_EFFECTS
                    for call in self._grounded_calls
                ):
                    return self._stop_incremental_graph(
                        reason="incremental_graph_terminal_declaration_without_write",
                        user_message="The remaining airline change has not been completed.",
                    )
            if not calls:
                if not _message_attr_str(strong_message, "content"):
                    return self._stop_incremental_graph(
                        reason="incremental_graph_empty_model_output",
                        user_message=(
                            "I could not safely continue because the airline "
                            "model returned no action or message."
                        ),
                    )
                return self._close_incremental_graph(strong_message)

            declared_closure = (
                _message_attr_str(strong_message, "content").strip()
                if successor_declares_closure
                else ""
            )
            if declared_closure:
                # The message is authored beside the actions and withheld here.
                # Nothing downstream may read it as sent: it is released only
                # once every receipt those actions produce has been accepted,
                # and discarded otherwise. Stripping it from the emitted message
                # is what keeps the acting turn silent to the user.
                self._declared_closure_text = declared_closure
                strong_message = _strip_message_content(strong_message)

            call_ids = [_message_attr_str(call, "id") for call in calls]
            if not all(call_ids) or len(set(call_ids)) != len(call_ids):
                return self._stop_incremental_graph(
                    reason="incremental_graph_call_ids_invalid",
                    user_message=(
                        "I could not safely identify the proposed airline "
                        "calls, so no new action was taken."
                    ),
                )
            available = {str(getattr(tool, "name", "")) for tool in self.tools}
            if any(_message_attr_str(call, "name") not in available for call in calls):
                return self._stop_incremental_graph(
                    reason="incremental_graph_tool_not_available",
                    user_message=(
                        "I could not safely continue because a proposed airline "
                        "action was unavailable."
                    ),
                )
            attempted_writes = sum(
                1
                for call in calls
                if _message_attr_str(call, "name") in AIRLINE_WRITE_EFFECTS
            )
            if self._graph_write_count + attempted_writes > int(write_budget):
                return self._stop_incremental_graph(
                    reason="residual_write_budget_exceeded",
                    user_message=(
                        "I could not safely complete the remaining airline "
                        "actions within the bounded graph."
                    ),
                )
            if (
                len(self._grounded_calls) + len(self._pending_live_calls) + len(calls)
                > int(max_strong_turns) - 1
            ):
                return self._stop_incremental_graph(
                    reason="incremental_graph_tool_call_budget_exceeded",
                    user_message=(
                        "I could not safely complete the remaining airline "
                        "actions within the graph node budget."
                    ),
                )

            visible_receipts = tuple(
                grounded.receipt_ref for grounded in self._grounded_calls
            )
            pending = [
                _PendingLiveCall(
                    node_id=(f"airline_call_{len(self._grounded_calls) + index:03d}"),
                    call=deepcopy(call),
                    tool_name=_message_attr_str(call, "name"),
                    arguments=dict(_tool_call_prompt_json(call).get("arguments") or {}),
                    proposal_round=self._graph_turn_count,
                    visible_receipt_refs=visible_receipts,
                )
                for index, call in enumerate(calls, start=1)
            ]
            provisional_calls = [
                *self._grounded_calls,
                *[
                    GroundedToolCall(
                        node_id=row.node_id,
                        tool_name=row.tool_name,
                        arguments=row.arguments,
                        result={"pending_live_receipt": row.node_id},
                        proposal_round=row.proposal_round,
                        visible_receipt_refs=row.visible_receipt_refs,
                    )
                    for row in pending
                ],
            ]
            if any(
                call.tool_name in AIRLINE_WRITE_EFFECTS
                or call.tool_name == TRANSFER_TOOL_NAME
                for call in provisional_calls
            ):
                try:
                    provisional_result, _ = self._compile_graph(
                        calls=provisional_calls,
                        terminal_text="pending incremental graph closure",
                    )
                except (KeyError, TypeError, ValueError):
                    provisional_result = None
                if provisional_result is None or not provisional_result.execute:
                    return self._stop_incremental_graph(
                        reason="incremental_graph_extension_rejected",
                        user_message=(
                            "I could not safely admit the next airline graph "
                            "extension, so no new action was taken."
                        ),
                    )
            self._pending_live_calls = pending
            self._graph_write_count += attempted_writes
            return _attach_metadata(
                strong_message,
                {
                    "schema": "papc.tau2.airline_incremental_graph_extension.v1",
                    "phase": "checked_graph_extension",
                    "admitted": True,
                    "node_count": len(pending),
                    "dependency_count": sum(
                        len(row.visible_receipt_refs) for row in pending
                    ),
                    "runtime_role": "strong",
                    "runtime_usage_accounted_in_transcript": True,
                    "public_safe": True,
                },
            )

        def _bind_live_receipts(self, incoming_message: Any) -> str:
            receipts = list(
                getattr(incoming_message, "tool_messages", None) or [incoming_message]
            )
            pending = list(self._pending_live_calls)
            self._pending_live_calls = []
            if len(receipts) != len(pending):
                self._graph_receipt_error_count += 1
                return "incremental_graph_receipt_count_mismatch"
            by_id = {_message_attr_str(receipt, "id"): receipt for receipt in receipts}
            expected_ids = {_message_attr_str(row.call, "id") for row in pending}
            if len(by_id) != len(receipts) or set(by_id) != expected_ids:
                self._graph_receipt_error_count += 1
                return "incremental_graph_receipt_mismatch"
            # Validate the entire batch before recording any receipt. Native
            # parallel tools may finish in a different order from their calls.
            for row in pending:
                receipt = by_id[_message_attr_str(row.call, "id")]
                refused = bool(_message_attr(receipt, "error"))
                if refused:
                    # A tool that refused the call changed nothing, so its
                    # receipt is evidence the successor can act on rather than
                    # a reason to abandon the graph. The write budget stays
                    # consumed, which keeps the repair bounded.
                    self._graph_receipt_error_count += 1
                self._grounded_calls.append(
                    GroundedToolCall(
                        node_id=row.node_id,
                        tool_name=row.tool_name,
                        arguments=row.arguments,
                        result=_receipt_result_value(_message_attr(receipt, "content")),
                        proposal_round=row.proposal_round,
                        succeeded=not refused,
                        visible_receipt_refs=row.visible_receipt_refs,
                    )
                )
            return ""

        def _compile_graph(
            self,
            *,
            calls: Sequence[GroundedToolCall],
            terminal_text: str,
        ) -> tuple[AirlineResidualAdmission, Mapping[str, Any]]:
            if self._graph_frontier is None or self._graph_compile_config is None:
                raise RuntimeError("incremental graph contract is not active")
            trace = GroundedProposalTrace(
                calls=tuple(calls),
                terminal_response=terminal_text,
                terminal_visible_receipt_refs=tuple(call.receipt_ref for call in calls),
                failed_proposal_call_count=self._graph_receipt_error_count,
            )
            bundle = bridge_grounded_proposal(
                trace=trace,
                graph_id="airline_incremental_receipt_graph",
                residual_obligation_ids=(RESIDUAL_OBLIGATION_ID,),
                evidence_fact_ids=(REACHED_STATE_EVIDENCE_ID,),
                terminal_authority_label="ell0",
                project_call=_project_airline_call,
            )
            compiler_result = compile_cfrc_suffix(
                frontier=self._graph_frontier,
                proposal_graph=bundle.proposal_graph,
                bindings=bundle.bindings,
                config=self._graph_compile_config,
            )
            selected = tuple(compiler_result.selected_node_ids)
            whole_graph_selected = (
                compiler_result.decision == "execute"
                and len(selected) == len(bundle.proposal_graph.nodes)
                and set(selected)
                == {node.node_id for node in bundle.proposal_graph.nodes}
            )
            proposal_public = {
                "schema": "papc.tau2.airline_incremental_receipt_graph.v1",
                **bundle.public_summary,
                "incremental_live_receipt_grounding": True,
                "private_reached_state_replica": False,
                "model_authored_relations": False,
                "receipt_derived_dependencies": True,
                "terminal_node_count": 1,
                "whole_graph_selected": whole_graph_selected,
                "graph": bundle.proposal_graph.to_json(),
            }
            result = AirlineResidualAdmission(
                execute=whole_graph_selected,
                reason=(
                    "compiled_airline_incremental_receipt_graph"
                    if whole_graph_selected
                    else "incremental_graph_compiler_rejected"
                ),
                selected_tool_names=(
                    tuple(call.tool_name for call in calls)
                    if whole_graph_selected
                    else ()
                ),
                proposal_public=proposal_public,
                compiler_result=compiler_result,
                diagnostics={
                    **self._graph_diagnostics(),
                    "successful_tool_result_count": len(calls),
                    "write_budget": int(write_budget),
                    "strong_turn_cap": successor_rounds,
                    "graph_tool_call_budget": int(max_strong_turns) - 1,
                    "closing_round_saved_by_declared_closure": (
                        self._declared_closure_released
                    ),
                    "public_safe": True,
                },
            )
            return result, bundle.public_summary

        def _close_incremental_graph(self, strong_message: Any):
            projection_start = time.perf_counter()
            try:
                status, strong_message = _successor_text_control(strong_message)
            except ValueError:
                self._last_successor_text_status = "invalid"
                return self._stop_incremental_graph(
                    reason="incremental_graph_text_status_invalid",
                    user_message="I could not safely interpret the airline continuation, so I stopped.",
                )
            self._last_successor_text_status = status or "legacy_unlabelled"
            if status == "blocked":
                return self._stop_incremental_graph(
                    reason="incremental_graph_declared_unfinished",
                    user_message=_message_attr_str(strong_message, "content"),
                )
            if status == "awaiting_user":
                if self._can_await_user_turn():
                    return self._await_user_turn(
                        strong_message, projection_start=projection_start,
                    )
                return self._stop_incremental_graph(
                    reason="incremental_graph_user_turn_budget_spent",
                    user_message=(
                        "The remaining airline request still needs your input, "
                        "but I could not continue within the bounded residual budget."
                    ),
                )
            mutating = [
                call
                for call in self._grounded_calls
                if call.succeeded
                and (
                    call.tool_name in AIRLINE_WRITE_EFFECTS
                    or call.tool_name == TRANSFER_TOOL_NAME
                )
            ]
            if mutating:
                try:
                    result, _ = self._compile_graph(
                        calls=self._grounded_calls,
                        terminal_text=_message_attr_str(strong_message, "content"),
                    )
                except (KeyError, TypeError, ValueError):
                    return self._stop_incremental_graph(
                        reason="incremental_graph_final_closure_rejected",
                        user_message=(
                            "I could not safely close the airline graph after "
                            "the observed receipts."
                        ),
                    )
            elif status is None and self._can_await_user_turn():
                return self._await_user_turn(
                    strong_message,
                    projection_start=projection_start,
                )
            else:
                result = AirlineResidualAdmission(
                    execute=False,
                    reason="incremental_graph_terminal_without_mutating_residual",
                    diagnostics=self._graph_diagnostics(),
                )
            self._write_graph_decision(
                result=result,
                projection_seconds=time.perf_counter() - projection_start,
            )
            metadata = {
                **result.public_json(),
                "incremental_graph_closed": result.execute,
                "residual_awaiting_user_turn": False,
                "hand_back_to_cheap_control": not result.execute,
                "runtime_role": "strong",
                "runtime_usage_accounted_in_transcript": True,
            }
            self._clear_incremental_graph()
            return _attach_metadata(strong_message, metadata)

        def _can_await_user_turn(self) -> bool:
            # Parking is only worth a user turn if the successor could still act
            # afterwards, so both the pause budget and the turn cap have to
            # leave room for the write the residual is waiting on.
            return (
                self._residual_pause_total < RESIDUAL_USER_TURN_BUDGET
                and self._graph_turn_count < successor_rounds - 1
                and self._graph_write_count < int(write_budget)
            )

        def _await_user_turn(self, strong_message: Any, *, projection_start: float):
            result = AirlineResidualAdmission(
                execute=False,
                reason="residual_awaiting_user_turn",
                diagnostics={
                    **self._graph_diagnostics(),
                    "user_turn_count": self._residual_pause_total + 1,
                    "user_turn_budget": RESIDUAL_USER_TURN_BUDGET,
                },
            )
            self._write_graph_decision(
                result=result,
                projection_seconds=time.perf_counter() - projection_start,
            )
            self._residual_pause_total += 1
            if residual_recertifies_after_user_turn:
                # The successor has authority over an executable residual, not
                # over the conversation that surrounds it. A pause means the
                # residual is not actionable until the user answers, so the
                # floor returns to the prefix model and the frontier is
                # certified again from the reached state the answer produces.
                self._clear_incremental_graph()
                self._residual_opened = False
                return _attach_metadata(
                    strong_message,
                    {
                        **result.public_json(),
                        "incremental_graph_closed": False,
                        "residual_awaiting_user_turn": False,
                        "hand_back_to_cheap_control": True,
                        "residual_recertifies_after_user_turn": True,
                        "runtime_role": "strong",
                        "runtime_usage_accounted_in_transcript": True,
                    },
                )
            self._graph_active = False
            self._residual_awaiting_user_turn = True
            self._residual_user_turn_count += 1
            self._intercepted_strong_messages = []
            return _attach_metadata(
                strong_message,
                {
                    **result.public_json(),
                    "incremental_graph_closed": False,
                    "residual_awaiting_user_turn": True,
                    "hand_back_to_cheap_control": False,
                    "runtime_role": "strong",
                    "runtime_usage_accounted_in_transcript": True,
                },
            )

        def _graph_diagnostics(self) -> dict[str, Any]:
            return {
                "schema": "papc.tau2.airline_incremental_graph_diagnostics.v1",
                "incremental_live_receipt_grounding": True,
                "strong_turn_count": self._graph_turn_count,
                "write_count": self._graph_write_count,
                "successful_tool_result_count": len(self._grounded_calls),
                "failed_tool_result_count": self._graph_receipt_error_count,
                "successor_text_status": self._last_successor_text_status,
                "request_contract_status": (
                    str(self._request_obligations.get("status") or "unavailable")
                    if self._request_obligations else "disabled"
                ),
                "request_contract_obligation_count": len(
                    (self._request_obligations or {}).get("obligations") or ()
                ),
                "closing_round_saved_by_declared_closure": (
                    self._declared_closure_released
                ),
                "public_safe": True,
            }

        def _successor_tools(self):
            if self._successor_tool_cache is None:
                self._successor_tool_cache = _compact_successor_tools(self.tools)
                if successor_declares_closure:
                    self._successor_tool_cache.append(CompactTool(
                        name=_TERMINAL_DECLARATION_TOOL,
                        openai_schema={"type": "function", "function": {
                            "name": _TERMINAL_DECLARATION_TOOL,
                            "description": (
                                "Declare the terminal node of the current residual graph. "
                                "Include this in the SAME batch as the final ready actions "
                                "when they discharge the entire remaining request. This "
                                "performs no domain action and sends no immediate message: "
                                "Checked Execution releases the message only after all "
                                "receipts pass. Use only already observed or confirmed "
                                "details; omit this declaration when a receipt must supply "
                                "a new value or a further action or user choice is needed."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {"message": {"type": "string"}},
                                "required": ["message"],
                                "additionalProperties": False,
                            },
                        }},
                    ))
            return self._successor_tool_cache

        def _close_on_declared_closure(self):
            """Release the closing message the successor wrote beside its actions.

            The message is held from the moment it is authored. Checked
            Execution is what releases it: every call the graph issued has to
            have returned, and every receipt has to have been accepted. One
            refusal, one mismatch, or one call still outstanding and the message
            is discarded and the successor keeps the closing round it would
            otherwise have saved, because the state it described is not the
            state that was reached.

            A message held beside a read is discarded for the same reason: the
            residual requires a state change, so a graph carrying no successful
            write or transfer has not reached the state any closing message
            could describe.
            """

            self._declared_closure_released = False
            text = self._declared_closure_text
            if not text:
                return None
            self._declared_closure_text = None
            if self._pending_live_calls or self._graph_receipt_error_count:
                return None
            if not self._grounded_calls or not all(
                call.succeeded for call in self._grounded_calls
            ):
                return None
            if not any(
                call.tool_name in AIRLINE_WRITE_EFFECTS
                or call.tool_name == TRANSFER_TOOL_NAME
                for call in self._grounded_calls
            ):
                return None
            try:
                status, _ = _successor_text_control(AssistantMessage.text(text))
            except ValueError:
                return None
            if status is None:
                # Text authored before receipts cannot inherit the legacy
                # post-receipt inference that any successful write closes the
                # request. A question or narration must keep the live round.
                return None
            self._declared_closure_released = True
            return self._close_incremental_graph(AssistantMessage.text(text))

        def _close_on_policy_literal(self):
            """Close a completed transfer with the sentence the policy dictates.

            The airline policy names the transfer message word for word: make
            the ``transfer_to_human_agents`` call, then send
            ``YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.`` So a
            strong round spent after a successful transfer would only reproduce
            text the policy already fixes.

            Returns ``None`` unless the last bound receipt is exactly that: one
            successful transfer, nothing else pending, no write outstanding. The
            successor keeps the round on every other shape.
            """

            if self._pending_live_calls or not self._grounded_calls:
                return None
            last = self._grounded_calls[-1]
            if last.tool_name != TRANSFER_TOOL_NAME or not last.succeeded:
                return None
            message = AssistantMessage.text(TRANSFER_POLICY_MESSAGE)
            return self._close_incremental_graph(message)

        def _stop_incremental_graph(self, *, reason: str, user_message: str):
            result = AirlineResidualAdmission(
                execute=False,
                reason=reason,
                diagnostics=self._graph_diagnostics(),
            )
            self._write_graph_decision(result=result, projection_seconds=0.0)
            self._clear_incremental_graph()
            return AssistantMessage.text(
                user_message,
                raw_data={
                    "papc_cfrc": {
                        **result.public_json(),
                        "fail_closed_before_next_live_write": True,
                    }
                },
            )

        def _settle_held_cheap_messages(self, *, reason: str, candidate: Any) -> None:
            """Bill intercepted generations even when no graph will open."""

            held = tuple(self._held_cheap_messages)
            self._held_cheap_messages = []
            if not held:
                return
            _write_airline_residual_decision(
                path=decision_log_path,
                task=getattr(self._task, "id", ""),
                strong_message=None,
                result=AirlineResidualAdmission(
                    execute=False,
                    reason=reason,
                    diagnostics={
                        "schema": (
                            "papc.tau2.airline_unopened_frontier_diagnostics.v1"
                        ),
                        "handoff_withheld_count": self._handoff_withheld_count,
                        "public_safe": True,
                    },
                ),
                receipt_summary={
                    "successful": 0,
                    "error": 0,
                    "unknown": 0,
                    "live_incremental": True,
                },
                projection_seconds=0.0,
                candidate_executed_before_frontier=False,
                cheap_call_count=self._cheap_call_count,
                strong_call_count=self._strong_call_count,
                runtime_ledger_summary=_runtime_ledger_for_messages(
                    cheap_messages=held,
                    strong_messages=(),
                ),
                strong_call_count_override=0,
                frontier_candidate_families=dict(
                    _cheap_frontier_candidate_summary(candidate)["tool_family_counts"]
                ),
            )

        def _write_graph_decision(
            self,
            *,
            result: AirlineResidualAdmission,
            projection_seconds: float,
        ) -> None:
            # A graph awaiting a user turn writes a row and then writes another
            # one when it finally closes. The held cheap frontier message never
            # reaches the transcript, so it is accounted here exactly once.
            cheap_messages = (
                (self._frontier_cheap_message,)
                if self._frontier_cheap_message is not None
                and not self._frontier_cheap_accounted
                else ()
            ) + tuple(self._held_cheap_messages)
            self._frontier_cheap_accounted = True
            self._held_cheap_messages = []
            _write_airline_residual_decision(
                path=decision_log_path,
                task=getattr(self._task, "id", ""),
                strong_message=None,
                result=result,
                receipt_summary={
                    "successful": len(self._grounded_calls),
                    "error": self._graph_receipt_error_count,
                    "unknown": 0,
                    "live_incremental": True,
                },
                projection_seconds=projection_seconds,
                candidate_executed_before_frontier=False,
                cheap_call_count=self._cheap_call_count,
                strong_call_count=self._strong_call_count,
                runtime_ledger_summary=_runtime_ledger_for_messages(
                    cheap_messages=cheap_messages,
                    strong_messages=tuple(self._intercepted_strong_messages),
                ),
                strong_call_count_override=self._graph_turn_count,
                frontier_candidate_families=self._frontier_candidate_families,
            )

        def _clear_incremental_graph(self) -> None:
            if residual_frontier_rearms_after_close:
                # Certification is a property of the reached state, not of the
                # episode. Once the graph is gone the floor is back with the
                # prefix model, so a later write candidate is again an
                # uncertified commitment frontier and has to be certified
                # before the successor may act on it. Leaving the latch set
                # would let every write after the first closed residual reach
                # the domain without crossing a frontier at all.
                self._residual_opened = False
            self._graph_active = False
            self._residual_awaiting_user_turn = False
            self._graph_frontier = None
            self._graph_compile_config = None
            self._residual_contract_context = {}
            self._last_successor_text_status = "none"
            self._frontier_cheap_message = None
            # The candidate families deliberately survive teardown: a graph that
            # pauses writes one decision row here and another when it finally
            # closes, and both rows describe the same frontier. A re-armed
            # frontier overwrites them on open.
            self._grounded_calls = []
            self._pending_live_calls = []
            self._intercepted_strong_messages = []
            self._graph_turn_count = 0
            self._graph_write_count = 0
            self._graph_receipt_error_count = 0
            self._declared_closure_text: str | None = None
            self._declared_closure_released = False

    def factory(
        tools: Any, domain_policy: str, llm: str, llm_args: Any = None, **kwargs: Any
    ):
        return AirlineResidualAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args or {},
            task=kwargs.get("task"),
        )

    if registry.get_agent_factory(agent_name) is None:
        registry.register_agent_factory(factory, agent_name)
    return agent_name


def _airline_frontier_from_prefix(messages: Sequence[Any]) -> CFRCFrontier:
    committed = _committed_airline_effect_units(messages)
    contract = StateWriteContract(
        surface="tau2_airline",
        derivation_timing="prefix",
        residual=frozenset(
            {
                Obligation(
                    obligation_id=RESIDUAL_OBLIGATION_ID,
                    write_footprint=frozenset(),
                    postcondition_witness="receipt_grounded_terminal_completion",
                    surface="tau2_airline",
                    status="open_bounded",
                )
            }
        ),
        forbidden_writes=frozenset(committed),
        legal_actions=frozenset({"bounded_repair", "abstain"}),
        state_change_required=True,
        user_facing_message_required=True,
        commitment_call_count=len(committed),
        commitment_footprint_size=len(committed),
        soundness_assumptions=(
            "visible_tool_receipts_define_reached_state",
            "each_live_call_is_admitted_by_the_incremental_graph",
            "terminal_closure_is_bound_to_live_receipts",
        ),
    )
    return CFRCFrontier(
        contract=contract,
        evidence_facts=(
            EvidenceFact(
                fact_id=REACHED_STATE_EVIDENCE_ID,
                fact_type="visible_receipt_state",
                provenance_refs=("accepted_prefix",),
                authority_labels=("ell0", "ell1"),
            ),
        ),
        frontier_timing="pre_write",
    )


def _airline_compile_config(*, graph_tool_call_budget: int) -> CFRCCompileConfig:
    return CFRCCompileConfig(
        mode="strict_core",
        authority_lattice=AuthorityLattice.from_ordered_labels(("ell0", "ell1")),
        strict_allowed_authorities=frozenset({"ell0", "ell1"}),
        allowed_authorities=frozenset({"ell0", "ell1"}),
        collision_budget=0.0,
        suffix_tool_call_budget=int(graph_tool_call_budget),
    )


def _project_airline_call(call: GroundedToolCall) -> GroundedNodeProjection:
    # A refused call changed nothing, so it carries no effect and discharges no
    # obligation. It stays in the graph as evidence, which is what lets the
    # successor repair from it instead of restarting.
    if call.succeeded and (
        call.tool_name in AIRLINE_WRITE_EFFECTS or call.tool_name == TRANSFER_TOOL_NAME
    ):
        family = (
            f"airline_write_{AIRLINE_WRITE_EFFECTS[call.tool_name]}"
            if call.tool_name in AIRLINE_WRITE_EFFECTS
            else "airline_transfer"
        )
        return GroundedNodeProjection(
            authority_label="ell1",
            semantic_effects=(_airline_effect_unit(call.tool_name, call.arguments),),
            effect_families=(family,),
            covered_obligation_ids=(RESIDUAL_OBLIGATION_ID,),
            evidence_fact_ids=(REACHED_STATE_EVIDENCE_ID,),
        )
    return GroundedNodeProjection(
        authority_label="ell0",
        evidence_fact_ids=(REACHED_STATE_EVIDENCE_ID,),
    )


def _airline_effect_unit(tool_name: str, arguments: Mapping[str, Any]) -> str:
    scope, object_ref = extract_object_ref(arguments)
    identity = (
        f"{scope}:{object_ref}"
        if object_ref
        else json.dumps(
            _json_safe_tool_value(dict(arguments)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    digest = hashlib.sha256(f"{tool_name}|{identity}".encode()).hexdigest()[:20]
    return f"airline_write_{digest}"


def _committed_airline_effect_units(messages: Sequence[Any]) -> tuple[str, ...]:
    pending: dict[str, tuple[str, Mapping[str, Any]]] = {}
    committed: list[str] = []
    for message in messages:
        if _message_attr_str(message, "role") == "assistant":
            for call in _message_tool_calls(message):
                row = _tool_call_prompt_json(call)
                name = str(row.get("name") or "")
                call_id = _message_attr_str(call, "id")
                if (
                    name in AIRLINE_WRITE_EFFECTS or name == TRANSFER_TOOL_NAME
                ) and call_id:
                    pending[call_id] = (name, dict(row.get("arguments") or {}))
            continue
        if _message_attr_str(message, "role") != "tool":
            continue
        call_id = _message_attr_str(message, "id")
        row = pending.pop(call_id, None)
        if row is None or bool(_message_attr(message, "error")):
            continue
        committed.append(_airline_effect_unit(row[0], row[1]))
    return tuple(dict.fromkeys(committed))


FRONTIER_ACTIVATION_POLICIES = ("all_write_or_transfer", "all_write")


def _opens_frontier(
    message: Any,
    *,
    policy: str = "all_write_or_transfer",
) -> bool:
    """Whether this cheap candidate is a commitment the successor must certify.

    A commitment frontier is the point where the prefix is about to make a
    commitment the successor has to preserve and complete from. A write is one.
    A hand-off to a human is not: it changes no state and leaves no residual
    behind it, so under ``all_write`` it does not open a frontier and the
    episode ends where the prefix ended it.

    ``all_write_or_transfer`` is retained as a compatibility policy and as the
    predicate every artifact before 2026-08-17 was produced under. It remains
    the default so those artifacts reproduce; current launches may select
    ``all_write`` explicitly.

    A gate that opened the frontier only on the escalations more likely to be
    wrong was implemented, measured and removed on 2026-08-17. It read nothing
    but the reached state, so it cleared the runtime prohibitions, but the
    reason it worked was that it predicted where the prefix fails, and a
    prediction of prefix failure is the router pattern this project refuses
    however it is computed. The measurement is kept in status.md.
    """

    if policy not in FRONTIER_ACTIVATION_POLICIES:
        return False
    return any(
        (name := _message_attr_str(call, "name")) in AIRLINE_WRITE_EFFECTS
        or (name == TRANSFER_TOOL_NAME and policy == "all_write_or_transfer")
        for call in _message_tool_calls(message)
    )


def _write_confirmation_is_open(messages: Sequence[Any]) -> bool:
    """Whether the reached state still owes the confirmation a write needs.

    The published airline policy admits a database update only after the agent
    has listed the action details and the user has answered. That answer is a
    fact about the reached state, not something a residual action can produce,
    so a write candidate raised before it arrives has no executable residual
    behind it yet. The test is structural: walking back from the candidate, the
    last exchange has to be a user turn that answers an assistant turn carrying
    text and no tool calls. No wording is matched and no tool argument is read.
    """

    last_user_index = -1
    for index, message in enumerate(messages):
        if _message_attr_str(message, "role") == "user":
            last_user_index = index
    if last_user_index < 0:
        return True
    for message in reversed(list(messages)[:last_user_index]):
        role = _message_attr_str(message, "role")
        if role == "tool":
            continue
        if role != "assistant":
            break
        answered = bool(
            _message_attr_str(message, "content")
        ) and not _assistant_has_tool_calls(message)
        return not answered
    return True


_WITHHELD_HANDOFF_INSTRUCTION = (
    "Runtime notice: the hand-off you just proposed was not sent, because it "
    "would end this conversation while the user's request is still open. "
    "Nothing about the reached state has changed and no tool ran. Recheck each "
    "unfinished part of the request against the published policy and visible "
    "receipts. A restriction on one kind of change does not automatically "
    "forbid other kinds of change. Check whether an available read tool can "
    "supply missing facts before treating them as unavailable. If one part is "
    "blocked, continue any independent parts that remain admissible. Preserve "
    "the user's constraints and obtain any required confirmation before "
    "writing. If no admissible continuation remains, propose the hand-off "
    "again and it will be sent."
)


_DEFERRED_FRONTIER_INSTRUCTION = (
    "Runtime notice: the action you just proposed updates the booking "
    "database, and the policy admits it only after you have listed its details "
    "and the user has answered. That confirmation is not on record yet, so the "
    "action was not run. Reply with the message that states the pending action "
    "and its details and asks the user to confirm, and make no tool call in "
    "this turn."
)


def _frontier_tool_family(name: str) -> str:
    if name in AIRLINE_WRITE_EFFECTS:
        return f"write:{AIRLINE_WRITE_EFFECTS[name]}"
    if name == TRANSFER_TOOL_NAME:
        return "transfer"
    if name:
        return "read_or_query"
    return "none"


def _residual_prompt_payload(
    *,
    visible_evidence: Sequence[Mapping[str, Any]],
    cheap_frontier_message: Any | None,
    prefix_trace: Mapping[str, Any],
    write_budget: int,
    used_write_count: int,
    pending_carrier_audit: bool = False,
    declared_closure: bool = False,
    request_obligations: Mapping[str, Any] | None = None,
    lean_packet: bool = False,
) -> dict[str, Any]:
    remaining_write_budget = max(0, int(write_budget) - int(used_write_count))
    # Every evidence row on this surface is a stub of something the same request
    # replays verbatim below the packet: the caller passes
    # ``user_requests_carried_verbatim=True`` and excludes the carried receipt
    # ids, so what survives is an ordering index over messages the model is
    # about to read in full. ``lean_packet`` drops the index when it carries
    # nothing else, and keeps it whenever any row still holds real content, so
    # the switch can never silently remove evidence.
    evidence_rows = [dict(event) for event in visible_evidence]
    evidence_is_all_stub = bool(evidence_rows) and all(
        bool(row.get("carried_verbatim_in_replica_transcript"))
        for row in evidence_rows
    )
    drop_evidence = bool(lean_packet) and evidence_is_all_stub
    payload: dict[str, Any] = {
        "schema": "papc.tau2.airline_incremental_graph_prompt.v1",
        "method_chain": (
            "cheap read prefix -> first write/transfer frontier -> "
            "strong incremental live-receipt graph"
        ),
        "authority": RESIDUAL_AUTHORITY,
        "remaining_write_budget": remaining_write_budget,
        "visible_evidence": [] if drop_evidence else evidence_rows,
        **(
            {
                "visible_evidence_note": (
                    "Every visible event is replayed verbatim in the "
                    "conversation below this packet; read it there."
                )
            }
            if drop_evidence
            else {}
        ),
        "cheap_frontier_candidate": _cheap_frontier_candidate_summary(
            cheap_frontier_message
        ),
        "prefix_graph_summary": {
            "write_count_bucket": count_bucket(
                int(prefix_trace.get("write_count") or 0)
            ),
            "read_count_bucket": read_count_bucket(
                int(prefix_trace.get("read_call_count") or 0)
            ),
            "transfer_seen": int(prefix_trace.get("transfer_call_count") or 0) > 0,
            "last_write_effect": str(prefix_trace.get("last_write_effect") or "none"),
            "effect_shape": str(prefix_trace.get("effect_shape") or "none"),
        },
        # Prose alone did not carry this. The packet already tells the
        # successor to cover the whole remaining request, and on the failing
        # rows it still stops after one write. A named ledger is something it
        # can read off rather than infer, and something a later checker could
        # refuse against.
        **(
            {
                "request_obligations": {
                    "status": str(request_obligations.get("status") or ""),
                    "reason": str(request_obligations.get("reason") or ""),
                    "owed": [
                        dict(row)
                        for row in (request_obligations.get("obligations") or ())
                    ],
                    "required_terminal_literals": [
                        str(item)
                        for item in (
                            request_obligations.get("required_terminal_literals")
                            or ()
                        )
                    ],
                    "communication_requirements": list(
                        request_obligations.get("communication_requirements") or ()
                    ),
                    "missing_evidence": list(
                        request_obligations.get("missing_evidence") or ()
                    ),
                    "terminal_response": str(
                        request_obligations.get("terminal_response") or ""
                    ),
                    "all_or_nothing": bool(request_obligations.get("all_or_nothing")),
                    "rule": (
                        "Each entry is one state effect the visible request "
                        "asks for and the receipts do not yet show. Discharge "
                        "every one of them and the communication requirements "
                        "before you close, in dependency order. Resolve missing "
                        "evidence before the affected action. Match entries "
                        "against current receipts so completed effects are not "
                        "repeated. A continuation that leaves an entry open is "
                        "under-complete even if the one write it made is "
                        "correct. When the ledger is unavailable, fall back to "
                        "reading the request yourself."
                    ),
                }
            }
            if request_obligations
            else {}
        ),
        "compiler_rules": [
            "Continue only from the reached conversation and environment state.",
            "Do not restart the task or ignore existing tool results.",
            "Use legal airline tools only.",
            (
                "Cover the whole remaining request: include every "
                "state-changing action it still requires, not only the next "
                "convenient one. A continuation that discharges one "
                "requirement and leaves another open is under-complete."
            ),
            (
                "Add no effect the request did not ask for, and do not redo "
                "an effect the visible receipts already show."
            ),
            (
                "When the request quantifies over a set, the obligation ranges "
                "over the whole set. Enumerate it from the receipts before "
                "acting, restate every member that meets the stated condition, "
                "and do not narrow it to the members most recently discussed."
            ),
            (
                "Do not ask the user for a value an accepted receipt already "
                "binds. Spend a question only on a decision that is the "
                "user's, or on a value no visible receipt contains."
            ),
            "You may read or verify before writing when required by policy.",
            "Do not exceed remaining_write_budget for mutating write tools.",
            ("When the request is complete, send a concise final user-facing message."),
        ],
        "forbidden_inputs": [
            "task_id",
            "gold",
            "reward",
            "reference_answer",
            "hidden_state",
            "benchmark-specific answer sketch",
        ],
    }
    if declared_closure:
        payload["declared_closure"] = _declared_closure_contract()
        payload["compiler_rules"] = [
            *payload["compiler_rules"],
            (
                "When the actions you issue in this turn discharge the whole "
                "remaining request, include cfrc_declare_completion(message) "
                "in the same tool-call batch as those actions. It represents "
                "the graph's terminal node, not a live domain action. Its "
                "message is held: the runtime delivers it only after "
                "every receipt those actions produce has been accepted, and "
                "discards it otherwise."
            ),
            (
                "The details the user already confirmed are part of the "
                "reached state, so restating them in that message is grounded, "
                "not speculation. Omit the terminal declaration "
                "only when it would have to quote a value that first appears "
                "in their receipts, when a further action may still be "
                "required, or when you are asking the user a question."
            ),
        ]
    if pending_carrier_audit:
        payload["pending_carrier_audit"] = _pending_carrier_audit_contract()
        payload["compiler_rules"] = [
            *payload["compiler_rules"],
            (
                "Before any write or transfer, audit the pending carrier "
                "against visible tool results and the latest user-confirmed "
                "constraints; a cheap proposal is not evidence."
            ),
            (
                "If the audit changes itinerary, price, payment, passenger, "
                "baggage, or cancellation scope, explain the correction and "
                "obtain confirmation before writing."
            ),
        ]
    return payload


def _declared_closure_contract() -> dict[str, Any]:
    return {
        "schema": "papc.tau2.airline_declared_closure.v4",
        "enabled": True,
        "terminal_declaration_tool": _TERMINAL_DECLARATION_TOOL,
        "declaration_is_not_a_live_domain_action": True,
        "message_beside_actions_is_held_not_sent": True,
        "released_only_if_every_receipt_is_accepted": True,
        "requires_explicit_residual_status": True,
        "discarded_if_any_receipt_refuses_or_mismatches": True,
        "may_restate_details_the_user_already_confirmed": True,
        "omit_when_a_value_first_appears_in_the_receipt": True,
        "omit_when_a_further_action_may_be_required": True,
        "omit_when_asking_the_user_a_question": True,
        "public_safe": True,
    }


def _pending_carrier_audit_contract() -> dict[str, Any]:
    return {
        "schema": "papc.tau2.airline_pending_carrier_audit.v1",
        "enabled": True,
        "candidate_status": "intercepted_not_executed",
        "audit_before_write_or_transfer": True,
        "cheap_proposal_is_not_evidence_of_correctness": True,
        "must_recompute_from_visible_tool_results": [
            "route",
            "dates",
            "cabin",
            "flight_numbers",
            "passengers",
            "baggage",
            "price_or_refund",
            "payment_methods",
            "cancellation_set",
        ],
        "must_reconfirm_if_correction_changes": [
            "route_or_date",
            "cabin",
            "price_or_refund",
            "payment_method",
            "passenger_set",
            "baggage_count",
            "reservation_cancelled_or_booked",
        ],
        "public_safe": True,
    }


def _residual_step_reminder(
    *,
    write_budget: int,
    used_write_count: int,
    resumed_after_user_turn: bool = False,
    declared_closure: bool = False,
) -> dict[str, Any]:
    payload = {
        "schema": "papc.tau2.airline_incremental_graph_step.v1",
        "authority": RESIDUAL_AUTHORITY,
        "remaining_write_budget": max(0, int(write_budget) - int(used_write_count)),
        "continue_from_live_receipts_only": True,
        "strong_full_restart_forbidden": True,
        "public_safe": True,
    }
    if declared_closure:
        payload["declared_closure"] = _declared_closure_contract()
    if resumed_after_user_turn:
        payload["resumed_after_user_turn"] = True
        payload["residual_state_change_still_open"] = True
        # The pause was spent to obtain the reply that is now in the
        # conversation. Reading it and acting on it is what the pause bought.
        payload["guidance"] = (
            "The user has answered the question you paused to ask, and the "
            "remaining state change is still open and still yours to complete. "
            "Read the reply. If it authorises the action, issue the tool call "
            "now instead of asking again. If it withdraws or changes the "
            "request, follow the reply rather than the earlier plan."
        )
    return payload


def _cheap_frontier_candidate_summary(message: Any | None) -> dict[str, Any]:
    if message is None:
        return {
            "available": False,
            "status": "absent",
            "tool_family_counts": {},
            "argument_key_counts": [],
        }
    calls = [_tool_call_prompt_json(call) for call in _message_tool_calls(message)]
    return {
        "available": bool(calls),
        "status": "unexecuted_pending_strong_audit",
        "tool_family_counts": dict(
            sorted(
                Counter(
                    _frontier_tool_family(str(call.get("name") or "")) for call in calls
                ).items()
            )
        ),
        "argument_key_counts": [
            len(dict(call.get("arguments") or {})) for call in calls
        ],
    }


def _receipt_result_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple, int, float, bool)) or value is None:
        return _json_safe_tool_value(value)
    text = str(value)
    try:
        return _json_safe_tool_value(json.loads(text))
    except json.JSONDecodeError:
        return text


def _carried_tool_receipt_ids(messages: Sequence[Any]) -> frozenset[str]:
    """Receipts this request already sends verbatim, so the packet may stub them.

    The successor request replays the live transcript in full next to the
    evidence packet built from that same transcript, so every receipt arrives
    twice and is billed twice on the one round that carries the packet. The
    packet keeps a stub naming each receipt, which is what makes the second copy
    redundant rather than lost: the model still sees the record, once.
    """

    return frozenset(
        receipt_id
        for message in messages
        if _message_attr_str(message, "role") == "tool"
        for receipt_id in (
            _message_attr_str(message, "id")
            or _message_attr_str(message, "tool_call_id"),
        )
        if receipt_id
    )


def _trace_from_messages(messages: Sequence[Any]) -> dict[str, Any]:
    write_nodes: list[dict[str, str]] = []
    read_count = 0
    transfer_count = 0
    for message in messages:
        if _message_attr_str(message, "role") != "assistant":
            continue
        for call in _message_tool_calls(message):
            row = _tool_call_prompt_json(call)
            name = str(row.get("name") or "")
            if name in AIRLINE_WRITE_EFFECTS:
                scope, ref = extract_object_ref(dict(row.get("arguments") or {}))
                write_nodes.append(
                    {
                        "tool_name": name,
                        "effect": AIRLINE_WRITE_EFFECTS[name],
                        "object_scope": scope,
                        "object_ref": ref,
                    }
                )
            elif name == TRANSFER_TOOL_NAME:
                transfer_count += 1
            elif name:
                read_count += 1
    return {
        "write_nodes": tuple(write_nodes),
        "write_count": len(write_nodes),
        "effect_shape": effect_shape(write_nodes),
        "last_write_effect": (
            str(write_nodes[-1]["effect"]) if write_nodes else "none"
        ),
        "read_call_count": read_count,
        "transfer_call_count": transfer_count,
    }


def _write_airline_residual_decision(
    *,
    path: Path,
    task: Any,
    strong_message: Any | None,
    result: AirlineResidualAdmission,
    receipt_summary: Mapping[str, Any],
    projection_seconds: float,
    candidate_executed_before_frontier: bool = True,
    cheap_call_count: int | None = None,
    strong_call_count: int | None = None,
    runtime_ledger_summary: Mapping[str, Any] | None = None,
    strong_call_count_override: int | None = None,
    frontier_candidate_families: Mapping[str, int] | None = None,
) -> None:
    ledger = RuntimeLedger()
    if strong_message is not None:
        ledger.add_usage(
            "strong",
            _mapping_or_none(getattr(strong_message, "usage", None)),
            latency_seconds=_optional_float(
                getattr(strong_message, "generation_time_seconds", None)
            ),
            cost_usd=_optional_float(getattr(strong_message, "cost", None)),
        )
    ledger_public = (
        dict(runtime_ledger_summary)
        if isinstance(runtime_ledger_summary, Mapping)
        else ledger.public_summary()
    )
    write_jsonl_append(
        path,
        {
            "schema": "papc.tau2.airline_residual_decision.v1",
            "task_hash": hashlib.sha256(str(task).encode("utf-8")).hexdigest()[:16],
            "protocol": "cheap_prefix_then_receipt_grounded_residual_graph",
            "frontier_verifier": {
                "schema": "papc.tau2.airline_residual_frontier.v1",
                "accepted": True,
                "predicate": "first_write_transfer_candidate",
                "frontier_timing": "pre_write_candidate"
                if not candidate_executed_before_frontier
                else "terminal_after_cheap_prefix",
                # Which kind of candidate opened the frontier, as the same
                # four-label histogram the successor already receives. A
                # frontier opened by a terminal escalation and one opened by a
                # write are indistinguishable in every other field of this row.
                "candidate_tool_families": dict(
                    sorted((frontier_candidate_families or {}).items())
                ),
                "candidate_tool_call_executed_before_frontier": bool(
                    candidate_executed_before_frontier
                ),
                "residual_activation_reason": FRONTIER_REASON,
            },
            "strong_call_count": (
                int(strong_call_count_override)
                if strong_call_count_override is not None
                else (1 if strong_message is not None else 0)
            ),
            "cheap_call_count_at_decision": (
                int(cheap_call_count) if cheap_call_count is not None else None
            ),
            "strong_call_count_at_decision": (
                int(strong_call_count) if strong_call_count is not None else None
            ),
            "projection_result": result.public_json(),
            "receipt_summary": dict(receipt_summary),
            "runtime_ledger": ledger_public,
            "deterministic_projection_seconds": round(projection_seconds, 6),
            "graph_residual_continuation": {
                "schema": "papc.tau2.graph_residual_continuation.v1",
                "enabled": True,
                "policy": "first_write_or_transfer_frontier",
                "after_residual_suffix": False,
                "after_abstain": False,
            },
            "raw_messages_excluded": True,
            "public_safe": False,
        },
    )


def _assistant_has_tool_calls(message: Any) -> bool:
    return bool(_message_tool_calls(message))


def _strip_message_content(message: Any) -> Any:
    """Return the message with its user-facing text removed, calls untouched."""

    stripped = deepcopy(message)
    try:
        stripped.content = None
    except (AttributeError, ValueError):
        return message
    return stripped


_SUCCESSOR_SYSTEM_PROMPT = (
    "You are a CFRC bounded graph-continuation agent. Continue only from the "
    "reached airline state and visible live tool results. Do not restart the task. "
    "Use legal tools, keep mutating writes within the supplied budget, and finish "
    "with a concise user-facing message when the request is complete. "
    "For a response without tool calls, return a JSON object with message "
    "(the user-facing text) and residual_status: awaiting_user when you need "
    "a user answer or confirmation, complete only when the whole remaining "
    "request is discharged, or blocked when it remains unfinished and no "
    "admissible continuation is available. One successful write does not "
    "complete a request that still needs another action or user decision. "
    "The status is runtime control and is removed before the user sees the text."
)


def _successor_text_control(message: Any) -> tuple[str | None, Any]:
    """Separate a declared continuation state from its user-facing text.

    Unlabelled text retains the legacy protocol. A declared status is checked
    strictly; in particular, a partial write cannot turn a question or refusal
    into a completion. A complete declaration still needs receipt admission.
    """

    try:
        payload = json.loads(_message_attr_str(message, "content"))
    except (TypeError, ValueError):
        return None, message
    if not isinstance(payload, dict) or "residual_status" not in payload:
        return None, message
    status = payload["residual_status"]
    text = payload.get("message")
    if (
        status not in ("awaiting_user", "complete", "blocked")
        or not isinstance(text, str)
        or not text.strip()
    ):
        raise ValueError("successor_text_status_invalid")
    visible = deepcopy(message)
    if isinstance(visible, dict):
        visible["content"] = text
    else:
        visible.content = text
    return status, visible


__all__ = [
    "RESIDUAL_AUTHORITY",
    "DEFAULT_RESIDUAL_WRITE_BUDGET",
    "AirlineResidualAdmission",
    "register_tau2_airline_residual_agent",
]


# The airline frontier contract carries one obligation with an empty write
# footprint, so it states that a residual exists without stating what the
# request still owes. The successor is told in prose to cover the whole
# remaining request, and on the rows that fail it stops after one write anyway:
# six of the eight failures on the registered old-pool arm are multi-write
# requests where part of the write set never lands, and the checker cannot
# refuse them because nothing in the contract names the missing write.
#
# This builds the same request-scoped ledger Retail uses. The module is
# surface-generic and airline is a tau2 domain, so the only new thing here is
# the wiring. It costs one prefix-model call at the frontier and it fails open:
# a builder that does not parse leaves the run exactly as it was.
def airline_request_obligations(
    *,
    generate_fn: Callable[..., Any],
    model: Any,
    llm_args: Mapping[str, Any],
    messages: Sequence[Any],
    system_messages: Sequence[Any],
    tools: Sequence[Any],
    visible_evidence: Sequence[Mapping[str, Any]],
    system_message_cls: Any,
    user_message_cls: Any,
) -> dict[str, Any]:
    """Enumerate what the visible request still owes, or say why it could not.

    The two message classes are passed in rather than imported here: the caller
    already holds them, and taking them as arguments is what lets this be
    exercised without the benchmark package on the path. The tool-schema bug
    this signature now guards against only surfaced once a paid run had started.
    """

    from papc.certspec.tau2.request_obligation_contract import (
        collect_visible_values,
        parse_request_contract,
        request_contract_prompt,
    )

    request_text = "\n\n".join(
        str(_message_attr(message, "content") or "").strip()
        for message in messages
        if _message_attr_str(message, "role") == "user"
        and str(_message_attr(message, "content") or "").strip()
    ).strip()
    if not request_text:
        return {"status": "unavailable", "reason": "no_visible_request"}
    # The callable surface as the builder reads it, taken from the tool's own
    # openai_schema the way the Retail builder takes it. Reaching for a
    # parameter attribute instead picks up the unbound Pydantic method and the
    # run dies on the first frontier.
    schemas: list[dict[str, Any]] = []
    allowed: list[str] = []
    for tool in tools:
        name = str(getattr(tool, "name", "") or "")
        if not name:
            continue
        allowed.append(name)
        schema = getattr(tool, "openai_schema", None)
        if isinstance(schema, Mapping):
            schemas.append(json.loads(json.dumps(dict(schema), default=str)))
        else:
            schemas.append({"name": name})
    policy = "\n\n".join(
        str(_message_attr(message, "content") or "").strip()
        for message in system_messages
        if str(_message_attr(message, "content") or "").strip()
    ).strip()
    # Assistant proposals explain a later "yes" but cannot supply the values
    # that make a write admissible. Keep them out of the grounding vocabulary.
    prefix_context = [
        dict(row) for row in visible_evidence
        if row.get("kind") == "untrusted_confirmation_context"
    ]
    authority_evidence = [
        dict(row) for row in visible_evidence
        if row.get("kind") != "untrusted_confirmation_context"
        and row.get("authority") is not False
    ]
    prompt = request_contract_prompt(
        surface="tau2_airline",
        visible_request=request_text,
        visible_evidence=authority_evidence,
        accepted_prefix_context=prefix_context,
        candidate_calls=[],
        public_tool_schemas=schemas,
        public_policy=policy,
    )
    message = generate_fn(
        model=model,
        messages=[
            system_message_cls(
                role="system", content=_AIRLINE_REQUEST_OBLIGATION_SYSTEM_PROMPT
            ),
            user_message_cls.text(
                json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
            ),
        ],
        call_name="airline_request_obligation_builder",
        **dict(llm_args),
    )
    raw = str(_message_attr(message, "content") or "")
    try:
        contract = parse_request_contract(
            raw,
            surface="tau2_airline",
            allowed_tool_names=sorted(allowed),
            visible_values=collect_visible_values(request_text, authority_evidence),
            visible_literal_text=request_text,
        )
    except (TypeError, ValueError) as exc:
        return {
            "status": "unavailable",
            "reason": f"parse_rejected:{str(exc).split(':', 1)[0]}",
            "builder_message": message,
        }
    return {
        **contract.private_json(),
        "builder_message": message,
    }


_AIRLINE_REQUEST_OBLIGATION_SYSTEM_PROMPT = (
    "You enumerate what a visible airline request still owes the user, as a "
    "ledger of atomic state-changing actions and required communication. "
    "Read every clause of the request "
    "against the receipts already in evidence and list one obligation per "
    "effect the request asks for and the receipts do not yet show. Preserve "
    "the user's scope restrictions when resolving later confirmations: an "
    "assistant proposal does not override those restrictions. A request "
    "that asks to change a flight and also add a bag owes two obligations, not "
    "one. Use only the public tool names and only values the request or a "
    "receipt already carries; never invent an identifier. Return only one JSON "
    "object whose keys are exactly the keys of output_schema, with no preamble "
    "and no commentary. When the receipts already discharge the request, "
    "return status no_state_change with an empty obligation list and a "
    "grounded terminal_response. If required evidence or a user choice is "
    "missing, return needs_more_evidence and identify it instead of inventing "
    "an argument."
)
