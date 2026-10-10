"""True-live STATE-Bench CFRC controller primitives.

This module owns the runtime seam that the old STATE-Bench residual runners did
not have: cheap acts inside the official harness, the visible prefix is checked
after each cheap step, and strong takes over only after a prefix-local frontier
is accepted.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
import traceback
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

from papc.certspec.contract_projection import compact_tool_schema
from papc.certspec.residual_closure import (
    contradicts,
    leaf_assignments,
    open_obligations,
    ungrounded_values,
    unrealized_write_fields,
)
from papc.certspec.runtime_ledger import RoleCall, RuntimeLedger
from papc.certspec.state_bench.online_frontier_stream import (
    ONLINE_EXECUTION_CONTRACT,
    StreamFrontierDecision,
    decide_stream_prefix,
)
from papc.certspec.state_bench.runtime_accounting import merge_phase_ledgers
from papc.certspec.state_bench.true_live_scaffold import (
    CompiledTrueLiveProposal,
    IncrementalResidualGraph,
    ResidualGraphNode,
    apply_true_live_precommit_guard,
    compile_native_true_live_proposal,
    latest_user_confirms_precommit,
    live_online_system_prompt_suffix,
    staged_true_live_precommit_message,
    strong_native_proposal_instruction,
    true_live_precommit_requires_confirmation,
)

SCHEMA = "papc.state_bench_true_live_online.v1"
MAX_PROVIDER_ATTEMPTS = 3
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class StateBenchCFRCProfile:
    """The runtime choices that make one STATE-Bench CFRC operating point.

    Each field changes what the successor may do or is told, so a run that
    changes one is a different arm and carries its own measurement. The
    defaults are the configuration that measured best on current source, not
    the widest available setting, so a run that passes no flag reproduces that
    point instead of an untested one.

    Publish this object rather than restating it. The published contract used
    to carry its own literal for the strong-call budget, and the two drifted:
    an artifact reported a budget of four over a run that had spent two.
    """

    max_residual_strong_calls: int = 2
    declare_remaining_responses: bool = True
    declare_closure_admissibility: bool = False
    refuse_unclosable_last_slot: bool = False
    closing_contract_style: str = "consequence"

    def public_json(self) -> dict[str, Any]:
        return {
            "max_residual_strong_calls": int(self.max_residual_strong_calls),
            "residual_round_budget_declared": bool(self.declare_remaining_responses),
            "closure_obligation_declared": bool(self.declare_closure_admissibility),
            "refuse_unclosable_last_slot": bool(self.refuse_unclosable_last_slot),
            "closing_contract_style": str(self.closing_contract_style),
        }


# Measured 2026-09-01 on current source over the full 150-task test split with
# GPT-5.4-mini high into GPT-5.5 high: 92 of 150 at $13.210066 cache neutral.
# The two declaration fields were each measured against this point. Declaring
# the remaining responses reached it from 86. Declaring closure admissibility
# moved the same pool to 93 but the GPT-5.6 Luna into Sol pool to 85 from 90,
# so it stays off rather than becoming a per-pool semantic boundary.
VALIDATED_MAINLINE_PROFILE = StateBenchCFRCProfile()

MAINLINE_MAX_RESIDUAL_STRONG_CALLS = (
    VALIDATED_MAINLINE_PROFILE.max_residual_strong_calls
)
MAINLINE_CHEAP_MAX_NO_RESULT_READS_PER_ROW = 32
# Measured arms. "mainline" is the sole method profile. "cheap_anchor" runs the
# cheap model over the whole system with every method-side component removed, so
# it measures what the cheap model alone reaches and at what cost. It is a
# denominator for the cost axis, not a method variant.
METHOD_ARMS = ("mainline", "cheap_anchor")

# Whether the method arm carries the STATE-Bench domain prompt suffix, and how
# much of it. The suffix has two independent parts: a statement of what the
# method itself does, and a restatement of domain policy the benchmark already
# publishes. "on" is both, "method_only" drops the domain policy, "off" drops
# the suffix entirely. The anchor arm reads the same setting, so an incumbent
# can be measured under the prompt the method arm receives.
DOMAIN_PROMPT_SUFFIX_MODES = ("on", "generic", "method_only", "off")
# The mode names predate the third form and do not line up with it: "on" and
# "method_only" both carry the method section, and only the domain section
# differs between them. Mapping them here keeps the flag stable for every
# recorded operating point while giving the value-free section a name.
_DOMAIN_DISCIPLINE_BY_SUFFIX_MODE = {
    "on": "full",
    "generic": "generic",
    "method_only": "none",
}


def online_execution_contract(
    profile: StateBenchCFRCProfile | None = None,
) -> dict[str, Any]:
    """Describe the STATE-Bench CFRC execution path that ``profile`` runs.

    Callers that hold the live configuration pass it, so the published contract
    is a projection of what ran rather than a second statement of it. Passing
    nothing describes the validated profile.
    """

    profile = VALIDATED_MAINLINE_PROFILE if profile is None else profile
    contract = dict(ONLINE_EXECUTION_CONTRACT)
    contract.update(
        {
            "execution_shape": "true_live_cheap_then_cfrc",
            "frontier_decided_on_prefix_only": True,
            "residual_representation": (
                "receipt_grounded_incremental_residual_graph"
            ),
            "strong_proposal_transport": "grounded_native_tool_loop",
            "strong_call_budget": profile.max_residual_strong_calls,
            **profile.public_json(),
            "runner_compiler_executes_or_abstains": True,
            "checked_execution": "graph_tool_surface_and_grounded_admission",
            "notes": (
                "Cheap acts until a prefix-local commitment frontier. Strong then "
                "authors residual action nodes incrementally from visible state and "
                "shadow receipts; receipt visibility induces graph dependencies. "
                "Only the graph-matched native calls and bounded terminal message "
                "reach the official harness."
            ),
        }
    )
    return contract


@dataclass(frozen=True)
class ProviderRoleConfig:
    model: str
    reasoning_effort: str


@dataclass(frozen=True)
class ShadowLoopState:
    """Public-safe state used by the shadow convergence policy."""

    successful_tool_result_count: int
    selected_tool_call_count: int
    remaining_strong_budget: int
    has_compile_or_handler_error: bool
    candidate_decision: str
    requires_precommit_confirmation: bool


@dataclass(frozen=True)
class ConvergenceDecision:
    action: Literal["continue", "reserve_closure", "safety_block"]
    reason: str
    state: ShadowLoopState


class OpenAICompatibleRuntimeClient:
    """Small Responses API client for true-live PAPC runtime roles."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float | None = None,
        default_reasoning_effort: str | None = None,
        openai_client: Any | None = None,
    ) -> None:
        if not api_key and openai_client is None:
            raise ValueError("OpenAICompatibleRuntimeClient requires an API key")
        if not base_url and openai_client is None:
            raise ValueError("OpenAICompatibleRuntimeClient requires a base URL")
        if not model:
            raise ValueError("OpenAICompatibleRuntimeClient requires a model")
        self.provider = "papc_openai_compatible"
        self.endpoint = base_url.rstrip("/")
        self.deployment = model
        self.deployments = [model]
        self.default_reasoning_effort = (
            str(default_reasoning_effort).strip() if default_reasoning_effort else None
        )
        self._papc_call_events: list[dict[str, Any]] = []
        self._papc_thread_local = threading.local()
        if openai_client is not None:
            self._client = openai_client
        else:
            kwargs: dict[str, Any] = {"api_key": api_key, "base_url": self.endpoint}
            if timeout is not None:
                kwargs["timeout"] = timeout
            self._client = OpenAI(**kwargs)

    @property
    def provider_name(self) -> str:
        return self.provider

    @property
    def model_name(self) -> str:
        return self.deployment

    def drain_papc_call_events(self) -> list[dict[str, Any]]:
        events = list(self._papc_current_thread_events())
        self._papc_current_thread_events().clear()
        return events

    def complete_chat(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "input": [dict(message) for message in messages],
            "max_output_tokens": max_tokens,
        }
        effort = reasoning_effort or self.default_reasoning_effort
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        elif temperature is not None:
            kwargs["temperature"] = temperature
        response = self._create_response(
            kwargs=kwargs,
            call_kind="complete_chat",
            reasoning_effort=effort,
        )
        return str(getattr(response, "output_text", "") or "")

    def complete_with_tools(
        self,
        *,
        instructions: str,
        input: list[Any],
        tools: list[dict[str, Any]],
        previous_response_id: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "instructions": instructions,
            "input": _normalise_items(list(input)),
            "tools": tools,
            "max_output_tokens": max_tokens,
        }
        if previous_response_id is not None:
            kwargs["previous_response_id"] = previous_response_id
        effort = reasoning_effort or self.default_reasoning_effort
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        elif temperature is not None:
            kwargs["temperature"] = temperature
        return self._create_response(
            kwargs=kwargs,
            call_kind="complete_with_tools",
            reasoning_effort=effort,
            previous_response_id_provided=previous_response_id is not None,
        )

    def _create_response(
        self,
        *,
        kwargs: dict[str, Any],
        call_kind: str,
        reasoning_effort: str | None,
        previous_response_id_provided: bool = False,
    ) -> Any:
        started = time.monotonic()
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self._client.responses.create(**kwargs)
                _raise_if_content_filtered(response)
                break
            except Exception as exc:
                if (
                    attempts >= MAX_PROVIDER_ATTEMPTS
                    or not _is_retryable_provider_error(exc)
                ):
                    self._record_papc_call_event(
                        {
                            "status": "error",
                            "model": self.deployment,
                            "call_kind": call_kind,
                            "reasoning_effort": reasoning_effort,
                            "previous_response_id_provided": previous_response_id_provided,
                            "latency_seconds": round(time.monotonic() - started, 6),
                            "attempt_count": attempts,
                            "error_type": type(exc).__name__,
                        }
                    )
                    raise
                time.sleep(min(2.0 * attempts, 8.0))
        self._record_papc_call_event(
            {
                "status": "ok",
                "model": self.deployment,
                "call_kind": call_kind,
                "reasoning_effort": reasoning_effort,
                "previous_response_id_provided": previous_response_id_provided,
                "response_id": getattr(response, "id", None),
                "latency_seconds": round(time.monotonic() - started, 6),
                "attempt_count": attempts,
                "usage": _response_usage_dict(response),
            }
        )
        return response

    def _record_papc_call_event(self, event: dict[str, Any]) -> None:
        event = {"schema": "papc.state_bench_client_call_telemetry.v1", **event}
        self._papc_call_events.append(event)
        self._papc_current_thread_events().append(event)

    def _papc_current_thread_events(self) -> list[dict[str, Any]]:
        events = getattr(self._papc_thread_local, "events", None)
        if events is None:
            events = []
            self._papc_thread_local.events = events
        return events


class TrueLiveCFRCAgent:
    """Official-harness agent that switches from cheap to strong at live z_f."""

    def __init__(
        self,
        *,
        domain: str,
        cheap_client: Any,
        strong_client: Any,
        cheap: ProviderRoleConfig,
        strong: ProviderRoleConfig,
        row_index: int = 0,
        min_cheap_assistant_turns: int = 1,
        max_tool_rounds_per_assistant_turn: int = 8,
        cheap_max_tool_calls_per_response: int = 32,
        cheap_max_no_result_reads_per_row: int = 32,
        max_residual_strong_calls: int = 4,
        cheap_system_prompt_suffix: str = "",
        strong_system_prompt_suffix: str = "",
        cheap_native_item_loop: bool = False,
        strong_proposer_mode: str = "shadow_tool_loop",
        precommit_guard_mode: str = "off",
        final_message_repair_mode: str = "closure",
        residual_round_budget_declared: bool = False,
        closure_obligation_declared: bool = False,
        closing_contract_style: str = "consequence",
        shadow_convergence_guard: bool = True,
        shadow_closure_short_circuit: bool = True,
        successor_tool_descriptions: bool = True,
        refuse_unclosable_last_slot: bool = False,
        final_repair_role: str = "strong",
        final_repair_reasoning_effort: str | None = None,
        shadow_tool_handlers_factory: Callable[
            [Sequence[Mapping[str, Any]]], Mapping[str, Callable[[dict[str, Any]], Any]]
        ]
        | None = None,
    ) -> None:
        self.domain = str(domain)
        self.cheap_client = cheap_client
        self.strong_client = strong_client
        self.cheap = cheap
        self.strong = strong
        self.row_index = int(row_index)
        self.min_cheap_assistant_turns = max(0, int(min_cheap_assistant_turns))
        self.max_tool_rounds_per_assistant_turn = max(
            1, int(max_tool_rounds_per_assistant_turn)
        )
        # Native-runtime output bound on the cheap agent's own batch size. The
        # round count is already capped; the number of tool calls inside one
        # response was not, and a weak cheap model can emit a degenerate
        # read burst (observed: 350 calls in one response, 1643 in one row)
        # that never commits, so no commitment frontier can be detected and the
        # episode is spent before the method can act. This trims the batch and
        # lets the harness return results; the model keeps its remaining rounds
        # and can re-issue. It is not a routing or escalation decision: it does
        # not choose who runs, and every write-tool call is preserved even
        # beyond the bound so no commitment can be dropped. 0 disables it and
        # restores the unbounded legacy batch byte-for-byte.
        self.cheap_max_tool_calls_per_response = max(
            0, int(cheap_max_tool_calls_per_response)
        )
        self._cheap_batch_truncated_response_count = 0
        self._cheap_batch_dropped_call_count = 0
        self._cheap_batch_max_calls_observed = 0
        self._cheap_batch_writes_kept_beyond_bound = 0
        # The bound above is per response, so a model that keeps searching can
        # spend the same burst across many responses instead of one. Observed on
        # STATE-Bench travel: single rows issuing 1,003 and 1,209 searches, 99%
        # of them returning nothing, with almost every call carrying a freshly
        # permuted argument set. A read that returned nothing is evidence about
        # those arguments; guessing different ones is not grounded in anything
        # the row has seen. This second bound therefore counts no-result reads
        # per row and stops exposing further reads once the row exceeds it,
        # while still preserving every write. 0 disables it.
        self.cheap_max_no_result_reads_per_row = max(
            0, int(cheap_max_no_result_reads_per_row)
        )
        self._cheap_no_result_read_bound_hit_count = 0
        self._cheap_no_result_reads_dropped = 0
        self._cheap_max_no_result_reads_observed = 0
        self.max_residual_strong_calls = max(0, int(max_residual_strong_calls))
        self.cheap_system_prompt_suffix = str(cheap_system_prompt_suffix or "")
        self.strong_system_prompt_suffix = str(strong_system_prompt_suffix or "")
        # Whether the cheap prefix model reads its own transcript as native
        # Responses items instead of the text-flattened rendering. Off restores
        # the flattened shape byte for byte, so every registered result stands.
        # It changes the carrier only: the same calls, arguments and results are
        # sent, no reasoning item is carried, and neither runtime bound below
        # depends on the rendering.
        self.cheap_native_item_loop = bool(cheap_native_item_loop)
        if strong_proposer_mode not in {"native_turn", "shadow_tool_loop"}:
            raise ValueError(
                "strong_proposer_mode must be native_turn or shadow_tool_loop"
            )
        self.strong_proposer_mode = strong_proposer_mode
        if precommit_guard_mode not in {"off", "conservative", "staged"}:
            raise ValueError(
                "precommit_guard_mode must be off, conservative, or staged"
            )
        self.precommit_guard_mode = precommit_guard_mode
        if final_message_repair_mode not in {"off", "closure"}:
            raise ValueError("final_message_repair_mode must be off or closure")
        self.final_message_repair_mode = final_message_repair_mode
        # The contract already carries a finite tenure budget. Declaring its
        # remaining response count changes no authority or runtime limit; it
        # only prevents the successor from planning as if another round were
        # guaranteed.
        self.residual_round_budget_declared = bool(
            residual_round_budget_declared
        )
        # The closure checker's conditions are deterministic and known before
        # the closing message is written. Declaring them to the proposer turns a
        # rewrite it would have paid for into a message written admissibly once.
        self.closure_obligation_declared = bool(closure_obligation_declared)
        self.closing_contract_style = str(closing_contract_style)
        self.shadow_convergence_guard = bool(shadow_convergence_guard)
        # Whether a round that both acted and closed ends the loop. Off restores
        # the shape where completion is only ever declared by a round that spends
        # a whole strong call to return no action at all.
        self.shadow_closure_short_circuit = bool(shadow_closure_short_circuit)
        # Whether the successor's copy of the catalog keeps each tool's own
        # description. Off is the contract projection: the successor is handed
        # the admissible action surface the contract compiled -- names, argument
        # names, types, enums, defaults, required flags -- and not the domain's
        # prose. The argument contract is untouched, which is what the successor
        # actually spends its output on.
        self.successor_tool_descriptions = bool(successor_tool_descriptions)
        # Whether the last remaining residual slot is refused when the proposal
        # that would spend it is empty. The round count a compiled residual gets
        # is min(max_tool_rounds, remaining_budget), so on the last slot a round
        # that answers with actions has nothing left to pay for the closing call
        # and the compiler discards the residual whole. Refusing reaches the same
        # transcript -- hand back to the cheap agent -- without the call. Off by
        # default: the two pools disagree on whether that is free.
        self.refuse_unclosable_last_slot = bool(refuse_unclosable_last_slot)
        # Which model writes the closing message when the draft does not close.
        # The successor never returns text beside its tool calls -- measured at
        # zero short-circuits across 473 recorded rows -- so this rewrite
        # is not an occasional repair but one further call on almost every
        # acting row. It reports receipts rather than choosing actions, and the
        # same deterministic closure test judges it either way.
        if final_repair_role not in {"strong", "cheap_prefix_model"}:
            raise ValueError("final_repair_role must be strong or cheap_prefix_model")
        self.final_repair_role = final_repair_role
        # How much deliberation that closing message is worth. It is a different
        # kind of work from the round it follows: the acting round decides which
        # actions the residual admits, while this one reports receipts that are
        # already fixed and already checked. The phase ledger separates the two
        # and the separation is visible in the numbers -- on the old pool the
        # repair spends 72.1 percent of its output on reasoning to produce 188
        # tokens of prose. Left as None the phase inherits the successor's
        # effort, which is what every registered arm ran; setting it lower buys
        # a cheaper closing message and nothing else, and whether that message
        # still discloses what the scorer reads is a measurement, not an
        # argument.
        self.final_repair_reasoning_effort = (
            str(final_repair_reasoning_effort).strip()
            if final_repair_reasoning_effort
            else None
        )
        self.shadow_tool_handlers_factory = shadow_tool_handlers_factory
        self._frontier_decision: StreamFrontierDecision | None = None
        self._attempted_frontier_ids: set[tuple[str, str, tuple[str, ...], str]] = set()
        self._compile_results: list[CompiledTrueLiveProposal] = []
        self._provider_events: list[dict[str, Any]] = []
        self._strong_call_count = 0
        self._strong_budget_exhausted = False
        self._residual_mode_active = False
        self._pending_suffix_final_message: str | None = None
        self._staged_precommit_suffix: CompiledTrueLiveProposal | None = None
        self._staged_precommit_preview_count = 0
        self._staged_precommit_execute_count = 0
        self._staged_precommit_cancel_count = 0
        self._final_repair_role_counts: dict[str, int] = {}
        self._final_repair_skipped_closed_count = 0
        self._final_repair_invoked_count = 0
        self._final_repair_rechecked_closed_count = 0
        self._final_repair_rechecked_open_count = 0
        self._closure_block_counts: dict[str, int] = {}
        self._shadow_completion_declared_count = 0
        self._shadow_closure_short_circuit_count = 0
        self._shadow_convergence_attempt_count = 0
        self._shadow_reserved_closure_count = 0
        self._shadow_non_progress_stop_count = 0
        self._shadow_action_budget_stop_count = 0
        self._shadow_convergence_safety_block_count = 0
        self._unclosable_last_slot_refused_count = 0
        self._shadow_convergence_skip_reasons: dict[str, int] = {}
        self._shadow_convergence_stage_precommit_once = False

    def uses_harness_tool_execution(self) -> bool:
        return True

    def memory_tool_schemas(self) -> list[dict[str, Any]]:
        return []

    def memory_tool_handlers(self) -> dict[str, Any]:
        return {}

    def prepare_conversation(self, conversation: list[Any]) -> list[Any]:
        return conversation

    @property
    def token_usage(self) -> Any:
        return _token_usage_from_ledger(self.runtime_ledger())

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self._pending_suffix_final_message is not None:
            message = self._pending_suffix_final_message
            self._pending_suffix_final_message = None
            self._residual_mode_active = False
            return {"text": message, "tool_calls": []}

        staged = self._staged_precommit_suffix
        if staged is not None:
            if latest_user_confirms_precommit(conversation):
                self._staged_precommit_suffix = None
                self._pending_suffix_final_message = staged.assistant_message
                self._residual_mode_active = False
                self._staged_precommit_execute_count += 1
                self._compile_results.append(staged)
                return {"text": "", "tool_calls": list(staged.tool_calls)}
            if _latest_message_role(conversation) == "user":
                self._staged_precommit_suffix = None
                self._residual_mode_active = False
                self._staged_precommit_cancel_count += 1

        if _current_tool_round_count(conversation) >= (
            self.max_tool_rounds_per_assistant_turn - 1
        ):
            return {
                "text": "I have completed the available tool checks and will stop here.",
                "tool_calls": [],
            }

        if self._residual_mode_active and self._frontier_decision is not None:
            residual_response = self._try_strong_residual_step(
                system_prompt=system_prompt,
                conversation=conversation,
                tools=tools,
                decision=self._frontier_decision,
            )
            if residual_response is not None:
                return residual_response

        decision = decide_stream_prefix(
            self.domain,
            {"conversation": conversation},
            row_index=self.row_index,
            min_assistant_turns=self.min_cheap_assistant_turns,
        )
        if decision.opens_strong_call:
            frontier_id = self._frontier_id(decision, conversation)
            if self._strong_call_count >= self.max_residual_strong_calls:
                self._strong_budget_exhausted = True
            elif frontier_id not in self._attempted_frontier_ids:
                self._frontier_decision = decision
                self._residual_mode_active = True
                self._attempted_frontier_ids.add(frontier_id)
                residual_response = self._try_strong_residual_step(
                    system_prompt=system_prompt,
                    conversation=conversation,
                    tools=tools,
                    decision=decision,
                )
                if residual_response is not None:
                    return residual_response

        return self._bound_cheap_tool_call_batch(
            self._call_provider(
                role="cheap",
                client=self.cheap_client,
                model=self.cheap.model,
                reasoning_effort=self.cheap.reasoning_effort,
                system_prompt=self._cheap_system_prompt(system_prompt),
                conversation=conversation,
                tools=tools,
                call_phase="cheap_live",
                native_items=self.cheap_native_item_loop,
            ),
            conversation=conversation,
        )

    def ingest_trajectory(self, trajectory: Any) -> None:
        metadata = getattr(trajectory, "metadata", None)
        if not isinstance(metadata, dict):
            return
        metadata["papc_true_live_cfrc"] = self.public_summary()
        metadata["papc_runtime_ledger"] = self.runtime_ledger()

    def live_profile(self) -> StateBenchCFRCProfile:
        """The operating point this agent is actually running.

        The public payload is built from this, so a report cannot describe a
        budget or a declaration the run did not use.
        """

        return StateBenchCFRCProfile(
            max_residual_strong_calls=self.max_residual_strong_calls,
            declare_remaining_responses=self.residual_round_budget_declared,
            declare_closure_admissibility=self.closure_obligation_declared,
            refuse_unclosable_last_slot=self.refuse_unclosable_last_slot,
            closing_contract_style=self.closing_contract_style,
        )

    def public_summary(self) -> dict[str, Any]:
        decision = self._frontier_decision
        residual_graph_summaries = [
            result.residual_graph.public_summary()
            for result in self._compile_results
            if result.residual_graph is not None
        ]
        return {
            "schema": SCHEMA,
            "execution_contract": online_execution_contract(self.live_profile()),
            "method_profile": "r4_grounded_schema_cfrc",
            "domain": self.domain,
            "frontier_found": decision is not None,
            "frontier": decision.to_public_json() if decision else None,
            "cheap_model": self.cheap.model,
            "cheap_reasoning_effort": self.cheap.reasoning_effort,
            "strong_model": self.strong.model,
            "strong_reasoning_effort": self.strong.reasoning_effort,
            "min_cheap_assistant_turns": self.min_cheap_assistant_turns,
            "max_tool_rounds_per_assistant_turn": self.max_tool_rounds_per_assistant_turn,
            "cheap_max_tool_calls_per_response": (
                self.cheap_max_tool_calls_per_response
            ),
            "cheap_batch_truncated_response_count": (
                self._cheap_batch_truncated_response_count
            ),
            "cheap_batch_dropped_call_count": self._cheap_batch_dropped_call_count,
            "cheap_batch_max_calls_observed": self._cheap_batch_max_calls_observed,
            "cheap_batch_writes_kept_beyond_bound": (
                self._cheap_batch_writes_kept_beyond_bound
            ),
            "cheap_max_no_result_reads_per_row": (
                self.cheap_max_no_result_reads_per_row
            ),
            "cheap_no_result_read_bound_hit_count": (
                self._cheap_no_result_read_bound_hit_count
            ),
            "cheap_no_result_reads_dropped": self._cheap_no_result_reads_dropped,
            "cheap_max_no_result_reads_observed": (
                self._cheap_max_no_result_reads_observed
            ),
            "max_residual_strong_calls": self.max_residual_strong_calls,
            "cheap_system_prompt_suffix_enabled": bool(
                self.cheap_system_prompt_suffix.strip()
            ),
            "cheap_native_item_loop": bool(self.cheap_native_item_loop),
            "strong_proposer_mode": self.strong_proposer_mode,
            "private_reached_state_replica": (
                self.strong_proposer_mode == "shadow_tool_loop"
            ),
            "incremental_live_receipt_grounding": True,
            "precommit_guard_mode": self.precommit_guard_mode,
            "final_message_repair_mode": self.final_message_repair_mode,
            "final_repair_policy": (
                "cheap_prefix_model"
                if self.final_repair_role == "cheap_prefix_model"
                else "strong_only"
            ),
            "final_repair_role": self.final_repair_role,
            "final_repair_reasoning_effort": self.final_repair_reasoning_effort,
            "final_repair_rechecked_closed_count": (
                self._final_repair_rechecked_closed_count
            ),
            "final_repair_rechecked_open_count": (
                self._final_repair_rechecked_open_count
            ),
            "final_repair_skipped_closed_count": (
                self._final_repair_skipped_closed_count
            ),
            "final_repair_invoked_count": self._final_repair_invoked_count,
            "residual_round_budget_declared": (
                self.residual_round_budget_declared
            ),
            "closure_obligation_declared": self.closure_obligation_declared,
            "closing_contract_style": self.closing_contract_style,
            "closure_block_counts": dict(sorted(self._closure_block_counts.items())),
            "strong_context_mode": "full_context",
            "strong_response_reuse": False,
            "shadow_convergence_guard": self.shadow_convergence_guard,
            "shadow_convergence_attempt_count": self._shadow_convergence_attempt_count,
            "shadow_reserved_closure_count": self._shadow_reserved_closure_count,
            "shadow_non_progress_stop_count": self._shadow_non_progress_stop_count,
            "shadow_completion_declared_count": (
                self._shadow_completion_declared_count
            ),
            "shadow_closure_short_circuit": self.shadow_closure_short_circuit,
            "successor_tool_descriptions": self.successor_tool_descriptions,
            "shadow_closure_short_circuit_count": (
                self._shadow_closure_short_circuit_count
            ),
            "shadow_action_budget_stop_count": self._shadow_action_budget_stop_count,
            "proposal_action_budget": self.max_tool_rounds_per_assistant_turn,
            "shadow_convergence_safety_block_count": (
                self._shadow_convergence_safety_block_count
            ),
            "refuse_unclosable_last_slot": self.refuse_unclosable_last_slot,
            "unclosable_last_slot_refused_count": (
                self._unclosable_last_slot_refused_count
            ),
            "shadow_convergence_skip_reasons": dict(
                sorted(self._shadow_convergence_skip_reasons.items())
            ),
            "shadow_converged_early_count": self._shadow_non_progress_stop_count,
            "shadow_forced_closure_count": self._shadow_reserved_closure_count,
            "adapter_shadow_tool_policy": "domain_hook_enabled",
            "travel_final_confirmation_no_approval_write_filter": (
                self.domain == "travel"
            ),
            "residual_strong_call_count": self._strong_call_count,
            "failed_strong_provider_attempt_count": (
                self._failed_strong_provider_attempt_count()
            ),
            "total_strong_provider_call_count": self._strong_provider_call_count(),
            "staged_precommit_pending": self._staged_precommit_suffix is not None,
            "staged_precommit_preview_count": self._staged_precommit_preview_count,
            "staged_precommit_execute_count": self._staged_precommit_execute_count,
            "staged_precommit_cancel_count": self._staged_precommit_cancel_count,
            "final_repair_role_counts": dict(
                sorted(self._final_repair_role_counts.items())
            ),
            "strong_budget_exhausted": self._strong_budget_exhausted,
            "residual_mode_active": self._residual_mode_active,
            "attempted_frontier_count": len(self._attempted_frontier_ids),
            "compiler_execute_count": sum(
                1 for result in self._compile_results if result.decision == "execute"
            ),
            "compiler_abstain_count": sum(
                1 for result in self._compile_results if result.decision == "abstain"
            ),
            "compiler_block_count": sum(
                1 for result in self._compile_results if result.decision == "block"
            ),
            "compiler_results": [
                result.public_summary() for result in self._compile_results
            ],
            "residual_graph_count": len(residual_graph_summaries),
            "residual_graph_node_count": sum(
                int(summary["node_count"]) for summary in residual_graph_summaries
            ),
            "residual_graph_tool_node_count": sum(
                int(summary["tool_node_count"]) for summary in residual_graph_summaries
            ),
            "residual_graph_edge_count": sum(
                int(summary["edge_count"]) for summary in residual_graph_summaries
            ),
            "provider_event_count": len(self._provider_events),
            "phase_ledger": _phase_ledger_from_events(self._provider_events),
            "raw_provider_payload_publicly_disclosed": False,
        }

    def runtime_ledger(self) -> dict[str, Any]:
        ledger = RuntimeLedger()
        for event in self._provider_events:
            role = str(event.get("role") or "")
            if event.get("status") == "ok":
                ledger.add_usage(
                    role,
                    event.get("usage")
                    if isinstance(event.get("usage"), Mapping)
                    else None,
                    latency_seconds=(
                        float(event.get("latency_seconds"))
                        if event.get("latency_seconds") is not None
                        else None
                    ),
                    cost_usd=None,
                    call_count=1,
                )
            else:
                ledger.add(
                    RoleCall(
                        role=role,
                        call_count=1,
                        latency_seconds=(
                            float(event.get("latency_seconds"))
                            if event.get("latency_seconds") is not None
                            else None
                        ),
                        token_usage_available=False,
                    )
                )
        summary = ledger.public_summary()
        summary["phase_ledger"] = _phase_ledger_from_events(self._provider_events)
        return summary

    def provider_events_public(self) -> list[dict[str, Any]]:
        return [
            {
                "role": event.get("role"),
                "status": event.get("status"),
                "model": event.get("model"),
                "reasoning_effort": event.get("reasoning_effort"),
                "latency_seconds": event.get("latency_seconds"),
                "call_phase": event.get("call_phase"),
                "usage": event.get("usage") if event.get("status") == "ok" else None,
                "error_type": event.get("error_type")
                if event.get("status") != "ok"
                else None,
            }
            for event in self._provider_events
        ]

    def _call_strong_compiler_proposal(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[Mapping[str, Any]],
        tools: list[dict[str, Any]],
        decision: StreamFrontierDecision,
    ) -> CompiledTrueLiveProposal:
        proposal_tools = _frontier_allowed_tools(
            decision,
            tools,
            domain=self.domain,
            conversation=conversation,
            keep_tool_description=self.successor_tool_descriptions,
        )
        if self.strong_proposer_mode == "shadow_tool_loop":
            return self._call_strong_shadow_tool_loop_proposal(
                system_prompt=system_prompt,
                conversation=conversation,
                tools=proposal_tools,
                decision=decision,
            )
        proposal = self._call_provider(
            role="strong",
            client=self.strong_client,
            model=self.strong.model,
            reasoning_effort=self.strong.reasoning_effort,
            system_prompt=strong_native_proposal_instruction(
                system_prompt=system_prompt,
                strong_system_prompt_suffix=self.strong_system_prompt_suffix,
                frontier=decision.to_public_json(),
                max_tool_calls=self.max_tool_rounds_per_assistant_turn,
            ),
            conversation=conversation,
            tools=proposal_tools,
            call_phase="strong_native_proposal",
        )
        self._strong_call_count += 1
        return compile_native_true_live_proposal(
            assistant_message=str(proposal.get("text") or ""),
            tool_calls=proposal.get("tool_calls")
            if isinstance(proposal.get("tool_calls"), list)
            else [],
            tools=proposal_tools,
            max_tool_calls=self.max_tool_rounds_per_assistant_turn,
        )

    def _proposal_action_budget_exceeded(
        self, proposed_tool_calls: Sequence[Mapping[str, Any]]
    ) -> bool:
        """Report whether the accumulated proposal can no longer compile.

        The compiler caps how many tool calls one compiled residual may carry,
        and the shadow loop only ever appends to that list. Once the cap is
        passed no later round can bring it back under, so every further strong
        call is spent on a proposal that is already certain to be blocked.
        """

        budget = max(0, int(self.max_tool_rounds_per_assistant_turn))
        return len(proposed_tool_calls) > budget

    def _call_strong_shadow_tool_loop_proposal(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[Mapping[str, Any]],
        tools: list[dict[str, Any]],
        decision: StreamFrontierDecision,
    ) -> CompiledTrueLiveProposal:
        tools = _frontier_allowed_tools(
            decision,
            tools,
            domain=self.domain,
            conversation=conversation,
            keep_tool_description=self.successor_tool_descriptions,
        )
        if self.shadow_tool_handlers_factory is None:
            return CompiledTrueLiveProposal(
                decision="block",
                compile_errors=("shadow_tool_handlers_unavailable",),
            )
        remaining_budget = self._strong_provider_calls_remaining()
        if remaining_budget <= 0:
            self._strong_budget_exhausted = True
            return CompiledTrueLiveProposal(
                decision="block",
                compile_errors=("strong_budget_exhausted",),
            )
        try:
            shadow_handlers = dict(self.shadow_tool_handlers_factory(conversation))
        except Exception as exc:
            return CompiledTrueLiveProposal(
                decision="block",
                compile_errors=(f"shadow_tool_replay_failed:{type(exc).__name__}",),
            )
        allowed_tool_names = _tool_schema_names(tools)
        proposed_tool_calls: list[dict[str, Any]] = []
        shadow_tool_results: list[dict[str, Any]] = []
        residual_graph_nodes: list[ResidualGraphNode] = []
        full_input_items: list[Any] = provider_safe_live_conversation(conversation)
        final_text = ""
        # The closing message the successor has written most recently, whether or
        # not that round also carried actions. A round that both acts and closes
        # already holds everything the user needs to be told, so the message it
        # returns is a real candidate rather than a discardable preamble.
        latest_round_text = ""
        compile_errors: list[str] = []
        executed_call_signatures: set[str] = set()
        reserved_closure = False

        max_rounds = min(self.max_tool_rounds_per_assistant_turn, remaining_budget)
        for _round_index in range(max_rounds):
            if (
                self.shadow_convergence_guard
                and self._strong_provider_calls_remaining() <= 1
            ):
                decision_ = self._shadow_convergence_decision(
                    reason="last_budget_slot",
                    conversation=conversation,
                    proposed_tool_calls=proposed_tool_calls,
                    shadow_tool_results=shadow_tool_results,
                    tools=tools,
                    compile_errors=compile_errors,
                )
                if decision_.action == "reserve_closure":
                    reserved_closure = True
                    self._shadow_reserved_closure_count += 1
                    # Reserving the slot no longer means handing the closing call
                    # a blank page: whatever the successor last wrote is a draft,
                    # and a draft that already closes spends no call at all.
                    final_text = latest_round_text
                    break
                if decision_.action == "safety_block":
                    return CompiledTrueLiveProposal(
                        decision="block",
                        compile_errors=(
                            f"shadow_convergence_safety_block:{decision_.reason}",
                        ),
                    )
            response = self._complete_with_tools_recorded(
                role="strong",
                client=self.strong_client,
                model=self.strong.model,
                reasoning_effort=self.strong.reasoning_effort,
                instructions=strong_native_proposal_instruction(
                    system_prompt=system_prompt,
                    strong_system_prompt_suffix=self.strong_system_prompt_suffix,
                    frontier=decision.to_public_json(),
                    frontier_in_input_packet=False,
                    max_tool_calls=self.max_tool_rounds_per_assistant_turn,
                    remaining_residual_rounds=(
                        self._strong_provider_calls_remaining()
                        if self.residual_round_budget_declared
                        else None
                    ),
                    closure_obligation_declared=self.closure_obligation_declared,
                    closing_contract_style=self.closing_contract_style,
                ),
                input_items=full_input_items,
                tools=tools,
                previous_response_id=None,
                call_phase="strong_shadow_round",
            )
            self._strong_call_count += 1
            response_output_items = _normalise_items(
                list(getattr(response, "output", []) or [])
            )
            response_tool_calls = _response_tool_calls(response)
            round_text = str(getattr(response, "output_text", "") or "").strip()
            if round_text:
                latest_round_text = round_text
            if not response_tool_calls:
                # A round that adds no action declares the residual complete and
                # carries the closing message with it. This is the loop's own
                # completion signal, not the budget running out.
                self._shadow_completion_declared_count += 1
                final_text = round_text
                break

            round_signatures = [
                _shadow_call_signature(call) for call in response_tool_calls
            ]
            if (
                self.shadow_convergence_guard
                and round_signatures
                and all(sig in executed_call_signatures for sig in round_signatures)
            ):
                self._shadow_non_progress_stop_count += 1
                decision_ = self._shadow_convergence_decision(
                    reason="non_progress_repeated_tool_calls",
                    conversation=conversation,
                    proposed_tool_calls=proposed_tool_calls,
                    shadow_tool_results=shadow_tool_results,
                    tools=tools,
                    compile_errors=compile_errors,
                )
                if decision_.action == "reserve_closure":
                    reserved_closure = True
                    self._shadow_reserved_closure_count += 1
                    # Reserving the slot no longer means handing the closing call
                    # a blank page: whatever the successor last wrote is a draft,
                    # and a draft that already closes spends no call at all.
                    final_text = latest_round_text
                    break
                return CompiledTrueLiveProposal(
                    decision="block",
                    compile_errors=(
                        f"shadow_convergence_safety_block:{decision_.reason}",
                    ),
                )

            tool_result_items: list[dict[str, str]] = []
            skipped_call_ids: set[str] = set()
            prior_node_ids = tuple(node.node_id for node in residual_graph_nodes)
            for call in response_tool_calls:
                name = str(call.get("name") or "")
                signature = _shadow_call_signature(call)
                if (
                    self.shadow_convergence_guard
                    and signature in executed_call_signatures
                ):
                    self._record_shadow_convergence_skip("duplicate_tool_call_skipped")
                    skipped_call_ids.add(str(call.get("call_id") or ""))
                    continue
                args = (
                    dict(call.get("arguments"))
                    if isinstance(call.get("arguments"), Mapping)
                    else {}
                )
                if allowed_tool_names and name not in allowed_tool_names:
                    compile_errors.append("proposal_tool_name_not_allowed")
                    continue
                handler = shadow_handlers.get(name)
                if handler is None:
                    compile_errors.append("shadow_tool_handler_missing")
                    continue
                try:
                    result = handler(args)
                except Exception as exc:
                    compile_errors.append(
                        f"shadow_tool_handler_failed:{type(exc).__name__}"
                    )
                    continue
                tool_call = {"name": name, "arguments": copy.deepcopy(args)}
                call_index = len(proposed_tool_calls)
                proposed_tool_calls.append(tool_call)
                shadow_tool_results.append(
                    {"name": name, "arguments": args, "result": result}
                )
                residual_graph_nodes.append(
                    ResidualGraphNode.from_tool_call(
                        node_id=f"action_{call_index:03d}",
                        round_index=_round_index,
                        dependency_ids=prior_node_ids,
                        tool_call=tool_call,
                        receipt=result,
                    )
                )
                executed_call_signatures.add(signature)
                tool_result_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(call.get("call_id") or ""),
                        "output": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
            if tool_result_items:
                if skipped_call_ids:
                    response_output_items = _drop_dangling_call_items(
                        response_output_items,
                        skipped_call_ids,
                    )
                full_input_items = [
                    *full_input_items,
                    *response_output_items,
                    *tool_result_items,
                ]
            if self._proposal_action_budget_exceeded(proposed_tool_calls):
                self._shadow_action_budget_stop_count += 1
                compile_errors.append("proposal_tool_call_budget_exceeded")
            if compile_errors:
                break
            # A round that acted and also stated the outcome has nothing left to
            # establish: the same closure test that decides whether the closing
            # message needs rewriting decides here that no further round is owed.
            # Without this the successor spends one more call only to say it is
            # done, and the message that call returns is the one it already wrote.
            if (
                self.shadow_closure_short_circuit
                and latest_round_text
                and proposed_tool_calls
                and self._draft_closes_executed_writes(
                    conversation=conversation,
                    shadow_tool_results=shadow_tool_results,
                    draft_final_message=latest_round_text,
                )
            ):
                self._shadow_closure_short_circuit_count += 1
                final_text = latest_round_text
                break

        if compile_errors:
            if self.shadow_convergence_guard:
                self._record_shadow_convergence_safety_block(compile_errors)
            return CompiledTrueLiveProposal(
                decision="block",
                compile_errors=tuple(sorted(set(compile_errors))),
            )
        if not final_text:
            if not reserved_closure:
                if self._strong_provider_calls_remaining() <= 0:
                    self._strong_budget_exhausted = True
                    reason = "strong_budget_exhausted_before_final"
                else:
                    reason = "shadow_tool_loop_no_final_message"
                return CompiledTrueLiveProposal(
                    decision="block",
                    compile_errors=(reason,),
                )
        if self.final_message_repair_mode == "closure" and proposed_tool_calls:
            if self._draft_closes_executed_writes(
                conversation=conversation,
                shadow_tool_results=shadow_tool_results,
                draft_final_message=final_text,
            ):
                self._final_repair_skipped_closed_count += 1
            else:
                self._final_repair_invoked_count += 1
                final_text = self._repair_final_message(
                    system_prompt=system_prompt,
                    conversation=conversation,
                    proposed_tool_calls=proposed_tool_calls,
                    shadow_tool_results=shadow_tool_results,
                    draft_final_message=final_text,
                )
                # The rewrite was never judged by the test that demanded it, so
                # a repair that still misreports the reached state reached the
                # user indistinguishable from one that did not. Reporting the
                # outcome does not change which message is sent; it is what
                # tells a repair policy that works from one that only costs.
                if self._draft_closes_executed_writes(
                    conversation=conversation,
                    shadow_tool_results=shadow_tool_results,
                    draft_final_message=final_text,
                ):
                    self._final_repair_rechecked_closed_count += 1
                else:
                    self._final_repair_rechecked_open_count += 1
        terminal_round = (
            max(
                (node.round_index for node in residual_graph_nodes),
                default=-1,
            )
            + 1
        )
        residual_graph = IncrementalResidualGraph(
            nodes=(
                *residual_graph_nodes,
                ResidualGraphNode.terminal(
                    round_index=terminal_round,
                    dependency_ids=tuple(node.node_id for node in residual_graph_nodes),
                    assistant_message=final_text,
                ),
            )
        )
        return compile_native_true_live_proposal(
            assistant_message=final_text,
            tool_calls=proposed_tool_calls,
            tools=tools,
            max_tool_calls=self.max_tool_rounds_per_assistant_turn,
            residual_graph=residual_graph,
        )

    def _shadow_convergence_decision(
        self,
        *,
        reason: str,
        conversation: Sequence[Mapping[str, Any]],
        proposed_tool_calls: Sequence[Mapping[str, Any]],
        shadow_tool_results: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        compile_errors: Sequence[str],
    ) -> ConvergenceDecision:
        self._shadow_convergence_attempt_count += 1
        state = self._shadow_loop_state(
            proposed_tool_calls=proposed_tool_calls,
            shadow_tool_results=shadow_tool_results,
            remaining_strong_budget=self._strong_provider_calls_remaining(),
            compile_errors=compile_errors,
            candidate=None,
            requires_precommit_confirmation=False,
        )
        if self.final_message_repair_mode != "closure":
            self._record_shadow_convergence_skip("final_message_repair_disabled")
            return ConvergenceDecision(
                action="continue",
                reason="final_message_repair_disabled",
                state=state,
            )
        if compile_errors:
            self._record_shadow_convergence_safety_block(compile_errors)
            return ConvergenceDecision(
                action="safety_block",
                reason="compile_or_handler_error",
                state=state,
            )
        if not proposed_tool_calls:
            unclosable_last_slot = (
                reason == "last_budget_slot" and self.refuse_unclosable_last_slot
            )
            action: Literal["continue", "safety_block"] = (
                "continue"
                if reason == "last_budget_slot" and not unclosable_last_slot
                else "safety_block"
            )
            self._record_shadow_convergence_skip("candidate_empty")
            if unclosable_last_slot:
                self._unclosable_last_slot_refused_count += 1
            if action == "safety_block":
                self._shadow_convergence_safety_block_count += 1
            return ConvergenceDecision(
                action=action,
                reason="candidate_empty",
                state=state,
            )
        if not shadow_tool_results:
            self._record_shadow_convergence_safety_block(
                ("no_successful_shadow_tool_result",)
            )
            return ConvergenceDecision(
                action="safety_block",
                reason="no_successful_shadow_tool_result",
                state=state,
            )
        candidate = compile_native_true_live_proposal(
            assistant_message="",
            tool_calls=proposed_tool_calls,
            tools=tools,
            max_tool_calls=self.max_tool_rounds_per_assistant_turn,
        )
        requires_precommit_confirmation = true_live_precommit_requires_confirmation(
            domain=self.domain,
            conversation=conversation,
            compiled=candidate,
        )
        state = self._shadow_loop_state(
            proposed_tool_calls=proposed_tool_calls,
            shadow_tool_results=shadow_tool_results,
            remaining_strong_budget=self._strong_provider_calls_remaining(),
            compile_errors=compile_errors,
            candidate=candidate,
            requires_precommit_confirmation=requires_precommit_confirmation,
        )
        if candidate.decision != "execute":
            reasons = candidate.compile_errors or ("candidate_compile_block",)
            self._record_shadow_convergence_safety_block(reasons)
            return ConvergenceDecision(
                action="safety_block",
                reason=str(reasons[0]),
                state=state,
            )
        if not candidate.tool_calls:
            self._record_shadow_convergence_safety_block(("selected_suffix_empty",))
            return ConvergenceDecision(
                action="safety_block",
                reason="selected_suffix_empty",
                state=state,
            )
        if self._strong_provider_calls_remaining() <= 0:
            self._record_shadow_convergence_safety_block(
                ("no_remaining_strong_budget",)
            )
            return ConvergenceDecision(
                action="safety_block",
                reason="no_remaining_strong_budget",
                state=state,
            )
        if requires_precommit_confirmation:
            self._shadow_convergence_stage_precommit_once = True
        return ConvergenceDecision(
            action="reserve_closure",
            reason=str(reason or "reserve_closure"),
            state=state,
        )

    def _shadow_loop_state(
        self,
        *,
        proposed_tool_calls: Sequence[Mapping[str, Any]],
        shadow_tool_results: Sequence[Mapping[str, Any]],
        remaining_strong_budget: int,
        compile_errors: Sequence[str],
        candidate: CompiledTrueLiveProposal | None,
        requires_precommit_confirmation: bool,
    ) -> ShadowLoopState:
        return ShadowLoopState(
            successful_tool_result_count=len(shadow_tool_results),
            selected_tool_call_count=len(proposed_tool_calls),
            remaining_strong_budget=max(0, int(remaining_strong_budget)),
            has_compile_or_handler_error=bool(compile_errors),
            candidate_decision=str(candidate.decision if candidate else ""),
            requires_precommit_confirmation=bool(requires_precommit_confirmation),
        )

    def _record_shadow_convergence_safety_block(
        self,
        reasons: Sequence[str],
    ) -> None:
        self._shadow_convergence_safety_block_count += 1
        for reason in reasons:
            self._record_shadow_convergence_skip(reason)

    def _record_shadow_convergence_skip(self, reason: str) -> None:
        token = _public_reason_token(reason)
        if not token:
            return
        self._shadow_convergence_skip_reasons[token] = (
            int(self._shadow_convergence_skip_reasons.get(token, 0)) + 1
        )

    def _try_strong_residual_step(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[Mapping[str, Any]],
        tools: list[dict[str, Any]],
        decision: StreamFrontierDecision,
    ) -> dict[str, Any] | None:
        if self._strong_provider_calls_remaining() <= 0:
            self._strong_budget_exhausted = True
            self._residual_mode_active = False
            return None
        compiled = self._call_strong_compiler_proposal(
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
            decision=decision,
        )
        force_staged_precommit = self._shadow_convergence_stage_precommit_once
        self._shadow_convergence_stage_precommit_once = False
        if force_staged_precommit and true_live_precommit_requires_confirmation(
            domain=self.domain,
            conversation=conversation,
            compiled=compiled,
        ):
            preview_message = staged_true_live_precommit_message(
                domain=self.domain,
                compiled=compiled,
            )
            self._staged_precommit_suffix = compiled
            self._staged_precommit_preview_count += 1
            self._residual_mode_active = False
            preview = CompiledTrueLiveProposal(
                decision="execute",
                assistant_message=preview_message,
                tool_calls=(),
                compile_errors=tuple(
                    sorted(
                        {
                            *compiled.compile_errors,
                            "precommit_confirmation_staged",
                        }
                    )
                ),
            )
            self._compile_results.append(preview)
            return {"text": preview_message, "tool_calls": []}
        if self.precommit_guard_mode == "conservative":
            compiled = apply_true_live_precommit_guard(
                domain=self.domain,
                conversation=conversation,
                compiled=compiled,
                strong_calls_remaining=self.max_residual_strong_calls
                - self._strong_provider_call_count(),
            )
        elif (
            self.precommit_guard_mode == "staged"
            and true_live_precommit_requires_confirmation(
                domain=self.domain,
                conversation=conversation,
                compiled=compiled,
            )
        ):
            preview_message = staged_true_live_precommit_message(
                domain=self.domain,
                compiled=compiled,
            )
            self._staged_precommit_suffix = compiled
            self._staged_precommit_preview_count += 1
            self._residual_mode_active = False
            preview = CompiledTrueLiveProposal(
                decision="execute",
                assistant_message=preview_message,
                tool_calls=(),
                compile_errors=tuple(
                    sorted(
                        {
                            *compiled.compile_errors,
                            "precommit_confirmation_staged",
                        }
                    )
                ),
            )
            self._compile_results.append(preview)
            return {"text": preview_message, "tool_calls": []}
        self._compile_results.append(compiled)
        if compiled.decision == "execute":
            if compiled.tool_calls:
                self._pending_suffix_final_message = compiled.assistant_message
                return {
                    "text": "",
                    "tool_calls": list(compiled.tool_calls),
                }
            return {
                "text": compiled.assistant_message,
                "tool_calls": list(compiled.tool_calls),
            }
        self._residual_mode_active = False
        return None

    def _domain_write_tool_names(self) -> frozenset[str]:
        try:
            from papc.certspec.state_bench.online_residual_selector import (
                adapter_for_domain,
            )

            return frozenset(
                str(name) for name in adapter_for_domain(self.domain).write_tool_names
            )
        except Exception:
            return frozenset()

    def _draft_closes_executed_writes(
        self,
        *,
        conversation: Sequence[Mapping[str, Any]],
        shadow_tool_results: Sequence[Mapping[str, Any]],
        draft_final_message: str,
    ) -> bool:
        """Whether the draft already discharges the writes the proposal executed.

        Fail closed in every direction the check cannot decide: without a
        declared write surface, without an executed write, or without a draft
        that states every value the writes newly established, the closing repair
        runs exactly as it did before. The check itself makes no provider call.
        """

        write_names = self._domain_write_tool_names()
        if not write_names:
            self._record_closure_block("no_declared_write_surface")
            return False
        report = draft_closure_report(
            draft_final_message=draft_final_message,
            shadow_tool_results=shadow_tool_results,
            write_tool_names=write_names,
            request=retained_request_text(conversation),
        )
        if not report.closes:
            for mark, present in (
                ("draft_absent", not report.draft_present),
                ("no_executed_write", report.checked_write_count <= 0),
                ("unrealized_write_fields", bool(report.unrealized_fields)),
                ("contradicted_fields", bool(report.contradicted_fields)),
                ("open_obligations", bool(report.open_obligation_marks)),
                ("ungrounded_values", bool(report.ungrounded_values)),
            ):
                if present:
                    self._record_closure_block(mark)
        return report.closes

    def _record_closure_block(self, mark: str) -> None:
        self._closure_block_counts[mark] = (
            int(self._closure_block_counts.get(mark, 0)) + 1
        )

    def _repair_final_message(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[Mapping[str, Any]],
        proposed_tool_calls: Sequence[Mapping[str, Any]],
        shadow_tool_results: Sequence[Mapping[str, Any]],
        draft_final_message: str,
    ) -> str:
        tool_calls = list(proposed_tool_calls)
        tool_results = list(shadow_tool_results)
        payload: dict[str, Any] = {
            "domain": self.domain,
            "conversation_tail": list(provider_safe_live_conversation(conversation))[
                -8:
            ],
            "selected_tool_results": tool_results,
            "draft_final_message": str(draft_final_message or ""),
        }
        # Each selected result already states the name and arguments of the call
        # that produced it, so the call list is carried only when it is not a
        # projection of the results.
        if not _tool_calls_are_projection_of_results(tool_calls, tool_results):
            payload["selected_tool_calls"] = tool_calls
        draft = str(draft_final_message or "").strip()
        if self.final_repair_role == "cheap_prefix_model":
            return self._cheap_prefix_final_repair(
                payload=payload,
                draft=draft,
                system_prompt=system_prompt,
            )
        return self._strong_only_final_repair(
            payload=payload,
            draft=draft,
            system_prompt=system_prompt,
        )

    def _cheap_prefix_final_repair(
        self,
        *,
        payload: Mapping[str, Any],
        draft: str,
        system_prompt: str,
    ) -> str:
        """Closure repair written by the prefix model instead of the successor.

        The closing message reports what the admitted actions established; it
        does not decide them. Every value it may state is already fixed by the
        receipts the payload carries, and the same deterministic closure test
        that admits a successor draft admits this one, so the authoring model is
        an operator choice rather than a method claim. Billing follows the model
        that ran, which keeps the strong ledger measuring completion only, and
        the residual budget is left untouched because no residual round is spent.
        """

        if self._strong_provider_calls_remaining() <= 0:
            self._strong_budget_exhausted = True
            return draft
        repaired = self._call_chat_provider(
            role="cheap",
            client=self.cheap_client,
            model=self.cheap.model,
            reasoning_effort=self.cheap.reasoning_effort,
            messages=_final_repair_messages(
                payload,
                system_prompt=system_prompt,
                system_prompt_suffix=self.strong_system_prompt_suffix,
            ),
            call_phase="cheap_final_repair",
        ).strip()
        # The slot is still spent even though the bill moved. Measured the other
        # way first: leaving the residual budget untouched let the shadow loop
        # take the freed slots, 149 rounds became 220, and the arm cost more
        # than the strong rewrite it replaced. What this policy changes is which
        # model writes the closing sentence, not how many rounds a residual gets.
        self._strong_call_count += 1
        self._record_final_repair_role("cheap")
        return repaired or draft

    def _strong_only_final_repair(
        self,
        *,
        payload: Mapping[str, Any],
        draft: str,
        system_prompt: str,
    ) -> str:
        """Historical closure repair: a single budget-gated strong rewrite.

        This is the single current STATE-Bench method-arm closure policy. It
        keeps the method in residual-completion form: the same residual strong
        model that inspected the frontier authors one bounded closing message,
        with no cheap rewrite and no alternate repair cascade.
        """

        if self._strong_provider_calls_remaining() <= 0:
            self._strong_budget_exhausted = True
            return draft
        repaired = self._call_chat_provider(
            role="strong",
            client=self.strong_client,
            model=self.strong.model,
            reasoning_effort=(
                self.final_repair_reasoning_effort or self.strong.reasoning_effort
            ),
            messages=_final_repair_messages(
                payload,
                system_prompt=system_prompt,
                system_prompt_suffix=self.strong_system_prompt_suffix,
            ),
            call_phase="strong_final_repair",
        ).strip()
        self._strong_call_count += 1
        self._record_final_repair_role("strong")
        return repaired or draft

    def _record_final_repair_role(self, role: str) -> None:
        self._final_repair_role_counts[role] = (
            int(self._final_repair_role_counts.get(role, 0)) + 1
        )

    def _strong_provider_call_count(self) -> int:
        return sum(
            1 for event in self._provider_events if event.get("role") == "strong"
        )

    def _failed_strong_provider_attempt_count(self) -> int:
        return sum(
            1
            for event in self._provider_events
            if event.get("role") == "strong" and event.get("status") != "ok"
        )

    def _strong_provider_calls_remaining(self) -> int:
        return max(
            0, self.max_residual_strong_calls - self._strong_provider_call_count()
        )

    def _call_chat_provider(
        self,
        *,
        role: str,
        client: Any,
        model: str,
        reasoning_effort: str,
        messages: list[dict[str, str]],
        call_phase: str,
    ) -> str:
        drain = getattr(client, "drain_papc_call_events", None)
        if callable(drain):
            drain()
        started = time.monotonic()
        try:
            response = client.complete_chat(
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            self._provider_events.append(
                {
                    "role": role,
                    "status": "error",
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "latency_seconds": round(time.monotonic() - started, 6),
                    "call_phase": call_phase,
                    "error_type": type(exc).__name__,
                }
            )
            raise
        if callable(drain):
            drained = list(drain())
        else:
            drained = []
        if drained:
            for event in drained:
                public_event = dict(event)
                public_event["role"] = role
                public_event.setdefault("model", model)
                public_event.setdefault("reasoning_effort", reasoning_effort)
                public_event["call_phase"] = call_phase
                self._provider_events.append(public_event)
        else:
            self._provider_events.append(
                {
                    "role": role,
                    "status": "ok",
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "latency_seconds": round(time.monotonic() - started, 6),
                    "call_phase": call_phase,
                    "usage": _response_usage_dict(response),
                }
            )
        return str(response)

    def _complete_with_tools_recorded(
        self,
        *,
        role: str,
        client: Any,
        model: str,
        reasoning_effort: str,
        instructions: str,
        input_items: list[Any],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        call_phase: str,
    ) -> Any:
        drain = getattr(client, "drain_papc_call_events", None)
        if callable(drain):
            drain()
        started = time.monotonic()
        try:
            response = client.complete_with_tools(
                instructions=instructions,
                input=input_items,
                tools=tools,
                previous_response_id=previous_response_id,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            self._provider_events.append(
                {
                    "role": role,
                    "status": "error",
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "latency_seconds": round(time.monotonic() - started, 6),
                    "call_phase": call_phase,
                    "error_type": type(exc).__name__,
                }
            )
            raise
        if callable(drain):
            drained = list(drain())
        else:
            drained = []
        if drained:
            for event in drained:
                public_event = dict(event)
                public_event["role"] = role
                public_event.setdefault("model", model)
                public_event.setdefault("reasoning_effort", reasoning_effort)
                public_event["call_phase"] = call_phase
                self._provider_events.append(public_event)
        else:
            self._provider_events.append(
                {
                    "role": role,
                    "status": "ok",
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "latency_seconds": round(time.monotonic() - started, 6),
                    "call_phase": call_phase,
                    "usage": _response_usage_dict(response),
                }
            )
        return response

    def _frontier_id(
        self,
        decision: StreamFrontierDecision,
        conversation: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str, tuple[str, ...], str]:
        return (
            decision.reason,
            decision.candidate_class,
            tuple(decision.class_relevant_families),
            _reached_state_signature(conversation),
        )

    def _call_provider(
        self,
        *,
        role: str,
        client: Any,
        model: str,
        reasoning_effort: str,
        system_prompt: str,
        conversation: Sequence[Mapping[str, Any]],
        tools: list[dict[str, Any]],
        call_phase: str,
        native_items: bool = False,
    ) -> dict[str, Any]:
        drain = getattr(client, "drain_papc_call_events", None)
        if callable(drain):
            drain()
        started = time.monotonic()
        try:
            response = client.complete_with_tools(
                instructions=system_prompt,
                input=(
                    native_item_live_conversation(conversation)
                    if native_items
                    else provider_safe_live_conversation(conversation)
                ),
                tools=tools,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            latency_seconds = round(time.monotonic() - started, 6)
            self._provider_events.append(
                {
                    "role": role,
                    "status": "error",
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "latency_seconds": latency_seconds,
                    "call_phase": call_phase,
                    "error_type": type(exc).__name__,
                }
            )
            raise
        if callable(drain):
            drained = list(drain())
        else:
            drained = []
        if drained:
            for event in drained:
                public_event = dict(event)
                public_event["role"] = role
                public_event.setdefault("model", model)
                public_event.setdefault("reasoning_effort", reasoning_effort)
                public_event["call_phase"] = call_phase
                self._provider_events.append(public_event)
        else:
            self._provider_events.append(
                {
                    "role": role,
                    "status": "ok",
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "latency_seconds": round(time.monotonic() - started, 6),
                    "call_phase": call_phase,
                    "usage": _response_usage_dict(response),
                }
            )
        return {
            "text": str(getattr(response, "output_text", "") or ""),
            "tool_calls": _response_tool_calls(response),
        }

    def _strong_system_prompt(self, system_prompt: str) -> str:
        if not self.strong_system_prompt_suffix.strip():
            return system_prompt
        return (
            system_prompt.rstrip() + "\n\n" + self.strong_system_prompt_suffix.strip()
        )

    def _bound_cheap_tool_call_batch(
        self,
        response: dict[str, Any],
        *,
        conversation: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """Trim a degenerate cheap tool-call burst without dropping a write.

        Keeps the first ``cheap_max_tool_calls_per_response`` calls in emission
        order, plus every later call that targets a domain write tool, so the
        commitment set the cheap agent expressed is preserved exactly and only
        the read burst is bounded. Responses at or below the bound are returned
        unchanged, so a cheap model that never bursts is byte-identical.
        """

        calls = response.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            return response
        self._cheap_batch_max_calls_observed = max(
            self._cheap_batch_max_calls_observed, len(calls)
        )
        bound = self.cheap_max_tool_calls_per_response
        no_result_bound = self.cheap_max_no_result_reads_per_row
        no_result_reads = (
            _no_result_read_count(conversation) if no_result_bound > 0 else 0
        )
        self._cheap_max_no_result_reads_observed = max(
            self._cheap_max_no_result_reads_observed, no_result_reads
        )
        row_exhausted = no_result_bound > 0 and no_result_reads >= no_result_bound
        if not row_exhausted and (bound <= 0 or len(calls) <= bound):
            return response
        try:
            from papc.certspec.state_bench.online_residual_selector import (
                adapter_for_domain,
            )

            write_names = frozenset(
                str(name) for name in adapter_for_domain(self.domain).write_tool_names
            )
        except Exception:
            # Without a declared write surface the bound cannot prove that a
            # dropped call is read-only, so the legacy batch is kept.
            return response
        if row_exhausted:
            kept = [
                call
                for call in calls
                if str((call or {}).get("name") or "") in write_names
            ]
            self._cheap_no_result_read_bound_hit_count += 1
            self._cheap_no_result_reads_dropped += len(calls) - len(kept)
            return {**response, "tool_calls": kept}
        head = list(calls[:bound])
        preserved_writes = [
            call
            for call in calls[bound:]
            if str((call or {}).get("name") or "") in write_names
        ]
        kept = head + preserved_writes
        self._cheap_batch_truncated_response_count += 1
        self._cheap_batch_dropped_call_count += len(calls) - len(kept)
        self._cheap_batch_writes_kept_beyond_bound += len(preserved_writes)
        return {**response, "tool_calls": kept}

    def _cheap_system_prompt(self, system_prompt: str) -> str:
        if not self.cheap_system_prompt_suffix.strip():
            return system_prompt
        return system_prompt.rstrip() + "\n\n" + self.cheap_system_prompt_suffix.strip()


def _build_mainline_agent(
    *,
    domain_name: str,
    cheap_client: Any,
    strong_client: Any,
    cheap: ProviderRoleConfig,
    strong: ProviderRoleConfig,
    row_index: int,
    min_cheap_assistant_turns: int,
    cheap_max_tool_calls_per_response: int = 32,
    max_residual_strong_calls: int = MAINLINE_MAX_RESIDUAL_STRONG_CALLS,
    successor_tool_descriptions: bool = True,
    residual_round_budget_declared: bool = False,
    closure_obligation_declared: bool = False,
    closing_contract_style: str = "consequence",
    refuse_unclosable_last_slot: bool = False,
    final_repair_role: str = "strong",
    final_repair_reasoning_effort: str | None = None,
    arm: str = "mainline",
    domain_prompt_suffix: str = "on",
    cheap_native_item_loop: bool = False,
    shadow_tool_handlers_factory: Callable[
        [Sequence[Mapping[str, Any]]], Mapping[str, Callable[[dict[str, Any]], Any]]
    ],
) -> TrueLiveCFRCAgent:
    """Construct the agent profile for one measured arm.

    ``mainline`` is the sole method profile and is unchanged. ``cheap_anchor``
    strips every method-side component: the residual strong budget is zero, so
    no strong call can be issued on any path, both method system-prompt
    suffixes are empty, and the two native-runtime cheap bounds are disabled.
    An unknown arm is rejected rather than silently treated as the method.
    """

    if arm not in METHOD_ARMS:
        raise ValueError(f"arm must be one of {METHOD_ARMS}: {arm!r}")
    cheap_anchor = arm == "cheap_anchor"
    if domain_prompt_suffix not in DOMAIN_PROMPT_SUFFIX_MODES:
        raise ValueError(
            f"domain_prompt_suffix must be one of {DOMAIN_PROMPT_SUFFIX_MODES}: "
            f"{domain_prompt_suffix!r}"
        )
    if domain_prompt_suffix == "off":
        suffix = ""
    else:
        # The anchor arm carries the suffix only when it is asked to. That is
        # the prompt-matched incumbent: without it the method arm would be the
        # only side that ever sees the domain policy the suffix restates.
        suffix = live_online_system_prompt_suffix(
            domain_name,
            discipline=_DOMAIN_DISCIPLINE_BY_SUFFIX_MODE[domain_prompt_suffix],
        )
    return TrueLiveCFRCAgent(
        domain=domain_name,
        cheap_client=cheap_client,
        strong_client=strong_client,
        cheap=cheap,
        strong=strong,
        row_index=row_index,
        min_cheap_assistant_turns=min_cheap_assistant_turns,
        cheap_max_tool_calls_per_response=(
            0 if cheap_anchor else cheap_max_tool_calls_per_response
        ),
        cheap_max_no_result_reads_per_row=(
            0 if cheap_anchor else MAINLINE_CHEAP_MAX_NO_RESULT_READS_PER_ROW
        ),
        max_residual_strong_calls=(
            0 if cheap_anchor else max(0, int(max_residual_strong_calls))
        ),
        successor_tool_descriptions=bool(successor_tool_descriptions),
        residual_round_budget_declared=(
            False if cheap_anchor else bool(residual_round_budget_declared)
        ),
        closure_obligation_declared=(
            False if cheap_anchor else bool(closure_obligation_declared)
        ),
        closing_contract_style=str(closing_contract_style),
        refuse_unclosable_last_slot=(
            False if cheap_anchor else bool(refuse_unclosable_last_slot)
        ),
        final_repair_role=(
            "strong" if cheap_anchor else str(final_repair_role or "strong")
        ),
        # The anchor issues no strong call at all, so the phase it would apply
        # to does not exist there.
        final_repair_reasoning_effort=(
            None if cheap_anchor else final_repair_reasoning_effort
        ),
        cheap_system_prompt_suffix=suffix,
        strong_system_prompt_suffix=suffix,
        # Both arms read this. It is not a method-side component: it is the
        # shape the benchmark's own loop already sends the cheap model, so an
        # anchor that could not carry it would differ from the method arm in the
        # transport as well as in the method.
        cheap_native_item_loop=bool(cheap_native_item_loop),
        strong_proposer_mode="shadow_tool_loop",
        precommit_guard_mode="off",
        final_message_repair_mode="closure",
        shadow_convergence_guard=True,
        shadow_tool_handlers_factory=shadow_tool_handlers_factory,
    )


def execute_true_live_task(
    *,
    state_bench_root: Path,
    domain_name: str,
    task: Any,
    output_path: Path,
    cheap_model: str,
    cheap_reasoning_effort: str,
    strong_model: str,
    strong_reasoning_effort: str,
    user_simulator_model: str,
    user_simulator_reasoning_effort: str,
    api_base: str,
    api_key_env: str,
    request_timeout_seconds: float,
    min_cheap_assistant_turns: int,
    row_index: int,
    cheap_max_tool_calls_per_response: int = 32,
    successor_tool_descriptions: bool = True,
    residual_round_budget_declared: bool = False,
    closure_obligation_declared: bool = False,
    closing_contract_style: str = "consequence",
    refuse_unclosable_last_slot: bool = False,
    final_repair_role: str = "strong",
    final_repair_reasoning_effort: str | None = None,
    max_residual_strong_calls: int = MAINLINE_MAX_RESIDUAL_STRONG_CALLS,
    arm: str = "mainline",
    domain_prompt_suffix: str = "on",
    cheap_native_item_loop: bool = False,
) -> dict[str, Any]:
    """Run one task with the profile selected by ``arm``."""

    root = state_bench_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    ensure_optional_azure_identity_importable()
    from state_bench.domain import get_domain_config
    from state_bench.env_loader import load_task_environment
    from state_bench.orchestrator import run_task

    api_key = os.environ.get(str(api_key_env))
    if not api_key:
        raise ValueError(f"API key env var is not set: {api_key_env}")
    domain = get_domain_config(domain_name)
    env_data, _ = load_task_environment(domain, task)
    cheap_client = OpenAICompatibleRuntimeClient(
        api_key=api_key,
        base_url=api_base,
        model=cheap_model,
        timeout=request_timeout_seconds,
    )
    strong_client = OpenAICompatibleRuntimeClient(
        api_key=api_key,
        base_url=api_base,
        model=strong_model,
        timeout=request_timeout_seconds,
    )
    user_simulator_client = OpenAICompatibleRuntimeClient(
        api_key=api_key,
        base_url=api_base,
        model=user_simulator_model,
        timeout=request_timeout_seconds,
        default_reasoning_effort=user_simulator_reasoning_effort,
    )
    agent = _build_mainline_agent(
        domain_name=domain_name,
        cheap_client=cheap_client,
        strong_client=strong_client,
        cheap=ProviderRoleConfig(cheap_model, cheap_reasoning_effort),
        strong=ProviderRoleConfig(strong_model, strong_reasoning_effort),
        row_index=row_index,
        min_cheap_assistant_turns=min_cheap_assistant_turns,
        cheap_max_tool_calls_per_response=cheap_max_tool_calls_per_response,
        max_residual_strong_calls=max_residual_strong_calls,
        successor_tool_descriptions=successor_tool_descriptions,
        residual_round_budget_declared=residual_round_budget_declared,
        closure_obligation_declared=closure_obligation_declared,
        closing_contract_style=closing_contract_style,
        refuse_unclosable_last_slot=refuse_unclosable_last_slot,
        final_repair_role=final_repair_role,
        final_repair_reasoning_effort=final_repair_reasoning_effort,
        arm=arm,
        domain_prompt_suffix=domain_prompt_suffix,
        cheap_native_item_loop=cheap_native_item_loop,
        shadow_tool_handlers_factory=(
            lambda conversation: build_shadow_tool_handlers(
                domain=domain,
                env_data=env_data,
                now=str(getattr(task, "now", "")),
                conversation=conversation,
            )
        ),
    )
    started = time.monotonic()
    trajectory_metadata = {
        "agent_name": "PAPCTrueLiveCFRCAgent",
        "papc_arm": arm,
        "papc_execution_shape": (
            "true_live_cheap_only"
            if arm == "cheap_anchor"
            else "true_live_cheap_then_cfrc"
        ),
        "papc_method_profile": (
            "cheap_anchor_single_model"
            if arm == "cheap_anchor"
            else "r4_grounded_schema_cfrc"
        ),
        "papc_method_components": (
            []
            if arm == "cheap_anchor"
            else [
                "reached_state_contract_construction",
                "executable_residual_completion",
                "checked_execution",
            ]
        ),
        "papc_cheap_model": cheap_model,
        "papc_cheap_reasoning_effort": cheap_reasoning_effort,
        "papc_residual_strong_model": strong_model,
        "papc_residual_strong_reasoning_effort": strong_reasoning_effort,
        "papc_strong_proposer_mode": "shadow_tool_loop",
        "papc_cheap_system_prompt_suffix_enabled": bool(
            agent.cheap_system_prompt_suffix.strip()
        ),
        "papc_cheap_native_item_loop": bool(agent.cheap_native_item_loop),
        "papc_precommit_guard_mode": "off",
        "papc_final_message_repair_mode": "closure",
        "papc_final_repair_policy": "strong_only",
        "papc_residual_round_budget_declared": bool(
            residual_round_budget_declared
        ),
        "papc_closure_obligation_declared": bool(closure_obligation_declared),
        "papc_closing_contract_style": str(closing_contract_style),
        "papc_strong_context_mode": "full_context",
        "papc_strong_response_reuse": False,
        "papc_shadow_convergence_guard": True,
        "papc_max_residual_strong_calls": agent.max_residual_strong_calls,
        "papc_cheap_max_tool_calls_per_response": (
            agent.cheap_max_tool_calls_per_response
        ),
        "papc_user_simulator_model": user_simulator_model,
        "papc_user_simulator_reasoning_effort": user_simulator_reasoning_effort,
    }
    try:
        trajectory = run_task(
            task,
            env_data,
            str(getattr(task, "user_id", "")),
            None,
            domain=domain,
            agent=agent,
            simulator_client=user_simulator_client,
            trajectory_metadata=trajectory_metadata,
        )
    except Exception as exc:
        elapsed = round(time.monotonic() - started, 6)
        summary = agent.public_summary()
        runtime_summary, user_simulator_events = _task_runtime_summary(
            agent_runtime_summary=agent.runtime_ledger(),
            user_simulator_events=[
                {
                    **dict(event),
                    "role": "user_simulator",
                    "call_phase": "user_simulator",
                }
                for event in user_simulator_client.drain_papc_call_events()
            ],
        )
        strong_calls = int(runtime_summary["per_role"]["strong"]["calls"])
        cheap_calls = int(runtime_summary["per_role"]["cheap"]["calls"])
        user_simulator_calls = int(
            runtime_summary["per_role"]["user_simulator"]["calls"]
        )
        return {
            "schema": "papc.state_bench_true_live_task_receipt.v1",
            "row_index": row_index,
            "domain": domain_name,
            "final_label": "provider_failure",
            "trajectory_written": False,
            "frontier_found": bool(summary["frontier_found"]),
            "cheap_provider_calls": cheap_calls,
            "strong_provider_calls": strong_calls,
            "user_simulator_provider_calls": user_simulator_calls,
            "provider_calls": cheap_calls + strong_calls + user_simulator_calls,
            "tool_call_count": 0,
            "tool_error_count": 0,
            "method_wall_clock_seconds": elapsed,
            "runtime_ledger": runtime_summary,
            "public_agent_summary": summary,
            "private": {
                "trajectory_path": str(output_path),
                "error_type": type(exc).__name__,
                "trace": traceback.format_exc(),
                "provider_events_public": [
                    *agent.provider_events_public(),
                    *_user_simulator_public_events(user_simulator_events),
                ],
            },
        }
    elapsed = round(time.monotonic() - started, 6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory.save(output_path)
    efficiency = getattr(trajectory, "efficiency", None)
    summary = agent.public_summary()
    user_simulator_events = [
        {**dict(event), "role": "user_simulator", "call_phase": "user_simulator"}
        for event in user_simulator_client.drain_papc_call_events()
    ]
    runtime_summary, user_simulator_events = _task_runtime_summary(
        agent_runtime_summary=agent.runtime_ledger(),
        user_simulator_events=user_simulator_events,
    )
    strong_calls = int(runtime_summary["per_role"]["strong"]["calls"])
    cheap_calls = int(runtime_summary["per_role"]["cheap"]["calls"])
    user_simulator_calls = int(runtime_summary["per_role"]["user_simulator"]["calls"])
    return {
        "schema": "papc.state_bench_true_live_task_receipt.v1",
        "row_index": row_index,
        "domain": domain_name,
        "final_label": "executed",
        "trajectory_written": True,
        "frontier_found": bool(summary["frontier_found"]),
        "cheap_provider_calls": cheap_calls,
        "strong_provider_calls": strong_calls,
        "user_simulator_provider_calls": user_simulator_calls,
        "provider_calls": cheap_calls + strong_calls + user_simulator_calls,
        "tool_call_count": int(getattr(efficiency, "tool_calls", 0) or 0),
        "tool_error_count": int(getattr(efficiency, "tool_errors", 0) or 0),
        "method_wall_clock_seconds": elapsed,
        "runtime_ledger": runtime_summary,
        "public_agent_summary": summary,
        "private": {
            "trajectory_path": str(output_path),
            "provider_events_public": [
                *agent.provider_events_public(),
                *_user_simulator_public_events(user_simulator_events),
            ],
        },
    }


def _task_runtime_summary(
    *,
    agent_runtime_summary: Mapping[str, Any],
    user_simulator_events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime_ledger = RuntimeLedger().merge_public_summary(agent_runtime_summary)
    normalized_user_events = [dict(event) for event in user_simulator_events]
    for event in normalized_user_events:
        if event.get("status") == "ok":
            runtime_ledger.add_usage(
                "user_simulator",
                event.get("usage") if isinstance(event.get("usage"), Mapping) else None,
                latency_seconds=(
                    float(event.get("latency_seconds"))
                    if event.get("latency_seconds") is not None
                    else None
                ),
                cost_usd=None,
                call_count=1,
            )
        else:
            runtime_ledger.add(
                RoleCall(
                    role="user_simulator",
                    call_count=1,
                    latency_seconds=(
                        float(event.get("latency_seconds"))
                        if event.get("latency_seconds") is not None
                        else None
                    ),
                    token_usage_available=False,
                )
            )
    runtime_summary = runtime_ledger.public_summary()
    runtime_summary["phase_ledger"] = merge_phase_ledgers(
        [
            agent_runtime_summary.get("phase_ledger")
            if isinstance(agent_runtime_summary, Mapping)
            else None,
            _phase_ledger_from_events(normalized_user_events),
        ]
    )
    return runtime_summary, normalized_user_events


def _user_simulator_public_events(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user_simulator",
            "status": event.get("status"),
            "model": event.get("model"),
            "reasoning_effort": event.get("reasoning_effort"),
            "latency_seconds": event.get("latency_seconds"),
            "call_phase": event.get("call_phase"),
            "usage": event.get("usage") if event.get("status") == "ok" else None,
            "error_type": event.get("error_type")
            if event.get("status") != "ok"
            else None,
        }
        for event in events
    ]


def build_shadow_tool_handlers(
    *,
    domain: Any,
    env_data: Any,
    now: str,
    conversation: Sequence[Mapping[str, Any]],
) -> Mapping[str, Callable[[dict[str, Any]], Any]]:
    """Rebuild reached state in a shadow STATE-Bench environment.

    The shadow environment is private to strong proposal generation. It is
    reconstructed from the task's initial environment plus tool calls already
    visible in the live transcript; candidate strong calls made against it never
    mutate the real harness environment.
    """

    shadow_env = domain.environment_class(env_data.deep_copy(), now=now)
    handlers = shadow_env.tool_handlers
    for call in _iter_visible_tool_calls(conversation):
        name = str(call.get("name") or "")
        args = (
            call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        )
        handler = handlers.get(name)
        if handler is None:
            raise ValueError(f"visible tool call has no handler: {name}")
        handler(dict(args))
    return handlers


def _iter_visible_tool_calls(
    conversation: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    calls: list[Mapping[str, Any]] = []
    for message in conversation:
        if not isinstance(message, Mapping):
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            calls.extend(call for call in tool_calls if isinstance(call, Mapping))
    return calls


def ensure_optional_azure_identity_importable() -> None:
    """Let official STATE-Bench imports work when Azure deps are absent."""

    try:
        if importlib.util.find_spec("azure.identity") is not None:
            return
    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    azure_module = sys.modules.get("azure")
    if azure_module is None:
        azure_module = types.ModuleType("azure")
        sys.modules["azure"] = azure_module
    identity_module = types.ModuleType("azure.identity")

    class DefaultAzureCredential:  # noqa: D401 - stub for OpenAI-only path
        """Unavailable Azure credential stub."""

    def get_bearer_token_provider(*_args: Any, **_kwargs: Any) -> Callable[[], str]:
        return lambda: ""

    identity_module.DefaultAzureCredential = DefaultAzureCredential
    identity_module.get_bearer_token_provider = get_bearer_token_provider
    setattr(azure_module, "identity", identity_module)
    sys.modules["azure.identity"] = identity_module


def provider_safe_live_conversation(
    conversation: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Convert a live transcript to provider-safe text messages.

    The harness appends every executed batch twice: once on the assistant
    message as ``tool_calls`` with the results embedded, and once as the
    following ``role=tool`` message. Both copies are then re-sent on every later
    request, so the same receipt is billed twice per turn. When the immediately
    following tool message carries an exactly matching batch -- same length and
    order, call ids where either side supplies one, names, arguments, and result
    values -- the assistant copy is rendered without its embedded ``result``,
    which the tool message still states in full.

    The assistant copy keeps its call names and arguments rather than being
    dropped outright. Dropping it was measured on the full official test split
    and cost quality: the assistant turn then no longer states which calls it
    made, and the same records reach the model only through a message whose role
    was coerced away from ``assistant``. The saving is not worth that. A failed
    proof check falls back to the legacy rendering byte for byte.
    """

    safe: list[dict[str, str]] = []
    raw_messages = list(conversation)
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, Mapping):
            continue
        raw_role = str(raw_message.get("role") or "user")
        role = raw_role if raw_role in {"system", "user", "assistant"} else "user"
        content = str(raw_message.get("content") or "")
        tool_calls = raw_message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            duplicated = _tool_batch_has_exact_following_result(
                tool_calls,
                raw_messages[index + 1] if index + 1 < len(raw_messages) else None,
            )
            rendered = _render_live_tool_records(
                tool_calls,
                include_results=not duplicated,
            )
            content = (
                f"{content}\n\n[Executed tool calls]\n{rendered}"
                if content
                else f"[Executed tool calls]\n{rendered}"
            )
        elif raw_role == "tool":
            content = "[Tool results]\n" + _render_live_tool_records(
                raw_message.get("content")
                if isinstance(raw_message.get("content"), list)
                else []
            )
        safe.append({"role": role, "content": content})
    return safe


def native_item_live_conversation(
    conversation: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Render the same live transcript as native Responses-API input items.

    ``provider_safe_live_conversation`` flattens every executed batch into a
    text block on the assistant message and coerces the ``role=tool`` carrier
    to ``user``. The benchmark's own loop chains native item arrays instead, so
    the flattened shape is this implementation's own departure from the harness
    it runs inside: a long user-role blob reads as fresh user input, while a
    ``function_call_output`` reads as a mechanical continuation of a call the
    model already made.

    The same records are emitted here as one ``function_call`` item per executed
    call followed by one ``function_call_output`` item per result. The harness
    appends every executed batch twice -- on the assistant message and again as
    the following ``role=tool`` message -- but in native form those two carriers
    are the two halves of one record, so the tool message is consumed as the
    outputs of the batch it repeats rather than rendered a second time.

    No ``reasoning`` item is ever carried. The accounting is cache-neutral, so a
    carried reasoning item would be re-billed as input on every later call in
    the row. For the same reason no ``previous_response_id`` is introduced: the
    input array stays the whole state that is sent.
    """

    items: list[dict[str, Any]] = []
    raw_messages = list(conversation)
    used_call_ids: set[str] = set()
    consumed_indices: set[int] = set()
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, Mapping):
            continue
        if index in consumed_indices:
            continue
        if str(raw_message.get("type") or "") == "reasoning":
            continue
        raw_role = str(raw_message.get("role") or "user")
        content = raw_message.get("content")
        tool_calls = raw_message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            speaker = (
                raw_role if raw_role in {"system", "user", "assistant"} else "assistant"
            )
            text = str(content or "")
            if text:
                items.append({"role": speaker, "content": text})
            following = (
                raw_messages[index + 1] if index + 1 < len(raw_messages) else None
            )
            paired = _native_paired_tool_records(tool_calls, following)
            if paired is not None:
                consumed_indices.add(index + 1)
            items.extend(
                _native_tool_call_items(
                    tool_calls,
                    paired if paired is not None else tool_calls,
                    batch_index=index,
                    used_call_ids=used_call_ids,
                )
            )
            continue
        if raw_role == "tool" and isinstance(content, list):
            # A tool message that was not consumed as the results of the batch
            # above it still has to arrive with a call to attach to, so it is
            # emitted as a self-contained call and output pair.
            items.extend(
                _native_tool_call_items(
                    content,
                    content,
                    batch_index=index,
                    used_call_ids=used_call_ids,
                )
            )
            continue
        role = raw_role if raw_role in {"system", "user", "assistant"} else "user"
        items.append({"role": role, "content": str(content or "")})
    return items


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def _native_paired_tool_records(
    assistant_records: Any,
    following_message: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]] | None:
    """Return the following tool message's records when they repeat this batch.

    Pairing is positional and requires the same record count, the same names and
    arguments in the same order, and equal call ids when either side supplies
    one. Results are not compared: the question here is only which carrier holds
    the outputs of this batch, not whether one of them may be dropped.
    """

    if not isinstance(assistant_records, list) or not assistant_records:
        return None
    if not isinstance(following_message, Mapping):
        return None
    if str(following_message.get("role") or "") != "tool":
        return None
    records = following_message.get("content")
    if not isinstance(records, list) or len(records) != len(assistant_records):
        return None
    for left, right in zip(assistant_records, records):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return None
        left_id = left.get("call_id")
        right_id = right.get("call_id")
        if left_id is not None or right_id is not None:
            if str(left_id or "") != str(right_id or ""):
                return None
        if str(left.get("name") or "") != str(right.get("name") or ""):
            return None
        left_args = (
            left.get("arguments") if isinstance(left.get("arguments"), Mapping) else {}
        )
        right_args = (
            right.get("arguments")
            if isinstance(right.get("arguments"), Mapping)
            else {}
        )
        if _compact_json(left_args) != _compact_json(right_args):
            return None
    return list(records)


def _native_tool_call_items(
    calls: Sequence[Any],
    results: Sequence[Any],
    *,
    batch_index: int,
    used_call_ids: set[str],
) -> list[dict[str, Any]]:
    """Emit one function_call item per call and one output item per result.

    The harness strips call ids before the transcript reaches the agent: an
    executed record is ``{name, arguments, result}`` only. A positional id is
    therefore synthesised when the transcript supplies none, which is what keeps
    each output attached to its own call.
    """

    call_items: list[dict[str, Any]] = []
    output_items: list[dict[str, Any]] = []
    for position, call in enumerate(calls):
        if not isinstance(call, Mapping):
            continue
        candidate = results[position] if position < len(results) else None
        result_record = candidate if isinstance(candidate, Mapping) else call
        call_id = _native_call_id(
            call,
            result_record,
            batch_index=batch_index,
            position=position,
            used_call_ids=used_call_ids,
        )
        arguments = (
            call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        )
        call_items.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": str(call.get("name") or ""),
                "arguments": _compact_json(arguments),
            }
        )
        result = (
            result_record.get("result")
            if "result" in result_record
            else call.get("result")
        )
        output_items.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": _compact_json(result),
            }
        )
    return [*call_items, *output_items]


def _native_call_id(
    call: Mapping[str, Any],
    result_record: Mapping[str, Any],
    *,
    batch_index: int,
    position: int,
    used_call_ids: set[str],
) -> str:
    candidate = str(call.get("call_id") or "") or str(
        result_record.get("call_id") or ""
    )
    if not candidate or candidate in used_call_ids:
        candidate = f"call_{batch_index}_{position}"
    while candidate in used_call_ids:
        candidate = f"{candidate}_x"
    used_call_ids.add(candidate)
    return candidate


def _reached_state_signature(conversation: Sequence[Mapping[str, Any]]) -> str:
    records: list[dict[str, Any]] = []
    for raw_message in conversation:
        if not isinstance(raw_message, Mapping):
            continue
        record: dict[str, Any] = {
            "role": str(raw_message.get("role") or ""),
            "content": str(raw_message.get("content") or ""),
        }
        tool_calls = raw_message.get("tool_calls")
        if isinstance(tool_calls, list):
            record["tool_calls"] = [
                _normalise_tool_record(item)
                for item in tool_calls
                if isinstance(item, Mapping)
            ]
        if record["role"] == "tool" and isinstance(raw_message.get("content"), list):
            record["tool_results"] = [
                _normalise_tool_record(item)
                for item in raw_message["content"]
                if isinstance(item, Mapping)
            ]
        records.append(record)
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _normalise_tool_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(record.get("name") or ""),
        "arguments": (
            dict(record.get("arguments"))
            if isinstance(record.get("arguments"), Mapping)
            else {}
        ),
        "result": record.get("result"),
    }


def _tool_batch_has_exact_following_result(
    assistant_records: Any,
    following_message: Mapping[str, Any] | None,
) -> bool:
    """Prove that an assistant batch's embedded results are duplicated.

    The check is intentionally strict. Only an adjacent tool message with the
    same record count and order is canonicalised. Call ids must agree when
    either side supplies one; names, arguments, and result values must match
    exactly under deterministic JSON encoding. Missing or malformed fields
    therefore fail closed to the legacy representation.
    """

    if not isinstance(assistant_records, list):
        return False
    if not isinstance(following_message, Mapping):
        return False
    if str(following_message.get("role") or "") != "tool":
        return False
    tool_records = following_message.get("content")
    if not isinstance(tool_records, list) or len(tool_records) != len(
        assistant_records
    ):
        return False

    def _encoded(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )

    for left, right in zip(assistant_records, tool_records):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        left_id = left.get("call_id")
        right_id = right.get("call_id")
        # When either side carries an id, equality is required so that a
        # positional coincidence cannot collapse two distinct calls.
        if left_id is not None or right_id is not None:
            if str(left_id or "") != str(right_id or ""):
                return False
        if str(left.get("name") or "") != str(right.get("name") or ""):
            return False
        left_args = (
            left.get("arguments") if isinstance(left.get("arguments"), Mapping) else {}
        )
        right_args = (
            right.get("arguments")
            if isinstance(right.get("arguments"), Mapping)
            else {}
        )
        if _encoded(left_args) != _encoded(right_args):
            return False
        if "result" not in left or "result" not in right:
            return False
        if _encoded(left.get("result")) != _encoded(right.get("result")):
            return False
    return True


def _render_live_tool_records(records: Any, *, include_results: bool = True) -> str:
    """Render executed tool records for one carrier.

    Values are encoded with compact separators. That is a byte-level encoding
    choice with identical JSON semantics: the model reads the same names,
    arguments, and results, and only the whitespace the provider would
    otherwise bill on every later request is gone.
    """

    if not isinstance(records, list):
        return ""
    rendered: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        name = str(record.get("name") or "")
        args = (
            record.get("arguments")
            if isinstance(record.get("arguments"), Mapping)
            else {}
        )
        line = (
            "- "
            + name
            + "("
            + json.dumps(
                args, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            + ")"
        )
        if include_results:
            line += " -> " + json.dumps(
                record.get("result"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        rendered.append(line)
    return "\n".join(rendered)


def _current_tool_round_count(conversation: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for message in reversed(list(conversation)):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "")
        if role == "user":
            break
        if role == "tool":
            count += 1
    return count


def _latest_message_role(conversation: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(list(conversation)):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "").strip()
        if role:
            return role
    return ""


def _normalise_item(item: Any) -> Any:
    if isinstance(item, Mapping):
        return dict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    return item


def _normalise_items(items: list[Any]) -> list[Any]:
    return [_normalise_item(item) for item in items]


def _raise_if_content_filtered(response: Any) -> None:
    details = getattr(response, "incomplete_details", None)
    if getattr(details, "reason", None) == "content_filter":
        raise RuntimeError("provider response was blocked by content filter")


def _is_retryable_provider_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    try:
        if int(status_code) in RETRYABLE_STATUS_CODES:
            return True
    except (TypeError, ValueError):
        pass
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def _response_tool_calls(response: Any) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for item in list(getattr(response, "output", []) or []):
        item_type = getattr(item, "type", None)
        if item_type is None and isinstance(item, Mapping):
            item_type = item.get("type")
        if item_type != "function_call":
            continue
        name = getattr(item, "name", None)
        raw_args = getattr(item, "arguments", None)
        call_id = getattr(item, "call_id", None)
        if isinstance(item, Mapping):
            name = item.get("name", name)
            raw_args = item.get("arguments", raw_args)
            call_id = item.get("call_id", call_id)
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, Mapping):
            args = dict(raw_args)
        else:
            args = {}
        if isinstance(name, str) and name:
            tool_calls.append(
                {
                    "name": name,
                    "arguments": args,
                    "call_id": str(call_id or ""),
                }
            )
    return tool_calls


def _drop_dangling_call_items(
    items: list[Any],
    skipped_call_ids: set[str],
) -> list[Any]:
    """Drop function_call items whose output will not be sent back.

    A shadow round can skip a duplicate tool call without executing it. The
    provider rejects an input that contains a function_call item with no
    matching function_call_output, so the skipped call items, and the reasoning
    items that must precede their siblings, cannot be resent.
    """

    filtered: list[Any] = []
    for item in items:
        item_type = item.get("type") if isinstance(item, Mapping) else None
        if item_type == "reasoning":
            continue
        if item_type == "function_call":
            call_id = (
                str(item.get("call_id") or "") if isinstance(item, Mapping) else ""
            )
            if call_id in skipped_call_ids:
                continue
        filtered.append(item)
    return filtered


def _shadow_call_signature(call: Mapping[str, Any]) -> str:
    """Stable (name, arguments) signature for shadow tool-call deduplication.

    Used by the convergence guard to detect a non-progress round in which the
    strong proposer only re-issues tool calls it already executed in this loop.
    """

    name = str(call.get("name") or "")
    arguments = (
        call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
    )
    encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return f"{name}:{encoded}"


def _public_reason_token(reason: str) -> str:
    token = str(reason or "").strip()
    if not token:
        return ""
    token = token.split("/", 1)[0].split("\\", 1)[0]
    return "".join(
        char for char in token[:96] if char.isalnum() or char in {":", "_", "-", "."}
    )


def _compact_tool_schema(
    tool: Mapping[str, Any], *, keep_tool_description: bool = True
) -> dict[str, Any]:
    """Use the shared projection for the successor's callable surface."""

    return compact_tool_schema(tool, keep_tool_description=keep_tool_description)


def _frontier_allowed_tools(
    decision: Any,
    tools: Sequence[Mapping[str, Any]],
    *,
    domain: str = "",
    conversation: Sequence[Mapping[str, Any]] = (),
    keep_tool_description: bool = True,
) -> list[dict[str, Any]]:
    allowed_names = _adapter_shadow_tool_names_for_frontier(
        domain=domain,
        conversation=conversation,
        decision=decision,
        tools=tools,
    )
    if allowed_names is None:
        return [
            _compact_tool_schema(tool, keep_tool_description=keep_tool_description)
            for tool in tools
        ]
    return [
        _compact_tool_schema(tool, keep_tool_description=keep_tool_description)
        for tool in tools
        if _tool_schema_name(tool) in allowed_names
    ]


def _adapter_shadow_tool_names_for_frontier(
    *,
    domain: str,
    conversation: Sequence[Mapping[str, Any]],
    decision: Any,
    tools: Sequence[Mapping[str, Any]],
) -> frozenset[str] | None:
    try:
        from papc.certspec.state_bench.online_residual_selector import (
            adapter_for_domain,
        )

        adapter = adapter_for_domain(domain)
    except Exception:
        return None
    hook = getattr(adapter, "shadow_tool_names_for_frontier", None)
    if hook is None:
        return None
    result = hook(
        {
            "conversation": [
                dict(item) for item in conversation if isinstance(item, Mapping)
            ]
        },
        _decision_public_json(decision),
        tool_names=tuple(_tool_schema_name(tool) for tool in tools),
    )
    if result is None:
        return None
    return frozenset(str(name) for name in result if str(name))


def _decision_public_json(decision: Any) -> Mapping[str, Any]:
    to_public_json = getattr(decision, "to_public_json", None)
    if callable(to_public_json):
        public = to_public_json()
        if isinstance(public, Mapping):
            return public
    if isinstance(decision, Mapping):
        return decision
    return {
        "authority_level": getattr(decision, "authority_level", ""),
        "candidate_class": getattr(decision, "candidate_class", ""),
        "class_relevant_families": tuple(
            getattr(decision, "class_relevant_families", ()) or ()
        ),
    }


def _tool_schema_name(tool: Mapping[str, Any]) -> str:
    direct = str(tool.get("name") or "").strip()
    if direct:
        return direct
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "").strip()
    return ""


def _tool_schema_names(tools: Sequence[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        name = _tool_schema_name(tool)
        if name:
            names.add(name)
    return names


def _result_is_empty(result: Any) -> bool:
    """Decide whether an executed tool record returned nothing usable.

    Conservative on purpose: an absent result, an empty sequence, or a mapping
    whose every list-valued field is empty counts as no result. A mapping with
    no list-valued field counts only when the mapping itself is empty, so a
    scalar answer is never mistaken for a miss.
    """

    if result is None:
        return True
    if isinstance(result, (list, tuple, set)):
        return not result
    if isinstance(result, Mapping):
        lists = [value for value in result.values() if isinstance(value, list)]
        if lists:
            return all(not value for value in lists)
        return not result
    return False


def _no_result_read_count(conversation: Sequence[Mapping[str, Any]]) -> int:
    """Count executed tool records in this row that returned nothing."""

    count = 0
    for message in conversation:
        if not isinstance(message, Mapping):
            continue
        records = message.get("tool_calls")
        if not isinstance(records, list):
            continue
        for record in records:
            if isinstance(record, Mapping) and _result_is_empty(record.get("result")):
                count += 1
    return count


def _tool_calls_are_projection_of_results(
    tool_calls: Sequence[Mapping[str, Any]],
    tool_results: Sequence[Mapping[str, Any]],
) -> bool:
    """Prove the call list states nothing the result list does not already state.

    Same length and order, matching names, byte-identical arguments under
    deterministic encoding, and no field on a call beyond ``name`` and
    ``arguments``. Anything else fails closed and both lists are carried.
    """

    if len(tool_calls) != len(tool_results):
        return False

    def _encoded(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )

    for call, result in zip(tool_calls, tool_results):
        if not isinstance(call, Mapping) or not isinstance(result, Mapping):
            return False
        if set(call) - {"name", "arguments"}:
            return False
        if str(call.get("name") or "") != str(result.get("name") or ""):
            return False
        if _encoded(call.get("arguments")) != _encoded(result.get("arguments")):
            return False
    return True


@dataclass(frozen=True)
class DraftClosureReport:
    """What a draft closing message leaves open over the executed writes."""

    draft_present: bool
    checked_write_count: int
    unrealized_fields: tuple[str, ...]
    contradicted_fields: tuple[str, ...]
    open_obligation_marks: tuple[str, ...]
    ungrounded_values: tuple[str, ...] = ()

    @property
    def closes(self) -> bool:
        return (
            self.draft_present
            and self.checked_write_count > 0
            and not self.unrealized_fields
            and not self.contradicted_fields
            and not self.open_obligation_marks
            and not self.ungrounded_values
        )


# Where the draft stops speaking about the field it just named.
_DRAFT_CLAUSE_END = re.compile(r"[.;:!?\n]")


def _shadow_write_envelope(arguments: Any) -> dict[str, Any]:
    """Read one native tool call as the write envelope closure is defined over.

    ``residual_closure`` reads a write as an interface envelope carrying a method
    and a body, while a STATE-Bench tool carries its body flat and answers with
    an operation result rather than an echo of the entity it wrote. The method is
    therefore the non-replacing one: a field the result omits is not evidence
    that the write was refused, while a field the result reports differently
    still is.
    """

    return {
        "method": "POST",
        "json": dict(arguments) if isinstance(arguments, Mapping) else {},
    }


def _draft_statement_for_field(draft: str, field: str) -> str | None:
    """What the draft says about one written field, if it names that field."""

    label = str(field).rsplit(".", 1)[-1].replace("_", " ").strip()
    if not label:
        return None
    match = re.search(rf"\b{re.escape(label)}\b", draft)
    if match is None:
        return None
    tail = _DRAFT_CLAUSE_END.split(draft[match.end() : match.end() + 120], 1)[0]
    return tail.strip() or None


def retained_request_text(conversation: Sequence[Mapping[str, Any]]) -> str:
    """The request the reached state still owes, in the words the user used."""

    parts: list[str] = []
    for message in conversation:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n".join(parts)


def draft_closure_report(
    *,
    draft_final_message: str,
    shadow_tool_results: Sequence[Mapping[str, Any]],
    write_tool_names: frozenset[str],
    request: str = "",
) -> DraftClosureReport:
    """Decide whether a draft already discharges what the proposal executed.

    Pure and provider free. Closure is the shared CFRC one, read over the writes
    the shadow loop actually committed:

    * a field the write's own receipt reports differently was never realized,
      whatever the draft says about it (``unrealized_write_fields``);
    * a value the draft states differently for a field the write set is a
      contradiction, judged by the shared closure test rather than by string
      equality, so a normalized restatement still counts as reporting;
    * a quantified obligation in the retained request ranges over a target set
      the reached state does not enumerate, so no draft can be certified as
      closing it (``open_obligations``);
    * a value the draft states that neither the request nor any receipt carries
      was invented by the draft, and nothing in the reached state witnesses it
      (``ungrounded_values``). This is what keeps the exemption above safe: a
      draft may leave the order id the user typed unsaid, but it may not name a
      different one.

    What the report deliberately does not test is whether the draft restates
    every value the writes carried. Requiring that made the check certify
    nothing at all across 300 measured rows, because the values it demanded back
    were protocol arguments rather than anything a user reads: a confirmation
    flag, an internal product id, a boolean for an option that was not taken.
    Closure is that the message does not misreport the reached state, and
    misreporting is contradiction or invention. An incomplete but accurate
    message is a quality question for the scorer, not a licence to overwrite a
    message the successor authored with the same receipts in hand.

    Reads are outside the report: their discharge lives in what the answer says,
    which arguments alone cannot witness. A proposal that committed no write is
    therefore never certified as closed.
    """

    draft = str(draft_final_message or "").strip().lower()
    unrealized: list[str] = []
    contradicted: list[str] = []
    checked = 0
    for entry in shadow_tool_results:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "")
        if name not in write_tool_names:
            continue
        checked += 1
        envelope = _shadow_write_envelope(entry.get("arguments"))
        unrealized.extend(
            f"{name}.{field}"
            for field in unrealized_write_fields(
                arguments=envelope,
                receipt=entry.get("result"),
            )
        )
        for path, value in sorted(leaf_assignments(envelope["json"]).items()):
            if value is None or isinstance(value, (list, tuple)):
                continue
            wrote = str(value).strip().lower()
            if not wrote:
                continue
            # An argument written in snake case is stated in prose as words. That
            # is the only rewriting the draft is credited with; nothing else
            # about the written value is normalized away.
            if draft and any(
                form in draft for form in (wrote, wrote.replace("_", " "))
            ):
                continue
            statement = _draft_statement_for_field(draft, path) if draft else None
            if statement is not None and contradicts(wrote=wrote, reported=statement):
                contradicted.append(f"{name}.{path}")
    marks: list[str] = []
    if checked:
        record = open_obligations(
            request=request,
            receipts=[
                {
                    "action": str(entry.get("name") or ""),
                    "observation": entry.get("result"),
                }
                for entry in shadow_tool_results
                if isinstance(entry, Mapping)
            ],
        )
        marks.extend(str(item) for item in record.get("quantifiers") or ())
        marks.extend(
            str(item) for item in record.get("unopened_enumerated_containers") or ()
        )
    return DraftClosureReport(
        draft_present=bool(draft),
        checked_write_count=checked,
        unrealized_fields=tuple(sorted(set(unrealized))),
        contradicted_fields=tuple(sorted(set(contradicted))),
        open_obligation_marks=tuple(sorted(set(marks))),
        ungrounded_values=ungrounded_values(
            draft=draft_final_message,
            evidence=[
                str(request or ""),
                *(
                    json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str)
                    for entry in shadow_tool_results
                    if isinstance(entry, Mapping)
                ),
            ],
        ),
    )


def _final_repair_messages(
    payload: Mapping[str, Any],
    *,
    system_prompt: str,
    system_prompt_suffix: str,
) -> list[dict[str, str]]:
    repair_contract = (
        "Rewrite the final user-facing message for a STATE-Bench CFRC "
        "residual suffix. Use only the supplied visible system policy, "
        "conversation tail, and selected tool results. Do not add tool "
        "calls, task ids, hidden scorer facts, or private paths. "
        "Preserve the selected action, but make supported policy, "
        "amounts, timelines, tradeoffs, and confirmation status "
        "explicit. Return only the final assistant message."
    )
    return [
        {
            "role": "system",
            "content": "\n\n".join(
                part
                for part in (
                    str(system_prompt or "").strip(),
                    str(system_prompt_suffix or "").strip(),
                    repair_contract,
                )
                if part
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            ),
        },
    ]


def _phase_ledger_from_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    phase_rows: dict[str, dict[str, Any]] = {}
    for event in events:
        phase = str(event.get("call_phase") or "unclassified")
        bucket = phase_rows.setdefault(phase, {"calls": 0})
        bucket["calls"] = int(bucket.get("calls") or 0) + 1
        if event.get("status") == "ok":
            bucket["ok_calls"] = int(bucket.get("ok_calls") or 0) + 1
        else:
            bucket["error_calls"] = int(bucket.get("error_calls") or 0) + 1
        usage = event.get("usage") if isinstance(event.get("usage"), Mapping) else {}
        if usage:
            bucket["input_tokens"] = int(bucket.get("input_tokens") or 0) + int(
                usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            )
            bucket["cached_input_tokens"] = int(
                bucket.get("cached_input_tokens") or 0
            ) + _event_cached_input_tokens(usage)
            bucket["output_tokens"] = int(bucket.get("output_tokens") or 0) + int(
                usage.get("output_tokens") or usage.get("completion_tokens") or 0
            )
            bucket["reasoning_tokens"] = int(
                bucket.get("reasoning_tokens") or 0
            ) + _event_reasoning_tokens(usage)
            bucket["total_tokens"] = int(bucket.get("total_tokens") or 0) + int(
                usage.get("total_tokens") or 0
            )
        latency = event.get("latency_seconds")
        if isinstance(latency, int | float):
            bucket["latency_seconds"] = float(
                bucket.get("latency_seconds") or 0.0
            ) + max(0.0, float(latency))
    return merge_phase_ledgers([phase_rows])


def _event_cached_input_tokens(usage: Mapping[str, Any]) -> int:
    if usage.get("cached_input_tokens") is not None:
        return max(0, int(usage.get("cached_input_tokens") or 0))
    details = usage.get("input_tokens_details")
    if isinstance(details, Mapping):
        return max(0, int(details.get("cached_tokens") or 0))
    return 0


def _event_reasoning_tokens(usage: Mapping[str, Any]) -> int:
    if usage.get("reasoning_output_tokens") is not None:
        return max(0, int(usage.get("reasoning_output_tokens") or 0))
    if usage.get("reasoning_tokens") is not None:
        return max(0, int(usage.get("reasoning_tokens") or 0))
    for key in ("output_tokens_details", "completion_tokens_details"):
        details = usage.get(key)
        if isinstance(details, Mapping):
            return max(0, int(details.get("reasoning_tokens") or 0))
    return 0


def _response_usage_dict(response: Any) -> dict[str, int]:
    def value(obj: Any, name: str) -> Any:
        return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)

    usage = value(response, "usage")
    if usage is None:
        return {}
    input_details = value(usage, "input_tokens_details")
    output_details = value(usage, "output_tokens_details")
    fields = {
        "input_tokens": value(usage, "input_tokens"),
        "cached_input_tokens": value(input_details, "cached_tokens"),
        "output_tokens": value(usage, "output_tokens"),
        "reasoning_output_tokens": value(output_details, "reasoning_tokens"),
        "total_tokens": value(usage, "total_tokens"),
    }
    # Missing fields are unmeasured, not explicit zero-cost usage.
    return {name: int(count) for name, count in fields.items() if count is not None}


def _token_usage_from_ledger(summary: Mapping[str, Any]) -> Any:
    totals = summary.get("totals") if isinstance(summary, Mapping) else {}
    totals = totals if isinstance(totals, Mapping) else {}

    class _Usage:
        def __init__(self) -> None:
            self.input_tokens = int(totals.get("input_tokens") or 0)
            self.cached_input_tokens = int(totals.get("cached_input_tokens") or 0)
            self.output_tokens = int(totals.get("output_tokens") or 0)
            self.reasoning_output_tokens = int(totals.get("reasoning_tokens") or 0)
            self.total_tokens = int(totals.get("total_tokens") or 0)
            self.input_cost_usd = 0.0
            self.cached_input_cost_usd = 0.0
            self.output_cost_usd = 0.0
            self.agent_turn_cost_usd = 0.0
            self.memory_ingestion_cost_usd = 0.0
            self.memory_retrieval_cost_usd = 0.0
            self.other_llm_cost_usd = 0.0
            self.total_cost_usd = 0.0

        def to_dict(self) -> dict[str, Any]:
            return {
                "input_tokens": self.input_tokens,
                "cached_input_tokens": self.cached_input_tokens,
                "output_tokens": self.output_tokens,
                "reasoning_output_tokens": self.reasoning_output_tokens,
                "total_tokens": self.total_tokens,
                "input_cost_usd": self.input_cost_usd,
                "cached_input_cost_usd": self.cached_input_cost_usd,
                "output_cost_usd": self.output_cost_usd,
                "agent_turn_cost_usd": self.agent_turn_cost_usd,
                "memory_ingestion_cost_usd": self.memory_ingestion_cost_usd,
                "memory_retrieval_cost_usd": self.memory_retrieval_cost_usd,
                "other_llm_cost_usd": self.other_llm_cost_usd,
                "total_cost_usd": self.total_cost_usd,
            }

    return _Usage()


__all__ = [
    "DOMAIN_PROMPT_SUFFIX_MODES",
    "MAINLINE_CHEAP_MAX_NO_RESULT_READS_PER_ROW",
    "MAINLINE_MAX_RESIDUAL_STRONG_CALLS",
    "METHOD_ARMS",
    "OpenAICompatibleRuntimeClient",
    "ProviderRoleConfig",
    "SCHEMA",
    "TrueLiveCFRCAgent",
    "build_shadow_tool_handlers",
    "ensure_optional_azure_identity_importable",
    "execute_true_live_task",
    "native_item_live_conversation",
    "online_execution_contract",
    "provider_safe_live_conversation",
]
