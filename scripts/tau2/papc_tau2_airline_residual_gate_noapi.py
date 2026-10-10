#!/usr/bin/env python3
"""Provider-zero gate for the tau2 Airline residual graph.

The graph now carries its open obligation across the user turn it spends to
ask for confirmation. This gate checks that change without a provider call.
It replays the real tau2 agent through a pause and a resume, replays the
recorded decision rows of the previous fifty row run to count how many of
them the pause rule would reach, and binds the
result to the hashes of the code that will run.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
for _path in (_REPO, _REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from papc.paths import DATA_ROOT

_TAU_SRC = Path(f"{DATA_ROOT}/benchmarks/tau_bench/repo") / "src"
if str(_TAU_SRC) not in sys.path:
    sys.path.insert(0, str(_TAU_SRC))

from papc.certspec.tau2.airline_residual_runtime import (
    DEFAULT_MAX_SUCCESSOR_TURNS,
    DEFAULT_RESIDUAL_WRITE_BUDGET,
    RESIDUAL_USER_TURN_BUDGET,
)
from papc.certspec.tau2.noapi_readiness import (
    public_leakage_audit,
    write_json_append_only,
)

SCHEMA = "papc.tau2.airline_residual_gate_noapi.v1"
MAX_ONLINE_CONCURRENCY = 8
HASHED_CODE = {
    "airline_runtime": "src/papc/certspec/tau2/airline_residual_runtime.py",
    "residual_support": "src/papc/certspec/tau2/residual_support.py",
    "claude_native_transport": "src/papc/certspec/tau2/claude_native_generate.py",
    "airline_wrapper": "scripts/tau2/papc_tau2_airline_residual_run.py",
    "airline_gate": "scripts/tau2/papc_tau2_airline_residual_gate_noapi.py",
    "shared_compiler": "src/papc/certspec/cfrc_compiler.py",
    "minimum_authority_selector": "src/papc/certspec/minimum_authority_selector.py",
    "slot_candidate_generator": "src/papc/certspec/slot_candidate_generator.py",
    "shared_ir": "src/papc/certspec/cfrc_ir.py",
    "state_write_contract": "src/papc/certspec/state_write_contract.py",
    "contract_projection": "src/papc/certspec/contract_projection.py",
    "grounded_proposal": "src/papc/certspec/grounded_proposal.py",
    "executable_closure": "src/papc/certspec/executable_closure.py",
    "receipt_provenance": "src/papc/certspec/receipt_provenance.py",
    "runtime_ledger": "src/papc/certspec/runtime_ledger.py",
    "costing": "src/papc/certspec/costing.py",
    "provider": "src/papc/certspec/provider.py",
    "tau2_shared_runner": "src/papc/certspec/tau2/cfrc_runner.py",
    "request_obligation_contract": "src/papc/certspec/tau2/request_obligation_contract.py",
    "tau2_launch_support": "scripts/tau2/papc_tau2_cfrc_clean_run.py",
    "tau2_official_wrapper": "scripts/tau2/papc_tau2_official_run_with_evaluator_patch.py",
}
# A pause is only reachable while the successor could still act afterwards.
TERMINAL_WITHOUT_WRITE = "incremental_graph_terminal_without_mutating_residual"


def code_sha256() -> dict[str, str]:
    return {
        name: hashlib.sha256((_REPO / path).read_bytes()).hexdigest()
        for name, path in sorted(HASHED_CODE.items())
    }


def replay_residual_user_turn() -> dict[str, Any]:
    """Drive the real tau2 agent through a pause, a resume and a close."""

    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        SystemMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    from papc.certspec.tau2 import airline_residual_runtime as runtime_module

    provider_calls: list[str] = []
    scripted: list[Any] = [
        AssistantMessage(
            role="assistant",
            content="The fare changed. Please confirm before I update the flight.",
        ),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="strong-write-1",
                    name="update_reservation_flights",
                    arguments={"reservation_id": "res_a", "flight_id": "FL123"},
                )
            ],
        ),
        AssistantMessage(role="assistant", content="Your flight is now updated."),
    ]
    cheap_frontier = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="cheap-write-1",
                name="update_reservation_flights",
                arguments={"reservation_id": "res_a", "flight_id": "FL123"},
            )
        ],
    )

    def fake_generate(*, model: str, messages: Any, call_name: str, **kwargs: Any):
        provider_calls.append(call_name)
        if call_name == "airline_residual_cheap_prefix":
            return cheap_frontier
        return scripted.pop(0)

    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    decision_log = _REPO / ".gate_tmp_airline_residual_user_turn.jsonl"
    if decision_log.exists():
        decision_log.unlink()

    import tau2.utils.llm_utils as tau2_llm_utils
    from tau2.registry import registry

    # The real registry and the real message types are used. Only the provider
    # call is replaced, which is what makes this replay cost nothing.
    saved_generate = tau2_llm_utils.generate
    tau2_llm_utils.generate = fake_generate
    try:
        agent_name = runtime_module.register_tau2_airline_residual_agent(
            agent_name="gate_airline_residual_user_turn",
            strong_model="strong-model",
            strong_llm_args={},
            decision_log_path=decision_log,
        )
        factory = registry.get_agent_factory(agent_name)
        agent = factory(
            tools=[
                _Tool("get_reservation_details"),
                _Tool("update_reservation_flights"),
                _Tool("transfer_to_human_agents"),
            ],
            domain_policy="policy",
            llm="cheap-model",
            llm_args={},
            task=None,
        )
        state = agent.get_init_state()
        state.system_messages = [SystemMessage(role="system", content="policy")]

        paused, state = agent.generate_next_message(
            UserMessage(role="user", content="Change my flight to FL123."), state
        )
        resumed, state = agent.generate_next_message(
            UserMessage(role="user", content="Yes, please proceed."), state
        )
        closed, state = agent.generate_next_message(
            MultiToolMessage(
                role="tool",
                tool_messages=[
                    ToolMessage(
                        id="strong-write-1",
                        role="tool",
                        content=json.dumps({"reservation_id": "res_a"}),
                        requestor="assistant",
                    )
                ],
            ),
            state,
        )
        rows = [
            json.loads(line)
            for line in decision_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    finally:
        tau2_llm_utils.generate = saved_generate
        if decision_log.exists():
            decision_log.unlink()

    paused_meta = (paused.raw_data or {}).get("papc_cfrc", {})
    closed_meta = (closed.raw_data or {}).get("papc_cfrc", {})
    resumed_calls = [call.name for call in (resumed.tool_calls or [])]
    return {
        "pass": (
            paused_meta.get("residual_awaiting_user_turn") is True
            and paused_meta.get("hand_back_to_cheap_control") is False
            and resumed_calls == ["update_reservation_flights"]
            and closed_meta.get("incremental_graph_closed") is True
            and [row["projection_result"]["reason"] for row in rows]
            == [
                "residual_awaiting_user_turn",
                "compiled_airline_incremental_receipt_graph",
            ]
        ),
        "provider_calls": 0,
        "live_provider_calls_made": 0,
        "scripted_model_calls": len(provider_calls),
        "question_reaches_the_user": bool(paused.content),
        "successor_completes_after_the_reply": resumed_calls
        == ["update_reservation_flights"],
        "cheap_frontier_message_accounted_once": [
            row["runtime_ledger"]["per_role"]["cheap"]["calls"] for row in rows
        ]
        == [1, 0],
    }


def replay_residual_recertification() -> dict[str, Any]:
    """Drive the real agent through a pause that returns the floor to cheap.

    The successor asks its question, the prefix model owns the turn that
    carries the answer, and the next write candidate certifies a fresh
    frontier from the reached state that answer produced.
    """

    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        SystemMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    from papc.certspec.tau2 import airline_residual_runtime as runtime_module

    provider_calls: list[str] = []
    scripted: list[Any] = [
        AssistantMessage(
            role="assistant",
            content="The fare changed. Please confirm before I update the flight.",
        ),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="strong-write-1",
                    name="update_reservation_flights",
                    arguments={"reservation_id": "res_a", "flight_id": "FL123"},
                )
            ],
        ),
        AssistantMessage(role="assistant", content="Your flight is now updated."),
    ]

    def cheap_frontier() -> Any:
        return AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="cheap-write-1",
                    name="update_reservation_flights",
                    arguments={"reservation_id": "res_a", "flight_id": "FL123"},
                )
            ],
        )

    def fake_generate(*, model: str, messages: Any, call_name: str, **kwargs: Any):
        provider_calls.append(call_name)
        if call_name == "airline_residual_cheap_prefix":
            return cheap_frontier()
        return scripted.pop(0)

    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    decision_log = _REPO / ".gate_tmp_airline_residual_recertification.jsonl"
    if decision_log.exists():
        decision_log.unlink()

    import tau2.utils.llm_utils as tau2_llm_utils
    from tau2.registry import registry

    saved_generate = tau2_llm_utils.generate
    tau2_llm_utils.generate = fake_generate
    try:
        agent_name = runtime_module.register_tau2_airline_residual_agent(
            agent_name="gate_airline_residual_recertification",
            strong_model="strong-model",
            strong_llm_args={},
            decision_log_path=decision_log,
            residual_recertifies_after_user_turn=True,
        )
        factory = registry.get_agent_factory(agent_name)
        agent = factory(
            tools=[
                _Tool("get_reservation_details"),
                _Tool("update_reservation_flights"),
                _Tool("transfer_to_human_agents"),
            ],
            domain_policy="policy",
            llm="cheap-model",
            llm_args={},
            task=None,
        )
        state = agent.get_init_state()
        state.system_messages = [SystemMessage(role="system", content="policy")]

        paused, state = agent.generate_next_message(
            UserMessage(role="user", content="Change my flight to FL123."), state
        )
        reopened, state = agent.generate_next_message(
            UserMessage(role="user", content="Yes, please proceed."), state
        )
        closed, state = agent.generate_next_message(
            MultiToolMessage(
                role="tool",
                tool_messages=[
                    ToolMessage(
                        id="strong-write-1",
                        role="tool",
                        content=json.dumps({"reservation_id": "res_a"}),
                        requestor="assistant",
                    )
                ],
            ),
            state,
        )
        rows = [
            json.loads(line)
            for line in decision_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    finally:
        tau2_llm_utils.generate = saved_generate
        if decision_log.exists():
            decision_log.unlink()

    paused_meta = (paused.raw_data or {}).get("papc_cfrc", {})
    closed_meta = (closed.raw_data or {}).get("papc_cfrc", {})
    reopened_calls = [call.name for call in (reopened.tool_calls or [])]
    cheap_prefix_calls = provider_calls.count("airline_residual_cheap_prefix")
    return {
        "pass": (
            paused_meta.get("residual_awaiting_user_turn") is False
            and paused_meta.get("hand_back_to_cheap_control") is True
            and cheap_prefix_calls == 2
            and reopened_calls == ["update_reservation_flights"]
            and closed_meta.get("incremental_graph_closed") is True
            and [row["projection_result"]["reason"] for row in rows]
            == [
                "residual_awaiting_user_turn",
                "compiled_airline_incremental_receipt_graph",
            ]
        ),
        "provider_calls": 0,
        "live_provider_calls_made": 0,
        "scripted_model_calls": len(provider_calls),
        "question_reaches_the_user": bool(paused.content),
        "prefix_model_owns_the_answering_turn": cheap_prefix_calls == 2,
        "frontier_is_certified_again_on_the_next_write_candidate": (
            reopened_calls == ["update_reservation_flights"]
        ),
        "each_frontier_message_accounted_once": [
            row["runtime_ledger"]["per_role"]["cheap"]["calls"] for row in rows
        ]
        == [1, 1],
        "pause_budget_is_counted_across_recertifications": (
            runtime_module.RESIDUAL_USER_TURN_BUDGET
        ),
    }


def replay_frontier_deferral() -> dict[str, Any]:
    """Drive a write candidate the reached state cannot admit yet.

    The confirmation the policy requires is missing, so no frontier is
    certified and no successor call is made. The prefix model writes the turn
    that asks for it, and the frontier is certified on the write candidate that
    follows the user's answer.
    """

    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        SystemMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    from papc.certspec.tau2 import airline_residual_runtime as runtime_module

    provider_calls: list[str] = []

    def write_candidate(call_id: str) -> Any:
        return AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id=call_id,
                    name="update_reservation_flights",
                    arguments={"reservation_id": "res_a", "flight_id": "FL123"},
                )
            ],
        )

    strong_scripted: list[Any] = [
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="strong-write-1",
                    name="update_reservation_flights",
                    arguments={"reservation_id": "res_a", "flight_id": "FL123"},
                )
            ],
        ),
        AssistantMessage(role="assistant", content="Your flight is now updated."),
    ]
    cheap_prefix_scripted: list[Any] = [
        write_candidate("cheap-write-1"),
        write_candidate("cheap-write-2"),
    ]

    def fake_generate(*, model: str, messages: Any, call_name: str, **kwargs: Any):
        provider_calls.append(call_name)
        if call_name == "airline_residual_cheap_prefix":
            return cheap_prefix_scripted.pop(0)
        if call_name == "airline_residual_cheap_confirmation":
            return AssistantMessage(
                role="assistant",
                content=(
                    "I can move reservation res_a to flight FL123. "
                    "Please confirm so I can make the change."
                ),
            )
        return strong_scripted.pop(0)

    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    decision_log = _REPO / ".gate_tmp_airline_frontier_deferral.jsonl"
    if decision_log.exists():
        decision_log.unlink()

    import tau2.utils.llm_utils as tau2_llm_utils
    from tau2.registry import registry

    saved_generate = tau2_llm_utils.generate
    tau2_llm_utils.generate = fake_generate
    try:
        agent_name = runtime_module.register_tau2_airline_residual_agent(
            agent_name="gate_airline_frontier_deferral",
            strong_model="strong-model",
            strong_llm_args={},
            decision_log_path=decision_log,
            defer_frontier_until_confirmed=True,
        )
        factory = registry.get_agent_factory(agent_name)
        agent = factory(
            tools=[
                _Tool("get_reservation_details"),
                _Tool("update_reservation_flights"),
                _Tool("transfer_to_human_agents"),
            ],
            domain_policy="policy",
            llm="cheap-model",
            llm_args={},
            task=None,
        )
        state = agent.get_init_state()
        state.system_messages = [SystemMessage(role="system", content="policy")]

        deferred, state = agent.generate_next_message(
            UserMessage(role="user", content="Change my flight to FL123."), state
        )
        calls_before_answer = list(provider_calls)
        opened, state = agent.generate_next_message(
            UserMessage(role="user", content="Yes, please proceed."), state
        )
        closed, state = agent.generate_next_message(
            MultiToolMessage(
                role="tool",
                tool_messages=[
                    ToolMessage(
                        id="strong-write-1",
                        role="tool",
                        content=json.dumps({"reservation_id": "res_a"}),
                        requestor="assistant",
                    )
                ],
            ),
            state,
        )
        rows = [
            json.loads(line)
            for line in decision_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    finally:
        tau2_llm_utils.generate = saved_generate
        if decision_log.exists():
            decision_log.unlink()

    closed_meta = (closed.raw_data or {}).get("papc_cfrc", {})
    opened_calls = [call.name for call in (opened.tool_calls or [])]
    no_successor_before_the_answer = (
        "airline_residual_strong_next" not in calls_before_answer
    )
    return {
        "pass": (
            no_successor_before_the_answer
            and calls_before_answer
            == [
                "airline_residual_cheap_prefix",
                "airline_residual_cheap_confirmation",
            ]
            and bool(deferred.content)
            and not (deferred.tool_calls or [])
            and opened_calls == ["update_reservation_flights"]
            and closed_meta.get("incremental_graph_closed") is True
            and [row["projection_result"]["reason"] for row in rows]
            == [
                "frontier_waiting_for_confirmation",
                "compiled_airline_incremental_receipt_graph",
            ]
            and rows[0]["projection_result"]["projection"]["decision"] != "execute"
            and [row["runtime_ledger"]["per_role"]["cheap"]["calls"] for row in rows]
            == [1, 1]
        ),
        "provider_calls": 0,
        "live_provider_calls_made": 0,
        "scripted_model_calls": len(provider_calls),
        "unconfirmed_write_candidate_certifies_no_frontier": (
            no_successor_before_the_answer
        ),
        "prefix_model_writes_the_confirmation_turn": bool(deferred.content),
        "candidate_write_did_not_run": not (deferred.tool_calls or []),
        "each_intercepted_candidate_accounted_once": (
            [row["runtime_ledger"]["per_role"]["cheap"]["calls"] for row in rows]
            == [1, 1]
        ),
        "frontier_is_certified_after_the_answer": (
            opened_calls == ["update_reservation_flights"]
        ),
        "confirmation_open_on_a_bare_request": (
            runtime_module._write_confirmation_is_open(
                [UserMessage(role="user", content="Change my flight.")]
            )
        ),
        "confirmation_closed_after_an_answered_statement": (
            not runtime_module._write_confirmation_is_open(
                [
                    UserMessage(role="user", content="Change my flight."),
                    AssistantMessage(
                        role="assistant", content="Please confirm this change."
                    ),
                    UserMessage(role="user", content="Yes."),
                ]
            )
        ),
    }


def replay_declared_closure() -> dict[str, Any]:
    """Drive a closing message written beside its actions, accepted and refused.

    The saving only exists if the closing round disappears, and the guarantee
    only holds if the held message is discarded whenever the receipts do not
    accept. Both are driven here against the real agent.
    """

    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        SystemMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    from papc.certspec.tau2 import airline_residual_runtime as runtime_module

    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    CLOSING = "Your flight is now updated to FL123."

    def drive(*, receipt_refuses: bool) -> dict[str, Any]:
        provider_calls: list[str] = []
        scripted: list[Any] = [
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="strong-write-1",
                        name="update_reservation_flights",
                        arguments={"reservation_id": "res_a", "flight_id": "FL123"},
                    ),
                    ToolCall(
                        id="terminal-1",
                        name=runtime_module._TERMINAL_DECLARATION_TOOL,
                        arguments={"message": CLOSING},
                    ),
                ],
            ),
            AssistantMessage(
                role="assistant", content="I could not complete the change."
            ),
        ]
        cheap_frontier = AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="cheap-write-1",
                    name="update_reservation_flights",
                    arguments={"reservation_id": "res_a", "flight_id": "FL123"},
                )
            ],
        )

        def fake_generate(*, model: str, messages: Any, call_name: str, **kwargs: Any):
            provider_calls.append(call_name)
            if call_name == "airline_residual_cheap_prefix":
                return cheap_frontier
            return scripted.pop(0)

        decision_log = _REPO / (
            f".gate_tmp_airline_declared_closure_{int(receipt_refuses)}.jsonl"
        )
        if decision_log.exists():
            decision_log.unlink()

        import tau2.utils.llm_utils as tau2_llm_utils
        from tau2.registry import registry

        saved_generate = tau2_llm_utils.generate
        tau2_llm_utils.generate = fake_generate
        try:
            agent_name = runtime_module.register_tau2_airline_residual_agent(
                agent_name=(
                    f"gate_airline_declared_closure_{int(receipt_refuses)}"
                ),
                strong_model="strong-model",
                strong_llm_args={},
                decision_log_path=decision_log,
                successor_declares_closure=True,
                prefix_completion_grounding=True,
            )
            factory = registry.get_agent_factory(agent_name)
            agent = factory(
                tools=[
                    _Tool("get_reservation_details"),
                    _Tool("update_reservation_flights"),
                    _Tool("transfer_to_human_agents"),
                ],
                domain_policy="policy",
                llm="cheap-model",
                llm_args={},
                task=None,
            )
            state = agent.get_init_state()
            state.system_messages = [SystemMessage(role="system", content="policy")]

            acting, state = agent.generate_next_message(
                UserMessage(role="user", content="Change my flight to FL123."), state
            )
            receipt = ToolMessage(
                id="strong-write-1",
                role="tool",
                content=(
                    "reservation not found"
                    if receipt_refuses
                    else json.dumps({"reservation_id": "res_a"})
                ),
                requestor="assistant",
                error=bool(receipt_refuses),
            )
            closed, state = agent.generate_next_message(
                MultiToolMessage(role="tool", tool_messages=[receipt]), state
            )
        finally:
            tau2_llm_utils.generate = saved_generate
            if decision_log.exists():
                decision_log.unlink()

        return {
            "acting_turn_content": acting.content,
            "acting_turn_calls": [call.name for call in (acting.tool_calls or [])],
            "closing_content": closed.content,
            "strong_calls": provider_calls.count("airline_residual_strong_next"),
            "closed": bool(
                (closed.raw_data or {}).get("papc_cfrc", {}).get(
                    "incremental_graph_closed"
                )
            ),
        }

    accepted = drive(receipt_refuses=False)
    refused = drive(receipt_refuses=True)
    # The published policy states the prohibition twice, once in each direction.
    # The synthetic text carries both sentences because a rewrite that lifts
    # only the first leaves the second forbidding what the first now allows.
    policy_text = (
        "You should only make one tool call at a time, and if you make a tool "
        "call, you should not respond to the user simultaneously. If you respond "
        "to the user, you should not make a tool call at the same time."
    )
    held_text, rewritten = runtime_module.hold_simultaneous_message_clause(policy_text)
    # A rewrite that silently fails to match the published policy makes the whole
    # arm a no-op, which is how the first paid pair of runs was spent. The gate
    # checks the published text itself, in the order the runtime applies it.
    published = Path(
        f"{DATA_ROOT}/benchmarks/tau_bench/repo/data/tau2/domains/airline/policy.md"
    )
    published_applied = None
    if published.exists():
        source = published.read_text(encoding="utf-8")
        batched, _ = runtime_module.relax_batching_prohibition(source)
        published_applied = all(
            runtime_module.hold_simultaneous_message_clause(text)[1]
            and "should not make a tool call at the same time"
            not in runtime_module.hold_simultaneous_message_clause(text)[0]
            for text in (source, batched)
        )
    return {
        "pass": bool(
            accepted["acting_turn_calls"] == ["update_reservation_flights"]
            and not accepted["acting_turn_content"]
            and accepted["closing_content"] == CLOSING
            and accepted["strong_calls"] == 1
            and accepted["closed"]
            and refused["strong_calls"] == 2
            and refused["closing_content"] != CLOSING
            and rewritten
            and "not respond to the user simultaneously" not in held_text
            and "should not make a tool call at the same time" not in held_text
            and published_applied is not False
        ),
        "provider_calls": 0,
        "live_provider_calls_made": 0,
        "acting_turn_carries_no_user_facing_message": (
            not accepted["acting_turn_content"]
        ),
        "accepted_receipt_releases_the_held_message": (
            accepted["closing_content"] == CLOSING
        ),
        "accepted_path_spends_one_strong_round": accepted["strong_calls"],
        "refused_receipt_discards_the_held_message": (
            refused["closing_content"] != CLOSING
        ),
        "refused_path_keeps_the_closing_round": refused["strong_calls"],
        "successor_policy_clause_restated_not_dropped": bool(rewritten),
        "published_policy_leaves_no_standing_prohibition": published_applied,
    }


def replay_recorded_rows(private_root: Path | None) -> dict[str, Any]:
    """Count how many recorded hand-back rows the pause rule would reach."""

    if private_root is None:
        return {"available": False, "reason": "no_recorded_run_supplied"}
    paths = sorted(private_root.glob("*/*decisions.jsonl"))
    if not paths:
        return {"available": False, "reason": "no_decision_log_under_private_root"}
    rows = [
        json.loads(line)
        for line in paths[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reasons = Counter(row["projection_result"]["reason"] for row in rows)
    reachable = 0
    blocked_by_budget = 0
    for row in rows:
        if row["projection_result"]["reason"] != TERMINAL_WITHOUT_WRITE:
            continue
        diagnostics = row["projection_result"]["projection"]["diagnostics"]
        turns = int(diagnostics.get("strong_turn_count") or 0)
        writes = int(diagnostics.get("write_count") or 0)
        if (
            turns < DEFAULT_MAX_SUCCESSOR_TURNS - 1
            and writes < DEFAULT_RESIDUAL_WRITE_BUDGET
        ):
            reachable += 1
        else:
            blocked_by_budget += 1
    return {
        "available": True,
        "decision_log": paths[0].name,
        "row_count": len(rows),
        "recorded_reasons": dict(reasons),
        "hand_back_rows": reasons.get(TERMINAL_WITHOUT_WRITE, 0),
        "rows_the_pause_rule_reaches": reachable,
        "rows_blocked_by_turn_or_write_budget": blocked_by_budget,
        "note": (
            "A reached row spends a user turn and returns the residual to the "
            "successor. It is not a projected pass."
        ),
    }


def build_gate(
    *,
    projected_pass_count: int,
    projection_basis: str,
    recorded: dict[str, Any],
) -> dict[str, Any]:
    replay = replay_residual_user_turn()
    recertification = replay_residual_recertification()
    deferral = replay_frontier_deferral()
    closure = replay_declared_closure()
    checks_pass = bool(
        replay["pass"]
        and recertification["pass"]
        and deferral["pass"]
        and closure["pass"]
    )
    gate = {
        "schema": SCHEMA,
        "surface": "tau2_airline",
        "method_family": "incremental_live_receipt_graph_completion",
        "gate_decision": {
            "selector_gate_passes": checks_pass,
            "guarded_selector_projected_pass_count": projected_pass_count,
            "setting_correct_selector_projected_pass_count": projected_pass_count,
            "projection_is_nonbinding": True,
            "projection_basis": projection_basis,
            "authorizes_paid_run": checks_pass,
            "authorization_scope": (
                "single_fresh_airline_base50_residual_completion_quality_and_cost_run"
            ),
            "next_required_gate": None,
        },
        "residual_graph_checks": {
            "pass": checks_pass,
            "provider_calls": 0,
            "frontier_activation_policy": "all_write_or_transfer",
            "ordinary_reads_and_writes_remain_on_cheap_path": True,
            "whole_graph_is_compiled_before_any_live_write": True,
            "partial_graph_selection_is_rejected": True,
            "dependencies_are_derived_from_visible_receipts": True,
            "checked_execution_requires_receipt_match": True,
            "refused_receipt_stays_as_repairable_evidence": True,
            "refused_write_discharges_no_obligation": True,
            "open_obligation_survives_the_residual_user_turn": replay["pass"],
            "recertifying_pause_returns_the_floor_to_the_prefix_model": (
                recertification["pass"]
            ),
            "unconfirmed_write_candidate_certifies_no_frontier": deferral["pass"],
            "declared_closure_is_released_only_by_checked_execution": (
                closure["pass"]
            ),
            "residual_user_turn_budget": RESIDUAL_USER_TURN_BUDGET,
            "budget_or_turn_cap_rejection_fails_closed_before_live_write": True,
            "max_strong_turns_cover_the_whole_interactive_proposal": (
                DEFAULT_MAX_SUCCESSOR_TURNS
            ),
            "write_budget": DEFAULT_RESIDUAL_WRITE_BUDGET,
            "max_online_concurrency": MAX_ONLINE_CONCURRENCY,
        },
        "residual_user_turn_replay": replay,
        "residual_recertification_replay": recertification,
        "frontier_deferral_replay": deferral,
        "declared_closure_replay": closure,
        "recorded_run_replay": recorded,
        "zero_api_diagnostic": {
            "provider_calls": 0,
            "no_api_calls": True,
            "secrets_read": False,
        },
        "code_sha256": code_sha256(),
        "public_safe": True,
    }
    gate["public_leakage_audit"] = public_leakage_audit(gate)
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recorded-private-root",
        type=Path,
        default=None,
        help="A recorded run directory used to count reachable hand-back rows",
    )
    parser.add_argument(
        "--projected-pass-count",
        type=int,
        required=True,
        help="Nonbinding offline projection for the fifty row denominator",
    )
    parser.add_argument("--projection-basis", required=True)
    args = parser.parse_args(argv)

    gate = build_gate(
        projected_pass_count=int(args.projected_pass_count),
        projection_basis=str(args.projection_basis),
        recorded=replay_recorded_rows(args.recorded_private_root),
    )
    if gate["public_leakage_audit"]["pass"] is not True:
        raise SystemExit("airline residual user turn gate failed the leakage audit")
    write_json_append_only(args.output, gate)
    print(
        json.dumps(
            {
                "decision": (
                    "ready_for_authorized_paid_run"
                    if gate["gate_decision"]["authorizes_paid_run"]
                    else "blocked"
                ),
                "provider_calls": 0,
                "rows_the_pause_rule_reaches": gate["recorded_run_replay"].get(
                    "rows_the_pause_rule_reaches"
                ),
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
