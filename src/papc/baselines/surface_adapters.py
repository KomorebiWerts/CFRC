"""Protected adapters for public Agent-Diff, STATE-Bench, and tau2 surfaces."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from papc.baselines.fixed_handoff import (
    TRANSCRIPT_ACCEPTED_TERMINAL,
    TRANSCRIPT_SUCCESSOR_READY,
    FixedHandoffPacket,
    PrivateFixedHandoffReplayBundle,
    canonical_environment_sha256,
    canonical_replay_receipt_sha256,
)
from papc.baselines.router_contract import FullEpisodeBinding

AGENT_DIFF_SURFACE = "agent_diff"
STATE_BENCH_SURFACE = "state_bench"
TAU2_RETAIL_SURFACE = "tau2_retail"
TAU2_AIRLINE_SURFACE = "tau2_airline"

AGENT_DIFF_SERVICES = frozenset({"box", "calendar", "linear", "slack"})
STATE_BENCH_DOMAINS = frozenset(
    {"customer_support", "shopping_assistant", "travel"}
)
TAU2_DOMAINS = frozenset({"retail", "airline"})

AGENT_DIFF_FULL_DENOMINATOR = 45
STATE_BENCH_FULL_DENOMINATOR = 150
TAU2_RETAIL_FULL_DENOMINATOR = 40
TAU2_AIRLINE_FULL_DENOMINATOR = 20
AGENT_DIFF_STRONG_CALL_CAP = 12
STATE_BENCH_STRONG_CALL_CAP = 4
AGENT_DIFF_STRONG_OUTPUT_TOKEN_CAP = 2400
STATE_BENCH_STRONG_OUTPUT_TOKEN_CAP = 4096
AGENT_DIFF_OBSERVATION_CHAR_CAP = 12000

AGENT_DIFF_NATIVE_TERMINAL = "react_done_tag"
STATE_BENCH_NATIVE_TERMINAL = "user_simulator_task_done_marker"
AGENT_DIFF_OFFICIAL_SCORER = "agent_diff_evaluate_run"
STATE_BENCH_OFFICIAL_SCORER = "state_bench_official_score_run"
TAU2_RETAIL_OFFICIAL_SCORER = "tau2_retail_official_reward"
TAU2_AIRLINE_OFFICIAL_SCORER = "tau2_airline_official_reward"
AGENT_DIFF_ENVIRONMENT_SCHEMA = "papc.agent_diff.environment_snapshot.v1"
STATE_BENCH_ENVIRONMENT_SCHEMA = "papc.state_bench.environment_snapshot.v1"
AGENT_DIFF_REPLAY_PROTOCOL = "papc.agent_diff.native_prefix_replay.v1"
STATE_BENCH_REPLAY_PROTOCOL = "papc.state_bench.native_prefix_replay.v1"
_AGENT_DIFF_SERVICE_CONFIG = {
    "slack": ("Slack", "https://slack.com/api", ""),
    "box": ("Box", "https://api.box.com/2.0", ""),
    "calendar": (
        "Google Calendar",
        "https://www.googleapis.com/calendar/v3",
        (
            "- **Current Date/Time**: Sunday, June 17, 2018 at 00:01 "
            "(midnight), timezone America/Los_Angeles.\n"
        ),
    ),
    "linear": ("Linear", "https://api.linear.app/graphql", ""),
}
_AGENT_DIFF_SYSTEM_PROMPT = """You are an AI assistant that completes tasks by interacting with APIs via bash commands.

## Current Session
- **Service**: {service_name}
- **Base URL**: {base_url}
{extra_context}
## Environment
- Authentication is handled automatically via proxy. Use placeholder tokens where credentials would go.
- You execute bash commands, primarily curl, to interact with the API.
- If you are not sure how to use the API, explore the endpoint, parameters, and learn how it works.

## Response Format
Respond using XML tags:

<thinking>Your reasoning</thinking>
<action>Your bash command</action>

When done:
<thinking>Your reasoning</thinking>
<done>Brief summary</done>

## Rules
1. Execute ONE command at a time, then wait for the result.
2. Parse API responses carefully. Extract IDs and data needed for subsequent calls.
3. If a command fails, analyze the error and try a different approach.
4. Only use <done> when the task is fully completed.
"""
_AGENT_DIFF_DONE_RE = re.compile(
    r"<done>(.*?)</done>",
    re.DOTALL | re.IGNORECASE,
)
_AGENT_DIFF_ACTION_RE = re.compile(
    r"<action>(.*?)</action>",
    re.DOTALL | re.IGNORECASE,
)


def agent_diff_public_tool_schemas() -> tuple[dict[str, Any], ...]:
    """Describe the public BashExecutorProxy action surface."""

    return (
        {
            "type": "native_executor",
            "name": "agent_diff_bash",
            "description": (
                "Execute one public Agent-Diff API command through the native "
                "BashExecutorProxy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "minLength": 1},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            "response_protocol": "react_xml_action",
        },
    )


def build_agent_diff_native_system_prompt(service: str) -> str:
    service_name = _allowed_variant(
        service,
        allowed=AGENT_DIFF_SERVICES,
        label="agent_diff_service",
    )
    display_name, base_url, extra_context = _AGENT_DIFF_SERVICE_CONFIG[
        service_name
    ]
    return _AGENT_DIFF_SYSTEM_PROMPT.format(
        service_name=display_name,
        base_url=base_url,
        extra_context=extra_context,
    )


def agent_diff_native_termination(message: str) -> bool:
    return agent_diff_native_done(message) is not None


def agent_diff_native_done(message: str) -> str | None:
    if not isinstance(message, str):
        raise TypeError("agent_diff_native_message_must_be_str")
    match = _AGENT_DIFF_DONE_RE.search(message)
    if not match:
        return None
    done = match.group(1).strip()
    return done or None


def agent_diff_native_action(message: str) -> str | None:
    if not isinstance(message, str):
        raise TypeError("agent_diff_native_message_must_be_str")
    match = _AGENT_DIFF_ACTION_RE.search(message)
    if not match:
        return None
    action = match.group(1).strip()
    return action or None


def state_bench_native_termination(user_message: str) -> bool:
    if not isinstance(user_message, str):
        raise TypeError("state_bench_native_user_message_must_be_str")
    return "[TASK_DONE]" in user_message


def build_agent_diff_fixed_handoff(
    *,
    service: str,
    original_request: str,
    accepted_transcript: Sequence[Mapping[str, Any]],
    reached_environment_sha256: str,
    cheap_model: str,
    cheap_reasoning_effort: str,
    strong_model: str,
    strong_reasoning_effort: str,
    strong_call_cap: int = AGENT_DIFF_STRONG_CALL_CAP,
    strong_output_token_cap: int = AGENT_DIFF_STRONG_OUTPUT_TOKEN_CAP,
    native_observation_char_cap: int = AGENT_DIFF_OBSERVATION_CHAR_CAP,
    accepted_transcript_status: str = TRANSCRIPT_SUCCESSOR_READY,
) -> FixedHandoffPacket:
    service_name = _allowed_variant(
        service,
        allowed=AGENT_DIFF_SERVICES,
        label="agent_diff_service",
    )
    return FixedHandoffPacket(
        surface=AGENT_DIFF_SURFACE,
        surface_variant=service_name,
        native_system_prompt=build_agent_diff_native_system_prompt(
            service_name
        ),
        original_request=original_request,
        accepted_transcript=tuple(dict(item) for item in accepted_transcript),
        reached_environment_schema=AGENT_DIFF_ENVIRONMENT_SCHEMA,
        reached_environment_sha256=reached_environment_sha256,
        native_tool_schemas=agent_diff_public_tool_schemas(),
        native_terminal_signal=AGENT_DIFF_NATIVE_TERMINAL,
        official_scorer=AGENT_DIFF_OFFICIAL_SCORER,
        full_denominator=AGENT_DIFF_FULL_DENOMINATOR,
        cheap_model=cheap_model,
        cheap_reasoning_effort=cheap_reasoning_effort,
        strong_model=strong_model,
        strong_reasoning_effort=strong_reasoning_effort,
        strong_call_cap=strong_call_cap,
        strong_output_token_cap=strong_output_token_cap,
        native_observation_char_cap=native_observation_char_cap,
        accepted_transcript_status=accepted_transcript_status,
    )


def build_agent_diff_full_episode_binding(
    *,
    service: str,
    original_request: str,
    initial_environment_sha256: str,
) -> FullEpisodeBinding:
    """Bind a pristine Agent-Diff episode without reached-state metadata."""

    service_name = _allowed_variant(
        service,
        allowed=AGENT_DIFF_SERVICES,
        label="agent_diff_service",
    )
    return FullEpisodeBinding(
        surface=AGENT_DIFF_SURFACE,
        surface_variant=service_name,
        original_request=original_request,
        initial_environment_sha256=initial_environment_sha256,
        official_scorer=AGENT_DIFF_OFFICIAL_SCORER,
        official_denominator=AGENT_DIFF_FULL_DENOMINATOR,
    )


def build_state_bench_fixed_handoff(
    *,
    domain: str,
    native_system_prompt: str,
    original_request: str,
    accepted_transcript: Sequence[Mapping[str, Any]],
    reached_environment_sha256: str,
    native_tool_schemas: Sequence[Mapping[str, Any]],
    cheap_model: str,
    cheap_reasoning_effort: str,
    strong_model: str,
    strong_reasoning_effort: str,
    strong_call_cap: int = STATE_BENCH_STRONG_CALL_CAP,
    strong_output_token_cap: int = STATE_BENCH_STRONG_OUTPUT_TOKEN_CAP,
) -> FixedHandoffPacket:
    domain_name = _allowed_variant(
        domain,
        allowed=STATE_BENCH_DOMAINS,
        label="state_bench_domain",
    )
    return FixedHandoffPacket(
        surface=STATE_BENCH_SURFACE,
        surface_variant=domain_name,
        native_system_prompt=native_system_prompt,
        original_request=original_request,
        accepted_transcript=tuple(dict(item) for item in accepted_transcript),
        reached_environment_schema=STATE_BENCH_ENVIRONMENT_SCHEMA,
        reached_environment_sha256=reached_environment_sha256,
        native_tool_schemas=tuple(dict(item) for item in native_tool_schemas),
        native_terminal_signal=STATE_BENCH_NATIVE_TERMINAL,
        official_scorer=STATE_BENCH_OFFICIAL_SCORER,
        full_denominator=STATE_BENCH_FULL_DENOMINATOR,
        cheap_model=cheap_model,
        cheap_reasoning_effort=cheap_reasoning_effort,
        strong_model=strong_model,
        strong_reasoning_effort=strong_reasoning_effort,
        strong_call_cap=strong_call_cap,
        strong_output_token_cap=strong_output_token_cap,
        native_observation_char_cap=None,
    )


def build_state_bench_full_episode_binding(
    *,
    domain: str,
    original_request: str,
    initial_environment_sha256: str,
) -> FullEpisodeBinding:
    """Bind a pristine STATE-Bench episode without reached-state metadata."""

    domain_name = _allowed_variant(
        domain,
        allowed=STATE_BENCH_DOMAINS,
        label="state_bench_domain",
    )
    return FullEpisodeBinding(
        surface=STATE_BENCH_SURFACE,
        surface_variant=domain_name,
        original_request=original_request,
        initial_environment_sha256=initial_environment_sha256,
        official_scorer=STATE_BENCH_OFFICIAL_SCORER,
        official_denominator=STATE_BENCH_FULL_DENOMINATOR,
    )


def build_tau2_full_episode_binding(
    *,
    domain: str,
    original_request: str,
    initial_environment_sha256: str,
) -> FullEpisodeBinding:
    """Bind a pristine official tau2 test episode without reached-state data."""

    domain_name = _allowed_variant(
        domain,
        allowed=TAU2_DOMAINS,
        label="tau2_domain",
    )
    if domain_name == "retail":
        surface = TAU2_RETAIL_SURFACE
        scorer = TAU2_RETAIL_OFFICIAL_SCORER
        denominator = TAU2_RETAIL_FULL_DENOMINATOR
    else:
        surface = TAU2_AIRLINE_SURFACE
        scorer = TAU2_AIRLINE_OFFICIAL_SCORER
        denominator = TAU2_AIRLINE_FULL_DENOMINATOR
    return FullEpisodeBinding(
        surface=surface,
        surface_variant=domain_name,
        original_request=original_request,
        initial_environment_sha256=initial_environment_sha256,
        official_scorer=scorer,
        official_denominator=denominator,
    )


def emit_agent_diff_live_prefix_bundle(
    *,
    service: str,
    original_request: str,
    accepted_transcript: Sequence[Mapping[str, Any]],
    executed_native_actions: Sequence[Mapping[str, Any]],
    reached_environment_snapshot: Any,
    cheap_model: str,
    cheap_reasoning_effort: str,
    strong_model: str,
    strong_reasoning_effort: str,
    strong_call_cap: int = AGENT_DIFF_STRONG_CALL_CAP,
    strong_output_token_cap: int = AGENT_DIFF_STRONG_OUTPUT_TOKEN_CAP,
    native_observation_char_cap: int = AGENT_DIFF_OBSERVATION_CHAR_CAP,
    accepted_transcript_status: str = TRANSCRIPT_SUCCESSOR_READY,
) -> PrivateFixedHandoffReplayBundle:
    """Bind a live Agent-Diff prefix without selecting its handoff cut."""

    service_name = _allowed_variant(
        service,
        allowed=AGENT_DIFF_SERVICES,
        label="agent_diff_service",
    )
    if accepted_transcript_status == TRANSCRIPT_ACCEPTED_TERMINAL:
        if not accepted_transcript:
            raise ValueError("agent_diff_terminal_transcript_empty")
        last = accepted_transcript[-1]
        if (
            not isinstance(last, Mapping)
            or last.get("role") != "assistant"
            or not isinstance(last.get("content"), str)
            or not agent_diff_native_termination(str(last["content"]))
        ):
            raise ValueError("agent_diff_terminal_transcript_signal_invalid")
    elif accepted_transcript_status != TRANSCRIPT_SUCCESSOR_READY:
        raise ValueError("agent_diff_transcript_status_invalid")
    steps, replay_bindings = _agent_diff_replay_steps(
        executed_native_actions
    )
    transcript_bindings = _agent_diff_transcript_bindings(
        accepted_transcript
    )
    if (
        tuple(command for command, _ in transcript_bindings)
        != tuple(command for command, _ in replay_bindings)
    ):
        raise ValueError("agent_diff_live_prefix_action_receipt_mismatch")
    if transcript_bindings != replay_bindings:
        raise ValueError(
            "agent_diff_live_prefix_observation_receipt_mismatch"
        )
    packet = build_agent_diff_fixed_handoff(
        service=service_name,
        original_request=original_request,
        accepted_transcript=accepted_transcript,
        reached_environment_sha256=canonical_environment_sha256(
            reached_environment_snapshot,
            surface=AGENT_DIFF_SURFACE,
            surface_variant=service_name,
            snapshot_schema=AGENT_DIFF_ENVIRONMENT_SCHEMA,
        ),
        cheap_model=cheap_model,
        cheap_reasoning_effort=cheap_reasoning_effort,
        strong_model=strong_model,
        strong_reasoning_effort=strong_reasoning_effort,
        strong_call_cap=strong_call_cap,
        strong_output_token_cap=strong_output_token_cap,
        native_observation_char_cap=native_observation_char_cap,
        accepted_transcript_status=accepted_transcript_status,
    )
    return PrivateFixedHandoffReplayBundle(
        packet=packet,
        replay_protocol=AGENT_DIFF_REPLAY_PROTOCOL,
        native_replay_steps=steps,
    )


def emit_state_bench_live_prefix_bundle(
    *,
    domain: str,
    native_system_prompt: str,
    original_request: str,
    live_native_items: Sequence[Mapping[str, Any]],
    reached_environment_snapshot: Any,
    native_tool_schemas: Sequence[Mapping[str, Any]],
    cheap_model: str,
    cheap_reasoning_effort: str,
    strong_model: str,
    strong_reasoning_effort: str,
    strong_call_cap: int = STATE_BENCH_STRONG_CALL_CAP,
    strong_output_token_cap: int = STATE_BENCH_STRONG_OUTPUT_TOKEN_CAP,
) -> PrivateFixedHandoffReplayBundle:
    """Bind a live successor-visible projection; scored rows are insufficient."""

    domain_name = _allowed_variant(
        domain,
        allowed=STATE_BENCH_DOMAINS,
        label="state_bench_domain",
    )
    native_prefix_items = state_bench_successor_visible_prefix(
        live_native_items
    )
    steps = _state_bench_replay_steps(native_prefix_items)
    allowed_tool_names = _native_tool_schema_names(native_tool_schemas)
    if any(str(step["name"]) not in allowed_tool_names for step in steps):
        raise ValueError("state_bench_native_prefix_tool_not_in_schema")
    packet = build_state_bench_fixed_handoff(
        domain=domain_name,
        native_system_prompt=native_system_prompt,
        original_request=original_request,
        accepted_transcript=native_prefix_items,
        reached_environment_sha256=canonical_environment_sha256(
            reached_environment_snapshot,
            surface=STATE_BENCH_SURFACE,
            surface_variant=domain_name,
            snapshot_schema=STATE_BENCH_ENVIRONMENT_SCHEMA,
        ),
        native_tool_schemas=native_tool_schemas,
        cheap_model=cheap_model,
        cheap_reasoning_effort=cheap_reasoning_effort,
        strong_model=strong_model,
        strong_reasoning_effort=strong_reasoning_effort,
        strong_call_cap=strong_call_cap,
        strong_output_token_cap=strong_output_token_cap,
    )
    return PrivateFixedHandoffReplayBundle(
        packet=packet,
        replay_protocol=STATE_BENCH_REPLAY_PROTOCOL,
        native_replay_steps=steps,
    )


def state_bench_successor_visible_prefix(
    live_native_items: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Remove hidden reasoning while retaining exact native call bindings."""

    projected: list[dict[str, Any]] = []
    for index, item in enumerate(live_native_items):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"state_bench_live_native_item_{index}_must_be_mapping"
            )
        item_type = str(item.get("type") or "")
        if item_type == "reasoning":
            continue
        if item_type and item_type not in {
            "message",
            "function_call",
            "function_call_output",
        }:
            raise ValueError(
                f"state_bench_live_native_item_{index}_unsupported_type:"
                f"{item_type}"
            )
        projected.append(dict(item))
    return tuple(projected)


def _agent_diff_replay_steps(
    events: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[tuple[str, str], ...],
]:
    steps: list[dict[str, Any]] = []
    bindings: list[tuple[str, str]] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise TypeError(
                f"agent_diff_native_action_{index}_must_be_mapping"
            )
        command = str(event.get("action") or "").strip()
        exit_code = event.get("tool_exit_code")
        observation = event.get("observation")
        if not command:
            raise ValueError(
                f"agent_diff_native_action_{index}_command_missing"
            )
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise TypeError(
                f"agent_diff_native_action_{index}_exit_code_missing"
            )
        if not isinstance(observation, str):
            raise TypeError(
                f"agent_diff_native_action_{index}_observation_missing"
            )
        bindings.append((command, observation))
        steps.append(
            {
                "kind": "agent_diff_bash",
                "command": command,
                "expected_exit_code": exit_code,
                "expected_receipt_sha256": canonical_replay_receipt_sha256(
                    {
                        "exit_code": exit_code,
                        "observation": observation,
                    },
                    replay_protocol=AGENT_DIFF_REPLAY_PROTOCOL,
                ),
            }
        )
    return tuple(steps), tuple(bindings)


def _agent_diff_transcript_bindings(
    items: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    bindings: list[tuple[str, str]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"agent_diff_transcript_item_{index}_must_be_mapping"
            )
        content = item.get("content")
        if item.get("role") != "assistant" or not isinstance(content, str):
            continue
        action = agent_diff_native_action(content)
        if action is None:
            continue
        if index + 1 >= len(items):
            raise ValueError(
                "agent_diff_live_prefix_observation_receipt_mismatch"
            )
        observation_item = items[index + 1]
        if (
            not isinstance(observation_item, Mapping)
            or observation_item.get("role") != "user"
            or not isinstance(observation_item.get("content"), str)
        ):
            raise ValueError(
                "agent_diff_live_prefix_observation_receipt_mismatch"
            )
        observation_content = str(observation_item["content"])
        prefix = "<observation>\n"
        suffix = "\n</observation>"
        if not (
            observation_content.startswith(prefix)
            and observation_content.endswith(suffix)
        ):
            raise ValueError(
                "agent_diff_live_prefix_observation_receipt_mismatch"
            )
        bindings.append(
            (
                action,
                observation_content[len(prefix) : -len(suffix)],
            )
        )
    return tuple(bindings)


def _state_bench_replay_steps(
    items: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    steps: list[dict[str, Any]] = []
    pending: dict[str, int] = {}
    seen_call_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"state_bench_native_prefix_item_{index}_must_be_mapping"
            )
        if item.get("tool_calls"):
            raise ValueError(
                "state_bench_lossy_scored_transcript_not_replayable"
            )
        item_type = str(item.get("type") or "")
        if item_type == "function_call":
            call_id = str(item.get("call_id") or "").strip()
            name = str(item.get("name") or "").strip()
            arguments = _state_bench_arguments(item.get("arguments"), index)
            if not call_id:
                raise ValueError(
                    f"state_bench_native_prefix_item_{index}_call_id_missing"
                )
            if call_id in seen_call_ids:
                raise ValueError(
                    "state_bench_native_prefix_call_id_not_unique"
                )
            if not name:
                raise ValueError(
                    f"state_bench_native_prefix_item_{index}_name_missing"
                )
            seen_call_ids.add(call_id)
            pending[call_id] = len(steps)
            steps.append(
                {
                    "kind": "state_bench_tool_call",
                    "name": name,
                    "arguments": arguments,
                    "call_id": call_id,
                    "expected_receipt_sha256": None,
                }
            )
        elif item_type == "function_call_output":
            call_id = str(item.get("call_id") or "").strip()
            step_index = pending.pop(call_id, None)
            if step_index is None:
                raise ValueError(
                    "state_bench_native_prefix_output_without_call"
                )
            steps[step_index]["expected_receipt_sha256"] = (
                canonical_replay_receipt_sha256(
                    _state_bench_output_value(item.get("output")),
                    replay_protocol=STATE_BENCH_REPLAY_PROTOCOL,
                )
            )
    if pending:
        raise ValueError("state_bench_native_prefix_call_output_missing")
    return tuple(steps)


def _state_bench_arguments(value: Any, index: int) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"state_bench_native_prefix_item_{index}_arguments_invalid"
            ) from exc
    if not isinstance(value, Mapping):
        raise TypeError(
            f"state_bench_native_prefix_item_{index}_arguments_invalid"
        )
    return dict(value)


def _state_bench_output_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _native_tool_schema_names(
    schemas: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    names: set[str] = set()
    for index, schema in enumerate(schemas):
        if not isinstance(schema, Mapping):
            raise TypeError(
                f"native_tool_schema_{index}_must_be_mapping"
            )
        function = schema.get("function")
        record = function if isinstance(function, Mapping) else schema
        name = str(record.get("name") or "").strip()
        if not name:
            raise ValueError(f"native_tool_schema_{index}_name_missing")
        names.add(name)
    return frozenset(names)


def _allowed_variant(
    value: Any,
    *,
    allowed: frozenset[str],
    label: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label}_must_be_str")
    normalized = value.strip()
    if normalized not in allowed:
        raise ValueError(f"{label}_unsupported:{normalized}")
    return normalized
