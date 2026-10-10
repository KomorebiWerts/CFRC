"""Protected successor runtimes over native surface callbacks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack
from copy import deepcopy
from typing import Any

from papc.baselines.fixed_handoff import (
    FixedHandoffPacket,
    PrivateFixedHandoffReplayBundle,
    canonical_environment_sha256,
    canonical_replay_receipt_sha256,
)
from papc.baselines.surface_adapters import (
    AGENT_DIFF_SURFACE,
    STATE_BENCH_SURFACE,
    agent_diff_native_action,
    agent_diff_native_done,
    state_bench_native_termination,
)
from papc.baselines.takeover import (
    IMPLICIT_DIRECT_TAKEOVER,
    PARADIGM_TAKEOVER,
    TakeoverArm,
)

AGENT_DIFF_RUNTIME_SCHEMA = "papc.agent_diff.protected_takeover_runtime.v1"
STATE_BENCH_RUNTIME_SCHEMA = "papc.state_bench.protected_takeover_runtime.v1"
MATCHED_REPLAY_SCHEMA = "papc.protected_matched_replay.v1"
FULL_CFRC_ARM = "full_cfrc"
MATCHED_REPLAY_ARM_ORDER = (
    IMPLICIT_DIRECT_TAKEOVER,
    PARADIGM_TAKEOVER,
    FULL_CFRC_ARM,
)

AgentDiffComplete = Callable[..., Mapping[str, Any]]
AgentDiffExecute = Callable[[str, int], Mapping[str, Any]]
StateBenchComplete = Callable[..., Mapping[str, Any]]
StateBenchToolHandler = Callable[[Mapping[str, Any]], Any]
OpenEnvironment = Callable[[], AbstractContextManager[Any]]
ReplayPrefix = Callable[
    [Any, tuple[dict[str, Any], ...]],
    Sequence[Any],
]
SnapshotEnvironment = Callable[[Any], Any]
EnvironmentBackend = Callable[[Any], Any]
CaptureInitialScorerAnchor = Callable[[Any], Any]
MatchedArmCallback = Callable[[Any, FixedHandoffPacket], Any]
MatchedScoreCallback = Callable[[Any, Any, Any], bool]


class _MatchedReplayCallbackFailure(Exception):
    def __init__(self, public_error: str) -> None:
        self.public_error = public_error


def _validated_matched_arm_order(arm_order: Sequence[str]) -> tuple[str, ...]:
    if isinstance(arm_order, (str, bytes)):
        raise TypeError("matched_replay_arm_order_must_be_sequence")
    resolved = tuple(arm_order)
    if not resolved:
        raise ValueError("matched_replay_arm_order_empty")
    if any(not isinstance(arm, str) or not arm.strip() for arm in resolved):
        raise ValueError("matched_replay_arm_order_invalid")
    if len(set(resolved)) != len(resolved):
        raise ValueError("matched_replay_arm_order_duplicate")
    return resolved


class _RedactedEnvironmentContext:
    def __init__(self, open_environment: OpenEnvironment) -> None:
        self._open_environment = open_environment
        self._manager: AbstractContextManager[Any] | None = None

    def __enter__(self) -> Any:
        factory_failed = False
        try:
            manager = self._open_environment()
        except Exception:
            factory_failed = True
            manager = None
        if factory_failed:
            raise _MatchedReplayCallbackFailure(
                "matched_replay_environment_factory_failed"
            ) from None

        enter_failed = False
        try:
            environment = manager.__enter__()
        except Exception:
            enter_failed = True
            environment = None
        if enter_failed:
            raise _MatchedReplayCallbackFailure(
                "matched_replay_environment_enter_failed"
            ) from None
        self._manager = manager
        return environment

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        exit_failed = False
        try:
            self._manager.__exit__(exc_type, exc_value, traceback)
        except Exception:
            exit_failed = True
        if exit_failed:
            raise _MatchedReplayCallbackFailure(
                "matched_replay_environment_exit_failed"
            ) from None
        return False


def _validated_matching_groups(
    matching_groups: Sequence[Sequence[str]] | None,
    *,
    arm_order: tuple[str, ...],
    arm_specific: bool,
) -> tuple[tuple[str, ...], ...]:
    if matching_groups is None:
        if arm_specific:
            raise ValueError("matched_replay_matching_groups_required")
        return (arm_order,)
    if isinstance(matching_groups, (str, bytes)):
        raise TypeError("matched_replay_matching_groups_must_be_sequence")
    groups: list[tuple[str, ...]] = []
    flattened: list[str] = []
    for raw_group in matching_groups:
        group = _validated_matched_arm_order(raw_group)
        groups.append(group)
        flattened.extend(group)
    if set(flattened) != set(arm_order) or len(flattened) != len(arm_order):
        raise ValueError("matched_replay_matching_groups_not_partition")
    positions = {arm: index for index, arm in enumerate(arm_order)}
    if any(
        list(group) != sorted(group, key=positions.__getitem__)
        for group in groups
    ):
        raise ValueError("matched_replay_matching_group_order_mismatch")
    return tuple(groups)


def run_matched_replay(
    *,
    bundle: PrivateFixedHandoffReplayBundle | None = None,
    arm_bundles: Mapping[str, PrivateFixedHandoffReplayBundle] | None = None,
    matching_groups: Sequence[Sequence[str]] | None = None,
    open_environment: OpenEnvironment,
    replay_prefix: ReplayPrefix,
    snapshot_environment: SnapshotEnvironment,
    environment_backend: EnvironmentBackend,
    capture_initial_scorer_anchor: CaptureInitialScorerAnchor,
    arm_callbacks: Mapping[str, MatchedArmCallback],
    score_arm: MatchedScoreCallback,
    arm_order: Sequence[str] = MATCHED_REPLAY_ARM_ORDER,
) -> dict[str, Any]:
    """Admit isolated arms after source-specific matched prefix replay.

    The legacy ``bundle`` form replays one byte-identical handoff into every
    arm.  ``arm_bundles`` supports a hierarchical experiment whose arms share
    an initial episode anchor but intentionally take over at different cuts.
    ``matching_groups`` partitions those arms into cuts that must remain
    byte-identical, such as the three contract-bearing component arms.

    ``environment_backend`` must return the state-owning backend object. The
    coordinator reads the immutable identity from that object and never accepts
    a caller-supplied identity string.
    """

    resolved_arm_order = _validated_matched_arm_order(arm_order)
    if (bundle is None) == (arm_bundles is None):
        raise ValueError("matched_replay_exactly_one_bundle_mode_required")
    arm_specific = arm_bundles is not None
    if bundle is not None:
        bundles = {arm: bundle for arm in resolved_arm_order}
    else:
        bundles = dict(arm_bundles or {})
        if set(bundles) != set(resolved_arm_order):
            raise ValueError("matched_replay_bundle_arm_set_mismatch")
        if any(
            not isinstance(bundles[arm], PrivateFixedHandoffReplayBundle)
            for arm in resolved_arm_order
        ):
            raise TypeError("matched_replay_bundle_invalid")
    groups = _validated_matching_groups(
        matching_groups,
        arm_order=resolved_arm_order,
        arm_specific=arm_specific,
    )

    callbacks = dict(arm_callbacks)
    expected_arms = set(resolved_arm_order)
    if set(callbacks) != expected_arms:
        raise ValueError("matched_replay_arm_set_mismatch")
    if not all(callable(callbacks[arm]) for arm in resolved_arm_order):
        raise TypeError("matched_replay_arm_callback_not_callable")
    for callback, label in (
        (open_environment, "open_environment"),
        (replay_prefix, "replay_prefix"),
        (snapshot_environment, "snapshot_environment"),
        (environment_backend, "environment_backend"),
        (score_arm, "score_arm"),
        (
            capture_initial_scorer_anchor,
            "capture_initial_scorer_anchor",
        ),
    ):
        if not callable(callback):
            raise TypeError(f"matched_replay_{label}_not_callable")
    for replay_bundle in bundles.values():
        replay_bundle.assert_integrity()

    callback_failure: str | None = None
    result: dict[str, Any] | None = None
    try:
        with ExitStack() as stack:
            environments = {
                arm: stack.enter_context(
                    _RedactedEnvironmentContext(open_environment)
                )
                for arm in resolved_arm_order
            }
            if len({id(environment) for environment in environments.values()}) != len(
                resolved_arm_order
            ):
                raise RuntimeError("matched_replay_environment_wrapper_reused")
            backend_identities = {
                arm: _environment_backend_identity(
                    environment_backend,
                    environment,
                )
                for arm, environment in environments.items()
            }
            backend_object_ids = {
                id(identity[0]) for identity in backend_identities.values()
            }
            backend_external_ids = [
                identity[1]
                for identity in backend_identities.values()
                if identity[1] is not None
            ]
            if (
                len(backend_object_ids) != len(resolved_arm_order)
                or len(set(backend_external_ids))
                != len(backend_external_ids)
            ):
                raise RuntimeError(
                    "matched_replay_environment_backend_not_fresh"
                )

            initial_scorer_anchors: dict[str, Any] = {}
            initial_scorer_anchor_digests: dict[str, str] = {}
            for arm, environment in environments.items():
                packet = bundles[arm].packet
                anchor = deepcopy(
                    _invoke_matched_replay_callback(
                        "matched_replay_initial_scorer_anchor_failed",
                        capture_initial_scorer_anchor,
                        environment,
                    )
                )
                initial_scorer_anchors[arm] = anchor
                initial_scorer_anchor_digests[arm] = (
                    canonical_environment_sha256(
                        anchor,
                        surface=packet.surface,
                        surface_variant=packet.surface_variant,
                        snapshot_schema=packet.reached_environment_schema,
                    )
                )
            if len(set(initial_scorer_anchor_digests.values())) != 1:
                raise RuntimeError(
                    "matched_replay_initial_scorer_anchor_digest_mismatch"
                )
            common_initial_scorer_anchor_digest = (
                initial_scorer_anchor_digests[resolved_arm_order[0]]
            )

            replay_receipts: dict[str, tuple[str, ...]] = {}
            for arm, environment in environments.items():
                replay_bundle = bundles[arm]
                replay_bundle.assert_integrity()
                replay_receipts[arm] = _canonical_replay_receipt_sha256s(
                    _invoke_matched_replay_callback(
                        "matched_replay_prefix_replay_failed",
                        replay_prefix,
                        environment,
                        replay_bundle.replay_steps_for_runtime(),
                    ),
                    replay_protocol=replay_bundle.replay_protocol,
                )
            if any(
                replay_receipts[arm]
                != bundles[arm].expected_replay_receipt_sha256s
                for arm in resolved_arm_order
            ):
                raise RuntimeError("matched_replay_receipt_digest_mismatch")

            reached_digests = {
                arm: _matched_replay_environment_digest(
                    snapshot_environment=snapshot_environment,
                    environment=environment,
                    packet=bundles[arm].packet,
                )
                for arm, environment in environments.items()
            }
            for arm in resolved_arm_order:
                if (
                    reached_digests[arm]
                    != bundles[arm].packet.reached_environment_sha256
                ):
                    raise RuntimeError(
                        "matched_replay_expected_reached_state_digest_mismatch"
                    )

            public_bindings = {
                arm: bundles[arm].public_matching_contract()
                for arm in resolved_arm_order
            }
            for group in groups:
                group_reached = {reached_digests[arm] for arm in group}
                if len(group_reached) != 1:
                    raise RuntimeError(
                        "matched_replay_group_reached_state_digest_mismatch"
                    )
                group_bindings = {
                    json.dumps(
                        public_bindings[arm],
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for arm in group
                }
                if len(group_bindings) != 1:
                    raise RuntimeError(
                        "matched_replay_group_handoff_binding_mismatch"
                    )

            arm_results: dict[str, dict[str, bool]] = {}
            for arm_index, arm in enumerate(resolved_arm_order):
                not_yet_run = resolved_arm_order[arm_index:]
                for pending_arm in not_yet_run:
                    current_identity = _environment_backend_identity(
                        environment_backend,
                        environments[pending_arm]
                    )
                    expected_identity = backend_identities[pending_arm]
                    if (
                        current_identity[0] is not expected_identity[0]
                        or current_identity[1] != expected_identity[1]
                    ):
                        raise RuntimeError(
                            "matched_replay_environment_backend_identity_changed"
                        )
                    current_digest = _matched_replay_environment_digest(
                        snapshot_environment=snapshot_environment,
                        environment=environments[pending_arm],
                        packet=bundles[pending_arm].packet,
                    )
                    if current_digest != reached_digests[pending_arm]:
                        raise RuntimeError(
                            "matched_replay_pre_arm_reached_state_digest_mismatch"
                        )
                for replay_bundle in bundles.values():
                    replay_bundle.assert_integrity()
                private_execution = _invoke_matched_replay_callback(
                    "matched_replay_arm_execution_failed",
                    callbacks[arm],
                    environments[arm],
                    bundles[arm].packet.copy_for_runtime(),
                )
                current_identity = _environment_backend_identity(
                    environment_backend,
                    environments[arm],
                )
                expected_identity = backend_identities[arm]
                if (
                    current_identity[0] is not expected_identity[0]
                    or current_identity[1] != expected_identity[1]
                ):
                    raise RuntimeError(
                        "matched_replay_environment_backend_identity_changed"
                    )
                official_primary_pass = _invoke_matched_replay_callback(
                    "matched_replay_official_scorer_failed",
                    score_arm,
                    environments[arm],
                    private_execution,
                    deepcopy(initial_scorer_anchors[arm]),
                )
                if type(official_primary_pass) is not bool:
                    raise TypeError(
                        "matched_replay_official_primary_pass_must_be_bool"
                    )
                arm_results[arm] = {
                    "official_primary_pass": official_primary_pass,
                }

            result = {
                "schema": MATCHED_REPLAY_SCHEMA,
                "initial_scorer_anchor_sha256": (
                    common_initial_scorer_anchor_digest
                ),
                "arm_order": list(resolved_arm_order),
                "matching_groups": [list(group) for group in groups],
                "reached_environment_sha256_by_arm": reached_digests,
                "matching_by_arm": public_bindings,
                "arm_results": arm_results,
            }
            if not arm_specific:
                result = {
                    "schema": MATCHED_REPLAY_SCHEMA,
                    "matching": public_bindings[resolved_arm_order[0]],
                    "initial_scorer_anchor_sha256": (
                        common_initial_scorer_anchor_digest
                    ),
                    "reached_environment_sha256": reached_digests[
                        resolved_arm_order[0]
                    ],
                    "arm_order": list(resolved_arm_order),
                    "arm_results": arm_results,
                }
    except _MatchedReplayCallbackFailure as exc:
        callback_failure = exc.public_error
    if callback_failure is not None:
        raise RuntimeError(callback_failure) from None
    if result is None:
        raise RuntimeError("matched_replay_result_missing")
    return result


def run_agent_diff_takeover(
    *,
    packet: FixedHandoffPacket,
    arm: TakeoverArm,
    complete: AgentDiffComplete,
    execute: AgentDiffExecute,
    command_timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Run one protected Agent-Diff successor to native done or call cap."""

    if packet.surface != AGENT_DIFF_SURFACE:
        raise ValueError("agent_diff_runtime_surface_mismatch")
    if command_timeout_seconds < 1:
        raise ValueError("command_timeout_seconds_must_be_positive")
    prompt = packet.prompt_for(arm)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt.system_instruction},
        *(deepcopy(list(prompt.messages))),
    ]
    trace: list[dict[str, Any]] = []
    usage_records: list[dict[str, Any]] = []
    tool_records: list[dict[str, Any]] = []
    model_latency_seconds = 0.0

    for call_index in range(1, packet.strong_call_cap + 1):
        receipt = dict(
            complete(
                model=packet.strong_model,
                reasoning_effort=packet.strong_reasoning_effort,
                messages=deepcopy(messages),
                max_output_tokens=packet.strong_output_token_cap,
            )
        )
        text = _required_response_text(receipt)
        usage_records.append(_usage_record(receipt))
        latency = receipt.get("latency_seconds")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            model_latency_seconds += max(0.0, float(latency))
        event: dict[str, Any] = {
            "call_index": call_index,
            "assistant_text_sha256": _text_sha256(text),
        }

        action = agent_diff_native_action(text)
        if action is not None:
            result = dict(execute(action, command_timeout_seconds))
            observation, exit_code = _agent_diff_observation(
                result,
                char_cap=packet.native_observation_char_cap,
            )
            tool_record = {
                "call_index": call_index,
                "action_sha256": _text_sha256(action),
                "exit_code": exit_code,
                "observation_sha256": _text_sha256(observation),
            }
            tool_records.append(tool_record)
            event.update({"kind": "native_action", **tool_record})
            trace.append(event)
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": f"<observation>\n{observation}\n</observation>",
                }
            )
            continue

        done = agent_diff_native_done(text)
        if done is not None:
            event["kind"] = "native_terminal"
            trace.append(event)
            return {
                "schema": AGENT_DIFF_RUNTIME_SCHEMA,
                "arm": arm,
                "status": "completed",
                "completion_reason": "react_done_tag",
                "provider_call_count": call_index,
                "model_latency_seconds": round(model_latency_seconds, 3),
                "usage_records": usage_records,
                "tool_records": tool_records,
                "trace": trace,
                "final_message": done,
            }

        event["kind"] = "native_format_error"
        trace.append(event)
        messages.append({"role": "assistant", "content": text})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Please respond using either <action>one bash "
                    "command</action> or <done>brief summary</done>."
                ),
            }
        )

    return {
        "schema": AGENT_DIFF_RUNTIME_SCHEMA,
        "arm": arm,
        "status": "incomplete",
        "completion_reason": "strong_call_cap_exhausted",
        "provider_call_count": packet.strong_call_cap,
        "model_latency_seconds": round(model_latency_seconds, 3),
        "usage_records": usage_records,
        "tool_records": tool_records,
        "trace": trace,
        "final_message": "",
    }


class StateBenchTakeoverRuntime:
    """Stateful strong successor for native STATE-Bench tool turns."""

    def __init__(
        self,
        *,
        packet: FixedHandoffPacket,
        arm: TakeoverArm,
        complete_with_tools: StateBenchComplete,
        tool_handlers: Mapping[str, StateBenchToolHandler],
    ) -> None:
        if packet.surface != STATE_BENCH_SURFACE:
            raise ValueError("state_bench_runtime_surface_mismatch")
        self.packet = packet
        self.arm = arm
        self._complete = complete_with_tools
        self._handlers = dict(tool_handlers)
        self._allowed_tool_names = _tool_schema_names(
            packet.native_tool_schemas
        )
        if set(self._handlers) != self._allowed_tool_names:
            raise ValueError("state_bench_native_tool_handler_set_mismatch")
        prompt = packet.prompt_for(arm)
        self.system_instruction = prompt.system_instruction
        self.messages: list[dict[str, Any]] = deepcopy(list(prompt.messages))
        self.provider_call_count = 0
        self.usage_records: list[dict[str, Any]] = []
        self.tool_records: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []
        self._awaiting_user = False
        self._closed = False

    def next_turn(self, user_message: str | None = None) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("state_bench_takeover_runtime_closed")
        if self._awaiting_user and user_message is None:
            raise ValueError("state_bench_next_user_message_required")
        if not self._awaiting_user and user_message is not None:
            raise ValueError("state_bench_unexpected_user_message")
        if self._awaiting_user:
            assert user_message is not None
            if state_bench_native_termination(user_message):
                self._closed = True
                return self._result(
                    status="completed",
                    completion_reason="user_simulator_task_done_marker",
                    final_message="",
                )
            if not user_message.strip():
                raise ValueError("state_bench_user_message_empty")
            self.messages.append({"role": "user", "content": user_message})
            self._awaiting_user = False

        provider_input = deepcopy(self.messages)
        previous_response_id: str | None = None
        while self.provider_call_count < self.packet.strong_call_cap:
            self.provider_call_count += 1
            response = dict(
                self._complete(
                    model=self.packet.strong_model,
                    reasoning_effort=self.packet.strong_reasoning_effort,
                    instructions=self.system_instruction,
                    input=provider_input,
                    tools=self.packet.tools_for_provider(),
                    previous_response_id=previous_response_id,
                    max_output_tokens=self.packet.strong_output_token_cap,
                )
            )
            text = str(response.get("text") or "")
            tool_calls = _state_bench_tool_calls(response.get("tool_calls"))
            self.usage_records.append(_usage_record(response))
            event: dict[str, Any] = {
                "call_index": self.provider_call_count,
                "assistant_text_sha256": _text_sha256(text),
                "native_tool_call_count": len(tool_calls),
            }
            if tool_calls:
                response_id = str(response.get("response_id") or "").strip()
                if not response_id:
                    raise ValueError(
                        "state_bench_native_tool_response_id_missing"
                    )
                executed: list[dict[str, Any]] = []
                provider_tool_results: list[dict[str, Any]] = []
                for call in tool_calls:
                    name = call["name"]
                    if name not in self._allowed_tool_names:
                        event["kind"] = "blocked_unknown_native_tool"
                        event["blocked_tool_name"] = name
                        self.trace.append(event)
                        self._closed = True
                        return self._result(
                            status="blocked",
                            completion_reason="unknown_native_tool",
                            final_message="",
                        )
                    result = self._handlers[name](call["arguments"])
                    record = {
                        "call_index": self.provider_call_count,
                        "name": name,
                        "call_id": call["call_id"],
                        "arguments_sha256": _jsonish_sha256(call["arguments"]),
                        "result_sha256": _jsonish_sha256(result),
                    }
                    self.tool_records.append(record)
                    executed.append(
                        {
                            "name": name,
                            "call_id": call["call_id"],
                            "arguments": deepcopy(call["arguments"]),
                            "result": deepcopy(result),
                        }
                    )
                    provider_tool_results.append(
                        {
                            "type": "function_call_output",
                            "call_id": call["call_id"],
                            "output": json.dumps(
                                result,
                                ensure_ascii=False,
                            ),
                        }
                    )
                event["kind"] = "native_tools_executed"
                self.trace.append(event)
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": deepcopy(tool_calls),
                    }
                )
                self.messages.append({"role": "tool", "content": executed})
                provider_input = provider_tool_results
                previous_response_id = response_id
                continue

            if not text.strip():
                event["kind"] = "blocked_empty_native_response"
                self.trace.append(event)
                self._closed = True
                return self._result(
                    status="blocked",
                    completion_reason="empty_native_assistant_response",
                    final_message="",
                )
            event["kind"] = "assistant_turn_complete"
            self.trace.append(event)
            self.messages.append({"role": "assistant", "content": text})
            self._awaiting_user = True
            return self._result(
                status="assistant_turn_complete",
                completion_reason="native_assistant_response",
                final_message=text,
            )

        self._closed = True
        return self._result(
            status="incomplete",
            completion_reason="strong_call_cap_exhausted",
            final_message="",
        )

    def _result(
        self,
        *,
        status: str,
        completion_reason: str,
        final_message: str,
    ) -> dict[str, Any]:
        return {
            "schema": STATE_BENCH_RUNTIME_SCHEMA,
            "arm": self.arm,
            "status": status,
            "completion_reason": completion_reason,
            "provider_call_count": self.provider_call_count,
            "usage_records": deepcopy(self.usage_records),
            "tool_records": deepcopy(self.tool_records),
            "trace": deepcopy(self.trace),
            "final_message": final_message,
        }


def _agent_diff_observation(
    result: Mapping[str, Any],
    *,
    char_cap: int | None,
) -> tuple[str, int]:
    raw_exit_code = result.get("exit_code")
    if isinstance(raw_exit_code, bool) or not isinstance(raw_exit_code, int):
        raise TypeError("agent_diff_native_exit_code_missing")
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    observation = stdout.strip() or "(empty output)"
    if raw_exit_code != 0:
        observation = f"{stdout}\n[stderr]: {stderr}".strip()
    if char_cap is not None:
        observation = observation[:char_cap]
    return observation, raw_exit_code


def _state_bench_tool_calls(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise TypeError("state_bench_native_tool_calls_must_be_list")
    calls: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"state_bench_native_tool_call_{index}_must_be_mapping"
            )
        name = str(raw.get("name") or "").strip()
        call_id = str(raw.get("call_id") or "").strip()
        arguments = raw.get("arguments")
        if not name:
            raise ValueError(
                f"state_bench_native_tool_call_{index}_name_missing"
            )
        if not isinstance(arguments, Mapping):
            raise TypeError(
                f"state_bench_native_tool_call_{index}_arguments_invalid"
            )
        if not call_id:
            raise ValueError(
                f"state_bench_native_tool_call_{index}_call_id_missing"
            )
        calls.append(
            {
                "name": name,
                "call_id": call_id,
                "arguments": dict(arguments),
            }
        )
    return calls


def _tool_schema_names(
    schemas: tuple[dict[str, Any], ...],
) -> set[str]:
    names: set[str] = set()
    for tool in schemas:
        function = tool.get("function")
        record = function if isinstance(function, Mapping) else tool
        names.add(str(record.get("name") or ""))
    return names


def _required_response_text(receipt: Mapping[str, Any]) -> str:
    text = receipt.get("text")
    if not isinstance(text, str):
        raise TypeError("agent_diff_provider_text_missing")
    return text


def _usage_record(receipt: Mapping[str, Any]) -> dict[str, Any]:
    usage = receipt.get("usage")
    return deepcopy(dict(usage)) if isinstance(usage, Mapping) else {}


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonish_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _environment_backend_identity(
    environment_backend: EnvironmentBackend,
    environment: Any,
) -> tuple[Any, str | None]:
    backend = _invoke_matched_replay_callback(
        "matched_replay_environment_backend_failed",
        environment_backend,
        environment,
    )
    if backend is None or isinstance(
        backend,
        (str, bytes, bytearray, bool, int, float),
    ):
        raise _MatchedReplayCallbackFailure(
            "matched_replay_environment_backend_identity_failed"
        ) from None

    missing = object()
    for attribute in ("environmentId", "environment_id", "instance_id"):
        try:
            value = getattr(backend, attribute, missing)
        except Exception:
            raise _MatchedReplayCallbackFailure(
                "matched_replay_environment_backend_identity_failed"
            ) from None
        if value is missing:
            continue
        if not isinstance(value, str) or not value.strip():
            raise _MatchedReplayCallbackFailure(
                "matched_replay_environment_backend_identity_failed"
            ) from None
        return backend, value.strip()
    return backend, None


def _matched_replay_environment_digest(
    *,
    snapshot_environment: SnapshotEnvironment,
    environment: Any,
    packet: FixedHandoffPacket,
) -> str:
    return canonical_environment_sha256(
        _invoke_matched_replay_callback(
            "matched_replay_environment_snapshot_failed",
            snapshot_environment,
            environment,
        ),
        surface=packet.surface,
        surface_variant=packet.surface_variant,
        snapshot_schema=packet.reached_environment_schema,
    )


def _invoke_matched_replay_callback(
    public_error: str,
    callback: Callable[..., Any],
    *args: Any,
) -> Any:
    callback_failed = False
    try:
        result = callback(*args)
    except Exception:
        callback_failed = True
        result = None
    if callback_failed:
        raise _MatchedReplayCallbackFailure(public_error) from None
    return result


def _canonical_replay_receipt_sha256s(
    value: Any,
    *,
    replay_protocol: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("matched_replay_receipts_must_be_sequence")
    return tuple(
        canonical_replay_receipt_sha256(
            receipt,
            replay_protocol=replay_protocol,
        )
        for receipt in value
    )
