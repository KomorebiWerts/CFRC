"""Method-neutral binding for matched reached-state baseline arms."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from papc.baselines.takeover import (
    IMPLICIT_DIRECT_TAKEOVER,
    PARADIGM_TAKEOVER,
    TakeoverArm,
    TakeoverPrompt,
    build_implicit_direct_takeover,
    build_paradigm_takeover,
)

SCHEMA = "papc.neutral_fixed_handoff.v2"
PRIVATE_REPLAY_SCHEMA = "papc.private_fixed_handoff_replay.v1"
PUBLIC_REPLAY_BINDING_SCHEMA = "papc.fixed_handoff_replay_binding.v1"
CANONICAL_ENVIRONMENT_SCHEMA = "papc.canonical_native_environment.v1"
CANONICAL_REPLAY_RECEIPT_SCHEMA = "papc.canonical_native_replay_receipt.v1"
TRANSCRIPT_SUCCESSOR_READY = "successor_ready"
TRANSCRIPT_ACCEPTED_TERMINAL = "accepted_terminal"
_TRANSCRIPT_STATUSES = frozenset({TRANSCRIPT_SUCCESSOR_READY, TRANSCRIPT_ACCEPTED_TERMINAL})
METHOD_PRIVATE_FIELD_NAMES = frozenset(
    {
        "frontier",
        "contract",
        "state_dump",
        "completed_action_list",
        "task_id",
        "gold_outcome",
        "direct_outcome",
        "fixed_object_id",
        "fixed_tool_args",
        "fixed_kb_query",
        "answer_sketch",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FixedHandoffPacket:
    """Arm-invariant inputs needed to bind one matched handoff."""

    surface: str
    surface_variant: str
    native_system_prompt: str = field(repr=False)
    original_request: str = field(repr=False)
    accepted_transcript: tuple[dict[str, Any], ...] = field(repr=False)
    reached_environment_schema: str
    reached_environment_sha256: str
    native_tool_schemas: tuple[dict[str, Any], ...] = field(repr=False)
    native_terminal_signal: str
    official_scorer: str
    full_denominator: int
    cheap_model: str
    cheap_reasoning_effort: str
    strong_model: str
    strong_reasoning_effort: str
    strong_call_cap: int
    strong_output_token_cap: int
    native_observation_char_cap: int | None = None
    accepted_transcript_status: str = TRANSCRIPT_SUCCESSOR_READY
    _accepted_transcript_sha256: str = field(init=False, repr=False)
    _native_tool_schema_sha256: str = field(init=False, repr=False)
    _matching_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for field_name in (
            "surface",
            "surface_variant",
            "reached_environment_schema",
            "native_terminal_signal",
            "official_scorer",
            "cheap_model",
            "cheap_reasoning_effort",
            "strong_model",
            "strong_reasoning_effort",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        transcript_status = _required_text(
            self.accepted_transcript_status,
            "accepted_transcript_status",
        )
        if transcript_status not in _TRANSCRIPT_STATUSES:
            raise ValueError("accepted_transcript_status_invalid")
        object.__setattr__(
            self,
            "accepted_transcript_status",
            transcript_status,
        )
        object.__setattr__(
            self,
            "native_system_prompt",
            _required_verbatim_text(
                self.native_system_prompt,
                "native_system_prompt",
            ),
        )

        paradigm = build_paradigm_takeover(self.original_request)
        request = str(paradigm.messages[0]["content"])
        implicit = build_implicit_direct_takeover(self.accepted_transcript)
        system_messages = [
            (index, message)
            for index, message in enumerate(implicit.messages)
            if message.get("role") == "system"
        ]
        if system_messages:
            if len(system_messages) != 1 or system_messages[0][0] != 0:
                raise ValueError("accepted_transcript_system_message_invalid")
            system_content = system_messages[0][1].get("content")
            if (
                not isinstance(system_content, str)
                or system_content != self.native_system_prompt
            ):
                raise ValueError(
                    "accepted_transcript_native_system_prompt_mismatch"
                )
            implicit = build_implicit_direct_takeover(
                implicit.messages[1:]
            )
        first_user = next(
            message
            for message in implicit.messages
            if message.get("role") == "user"
        )
        if not isinstance(first_user.get("content"), str):
            raise TypeError("accepted_transcript_first_user_content_must_be_str")
        if str(first_user["content"]).strip() != request:
            raise ValueError("accepted_transcript_original_request_mismatch")
        last_item = implicit.messages[-1]
        if self.accepted_transcript_status == TRANSCRIPT_ACCEPTED_TERMINAL:
            if (
                last_item.get("role") != "assistant"
                and not (
                    last_item.get("type") == "message"
                    and last_item.get("role") == "assistant"
                )
            ):
                raise ValueError("accepted_terminal_transcript_not_assistant")
        elif (
            last_item.get("role") not in {"user", "tool"}
            and last_item.get("type") != "function_call_output"
        ):
            raise ValueError("accepted_transcript_not_successor_ready")

        environment_sha256 = _required_text(
            self.reached_environment_sha256,
            "reached_environment_sha256",
        )
        if not _SHA256_RE.fullmatch(environment_sha256):
            raise ValueError("reached_environment_sha256_invalid")
        tools = _validated_tool_schemas(self.native_tool_schemas)
        _required_positive_int(self.full_denominator, "full_denominator")
        _required_positive_int(self.strong_call_cap, "strong_call_cap")
        _required_positive_int(
            self.strong_output_token_cap,
            "strong_output_token_cap",
        )
        if self.native_observation_char_cap is not None:
            _required_positive_int(
                self.native_observation_char_cap,
                "native_observation_char_cap",
            )
        _canonical_bytes(
            {
                "accepted_transcript": implicit.messages,
                "native_tool_schemas": tools,
            }
        )

        object.__setattr__(self, "original_request", request)
        object.__setattr__(
            self,
            "accepted_transcript",
            tuple(deepcopy(message) for message in implicit.messages),
        )
        object.__setattr__(self, "reached_environment_sha256", environment_sha256)
        object.__setattr__(
            self,
            "native_tool_schemas",
            tuple(deepcopy(tool) for tool in tools),
        )
        object.__setattr__(
            self,
            "_accepted_transcript_sha256",
            _digest(self.accepted_transcript),
        )
        object.__setattr__(
            self,
            "_native_tool_schema_sha256",
            _digest(self.native_tool_schemas),
        )
        object.__setattr__(
            self,
            "_matching_sha256",
            _digest(self._matching_payload()),
        )

    @property
    def accepted_transcript_sha256(self) -> str:
        return self._accepted_transcript_sha256

    @property
    def native_tool_schema_sha256(self) -> str:
        return self._native_tool_schema_sha256

    @property
    def matching_sha256(self) -> str:
        return self._matching_sha256

    def prompt_for(self, arm: TakeoverArm) -> TakeoverPrompt:
        self.assert_integrity()
        if arm == IMPLICIT_DIRECT_TAKEOVER:
            prompt = build_implicit_direct_takeover(self.accepted_transcript)
        elif arm == PARADIGM_TAKEOVER:
            prompt = build_paradigm_takeover(self.original_request)
        else:
            raise ValueError(f"unknown_takeover_arm:{arm}")
        return TakeoverPrompt(
            arm=prompt.arm,
            system_instruction=(
                f"{self.native_system_prompt}\n\n"
                f"{prompt.system_instruction}"
            ),
            messages=prompt.messages,
            exposure=prompt.exposure,
        )

    def tools_for_provider(self) -> list[dict[str, Any]]:
        self.assert_integrity()
        return deepcopy(list(self.native_tool_schemas))

    def copy_for_runtime(self) -> FixedHandoffPacket:
        self.assert_integrity()
        return deepcopy(self)

    def matching_contract(self) -> dict[str, Any]:
        self.assert_integrity()
        return {
            "schema": SCHEMA,
            "surface": self.surface,
            "surface_variant": self.surface_variant,
            "native_system_prompt_sha256": hashlib.sha256(
                self.native_system_prompt.encode("utf-8")
            ).hexdigest(),
            "original_request_sha256": hashlib.sha256(
                self.original_request.encode("utf-8")
            ).hexdigest(),
            "accepted_transcript_sha256": self.accepted_transcript_sha256,
            "reached_environment_schema": self.reached_environment_schema,
            "reached_environment_sha256": self.reached_environment_sha256,
            "native_tool_schema_count": len(self.native_tool_schemas),
            "native_tool_schema_sha256": self.native_tool_schema_sha256,
            "native_terminal_signal": self.native_terminal_signal,
            "official_scorer": self.official_scorer,
            "full_denominator": self.full_denominator,
            "model_plan": {
                "cheap_model": self.cheap_model,
                "cheap_reasoning_effort": self.cheap_reasoning_effort,
                "strong_model": self.strong_model,
                "strong_reasoning_effort": self.strong_reasoning_effort,
            },
            "strong_call_cap": self.strong_call_cap,
            "strong_output_token_cap": self.strong_output_token_cap,
            "native_observation_char_cap": (
                self.native_observation_char_cap
            ),
            "accepted_transcript_status": self.accepted_transcript_status,
            "matching_sha256": self.matching_sha256,
        }

    def assert_integrity(self) -> None:
        if _digest(self.accepted_transcript) != self.accepted_transcript_sha256:
            raise RuntimeError("fixed_handoff_transcript_mutated")
        if _digest(self.native_tool_schemas) != self.native_tool_schema_sha256:
            raise RuntimeError("fixed_handoff_tool_schemas_mutated")
        if _digest(self._matching_payload()) != self.matching_sha256:
            raise RuntimeError("fixed_handoff_matching_binding_mutated")

    def _matching_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "surface": self.surface,
            "surface_variant": self.surface_variant,
            "native_system_prompt_sha256": hashlib.sha256(
                self.native_system_prompt.encode("utf-8")
            ).hexdigest(),
            "original_request_sha256": hashlib.sha256(
                self.original_request.encode("utf-8")
            ).hexdigest(),
            "accepted_transcript_sha256": self.accepted_transcript_sha256,
            "reached_environment_schema": self.reached_environment_schema,
            "reached_environment_sha256": self.reached_environment_sha256,
            "native_tool_schema_sha256": self.native_tool_schema_sha256,
            "native_terminal_signal": self.native_terminal_signal,
            "official_scorer": self.official_scorer,
            "full_denominator": self.full_denominator,
            "cheap_model": self.cheap_model,
            "cheap_reasoning_effort": self.cheap_reasoning_effort,
            "strong_model": self.strong_model,
            "strong_reasoning_effort": self.strong_reasoning_effort,
            "strong_call_cap": self.strong_call_cap,
            "strong_output_token_cap": self.strong_output_token_cap,
            "native_observation_char_cap": self.native_observation_char_cap,
            "accepted_transcript_status": self.accepted_transcript_status,
        }


@dataclass(frozen=True)
class PrivateFixedHandoffReplayBundle:
    """Private native prefix needed to reproduce one frozen reached state."""

    packet: FixedHandoffPacket
    replay_protocol: str
    native_replay_steps: tuple[dict[str, Any], ...] = field(repr=False)
    _native_replay_steps_sha256: str = field(init=False, repr=False)
    _matching_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        protocol = _required_text(self.replay_protocol, "replay_protocol")
        if not re.fullmatch(r"papc\.[a-z0-9_.]+\.v[1-9][0-9]*", protocol):
            raise ValueError("replay_protocol_not_versioned")
        if not protocol.startswith(f"papc.{self.packet.surface}."):
            raise ValueError("replay_protocol_surface_mismatch")
        steps = _validated_replay_steps(self.native_replay_steps)
        if any(
            not str(step["kind"]).startswith(f"{self.packet.surface}_")
            for step in steps
        ):
            raise ValueError("native_replay_step_surface_mismatch")
        object.__setattr__(self, "replay_protocol", protocol)
        object.__setattr__(
            self,
            "native_replay_steps",
            tuple(deepcopy(step) for step in steps),
        )
        object.__setattr__(
            self,
            "_native_replay_steps_sha256",
            _digest(self.native_replay_steps),
        )
        object.__setattr__(
            self,
            "_matching_sha256",
            _digest(self._matching_payload()),
        )

    @property
    def native_replay_steps_sha256(self) -> str:
        return self._native_replay_steps_sha256

    @property
    def matching_sha256(self) -> str:
        return self._matching_sha256

    @property
    def expected_replay_receipt_sha256s(self) -> tuple[str, ...]:
        return tuple(
            str(step["expected_receipt_sha256"])
            for step in self.native_replay_steps
        )

    def replay_steps_for_runtime(self) -> tuple[dict[str, Any], ...]:
        """Return executable fields without exposing expected receipt digests."""

        self.assert_integrity()
        return tuple(
            {
                key: deepcopy(value)
                for key, value in step.items()
                if key != "expected_receipt_sha256"
            }
            for step in self.native_replay_steps
        )

    def public_matching_contract(self) -> dict[str, Any]:
        """Return hashes and counts only; exact replay actions stay private."""

        self.assert_integrity()
        return {
            "schema": PUBLIC_REPLAY_BINDING_SCHEMA,
            "fixed_handoff": self.packet.matching_contract(),
            "replay_protocol": self.replay_protocol,
            "native_replay_step_count": len(self.native_replay_steps),
            "native_replay_steps_sha256": self.native_replay_steps_sha256,
            "matching_sha256": self.matching_sha256,
        }

    def assert_integrity(self) -> None:
        self.packet.assert_integrity()
        if (
            _digest(self.native_replay_steps)
            != self.native_replay_steps_sha256
        ):
            raise RuntimeError("fixed_handoff_replay_steps_mutated")
        if _digest(self._matching_payload()) != self.matching_sha256:
            raise RuntimeError("fixed_handoff_replay_binding_mutated")

    def _matching_payload(self) -> dict[str, Any]:
        return {
            "schema": PRIVATE_REPLAY_SCHEMA,
            "packet_matching_sha256": self.packet.matching_sha256,
            "replay_protocol": self.replay_protocol,
            "native_replay_steps_sha256": self.native_replay_steps_sha256,
        }


def canonical_environment_sha256(
    snapshot: Any,
    *,
    surface: str,
    surface_variant: str,
    snapshot_schema: str,
) -> str:
    """Hash a JSON snapshot with an explicit surface-owned schema."""

    return _digest(
        {
            "schema": CANONICAL_ENVIRONMENT_SCHEMA,
            "surface": _required_text(surface, "surface"),
            "surface_variant": _required_text(
                surface_variant,
                "surface_variant",
            ),
            "snapshot_schema": _required_text(
                snapshot_schema,
                "reached_environment_schema",
            ),
            "snapshot": snapshot,
        }
    )


def canonical_replay_receipt_sha256(
    receipt: Any,
    *,
    replay_protocol: str,
) -> str:
    """Hash one native replay receipt under its versioned protocol."""

    protocol = _required_text(replay_protocol, "replay_protocol")
    return _digest(
        {
            "schema": CANONICAL_REPLAY_RECEIPT_SCHEMA,
            "replay_protocol": protocol,
            "receipt": receipt,
        }
    )


def _validated_replay_steps(
    source: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(source, (str, bytes)):
        raise TypeError("native_replay_steps_must_be_sequence")
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(source):
        if not isinstance(item, Mapping):
            raise TypeError(f"native_replay_step_{index}_must_be_mapping")
        step = deepcopy(dict(item))
        kind = str(step.get("kind") or "").strip()
        if not kind:
            raise ValueError(f"native_replay_step_{index}_kind_missing")
        step["kind"] = kind
        if "expected_receipt_sha256" not in step:
            raise ValueError(
                f"native_replay_step_{index}_expected_receipt_sha256_missing"
            )
        for key, value in step.items():
            if key.endswith("_sha256") and not _SHA256_RE.fullmatch(
                str(value)
            ):
                raise ValueError(
                    f"native_replay_step_{index}_{key}_invalid"
                )
        _canonical_bytes(step)
        steps.append(step)
    return tuple(steps)


def _validated_tool_schemas(
    source: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(source, (str, bytes)) or not source:
        raise ValueError("native_tool_schemas_empty")
    tools: list[dict[str, Any]] = []
    names: list[str] = []
    for index, item in enumerate(source):
        if not isinstance(item, Mapping):
            raise TypeError(f"native_tool_schema_{index}_must_be_mapping")
        tool = deepcopy(dict(item))
        record = tool.get("function")
        record = record if isinstance(record, Mapping) else tool
        name = str(record.get("name") or "").strip()
        if not name:
            raise ValueError(f"native_tool_schema_{index}_name_missing")
        tools.append(tool)
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("native_tool_schema_names_not_unique")
    return tuple(tools)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label}_must_be_str")
    text = value.strip()
    if not text:
        raise ValueError(f"{label}_empty")
    return text


def _required_verbatim_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label}_must_be_str")
    if not value.strip():
        raise ValueError(f"{label}_empty")
    return value


def _required_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label}_must_be_positive_int")
    return value


def _canonical_bytes(value: Any) -> bytes:
    _validate_json_value(value)
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("fixed_handoff_payload_not_json_serializable") from exc
    return rendered.encode("utf-8")


def _validate_json_value(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"fixed_handoff_payload_nonfinite_number:{path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"fixed_handoff_payload_nonstring_key:{path}"
                )
            _validate_json_value(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    raise TypeError(
        f"fixed_handoff_payload_not_json_value:{path}:{type(value).__name__}"
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()
