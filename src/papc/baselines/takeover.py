"""Frozen prompt contracts for the two protected takeover baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Literal

IMPLICIT_DIRECT_TAKEOVER = "implicit_direct_takeover"
PARADIGM_TAKEOVER = "paradigm_takeover"
TakeoverArm = Literal["implicit_direct_takeover", "paradigm_takeover"]

_MESSAGE_KEYS = frozenset(
    {"role", "content", "name", "tool_call_id", "tool_calls"}
)
_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})
_NATIVE_INPUT_ITEM_KEYS = {
    "message": frozenset({"id", "type", "role", "content", "status"}),
    "function_call": frozenset(
        {"id", "type", "call_id", "name", "arguments", "status"}
    ),
    "function_call_output": frozenset(
        {"type", "call_id", "output"}
    ),
}

_B1_INSTRUCTION = (
    "Continue the original request from the environment's current reached "
    "state. The complete accepted native transcript is provided below. Use "
    "the available native tools and their ordinary read operations as needed, "
    "then stop only through the surface's native terminal signal."
)
_B2_INSTRUCTION = (
    "Complete the original request in the current environment. Another agent "
    "may already have modified the environment. You are not given that "
    "agent's prior transcript or a state summary. Use the available native "
    "read operations to reconstruct the current state, then stop only through "
    "the surface's native terminal signal."
)


@dataclass(frozen=True)
class ExposureContract:
    original_request_visible: bool
    accepted_transcript_visible: bool
    reached_environment_read_access: bool
    full_native_tool_schemas_visible: bool
    native_terminal_signal_used: bool
    frontier_visible: bool = False
    contract_visible: bool = False
    state_dump_visible: bool = False
    completed_action_list_visible: bool = False
    task_id_visible: bool = False
    gold_outcome_visible: bool = False


@dataclass(frozen=True)
class TakeoverPrompt:
    arm: TakeoverArm
    system_instruction: str
    messages: tuple[dict[str, Any], ...]
    exposure: ExposureContract

    def public_contract(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "system_instruction": self.system_instruction,
            "message_count": len(self.messages),
            "exposure": asdict(self.exposure),
        }


def build_implicit_direct_takeover(
    accepted_transcript: Sequence[Mapping[str, Any]],
) -> TakeoverPrompt:
    """Build B1 from the complete accepted native transcript."""

    messages = _validated_transcript(accepted_transcript)
    if not any(message.get("role") == "user" for message in messages):
        raise ValueError("accepted_transcript_missing_user_request")
    return TakeoverPrompt(
        arm=IMPLICIT_DIRECT_TAKEOVER,
        system_instruction=_B1_INSTRUCTION,
        messages=messages,
        exposure=ExposureContract(
            original_request_visible=True,
            accepted_transcript_visible=True,
            reached_environment_read_access=True,
            full_native_tool_schemas_visible=True,
            native_terminal_signal_used=True,
        ),
    )


def build_paradigm_takeover(original_request: str) -> TakeoverPrompt:
    """Build B2 without accepting a transcript or reached-state summary."""

    if not isinstance(original_request, str):
        raise TypeError("original_request_must_be_str")
    request = original_request.strip()
    if not request:
        raise ValueError("original_request_empty")
    return TakeoverPrompt(
        arm=PARADIGM_TAKEOVER,
        system_instruction=_B2_INSTRUCTION,
        messages=({"role": "user", "content": request},),
        exposure=ExposureContract(
            original_request_visible=True,
            accepted_transcript_visible=False,
            reached_environment_read_access=True,
            full_native_tool_schemas_visible=True,
            native_terminal_signal_used=True,
        ),
    )


def _validated_transcript(
    transcript: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if not transcript:
        raise ValueError("accepted_transcript_empty")
    messages: list[dict[str, Any]] = []
    for index, source in enumerate(transcript):
        message = dict(source)
        item_type = str(message.get("type") or "")
        if item_type:
            messages.append(
                deepcopy(_validated_native_input_item(message, index=index))
            )
            continue
        unknown = sorted(set(message) - _MESSAGE_KEYS)
        if unknown:
            raise ValueError(
                f"accepted_transcript_message_{index}_unsupported_keys:"
                + ",".join(unknown)
            )
        role = str(message.get("role") or "")
        if role not in _MESSAGE_ROLES:
            raise ValueError(
                f"accepted_transcript_message_{index}_invalid_role:{role}"
            )
        message["role"] = role
        messages.append(deepcopy(message))
    return tuple(messages)


def _validated_native_input_item(
    item: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    item_type = str(item.get("type") or "")
    allowed = _NATIVE_INPUT_ITEM_KEYS.get(item_type)
    if allowed is None:
        raise ValueError(
            f"accepted_transcript_item_{index}_unsupported_type:{item_type}"
        )
    unknown = sorted(set(item) - allowed)
    if unknown:
        raise ValueError(
            f"accepted_transcript_item_{index}_unsupported_keys:"
            + ",".join(unknown)
        )

    if item_type == "message":
        role = str(item.get("role") or "")
        if role not in {"user", "assistant"}:
            raise ValueError(
                f"accepted_transcript_item_{index}_invalid_role:{role}"
            )
        if not isinstance(item.get("content"), (str, list)):
            raise TypeError(
                f"accepted_transcript_item_{index}_content_invalid"
            )
        item["role"] = role
        return item

    call_id = str(item.get("call_id") or "").strip()
    if not call_id:
        raise ValueError(
            f"accepted_transcript_item_{index}_call_id_missing"
        )
    item["call_id"] = call_id
    if item_type == "function_call":
        name = str(item.get("name") or "").strip()
        arguments = item.get("arguments")
        if not name:
            raise ValueError(
                f"accepted_transcript_item_{index}_name_missing"
            )
        if not isinstance(arguments, (str, Mapping)):
            raise TypeError(
                f"accepted_transcript_item_{index}_arguments_invalid"
            )
        item["name"] = name
    elif not isinstance(item.get("output"), (str, Mapping, list)):
        raise TypeError(
            f"accepted_transcript_item_{index}_output_invalid"
        )
    return item
