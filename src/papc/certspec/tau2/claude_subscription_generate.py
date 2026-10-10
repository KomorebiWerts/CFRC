"""Route one tau2 model through Claude subscription Agent SDK.

The dispatcher is intentionally narrow: one pinned Claude model is translated
to tau2's next-message shape, while every other model is delegated unchanged to
tau2's existing provider function.  It never exposes Claude OAuth as an API key.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from papc.certspec.claude_subscription import ClaudeSubscriptionResponse

__all__ = [
    "Tau2ClaudeBridgeError",
    "install_tau2_claude_generate_bridge",
    "make_tau2_generate_dispatcher",
]


class Tau2ClaudeBridgeError(RuntimeError):
    """Claude returned a proposal that cannot be represented in tau2."""


class SubscriptionCompletionClient(Protocol):
    model: str

    async def complete(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        output_format: Mapping[str, Any] | None = None,
    ) -> ClaudeSubscriptionResponse: ...


GenerateFunction = Callable[..., Any]


def make_tau2_generate_dispatcher(
    *,
    claude_model: str,
    claude_client: SubscriptionCompletionClient,
    fallback_generate: GenerateFunction,
) -> GenerateFunction:
    """Build a generate function that changes only the pinned Claude model."""

    if claude_client.model != claude_model:
        raise Tau2ClaudeBridgeError(
            f"Claude client model mismatch: {claude_client.model} != {claude_model}"
        )

    def generate(
        *,
        model: str,
        messages: Sequence[Any],
        tools: Sequence[Any] | None = None,
        call_name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        # tau2/LiteLLM prefixes models sent through an OpenAI-compatible base.
        # Treat that one transport prefix as the same pinned Claude model while
        # preserving every other model name for the existing provider path.
        routed_model = model.removeprefix("openai/")
        if routed_model != claude_model:
            return fallback_generate(
                model=model,
                messages=messages,
                tools=tools,
                call_name=call_name,
                **kwargs,
            )
        return _generate_claude_tau2_message(
            client=claude_client,
            messages=messages,
            tools=tools or (),
            call_name=call_name,
        )

    return generate


def install_tau2_claude_generate_bridge(
    *,
    claude_model: str,
    claude_client: SubscriptionCompletionClient,
) -> GenerateFunction:
    """Install the dispatcher in tau2's two text-agent generate seams."""

    import tau2.agent.llm_agent as llm_agent_module
    import tau2.utils.llm_utils as llm_utils_module

    original_generate = llm_utils_module.generate
    dispatcher = make_tau2_generate_dispatcher(
        claude_model=claude_model,
        claude_client=claude_client,
        fallback_generate=original_generate,
    )
    llm_utils_module.generate = dispatcher
    llm_agent_module.generate = dispatcher
    return dispatcher


def _generate_claude_tau2_message(
    *,
    client: SubscriptionCompletionClient,
    messages: Sequence[Any],
    tools: Sequence[Any],
    call_name: str | None,
) -> Any:
    from tau2.data_model.message import AssistantMessage, ToolCall

    tool_by_name = {str(tool.name): tool for tool in tools}
    tool_schemas = [dict(tool.openai_schema) for tool in tools]
    response_schema = _response_schema(sorted(tool_by_name))
    system_prompt = _system_prompt(messages=messages, tool_schemas=tool_schemas)
    prompt = json.dumps(
        {
            "conversation": [
                _visible_message(message)
                for message in messages
                if getattr(message, "role", "") != "system"
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    # Keep every Claude turn on the same typed envelope.  Asking the model to
    # spell a tool call in prose and then silently treating malformed text as
    # an answer can turn a real write into an apparent refusal.  The envelope
    # also applies to no-tool turns, so the bridge has one behavior regardless
    # of which tools tau2 exposes on that turn.
    response = asyncio.run(
        client.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            output_format={"type": "json_schema", "schema": response_schema},
        )
    )
    payload = response.structured_output
    if not isinstance(payload, Mapping):
        raise Tau2ClaudeBridgeError("Claude structured output is missing")

    content_value = payload.get("content")
    content = content_value.strip() if isinstance(content_value, str) else ""
    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list):
        raise Tau2ClaudeBridgeError("Claude tool_calls must be a list")
    if content and raw_calls:
        raise Tau2ClaudeBridgeError("Claude must return either text or tool calls")
    if not content and not raw_calls:
        raise Tau2ClaudeBridgeError("Claude returned neither text nor tool calls")

    tool_calls: list[Any] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            raise Tau2ClaudeBridgeError("Claude tool call must be an object")
        name = str(raw_call.get("name") or "")
        arguments = raw_call.get("arguments")
        if name not in tool_by_name:
            raise Tau2ClaudeBridgeError(f"Claude proposed unknown tau2 tool: {name}")
        if not isinstance(arguments, Mapping):
            raise Tau2ClaudeBridgeError(
                f"Claude arguments for {name} must be an object"
            )
        params = getattr(tool_by_name[name], "params", None)
        if params is not None and hasattr(params, "model_validate"):
            try:
                params.model_validate(dict(arguments))
            except Exception as exc:
                raise Tau2ClaudeBridgeError(
                    f"Claude arguments failed tau2 validation for {name}: {exc}"
                ) from exc
        tool_calls.append(
            ToolCall(
                id=f"papc_claude_{uuid.uuid4().hex}",
                name=name,
                arguments=dict(arguments),
            )
        )

    return AssistantMessage(
        role="assistant",
        content=content or None,
        tool_calls=tool_calls or None,
        cost=response.estimated_cost_usd,
        usage=_tau2_noncache_usage(response.usage),
        generation_time_seconds=response.duration_ms / 1000,
        raw_data={
            "papc_provider": response.provider,
            "papc_model": response.model,
            "papc_call_name": call_name,
            "papc_subscription_usage": True,
            "papc_cache_usage_diagnostic": _cache_usage_diagnostic(response.usage),
        },
    )


def _response_schema(tool_names: Sequence[str]) -> dict[str, Any]:
    name_schema: dict[str, Any] = {"type": "string"}
    if tool_names:
        name_schema["enum"] = list(tool_names)
    return {
        "type": "object",
        "properties": {
            "content": {"type": ["string", "null"]},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": name_schema,
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["content", "tool_calls"],
        "additionalProperties": False,
    }


def _tau2_noncache_usage(usage: Mapping[str, Any] | None) -> dict[str, int] | None:
    """Map Anthropic non-cache tokens to tau2's historical usage keys."""

    if not isinstance(usage, Mapping):
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    return {
        "prompt_tokens": max(0, int(input_tokens or 0)),
        "completion_tokens": max(0, int(output_tokens or 0)),
    }


def _cache_usage_diagnostic(usage: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep cache counters visible without feeding primary token accounting."""

    usage_map = usage if isinstance(usage, Mapping) else {}
    return {
        "cache_creation_input_tokens": max(
            0, int(usage_map.get("cache_creation_input_tokens") or 0)
        ),
        "cache_read_input_tokens": max(
            0, int(usage_map.get("cache_read_input_tokens") or 0)
        ),
        "excluded_from_primary_token_accounting": True,
    }


def _system_prompt(
    *, messages: Sequence[Any], tool_schemas: Sequence[Mapping[str, Any]]
) -> str:
    inherited = "\n\n".join(
        str(getattr(message, "content", "") or "")
        for message in messages
        if getattr(message, "role", "") == "system"
    ).strip()
    return (
        f"{inherited}\n\n"
        "Return the next tau2 assistant action. Use exactly one mode: either "
        "a non-empty content string with an empty tool_calls list, or null content "
        "with one or more tool calls. Never claim a tool succeeded before its tool "
        "result appears in the conversation. Use only the listed tool schemas. "
        "Return only one JSON object with exactly the keys content and tool_calls; "
        "do not use markdown fences or explanatory text. Each tool_calls item must "
        "have exactly name and arguments.\n"
        "Tool schemas: "
        f"{json.dumps(list(tool_schemas), ensure_ascii=False, separators=(',', ':'))}"
    ).strip()


def _visible_message(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": str(getattr(message, "role", "")),
        "content": getattr(message, "content", None),
    }
    identifier = getattr(message, "id", None)
    if identifier:
        payload["tool_call_id"] = str(identifier)
    calls = getattr(message, "tool_calls", None)
    if calls:
        payload["tool_calls"] = [
            {
                "name": str(getattr(call, "name", "")),
                "arguments": dict(getattr(call, "arguments", {}) or {}),
            }
            for call in calls
        ]
    return payload
